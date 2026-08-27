"""
TaskAuthorityChecker — Tool Authority Gate for Agent Execution
==============================================================
Before the ExecutionLoop calls Brain.process() for any task, the
TaskAuthorityChecker inspects the task utterance to determine the
required authority level.

This ensures destructive or sensitive actions require user confirmation
even when the LLM generated the task — the LLM cannot bypass this gate.

Authority levels (mapped from PermissionTier)
---------------------------------------------
SAFE     (T0) — read-only and non-destructive: open apps, screenshots, search
AWARE    (T1) — write-once, low-risk: save notes, set timers, write files
CONFIRM  (T2) — require user confirmation: upload, post, send, delete files
CRITICAL (T3) — always require admin unlock: shutdown, restart, mass-delete

Design
------
- Keyword matching on task.utterance (no LLM, no network, instant)
- Ordered by authority level (CRITICAL checked before CONFIRM, etc.)
- LLM-generated tasks go through the same gate as user-typed tasks
- STOP command priority is NOT managed here (InterruptionHandler handles it)
- Publishing an AgentConfirmationRequiredEvent is the ExecutionLoop's job;
  the checker only returns the level.

Usage
-----
    checker = TaskAuthorityChecker()
    level = checker.check(task)
    if checker.requires_confirmation(task):
        # pause and publish event — DO NOT call Brain.process() yet
        ...
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.agent.types import TaskRecord

log = get_logger(__name__)


# ── Authority Levels ───────────────────────────────────────────────────────────


class AuthorityLevel(str, Enum):
    """
    Required authority level for a task.

    Maps to the existing PermissionTier system:
      SAFE     → T0   (always allowed)
      AWARE    → T1   (allowed with logging)
      CONFIRM  → T2   (explicit confirmation every time)
      CRITICAL → T3   (admin unlock required)
    """
    SAFE     = "safe"       # T0
    AWARE    = "aware"      # T1
    CONFIRM  = "confirm"    # T2
    CRITICAL = "critical"   # T3


# ── Keyword Tables ─────────────────────────────────────────────────────────────

# CRITICAL — T3: system-level destructive actions
_CRITICAL_KEYWORDS: list[str] = [
    "shutdown", "shut down", "power off", "restart", "reboot",
    "format", "wipe", "factory reset",
    "delete all", "remove all", "bulk delete", "mass delete",
    "uninstall all",
    "disable firewall", "disable antivirus",
    "modify registry", "edit registry",
    "change security", "modify security",
    "delete account", "remove account",
    "rm -rf", "del /f /s /q",
]

# CONFIRM — T2: user-data-affecting or externally-visible actions
_CONFIRM_KEYWORDS: list[str] = [
    # Social / messaging
    "upload", "post to", "publish", "share to",
    "send message", "send email", "send mail", "reply to",
    "tweet", "instagram", "facebook post", "linkedin post",
    "submit form", "place order", "buy", "purchase",
    # File destructive
    "delete", "remove", "erase", "trash", "destroy",
    "delete file", "delete folder", "remove file", "remove folder",
    "overwrite", "replace file",
    "move to trash", "empty trash",
    # System changes
    "install software", "install application", "install app",
    "uninstall",
    "change password", "reset password",
    # Code execution escape hatches
    "code_exec", "execute_code", "run_code", "run arbitrary code", "eval(",
    # Finance
    "transfer money", "send payment", "wire transfer",
    # Git
    "git push", "force push", "git rm",
]


# AWARE — T1: write/modify but low-risk
_AWARE_KEYWORDS: list[str] = [
    "save file", "write file", "create file", "modify file",
    "save note", "write note",
    "set timer", "set alarm", "set reminder",
    "change volume", "set volume",
    "rename file", "rename folder",
    "copy file", "move file",
    "download", "install package", "pip install",
    "npm install", "run script", "execute script",
    "open terminal", "run command",
    "git commit", "git add",
    "create folder", "make folder", "mkdir",
]

# Everything else → SAFE (T0)


class TaskAuthorityChecker:
    """
    Determines the authority level required for a task before execution.

    Parameters
    ----------
    min_confirm_level:
        Tasks at or above this level require confirmation.
        Default: CONFIRM (T2).
    """

    def __init__(
        self,
        min_confirm_level: AuthorityLevel = AuthorityLevel.CONFIRM,
    ) -> None:
        self._min_confirm_level = min_confirm_level
        _LEVEL_ORDER = [
            AuthorityLevel.SAFE,
            AuthorityLevel.AWARE,
            AuthorityLevel.CONFIRM,
            AuthorityLevel.CRITICAL,
        ]
        self._confirm_ordinal = _LEVEL_ORDER.index(min_confirm_level)
        self._level_order = _LEVEL_ORDER

    def check(self, task: "TaskRecord") -> AuthorityLevel:
        """
        Return the authority level required to execute this task.

        Parameters
        ----------
        task:
            The TaskRecord whose utterance is inspected.

        Returns
        -------
        AuthorityLevel
            The required authority level.
        """
        utterance_lower = (task.utterance or "").lower()
        description_lower = (task.description or "").lower()
        combined = f"{utterance_lower} {description_lower}"

        # M18: also scan structured action fields (action, target) to prevent bypass
        if task.action and isinstance(task.action, dict):
            action_str = " ".join(str(v) for v in task.action.values() if v)
            combined = f"{combined} {action_str.lower()}"

        # Check from most restrictive to least
        if any(kw in combined for kw in _CRITICAL_KEYWORDS):
            log.info(
                "[AGENT] Authority: CRITICAL task detected: '{desc}'",
                desc=task.description[:60],
            )
            return AuthorityLevel.CRITICAL

        if any(kw in combined for kw in _CONFIRM_KEYWORDS):
            log.info(
                "[AGENT] Authority: CONFIRM task detected: '{desc}'",
                desc=task.description[:60],
            )
            return AuthorityLevel.CONFIRM

        if any(kw in combined for kw in _AWARE_KEYWORDS):
            log.debug(
                "[AGENT] Authority: AWARE task: '{desc}'",
                desc=task.description[:60],
            )
            return AuthorityLevel.AWARE

        return AuthorityLevel.SAFE

    def requires_confirmation(self, task: "TaskRecord") -> bool:
        """
        Return True if this task needs explicit user confirmation.

        The LLM cannot bypass this check — it operates purely on the
        task utterance text, before Brain.process() is called.
        """
        level = self.check(task)
        ordinal = self._level_order.index(level)
        return ordinal >= self._confirm_ordinal

    def level_label(self, task: "TaskRecord") -> str:
        """Return a short human-readable authority label for logging."""
        level = self.check(task)
        labels = {
            AuthorityLevel.SAFE: "T0/SAFE",
            AuthorityLevel.AWARE: "T1/AWARE",
            AuthorityLevel.CONFIRM: "T2/CONFIRM",
            AuthorityLevel.CRITICAL: "T3/CRITICAL",
        }
        return labels[level]
