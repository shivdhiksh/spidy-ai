"""
ReflectionEngine — Post-Task Outcome Evaluation (Milestone 13)
===============================================================
After every task execution, the ReflectionEngine analyses the Brain's
response and decides what the ExecutionLoop should do next.

Decision outcomes
-----------------
CONTINUE    — task succeeded; proceed to the next task
RETRY       — task failed but may succeed on retry (attempt < max)
ALTERNATIVE — task failed; try an alternative utterance
ASK_USER    — ambiguous or blocked; prompt the user for input
ABORT       — unrecoverable failure; stop the goal

Evaluation signals
------------------
- Brain response text (contains success/failure language)
- result_success flag from ToolResult (propagated as bool)
- Current attempt count vs. max_retries
- Failure patterns (error keywords, clarification patterns)
- Success patterns (confirmation words)

Design
------
- Fully synchronous (pure function — no async I/O)
- Zero LLM calls (pattern matching only — fast and offline-capable)
- Returns a (decision, reason) pair for logging and event publishing
- Conservative: defaults to CONTINUE on ambiguous cases
"""

from __future__ import annotations

import re

from spidy.agent.types import ReflectionDecision, TaskRecord, TaskState
from spidy.logging.logger import get_logger

log = get_logger(__name__)

# ─── Signal Patterns ──────────────────────────────────────────────────────────

# Patterns in the Brain's response text that indicate clear SUCCESS
_SUCCESS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bi'?ve (opened|created|installed|launched|found|started|done|completed|set|taken)\b", re.I),
    re.compile(r"\b(done|all set|got it|completed|success|finished|ready)\b", re.I),
    re.compile(r"\b(sure thing|no problem|consider it done)\b", re.I),
    re.compile(r"\bopened\b.{0,40}\bfor you\b", re.I),
    re.compile(r"\binstalled\b", re.I),
    re.compile(r"\bcreated\b.{0,40}\bfile\b", re.I),
    re.compile(r"\bsearching\b", re.I),
]

# Patterns in the response text that indicate FAILURE
_FAILURE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(couldn'?t|could not|failed|error|unable|can'?t|cannot)\b", re.I),
    re.compile(r"\b(no skill|permission denied|not found|traceback)\b", re.I),
    re.compile(r"\b(i'?m not able|i don'?t have a skill for)\b", re.I),
    re.compile(r"\bsomething went wrong\b", re.I),
]

# Patterns that indicate the user needs to clarify something
_CLARIFICATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(could you|can you|would you|please)\s+(clarify|rephrase|be more specific)\b", re.I),
    re.compile(r"\bi'?m not sure (what|which|how)\b", re.I),
    re.compile(r"\bI didn'?t understand\b", re.I),
    re.compile(r"\b(what do you mean|which one did you mean)\b", re.I),
]

# Patterns that indicate a retryable transient issue
_RETRY_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(try again|retry|timed out|connection|network|temporarily)\b", re.I),
    re.compile(r"\bsnag\b", re.I),
]

# Minimum response text length to consider a success (avoids treating empty/short errors as success)
_MIN_SUCCESS_RESPONSE_LENGTH = 8


from enum import Enum

class RecoveryCategory(str, Enum):
    WINDOW_NOT_FOUND = "WINDOW_NOT_FOUND"
    CONTROL_NOT_FOUND = "CONTROL_NOT_FOUND"
    WRONG_WINDOW = "WRONG_WINDOW"
    FOCUS_LOST = "FOCUS_LOST"
    TIMEOUT = "TIMEOUT"
    APP_NOT_READY = "APP_NOT_READY"
    NAVIGATION_FAILED = "NAVIGATION_FAILED"
    ELEMENT_CHANGED = "ELEMENT_CHANGED"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    TEXT_NOT_ENTERED = "TEXT_NOT_ENTERED"
    SAVE_FAILED = "SAVE_FAILED"
    PROCESS_CRASHED = "PROCESS_CRASHED"
    TEST_FAILED = "TEST_FAILED"
    CODE_ERROR = "CODE_ERROR"
    SCREEN_UNREADABLE = "SCREEN_UNREADABLE"
    VISION_UNCERTAIN = "VISION_UNCERTAIN"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class ReflectionEngine:
    """
    Evaluates a task result and decides the next ExecutionLoop action.

    Parameters
    ----------
    max_retries:
        Maximum number of times a task can be retried before ABORT/next.
    """

    def __init__(self, max_retries: int = 2) -> None:
        self._max_retries = max_retries

    @staticmethod
    def diagnose_error(text: str) -> RecoveryCategory:
        """Categorize error for deterministic recovery routing."""
        t = text.lower()
        if "window" in t and ("not found" in t or "could not find" in t):
            return RecoveryCategory.WINDOW_NOT_FOUND
        if "element" in t or "control" in t or "button" in t:
            return RecoveryCategory.CONTROL_NOT_FOUND
        if "focus" in t or "foreground" in t:
            return RecoveryCategory.FOCUS_LOST
        if "timed out" in t or "timeout" in t:
            return RecoveryCategory.TIMEOUT
        if "file not found" in t or "no such file" in t or "could not find file" in t:
            return RecoveryCategory.FILE_NOT_FOUND
        if "permission" in t or "access denied" in t:
            return RecoveryCategory.PERMISSION_DENIED
        if "syntaxerror" in t or "nameerror" in t or "typeerror" in t or "traceback" in t:
            return RecoveryCategory.CODE_ERROR
        if "test" in t and ("failed" in t or "failure" in t):
            return RecoveryCategory.TEST_FAILED
        if "navigate" in t or "dns" in t or "url" in t:
            return RecoveryCategory.NAVIGATION_FAILED
        return RecoveryCategory.UNKNOWN_ERROR

    def reflect(
        self,
        task: TaskRecord,
        response_text: str,
        result_success: bool,
    ) -> tuple[ReflectionDecision, str]:
        """
        Evaluate the outcome of a completed task.
        """
        text = (response_text or "").strip()

        # ── Clarification needed ───────────────────────────────────────────
        if self._matches_any(text, _CLARIFICATION_PATTERNS):
            return (
                ReflectionDecision.ASK_USER,
                "Brain asked for clarification.",
            )

        # ── Explicit success ───────────────────────────────────────────────
        if result_success and self._matches_any(text, _SUCCESS_PATTERNS):
            return (ReflectionDecision.CONTINUE, "Task succeeded.")

        if result_success and len(text) >= _MIN_SUCCESS_RESPONSE_LENGTH:
            # Response is reasonably long — assume success even if no keyword matched
            return (ReflectionDecision.CONTINUE, "Task likely succeeded (response received).")

        # ── Transient / retryable failure ──────────────────────────────────
        if self._matches_any(text, _RETRY_PATTERNS):
            if task.attempt < self._max_retries:
                return (
                    ReflectionDecision.RETRY,
                    f"Transient failure detected (attempt {task.attempt}/{self._max_retries}). Retrying.",
                )

        # ── Hard failure ───────────────────────────────────────────────────
        if self._matches_any(text, _FAILURE_PATTERNS) or not result_success:
            if task.attempt < self._max_retries:
                return (
                    ReflectionDecision.RETRY,
                    f"Task failed (attempt {task.attempt}/{self._max_retries}). Retrying.",
                )
            # Retries exhausted — try alternative only if task has no alternative utterance yet
            if not task.extra.get("alternative_tried"):
                return (
                    ReflectionDecision.ALTERNATIVE,
                    "All retries failed. Attempting an alternative approach.",
                )
            return (
                ReflectionDecision.ABORT,
                f"Task '{task.description}' failed after {task.attempt} attempts. Cannot recover.",
            )

        if text:
            return (ReflectionDecision.CONTINUE, "Task completed (response received).")

        # Empty response — treat as soft failure
        if task.attempt < self._max_retries:
            return (
                ReflectionDecision.RETRY,
                f"Empty response (attempt {task.attempt}). Retrying.",
            )
        return (
            ReflectionDecision.CONTINUE,
            "Empty response after retries — moving to next task.",
        )

    def build_alternative_utterance(self, task: TaskRecord) -> str:
        """
        Generate an alternative utterance for a failed task.
        """
        original = task.utterance
        alternatives: dict[str, str] = {
            "open a terminal": "launch command prompt or powershell",
            "create a folder": "make a new directory",
            "install flask": "pip install flask in the command line",
            "create a file": "make a new file",
            "open vs code": "launch visual studio code",
            "run the flask": "start the flask application with python",
            "search google": "open google.com and search",
            "open the web browser": "launch chrome or edge browser",
        }

        original_lower = original.lower()
        for pattern, replacement in alternatives.items():
            if pattern in original_lower:
                alt = original_lower.replace(pattern, replacement)
                log.debug(
                    "ReflectionEngine: alternative utterance: '{alt}'",
                    alt=alt[:80],
                )
                return alt

        # Generic rephrase
        return f"please try to {original}"

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _matches_any(text: str, patterns: list[re.Pattern]) -> bool:
        """Return True if any pattern matches the text."""
        return any(p.search(text) for p in patterns)
