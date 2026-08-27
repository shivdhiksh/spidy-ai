"""
TaskObserver — Environment Observation After Task Execution
===========================================================
After each task executes (Brain.process returns), the TaskObserver
inspects the environment to determine what actually happened.

Observation methods (tried cheapest-first)
-------------------------------------------
1. Response text analysis   — always performed; baseline signal
2. App state check          — is a named application running?
3. File existence check     — did a file get created/modified?
4. Browser URL check        — did navigation succeed?
5. Screenshot + Vision      — for complex visual tasks (optional, configurable)

Design
------
- Observation is ADDITIVE to the existing ReflectionEngine, not a replacement.
  ReflectionEngine still provides the primary success/failure signal.
  TaskObserver provides secondary environmental evidence.
- Observation is SKIPPED for terminal single-step tasks (fast path).
- All observation methods are fail-open: if a check cannot run, it returns
  None for that field rather than raising.
- No LLM calls — observation is deterministic and fast.
- Vision (screenshot) is optional and only used when vision_enabled=True
  AND the task hints at a visual outcome.

Usage
-----
    observer = TaskObserver(vision=vision_engine)
    obs = await observer.observe(task, response_text)
    # obs.success_signal: True | False | None
    # obs.summary: human-readable description of what was observed
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.agent.types import TaskRecord
    from spidy.brain.interfaces import VisionInterface

log = get_logger(__name__)


# ── Observation Result ─────────────────────────────────────────────────────────


@dataclass
class Observation:
    """
    The result of observing the environment after a task execution.

    Attributes
    ----------
    method:
        Which observation method produced the primary signal.
        One of: "response_text", "app_state", "file_check",
                "browser_url", "vision", "skipped"
    summary:
        Human-readable description of what was observed.
    success_signal:
        True  = environment evidence supports success.
        False = environment evidence suggests failure.
        None  = inconclusive (fail-open).
    app_running:
        Whether a named app was detected as running (None = not checked).
    file_exists:
        Whether a expected file was found (None = not checked).
    browser_url:
        The current browser URL if checked (empty = not checked).
    screenshot_path:
        Absolute path to the captured screenshot (empty = no screenshot).
    extra:
        Any additional observation metadata.
    """
    method: str = "skipped"
    summary: str = ""
    success_signal: bool | None = None
    app_running: bool | None = None
    file_exists: bool | None = None
    browser_url: str = ""
    screenshot_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# ── Keywords for observation routing ──────────────────────────────────────────

# File-related task keywords → check if a file was created/modified
_FILE_KEYWORDS = frozenset([
    "create", "write", "save", "make", "generate",
    "file", "folder", "directory", "mkdir", ".py", ".txt", ".md",
    ".html", ".js", ".css", ".json", ".yaml", ".toml",
])

# App-related task keywords → check if the app is running
_APP_KEYWORDS = frozenset([
    "open", "launch", "start", "close", "quit", "exit",
    "minimize", "maximize", "focus", "bring", "switch",
])

# Browser-related task keywords
_BROWSER_KEYWORDS = frozenset([
    "search", "navigate", "browse", "go to", "open", "url",
    "website", "youtube", "google", "instagram", "twitter",
])

# Keywords that suggest a visual outcome worth screenshotting
_VISUAL_KEYWORDS = frozenset([
    "upload", "post", "submit", "instagram", "twitter", "facebook",
    "form", "button", "click", "screenshot",
])

# App name extraction — patterns to look for
_KNOWN_APP_NAMES = [
    "chrome", "google chrome",
    "edge", "microsoft edge",
    "firefox",
    "notepad",
    "calculator",
    "vs code", "vscode", "visual studio code",
    "discord",
    "spotify",
    "terminal", "cmd", "powershell", "command prompt",
    "paint",
    "task manager",
    "explorer", "file explorer",
    "settings", "windows settings",
]


class TaskObserver:
    """
    Observes the environment after a task to produce an Observation.

    Parameters
    ----------
    vision:
        Optional VisionInterface for screenshot + visual analysis.
        When None, visual observation is skipped.
    vision_enabled:
        Master switch for visual observation. Default False (perf-safe).
    """

    def __init__(
        self,
        vision: "VisionInterface | None" = None,
        vision_enabled: bool = False,
    ) -> None:
        self._vision = vision
        self._vision_enabled = vision_enabled

    # ── Public API ─────────────────────────────────────────────────────────

    async def observe(
        self,
        task: "TaskRecord",
        response_text: str,
        skip_for_terminal: bool = True,
    ) -> Observation:
        """
        Observe the environment after a task execution.

        Parameters
        ----------
        task:
            The TaskRecord that was just executed.
        response_text:
            Brain's response text for this task.
        skip_for_terminal:
            If True (default), skip expensive observation for terminal
            single-step tasks (e.g. "Open Calculator"). The response
            text alone is sufficient.

        Returns
        -------
        Observation
            Environment observation result. Never raises.
        """
        # ── Fast path: terminal single-step tasks (never skip browser tasks) ─
        is_browser_task = False
        if task.action and isinstance(task.action, dict):
            skill = (task.action.get("skill") or "").lower()
            act = (task.action.get("action") or "").lower()
            if skill == "browser" or act in ("navigate", "search", "open_url", "search_youtube", "search_google"):
                is_browser_task = True
        elif any(kw in (task.utterance or "").lower() for kw in ("youtube", "google", "browser", "navigate", "search")):
            is_browser_task = True

        if skip_for_terminal and task.terminal and task.attempt <= 1 and not is_browser_task:
            log.debug(
                "[AGENT] Observer: skipping observation for terminal task '{desc}'",
                desc=task.description[:60],
            )
            # Still analyse response text (zero cost)
            sig = self._analyse_response(response_text)
            return Observation(
                method="skipped",
                summary=f"Terminal task — using response text signal only.",
                success_signal=sig,
            )

        utterance_lower = (task.utterance or "").lower()

        # M18: extract expected_outcome from structured action
        expected_outcome: str = ""
        if task.action and isinstance(task.action, dict):
            expected_outcome = task.action.get("expected_outcome", "")

        # ── Step 1: Response text analysis ─────────────────────────────────
        text_signal = self._analyse_response(response_text)

        # ── Step 2: Browser URL extraction & Verification ───────────────────
        browser_url = self._extract_browser_url(response_text)
        browser_verified: bool | None = None
        browser_reason: str = ""

        if is_browser_task:
            act = (task.action.get("action") or "").lower() if task.action else ""
            target = (task.action.get("target") or "").lower() if task.action else ""
            req_query = (task.action.get("query") or "").lower() if task.action else ""
            req_url = (task.action.get("url") or "").lower() if task.action else ""

            if act == "navigate" or "navigat" in task.description.lower() or "open" in task.description.lower():
                expected_host = ""
                if "youtube" in target or "youtube" in req_url or "youtube" in task.description.lower():
                    expected_host = "youtube.com"
                elif "google" in target or "google" in req_url or "google" in task.description.lower():
                    expected_host = "google.com"
                elif req_url:
                    expected_host = req_url.replace("https://", "").replace("http://", "").split("/")[0]

                if browser_url:
                    if expected_host:
                        if expected_host in browser_url.lower():
                            browser_verified = True
                            browser_reason = f"Verified browser navigated to {expected_host} (URL: {browser_url})"
                        else:
                            browser_verified = False
                            browser_reason = f"Browser URL mismatch: expected {expected_host}, got {browser_url}"
                    else:
                        browser_verified = True
                        browser_reason = f"Verified browser opened with URL: {browser_url}"
                elif "http" in response_text or "navigat" in response_text.lower() or "open" in response_text.lower():
                    browser_verified = None
                    browser_reason = "Browser state unverified (URL not extracted)"

            elif act in ("extract_text", "read_page", "read_result", "get_text") or "extract" in task.description.lower() or "read" in task.description.lower():
                if len(response_text.strip()) > 5 and "error" not in response_text.lower():
                    browser_verified = True
                    browser_reason = f"Verified text extracted from browser ({len(response_text)} chars)"
                else:
                    browser_verified = False
                    browser_reason = "Failed to extract text from browser page"

            elif act == "search" or "search" in task.description.lower():
                query_disp = (task.action.get("query") or "").strip() if task.action else ""
                if not query_disp and "for" in task.description.lower():
                    query_disp = task.description.split("for", 1)[1].strip()

                if browser_url:
                    norm_q = req_query.replace(" ", "+") if req_query else ""
                    norm_q_raw = req_query if req_query else ""
                    if norm_q and (norm_q in browser_url.lower() or all(w in browser_url.lower() for w in norm_q_raw.split())):
                        browser_verified = True
                        browser_reason = f"Verified search results for '{query_disp or req_query}' loaded (URL: {browser_url})"
                    else:
                        browser_verified = False
                        browser_reason = f"Browser search query mismatch: '{query_disp or req_query}' not found in URL {browser_url}"
                elif "search" in response_text.lower() or "result" in response_text.lower():
                    browser_verified = None
                    browser_reason = "Browser search state unverified (URL not extracted)"

        # ── Step 3: App state check ─────────────────────────────────────────
        app_running: bool | None = None
        app_name: str | None = None
        if task.action and isinstance(task.action, dict):
            t = (task.action.get("target") or task.action.get("browser") or "").lower().strip()
            if t in _KNOWN_APP_NAMES or any(a in t for a in ("edge", "chrome", "firefox")):
                app_name = t
        if not app_name:
            app_name = self._extract_app_name(utterance_lower)
        if app_name:
            app_running = self._check_app_running(app_name)

        # ── Step 4: File existence check ────────────────────────────────────
        file_exists: bool | None = None
        file_path = self._extract_file_path(utterance_lower, response_text)
        if file_path:
            file_exists = os.path.exists(file_path)
            log.debug(
                "[AGENT] Observer: file check '{path}' → {exists}",
                path=file_path[:80],
                exists=file_exists,
            )

        # ── Step 5: Visual observation (screenshot) ─────────────────────────
        screenshot_path = ""
        if self._vision_enabled and self._vision is not None:
            is_visual = any(kw in utterance_lower for kw in _VISUAL_KEYWORDS)
            if is_visual:
                screenshot_path = await self._take_screenshot()

        # ── Build consolidated signal ────────────────────────────────────────
        success_signal, method, summary = self._consolidate(
            text_signal=text_signal,
            app_name=app_name,
            app_running=app_running,
            file_path=file_path,
            file_exists=file_exists,
            screenshot_path=screenshot_path,
            browser_url=browser_url,
            browser_verified=browser_verified,
            browser_reason=browser_reason,
            is_browser_task=is_browser_task,
        )

        # M18: enrich summary with expected_outcome when present
        extra: dict = {}
        if expected_outcome:
            extra["expected_outcome"] = expected_outcome
            if success_signal is True:
                summary = f"{summary} | Expected: {expected_outcome[:60]}"
            elif success_signal is False:
                summary = f"{summary} | Wanted: {expected_outcome[:60]}"

        obs = Observation(
            method=method,
            summary=summary,
            success_signal=success_signal,
            app_running=app_running,
            file_exists=file_exists,
            browser_url=browser_url,
            screenshot_path=screenshot_path,
            extra=extra,
        )

        log.info(
            "[AGENT] Observation: task='{desc}' method={m} signal={s} summary='{sum}'",
            desc=task.description[:60],
            m=method,
            s=success_signal,
            sum=summary[:80],
        )

        return obs

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _analyse_response(response_text: str) -> bool | None:
        """Analyse Brain's response text for success/failure signals."""
        if not response_text:
            return None
        text = response_text.lower()
        _SUCCESS = ["opened", "created", "installed", "launched", "done",
                    "completed", "found", "saved", "set", "started", "finished",
                    "all set", "sure thing", "searching", "navigating"]
        _FAILURE = ["couldn't", "could not", "failed", "error", "unable",
                    "can't", "cannot", "not found", "permission denied",
                    "no skill", "something went wrong", "i'm not able"]
        if any(w in text for w in _FAILURE):
            return False
        if any(w in text for w in _SUCCESS):
            return True
        # Non-empty response with no clear signal → optimistic
        if len(response_text.strip()) > 8:
            return True
        return None

    @staticmethod
    def _extract_app_name(utterance_lower: str) -> str | None:
        """Extract a known app name from the utterance, if any."""
        if not any(kw in utterance_lower for kw in _APP_KEYWORDS):
            return None
        for app in _KNOWN_APP_NAMES:
            if app in utterance_lower:
                return app
        return None

    @staticmethod
    def _check_app_running(app_name: str) -> bool | None:
        """
        Check if a named application process is running.
        Returns None if the check cannot be performed.
        """
        try:
            import subprocess
            # Windows: tasklist
            result = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode != 0:
                return None
            output_lower = result.stdout.lower()
            # Map app names to process names
            _PROCESS_MAP = {
                "chrome": "chrome.exe",
                "google chrome": "chrome.exe",
                "edge": "msedge.exe",
                "microsoft edge": "msedge.exe",
                "firefox": "firefox.exe",
                "notepad": "notepad.exe",
                "calculator": "calculatorapp.exe",
                "vs code": "code.exe",
                "vscode": "code.exe",
                "visual studio code": "code.exe",
                "discord": "discord.exe",
                "spotify": "spotify.exe",
                "paint": "mspaint.exe",
                "task manager": "taskmgr.exe",
                "powershell": "powershell.exe",
                "cmd": "cmd.exe",
                "terminal": "windowsterminal.exe",
                "command prompt": "cmd.exe",
            }
            process = _PROCESS_MAP.get(app_name.lower(), "")
            if process:
                return process in output_lower
            return None
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _extract_file_path(utterance_lower: str, response_text: str) -> str | None:
        """
        Try to extract a file path from utterance or response text.
        Returns None if no useful path can be extracted.
        """
        import re
        # Look for paths in response text (e.g. "Created C:\Users\...\file.py")
        path_pattern = re.compile(
            r'[A-Za-z]:[/\\][^\s\'"<>|]{3,}',
            re.IGNORECASE,
        )
        for text in [response_text, utterance_lower]:
            m = path_pattern.search(text)
            if m:
                candidate = m.group(0).rstrip(".,;:)")
                # Filter out obviously wrong matches
                if any(candidate.endswith(ext) for ext in
                       [".py", ".txt", ".md", ".js", ".html", ".css",
                        ".json", ".yaml", ".toml", ".csv", ".pdf", ".docx"]):
                    return candidate
        return None

    async def _take_screenshot(self) -> str:
        """Take a screenshot using the VisionInterface. Returns path or ''."""
        try:
            if self._vision is None:
                return ""
            result = await self._vision.capture_screenshot()
            return getattr(result, "path", "") or ""
        except Exception as exc:  # noqa: BLE001
            log.debug("[AGENT] Observer: screenshot failed (non-fatal): {exc}", exc=exc)
            return ""

    @staticmethod
    def _extract_browser_url(response_text: str) -> str:
        """Extract URL from response text."""
        import re
        m = re.search(r'https?://[^\s\'"<>]+', response_text)
        if m:
            return m.group(0).rstrip(".,;)")
        return ""

    @staticmethod
    def _consolidate(
        text_signal: bool | None,
        app_name: str | None,
        app_running: bool | None,
        file_path: str | None,
        file_exists: bool | None,
        screenshot_path: str,
        browser_url: str = "",
        browser_verified: bool | None = None,
        browser_reason: str = "",
        is_browser_task: bool = False,
    ) -> tuple[bool | None, str, str]:
        """
        Combine all observation signals into a single verdict.

        Returns (success_signal, method, summary).
        Strongest available signal wins; ties resolved conservatively.
        """
        parts: list[str] = []
        signals: list[bool] = []

        if browser_verified is not None:
            parts.append(browser_reason or f"Browser URL: {browser_url}")
            return browser_verified, "browser_url", browser_reason or f"Browser verified ({browser_url})"

        if is_browser_task and browser_verified is None:
            # Browser task executed but actual state unverified
            return None, "browser_url", browser_reason or "Browser state unverified directly"

        if app_running is not None:
            sig_str = "running" if app_running else "NOT running"
            parts.append(f"App '{app_name}' is {sig_str}")
            signals.append(app_running)

        if file_exists is not None:
            sig_str = "exists" if file_exists else "NOT found"
            parts.append(f"File '{file_path}' {sig_str}")
            signals.append(file_exists)

        if browser_url:
            parts.append(f"Browser URL: {browser_url}")

        if screenshot_path:
            parts.append(f"Screenshot captured: {screenshot_path}")

        # Determine method
        if signals:
            # Concrete environmental evidence overrides text analysis
            success = all(signals)  # conservative: all must be True
            method = "app_state" if app_running is not None else "file_check"
            if parts:
                summary = "; ".join(parts)
            else:
                summary = "Environmental check performed."
            return success, method, summary

        # Fall back to response text analysis
        if text_signal is not None:
            return (
                text_signal,
                "response_text",
                "Signal from Brain response text.",
            )

        return None, "response_text", "Inconclusive — no strong signal."

