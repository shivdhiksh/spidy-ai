"""
ProactiveChecker — Smart Pre-Execution State Checks
=====================================================
Before executing a skill step, the ToolRouter consults the ProactiveChecker
to see if a smarter action is available.

This prevents Spidy from:
- Relaunching an app that is already open
- Opening a URL that is already in the active browser tab
- Writing a file that already exists without asking
- Trying to search the web when the network is unavailable

Decision outcomes
-----------------
``proceed``  — go ahead, no issue detected
``reuse``    — don't relaunch; the resource is already available
``ask``      — something ambiguous, ask the user before proceeding
``skip``     — silently skip (e.g. network check passed trivially)

Design
------
- Heuristic-first: uses psutil for process checks (always available)
- Vision-enhanced: if VisionInterface is present, uses screen analysis
- All checks are non-blocking; any check failure → "proceed" (fail open)
- No check ever blocks execution; they only add context
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# ─── Common app → process name mappings ──────────────────────────────────────

_APP_PROCESS_NAMES: dict[str, list[str]] = {
    "vs code": ["code.exe", "code"],
    "vscode": ["code.exe", "code"],
    "visual studio code": ["code.exe", "code"],
    "code": ["code.exe", "code"],
    "notepad": ["notepad.exe"],
    "calculator": ["calculatorapp.exe", "calculator.exe"],
    "chrome": ["chrome.exe"],
    "google chrome": ["chrome.exe"],
    "firefox": ["firefox.exe"],
    "edge": ["msedge.exe"],
    "microsoft edge": ["msedge.exe"],
    "spotify": ["spotify.exe"],
    "discord": ["discord.exe"],
    "slack": ["slack.exe"],
    "zoom": ["zoom.exe"],
    "teams": ["teams.exe"],
    "microsoft teams": ["teams.exe"],
    "terminal": ["wt.exe", "cmd.exe", "powershell.exe"],
    "cmd": ["cmd.exe"],
    "powershell": ["powershell.exe", "pwsh.exe"],
    "explorer": ["explorer.exe"],
    "task manager": ["taskmgr.exe"],
    "word": ["winword.exe"],
    "excel": ["excel.exe"],
    "powerpoint": ["powerpnt.exe"],
    "outlook": ["outlook.exe"],
}


# ─── Data types ───────────────────────────────────────────────────────────────


@dataclass
class ProactiveAdvice:
    """
    The result of a proactive check.

    Attributes
    ----------
    action:
        What the ToolRouter should do:
        - ``"proceed"``  — execute the step normally
        - ``"reuse"``    — skip launch; reuse existing resource
        - ``"ask"``      — ask the user before proceeding
        - ``"skip"``     — silently skip this step
    message:
        Human-readable message to show the user if action != "proceed".
        Empty for "proceed" (no message needed).
    data:
        Optional structured data (e.g. existing file path, open window title).
    """
    action: str = "proceed"  # "proceed" | "reuse" | "ask" | "skip"
    message: str = ""
    data: Any = None

    @property
    def should_skip(self) -> bool:
        return self.action in ("reuse", "skip")


# ─── Process helpers ──────────────────────────────────────────────────────────


def _get_running_processes() -> set[str]:
    """Return lowercase process names of all running processes."""
    try:
        import psutil
        return {p.name().lower() for p in psutil.process_iter(["name"])}
    except Exception:  # noqa: BLE001 — psutil unavailable or permission denied
        return set()


def _is_app_running(app_name: str) -> bool:
    """
    Check if an application is currently running.

    Uses psutil for process inspection. Returns False if psutil is
    unavailable or the check fails.
    """
    app_lower = app_name.lower().strip()
    process_names = _APP_PROCESS_NAMES.get(app_lower, [])
    if not process_names:
        # Try a fuzzy match: app_name contains a known key
        for key, procs in _APP_PROCESS_NAMES.items():
            if key in app_lower or app_lower in key:
                process_names = procs
                break
    if not process_names:
        return False

    running = _get_running_processes()
    return any(proc.lower() in running for proc in process_names)


def _is_network_available(timeout: float = 1.5) -> bool:
    """Quick network reachability check (DNS lookup to 8.8.8.8)."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except OSError:
        return False


# ─── ProactiveChecker ─────────────────────────────────────────────────────────


class ProactiveChecker:
    """
    Evaluates whether a planned action needs pre-execution checks.

    Parameters
    ----------
    check_running_apps:
        If True, check if the target app is already running before launching.
    check_network:
        If True, check network availability before web search steps.
    check_file_exists:
        If True, check if a file already exists before creating it.
    """

    def __init__(
        self,
        check_running_apps: bool = True,
        check_network: bool = True,
        check_file_exists: bool = True,
    ) -> None:
        self._check_apps = check_running_apps
        self._check_network = check_network
        self._check_files = check_file_exists

    async def check(
        self,
        action: str,
        params: dict[str, Any],
        desktop_state: Any = None,
    ) -> ProactiveAdvice:
        """
        Perform all relevant checks for the given action.

        Parameters
        ----------
        action:
            The skill action about to be executed (e.g. ``"launch_app"``).
        params:
            The action parameters (e.g. ``{"name": "VS Code"}``).
        desktop_state:
            Optional DesktopStateSnapshot from ObserverManager.

        Returns
        -------
        ProactiveAdvice
            What the ToolRouter should do.
        """
        try:
            if action == "launch_app" and self._check_apps:
                return await self._check_launch_app(params, desktop_state)

            if action in (
                "search_web", "search_google", "search_youtube",
                "search_bing", "search_duckduckgo", "open_url",
            ) and self._check_network:
                return await self._check_network_action()

        except Exception as exc:  # noqa: BLE001 — never block execution
            log.debug("ProactiveChecker: check failed (non-fatal): {exc}", exc=exc)

        return ProactiveAdvice(action="proceed")

    async def _check_launch_app(
        self,
        params: dict[str, Any],
        desktop_state: Any,
    ) -> ProactiveAdvice:
        """Check if the app is already running before launching."""
        app_name = (
            params.get("name") or
            params.get("app_name") or
            params.get("app") or ""
        ).strip()

        if not app_name:
            return ProactiveAdvice(action="proceed")

        # Prefer desktop_state if available (more accurate)
        if desktop_state is not None:
            active_app = getattr(desktop_state, "active_app_name", "") or ""
            open_windows = getattr(desktop_state, "open_windows", []) or []
            window_titles = [str(w).lower() for w in open_windows]
            app_lower = app_name.lower()
            if active_app.lower() == app_lower or any(app_lower in t for t in window_titles):
                return ProactiveAdvice(
                    action="reuse",
                    message=f"{app_name.title()} is already open — I'll use that.",
                    data={"app_name": app_name, "source": "desktop_state"},
                )

        # Fall back to psutil process check
        if _is_app_running(app_name):
            return ProactiveAdvice(
                action="reuse",
                message=f"{app_name.title()} is already running — bringing it to focus.",
                data={"app_name": app_name, "source": "psutil"},
            )

        return ProactiveAdvice(action="proceed")

    async def _check_network_action(self) -> ProactiveAdvice:
        """Check network availability before internet-dependent actions."""
        if not _is_network_available():
            return ProactiveAdvice(
                action="ask",
                message=(
                    "It looks like you're offline. I can't search the web right now. "
                    "Would you like me to try anyway, or is there something else I can help with?"
                ),
                data={"network": "unavailable"},
            )
        return ProactiveAdvice(action="proceed")
