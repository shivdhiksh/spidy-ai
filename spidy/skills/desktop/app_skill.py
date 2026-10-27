"""
AppSkill — Windows Application Agent
=====================================
Provides application management capabilities: detect, launch, focus, close.

Permission Tiers
----------------
T0 — Read-only (detect running apps)
T1 — Non-destructive OS actions (launch, bring to foreground)
T2 — Destructive process termination (close_app)

Actions
-------
detect_running_apps      — List currently running processes          (T0)
launch_app               — Start an installed application            (T1)
bring_app_to_foreground  — Focus a running application window        (T1)
close_app                — Terminate an application process          (T2)

Architecture
------------
- Process detection uses psutil (cross-platform, already required).
- Launching uses subprocess.Popen() (PATH + fallback search).
- Window focus uses ctypes.windll.user32 (Windows-only, guarded).
- Process termination uses psutil .terminate() → .kill() (graceful first).
- All results are published to the EventBus after execution.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

from spidy.logging.logger import get_logger
from spidy.skills.desktop.app_catalog import (
    StartApp,
    find_start_app,
    launch_start_app,
    normalize_app_name,
)
from spidy.skills.base import (
    BaseSkill,
    ParamSchema,
    SkillCapability,
    SkillContext,
    SkillResult,
)

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)

# Common Windows app aliases: spoken name → executable
#
# Cross-platform OS aliases are included so macOS / Linux vocabulary
# (e.g. "Finder", "Trash", "Terminal", "Applications") is transparently
# mapped to the Windows equivalent without confusing the user.
_APP_ALIASES: dict[str, str] = {
    # ── Core Windows apps ────────────────────────────────────────────────
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "paint": "mspaint.exe",
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "firefox": "firefox.exe",
    "edge": "msedge.exe",
    "microsoft edge": "msedge.exe",
    "word": "WINWORD.EXE",
    "excel": "EXCEL.EXE",
    "powerpoint": "POWERPNT.EXE",
    "outlook": "OUTLOOK.EXE",
    "explorer": "explorer.exe",
    "task manager": "taskmgr.exe",
    "control panel": "control.exe",
    "command prompt": "cmd.exe",
    "cmd": "cmd.exe",
    "powershell": "powershell.exe",
    "terminal": "wt.exe",
    "windows terminal": "wt.exe",
    "vs code": "code.exe",
    "vscode": "code.exe",
    "visual studio code": "code.exe",
    "spotify": "Spotify.exe",
    "discord": "Discord.exe",
    "slack": "slack.exe",
    "teams": "Teams.exe",
    "microsoft teams": "Teams.exe",
    "zoom": "Zoom.exe",
    "vlc": "vlc.exe",
    "snipping tool": "SnippingTool.exe",

    # ── macOS → Windows cross-platform aliases ───────────────────────────
    # macOS "Finder" is the file manager → Windows File Explorer
    "finder": "explorer.exe",
    "mac finder": "explorer.exe",
    # macOS "Applications" folder → Windows "Programs" view in Explorer
    "applications": "explorer.exe",
    "apps folder": "explorer.exe",
    # macOS / Linux generic file manager aliases
    "file manager": "explorer.exe",
    "file browser": "explorer.exe",
    "files": "explorer.exe",   # GNOME Files / elementary Files
    "nautilus": "explorer.exe",
    "file explorer": "explorer.exe",
    "windows explorer": "explorer.exe",
    # macOS Terminal / iTerm2 → Windows Terminal (with cmd.exe fallback)
    "iterm": "wt.exe",
    "iterm2": "wt.exe",
    "bash": "wt.exe",
    "zsh": "wt.exe",
    "shell": "wt.exe",
    "console": "wt.exe",
    # Linux / macOS Trash → Windows Recycle Bin shell
    # (Note: emptying Trash/Recycle Bin is handled by SystemControlSkill;
    #  opening the Recycle Bin view uses the explorer shell command below.)
    "trash": "explorer.exe",
    "recycle bin": "explorer.exe",
    "bin": "explorer.exe",
}

# Fallback full paths for apps not in PATH (e.g. Chrome installed to Program Files)
_KNOWN_PATHS: dict[str, list[str]] = {
    "chrome.exe": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
    ],
    "msedge.exe": [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ],
    "firefox.exe": [
        r"C:\Program Files\Mozilla Firefox\firefox.exe",
        r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
    ],
    "code.exe": [
        r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe",
        r"C:\Program Files\Microsoft VS Code\Code.exe",
    ],
    "spotify.exe": [
        r"%APPDATA%\Spotify\Spotify.exe",
    ],
}


_BROWSER_PROCESSES = frozenset({
    "chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe", "vivaldi.exe",
})


class AppSkill(BaseSkill):
    """
    Application management agent for Windows.

    Parameters
    ----------
    bus:
        EventBus for publishing desktop events. May be None.
    """

    name = "app_skill"
    version = "1.0.0"

    def __init__(self, bus: "EventBus | None" = None) -> None:
        self._bus = bus

    # ── Capabilities ──────────────────────────────────────────────────────

    def capabilities(self) -> list[SkillCapability]:
        return [
            SkillCapability(
                action="detect_running_apps",
                description=(
                    "List currently running applications and processes with their PIDs."
                ),
                permission_tier="T0",
                params=[
                    ParamSchema("filter", "string", required=False,
                                description="Optional name filter to narrow results."),
                    ParamSchema("limit", "int", required=False, default=20,
                                description="Maximum number of processes to return."),
                ],
                examples=[
                    "what apps are running", "list running processes",
                    "is Chrome running", "show me open applications",
                    "check if Spotify is open",
                ],
            ),
            SkillCapability(
                action="launch_app",
                description="Launch an installed application by name or path.",
                permission_tier="T1",
                params=[
                    ParamSchema("name", "string", required=True,
                                description="Application name (e.g. 'notepad', 'chrome') or full path."),
                    ParamSchema("args", "string", required=False,
                                description="Space-separated arguments to pass to the app."),
                ],
                examples=[
                    "open notepad", "launch chrome", "start calculator",
                    "open Microsoft Word", "run powershell",
                ],
            ),
            SkillCapability(
                action="bring_app_to_foreground",
                description="Bring a running application to the foreground (give it focus).",
                permission_tier="T1",
                params=[
                    ParamSchema("name", "string", required=True,
                                description="Application name or window title to bring forward."),
                ],
                examples=[
                    "switch to Chrome", "focus notepad", "bring VS Code to front",
                    "show the explorer window", "switch to terminal",
                ],
            ),
            SkillCapability(
                action="minimize_window",
                description="Minimize an application window to the taskbar.",
                permission_tier="T1",
                params=[
                    ParamSchema("name", "string", required=True,
                                description="Application name or window title to minimize."),
                ],
                examples=[
                    "minimize notepad", "minimize edge", "minimize chrome",
                    "minimise vs code", "minimize calculator",
                ],
            ),
            SkillCapability(
                action="maximize_window",
                description="Maximize (or restore) an application window.",
                permission_tier="T1",
                params=[
                    ParamSchema("name", "string", required=True,
                                description="Application name or window title to maximize."),
                ],
                examples=[
                    "maximize edge", "maximise chrome", "restore notepad",
                    "maximize vs code", "maximise calculator",
                ],
            ),
            SkillCapability(
                action="close_app",
                description="Close (terminate) a running application.",
                permission_tier="T2",
                params=[
                    ParamSchema("name", "string", required=True,
                                description="Application name or process name to close."),
                    ParamSchema("force", "bool", required=False, default=False,
                                description="If true, force-kill the process without grace period."),
                ],
                examples=[
                    "close notepad", "kill Chrome", "force close the app",
                    "terminate spotify", "close all Chrome windows",
                ],
            ),
            SkillCapability(
                action="run_python_script",
                description="Execute a local Python script and return its stdout/stderr output.",
                permission_tier="T1",
                params=[
                    ParamSchema("path", "string", required=True,
                                description="Path to the Python script (e.g. Desktop/area.py or full path)."),
                    ParamSchema("args", "string", required=False, default="",
                                description="Optional space-separated arguments to pass to the script."),
                    ParamSchema("timeout", "int", required=False, default=15,
                                description="Execution timeout in seconds (default 15, max 60)."),
                ],
                examples=[
                    "run python script", "run the script on my desktop", "execute area.py",
                ],
            ),
            SkillCapability(
                action="execute_script",
                description="Execute a specific command or script with safety isolation.",
                permission_tier="T1",
                params=[
                    ParamSchema("command", "string", required=True,
                                description="Command to execute (e.g. 'python script.py')."),
                    ParamSchema("timeout", "int", required=False, default=15,
                                description="Execution timeout in seconds (default 15, max 60)."),
                ],
                examples=[
                    "run command", "execute command",
                ],
            ),
        ]

    # ── Dispatch ──────────────────────────────────────────────────────────

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        if action == "detect_running_apps":
            return await self._detect_running_apps(context)
        if action == "launch_app":
            return await self._launch_app(context)
        if action == "bring_app_to_foreground":
            return await self._bring_to_foreground(context)
        if action == "minimize_window":
            return await self._minimize_window(context)
        if action == "maximize_window":
            return await self._maximize_window(context)
        if action == "close_app":
            return await self._close_app(context)
        if action == "run_python_script":
            return await self._run_python_script(context)
        if action == "execute_script":
            return await self._execute_script(context)
        return SkillResult.fail(f"AppSkill: unknown action '{action}'.")


    # ── Actions ───────────────────────────────────────────────────────────

    async def _detect_running_apps(self, context: SkillContext) -> SkillResult:
        name_filter: str = (context.get("filter") or "").lower()
        limit = int(context.get("limit") or 20)
        limit = max(1, min(limit, 200))

        apps = await asyncio.to_thread(self._get_running_processes, name_filter, limit)

        await self._emit_running_apps(apps, context.session_id)

        if not apps:
            msg = (
                f"No running processes matching '{name_filter}'."
                if name_filter
                else "No running processes found."
            )
            return SkillResult.ok(msg, data={"apps": []}, action_taken="detect_running_apps")

        lines = [f"  {a['name']} (PID {a['pid']})" for a in apps[:20]]
        suffix = f"\n… and {len(apps) - 20} more." if len(apps) > 20 else ""
        return SkillResult.ok(
            f"Running applications ({len(apps)}):\n" + "\n".join(lines) + suffix,
            data={"apps": apps},
            action_taken="detect_running_apps",
        )

    async def _launch_app(self, context: SkillContext) -> SkillResult:
        name: str = context.get("name", "")
        if not name:
            return SkillResult.fail(
                "AppSkill: 'name' parameter is required for launch_app.",
                non_retryable=True,
            )

        args_raw = context.get("args", "") or ""
        if isinstance(args_raw, (list, tuple)):
            args = [str(arg) for arg in args_raw]
        else:
            # Windows paths routinely contain spaces; use Windows-aware
            # splitting rather than turning one file argument into many.
            import shlex
            args = shlex.split(str(args_raw), posix=False) if args_raw else []

        # App launches are idempotent when the caller only asks to open an
        # application.  VS Code titles use "Visual Studio Code" while its
        # executable is Code.exe, so this must use the same alias-aware window
        # lookup as readiness/focus rather than blindly starting a new process.
        # Supplying args is intentionally different: ``code <file>`` is a
        # request to open that specific file in the existing editor instance.
        if not args:
            existing_hwnd, existing_name = await asyncio.to_thread(
                self._find_window_by_name, name
            )
            if existing_hwnd:
                focused = await asyncio.to_thread(self._set_foreground, existing_hwnd)
                if not focused:
                    return SkillResult.fail(
                        f"AppSkill: '{existing_name or name}' is already running, but Windows would not focus it."
                    )
                await self._emit_app_foregrounded(existing_name or name, existing_hwnd, context.session_id)
                return SkillResult.ok(
                    f"'{existing_name or name}' is already running and focused.",
                    data={
                        "app_name": name,
                        "hwnd": existing_hwnd,
                        "ready": True,
                        "already_running": True,
                        "verified": True,
                    },
                    action_taken="launch_app",
                )

        # Spoken names are often mis-heard ("VS Core"); look up the cleaned-up name.
        spoken = normalize_app_name(name) or name.lower()

        # How to start it, best first: known executable → Start menu (covers Store
        # apps such as WhatsApp) → whatever is on PATH.
        proc: subprocess.Popen | None = None
        start_app: StartApp | None = None
        display = spoken  # cleaned-up name ("VS Core" is reported as "vs code")
        alias_exe = _APP_ALIASES.get(spoken) or _APP_ALIASES.get(name.lower())
        if alias_exe:
            exe_path = self._resolve_exe_path(alias_exe)
            if os.path.isabs(exe_path) or shutil.which(exe_path):
                try:
                    proc = await asyncio.to_thread(
                        subprocess.Popen, [exe_path] + args, shell=False,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("AppSkill: '{exe}' failed to start ({exc}); trying Start menu.",
                                exe=exe_path, exc=exc)
        if proc is None and not args:
            start_app = await asyncio.to_thread(find_start_app, spoken)
            if start_app is not None:
                display = start_app.name
                try:
                    proc = await asyncio.to_thread(launch_start_app, start_app)
                except Exception as exc:  # noqa: BLE001
                    return SkillResult.fail(
                        f"AppSkill: failed to start '{display}': {exc}", error=exc,
                    )
        if proc is None:
            try:
                proc = await asyncio.to_thread(
                    subprocess.Popen, [self._resolve_exe_path(alias_exe or name)] + args,
                    shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                return SkillResult.fail(
                    f"I couldn't find an app called '{name}' on this computer.",
                    non_retryable=True,
                )
            except Exception as exc:  # noqa: BLE001
                return SkillResult.fail(
                    f"AppSkill: failed to launch '{name}': {exc}", error=exc
                )

        # Quick exit check to catch immediate launch failures
        await asyncio.sleep(0.05)
        exit_code = proc.poll()
        if exit_code is not None and exit_code != 0 and start_app is None:
            return SkillResult.fail(
                f"AppSkill: '{name}' exited immediately with code {exit_code}. "
                "Check that the application is installed correctly."
            )

        pid = proc.pid
        log.info("AppSkill: launched '{name}' as PID {pid}", name=display, pid=pid)

        # Only claim success once a window really appeared (Electron/Store apps can
        # take several seconds). "Started" is not "opened".
        ready, hwnd = await self._wait_for_any_window(
            [name, spoken, display], timeout=15.0,
        )
        if not ready:
            log.warning("AppSkill: '{name}' started but no window appeared.", name=display)
            return SkillResult.fail(
                f"I started {display}, but no window appeared, so I can't confirm it opened.",
                non_retryable=True,
            )

        log.info("AppSkill: '{name}' window ready (hwnd={hwnd})", name=display, hwnd=hwnd)
        await self._emit_app_foregrounded(display, hwnd, context.session_id)
        await self._emit_app_launched(display, pid, context.session_id)
        return SkillResult.ok(
            f"Opened {display}.",
            data={"app_name": display, "pid": pid, "hwnd": hwnd, "ready": True, "verified": True},
            action_taken="launch_app",
        )

    async def _wait_for_any_window(
        self,
        names: list[str],
        timeout: float = 12.0,
        poll_interval: float = 0.25,
    ) -> tuple[bool, int]:
        """Poll until a visible window matching any of ``names`` exists, then focus it."""
        import time as _time
        candidates = [n for n in dict.fromkeys(names) if n]
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            for candidate in candidates:
                hwnd, _ = await asyncio.to_thread(self._find_window_by_name, candidate)
                if hwnd:
                    await asyncio.to_thread(self._set_foreground, hwnd)
                    return True, hwnd
            await asyncio.sleep(poll_interval)
        return False, 0

    async def _wait_for_app_ready(
        self,
        name: str,
        pid: int,
        timeout: float = 2.5,
        poll_interval: float = 0.05,
    ) -> tuple[bool, int]:
        """
        Poll boundedly until the application window is found and focused.
        """
        import time as _time
        deadline = _time.monotonic() + timeout
        hwnd = 0
        while _time.monotonic() < deadline:
            hwnd, _ = await asyncio.to_thread(self._find_window_by_name, name)
            if hwnd != 0:
                await asyncio.to_thread(self._set_foreground, hwnd)
                return True, hwnd
            await asyncio.sleep(poll_interval)
        return False, hwnd

    async def _bring_to_foreground(self, context: SkillContext) -> SkillResult:
        name: str = context.get("name", "")
        if not name:
            return SkillResult.fail(
                "AppSkill: 'name' parameter is required for bring_app_to_foreground.",
                non_retryable=True,
            )

        hwnd, matched_name = await asyncio.to_thread(self._find_window_by_name, name)

        if hwnd == 0:
            return SkillResult.fail(
                f"AppSkill: no running window found matching '{name}'. "
                "Is the application running?"
            )

        success = await asyncio.to_thread(self._set_foreground, hwnd)
        if not success:
            return SkillResult.fail(
                f"AppSkill: could not bring '{matched_name}' to foreground "
                "(may be blocked by Windows focus policy)."
            )

        await self._emit_app_foregrounded(matched_name, hwnd, context.session_id)
        return SkillResult.ok(
            f"Brought '{matched_name}' to the foreground.",
            data={"app_name": matched_name, "hwnd": hwnd},
            action_taken="bring_app_to_foreground",
        )

    async def _close_app(self, context: SkillContext) -> SkillResult:
        name: str = context.get("name", "")
        if not name:
            return SkillResult.fail(
                "AppSkill: 'name' parameter is required for close_app.",
                non_retryable=True,
            )

        force_raw = context.get("force", False)
        if isinstance(force_raw, str):
            force = force_raw.lower() not in ("false", "0", "no", "")
        else:
            force = bool(force_raw)

        processes = await asyncio.to_thread(self._find_processes_by_name, name)
        if not processes:
            return SkillResult.fail(
                f"AppSkill: no running process found matching '{name}'."
            )

        closed: list[dict] = []
        errors: list[str] = []

        for proc_info in processes:
            try:
                import psutil
                proc = psutil.Process(proc_info["pid"])
                if force:
                    await asyncio.to_thread(proc.kill)
                    log.info("AppSkill: force-killed '{n}' (PID {p})",
                             n=proc_info["name"], p=proc_info["pid"])
                else:
                    await asyncio.to_thread(proc.terminate)
                    log.info("AppSkill: terminated '{n}' (PID {p})",
                             n=proc_info["name"], p=proc_info["pid"])
                closed.append(proc_info)
                await self._emit_app_closed(proc_info["name"], proc_info["pid"],
                                            forced=force, session_id=context.session_id)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"PID {proc_info['pid']}: {exc}")

        if not closed:
            return SkillResult.fail(
                f"AppSkill: could not close '{name}': " + "; ".join(errors)
            )

        names = ", ".join(f"'{p['name']}' (PID {p['pid']})" for p in closed)
        action_word = "Force-killed" if force else "Closed"
        return SkillResult.ok(
            f"{action_word} {names}.",
            data={"closed": closed, "errors": errors},
            action_taken="close_app",
        )

    async def _minimize_window(self, context: SkillContext) -> SkillResult:
        """
        Minimize an application window to the taskbar.

        BUG 3 FIX: Previously there was no minimize action; the intent fell
        through to chat and produced no window operation.
        """
        name: str = context.get("name", "")
        if not name:
            return SkillResult.fail(
                "AppSkill: 'name' parameter is required for minimize_window.",
                non_retryable=True,
            )

        hwnd, matched = await asyncio.to_thread(self._find_window_by_name, name)
        if not hwnd:
            return SkillResult.fail(
                f"AppSkill: no visible window found for '{name}'. "
                "The application may not be running or may have no visible window."
            )

        try:
            import win32gui  # type: ignore[import-untyped]
            import win32con  # type: ignore[import-untyped]
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            log.info("Minimized window '{name}' (hwnd={hwnd})", name=matched, hwnd=hwnd)
            return SkillResult.ok(
                f"Minimized '{matched}'.",
                data={"hwnd": hwnd, "name": matched},
                action_taken="minimize_window",
            )
        except ImportError:
            # Fallback via ctypes if win32con not available
            try:
                import ctypes
                SW_MINIMIZE = 6
                ctypes.windll.user32.ShowWindow(hwnd, SW_MINIMIZE)
                log.info("Minimized window '{name}' via ctypes", name=matched)
                return SkillResult.ok(
                    f"Minimized '{matched}'.",
                    data={"hwnd": hwnd, "name": matched},
                    action_taken="minimize_window",
                )
            except Exception as exc:  # noqa: BLE001
                return SkillResult.fail(f"AppSkill: minimize failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(f"AppSkill: minimize failed: {exc}")

    async def _maximize_window(self, context: SkillContext) -> SkillResult:
        """
        Maximize (or restore) an application window.

        BUG 3 FIX: Previously there was no maximize action.
        Uses SW_MAXIMIZE to maximize or SW_RESTORE if already maximized.
        """
        name: str = context.get("name", "")
        if not name:
            return SkillResult.fail(
                "AppSkill: 'name' parameter is required for maximize_window.",
                non_retryable=True,
            )

        hwnd, matched = await asyncio.to_thread(self._find_window_by_name, name)
        if not hwnd:
            return SkillResult.fail(
                f"AppSkill: no visible window found for '{name}'. "
                "The application may not be running or may have no visible window."
            )

        try:
            import win32gui  # type: ignore[import-untyped]
            import win32con  # type: ignore[import-untyped]
            # SW_MAXIMIZE = 3; SW_RESTORE = 9 (restores if minimized first)
            placement = win32gui.GetWindowPlacement(hwnd)
            if placement[1] == win32con.SW_SHOWMAXIMIZED:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                verb = "Restored"
            else:
                win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
                verb = "Maximized"
            log.info("{verb} window '{name}' (hwnd={hwnd})", verb=verb, name=matched, hwnd=hwnd)
            return SkillResult.ok(
                f"{verb} '{matched}'.",
                data={"hwnd": hwnd, "name": matched},
                action_taken="maximize_window",
            )
        except ImportError:
            try:
                import ctypes
                SW_MAXIMIZE = 3
                ctypes.windll.user32.ShowWindow(hwnd, SW_MAXIMIZE)
                log.info("Maximized window '{name}' via ctypes", name=matched)
                return SkillResult.ok(
                    f"Maximized '{matched}'.",
                    data={"hwnd": hwnd, "name": matched},
                    action_taken="maximize_window",
                )
            except Exception as exc:  # noqa: BLE001
                return SkillResult.fail(f"AppSkill: maximize failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(f"AppSkill: maximize failed: {exc}")

    async def _run_python_script(self, context: SkillContext) -> SkillResult:
        """
        Execute a Python script safely with timeout and output capture.
        """
        import os
        import sys
        from pathlib import Path

        path_str = context.get("path") or context.get("script_path") or context.get("target") or ""
        if not path_str:
            return SkillResult.fail(
                "AppSkill: 'path' parameter is required for run_python_script.",
                non_retryable=True,
            )

        timeout = int(context.get("timeout") or 15)
        timeout = max(1, min(timeout, 60))
        extra_args_str = context.get("args") or ""

        # Resolve candidate paths
        candidate_paths = [
            Path(path_str).expanduser().resolve(),
            Path.home() / "Desktop" / Path(path_str).name,
            Path.home() / "OneDrive" / "Desktop" / Path(path_str).name,
            Path.cwd() / path_str,
        ]

        resolved_path = None
        for p in candidate_paths:
            if p.is_file():
                resolved_path = p
                break

        if resolved_path is None:
            # Try finding on disk
            if os.path.exists(path_str):
                resolved_path = Path(path_str).resolve()

        if resolved_path is None:
            return SkillResult.fail(
                f"AppSkill: Python script not found at '{path_str}'.",
                non_retryable=True,
            )

        cmd = [sys.executable, str(resolved_path)]
        if extra_args_str:
            import shlex
            cmd.extend(shlex.split(extra_args_str))

        log.info("AppSkill: executing python script '{p}' with timeout {t}s", p=resolved_path, t=timeout)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=float(timeout),
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
                return SkillResult.fail(
                    f"AppSkill: Script execution timed out after {timeout} seconds.",
                    non_retryable=True,
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            returncode = proc.returncode

            if returncode == 0:
                out_msg = f"Script completed successfully (exit 0).\nOutput:\n{stdout}" if stdout else "Script completed with no output (exit 0)."
                return SkillResult.ok(
                    out_msg,
                    data={"stdout": stdout, "stderr": stderr, "returncode": returncode},
                    action_taken="run_python_script",
                )
            else:
                err_msg = f"Script exited with code {returncode}.\nError:\n{stderr or stdout}"
                return SkillResult.fail(
                    err_msg,
                    data={"stdout": stdout, "stderr": stderr, "returncode": returncode},
                )

        except Exception as exc:  # noqa: BLE001
            log.error("AppSkill._run_python_script error: {exc}", exc=exc)
            return SkillResult.fail(f"AppSkill: Failed to execute script: {exc}")

    async def _execute_script(self, context: SkillContext) -> SkillResult:
        """Execute a general script or command with timeout."""
        cmd_str = context.get("command") or ""
        if not cmd_str:
            return SkillResult.fail("AppSkill: 'command' parameter is required for execute_script.")

        import shlex
        import sys
        timeout = int(context.get("timeout") or 15)
        timeout = max(1, min(timeout, 60))

        args = shlex.split(cmd_str)
        if not args:
            return SkillResult.fail("AppSkill: Invalid command.")

        # If it's a python command, use current interpreter
        if args[0].lower() in ("python", "python3", "py"):
            args[0] = sys.executable

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=float(timeout),
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            returncode = proc.returncode

            if returncode == 0:
                return SkillResult.ok(
                    f"Command completed successfully.\nOutput:\n{stdout}",
                    data={"stdout": stdout, "stderr": stderr, "returncode": returncode},
                    action_taken="execute_script",
                )
            return SkillResult.fail(
                f"Command exited with code {returncode}.\nError:\n{stderr or stdout}",
                data={"stdout": stdout, "stderr": stderr, "returncode": returncode},
            )
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(f"AppSkill: execute_script error: {exc}")

    # ── Internal helpers ──────────────────────────────────────────────────


    @staticmethod
    def _resolve_exe_path(exe_name: str) -> str:
        """
        Resolve an executable name to a full path.

        Tries (in order):
        1. The name as-is (works when it is already a full path or is on PATH).
        2. Known install paths from _KNOWN_PATHS (with env-var expansion).
        3. Special fallback: wt.exe → cmd.exe when Windows Terminal is absent.

        Returns the first path that exists on disk, or the original name if
        nothing is found (Popen will raise FileNotFoundError as normal).
        """
        import os, shutil
        # Already a full path or on PATH
        if os.path.isabs(exe_name) or shutil.which(exe_name):
            return exe_name
        # Check known locations
        key = exe_name.lower()
        for candidate in _KNOWN_PATHS.get(key, []):
            expanded = os.path.expandvars(candidate)
            if os.path.isfile(expanded):
                return expanded
        # Special case: Windows Terminal not installed → fall back to cmd.exe
        if key == "wt.exe" and shutil.which("cmd.exe"):
            log.info(
                "AppSkill: wt.exe not found; falling back to cmd.exe"
            )
            return "cmd.exe"
        # Return original — Popen will raise FileNotFoundError if not found
        return exe_name

    @staticmethod
    def _get_running_processes(name_filter: str, limit: int) -> list[dict]:
        """Return running processes as a list of dicts with name, pid, status."""
        try:
            import psutil
        except ImportError:
            log.warning("AppSkill: psutil not installed. Cannot list processes.")
            return []

        results: list[dict] = []
        for proc in psutil.process_iter(["name", "pid", "status"]):
            try:
                pname = (proc.info.get("name") or "").lower()
                if name_filter and name_filter not in pname:
                    continue
                results.append({
                    "name": proc.info.get("name", ""),
                    "pid": proc.info.get("pid", 0),
                    "status": proc.info.get("status", ""),
                })
                if len(results) >= limit:
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return results

    @staticmethod
    def _find_processes_by_name(name: str) -> list[dict]:
        """Find all processes matching the given name (case-insensitive)."""
        try:
            import psutil
        except ImportError:
            return []

        name_lower = name.lower()
        # Check aliases
        resolved = _APP_ALIASES.get(name_lower, name_lower)
        exe_name = resolved.lower().replace(".exe", "")

        matches: list[dict] = []
        for proc in psutil.process_iter(["name", "pid"]):
            try:
                pname = (proc.info.get("name") or "").lower()
                if exe_name in pname or name_lower in pname:
                    matches.append({
                        "name": proc.info.get("name", ""),
                        "pid": proc.info.get("pid", 0),
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return matches

    @staticmethod
    def _find_window_by_name(name: str) -> tuple[int, str]:
        """
        Find an HWND for a window whose title or process name matches ``name``.
        Returns (hwnd, matched_name). hwnd=0 if not found.
        """
        try:
            import win32gui  # type: ignore[import-untyped]
            import win32process  # type: ignore[import-untyped]
            import psutil
        except ImportError:
            log.warning("AppSkill: win32gui/psutil not available for window search.")
            return 0, ""

        name_lower = name.lower().strip()
        resolved = _APP_ALIASES.get(name_lower, name_lower).lower()
        # Match both a spoken alias and the real Windows process/title.  The
        # old exact-string comparison could never find a running VS Code
        # window for the spoken name "VS Code".
        needles = {name_lower, resolved.removesuffix(".exe")}
        if name_lower in ("vs code", "vscode", "visual studio code"):
            needles.update({"visual studio code", "code"})
        needles.discard("")
        wants_browser = any(
            b in needle for needle in needles
            for b in ("chrome", "edge", "firefox", "brave", "opera", "browser")
        )
        found_hwnd = 0
        found_name = ""

        def _enum_callback(hwnd: int, _: Any) -> None:
            nonlocal found_hwnd, found_name
            if found_hwnd:
                return
            try:
                if not win32gui.IsWindowVisible(hwnd):
                    return
                title = win32gui.GetWindowText(hwnd).lower()
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                try:
                    proc_name = psutil.Process(pid).name().lower()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    proc_name = ""
                # A browser tab called "WhatsApp Web" is not the WhatsApp app: ignore
                # browser windows unless a browser itself was asked for.
                if proc_name in _BROWSER_PROCESSES and not wants_browser:
                    return
                if any(needle in title for needle in needles):
                    found_hwnd = hwnd
                    found_name = win32gui.GetWindowText(hwnd)
                    return
                # Also check process name
                try:
                    proc = psutil.Process(pid)
                    process_name = proc.name().lower()
                    if any(needle in process_name for needle in needles):
                        found_hwnd = hwnd
                        found_name = proc.name()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            except Exception:  # noqa: BLE001
                pass

        try:
            win32gui.EnumWindows(_enum_callback, None)
        except Exception:  # noqa: BLE001
            pass

        return found_hwnd, found_name

    @staticmethod
    def _set_foreground(hwnd: int) -> bool:
        """Bring the given window to the foreground. Returns True on success."""
        try:
            import ctypes
            # Attach to the foreground window's thread first (Windows focus policy)
            import win32gui  # type: ignore[import-untyped]
            win32gui.ShowWindow(hwnd, 9)  # SW_RESTORE = 9
            result = ctypes.windll.user32.SetForegroundWindow(hwnd)
            return bool(result)
        except Exception:  # noqa: BLE001
            return False

    # ── EventBus publishers ───────────────────────────────────────────────

    async def _emit_running_apps(self, apps: list[dict], session_id: str) -> None:
        if self._bus is None:
            return
        from spidy.skills.desktop.events import RunningAppsResultEvent
        await self._bus.publish(RunningAppsResultEvent(apps=apps, session_id=session_id))

    async def _emit_app_launched(self, name: str, pid: int, session_id: str) -> None:
        if self._bus is None:
            return
        from spidy.skills.desktop.events import AppLaunchedEvent
        await self._bus.publish(AppLaunchedEvent(app_name=name, pid=pid, session_id=session_id))

    async def _emit_app_foregrounded(self, name: str, hwnd: int, session_id: str) -> None:
        if self._bus is None:
            return
        from spidy.skills.desktop.events import AppForegroundedEvent
        await self._bus.publish(AppForegroundedEvent(
            app_name=name, hwnd=hwnd, session_id=session_id
        ))

    async def _emit_app_closed(
        self, name: str, pid: int, forced: bool, session_id: str
    ) -> None:
        if self._bus is None:
            return
        from spidy.skills.desktop.events import AppClosedEvent
        await self._bus.publish(AppClosedEvent(
            app_name=name, pid=pid, forced=forced, session_id=session_id
        ))
