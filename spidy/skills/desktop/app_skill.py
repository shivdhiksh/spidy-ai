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
import subprocess
from typing import TYPE_CHECKING, Any

from spidy.logging.logger import get_logger
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
        ]

    # ── Dispatch ──────────────────────────────────────────────────────────

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        if action == "detect_running_apps":
            return await self._detect_running_apps(context)
        if action == "launch_app":
            return await self._launch_app(context)
        if action == "bring_app_to_foreground":
            return await self._bring_to_foreground(context)
        if action == "close_app":
            return await self._close_app(context)
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
            return SkillResult.fail("AppSkill: 'name' parameter is required for launch_app.")

        args_str: str = context.get("args", "") or ""
        args: list[str] = args_str.split() if args_str else []

        # Resolve alias
        resolved = _APP_ALIASES.get(name.lower(), name)

        # Expand known install paths for apps not on PATH
        exe_path = self._resolve_exe_path(resolved)
        cmd = [exe_path] + args

        proc: subprocess.Popen | None = None
        try:
            proc = await asyncio.to_thread(
                subprocess.Popen,
                cmd,
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return SkillResult.fail(
                f"AppSkill: could not launch '{name}' — not found in PATH or known install locations.",
            )
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(
                f"AppSkill: failed to launch '{name}': {exc}", error=exc
            )

        # Verify the process is still alive after a short settle period.
        # Without this, shell-spawned wrappers report success even when the
        # real executable was never found (e.g. chrome.exe not in PATH).
        await asyncio.sleep(0.35)
        exit_code = proc.poll()
        if exit_code is not None:
            return SkillResult.fail(
                f"AppSkill: '{name}' exited immediately with code {exit_code}. "
                "Check that the application is installed correctly."
            )

        pid = proc.pid
        log.info("AppSkill: launched '{name}' as PID {pid}", name=resolved, pid=pid)
        await self._emit_app_launched(name, pid, context.session_id)
        return SkillResult.ok(
            f"Launched '{name}' (PID {pid}).",
            data={"app_name": name, "pid": pid},
            action_taken="launch_app",
        )

    async def _bring_to_foreground(self, context: SkillContext) -> SkillResult:
        name: str = context.get("name", "")
        if not name:
            return SkillResult.fail(
                "AppSkill: 'name' parameter is required for bring_app_to_foreground."
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
            return SkillResult.fail("AppSkill: 'name' parameter is required for close_app.")

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

        name_lower = name.lower()
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
                if name_lower in title:
                    found_hwnd = hwnd
                    found_name = win32gui.GetWindowText(hwnd)
                    return
                # Also check process name
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                try:
                    proc = psutil.Process(pid)
                    if name_lower in proc.name().lower():
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
