"""
ActiveWindowObserver — Foreground Window & Application Monitor
==============================================================
Polls the Windows foreground window every N seconds and publishes
events when the window title or active application changes.

Published events
----------------
  context.window_changed   — when the window title changes
  context.app_changed      — when the EXE name changes (app switch)

Windows API used
----------------
  win32gui.GetForegroundWindow()   → HWND of the active window
  win32gui.GetWindowText(hwnd)     → Window title string
  win32process.GetWindowThreadProcessId(hwnd) → (tid, pid)
  psutil.Process(pid).name()       → Process name (e.g. "Code.exe")
  psutil.Process(pid).exe()        → Full exe path

Fallback
--------
If pywin32 is not installed, the observer logs a warning and skips
polling silently. The rest of Spidy continues without context.
"""

from __future__ import annotations

import asyncio

from spidy.core.event_bus import EventBus
from spidy.logging.logger import get_logger
from spidy.perception.context.base import BaseObserver
from spidy.perception.context.events import AppChangedEvent, WindowChangedEvent
from spidy.perception.context.snapshot import WindowInfo

log = get_logger(__name__)

# Sentinel for "not yet seen"
_EMPTY_WINDOW = WindowInfo()


class ActiveWindowObserver(BaseObserver):
    """
    Monitors the foreground window and fires events on change.

    Parameters
    ----------
    bus:
        Application EventBus.
    poll_interval:
        How often to check the foreground window (seconds). Default: 0.5s.
    """

    name = "active_window"

    def __init__(self, bus: EventBus, poll_interval: float = 0.5) -> None:
        super().__init__(bus, poll_interval)
        self._last_window = _EMPTY_WINDOW
        self._win32_available = False

    async def on_start(self) -> None:
        try:
            import win32gui  # noqa: F401
            import win32process  # noqa: F401
            self._win32_available = True
            log.debug("ActiveWindowObserver: pywin32 available.")
        except ImportError:
            log.warning(
                "ActiveWindowObserver: pywin32 not installed. "
                "Window tracking will be disabled."
            )

    async def poll(self) -> None:
        if not self._win32_available:
            return

        window = await asyncio.to_thread(self._get_foreground_window)

        if window.title == self._last_window.title and \
                window.app_name == self._last_window.app_name:
            return  # No change

        old = self._last_window
        self._last_window = window

        # Always fire window_changed
        await self._bus.publish(WindowChangedEvent(
            title=window.title,
            hwnd=window.hwnd,
            app_name=window.app_name,
            pid=window.pid,
        ))

        # Fire app_changed only when the application itself switches
        if window.app_name != old.app_name:
            await self._bus.publish(AppChangedEvent(
                app_name=window.app_name,
                exe_path=window.exe_path,
                pid=window.pid,
            ))
            log.info(
                "App changed: {old} → {new}",
                old=old.app_name or "?",
                new=window.app_name or "?",
            )

    def current_window(self) -> WindowInfo:
        """Return the last observed window state (thread-safe read)."""
        return self._last_window

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_foreground_window() -> WindowInfo:
        """
        Read foreground window state synchronously.
        Must be called inside asyncio.to_thread().
        """
        try:
            import psutil
            import win32gui
            import win32process

            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return _EMPTY_WINDOW

            title = win32gui.GetWindowText(hwnd) or ""
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
            except Exception:
                return WindowInfo(title=title, hwnd=hwnd)

            try:
                proc = psutil.Process(pid)
                app_name = proc.name()
                try:
                    exe_path = proc.exe()
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    exe_path = ""
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                app_name = ""
                exe_path = ""

            return WindowInfo(
                title=title,
                app_name=app_name,
                exe_path=exe_path,
                pid=pid,
                hwnd=hwnd,
            )

        except Exception as exc:  # noqa: BLE001
            log.debug("_get_foreground_window error: {exc}", exc=exc)
            return _EMPTY_WINDOW
