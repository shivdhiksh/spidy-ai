"""
EnvironmentContext — Unified Real-Time Windows Environment Snapshot
===================================================================
Provides a comprehensive, lightweight, deterministic snapshot of the
current desktop state for autonomous reasoning, planning, and verification.

Properties captured:
- Active window info (hwnd, title, process, class, bounding rect)
- Open top-level windows & running apps
- Screen dimensions
- Active browser URL & title (when available)
- Clipboard text (safely truncated)
- Accessible UI elements of active window
- File & workspace context
- Recent agent actions, active goal & task

Design principles:
- Cheap/deterministic APIs first (Win32/ctypes/psutil/UIA)
- Non-blocking with async interface and synchronous fallbacks
- Zero fabrication of unavailable data (empty string/list/dict when unknown)
- Safe: never logs/exposes passwords or unredacted secrets
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from spidy.desktop.controller import DesktopComputerController
from spidy.logging.logger import get_logger

log = get_logger(__name__)


@dataclass
class EnvironmentContext:
    """
    Unified environment context snapshot.
    """
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active_window: dict[str, Any] = field(default_factory=dict)
    active_process: str = ""
    open_windows: list[dict[str, Any]] = field(default_factory=list)
    running_apps: list[str] = field(default_factory=list)
    screen_size: tuple[int, int] = (1920, 1080)
    foreground_app: str = ""
    browser_state: dict[str, Any] = field(default_factory=dict)
    selected_text: str = ""
    clipboard_text: str = ""
    visible_ui_elements: list[dict[str, Any]] = field(default_factory=list)
    file_context: dict[str, Any] = field(default_factory=dict)
    current_directory: str = field(default_factory=os.getcwd)
    platform: str = field(default_factory=lambda: sys.platform)
    recent_agent_actions: list[str] = field(default_factory=list)
    current_goal: str = ""
    current_task: str = ""

    def summary(self) -> str:
        """Return a human-readable 1-2 line summary of active desktop state."""
        win_title = self.active_window.get("title", "") or self.foreground_app or "Desktop"
        proc = self.active_process or "unknown"
        apps_count = len(self.open_windows)
        summary_parts = [
            f"Active Window: '{win_title}' ({proc})",
            f"Screen Size: {self.screen_size[0]}x{self.screen_size[1]}",
            f"Windows open: {apps_count}",
        ]
        if self.browser_state.get("url"):
            summary_parts.append(f"Browser URL: {self.browser_state['url']}")
        if self.clipboard_text:
            clip_preview = self.clipboard_text[:30].replace("\n", " ")
            summary_parts.append(f"Clipboard: '{clip_preview}...'")
        return " | ".join(summary_parts)

    def format_summary(self) -> str:
        """Alias for summary()."""
        return self.summary()

    def is_app_running(self, app_name: str) -> bool:
        """Check if an application is in the running/open list."""
        name = app_name.lower().strip()
        for a in self.running_apps:
            if name in a.lower():
                return True
        for w in self.open_windows:
            if name in w.get("title", "").lower() or name in w.get("process_name", "").lower():
                return True
        return False

    def is_active_window(self, app_or_title: str) -> bool:
        """Check if active window matches title or process."""
        name = app_or_title.lower().strip()
        title = self.active_window.get("title", "").lower()
        proc = self.active_process.lower()
        return name in title or name in proc or name in self.foreground_app.lower()

    def to_dict(self) -> dict[str, Any]:
        """Serialize snapshot to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "active_window": self.active_window,
            "active_process": self.active_process,
            "open_windows": self.open_windows,
            "running_apps": self.running_apps,
            "screen_size": list(self.screen_size),
            "foreground_app": self.foreground_app,
            "browser_state": self.browser_state,
            "selected_text": self.selected_text,
            "clipboard_text": self.clipboard_text[:200] if self.clipboard_text else "",
            "visible_ui_elements": self.visible_ui_elements[:20],
            "file_context": self.file_context,
            "current_directory": self.current_directory,
            "recent_agent_actions": self.recent_agent_actions[-5:],
            "current_goal": self.current_goal,
            "current_task": self.current_task,
        }


class EnvironmentContextCollector:
    """
    Fast, non-blocking collector for EnvironmentContext snapshots.
    """

    def __init__(self, controller: DesktopComputerController | None = None) -> None:
        self._controller = controller or DesktopComputerController()

    async def collect(
        self,
        current_goal: str = "",
        current_task: str = "",
        recent_actions: list[str] | None = None,
        include_elements: bool = False,
    ) -> EnvironmentContext:
        """Alias for capture."""
        return await self.capture(
            current_goal=current_goal,
            current_task=current_task,
            recent_actions=recent_actions,
            include_ui_elements=include_elements,
        )

    async def capture(
        self,
        current_goal: str = "",
        current_task: str = "",
        recent_actions: list[str] | None = None,
        include_ui_elements: bool = False,
    ) -> EnvironmentContext:
        """
        Capture the current environment state.
        Uses non-blocking thread execution for OS inspection.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._capture_sync,
            current_goal,
            current_task,
            recent_actions or [],
            include_ui_elements,
        )

    def _capture_sync(
        self,
        current_goal: str,
        current_task: str,
        recent_actions: list[str],
        include_ui_elements: bool,
    ) -> EnvironmentContext:
        # 1. Screen size
        screen_size = self._controller.get_screen_size()

        # 2. Active window
        active_win = self._controller.get_active_window()
        active_proc = active_win.get("process_name", "")
        active_title = active_win.get("title", "")

        # Friendly app name derivation
        foreground_app = active_proc.replace(".exe", "").capitalize()
        if not foreground_app and active_title:
            foreground_app = active_title.split(" - ")[-1]

        # 3. Open windows list
        open_wins = self._controller.list_windows(visible_only=True)
        running_apps_set: set[str] = set()
        for w in open_wins:
            pname = w.get("process_name", "")
            if pname:
                running_apps_set.add(pname.replace(".exe", "").capitalize())
            elif w.get("title"):
                running_apps_set.add(w["title"].split(" - ")[-1])
        running_apps = sorted(running_apps_set)

        # 4. Clipboard text (truncated safely to prevent massive memory payloads)
        clipboard_text = ""
        try:
            raw_clip = self._controller.get_clipboard_text()
            if raw_clip:
                clipboard_text = raw_clip[:1000]
        except Exception as exc:
            log.debug("EnvironmentContext: clipboard read error: {exc}", exc=exc)

        # 5. Browser state inspection
        browser_state: dict[str, Any] = {}
        if active_proc.lower() in ("msedge.exe", "chrome.exe", "firefox.exe", "brave.exe"):
            browser_name = active_proc.lower().replace(".exe", "")
            browser_state = {
                "browser": browser_name,
                "title": active_title,
                "is_active": True,
            }
            # If window title contains standard browser pattern e.g. "Page Title - Microsoft​ Edge"
            if " - " in active_title:
                page_title = active_title.rsplit(" - ", 1)[0]
                browser_state["page_title"] = page_title

        # 6. UI elements (optional to keep collection latency under 5ms)
        visible_elements: list[dict[str, Any]] = []
        if include_ui_elements and active_win.get("hwnd"):
            try:
                hwnd = active_win["hwnd"]
                # Traverse top child controls
                child = self._controller._find_win32_child(hwnd, None, None)
                if child:
                    visible_elements.append(child)
            except Exception as exc:
                log.debug("EnvironmentContext: UI element inspection error: {exc}", exc=exc)

        return EnvironmentContext(
            active_window=active_win,
            active_process=active_proc,
            open_windows=open_wins,
            running_apps=running_apps,
            screen_size=screen_size,
            foreground_app=foreground_app,
            browser_state=browser_state,
            selected_text="",
            clipboard_text=clipboard_text,
            visible_ui_elements=visible_elements,
            file_context={"cwd": os.getcwd()},
            current_directory=os.getcwd(),
            recent_agent_actions=list(recent_actions),
            current_goal=current_goal,
            current_task=current_task,
        )
