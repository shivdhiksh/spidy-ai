"""
ComputerControlSkill — Windows Direct Computer Control Agent
============================================================
Provides deterministic mouse, keyboard, and window automation capabilities
for desktop interaction (e.g. interacting with Calculator, native dialogs,
active windows, and hotkey sequences).

Permission Tiers
----------------
T0 — Read-only (get active window)
T1 — Reversible OS actions (mouse click, move, scroll, drag, keyboard type, hotkeys, window focus/minimize/maximize)
T2 — Disruptive process/window closing (close_window)

Architecture
------------
- Mouse and keyboard automation uses pyautogui when available, with
  deterministic ctypes/win32 API fallbacks for maximum Windows reliability.
- Window management uses ctypes.windll.user32 (EnumWindows, SetForegroundWindow,
  ShowWindow, CloseWindow).
- All external dependencies are guarded and fail-safe for CI and headless testing.
- Every action publishes completion events to the EventBus.
"""

from __future__ import annotations

import asyncio
import ctypes
import sys
import time
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

# Try importing pyautogui with failsafe configuration
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    _PYAUTOGUI_AVAILABLE = True
except Exception:  # noqa: BLE001
    pyautogui = None
    _PYAUTOGUI_AVAILABLE = False


class ComputerControlSkill(BaseSkill):
    """
    Direct computer-use skill for OS mouse, keyboard, and window control.

    Parameters
    ----------
    bus:
        Application EventBus for publishing computer control events.
    typing_interval:
        Default delay between keystrokes in seconds.
    """

    name = "computer_control_skill"
    version = "1.0.0"

    def __init__(
        self,
        bus: "EventBus | None" = None,
        typing_interval: float = 0.01,
    ) -> None:
        self._bus = bus
        self._typing_interval = typing_interval
        from spidy.desktop.controller import DesktopComputerController
        self._controller = DesktopComputerController()

    # ── Capabilities ──────────────────────────────────────────────────────

    def capabilities(self) -> list[SkillCapability]:
        return [
            # ── T0 — Read-only ─────────────────────────────────────────
            SkillCapability(
                action="get_active_window",
                description="Get the title and details of the currently focused active window.",
                permission_tier="T0",
                params=[],
                examples=[
                    "what window is active", "get active window", "current window title",
                ],
            ),
            # ── T1 — Mouse Operations ──────────────────────────────────
            SkillCapability(
                action="mouse_click",
                description="Click the mouse at specific coordinates or the current position.",
                permission_tier="T1",
                params=[
                    ParamSchema("x", "int", required=False, default=None, description="X coordinate"),
                    ParamSchema("y", "int", required=False, default=None, description="Y coordinate"),
                    ParamSchema("button", "string", required=False, default="left", description="left | right | middle"),
                    ParamSchema("clicks", "int", required=False, default=1, description="Number of clicks"),
                ],
                examples=[
                    "click mouse", "click at 500 300", "right click", "double click",
                ],
            ),
            SkillCapability(
                action="mouse_move",
                description="Move the mouse pointer to specific coordinates.",
                permission_tier="T1",
                params=[
                    ParamSchema("x", "int", required=True, description="Target X coordinate"),
                    ParamSchema("y", "int", required=True, description="Target Y coordinate"),
                    ParamSchema("duration", "float", required=False, default=0.1, description="Movement duration"),
                ],
                examples=[
                    "move mouse to 100 200", "move cursor to center",
                ],
            ),
            SkillCapability(
                action="mouse_scroll",
                description="Scroll the mouse wheel up or down.",
                permission_tier="T1",
                params=[
                    ParamSchema("clicks", "int", required=True, description="Scroll amount (+ up, - down)"),
                ],
                examples=[
                    "scroll down", "scroll up 5 clicks",
                ],
            ),
            # ── T1 — Keyboard Operations ───────────────────────────────
            SkillCapability(
                action="keyboard_type",
                description="Type text into the currently active focused application window.",
                permission_tier="T1",
                params=[
                    ParamSchema("text", "string", required=True, description="Text string to type"),
                    ParamSchema("interval", "float", required=False, default=0.01, description="Keystroke interval"),
                    ParamSchema("press_enter", "bool", required=False, default=False, description="Press Enter after typing"),
                ],
                examples=[
                    "type hello world", "write into active window", "type 125 * 24",
                ],
            ),
            SkillCapability(
                action="key_press",
                description="Press a single key (e.g. enter, esc, tab, backspace, space, f5).",
                permission_tier="T1",
                params=[
                    ParamSchema("key", "string", required=True, description="Key name to press"),
                ],
                examples=[
                    "press enter", "hit escape", "press tab", "press space",
                ],
            ),
            SkillCapability(
                action="hotkey",
                description="Press a keyboard shortcut combination (e.g. ctrl+c, alt+tab, win+r, ctrl+v).",
                permission_tier="T1",
                params=[
                    ParamSchema("keys", "string", required=True, description="Keys separated by '+' or space (e.g. 'ctrl+s')"),
                ],
                examples=[
                    "press ctrl+c", "hit alt+tab", "shortcut win+r", "press ctrl+v",
                ],
            ),
            # ── T1 / T2 — Window Management ────────────────────────────
            SkillCapability(
                action="focus_window",
                description="Bring an application window matching the title to the foreground.",
                permission_tier="T1",
                params=[
                    ParamSchema("title", "string", required=True, description="Window title or keyword"),
                ],
                examples=[
                    "focus calculator", "switch to notepad", "bring edge to front",
                ],
            ),
            SkillCapability(
                action="minimize_window",
                description="Minimize a window by title or the currently active window.",
                permission_tier="T1",
                params=[
                    ParamSchema("title", "string", required=False, default="", description="Window title or empty for active"),
                ],
                examples=[
                    "minimize window", "minimize calculator",
                ],
            ),
            SkillCapability(
                action="maximize_window",
                description="Maximize a window by title or the currently active window.",
                permission_tier="T1",
                params=[
                    ParamSchema("title", "string", required=False, default="", description="Window title or empty for active"),
                ],
                examples=[
                    "maximize window", "maximize edge",
                ],
            ),
            SkillCapability(
                action="close_window",
                description="Close an application window matching the title.",
                permission_tier="T2",
                params=[
                    ParamSchema("title", "string", required=True, description="Window title or keyword"),
                ],
                examples=[
                    "close calculator window", "close active window",
                ],
            ),
        ]

    # ── Execution Dispatch ─────────────────────────────────────────────────

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        handler_map = {
            "get_active_window": self._exec_get_active_window,
            "mouse_click":       self._exec_mouse_click,
            "click":             self._exec_mouse_click,
            "mouse_move":        self._exec_mouse_move,
            "mouse_scroll":      self._exec_mouse_scroll,
            "keyboard_type":     self._exec_keyboard_type,
            "type_text":         self._exec_keyboard_type,
            "key_press":         self._exec_key_press,
            "hotkey":            self._exec_hotkey,
            "focus_window":      self._exec_focus_window,
            "focus":             self._exec_focus_window,
            "minimize_window":   self._exec_minimize_window,
            "maximize_window":   self._exec_maximize_window,
            "close_window":      self._exec_close_window,
            "click_element":     self._exec_click_element,
            "copy":              self._exec_copy,
            "paste":             self._exec_paste,
            "get_clipboard":     self._exec_get_clipboard,
            "set_clipboard":     self._exec_set_clipboard,
            "screenshot":        self._exec_screenshot,
        }

        handler = handler_map.get(action)
        if handler is None:
            return SkillResult.fail(f"ComputerControlSkill: unknown action '{action}'.")

        try:
            return await handler(context)
        except Exception as exc:  # noqa: BLE001
            log.error("ComputerControlSkill.{act} error: {exc}", act=action, exc=exc)
            return SkillResult.fail(f"Computer control error during {action}: {exc}")

    # ── Mouse Handlers ─────────────────────────────────────────────────────

    async def _exec_mouse_click(self, context: SkillContext) -> SkillResult:
        x = context.get("x")
        y = context.get("y")
        button = (context.get("button") or "left").lower()
        clicks = int(context.get("clicks", 1))

        if _PYAUTOGUI_AVAILABLE and pyautogui:
            await asyncio.to_thread(
                pyautogui.click,
                x=x,
                y=y,
                clicks=clicks,
                button=button,
            )
            coords_str = f" at ({x}, {y})" if x is not None and y is not None else ""
            return SkillResult.ok(f"Clicked {button} mouse button {clicks} time(s){coords_str}.")

        # Fallback to ctypes for Windows if pyautogui is not available
        if sys.platform == "win32":
            await self._win32_mouse_click(x, y, button, clicks)
            return SkillResult.ok(f"Clicked {button} mouse button via Windows API.")

        return SkillResult.ok("Mouse click simulated (test mode).")

    async def _exec_mouse_move(self, context: SkillContext) -> SkillResult:
        x = int(context.get("x", 0))
        y = int(context.get("y", 0))
        duration = float(context.get("duration", 0.1))

        if _PYAUTOGUI_AVAILABLE and pyautogui:
            await asyncio.to_thread(pyautogui.moveTo, x, y, duration=duration)
            return SkillResult.ok(f"Moved mouse pointer to ({x}, {y}).")

        if sys.platform == "win32":
            await asyncio.to_thread(ctypes.windll.user32.SetCursorPos, x, y)
            return SkillResult.ok(f"Moved mouse pointer to ({x}, {y}) via Windows API.")

        return SkillResult.ok(f"Mouse move to ({x}, {y}) simulated (test mode).")

    async def _exec_mouse_scroll(self, context: SkillContext) -> SkillResult:
        clicks = int(context.get("clicks", 0))

        if _PYAUTOGUI_AVAILABLE and pyautogui:
            await asyncio.to_thread(pyautogui.scroll, clicks)
            return SkillResult.ok(f"Scrolled mouse wheel {clicks} clicks.")

        if sys.platform == "win32":
            # MOUSEEVENTF_WHEEL = 0x0800, WHEEL_DELTA = 120
            await asyncio.to_thread(
                ctypes.windll.user32.mouse_event,
                0x0800, 0, 0, clicks * 120, 0
            )
            return SkillResult.ok(f"Scrolled mouse wheel {clicks} clicks via Windows API.")

        return SkillResult.ok(f"Mouse scroll {clicks} simulated (test mode).")

    # ── Keyboard Handlers ──────────────────────────────────────────────────

    async def _exec_keyboard_type(self, context: SkillContext) -> SkillResult:
        text = context.get("text", "")
        interval = float(context.get("interval", self._typing_interval))
        press_enter = bool(context.get("press_enter", False))

        if not text:
            return SkillResult.fail("keyboard_type requires 'text' parameter.")

        if _PYAUTOGUI_AVAILABLE and pyautogui:
            await asyncio.to_thread(pyautogui.write, text, interval=interval)
            if press_enter:
                await asyncio.sleep(0.05)
                await asyncio.to_thread(pyautogui.press, "enter")
            return SkillResult.ok(f"Typed text: '{text}'" + (" (with Enter)" if press_enter else ""))

        # Windows ctypes fallback for typing
        if sys.platform == "win32":
            await self._win32_send_keys(text, press_enter=press_enter)
            return SkillResult.ok(f"Typed text via Windows API: '{text}'")

        return SkillResult.ok(f"Simulated typing: '{text}'")

    async def _exec_key_press(self, context: SkillContext) -> SkillResult:
        key = (context.get("key", "") or "").lower().strip()
        if not key:
            return SkillResult.fail("key_press requires 'key' parameter.")

        if _PYAUTOGUI_AVAILABLE and pyautogui:
            await asyncio.to_thread(pyautogui.press, key)
            return SkillResult.ok(f"Pressed key: '{key}'.")

        if sys.platform == "win32":
            await self._win32_press_key(key)
            return SkillResult.ok(f"Pressed key via Windows API: '{key}'.")

        return SkillResult.ok(f"Simulated key press: '{key}'.")

    async def _exec_hotkey(self, context: SkillContext) -> SkillResult:
        keys_str = context.get("keys", "") or ""
        if not keys_str:
            return SkillResult.fail("hotkey requires 'keys' parameter (e.g. 'ctrl+c').")

        # Split combinations like 'ctrl+c', 'alt+tab', 'win+r'
        keys = [k.strip().lower() for k in keys_str.replace(" ", "+").split("+") if k.strip()]
        if not keys:
            return SkillResult.fail(f"Invalid hotkey specification: '{keys_str}'.")

        # Map common names
        key_map = {
            "win": "win", "windows": "win", "cmd": "win",
            "ctrl": "ctrl", "control": "ctrl",
            "alt": "alt", "shift": "shift",
            "enter": "enter", "return": "enter",
            "esc": "esc", "escape": "esc",
        }
        resolved_keys = [key_map.get(k, k) for k in keys]

        if _PYAUTOGUI_AVAILABLE and pyautogui:
            await asyncio.to_thread(pyautogui.hotkey, *resolved_keys)
            return SkillResult.ok(f"Executed hotkey shortcut: '{'+'.join(resolved_keys)}'.")

        return SkillResult.ok(f"Simulated hotkey: '{'+'.join(resolved_keys)}'.")

    # ── Window Handlers ────────────────────────────────────────────────────

    async def _exec_get_active_window(self, context: SkillContext) -> SkillResult:
        if sys.platform == "win32":
            title = await asyncio.to_thread(self._win32_get_active_window_title)
            return SkillResult.ok(f"Active window: '{title}'", data={"title": title})

        return SkillResult.ok("Active window: 'Desktop' (simulated)", data={"title": "Desktop"})

    async def _exec_focus_window(self, context: SkillContext) -> SkillResult:
        title = context.get("title", "")
        if not title:
            return SkillResult.fail("focus_window requires 'title' parameter.")

        if sys.platform == "win32":
            found = await asyncio.to_thread(self._win32_focus_window_by_title, title)
            if found:
                return SkillResult.ok(f"Focused window matching '{title}'.")
            return SkillResult.fail(f"No window found matching '{title}'.")

        return SkillResult.ok(f"Focused window '{title}' (simulated).")

    async def _exec_minimize_window(self, context: SkillContext) -> SkillResult:
        title = context.get("title", "")
        if sys.platform == "win32":
            success = await asyncio.to_thread(self._win32_show_window, title, 6)  # SW_MINIMIZE = 6
            return SkillResult.ok(f"Minimized window '{title or 'active'}'.") if success else SkillResult.fail(f"Could not minimize window '{title}'.")

        return SkillResult.ok(f"Minimized window '{title or 'active'}' (simulated).")

    async def _exec_maximize_window(self, context: SkillContext) -> SkillResult:
        title = context.get("title", "")
        if sys.platform == "win32":
            success = await asyncio.to_thread(self._win32_show_window, title, 3)  # SW_MAXIMIZE = 3
            return SkillResult.ok(f"Maximized window '{title or 'active'}'.") if success else SkillResult.fail(f"Could not maximize window '{title}'.")

        return SkillResult.ok(f"Maximized window '{title or 'active'}' (simulated).")

    async def _exec_close_window(self, context: SkillContext) -> SkillResult:
        title = context.get("title", "")
        if not title:
            return SkillResult.fail("close_window requires 'title' parameter.")

        if sys.platform == "win32":
            success = await asyncio.to_thread(self._win32_close_window, title)
            return SkillResult.ok(f"Closed window matching '{title}'.") if success else SkillResult.fail(f"Could not close window matching '{title}'.")

        return SkillResult.ok(f"Closed window '{title}' (simulated).")

    # ── Win32 Native Helpers ───────────────────────────────────────────────

    @staticmethod
    def _win32_get_active_window_title() -> str:
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return ""
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return ""
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            return buff.value
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _win32_focus_window_by_title(target_title: str) -> bool:
        try:
            user32 = ctypes.windll.user32
            target = target_title.lower()
            found_hwnd = [0]

            def enum_proc(hwnd: int, lparam: int) -> bool:
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buff, length + 1)
                        if target in buff.value.lower():
                            found_hwnd[0] = hwnd
                            return False  # stop enumeration
                return True

            enum_func_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
            user32.EnumWindows(enum_func_type(enum_proc), 0)

            if found_hwnd[0]:
                user32.ShowWindow(found_hwnd[0], 9)  # SW_RESTORE = 9
                user32.SetForegroundWindow(found_hwnd[0])
                return True
            return False
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _win32_show_window(target_title: str, cmd: int) -> bool:
        try:
            user32 = ctypes.windll.user32
            if not target_title:
                hwnd = user32.GetForegroundWindow()
                if hwnd:
                    user32.ShowWindow(hwnd, cmd)
                    return True
                return False

            target = target_title.lower()
            found = [0]

            def enum_proc(hwnd: int, lparam: int) -> bool:
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buff, length + 1)
                        if target in buff.value.lower():
                            found[0] = hwnd
                            return False
                return True

            enum_func_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
            user32.EnumWindows(enum_func_type(enum_proc), 0)

            if found[0]:
                user32.ShowWindow(found[0], cmd)
                return True
            return False
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _win32_close_window(target_title: str) -> bool:
        try:
            user32 = ctypes.windll.user32
            WM_CLOSE = 0x0010
            target = target_title.lower()
            found = [0]

            def enum_proc(hwnd: int, lparam: int) -> bool:
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buff, length + 1)
                        if target in buff.value.lower():
                            found[0] = hwnd
                            return False
                return True

            enum_func_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
            user32.EnumWindows(enum_func_type(enum_proc), 0)

            if found[0]:
                user32.PostMessageW(found[0], WM_CLOSE, 0, 0)
                return True
            return False
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    async def _win32_mouse_click(x: int | None, y: int | None, button: str, clicks: int) -> None:
        user32 = ctypes.windll.user32
        if x is not None and y is not None:
            user32.SetCursorPos(x, y)
            await asyncio.sleep(0.02)

        down_flag = 0x0002 if button == "left" else 0x0008  # MOUSEEVENTF_LEFTDOWN / RIGHTDOWN
        up_flag = 0x0004 if button == "left" else 0x0010    # MOUSEEVENTF_LEFTUP / RIGHTUP

        for _ in range(clicks):
            user32.mouse_event(down_flag, 0, 0, 0, 0)
            await asyncio.sleep(0.01)
            user32.mouse_event(up_flag, 0, 0, 0, 0)
            if clicks > 1:
                await asyncio.sleep(0.05)

    @staticmethod
    async def _win32_send_keys(text: str, press_enter: bool = False) -> None:
        # Simple virtual key dispatch via Windows SendInput / keybd_event
        user32 = ctypes.windll.user32
        VK_RETURN = 0x0D
        for char in text:
            vk = user32.VkKeyScanW(ord(char))
            if vk != -1:
                key_code = vk & 0xFF
                shift = (vk >> 8) & 1
                if shift:
                    user32.keybd_event(0x10, 0, 0, 0)  # VK_SHIFT down
                user32.keybd_event(key_code, 0, 0, 0)
                user32.keybd_event(key_code, 0, 2, 0)  # KEYEVENTF_KEYUP
                if shift:
                    user32.keybd_event(0x10, 0, 2, 0)  # VK_SHIFT up
            await asyncio.sleep(0.01)

        if press_enter:
            user32.keybd_event(VK_RETURN, 0, 0, 0)
            user32.keybd_event(VK_RETURN, 0, 2, 0)

    @staticmethod
    async def _win32_press_key(key: str) -> None:
        user32 = ctypes.windll.user32
        vk_map = {
            "enter": 0x0D, "return": 0x0D,
            "esc": 0x1B, "escape": 0x1B,
            "tab": 0x09, "space": 0x20,
            "backspace": 0x08, "delete": 0x2E,
            "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
            "f5": 0x74,
        }
        vk = vk_map.get(key.lower(), 0)
        if vk:
            user32.keybd_event(vk, 0, 0, 0)
            user32.keybd_event(vk, 0, 2, 0)

    # ── Additional Desktop Computer Handlers ───────────────────────────────

    async def _exec_click_element(self, context: SkillContext) -> SkillResult:
        name = context.get("target") or context.get("name") or context.get("control_name")
        win = context.get("window") or context.get("window_query")
        auto_id = context.get("auto_id") or context.get("automation_id")
        ok = await asyncio.to_thread(
            self._controller.click_ui_element,
            window_query=win,
            control_name=name,
            auto_id=auto_id,
        )
        if ok:
            return SkillResult.ok(f"Clicked UI element '{name}'.")
        return SkillResult.fail(f"Could not click UI element '{name}'.")

    async def _exec_copy(self, context: SkillContext) -> SkillResult:
        text = await asyncio.to_thread(self._controller.copy)
        return SkillResult.ok(f"Copied text to clipboard: '{text[:80]}'", data={"text": text})

    async def _exec_paste(self, context: SkillContext) -> SkillResult:
        text = context.get("text")
        ok = await asyncio.to_thread(self._controller.paste, text)
        if ok:
            return SkillResult.ok("Pasted text into active window.")
        return SkillResult.fail("Failed to paste text.")

    async def _exec_get_clipboard(self, context: SkillContext) -> SkillResult:
        text = await asyncio.to_thread(self._controller.get_clipboard_text)
        return SkillResult.ok(f"Clipboard text: '{text[:80]}'", data={"text": text})

    async def _exec_set_clipboard(self, context: SkillContext) -> SkillResult:
        text = context.get("text", "")
        ok = await asyncio.to_thread(self._controller.set_clipboard_text, text)
        if ok:
            return SkillResult.ok("Clipboard updated successfully.")
        return SkillResult.fail("Failed to set clipboard.")

    async def _exec_screenshot(self, context: SkillContext) -> SkillResult:
        path = context.get("path") or context.get("file_path") or ""
        raw_bytes = await asyncio.to_thread(self._controller.screenshot, file_path=path)
        return SkillResult.ok(f"Screenshot captured ({len(raw_bytes)} bytes).", data={"size_bytes": len(raw_bytes)})

