"""
Desktop Computer Controller — Native Windows Automation & Computer Use
======================================================================
Implements deterministic, state-aware desktop automation for Windows:
- Screen metrics & native GDI screenshot capture
- Window management (focus, list, minimize, maximize, close, active title)
- Mouse control (move, click, double click, right click, scroll)
- Keyboard control (typing, key press, hotkey combos via SendInput)
- Clipboard integration (copy, paste, get/set text)
- Windows UI Automation / Accessible control lookup and invocation
- Structured telemetry logging with [COMPUTER] tags
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import subprocess
import sys
import time
from typing import Any

from spidy.logging.logger import get_logger

log = get_logger(__name__)

# Try importing win32gui/win32con/win32api
try:
    import win32api
    import win32con
    import win32gui
    import win32process
    _WIN32_AVAILABLE = True
except ImportError:
    win32api = None
    win32con = None
    win32gui = None
    win32process = None
    _WIN32_AVAILABLE = False


class DesktopComputerController:
    """
    Unified Windows Desktop Computer-Use Controller.
    """

    def __init__(self) -> None:
        self._is_windows = sys.platform == "win32"

    # ── 1. Screen & Display ───────────────────────────────────────────────────

    def get_screen_size(self) -> tuple[int, int]:
        """Return (width, height) of the primary monitor in pixels."""
        if self._is_windows:
            try:
                user32 = ctypes.windll.user32
                w = user32.GetSystemMetrics(0)
                h = user32.GetSystemMetrics(1)
                return w, h
            except Exception as exc:
                log.debug("[COMPUTER] get_screen_size error: {exc}", exc=exc)
        return 1920, 1080

    def screenshot(
        self,
        region: tuple[int, int, int, int] | None = None,
        file_path: str = "",
    ) -> bytes:
        """
        Capture a screenshot of the entire screen or a sub-region (x, y, w, h).
        Saves to file_path if provided and returns raw PNG/BMP bytes.
        """
        log.info("[COMPUTER] action=start type=screenshot region={reg}", reg=region)
        img_bytes = b""
        if self._is_windows:
            try:
                img_bytes = self._capture_gdi(region)
            except Exception as exc:
                log.warning("[COMPUTER] screenshot GDI error: {exc}", exc=exc)

        if not img_bytes:
            # Fallback synthetic BMP header for mock / headless mode
            img_bytes = b"BM" + b"\x00" * 52

        if file_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(img_bytes)
                log.info("[COMPUTER] action=complete type=screenshot path={path}", path=file_path)
            except Exception as exc:
                log.error("[COMPUTER] failed to write screenshot to {path}: {exc}", path=file_path, exc=exc)

        return img_bytes

    def _capture_gdi(self, region: tuple[int, int, int, int] | None = None) -> bytes:
        """Capture screen via Windows GDI API with zero third-party dependencies."""
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        if region:
            x, y, w, h = region
        else:
            x, y = 0, 0
            w = user32.GetSystemMetrics(0)
            h = user32.GetSystemMetrics(1)

        hdc_screen = user32.GetDC(0)
        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        hbm = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
        hbm_old = gdi32.SelectObject(hdc_mem, hbm)

        # SRCCOPY = 0x00CC0020
        gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, x, y, 0x00CC0020)

        # Extract DIB bits
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", ctypes.c_uint32),
                ("biWidth", ctypes.c_int32),
                ("biHeight", ctypes.c_int32),
                ("biPlanes", ctypes.c_uint16),
                ("biBitCount", ctypes.c_uint16),
                ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32),
                ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32),
                ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32),
            ]

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = w
        bmi.biHeight = h
        bmi.biPlanes = 1
        bmi.biBitCount = 24
        bmi.biCompression = 0  # BI_RGB

        row_stride = ((w * 3 + 3) & ~3)
        buf_size = row_stride * h
        buffer = ctypes.create_string_buffer(buf_size)

        gdi32.GetDIBits(hdc_mem, hbm, 0, h, buffer, ctypes.byref(bmi), 0)

        # Cleanup GDI handles
        gdi32.SelectObject(hdc_mem, hbm_old)
        gdi32.DeleteObject(hbm)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)

        # Build BMP in memory (14-byte file header + 40-byte DIB header + pixel data)
        file_header = (
            b"BM"
            + (14 + 40 + buf_size).to_bytes(4, "little")
            + (0).to_bytes(4, "little")
            + (14 + 40).to_bytes(4, "little")
        )
        dib_header = bytes(bmi)
        return file_header + dib_header + buffer.raw

    # ── 2. Window Management ──────────────────────────────────────────────────

    def get_active_window(self) -> dict[str, Any]:
        """Return information about the currently focused active window."""
        if not self._is_windows:
            return {"hwnd": 0, "title": "Desktop", "class_name": "", "pid": 0, "process_name": "explorer.exe"}

        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return {"hwnd": 0, "title": "", "class_name": "", "pid": 0, "process_name": ""}

            title = self._get_window_title(hwnd)
            class_name = self._get_window_class(hwnd)

            pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            process_name = self._get_process_name(pid.value)

            rect = self._get_window_rect(hwnd)
            info = {
                "hwnd": hwnd,
                "title": title,
                "class_name": class_name,
                "pid": pid.value,
                "process_name": process_name,
                "rect": rect,
            }
            log.debug("[COMPUTER] observation active_window='{title}' proc={proc}", title=title, proc=process_name)
            return info
        except Exception as exc:
            log.debug("[COMPUTER] get_active_window error: {exc}", exc=exc)
            return {"hwnd": 0, "title": "", "class_name": "", "pid": 0, "process_name": ""}

    def list_windows(self, visible_only: bool = True) -> list[dict[str, Any]]:
        """List open top-level application windows."""
        windows: list[dict[str, Any]] = []
        if not self._is_windows:
            return windows

        user32 = ctypes.windll.user32

        def enum_proc(hwnd: int, lparam: int) -> bool:
            if visible_only and not user32.IsWindowVisible(hwnd):
                return True
            title = self._get_window_title(hwnd)
            if visible_only and not title.strip():
                return True

            pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            proc_name = self._get_process_name(pid.value)
            rect = self._get_window_rect(hwnd)

            windows.append({
                "hwnd": hwnd,
                "title": title,
                "class_name": self._get_window_class(hwnd),
                "pid": pid.value,
                "process_name": proc_name,
                "rect": rect,
            })
            return True

        enum_func_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
        user32.EnumWindows(enum_func_type(enum_proc), 0)
        return windows

    def find_window(self, query: str) -> dict[str, Any] | None:
        """Find a window by title substring or process name."""
        q = query.lower().strip()
        for win in self.list_windows(visible_only=True):
            if q in win["title"].lower() or q in win["process_name"].lower():
                return win
        return None

    def focus_window(self, query: str | int) -> bool:
        """Bring target window to the foreground."""
        log.info("[COMPUTER] action=start type=focus_window target='{q}'", q=query)
        if not self._is_windows:
            log.info("[COMPUTER] action=complete type=focus_window success=True (mock)")
            return True

        hwnd = query if isinstance(query, int) else None
        if hwnd is None:
            win = self.find_window(str(query))
            if win:
                hwnd = win["hwnd"]

        if not hwnd:
            log.warning("[COMPUTER] focus_window: window '{q}' not found", q=query)
            return False

        try:
            user32 = ctypes.windll.user32
            # SW_RESTORE = 9
            user32.ShowWindow(hwnd, 9)
            user32.SetForegroundWindow(hwnd)
            time.sleep(0.05)
            log.info("[COMPUTER] action=complete type=focus_window success=True hwnd={h}", h=hwnd)
            return True
        except Exception as exc:
            log.error("[COMPUTER] focus_window failed: {exc}", exc=exc)
            return False

    def open_application(self, name_or_path: str, args: str = "") -> tuple[bool, str]:
        """Launch or focus a Windows application by name or executable path."""
        log.info("[COMPUTER] action=start type=open_app target='{t}'", t=name_or_path)
        app_map = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "calc": "calc.exe",
            "file explorer": "explorer.exe",
            "explorer": "explorer.exe",
            "settings": "ms-settings:",
            "windows settings": "ms-settings:",
            "paint": "mspaint.exe",
            "cmd": "cmd.exe",
            "terminal": "wt.exe",
            "edge": "msedge.exe",
            "chrome": "chrome.exe",
        }
        cmd = app_map.get(name_or_path.lower().strip(), name_or_path)

        # Check if already running and visible (except Settings which uses protocol)
        if not cmd.startswith("ms-settings:"):
            existing = self.find_window(name_or_path)
            if existing:
                self.focus_window(existing["hwnd"])
                log.info("[COMPUTER] action=complete type=open_app existing_window_focused=True")
                return True, f"Focused existing '{name_or_path}' window."

        try:
            if cmd.startswith("ms-settings:"):
                os.startfile(cmd)
            else:
                full_cmd = [cmd]
                if args:
                    full_cmd.extend(args.split())
                subprocess.Popen(full_cmd, shell=False)
            time.sleep(0.4)
            log.info("[COMPUTER] action=complete type=open_app success=True")
            return True, f"Launched '{name_or_path}'."
        except Exception as exc:
            log.error("[COMPUTER] open_app failed: {exc}", exc=exc)
            return False, f"Failed to open '{name_or_path}': {exc}"

    def close_window(self, query: str | int, force: bool = False) -> bool:
        """Close an application window."""
        log.info("[COMPUTER] action=start type=close_window target='{q}' force={f}", q=query, f=force)
        if not self._is_windows:
            return True

        hwnd = query if isinstance(query, int) else None
        if hwnd is None:
            win = self.find_window(str(query))
            if win:
                hwnd = win["hwnd"]

        if not hwnd:
            return False

        try:
            user32 = ctypes.windll.user32
            # WM_CLOSE = 0x0010
            user32.PostMessageW(hwnd, 0x0010, 0, 0)
            time.sleep(0.1)
            if force:
                pid = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value:
                    try:
                        import subprocess
                        subprocess.run(
                            ["taskkill", "/F", "/PID", str(pid.value)],
                            capture_output=True,
                            timeout=2.0,
                        )
                    except Exception:
                        pass
            log.info("[COMPUTER] action=complete type=close_window success=True")
            return True
        except Exception as exc:
            log.error("[COMPUTER] close_window failed: {exc}", exc=exc)
            return False

    def minimize_window(self, query: str | int | None = None) -> bool:
        """Minimize target window or active window."""
        return self._show_window_cmd(query, 6)  # SW_MINIMIZE = 6

    def maximize_window(self, query: str | int | None = None) -> bool:
        """Maximize target window or active window."""
        return self._show_window_cmd(query, 3)  # SW_MAXIMIZE = 3

    def restore_window(self, query: str | int | None = None) -> bool:
        """Restore target window or active window."""
        return self._show_window_cmd(query, 9)  # SW_RESTORE = 9

    def _show_window_cmd(self, query: str | int | None, cmd: int) -> bool:
        if not self._is_windows:
            return True
        user32 = ctypes.windll.user32
        hwnd = None
        if query is None:
            hwnd = user32.GetForegroundWindow()
        elif isinstance(query, int):
            hwnd = query
        else:
            win = self.find_window(str(query))
            if win:
                hwnd = win["hwnd"]

        if not hwnd:
            return False
        try:
            user32.ShowWindow(hwnd, cmd)
            return True
        except Exception as exc:
            log.debug("[COMPUTER] ShowWindow error: {exc}", exc=exc)
            return False

    # ── 3. Mouse Control ──────────────────────────────────────────────────────

    def get_mouse_position(self) -> tuple[int, int]:
        """Get current mouse cursor position (x, y)."""
        if self._is_windows:
            try:
                class POINT(ctypes.Structure):
                    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
                pt = POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                return pt.x, pt.y
            except Exception:
                pass
        return 0, 0

    def move_mouse(self, x: int, y: int, duration: float = 0.0) -> bool:
        """Move the mouse cursor to (x, y)."""
        log.info("[COMPUTER] action=start type=move_mouse x={x} y={y}", x=x, y=y)
        if not self._is_windows:
            return True
        try:
            ctypes.windll.user32.SetCursorPos(int(x), int(y))
            if duration > 0:
                time.sleep(duration)
            log.info("[COMPUTER] action=complete type=move_mouse success=True")
            return True
        except Exception as exc:
            log.error("[COMPUTER] move_mouse failed: {exc}", exc=exc)
            return False

    def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
        clicks: int = 1,
    ) -> bool:
        """Click mouse button at specified coordinates or current position."""
        log.info(
            "[COMPUTER] action=start type=click x={x} y={y} button={b} clicks={c}",
            x=x, y=y, b=button, c=clicks,
        )
        if not self._is_windows:
            return True

        try:
            user32 = ctypes.windll.user32
            if x is not None and y is not None:
                user32.SetCursorPos(int(x), int(y))
                time.sleep(0.02)

            btn = button.lower()
            down_flag = 0x0002 if btn == "left" else (0x0008 if btn == "right" else 0x0020)
            up_flag = 0x0004 if btn == "left" else (0x0010 if btn == "right" else 0x0040)

            for _ in range(clicks):
                user32.mouse_event(down_flag, 0, 0, 0, 0)
                time.sleep(0.01)
                user32.mouse_event(up_flag, 0, 0, 0, 0)
                if clicks > 1:
                    time.sleep(0.05)

            log.info("[COMPUTER] action=complete type=click success=True")
            return True
        except Exception as exc:
            log.error("[COMPUTER] click failed: {exc}", exc=exc)
            return False

    def double_click(self, x: int | None = None, y: int | None = None) -> bool:
        """Perform a mouse double-click."""
        return self.click(x=x, y=y, button="left", clicks=2)

    def right_click(self, x: int | None = None, y: int | None = None) -> bool:
        """Perform a mouse right-click."""
        return self.click(x=x, y=y, button="right", clicks=1)

    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> bool:
        """Scroll mouse wheel (+ up, - down)."""
        log.info("[COMPUTER] action=start type=scroll clicks={c}", c=clicks)
        if not self._is_windows:
            return True
        try:
            user32 = ctypes.windll.user32
            if x is not None and y is not None:
                user32.SetCursorPos(int(x), int(y))
            # MOUSEEVENTF_WHEEL = 0x0800, WHEEL_DELTA = 120
            user32.mouse_event(0x0800, 0, 0, clicks * 120, 0)
            log.info("[COMPUTER] action=complete type=scroll success=True")
            return True
        except Exception as exc:
            log.error("[COMPUTER] scroll failed: {exc}", exc=exc)
            return False

    # ── 4. Keyboard & Typing ──────────────────────────────────────────────────

    def type_text(self, text: str, interval: float = 0.01, press_enter: bool = False) -> bool:
        """Type text string into the currently focused window."""
        log.info(
            "[COMPUTER] action=start type=type_text text_len={l} enter={e}",
            l=len(text), e=press_enter,
        )
        if not self._is_windows:
            log.info("[COMPUTER] action=complete type=type_text success=True (mock)")
            return True

        try:
            user32 = ctypes.windll.user32
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
                else:
                    # Unicode character via SendInput
                    self._send_unicode_char(char)
                if interval > 0:
                    time.sleep(interval)

            if press_enter:
                time.sleep(0.02)
                user32.keybd_event(0x0D, 0, 0, 0)  # VK_RETURN down
                user32.keybd_event(0x0D, 0, 2, 0)  # VK_RETURN up

            log.info("[COMPUTER] action=complete type=type_text success=True")
            return True
        except Exception as exc:
            log.error("[COMPUTER] type_text failed: {exc}", exc=exc)
            return False

    def press_key(self, key: str) -> bool:
        """Press and release a single key (e.g. 'enter', 'tab', 'esc', 'backspace', 'f5')."""
        log.info("[COMPUTER] action=start type=press_key key='{k}'", k=key)
        if not self._is_windows:
            return True
        vk_map = {
            "enter": 0x0D, "return": 0x0D, "esc": 0x1B, "escape": 0x1B,
            "tab": 0x09, "space": 0x20, "backspace": 0x08, "delete": 0x2E,
            "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
            "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74,
            "f6": 0x75, "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79,
            "f11": 0x7A, "f12": 0x7B,
        }
        vk = vk_map.get(key.lower(), 0)
        if not vk and len(key) == 1:
            vk = ctypes.windll.user32.VkKeyScanW(ord(key)) & 0xFF

        if not vk:
            log.warning("[COMPUTER] press_key: unknown key '{k}'", k=key)
            return False

        try:
            user32 = ctypes.windll.user32
            user32.keybd_event(vk, 0, 0, 0)
            time.sleep(0.02)
            user32.keybd_event(vk, 0, 2, 0)
            log.info("[COMPUTER] action=complete type=press_key success=True")
            return True
        except Exception as exc:
            log.error("[COMPUTER] press_key failed: {exc}", exc=exc)
            return False

    def hotkey(self, *keys: str) -> bool:
        """Execute a key combination (e.g. 'ctrl', 's' or 'alt', 'f4')."""
        key_list = []
        for k in keys:
            parts = k.replace("+", " ").split()
            key_list.extend(parts)

        log.info("[COMPUTER] action=start type=hotkey combo='{c}'", c="+".join(key_list))
        if not self._is_windows:
            return True

        vk_map = {
            "ctrl": 0x11, "control": 0x11, "shift": 0x10, "alt": 0x12,
            "win": 0x5B, "windows": 0x5B, "cmd": 0x5B,
            "enter": 0x0D, "return": 0x0D, "esc": 0x1B, "escape": 0x1B,
            "tab": 0x09, "space": 0x20, "backspace": 0x08, "delete": 0x2E,
            "s": 0x53, "c": 0x43, "v": 0x56, "x": 0x58, "a": 0x41, "z": 0x5A,
            "f4": 0x73, "n": 0x4E, "o": 0x4F, "w": 0x57, "r": 0x52,
        }
        vks: list[int] = []
        for k in key_list:
            norm = k.lower().strip()
            vk = vk_map.get(norm)
            if not vk and len(norm) == 1:
                vk = ctypes.windll.user32.VkKeyScanW(ord(norm)) & 0xFF
            if vk:
                vks.append(vk)

        if not vks:
            return False

        try:
            user32 = ctypes.windll.user32
            # Key down in forward order
            for vk in vks:
                user32.keybd_event(vk, 0, 0, 0)
                time.sleep(0.01)
            time.sleep(0.05)
            # Key up in reverse order
            for vk in reversed(vks):
                user32.keybd_event(vk, 0, 2, 0)
                time.sleep(0.01)

            log.info("[COMPUTER] action=complete type=hotkey success=True")
            return True
        except Exception as exc:
            log.error("[COMPUTER] hotkey failed: {exc}", exc=exc)
            return False

    def _send_unicode_char(self, char: str) -> None:
        """Send a Unicode character via keybd_event / SendInput."""
        try:
            user32 = ctypes.windll.user32
            # KEYEVENTF_UNICODE = 0x0004
            user32.keybd_event(0, ord(char), 0x0004, 0)
            user32.keybd_event(0, ord(char), 0x0004 | 2, 0)
        except Exception:
            pass

    # ── 5. Clipboard Integration ──────────────────────────────────────────────

    def get_clipboard_text(self) -> str:
        """Get current text from the Windows clipboard."""
        if not self._is_windows:
            return ""
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            user32.OpenClipboard.argtypes = [ctypes.c_void_p]
            user32.OpenClipboard.restype = ctypes.c_bool
            user32.GetClipboardData.argtypes = [ctypes.c_uint]
            user32.GetClipboardData.restype = ctypes.c_void_p
            user32.CloseClipboard.restype = ctypes.c_bool

            kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalLock.restype = ctypes.c_void_p
            kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalUnlock.restype = ctypes.c_bool

            if not user32.OpenClipboard(None):
                return ""
            # CF_UNICODETEXT = 13
            h_data = user32.GetClipboardData(13)
            if not h_data:
                user32.CloseClipboard()
                return ""
            data_ptr = kernel32.GlobalLock(h_data)
            if not data_ptr:
                user32.CloseClipboard()
                return ""
            text = ctypes.c_wchar_p(data_ptr).value or ""
            kernel32.GlobalUnlock(h_data)
            user32.CloseClipboard()
            return text
        except Exception as exc:
            log.debug("[COMPUTER] get_clipboard_text error: {exc}", exc=exc)
            return ""

    def set_clipboard_text(self, text: str) -> bool:
        """Set text onto the Windows clipboard."""
        if not self._is_windows:
            return True
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            user32.OpenClipboard.argtypes = [ctypes.c_void_p]
            user32.OpenClipboard.restype = ctypes.c_bool
            user32.EmptyClipboard.restype = ctypes.c_bool
            user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
            user32.SetClipboardData.restype = ctypes.c_void_p
            user32.CloseClipboard.restype = ctypes.c_bool

            kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
            kernel32.GlobalAlloc.restype = ctypes.c_void_p
            kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalLock.restype = ctypes.c_void_p
            kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalUnlock.restype = ctypes.c_bool

            # Try opening clipboard with retry
            opened = False
            for _ in range(5):
                if user32.OpenClipboard(None):
                    opened = True
                    break
                time.sleep(0.02)
            if not opened:
                return False

            user32.EmptyClipboard()
            # GMEM_MOVEABLE | GMEM_ZEROINIT = 0x0042
            buf_bytes = (text + "\x00").encode("utf-16le")
            h_data = kernel32.GlobalAlloc(0x0042, len(buf_bytes))
            if not h_data:
                user32.CloseClipboard()
                return False

            data_ptr = kernel32.GlobalLock(h_data)
            if not data_ptr:
                user32.CloseClipboard()
                return False

            ctypes.memmove(data_ptr, buf_bytes, len(buf_bytes))
            kernel32.GlobalUnlock(h_data)
            res = user32.SetClipboardData(13, h_data)
            user32.CloseClipboard()
            return bool(res)
        except Exception as exc:
            log.debug("[COMPUTER] set_clipboard_text error: {exc}", exc=exc)
            return False

    def copy(self) -> str:
        """Execute Ctrl+C and return clipboard content."""
        self.hotkey("ctrl", "c")
        time.sleep(0.05)
        return self.get_clipboard_text()

    def paste(self, text: str | None = None) -> bool:
        """Paste text (sets clipboard if text provided, then executes Ctrl+V)."""
        if text is not None:
            self.set_clipboard_text(text)
        return self.hotkey("ctrl", "v")

    # ── 6. UI Automation & Semantic Control Lookup ────────────────────────────

    def find_ui_element(
        self,
        window_query: str | None = None,
        control_name: str | None = None,
        control_type: str | None = None,
        auto_id: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Find an accessible UI element inside a window by name, class, or automation ID.
        Uses native Win32 child window traversal and UI Automation queries.
        """
        log.info(
            "[COMPUTER] action=start type=find_ui_element win='{w}' name='{n}' type='{t}' id='{i}'",
            w=window_query, n=control_name, t=control_type, i=auto_id,
        )
        if not self._is_windows:
            return None

        # Find target parent window
        parent_hwnd = 0
        if window_query:
            win = self.find_window(window_query)
            if win:
                parent_hwnd = win["hwnd"]
        else:
            parent_hwnd = ctypes.windll.user32.GetForegroundWindow()

        if not parent_hwnd:
            return None

        # 1. Native Win32 child window search
        found_ctrl = self._find_win32_child(parent_hwnd, control_name, control_type)
        if found_ctrl:
            log.info("[COMPUTER] action=complete type=find_ui_element found=True (win32)")
            return found_ctrl

        # 2. PowerShell System.Windows.Automation lookup
        ps_ctrl = self._find_uia_powershell(parent_hwnd, control_name, auto_id)
        if ps_ctrl:
            log.info("[COMPUTER] action=complete type=find_ui_element found=True (uia)")
            return ps_ctrl

        log.debug("[COMPUTER] find_ui_element: control not found")
        return None

    def click_ui_element(
        self,
        window_query: str | None = None,
        control_name: str | None = None,
        control_type: str | None = None,
        auto_id: str | None = None,
    ) -> bool:
        """Find a UI element semantically and click it."""
        log.info("[COMPUTER] action=start type=click_ui_element name='{n}'", n=control_name)
        elem = self.find_ui_element(
            window_query=window_query,
            control_name=control_name,
            control_type=control_type,
            auto_id=auto_id,
        )
        if not elem:
            log.warning("[COMPUTER] click_ui_element: element not found")
            return False

        rect = elem.get("rect")
        if rect:
            # Click center of element rect (left, top, right, bottom)
            cx = (rect[0] + rect[2]) // 2
            cy = (rect[1] + rect[3]) // 2
            return self.click(x=cx, y=cy)

        hwnd = elem.get("hwnd")
        if hwnd:
            # BM_CLICK = 0x00F5
            ctypes.windll.user32.SendMessageW(hwnd, 0x00F5, 0, 0)
            log.info("[COMPUTER] action=complete type=click_ui_element success=True (BM_CLICK)")
            return True

        return False

    def _find_win32_child(
        self,
        parent_hwnd: int,
        target_name: str | None,
        target_class: str | None,
    ) -> dict[str, Any] | None:
        user32 = ctypes.windll.user32
        found: list[dict[str, Any]] = []

        def enum_child(hwnd: int, lparam: int) -> bool:
            title = self._get_window_title(hwnd)
            class_name = self._get_window_class(hwnd)

            name_match = True if not target_name else (target_name.lower() in title.lower())
            class_match = True if not target_class else (target_class.lower() in class_name.lower())

            if name_match and class_match:
                rect = self._get_window_rect(hwnd)
                found.append({
                    "hwnd": hwnd,
                    "title": title,
                    "class_name": class_name,
                    "rect": rect,
                })
                return False
            return True

        enum_func_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
        user32.EnumChildWindows(parent_hwnd, enum_func_type(enum_child), 0)
        return found[0] if found else None

    def _find_uia_powershell(
        self,
        parent_hwnd: int,
        target_name: str | None,
        auto_id: str | None,
    ) -> dict[str, Any] | None:
        """Query UIAutomationElement via PowerShell .NET reflection."""
        try:
            ps_script = f"""
            Add-Type -AssemblyName UIAutomationClient
            $elem = [System.Windows.Automation.AutomationElement]::FromHandle({parent_hwnd})
            if ($elem) {{
                $conds = @()
                if ('{target_name or ""}') {{
                    $conds += New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::NameProperty, '{target_name}')
                }}
                if ('{auto_id or ""}') {{
                    $conds += New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::AutomationIdProperty, '{auto_id}')
                }}
                if ($conds.Count -gt 1) {{
                    $cond = New-Object System.Windows.Automation.AndCondition($conds)
                }} elseif ($conds.Count -eq 1) {{
                    $cond = $conds[0]
                }} else {{
                    $cond = [System.Windows.Automation.Condition]::TrueCondition
                }}
                $found = $elem.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $cond)
                if ($found) {{
                    $r = $found.Current.BoundingRectangle
                    "$($r.Left),$($r.Top),$($r.Right),$($r.Bottom)|$($found.Current.Name)"
                }}
            }}
            """
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True, text=True, timeout=2,
            )
            out = res.stdout.strip()
            if "|" in out:
                coords_str, name = out.split("|", 1)
                parts = [int(float(c)) for c in coords_str.split(",")]
                if len(parts) == 4:
                    return {
                        "name": name,
                        "rect": (parts[0], parts[1], parts[2], parts[3]),
                    }
        except Exception as exc:
            log.debug("[COMPUTER] _find_uia_powershell error: {exc}", exc=exc)
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_window_title(hwnd: int) -> str:
        try:
            user32 = ctypes.windll.user32
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return ""
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            return buff.value
        except Exception:
            return ""

    @staticmethod
    def _get_window_class(hwnd: int) -> str:
        try:
            user32 = ctypes.windll.user32
            buff = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, buff, 256)
            return buff.value
        except Exception:
            return ""

    @staticmethod
    def _get_window_rect(hwnd: int) -> tuple[int, int, int, int]:
        try:
            class RECT(ctypes.Structure):
                _fields_ = [
                    ("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long),
                ]
            r = RECT()
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(r))
            return r.left, r.top, r.right, r.bottom
        except Exception:
            return 0, 0, 0, 0

    @staticmethod
    def _get_process_name(pid: int) -> str:
        if pid <= 0:
            return ""
        try:
            # PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h_proc = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not h_proc:
                return ""
            buff = ctypes.create_unicode_buffer(1024)
            size = ctypes.c_ulong(1024)
            ctypes.windll.kernel32.QueryFullProcessImageNameW(h_proc, 0, buff, ctypes.byref(size))
            ctypes.windll.kernel32.CloseHandle(h_proc)
            full_path = buff.value
            return os.path.basename(full_path) if full_path else ""
        except Exception:
            return ""
