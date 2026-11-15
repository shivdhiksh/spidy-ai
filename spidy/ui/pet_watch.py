"""
What the desktop pet notices around him (Windows):

* ForegroundWatcher -- the active window's title / app / position, polled cheaply: a new browser tab, a new
  window or a PDF being opened.
* scroll detection -- Windows Raw Input (WM_INPUT with RIDEV_INPUTSINK) on the pet's own window: he hears
  mouse-wheel scrolls anywhere without a global hook, so the mouse never lags.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0
RIDEV_INPUTSINK = 0x00000100
WM_INPUT = 0x00FF
RI_MOUSE_WHEEL = 0x0400
RI_MOUSE_HWHEEL = 0x0800


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [("usUsagePage", wintypes.USHORT), ("usUsage", wintypes.USHORT),
                ("dwFlags", wintypes.DWORD), ("hwndTarget", wintypes.HWND)]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [("dwType", wintypes.DWORD), ("dwSize", wintypes.DWORD),
                ("hDevice", wintypes.HANDLE), ("wParam", wintypes.WPARAM)]


def register_scroll_sink(hwnd: int) -> bool:
    """Ask Windows to send this window raw mouse input even when it is not focused."""
    if sys.platform != "win32":
        return False
    try:
        dev = RAWINPUTDEVICE(0x01, 0x02, RIDEV_INPUTSINK, wintypes.HWND(hwnd))
        return bool(ctypes.windll.user32.RegisterRawInputDevices(ctypes.byref(dev), 1, ctypes.sizeof(dev)))
    except Exception:  # noqa: BLE001
        return False


def is_wheel_message(message_ptr: int) -> bool:
    """True if the native message is a raw-input mouse-wheel event."""
    try:
        msg = wintypes.MSG.from_address(int(message_ptr))
        if msg.message != WM_INPUT:
            return False
        user32 = ctypes.windll.user32
        size = wintypes.UINT(0)
        hdr = ctypes.sizeof(RAWINPUTHEADER)
        user32.GetRawInputData(wintypes.HANDLE(msg.lParam), RID_INPUT, None, ctypes.byref(size), hdr)
        if size.value <= 0 or size.value > 256:
            return False
        buf = ctypes.create_string_buffer(size.value)
        if user32.GetRawInputData(wintypes.HANDLE(msg.lParam), RID_INPUT, buf, ctypes.byref(size), hdr) <= 0:
            return False
        header = RAWINPUTHEADER.from_buffer_copy(buf)
        if header.dwType != RIM_TYPEMOUSE:
            return False
        flags = int.from_bytes(buf.raw[hdr + 4:hdr + 6], "little")             # RAWMOUSE.usButtonFlags
        return bool(flags & (RI_MOUSE_WHEEL | RI_MOUSE_HWHEEL))
    except Exception:  # noqa: BLE001
        return False


class ForegroundWatcher:
    """Reports what changed in the active window since the last poll."""

    def __init__(self) -> None:
        self.title = ""
        self.app = ""
        self.hwnd = 0

    def poll(self) -> tuple[str, str, tuple[int, int, int, int]] | None:
        """Returns (kind, title, rect) when something new happened: "pdf", "tab", "window"; else None."""
        if sys.platform != "win32":
            return None
        try:
            import win32gui
            import win32process
            import psutil
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd) or ""
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            app = psutil.Process(pid).name().lower() if pid else ""
            rect = win32gui.GetWindowRect(hwnd)
        except Exception:  # noqa: BLE001
            return None
        if app.startswith("python") or not title or title in ("Spidy", "Spidy — click to open"):
            return None                                                      # ignore Spidy's own windows
        old_title, old_app, old_hwnd = self.title, self.app, self.hwnd
        self.title, self.app, self.hwnd = title, app, hwnd
        if not old_title or title == old_title:
            return None
        if ".pdf" in title.lower() and ".pdf" not in old_title.lower():
            return "pdf", title, rect
        if hwnd == old_hwnd and app == old_app:
            return "tab", title, rect                                        # same window, new title: new tab/page
        if hwnd != old_hwnd:
            return "window", title, rect
        return None
