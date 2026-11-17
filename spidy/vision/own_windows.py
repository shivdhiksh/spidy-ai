"""
Keep Spidy out of her own screenshots
=====================================
When you ask "what's on my screen?" Spidy should see your screen, not her own HUD, pet or webs.

``hidden_from_capture()`` marks every visible window of this process with Windows'
WDA_EXCLUDEFROMCAPTURE for the instant a screenshot is taken (they stay visible to you; screen
captures see straight through them to what's behind), then restores them. Only during that
instant, so your own screen recordings still show Spidy.

``is_own_window(hwnd)`` tells whether a window (e.g. the foreground one) belongs to Spidy.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import time

WDA_NONE = 0x0
WDA_EXCLUDEFROMCAPTURE = 0x11          # Windows 10 2004+


def _own_windows() -> list[int]:
    try:
        user32 = ctypes.windll.user32
    except Exception:  # noqa: BLE001 -- not Windows
        return []
    pid = os.getpid()
    found: list[int] = []
    proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def cb(hwnd, _):
        wpid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
        if wpid.value == pid and user32.IsWindowVisible(hwnd):
            found.append(int(hwnd))
        return True
    user32.EnumWindows(proc(cb), 0)
    return found


def is_own_window(hwnd: int) -> bool:
    try:
        wpid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
        return wpid.value == os.getpid()
    except Exception:  # noqa: BLE001
        return False


def window_behind_spidy(hwnd: int) -> int | None:
    """If ``hwnd`` is one of Spidy's windows, the next real app window below it (else None)."""
    if not is_own_window(hwnd):
        return None
    try:
        user32 = ctypes.windll.user32
        GW_HWNDNEXT = 2
        h = user32.GetWindow(ctypes.c_void_p(hwnd), GW_HWNDNEXT)
        for _ in range(500):
            if not h:
                return None
            if user32.IsWindowVisible(ctypes.c_void_p(h)) and not is_own_window(h)                     and user32.GetWindowTextLengthW(ctypes.c_void_p(h)) > 0 and not user32.IsIconic(ctypes.c_void_p(h)):
                return int(h)
            h = user32.GetWindow(ctypes.c_void_p(h), GW_HWNDNEXT)
    except Exception:  # noqa: BLE001
        pass
    return None


@contextlib.contextmanager
def hidden_from_capture():
    """Screenshots taken inside this block don't include any of Spidy's windows."""
    hidden: list[int] = []
    try:
        user32 = ctypes.windll.user32
        for h in _own_windows():
            if user32.SetWindowDisplayAffinity(ctypes.c_void_p(h), WDA_EXCLUDEFROMCAPTURE):
                hidden.append(h)
        if hidden:
            time.sleep(0.03)                 # let the compositor apply it before the grab
    except Exception:  # noqa: BLE001 -- never block a screenshot over this
        pass
    try:
        yield
    finally:
        try:
            user32 = ctypes.windll.user32
            for h in hidden:
                user32.SetWindowDisplayAffinity(ctypes.c_void_p(h), WDA_NONE)
        except Exception:  # noqa: BLE001
            pass
