"""
spidy.desktop.autostart — Windows User-Level Auto-Start Manager
==============================================================
Provides non-elevated auto-start registration for Spidy at Windows login
via the user registry:
    HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run

Features:
- Zero administrator permissions needed.
- Uses pythonw.exe (windowless python) so Spidy starts silently with NO console window.
- Supports --background flag to start with HUD hidden in tray.
- Inspects, installs, and uninstalls cleanly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import winreg
except ImportError:
    winreg = None

from spidy.config.manager import PROJECT_ROOT
from spidy.logging.logger import get_logger

log = get_logger(__name__)

_REG_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_APP_KEY_NAME = "Spidy"
APP_NAME = _APP_KEY_NAME
APP_KEY_NAME = _APP_KEY_NAME


def _get_pythonw_executable() -> str:
    """Return path to windowless python executable (pythonw.exe)."""
    current_python = sys.executable
    parent = Path(current_python).parent
    pythonw = parent / "pythonw.exe"
    if pythonw.is_file():
        return str(pythonw)
    return current_python


def get_default_autostart_command(background: bool = True) -> str:
    """
    Construct the recommended command line to start Spidy on Windows boot.
    """
    pythonw = _get_pythonw_executable()
    flag = " --background" if background else ""
    # Ensure working directory / PYTHONPATH is preserved or cd / script run
    cmd = f'"{pythonw}" -m spidy.main{flag}'
    return cmd


get_autostart_command = get_default_autostart_command


def is_autostart_enabled() -> bool:
    """Check whether Spidy is currently registered in HKCU Run key."""
    if sys.platform != "win32" or winreg is None:
        return False
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            _REG_RUN_KEY,
            0,
            winreg.KEY_READ,
        ) as key:
            try:
                val, _ = winreg.QueryValueEx(key, _APP_KEY_NAME)
                return bool(val)
            except FileNotFoundError:
                return False
    except Exception as exc:
        log.debug("Failed to check autostart registry key: {exc}", exc=exc)
        return False


def get_autostart_command() -> str | None:
    """Return current command line registered in registry, or None."""
    if sys.platform != "win32" or winreg is None:
        return None
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            _REG_RUN_KEY,
            0,
            winreg.KEY_READ,
        ) as key:
            try:
                val, _ = winreg.QueryValueEx(key, _APP_KEY_NAME)
                return str(val)
            except FileNotFoundError:
                return None
    except Exception as exc:
        log.debug("Failed to read autostart command: {exc}", exc=exc)
        return None


def enable_autostart(background: bool = True, custom_cmd: str | None = None) -> bool:
    """
    Register Spidy to start automatically at Windows login.
    """
    if sys.platform != "win32" or winreg is None:
        log.warning("Autostart is only supported on Windows platforms.")
        return False

    cmd = custom_cmd or get_default_autostart_command(background=background)
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            _REG_RUN_KEY,
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            winreg.SetValueEx(key, _APP_KEY_NAME, 0, winreg.REG_SZ, cmd)
        log.info(
            f"[SPIDY AUTOSTART] Registered Spidy in HKCU Run: {cmd}"
        )
        return True
    except Exception as exc:
        log.error("Failed to register autostart: {exc}", exc=exc)
        return False


def disable_autostart() -> bool:
    """
    Remove Spidy from Windows login registry.
    """
    if sys.platform != "win32" or winreg is None:
        return False
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            _REG_RUN_KEY,
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            try:
                winreg.DeleteValue(key, _APP_KEY_NAME)
                log.info("[SPIDY AUTOSTART] Removed Spidy from HKCU Run.")
                return True
            except FileNotFoundError:
                return True
    except Exception as exc:
        log.error("Failed to remove autostart: {exc}", exc=exc)
        return False
