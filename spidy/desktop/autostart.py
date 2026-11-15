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
    # --supervise: like a phone's assistant service, Spidy comes back by itself if it ever crashes
    cmd = f'"{pythonw}" -m spidy.main --supervise{flag}'
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


# ── Task Scheduler (reliable start at every login / unlock) ───────────────────
# The registry Run list proved unreliable on this PC: after a reboot Windows ran only three of its
# entries and silently skipped Spidy. A per-user scheduled task (no admin needed) starts Spidy at
# login (after a short delay so audio/OneDrive are ready) and again whenever the PC is unlocked; a
# Spidy that is already running ignores the extra start (single-instance guard / IgnoreNew).

TASK_NAME = "Spidy Assistant"


def _task_xml(background: bool = True) -> str:
    import getpass
    from xml.sax.saxutils import escape
    user = escape(os.environ.get("USERDOMAIN", "") + "\\" + getpass.getuser())
    exe = escape(_get_pythonw_executable())
    args = escape("-m spidy.main --supervise" + (" --background" if background else ""))
    cwd = escape(str(PROJECT_ROOT))
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>Starts Spidy so "Hey Spidy" works after every boot, login and unlock.</Description></RegistrationInfo>
  <Triggers>
    <LogonTrigger><Enabled>true</Enabled><UserId>{user}</UserId><Delay>PT15S</Delay></LogonTrigger>
    <SessionStateChangeTrigger><Enabled>true</Enabled><StateChange>SessionUnlock</StateChange><UserId>{user}</UserId><Delay>PT5S</Delay></SessionStateChangeTrigger>
  </Triggers>
  <Principals><Principal id="Author"><UserId>{user}</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure><Interval>PT1M</Interval><Count>10</Count></RestartOnFailure>
    <Priority>6</Priority>
  </Settings>
  <Actions Context="Author"><Exec><Command>{exe}</Command><Arguments>{args}</Arguments><WorkingDirectory>{cwd}</WorkingDirectory></Exec></Actions>
</Task>"""


def install_logon_task(background: bool = True) -> bool:
    """Register the per-user scheduled task (replaces the Run-key entry). Returns True on success."""
    if sys.platform != "win32":
        return False
    import subprocess
    import tempfile
    path = Path(tempfile.gettempdir()) / "spidy_task.xml"
    path.write_text(_task_xml(background), encoding="utf-16")
    r = subprocess.run(["schtasks", "/Create", "/TN", TASK_NAME, "/XML", str(path), "/F"],
                       capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        path.unlink()
    except OSError:
        pass
    if r.returncode != 0:
        log.warning("[SPIDY AUTOSTART] scheduled task not created: {e}", e=(r.stderr or r.stdout).strip())
        return False
    log.info("[SPIDY AUTOSTART] scheduled task '{t}' registered (login + unlock).", t=TASK_NAME)
    return True


def is_logon_task_installed() -> bool:
    if sys.platform != "win32":
        return False
    import subprocess
    r = subprocess.run(["schtasks", "/Query", "/TN", TASK_NAME], capture_output=True, text=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return r.returncode == 0


def remove_logon_task() -> bool:
    if sys.platform != "win32":
        return False
    import subprocess
    r = subprocess.run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"], capture_output=True, text=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return r.returncode == 0


def ensure_reliable_autostart(background: bool = True) -> str:
    """Prefer the scheduled task; fall back to the Run key. Returns "task" or "registry"."""
    if install_logon_task(background):
        disable_autostart()                    # one start mechanism, not two
        return "task"
    enable_autostart(background)
    return "registry"
