"""
Shutdown watchdog and hard exit.

"Close yourself" used to leave Spidy half-closed in two ways:

1. The UI deadlocked (Qt called from the wrong thread), so graceful shutdown never ended.
2. Even a normal Python exit can hang inside Windows' ExitProcess: every DLL (CUDA, audio,
   onnxruntime...) runs its detach code, and one of them can block forever. The process then
   lingers with one thread, still holding the single-instance lock and the microphone.

So Spidy ends itself with ``TerminateProcess`` (no DLL detach code), and while shutting down
it arms two backstops that work even if Python itself is deadlocked:

- ``faulthandler.dump_traceback_later`` writes every thread's stack to
  ``%APPDATA%\\Spidy\\logs\\spidy_crash.log`` so the cause can be found, and
- a tiny detached helper process force-kills Spidy a few seconds later.
"""

from __future__ import annotations

import faulthandler
import os
import subprocess
import sys
from pathlib import Path

_watchdog_file = None  # must stay open while the timer is armed
_enabled = False       # only the real application turns this on (see spidy.main)


def enable() -> None:
    """Allow ``arm_shutdown_watchdog`` to act. Called by ``spidy.main`` only, never by tests/libraries."""
    global _enabled
    _enabled = True


def kill_self(exit_code: int = 0) -> None:
    """End this process immediately, skipping DLL detach code that can hang on exit."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.TerminateProcess(kernel32.GetCurrentProcess(), exit_code)
        except Exception:  # noqa: BLE001
            pass
    os._exit(exit_code)


def _spawn_external_killer(seconds: float) -> None:
    """Detached helper that force-kills this PID after ``seconds`` (Windows only)."""
    if sys.platform != "win32":
        return
    pid = os.getpid()
    # `ping -n N` is a console-less sleep (~1 s per count); taskkill uses TerminateProcess.
    # Only kill if the PID is still a Python process (PIDs get reused after Spidy exits).
    command = (
        f"ping -n {int(seconds) + 1} 127.0.0.1 >nul"
        f" & taskkill /F /FI \"PID eq {pid}\" /FI \"IMAGENAME eq pythonw.exe\" >nul 2>&1"
        f" & taskkill /F /FI \"PID eq {pid}\" /FI \"IMAGENAME eq python.exe\" >nul 2>&1"
    )
    # CREATE_NO_WINDOW alone: combining it with DETACHED_PROCESS made Windows open a visible
    # console showing the ping countdown.
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(
        ["cmd", "/c", command],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=flags, close_fds=True,
    )


def arm_shutdown_watchdog(seconds: float = 15.0) -> None:
    """If Spidy is still alive ``seconds`` after this call, dump all stacks and kill it. Idempotent."""
    global _watchdog_file
    if not _enabled or _watchdog_file is not None:
        return
    try:
        log_dir = Path(os.environ.get("APPDATA", str(Path.home()))) / "Spidy" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _watchdog_file = open(log_dir / "spidy_crash.log", "a", buffering=1, encoding="utf-8")
        _watchdog_file.write(
            f"\n=== shutdown watchdog armed: {seconds:.0f}s to exit; threads below if it stalls ===\n"
        )
        # exit=False: faulthandler's own exit uses _exit/ExitProcess, which can hang (see above).
        faulthandler.dump_traceback_later(seconds, exit=False, file=_watchdog_file)
        _spawn_external_killer(seconds + 2)
    except Exception:  # noqa: BLE001 — a diagnostics failure must never block shutdown
        _watchdog_file = None
