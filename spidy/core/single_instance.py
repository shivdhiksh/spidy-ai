"""
spidy.core.single_instance — Single Instance Process Guard & Inter-Process Notifier
==================================================================================
Ensures only one instance of Spidy runs at any time on the system.
If a second instance is launched (e.g., via shortcut or Windows login):
1. Detects the existing instance via a Windows named kernel mutex.
2. Connects to the existing instance's local IPC listener.
3. Requests the existing instance to restore and foreground its HUD.
4. Cleanly exits the second process with code 0.
"""

from __future__ import annotations

import ctypes
import os
import socket
import sys
import threading
from pathlib import Path
from typing import Callable

from spidy.config.manager import PROJECT_ROOT
from spidy.logging.logger import get_logger

log = get_logger(__name__)

_MUTEX_NAME = r"Local\Spidy_Desktop_Companion_SingleInstance_Mutex"
_DEFAULT_PORT = 49281
_LOCKFILE_PATH = Path(os.path.expandvars(r"%APPDATA%\Spidy\.single_instance.port"))


class _CallableBool(int):
    def __call__(self) -> bool:
        return bool(self)


class SingleInstanceGuard:
    """
    Guarantees a single Spidy process and coordinates IPC between launches.
    """

    def __init__(
        self,
        mutex_name: str = _MUTEX_NAME,
        port: int = _DEFAULT_PORT,
        on_command: Callable[[str], None] | None = None,
    ) -> None:
        self._mutex_name = mutex_name
        self._port = port
        self._mutex_handle = None
        self._is_primary = False
        self._server_sock: socket.socket | None = None
        self._listener_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._on_command = on_command

    @property
    def is_primary(self) -> _CallableBool:
        return _CallableBool(self._is_primary)

    def acquire(self) -> bool:
        """
        Attempt to acquire primary ownership.
        Returns True if this is the first running instance.
        Returns False if an existing instance is already running.
        """
        if sys.platform == "win32":
            kernel32 = ctypes.windll.kernel32
            # CreateMutexW(security_attributes, initial_owner, name)
            self._mutex_handle = kernel32.CreateMutexW(None, True, self._mutex_name)
            last_error = kernel32.GetLastError()
            # ERROR_ALREADY_EXISTS = 183
            if last_error == 183 or self._mutex_handle == 0:
                log.info(
                    "[SPIDY LIFECYCLE] Existing Spidy instance detected via kernel mutex."
                )
                self._is_primary = False
                return False
        else:
            # Unix-like socket binding fallback for testing
            try:
                test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_sock.bind(("127.0.0.1", self._port))
                test_sock.close()
            except OSError:
                log.info("[SPIDY LIFECYCLE] Existing Spidy instance detected via socket bind.")
                self._is_primary = False
                return False

        self._is_primary = True
        if self._on_command:
            self.start_listener(self._on_command)
        return True

    def notify_existing_instance(self, message: str = "RESTORE_HUD") -> bool:
        """
        Send an IPC activation command to the currently running instance.
        """
        target_port = self._port
        # Check lockfile for dynamic port if available
        if _LOCKFILE_PATH.is_file():
            try:
                content = _LOCKFILE_PATH.read_text(encoding="utf-8").strip()
                if content.isdigit():
                    target_port = int(content)
            except Exception:
                pass

        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(2.0)
            client.connect(("127.0.0.1", target_port))
            client.sendall(message.strip().encode("utf-8") + b"\n")
            response = client.recv(1024)
            client.close()
            log.info(
                "[SPIDY LIFECYCLE] Sent '{msg}' to existing instance, got ack='{ack}'",
                msg=message,
                ack=response.decode("utf-8", errors="ignore").strip(),
            )
            return True
        except Exception as exc:
            log.warning(
                "[SPIDY LIFECYCLE] Could not notify existing instance: {exc}", exc=exc
            )
            return False

    notify_primary = notify_existing_instance

    def start_listener(self, on_command: Callable[[str], None]) -> None:
        """
        Start the background socket listener for IPC commands from new launches.
        """
        if not self._is_primary:
            return

        self._on_command = on_command
        self._stop_event.clear()

        # Try binding default port or fallback to ephemeral port
        bound = False
        for p in (self._port, 0):
            try:
                self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._server_sock.bind(("127.0.0.1", p))
                self._server_sock.listen(5)
                self._server_sock.settimeout(1.0)
                actual_port = self._server_sock.getsockname()[1]
                self._port = actual_port
                bound = True
                break
            except Exception as exc:
                log.debug("Could not bind port {p}: {exc}", p=p, exc=exc)

        if not bound or self._server_sock is None:
            log.warning("[SPIDY LIFECYCLE] Failed to bind IPC socket listener.")
            return

        # Write actual port to lockfile
        try:
            _LOCKFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _LOCKFILE_PATH.write_text(str(self._port), encoding="utf-8")
        except Exception as exc:
            log.debug("Failed to write single instance lockfile: {exc}", exc=exc)

        def _listen_loop() -> None:
            log.info("[SPIDY LIFECYCLE] Single instance IPC listener active on port {p}", p=self._port)
            while not self._stop_event.is_set():
                try:
                    conn, _ = self._server_sock.accept()
                    conn.settimeout(2.0)
                    data = conn.recv(1024)
                    if data:
                        cmd = data.decode("utf-8", errors="ignore").strip()
                        log.info("[SPIDY LIFECYCLE] Received IPC command: '{cmd}'", cmd=cmd)
                        conn.sendall(b"ACK\n")
                        conn.close()
                        if self._on_command:
                            try:
                                self._on_command(cmd)
                            except Exception as cb_exc:
                                log.warning("Error in single instance command callback: {exc}", exc=cb_exc)
                    else:
                        conn.close()
                except socket.timeout:
                    continue
                except Exception:
                    break

        self._listener_thread = threading.Thread(
            target=_listen_loop,
            name="SpidySingleInstanceListener",
            daemon=True,
        )
        self._listener_thread.start()

    def release(self) -> None:
        """Release the mutex, stop listener, and clean up lockfile."""
        self._stop_event.set()
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None

        if _LOCKFILE_PATH.is_file():
            try:
                _LOCKFILE_PATH.unlink(missing_ok=True)
            except Exception:
                pass

        if self._mutex_handle and sys.platform == "win32":
            kernel32 = ctypes.windll.kernel32
            kernel32.ReleaseMutex(self._mutex_handle)
            kernel32.CloseHandle(self._mutex_handle)
            self._mutex_handle = None

        self._is_primary = False
