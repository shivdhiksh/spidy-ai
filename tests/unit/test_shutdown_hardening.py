"""
"Close yourself" used to leave Spidy half-closed:
- QApplication.quit() was called from the asyncio thread and deadlocked with the Qt thread's
  paintEvent, so the HUD froze ("Not responding") and shutdown never finished;
- a normal exit can hang in a DLL's detach code, leaving a process that still owns the
  single-instance lock;
- a stale lock then made Spidy silently refuse to start.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Slot  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from spidy.ui.overlay import UISignalBridge  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_quit_request_from_another_thread_runs_on_the_qt_thread(qapp):
    seen: list[int] = []

    class Receiver(QObject):
        @Slot()
        def on_quit(self):
            seen.append(threading.get_ident())

    receiver = Receiver()
    bridge = UISignalBridge()
    bridge.quit_app_requested.connect(receiver.on_quit)

    t = threading.Thread(target=bridge.request_quit_app)
    t.start()
    t.join()
    assert seen == [], "must be queued, not executed on the calling thread"
    qapp.processEvents()
    assert seen == [threading.get_ident()]


def test_spidyapp_quit_never_calls_qt_directly_when_a_bridge_exists():
    from spidy.ui.app import SpidyApp

    app = object.__new__(SpidyApp)
    app._on_quit = None
    app._bridge = MagicMock()
    app._qt_app = MagicMock()
    app.quit()
    app._bridge.request_quit_app.assert_called_once()
    app._qt_app.quit.assert_not_called()


def test_kill_self_ends_the_process_even_with_a_live_non_daemon_thread():
    code = (
        "import threading, time\n"
        "from spidy.core.watchdog import kill_self\n"
        "threading.Thread(target=lambda: time.sleep(60)).start()\n"
        "kill_self(7)\n"
    )
    started = time.monotonic()
    proc = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, timeout=30,
                          env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert proc.returncode == 7
    assert time.monotonic() - started < 20


def test_watchdog_does_nothing_unless_the_real_app_enabled_it():
    """Otherwise any test calling request_shutdown() would get the whole test run killed."""
    from spidy.core import watchdog

    assert watchdog._enabled is False
    with patch.object(watchdog, "_spawn_external_killer") as killer, \
            patch.object(watchdog.faulthandler, "dump_traceback_later") as dump:
        watchdog.arm_shutdown_watchdog(5)
    killer.assert_not_called()
    dump.assert_not_called()


@pytest.mark.asyncio
async def test_stale_single_instance_lock_does_not_stop_spidy_starting():
    from spidy.core.app import SpidyCore

    core = SpidyCore(single_instance=True)
    core._initialise = AsyncMock()
    core._run = AsyncMock()
    core._stop = AsyncMock()
    guard = MagicMock()
    guard.acquire.return_value = False                      # lock is held ...
    guard.notify_existing_instance.return_value = False     # ... but nobody answers
    with patch("spidy.core.single_instance.SingleInstanceGuard", return_value=guard):
        await core.start()
    core._initialise.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_second_instance_still_defers_to_the_running_one():
    from spidy.core.app import SpidyCore

    core = SpidyCore(single_instance=True)
    core._initialise = AsyncMock()
    core._run = AsyncMock()
    guard = MagicMock()
    guard.acquire.return_value = False
    guard.notify_existing_instance.return_value = True      # running instance answered
    with patch("spidy.core.single_instance.SingleInstanceGuard", return_value=guard):
        await core.start()
    core._initialise.assert_not_awaited()
