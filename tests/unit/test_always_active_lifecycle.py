"""
test_always_active_lifecycle.py -- Unit tests for SingleInstanceGuard, Windows Autostart, and SpidyCore lifecycle idempotency.
=============================================================================================================================
"""

import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.core.single_instance import SingleInstanceGuard
from spidy.desktop.autostart import (
    APP_NAME,
    enable_autostart,
    disable_autostart,
    is_autostart_enabled,
    get_default_autostart_command,
)
from spidy.core.lifecycle import AppLifecycleState, LifecycleState


# ─── SingleInstanceGuard Tests ───────────────────────────────────────────────

def test_single_instance_primary_acquisition():
    """Primary instance successfully acquires lock."""
    # Use a test-specific mutex name to avoid clashing with any running instances
    guard = SingleInstanceGuard(
        mutex_name="Local\\Spidy_Test_Primary_Mutex_12345",
        port=59281,
    )
    acquired = guard.acquire()
    try:
        assert acquired is True
        assert guard.is_primary() is True
    finally:
        guard.release()


def test_single_instance_secondary_detection_and_ipc():
    """Secondary instance detects existing primary and notifies via IPC."""
    received_commands = []

    def on_command(cmd: str):
        received_commands.append(cmd)

    mutex_name = f"Local\\Spidy_Test_IPC_Mutex_{int(time.time())}"
    port = 59282

    guard1 = SingleInstanceGuard(mutex_name=mutex_name, port=port, on_command=on_command)
    assert guard1.acquire() is True

    # Secondary instance attempts to acquire same mutex
    guard2 = SingleInstanceGuard(mutex_name=mutex_name, port=port)
    assert guard2.acquire() is False
    assert guard2.is_primary() is False

    # Secondary notifies primary to restore HUD
    notified = guard2.notify_primary("RESTORE_HUD")
    assert notified is True

    # Allow IPC message to be received by primary
    time.sleep(0.15)
    assert "RESTORE_HUD" in received_commands

    guard2.release()
    guard1.release()


# ─── Autostart Manager Tests ────────────────────────────────────────────────

def test_get_autostart_command():
    """Autostart command should use pythonw.exe and --background flag."""
    cmd = get_default_autostart_command()
    assert "pythonw.exe" in cmd.lower()
    assert "-m spidy.main --background" in cmd


def test_autostart_registry_enable_disable():
    """Test registry manipulation with mocked winreg."""
    mock_reg = MagicMock()
    mock_key = MagicMock()
    mock_reg.OpenKey.return_value = mock_key
    mock_reg.CreateKey.return_value = mock_key
    mock_reg.QueryValueEx.return_value = ("pythonw.exe -m spidy.main --background", 1)

    with patch("spidy.desktop.autostart.winreg", mock_reg):
        # Check enabled
        assert is_autostart_enabled() is True

        # Check enable
        enable_autostart()
        mock_reg.SetValueEx.assert_called()

        # Check disable
        disable_autostart()
        mock_reg.DeleteValue.assert_called_with(mock_key.__enter__(), APP_NAME)


# ─── SpidyCore Lifecycle & Idempotency Tests ────────────────────────────────

@pytest.mark.asyncio
async def test_spidy_core_lifecycle_states():
    """Test AppLifecycleState definitions and transitions."""
    assert AppLifecycleState.STARTING.value == "STARTING"
    assert AppLifecycleState.RUNNING.value == "RUNNING"
    assert AppLifecycleState.HUD_VISIBLE.value == "HUD_VISIBLE"
    assert AppLifecycleState.HUD_HIDDEN.value == "HUD_HIDDEN"
    assert AppLifecycleState.AWAKE.value == "AWAKE"
    assert AppLifecycleState.LISTENING.value == "LISTENING"
    assert AppLifecycleState.STOPPING.value == "STOPPING"
    assert AppLifecycleState.STOPPED.value == "STOPPED"


@pytest.mark.asyncio
async def test_spidy_core_idempotent_start():
    """Calling start() multiple times should be safe and idempotent."""
    from spidy.core.app import SpidyCore

    mock_bus = MagicMock()
    core = SpidyCore(
        bus=mock_bus,
        single_instance=False,
        start_hidden=True,
    )
    core._state = LifecycleState.RUNNING

    # Duplicate start on running instance should return safely without error
    await core.start()
    assert core.state == LifecycleState.RUNNING

    # Stop should clean up safely
    await core.stop()
    assert core.state == LifecycleState.STOPPED

    # Duplicate stop should also be safe
    await core.stop()
    assert core.state == LifecycleState.STOPPED


def test_cli_argument_parsing():
    """Test CLI parser for new background and autostart flags."""
    import argparse
    from spidy.main import _build_cli_parser

    parser = _build_cli_parser()

    # Default
    args = parser.parse_args([])
    assert args.background is False
    assert args.install_autostart is False
    assert args.uninstall_autostart is False
    assert args.status_autostart is False

    # --background
    args = parser.parse_args(["--background"])
    assert args.background is True

    # --hidden alias
    args = parser.parse_args(["--hidden"])
    assert args.background is True

    # Autostart flags
    args = parser.parse_args(["--install-autostart"])
    assert args.install_autostart is True

    args = parser.parse_args(["--status-autostart"])
    assert args.status_autostart is True
