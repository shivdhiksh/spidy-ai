"""
Test suite for spidy.core.app (SpidyCore)
==========================================
Tests lifecycle transitions, startup, and shutdown.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from spidy.core.app import SpidyCore
from spidy.core.lifecycle import LifecycleState


class TestSpidyCoreLifecycle:
    """SpidyCore should drive through lifecycle states correctly."""

    @pytest.mark.asyncio
    async def test_initial_state_is_created(self, tmp_path: Path):
        """A freshly constructed SpidyCore should be in CREATED state."""
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("", encoding="utf-8")
        core = SpidyCore(config_path=cfg_file)
        assert core.state == LifecycleState.CREATED

    @pytest.mark.asyncio
    async def test_core_reaches_running_state(self, tmp_path: Path):
        """SpidyCore.start() should reach RUNNING, then shutdown cleanly."""
        cfg_file = tmp_path / "cfg.yaml"
        # Explicitly disable the Qt overlay so this test doesn't create a
        # QApplication in a background thread, which would contaminate the
        # Qt timer tests in test_ui_overlay.py that run later in the suite.
        cfg_file.write_text("ui:\n  enabled: false\n", encoding="utf-8")
        core = SpidyCore(config_path=cfg_file)

        async def shutdown_after_start(event):
            await core.request_shutdown()

        # Run start() but trigger shutdown immediately after the started event
        # by subscribing before the core is started
        async def run():
            # We must subscribe after the bus is created (during initialise)
            # So we monkey-patch _run to immediately shut down
            original_run = core._run
            async def patched_run():
                core._bus.subscribe("system.started", shutdown_after_start)
                await original_run()
            core._run = patched_run
            await core.start()

        await run()
        assert core.state == LifecycleState.STOPPED

    @pytest.mark.asyncio
    async def test_settings_raises_before_start(self, tmp_path: Path):
        """Accessing settings before start() should raise RuntimeError."""
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("", encoding="utf-8")
        core = SpidyCore(config_path=cfg_file)
        with pytest.raises(RuntimeError):
            _ = core.settings

    @pytest.mark.asyncio
    async def test_bus_raises_before_start(self, tmp_path: Path):
        """Accessing bus before start() should raise RuntimeError."""
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("", encoding="utf-8")
        core = SpidyCore(config_path=cfg_file)
        with pytest.raises(RuntimeError):
            _ = core.bus


class TestLifecycleStateTransitions:
    """LifecycleState.can_transition_to() should enforce the state machine."""

    def test_created_can_go_to_initialising(self):
        assert LifecycleState.CREATED.can_transition_to(LifecycleState.INITIALISING)

    def test_created_cannot_go_to_running(self):
        assert not LifecycleState.CREATED.can_transition_to(LifecycleState.RUNNING)

    def test_running_can_go_to_stopping(self):
        assert LifecycleState.RUNNING.can_transition_to(LifecycleState.STOPPING)

    def test_stopped_cannot_go_anywhere(self):
        for state in LifecycleState:
            assert not LifecycleState.STOPPED.can_transition_to(state)

    def test_any_state_can_go_to_error(self):
        # All states except STOPPED and ERROR itself should be able to go to ERROR
        for state in (LifecycleState.CREATED, LifecycleState.INITIALISING,
                      LifecycleState.READY, LifecycleState.RUNNING, LifecycleState.STOPPING):
            assert state.can_transition_to(LifecycleState.ERROR)
