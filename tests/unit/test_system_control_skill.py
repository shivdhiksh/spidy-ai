"""
Tests for Milestone 6 — SystemControlSkill
============================================
Unit tests for all SystemControlSkill actions:
- set_volume
- set_brightness
- lock_workstation
- sleep_system
- shutdown_system
- restart_system
- empty_recycle_bin
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.skills.base import SkillContext, SkillResult
from spidy.skills.desktop.system_control_skill import SystemControlSkill


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ctx(action: str = "", params: dict | None = None, session_id: str = "test") -> SkillContext:
    return SkillContext(action=action, params=params or {}, session_id=session_id)


def _make_skill(bus=None) -> SystemControlSkill:
    return SystemControlSkill(bus=bus)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Capability declarations
# ──────────────────────────────────────────────────────────────────────────────


class TestSystemControlSkillCapabilities:
    def test_all_actions_declared(self):
        skill = _make_skill()
        actions = set(skill.capability_names())
        expected = {
            "set_volume", "set_brightness", "lock_workstation",
            "sleep_system", "shutdown_system", "restart_system",
            "empty_recycle_bin",
            "show_desktop",   # M13.2
        }
        assert expected == actions

    def test_t1_actions(self):
        skill = _make_skill()
        t1_actions = {"set_volume", "set_brightness", "lock_workstation"}
        for cap in skill.capabilities():
            if cap.action in t1_actions:
                assert cap.permission_tier == "T1", f"{cap.action} should be T1"

    def test_t2_actions(self):
        skill = _make_skill()
        t2_actions = {"sleep_system", "empty_recycle_bin"}
        for cap in skill.capabilities():
            if cap.action in t2_actions:
                assert cap.permission_tier == "T2", f"{cap.action} should be T2"

    def test_t3_actions(self):
        skill = _make_skill()
        t3_actions = {"shutdown_system", "restart_system"}
        for cap in skill.capabilities():
            if cap.action in t3_actions:
                assert cap.permission_tier == "T3", f"{cap.action} should be T3"

    def test_skill_name_and_version(self):
        assert SystemControlSkill.name == "system_control_skill"
        assert SystemControlSkill.version == "1.0.0"

    def test_all_actions_supported(self):
        skill = _make_skill()
        for action in ("set_volume", "set_brightness", "lock_workstation",
                       "sleep_system", "shutdown_system", "restart_system",
                       "empty_recycle_bin"):
            assert skill.supports(action)


# ──────────────────────────────────────────────────────────────────────────────
# 2. set_volume
# ──────────────────────────────────────────────────────────────────────────────


class TestSetVolume:
    @pytest.mark.asyncio
    async def test_set_volume_to_level(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_set_volume", return_value=(75, False)):
            ctx = _ctx("set_volume", {"level": "75"})
            result = await skill.execute("set_volume", ctx)
        assert result.success
        assert "75" in result.message
        assert result.data["level"] == 75

    @pytest.mark.asyncio
    async def test_set_volume_up(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_set_volume", return_value=(60, False)):
            ctx = _ctx("set_volume", {"direction": "up"})
            result = await skill.execute("set_volume", ctx)
        assert result.success

    @pytest.mark.asyncio
    async def test_set_volume_down(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_set_volume", return_value=(40, False)):
            ctx = _ctx("set_volume", {"direction": "down"})
            result = await skill.execute("set_volume", ctx)
        assert result.success

    @pytest.mark.asyncio
    async def test_set_volume_mute(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_set_volume", return_value=(50, True)):
            ctx = _ctx("set_volume", {"direction": "mute"})
            result = await skill.execute("set_volume", ctx)
        assert result.success
        assert result.data["muted"] is True

    @pytest.mark.asyncio
    async def test_set_volume_unmute(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_set_volume", return_value=(50, False)):
            ctx = _ctx("set_volume", {"direction": "unmute"})
            result = await skill.execute("set_volume", ctx)
        assert result.success
        assert "unmuted" in result.message.lower()

    @pytest.mark.asyncio
    async def test_set_volume_pycaw_not_installed(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_set_volume",
                          side_effect=RuntimeError("pycaw is not installed")):
            ctx = _ctx("set_volume", {"level": "50"})
            result = await skill.execute("set_volume", ctx)
        assert not result.success
        assert "pycaw" in result.message.lower()

    @pytest.mark.asyncio
    async def test_set_volume_publishes_event(self):
        bus = MagicMock()
        bus.publish = AsyncMock()
        skill = _make_skill(bus=bus)
        with patch.object(SystemControlSkill, "_do_set_volume", return_value=(50, False)):
            ctx = _ctx("set_volume", {"level": "50"})
            await skill.execute("set_volume", ctx)
        bus.publish.assert_called_once()
        from spidy.skills.desktop.events import VolumeChangedEvent
        assert isinstance(bus.publish.call_args[0][0], VolumeChangedEvent)


# ──────────────────────────────────────────────────────────────────────────────
# 3. set_brightness
# ──────────────────────────────────────────────────────────────────────────────


class TestSetBrightness:
    @pytest.mark.asyncio
    async def test_set_brightness_success(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_set_brightness", return_value=70):
            ctx = _ctx("set_brightness", {"level": "70"})
            result = await skill.execute("set_brightness", ctx)
        assert result.success
        assert "70" in result.message

    @pytest.mark.asyncio
    async def test_set_brightness_clamps_to_range(self):
        skill = _make_skill()
        # Level 150 should be clamped to 100
        with patch.object(SystemControlSkill, "_do_set_brightness", return_value=100) as mock_fn:
            ctx = _ctx("set_brightness", {"level": "150"})
            result = await skill.execute("set_brightness", ctx)
        assert result.success
        # Should have been called with 100 (clamped)
        mock_fn.assert_called_with(100)

    @pytest.mark.asyncio
    async def test_set_brightness_requires_level(self):
        skill = _make_skill()
        result = await skill.execute("set_brightness", _ctx("set_brightness", {}))
        assert not result.success

    @pytest.mark.asyncio
    async def test_set_brightness_not_supported(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_set_brightness",
                          side_effect=RuntimeError("brightness control is not supported")):
            ctx = _ctx("set_brightness", {"level": "50"})
            result = await skill.execute("set_brightness", ctx)
        assert not result.success

    @pytest.mark.asyncio
    async def test_set_brightness_publishes_event(self):
        bus = MagicMock()
        bus.publish = AsyncMock()
        skill = _make_skill(bus=bus)
        with patch.object(SystemControlSkill, "_do_set_brightness", return_value=60):
            ctx = _ctx("set_brightness", {"level": "60"})
            await skill.execute("set_brightness", ctx)
        bus.publish.assert_called_once()
        from spidy.skills.desktop.events import BrightnessChangedEvent
        assert isinstance(bus.publish.call_args[0][0], BrightnessChangedEvent)


# ──────────────────────────────────────────────────────────────────────────────
# 4. lock_workstation
# ──────────────────────────────────────────────────────────────────────────────


class TestLockWorkstation:
    @pytest.mark.asyncio
    async def test_lock_success(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_lock_workstation", return_value=True):
            result = await skill.execute("lock_workstation", _ctx("lock_workstation"))
        assert result.success
        assert "locked" in result.message.lower()

    @pytest.mark.asyncio
    async def test_lock_failure(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_lock_workstation", return_value=False):
            result = await skill.execute("lock_workstation", _ctx("lock_workstation"))
        assert not result.success

    @pytest.mark.asyncio
    async def test_lock_not_supported_non_windows(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_lock_workstation",
                          side_effect=RuntimeError("only supported on Windows")):
            result = await skill.execute("lock_workstation", _ctx("lock_workstation"))
        assert not result.success

    @pytest.mark.asyncio
    async def test_lock_publishes_event(self):
        bus = MagicMock()
        bus.publish = AsyncMock()
        skill = _make_skill(bus=bus)
        with patch.object(SystemControlSkill, "_do_lock_workstation", return_value=True):
            await skill.execute("lock_workstation", _ctx("lock_workstation"))
        bus.publish.assert_called_once()
        from spidy.skills.desktop.events import WorkstationLockedEvent
        assert isinstance(bus.publish.call_args[0][0], WorkstationLockedEvent)


# ──────────────────────────────────────────────────────────────────────────────
# 5. sleep_system
# ──────────────────────────────────────────────────────────────────────────────


class TestSleepSystem:
    @pytest.mark.asyncio
    async def test_sleep_success(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_sleep", return_value=None):
            result = await skill.execute("sleep_system", _ctx("sleep_system"))
        assert result.success
        assert "sleep" in result.message.lower()

    @pytest.mark.asyncio
    async def test_sleep_publishes_event(self):
        bus = MagicMock()
        bus.publish = AsyncMock()
        skill = _make_skill(bus=bus)
        with patch.object(SystemControlSkill, "_do_sleep", return_value=None):
            await skill.execute("sleep_system", _ctx("sleep_system"))
        bus.publish.assert_called_once()
        from spidy.skills.desktop.events import SystemSleepInitiatedEvent
        assert isinstance(bus.publish.call_args[0][0], SystemSleepInitiatedEvent)

    @pytest.mark.asyncio
    async def test_sleep_failure(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_sleep",
                          side_effect=Exception("Cannot sleep")):
            result = await skill.execute("sleep_system", _ctx("sleep_system"))
        assert not result.success


# ──────────────────────────────────────────────────────────────────────────────
# 6. shutdown_system
# ──────────────────────────────────────────────────────────────────────────────


class TestShutdownSystem:
    @pytest.mark.asyncio
    async def test_shutdown_immediate(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_shutdown", return_value=None) as mock_fn:
            result = await skill.execute("shutdown_system", _ctx("shutdown_system", {"delay": "0"}))
        assert result.success
        assert "shutting down" in result.message.lower()
        mock_fn.assert_called_with(0, restart=False)

    @pytest.mark.asyncio
    async def test_shutdown_with_delay(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_shutdown", return_value=None):
            result = await skill.execute("shutdown_system",
                                         _ctx("shutdown_system", {"delay": "60"}))
        assert result.success
        assert "60" in result.message
        assert "abort" in result.message.lower()

    @pytest.mark.asyncio
    async def test_shutdown_publishes_event(self):
        bus = MagicMock()
        bus.publish = AsyncMock()
        skill = _make_skill(bus=bus)
        with patch.object(SystemControlSkill, "_do_shutdown", return_value=None):
            await skill.execute("shutdown_system", _ctx("shutdown_system"))
        bus.publish.assert_called_once()
        from spidy.skills.desktop.events import SystemShutdownInitiatedEvent
        assert isinstance(bus.publish.call_args[0][0], SystemShutdownInitiatedEvent)


# ──────────────────────────────────────────────────────────────────────────────
# 7. restart_system
# ──────────────────────────────────────────────────────────────────────────────


class TestRestartSystem:
    @pytest.mark.asyncio
    async def test_restart_immediate(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_shutdown", return_value=None) as mock_fn:
            result = await skill.execute("restart_system", _ctx("restart_system"))
        assert result.success
        assert "restart" in result.message.lower()
        mock_fn.assert_called_with(0, restart=True)

    @pytest.mark.asyncio
    async def test_restart_with_delay(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_shutdown", return_value=None):
            result = await skill.execute("restart_system",
                                         _ctx("restart_system", {"delay": "30"}))
        assert result.success
        assert "30" in result.message

    @pytest.mark.asyncio
    async def test_restart_publishes_event(self):
        bus = MagicMock()
        bus.publish = AsyncMock()
        skill = _make_skill(bus=bus)
        with patch.object(SystemControlSkill, "_do_shutdown", return_value=None):
            await skill.execute("restart_system", _ctx("restart_system"))
        bus.publish.assert_called_once()
        from spidy.skills.desktop.events import SystemRestartInitiatedEvent
        assert isinstance(bus.publish.call_args[0][0], SystemRestartInitiatedEvent)


# ──────────────────────────────────────────────────────────────────────────────
# 8. empty_recycle_bin
# ──────────────────────────────────────────────────────────────────────────────


class TestEmptyRecycleBin:
    @pytest.mark.asyncio
    async def test_empty_success(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_empty_recycle_bin", return_value=None):
            result = await skill.execute("empty_recycle_bin", _ctx("empty_recycle_bin"))
        assert result.success
        assert "emptied" in result.message.lower()

    @pytest.mark.asyncio
    async def test_empty_failure(self):
        skill = _make_skill()
        with patch.object(SystemControlSkill, "_do_empty_recycle_bin",
                          side_effect=RuntimeError("only supported on Windows")):
            result = await skill.execute("empty_recycle_bin", _ctx("empty_recycle_bin"))
        assert not result.success

    @pytest.mark.asyncio
    async def test_empty_publishes_event(self):
        bus = MagicMock()
        bus.publish = AsyncMock()
        skill = _make_skill(bus=bus)
        with patch.object(SystemControlSkill, "_do_empty_recycle_bin", return_value=None):
            await skill.execute("empty_recycle_bin", _ctx("empty_recycle_bin"))
        bus.publish.assert_called_once()
        from spidy.skills.desktop.events import RecycleBinEmptiedEvent
        assert isinstance(bus.publish.call_args[0][0], RecycleBinEmptiedEvent)


# ──────────────────────────────────────────────────────────────────────────────
# 9. Unknown action
# ──────────────────────────────────────────────────────────────────────────────


class TestUnknownAction:
    @pytest.mark.asyncio
    async def test_unknown_action_fails_gracefully(self):
        skill = _make_skill()
        result = await skill.execute("do_something_weird", _ctx("do_something_weird"))
        assert not result.success
        assert "unknown action" in result.message.lower()
