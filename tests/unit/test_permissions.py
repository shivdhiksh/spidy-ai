"""
Tests for Milestone 4 — Permission Manager
===========================================
Unit tests for PermissionTier enum, PermissionManager tier evaluation,
event publishing, and session management.
"""

from __future__ import annotations

import asyncio

import pytest

from spidy.permissions.manager import PermissionManager, PermissionTier


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_config(require_confirmation_for: list[str] | None = None):
    from spidy.config.manager import PermissionsConfig
    return PermissionsConfig(
        require_confirmation_for=require_confirmation_for or [],
        audit_log_enabled=False,  # Don't write files in tests
    )


def _make_manager(require: list[str] | None = None, bus=None):
    return PermissionManager(config=_make_config(require), bus=bus)


# ──────────────────────────────────────────────────────────────────────────────
# 1. PermissionTier Enum
# ──────────────────────────────────────────────────────────────────────────────

class TestPermissionTierEnum:
    def test_tier_values(self):
        assert PermissionTier.T0.value == "T0"
        assert PermissionTier.T1.value == "T1"
        assert PermissionTier.T2.value == "T2"
        assert PermissionTier.T3.value == "T3"

    def test_tier_ordering(self):
        assert PermissionTier.T0 < PermissionTier.T1
        assert PermissionTier.T1 < PermissionTier.T2
        assert PermissionTier.T2 < PermissionTier.T3

    def test_tier_lte(self):
        assert PermissionTier.T0 <= PermissionTier.T0
        assert PermissionTier.T0 <= PermissionTier.T1

    def test_tier_ordinal(self):
        assert PermissionTier.T0.ordinal == 0
        assert PermissionTier.T1.ordinal == 1
        assert PermissionTier.T2.ordinal == 2
        assert PermissionTier.T3.ordinal == 3

    def test_tier_from_string(self):
        assert PermissionTier("T0") == PermissionTier.T0
        assert PermissionTier("T2") == PermissionTier.T2


# ──────────────────────────────────────────────────────────────────────────────
# 2. T0 — Always Allowed
# ──────────────────────────────────────────────────────────────────────────────

class TestPermissionT0:
    @pytest.mark.asyncio
    async def test_t0_always_granted(self):
        manager = _make_manager()
        assert await manager.check("show_help", tier="T0")

    @pytest.mark.asyncio
    async def test_t0_no_session_needed(self):
        manager = _make_manager()
        assert await manager.check("get_time", tier="T0", session_id="")

    @pytest.mark.asyncio
    async def test_t0_multiple_calls(self):
        manager = _make_manager()
        for _ in range(5):
            assert await manager.check("get_time", tier="T0")


# ──────────────────────────────────────────────────────────────────────────────
# 3. T1 — Auto-Approved Per Session
# ──────────────────────────────────────────────────────────────────────────────

class TestPermissionT1:
    @pytest.mark.asyncio
    async def test_t1_auto_approved(self):
        manager = _make_manager()
        assert await manager.check("take_note", tier="T1", session_id="sess-1")

    @pytest.mark.asyncio
    async def test_t1_approved_multiple_times_same_session(self):
        manager = _make_manager()
        # Should be approved both times (auto-approve T1 in M4)
        assert await manager.check("take_note", tier="T1", session_id="sess-1")
        assert await manager.check("take_note", tier="T1", session_id="sess-1")

    @pytest.mark.asyncio
    async def test_t1_tracks_per_session(self):
        manager = _make_manager()
        await manager.check("take_note", tier="T1", session_id="sess-A")
        assert ("take_note", "sess-A") in manager._t1_confirmed

    @pytest.mark.asyncio
    async def test_reset_session_clears_t1(self):
        manager = _make_manager()
        await manager.check("take_note", tier="T1", session_id="sess-X")
        manager.reset_session("sess-X")
        assert ("take_note", "sess-X") not in manager._t1_confirmed


# ──────────────────────────────────────────────────────────────────────────────
# 4. T2 — Auto-Denied
# ──────────────────────────────────────────────────────────────────────────────

class TestPermissionT2:
    @pytest.mark.asyncio
    async def test_t2_auto_denied(self):
        manager = _make_manager()
        result = await manager.check("delete_file", tier="T2")
        assert not result

    @pytest.mark.asyncio
    async def test_t2_denied_consistently(self):
        manager = _make_manager()
        for _ in range(3):
            assert not await manager.check("delete_file", tier="T2")


# ──────────────────────────────────────────────────────────────────────────────
# 5. T3 — Locked by Default
# ──────────────────────────────────────────────────────────────────────────────

class TestPermissionT3:
    @pytest.mark.asyncio
    async def test_t3_denied_without_unlock(self):
        manager = _make_manager()
        assert not await manager.check("admin_action", tier="T3")

    @pytest.mark.asyncio
    async def test_t3_granted_after_unlock(self):
        manager = _make_manager()
        manager.unlock_t3("admin_action")
        assert await manager.check("admin_action", tier="T3")

    @pytest.mark.asyncio
    async def test_t3_unlock_is_action_specific(self):
        manager = _make_manager()
        manager.unlock_t3("admin_action")
        # other_action is still locked
        assert not await manager.check("other_action", tier="T3")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Policy Overrides
# ──────────────────────────────────────────────────────────────────────────────

class TestPermissionPolicyOverride:
    @pytest.mark.asyncio
    async def test_t0_action_overridden_to_t2(self):
        # Action declared T0 but in require_confirmation_for → treated as T2
        manager = _make_manager(require=["some_action"])
        result = await manager.check("some_action", tier="T0")
        assert not result  # T2 = auto-deny in M4

    @pytest.mark.asyncio
    async def test_t1_action_overridden_to_t2(self):
        manager = _make_manager(require=["take_note"])
        result = await manager.check("take_note", tier="T1")
        assert not result

    @pytest.mark.asyncio
    async def test_t2_action_not_downgraded(self):
        # T2 action not in require list stays T2
        manager = _make_manager(require=[])
        result = await manager.check("delete_all", tier="T2")
        assert not result


# ──────────────────────────────────────────────────────────────────────────────
# 7. EventBus Integration
# ──────────────────────────────────────────────────────────────────────────────

class TestPermissionEvents:
    @pytest.mark.asyncio
    async def test_granted_event_published(self):
        from spidy.core.event_bus import EventBus
        from spidy.permissions.events import PermissionGrantedEvent

        bus = EventBus()
        events = []
        bus.subscribe("permission.granted", lambda e: events.append(e))

        manager = _make_manager(bus=bus)
        await manager.check("get_time", tier="T0", session_id="s1")

        await asyncio.sleep(0)
        assert any(isinstance(e, PermissionGrantedEvent) for e in events)

    @pytest.mark.asyncio
    async def test_denied_event_published(self):
        from spidy.core.event_bus import EventBus
        from spidy.permissions.events import PermissionDeniedEvent

        bus = EventBus()
        events = []
        bus.subscribe("permission.denied", lambda e: events.append(e))

        manager = _make_manager(bus=bus)
        await manager.check("delete_file", tier="T2", session_id="s2")

        await asyncio.sleep(0)
        assert any(isinstance(e, PermissionDeniedEvent) for e in events)

    @pytest.mark.asyncio
    async def test_no_event_bus_does_not_crash(self):
        manager = _make_manager(bus=None)
        # Should not raise even without a bus
        result = await manager.check("get_time", tier="T0")
        assert result

    def test_permission_events_topics(self):
        from spidy.permissions.events import (
            PermissionDeniedEvent,
            PermissionGrantedEvent,
            PermissionRequestedEvent,
        )
        assert PermissionGrantedEvent.topic == "permission.granted"
        assert PermissionDeniedEvent.topic == "permission.denied"
        assert PermissionRequestedEvent.topic == "permission.requested"
