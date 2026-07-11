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
# 4. T2 — Requires Confirmation (M6+)
# ──────────────────────────────────────────────────────────────────────────────

class TestPermissionT2:
    @pytest.mark.asyncio
    async def test_t2_denied_without_bus(self):
        """T2 without a bus → instant deny (no UI can respond)."""
        manager = _make_manager(bus=None)
        result = await manager.check("delete_file", tier="T2")
        assert not result

    @pytest.mark.asyncio
    async def test_t2_denied_consistently_without_bus(self):
        """T2 without bus → always instantly denied."""
        manager = _make_manager(bus=None)
        for _ in range(3):
            assert not await manager.check("delete_file", tier="T2")

    @pytest.mark.asyncio
    async def test_t2_approved_via_respond_to_confirmation(self):
        """T2 with bus → approved when respond_to_confirmation(approved=True) is called."""
        from spidy.core.event_bus import EventBus
        bus = EventBus()
        config = _make_config()
        # Short timeout so test doesn't hang if something goes wrong
        config.t2_confirmation_timeout_seconds = 2.0
        manager = PermissionManager(config=config, bus=bus)

        # Schedule the approval after a short delay (simulating user clicking OK)
        async def _approve():
            await asyncio.sleep(0.05)
            manager.respond_to_confirmation("close_app", "sess-t2", approved=True)

        check_task = asyncio.ensure_future(manager.check("close_app", tier="T2", session_id="sess-t2"))
        asyncio.ensure_future(_approve())
        result = await check_task
        assert result

    @pytest.mark.asyncio
    async def test_t2_denied_via_respond_to_confirmation(self):
        """T2 with bus → denied when respond_to_confirmation(approved=False) is called."""
        from spidy.core.event_bus import EventBus
        bus = EventBus()
        config = _make_config()
        config.t2_confirmation_timeout_seconds = 2.0
        manager = PermissionManager(config=config, bus=bus)

        async def _deny():
            await asyncio.sleep(0.05)
            manager.respond_to_confirmation("close_app", "sess-t2-deny", approved=False)

        check_task = asyncio.ensure_future(
            manager.check("close_app", tier="T2", session_id="sess-t2-deny")
        )
        asyncio.ensure_future(_deny())
        result = await check_task
        assert not result

    @pytest.mark.asyncio
    async def test_t2_times_out_and_denies(self):
        """T2 with bus but no response → denied after timeout."""
        from spidy.core.event_bus import EventBus
        bus = EventBus()
        config = _make_config()
        config.t2_confirmation_timeout_seconds = 0.1  # Very short timeout
        manager = PermissionManager(config=config, bus=bus)

        result = await manager.check("risky_action", tier="T2", session_id="sess-timeout")
        assert not result


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
        # Without bus → instant deny
        manager = _make_manager(require=["some_action"], bus=None)
        result = await manager.check("some_action", tier="T0")
        assert not result

    @pytest.mark.asyncio
    async def test_t1_action_overridden_to_t2(self):
        manager = _make_manager(require=["take_note"], bus=None)
        result = await manager.check("take_note", tier="T1")
        assert not result

    @pytest.mark.asyncio
    async def test_t2_action_not_downgraded(self):
        # T2 action not in require list stays T2 → instant deny (no bus)
        manager = _make_manager(require=[], bus=None)
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
        """T2 with bus: a PermissionRequestedEvent is emitted; result is denied if no response."""
        from spidy.core.event_bus import EventBus
        from spidy.permissions.events import PermissionRequestedEvent

        bus = EventBus()
        events = []
        bus.subscribe("permission.requested", lambda e: events.append(e))

        config = _make_config()
        config.t2_confirmation_timeout_seconds = 0.1  # Fast timeout
        manager = PermissionManager(config=config, bus=bus)
        result = await manager.check("delete_file", tier="T2", session_id="s2")

        await asyncio.sleep(0)
        assert not result  # Timed out → denied
        assert any(isinstance(e, PermissionRequestedEvent) for e in events)

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
            PermissionResponseEvent,
        )
        assert PermissionGrantedEvent.topic == "permission.granted"
        assert PermissionDeniedEvent.topic == "permission.denied"
        assert PermissionRequestedEvent.topic == "permission.requested"
        assert PermissionResponseEvent.topic == "permission.response"


# ──────────────────────────────────────────────────────────────────────────────
# 8. T2 via EventBus PermissionResponseEvent (M6)
# ──────────────────────────────────────────────────────────────────────────────

class TestPermissionT2ViaBus:
    @pytest.mark.asyncio
    async def test_t2_resolved_via_event_bus(self):
        """Simulates the UI publishing a PermissionResponseEvent to approve a T2 action."""
        from spidy.core.event_bus import EventBus
        from spidy.permissions.events import PermissionResponseEvent

        bus = EventBus()
        config = _make_config()
        config.t2_confirmation_timeout_seconds = 2.0
        manager = PermissionManager(config=config, bus=bus)

        async def _ui_approve():
            await asyncio.sleep(0.05)
            await bus.publish(PermissionResponseEvent(
                action="empty_recycle_bin",
                session_id="ui-session",
                approved=True,
            ))

        check_task = asyncio.ensure_future(
            manager.check("empty_recycle_bin", tier="T2", session_id="ui-session")
        )
        asyncio.ensure_future(_ui_approve())
        result = await check_task
        assert result

    @pytest.mark.asyncio
    async def test_t2_rejected_via_event_bus(self):
        """UI publishes PermissionResponseEvent with approved=False."""
        from spidy.core.event_bus import EventBus
        from spidy.permissions.events import PermissionResponseEvent

        bus = EventBus()
        config = _make_config()
        config.t2_confirmation_timeout_seconds = 2.0
        manager = PermissionManager(config=config, bus=bus)

        async def _ui_reject():
            await asyncio.sleep(0.05)
            await bus.publish(PermissionResponseEvent(
                action="sleep_system",
                session_id="ui-session-deny",
                approved=False,
            ))

        check_task = asyncio.ensure_future(
            manager.check("sleep_system", tier="T2", session_id="ui-session-deny")
        )
        asyncio.ensure_future(_ui_reject())
        result = await check_task
        assert not result
