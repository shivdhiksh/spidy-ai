"""
tests/unit/test_progress_tracker.py — Unit tests for ProgressTracker
=====================================================================
Tests event publishing, percentage calculation, and lifecycle methods.
"""

from __future__ import annotations

import asyncio

import pytest

from spidy.agent.progress_tracker import ProgressTracker
from spidy.core.event_bus import EventBus


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_bus() -> EventBus:
    """Create a bus; sets loop only if one is running (async context)."""
    bus = EventBus()
    try:
        loop = asyncio.get_running_loop()
        bus.set_loop(loop)
    except RuntimeError:
        pass  # sync context — no loop needed for these tests
    return bus


def make_tracker() -> tuple[ProgressTracker, EventBus]:
    bus = make_bus()
    tracker = ProgressTracker(bus=bus)
    return tracker, bus


# ═══════════════════════════════════════════════════════════════════════════════
# Initialisation
# ═══════════════════════════════════════════════════════════════════════════════


class TestProgressTrackerInit:
    def test_start_sets_total(self) -> None:
        bus = EventBus()  # plain bus — no loop needed for sync init
        tracker = ProgressTracker(bus=bus)
        tracker.start(goal_id="g1", total_tasks=5, session_id="s1")
        assert tracker._total_tasks == 5
        assert tracker._goal_id == "g1"
        assert tracker._session_id == "s1"

    def test_start_with_zero_tasks_safe(self) -> None:
        bus = EventBus()
        tracker = ProgressTracker(bus=bus)
        tracker.start(goal_id="g1", total_tasks=0)
        assert tracker._total_tasks == 1  # clamped to 1


# ═══════════════════════════════════════════════════════════════════════════════
# Percentage calculation
# ═══════════════════════════════════════════════════════════════════════════════


class TestPercentageCalculation:
    @pytest.mark.asyncio
    async def test_advance_first_task_gives_0_percent(self) -> None:
        tracker, _ = make_tracker()
        tracker.start(goal_id="g1", total_tasks=4)
        pct = await tracker.advance(task_index=0, task_description="Task 1")
        assert pct == 0  # 0/4 before task 0 completes

    @pytest.mark.asyncio
    async def test_advance_second_task_gives_25_percent(self) -> None:
        tracker, _ = make_tracker()
        tracker.start(goal_id="g1", total_tasks=4)
        pct = await tracker.advance(task_index=1, task_description="Task 2")
        assert pct == 25  # 1/4 = 25%

    @pytest.mark.asyncio
    async def test_task_done_gives_correct_percent(self) -> None:
        tracker, _ = make_tracker()
        tracker.start(goal_id="g1", total_tasks=4)
        pct = await tracker.task_done(task_index=1, task_description="Task 2")
        assert pct == 50  # 2/4 = 50%

    @pytest.mark.asyncio
    async def test_task_done_last_gives_100(self) -> None:
        tracker, _ = make_tracker()
        tracker.start(goal_id="g1", total_tasks=2)
        pct = await tracker.task_done(task_index=1, task_description="Last task")
        assert pct == 100


# ═══════════════════════════════════════════════════════════════════════════════
# Event publishing
# ═══════════════════════════════════════════════════════════════════════════════


class TestEventPublishing:
    @pytest.mark.asyncio
    async def test_planning_publishes_agent_progress_event(self) -> None:
        tracker, bus = make_tracker()
        tracker.start(goal_id="g1", total_tasks=3, session_id="s1")

        agent_events = []
        bus.subscribe("agent.progress", lambda e: agent_events.append(e))

        await tracker.publish_planning()

        assert len(agent_events) == 1
        assert agent_events[0].message == "Planning..."
        assert agent_events[0].progress_percent == 0

    @pytest.mark.asyncio
    async def test_planning_also_publishes_brain_progress_event(self) -> None:
        tracker, bus = make_tracker()
        tracker.start(goal_id="g1", total_tasks=3, session_id="s1")

        brain_events = []
        bus.subscribe("brain.progress", lambda e: brain_events.append(e))

        await tracker.publish_planning()

        assert len(brain_events) == 1

    @pytest.mark.asyncio
    async def test_advance_publishes_event_with_task_name(self) -> None:
        tracker, bus = make_tracker()
        tracker.start(goal_id="g1", total_tasks=3, session_id="s1")

        events = []
        bus.subscribe("agent.progress", lambda e: events.append(e))

        await tracker.advance(task_index=0, task_description="Installing Flask")

        assert any("Installing Flask" in e.message for e in events)

    @pytest.mark.asyncio
    async def test_completed_publishes_100_percent_event(self) -> None:
        tracker, bus = make_tracker()
        tracker.start(goal_id="g1", total_tasks=3, session_id="s1")

        events = []
        bus.subscribe("agent.progress", lambda e: events.append(e))

        await tracker.publish_completed()

        completion_events = [e for e in events if e.progress_percent == 100]
        assert len(completion_events) >= 1

    @pytest.mark.asyncio
    async def test_failed_publishes_event(self) -> None:
        tracker, bus = make_tracker()
        tracker.start(goal_id="g1", total_tasks=3, session_id="s1")

        events = []
        bus.subscribe("agent.progress", lambda e: events.append(e))

        await tracker.publish_failed("Something broke")

        assert len(events) >= 1
        assert any("went wrong" in e.message.lower() or "broke" in e.message.lower() for e in events)

    @pytest.mark.asyncio
    async def test_cancelled_publishes_event(self) -> None:
        tracker, bus = make_tracker()
        tracker.start(goal_id="g1", total_tasks=3, session_id="s1")

        events = []
        bus.subscribe("agent.progress", lambda e: events.append(e))

        await tracker.publish_cancelled()

        assert len(events) >= 1
        assert any("cancel" in e.message.lower() for e in events)

    @pytest.mark.asyncio
    async def test_agent_progress_event_has_correct_goal_id(self) -> None:
        tracker, bus = make_tracker()
        tracker.start(goal_id="goal-xyz", total_tasks=2, session_id="sess-1")

        events = []
        bus.subscribe("agent.progress", lambda e: events.append(e))

        await tracker.advance(task_index=0, task_description="Task A")

        assert all(e.goal_id == "goal-xyz" for e in events)

    @pytest.mark.asyncio
    async def test_event_publish_failure_does_not_crash(self) -> None:
        """ProgressTracker must never crash the execution loop on event failure."""
        bus = EventBus()  # no loop set — will cause publish failure gracefully

        tracker = ProgressTracker(bus=bus)
        tracker.start(goal_id="g1", total_tasks=3)

        # Should not raise even with a broken bus
        await tracker.publish_planning()  # no assertion — just must not raise
