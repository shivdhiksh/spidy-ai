"""
tests/unit/test_goal_manager.py — Unit tests for GoalManager
=============================================================
Tests goal CRUD, state transitions, event publishing, history management,
and error cases.
"""

from __future__ import annotations

import asyncio

import pytest

from spidy.agent.goal_manager import GoalManager
from spidy.agent.types import GoalState, TaskRecord, TaskState
from spidy.core.event_bus import EventBus


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_bus() -> EventBus:
    bus = EventBus()
    loop = asyncio.get_running_loop()
    bus.set_loop(loop)
    return bus


def make_manager(**kwargs) -> GoalManager:
    bus = make_bus()
    return GoalManager(bus=bus, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Goal creation
# ═══════════════════════════════════════════════════════════════════════════════


class TestGoalCreation:
    @pytest.mark.asyncio
    async def test_create_goal_returns_record(self) -> None:
        mgr = make_manager()
        goal = await mgr.create_goal("Create a Flask project")
        assert goal.description == "Create a Flask project"
        assert goal.state == GoalState.PENDING
        assert goal.goal_id

    @pytest.mark.asyncio
    async def test_create_goal_publishes_event(self) -> None:
        bus = make_bus()
        mgr = GoalManager(bus=bus)

        events = []
        bus.subscribe("agent.goal_created", lambda e: events.append(e))

        await mgr.create_goal("Test goal")

        assert len(events) == 1
        assert events[0].description == "Test goal"

    @pytest.mark.asyncio
    async def test_create_goal_sets_active(self) -> None:
        mgr = make_manager()
        assert mgr.active_goal is None
        goal = await mgr.create_goal("Goal 1")
        assert mgr.active_goal is not None
        assert mgr.active_goal.goal_id == goal.goal_id

    @pytest.mark.asyncio
    async def test_create_second_goal_raises_while_active(self) -> None:
        mgr = make_manager()
        await mgr.create_goal("Goal 1")
        # Before transitioning to terminal state, creating another raises
        with pytest.raises(RuntimeError, match="already active"):
            await mgr.create_goal("Goal 2")

    @pytest.mark.asyncio
    async def test_create_goal_after_completion(self) -> None:
        mgr = make_manager()
        goal = await mgr.create_goal("Goal 1")
        await mgr.complete_goal(goal.goal_id)
        # Now a new goal can be created
        goal2 = await mgr.create_goal("Goal 2")
        assert goal2.description == "Goal 2"


# ═══════════════════════════════════════════════════════════════════════════════
# State transitions
# ═══════════════════════════════════════════════════════════════════════════════


class TestStateTransitions:
    @pytest.mark.asyncio
    async def test_begin_planning(self) -> None:
        mgr = make_manager()
        goal = await mgr.create_goal("Goal")
        planning = await mgr.begin_planning(goal.goal_id)
        assert planning.state == GoalState.PLANNING

    @pytest.mark.asyncio
    async def test_begin_execution(self) -> None:
        mgr = make_manager()
        goal = await mgr.create_goal("Goal")
        await mgr.begin_planning(goal.goal_id)
        tasks = [
            TaskRecord(description="t1", utterance="u1"),
            TaskRecord(description="t2", utterance="u2"),
        ]
        executing = await mgr.begin_execution(goal.goal_id, tasks=tasks)
        assert executing.state == GoalState.EXECUTING
        assert len(executing.tasks) == 2

    @pytest.mark.asyncio
    async def test_begin_execution_publishes_started_event(self) -> None:
        bus = make_bus()
        mgr = GoalManager(bus=bus)
        events = []
        bus.subscribe("agent.goal_started", lambda e: events.append(e))

        goal = await mgr.create_goal("Goal")
        await mgr.begin_planning(goal.goal_id)
        await mgr.begin_execution(goal.goal_id, tasks=[
            TaskRecord(description="t1", utterance="u1"),
        ])

        assert len(events) == 1
        assert events[0].task_count == 1

    @pytest.mark.asyncio
    async def test_complete_goal(self) -> None:
        mgr = make_manager()
        goal = await mgr.create_goal("Goal")
        completed = await mgr.complete_goal(goal.goal_id, summary="All done!")
        assert completed.state == GoalState.COMPLETED
        assert completed.summary == "All done!"
        assert completed.completed_at is not None
        assert mgr.active_goal is None

    @pytest.mark.asyncio
    async def test_fail_goal(self) -> None:
        mgr = make_manager()
        goal = await mgr.create_goal("Goal")
        failed = await mgr.fail_goal(goal.goal_id, error="Network error")
        assert failed.state == GoalState.FAILED
        assert failed.error == "Network error"
        assert mgr.active_goal is None

    @pytest.mark.asyncio
    async def test_cancel_goal(self) -> None:
        mgr = make_manager()
        goal = await mgr.create_goal("Goal")
        cancelled = await mgr.cancel_goal(goal.goal_id)
        assert cancelled.state == GoalState.CANCELLED
        assert mgr.active_goal is None

    @pytest.mark.asyncio
    async def test_cancel_publishes_event(self) -> None:
        bus = make_bus()
        mgr = GoalManager(bus=bus)
        events = []
        bus.subscribe("agent.goal_cancelled", lambda e: events.append(e))

        goal = await mgr.create_goal("Goal")
        await mgr.cancel_goal(goal.goal_id)

        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_update_task(self) -> None:
        mgr = make_manager()
        task = TaskRecord(description="task", utterance="do it")
        goal = await mgr.create_goal("Goal")
        await mgr.begin_execution(goal.goal_id, tasks=[task])

        updated_task = task.mark_completed("Done!")
        updated_goal = await mgr.update_task(goal.goal_id, updated_task)

        assert updated_goal.tasks[0].state == TaskState.COMPLETED


# ═══════════════════════════════════════════════════════════════════════════════
# History
# ═══════════════════════════════════════════════════════════════════════════════


class TestHistory:
    @pytest.mark.asyncio
    async def test_completed_goal_in_history(self) -> None:
        mgr = make_manager()
        goal = await mgr.create_goal("Goal 1")
        await mgr.complete_goal(goal.goal_id)

        history = mgr.get_history()
        assert len(history) == 1
        assert history[0].state == GoalState.COMPLETED

    @pytest.mark.asyncio
    async def test_history_respects_limit(self) -> None:
        mgr = make_manager()
        goal = await mgr.create_goal("Goal 1")
        await mgr.complete_goal(goal.goal_id)
        history = mgr.get_history(limit=1)
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_max_history_cap(self) -> None:
        mgr = make_manager(max_history=3)
        for i in range(5):
            goal = await mgr.create_goal(f"Goal {i}")
            await mgr.complete_goal(goal.goal_id)

        assert len(mgr.get_history()) == 3

    @pytest.mark.asyncio
    async def test_find_goal_in_history(self) -> None:
        mgr = make_manager()
        goal = await mgr.create_goal("Goal")
        await mgr.complete_goal(goal.goal_id)

        found = mgr.find_goal(goal.goal_id)
        assert found is not None
        assert found.goal_id == goal.goal_id

    @pytest.mark.asyncio
    async def test_find_active_goal(self) -> None:
        mgr = make_manager()
        goal = await mgr.create_goal("Active goal")
        found = mgr.find_goal(goal.goal_id)
        assert found is not None

    @pytest.mark.asyncio
    async def test_find_unknown_goal_returns_none(self) -> None:
        mgr = make_manager()
        assert mgr.find_goal("nonexistent-id") is None


# ═══════════════════════════════════════════════════════════════════════════════
# Error cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestErrorCases:
    @pytest.mark.asyncio
    async def test_wrong_goal_id_raises(self) -> None:
        mgr = make_manager()
        await mgr.create_goal("Goal")
        with pytest.raises(RuntimeError):
            await mgr.complete_goal("wrong-id")

    @pytest.mark.asyncio
    async def test_events_published_for_fail(self) -> None:
        bus = make_bus()
        mgr = GoalManager(bus=bus)
        events = []
        bus.subscribe("agent.goal_failed", lambda e: events.append(e))

        goal = await mgr.create_goal("Goal")
        await mgr.fail_goal(goal.goal_id, error="Boom")

        assert len(events) == 1
        assert events[0].error == "Boom"

    @pytest.mark.asyncio
    async def test_completed_event_published(self) -> None:
        bus = make_bus()
        mgr = GoalManager(bus=bus)
        events = []
        bus.subscribe("agent.goal_completed", lambda e: events.append(e))

        goal = await mgr.create_goal("Goal")
        await mgr.complete_goal(goal.goal_id, summary="Great!")

        assert len(events) == 1
        assert events[0].summary == "Great!"
