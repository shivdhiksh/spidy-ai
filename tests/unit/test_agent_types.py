"""
tests/unit/test_agent_types.py — Unit tests for spidy.agent.types
==================================================================
Tests all data types, enumerations, state transition helpers,
and derived properties.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from spidy.agent.types import (
    ExecutionContext,
    GoalRecord,
    GoalState,
    ReflectionDecision,
    TaskRecord,
    TaskState,
)


# ═══════════════════════════════════════════════════════════════════════════════
# GoalState enum
# ═══════════════════════════════════════════════════════════════════════════════

class TestGoalState:
    def test_all_states_exist(self) -> None:
        states = {s.value for s in GoalState}
        assert "pending" in states
        assert "planning" in states
        assert "executing" in states
        assert "completed" in states
        assert "failed" in states
        assert "cancelled" in states
        assert "paused" in states

    def test_state_is_str(self) -> None:
        assert GoalState.COMPLETED == "completed"


# ═══════════════════════════════════════════════════════════════════════════════
# TaskState enum
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskState:
    def test_all_states_exist(self) -> None:
        states = {s.value for s in TaskState}
        assert "pending" in states
        assert "running" in states
        assert "completed" in states
        assert "failed" in states
        assert "skipped" in states


# ═══════════════════════════════════════════════════════════════════════════════
# ReflectionDecision enum
# ═══════════════════════════════════════════════════════════════════════════════

class TestReflectionDecision:
    def test_all_decisions_exist(self) -> None:
        decisions = {d.value for d in ReflectionDecision}
        assert "continue" in decisions
        assert "retry" in decisions
        assert "alternative" in decisions
        assert "ask_user" in decisions
        assert "abort" in decisions


# ═══════════════════════════════════════════════════════════════════════════════
# TaskRecord
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskRecord:
    def _make_task(self, **kwargs) -> TaskRecord:
        return TaskRecord(
            description="Test task",
            utterance="do something",
            **kwargs,
        )

    def test_default_state_is_pending(self) -> None:
        task = self._make_task()
        assert task.state == TaskState.PENDING

    def test_auto_uuid(self) -> None:
        t1 = self._make_task()
        t2 = self._make_task()
        assert t1.task_id != t2.task_id

    def test_mark_running(self) -> None:
        task = self._make_task()
        running = task.mark_running()
        assert running.state == TaskState.RUNNING
        assert running.attempt == 1
        assert running.started_at is not None
        # Original is unchanged
        assert task.state == TaskState.PENDING

    def test_mark_running_increments_attempt(self) -> None:
        task = self._make_task()
        r1 = task.mark_running()
        assert r1.attempt == 1
        # Simulate a second attempt
        r2 = r1.mark_running()
        assert r2.attempt == 2

    def test_mark_completed(self) -> None:
        task = self._make_task()
        completed = task.mark_running().mark_completed("Done!")
        assert completed.state == TaskState.COMPLETED
        assert completed.result_message == "Done!"
        assert completed.result_success is True
        assert completed.completed_at is not None

    def test_mark_failed(self) -> None:
        task = self._make_task()
        failed = task.mark_running().mark_failed(error="Network error", message="Couldn't connect")
        assert failed.state == TaskState.FAILED
        assert failed.result_success is False
        assert failed.error == "Network error"
        assert failed.result_message == "Couldn't connect"

    def test_mark_skipped(self) -> None:
        task = self._make_task()
        skipped = task.mark_skipped("User cancelled")
        assert skipped.state == TaskState.SKIPPED
        assert skipped.result_message == "User cancelled"

    def test_task_ids_preserved_across_transitions(self) -> None:
        task = self._make_task()
        original_id = task.task_id
        running = task.mark_running()
        completed = running.mark_completed("ok")
        assert completed.task_id == original_id


# ═══════════════════════════════════════════════════════════════════════════════
# GoalRecord
# ═══════════════════════════════════════════════════════════════════════════════

class TestGoalRecord:
    def _make_goal(self, **kwargs) -> GoalRecord:
        return GoalRecord(description="Test goal", **kwargs)

    def test_default_state_is_pending(self) -> None:
        goal = self._make_goal()
        assert goal.state == GoalState.PENDING

    def test_auto_uuid(self) -> None:
        g1 = self._make_goal()
        g2 = self._make_goal()
        assert g1.goal_id != g2.goal_id

    def test_is_terminal_on_completed(self) -> None:
        goal = self._make_goal(state=GoalState.COMPLETED)
        assert goal.is_terminal is True

    def test_is_terminal_on_failed(self) -> None:
        goal = self._make_goal(state=GoalState.FAILED)
        assert goal.is_terminal is True

    def test_is_terminal_on_cancelled(self) -> None:
        goal = self._make_goal(state=GoalState.CANCELLED)
        assert goal.is_terminal is True

    def test_not_terminal_on_executing(self) -> None:
        goal = self._make_goal(state=GoalState.EXECUTING)
        assert goal.is_terminal is False

    def test_is_active_on_planning(self) -> None:
        goal = self._make_goal(state=GoalState.PLANNING)
        assert goal.is_active is True

    def test_is_active_on_executing(self) -> None:
        goal = self._make_goal(state=GoalState.EXECUTING)
        assert goal.is_active is True

    def test_not_active_on_completed(self) -> None:
        goal = self._make_goal(state=GoalState.COMPLETED)
        assert goal.is_active is False

    def test_task_count(self) -> None:
        tasks = [
            TaskRecord(description="t1", utterance="u1"),
            TaskRecord(description="t2", utterance="u2"),
        ]
        goal = self._make_goal(tasks=tasks)
        assert goal.task_count == 2

    def test_completed_task_count(self) -> None:
        tasks = [
            TaskRecord(description="t1", utterance="u1", state=TaskState.COMPLETED),
            TaskRecord(description="t2", utterance="u2", state=TaskState.RUNNING),
            TaskRecord(description="t3", utterance="u3", state=TaskState.COMPLETED),
        ]
        goal = self._make_goal(tasks=tasks)
        assert goal.completed_task_count == 2

    def test_failed_task_count(self) -> None:
        tasks = [
            TaskRecord(description="t1", utterance="u1", state=TaskState.FAILED),
            TaskRecord(description="t2", utterance="u2", state=TaskState.COMPLETED),
        ]
        goal = self._make_goal(tasks=tasks)
        assert goal.failed_task_count == 1

    def test_progress_percent_empty(self) -> None:
        goal = self._make_goal()
        assert goal.progress_percent == 0

    def test_progress_percent_half_done(self) -> None:
        tasks = [
            TaskRecord(description="t1", utterance="u1", state=TaskState.COMPLETED),
            TaskRecord(description="t2", utterance="u2", state=TaskState.COMPLETED),
            TaskRecord(description="t3", utterance="u3", state=TaskState.PENDING),
            TaskRecord(description="t4", utterance="u4", state=TaskState.PENDING),
        ]
        goal = self._make_goal(tasks=tasks)
        assert goal.progress_percent == 50

    def test_progress_percent_all_done(self) -> None:
        tasks = [
            TaskRecord(description="t1", utterance="u1", state=TaskState.COMPLETED),
            TaskRecord(description="t2", utterance="u2", state=TaskState.SKIPPED),
        ]
        goal = self._make_goal(tasks=tasks)
        assert goal.progress_percent == 100

    def test_with_state_returns_new_instance(self) -> None:
        goal = self._make_goal()
        updated = goal.with_state(GoalState.COMPLETED, summary="Done")
        assert updated.state == GoalState.COMPLETED
        assert updated.summary == "Done"
        # Original unchanged
        assert goal.state == GoalState.PENDING

    def test_with_state_preserves_fields(self) -> None:
        goal = GoalRecord(description="My goal")
        updated = goal.with_state(GoalState.EXECUTING)
        assert updated.description == "My goal"
        assert updated.goal_id == goal.goal_id


# ═══════════════════════════════════════════════════════════════════════════════
# ExecutionContext
# ═══════════════════════════════════════════════════════════════════════════════

class TestExecutionContext:
    def test_cancel_sets_flag(self) -> None:
        goal = GoalRecord(description="test")
        ctx = ExecutionContext(goal=goal)
        assert ctx.cancelled is False
        ctx.cancel()
        assert ctx.cancelled is True

    def test_default_max_retries(self) -> None:
        goal = GoalRecord(description="test")
        ctx = ExecutionContext(goal=goal)
        assert ctx.max_task_retries == 2
