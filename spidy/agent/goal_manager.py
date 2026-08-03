"""
GoalManager — Goal Lifecycle Management (Milestone 13)
=======================================================
Creates, tracks, completes, cancels, and archives autonomous goals.

The GoalManager is the single source of truth for goal state.
It does not execute anything — it only manages records.

Design
------
- One GoalManager instance per AutonomousAgent (injected at construction)
- Goals are stored in memory (list); persistent storage is a future milestone
- Only one goal can be *active* (PLANNING or EXECUTING) at a time
- Completed/cancelled goals are kept in history (capped at max_history)
- All state transitions publish EventBus events

Usage
-----
    manager = GoalManager(bus=bus, max_history=50)

    goal = await manager.create_goal("Create a Flask project", session_id="s1")
    goal = await manager.begin_execution(goal.goal_id, tasks=[...])
    goal = await manager.complete_goal(goal.goal_id, summary="All done!")
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from spidy.agent.types import GoalRecord, GoalState, TaskRecord
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)

_DEFAULT_MAX_HISTORY = 100


class GoalManager:
    """
    Manages the lifecycle of autonomous goals.

    Parameters
    ----------
    bus:
        Application EventBus for publishing goal events.
    max_history:
        Maximum number of completed/cancelled goals to retain.
    """

    def __init__(
        self,
        bus: "EventBus",
        max_history: int = _DEFAULT_MAX_HISTORY,
    ) -> None:
        self._bus = bus
        self._max_history = max_history
        self._active_goal: GoalRecord | None = None
        self._history: deque[GoalRecord] = deque()

    # ── Goal creation ──────────────────────────────────────────────────────

    async def create_goal(
        self,
        description: str,
        session_id: str = "",
    ) -> GoalRecord:
        """
        Register a new autonomous goal.

        Raises
        ------
        RuntimeError
            If another goal is already active (PLANNING or EXECUTING).
        """
        if self._active_goal and not self._active_goal.is_terminal:
            raise RuntimeError(
                f"Goal '{self._active_goal.description[:60]}' is already active. "
                "Cancel it first before starting a new goal."
            )

        goal = GoalRecord(
            description=description,
            state=GoalState.PENDING,
            session_id=session_id,
        )
        self._active_goal = goal

        from spidy.agent.events import GoalCreatedEvent
        await self._bus.publish(GoalCreatedEvent(
            goal_id=goal.goal_id,
            description=goal.description,
            session_id=session_id,
        ))

        log.info(
            "Goal created: '{desc}' | id={gid}",
            desc=description[:80],
            gid=goal.goal_id,
        )
        return goal

    # ── State transitions ──────────────────────────────────────────────────

    async def begin_planning(self, goal_id: str) -> GoalRecord:
        """Transition the goal to PLANNING state."""
        goal = self._require_active(goal_id)
        updated = goal.with_state(GoalState.PLANNING)
        self._active_goal = updated
        log.debug("Goal planning: {gid}", gid=goal_id)
        return updated

    async def begin_execution(
        self,
        goal_id: str,
        tasks: list[TaskRecord],
    ) -> GoalRecord:
        """
        Transition the goal to EXECUTING state with its task list.

        Parameters
        ----------
        goal_id:
            The goal to start executing.
        tasks:
            Ordered list of tasks produced by TaskDecomposer.
        """
        goal = self._require_active(goal_id)
        updated = goal.with_state(
            GoalState.EXECUTING,
            tasks=tasks,
            started_at=datetime.now(timezone.utc),
        )
        self._active_goal = updated

        from spidy.agent.events import GoalStartedEvent
        await self._bus.publish(GoalStartedEvent(
            goal_id=updated.goal_id,
            description=updated.description,
            task_count=len(tasks),
            session_id=updated.session_id,
        ))

        log.info(
            "Goal executing: '{desc}' | {n} tasks",
            desc=updated.description[:80],
            n=len(tasks),
        )
        return updated

    async def update_task(
        self,
        goal_id: str,
        task: TaskRecord,
    ) -> GoalRecord:
        """
        Replace a TaskRecord in the active goal's task list.

        Called by ExecutionLoop after each task state change.
        """
        goal = self._require_active(goal_id)
        new_tasks = [
            task if t.task_id == task.task_id else t
            for t in goal.tasks
        ]
        updated = goal.with_state(goal.state, tasks=new_tasks)
        self._active_goal = updated
        return updated

    async def complete_goal(
        self,
        goal_id: str,
        summary: str = "",
    ) -> GoalRecord:
        """Mark the goal as successfully COMPLETED."""
        goal = self._require_active(goal_id)
        updated = goal.with_state(
            GoalState.COMPLETED,
            completed_at=datetime.now(timezone.utc),
            summary=summary,
        )
        self._active_goal = None
        self._archive(updated)

        from spidy.agent.events import GoalCompletedEvent
        await self._bus.publish(GoalCompletedEvent(
            goal_id=updated.goal_id,
            description=updated.description,
            summary=summary,
            completed_task_count=updated.completed_task_count,
            session_id=updated.session_id,
        ))

        log.info(
            "Goal completed: '{desc}' | {n}/{total} tasks",
            desc=updated.description[:80],
            n=updated.completed_task_count,
            total=updated.task_count,
        )
        return updated

    async def fail_goal(
        self,
        goal_id: str,
        error: str = "",
    ) -> GoalRecord:
        """Mark the goal as permanently FAILED."""
        goal = self._require_active(goal_id)
        updated = goal.with_state(
            GoalState.FAILED,
            completed_at=datetime.now(timezone.utc),
            error=error,
        )
        self._active_goal = None
        self._archive(updated)

        from spidy.agent.events import GoalFailedEvent
        await self._bus.publish(GoalFailedEvent(
            goal_id=updated.goal_id,
            description=updated.description,
            error=error,
            completed_task_count=updated.completed_task_count,
            failed_task_count=updated.failed_task_count,
            session_id=updated.session_id,
        ))

        log.warning(
            "Goal failed: '{desc}' | error={err}",
            desc=updated.description[:80],
            err=error[:120],
        )
        return updated

    async def cancel_goal(self, goal_id: str) -> GoalRecord:
        """Cancel the active goal immediately."""
        goal = self._require_active(goal_id)
        updated = goal.with_state(
            GoalState.CANCELLED,
            completed_at=datetime.now(timezone.utc),
        )
        self._active_goal = None
        self._archive(updated)

        from spidy.agent.events import GoalCancelledEvent
        await self._bus.publish(GoalCancelledEvent(
            goal_id=updated.goal_id,
            description=updated.description,
            completed_task_count=updated.completed_task_count,
            session_id=updated.session_id,
        ))

        log.info("Goal cancelled: {gid}", gid=goal_id)
        return updated

    # ── Query API ──────────────────────────────────────────────────────────

    @property
    def active_goal(self) -> GoalRecord | None:
        """The currently active goal, or None."""
        return self._active_goal

    def get_history(self, limit: int = 20) -> list[GoalRecord]:
        """Return the most recent completed/failed/cancelled goals."""
        history = list(self._history)
        return history[-limit:]

    def find_goal(self, goal_id: str) -> GoalRecord | None:
        """Find any goal by ID (active or in history)."""
        if self._active_goal and self._active_goal.goal_id == goal_id:
            return self._active_goal
        for g in self._history:
            if g.goal_id == goal_id:
                return g
        return None

    # ── Internal ──────────────────────────────────────────────────────────

    def _require_active(self, goal_id: str) -> GoalRecord:
        """Retrieve the active goal or raise if it doesn't match."""
        if self._active_goal and self._active_goal.goal_id == goal_id:
            return self._active_goal
        raise RuntimeError(
            f"Goal '{goal_id}' is not the active goal. "
            f"Active: {self._active_goal.goal_id if self._active_goal else 'none'}"
        )

    def _archive(self, goal: GoalRecord) -> None:
        """Move a terminal goal into history, respecting max_history."""
        self._history.append(goal)
        while len(self._history) > self._max_history:
            self._history.popleft()
