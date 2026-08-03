"""
ProgressTracker — Real-Time Goal Progress Reporting (Milestone 13)
===================================================================
Maintains step-level progress counters and publishes AgentProgressEvent
updates so the UI can display live progress during long-running goals.

Design
------
- Stateful: tracks current step index and total step count
- Percentage calculation: (completed + 1) / total * 100
- Publishes both AgentProgressEvent AND BrainProgressEvent
  (so existing UI subscribers of brain.progress also see updates)
- All publish calls are async and fire-and-forget safe

Usage
-----
    tracker = ProgressTracker(bus=bus)
    tracker.start(goal_id=goal.goal_id, total_tasks=7, session_id=sid)

    for i, task in enumerate(tasks):
        await tracker.advance(
            task_index=i,
            task_description=task.description,
        )
        # ... execute task ...

    await tracker.complete()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)

# Progress stage messages for display
_STAGE_MESSAGES: dict[str, str] = {
    "planning": "Planning...",
    "executing": "Working on it...",
    "completed": "All done!",
    "failed": "Something went wrong.",
    "cancelled": "Cancelled.",
}


class ProgressTracker:
    """
    Tracks and publishes progress for an autonomous goal execution.

    Parameters
    ----------
    bus:
        EventBus for publishing progress events.
    """

    def __init__(self, bus: "EventBus") -> None:
        self._bus = bus
        self._goal_id: str = ""
        self._session_id: str = ""
        self._total_tasks: int = 0
        self._current_index: int = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(
        self,
        goal_id: str,
        total_tasks: int,
        session_id: str = "",
    ) -> None:
        """
        Initialise tracking for a new goal execution.

        Parameters
        ----------
        goal_id:
            The goal being tracked.
        total_tasks:
            Total number of tasks in the goal.
        session_id:
            For event publishing.
        """
        self._goal_id = goal_id
        self._session_id = session_id
        self._total_tasks = max(total_tasks, 1)
        self._current_index = 0

    # ── Progress updates ──────────────────────────────────────────────────

    async def publish_planning(self) -> None:
        """Publish a 'Planning...' progress event at 0%."""
        await self._publish(
            message=_STAGE_MESSAGES["planning"],
            percent=0,
            current_task="",
            completed=0,
        )

    async def advance(
        self,
        task_index: int,
        task_description: str,
    ) -> int:
        """
        Publish a progress event for the current task about to execute.

        Parameters
        ----------
        task_index:
            Zero-based index of the task being started.
        task_description:
            Human-readable description of the current task.

        Returns
        -------
        int
            The progress percentage (0–100).
        """
        self._current_index = task_index
        # Percentage: reflect that THIS task is about to begin (not completed)
        percent = int((task_index / self._total_tasks) * 100)
        message = f"{task_description}..."

        await self._publish(
            message=message,
            percent=percent,
            current_task=task_description,
            completed=task_index,
        )
        return percent

    async def task_done(
        self,
        task_index: int,
        task_description: str,
    ) -> int:
        """
        Publish a progress event after a task completes.

        Parameters
        ----------
        task_index:
            Zero-based index of the task that just completed.
        task_description:
            Human-readable description.

        Returns
        -------
        int
            The updated progress percentage.
        """
        completed = task_index + 1
        percent = int((completed / self._total_tasks) * 100)

        await self._publish(
            message=f"Completed: {task_description}",
            percent=percent,
            current_task=task_description,
            completed=completed,
        )
        return percent

    async def publish_completed(self) -> None:
        """Publish a 100% completion progress event."""
        await self._publish(
            message=_STAGE_MESSAGES["completed"],
            percent=100,
            current_task="",
            completed=self._total_tasks,
        )

    async def publish_failed(self, reason: str = "") -> None:
        """Publish a failure progress event."""
        msg = _STAGE_MESSAGES["failed"]
        if reason:
            msg = f"{msg} {reason}"
        await self._publish(
            message=msg,
            percent=self._current_percent,
            current_task="",
            completed=self._current_index,
        )

    async def publish_cancelled(self) -> None:
        """Publish a cancellation progress event."""
        await self._publish(
            message=_STAGE_MESSAGES["cancelled"],
            percent=self._current_percent,
            current_task="",
            completed=self._current_index,
        )

    # ── Internal ──────────────────────────────────────────────────────────

    @property
    def _current_percent(self) -> int:
        if not self._total_tasks:
            return 0
        return int((self._current_index / self._total_tasks) * 100)

    async def _publish(
        self,
        message: str,
        percent: int,
        current_task: str,
        completed: int,
    ) -> None:
        """Publish both AgentProgressEvent and BrainProgressEvent."""
        from spidy.agent.events import AgentProgressEvent
        from spidy.brain.events import BrainProgressEvent

        log.debug(
            "Progress [{pct}%]: {msg}",
            pct=percent,
            msg=message[:80],
        )

        try:
            await self._bus.publish(AgentProgressEvent(
                goal_id=self._goal_id,
                session_id=self._session_id,
                message=message,
                progress_percent=percent,
                current_task=current_task,
                completed_tasks=completed,
                total_tasks=self._total_tasks,
            ))

            # Also publish BrainProgressEvent for backward-compat UI subscribers
            await self._bus.publish(BrainProgressEvent(
                session_id=self._session_id,
                message=message,
                progress_percent=percent,
            ))
        except Exception as exc:  # noqa: BLE001 — progress events must never crash the loop
            log.warning("ProgressTracker: event publish failed (non-fatal): {exc}", exc=exc)
