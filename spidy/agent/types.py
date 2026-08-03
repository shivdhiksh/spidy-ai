"""
Agent — Shared Data Types (Milestone 13)
=========================================
Typed data structures used across all autonomous agent components.

The agent layer sits above the Brain pipeline and manages persistent
goal state across multiple Brain.process() invocations.

Pipeline
--------
User goal → GoalRecord → TaskRecord[] → ExecutionLoop → Brain.process() × N → completion

GoalState transitions
---------------------
  PENDING → PLANNING → EXECUTING → COMPLETED
                    ↘              ↗
                      FAILED
                    ↘
                      CANCELLED
                    ↘
                      PAUSED (future: pause/resume)

TaskState transitions
---------------------
  PENDING → RUNNING → COMPLETED
                    ↘ FAILED
                    ↘ SKIPPED
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ─── Enumerations ─────────────────────────────────────────────────────────────


class GoalState(str, Enum):
    """Lifecycle state of an autonomous goal."""
    PENDING    = "pending"     # Created, not yet started
    PLANNING   = "planning"    # Decomposing into tasks
    EXECUTING  = "executing"   # Actively running tasks
    PAUSED     = "paused"      # Temporarily halted (future)
    COMPLETED  = "completed"   # All tasks finished successfully
    FAILED     = "failed"      # Unrecoverable failure
    CANCELLED  = "cancelled"   # Explicitly cancelled by user


class TaskState(str, Enum):
    """Lifecycle state of a single task within a goal."""
    PENDING    = "pending"     # Not yet started
    RUNNING    = "running"     # Currently executing
    COMPLETED  = "completed"   # Finished successfully
    FAILED     = "failed"      # Failed, all retries exhausted
    SKIPPED    = "skipped"     # Intentionally skipped (e.g. optional step)


class ReflectionDecision(str, Enum):
    """Decision produced by the ReflectionEngine after a task result."""
    CONTINUE     = "continue"     # Task succeeded — proceed to next
    RETRY        = "retry"        # Task failed — retry with same approach
    ALTERNATIVE  = "alternative"  # Task failed — try a different utterance
    ASK_USER     = "ask_user"     # Ambiguous — need clarification
    ABORT        = "abort"        # Unrecoverable — halt the goal


# ─── Task Record ──────────────────────────────────────────────────────────────


@dataclass
class TaskRecord:
    """
    A single atomic task within an autonomous goal.

    Attributes
    ----------
    task_id:
        Unique identifier for this task.
    goal_id:
        Parent goal this task belongs to.
    description:
        Human-readable description (e.g. "Install Flask via pip").
    utterance:
        The natural-language utterance sent to Brain.process() to execute
        this task (e.g. "install flask using pip in the terminal").
    state:
        Current lifecycle state.
    result_message:
        The Brain's response text for this task.
    result_success:
        Whether the Brain considered the task successful.
    attempt:
        How many times this task has been attempted (1-based).
    error:
        Error description if the task failed.
    started_at:
        When execution began.
    completed_at:
        When execution ended (success or final failure).
    extra:
        Arbitrary metadata (e.g. skill name, action used).
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal_id: str = ""
    description: str = ""
    utterance: str = ""
    state: TaskState = TaskState.PENDING
    result_message: str = ""
    result_success: bool = False
    attempt: int = 0
    error: str = ""
    started_at: datetime | None = None
    completed_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def mark_running(self) -> "TaskRecord":
        """Return a new TaskRecord in RUNNING state."""
        return TaskRecord(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            utterance=self.utterance,
            state=TaskState.RUNNING,
            attempt=self.attempt + 1,
            started_at=datetime.now(timezone.utc),
            extra=self.extra,
        )

    def mark_completed(self, message: str) -> "TaskRecord":
        """Return a new TaskRecord in COMPLETED state."""
        return TaskRecord(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            utterance=self.utterance,
            state=TaskState.COMPLETED,
            result_message=message,
            result_success=True,
            attempt=self.attempt,
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc),
            extra=self.extra,
        )

    def mark_failed(self, error: str, message: str = "") -> "TaskRecord":
        """Return a new TaskRecord in FAILED state."""
        return TaskRecord(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            utterance=self.utterance,
            state=TaskState.FAILED,
            result_message=message,
            result_success=False,
            attempt=self.attempt,
            error=error,
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc),
            extra=self.extra,
        )

    def mark_skipped(self, reason: str = "") -> "TaskRecord":
        """Return a new TaskRecord in SKIPPED state."""
        return TaskRecord(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            utterance=self.utterance,
            state=TaskState.SKIPPED,
            result_message=reason,
            attempt=self.attempt,
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc),
            extra=self.extra,
        )


# ─── Goal Record ──────────────────────────────────────────────────────────────


@dataclass
class GoalRecord:
    """
    An autonomous goal owned by GoalManager.

    A goal is the top-level user intention; it is decomposed into
    an ordered list of TaskRecords by the TaskDecomposer.

    Attributes
    ----------
    goal_id:
        Unique identifier.
    description:
        The original user goal description (e.g. "Create a Flask project").
    state:
        Current lifecycle state.
    tasks:
        Ordered list of tasks produced by TaskDecomposer.
        Empty until the PLANNING phase completes.
    session_id:
        The Brain session this goal belongs to.
    created_at:
        When the goal was created.
    started_at:
        When execution began.
    completed_at:
        When execution finished (any terminal state).
    error:
        Error description if the goal failed.
    summary:
        Natural language summary of the outcome (populated on completion).
    """
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    state: GoalState = GoalState.PENDING
    tasks: list[TaskRecord] = field(default_factory=list)
    session_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str = ""
    summary: str = ""

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def is_terminal(self) -> bool:
        """True if the goal has reached a terminal state."""
        return self.state in (GoalState.COMPLETED, GoalState.FAILED, GoalState.CANCELLED)

    @property
    def is_active(self) -> bool:
        """True if the goal is currently being worked on."""
        return self.state in (GoalState.PLANNING, GoalState.EXECUTING)

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def completed_task_count(self) -> int:
        return sum(1 for t in self.tasks if t.state == TaskState.COMPLETED)

    @property
    def failed_task_count(self) -> int:
        return sum(1 for t in self.tasks if t.state == TaskState.FAILED)

    @property
    def progress_percent(self) -> int:
        """0–100 progress percentage based on completed tasks."""
        if not self.tasks:
            return 0
        done = sum(
            1 for t in self.tasks
            if t.state in (TaskState.COMPLETED, TaskState.SKIPPED)
        )
        return int((done / len(self.tasks)) * 100)

    def with_state(self, state: GoalState, **kwargs: Any) -> "GoalRecord":
        """Return a shallow copy of this record with a new state."""
        import dataclasses
        updates = {"state": state, **kwargs}
        return dataclasses.replace(self, **updates)


# ─── Execution Context ────────────────────────────────────────────────────────


@dataclass
class ExecutionContext:
    """
    Runtime context passed through the execution loop.

    Carries the goal, mutable cancellation flag, and session info
    without any mutable shared state on the GoalRecord itself.
    """
    goal: GoalRecord
    session_id: str = ""
    cancelled: bool = False
    max_task_retries: int = 2

    def cancel(self) -> None:
        """Signal the execution loop to stop after the current task."""
        self.cancelled = True
