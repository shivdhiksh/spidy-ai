"""
Agent — Shared Data Types (Milestones 13 + 18)
===============================================
Typed data structures used across all autonomous agent components.

The agent layer sits above the Brain pipeline and manages persistent
goal state across multiple Brain.process() invocations.

Pipeline
--------
User goal → GoalRecord → TaskRecord[] → ExecutionLoop
    → IF task.action: StructuredRouter (direct skill dispatch)
    → ELSE: Brain.process() × N
    → completion

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

Structured Action Schema (M18)
------------------------------
TaskRecord.action is an optional dict with the following keys:
  skill           str  — which skill to route to ("browser", "desktop", "file")
  action          str  — what to do ("open_app", "navigate", "search", "verify", ...)
  target          str  — target app / site name ("Edge", "YouTube", "Google")
  query           str  — search query, preserved verbatim
  url             str  — explicit URL override
  expected_outcome str — what success looks like (used by Observer + GoalVerifier)

All fields are optional; the router uses whatever is present.
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


class TaskPriority(str, Enum):
    """Execution priority of a task."""
    HIGH   = "high"    # Cancellation, critical recovery, user confirmations
    NORMAL = "normal"  # Standard task steps
    LOW    = "low"     # Optional cleanup, telemetry


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
        The natural-language command to execute this step (what a user would say to an AI assistant)
    action:
        Optional structured action contract. When present, the ExecutionLoop
        routes directly to the appropriate skill without re-parsing through Brain.
    input_from:
        Optional task_id or reference expression (e.g. "${task_1}.text", "<input_from:step_2>").
    priority:
        Task priority (HIGH, NORMAL, LOW).
    state:
        Current lifecycle state.
    result_message:
        The Brain's response text for this task.
    result_success:
        Whether the task succeeded.
    attempt:
        How many times this task has been attempted.
    error:
        Error description if the task failed.
    started_at:
        When execution began.
    completed_at:
        When execution ended.
    extra:
        Arbitrary metadata.
    terminal:
        If True, the ExecutionLoop will complete the goal immediately after this task.
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal_id: str = ""
    description: str = ""
    utterance: str = ""
    action: dict[str, Any] | None = None   # structured action contract
    input_from: str = ""                   # inject result of prior task
    priority: TaskPriority = TaskPriority.NORMAL
    state: TaskState = TaskState.PENDING
    result_message: str = ""
    result_success: bool = False
    attempt: int = 0
    error: str = ""
    started_at: datetime | None = None
    completed_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    terminal: bool = False

    def mark_running(self) -> "TaskRecord":
        """Return a new TaskRecord in RUNNING state."""
        return TaskRecord(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            utterance=self.utterance,
            action=self.action,
            input_from=self.input_from,
            priority=self.priority,
            state=TaskState.RUNNING,
            attempt=self.attempt + 1,
            started_at=datetime.now(timezone.utc),
            extra=self.extra,
            terminal=self.terminal,
        )

    def mark_completed(self, message: str) -> "TaskRecord":
        """Return a new TaskRecord in COMPLETED state."""
        return TaskRecord(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            utterance=self.utterance,
            action=self.action,
            input_from=self.input_from,
            priority=self.priority,
            state=TaskState.COMPLETED,
            result_message=message,
            result_success=True,
            attempt=self.attempt,
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc),
            extra=self.extra,
            terminal=self.terminal,
        )

    def mark_failed(self, error: str, message: str = "") -> "TaskRecord":
        """Return a new TaskRecord in FAILED state."""
        return TaskRecord(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            utterance=self.utterance,
            action=self.action,
            input_from=self.input_from,
            priority=self.priority,
            state=TaskState.FAILED,
            result_message=message,
            result_success=False,
            attempt=self.attempt,
            error=error,
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc),
            extra=self.extra,
            terminal=self.terminal,
        )

    def mark_skipped(self, reason: str = "") -> "TaskRecord":
        """Return a new TaskRecord in SKIPPED state."""
        return TaskRecord(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            utterance=self.utterance,
            action=self.action,
            input_from=self.input_from,
            priority=self.priority,
            state=TaskState.SKIPPED,
            result_message=reason,
            attempt=self.attempt,
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc),
            extra=self.extra,
            terminal=self.terminal,
        )


# ─── Goal Record ──────────────────────────────────────────────────────────────


@dataclass
class GoalRecord:
    """
    Persistent state of a multi-task autonomous goal.
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

    Carries the goal, mutable cancellation flag, session info, and
    a structured step-result store so later tasks can reference earlier results.
    """
    goal: GoalRecord
    session_id: str = ""
    cancelled: bool = False
    max_task_retries: int = 2
    step_results: dict[str, Any] = field(default_factory=dict)
    _ordered_task_ids: list[str] = field(default_factory=list)

    def cancel(self) -> None:
        """Signal the execution loop to stop after the current task."""
        self.cancelled = True

    def add_result(self, task_id: str, result: Any) -> None:
        """Store a task result so later tasks can reference it via input_from or templating."""
        self.step_results[task_id] = result
        if task_id not in self._ordered_task_ids:
            self._ordered_task_ids.append(task_id)

    def add_structured_result(
        self,
        task_id: str,
        data: dict[str, Any] | str,
        step_index: int = -1,
    ) -> None:
        """Store a structured step result dictionary."""
        if isinstance(data, str):
            structured = {"result": data, "text": data}
        else:
            structured = dict(data)
            if "result" not in structured and "text" in structured:
                structured["result"] = structured["text"]

        self.step_results[task_id] = structured
        if task_id not in self._ordered_task_ids:
            self._ordered_task_ids.append(task_id)

        # Also alias step_N for intuitive reference
        if step_index >= 0:
            self.step_results[f"step_{step_index + 1}"] = structured
            self.step_results[f"step{step_index + 1}"] = structured

    def get_result(self, key: str, default: Any = "") -> Any:
        """Retrieve a raw or structured result by key."""
        return self.step_results.get(key, default)

    def resolve_variable(self, expr: str, current_task_id: str = "") -> Any:
        """
        Resolve a variable expression against stored step results.

        Supported expressions:
          - "${task_id.field}" or "${task_id}"
          - "${prev.field}" or "${prev}"
          - "<input_from:task_id>"
          - "<input_from:step_N>"
          - plain task_id string
        """
        if not expr or not isinstance(expr, str):
            return expr

        # 1. Plain task ID match in step_results
        if expr in self.step_results:
            val = self.step_results[expr]
            if isinstance(val, dict):
                return val.get("result") or val.get("text") or val.get("url") or str(val)
            return val

        # 2. Extract <input_from:...> syntax
        if expr.startswith("<input_from:") and expr.endswith(">"):
            ref_key = expr[12:-1].strip()
            if ref_key not in self.step_results and not ref_key.startswith("step"):
                return ""
            return self.resolve_variable(ref_key, current_task_id)

        # 3. Match ${...} interpolation patterns
        if "${" in expr:
            import re

            def _replace_match(match: re.Match) -> str:
                token = match.group(1).strip()
                parts = token.split(".", 1)
                target_ref = parts[0]
                field_name = parts[1] if len(parts) > 1 else None

                # Resolve 'prev' or 'previous'
                if target_ref in ("prev", "previous", "last"):
                    if self._ordered_task_ids:
                        if current_task_id and current_task_id in self._ordered_task_ids:
                            idx = self._ordered_task_ids.index(current_task_id)
                            target_ref = self._ordered_task_ids[idx - 1] if idx > 0 else ""
                        else:
                            target_ref = self._ordered_task_ids[-1]

                if not target_ref or target_ref not in self.step_results:
                    return ""

                stored = self.step_results[target_ref]
                if isinstance(stored, dict):
                    if field_name:
                        return str(stored.get(field_name, ""))
                    return str(stored.get("result") or stored.get("text") or stored.get("url") or "")
                return str(stored)

            return re.sub(r"\$\{([^}]+)\}", _replace_match, expr)

        return expr
