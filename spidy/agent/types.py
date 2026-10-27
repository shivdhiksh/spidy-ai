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
    PENDING          = "pending"           # Created, not yet started
    PLANNING         = "planning"          # Decomposing into tasks
    EXECUTING        = "executing"         # Actively running tasks
    WAITING_FOR_USER = "waiting_for_user"  # Paused waiting for interactive user clarification
    PAUSED           = "paused"            # Temporarily halted (future)
    COMPLETED        = "completed"         # All tasks finished successfully
    FAILED           = "failed"            # Unrecoverable failure
    CANCELLED        = "cancelled"         # Explicitly cancelled by user


class TaskState(str, Enum):
    """Lifecycle state of a single task within a goal."""
    PENDING          = "pending"           # Not yet started
    RUNNING          = "running"           # Currently executing
    SUCCEEDED        = "succeeded"         # Finished successfully with verified outcome
    COMPLETED        = "completed"         # Finished successfully (alias/compatibility)
    FAILED           = "failed"            # Failed, all retries exhausted
    RETRYING         = "retrying"          # Attempt failed, retry underway
    BLOCKED          = "blocked"           # Precondition or dependency not met
    CANCELLED        = "cancelled"         # Explicitly cancelled by user or loop
    WAITING_FOR_USER = "waiting_for_user"  # Paused waiting for user input
    SKIPPED          = "skipped"           # Intentionally skipped (e.g. optional step)


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
    ASK_USER     = "ask_user"     # Clarification needed from user
    ABORT        = "abort"        # Unrecoverable — fail the goal immediately


class TaskErrorCode(str, Enum):
    """Taxonomy of deterministic and recoverable execution error categories."""
    APP_NOT_FOUND        = "APP_NOT_FOUND"
    WINDOW_NOT_FOUND     = "WINDOW_NOT_FOUND"
    FOCUS_LOST           = "FOCUS_LOST"
    CONTROL_NOT_FOUND    = "CONTROL_NOT_FOUND"
    TIMEOUT              = "TIMEOUT"
    FILE_NOT_FOUND       = "FILE_NOT_FOUND"
    PERMISSION_DENIED    = "PERMISSION_DENIED"
    BROWSER_FAILURE      = "BROWSER_FAILURE"
    NETWORK_FAILURE      = "NETWORK_FAILURE"
    TEXT_NOT_ENTERED     = "TEXT_NOT_ENTERED"
    SAVE_FAILED          = "SAVE_FAILED"
    PROCESS_CRASHED      = "PROCESS_CRASHED"
    CODE_ERROR           = "CODE_ERROR"
    TEST_FAILED          = "TEST_FAILED"
    SCREEN_UNREADABLE    = "SCREEN_UNREADABLE"
    AMBIGUOUS_INTENT     = "AMBIGUOUS_INTENT"
    UNKNOWN              = "UNKNOWN"


class Confidence(str, Enum):
    """How certain the evaluator is about the task outcome."""
    HIGH    = "high"     # Multiple signals agree
    MEDIUM  = "medium"   # Single reliable signal
    LOW     = "low"      # Weak or conflicting signals
    UNKNOWN = "unknown"  # No signal available


@dataclass
class TaskOutcome:
    """
    Structured result of evaluating a task's execution.
    """
    success: bool = False
    confidence: Confidence = Confidence.UNKNOWN
    reason: str = ""
    error_code: TaskErrorCode = TaskErrorCode.UNKNOWN
    recovery_strategy: str = ""
    should_replan: bool = False
    replan_hint: str = ""
    reflection_decision: ReflectionDecision = ReflectionDecision.CONTINUE


# ─── Task Record ──────────────────────────────────────────────────────────────


@dataclass
class StepResult:
    """
    Standardized result contract returned by any skill or task execution.
    """
    success: bool = True
    data: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.data:
            return self.data[key]
        if hasattr(self, key):
            return getattr(self, key)
        return default


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
    dependencies:
        List of task_ids that MUST be completed successfully before this task can start.
    parameters:
        Structured action parameters dictionary.
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
    timeout:
        Task timeout in seconds.
    verification_required:
        Whether outcome verification is mandatory for this task.
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal_id: str = ""
    description: str = ""
    utterance: str = ""
    action: dict[str, Any] | None = None   # structured action contract
    dependencies: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
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
    timeout: float = 30.0
    verification_required: bool = True
    evidence: dict[str, Any] | None = None

    def mark_running(self) -> "TaskRecord":
        """Return a new TaskRecord in RUNNING state."""
        return TaskRecord(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            utterance=self.utterance,
            action=self.action,
            dependencies=list(self.dependencies),
            parameters=dict(self.parameters),
            input_from=self.input_from,
            priority=self.priority,
            state=TaskState.RUNNING,
            attempt=self.attempt + 1,
            started_at=datetime.now(timezone.utc),
            extra=dict(self.extra),
            terminal=self.terminal,
            timeout=self.timeout,
            verification_required=self.verification_required,
            evidence=self.evidence,
        )

    def mark_succeeded(self, message: str, evidence: dict[str, Any] | None = None) -> "TaskRecord":
        """Return a new TaskRecord in SUCCEEDED state with evidence."""
        extra_copy = dict(self.extra)
        if evidence:
            extra_copy["evidence"] = evidence
        return TaskRecord(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            utterance=self.utterance,
            action=self.action,
            dependencies=list(self.dependencies),
            parameters=dict(self.parameters),
            input_from=self.input_from,
            priority=self.priority,
            state=TaskState.SUCCEEDED,
            result_message=message,
            result_success=True,
            attempt=self.attempt,
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc),
            extra=extra_copy,
            terminal=self.terminal,
            timeout=self.timeout,
            verification_required=self.verification_required,
            evidence=evidence or self.evidence,
        )

    def mark_completed(self, message: str, evidence: dict[str, Any] | None = None) -> "TaskRecord":
        """Return a new TaskRecord in COMPLETED state."""
        extra_copy = dict(self.extra)
        if evidence:
            extra_copy["evidence"] = evidence
        return TaskRecord(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            utterance=self.utterance,
            action=self.action,
            dependencies=list(self.dependencies),
            parameters=dict(self.parameters),
            input_from=self.input_from,
            priority=self.priority,
            state=TaskState.COMPLETED,
            result_message=message,
            result_success=True,
            attempt=self.attempt,
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc),
            extra=extra_copy,
            terminal=self.terminal,
            timeout=self.timeout,
            verification_required=self.verification_required,
            evidence=evidence or self.evidence,
        )

    def mark_failed(self, error: str, message: str = "") -> "TaskRecord":
        """Return a new TaskRecord in FAILED state."""
        return TaskRecord(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            utterance=self.utterance,
            action=self.action,
            dependencies=list(self.dependencies),
            parameters=dict(self.parameters),
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
            timeout=self.timeout,
            verification_required=self.verification_required,
        )

    def mark_waiting_for_user(self, prompt: str) -> "TaskRecord":
        """Return a new TaskRecord in WAITING_FOR_USER state."""
        return TaskRecord(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            utterance=self.utterance,
            action=self.action,
            dependencies=list(self.dependencies),
            parameters=dict(self.parameters),
            input_from=self.input_from,
            priority=self.priority,
            state=TaskState.WAITING_FOR_USER,
            result_message=prompt,
            result_success=False,
            attempt=self.attempt,
            started_at=self.started_at,
            extra=self.extra,
            terminal=self.terminal,
            timeout=self.timeout,
            verification_required=self.verification_required,
        )

    def mark_skipped(self, reason: str = "") -> "TaskRecord":
        """Return a new TaskRecord in SKIPPED state."""
        return TaskRecord(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            utterance=self.utterance,
            action=self.action,
            dependencies=list(self.dependencies),
            parameters=dict(self.parameters),
            input_from=self.input_from,
            priority=self.priority,
            state=TaskState.SKIPPED,
            result_message=reason,
            attempt=self.attempt,
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc),
            extra=self.extra,
            terminal=self.terminal,
            timeout=self.timeout,
            verification_required=self.verification_required,
            evidence=self.evidence,
        )

    def mark_retrying(self, reason: str = "") -> "TaskRecord":
        """Return a new TaskRecord in RETRYING state."""
        return TaskRecord(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            utterance=self.utterance,
            action=self.action,
            dependencies=list(self.dependencies),
            parameters=dict(self.parameters),
            input_from=self.input_from,
            priority=self.priority,
            state=TaskState.RETRYING,
            result_message=reason,
            attempt=self.attempt,
            started_at=self.started_at,
            extra=self.extra,
            terminal=self.terminal,
            timeout=self.timeout,
            verification_required=self.verification_required,
            evidence=self.evidence,
        )

    def mark_blocked(self, reason: str = "") -> "TaskRecord":
        """Return a new TaskRecord in BLOCKED state."""
        return TaskRecord(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            utterance=self.utterance,
            action=self.action,
            dependencies=list(self.dependencies),
            parameters=dict(self.parameters),
            input_from=self.input_from,
            priority=self.priority,
            state=TaskState.BLOCKED,
            result_message=reason,
            attempt=self.attempt,
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc),
            extra=self.extra,
            terminal=self.terminal,
            timeout=self.timeout,
            verification_required=self.verification_required,
            evidence=self.evidence,
        )

    def mark_cancelled(self, reason: str = "Cancelled by user.") -> "TaskRecord":
        """Return a new TaskRecord in CANCELLED state."""
        return TaskRecord(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            utterance=self.utterance,
            action=self.action,
            dependencies=list(self.dependencies),
            parameters=dict(self.parameters),
            input_from=self.input_from,
            priority=self.priority,
            state=TaskState.CANCELLED,
            result_message=reason,
            attempt=self.attempt,
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc),
            extra=self.extra,
            terminal=self.terminal,
            timeout=self.timeout,
            verification_required=self.verification_required,
            evidence=self.evidence,
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
        """True if the goal is currently being worked on or waiting for user."""
        return self.state in (GoalState.PLANNING, GoalState.EXECUTING, GoalState.WAITING_FOR_USER)

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def completed_task_count(self) -> int:
        return sum(1 for t in self.tasks if t.state in (TaskState.COMPLETED, TaskState.SUCCEEDED))

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
            if t.state in (TaskState.COMPLETED, TaskState.SUCCEEDED, TaskState.SKIPPED)
        )
        return int((done / len(self.tasks)) * 100)

    def with_state(self, state: GoalState, **kwargs: Any) -> "GoalRecord":
        """Return a shallow copy of this record with a new state."""
        import dataclasses
        updates = {"state": state, **kwargs}
        return dataclasses.replace(self, **updates)


# ─── Goal Context ─────────────────────────────────────────────────────────────


@dataclass
class GoalContext:
    """
    Persistent, non-lossy Goal Context holding the uncompromised source of truth.
    Survives all task transitions, retries, recoveries, clarifications, and replanning.
    """
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_goal: str = ""
    normalized_goal: str = ""
    original_transcript: str = ""
    normalized_transcript: str = ""
    intent: str = ""
    entities: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    required_outcomes: list[str] = field(default_factory=list)
    tasks: list[TaskRecord] = field(default_factory=list)
    completed_tasks: list[TaskRecord] = field(default_factory=list)
    current_task: TaskRecord | None = None
    task_dependencies: dict[str, list[str]] = field(default_factory=dict)
    step_results: dict[str, Any] = field(default_factory=dict)
    environment_snapshot: dict[str, Any] = field(default_factory=dict)
    pending_clarification: dict[str, Any] | None = None
    recovery_history: list[dict[str, Any]] = field(default_factory=list)
    verification_state: str = "pending"
    final_result: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float | int] = field(default_factory=dict)

    def add_recovery(self, task_id: str, strategy: str, error: str, result: str) -> None:
        self.recovery_history.append({
            "task_id": task_id,
            "strategy": strategy,
            "error": error,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


# ─── Execution Context ────────────────────────────────────────────────────────


@dataclass
class ExecutionContext:
    """
    Runtime context passed through the execution loop.
    Carries the goal, mutable cancellation flag, session info, and
    a structured step-result store so later tasks can reference earlier results.
    """
    goal: GoalRecord = field(default_factory=lambda: GoalRecord(goal_id="default", description="default"))
    session_id: str = ""
    cancelled: bool = False
    max_task_retries: int = 2
    step_results: dict[str, Any] = field(default_factory=dict)
    _ordered_task_ids: list[str] = field(default_factory=list)
    waiting_task_id: str = ""
    waiting_prompt: str = ""
    waiting_candidates: list[dict[str, Any]] = field(default_factory=list)
    goal_context: GoalContext | None = None
    step_evidences: list[dict[str, Any]] = field(default_factory=list)

    def cancel(self) -> None:
        """Signal the execution loop to stop after the current task."""
        self.cancelled = True

    def add_evidence(self, evidence: dict[str, Any]) -> None:
        """Store verifiable step evidence."""
        if evidence and isinstance(evidence, dict):
            self.step_evidences.append(dict(evidence))

    def add_result(self, task_id: str, result: Any) -> None:
        """Store a task result so later tasks can reference it via input_from or templating."""
        self.step_results[task_id] = result
        if task_id not in self._ordered_task_ids:
            self._ordered_task_ids.append(task_id)
        if self.goal_context:
            self.goal_context.step_results[task_id] = result

    def add_structured_result(
        self,
        task_id: str,
        data: dict[str, Any] | str | StepResult,
        step_index: int = -1,
    ) -> None:
        """Store a structured step result dictionary."""
        if isinstance(data, StepResult):
            structured = dict(data.data)
            structured["success"] = data.success
            structured["error"] = data.error
            structured["observation"] = data.observation
        elif isinstance(data, str):
            structured = {"result": data, "text": data, "success": True}
        else:
            structured = dict(data)
            if "result" not in structured and "text" in structured:
                structured["result"] = structured["text"]

        # If evidence was captured, track in step_evidences
        if isinstance(structured, dict):
            if "evidence_data" in structured and isinstance(structured["evidence_data"], dict):
                self.add_evidence(structured["evidence_data"])
            elif "evidence" in structured and "step" in structured:
                self.add_evidence(structured)

        self.step_results[task_id] = structured
        if task_id not in self._ordered_task_ids:
            self._ordered_task_ids.append(task_id)

        # Also alias step_N for intuitive reference
        if step_index >= 0:
            self.step_results[f"step_{step_index + 1}"] = structured
            self.step_results[f"step{step_index + 1}"] = structured

        if self.goal_context:
            self.goal_context.step_results[task_id] = structured
            if step_index >= 0:
                self.goal_context.step_results[f"step_{step_index + 1}"] = structured

    def get_result(self, key: str, default: Any = "") -> Any:
        """Retrieve a raw or structured result by key."""
        return self.step_results.get(key, default)

    def get_latest_result(self) -> dict[str, Any] | Any:
        """Retrieve the most recent step result."""
        if not self._ordered_task_ids:
            return {}
        last_id = self._ordered_task_ids[-1]
        return self.step_results.get(last_id, {})

    def resolve_variable(self, expr: str, current_task_id: str = "") -> Any:
        """
        Resolve a variable expression against stored step results and natural language references.

        Supported expressions:
          - "${task_id.field}" or "${task_id}"
          - "${prev.field}" or "${prev}"
          - "<input_from:task_id.field>" or "<input_from:task_id>"
          - "<task_result>" or "<result>"
          - Natural language references: "the file", "the search query", "the result", "the query", "it", "that", "what you found"
        """
        if not expr or not isinstance(expr, str):
            return expr

        norm_expr = expr.strip().lower()

        # Natural language / pronoun resolution against latest stored data
        if norm_expr in (
            "the file", "the resume", "the found file", "it", "that",
            "the document", "the path", "the selected file"
        ):
            latest = self.get_latest_result()
            if isinstance(latest, dict):
                path_val = latest.get("path") or latest.get("file_path") or latest.get("target") or latest.get("result")
                if path_val:
                    return path_val
            return self.resolve_variable("${prev.path}", current_task_id)

        if norm_expr in (
            "the query", "the search query", "the search term", "search query",
            "the query you searched", "the term"
        ):
            latest = self.get_latest_result()
            if isinstance(latest, dict):
                query_val = latest.get("query") or latest.get("input")
                if query_val:
                    return query_val
            # Scan backward through ordered results for a query
            for tid in reversed(self._ordered_task_ids):
                res = self.step_results.get(tid)
                if isinstance(res, dict) and res.get("query"):
                    return res["query"]
            return self.resolve_variable("${prev.query}", current_task_id)

        if norm_expr in (
            "the result", "the search result", "what you found",
            "the answer", "the output", "the text", "the code"
        ):
            latest = self.get_latest_result()
            if isinstance(latest, dict):
                val = latest.get("text") or latest.get("result") or latest.get("output") or latest.get("data")
                if val:
                    return val
            return self.resolve_variable("${prev.result}", current_task_id)

        # 0. Global alias for previous task result
        if expr in ("<task_result>", "<result>", "prev", "previous"):
            return self.resolve_variable("${prev.result}", current_task_id)

        # 1. Extract <input_from:...> syntax
        if expr.startswith("<input_from:") and expr.endswith(">"):
            ref_key = expr[12:-1].strip()
            return self.resolve_variable(f"${{{ref_key}}}", current_task_id)

        # 2. Match ${...} interpolation patterns
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
                    return str(
                        stored.get("result")
                        or stored.get("text")
                        or stored.get("url")
                        or stored.get("query")
                        or stored.get("file_path")
                        or stored.get("path")
                        or ""
                    )
                return str(stored)

            return re.sub(r"\$\{([^}]+)\}", _replace_match, expr)

        # 3. Direct lookup in step_results with possible dot notation
        if "." in expr:
            parts = expr.split(".", 1)
            target_ref, field_name = parts[0], parts[1]
            if target_ref in self.step_results:
                stored = self.step_results[target_ref]
                if isinstance(stored, dict):
                    return stored.get(field_name, "")
                return getattr(stored, field_name, "")

        # 4. Plain task ID match in step_results
        if expr in self.step_results:
            val = self.step_results[expr]
            if isinstance(val, dict):
                return (
                    val.get("result")
                    or val.get("text")
                    or val.get("url")
                    or val.get("query")
                    or val.get("file_path")
                    or val.get("path")
                    or str(val)
                )
            return val

        return expr

