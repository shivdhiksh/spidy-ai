"""
Agent Events (Milestone 13 → Milestone 16 upgrade)
====================================================
All EventBus events published by the Autonomous Agent layer.

Topic namespace: ``agent.*``

These events allow the UI, logging, and any other subscriber to
observe goal lifecycle without coupling to the agent internals.

Events
------
agent.goal_created      — a new goal was registered
agent.goal_started      — execution of a goal began
agent.goal_completed    — goal finished successfully
agent.goal_failed       — goal failed permanently
agent.goal_cancelled    — goal was cancelled by user
agent.goal_verified     — final goal verification result
agent.task_started      — a task within a goal began executing
agent.task_completed    — a task finished successfully
agent.task_failed       — a task failed (may still be retried)
agent.task_skipped      — a task was skipped
agent.progress          — percentage-based progress update
agent.reflection        — reflection engine decision logged
agent.clarification     — agent needs user input to continue
agent.observation       — environment observation after a task
agent.evaluation        — structured task outcome evaluation
agent.replanning        — replanner triggered after soft failure
agent.confirmation_required — task requires user approval before proceeding
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spidy.core.event_bus import Event


@dataclass
class GoalCreatedEvent(Event):
    """Emitted when a new autonomous goal is registered."""
    topic = "agent.goal_created"
    goal_id: str = ""
    description: str = ""
    session_id: str = ""


@dataclass
class GoalStartedEvent(Event):
    """Emitted when execution of a goal begins (after decomposition)."""
    topic = "agent.goal_started"
    goal_id: str = ""
    description: str = ""
    task_count: int = 0
    session_id: str = ""


@dataclass
class GoalCompletedEvent(Event):
    """Emitted when all tasks in a goal finish successfully."""
    topic = "agent.goal_completed"
    goal_id: str = ""
    description: str = ""
    summary: str = ""
    completed_task_count: int = 0
    session_id: str = ""


@dataclass
class GoalFailedEvent(Event):
    """Emitted when a goal fails permanently (cannot be recovered)."""
    topic = "agent.goal_failed"
    goal_id: str = ""
    description: str = ""
    error: str = ""
    completed_task_count: int = 0
    failed_task_count: int = 0
    session_id: str = ""


@dataclass
class GoalCancelledEvent(Event):
    """Emitted when a goal is explicitly cancelled by the user."""
    topic = "agent.goal_cancelled"
    goal_id: str = ""
    description: str = ""
    completed_task_count: int = 0
    session_id: str = ""


@dataclass
class GoalVerifiedEvent(Event):
    """
    Emitted after GoalVerifier.verify() completes.

    Carries the final verification verdict and human-readable summary.
    Subscribers (UI, logging) can use this to display a definitive outcome.
    """
    topic = "agent.goal_verified"
    goal_id: str = ""
    description: str = ""
    verified: bool = True
    confidence: str = "medium"
    summary: str = ""
    method: str = "task_results"
    session_id: str = ""


@dataclass
class TaskStartedEvent(Event):
    """Emitted when a task within a goal begins executing."""
    topic = "agent.task_started"
    goal_id: str = ""
    task_id: str = ""
    description: str = ""
    task_index: int = 0
    total_tasks: int = 0
    attempt: int = 1
    authority_level: str = "safe"
    session_id: str = ""


@dataclass
class TaskCompletedEvent(Event):
    """Emitted when a task finishes successfully."""
    topic = "agent.task_completed"
    goal_id: str = ""
    task_id: str = ""
    description: str = ""
    result_message: str = ""
    task_index: int = 0
    total_tasks: int = 0
    session_id: str = ""


@dataclass
class TaskFailedEvent(Event):
    """Emitted when a task fails (may still be retried at the goal level)."""
    topic = "agent.task_failed"
    goal_id: str = ""
    task_id: str = ""
    description: str = ""
    error: str = ""
    attempt: int = 1
    will_retry: bool = False
    session_id: str = ""


@dataclass
class TaskSkippedEvent(Event):
    """Emitted when a task is intentionally skipped."""
    topic = "agent.task_skipped"
    goal_id: str = ""
    task_id: str = ""
    description: str = ""
    reason: str = ""
    session_id: str = ""


@dataclass
class AgentProgressEvent(Event):
    """
    Emitted to provide real-time progress updates to the UI.

    Published after every task step — includes both a human-readable
    message and a numeric percentage for progress bars.
    """
    topic = "agent.progress"
    goal_id: str = ""
    session_id: str = ""
    message: str = ""
    progress_percent: int = 0
    current_task: str = ""
    completed_tasks: int = 0
    total_tasks: int = 0


@dataclass
class AgentReflectionEvent(Event):
    """
    Emitted when the ReflectionEngine makes a decision.

    Useful for debugging and audit logs.
    """
    topic = "agent.reflection"
    goal_id: str = ""
    task_id: str = ""
    decision: str = ""  # ReflectionDecision.value
    reason: str = ""
    session_id: str = ""


@dataclass
class AgentClarificationEvent(Event):
    """
    Emitted when the agent needs user input to continue.

    The UI should prompt the user and feed the answer back via Brain.process().
    """
    topic = "agent.clarification"
    goal_id: str = ""
    task_id: str = ""
    question: str = ""
    session_id: str = ""


# ── New events (Milestone 16 autonomous agent upgrade) ─────────────────────────


@dataclass
class AgentObservationEvent(Event):
    """
    Emitted after TaskObserver.observe() completes for a task.

    Carries the observation method used and the resulting success signal.
    Useful for debugging, audit trails, and UI progress display.
    """
    topic = "agent.observation"
    goal_id: str = ""
    task_id: str = ""
    task_description: str = ""
    method: str = ""
    summary: str = ""
    success_signal: bool | None = None
    session_id: str = ""


@dataclass
class AgentEvaluationEvent(Event):
    """
    Emitted after TaskEvaluator.evaluate() produces a TaskOutcome.

    Allows UI and logging subscribers to see the structured verdict
    (success, confidence, whether replanning was triggered).
    """
    topic = "agent.evaluation"
    goal_id: str = ""
    task_id: str = ""
    task_description: str = ""
    success: bool = False
    confidence: str = "unknown"
    should_replan: bool = False
    reason: str = ""
    session_id: str = ""


@dataclass
class AgentReplanningEvent(Event):
    """
    Emitted when the Replanner is triggered after a soft task failure.

    Carries the attempt counter so the UI can show "Replanning (1/2)".
    """
    topic = "agent.replanning"
    goal_id: str = ""
    failed_task_id: str = ""
    failed_task_description: str = ""
    attempt: int = 1
    max_attempts: int = 2
    reason: str = ""
    session_id: str = ""


@dataclass
class AgentConfirmationRequiredEvent(Event):
    """
    Emitted when a task requires explicit user confirmation before proceeding.

    The UI must prompt the user. The ExecutionLoop pauses until:
    - user confirms (publishes agent.confirmation_granted)
    - user denies  (publishes agent.confirmation_denied)
    - or the goal is cancelled

    This event is published BEFORE Brain.process() — the LLM never sees it.
    """
    topic = "agent.confirmation_required"
    goal_id: str = ""
    task_id: str = ""
    task_description: str = ""
    utterance: str = ""
    authority_level: str = "confirm"
    prompt: str = ""
    session_id: str = ""
