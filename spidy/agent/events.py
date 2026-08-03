"""
Agent Events (Milestone 13)
============================
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
agent.task_started      — a task within a goal began executing
agent.task_completed    — a task finished successfully
agent.task_failed       — a task failed (may still be retried)
agent.task_skipped      — a task was skipped
agent.progress          — percentage-based progress update
agent.reflection        — reflection engine decision logged
agent.clarification     — agent needs user input to continue
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
class TaskStartedEvent(Event):
    """Emitted when a task within a goal begins executing."""
    topic = "agent.task_started"
    goal_id: str = ""
    task_id: str = ""
    description: str = ""
    task_index: int = 0
    total_tasks: int = 0
    attempt: int = 1
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
