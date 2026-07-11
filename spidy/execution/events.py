"""
Execution Events — EventBus events for the SkillExecutor
=========================================================
Topic namespace: ``execution.*``
"""

from __future__ import annotations

from dataclasses import dataclass

from spidy.core.event_bus import Event


@dataclass
class SkillExecutionStartedEvent(Event):
    """Emitted just before a skill action is executed."""
    topic = "execution.started"
    skill_name: str = ""
    action: str = ""
    session_id: str = ""


@dataclass
class SkillExecutionCompletedEvent(Event):
    """Emitted after a skill action completes successfully."""
    topic = "execution.completed"
    skill_name: str = ""
    action: str = ""
    success: bool = True
    session_id: str = ""


@dataclass
class SkillExecutionFailedEvent(Event):
    """Emitted when a skill action fails after all retries."""
    topic = "execution.failed"
    skill_name: str = ""
    action: str = ""
    error: str = ""
    session_id: str = ""


@dataclass
class SkillRetryEvent(Event):
    """Emitted when a skill is being retried after a transient failure."""
    topic = "execution.retry"
    skill_name: str = ""
    action: str = ""
    attempt: int = 1
    session_id: str = ""
