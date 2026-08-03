"""
spidy.agent — Autonomous Agent Package (Milestone 13)

Public API
----------
    from spidy.agent import AutonomousAgent
    from spidy.agent.types import GoalState, TaskState, GoalRecord, TaskRecord
    from spidy.agent.events import GoalCreatedEvent, AgentProgressEvent
"""

from spidy.agent.agent import AutonomousAgent
from spidy.agent.goal_manager import GoalManager
from spidy.agent.types import (
    ExecutionContext,
    GoalRecord,
    GoalState,
    ReflectionDecision,
    TaskRecord,
    TaskState,
)

__all__ = [
    "AutonomousAgent",
    "GoalManager",
    "GoalState",
    "TaskState",
    "ReflectionDecision",
    "GoalRecord",
    "TaskRecord",
    "ExecutionContext",
]
