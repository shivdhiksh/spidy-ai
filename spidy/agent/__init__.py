"""
spidy.agent — Autonomous Agent Package (Milestone 16)

Public API
----------
    from spidy.agent import AutonomousAgent
    from spidy.agent.types import GoalState, TaskState, GoalRecord, TaskRecord
    from spidy.agent.events import GoalCreatedEvent, AgentProgressEvent
    from spidy.agent.observer import TaskObserver, Observation
    from spidy.agent.evaluator import TaskEvaluator, TaskOutcome, Confidence
    from spidy.agent.replanner import Replanner
    from spidy.agent.verifier import GoalVerifier, VerificationResult
    from spidy.agent.authority import TaskAuthorityChecker, AuthorityLevel
    from spidy.agent.task_logger import AgentTaskLogger
"""

from spidy.agent.agent import AutonomousAgent
from spidy.agent.authority import AuthorityLevel, TaskAuthorityChecker
from spidy.agent.evaluator import Confidence, TaskEvaluator, TaskOutcome
from spidy.agent.goal_manager import GoalManager
from spidy.agent.observer import Observation, TaskObserver
from spidy.agent.replanner import Replanner
from spidy.agent.task_logger import AgentTaskLogger
from spidy.agent.types import (
    ExecutionContext,
    GoalRecord,
    GoalState,
    ReflectionDecision,
    TaskRecord,
    TaskState,
)
from spidy.agent.verifier import GoalVerifier, VerificationResult

__all__ = [
    # Core
    "AutonomousAgent",
    "GoalManager",
    # Types
    "GoalState",
    "TaskState",
    "ReflectionDecision",
    "GoalRecord",
    "TaskRecord",
    "ExecutionContext",
    # M16 — Observer
    "TaskObserver",
    "Observation",
    # M16 — Evaluator
    "TaskEvaluator",
    "TaskOutcome",
    "Confidence",
    # M16 — Replanner
    "Replanner",
    # M16 — Verifier
    "GoalVerifier",
    "VerificationResult",
    # M16 — Authority
    "TaskAuthorityChecker",
    "AuthorityLevel",
    # M16 — Logger
    "AgentTaskLogger",
]
