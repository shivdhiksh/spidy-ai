"""
spidy.brain — The Brain Core Package (Milestone 3+)

Public API
----------
    from spidy.brain import Brain
    from spidy.brain import AutonomousAgent          # M13
    from spidy.brain.interfaces import MemoryInterface, KnowledgeInterface, LearningInterface
    from spidy.brain.types import Intent, Decision, Plan, ToolResult
    from spidy.brain.events import BrainResponseReadyEvent
"""

from spidy.brain.brain import Brain

__all__ = ["Brain"]

# M13: Lazy import to avoid circular dependency at package load time
# Usage: from spidy.brain import AutonomousAgent
def __getattr__(name: str):  # noqa: N807
    if name == "AutonomousAgent":
        from spidy.agent.agent import AutonomousAgent
        return AutonomousAgent
    raise AttributeError(f"module 'spidy.brain' has no attribute {name!r}")
