"""
spidy.brain — The Brain Core Package (Milestone 3)

Public API
----------
    from spidy.brain import Brain
    from spidy.brain.interfaces import MemoryInterface, KnowledgeInterface, LearningInterface
    from spidy.brain.types import Intent, Decision, Plan, ToolResult
    from spidy.brain.events import BrainResponseReadyEvent
"""

from spidy.brain.brain import Brain

__all__ = ["Brain"]
