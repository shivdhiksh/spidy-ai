# spidy/learning package — Learning Engine
"""
Spidy Learning Engine
=====================
Provides personalised, local-only learning from user interaction patterns.

No LLM fine-tuning. No telemetry. No cloud calls.
Everything is stored in ``spidy_learning.db`` on the user's device.

Public API
----------
    from spidy.learning import LearningManager

    manager = LearningManager(config=settings.learning, bus=bus, learning_dir=path)
    await manager.initialize()

    # Record feedback
    await manager.record_feedback(session_id, utterance, response, rating=+1.0)

    # Get preferences (for LLM prompt injection)
    context_str = await manager.get_preference_context()

    # Detect habits
    habit = await manager.get_habit({"active_app": "code.exe", "topic": "code"})
"""

from spidy.learning.manager import LearningManager
from spidy.learning.types import (
    FeedbackSignal,
    Habit,
    LearningContext,
    Preference,
    Workflow,
)
from spidy.learning.events import (
    LearningPreferenceUpdatedEvent,
    LearningHabitDetectedEvent,
    LearningHabitSuggestedEvent,
    LearningWorkflowDetectedEvent,
    LearningWorkflowSuggestedEvent,
    LearningFeedbackRecordedEvent,
    LearningErrorEvent,
)

__all__ = [
    # Manager (main entry point)
    "LearningManager",
    # Types
    "Preference",
    "Habit",
    "Workflow",
    "FeedbackSignal",
    "LearningContext",
    # Events
    "LearningPreferenceUpdatedEvent",
    "LearningHabitDetectedEvent",
    "LearningHabitSuggestedEvent",
    "LearningWorkflowDetectedEvent",
    "LearningWorkflowSuggestedEvent",
    "LearningFeedbackRecordedEvent",
    "LearningErrorEvent",
]
