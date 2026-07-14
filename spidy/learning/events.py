"""
Learning Engine — EventBus Events
===================================
Typed events published by the Learning Engine under the ``learning.*`` namespace.

All events are dataclasses that extend the base ``Event`` class from
``spidy.core.event_bus``. They follow the same conventions as the
``memory.*`` and ``knowledge.*`` events.

Event Catalogue
---------------
learning.preference_updated   — a preference was set or updated
learning.habit_detected       — a new habit crossed the confidence threshold
learning.habit_suggested      — a habit match was found for the current context
learning.workflow_detected    — a new workflow was identified
learning.workflow_suggested   — Spidy is suggesting to automate a workflow
learning.feedback_recorded    — a feedback signal was processed
learning.error                — a non-fatal error inside the Learning Engine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spidy.core.event_bus import Event


# ─── Preference Events ────────────────────────────────────────────────────────


@dataclass
class LearningPreferenceUpdatedEvent(Event):
    """
    Published when a user preference is created or updated.

    Attributes
    ----------
    key:
        The preference key (e.g. ``"preferred_browser"``).
    old_value:
        Previous value (None if this is a new preference).
    new_value:
        The updated value.
    confidence:
        New confidence score (0.0–1.0).
    source:
        How the update was triggered: ``"explicit"`` | ``"implicit"`` | ``"inferred"``.
    """

    topic: str = field(default="learning.preference_updated", init=False)
    key: str = ""
    old_value: Any = None
    new_value: Any = None
    confidence: float = 0.0
    source: str = "implicit"


# ─── Habit Events ─────────────────────────────────────────────────────────────


@dataclass
class LearningHabitDetectedEvent(Event):
    """
    Published when a new habit crosses the minimum confidence threshold
    and is stored for the first time.

    Attributes
    ----------
    habit_id:
        Stable habit identifier.
    description:
        Human-readable habit description.
    confidence:
        Habit confidence at time of detection.
    observed_count:
        Number of observations that triggered detection.
    """

    topic: str = field(default="learning.habit_detected", init=False)
    habit_id: str = ""
    description: str = ""
    confidence: float = 0.0
    observed_count: int = 0


@dataclass
class LearningHabitSuggestedEvent(Event):
    """
    Published when a habit matches the current context and could be suggested
    to the user.

    Attributes
    ----------
    habit_id:
        The matching habit's ID.
    action:
        The suggested action.
    confidence:
        Match confidence.
    context_summary:
        Brief description of the context that triggered the match.
    """

    topic: str = field(default="learning.habit_suggested", init=False)
    habit_id: str = ""
    action: str = ""
    confidence: float = 0.0
    context_summary: str = ""


# ─── Workflow Events ──────────────────────────────────────────────────────────


@dataclass
class LearningWorkflowDetectedEvent(Event):
    """
    Published when a multi-step workflow is detected for the first time.

    Attributes
    ----------
    workflow_id:
        Stable workflow identifier.
    name:
        Human-readable workflow name (e.g. "Code → Review → Browse").
    step_count:
        Number of steps in the workflow.
    trigger_count:
        How many times the sequence was observed.
    """

    topic: str = field(default="learning.workflow_detected", init=False)
    workflow_id: str = ""
    name: str = ""
    step_count: int = 0
    trigger_count: int = 0


@dataclass
class LearningWorkflowSuggestedEvent(Event):
    """
    Published when Spidy detects a workflow that could be automated.

    Attributes
    ----------
    workflow_id:
        The workflow to automate.
    name:
        Human-readable name.
    next_step:
        The predicted next step in the workflow.
    """

    topic: str = field(default="learning.workflow_suggested", init=False)
    workflow_id: str = ""
    name: str = ""
    next_step: str = ""


# ─── Feedback Events ──────────────────────────────────────────────────────────


@dataclass
class LearningFeedbackRecordedEvent(Event):
    """
    Published after a feedback signal is successfully processed.

    Attributes
    ----------
    signal_id:
        The FeedbackSignal ID.
    session_id:
        Brain session the feedback belongs to.
    rating:
        The processed rating value (-1.0 to +1.0).
    tags:
        Tags attached to this signal.
    """

    topic: str = field(default="learning.feedback_recorded", init=False)
    signal_id: str = ""
    session_id: str = ""
    rating: float = 0.0
    tags: list[str] = field(default_factory=list)


# ─── Error Event ──────────────────────────────────────────────────────────────


@dataclass
class LearningErrorEvent(Event):
    """
    Published when a non-fatal error occurs inside the Learning Engine.

    Learning errors must never crash the Brain. This event allows
    subscribers (e.g. a monitoring dashboard) to track failures.

    Attributes
    ----------
    component:
        The Learning subsystem that raised the error
        (e.g. ``"PreferenceLearner"``, ``"HabitDetector"``).
    error_message:
        Human-readable description of the error.
    """

    topic: str = field(default="learning.error", init=False)
    component: str = ""
    error_message: str = ""
