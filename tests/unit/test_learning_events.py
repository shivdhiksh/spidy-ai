"""Tests for spidy.learning.events - Learning Engine EventBus events."""
from spidy.learning.events import (
    LearningPreferenceUpdatedEvent, LearningHabitDetectedEvent,
    LearningHabitSuggestedEvent, LearningWorkflowDetectedEvent,
    LearningWorkflowSuggestedEvent, LearningFeedbackRecordedEvent, LearningErrorEvent,
)
from spidy.core.event_bus import Event

def test_all_inherit_event():
    for cls in [LearningPreferenceUpdatedEvent, LearningHabitDetectedEvent,
                LearningHabitSuggestedEvent, LearningWorkflowDetectedEvent,
                LearningWorkflowSuggestedEvent, LearningFeedbackRecordedEvent,
                LearningErrorEvent]:
        assert issubclass(cls, Event)

def test_preference_updated_topic():
    assert LearningPreferenceUpdatedEvent().topic == "learning.preference_updated"

def test_preference_updated_fields():
    e = LearningPreferenceUpdatedEvent(key="k", new_value="v", confidence=0.9, source="explicit")
    assert e.key == "k" and e.new_value == "v" and e.confidence == 0.9

def test_preference_updated_defaults():
    e = LearningPreferenceUpdatedEvent()
    assert e.old_value is None and e.source == "implicit"

def test_habit_detected_topic():
    assert LearningHabitDetectedEvent().topic == "learning.habit_detected"

def test_habit_detected_fields():
    e = LearningHabitDetectedEvent(habit_id="h1", description="d", confidence=0.5, observed_count=3)
    assert e.habit_id == "h1" and e.observed_count == 3

def test_habit_suggested_topic():
    assert LearningHabitSuggestedEvent().topic == "learning.habit_suggested"

def test_workflow_detected_topic():
    assert LearningWorkflowDetectedEvent().topic == "learning.workflow_detected"

def test_workflow_detected_fields():
    e = LearningWorkflowDetectedEvent(workflow_id="w1", name="n", step_count=2, trigger_count=4)
    assert e.step_count == 2 and e.trigger_count == 4

def test_workflow_suggested_topic():
    assert LearningWorkflowSuggestedEvent().topic == "learning.workflow_suggested"

def test_workflow_suggested_fields():
    e = LearningWorkflowSuggestedEvent(workflow_id="w", name="n", next_step="git_commit")
    assert e.next_step == "git_commit"

def test_feedback_recorded_topic():
    assert LearningFeedbackRecordedEvent().topic == "learning.feedback_recorded"

def test_feedback_recorded_fields():
    e = LearningFeedbackRecordedEvent(signal_id="s1", session_id="sess", rating=-1.0, tags=["t"])
    assert e.rating == -1.0 and "t" in e.tags

def test_error_topic():
    assert LearningErrorEvent().topic == "learning.error"

def test_error_fields():
    e = LearningErrorEvent(component="HabitDetector", error_message="fail")
    assert e.component == "HabitDetector" and e.error_message == "fail"