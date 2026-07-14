"""Tests for spidy.learning.types - Learning Engine data types."""
from __future__ import annotations
from datetime import datetime, timezone
import pytest
from spidy.learning.types import (
    FeedbackSignal, Habit, LearningContext, Preference, Workflow,
    _short_id, _stable_id, _utcnow,
)


class TestHelpers:
    def test_utcnow_timezone_aware(self):
        dt = _utcnow()
        assert dt.tzinfo is not None

    def test_short_id_length(self):
        assert len(_short_id()) == 12

    def test_short_id_unique(self):
        assert len({ _short_id() for _ in range(50)}) == 50

    def test_stable_id_deterministic(self):
        assert _stable_id(["a","b"]) == _stable_id(["a","b"])

    def test_stable_id_order_sensitive(self):
        assert _stable_id(["a","b"]) != _stable_id(["b","a"])

    def test_stable_id_length(self):
        assert len(_stable_id(["x"])) == 12


class TestPreference:
    def test_create_defaults(self):
        p = Preference.create("key", "val")
        assert p.key == "key"
        assert p.confidence == 0.5
        assert p.source == "implicit"

    def test_conf_clamped_low(self):
        assert Preference.create("k","v",confidence=-5.0).confidence == 0.0

    def test_conf_clamped_high(self):
        assert Preference.create("k","v",confidence=5.0).confidence == 1.0

    def test_explicit_source(self):
        assert Preference.create("k","v",source="explicit").source == "explicit"

    def test_metadata(self):
        p = Preference.create("k","v",metadata={"n": 3})
        assert p.metadata["n"] == 3

    def test_with_conf_immutable(self):
        p = Preference.create("k","v",confidence=0.5)
        p2 = p.with_confidence(0.9)
        assert p.confidence == 0.5
        assert p2.confidence == 0.9

    def test_with_conf_clamped(self):
        p = Preference.create("k","v",confidence=0.5)
        assert p.with_confidence(5.0).confidence == 1.0
        assert p.with_confidence(-5.0).confidence == 0.0

    def test_with_value(self):
        p = Preference.create("k","old",confidence=0.5)
        p2 = p.with_value("new", new_confidence=0.8)
        assert p.value == "old"
        assert p2.value == "new"
        assert p2.confidence == 0.8

    def test_frozen(self):
        p = Preference.create("k","v")
        with pytest.raises((TypeError, AttributeError)):
            p.key = "x"

    def test_timestamp_utc(self):
        p = Preference.create("k","v")
        assert p.last_updated.tzinfo == timezone.utc


class TestHabit:
    def _h(self, count=3):
        return Habit.create(trigger={"active_app":"code.exe","topic":"code"}, action="intent:code", observed_count=count)

    def test_stable_id(self):
        assert self._h().id == self._h().id

    def test_conf_at_10(self):
        assert self._h(10).confidence == 1.0

    def test_conf_at_5(self):
        assert abs(self._h(5).confidence - 0.5) < 0.001

    def test_conf_capped(self):
        assert self._h(20).confidence == 1.0

    def test_description_nonempty(self):
        assert len(self._h().description) > 0

    def test_observed_increments(self):
        h = self._h(3)
        h2 = h.observed()
        assert h.observed_count == 3
        assert h2.observed_count == 4

    def test_observed_immutable(self):
        h = self._h(3)
        h.observed()
        assert h.observed_count == 3

    def test_custom_desc(self):
        h = Habit.create(trigger={"active_app":"code.exe"}, action="open", description="My desc")
        assert h.description == "My desc"

    def test_frozen(self):
        h = self._h()
        with pytest.raises((TypeError, AttributeError)):
            h.action = "x"


class TestWorkflow:
    def _w(self, count=3):
        return Workflow.create(steps=["open_file","edit_code","git_commit"], trigger_count=count)

    def test_stable_id(self):
        assert self._w().id == self._w().id

    def test_auto_name(self):
        w = self._w()
        assert "open_file" in w.name
        assert "\u2192" in w.name

    def test_custom_name(self):
        w = Workflow.create(steps=["a","b"], name="Custom")
        assert w.name == "Custom"

    def test_conf_at_5(self):
        assert self._w(5).confidence == 1.0

    def test_conf_at_1(self):
        assert abs(self._w(1).confidence - 0.2) < 0.001

    def test_steps_tuple(self):
        assert isinstance(self._w().steps, tuple)

    def test_suggested_false(self):
        assert self._w().suggested is False

    def test_executed_increments(self):
        w = self._w(2)
        w2 = w.executed()
        assert w.trigger_count == 2
        assert w2.trigger_count == 3

    def test_frozen(self):
        w = self._w()
        with pytest.raises((TypeError, AttributeError)):
            w.name = "x"


class TestFeedbackSignal:
    def test_positive(self):
        s = FeedbackSignal.create("s","u","r",rating=1.0)
        assert s.is_positive and not s.is_negative and not s.is_neutral

    def test_negative(self):
        s = FeedbackSignal.create("s","u","r",rating=-1.0)
        assert s.is_negative and not s.is_positive

    def test_neutral(self):
        assert FeedbackSignal.create("s","u","r",rating=0.0).is_neutral

    def test_rating_clamped_high(self):
        assert FeedbackSignal.create("s","u","r",rating=5.0).rating == 1.0

    def test_rating_clamped_low(self):
        assert FeedbackSignal.create("s","u","r",rating=-5.0).rating == -1.0

    def test_unique_ids(self):
        s1 = FeedbackSignal.create("s","u","r",rating=0.0)
        s2 = FeedbackSignal.create("s","u","r",rating=0.0)
        assert s1.id != s2.id

    def test_utterance_truncated(self):
        s = FeedbackSignal.create("s","x"*1000,"r",rating=0.0)
        assert len(s.utterance) <= 500

    def test_response_truncated(self):
        s = FeedbackSignal.create("s","u","y"*1000,rating=0.0)
        assert len(s.response) <= 500

    def test_tags_tuple(self):
        s = FeedbackSignal.create("s","u","r",rating=0.0,tags=["a","b"])
        assert isinstance(s.tags, tuple)
        assert "a" in s.tags

    def test_neutral_boundary_pos(self):
        assert FeedbackSignal.create("s","u","r",rating=0.3).is_neutral

    def test_neutral_boundary_neg(self):
        assert FeedbackSignal.create("s","u","r",rating=-0.3).is_neutral

    def test_pos_above_boundary(self):
        assert FeedbackSignal.create("s","u","r",rating=0.31).is_positive

    def test_neg_below_boundary(self):
        assert FeedbackSignal.create("s","u","r",rating=-0.31).is_negative


class TestLearningContext:
    def test_from_dict_full(self):
        ctx = LearningContext.from_dict({"hour_of_day":9,"day_of_week":1,"active_app":"Code.exe","active_window_title":"main.py","topic":"code"})
        assert ctx.hour_of_day == 9
        assert ctx.topic == "code"
        assert ctx.active_app == "Code.exe"

    def test_from_dict_defaults(self):
        now = datetime.now(tz=timezone.utc)
        ctx = LearningContext.from_dict({})
        assert ctx.hour_of_day == now.hour
        assert ctx.active_app == ""

    def test_time_bucket_morning(self):
        ctx = LearningContext(hour_of_day=8,day_of_week=0,active_app="",active_window_title="",topic="")
        assert ctx.time_bucket == "morning"

    def test_time_bucket_afternoon(self):
        ctx = LearningContext(hour_of_day=14,day_of_week=0,active_app="",active_window_title="",topic="")
        assert ctx.time_bucket == "afternoon"

    def test_time_bucket_evening(self):
        ctx = LearningContext(hour_of_day=19,day_of_week=0,active_app="",active_window_title="",topic="")
        assert ctx.time_bucket == "evening"

    def test_time_bucket_night(self):
        ctx = LearningContext(hour_of_day=1,day_of_week=0,active_app="",active_window_title="",topic="")
        assert ctx.time_bucket == "night"

    def test_time_bucket_morning_start(self):
        ctx = LearningContext(hour_of_day=6,day_of_week=0,active_app="",active_window_title="",topic="")
        assert ctx.time_bucket == "morning"

    def test_frozen(self):
        ctx = LearningContext(hour_of_day=9,day_of_week=0,active_app="",active_window_title="",topic="")
        with pytest.raises((TypeError, AttributeError)):
            ctx.hour_of_day = 10
