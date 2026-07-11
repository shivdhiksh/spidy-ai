"""
Tests for spidy.vision.events
"""
import pytest
from spidy.core.event_bus import Event
from spidy.vision.events import (
    VisionAnalysisCompletedEvent,
    VisionCaptureCompletedEvent,
    VisionCaptureStartedEvent,
    VisionErrorEvent,
    VisionOCRCompletedEvent,
)


# ─── Topic Constants ──────────────────────────────────────────────────────────

EXPECTED_TOPICS = {
    VisionCaptureStartedEvent: "vision.capture_started",
    VisionCaptureCompletedEvent: "vision.capture_completed",
    VisionOCRCompletedEvent: "vision.ocr_completed",
    VisionAnalysisCompletedEvent: "vision.analysis_completed",
    VisionErrorEvent: "vision.error",
}


class TestVisionEventTopics:
    @pytest.mark.parametrize("event_cls, expected_topic", list(EXPECTED_TOPICS.items()))
    def test_topic_string(self, event_cls, expected_topic):
        """Every event has the correct topic class attribute."""
        assert event_cls.topic == expected_topic

    @pytest.mark.parametrize("event_cls", list(EXPECTED_TOPICS.keys()))
    def test_is_event_subclass(self, event_cls):
        """Every vision event extends the base Event."""
        assert issubclass(event_cls, Event)

    @pytest.mark.parametrize("event_cls, expected_topic", list(EXPECTED_TOPICS.items()))
    def test_topic_starts_with_vision(self, event_cls, expected_topic):
        assert expected_topic.startswith("vision.")

    @pytest.mark.parametrize("event_cls", list(EXPECTED_TOPICS.keys()))
    def test_instance_topic_matches_class_topic(self, event_cls):
        """Instantiated event has the same topic as the class attribute."""
        obj = event_cls()
        assert obj.topic == event_cls.topic


# ─── VisionCaptureStartedEvent ────────────────────────────────────────────────


class TestVisionCaptureStartedEvent:
    def test_defaults(self):
        e = VisionCaptureStartedEvent()
        assert e.topic == "vision.capture_started"
        assert e.source == "fullscreen"
        assert e.monitor_index == 0

    def test_with_values(self):
        e = VisionCaptureStartedEvent(source="window", monitor_index=1)
        assert e.source == "window"
        assert e.monitor_index == 1


# ─── VisionCaptureCompletedEvent ─────────────────────────────────────────────


class TestVisionCaptureCompletedEvent:
    def test_defaults(self):
        e = VisionCaptureCompletedEvent()
        assert e.topic == "vision.capture_completed"
        assert e.source == "fullscreen"
        assert e.width == 0
        assert e.height == 0
        assert e.size_bytes == 0
        assert e.duration_ms == 0.0
        assert e.monitor_index == 0

    def test_with_values(self):
        e = VisionCaptureCompletedEvent(
            source="region",
            width=800,
            height=600,
            size_bytes=12345,
            duration_ms=12.5,
            monitor_index=2,
        )
        assert e.width == 800
        assert e.height == 600
        assert e.size_bytes == 12345
        assert e.duration_ms == 12.5
        assert e.monitor_index == 2


# ─── VisionOCRCompletedEvent ──────────────────────────────────────────────────


class TestVisionOCRCompletedEvent:
    def test_defaults(self):
        e = VisionOCRCompletedEvent()
        assert e.topic == "vision.ocr_completed"
        assert e.text_length == 0
        assert e.confidence == 0.0
        assert e.block_count == 0
        assert e.language == "en"
        assert e.available is True

    def test_unavailable(self):
        e = VisionOCRCompletedEvent(available=False)
        assert e.available is False

    def test_with_values(self):
        e = VisionOCRCompletedEvent(
            text_length=250,
            confidence=0.87,
            block_count=12,
            language="fr",
        )
        assert e.text_length == 250
        assert e.confidence == 0.87
        assert e.block_count == 12
        assert e.language == "fr"


# ─── VisionAnalysisCompletedEvent ────────────────────────────────────────────


class TestVisionAnalysisCompletedEvent:
    def test_defaults(self):
        e = VisionAnalysisCompletedEvent()
        assert e.topic == "vision.analysis_completed"
        assert e.active_app == ""
        assert e.active_window == ""
        assert e.window_count == 0
        assert e.region_count == 0
        assert e.available is True

    def test_with_values(self):
        e = VisionAnalysisCompletedEvent(
            active_app="Code.exe",
            active_window="editor",
            window_count=5,
            region_count=3,
        )
        assert e.active_app == "Code.exe"
        assert e.window_count == 5
        assert e.region_count == 3


# ─── VisionErrorEvent ────────────────────────────────────────────────────────


class TestVisionErrorEvent:
    def test_defaults(self):
        e = VisionErrorEvent()
        assert e.topic == "vision.error"
        assert e.operation == ""
        assert e.error == ""
        assert e.source == ""

    def test_with_values(self):
        e = VisionErrorEvent(
            operation="capture",
            error="mss not installed",
            source="screenshot_engine",
        )
        assert e.operation == "capture"
        assert e.error == "mss not installed"
        assert e.source == "screenshot_engine"


# ─── All events usable without arguments ─────────────────────────────────────


class TestAllEventsInstantiable:
    def test_all_events_instantiable_with_defaults(self):
        """All vision events must be constructable with no arguments."""
        event_classes = [
            VisionCaptureStartedEvent,
            VisionCaptureCompletedEvent,
            VisionOCRCompletedEvent,
            VisionAnalysisCompletedEvent,
            VisionErrorEvent,
        ]
        for cls in event_classes:
            obj = cls()
            assert isinstance(obj, Event)
