"""
test_wake_state_machine.py -- Unit tests for wake event HUD restoration, continuous voice wake recovery, and wake-word command suppression.
======================================================================================================================================
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from PySide6.QtWidgets import QApplication

from spidy.perception.voice.engine import WakeWordDetectedEvent, VoiceTranscriptEvent
from spidy.ui.events import UIShowEvent, UIRestoreEvent
from spidy.ui.overlay import MIN_WIDTH, MIN_HEIGHT
from spidy.voice.continuous import ContinuousVoiceController
from spidy.voice.interruption import InterruptionHandler
from spidy.voice.session import VoiceSessionManager
from spidy.voice.streaming_tts import StreamingTTSWrapper


@pytest.fixture(scope="session")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.ui.enabled = True
    config.ui.theme = "dark"
    config.ui.width = 1000
    config.ui.height = 600
    config.ui.position = "center"
    config.ui.always_on_top = True
    config.ui.animate = False
    config.ui.hotkey = "ctrl+space"
    return config


# ── Test Doubles ─────────────────────────────────────────────────────────────

class FakeTTSEngine:
    def __init__(self):
        self.spoken = []
        self._speaking = False

    async def speak(self, text: str) -> None:
        self.spoken.append(text)

    def stop(self) -> None:
        pass


class FakeVoiceEngine:
    def __init__(self):
        self._tts = FakeTTSEngine()
        self.conversation_mode = False
        self.paused = False
        self.relisten_calls = 0
        self._capture_engine = MagicMock()
        self._wake_model = MagicMock()

    def signal_relisten(self) -> None:
        self.relisten_calls += 1

    async def _on_brain_response(self, event) -> None:
        pass


class FakeBrain:
    def __init__(self):
        self.process = AsyncMock(return_value="Here is your answer.")
        self.run_goal = AsyncMock(return_value="Goal completed.")
        self.cancel_goal = AsyncMock(return_value=True)


class FakeBus:
    def __init__(self):
        self._subscribers = {}
        self.published = []

    async def publish(self, event) -> None:
        self.published.append(event)
        topic = getattr(event, "topic", "")
        for cb in self._subscribers.get(topic, []):
            if asyncio.iscoroutinefunction(cb):
                await cb(event)
            else:
                cb(event)

    def subscribe(self, topic: str, callback) -> None:
        self._subscribers.setdefault(topic, []).append(callback)

    def unsubscribe(self, topic: str, callback) -> None:
        try:
            self._subscribers.get(topic, []).remove(callback)
        except ValueError:
            pass

    def publish_threadsafe(self, event, loop=None) -> None:
        self.published.append(event)


def make_controller(fake_bus, fake_brain, fake_voice_engine):
    session_mgr = VoiceSessionManager(
        bus=fake_bus,
        timeout_seconds=60.0,
        continuous_mode=True,
    )
    streaming_tts = StreamingTTSWrapper(engine=fake_voice_engine._tts)
    interruption = InterruptionHandler(
        bus=fake_bus,
        brain=fake_brain,
        tts=fake_voice_engine._tts,
        session=session_mgr,
    )
    return ContinuousVoiceController(
        voice_engine=fake_voice_engine,
        brain=fake_brain,
        bus=fake_bus,
        session_manager=session_mgr,
        streaming_tts=streaming_tts,
        interruption_handler=interruption,
        continuous_mode=True,
        measure_latency=False,
    )


# ─── 1. SpidyApp Wake Event HUD Restoration Tests ─────────────────────────────

@pytest.mark.asyncio
async def test_wake_event_restores_hidden_hud(qapp, mock_config):
    """When the HUD is minimized/hidden, wake word event restores it to full size."""
    from spidy.ui.app import SpidyApp

    bus = MagicMock()
    app = SpidyApp(bus=bus, config=mock_config)
    app.start()

    # Minimize HUD
    app.minimize_hud()
    qapp.processEvents()
    assert app._overlay.isVisible() is False
    assert app._overlay._is_hidden is True

    # Receive wake detected event
    wake_event = WakeWordDetectedEvent(wake_phrase="hey spidy", confidence=0.85)
    await app._handle_wake_detected(wake_event)
    qapp.processEvents()

    # HUD must be visible and restored to full size
    assert app._overlay.isVisible() is True
    assert app._overlay._is_hidden is False
    assert app._overlay.width() >= MIN_WIDTH
    assert app._overlay.height() >= MIN_HEIGHT

    app._overlay.close()


@pytest.mark.asyncio
async def test_ui_restore_event_restores_hud(qapp, mock_config):
    """Publishing UIRestoreEvent restores HUD to visible with proper geometry."""
    from spidy.ui.app import SpidyApp

    bus = MagicMock()
    app = SpidyApp(bus=bus, config=mock_config)
    app.start()

    # Minimize HUD
    app.minimize_hud()
    qapp.processEvents()
    assert app._overlay.isVisible() is False

    # Receive restore event (e.g. from single-instance IPC)
    restore_event = UIRestoreEvent(source="single_instance_ipc")
    await app._handle_restore(restore_event)
    qapp.processEvents()

    assert app._overlay.isVisible() is True
    assert app._overlay.width() >= MIN_WIDTH
    assert app._overlay.height() >= MIN_HEIGHT

    app._overlay.close()


# ─── 2. Continuous Voice Controller Wake Recovery & Suppression Tests ─────────

@pytest.mark.asyncio
async def test_continuous_voice_speech_recovery_for_wake_phrase():
    """When idle, saying 'wake up spidy' triggers deterministic wake recovery without calling LLM."""
    bus = FakeBus()
    brain = FakeBrain()
    engine = FakeVoiceEngine()
    controller = make_controller(bus, brain, engine)
    await controller.start()

    transcript_event = VoiceTranscriptEvent(
        text="wake up spidy",
        confidence=0.92,
    )

    await controller._on_transcript(transcript_event)

    # Voice session should be awakened via wake_word.detected publication
    assert controller._session.is_active is True

    # UIShowEvent must have been published
    show_events = [e for e in bus.published if isinstance(e, UIShowEvent)]
    assert len(show_events) >= 1
    assert show_events[0].source == "wake_word"

    # LLM must NOT be called for wake phrases
    brain.process.assert_not_called()
    brain.run_goal.assert_not_called()


@pytest.mark.asyncio
async def test_wake_words_never_dispatched_as_commands_to_brain():
    """Saying wake words must be filtered/suppressed and never forwarded as regular user commands to LLM."""
    bus = FakeBus()
    brain = FakeBrain()
    engine = FakeVoiceEngine()
    controller = make_controller(bus, brain, engine)
    await controller.start()

    # Start an active session
    await controller._session.activate(wake_word="hey spidy")
    assert controller._session.is_active is True

    # Standalone wake word transcript while active
    transcript_event = VoiceTranscriptEvent(
        text="Hey Spidy",
        confidence=0.95,
    )

    await controller._on_transcript(transcript_event)

    # Brain must NOT be asked to process the wake word as a command
    brain.process.assert_not_called()
    brain.run_goal.assert_not_called()


@pytest.mark.asyncio
async def test_wake_word_handler_publishes_ui_show_and_starts_session():
    """WakeWordDetectedEvent from engine starts voice session and fires UIShowEvent."""
    bus = FakeBus()
    brain = FakeBrain()
    engine = FakeVoiceEngine()
    controller = make_controller(bus, brain, engine)
    await controller.start()

    wake_event = WakeWordDetectedEvent(wake_phrase="hey spidy", confidence=0.88)
    await controller._on_wake_word(wake_event)

    assert controller._session.is_active is True
    show_events = [e for e in bus.published if isinstance(e, UIShowEvent)]
    assert len(show_events) >= 1
    assert show_events[0].source == "wake_word"
