"""
M14 Integration Tests — Voice Companion Pipeline
==================================================
Tests the full M14 voice pipeline end-to-end with faked hardware:
  wake word → session open → STT → Brain → StreamingTTS → session state

All external dependencies (audio, LLM, models) are mocked.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from spidy.voice.continuous import ContinuousVoiceController
from spidy.voice.session import VoiceSessionManager, VoiceSessionState
from spidy.voice.streaming_tts import StreamingTTSWrapper
from spidy.voice.interruption import InterruptionHandler, InterruptCommand
from spidy.voice.settings import VoiceSettings


# ── Shared fakes ──────────────────────────────────────────────────────────────


class FakeTTSEngine:
    def __init__(self):
        self.spoken: list[str] = []
        self._speaking = False

    async def speak(self, text: str) -> None:
        self.spoken.append(text)
        await asyncio.sleep(0)

    async def synthesize(self, text: str):
        return None

    def stop(self) -> None:
        pass

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    @property
    def voice_name(self) -> str:
        return "fake_voice"

    @property
    def is_loaded(self) -> bool:
        return True


class FakeBus:
    def __init__(self):
        self._subscribers: dict[str, list] = {}
        self.published: list = []

    async def publish(self, event):
        self.published.append(event)
        topic = getattr(event, "topic", "")
        for cb in list(self._subscribers.get(topic, [])):
            if asyncio.iscoroutinefunction(cb):
                await cb(event)
            else:
                cb(event)

    def subscribe(self, topic: str, callback) -> None:
        self._subscribers.setdefault(topic, []).append(callback)

    def unsubscribe(self, topic: str, callback) -> None:
        subs = self._subscribers.get(topic, [])
        if callback in subs:
            subs.remove(callback)

    def publish_threadsafe(self, event, loop=None) -> None:
        pass


class FakeBrain:
    def __init__(self):
        self.process = AsyncMock(return_value="I can help with that.")
        self.run_goal = AsyncMock(return_value="Goal completed successfully.")
        self.cancel_goal = AsyncMock(return_value=True)


class FakeVoiceEngine:
    def __init__(self):
        self._tts = FakeTTSEngine()
        self.conversation_mode_enabled = False

    def enable_conversation_mode(self, timeout_seconds=60.0):
        self.conversation_mode_enabled = True


# ── Fixture factory ───────────────────────────────────────────────────────────


def _make_stack(continuous_mode=True):
    """Build a complete M14 voice stack with fakes."""
    bus = FakeBus()
    brain = FakeBrain()
    engine = FakeVoiceEngine()

    session = VoiceSessionManager(
        bus=bus,
        timeout_seconds=60.0,
        max_turns=50,
        continuous_mode=continuous_mode,
    )
    streaming_tts = StreamingTTSWrapper(engine=engine._tts)
    interruption = InterruptionHandler(
        bus=bus, brain=brain, tts=engine._tts, session=session
    )
    controller = ContinuousVoiceController(
        voice_engine=engine,
        brain=brain,
        bus=bus,
        session_manager=session,
        streaming_tts=streaming_tts,
        interruption_handler=interruption,
        continuous_mode=continuous_mode,
        measure_latency=False,
    )
    return controller, session, brain, engine, bus


# ── Test: Basic voice flow ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_basic_voice_flow():
    """
    Full pipeline: wake → session opens → transcript → Brain → response.
    """
    controller, session, brain, engine, bus = _make_stack()
    await controller.start()

    # 1. Simulate wake word
    class WakeEvent:
        topic = "wake_word.detected"
        model_name = "hey_spidy"

    await bus.publish(WakeEvent())
    assert session.is_active

    # 2. Simulate transcript
    class TranscriptEvent:
        topic = "voice.transcript"
        text = "what time is it"

    await bus.publish(TranscriptEvent())

    # Brain should have been called
    assert brain.process.called or brain.run_goal.called

    await controller.stop()


# ── Test: Continuous conversation (3 turns) ────────────────────────────────────


@pytest.mark.asyncio
async def test_continuous_conversation_three_turns():
    """
    After each response, session stays AWAKE for next utterance
    (no wake word needed between turns).
    """
    controller, session, brain, engine, bus = _make_stack(continuous_mode=True)
    await controller.start()

    # Wake up
    class WakeEvent:
        topic = "wake_word.detected"
        model_name = "hey_spidy"

    await bus.publish(WakeEvent())

    # 3 turns without re-waking
    for i, utterance in enumerate([
        "open chrome",
        "search youtube",
        "open first result",
    ]):
        class TEvent:
            topic = "voice.transcript"
            text = utterance

        await bus.publish(TEvent())
        # Session should still be active after each turn in continuous mode
        assert session.is_active, f"Session should be active after turn {i+1}"

    # All 3 turns processed by brain
    total_calls = brain.process.call_count + brain.run_goal.call_count
    assert total_calls == 3

    await controller.stop()


# ── Test: Interruption mid-conversation ───────────────────────────────────────


@pytest.mark.asyncio
async def test_interruption_cancel_stops_goal():
    """
    User says "cancel" mid-conversation — goal is cancelled and session closes.
    """
    controller, session, brain, engine, bus = _make_stack()
    await controller.start()

    # Wake up
    class WakeEvent:
        topic = "wake_word.detected"
        model_name = "hey_spidy"

    await bus.publish(WakeEvent())

    # User speaks a goal
    class GoalEvent:
        topic = "voice.transcript"
        text = "open vs code"

    await bus.publish(GoalEvent())

    # User interrupts
    class CancelEvent:
        topic = "voice.transcript"
        text = "cancel"

    await bus.publish(CancelEvent())

    # Brain.cancel_goal should have been called
    brain.cancel_goal.assert_awaited()

    await controller.stop()


# ── Test: Pause and resume ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pause_and_resume_via_voice():
    """
    User says "wait" → session pauses.
    User says "resume" → session resumes and accepts new utterances.
    """
    controller, session, brain, engine, bus = _make_stack()
    await controller.start()

    class WakeEvent:
        topic = "wake_word.detected"
        model_name = "hey_spidy"

    await bus.publish(WakeEvent())

    # Pause
    class WaitEvent:
        topic = "voice.transcript"
        text = "wait"

    await bus.publish(WaitEvent())
    assert session.is_paused

    # Resume
    class ResumeEvent:
        topic = "voice.transcript"
        text = "resume"

    await bus.publish(ResumeEvent())
    assert not session.is_paused

    await controller.stop()


# ── Test: Session timeout ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_timeout():
    """Session deactivates after inactivity timeout."""
    bus = FakeBus()
    brain = FakeBrain()
    engine = FakeVoiceEngine()

    # Create session with very short timeout directly
    session = VoiceSessionManager(
        bus=bus,
        timeout_seconds=0.1,   # Short timeout for test speed
        max_turns=50,
        continuous_mode=True,
    )
    streaming_tts = StreamingTTSWrapper(engine=engine._tts)
    interruption = InterruptionHandler(
        bus=bus, brain=brain, tts=engine._tts, session=session
    )
    controller = ContinuousVoiceController(
        voice_engine=engine,
        brain=brain,
        bus=bus,
        session_manager=session,
        streaming_tts=streaming_tts,
        interruption_handler=interruption,
        continuous_mode=True,
        measure_latency=False,
    )

    await controller.start()

    class WakeEvent:
        topic = "wake_word.detected"
        model_name = "hey_spidy"

    await bus.publish(WakeEvent())
    assert session.is_active

    # Poll until session times out (timeout is 0.1s, polling 2 seconds max)
    for _ in range(100):
        if not session.is_active:
            break
        await asyncio.sleep(0.02)

    assert not session.is_active, "Session should have timed out after 0.1s inactivity"

    await controller.stop()


# ── Test: VoiceSettings ───────────────────────────────────────────────────────


def test_voice_settings_defaults():
    bus = FakeBus()
    settings = VoiceSettings(bus=bus)

    assert settings.get("volume") == 1.0
    assert settings.get("conversation_timeout") == 60.0
    assert settings.get("wake_word_enabled") is True


def test_voice_settings_set_valid():
    bus = FakeBus()
    settings = VoiceSettings(bus=bus)

    settings.set("volume", 0.7)
    assert settings.get("volume") == pytest.approx(0.7)


def test_voice_settings_set_out_of_range():
    bus = FakeBus()
    settings = VoiceSettings(bus=bus)

    settings.set("volume", 2.0)  # Invalid: > 1.0
    assert settings.get("volume") == 1.0  # Unchanged


def test_voice_settings_unknown_key():
    bus = FakeBus()
    settings = VoiceSettings(bus=bus)
    # Should not raise
    settings.set("nonexistent_key", "value")


def test_voice_settings_apply_to_session():
    bus = FakeBus()
    settings = VoiceSettings(bus=bus)
    settings.set("conversation_timeout", 30.0)

    session = VoiceSessionManager(bus=bus, timeout_seconds=60.0)
    settings.apply_to(session_manager=session)
    assert session.timeout_seconds == pytest.approx(30.0)


def test_voice_settings_all_settings():
    bus = FakeBus()
    settings = VoiceSettings(bus=bus)
    all_s = settings.all_settings
    assert "volume" in all_s
    assert "conversation_timeout" in all_s
    assert "wake_word_enabled" in all_s


# ── Test: NullWakeWordModel ───────────────────────────────────────────────────


def test_null_wake_word_model_trigger():
    from spidy.perception.voice.wake_word.openwakeword import NullWakeWordModel
    import numpy as np

    model = NullWakeWordModel(trigger_after_chunks=3)
    model.load()

    dummy = np.zeros(1280, dtype=np.float32)
    # First 2 calls should return 0
    assert model.process_chunk(dummy) == 0.0
    assert model.process_chunk(dummy) == 0.0
    # 3rd call should trigger
    assert model.process_chunk(dummy) == 1.0


def test_null_wake_word_force_trigger():
    from spidy.perception.voice.wake_word.openwakeword import NullWakeWordModel
    import numpy as np

    model = NullWakeWordModel()
    model.load()
    model.force_trigger()

    dummy = np.zeros(1280, dtype=np.float32)
    assert model.process_chunk(dummy) == 1.0
    # Second call should NOT trigger (forced is one-shot)
    assert model.process_chunk(dummy) == 0.0
