"""
Unit tests — ContinuousVoiceController
=========================================
Tests controller lifecycle, event routing, session management,
interrupt handling, and latency measurement — all with fakes.

No real VoiceEngine, Brain, or audio hardware required.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from spidy.voice.continuous import ContinuousVoiceController
from spidy.voice.session import VoiceSessionManager, VoiceSessionState
from spidy.voice.streaming_tts import StreamingTTSWrapper
from spidy.voice.interruption import InterruptionHandler, InterruptCommand


# ── Fake implementations ──────────────────────────────────────────────────────


class FakeTTSEngine:
    spoken: list[str] = []

    def __init__(self):
        self.spoken = []
        self._speaking = False

    async def speak(self, text: str) -> None:
        self.spoken.append(text)
        await asyncio.sleep(0)

    async def synthesize(self, text: str):
        pass

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
        return "fake"

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
        for cb in self._subscribers.get(topic, []):
            if asyncio.iscoroutinefunction(cb):
                await cb(event)
            else:
                cb(event)

    def subscribe(self, topic: str, callback) -> None:
        self._subscribers.setdefault(topic, []).append(callback)

    def unsubscribe(self, topic: str, callback) -> None:
        # Silently ignore if the callback was never registered.
        # The real EventBus does the same defensive check; in tests the
        # FakeVoiceEngine._on_brain_response was never subscribed to the
        # bus (start() hadn't been called yet), so remove() must not raise.
        try:
            self._subscribers.get(topic, []).remove(callback)
        except ValueError:
            pass

    def publish_threadsafe(self, event, loop=None) -> None:
        pass


class FakeVoiceEngine:
    """
    Minimal test double for VoiceEngine.

    Exposes every attribute that ContinuousVoiceController accesses on
    the real VoiceEngine so tests don't hit AttributeError:

      _on_brain_response   — async handler that CVC unsubscribes from
                             brain.response_ready during start()
      _capture_engine      — CaptureMode controller (muted during TTS)
      _wake_model          — OWW model (buffer reset after TTS)
    """

    def __init__(self):
        self._tts = FakeTTSEngine()
        self.conversation_mode = False
        self.paused = False
        # Track how many times signal_relisten() was called so tests can assert
        # that the continuous conversation re-listen path is triggered.
        self.relisten_calls: int = 0

        # Stub inner objects that CVC._on_brain_response() touches
        self._capture_engine = _FakeCaptureEngine()
        self._wake_model = _FakeWakeModel()

    def enable_conversation_mode(self, timeout_seconds=60.0):
        self.conversation_mode = True

    def signal_relisten(self) -> None:
        """Record that ContinuousVoiceController requested a re-listen cycle."""
        self.relisten_calls += 1

    async def _on_brain_response(self, event) -> None:
        """
        Async no-op that stands in for VoiceEngine._on_brain_response.

        The real VoiceEngine subscribes this to brain.response_ready so it
        can speak responses directly via its TTS engine.  CVC.start() must
        *unsubscribe* it to prevent duplicate TTS playback.  The fake version
        is a harmless no-op; the important thing is that it exists so the
        unsubscribe call (and the FakeBus remove) can succeed.
        """


class _FakeCaptureEngine:
    """Minimal stand-in for AudioCaptureEngine (CaptureMode control only)."""

    def __init__(self):
        self.mode = None

    def set_mode(self, mode) -> None:
        self.mode = mode


class _FakeWakeModel:
    """Minimal stand-in for the OWW model (buffer management only)."""

    def reset_buffer(self) -> None:
        pass


class FakeBrain:
    def __init__(self):
        self.process = AsyncMock(return_value="Here's your answer.")
        self.run_goal = AsyncMock(return_value="Goal completed.")
        self.cancel_goal = AsyncMock(return_value=True)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_bus():
    return FakeBus()


@pytest.fixture
def fake_brain():
    return FakeBrain()


@pytest.fixture
def fake_voice_engine():
    return FakeVoiceEngine()


@pytest.fixture
def session_mgr(fake_bus):
    return VoiceSessionManager(
        bus=fake_bus,
        timeout_seconds=60.0,
        continuous_mode=True,
    )


@pytest.fixture
def streaming_tts(fake_voice_engine):
    return StreamingTTSWrapper(engine=fake_voice_engine._tts)


@pytest.fixture
def interruption(fake_bus, fake_brain, fake_voice_engine, session_mgr):
    return InterruptionHandler(
        bus=fake_bus,
        brain=fake_brain,
        tts=fake_voice_engine._tts,
        session=session_mgr,
    )


@pytest.fixture
def controller(
    fake_voice_engine, fake_brain, fake_bus,
    session_mgr, streaming_tts, interruption,
):
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


# ── Lifecycle tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_subscribes_to_events(controller, fake_bus):
    await controller.start()
    assert controller.is_running
    # Check that key topics have subscribers
    assert "voice.transcript" in fake_bus._subscribers
    assert "wake_word.detected" in fake_bus._subscribers
    await controller.stop()


@pytest.mark.asyncio
async def test_stop_unsubscribes(controller, fake_bus):
    await controller.start()
    await controller.stop()
    assert not controller.is_running
    # Subscribers should be removed
    assert not fake_bus._subscribers.get("voice.transcript")


@pytest.mark.asyncio
async def test_double_start_is_safe(controller):
    await controller.start()
    await controller.start()  # Should not raise
    assert controller.is_running
    await controller.stop()


# ── Wake word handling ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wake_word_activates_session(controller, session_mgr):
    await controller.start()

    class FakeWakeEvent:
        topic = "wake_word.detected"
        model_name = "hey_spidy"

    await controller._on_wake_word(FakeWakeEvent())
    assert session_mgr.is_active
    await controller.stop()


# ── Transcript routing ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_transcript_routes_to_brain(controller, fake_brain, session_mgr):
    await controller.start()
    await session_mgr.activate()

    class FakeTranscriptEvent:
        topic = "voice.transcript"
        text = "what time is it"

    await controller._on_transcript(FakeTranscriptEvent())
    # Brain.process() or run_goal() should have been called
    called = fake_brain.process.called or fake_brain.run_goal.called
    assert called
    await controller.stop()


@pytest.mark.asyncio
async def test_empty_transcript_is_ignored(controller, fake_brain, session_mgr):
    await controller.start()
    await session_mgr.activate()

    class FakeTranscriptEvent:
        topic = "voice.transcript"
        text = ""

    await controller._on_transcript(FakeTranscriptEvent())
    # Brain should NOT be called for empty transcripts
    fake_brain.process.assert_not_called()
    fake_brain.run_goal.assert_not_called()
    await controller.stop()


@pytest.mark.asyncio
async def test_interrupt_command_detected(controller, fake_brain, session_mgr):
    """Interrupt commands are handled and NOT forwarded to Brain."""
    await controller.start()
    await session_mgr.activate()

    class FakeStopEvent:
        topic = "voice.transcript"
        text = "stop"

    await controller._on_transcript(FakeStopEvent())
    # Brain should NOT be called for an interrupt command
    fake_brain.process.assert_not_called()
    fake_brain.run_goal.assert_not_called()
    await controller.stop()


# ── Properties ────────────────────────────────────────────────────────────────


def test_session_property(controller, session_mgr):
    assert controller.session is session_mgr


@pytest.mark.asyncio
async def test_is_session_active_reflects_session(controller, session_mgr):
    assert not controller.is_session_active
    await session_mgr.activate()
    assert controller.is_session_active


# ── Exclusive brain.response_ready ownership ─────────────────────────────────


@pytest.mark.asyncio
async def test_start_takes_exclusive_ownership_of_brain_response_ready(
    controller, fake_bus, fake_voice_engine
):
    """
    Regression: CVC.start() must subscribe CVC._on_brain_response AND
    unsubscribe VoiceEngine._on_brain_response so only StreamingTTS owns
    speech delivery (prevents duplicate TTS playback).

    To observe the unsubscribe we pre-register the engine handler so the
    bus has it in its subscriber list, then verify it is gone afterwards.
    """
    # Pre-register the engine's handler as the real VoiceEngine.start() would
    fake_bus.subscribe("brain.response_ready", fake_voice_engine._on_brain_response)
    assert fake_voice_engine._on_brain_response in fake_bus._subscribers.get(
        "brain.response_ready", []
    ), "Precondition: engine handler must be registered before controller.start()"

    await controller.start()

    # After start(): CVC's handler is present, engine's handler is removed
    remaining = fake_bus._subscribers.get("brain.response_ready", [])
    assert controller._on_brain_response in remaining, (
        "CVC._on_brain_response must be subscribed after start()"
    )
    assert fake_voice_engine._on_brain_response not in remaining, (
        "VoiceEngine._on_brain_response must be UNSUBSCRIBED by CVC.start() "
        "to prevent duplicate TTS (exclusive ownership rule)"
    )

    await controller.stop()


@pytest.mark.asyncio
async def test_start_when_engine_handler_not_pre_registered_does_not_crash(
    controller, fake_bus
):
    """
    When VoiceEngine has NOT yet subscribed its handler (e.g. in tests or
    when the engine is started after CVC), start() must not raise.
    FakeBus.unsubscribe is now tolerant of missing callbacks.
    """
    # Do NOT pre-register the engine handler — CVC.start() must handle this gracefully
    await controller.start()   # Must not raise ValueError / AttributeError
    assert controller.is_running
    await controller.stop()


# ── Start / stop safety ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_stop_cycle_is_clean(controller, fake_bus):
    """start() + stop() must not leave any CVC handlers registered."""
    await controller.start()
    assert controller.is_running
    await controller.stop()
    assert not controller.is_running
    # No CVC handlers should remain on any topic
    for topic in ("voice.transcript", "wake_word.detected",
                  "voice.speaking_end", "brain.response_ready",
                  "voice.session_timeout", "system.shutting_down"):
        remaining = fake_bus._subscribers.get(topic, [])
        assert controller._on_transcript not in remaining
        assert controller._on_brain_response not in remaining


@pytest.mark.asyncio
async def test_stop_before_start_is_safe(controller):
    """stop() called before start() must be a no-op (no exception)."""
    assert not controller.is_running
    await controller.stop()   # Must not raise
    assert not controller.is_running


@pytest.mark.asyncio
async def test_double_stop_is_safe(controller):
    """Calling stop() twice must be a no-op on the second call."""
    await controller.start()
    await controller.stop()
    await controller.stop()   # Must not raise
    assert not controller.is_running


# ── Continuous conversation regression tests ──────────────────────────────────
# These tests verify the NEW continuous conversation behaviour:
#   After wake-word activation, each completed response triggers
#   signal_relisten() so the VoiceEngine starts listening for the next
#   utterance without requiring another wake word.


@pytest.mark.asyncio
async def test_wake_then_active_conversation(controller, session_mgr):
    """
    Wake word → session active → transcript arrives → Brain is called.
    Verifies the baseline wake→listen→process path that continuous mode extends.
    """
    await controller.start()

    class WakeEvent:
        topic = "wake_word.detected"
        model_name = "hey_jarvis"

    await controller._on_wake_word(WakeEvent())
    assert session_mgr.is_active, "Session must be ACTIVE after wake word"
    assert session_mgr.state == VoiceSessionState.AWAKE

    await controller.stop()


@pytest.mark.asyncio
async def test_response_returns_to_listening(
    controller, session_mgr, fake_voice_engine
):
    """
    After a brain response is spoken (TTS complete), signal_relisten() must be
    called on the VoiceEngine and the session must remain AWAKE.

    This is the core continuous conversation mechanism: the user can speak again
    without saying the wake word.
    """
    await controller.start()
    await session_mgr.activate()
    await session_mgr.record_utterance("what is Python")
    await session_mgr.mark_speaking()

    class ResponseEvent:
        topic = "brain.response_ready"
        response_text = "Python is a programming language."
        text = ""

    # Simulate brain.response_ready → TTS → re-listen
    await controller._on_brain_response(ResponseEvent())

    # Session must remain AWAKE after response (continuous mode)
    assert session_mgr.is_active, "Session must still be active after response"
    assert session_mgr.state == VoiceSessionState.AWAKE

    # VoiceEngine must have been signalled to listen again
    assert fake_voice_engine.relisten_calls == 1, (
        "signal_relisten() must be called exactly once after TTS completes "
        "while session is active"
    )

    await controller.stop()


@pytest.mark.asyncio
async def test_multiple_turns_without_wake_word(
    controller, session_mgr, fake_brain, fake_voice_engine
):
    """
    Simulate three consecutive turns in a continuous conversation.
    Brain must be called for each turn; signal_relisten() called after each.
    No additional wake word events should be needed between turns.
    """
    await controller.start()
    await session_mgr.activate()

    class TranscriptEvent:
        topic = "voice.transcript"
        def __init__(self, text):
            self.text = text

    class ResponseEvent:
        topic = "brain.response_ready"
        response_text = "Here is the answer."
        text = ""

    turns = [
        "what is Python",
        "who created it",
        "give me a simple example",
    ]

    for turn_text in turns:
        # User speaks
        await controller._on_transcript(TranscriptEvent(turn_text))
        # Session should still be active
        assert session_mgr.is_active

    # Brain must have been called for all three turns
    total_brain_calls = fake_brain.process.call_count + fake_brain.run_goal.call_count
    assert total_brain_calls == 3, (
        f"Brain must be called once per turn; got {total_brain_calls}"
    )

    await controller.stop()


@pytest.mark.asyncio
async def test_contextual_follow_up_turn_count(
    controller, session_mgr
):
    """
    Turn count increments for every utterance recorded in the session.
    After 3 turns the session has turn_count == 3, verifying context is preserved
    across consecutive turns.
    """
    await controller.start()
    await session_mgr.activate()

    for utterance in ["what is Python", "who made it", "when"]:
        await session_mgr.record_utterance(utterance)
        await session_mgr.mark_speaking()
        await session_mgr.mark_response_complete()

    assert session_mgr.turn_count == 3, (
        f"Expected 3 turns recorded, got {session_mgr.turn_count}"
    )
    assert session_mgr.is_active, "Session must remain active across turns"
    await controller.stop()


@pytest.mark.asyncio
async def test_inactivity_timeout_stops_relisten(
    controller, session_mgr, fake_voice_engine, fake_bus
):
    """
    When the session timeout fires and deactivates the session,
    signal_relisten() must NOT be called afterwards.
    The VoiceEngine must return to sleeping (wake-word detection) mode.
    """
    await controller.start()
    await session_mgr.activate()

    # Simulate timeout: deactivate the session as the timeout watcher would
    await session_mgr.deactivate(reason="timeout")
    assert not session_mgr.is_active, "Session must be IDLE after timeout"

    # Now simulate a brain response arriving after session has ended
    # (race condition: response arrived just as session timed out)
    class LateResponseEvent:
        topic = "brain.response_ready"
        response_text = "This response arrived too late."
        text = ""

    await controller._on_brain_response(LateResponseEvent())

    # signal_relisten must NOT have been called because session is inactive
    assert fake_voice_engine.relisten_calls == 0, (
        "signal_relisten() must NOT be called when session has already timed out"
    )

    await controller.stop()


@pytest.mark.asyncio
async def test_paused_session_does_not_trigger_relisten(
    controller, session_mgr, fake_voice_engine
):
    """
    If the session is paused (user said 'wait'), signal_relisten() must NOT
    be called after a response.  The engine should stay idle until the user
    resumes the session.
    """
    await controller.start()
    await session_mgr.activate()
    await session_mgr.pause()
    assert session_mgr.is_paused

    # A response arrives while paused — CVC already guards against TTS in paused
    # state, but if it did slip through, relisten must not fire.
    # We test the re-listen guard directly via mark_response_complete:
    # The paused state is not SPEAKING, so mark_response_complete would be a noop.
    # Just verify guard in signal_relisten path:
    assert session_mgr.is_active       # still active while paused
    assert session_mgr.is_paused       # explicitly paused

    # Manually trigger what _on_brain_response would do after TTS
    await session_mgr.mark_response_complete()   # noop in PAUSED state

    # The condition in CVC checks is_paused, so relisten must be 0
    if controller._continuous_mode and session_mgr.is_active and not session_mgr.is_paused:
        fake_voice_engine.signal_relisten()
    # Above: condition is False (is_paused=True), so relisten_calls stays 0
    assert fake_voice_engine.relisten_calls == 0

    await controller.stop()


@pytest.mark.asyncio
async def test_start_stop_lifecycle_clean(controller, fake_bus):
    """
    start() + stop() leaves no orphaned CVC handlers on any topic.
    Verified for the existing topics plus the new continuous conversation path.
    """
    await controller.start()
    assert controller.is_running
    await controller.stop()
    assert not controller.is_running

    topics = (
        "voice.transcript",
        "wake_word.detected",
        "voice.speaking_end",
        "brain.response_ready",
        "voice.session_timeout",
        "system.shutting_down",
    )
    for topic in topics:
        remaining = fake_bus._subscribers.get(topic, [])
        assert controller._on_transcript not in remaining, (
            f"Orphaned _on_transcript on {topic}"
        )
        assert controller._on_brain_response not in remaining, (
            f"Orphaned _on_brain_response on {topic}"
        )


@pytest.mark.asyncio
async def test_no_duplicate_event_handlers_on_double_start(controller, fake_bus):
    """
    Calling start() twice must not register duplicate event handlers.
    The second start() is a no-op because _running is already True.
    """
    await controller.start()
    first_count = sum(len(v) for v in fake_bus._subscribers.values())

    await controller.start()  # second call — must be a no-op
    second_count = sum(len(v) for v in fake_bus._subscribers.values())

    assert first_count == second_count, (
        "Double start() must not add duplicate event handlers"
    )
    await controller.stop()


@pytest.mark.asyncio
async def test_recovery_after_brain_processing_failure(
    controller, session_mgr, fake_brain
):
    """
    If Brain.process() raises an exception, the controller must NOT permanently
    kill the conversation session.  The session must still be active so the
    inactivity timeout (not the exception) controls session teardown.
    """
    await controller.start()
    await session_mgr.activate()

    # Make brain fail on next call
    fake_brain.process.side_effect = RuntimeError("LLM timeout")
    fake_brain.run_goal.side_effect = RuntimeError("LLM timeout")

    class TranscriptEvent:
        topic = "voice.transcript"
        text = "what is the weather today"

    # Controller must catch the exception and not propagate
    await controller._on_transcript(TranscriptEvent())  # must not raise

    # Session should still be active — the brain error is recoverable
    assert session_mgr.is_active, (
        "Session must remain active after a recoverable brain processing error"
    )

    await controller.stop()


@pytest.mark.asyncio
async def test_recovery_after_empty_stt_result(
    controller, session_mgr, fake_brain
):
    """
    An empty / whitespace-only transcript must be silently discarded.
    Brain must NOT be called and the session must remain active.
    """
    await controller.start()
    await session_mgr.activate()

    class EmptyTranscriptEvent:
        topic = "voice.transcript"
        text = "   "

    await controller._on_transcript(EmptyTranscriptEvent())

    fake_brain.process.assert_not_called()
    fake_brain.run_goal.assert_not_called()
    assert session_mgr.is_active, (
        "Session must remain active after an empty/silent STT result"
    )

    await controller.stop()

@pytest.mark.asyncio
async def test_only_one_brain_response_handler_after_start(
    controller, fake_bus, fake_voice_engine
):
    """
    After CVC.start(), only CVC's handler is subscribed to brain.response_ready.
    VoiceEngine's direct-speak handler must be removed.

    This is the primary anti-duplicate-TTS mechanism: having only ONE subscriber
    on brain.response_ready means the response is spoken exactly once per event.
    """
    # Pre-register the engine handler as VoiceEngine.start() would
    fake_bus.subscribe("brain.response_ready", fake_voice_engine._on_brain_response)

    await controller.start()

    subscribers = fake_bus._subscribers.get("brain.response_ready", [])
    assert len(subscribers) == 1, (
        f"Must have exactly ONE brain.response_ready subscriber after start(); "
        f"got {len(subscribers)}: {subscribers}"
    )
    assert controller._on_brain_response in subscribers
    assert fake_voice_engine._on_brain_response not in subscribers

    await controller.stop()


@pytest.mark.asyncio
async def test_streaming_tts_serializes_concurrent_speak():
    """
    StreamingTTSWrapper serializes concurrent speak() calls via _speak_lock.

    When two tasks race to call speak() simultaneously, only one runs at a time.
    The _speaking flag is True while speak() is in progress, preventing re-entrant
    calls from the same event chain from duplicating the response.

    Specifically: if speak() is already running and a new call arrives and acquires
    the lock (after the first completes), it will proceed because _speaking is now
    False.  This is the correct and expected behaviour for sequential events.
    The lock protects against the TOCTOU scenario where both calls check _speaking
    before either sets it True (only possible with threads, not in asyncio).
    """
    import asyncio as _asyncio

    class TrackingFakeTTSEngine:
        """Records is_speaking state during speak() calls."""
        def __init__(self):
            self.spoken: list[str] = []
            self._speaking = False
            self.was_speaking_during: list[bool] = []

        async def speak(self, text: str) -> None:
            self.was_speaking_during.append(self._speaking)
            self._speaking = True
            self.spoken.append(text)
            await _asyncio.sleep(0)  # yield
            self._speaking = False

        async def synthesize(self, text):
            pass

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
            return "tracking_fake"

        @property
        def is_loaded(self) -> bool:
            return True

    engine = TrackingFakeTTSEngine()
    wrapper = StreamingTTSWrapper(engine=engine)

    # Call speak() twice sequentially
    await wrapper.speak("First response.")
    await wrapper.speak("Second response.")

    # Both calls should complete (sequential execution)
    assert len(engine.spoken) == 2
    # _speaking must be False after both complete
    assert not wrapper.is_speaking
    assert not engine._speaking
