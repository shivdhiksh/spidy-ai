"""
Voice Pipeline Diagnostics — Regression Tests
================================================
Regression suite for:
    A. Provider timing logged at INFO level
    B. NVIDIA provider name in logs
    C. Ollama fallback in logs
    D. Wake word ignored when session already active
    E. Second wake does not create duplicate VoiceSession
    F. "Hey Jarvis, Open Edge" -> "Open Edge"
    G. "Hey Jarvis, What is Python?" -> "What is Python?"
    H. "Hey Jarvis." -> empty
    I. Clearly repetitive/garbled transcript is rejected
    J. Continuous conversation still works
    K. TTS -> listening transition occurs
    L. No duplicate event handlers after repeated start
    M. Wake-word detection active while sleeping
    N. "Who is Jarvis?" NOT stripped
    O. Legitimate short commands not rejected
    P. Whisper metadata propagated correctly
    Q. NVIDIA failure -> Ollama fallback functional

No real VoiceEngine, Brain, LLM, or audio hardware required.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger as _loguru_logger

# -- Imports under test -------------------------------------------------------

from spidy.llm.router import LLMRouter
from spidy.llm.client import LLMMessage, LLMResponse, BaseLLMClient
from spidy.voice.wake_word_stripper import WakeWordStripper
from spidy.voice.transcript_guard import TranscriptQualityGuard, SegmentQualitySignals
from spidy.voice.continuous import ContinuousVoiceController
from spidy.voice.session import VoiceSessionManager, VoiceSessionState
from spidy.voice.streaming_tts import StreamingTTSWrapper
from spidy.voice.interruption import InterruptionHandler
from spidy.perception.voice.stt.base import TranscriptResult


# =============================================================================
# Shared fakes
# =============================================================================


class FakeTTSEngine:
    def __init__(self):
        self.spoken: list[str] = []
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
        for cb in list(self._subscribers.get(topic, [])):
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
        pass


class _FakeCaptureEngine:
    def __init__(self):
        self.mode = None

    def set_mode(self, mode) -> None:
        self.mode = mode


class _FakeWakeModel:
    def reset_buffer(self) -> None:
        pass


class FakeVoiceEngine:
    def __init__(self):
        self._tts = FakeTTSEngine()
        self.conversation_mode = False
        self.paused = False
        self.relisten_calls: int = 0
        self._capture_engine = _FakeCaptureEngine()
        self._wake_model = _FakeWakeModel()

    def enable_conversation_mode(self, timeout_seconds=60.0):
        self.conversation_mode = True

    def signal_relisten(self) -> None:
        self.relisten_calls += 1

    async def _on_brain_response(self, event) -> None:
        pass


class FakeBrain:
    def __init__(self):
        self.process = AsyncMock(return_value="Here is the answer.")
        self.run_goal = AsyncMock(return_value="Goal completed.")
        self.cancel_goal = AsyncMock(return_value=True)


# -- Fake LLM clients for router tests ----------------------------------------


class FakeLLMClient(BaseLLMClient):
    """Configurable success/failure LLM client for testing."""

    def __init__(
        self,
        name: str = "fake",
        model: str = "fake-model",
        success: bool = True,
        response_text: str = "ok",
    ):
        self.provider_name = name
        self._model = model
        self._success = success
        self._response_text = response_text
        self.call_count = 0

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature=None,
        max_tokens=None,
    ) -> LLMResponse:
        self.call_count += 1
        if self._success:
            return LLMResponse(text=self._response_text, model=self._model, success=True)
        return LLMResponse.failure("provider unavailable", model=self._model)

    async def stream(self, messages, temperature=None, max_tokens=None):
        if False:  # pragma: no cover
            yield ""

    async def close(self) -> None:
        pass


# =============================================================================
# Fixtures
# =============================================================================


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
        measure_latency=True,
    )


@pytest.fixture
def loguru_caplog(caplog):
    """
    Bridge loguru output into pytest's caplog fixture.

    Loguru writes to its own handler chain (not Python's stdlib logging).
    This fixture adds a temporary loguru sink that forwards every record to
    the stdlib logger with the matching level, so caplog can capture it.
    """
    handler_id = _loguru_logger.add(
        lambda msg: logging.getLogger(msg.record["extra"].get("module", "spidy")).log(
            msg.record["level"].no, msg.record["message"]
        ),
        format="{message}",
        level="DEBUG",
    )
    caplog.set_level(logging.DEBUG)
    yield caplog
    _loguru_logger.remove(handler_id)


# =============================================================================
# A. Provider timing logged at INFO level
# =============================================================================


@pytest.mark.asyncio
async def test_a_llm_router_logs_provider_timing_at_info(loguru_caplog):
    """LLMRouter must log 'LLM request' and 'LLM response' at INFO level."""
    nvidia = FakeLLMClient(name="nvidia", model="nvidia/nemotron-3-ultra-550b-a55b", success=True)
    router = LLMRouter(providers=[nvidia], bus=None, auto_fallback=False)

    msgs = [LLMMessage(role="user", content="Hello")]
    with loguru_caplog.at_level(logging.INFO):
        resp = await router.complete(msgs)

    assert resp.success
    combined = loguru_caplog.text
    assert "LLM request" in combined, f"Expected 'LLM request' in logs, got: {combined}"
    assert "LLM response" in combined, f"Expected 'LLM response' in logs, got: {combined}"
    assert "duration" in combined, f"Response log must include 'duration', got: {combined}"


# =============================================================================
# B. NVIDIA provider name in logs
# =============================================================================


@pytest.mark.asyncio
async def test_b_nvidia_provider_name_in_logs(loguru_caplog):
    """LLMRouter must log 'provider=nvidia' for an NVIDIA client."""
    nvidia = FakeLLMClient(name="nvidia", model="nvidia/nemotron-3-ultra-550b-a55b", success=True)
    router = LLMRouter(providers=[nvidia], bus=None)

    msgs = [LLMMessage(role="user", content="ping")]
    with loguru_caplog.at_level(logging.INFO):
        await router.complete(msgs)

    combined = loguru_caplog.text
    assert "provider=nvidia" in combined, (
        f"Expected 'provider=nvidia' in logs, got: {combined}"
    )


# =============================================================================
# C. Ollama fallback in logs
# =============================================================================


@pytest.mark.asyncio
async def test_c_ollama_fallback_logged_on_nvidia_failure(loguru_caplog):
    """When NVIDIA fails, router must log fallback and succeed via Ollama."""
    nvidia = FakeLLMClient(name="nvidia", success=False)
    ollama = FakeLLMClient(name="ollama", model="llama3.2:3b", success=True)
    router = LLMRouter(providers=[nvidia, ollama], bus=None, auto_fallback=True)

    msgs = [LLMMessage(role="user", content="ping")]
    with loguru_caplog.at_level(logging.INFO):
        resp = await router.complete(msgs)

    assert resp.success, "Fallback to ollama must succeed"
    combined = loguru_caplog.text
    assert "nvidia" in combined and "ollama" in combined, (
        f"Both provider names must appear in fallback logs. Got: {combined}"
    )


def test_c_ollama_client_has_provider_name():
    """OllamaClient must have provider_name = 'ollama'."""
    from spidy.llm.backends.ollama import OllamaClient
    client = OllamaClient()
    assert getattr(client, "provider_name", None) == "ollama"


# =============================================================================
# D. Wake word ignored when session already active
# =============================================================================


@pytest.mark.asyncio
async def test_d_wake_word_ignored_during_active_session(
    controller, session_mgr, loguru_caplog,
):
    """Second wake event while session is active must be ignored + logged."""
    await controller.start()

    class WakeEvent:
        topic = "wake_word.detected"
        model_name = "hey_jarvis"

    await controller._on_wake_word(WakeEvent())
    first_sid = session_mgr.session_id
    assert session_mgr.is_active

    with loguru_caplog.at_level(logging.INFO):
        await controller._on_wake_word(WakeEvent())

    # Session ID unchanged
    assert session_mgr.session_id == first_sid, "No new session must be created"
    combined = loguru_caplog.text
    assert "ignoring" in combined.lower() or "already active" in combined.lower(), (
        f"Must log that the wake was ignored. Got: {combined}"
    )

    await controller.stop()


# =============================================================================
# E. No duplicate VoiceSession on second wake
# =============================================================================


@pytest.mark.asyncio
async def test_e_no_duplicate_session_on_second_wake(controller, session_mgr):
    """Two consecutive wakes must yield exactly one session with the same ID."""
    await controller.start()

    class WakeEvent:
        topic = "wake_word.detected"
        model_name = "hey_jarvis"

    await controller._on_wake_word(WakeEvent())
    sid1 = session_mgr.session_id
    assert session_mgr.is_active

    await controller._on_wake_word(WakeEvent())
    sid2 = session_mgr.session_id

    assert sid1 == sid2, f"Session ID must not change; {sid1} != {sid2}"
    assert session_mgr.turn_count == 0, "No turns must be added by the second wake"

    await controller.stop()


# =============================================================================
# F. "Hey Jarvis, Open Edge" -> "Open Edge"
# =============================================================================


def test_f_strip_hey_jarvis_comma_open_edge():
    s = WakeWordStripper()
    assert s.strip_wake_prefix("Hey Jarvis, Open Edge.") == "Open Edge."


def test_f_strip_hey_jarvis_open_edge_lowercase():
    s = WakeWordStripper()
    assert s.strip_wake_prefix("hey jarvis open edge") == "open edge"


# =============================================================================
# G. "Hey Jarvis, What is Python?" -> "What is Python?"
# =============================================================================


def test_g_strip_hey_jarvis_what_is_python_comma():
    s = WakeWordStripper()
    assert s.strip_wake_prefix("Hey Jarvis, What is Python?") == "What is Python?"


def test_g_strip_hey_jarvis_dot_what_is_python():
    s = WakeWordStripper()
    assert s.strip_wake_prefix("Hey Jarvis. What is Python?") == "What is Python?"


# =============================================================================
# H. "Hey Jarvis." -> empty
# =============================================================================


def test_h_wake_only_returns_empty():
    s = WakeWordStripper()
    assert s.strip_wake_prefix("Hey Jarvis.") == ""


def test_h_wake_only_trailing_space_empty():
    s = WakeWordStripper()
    assert s.strip_wake_prefix("hey jarvis  ") == ""


# =============================================================================
# I. Repetitive/garbled transcript rejected
# =============================================================================


def test_i_repetitive_rejected():
    g = TranscriptQualityGuard()
    assert g.is_suspicious("word word word word word word word") is True


def test_i_beep_pattern_rejected():
    g = TranscriptQualityGuard()
    assert g.is_suspicious("Beep. Beep. Beep. Beep.") is True


def test_i_high_no_speech_prob_rejected():
    g = TranscriptQualityGuard()
    signals = [SegmentQualitySignals(no_speech_prob=0.95, avg_logprob=-0.5)]
    assert g.is_suspicious("some text", segments=signals) is True


def test_i_very_low_logprob_rejected():
    g = TranscriptQualityGuard()
    signals = [SegmentQualitySignals(no_speech_prob=0.1, avg_logprob=-2.0)]
    assert g.is_suspicious("some text", segments=signals) is True


# =============================================================================
# J. Continuous conversation still works
# =============================================================================


@pytest.mark.asyncio
async def test_j_continuous_conversation_basic(controller, session_mgr, fake_brain):
    """Wake -> transcript -> Brain called -> session stays active."""
    await controller.start()

    class WakeEvent:
        topic = "wake_word.detected"
        model_name = "hey_jarvis"

    class TranscriptEvent:
        topic = "voice.transcript"
        text = "what time is it"
        segment_signals = None

    await controller._on_wake_word(WakeEvent())
    assert session_mgr.is_active

    await controller._on_transcript(TranscriptEvent())
    called = fake_brain.process.called or fake_brain.run_goal.called
    assert called, "Brain must be called for a valid transcript"
    assert session_mgr.is_active

    await controller.stop()


@pytest.mark.asyncio
async def test_j_three_follow_up_utterances_all_reach_brain(
    controller, session_mgr, fake_brain,
):
    """Multiple follow-up utterances without wake word all reach Brain once each."""
    await controller.start()
    await session_mgr.activate()

    class T:
        topic = "voice.transcript"
        def __init__(self, text):
            self.text = text
            self.segment_signals = None

    for u in ["open calculator", "what is 2 plus 2", "set a timer for 5 minutes"]:
        await controller._on_transcript(T(u))

    total = fake_brain.process.call_count + fake_brain.run_goal.call_count
    assert total == 3, f"Expected 3 Brain calls, got {total}"
    assert session_mgr.is_active

    await controller.stop()


# =============================================================================
# K. TTS -> listening transition
# =============================================================================


@pytest.mark.asyncio
async def test_k_tts_complete_triggers_relisten(
    controller, session_mgr, fake_voice_engine,
):
    """After TTS completes, signal_relisten() is called and session is AWAKE."""
    await controller.start()
    await session_mgr.activate()
    await session_mgr.record_utterance("open chrome")
    await session_mgr.mark_speaking()

    class ResponseEvent:
        topic = "brain.response_ready"
        response_text = "Opening Chrome."
        text = ""

    await controller._on_brain_response(ResponseEvent())

    assert session_mgr.state == VoiceSessionState.AWAKE
    assert fake_voice_engine.relisten_calls == 1, (
        f"signal_relisten must be called once, got {fake_voice_engine.relisten_calls}"
    )

    await controller.stop()


@pytest.mark.asyncio
async def test_k_capture_idle_during_tts(controller, session_mgr, fake_voice_engine):
    """Capture engine must be IDLE during TTS and DETECTING after."""
    from spidy.perception.voice.audio_capture import CaptureMode

    await controller.start()
    await session_mgr.activate()
    await session_mgr.record_utterance("hello")
    await session_mgr.mark_speaking()

    modes_during: list = []
    original_speak = controller._streaming_tts.speak

    async def recording_speak(text: str):
        modes_during.append(fake_voice_engine._capture_engine.mode)
        await original_speak(text)

    controller._streaming_tts.speak = recording_speak

    class ResponseEvent:
        topic = "brain.response_ready"
        response_text = "Hello."
        text = ""

    await controller._on_brain_response(ResponseEvent())

    assert CaptureMode.IDLE in modes_during, "IDLE must be set before TTS"
    assert fake_voice_engine._capture_engine.mode == CaptureMode.DETECTING, (
        "DETECTING must be restored after TTS"
    )

    await controller.stop()


# =============================================================================
# L. No duplicate event handlers after repeated start
# =============================================================================


@pytest.mark.asyncio
async def test_l_no_duplicate_handlers_on_double_start(controller, fake_bus):
    """Double start() must not add duplicate event handlers."""
    await controller.start()
    count1 = sum(len(v) for v in fake_bus._subscribers.values())

    await controller.start()  # no-op
    count2 = sum(len(v) for v in fake_bus._subscribers.values())

    assert count1 == count2, f"Handler count changed on double start: {count1} -> {count2}"

    await controller.stop()


# =============================================================================
# M. Wake-word detection active while sleeping
# =============================================================================


@pytest.mark.asyncio
async def test_m_wake_from_idle_activates_session(controller, session_mgr):
    """From IDLE, a wake event must move session to AWAKE."""
    await controller.start()
    assert session_mgr.state == VoiceSessionState.IDLE

    class WakeEvent:
        topic = "wake_word.detected"
        model_name = "hey_jarvis"

    await controller._on_wake_word(WakeEvent())
    assert session_mgr.state == VoiceSessionState.AWAKE

    await controller.stop()


@pytest.mark.asyncio
async def test_m_after_session_end_next_wake_creates_new_session(
    controller, session_mgr,
):
    """After timeout, a fresh wake must create a NEW session with a different ID."""
    await controller.start()

    class WakeEvent:
        topic = "wake_word.detected"
        model_name = "hey_jarvis"

    await controller._on_wake_word(WakeEvent())
    first_sid = session_mgr.session_id

    await session_mgr.deactivate(reason="timeout")
    assert not session_mgr.is_active

    await controller._on_wake_word(WakeEvent())
    assert session_mgr.is_active
    assert session_mgr.session_id != first_sid, "New session must have a different ID"

    await controller.stop()


# =============================================================================
# N. "Who is Jarvis?" NOT stripped
# =============================================================================


def test_n_who_is_jarvis_unchanged():
    s = WakeWordStripper()
    assert s.strip_wake_prefix("Who is Jarvis?") == "Who is Jarvis?"


def test_n_tell_me_about_jarvis_unchanged():
    s = WakeWordStripper()
    assert s.strip_wake_prefix("Tell me about Jarvis") == "Tell me about Jarvis"


def test_n_i_asked_jarvis_unchanged():
    s = WakeWordStripper()
    result = s.strip_wake_prefix("I asked Jarvis about the weather")
    assert result == "I asked Jarvis about the weather"


# =============================================================================
# O. Legitimate short commands not rejected
# =============================================================================


@pytest.mark.parametrize("cmd", [
    "Open Edge",
    "Open Calculator",
    "Stop",
    "Yes",
    "No",
    "Thanks",
    "what time is it",
    "open VS Code",
    "play music",
    "volume up",
    "set a timer",
])
def test_o_short_commands_pass_quality_guard(cmd):
    """Legitimate short commands must never be rejected."""
    g = TranscriptQualityGuard()
    assert g.is_suspicious(cmd) is False, f"'{cmd}' must not be rejected"


def test_o_no_metadata_fails_open():
    """Without segment signals, guard must fail open for normal text."""
    g = TranscriptQualityGuard()
    assert g.is_suspicious("I would like to open Chrome please", segments=None) is False


# =============================================================================
# P. Whisper metadata propagated correctly
# =============================================================================


def test_p_transcript_result_accepts_segment_signals():
    """TranscriptResult must store segment_signals without error."""
    signals = [
        SegmentQualitySignals(no_speech_prob=0.1, avg_logprob=-0.3, compression_ratio=1.1)
    ]
    result = TranscriptResult(text="open chrome", confidence=1.0, segment_signals=signals)
    assert result.segment_signals is not None
    assert len(result.segment_signals) == 1
    assert result.segment_signals[0].no_speech_prob == pytest.approx(0.1)
    assert result.segment_signals[0].avg_logprob == pytest.approx(-0.3)


def test_p_transcript_result_backward_compatible():
    """Existing code that omits segment_signals must still work (defaults to None)."""
    result = TranscriptResult(text="hello")
    assert result.segment_signals is None


def test_p_quality_guard_passes_good_signals():
    g = TranscriptQualityGuard()
    good = [SegmentQualitySignals(no_speech_prob=0.05, avg_logprob=-0.2)]
    assert g.is_suspicious("open edge", segments=good) is False


def test_p_quality_guard_rejects_bad_signals():
    g = TranscriptQualityGuard()
    bad = [SegmentQualitySignals(no_speech_prob=0.92, avg_logprob=-0.2)]
    assert g.is_suspicious("some text", segments=bad) is True


def test_p_whisper_transcribe_sync_captures_segment_metadata():
    """_transcribe_sync must populate segment_signals from faster-whisper data."""
    from spidy.perception.voice.stt.whisper import FasterWhisperRecognizer
    import numpy as np

    class FakeSegment:
        text = "open edge"
        no_speech_prob = 0.05
        avg_logprob = -0.3
        compression_ratio = 1.1

    class FakeInfo:
        language = "en"

    recognizer = FasterWhisperRecognizer(model_size="base.en")
    fake_model = MagicMock()
    fake_model.transcribe.return_value = ([FakeSegment()], FakeInfo())
    recognizer._model = fake_model

    audio = np.zeros(16000, dtype=np.float32)
    result = recognizer._transcribe_sync(audio)

    assert result.text == "open edge"
    assert result.segment_signals is not None, "segment_signals must not be None"
    assert len(result.segment_signals) == 1
    assert result.segment_signals[0].no_speech_prob == pytest.approx(0.05)
    assert result.segment_signals[0].avg_logprob == pytest.approx(-0.3)
    assert result.segment_signals[0].compression_ratio == pytest.approx(1.1)


# =============================================================================
# Q. NVIDIA failure -> Ollama fallback functional
# =============================================================================


@pytest.mark.asyncio
async def test_q_nvidia_failure_falls_back_to_ollama():
    """When NVIDIA fails, LLMRouter falls back to Ollama and returns success."""
    nvidia = FakeLLMClient(name="nvidia", success=False)
    ollama = FakeLLMClient(name="ollama", model="llama3.2:3b", success=True,
                           response_text="Hello from Ollama")
    router = LLMRouter(providers=[nvidia, ollama], bus=None, auto_fallback=True)

    msgs = [LLMMessage(role="user", content="hello")]
    resp = await router.complete(msgs)

    assert resp.success, "Fallback response must succeed"
    assert resp.text == "Hello from Ollama"
    assert nvidia.call_count == 1, "NVIDIA must be attempted once"
    assert ollama.call_count == 1, "Ollama must be used as fallback"


@pytest.mark.asyncio
async def test_q_both_providers_fail_returns_failure():
    """When all providers fail, router returns a failure response."""
    nvidia = FakeLLMClient(name="nvidia", success=False)
    ollama = FakeLLMClient(name="ollama", success=False)
    router = LLMRouter(providers=[nvidia, ollama], bus=None, auto_fallback=True)

    msgs = [LLMMessage(role="user", content="ping")]
    resp = await router.complete(msgs)

    assert resp.success is False
    assert nvidia.call_count == 1
    assert ollama.call_count == 1


@pytest.mark.asyncio
async def test_q_fallback_logs_both_provider_names(loguru_caplog):
    """Fallback must log both provider names."""
    nvidia = FakeLLMClient(name="nvidia", success=False)
    ollama = FakeLLMClient(name="ollama", success=True)
    router = LLMRouter(providers=[nvidia, ollama], bus=None, auto_fallback=True)

    msgs = [LLMMessage(role="user", content="ping")]
    with loguru_caplog.at_level(logging.INFO):
        await router.complete(msgs)

    combined = loguru_caplog.text
    assert "nvidia" in combined and "ollama" in combined, (
        f"Both provider names must appear in logs. Got: {combined}"
    )


# =============================================================================
# Additional integration: stripper + guard wired in controller
# =============================================================================


@pytest.mark.asyncio
async def test_integration_wake_strip_wired_in_controller(
    controller, session_mgr, fake_brain,
):
    """
    'Hey Jarvis, open Edge.' -> Brain receives 'open Edge.' (not full string).
    Proves WakeWordStripper is wired into _on_transcript().
    """
    await controller.start()
    await session_mgr.activate()

    class TranscriptEvent:
        topic = "voice.transcript"
        text = "Hey Jarvis, open Edge."
        segment_signals = None

    await controller._on_transcript(TranscriptEvent())

    total = fake_brain.process.call_count + fake_brain.run_goal.call_count
    assert total == 1, f"Brain must be called once; got {total}"

    call_args = fake_brain.process.call_args or fake_brain.run_goal.call_args
    received = call_args[0][0] if call_args else ""
    assert "Hey Jarvis" not in received, (
        f"Wake prefix must be stripped; got: '{received}'"
    )

    await controller.stop()


@pytest.mark.asyncio
async def test_integration_wake_only_not_forwarded_to_brain(
    controller, session_mgr, fake_brain,
):
    """'Hey Jarvis.' (no command) must NOT trigger a Brain call."""
    await controller.start()
    await session_mgr.activate()

    class TranscriptEvent:
        topic = "voice.transcript"
        text = "Hey Jarvis."
        segment_signals = None

    await controller._on_transcript(TranscriptEvent())

    fake_brain.process.assert_not_called()
    fake_brain.run_goal.assert_not_called()

    await controller.stop()


@pytest.mark.asyncio
async def test_integration_garbled_not_forwarded_to_brain(
    controller, session_mgr, fake_brain,
):
    """Garbled transcript must be rejected by quality guard, Brain not called."""
    await controller.start()
    await session_mgr.activate()

    class GarbledEvent:
        topic = "voice.transcript"
        text = "word word word word word word word word word"
        segment_signals = None

    await controller._on_transcript(GarbledEvent())

    fake_brain.process.assert_not_called()
    fake_brain.run_goal.assert_not_called()

    await controller.stop()
