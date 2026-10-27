"""
test_wake_listener_survives_session_timeout.py
=============================================
Unit tests verifying that the VoiceEngine wake-word listener, OpenWakeWord
models, audio capture stream, and async processing task survive session timeouts
and cleanly activate new sessions across multiple wake words.
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from spidy.core.event_bus import EventBus
from spidy.perception.voice.audio_capture import AudioCaptureEngine, CaptureMode
from spidy.perception.voice.engine import (
    VoiceEngine,
    VoiceState,
    VoiceTranscriptEvent,
    WakeWordDetectedEvent,
)
from spidy.perception.voice.stt.base import SpeechRecognizer, TranscriptResult
from spidy.perception.voice.tts.base import TTSEngine
from spidy.perception.voice.wake_word.base import WakeWordModel
from spidy.perception.voice.wake_word.openwakeword import NullWakeWordModel, OpenWakeWordModel
from spidy.ui.events import UIStateChangeEvent
from spidy.voice.continuous import ContinuousVoiceController
from spidy.voice.session import (
    VoiceSessionEndedEvent,
    VoiceSessionManager,
    VoiceSessionStartedEvent,
    VoiceSessionState,
    VoiceSessionTimeoutEvent,
)
from spidy.config.manager import WakeAckConfig
from spidy.voice.wake_ack import WakeAcknowledger


class _FakeSTT(SpeechRecognizer):
    def __init__(self, transcript: str = "open notepad"):
        self._transcript = transcript

    async def transcribe(self, audio: np.ndarray) -> TranscriptResult:
        return TranscriptResult(
            text=self._transcript,
            confidence=0.95,
            language="en",
            duration_seconds=1.0,
        )

    def load(self, model_name: str | None = None) -> None:
        pass

    def unload(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "fake_stt"

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def is_loaded(self) -> bool:
        return True


class _FakeTTS(TTSEngine):
    def __init__(self):
        self._speaking = False
        self.spoken_texts: list[str] = []

    async def synthesize(self, text: str):
        from spidy.perception.voice.tts.base import AudioBuffer
        return AudioBuffer(samples=np.zeros(1600, dtype=np.float32), sample_rate=22050, duration_seconds=0.1)

    async def speak(self, text: str) -> None:
        self._speaking = True
        self.spoken_texts.append(text)
        await asyncio.sleep(0.01)
        self._speaking = False

    def stop(self) -> None:
        self._speaking = False

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    @property
    def voice_name(self) -> str:
        return "fake_voice"

    @property
    def is_speaking(self) -> bool:
        return self._speaking


class _MockOWWModel(WakeWordModel):
    def __init__(self, model_names: list[str] | None = None):
        self._active_models = model_names or ["hey_spidy", "wake_up_spidy", "hey_jarvis"]
        self._model_name = self._active_models[0]
        self._loaded = True
        self._is_detected = False
        self._last_detected_model = ""
        self._last_detected_phrase = ""
        self.reset_count = 0

    def load(self, model_path: str | None = None) -> None:
        self._loaded = True

    def process_chunk(self, chunk: np.ndarray) -> float:
        if self._is_detected:
            return 1.0
        return 0.0

    def trigger(self, model: str, phrase: str) -> None:
        self._is_detected = True
        self._last_detected_model = model
        self._last_detected_phrase = phrase

    def reset_buffer(self) -> None:
        self.reset_count += 1
        self._is_detected = False

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_detected(self) -> bool:
        return self._is_detected

    @property
    def active_model_names(self) -> list[str]:
        return list(self._active_models)

    @property
    def model_name(self) -> str:
        return self._last_detected_model or self._model_name

    @property
    def wake_phrase(self) -> str:
        return self._last_detected_phrase or self._model_name

    @property
    def last_detected_model(self) -> str:
        return self._last_detected_model

    @property
    def last_detected_phrase(self) -> str:
        return self._last_detected_phrase

    @property
    def chunk_size(self) -> int:
        return 1280


@pytest.fixture
async def voice_stack():
    bus = EventBus()
    bus.set_loop(asyncio.get_running_loop())
    wake_model = _MockOWWModel(["hey_spidy", "wake_up_spidy", "hey_jarvis"])
    stt = _FakeSTT()
    tts = _FakeTTS()

    with patch("sounddevice.InputStream"), patch("sounddevice.query_devices"):
        engine = VoiceEngine(
            bus=bus,
            wake_word_model=wake_model,
            recognizer=stt,
            tts=tts,
            sample_rate=16000,
        )

    session_mgr = VoiceSessionManager(
        bus=bus,
        timeout_seconds=0.1,  # Fast timeout for tests
        continuous_mode=True,
    )

    streaming_tts = MagicMock()
    streaming_tts.speak = AsyncMock()
    streaming_tts.is_speaking = False

    brain = MagicMock()
    brain.process = AsyncMock()
    interruption_handler = MagicMock()

    controller = ContinuousVoiceController(
        voice_engine=engine,
        brain=brain,
        bus=bus,
        session_manager=session_mgr,
        streaming_tts=streaming_tts,
        interruption_handler=interruption_handler,
        continuous_mode=True,
    )

    return {
        "bus": bus,
        "engine": engine,
        "wake_model": wake_model,
        "session_mgr": session_mgr,
        "controller": controller,
        "brain": brain,
        "stt": stt,
        "tts": tts,
        "streaming_tts": streaming_tts,
    }


# ── Required Tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wake_listener_remains_active_after_session_timeout(voice_stack):
    """1. Test that the wake listener is active and in SLEEPING state after session timeout."""
    engine = voice_stack["engine"]
    session_mgr = voice_stack["session_mgr"]
    controller = voice_stack["controller"]

    await engine.start()
    await controller.start()

    # Initial state
    assert engine.state == VoiceState.SLEEPING
    assert engine._capture_engine.mode == CaptureMode.DETECTING

    # Activate session
    sid = await session_mgr.activate(wake_word="hey_spidy")
    assert session_mgr.state == VoiceSessionState.AWAKE

    # Wait for fast session timeout (0.1s)
    await asyncio.sleep(0.25)

    # Session timed out
    assert session_mgr.state == VoiceSessionState.IDLE

    # Engine must still be SLEEPING and in DETECTING mode (wake listener active)
    assert engine.state == VoiceState.SLEEPING
    assert engine._capture_engine.mode == CaptureMode.DETECTING

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_processing_task_survives_session_timeout(voice_stack):
    """2. Test that VoiceEngine's async processing task remains alive across timeouts."""
    engine = voice_stack["engine"]
    session_mgr = voice_stack["session_mgr"]
    controller = voice_stack["controller"]

    await engine.start()
    await controller.start()

    task = engine._processing_task
    assert task is not None
    assert not task.done()

    # Activate session & let it timeout
    await session_mgr.activate(wake_word="hey_spidy")
    await asyncio.sleep(0.25)

    assert session_mgr.state == VoiceSessionState.IDLE
    # Processing task must be the exact same alive task
    assert engine._processing_task is task
    assert not engine._processing_task.done()

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_capture_stream_remains_active_after_timeout(voice_stack):
    """3. Test that AudioCaptureEngine thread remains alive and in DETECTING mode."""
    engine = voice_stack["engine"]
    session_mgr = voice_stack["session_mgr"]
    controller = voice_stack["controller"]

    await engine.start()
    await controller.start()

    assert engine._capture_engine.is_running
    assert engine._capture_engine.mode == CaptureMode.DETECTING

    # Session lifecycle
    await session_mgr.activate(wake_word="hey_spidy")
    await asyncio.sleep(0.25)

    assert engine._capture_engine.is_running
    assert engine._capture_engine.mode == CaptureMode.DETECTING

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_openwakeword_models_remain_loaded_after_timeout(voice_stack):
    """4. Test that all 3 registered wake word models remain loaded after session timeout."""
    engine = voice_stack["engine"]
    wake_model = voice_stack["wake_model"]
    session_mgr = voice_stack["session_mgr"]
    controller = voice_stack["controller"]

    await engine.start()
    await controller.start()

    assert wake_model.active_model_names == ["hey_spidy", "wake_up_spidy", "hey_jarvis"]

    # Session timeout
    await session_mgr.activate(wake_word="hey_spidy")
    await asyncio.sleep(0.25)

    # Models must remain loaded and unchanged
    assert wake_model._loaded is True
    assert wake_model.active_model_names == ["hey_spidy", "wake_up_spidy", "hey_jarvis"]

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_new_session_can_activate_after_timeout(voice_stack):
    """5. Test that a brand new session with a new ID can be created after timeout."""
    session_mgr = voice_stack["session_mgr"]
    controller = voice_stack["controller"]
    engine = voice_stack["engine"]

    await engine.start()
    await controller.start()

    # Session 1
    sid1 = await session_mgr.activate(wake_word="hey_spidy")
    assert sid1 != ""
    assert session_mgr.state == VoiceSessionState.AWAKE

    await asyncio.sleep(0.25)
    assert session_mgr.state == VoiceSessionState.IDLE

    # Session 2
    sid2 = await session_mgr.activate(wake_word="hey_spidy")
    assert sid2 != ""
    assert sid2 != sid1  # Distinct session ID
    assert session_mgr.state == VoiceSessionState.AWAKE

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_second_wake_word_detected_after_timeout(voice_stack):
    """6. Test that a second wake word event triggers full session activation after timeout."""
    bus = voice_stack["bus"]
    engine = voice_stack["engine"]
    session_mgr = voice_stack["session_mgr"]
    controller = voice_stack["controller"]
    wake_model = voice_stack["wake_model"]

    await engine.start()
    await controller.start()

    # First wake
    wake_model.trigger("hey_spidy", "Hey Spidy")
    chunk = np.zeros(1280, dtype=np.float32)
    engine._process_wake_word(chunk)
    await asyncio.sleep(0.05)

    assert session_mgr.state == VoiceSessionState.AWAKE
    sid1 = session_mgr.session_id

    # Wait for session timeout
    await asyncio.sleep(0.25)
    assert session_mgr.state == VoiceSessionState.IDLE

    # Second wake
    wake_model.trigger("hey_spidy", "Hey Spidy")
    engine._process_wake_word(chunk)
    await asyncio.sleep(0.05)

    assert session_mgr.state == VoiceSessionState.AWAKE
    sid2 = session_mgr.session_id
    assert sid2 != sid1

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_alternating_wake_words_after_timeout(voice_stack):
    """7. Test alternating between different configured wake models across timeouts."""
    engine = voice_stack["engine"]
    session_mgr = voice_stack["session_mgr"]
    controller = voice_stack["controller"]
    wake_model = voice_stack["wake_model"]

    await engine.start()
    await controller.start()

    chunk = np.zeros(1280, dtype=np.float32)

    # Wake with Hey Spidy
    wake_model.trigger("hey_spidy", "Hey Spidy")
    engine._process_wake_word(chunk)
    await asyncio.sleep(0.05)
    assert session_mgr.state == VoiceSessionState.AWAKE
    assert session_mgr._last_wake_word == "Hey Spidy"

    # Timeout
    await asyncio.sleep(0.25)
    assert session_mgr.state == VoiceSessionState.IDLE

    # Wake with Wake up Spidy
    wake_model.trigger("wake_up_spidy", "Wake up Spidy")
    engine._process_wake_word(chunk)
    await asyncio.sleep(0.05)
    assert session_mgr.state == VoiceSessionState.AWAKE
    assert session_mgr._last_wake_word == "Wake up Spidy"

    # Timeout
    await asyncio.sleep(0.25)
    assert session_mgr.state == VoiceSessionState.IDLE

    # Wake with Hey Jarvis
    wake_model.trigger("hey_jarvis", "Hey Jarvis")
    engine._process_wake_word(chunk)
    await asyncio.sleep(0.05)
    assert session_mgr.state == VoiceSessionState.AWAKE
    assert session_mgr._last_wake_word == "Hey Jarvis"

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_timeout_does_not_create_duplicate_listener(voice_stack):
    """8. Test that session timeouts do not spawn duplicate listener tasks."""
    engine = voice_stack["engine"]
    session_mgr = voice_stack["session_mgr"]
    controller = voice_stack["controller"]

    await engine.start()
    await controller.start()

    initial_task = engine._processing_task

    # Cycle through 3 timeouts
    for _ in range(3):
        await session_mgr.activate(wake_word="hey_spidy")
        await asyncio.sleep(0.20)
        assert session_mgr.state == VoiceSessionState.IDLE

    # Still exactly one active processing task
    assert engine._processing_task is initial_task
    assert not engine._processing_task.done()

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_timeout_clears_stale_session_state(voice_stack):
    """9. Test that session state, goals, and turns are reset on timeout."""
    session_mgr = voice_stack["session_mgr"]
    controller = voice_stack["controller"]
    engine = voice_stack["engine"]

    await engine.start()
    await controller.start()

    await session_mgr.activate(wake_word="hey_spidy")
    await session_mgr.record_utterance("open file manager")
    assert session_mgr.turn_count == 1
    assert session_mgr.current_goal == "open file manager"

    # Complete response cycle: PROCESSING -> SPEAKING -> AUDIO_FLUSH -> AWAKE
    await session_mgr.mark_speaking()
    await session_mgr.mark_audio_flush()
    await session_mgr.mark_response_complete()
    assert session_mgr.state == VoiceSessionState.AWAKE

    # Timeout after 0.1s inactivity
    await asyncio.sleep(0.25)
    assert session_mgr.state == VoiceSessionState.IDLE
    assert session_mgr.current_goal == ""

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_timeout_resets_transient_audio_buffers(voice_stack):
    """10. Test that STT buffer and OpenWakeWord temporal buffers are reset on timeout."""
    engine = voice_stack["engine"]
    wake_model = voice_stack["wake_model"]
    session_mgr = voice_stack["session_mgr"]
    controller = voice_stack["controller"]

    await engine.start()
    await controller.start()

    # Populate capture buffer
    with engine._buffer_lock:
        engine._capture_buffer.append(np.ones(1280, dtype=np.float32))

    # Activate & timeout
    await session_mgr.activate(wake_word="hey_spidy")
    await asyncio.sleep(0.25)

    # Buffers must be clean
    with engine._buffer_lock:
        assert len(engine._capture_buffer) == 0
    assert wake_model.reset_count >= 1

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_wake_ack_works_after_timeout(voice_stack):
    """11. Test that wake acknowledgement is spoken on subsequent sessions after timeout."""
    bus = voice_stack["bus"]
    engine = voice_stack["engine"]
    session_mgr = voice_stack["session_mgr"]
    controller = voice_stack["controller"]
    wake_model = voice_stack["wake_model"]

    ack_cfg = WakeAckConfig(enabled=True, phrases=["Yes Shiva."])
    acknowledger = WakeAcknowledger(ack_cfg)
    controller._wake_acknowledger = acknowledger
    controller._post_ack_flush_ms = 20  # Fast flush for tests

    await engine.start()
    await controller.start()

    chunk = np.zeros(1280, dtype=np.float32)

    # First session with ack
    wake_model.trigger("hey_spidy", "Hey Spidy")
    engine._process_wake_word(chunk)
    # Wait for ack + post-ack flush (20ms) + session timeout (100ms)
    await asyncio.sleep(0.25)
    assert session_mgr.state == VoiceSessionState.IDLE

    # Give second session a comfortable timeout window (2.0s) so it doesn't immediately expire
    session_mgr._timeout_seconds = 2.0

    # Second session with ack
    wake_model.trigger("wake_up_spidy", "Wake up Spidy")
    engine._process_wake_word(chunk)
    # Wait for ack and post-ack settle (20ms) to complete
    await asyncio.sleep(0.08)

    assert session_mgr.state == VoiceSessionState.AWAKE

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_hud_wake_reopen_works_after_timeout(voice_stack):
    """12. Test that HUD UI events transition cleanly across wake, timeout, and re-wake."""
    bus = voice_stack["bus"]
    engine = voice_stack["engine"]
    session_mgr = voice_stack["session_mgr"]
    controller = voice_stack["controller"]
    wake_model = voice_stack["wake_model"]

    ui_events: list[str] = []

    async def _on_ui_event(evt: UIStateChangeEvent):
        ui_events.append(evt.state)

    bus.subscribe("ui.state_change", _on_ui_event)

    await engine.start()
    await controller.start()

    chunk = np.zeros(1280, dtype=np.float32)

    # 1. Wake -> UI listening
    wake_model.trigger("hey_spidy", "Hey Spidy")
    engine._process_wake_word(chunk)
    await asyncio.sleep(0.05)
    assert "listening" in ui_events

    # 2. Timeout -> UI idle
    await asyncio.sleep(0.25)
    assert "idle" in ui_events

    # 3. Second wake -> UI listening again
    ui_events.clear()
    wake_model.trigger("hey_spidy", "Hey Spidy")
    engine._process_wake_word(chunk)
    await asyncio.sleep(0.05)
    assert "listening" in ui_events

    await controller.stop()
    await engine.stop()
