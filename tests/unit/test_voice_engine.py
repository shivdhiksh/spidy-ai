"""
Integration tests for VoiceEngine state machine
=================================================
Tests the full SLEEPING → WAKING → LISTENING → PROCESSING → SLEEPING
state transition loop using mocked audio components.

No real microphone, no real AI model, no real TTS — all replaced
with fakes that simulate the expected behaviour.

Run with:
    pytest tests/unit/test_voice_engine.py -v
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spidy.core.event_bus import EventBus
from spidy.perception.voice.audio_capture import AudioCaptureEngine, CaptureMode
from spidy.perception.voice.engine import (
    VoiceEngine,
    VoiceState,
    VoiceTranscriptEvent,
    WakeWordDetectedEvent,
    VoiceListeningEvent,
    VoiceSpeakingStartEvent,
    VoiceSpeakingEndEvent,
)
from spidy.perception.voice.stt.base import TranscriptResult, SpeechRecognizer
from spidy.perception.voice.tts.base import AudioBuffer, TTSEngine
from spidy.perception.voice.wake_word.base import WakeWordModel


# ─── Fakes ────────────────────────────────────────────────────────────────────


class FakeWakeWordModel(WakeWordModel):
    """Controllable wake word model: returns _score on every chunk."""

    def __init__(self) -> None:
        self._score = 0.0
        self._chunk_count = 0

    def load(self, model_path=None) -> None:
        pass

    def unload(self) -> None:
        pass

    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        self._chunk_count += 1
        return self._score

    def trigger(self) -> None:
        """Simulate wake word detection."""
        self._score = 1.0

    def silence(self) -> None:
        """Back to silent mode."""
        self._score = 0.0

    @property
    def model_name(self) -> str:
        return "fake_wake_word"

    @property
    def chunk_size(self) -> int:
        return 1280


class FakeSpeechRecognizer(SpeechRecognizer):
    """Returns pre-set transcript text."""

    def __init__(self, transcript: str = "hello spidy") -> None:
        self._transcript = transcript

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    async def transcribe(self, audio_data: np.ndarray) -> TranscriptResult:
        await asyncio.sleep(0.01)  # Simulate processing delay
        return TranscriptResult(text=self._transcript)

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def model_name(self) -> str:
        return "fake_stt"


class FakeTTSEngine(TTSEngine):
    """Records calls to speak(), completes instantly."""

    def __init__(self) -> None:
        self._speaking = False
        self.speak_calls: list[str] = []

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    async def synthesize(self, text: str) -> AudioBuffer:
        return AudioBuffer(
            samples=np.zeros(100, dtype=np.float32),
            sample_rate=22050,
            duration_seconds=0.01,
        )

    async def speak(self, text: str) -> None:
        self._speaking = True
        self.speak_calls.append(text)
        await asyncio.sleep(0.02)  # Simulate short playback
        self._speaking = False

    def stop(self) -> None:
        self._speaking = False

    @property
    def voice_name(self) -> str:
        return "fake_voice"

    @property
    def is_speaking(self) -> bool:
        return self._speaking


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_engine(
    bus: EventBus,
    wake_model: FakeWakeWordModel | None = None,
    recognizer: FakeSpeechRecognizer | None = None,
    tts: FakeTTSEngine | None = None,
    max_record_seconds: float = 0.3,
    silence_timeout_seconds: float = 0.1,
    wake_threshold: float = 0.5,
) -> VoiceEngine:
    return VoiceEngine(
        bus=bus,
        wake_word_model=wake_model or FakeWakeWordModel(),
        recognizer=recognizer or FakeSpeechRecognizer(),
        tts=tts or FakeTTSEngine(),
        sample_rate=16000,
        max_record_seconds=max_record_seconds,
        silence_timeout_seconds=silence_timeout_seconds,
        wake_threshold=wake_threshold,
    )


# ─── AudioCaptureEngine Tests ─────────────────────────────────────────────────


class TestAudioCaptureEngine:
    """Unit tests for AudioCaptureEngine without real microphone."""

    def test_initial_state_is_idle(self):
        engine = AudioCaptureEngine()
        assert engine.mode == CaptureMode.IDLE

    def test_set_mode_changes_mode(self):
        engine = AudioCaptureEngine()
        engine.set_mode(CaptureMode.DETECTING)
        assert engine.mode == CaptureMode.DETECTING
        engine.set_mode(CaptureMode.RECORDING)
        assert engine.mode == CaptureMode.RECORDING

    def test_dispatch_routes_to_wake_callback(self):
        received: list[np.ndarray] = []

        def on_wake(chunk: np.ndarray) -> None:
            received.append(chunk)

        engine = AudioCaptureEngine(on_wake_chunk=on_wake)
        engine.set_mode(CaptureMode.DETECTING)

        chunk = np.zeros(1280, dtype=np.float32)
        engine._dispatch(chunk)

        assert len(received) == 1
        np.testing.assert_array_equal(received[0], chunk)

    def test_dispatch_routes_to_record_callback(self):
        received: list[np.ndarray] = []

        def on_record(chunk: np.ndarray) -> None:
            received.append(chunk)

        engine = AudioCaptureEngine(on_record_chunk=on_record)
        engine.set_mode(CaptureMode.RECORDING)

        chunk = np.ones(1280, dtype=np.float32)
        engine._dispatch(chunk)

        assert len(received) == 1

    def test_dispatch_idle_discards_chunk(self):
        received: list[np.ndarray] = []

        engine = AudioCaptureEngine(
            on_wake_chunk=lambda c: received.append(c),
            on_record_chunk=lambda c: received.append(c),
        )
        engine.set_mode(CaptureMode.IDLE)
        engine._dispatch(np.zeros(1280, dtype=np.float32))

        assert len(received) == 0

    def test_callback_exception_does_not_crash_dispatch(self):
        def bad_callback(chunk: np.ndarray) -> None:
            raise RuntimeError("boom")

        engine = AudioCaptureEngine(on_wake_chunk=bad_callback)
        engine.set_mode(CaptureMode.DETECTING)
        # Must not raise
        engine._dispatch(np.zeros(1280, dtype=np.float32))

    def test_double_start_raises(self):
        """Starting an already-running engine must raise."""
        engine = AudioCaptureEngine()
        # Patch sd.InputStream so no real microphone needed
        with patch("spidy.perception.voice.audio_capture.AudioCaptureEngine._capture_loop"):
            engine._thread = threading.Thread(target=lambda: time.sleep(1), daemon=True)
            engine._thread.start()
            with pytest.raises(RuntimeError, match="already running"):
                engine.start()
            engine._stop_event.set()


# ─── VoiceEngine State Machine Tests ─────────────────────────────────────────


class TestVoiceEngineStateMachine:
    """Tests that VoiceEngine state transitions work correctly."""

    @pytest.mark.asyncio
    async def test_initial_state_is_sleeping(self):
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        engine = make_engine(bus)
        assert engine.state == VoiceState.SLEEPING

    @pytest.mark.asyncio
    async def test_wake_word_transitions_to_waking(self):
        """Simulating a wake event should set state to WAKING."""
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        wake_model = FakeWakeWordModel()
        engine = make_engine(bus, wake_model=wake_model)

        # Simulate wake word detection directly via the callback
        wake_model.trigger()
        engine._process_wake_word(np.zeros(1280, dtype=np.float32))

        assert engine.state == VoiceState.WAKING
        assert engine._wake_detected.is_set()

    @pytest.mark.asyncio
    async def test_wake_event_published_on_detection(self):
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        wake_model = FakeWakeWordModel()
        engine = make_engine(bus, wake_model=wake_model)

        received_events: list[WakeWordDetectedEvent] = []

        async def on_wake(evt: WakeWordDetectedEvent) -> None:
            received_events.append(evt)

        bus.subscribe("wake_word.detected", on_wake)

        # Trigger wake word
        wake_model.trigger()
        engine._process_wake_word(np.zeros(1280, dtype=np.float32))

        # Give the threadsafe publish a moment to deliver
        await asyncio.sleep(0.1)

        assert len(received_events) == 1
        assert received_events[0].confidence == 1.0

    @pytest.mark.asyncio
    async def test_below_threshold_does_not_trigger(self):
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        wake_model = FakeWakeWordModel()
        engine = make_engine(bus, wake_model=wake_model, wake_threshold=0.9)

        wake_model._score = 0.4  # below threshold
        engine._process_wake_word(np.zeros(1280, dtype=np.float32))

        assert engine.state == VoiceState.SLEEPING
        assert not engine._wake_detected.is_set()

    @pytest.mark.asyncio
    async def test_speak_publishes_events(self):
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        tts = FakeTTSEngine()
        engine = make_engine(bus, tts=tts)

        start_events: list[VoiceSpeakingStartEvent] = []
        end_events: list[VoiceSpeakingEndEvent] = []

        bus.subscribe("voice.speaking_start", lambda e: start_events.append(e))
        bus.subscribe("voice.speaking_end", lambda e: end_events.append(e))

        await engine.speak("Hello Shiva")

        assert len(start_events) == 1
        assert start_events[0].text == "Hello Shiva"
        assert len(end_events) == 1
        assert tts.speak_calls == ["Hello Shiva"]

    @pytest.mark.asyncio
    async def test_speak_empty_string_does_nothing(self):
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        tts = FakeTTSEngine()
        engine = make_engine(bus, tts=tts)

        await engine.speak("")
        await engine.speak("   ")

        assert tts.speak_calls == []

    @pytest.mark.asyncio
    async def test_speak_returns_to_sleeping(self):
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        engine = make_engine(bus)

        await engine.speak("Testing")
        assert engine.state == VoiceState.SLEEPING

    @pytest.mark.asyncio
    async def test_speak_switches_to_detecting_after(self):
        """After speaking, AudioCaptureEngine must be in DETECTING mode."""
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        engine = make_engine(bus)

        await engine.speak("Testing")
        assert engine._capture_engine.mode == CaptureMode.DETECTING

    @pytest.mark.asyncio
    async def test_record_until_silence_returns_none_when_stopped(self):
        """If stop_event is set, recording should abort and return None."""
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        engine = make_engine(bus)
        engine._stop_event.set()

        result = engine._record_until_silence()
        assert result is None

    @pytest.mark.asyncio
    async def test_record_until_silence_returns_audio_from_buffer(self):
        """Filling the buffer with non-silent audio should return it."""
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        engine = make_engine(bus, max_record_seconds=0.5, silence_timeout_seconds=0.1)

        # Pre-fill buffer with loud audio (energy > threshold)
        loud_chunk = np.ones(1280, dtype=np.float32) * 0.5
        with engine._buffer_lock:
            for _ in range(5):
                engine._capture_buffer.append(loud_chunk)
            # Then a bunch of silent chunks to trigger silence detection
            silent_chunk = np.zeros(1280, dtype=np.float32)
            for _ in range(10):
                engine._capture_buffer.append(silent_chunk)

        result = engine._record_until_silence()
        assert result is not None
        assert len(result) > 0


# ─── SkillRegistry Tests ──────────────────────────────────────────────────────


class TestSkillRegistryFromVoiceModule:
    """Basic smoke test that SkillRegistry imports and works."""

    def test_registry_import(self):
        from spidy.skills.registry import SkillRegistry
        r = SkillRegistry()
        assert len(r) == 0
        assert repr(r) == "SkillRegistry([])"
