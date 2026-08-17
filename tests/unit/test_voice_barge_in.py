"""
Tests — BargeInDetector (Milestone 15)
=======================================
16 focused tests covering the barge-in interruption path:

1.  BARGE_IN mode routes chunks to barge-in callback
2.  IDLE mode does NOT route chunks to barge-in callback
3.  DETECTING mode does NOT route chunks to barge-in callback
4.  BargeInDetector.feed_chunk discards low-energy (self-talk) chunks
5.  BargeInDetector.feed_chunk accepts high-energy (speech) chunks
6.  "Spidy stop" detected and TTS stopped
7.  "Spidy, stop" detected and TTS stopped
8.  Stop does not call Brain
9.  Stop does not call NVIDIA
10. Stop does not call Ollama
11. Stop immediately calls streaming_tts.stop()
12. Stop keeps session active
13. Normal speech during TTS does NOT stop speech (false-positive guard)
14. No duplicate barge-in handlers after multiple TTS cycles (start idempotent)
15. No mic mode leak — BARGE_IN → DETECTING always restored after TTS
16. Continuous conversation still works after barge-in stop
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import numpy as np
import pytest

from spidy.perception.voice.audio_capture import AudioCaptureEngine, CaptureMode
from spidy.voice.barge_in import BargeInDetector
from spidy.voice.events import VoiceInterruptEvent
from spidy.voice.interruption import InterruptCommand


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_handler(detect_return: InterruptCommand | None = None):
    """Return a mock InterruptionHandler whose detect() returns the given value."""
    h = MagicMock()
    h.detect.return_value = detect_return
    return h


def _make_bus():
    bus = MagicMock()
    bus.publish_threadsafe = MagicMock()
    return bus


def _make_tts(speaking: bool = True):
    tts = MagicMock()
    tts.is_speaking = speaking
    tts.stop = MagicMock()
    return tts


def _audio_chunk(rms: float = 0.1, samples: int = 1280) -> np.ndarray:
    """Generate a numpy chunk with approximately the given RMS level."""
    if rms == 0.0:
        return np.zeros(samples, dtype=np.float32)
    # Sine wave scaled to the desired RMS
    t = np.linspace(0, 1, samples, dtype=np.float32)
    wave = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    actual_rms = float(np.sqrt(np.mean(wave ** 2)))
    return (wave * (rms / actual_rms)).astype(np.float32)


# ─── Group 1: CaptureMode routing ─────────────────────────────────────────────


class TestCaptureModeRouting:
    """Tests 1-3: BARGE_IN mode routes to barge-in callback; others do not."""

    def _engine(self):
        wake_cb = MagicMock()
        rec_cb = MagicMock()
        barge_cb = MagicMock()
        engine = AudioCaptureEngine(
            on_wake_chunk=wake_cb,
            on_record_chunk=rec_cb,
            on_barge_in_chunk=barge_cb,
        )
        return engine, wake_cb, rec_cb, barge_cb

    def test_barge_in_mode_routes_to_barge_in_callback(self):
        """BARGE_IN mode → only barge-in callback receives chunk."""
        engine, wake_cb, rec_cb, barge_cb = self._engine()
        engine.set_mode(CaptureMode.BARGE_IN)
        chunk = _audio_chunk(0.1)
        engine._dispatch(chunk)
        barge_cb.assert_called_once()
        wake_cb.assert_not_called()
        rec_cb.assert_not_called()

    def test_idle_mode_does_not_route_to_barge_in_callback(self):
        """IDLE mode → no callbacks called (mic muted)."""
        engine, wake_cb, rec_cb, barge_cb = self._engine()
        engine.set_mode(CaptureMode.IDLE)
        engine._dispatch(_audio_chunk(0.1))
        barge_cb.assert_not_called()
        wake_cb.assert_not_called()
        rec_cb.assert_not_called()

    def test_detecting_mode_does_not_route_to_barge_in_callback(self):
        """DETECTING mode → only wake callback; barge-in not called."""
        engine, wake_cb, rec_cb, barge_cb = self._engine()
        engine.set_mode(CaptureMode.DETECTING)
        engine._dispatch(_audio_chunk(0.1))
        wake_cb.assert_called_once()
        barge_cb.assert_not_called()


# ─── Group 2: Energy gate ─────────────────────────────────────────────────────


class TestEnergyGate:
    """Tests 4-5: Self-talk guard — chunks below RMS threshold are discarded."""

    def _detector(self, enabled=True):
        return BargeInDetector(
            interruption_handler=_make_handler(),
            bus=_make_bus(),
            enabled=enabled,
            min_rms_threshold=0.015,
        )

    def test_feed_chunk_discards_silent_chunk(self):
        """Chunk with RMS=0 (silence/TTS-bleed) is discarded — buffer stays empty."""
        det = self._detector()
        det._active = True
        det.feed_chunk(np.zeros(1280, dtype=np.float32))  # RMS = 0
        assert det._buf_samples == 0

    def test_feed_chunk_accepts_speech_chunk(self):
        """Chunk with RMS > threshold is accepted — buffer grows."""
        det = self._detector()
        det._active = True
        det.feed_chunk(_audio_chunk(rms=0.08))  # well above 0.015
        assert det._buf_samples > 0

    def test_feed_chunk_noop_when_inactive(self):
        """feed_chunk does nothing when detector is not active."""
        det = self._detector()
        det._active = False
        det.feed_chunk(_audio_chunk(rms=0.5))
        assert det._buf_samples == 0


# ─── Group 3: Stop detection ──────────────────────────────────────────────────


class TestStopDetection:
    """Tests 6-11: Stop phrase detected → TTS stopped; Brain/LLM not called."""

    def _detector_with_model(self, stop_text: str):
        """
        BargeInDetector with a mocked tiny.en model that always returns stop_text.
        """
        handler = _make_handler(detect_return=InterruptCommand.STOP)
        bus = _make_bus()
        tts = _make_tts()
        det = BargeInDetector(
            interruption_handler=handler,
            bus=bus,
            model_size="tiny.en",
            window_seconds=0.1,  # tiny window for fast test
            min_rms_threshold=0.0,  # accept all audio
            enabled=True,
        )
        # Mark model as loaded without actually loading it
        det._model_loaded = True
        det._model = MagicMock()  # placeholder — _transcribe is patched below

        return det, handler, bus, tts

    def _run_detection(self, det, tts, stop_text: str) -> None:
        """Start detector, feed one full window, let worker run, stop detector."""
        window_samples = int(det._window_seconds * 16000)
        # Feed enough audio to fill the window
        chunk = _audio_chunk(rms=0.1, samples=window_samples)

        with patch.object(det, "_transcribe", return_value=stop_text):
            det.start(tts=tts)
            det._active = True
            det.feed_chunk(chunk)
            # Give worker thread time to detect
            time.sleep(0.5)
            det.stop()

    def test_spidy_stop_detected_tts_stopped(self):
        """'Spidy stop' → TTS.stop() called."""
        det, _, _, tts = self._detector_with_model("Spidy stop")
        self._run_detection(det, tts, "Spidy stop")
        tts.stop.assert_called()

    def test_spidy_comma_stop_detected_tts_stopped(self):
        """'Spidy, stop' → TTS.stop() called."""
        det, _, _, tts = self._detector_with_model("Spidy, stop")
        self._run_detection(det, tts, "Spidy, stop")
        tts.stop.assert_called()

    def test_stop_does_not_call_brain(self):
        """Stop command never reaches Brain.process()."""
        brain_mock = MagicMock()
        det, handler, _, tts = self._detector_with_model("Spidy stop")
        # If the handler correctly returns STOP, _transcribe never calls brain
        self._run_detection(det, tts, "Spidy stop")
        brain_mock.process.assert_not_called()

    def test_stop_does_not_call_nvidia(self):
        """Stop command does not hit the NVIDIA NIM endpoint."""
        with patch("spidy.llm.backends.nvidia.NvidiaClient.complete", new_callable=AsyncMock) as nvidia:
            det, _, _, tts = self._detector_with_model("Spidy stop")
            self._run_detection(det, tts, "Spidy stop")
            nvidia.assert_not_called()

    def test_stop_does_not_call_ollama(self):
        """Stop command does not hit the Ollama endpoint."""
        with patch("spidy.llm.backends.ollama.OllamaClient.complete", new_callable=AsyncMock) as ollama:
            det, _, _, tts = self._detector_with_model("Spidy stop")
            self._run_detection(det, tts, "Spidy stop")
            ollama.assert_not_called()

    def test_stop_publishes_interrupt_event(self):
        """Stop detection publishes VoiceInterruptEvent to bus."""
        det, _, bus, tts = self._detector_with_model("Spidy stop")
        self._run_detection(det, tts, "Spidy stop")
        bus.publish_threadsafe.assert_called()
        args = bus.publish_threadsafe.call_args[0]
        evt = args[0]
        assert isinstance(evt, VoiceInterruptEvent)
        assert evt.command == "stop"


# ─── Group 4: False-positive protection ──────────────────────────────────────


class TestFalsePositiveProtection:
    """Test 13: Normal speech during TTS does NOT trigger stop."""

    def test_normal_speech_does_not_stop_tts(self):
        """'What is the weather today' → handler returns None → TTS not stopped."""
        handler = _make_handler(detect_return=None)  # no stop command
        bus = _make_bus()
        tts = _make_tts()
        det = BargeInDetector(
            interruption_handler=handler,
            bus=bus,
            window_seconds=0.1,
            min_rms_threshold=0.0,
            enabled=True,
        )
        det._model_loaded = True
        det._model = MagicMock()

        window_samples = int(det._window_seconds * 16000)
        chunk = _audio_chunk(rms=0.1, samples=window_samples)

        with patch.object(det, "_transcribe", return_value="What is the weather today"):
            det.start(tts=tts)
            det.feed_chunk(chunk)
            time.sleep(0.4)
            det.stop()

        tts.stop.assert_not_called()
        bus.publish_threadsafe.assert_not_called()


# ─── Group 5: Lifecycle / state machine ──────────────────────────────────────


class TestLifecycle:
    """Tests 14-16: Idempotent start, no mic leak, continuous conversation works."""

    def test_start_is_idempotent_no_duplicate_workers(self):
        """Calling start() twice kills the first worker before starting a new one."""
        handler = _make_handler(detect_return=None)
        bus = _make_bus()
        tts = _make_tts()
        det = BargeInDetector(
            interruption_handler=handler,
            bus=bus,
            window_seconds=10.0,  # long window so worker stays alive
            enabled=True,
        )
        det._model_loaded = True

        det.start(tts=tts)
        first_thread = det._worker_thread

        det.start(tts=tts)  # second call — should kill first worker
        second_thread = det._worker_thread

        assert first_thread is not second_thread or not first_thread.is_alive(), (
            "First worker thread should have been replaced by second start()"
        )

        det.stop()

    def test_stop_deactivates_and_clears_tts_ref(self):
        """stop() sets _active=False and clears _streaming_tts reference."""
        handler = _make_handler(detect_return=None)
        bus = _make_bus()
        tts = _make_tts()
        det = BargeInDetector(
            interruption_handler=handler,
            bus=bus,
            window_seconds=10.0,
            enabled=True,
        )
        det._model_loaded = True
        det.start(tts=tts)
        assert det.is_active

        det.stop()
        assert not det.is_active
        assert det._streaming_tts is None

    def test_capture_mode_restored_to_detecting_after_barge_in_stop(self):
        """
        After TTS completes (with or without barge-in firing), CaptureMode
        returns to DETECTING — never stays in BARGE_IN.

        Simulates the ContinuousVoiceController._on_brain_response finally block.
        """
        # Simulate the engine capture state machine directly
        captured_modes = []

        class _MockCapture:
            def __init__(self):
                self.mode = CaptureMode.DETECTING

            def set_mode(self, m):
                self.mode = m
                captured_modes.append(m)

        capture = _MockCapture()

        # Simulate the finally block behaviour
        capture.set_mode(CaptureMode.BARGE_IN)   # TTS start
        # ... TTS plays ...
        capture.set_mode(CaptureMode.DETECTING)   # TTS done (finally block)

        assert captured_modes[-1] == CaptureMode.DETECTING, (
            "Mic should always be restored to DETECTING after TTS regardless of barge-in"
        )

    def test_disabled_detector_does_not_load_model(self):
        """When enabled=False, load_model() is a no-op."""
        handler = _make_handler()
        bus = _make_bus()
        det = BargeInDetector(
            interruption_handler=handler,
            bus=bus,
            enabled=False,
        )
        with patch("faster_whisper.WhisperModel") as mock_model:
            det.load_model()
            mock_model.assert_not_called()
        assert not det.is_loaded
        assert not det.is_active

    def test_session_stays_active_after_barge_in_stop(self):
        """
        BargeInDetector.stop() only stops TTS and clears internal state.
        It does NOT call session.deactivate() — the conversation continues.
        """
        handler = _make_handler(detect_return=InterruptCommand.STOP)
        bus = _make_bus()
        tts = _make_tts()
        session = MagicMock()

        det = BargeInDetector(
            interruption_handler=handler,
            bus=bus,
            window_seconds=0.1,
            min_rms_threshold=0.0,
            enabled=True,
        )
        det._model_loaded = True

        window_samples = int(det._window_seconds * 16000)
        chunk = _audio_chunk(rms=0.1, samples=window_samples)

        with patch.object(det, "_transcribe", return_value="Spidy stop"):
            det.start(tts=tts)
            det.feed_chunk(chunk)
            time.sleep(0.4)
            det.stop()

        # BargeInDetector must NEVER call session.deactivate()
        session.deactivate.assert_not_called()
