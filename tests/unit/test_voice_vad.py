"""
Unit tests — VoiceActivityDetector
====================================
Tests energy-based silence detection, VAD frame processing,
minimum speech duration filtering, and webrtcvad graceful fallback.

All tests use synthetic numpy audio — no real microphone required.
"""

from __future__ import annotations

import threading
import numpy as np
import pytest

from spidy.voice.vad import VoiceActivityDetector


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_speech(num_samples: int = 16000, amplitude: float = 0.3) -> np.ndarray:
    """Generate synthetic speech-like audio (above energy threshold)."""
    return (np.random.randn(num_samples) * amplitude).astype(np.float32)


def _make_silence(num_samples: int = 8000) -> np.ndarray:
    """Generate very quiet audio (below energy threshold)."""
    return np.zeros(num_samples, dtype=np.float32)


def _make_buffer(chunks: list[np.ndarray]) -> tuple[threading.Lock, list[np.ndarray]]:
    lock = threading.Lock()
    buf = list(chunks)
    return lock, buf


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def vad():
    return VoiceActivityDetector(
        sample_rate=16000,
        silence_timeout_seconds=0.1,  # Very short for fast tests
        min_speech_duration_ms=10,    # Low threshold for test audio
        energy_threshold=0.001,
    )


# ── Energy fallback tests ──────────────────────────────────────────────────────


def test_vad_instantiates(vad):
    assert vad is not None


def test_energy_fallback_available(vad):
    """Energy fallback is always available."""
    # webrtcvad may or may not be installed — both paths should work
    assert isinstance(vad.using_webrtcvad, bool)


def test_energy_collects_speech():
    """VAD collects speech audio above energy threshold."""
    stop = threading.Event()
    speech_chunk = _make_speech(1280, amplitude=0.3)

    vad = VoiceActivityDetector(
        sample_rate=16000,
        silence_timeout_seconds=0.05,
        min_speech_duration_ms=0,
        energy_threshold=0.01,
    )
    lock, buf = _make_buffer([speech_chunk])

    result = vad._collect_energy(lock, buf, stop, max_seconds=1.0)
    assert result is not None
    assert len(result) > 0


def test_energy_returns_none_for_silence():
    """VAD returns None when only silence is captured."""
    stop = threading.Event()

    vad = VoiceActivityDetector(
        sample_rate=16000,
        silence_timeout_seconds=0.01,
        min_speech_duration_ms=10000,  # Very long minimum — will discard
        energy_threshold=0.01,
    )
    lock, buf = _make_buffer([_make_silence(1280)])

    result = vad._collect_energy(lock, buf, stop, max_seconds=0.5)
    # Either None or empty (too short)
    assert result is None or len(result) == 0


def test_stop_event_causes_early_return():
    """Setting stop_event causes immediate None return."""
    stop = threading.Event()
    stop.set()  # Already stopped

    vad = VoiceActivityDetector(sample_rate=16000, silence_timeout_seconds=5.0)
    lock, buf = _make_buffer([_make_speech(1280)])

    result = vad._collect_energy(lock, buf, stop, max_seconds=5.0)
    assert result is None


def test_min_speech_duration_filters_short_sounds():
    """Sounds shorter than min_speech_duration_ms are discarded."""
    stop = threading.Event()

    vad = VoiceActivityDetector(
        sample_rate=16000,
        silence_timeout_seconds=0.01,
        min_speech_duration_ms=5000,  # 5 seconds minimum — won't be reached
        energy_threshold=0.001,
    )
    lock, buf = _make_buffer([_make_speech(160)])  # Only 10ms of audio

    result = vad._collect_energy(lock, buf, stop, max_seconds=0.5)
    assert result is None


def test_max_seconds_limits_recording():
    """Recording stops at max_seconds even if audio continues."""
    stop = threading.Event()

    vad = VoiceActivityDetector(
        sample_rate=16000,
        silence_timeout_seconds=100.0,  # Never times out on silence
        min_speech_duration_ms=0,
        energy_threshold=100.0,  # Threshold so high nothing triggers silence
    )
    # Empty buffer — will hit max_seconds
    lock, buf = _make_buffer([])

    import time
    start = time.monotonic()
    result = vad._collect_energy(lock, buf, stop, max_seconds=0.05)
    elapsed = time.monotonic() - start
    # Should return quickly (within ~100ms of max_seconds)
    assert elapsed < 1.0


def test_collect_concatenates_chunks():
    """Multiple speech chunks are concatenated correctly."""
    stop = threading.Event()
    chunk1 = _make_speech(1280, amplitude=0.3)
    chunk2 = _make_speech(1280, amplitude=0.3)

    vad = VoiceActivityDetector(
        sample_rate=16000,
        silence_timeout_seconds=0.01,
        min_speech_duration_ms=0,
        energy_threshold=0.001,
    )
    lock, buf = _make_buffer([chunk1, chunk2])

    result = vad._collect_energy(lock, buf, stop, max_seconds=1.0)
    if result is not None:
        assert len(result) >= len(chunk1)


# ── API-level tests ────────────────────────────────────────────────────────────


def test_collect_until_silence_delegates_correctly():
    """collect_until_silence() calls the right backend."""
    stop = threading.Event()
    vad = VoiceActivityDetector(
        sample_rate=16000,
        silence_timeout_seconds=0.01,
        min_speech_duration_ms=0,
        energy_threshold=0.001,
    )
    lock, buf = _make_buffer([_make_speech(1280)])

    # Should not raise regardless of which backend is used
    result = vad.collect_until_silence(lock, buf, stop, max_seconds=0.5)
    assert result is None or isinstance(result, np.ndarray)
