"""
Tests — STT Device Placement & RAM Optimizations
=================================================
10 focused tests verifying the GPU+RAM optimization changes.

 1. CPU fallback when ctranslate2 CUDA probe returns False
 2. CUDA device used when ctranslate2 probe returns True
 3. Explicit device="cpu" bypasses auto-detection (probe not called)
 4. Explicit device="cuda" bypasses auto-detection (probe not called)
 5a. compute_type=auto + cpu  -> int8
 5b. compute_type=auto + cuda -> float16
 6. CUDA model load failure falls back silently to CPU (no crash)
 7. load() idempotent — WhisperModel constructed exactly once
 8. BargeInDetector always passes device="cpu" to WhisperModel
 9. schedule_lazy_load() spawns a daemon thread that calls load_model
10. schedule_lazy_load() is a no-op when barge-in is disabled

Run with:
    pytest tests/unit/test_stt_device_placement.py -v
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from spidy.perception.voice.stt.whisper import (
    FasterWhisperRecognizer,
    _detect_cuda_via_ctranslate2,
)
from spidy.voice.barge_in import BargeInDetector


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _mock_wm():
    m = MagicMock()
    m.transcribe.return_value = (iter([]), MagicMock())
    return m


def _make_barge_in(enabled: bool = True) -> BargeInDetector:
    handler = MagicMock()
    handler.detect.return_value = None
    bus = MagicMock()
    bus.publish_threadsafe = MagicMock()
    return BargeInDetector(
        interruption_handler=handler,
        bus=bus,
        model_size="tiny.en",
        enabled=enabled,
    )


# ─── Group 1: Device Detection ────────────────────────────────────────────────


class TestCUDADetection:

    def test_cpu_fallback_when_cuda_probe_fails(self):
        """1. CUDA probe False -> device=cpu."""
        with patch(
            "spidy.perception.voice.stt.whisper._detect_cuda_via_ctranslate2",
            return_value=(False, "ctranslate2 CUDA build not present"),
        ):
            with patch("faster_whisper.WhisperModel", return_value=_mock_wm()) as mock_wm:
                rec = FasterWhisperRecognizer(model_size="tiny.en", device="auto", compute_type="auto")
                rec.load()
                assert rec.device == "cpu"
                # verify WhisperModel was called with device="cpu"
                call_kwargs = mock_wm.call_args[1]
                assert call_kwargs.get("device") == "cpu"

    def test_cuda_device_when_probe_succeeds(self):
        """2. CUDA probe True -> device=cuda."""
        with patch(
            "spidy.perception.voice.stt.whisper._detect_cuda_via_ctranslate2",
            return_value=(True, "cuda ok"),
        ):
            with patch("faster_whisper.WhisperModel", return_value=_mock_wm()) as mock_wm:
                rec = FasterWhisperRecognizer(model_size="tiny.en", device="auto", compute_type="auto")
                rec.load()
                assert rec.device == "cuda"
                call_kwargs = mock_wm.call_args[1]
                assert call_kwargs.get("device") == "cuda"


# ─── Group 2: Explicit Config Overrides ───────────────────────────────────────


class TestExplicitDeviceConfig:

    def test_explicit_cpu_bypasses_detection(self):
        """3. device='cpu' -> probe never called."""
        with patch(
            "spidy.perception.voice.stt.whisper._detect_cuda_via_ctranslate2"
        ) as mock_detect:
            with patch("faster_whisper.WhisperModel", return_value=_mock_wm()):
                rec = FasterWhisperRecognizer(model_size="tiny.en", device="cpu")
                rec.load()
                mock_detect.assert_not_called()
                assert rec.device == "cpu"

    def test_explicit_cuda_bypasses_detection(self):
        """4. device='cuda' -> probe never called."""
        with patch(
            "spidy.perception.voice.stt.whisper._detect_cuda_via_ctranslate2"
        ) as mock_detect:
            with patch("faster_whisper.WhisperModel", return_value=_mock_wm()):
                rec = FasterWhisperRecognizer(model_size="tiny.en", device="cuda", compute_type="float16")
                rec.load()
                mock_detect.assert_not_called()
                assert rec.device == "cuda"


# ─── Group 3: Compute-Type Auto-Resolution ────────────────────────────────────


class TestComputeTypeResolution:

    def test_auto_compute_type_is_int8_on_cpu(self):
        """5a. compute_type=auto + cpu -> int8."""
        with patch(
            "spidy.perception.voice.stt.whisper._detect_cuda_via_ctranslate2",
            return_value=(False, "no cuda"),
        ):
            with patch("faster_whisper.WhisperModel", return_value=_mock_wm()):
                rec = FasterWhisperRecognizer(model_size="tiny.en", device="auto", compute_type="auto")
                rec.load()
                assert rec.compute_type == "int8"

    def test_auto_compute_type_is_float16_on_cuda(self):
        """5b. compute_type=auto + cuda -> float16."""
        with patch(
            "spidy.perception.voice.stt.whisper._detect_cuda_via_ctranslate2",
            return_value=(True, "cuda ok"),
        ):
            with patch("faster_whisper.WhisperModel", return_value=_mock_wm()):
                rec = FasterWhisperRecognizer(model_size="tiny.en", device="auto", compute_type="auto")
                rec.load()
                assert rec.compute_type == "float16"


# ─── Group 4: Safe CUDA Failure Fallback ──────────────────────────────────────


class TestCUDAFallback:

    def test_cuda_load_failure_falls_back_to_cpu_silently(self):
        """6. WhisperModel raises on CUDA -> retries on CPU, no exception propagated."""
        call_count = {"n": 0}
        real_model = _mock_wm()

        def side_effect(model_size, device, compute_type, **kw):
            call_count["n"] += 1
            if device == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
            return real_model

        with patch(
            "spidy.perception.voice.stt.whisper._detect_cuda_via_ctranslate2",
            return_value=(True, "probe ok"),
        ):
            with patch("faster_whisper.WhisperModel", side_effect=side_effect):
                rec = FasterWhisperRecognizer(model_size="tiny.en", device="auto", compute_type="auto")
                rec.load()  # must not raise

        assert call_count["n"] == 2, "Expected 2 calls (cuda fail + cpu retry)"
        assert rec.device == "cpu"
        assert rec.compute_type == "int8"


# ─── Group 5: Model Load Idempotency ──────────────────────────────────────────


class TestModelLoadIdempotency:

    def test_load_called_twice_creates_model_once(self):
        """7. load() twice -> WhisperModel constructor called exactly once."""
        with patch(
            "spidy.perception.voice.stt.whisper._detect_cuda_via_ctranslate2",
            return_value=(False, "no cuda"),
        ):
            with patch("faster_whisper.WhisperModel", return_value=_mock_wm()) as mock_wm:
                rec = FasterWhisperRecognizer(model_size="tiny.en", device="auto")
                rec.load()
                rec.load()
                assert mock_wm.call_count == 1


# ─── Group 6: Barge-in Always CPU ─────────────────────────────────────────────


class TestBargeInAlwaysCPU:

    def test_barge_in_load_model_uses_cpu(self):
        """8. BargeInDetector.load_model() always passes device='cpu', compute_type='int8'."""
        bi = _make_barge_in()
        with patch("faster_whisper.WhisperModel", return_value=_mock_wm()) as mock_wm:
            bi.load_model()
        _, kwargs = mock_wm.call_args
        assert kwargs.get("device") == "cpu", f"got device={kwargs.get('device')!r}"
        assert kwargs.get("compute_type") == "int8", f"got compute_type={kwargs.get('compute_type')!r}"


# ─── Group 7: Lazy-Load Thread Scheduling ─────────────────────────────────────


class TestLazyLoadScheduling:

    def test_schedule_lazy_load_spawns_daemon_thread_and_calls_load(self):
        """9. schedule_lazy_load() starts spidy-barge-in-loader thread that calls load_model."""
        bi = _make_barge_in(enabled=True)
        load_called = threading.Event()

        def _fake_load():
            load_called.set()

        bi.load_model = _fake_load

        bi.schedule_lazy_load(delay_seconds=0.05)

        # Thread must exist right after scheduling
        thread_names = [t.name for t in threading.enumerate()]
        assert "spidy-barge-in-loader" in thread_names, (
            f"spidy-barge-in-loader not found; threads={thread_names}"
        )

        # load_model must be called within 2 s
        assert load_called.wait(timeout=2.0), "load_model was not called within 2 s"

    def test_schedule_lazy_load_noop_when_disabled(self):
        """10. schedule_lazy_load() does nothing when barge-in is disabled."""
        bi = _make_barge_in(enabled=False)
        load_called = threading.Event()
        bi.load_model = lambda: load_called.set()

        bi.schedule_lazy_load(delay_seconds=0.05)

        # No loader thread
        thread_names = [t.name for t in threading.enumerate()]
        assert "spidy-barge-in-loader" not in thread_names

        # load_model must NOT be called
        assert not load_called.wait(timeout=0.3), "load_model should not be called when disabled"
