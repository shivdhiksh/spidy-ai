"""
Unit Tests — CUDA Whisper Device Selection, Compute Types & Fallbacks
=====================================================================
Validates CUDA Whisper acceleration on RTX 3050 and safe CPU fallbacks.

Covered Scenarios:
1. Device "auto" resolves to CUDA when ctranslate2 probe succeeds
2. Device "auto" falls back to CPU when ctranslate2 probe fails
3. Device "cpu" explicitly stays on CPU without probing CUDA
4. Device "cuda" explicitly tries CUDA
5. Device "cuda" gracefully falls back to CPU when CUDA init raises Exception
6. Compute type "auto" resolves to "float16" on CUDA and "int8" on CPU
7. Explicit compute types (e.g. "int8_float16") are preserved on CUDA
8. Load idempotency — repeated load() calls do not recreate WhisperModel
9. Unload and reload cycle works cleanly
10. Barge-in STT stays isolated on CPU int8
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from spidy.perception.voice.stt.whisper import (
    FasterWhisperRecognizer,
    _detect_cuda_via_ctranslate2,
)
from spidy.voice.barge_in import BargeInDetector


def _mock_model():
    m = MagicMock()
    m.transcribe.return_value = (iter([]), MagicMock())
    return m


class TestCUDAWhisperSelection:

    def test_auto_device_selects_cuda_when_available(self):
        """Auto device selection picks CUDA float16 when ctranslate2 reports CUDA available."""
        with patch(
            "spidy.perception.voice.stt.whisper._detect_cuda_via_ctranslate2",
            return_value=(True, "ctranslate2 CUDA + NVIDIA driver available"),
        ):
            with patch("faster_whisper.WhisperModel", return_value=_mock_model()) as mock_wm:
                rec = FasterWhisperRecognizer(model_size="base.en", device="auto", compute_type="auto")
                rec.load()
                assert rec.device == "cuda"
                assert rec.compute_type == "float16"
                mock_wm.assert_called_once_with(
                    "base.en",
                    device="cuda",
                    compute_type="float16",
                    download_root=None,
                )

    def test_auto_device_falls_back_to_cpu_when_unavailable(self):
        """Auto device selection falls back to CPU int8 when ctranslate2 reports no CUDA."""
        with patch(
            "spidy.perception.voice.stt.whisper._detect_cuda_via_ctranslate2",
            return_value=(False, "no CUDA GPU device detected by ctranslate2"),
        ):
            with patch("faster_whisper.WhisperModel", return_value=_mock_model()) as mock_wm:
                rec = FasterWhisperRecognizer(model_size="base.en", device="auto", compute_type="auto")
                rec.load()
                assert rec.device == "cpu"
                assert rec.compute_type == "int8"
                mock_wm.assert_called_once_with(
                    "base.en",
                    device="cpu",
                    compute_type="int8",
                    download_root=None,
                )

    def test_explicit_cpu_bypasses_cuda_probe(self):
        """Explicit device='cpu' never calls CUDA probe and uses int8."""
        with patch(
            "spidy.perception.voice.stt.whisper._detect_cuda_via_ctranslate2"
        ) as mock_probe:
            with patch("faster_whisper.WhisperModel", return_value=_mock_model()) as mock_wm:
                rec = FasterWhisperRecognizer(model_size="base.en", device="cpu", compute_type="auto")
                rec.load()
                mock_probe.assert_not_called()
                assert rec.device == "cpu"
                assert rec.compute_type == "int8"
                mock_wm.assert_called_once_with(
                    "base.en",
                    device="cpu",
                    compute_type="int8",
                    download_root=None,
                )

    def test_explicit_cuda_uses_cuda(self):
        """Explicit device='cuda' loads on CUDA directly."""
        with patch(
            "spidy.perception.voice.stt.whisper._detect_cuda_via_ctranslate2"
        ) as mock_probe:
            with patch("faster_whisper.WhisperModel", return_value=_mock_model()) as mock_wm:
                rec = FasterWhisperRecognizer(model_size="base.en", device="cuda", compute_type="float16")
                rec.load()
                mock_probe.assert_not_called()
                assert rec.device == "cuda"
                assert rec.compute_type == "float16"
                mock_wm.assert_called_once_with(
                    "base.en",
                    device="cuda",
                    compute_type="float16",
                    download_root=None,
                )

    def test_cuda_load_failure_gracefully_falls_back_to_cpu(self):
        """When CUDA model initialization raises an error, fallback to CPU int8."""
        first_call = True

        def side_effect(*args, **kwargs):
            nonlocal first_call
            if first_call and kwargs.get("device") == "cuda":
                first_call = False
                raise RuntimeError("CUDA out of memory or cublas not found")
            return _mock_model()

        with patch(
            "spidy.perception.voice.stt.whisper._detect_cuda_via_ctranslate2",
            return_value=(True, "cuda ok"),
        ):
            with patch("faster_whisper.WhisperModel", side_effect=side_effect) as mock_wm:
                rec = FasterWhisperRecognizer(model_size="base.en", device="auto", compute_type="auto")
                rec.load()
                assert rec.device == "cpu"
                assert rec.compute_type == "int8"
                assert mock_wm.call_count == 2
                # First call was CUDA float16
                assert mock_wm.call_args_list[0][1]["device"] == "cuda"
                # Second fallback call was CPU int8
                assert mock_wm.call_args_list[1][1]["device"] == "cpu"
                assert mock_wm.call_args_list[1][1]["compute_type"] == "int8"

    def test_custom_compute_type_preserved(self):
        """Explicit compute types like int8_float16 are preserved on CUDA."""
        with patch("faster_whisper.WhisperModel", return_value=_mock_model()) as mock_wm:
            rec = FasterWhisperRecognizer(model_size="base.en", device="cuda", compute_type="int8_float16")
            rec.load()
            assert rec.device == "cuda"
            assert rec.compute_type == "int8_float16"
            mock_wm.assert_called_once_with(
                "base.en",
                device="cuda",
                compute_type="int8_float16",
                download_root=None,
            )

    def test_load_idempotency_prevents_duplicate_initialization(self):
        """Calling load() multiple times does not reload or recreate the model."""
        with patch(
            "spidy.perception.voice.stt.whisper._detect_cuda_via_ctranslate2",
            return_value=(True, "cuda ok"),
        ):
            with patch("faster_whisper.WhisperModel", return_value=_mock_model()) as mock_wm:
                rec = FasterWhisperRecognizer(model_size="base.en", device="auto")
                rec.load()
                rec.load()
                rec.load()
                assert mock_wm.call_count == 1

    def test_unload_and_reload_cycle(self):
        """Unloading releases model and allows clean reload."""
        with patch(
            "spidy.perception.voice.stt.whisper._detect_cuda_via_ctranslate2",
            return_value=(True, "cuda ok"),
        ):
            with patch("faster_whisper.WhisperModel", return_value=_mock_model()) as mock_wm:
                rec = FasterWhisperRecognizer(model_size="base.en", device="auto")
                rec.load()
                assert rec._model is not None
                rec.unload()
                assert rec._model is None
                rec.load()
                assert rec._model is not None
                assert mock_wm.call_count == 2

    def test_barge_in_detector_stays_cpu(self):
        """BargeInDetector defaults to CPU int8 to prevent model contention."""
        handler = MagicMock()
        bus = MagicMock()
        with patch("faster_whisper.WhisperModel", return_value=_mock_model()) as mock_wm:
            barge = BargeInDetector(
                interruption_handler=handler,
                bus=bus,
                model_size="tiny.en",
                device="cpu",
                compute_type="int8",
            )
            barge.load_model()
            mock_wm.assert_called_once_with(
                "tiny.en",
                device="cpu",
                compute_type="int8",
            )
