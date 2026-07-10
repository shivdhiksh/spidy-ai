"""
FasterWhisperRecognizer — Speech-to-Text Engine
================================================
Uses faster-whisper (CTranslate2 backend) for high-accuracy,
low-latency speech recognition.

Why faster-whisper?
-------------------
- 4x faster than openai-whisper on the same hardware
- GPU: float16 inference on CUDA
- CPU: int8 quantised inference (fast enough for real-time)
- VAD filter: skips silent segments automatically
- Fully offline — no API calls

Performance targets:
  GPU (RTX 3050): base.en transcription < 300ms for 5-second utterance
  CPU (fallback): base.en transcription < 1200ms for 5-second utterance

Model sizes:
  tiny.en   ~39M params  — fastest, less accurate
  base.en   ~74M params  — good balance (default)
  small.en  ~244M params — better accuracy, 2x slower
  medium.en ~769M params — best, needs more VRAM
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np

from spidy.logging.logger import get_logger
from spidy.perception.voice.stt.base import SpeechRecognizer, TranscriptResult

if TYPE_CHECKING:
    from faster_whisper import WhisperModel as FWModel

log = get_logger(__name__)

_SAMPLE_RATE = 16000


class FasterWhisperRecognizer(SpeechRecognizer):
    """
    faster-whisper powered speech recogniser.

    Parameters
    ----------
    model_size:
        Whisper model variant. One of: tiny, tiny.en, base, base.en,
        small, small.en, medium, medium.en, large-v3.
        Default: base.en (English-only, good balance of speed/accuracy).
    device:
        Compute device. "cuda" | "cpu" | "auto".
        "auto" resolves to "cuda" if available, else "cpu".
    compute_type:
        Quantisation type. "float16" (GPU) | "int8" (CPU) | "auto".
        "auto" picks based on device.
    language:
        Language code, e.g. "en". None enables auto-detection (slower).
    vad_filter:
        Apply VAD to skip silent audio segments (reduces hallucinations).
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "auto",
        compute_type: str = "auto",
        language: str | None = "en",
        vad_filter: bool = True,
        vad_threshold: float = 0.5,
        model_dir: str | None = None,
    ) -> None:
        self._model_size = model_size
        self._requested_device = device
        self._requested_compute_type = compute_type
        self._language = language
        self._vad_filter = vad_filter
        self._vad_threshold = vad_threshold
        self._model_dir = model_dir
        self._model: "FWModel | None" = None
        self._resolved_device: str = "cpu"
        self._resolved_compute_type: str = "int8"

    # ── SpeechRecognizer interface ────────────────────────────────────────

    def load(self) -> None:
        """
        Load the Whisper model into memory.

        Resolves device and compute_type from "auto" based on CUDA
        availability. This is a blocking call — run at startup only.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install it with: pip install faster-whisper"
            ) from exc

        # Resolve device
        if self._requested_device == "auto":
            try:
                import torch
                self._resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._resolved_device = "cpu"
        else:
            self._resolved_device = self._requested_device

        # Resolve compute type
        if self._requested_compute_type == "auto":
            self._resolved_compute_type = (
                "float16" if self._resolved_device == "cuda" else "int8"
            )
        else:
            self._resolved_compute_type = self._requested_compute_type

        log.info(
            "Loading faster-whisper '{model}' | device={dev} | compute={ct}",
            model=self._model_size,
            dev=self._resolved_device,
            ct=self._resolved_compute_type,
        )

        try:
            self._model = WhisperModel(
                self._model_size,
                device=self._resolved_device,
                compute_type=self._resolved_compute_type,
                download_root=self._model_dir,
            )
            log.info("faster-whisper model loaded successfully.")
        except Exception as exc:
            # Fallback: if CUDA load fails, try CPU
            if self._resolved_device == "cuda":
                log.warning(
                    "CUDA model load failed ({exc}). Falling back to CPU.",
                    exc=exc,
                )
                self._resolved_device = "cpu"
                self._resolved_compute_type = "int8"
                self._model = WhisperModel(
                    self._model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root=self._model_dir,
                )
                log.info("faster-whisper loaded on CPU (fallback).")
            else:
                raise

    def unload(self) -> None:
        """Release model resources."""
        if self._model is not None:
            del self._model
            self._model = None
            log.debug("faster-whisper model unloaded.")

    async def transcribe(self, audio_data: np.ndarray) -> TranscriptResult:
        """
        Transcribe audio asynchronously.

        Runs faster-whisper inference in a thread pool to avoid blocking
        the asyncio event loop.

        Parameters
        ----------
        audio_data:
            Float32 numpy array at 16kHz mono.

        Returns
        -------
        TranscriptResult
            The transcription. Returns empty result on any error.
        """
        if self._model is None:
            log.error("transcribe() called before load(). Call load() first.")
            return TranscriptResult(text="", is_empty=True)

        try:
            result = await asyncio.to_thread(
                self._transcribe_sync, audio_data
            )
            return result
        except Exception as exc:  # noqa: BLE001
            log.error("Transcription failed: {exc}", exc=exc)
            return TranscriptResult(text="", is_empty=True)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _transcribe_sync(self, audio_data: np.ndarray) -> TranscriptResult:
        """
        Synchronous transcription — runs in thread pool via asyncio.to_thread.
        """
        duration = len(audio_data) / _SAMPLE_RATE

        # faster-whisper transcribe() returns (segments_generator, info)
        segments, info = self._model.transcribe(
            audio_data,
            language=self._language,
            vad_filter=self._vad_filter,
            vad_parameters={"threshold": self._vad_threshold},
            beam_size=5,
            best_of=5,
            temperature=0.0,    # greedy — more consistent
            condition_on_previous_text=False,
        )

        # Consume generator and join segments
        text_parts: list[str] = []
        for segment in segments:
            text_parts.append(segment.text)

        full_text = " ".join(text_parts).strip()
        detected_lang = getattr(info, "language", self._language or "en")

        if not full_text:
            log.debug("Transcription result: <empty>")
            return TranscriptResult(
                text="",
                language=detected_lang,
                duration_seconds=duration,
                is_empty=True,
            )

        log.debug(
            "Transcribed ({dur:.1f}s): '{text}'",
            dur=duration,
            text=full_text[:80] + ("..." if len(full_text) > 80 else ""),
        )

        return TranscriptResult(
            text=full_text,
            confidence=1.0,         # faster-whisper doesn't expose per-result confidence
            language=detected_lang,
            duration_seconds=duration,
            is_empty=False,
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def device(self) -> str:
        return self._resolved_device

    @property
    def model_name(self) -> str:
        return f"whisper-{self._model_size}"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
