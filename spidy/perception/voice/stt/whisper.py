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

CUDA Device Detection
---------------------
Device resolution uses ctranslate2 directly (NOT torch.cuda.is_available()),
because torch may be installed as a CPU-only build while ctranslate2 retains
full CUDA support. The helper ``_detect_cuda_via_ctranslate2()`` probes
ctranslate2's own compute-type registry and attempts a lightweight runtime
check. If cublas/cudart DLLs are absent, the probe fails and CPU is used.

Performance targets (RTX 3050):
  GPU float16: base.en transcription < 350ms for 3-second utterance
  CPU int8:    base.en transcription < 2500ms for 3-second utterance

Model sizes:
  tiny.en   ~39M params  — fastest, less accurate
  base.en   ~74M params  — good balance (default)
  small.en  ~244M params — better accuracy, 2x slower
  medium.en ~769M params — best, needs more VRAM
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncGenerator

import numpy as np

from spidy.logging.logger import get_logger
from spidy.perception.voice.stt.base import SpeechRecognizer, TranscriptResult
from spidy.voice.events import VoicePartialTranscriptEvent

if TYPE_CHECKING:
    from faster_whisper import WhisperModel as FWModel

log = get_logger(__name__)

_SAMPLE_RATE = 16000


def _detect_cuda_via_ctranslate2() -> tuple[bool, str]:
    """
    Probe CUDA availability via ctranslate2 — independent of torch.

    ctranslate2 bundles its own CUDA detection and does not require
    ``torch.cuda.is_available()`` to return True.  This matters when torch
    is installed as a CPU-only build (``torch==x.y.z+cpu``) while
    ctranslate2 was built with CUDA support.

    Returns
    -------
    (cuda_ok, reason)
        cuda_ok:  True if CUDA is usable by ctranslate2.
        reason:   Human-readable explanation for the startup log.
    """
    try:
        from spidy.core.device import ensure_cuda_dlls
        ensure_cuda_dlls()
    except Exception:
        pass

    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() <= 0:
            return False, "no CUDA GPU device detected by ctranslate2"
        supported = ctranslate2.get_supported_compute_types("cuda")
        if "float16" not in supported:
            return False, "ctranslate2 CUDA build does not support float16"
    except Exception as exc:
        return False, f"ctranslate2 CUDA probe failed: {exc}"

    # Second check: try to verify that the required CUDA runtime DLLs are
    # accessible.  We do this by attempting to import ctypes and load
    # nvcuda.dll (the NVIDIA CUDA driver interface — always present when a
    # driver is installed).  The real runtime DLL (cublas) is validated
    # lazily at model load time with a graceful CPU fallback.
    try:
        import ctypes
        ctypes.CDLL("nvcuda.dll")
    except OSError:
        return False, "NVIDIA driver (nvcuda.dll) not accessible"

    return True, "ctranslate2 CUDA + NVIDIA driver available"


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
        initial_prompt: str | None = None,
    ) -> None:
        self._model_size = model_size
        self._requested_device = device
        self._requested_compute_type = compute_type
        self._language = language
        self._vad_filter = vad_filter
        self._vad_threshold = vad_threshold
        self._model_dir = model_dir
        # Issue 3 fix: initial_prompt primes Whisper with command vocabulary.
        # Without this, Whisper's language model picks the most probable words
        # from its full vocabulary — 'Notepad' becomes 'not bad', 'Edge' becomes
        # 'the edge of my body'. A prompt biases the decoder toward these terms.
        # Default covers the most common Windows app/command names.
        self._initial_prompt: str = initial_prompt or (
            "Open Notepad. Open Chrome. Open Edge. Open File Explorer. "
            "Open Settings. Open Calculator. Open Paint. Open Teams. "
            "Open Spotify. Open Discord. Close this. Minimize. Maximize. "
            "Search for. Play music. Volume up. Volume down."
        )
        self._model: "FWModel | None" = None
        self._resolved_device: str = "cpu"
        self._resolved_compute_type: str = "int8"

    # ── SpeechRecognizer interface ────────────────────────────────────────

    def load(self) -> None:
        """
        Load the Whisper model into memory.

        Resolves device and compute_type from "auto" using ctranslate2-native
        CUDA detection (independent of torch).  This is a blocking call —
        run at startup only via asyncio.to_thread or the voice factory.

        Device resolution order
        -----------------------
        "cuda"  (explicit)  → always try CUDA; CPU fallback on failure.
        "cpu"   (explicit)  → always CPU, no CUDA attempt.
        "auto"              → probe ctranslate2; cuda if available, else cpu.

        Startup diagnostic log
        ----------------------
        Always emits a clear one-line summary:
          [STT] model=base.en | device=cuda | compute_type=float16
        or on CPU fallback:
          [STT] model=base.en | device=cpu | compute_type=int8 | reason=<why>
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install it with: pip install faster-whisper"
            ) from exc

        # Idempotency guard — model already loaded, nothing to do.
        if self._model is not None:
            return

        # ── Resolve device ────────────────────────────────────────────────
        cuda_reason = ""
        if self._requested_device == "auto":
            cuda_ok, cuda_reason = _detect_cuda_via_ctranslate2()
            self._resolved_device = "cuda" if cuda_ok else "cpu"
        else:
            self._resolved_device = self._requested_device

        # ── Resolve compute type ──────────────────────────────────────────
        if self._requested_compute_type in ("auto", "int8"):
            # Always use int8 on CPU.  On CUDA, prefer float16 (best
            # accuracy/speed on Ampere+), but respect explicit overrides.
            if self._requested_compute_type == "auto":
                self._resolved_compute_type = (
                    "float16" if self._resolved_device == "cuda" else "int8"
                )
            else:
                # explicit "int8" requested — honour it on both CPU and GPU
                self._resolved_compute_type = "int8"
        else:
            self._resolved_compute_type = self._requested_compute_type

        # ── Startup diagnostic log ────────────────────────────────────────
        if self._resolved_device == "cuda":
            log.info(
                "[STT] model={model} | device=cuda | compute_type={ct}",
                model=self._model_size,
                ct=self._resolved_compute_type,
            )
        else:
            reason_suffix = f" | reason={cuda_reason}" if cuda_reason else ""
            log.info(
                "[STT] model={model} | device=cpu | compute_type={ct}{reason}",
                model=self._model_size,
                ct=self._resolved_compute_type,
                reason=reason_suffix,
            )

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
            log.info(
                "faster-whisper '{model}' loaded successfully on {dev}.",
                model=self._model_size,
                dev=self._resolved_device,
            )
        except Exception as exc:
            # Fallback: if CUDA inference fails (e.g. missing cublas DLL),
            # gracefully reload on CPU so Spidy always starts correctly.
            if self._resolved_device == "cuda":
                log.warning(
                    "[STT] CUDA load failed — falling back to CPU. "
                    "Install nvidia-cublas-cu12 to enable GPU Whisper. "
                    "Error: {exc}",
                    exc=exc,
                )
                self._resolved_device = "cpu"
                self._resolved_compute_type = "int8"
                log.info(
                    "[STT] model={model} | device=cpu | compute_type=int8 "
                    "| reason=CUDA inference failed (cublas unavailable)",
                    model=self._model_size,
                )
                self._model = WhisperModel(
                    self._model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root=self._model_dir,
                )
                log.info(
                    "faster-whisper '{model}' loaded on CPU (fallback).",
                    model=self._model_size,
                )
            else:
                raise

    def unload(self) -> None:
        """Release model resources."""
        if self._model is not None:
            del self._model
            self._model = None
            log.debug("faster-whisper model unloaded.")

    # ── Diagnostic properties ──────────────────────────────────────────────

    @property
    def device(self) -> str:
        """
        The device that was used to load the model.

        Returns the resolved device string (``"cpu"`` or ``"cuda:0"``/``"cuda"``)
        set by the last successful ``load()`` call.  Returns ``"cpu"`` before
        ``load()`` is called (the default fallback).
        """
        return self._resolved_device

    @property
    def compute_type(self) -> str:
        """
        The compute type that was used to load the model.

        Returns the resolved compute type string (``"int8"`` or ``"float16"``)
        set by the last successful ``load()`` call.  Returns ``"int8"`` before
        ``load()`` is called.
        """
        return self._resolved_compute_type

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

    async def transcribe_streaming(
        self,
        audio_data: np.ndarray,
        bus=None,
        session_id: str = "",
    ) -> AsyncGenerator[TranscriptResult, None]:
        """
        Stream partial transcription results using faster-whisper's
        segment generator (Milestone 14).

        Yields each segment as it completes, then a final merged result.
        Publishes VoicePartialTranscriptEvent to the bus for each segment
        if ``bus`` is provided.

        Parameters
        ----------
        audio_data:
            Float32 audio at 16kHz mono.
        bus:
            Optional EventBus for publishing partial events.
        session_id:
            Session ID passed into partial events.
        """
        if self._model is None:
            log.error("transcribe_streaming() called before load().")
            yield TranscriptResult(text="", is_empty=True)
            return

        duration = len(audio_data) / _SAMPLE_RATE
        accumulated = []

        try:
            segments, info = await asyncio.to_thread(
                lambda: self._model.transcribe(
                    audio_data,
                    language=self._language,
                    vad_filter=self._vad_filter,
                    vad_parameters={"threshold": self._vad_threshold},
                    beam_size=5,
                    best_of=5,
                    temperature=0.0,
                    condition_on_previous_text=False,
                )
            )

            # Iterate segments from the generator
            def _next_segment():
                try:
                    return next(iter([]))
                except StopIteration:
                    return None

            # Consume the segment generator in a thread-safe way
            segment_list = await asyncio.to_thread(list, segments)
            detected_lang = getattr(info, "language", self._language or "en")

            for segment in segment_list:
                seg_text = segment.text.strip()
                if not seg_text:
                    continue

                accumulated.append(seg_text)
                partial_text = " ".join(accumulated)

                partial_result = TranscriptResult(
                    text=partial_text,
                    language=detected_lang,
                    duration_seconds=duration,
                    is_partial=True,
                )

                # Publish partial event to bus if available
                if bus is not None:
                    try:
                        await bus.publish(VoicePartialTranscriptEvent(
                            text=partial_text,
                            is_final=False,
                        ))
                    except Exception:  # noqa: BLE001
                        pass

                yield partial_result

            # Yield final result
            full_text = " ".join(accumulated).strip()
            final_result = TranscriptResult(
                text=full_text,
                confidence=1.0,
                language=detected_lang,
                duration_seconds=duration,
                is_empty=not bool(full_text),
                is_partial=False,
            )

            if bus is not None:
                try:
                    await bus.publish(VoicePartialTranscriptEvent(
                        text=full_text,
                        is_final=True,
                    ))
                except Exception:  # noqa: BLE001
                    pass

            yield final_result

        except Exception as exc:  # noqa: BLE001
            log.error("Streaming transcription failed: {exc}", exc=exc)
            yield TranscriptResult(text="", is_empty=True)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _transcribe_sync(self, audio_data: np.ndarray) -> TranscriptResult:
        """
        Synchronous transcription — runs in thread pool via asyncio.to_thread.
        """
        from spidy.voice.transcript_guard import SegmentQualitySignals

        duration = len(audio_data) / _SAMPLE_RATE

        # faster-whisper transcribe() returns (segments_generator, info)
        segments, info = self._model.transcribe(
            audio_data,
            language=self._language,
            vad_filter=self._vad_filter,
            # Fix 3C: VAD threshold lowered 0.5 → 0.3 (less aggressive speech gate).
            # At 0.5, Silero frequently truncates short commands like 'Open Notepad'
            # because the trailing phonemes of a 2-word command fall below the gate.
            # min_speech_duration_ms=250 prevents Silero from dropping segments
            # shorter than 250ms (a single syllable) which Whisper would then skip.
            vad_parameters={
                "threshold": 0.3,
                "min_speech_duration_ms": 250,
            },
            beam_size=5,
            best_of=5,
            # Fix 3D: temperature fallback chain instead of pure greedy (0.0).
            # With temperature=0.0, Whisper commits to the highest-probability token
            # with no fallback. For short/noisy audio this locks in wrong words
            # ('not bad' instead of 'Notepad'). Passing a list enables Whisper's
            # internal fallback: if temperature=0.0 produces a result with low
            # confidence (high compression ratio or high no_speech_prob), it retries
            # at 0.2, then 0.4. This matches OpenAI's production API behaviour.
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,
            initial_prompt=self._initial_prompt,
            no_speech_threshold=0.6,
        )

        # Consume generator, join segments, and collect quality signals
        text_parts: list[str] = []
        quality_signals: list[SegmentQualitySignals] = []
        for segment in segments:
            text_parts.append(segment.text)
            # Collect segment-level quality metadata for TranscriptQualityGuard.
            # These attributes are always present on faster-whisper segment objects.
            quality_signals.append(SegmentQualitySignals(
                no_speech_prob=getattr(segment, "no_speech_prob", 0.0),
                avg_logprob=getattr(segment, "avg_logprob", 0.0),
                compression_ratio=getattr(segment, "compression_ratio", 1.0),
            ))

        full_text = " ".join(text_parts).strip()
        detected_lang = getattr(info, "language", self._language or "en")

        if not full_text:
            log.debug("Transcription result: <empty>")
            return TranscriptResult(
                text="",
                language=detected_lang,
                duration_seconds=duration,
                is_empty=True,
                segment_signals=quality_signals or None,
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
            segment_signals=quality_signals or None,
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
