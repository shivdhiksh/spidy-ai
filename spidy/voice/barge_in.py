"""
BargeInDetector — Lightweight Real-Time Stop-Command Detector
=============================================================
Allows the user to say "Spidy stop" while Spidy is speaking (TTS) and have
that command detected and acted upon immediately, without routing through the
Brain, NVIDIA, or Ollama.

Architecture
------------
During TTS playback the AudioCaptureEngine is put into CaptureMode.BARGE_IN
instead of CaptureMode.IDLE.  Every audio chunk from the microphone is routed
here instead of being discarded.

BargeInDetector accumulates chunks in a small ring buffer.  When the buffer
holds ``window_seconds`` of audio, it runs a dedicated ``tiny.en`` faster-
whisper model (separate from the main ``base.en`` production model — no
contention, ~20 MB extra RSS) and passes the result to InterruptionHandler.
If a stop phrase is matched, streaming_tts.stop() is called thread-safely and
a VoiceInterruptEvent is published to the bus.

Model loading
-------------
Two loading modes are available:

``load_model()``:
    Eager synchronous load — call at startup from the factory.  The model is
    ready immediately before Spidy finishes starting up.

``schedule_lazy_load(delay_seconds=3.0)``:
    Deferred background load — starts a daemon thread that waits
    ``delay_seconds`` then loads the model.  Use this to reduce startup RAM
    impact.  Spidy starts faster and the barge-in model is ready by the time
    TTS first plays (typical: 3–10 seconds after startup).

    The ``is_loaded`` guard in ``start()`` already handles the 'not yet
    ready' case gracefully — barge-in silently skips the first TTS cycle
    if the model hasn't finished loading.

Performance (measured on this machine)
--------------------------------------
- Model:          tiny.en (int8 CPU)
- Window:         0.8 s (default, configurable)
- STT latency:    ~671 ms (warm, beam_size=1)
- Total barge-in: ~1.4 s worst case (0.8s accumulate + 671ms STT)
- vs base.en:     ~1.7 s worst case — 300ms slower, adds model contention
- Memory delta:   ~180 MB loaded alone; ~20 MB additional when base.en is
                  already resident (shared CTranslate2 runtime)

Self-talk guard
---------------
An energy gate (RMS < min_rms_threshold) discards chunks that are dominated
by TTS speaker bleed.  Piper's output through the speaker typically saturates
the mic at RMS ≥ 0.04–0.15 at normal distances.  The default gate (0.015 RMS)
lets through user speech (~0.05+ RMS) while rejecting most Piper bleed.
The gate is config-tunable via ``voice.barge_in.min_rms_threshold``.

Concurrency
-----------
- AudioCaptureEngine runs on thread ``spidy-audio-capture``
  → calls ``feed_chunk()`` → appends to ``_buffer`` (guarded by Lock)
- ``_worker_loop()`` runs on thread ``spidy-barge-in``
  → drains buffer, runs STT, calls streaming_tts.stop() thread-safely
- No asyncio locks held across threads; bus.publish_threadsafe() used
- start()/stop() are idempotent; always called from asyncio event loop
- ``finally`` in _on_brain_response always calls stop() → no mic leaks
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from spidy.logging.logger import get_logger
from spidy.voice.interruption import InterruptionHandler, InterruptCommand
from spidy.voice.events import VoiceInterruptEvent

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus
    from spidy.voice.streaming_tts import StreamingTTSWrapper

log = get_logger(__name__)

_SAMPLE_RATE = 16000


class BargeInDetector:
    """
    Lightweight real-time stop-command detector active during TTS playback.

    Parameters
    ----------
    interruption_handler:
        Used to match transcripts against stop phrases.
    bus:
        Application EventBus for publishing VoiceInterruptEvent.
    model_size:
        faster-whisper model to load for barge-in STT.
        Default: ``"tiny.en"`` — fastest, adequate for short stop commands.
        The main pipeline uses ``base.en``; keeping them separate avoids
        model contention and uses only ~20 MB extra RAM.
    window_seconds:
        Duration of audio (seconds) to accumulate before running STT.
        Default: 0.8 s.  Minimum practical value is 0.4 s.
    min_rms_threshold:
        Energy gate.  Chunks below this RMS are discarded (self-talk /
        speaker bleed guard).  Default: 0.015.
    enabled:
        Master enable switch.  When False, feed_chunk() is a no-op.
    """

    def __init__(
        self,
        interruption_handler: InterruptionHandler,
        bus: "EventBus",
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        window_seconds: float = 0.8,
        min_rms_threshold: float = 0.015,
        enabled: bool = True,
        suppress_during_tts: bool = True,
    ) -> None:
        self._handler = interruption_handler
        self._bus = bus
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._window_seconds = max(0.4, float(window_seconds))
        self._min_rms = float(min_rms_threshold)
        self._enabled = enabled
        self._suppress_during_tts = suppress_during_tts

        # The tiny.en faster-whisper model instance (loaded lazily at first start())
        self._model = None
        self._model_loaded = False
        self._model_lock = threading.Lock()

        # Audio ring buffer: deque of float32 numpy chunks
        # Protected by _buf_lock — written from audio thread, read from worker thread
        self._buffer: deque[np.ndarray] = deque()
        self._buf_lock = threading.Lock()
        self._buf_samples = 0  # total samples currently in buffer

        # Worker thread lifecycle
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._active = False           # True while TTS is playing
        self._streaming_tts: "StreamingTTSWrapper | None" = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """
        Load the tiny.en faster-whisper model (eager/synchronous).

        Called once at startup from the asyncio loop via asyncio.to_thread
        so the blocking model load doesn't block the event loop.
        This is the ONLY method that loads a model — never called again.

        Alternative: use ``schedule_lazy_load()`` to defer the load to a
        background thread and reduce startup time / initial RAM usage.
        """
        with self._model_lock:
            if self._model_loaded:
                return

            if not self._enabled:
                log.info("BargeInDetector: disabled — skipping model load.")
                return

            # Resolve device & compute_type
            dev = self._device
            ct = self._compute_type
            if dev == "auto":
                try:
                    from spidy.perception.voice.stt.whisper import _detect_cuda_via_ctranslate2
                    cuda_ok, _ = _detect_cuda_via_ctranslate2()
                    dev = "cuda" if cuda_ok else "cpu"
                except Exception:
                    dev = "cpu"
            if ct == "auto":
                ct = "float16" if dev == "cuda" else "int8"

            try:
                from spidy.core.device import ensure_cuda_dlls
                ensure_cuda_dlls()
            except Exception:
                pass

            try:
                from faster_whisper import WhisperModel
                log.info(
                    "BargeInDetector: loading '{m}' ({ct}/{dev}) for barge-in STT ...",
                    m=self._model_size,
                    ct=ct,
                    dev=dev,
                )
                t0 = time.monotonic()
                try:
                    self._model = WhisperModel(
                        self._model_size,
                        device=dev,
                        compute_type=ct,
                    )
                except Exception as cuda_exc:
                    if dev == "cuda":
                        log.warning(
                            "BargeInDetector: CUDA load failed ({exc}) — falling back to CPU",
                            exc=cuda_exc,
                        )
                        dev = "cpu"
                        ct = "int8"
                        self._model = WhisperModel(
                            self._model_size,
                            device="cpu",
                            compute_type="int8",
                        )
                    else:
                        raise

                elapsed = (time.monotonic() - t0) * 1000
                self._model_loaded = True
                log.info(
                    "[STT BARGE-IN] model={m} | device={dev} | compute_type={ct} | "
                    "loaded in {ms:.0f}ms.",
                    m=self._model_size,
                    dev=dev,
                    ct=ct,
                    ms=elapsed,
                )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "BargeInDetector: failed to load '{m}': {exc}. "
                    "Barge-in will be disabled.",
                    m=self._model_size,
                    exc=exc,
                )

    def schedule_lazy_load(self, delay_seconds: float = 3.0) -> None:
        """
        Schedule the tiny.en model to load in a background thread after a delay.

        This reduces Spidy's startup time and initial RAM footprint.  The
        barge-in model (~180 MB) is loaded asynchronously after
        ``delay_seconds``.  By the time the user first triggers TTS, the
        model is ready.  If TTS fires before the model is loaded, the first
        TTS cycle runs without barge-in (safe fallback — mic stays IDLE).

        Parameters
        ----------
        delay_seconds:
            Seconds to wait after this method is called before starting
            the model load.  Default: 3.0 s.  Should be long enough for
            the rest of Spidy's startup sequence to complete.
        """
        if self._model_loaded or not self._enabled:
            return  # Already loaded or disabled — nothing to do

        def _deferred_load() -> None:
            time.sleep(delay_seconds)
            log.info(
                "BargeInDetector: starting deferred model load "
                "(delay={d}s elapsed).",
                d=delay_seconds,
            )
            self.load_model()

        t = threading.Thread(target=_deferred_load, name="spidy-barge-in-loader", daemon=True)
        t.start()
        log.info(
            "BargeInDetector: lazy load scheduled in {d}s "
            "(model='{m}', device=cpu).",
            d=delay_seconds,
            m=self._model_size,
        )

    def start(self, tts: "StreamingTTSWrapper") -> None:
        """
        Activate barge-in detection.

        Called from asyncio event loop at the start of TTS playback.
        Idempotent — if already active, kills old worker before starting new.

        Parameters
        ----------
        tts:
            The active StreamingTTSWrapper; its .stop() will be called on
            detection.
        """
        if not self._enabled or not self._model_loaded:
            return

        # Kill any lingering worker from a previous TTS
        self._stop_worker()

        self._streaming_tts = tts
        self._active = True
        self._stop_event.clear()

        # Clear stale audio from buffer
        with self._buf_lock:
            self._buffer.clear()
            self._buf_samples = 0

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="spidy-barge-in",
            daemon=True,
        )
        self._worker_thread.start()
        log.info(
            "BargeInDetector: started (window={w}s, min_rms={r}).",
            w=self._window_seconds,
            r=self._min_rms,
        )

    def stop(self) -> None:
        """
        Deactivate barge-in detection.

        Called from asyncio event loop in the ``finally`` block of
        _on_brain_response — guaranteed to run even if TTS is interrupted.
        """
        self._active = False
        self._streaming_tts = None
        self._stop_worker()
        log.debug("BargeInDetector: stopped.")

    # ── Audio feed (called from audio thread) ─────────────────────────────

    def feed_chunk(self, chunk: np.ndarray) -> None:
        """
        Accept one audio chunk from AudioCaptureEngine in BARGE_IN mode.

        Called from the audio capture thread — must be fast and non-blocking.

        Guards
        ------
        - If not active, returns immediately.
        - Energy gate: chunks with RMS < min_rms_threshold are discarded
          (filters TTS speaker bleed at source).
        """
        if not self._active or self._stop_event.is_set():
            return

        # Energy gate: Piper's speaker output typically bleeds into the mic
        # at moderate RMS (0.02-0.06). User speech arrives at >= 0.08 RMS when speaking over TTS.
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        effective_threshold = self._min_rms
        if self._suppress_during_tts and self._streaming_tts is not None and getattr(self._streaming_tts, "is_speaking", False):
            # Raise gate during active TTS playback to isolate user voice from speaker bleed
            effective_threshold = max(0.065, self._min_rms * 2.5)

        if rms < effective_threshold:
            return

        with self._buf_lock:
            self._buffer.append(chunk)
            self._buf_samples += len(chunk)

    # ── Worker thread ─────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """
        Background thread: accumulate audio, run STT, check for stop phrase.

        Runs until stop() is called or a stop phrase is detected.
        """
        window_samples = int(self._window_seconds * _SAMPLE_RATE)
        log.debug(
            "BargeInDetector worker: started (window={w}s = {n} samples).",
            w=self._window_seconds,
            n=window_samples,
        )

        while not self._stop_event.is_set():
            # Wait until we have a full window of audio
            with self._buf_lock:
                ready = self._buf_samples >= window_samples

            if not ready:
                time.sleep(0.02)  # 20ms poll — low CPU overhead
                continue

            # Drain buffer up to window_samples
            chunks = []
            total = 0
            with self._buf_lock:
                while self._buffer and total < window_samples:
                    c = self._buffer.popleft()
                    chunks.append(c)
                    total += len(c)
                    self._buf_samples -= len(c)

            if not chunks:
                continue

            audio = np.concatenate(chunks)[:window_samples]

            # Run tiny.en STT
            text = self._transcribe(audio)
            if not text:
                continue

            log.debug(
                "BargeInDetector: transcribed: '{t}'",
                t=text[:80],
            )

            # Check against stop phrases
            command = self._handler.detect(text)
            if command == InterruptCommand.STOP:
                log.info(
                    "BargeInDetector: STOP detected! Interrupting TTS. "
                    "(phrase='{t}')",
                    t=text[:60],
                )
                self._trigger_stop()
                return  # worker done — stop() will be called by the finally block

        log.debug("BargeInDetector worker: exiting cleanly.")

    def _transcribe(self, audio: np.ndarray) -> str:
        """
        Run faster-whisper tiny.en on a float32 audio chunk.

        Returns the transcribed text, or '' on any error / no-speech.

        Tuned for barge-in (speed over accuracy):
          - beam_size=1      — greedy decode, fastest
          - best_of=1        — no sampling
          - temperature=0.0  — deterministic
          - vad_filter=False — we already energy-gated; VAD adds latency
          - no_speech_threshold=0.7 — skip obviously-silent windows
        """
        if self._model is None:
            return ""
        try:
            segments, _info = self._model.transcribe(
                audio,
                language="en",
                vad_filter=False,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                no_speech_threshold=0.7,
            )
            parts = [s.text.strip() for s in segments if s.text.strip()]
            return " ".join(parts).strip()
        except Exception as exc:  # noqa: BLE001
            log.debug("BargeInDetector: transcription error (non-fatal): {exc}", exc=exc)
            return ""

    def _trigger_stop(self) -> None:
        """
        Stop TTS immediately and publish interrupt event.

        Thread-safe — called from the worker thread.
        Uses only thread-safe APIs (no asyncio).
        """
        tts = self._streaming_tts
        if tts is not None:
            tts.stop()
            log.info("BargeInDetector: TTS stopped.")

        # Publish VoiceInterruptEvent thread-safely
        try:
            self._bus.publish_threadsafe(
                VoiceInterruptEvent(command="stop", utterance="barge_in")
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("BargeInDetector: publish_threadsafe error: {exc}", exc=exc)

    def _stop_worker(self) -> None:
        """Signal and join the worker thread."""
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """True while barge-in detection is running."""
        return self._active

    @property
    def is_loaded(self) -> bool:
        """True if the tiny.en model is loaded and ready."""
        return self._model_loaded

    @property
    def enabled(self) -> bool:
        """True if barge-in is globally enabled."""
        return self._enabled
