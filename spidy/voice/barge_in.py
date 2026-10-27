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

import difflib
import re
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Callable

import numpy as np

from spidy.logging.logger import get_logger
from spidy.voice.echo_canceller import EchoCanceller, MicClock, PlaybackReference
from spidy.voice.interruption import InterruptionHandler, InterruptCommand
from spidy.voice.events import VoiceInterruptEvent

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus
    from spidy.voice.streaming_tts import StreamingTTSWrapper

log = get_logger(__name__)

_SAMPLE_RATE = 16000

# Speech-over-Spidy detection (interrupt by simply talking, no stop phrase needed).
_HISTORY_CHUNKS = 25            # ~2 s of per-chunk levels: Spidy's own echo level while speaking
_MIN_HISTORY_CHUNKS = 6         # ~0.5 s of echo history needed before anything can trigger
_HOT_CHUNKS = 3                 # consecutive loud chunks (~0.25 s) that open a candidate
_PRE_ROLL_CHUNKS = 4            # chunks kept from just before the onset (the first syllables)
_FIRST_CHECK_SECONDS = 0.7      # transcribe once this much candidate audio is buffered
_RECHECK_SECONDS = 0.5          # then re-check every time this much more arrives
_GIVE_UP_SECONDS = 3.2          # a candidate that is still unconvincing by now is dropped
_COOLDOWN_SECONDS = 1.0         # after a rejected candidate, ignore the mic for a moment
_ECHO_OVERLAP = 0.75            # heard words that match what Spidy is saying => it is her own echo
_UNCONVERGED_RATIO = 2.6        # trigger ratio while the echo canceller has not learnt the room yet

_STOPWORDS = frozenset(
    "a an the and or but of to in on at for with is are was were be been it its this that these those "
    "i you he she we they me my your our their do does did so just very can could would should will "
    "what how why when where who which there here as by from up out about into than then".split()
)
# What Whisper tends to invent from room noise or speaker bleed.
_FILLER = frozenset(
    {"you", "thank", "thanks", "bye", "okay", "ok", "uh", "um", "hmm", "mm", "huh", "yeah", "the", "so",
     "see", "later", "very", "much", "next", "time", "watching", "for", "subtitles", "good", "night"}
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def is_echo_of(heard: str, spoken: str) -> bool:
    """
    True if ``heard`` is (mostly) Spidy's own voice coming back through the microphone.

    Each heard content word is matched against the words Spidy is saying (fuzzily, since tiny.en
    garbles speaker bleed). A short real question that shares one common word with the answer
    ("what is the weather" while Spidy talks about weather) is not treated as echo.
    """
    h = _words(heard)
    s = set(_words(spoken))
    if not h or not s:
        return False
    content = [w for w in h if w not in _STOPWORDS] or h
    vocab = list(s)
    hits = sum(
        1 for w in content
        if w in s or difflib.get_close_matches(w, vocab, n=1, cutoff=0.78)
    )
    overlap = hits / len(content)
    return overlap >= _ECHO_OVERLAP and (len(content) >= 2 or len(h) <= 2)


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
        interrupt_on_speech: bool = True,
        trigger_ratio: float = 1.5,
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
        self._interrupt_on_speech = interrupt_on_speech
        self._trigger_ratio = max(1.1, float(trigger_ratio))

        # The tiny.en faster-whisper model instance (loaded lazily at first start())
        self._model = None
        self._model_loaded = False
        self._model_lock = threading.Lock()
        self._loader: threading.Thread | None = None

        # Speech-over-Spidy candidate audio. Protected by _buf_lock — written from the audio
        # thread, read from the worker thread and (once) from the asyncio loop at hand-off.
        self._buf_lock = threading.Lock()
        self._ring: deque[np.ndarray] = deque(maxlen=_PRE_ROLL_CHUNKS)   # newest chunks, always
        self._cand: list[np.ndarray] = []      # audio since a possible interruption began
        self._cand_samples = 0
        self._cand_active = False
        self._triggered = False                # an interruption fired; _cand now belongs to the engine
        self._handed_off = False               # the engine took _cand; late chunks go to _late_sink
        self._late_sink: Callable[[np.ndarray], None] | None = None
        self._interrupt_text = ""
        # Audio-thread-only detection state
        self._reference: PlaybackReference | None = None
        self._aec: EchoCanceller | None = None     # kept between utterances: the room does not change
        self._clock = MicClock()                   # exact end time of every mic chunk (audio clock)
        # The capture engine lowers the mic gain while Spidy speaks (her echo clips otherwise).
        # ratio = barge-in gain / normal gain; chunks are scaled back by 1/ratio for the recorder.
        self._gain_ratio = 1.0
        self._levels: deque[float] = deque(maxlen=_HISTORY_CHUNKS)
        self._hot_run = 0
        self._cooldown_until = 0.0
        self._spoken_text = ""

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

    def prewarm(self) -> None:
        """Load the model in the background if it is not loaded (wake word is sounding)."""
        if self._model_loaded or not self._enabled:
            return
        t = self._loader
        if t is not None and t.is_alive():
            return
        self._loader = threading.Thread(target=self.load_model, name="spidy-barge-in-prewarm", daemon=True)
        self._loader.start()

    def unload_model(self) -> None:
        """Free the model (idle). Never while an utterance is being listened to."""
        with self._model_lock:
            if self._active or not self._model_loaded:
                return
            self._model = None
            self._model_loaded = False
        import gc
        gc.collect()
        log.info("[STT BARGE-IN] model unloaded (idle).")

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

    def set_handoff_sink(self, sink: Callable[[np.ndarray], None] | None) -> None:
        """Where a chunk that arrives after :meth:`consume_interrupt` must go (the engine's buffer)."""
        self._late_sink = sink

    def start(self, tts: "StreamingTTSWrapper", spoken_text: str = "") -> None:
        """
        Activate barge-in detection.

        Called from asyncio event loop at the start of TTS playback.
        Idempotent — if already active, kills old worker before starting new.

        Parameters
        ----------
        tts:
            The active StreamingTTSWrapper; its .stop() will be called on
            detection.
        spoken_text:
            What Spidy is about to say. Used to tell the user's voice from Spidy's own echo.
        """
        if not self._enabled or not self._model_loaded:
            return

        # Kill any lingering worker from a previous TTS
        self._stop_worker()

        self._streaming_tts = tts
        self._spoken_text = spoken_text
        self._reference = getattr(tts, "reference", None)
        if self._reference is not None and not isinstance(self._reference, PlaybackReference):
            self._reference = None                  # not a real reference (e.g. a test double)
        if self._aec is not None:
            self._aec.reset_history()
        self._stop_event.clear()

        # Clear stale audio and detection state
        with self._buf_lock:
            self._ring.clear()
            self._cand = []
            self._cand_samples = 0
            self._cand_active = False
            self._triggered = False
            self._handed_off = False
            self._interrupt_text = ""
        self._levels.clear()
        self._hot_run = 0
        self._cooldown_until = 0.0
        self._active = True

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

        Spidy's own voice reaches the mic through the speakers, so the level of the last ~2 s is
        taken as the "echo level". A few consecutive chunks well above it open a *candidate*: the
        worker transcribes it and, if it is not Spidy's own words, the user is interrupting.
        """
        now = time.monotonic()
        t_end = self._clock.tick(now, len(chunk))
        sink = None
        with self._buf_lock:
            handed_off = self._handed_off
            if handed_off:
                sink = self._late_sink              # the engine owns the audio now
            elif not self._active or self._stop_event.is_set():
                return
        if handed_off:
            if sink is not None:
                sink(self._normal_level(chunk))
            return

        # Take Spidy's own voice out of the microphone signal, if we know what she is playing.
        raw = chunk
        chunk = self._cancel_echo(raw, t_end)

        with self._buf_lock:
            if self._triggered or self._cand_active:
                self._cand.append(chunk)
                self._cand_samples += len(chunk)
                return
            self._ring.append(chunk)

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        hot = False
        if len(self._levels) >= _MIN_HISTORY_CHUNKS and time.monotonic() >= self._cooldown_until:
            echo_level = float(np.percentile(self._levels, 75))
            # Until the canceller has learnt the room the "residual" is mostly raw echo, whose peaks
            # reach 3x its own level: ask for a much louder voice.
            converged = self._aec is not None and self._aec.converged
            ratio = self._trigger_ratio if converged else max(self._trigger_ratio, _UNCONVERGED_RATIO)
            hot = rms >= max(self._min_rms * 2.0 * self._gain_ratio, echo_level * ratio)
        self._levels.append(rms)

        self._hot_run = self._hot_run + 1 if hot else 0
        if self._hot_run >= _HOT_CHUNKS:
            self._hot_run = 0
            with self._buf_lock:
                self._cand = list(self._ring)       # includes the loud chunks and a bit before
                self._cand_samples = sum(len(c) for c in self._cand)
                self._cand_active = True

    def set_gain_ratio(self, ratio: float) -> None:
        """barge-in mic gain / normal mic gain (see AudioCaptureEngine.barge_in_gain_ratio)."""
        self._gain_ratio = max(1e-3, float(ratio))

    def _normal_level(self, chunk: np.ndarray) -> np.ndarray:
        return chunk if self._gain_ratio == 1.0 else np.clip(chunk / self._gain_ratio, -1.0, 1.0)

    def attach_reference(self, reference: PlaybackReference | None) -> None:
        """Tell the detector what Spidy plays, so it can learn the echo path from every playback."""
        self._reference = reference if isinstance(reference, PlaybackReference) else None

    def train_chunk(self, chunk: np.ndarray) -> None:
        """Audio thread: a chunk heard outside BARGE_IN mode. While Spidy plays, adapt the canceller."""
        t_end = self._clock.tick(time.monotonic(), len(chunk))     # every chunk, so the clock stays exact
        ref = self._reference
        if ref is None or not self._interrupt_on_speech or not ref.playing:
            return
        self._cancel_echo(chunk, t_end)

    def _cancel_echo(self, chunk: np.ndarray, t_end: float) -> np.ndarray:
        """The microphone chunk minus the echo of what Spidy is playing (the chunk itself if unknown)."""
        ref = self._reference
        if ref is None:
            return chunk
        if self._aec is None or self._aec.block != len(chunk):
            self._aec = EchoCanceller(block=len(chunk))
        try:
            return self._aec.process(chunk, ref.block(t_end, len(chunk)))
        except Exception as exc:  # noqa: BLE001 -- never lose the mic over the canceller
            log.debug("BargeInDetector: echo cancellation failed: {exc}", exc=exc)
            return chunk

    # ── Worker thread ─────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """
        Background thread: transcribe candidate audio and decide whether the user is interrupting.

        Runs until stop() is called or an interruption fires.
        """
        log.debug("BargeInDetector worker: started.")
        checked = 0
        while not self._stop_event.is_set():
            with self._buf_lock:
                active, samples = self._cand_active, self._cand_samples
            if not active:
                checked = 0
                time.sleep(0.02)  # 20ms poll — low CPU overhead
                continue

            need = _FIRST_CHECK_SECONDS if checked == 0 else checked / _SAMPLE_RATE + _RECHECK_SECONDS
            if samples < need * _SAMPLE_RATE:
                time.sleep(0.02)
                continue

            with self._buf_lock:
                audio = np.concatenate(self._cand)[-int(3 * _SAMPLE_RATE):]
            checked = samples

            text = self._transcribe(self._normal_level(audio))
            verdict = self._judge(text, samples)
            log.debug("BargeInDetector: heard '{t}' -> {v}", t=text[:80], v=verdict)

            if verdict == "stop":
                log.info("BargeInDetector: STOP detected! Interrupting TTS. (phrase='{t}')", t=text[:60])
                self._trigger_stop()
                return  # worker done — stop() will be called by the finally block
            if verdict == "interrupt":
                log.info("BargeInDetector: user is speaking over Spidy -- interrupting. ('{t}')", t=text[:80])
                self._trigger_interrupt(text)
                return
            if verdict == "reject" or samples >= _GIVE_UP_SECONDS * _SAMPLE_RATE:
                self._drop_candidate()
                checked = 0

        log.debug("BargeInDetector worker: exiting cleanly.")

    def _judge(self, text: str, samples: int) -> str:
        """Classify what a candidate transcribed to: stop | interrupt | reject | wait."""
        if self._handler.detect(text) == InterruptCommand.STOP:
            return "stop"
        if not self._interrupt_on_speech:
            return "reject"
        words = _words(text)
        long_enough = samples >= 1.6 * _SAMPLE_RATE
        if not words:
            return "reject" if long_enough else "wait"
        if all(w in _FILLER for w in words):
            return "reject" if long_enough else "wait"
        if self._spoken_text and is_echo_of(text, self._spoken_text):
            return "reject"
        if len(words) >= 2 or (len(words[0]) >= 5 and samples >= 1.2 * _SAMPLE_RATE):
            return "interrupt"
        return "wait"

    def _drop_candidate(self) -> None:
        """Forget an unconvincing candidate and ignore the mic briefly (echo comes in bursts)."""
        with self._buf_lock:
            self._cand = []
            self._cand_samples = 0
            self._cand_active = False
        self._hot_run = 0
        self._cooldown_until = time.monotonic() + _COOLDOWN_SECONDS

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

    def _trigger_interrupt(self, text: str) -> None:
        """
        The user started talking over Spidy: cut her off now.

        The candidate audio is kept (the audio thread keeps appending to it) so the controller can
        hand it to the recorder via :meth:`consume_interrupt` and nothing the user said is lost.
        """
        with self._buf_lock:
            self._triggered = True
            self._interrupt_text = text
        tts = self._streaming_tts
        if tts is not None:
            tts.stop()
        try:
            self._bus.publish_threadsafe(VoiceInterruptEvent(command="barge_in", utterance=text))
        except Exception as exc:  # noqa: BLE001
            log.debug("BargeInDetector: publish_threadsafe error: {exc}", exc=exc)

    def consume_interrupt(self) -> list[np.ndarray] | None:
        """
        Take the audio captured since the user started speaking (None if nobody interrupted).

        Called once from the asyncio loop after the engine's capture mode has left BARGE_IN. A chunk
        that was already in flight to :meth:`feed_chunk` is forwarded to the hand-off sink.
        """
        with self._buf_lock:
            if not self._triggered:
                return None
            chunks, self._cand = self._cand, []
            self._cand_samples = 0
            self._handed_off = True
        return [self._normal_level(c) for c in chunks]

    @property
    def interrupted(self) -> bool:
        """True once the user has talked over Spidy during the current utterance."""
        return self._triggered

    @property
    def interrupt_text(self) -> str:
        return self._interrupt_text

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
