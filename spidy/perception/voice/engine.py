"""
VoiceEngine — Voice Pipeline Coordinator
==========================================
The VoiceEngine owns the complete voice interaction loop:

  SLEEPING → wake word detected → LISTENING → speech recorded →
  PROCESSING → transcript published → SPEAKING → playback done → SLEEPING

It coordinates:
- AudioCaptureEngine (continuous microphone stream)
- WakeWordModel (wake detection in audio thread)
- SpeechRecognizer (STT on captured speech)
- TTSEngine (speak responses)
- EventBus (publish events to Brain, UI, etc.)

State Machine
-------------

  ┌─────────────┐    wake_word.detected    ┌─────────────┐
  │  SLEEPING   │ ───────────────────────► │   WAKING    │
  └─────────────┘                          └──────┬──────┘
        ▲                                         │ VAD starts
        │                                         ▼
  ┌─────┴───────┐    voice.transcript      ┌─────────────┐
  │  SPEAKING   │ ◄─────────────────────── │  LISTENING  │
  └─────────────┘    (after response TTS)  └──────┬──────┘
                                                   │ silence detected
                                                   ▼
                                           ┌─────────────┐
                                           │ PROCESSING  │
                                           └─────────────┘

Events Published
----------------
  wake_word.detected    → Brain (start processing), UI (animate)
  voice.listening       → UI (show mic active)
  voice.transcript      → Brain (user spoke this text)
  voice.speaking_start  → UI (show speaking indicator)
  voice.speaking_end    → UI (hide speaking indicator)
  voice.error           → UI (show error), Logger

Events Subscribed
-----------------
  brain.response_ready  → speak() the response text
  system.shutting_down  → stop all audio, clean up

Thread Architecture
-------------------
Audio capture runs in a dedicated background thread via AudioCaptureEngine.
Wake word detection runs inside that thread.
All EventBus publications are threadsafe (publish_threadsafe).
STT and TTS are awaited in the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from spidy.core.event_bus import Event, EventBus
from spidy.logging.logger import get_logger
from spidy.perception.voice.audio_capture import AudioCaptureEngine, CaptureMode
from spidy.perception.voice.stt.base import SpeechRecognizer
from spidy.perception.voice.tts.base import TTSEngine
from spidy.perception.voice.wake_word.base import WakeWordModel
from spidy.perception.voice.wake_word.verifier import WakeVerifier

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Wake score at which the speech models start loading, below the detection threshold so they are
# (nearly) ready by the time the wake word is confirmed.
_PREWARM_SCORE = 0.08
# A detection this confident with *no words at all* heard by Whisper is accepted (see _verify_and_fire).
_EMPTY_ACCEPT_SCORE = 0.85


# ─── Voice Events ─────────────────────────────────────────────────────────────


@dataclass
class WakeWordDetectedEvent(Event):
    """Published when the wake word is confirmed above threshold."""
    topic = "wake_word.detected"
    confidence: float = 0.0
    model_name: str = ""
    wake_phrase: str = ""



@dataclass
class VoiceListeningEvent(Event):
    """Published when Spidy starts recording user speech."""
    topic = "voice.listening"


@dataclass
class VoiceTranscriptEvent(Event):
    """Published when speech is transcribed to text."""
    topic = "voice.transcript"
    text: str = ""
    confidence: float = 1.0
    language: str = "en"
    source: str = "user"          # "user" | "spidy_tts" | "echo" | "unknown"
    session_id: str = ""
    capture_state: str = ""
    tts_active: bool = False
    flush_active: bool = False
    timestamp: float = 0.0
    segment_signals: Any = None


@dataclass
class VoiceSpeakingStartEvent(Event):
    """Published when Spidy begins speaking a response."""
    topic = "voice.speaking_start"
    text: str = ""


@dataclass
class VoiceSpeakingEndEvent(Event):
    """Published when Spidy finishes speaking."""
    topic = "voice.speaking_end"


@dataclass
class VoiceErrorEvent(Event):
    """Published on voice pipeline errors."""
    topic = "voice.error"
    source: str = ""
    message: str = ""


# BrainResponseReadyEvent is defined in brain.events (canonical).
# Import it here so the subscriber type annotation resolves correctly.
# Do NOT redefine it — the duplicate caused field name mismatch (text vs response_text).
from spidy.brain.events import BrainResponseReadyEvent  # noqa: E402


# ─── State Machine ────────────────────────────────────────────────────────────


class VoiceState(Enum):
    SLEEPING = auto()       # Wake word detector active / IDLE
    WAKING = auto()         # Wake word detected, preparing to listen
    LISTENING = auto()      # Recording user speech (LISTENING_FOR_USER)
    PROCESSING = auto()     # Running STT transcription
    SPEAKING = auto()       # Playing TTS response
    AUDIO_FLUSH = auto()    # Post-TTS echo & room reverb settle flush
    STOPPED = auto()        # Shutdown complete


# ─── VoiceEngine ──────────────────────────────────────────────────────────────


class VoiceEngine:
    """
    Coordinates the complete voice interaction pipeline.

    Parameters
    ----------
    bus:
        The application EventBus. VoiceEngine subscribes to brain events
        and publishes voice events.
    wake_word_model:
        A loaded WakeWordModel instance.
    recognizer:
        A loaded SpeechRecognizer instance.
    tts:
        A loaded TTSEngine instance.
    sample_rate:
        Audio sample rate. Must match wake_word_model.chunk_size assumption.
        Default: 16000 Hz.
    max_record_seconds:
        Maximum time to record after wake word. Default: 10s.
    silence_timeout_seconds:
        Stop recording after this many seconds of silence. Default: 1.5s.
    wake_threshold:
        Minimum wake word confidence to trigger listening. Default: 0.5.
    """

    def __init__(
        self,
        bus: EventBus,
        wake_word_model: WakeWordModel,
        recognizer: SpeechRecognizer,
        tts: TTSEngine,
        sample_rate: int = 16000,
        max_record_seconds: float = 10.0,
        silence_timeout_seconds: float = 1.5,
        wake_threshold: float = 0.5,
        mic_gain: float = 1.0,
        wake_verify: bool = False,
        wake_verify_window: float = 2.0,
        min_speech_ratio: float = 0.0,
        idle_unload_seconds: float = 0.0,
    ) -> None:
        self._bus = bus
        self._min_speech_ratio = min_speech_ratio
        self._wake_model = wake_word_model
        self._recognizer = recognizer
        self._tts = tts
        self._sample_rate = sample_rate
        self._max_record_seconds = max_record_seconds
        self._silence_timeout = silence_timeout_seconds
        self._wake_threshold = wake_threshold
        self._mic_gain = mic_gain

        self._state = VoiceState.SLEEPING
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._tts_active: bool = False
        self._flush_active: bool = False

        # Rolling chunk energies while idle (~6 s at 80 ms/chunk): the room noise floor.
        self._noise_energies: deque[float] = deque(maxlen=75)
        self._noise_lock = threading.Lock()

        # Stage-2 wake verification (Whisper confirms the name "Spidy" was said).
        # Silently disabled for recognizers without quick_transcribe (tests, null STT).
        self._verifying: bool = False
        # Set when the user interrupts Spidy: the capture buffer already holds the start of their
        # request, so the next listen cycle must not wipe it.
        self._keep_buffered_audio: bool = False
        self._wake_verifier: WakeVerifier | None = None
        self._wake_verify_pool: ThreadPoolExecutor | None = None
        quick = getattr(recognizer, "quick_transcribe", None)
        if wake_verify and callable(quick):
            self._wake_verifier = WakeVerifier(
                transcribe=quick,
                sample_rate=sample_rate,
                window_seconds=wake_verify_window,
            )
            self._wake_verify_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="wake-verify",
            )
            log.info("[WAKE VERIFY] enabled | window={w}s", w=wake_verify_window)

        # Phone-style resource use: only the small wake model stays resident while idle. The speech
        # models are loaded when a wake word starts to sound (prewarm) and freed after
        # ``idle_unload_seconds`` of nothing happening (0 = keep them loaded).
        self._idle_unload = float(idle_unload_seconds)
        self._last_active = time.monotonic()
        self._prewarm_hooks: list[Callable[[], None]] = []
        self._unload_hooks: list[Callable[[], None]] = []
        self._governor_task: asyncio.Task | None = None

        # Audio buffers (populated by AudioCaptureEngine in RECORDING mode)
        self._capture_buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()

        # AudioCaptureEngine — wired with our callbacks
        self._capture_engine = AudioCaptureEngine(
            sample_rate=sample_rate,
            chunk_size=wake_word_model.chunk_size,
            on_wake_chunk=self._process_wake_word,
            on_record_chunk=self._buffer_audio,
            mic_gain=mic_gain,
        )

        # Signalling
        self._wake_detected = threading.Event()
        self._poll_interval = 0.05  # seconds between wake-check polls

        # ── Milestone 14: Continuous conversation ──────────────────────────
        self._conversation_mode: bool = False
        self._conversation_timeout: float = 60.0
        self._paused: bool = False
        self._pause_event = threading.Event()

        # ── Diagnostics ────────────────────────────────────────────────────
        self._wake_chunk_counter: int = 0   # counts audio chunks processed by wake model
        self._processing_task: asyncio.Task | None = None  # reference kept to prevent GC


    # ── Public API ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Start the voice engine.

        - Subscribes to brain.response_ready events
        - Starts the audio capture background thread
        - Starts the async response handler loop
        """
        log.info("VoiceEngine starting...")

        # Subscribe to Brain responses
        self._bus.subscribe("brain.response_ready", self._on_brain_response)
        self._bus.subscribe("system.shutting_down", self._on_shutdown)

        # [DIAG] Start AudioCaptureEngine (owns the microphone stream)
        self._capture_engine.start()

        # Give the audio thread a moment to open the device, then confirm it's alive.
        import time as _time
        _time.sleep(0.15)
        if self._capture_engine.is_running:
            log.info(
                "[VOICE DIAG-5b] AudioCaptureEngine thread is ALIVE — "
                "microphone stream opened successfully."
            )
        else:
            log.error(
                "[VOICE DIAG-5b] AudioCaptureEngine thread is DEAD — "
                "microphone failed to open. Wake word detection will NOT work."
            )

        # Start async processing loop — keep reference to prevent GC and catch crashes.
        self._processing_task = asyncio.create_task(
            self._processing_loop(), name="voice-processing-loop"
        )
        self._processing_task.add_done_callback(self._on_processing_task_done)
        if self._idle_unload > 0 and hasattr(self._recognizer, "ensure_loaded"):
            self._governor_task = asyncio.create_task(self._resource_governor(), name="voice-resource-governor")

        log.info(
            "VoiceEngine started | wake_model={m} | threshold={t} | "
            "processing_task={tid}",
            m=self._wake_model.model_name,
            t=self._wake_threshold,
            tid=id(self._processing_task),
        )
        log.info("[WAKE] listener active")

    async def stop(self) -> None:
        """Stop the voice engine and release all resources."""
        log.info("VoiceEngine stopping...")
        self._stop_event.set()
        if self._governor_task is not None:
            self._governor_task.cancel()
        if self._wake_verify_pool is not None:
            self._wake_verify_pool.shutdown(wait=False, cancel_futures=True)
        self._tts.stop()
        self._set_state(VoiceState.STOPPED)

        # Stop AudioCaptureEngine (releases microphone)
        self._capture_engine.stop()

        self._wake_model.unload()
        self._recognizer.unload()
        self._tts.unload()
        log.info("VoiceEngine stopped.")

    async def speak(self, text: str) -> None:
        """
        Speak a response. Interrupts any active playback first.

        Parameters
        ----------
        text:
            The text to synthesise and speak aloud.
        """
        if not text.strip():
            return

        # Interrupt any active speech
        if self._tts.is_speaking:
            self._tts.stop()
            await asyncio.sleep(0.05)  # give stop time to propagate

        self.set_speaking(True)
        await self._bus.publish(VoiceSpeakingStartEvent(text=text))

        log.info("Speaking: '{text}'", text=text[:60] + ("..." if len(text) > 60 else ""))

        # Mute microphone during TTS playback to block echo loop.
        self._capture_engine.set_mode(CaptureMode.IDLE)
        try:
            await self._tts.speak(text)
        finally:
            await self.enter_audio_flush(settle_seconds=0.20)
            self._switch_to_detecting()
            self._set_state(VoiceState.SLEEPING)
            await self._bus.publish(VoiceSpeakingEndEvent())

    def set_speaking(self, active: bool) -> None:
        """Set or clear TTS active speaking state across the voice engine."""
        with self._state_lock:
            self._tts_active = active
            if active:
                self._state = VoiceState.SPEAKING
                log.info("[VOICE STATE]\n{old} → {new}", old=self._state.name, new=VoiceState.SPEAKING.name)

    async def enter_audio_flush(self, settle_seconds: float = 0.20) -> None:
        """
        Perform a bounded post-TTS settle and audio buffer flush.
        Discards DAC audio echo, room reverberation, and resets wake/capture buffers.
        """
        self._set_state(VoiceState.AUDIO_FLUSH)
        with self._state_lock:
            self._flush_active = True
            self._tts_active = False

        # Reset wake model temporal buffer
        if hasattr(self._wake_model, "reset_buffer"):
            self._wake_model.reset_buffer()

        if settle_seconds > 0:
            await asyncio.sleep(settle_seconds)

        # Discard any audio chunks captured during settle
        with self._buffer_lock:
            self._capture_buffer.clear()

        if hasattr(self._wake_model, "reset_buffer"):
            self._wake_model.reset_buffer()
        with self._state_lock:
            self._flush_active = False
            self._state = VoiceState.SLEEPING

    def exit_audio_flush(self) -> None:
        """Exit audio flush state and restore normal listening."""
        with self._state_lock:
            self._flush_active = False
            self._tts_active = False
            self._state = VoiceState.SLEEPING

    def return_to_sleeping(self) -> None:
        """
        Return VoiceEngine and AudioCaptureEngine to clean wake-listening SLEEPING state.
        Called after voice session ends, times out, or when resetting the conversation.
        """
        with self._state_lock:
            old_state = self._state
            self._tts_active = False
            self._flush_active = False
            self._state = VoiceState.SLEEPING
        self._wake_detected.clear()
        with self._buffer_lock:
            self._capture_buffer.clear()
        if hasattr(self, "_wake_model") and hasattr(self._wake_model, "reset_buffer"):
            self._wake_model.reset_buffer()
        if hasattr(self, "_capture_engine") and self._capture_engine is not None:
            self._capture_engine.set_mode(CaptureMode.DETECTING)
        log.info("[VOICE STATE]\n{old} → SLEEPING (wake listener restored)", old=old_state.name)
        self.log_health()

    def discard_pending_wake(self, restore_state: VoiceState = VoiceState.SLEEPING) -> None:
        """Discard a wake event that the session layer deliberately rejected.

        Wake detection runs on the audio thread and moves the engine to
        ``WAKING`` before publishing its event.  If the session controller
        rejects that event (for example because the previous turn is still
        processing), leaving the engine in ``WAKING`` permanently disables
        future wake detection.  This method is the matching acknowledgement
        for that rejection; it does *not* change the capture mode, which may
        intentionally be IDLE while TTS is playing.
        """
        with self._state_lock:
            old_state = self._state
            self._wake_detected.clear()
            self._state = restore_state
        if hasattr(self._wake_model, "reset_buffer"):
            self._wake_model.reset_buffer()
        log.info(
            "[VOICE STATE] discarded rejected wake: {old} → {new}",
            old=old_state.name,
            new=restore_state.name,
        )

    def log_health(self) -> None:
        """Log health telemetry for voice processing loop, capture stream, and wake detector."""
        task_alive = self._processing_task is not None and not self._processing_task.done()
        stream_alive = self._capture_engine.is_running if hasattr(self, "_capture_engine") and self._capture_engine is not None else False
        wake_active = (
            self._state == VoiceState.SLEEPING
            and not self._tts_active
            and not self._flush_active
            and hasattr(self, "_capture_engine")
            and self._capture_engine is not None
            and self._capture_engine.mode == CaptureMode.DETECTING
        )
        log.info(
            "[VOICE HEALTH] processing_task_alive={pt} | capture_stream_alive={cs} | wake_detection_active={wd}",
            pt=task_alive,
            cs=stream_alive,
            wd=wake_active,
        )
        loaded_models = getattr(self._wake_model, "active_model_names", None) or [getattr(self._wake_model, "model_name", "hey_jarvis")]
        log.info(
            "[VOICE WAKE] models_active={models} | listener_active={la}",
            models=loaded_models,
            la=wake_active,
        )

    def discard_pending_wake(self, restore_state: VoiceState = VoiceState.SLEEPING) -> None:
        """
        Discard a pending wake event that was rejected by higher layers (e.g. duplicate wake
        or wake during active TTS/processing), preventing the engine from staying stuck in WAKING.
        """
        self._wake_detected.clear()
        with self._state_lock:
            if self._state == VoiceState.WAKING:
                self._state = restore_state
                log.info(
                    "[VOICE STATE]\nWAKING → {s} (discarded duplicate/rejected wake)",
                    s=restore_state.name,
                )

    def ensure_processing_task(self) -> bool:
        """Ensure the single async processing loop task is alive; restart if crashed."""
        if self._stop_event.is_set():
            return False
        if self._processing_task is None or self._processing_task.done():
            log.warning("[VOICE HEALTH] Processing task was not running! Restarting loop...")
            try:
                loop = asyncio.get_running_loop()
                self._processing_task = loop.create_task(
                    self._processing_loop(), name="voice-processing-loop"
                )
                self._processing_task.add_done_callback(self._on_processing_task_done)
            except RuntimeError:
                pass
        return self._processing_task is not None and not self._processing_task.done()

    @property
    def is_speaking(self) -> bool:
        """True if TTS is currently active or speaking."""
        return self._tts_active or self._tts.is_speaking

    @property
    def is_flushing(self) -> bool:
        """True if post-TTS audio flush is currently active."""
        return self._flush_active

    @property
    def state(self) -> VoiceState:
        return self._state

    # ── Milestone 14: Conversation mode & microphone ──────────────────────

    def enable_conversation_mode(
        self,
        timeout_seconds: float = 60.0,
    ) -> None:
        """
        Enable continuous conversation mode.

        When enabled, the engine stays LISTENING after each response
        instead of returning to SLEEPING (wake word mode), for up to
        ``timeout_seconds`` of inactivity.
        """
        self._conversation_mode = True
        self._conversation_timeout = timeout_seconds
        log.info(
            "VoiceEngine: conversation mode enabled (timeout={t}s).",
            t=timeout_seconds,
        )

    def disable_conversation_mode(self) -> None:
        """Disable continuous conversation mode (return to wake-word mode)."""
        self._conversation_mode = False
        log.info("VoiceEngine: conversation mode disabled.")

    def add_resource_hooks(
        self,
        prewarm: Callable[[], None] | None = None,
        unload: Callable[[], None] | None = None,
    ) -> None:
        """Register extra models to load when a wake word starts to sound / free when idle."""
        if prewarm is not None:
            self._prewarm_hooks.append(prewarm)
        if unload is not None:
            self._unload_hooks.append(unload)

    def _prewarm_models(self) -> None:
        """Audio thread: something that sounds like the wake word is being heard -- start loading."""
        self._last_active = time.monotonic()
        warm = getattr(self._recognizer, "prewarm", None)
        if callable(warm):
            warm()
        for hook in self._prewarm_hooks:
            try:
                hook()
            except Exception as exc:  # noqa: BLE001
                log.debug("prewarm hook failed: {exc}", exc=exc)

    async def _resource_governor(self) -> None:
        """Free the speech models after a quiet spell; they reload on demand."""
        while not self._stop_event.is_set():
            await asyncio.sleep(10.0)
            if self._should_release_models():
                await asyncio.to_thread(self._release_models)

    def _should_release_models(self) -> bool:
        with self._state_lock:
            idle_state = (
                self._state == VoiceState.SLEEPING and not self._tts_active and not self._flush_active
            )
        if not idle_state or self._verifying or self._wake_detected.is_set():
            return False
        quiet = time.monotonic() - self._last_active
        used = getattr(self._recognizer, "idle_seconds", quiet)
        if min(quiet, used) < self._idle_unload:
            return False
        if not getattr(self._recognizer, "is_loaded", False):
            return False
        log.info("[RESOURCES] idle for {s:.0f}s -- releasing speech models.", s=min(quiet, used))
        return True

    def _release_models(self) -> None:
        for fn in [self._recognizer.unload, *self._unload_hooks]:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                log.debug("unload failed: {exc}", exc=exc)

    def signal_relisten(self) -> None:
        """
        Signal the processing loop to start a new listen cycle immediately,
        without requiring a new wake word.

        Called by ContinuousVoiceController after TTS completes and the
        voice conversation session is still active.  Setting _wake_detected
        causes _processing_loop to wake from its poll, switch the capture
        engine to RECORDING mode, and collect the next user utterance —
        exactly the same path as a genuine wake word trigger.

        Guards
        ------
        Only fires when the engine is SLEEPING (idle, waiting for wake).
        If the engine is already LISTENING, PROCESSING, or SPEAKING,
        the current pipeline takes precedence and the signal is dropped
        to avoid overlapping listen cycles.
        """
        with self._state_lock:
            current = self._state
        if current not in (VoiceState.SLEEPING,):
            log.debug(
                "signal_relisten: engine busy ({s}), skipping re-listen.",
                s=current.name,
            )
            return
        log.info(
            "VoiceEngine: continuous conversation re-listen signal received."
        )
        self._wake_detected.set()

    def resume_listening_after_interrupt(self, get_preroll: Callable[[], list[np.ndarray] | None]) -> None:
        """
        The user talked over Spidy: record what they are saying right now, no wake word needed.

        Unlike :meth:`signal_relisten` this keeps the audio already heard. The capture mode leaves
        BARGE_IN first (so new chunks land in the capture buffer), then ``get_preroll`` yields the
        chunks barge-in collected up to that moment and they are put in front.
        """
        with self._state_lock:
            self._tts_active = False
            self._flush_active = False
            self._state = VoiceState.SLEEPING
        if hasattr(self._wake_model, "reset_buffer"):
            self._wake_model.reset_buffer()
        self._keep_buffered_audio = True
        self._capture_engine.set_mode(CaptureMode.RECORDING)
        preroll = get_preroll() or []
        with self._buffer_lock:
            self._capture_buffer[:0] = list(preroll)
        log.info(
            "[VOICE STATE] interrupted by user -> recording the new request (pre-roll {n} chunks).",
            n=len(preroll),
        )
        self._wake_detected.set()

    def pause(self) -> None:
        """
        Pause voice listening (user said 'wait' or 'pause').

        Audio capture continues but the processing loop will not react
        to wake words or recordings until resume() is called.
        """
        if not self._paused:
            self._paused = True
            self._pause_event.clear()
            log.info("VoiceEngine: paused.")

    def resume(self) -> None:
        """Resume from paused state."""
        if self._paused:
            self._paused = False
            self._pause_event.set()
            log.info("VoiceEngine: resumed.")

    def set_microphone(self, device_index: int | None) -> None:
        """
        Switch the input microphone at runtime.

        Parameters
        ----------
        device_index:
            sounddevice input device index, or None for the system default.
        """
        log.info("VoiceEngine: switching microphone to device {d}.", d=device_index)
        self._capture_engine.stop()
        self._capture_engine = AudioCaptureEngine(
            sample_rate=self._sample_rate,
            chunk_size=self._wake_model.chunk_size,
            on_wake_chunk=self._process_wake_word,
            on_record_chunk=self._buffer_audio,
            input_device=device_index,
            mic_gain=self._mic_gain,  # preserve gain setting on device switch
        )
        self._capture_engine.start()
        log.info("VoiceEngine: microphone switched.")

    @property
    def conversation_mode(self) -> bool:
        """True if continuous conversation mode is active."""
        return self._conversation_mode

    @property
    def is_paused(self) -> bool:
        """True while the voice session is paused."""
        return self._paused

    # ── Mode switching (delegates to AudioCaptureEngine) ─────────────────

    def _switch_to_recording(self) -> None:
        """Switch AudioCaptureEngine to RECORDING mode and clear the buffer."""
        with self._buffer_lock:
            self._capture_buffer.clear()
        self._capture_engine.set_mode(CaptureMode.RECORDING)

    def _switch_to_detecting(self) -> None:
        """Switch AudioCaptureEngine back to DETECTING mode."""
        self._capture_engine.set_mode(CaptureMode.DETECTING)

    # ── Wake word and audio buffering callbacks (called from audio thread) ─

    def _process_wake_word(self, chunk: np.ndarray) -> None:
        """Run wake word detection on one audio chunk."""
        self._wake_chunk_counter += 1
        if self._wake_verifier is not None:
            self._wake_verifier.push(chunk)
        # Track the room noise level (~6 s) so end-of-speech detection can adapt to it.
        with self._noise_lock:
            self._noise_energies.append(float(np.sqrt(np.mean(chunk ** 2))))

        try:
            score = self._wake_model.process_chunk(chunk)
        except Exception as exc:  # noqa: BLE001
            log.debug("Wake word processing error: {exc}", exc=exc)
            return

        # [DIAG-10] Throttled proof-of-life: every 200 chunks (~16 s at 80ms/chunk)
        # confirms audio frames ARE reaching the wake model.
        if self._wake_chunk_counter % 200 == 0:
            log.info(
                "[VOICE DIAG-10] Wake model alive: chunks_processed={n} | "
                "latest_score={s:.4f} | threshold={t}",
                n=self._wake_chunk_counter,
                s=score,
                t=self._wake_threshold,
            )

        # [DIAG-11] Log every non-trivial score so we can see when detection gets close.
        # Threshold 0.05 catches partial matches (model needs 0.4 to trigger).
        if score > _PREWARM_SCORE:
            self._prewarm_models()      # no-op when the models are already loaded
        if score > 0.05:
            log.info(
                "[VOICE DIAG-11] Wake score={s:.4f} | "
                "threshold={t} | chunks={n}",
                s=score,
                t=self._wake_threshold,
                n=self._wake_chunk_counter,
            )

        with self._state_lock:
            current_state = self._state
            is_busy = (
                current_state in (VoiceState.SPEAKING, VoiceState.AUDIO_FLUSH, VoiceState.WAKING, VoiceState.STOPPED)
                or self._tts_active
                or self._flush_active
            )

        if is_busy:
            if score > 0.30:
                log.info(
                    "[VOICE DIAG-GATE] Wake suppressed | score={s:.4f} | state={st} | "
                    "tts_active={tts} | flush_active={fl}",
                    s=score,
                    st=(current_state.name if hasattr(current_state, "name") else str(current_state)),
                    tts=self._tts_active,
                    fl=self._flush_active,
                )
            return

        if self._verifying:
            return

        is_detected = getattr(self._wake_model, "is_detected", False)
        if not is_detected and not hasattr(self._wake_model, "_min_consecutive_frames"):
            is_detected = (score >= self._wake_threshold)

        if score > 0.30 and not is_detected:
            log.info(
                "[VOICE DIAG-GATE] Score above threshold but is_detected=False | score={s:.4f} | threshold={t}",
                s=score,
                t=self._wake_threshold,
            )

        if is_detected:
            detected_model = getattr(self._wake_model, "last_detected_model", "") or self._wake_model.model_name
            detected_phrase = getattr(self._wake_model, "last_detected_phrase", "") or getattr(self._wake_model, "wake_phrase", detected_model)
            log.info(
                "[WAKE] detected phrase='{phrase}' model='{model}' confidence={score:.3f}",
                phrase=detected_phrase,
                model=detected_model,
                score=score,
            )
            if self._wake_verifier is not None:
                # Stage 2: confirm with Whisper off the audio thread.
                self._verifying = True
                self._wake_model.reset_buffer()
                # One long-lived worker thread: GPU Whisper calls from a fresh thread
                # each time create/destroy CUDA handles per thread and crashed the
                # process (heap corruption, 0xc0000374) in real use.
                self._wake_verify_pool.submit(
                    self._verify_and_fire, detected_model, detected_phrase, score,
                )
                return
            self._fire_wake(detected_model, detected_phrase, score)

    def _verify_and_fire(self, model: str, phrase: str, score: float) -> None:
        """Worker thread: Whisper-check a wake detection, then fire or discard it."""
        try:
            audio = self._wake_verifier.collect()  # type: ignore[union-attr]
            ok, text = self._wake_verifier.verify(audio)  # type: ignore[union-attr]
            if not ok and not text.strip() and score >= _EMPTY_ACCEPT_SCORE:
                # Whisper heard no words at all although the wake model is very sure: the phrase may
                # simply have been clipped. Listen a little longer; if there is still nothing to
                # contradict the wake model (false triggers always come with *other* words), accept.
                audio = self._wake_verifier.collect()  # type: ignore[union-attr]
                ok, text = self._wake_verifier.verify(audio)  # type: ignore[union-attr]
                if not ok and not text.strip():
                    log.info("[WAKE VERIFY] no words heard but wake score {s:.2f} is high -- accepting.", s=score)
                    ok = True
            if not ok:
                rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
                log.info("[WAKE VERIFY] audio rms={r:.3f} peak={p:.2f}", r=rms, p=float(np.abs(audio).max()) if len(audio) else 0.0)
            if ok and not self._stop_event.is_set():
                log.info("[WAKE VERIFY] accepted | heard='{t}'", t=text)
                self._fire_wake(model, phrase, score)
            else:
                log.info(
                    "[WAKE VERIFY] rejected | model='{m}' score={s:.2f} heard='{t}'",
                    m=model, s=score, t=text,
                )
        finally:
            self._verifying = False

    def _fire_wake(self, model: str, phrase: str, score: float) -> None:
        """Enter WAKING and announce the wake word (audio thread or verify thread)."""
        self._set_state(VoiceState.WAKING)
        # Reset OWW's temporal smoothing buffer immediately after detection.
        self._wake_model.reset_buffer()
        self._bus.publish_threadsafe(
            WakeWordDetectedEvent(
                confidence=score,
                model_name=model,
                wake_phrase=phrase,
            )
        )
        # Signal async loop to start listening if not in conversation mode
        # (In conversation mode, ContinuousVoiceController coordinates ack and listening)
        if not self._conversation_mode:
            self._wake_detected.set()

    def _buffer_audio(self, chunk: np.ndarray) -> None:
        """Buffer audio chunks during the LISTENING state."""
        # Never buffer chunks when TTS is active or during audio flush
        if self._tts_active or self._flush_active:
            return
        with self._buffer_lock:
            self._capture_buffer.append(chunk)

    # ── Async processing loop ─────────────────────────────────────────────

    async def _processing_loop(self) -> None:
        """
        Async loop that handles:
        1. Detecting wake word signal → switch to LISTENING
        2. Waiting for speech to complete → run STT → publish transcript
        """
        while not self._stop_event.is_set():
            # Wait for wake word signal — poll every 50ms to stay responsive
            await asyncio.sleep(self._poll_interval)

            if self._stop_event.is_set():
                break

            if not self._wake_detected.is_set():
                continue

            # Check if TTS or post-TTS flush is active before starting recording
            if self._tts_active or self._flush_active or self._state in (VoiceState.SPEAKING, VoiceState.AUDIO_FLUSH):
                await asyncio.sleep(0.05)
                continue

            self._wake_detected.clear()

            # Transition to LISTENING: tell AudioCaptureEngine to buffer audio
            self._set_state(VoiceState.LISTENING)
            keep_audio, self._keep_buffered_audio = self._keep_buffered_audio, False
            if not keep_audio:
                self._switch_to_recording()
            await self._bus.publish(VoiceListeningEvent())

            if not keep_audio:
                # 300ms pre-recording delay: let acoustic decay clear
                await asyncio.sleep(0.30)
                with self._buffer_lock:
                    self._capture_buffer.clear()  # discard wake-phrase tail
            # else: the user is already mid-sentence (interrupted Spidy) -- every chunk counts

            # Wait for speech to complete (silence detection)
            audio = await asyncio.to_thread(self._record_until_silence)

            if audio is None or len(audio) == 0:
                log.debug("No speech detected after wake word.")
                if self._conversation_mode and self._state == VoiceState.LISTENING:
                    # Session still open: keep listening instead of dropping back to the
                    # wake word; the session manager ends it after its idle timeout.
                    self._wake_detected.set()
                    continue
                self._switch_to_detecting()
                self._set_state(VoiceState.SLEEPING)
                continue

            # Run STT
            self._set_state(VoiceState.PROCESSING)
            result = await self._recognizer.transcribe(audio)

            if result.is_empty:
                log.debug("STT returned empty result.")
                self._switch_to_detecting()
                self._set_state(VoiceState.SLEEPING)
                continue

            # TTS echo check on completed transcript
            if self._tts_active or self._flush_active or self._state in (VoiceState.SPEAKING, VoiceState.AUDIO_FLUSH):
                log.info("[VOICE] transcript=rejected reason=tts_echo")
                log.info("[VOICE AUDIO] rejected source=tts_echo")
                self._switch_to_detecting()
                self._set_state(VoiceState.SLEEPING)
                continue

            log.info("User said: '{text}'", text=result.text)

            self._switch_to_detecting()
            # In continuous mode the session controller is now processing
            # this turn.  Keeping the hardware engine in PROCESSING closes a
            # race where a second wake was accepted between STT and TTS, then
            # rejected by the controller and stranded the engine in WAKING.
            # The response path transitions to SPEAKING and eventually back
            # to SLEEPING before it signals the next listen cycle.
            if not self._conversation_mode:
                self._set_state(VoiceState.SLEEPING)

            import time as _time
            await self._bus.publish(
                VoiceTranscriptEvent(
                    text=result.text,
                    confidence=result.confidence,
                    language=result.language or "en",
                    source="user",
                    capture_state=VoiceState.PROCESSING.name,
                    tts_active=False,
                    flush_active=False,
                    timestamp=_time.monotonic(),
                    segment_signals=getattr(result, "segment_signals", None),
                )
            )

    def _record_until_silence(self) -> np.ndarray | None:
        """
        Collect audio until silence is detected or max duration reached.

        Runs in a thread. Uses a simple energy-based VAD for now.
        Returns the concatenated audio array.
        """
        import time

        chunk_size = self._wake_model.chunk_size

        # Silence ends the command after _SILENCE_TIMEOUT_SECS of quiet. "Quiet" is
        # relative to the room: with mic_gain=8 the background alone measures ~0.06
        # RMS, so the old fixed 0.03 threshold was never met and every command was
        # recorded for the full 10 s cap (then Whisper hallucinated on the noise).
        _MIN_SPEECH_ENERGY = 0.03   # absolute floor for the threshold
        _NOISE_MARGIN = 2.5         # speech must exceed noise floor by this factor
        _MAX_THRESHOLD = 0.25       # never demand more than this (very noisy rooms)
        _MIN_SPEECH_CHUNKS = 3      # speech chunks required before silence can end
        _SILENCE_TIMEOUT_SECS = 2.0
        _STRAY_CHUNKS = 1           # tolerated noise blips inside the silence window

        silence_chunks = int(
            _SILENCE_TIMEOUT_SECS * self._sample_rate / chunk_size
        )

        threshold = self._silence_threshold(None, _MIN_SPEECH_ENERGY, _NOISE_MARGIN, _MAX_THRESHOLD)
        recent_speech: deque[bool] = deque(maxlen=silence_chunks)
        all_chunks: list[np.ndarray] = []
        energies: list[float] = []
        speech_chunks_seen = 0
        ended_on_silence = False

        start_time = time.monotonic()

        while True:
            # Check max duration
            elapsed = time.monotonic() - start_time
            if elapsed >= self._max_record_seconds:
                log.debug("Max record duration reached.")
                break

            if self._stop_event.is_set() or self._tts_active or self._flush_active:
                return None

            # Drain the capture buffer
            with self._buffer_lock:
                new_chunks = list(self._capture_buffer)
                self._capture_buffer.clear()

            for chunk in new_chunks:
                all_chunks.append(chunk)
                energy = float(np.sqrt(np.mean(chunk ** 2)))
                energies.append(energy)

                # No idle history (e.g. hotkey start): learn the floor from this clip.
                if threshold is None and len(energies) >= 12:
                    threshold = self._silence_threshold(
                        energies, _MIN_SPEECH_ENERGY, _NOISE_MARGIN, _MAX_THRESHOLD,
                    )
                limit = threshold if threshold is not None else _MIN_SPEECH_ENERGY

                is_speech = energy >= limit
                recent_speech.append(is_speech)
                if is_speech:
                    speech_chunks_seen += 1

                # Only stop on silence after we've seen actual speech content
                if (len(recent_speech) == silence_chunks
                        and sum(recent_speech) <= _STRAY_CHUNKS
                        and speech_chunks_seen >= _MIN_SPEECH_CHUNKS):
                    log.debug(
                        "Silence detected after {n} chunks (speech={s}, threshold={t:.3f}).",
                        n=len(all_chunks),
                        s=speech_chunks_seen,
                        t=limit,
                    )
                    ended_on_silence = True
                    break
            else:
                time.sleep(0.01)
                continue
            break

        if not all_chunks:
            return None

        if speech_chunks_seen < _MIN_SPEECH_CHUNKS:
            # Only room noise: sending it to Whisper makes it invent text.
            log.info(
                "[VOICE] no speech above noise floor (threshold={t}, chunks={n}); skipping STT.",
                t=f"{threshold:.3f}" if threshold is not None else "n/a",
                n=len(all_chunks),
            )
            return None

        # How loud was the speech compared with the room? Background chatter sits just
        # above the threshold; someone talking to the mic is far above it.
        limit = threshold if threshold is not None else _MIN_SPEECH_ENERGY
        speech_energies = [e for e in energies if e >= limit]
        median_speech = float(np.median(speech_energies)) if speech_energies else 0.0
        ratio = median_speech / limit if limit else 0.0
        log.info(
            "[VOICE] recorded {s:.1f}s | end={end} | threshold={t} | speech_median={m:.3f} | ratio={r:.2f}",
            s=len(all_chunks) * chunk_size / self._sample_rate,
            end="silence" if ended_on_silence else "max_duration",
            t=f"{limit:.3f}",
            m=median_speech,
            r=ratio,
        )
        if self._min_speech_ratio > 0 and ratio < self._min_speech_ratio:
            log.info(
                "[VOICE] too quiet to be a command (ratio {r:.2f} < {m:.2f}); ignoring as background speech.",
                r=ratio, m=self._min_speech_ratio,
            )
            return None
        return np.concatenate(all_chunks)

    def _silence_threshold(
        self,
        recorded: list[float] | None,
        floor: float,
        margin: float,
        ceiling: float,
    ) -> float | None:
        """
        Speech/silence energy threshold relative to the room's noise level.

        Uses the 25th percentile of recent idle chunk energies (or of ``recorded``
        when there is no idle history). Returns None if there is not enough data.
        """
        if recorded is None:
            with self._noise_lock:
                history = list(self._noise_energies)
        else:
            history = recorded
        if len(history) < 12:
            return None
        noise_floor = float(np.percentile(history, 25))
        return min(ceiling, max(floor, noise_floor * margin))


    # ── Event handlers ────────────────────────────────────────────────────

    async def _on_brain_response(self, event: BrainResponseReadyEvent) -> None:
        """Called when the Brain has a response to speak."""
        # event.response_text is the canonical field from brain.events.BrainResponseReadyEvent.
        await self.speak(event.response_text)

    async def _on_shutdown(self, event: Event) -> None:
        """Called on system shutdown event."""
        await self.stop()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _on_processing_task_done(self, task: asyncio.Task) -> None:
        """Callback fired when _processing_loop exits — logs any silent crash."""
        try:
            exc = task.exception()
            if exc is not None:
                log.error(
                    "[VOICE DIAG] Processing loop CRASHED silently: {exc}",
                    exc=exc,
                )
            else:
                log.info("[VOICE DIAG] Processing loop exited cleanly.")
        except asyncio.CancelledError:
            log.info("[VOICE DIAG] Processing loop was cancelled.")

    def _set_state(self, new_state: VoiceState) -> None:
        with self._state_lock:
            if self._state != new_state:
                log.info(
                    "[VOICE STATE]\n{old} → {new}",
                    old=self._state.name,
                    new=new_state.name,
                )
                self._state = new_state
            if new_state != VoiceState.SLEEPING:
                self._last_active = time.monotonic()
