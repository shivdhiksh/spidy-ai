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
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from spidy.core.event_bus import Event, EventBus
from spidy.logging.logger import get_logger
from spidy.perception.voice.audio_capture import AudioCaptureEngine, CaptureMode
from spidy.perception.voice.stt.base import SpeechRecognizer
from spidy.perception.voice.tts.base import TTSEngine
from spidy.perception.voice.wake_word.base import WakeWordModel

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


# ─── Voice Events ─────────────────────────────────────────────────────────────


@dataclass
class WakeWordDetectedEvent(Event):
    """Published when the wake word is confirmed above threshold."""
    topic = "wake_word.detected"
    confidence: float = 0.0
    model_name: str = ""


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
    SLEEPING = auto()       # Wake word detector active
    WAKING = auto()         # Wake word detected, preparing to listen
    LISTENING = auto()      # Recording user speech
    PROCESSING = auto()     # Running STT transcription
    SPEAKING = auto()       # Playing TTS response
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
    ) -> None:
        self._bus = bus
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

        log.info(
            "VoiceEngine started | wake_model={m} | threshold={t} | "
            "processing_task={tid}",
            m=self._wake_model.model_name,
            t=self._wake_threshold,
            tid=id(self._processing_task),
        )

    async def stop(self) -> None:
        """Stop the voice engine and release all resources."""
        log.info("VoiceEngine stopping...")
        self._stop_event.set()
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

        self._set_state(VoiceState.SPEAKING)
        await self._bus.publish(VoiceSpeakingStartEvent(text=text))

        log.info("Speaking: '{text}'", text=text[:60] + ("..." if len(text) > 60 else ""))

        # Issue 1 fix: mute the microphone during TTS playback.
        # Without this, AudioCaptureEngine stays in DETECTING mode and the
        # wake-word model (and subsequently STT) processes Spidy's own speaker
        # output, producing fake transcripts. IDLE mode discards all chunks.
        self._capture_engine.set_mode(CaptureMode.IDLE)
        try:
            await self._tts.speak(text)
        finally:
            # Restore detection only after playback is fully complete.
            # Also reset the OWW temporal buffer so stale TTS audio does not
            # bleed into the next detection window (Issue 2 partial fix).
            self._wake_model.reset_buffer()
            self._switch_to_detecting()
            self._set_state(VoiceState.SLEEPING)
            await self._bus.publish(VoiceSpeakingEndEvent())

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
        if score > 0.05:
            log.info(
                "[VOICE DIAG-11] Wake score={s:.4f} | "
                "threshold={t} | chunks={n}",
                s=score,
                t=self._wake_threshold,
                n=self._wake_chunk_counter,
            )

        if score >= self._wake_threshold:
            log.info(
                "Wake word detected! confidence={score:.3f}",
                score=score,
            )
            self._set_state(VoiceState.WAKING)
            # Issue 2 fix: reset OWW's temporal smoothing buffer immediately
            # after detection. This prevents the high-score frame from
            # re-triggering on the very next chunk (double-fire) and ensures
            # the buffer is clean for the next wake-word attempt.
            self._wake_model.reset_buffer()
            self._bus.publish_threadsafe(
                WakeWordDetectedEvent(
                    confidence=score,
                    model_name=self._wake_model.model_name,
                )
            )
            # Signal async loop to start listening
            self._wake_detected.set()

    def _buffer_audio(self, chunk: np.ndarray) -> None:
        """Buffer audio chunks during the LISTENING state."""
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

            self._wake_detected.clear()

            # Transition to LISTENING: tell AudioCaptureEngine to buffer audio
            self._set_state(VoiceState.LISTENING)
            self._switch_to_recording()
            await self._bus.publish(VoiceListeningEvent())

            # Fix 3A: 300ms pre-recording delay.
            # The tail of the wake phrase ('...rvis') is still ringing in the
            # microphone capsule and room reverb when recording starts.
            # A 300ms gap lets the acoustic decay clear before buffering the command.
            # The AudioCaptureEngine is in RECORDING mode during this pause, so
            # early chunks (wake phrase tail) are discarded by clearing the buffer
            # immediately after the sleep.
            await asyncio.sleep(0.30)
            with self._buffer_lock:
                self._capture_buffer.clear()  # discard wake-phrase tail

            # Wait for speech to complete (silence detection)
            audio = await asyncio.to_thread(self._record_until_silence)

            if audio is None or len(audio) == 0:
                log.debug("No speech detected after wake word.")
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

            log.info("User said: '{text}'", text=result.text)

            # BUG 1 FIX: Return to DETECTING mode immediately after STT completes.
            # Without this, AudioCaptureEngine stays in RECORDING mode between
            # utterances and the capture buffer keeps accumulating new microphone
            # chunks. On the next wake event, _switch_to_recording() clears the
            # buffer — but there is a race between the clear() and the audio
            # thread appending the next chunk. If the timing is bad the old PCM
            # bleeds into the new recording, causing Whisper to transcribe the
            # previous utterance again ("Open Camera. Open Camera. Open Camera.")
            self._switch_to_detecting()
            self._set_state(VoiceState.SLEEPING)

            await self._bus.publish(
                VoiceTranscriptEvent(
                    text=result.text,
                    confidence=result.confidence,
                    language=result.language or "en",
                )
            )

            # Brain response handling (TTS) is driven by _on_brain_response.
            # We are now back in SLEEPING/DETECTING mode ready for the next wake.

    def _record_until_silence(self) -> np.ndarray | None:
        """
        Collect audio until silence is detected or max duration reached.

        Runs in a thread. Uses a simple energy-based VAD for now.
        Returns the concatenated audio array.
        """
        import time

        max_samples = int(self._max_record_seconds * self._sample_rate)  # noqa: F841

        chunk_size = self._wake_model.chunk_size

        # Fix 3B: silence timeout for command recognition raised to 2.5s.
        # Short commands like 'Open Notepad' are ~0.8s long, followed by natural
        # silence before the user speaks again. The old self._silence_timeout (1.5s)
        # was too short — it fired before the user finished speaking slowly.
        # 2.5s gives a comfortable window without making the system feel sluggish.
        _SILENCE_ENERGY = 0.03
        _MIN_SPEECH_CHUNKS = 3  # must see at least 3 speech chunks before silence can end
        _SILENCE_TIMEOUT_SECS = 2.5

        silence_chunks = int(
            _SILENCE_TIMEOUT_SECS * self._sample_rate / chunk_size
        )

        consecutive_silence = 0
        total_samples = 0
        all_chunks: list[np.ndarray] = []
        speech_chunks_seen = 0

        start_time = time.monotonic()

        while True:
            # Check max duration
            elapsed = time.monotonic() - start_time
            if elapsed >= self._max_record_seconds:
                log.debug("Max record duration reached.")
                break

            if self._stop_event.is_set():
                return None

            # Drain the capture buffer
            with self._buffer_lock:
                new_chunks = list(self._capture_buffer)
                self._capture_buffer.clear()

            for chunk in new_chunks:
                all_chunks.append(chunk)
                total_samples += len(chunk)

                # Energy-based silence detection (threshold calibrated for gained signal)
                energy = float(np.sqrt(np.mean(chunk ** 2)))
                if energy < _SILENCE_ENERGY:
                    consecutive_silence += 1
                else:
                    consecutive_silence = 0
                    speech_chunks_seen += 1

                # Only stop on silence after we've seen actual speech content
                if (consecutive_silence >= silence_chunks
                        and speech_chunks_seen >= _MIN_SPEECH_CHUNKS):
                    log.debug(
                        "Silence detected after {n} chunks (speech={s}).",
                        n=len(all_chunks),
                        s=speech_chunks_seen,
                    )
                    break
            else:
                time.sleep(0.01)
                continue
            break

        if not all_chunks:
            return None

        return np.concatenate(all_chunks)


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
                log.debug(
                    "VoiceEngine: {old} → {new}",
                    old=self._state.name,
                    new=new_state.name,
                )
                self._state = new_state
