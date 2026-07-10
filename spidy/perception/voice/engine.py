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


@dataclass
class BrainResponseReadyEvent(Event):
    """Subscribed from Brain: response text to speak."""
    topic = "brain.response_ready"
    text: str = ""
    session_id: str = ""


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
    ) -> None:
        self._bus = bus
        self._wake_model = wake_word_model
        self._recognizer = recognizer
        self._tts = tts
        self._sample_rate = sample_rate
        self._max_record_seconds = max_record_seconds
        self._silence_timeout = silence_timeout_seconds
        self._wake_threshold = wake_threshold

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
        )

        # Signalling
        self._wake_detected = threading.Event()
        self._poll_interval = 0.05  # seconds between wake-check polls


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

        # Start AudioCaptureEngine (owns the microphone stream)
        self._capture_engine.start()

        # Start async processing loop
        asyncio.create_task(self._processing_loop())

        log.info("VoiceEngine started | wake_model={m} | threshold={t}",
                 m=self._wake_model.model_name, t=self._wake_threshold)

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

        try:
            await self._tts.speak(text)
        finally:
            self._switch_to_detecting()
            self._set_state(VoiceState.SLEEPING)
            await self._bus.publish(VoiceSpeakingEndEvent())

    @property
    def state(self) -> VoiceState:
        return self._state

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
        try:
            score = self._wake_model.process_chunk(chunk)
        except Exception as exc:  # noqa: BLE001
            log.debug("Wake word processing error: {exc}", exc=exc)
            return

        if score >= self._wake_threshold:
            log.info(
                "Wake word detected! confidence={score:.3f}",
                score=score,
            )
            self._set_state(VoiceState.WAKING)
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

            await self._bus.publish(
                VoiceTranscriptEvent(
                    text=result.text,
                    confidence=result.confidence,
                    language=result.language or "en",
                )
            )

            # Stay in PROCESSING until Brain responds (handled by _on_brain_response)

    def _record_until_silence(self) -> np.ndarray | None:
        """
        Collect audio until silence is detected or max duration reached.

        Runs in a thread. Uses a simple energy-based VAD for now.
        Returns the concatenated audio array.
        """
        import time

        max_samples = int(self._max_record_seconds * self._sample_rate)  # noqa: F841

        chunk_size = self._wake_model.chunk_size
        silence_chunks = int(
            self._silence_timeout * self._sample_rate / chunk_size
        )

        consecutive_silence = 0
        total_samples = 0
        all_chunks: list[np.ndarray] = []

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

                # Energy-based silence detection
                energy = np.sqrt(np.mean(chunk ** 2))
                if energy < 0.01:   # ~-40dBFS threshold
                    consecutive_silence += 1
                else:
                    consecutive_silence = 0

                if consecutive_silence >= silence_chunks:
                    log.debug(
                        "Silence detected after {n} chunks. Done recording.",
                        n=len(all_chunks),
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
        await self.speak(event.text)

    async def _on_shutdown(self, event: Event) -> None:
        """Called on system shutdown event."""
        await self.stop()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _set_state(self, new_state: VoiceState) -> None:
        with self._state_lock:
            if self._state != new_state:
                log.debug(
                    "VoiceEngine: {old} → {new}",
                    old=self._state.name,
                    new=new_state.name,
                )
                self._state = new_state
