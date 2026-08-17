"""
AudioCaptureEngine — Microphone Input Pipeline
===============================================
Manages the raw microphone stream independently from wake word logic.

Design
------
- Owns the sounddevice InputStream in a dedicated background thread
- Exposes two modes: DETECTING (feed chunks to wake word) and
  RECORDING (buffer chunks for STT)
- The VoiceEngine controls which mode is active
- Zero-copy: numpy arrays are passed by reference, not copied

Why a separate class?
---------------------
Previously audio capture was inlined in VoiceEngine. Separating it:
1. Makes AudioCaptureEngine independently testable (mock the stream)
2. Lets us swap sounddevice for another backend (e.g. PyAudio) via config
3. Keeps VoiceEngine focused on state transitions, not I/O

Audio spec
----------
- Sample rate: 16000 Hz (required by openWakeWord and faster-whisper)
- Channels: 1 (mono)
- Dtype: float32 (values in [-1.0, 1.0])
- Block size: matches wake word model chunk_size (1280 samples = 80ms)
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from enum import Enum, auto
from typing import Any

import numpy as np

from spidy.logging.logger import get_logger

log = get_logger(__name__)

# Callback type: receives one float32 audio chunk
AudioCallback = Callable[[np.ndarray], None]


class CaptureMode(Enum):
    """Operating mode of the capture engine."""
    DETECTING = auto()   # Feed chunks to wake word callback
    RECORDING = auto()   # Buffer chunks for STT
    IDLE = auto()        # Neither (between transitions / legacy mute)
    BARGE_IN = auto()    # Feed chunks to barge-in interruption detector only
                         # Used during TTS playback so the user can say "Spidy stop".
                         # Neither the wake-word model nor the main STT buffer receive
                         # these chunks — only the lightweight BargeInDetector does.


class AudioCaptureEngine:
    """
    Microphone input manager. Runs in a dedicated background thread.

    Parameters
    ----------
    sample_rate:
        Audio sample rate in Hz. Default: 16000.
    chunk_size:
        Number of samples per audio chunk. Default: 1280 (80ms at 16kHz).
    input_device:
        sounddevice device index. None = system default.
    on_wake_chunk:
        Callback called with each audio chunk in DETECTING mode.
        Called from the audio thread — must be thread-safe.
    on_record_chunk:
        Callback called with each audio chunk in RECORDING mode.
        Called from the audio thread — must be thread-safe.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1280,
        input_device: int | None = None,
        on_wake_chunk: AudioCallback | None = None,
        on_record_chunk: AudioCallback | None = None,
        on_barge_in_chunk: AudioCallback | None = None,
        mic_gain: float = 1.0,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._input_device = input_device
        self._on_wake_chunk = on_wake_chunk
        self._on_record_chunk = on_record_chunk
        self._on_barge_in_chunk = on_barge_in_chunk
        self._mic_gain = float(mic_gain)  # Software amplification factor (applied before OWW/STT)
        self._low_signal_warned = False   # One-shot warning for very quiet mic

        self._mode = CaptureMode.IDLE
        self._mode_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start the audio capture background thread.

        Raises
        ------
        RuntimeError
            If already running.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("AudioCaptureEngine is already running.")

        self._stop_event.clear()
        self._mode = CaptureMode.DETECTING  # Start in detection mode

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="spidy-audio-capture",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "AudioCaptureEngine started | sr={sr} | chunk={c}",
            sr=self._sample_rate,
            c=self._chunk_size,
        )

    def stop(self) -> None:
        """Stop the audio capture thread and release the microphone."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._mode = CaptureMode.IDLE
        log.debug("AudioCaptureEngine stopped.")

    def set_mode(self, mode: CaptureMode) -> None:
        """
        Switch the operating mode.

        Thread-safe — can be called from any thread.

        Parameters
        ----------
        mode:
            DETECTING: feed chunks to on_wake_chunk
            RECORDING: feed chunks to on_record_chunk
            IDLE: discard all chunks
        """
        with self._mode_lock:
            if self._mode != mode:
                log.debug(
                    "AudioCaptureEngine mode: {old} → {new}",
                    old=self._mode.name,
                    new=mode.name,
                )
                self._mode = mode

    @property
    def mode(self) -> CaptureMode:
        """Current operating mode."""
        return self._mode

    @property
    def is_running(self) -> bool:
        """True while the capture thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ── Internal capture loop ─────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """
        Main audio capture loop. Runs in background thread.
        Never raises — all exceptions are caught and logged.
        """
        try:
            import sounddevice as sd
        except ImportError:
            log.error(
                "sounddevice is not installed. "
                "Install it with: pip install sounddevice"
            )
            return

        try:
            stream_kwargs: dict[str, Any] = {
                "samplerate": self._sample_rate,
                "channels": 1,
                "dtype": "float32",
                "blocksize": self._chunk_size,
            }
            if self._input_device is not None:
                stream_kwargs["device"] = self._input_device

            # [DIAG-6] Log exactly which device sounddevice will open.
            try:
                device_info = sd.query_devices(
                    self._input_device, kind="input"
                )
                log.info(
                    "[VOICE DIAG-6] Opening microphone: index={idx} | "
                    "name='{name}' | channels={ch} | rate={sr}",
                    idx=self._input_device if self._input_device is not None
                        else sd.default.device[0],
                    name=device_info.get("name", "?"),
                    ch=device_info.get("max_input_channels", "?"),
                    sr=int(device_info.get("default_samplerate", 0)),
                )
            except Exception as _dev_exc:  # noqa: BLE001
                log.warning(
                    "[VOICE DIAG-6] Could not query device info: {e}",
                    e=_dev_exc,
                )

            with sd.InputStream(**stream_kwargs) as stream:
                log.info(
                    "[VOICE DIAG-6b] sd.InputStream opened successfully — "
                    "audio capture is ACTIVE. mic_gain={g}x",
                    g=self._mic_gain,
                )
                _chunk_counter = 0
                _rms_accum = 0.0
                while not self._stop_event.is_set():
                    audio_chunk, overflowed = stream.read(self._chunk_size)
                    if overflowed:
                        log.debug("Audio buffer overflow — some samples dropped.")

                    audio_flat = audio_chunk.flatten()

                    # ── Software microphone gain ──────────────────────────────
                    if self._mic_gain != 1.0:
                        audio_flat = np.clip(
                            audio_flat * self._mic_gain, -1.0, 1.0
                        )

                    # ── Periodic low-signal warning (every 500 chunks ≈ 40s) ─
                    _chunk_counter += 1
                    _rms_accum += float(np.sqrt(np.mean(audio_flat ** 2)))
                    if _chunk_counter % 500 == 0:
                        avg_rms = _rms_accum / 500
                        _rms_accum = 0.0
                        if avg_rms < 0.005 and not self._low_signal_warned:
                            self._low_signal_warned = True
                            log.warning(
                                "[VOICE] Microphone signal is very weak "
                                "(avg RMS={r:.5f} over last 500 chunks). "
                                "Wake word detection may fail. "
                                "Increase Windows mic volume or set "
                                "voice.audio.mic_gain in config (currently {g}x).",
                                r=avg_rms,
                                g=self._mic_gain,
                            )
                        elif avg_rms >= 0.005:
                            self._low_signal_warned = False  # reset if signal recovers

                    self._dispatch(audio_flat)

        except Exception as exc:  # noqa: BLE001
            log.error("AudioCaptureEngine error: {exc}", exc=exc)

    def _dispatch(self, chunk: np.ndarray) -> None:
        """Route the chunk to the correct callback based on current mode."""
        with self._mode_lock:
            mode = self._mode

        if mode == CaptureMode.DETECTING and self._on_wake_chunk:
            try:
                self._on_wake_chunk(chunk)
            except Exception as exc:  # noqa: BLE001
                log.debug("wake chunk callback error: {exc}", exc=exc)

        elif mode == CaptureMode.RECORDING and self._on_record_chunk:
            try:
                self._on_record_chunk(chunk)
            except Exception as exc:  # noqa: BLE001
                log.debug("record chunk callback error: {exc}", exc=exc)

        elif mode == CaptureMode.BARGE_IN and self._on_barge_in_chunk:
            try:
                self._on_barge_in_chunk(chunk)
            except Exception as exc:  # noqa: BLE001
                log.debug("barge-in chunk callback error: {exc}", exc=exc)
        # IDLE → chunk is discarded (no callback called)
