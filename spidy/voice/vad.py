"""
VoiceActivityDetector — Enhanced Voice Activity Detection
==========================================================
Sits between the microphone buffer and the STT engine.

Responsibilities
----------------
1. Collect audio chunks from the recording buffer
2. Apply energy-based silence detection (always available)
3. Optionally apply webrtcvad (better noise tolerance, low CPU)
4. Return clean speech audio or None (silence / timeout)

Strategy
--------
Primary:   webrtcvad (aggressiveness 0–3) — handles background noise well
Fallback:  Energy threshold (RMS-based) — always works, no extra deps

webrtcvad requires audio chunks of exactly 10ms, 20ms, or 30ms at
8000, 16000, or 32000 Hz. We use 20ms frames at 16kHz → 320 samples.

Energy fallback uses the same approach as the existing VoiceEngine
``_record_until_silence()`` method but is extracted here for reuse
and testability.

Usage
-----
    vad = VoiceActivityDetector(
        sample_rate=16000,
        silence_timeout_seconds=1.5,
        aggressiveness=2,
    )
    audio = vad.collect_until_silence(
        buffer_lock=engine._buffer_lock,
        buffer=engine._capture_buffer,
        stop_event=engine._stop_event,
        max_seconds=10.0,
    )
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from spidy.logging.logger import get_logger

log = get_logger(__name__)

# webrtcvad frame parameters
_VAD_FRAME_MS = 20          # 20ms frames
_VAD_SAMPLE_RATE = 16000
_VAD_FRAME_SAMPLES = int(_VAD_SAMPLE_RATE * _VAD_FRAME_MS / 1000)   # 320 samples
_ENERGY_SILENCE_THRESHOLD = 0.01   # ~-40 dBFS


class VoiceActivityDetector:
    """
    Enhanced VAD that bridges the recording buffer and the STT engine.

    Parameters
    ----------
    sample_rate:
        Audio sample rate in Hz. Must be 8000, 16000, or 32000 for
        webrtcvad. Defaults to 16000.
    silence_timeout_seconds:
        Stop recording after this many consecutive seconds of silence.
        Default: 1.5 s.
    aggressiveness:
        webrtcvad aggressiveness level 0–3 (0=lenient, 3=aggressive).
        Higher values filter more background noise. Default: 2.
    min_speech_duration_ms:
        Minimum speech duration before considering it valid input.
        Prevents very short sounds from triggering transcription.
        Default: 250 ms.
    energy_threshold:
        RMS amplitude below which a chunk is considered silent (fallback).
        Default: 0.01.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_timeout_seconds: float = 1.5,
        aggressiveness: int = 2,
        min_speech_duration_ms: int = 250,
        energy_threshold: float = _ENERGY_SILENCE_THRESHOLD,
    ) -> None:
        self._sample_rate = sample_rate
        self._silence_timeout = silence_timeout_seconds
        self._aggressiveness = aggressiveness
        self._min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)
        self._energy_threshold = energy_threshold

        # Try to load webrtcvad
        self._vad = self._load_webrtcvad()

    # ── Public API ────────────────────────────────────────────────────────

    def collect_until_silence(
        self,
        buffer_lock: threading.Lock,
        buffer: list[np.ndarray],
        stop_event: threading.Event,
        max_seconds: float = 10.0,
    ) -> np.ndarray | None:
        """
        Drain ``buffer`` until silence is detected or max_seconds reached.

        This method runs synchronously and should be called via
        ``asyncio.to_thread()`` from the async VoiceEngine loop.

        Parameters
        ----------
        buffer_lock:
            Threading lock protecting the buffer.
        buffer:
            Mutable list of np.ndarray audio chunks.  This method drains it.
        stop_event:
            Set when the engine is shutting down — causes early return.
        max_seconds:
            Maximum recording time before forced stop.

        Returns
        -------
        np.ndarray or None
            Concatenated speech audio, or None if nothing was captured.
        """
        if self._vad is not None:
            return self._collect_webrtcvad(
                buffer_lock, buffer, stop_event, max_seconds
            )
        return self._collect_energy(
            buffer_lock, buffer, stop_event, max_seconds
        )

    @property
    def using_webrtcvad(self) -> bool:
        """True if webrtcvad is active, False if using energy fallback."""
        return self._vad is not None

    # ── webrtcvad implementation ──────────────────────────────────────────

    def _collect_webrtcvad(
        self,
        buffer_lock: threading.Lock,
        buffer: list[np.ndarray],
        stop_event: threading.Event,
        max_seconds: float,
    ) -> np.ndarray | None:
        """Collect audio with webrtcvad silence detection."""
        assert self._vad is not None

        start_time = time.monotonic()
        all_chunks: list[np.ndarray] = []
        silence_start: float | None = None
        speech_samples = 0

        while True:
            if stop_event.is_set():
                return None
            if time.monotonic() - start_time >= max_seconds:
                log.debug("VAD: max recording duration reached.")
                break

            # Drain buffer
            with buffer_lock:
                new_chunks = list(buffer)
                buffer.clear()

            for chunk in new_chunks:
                all_chunks.append(chunk)
                speech_samples += len(chunk)

                # Process in VAD-sized frames (320 samples = 20ms at 16kHz)
                int16_chunk = (chunk * 32767).astype(np.int16)
                is_voiced = self._is_voiced_chunk(int16_chunk)

                if is_voiced:
                    silence_start = None
                else:
                    if silence_start is None:
                        silence_start = time.monotonic()
                    elapsed_silence = time.monotonic() - silence_start
                    if elapsed_silence >= self._silence_timeout:
                        log.debug(
                            "VAD(webrtcvad): silence detected after {s:.2f}s.",
                            s=elapsed_silence,
                        )
                        break
            else:
                time.sleep(0.01)
                continue
            break

        if not all_chunks:
            return None

        result = np.concatenate(all_chunks)

        # Discard very short captures (less than min_speech_duration_ms)
        if len(result) < self._min_speech_samples:
            log.debug("VAD: capture too short ({n} samples), discarding.", n=len(result))
            return None

        return result

    def _is_voiced_chunk(self, int16_chunk: np.ndarray) -> bool:
        """Check if the chunk contains speech using webrtcvad."""
        assert self._vad is not None
        # Process in _VAD_FRAME_SAMPLES-sized frames
        for i in range(0, len(int16_chunk) - _VAD_FRAME_SAMPLES, _VAD_FRAME_SAMPLES):
            frame = int16_chunk[i: i + _VAD_FRAME_SAMPLES]
            frame_bytes = frame.tobytes()
            try:
                if self._vad.is_speech(frame_bytes, self._sample_rate):
                    return True
            except Exception:  # noqa: BLE001
                # webrtcvad raises on wrong frame length — skip
                continue
        return False

    # ── Energy fallback implementation ────────────────────────────────────

    def _collect_energy(
        self,
        buffer_lock: threading.Lock,
        buffer: list[np.ndarray],
        stop_event: threading.Event,
        max_seconds: float,
    ) -> np.ndarray | None:
        """Collect audio using energy-based silence detection (no deps)."""
        start_time = time.monotonic()
        all_chunks: list[np.ndarray] = []
        consecutive_silence = 0
        chunk_size_approx = 1280  # Default openWakeWord chunk size

        # Number of consecutive silent chunks = timeout / chunk_duration
        # chunk_duration = chunk_size / sample_rate
        chunk_dur = chunk_size_approx / self._sample_rate
        silence_chunks_needed = int(self._silence_timeout / chunk_dur)

        while True:
            if stop_event.is_set():
                return None
            if time.monotonic() - start_time >= max_seconds:
                log.debug("VAD(energy): max duration reached.")
                break

            with buffer_lock:
                new_chunks = list(buffer)
                buffer.clear()

            for chunk in new_chunks:
                all_chunks.append(chunk)
                energy = float(np.sqrt(np.mean(chunk ** 2)))
                if energy < self._energy_threshold:
                    consecutive_silence += 1
                else:
                    consecutive_silence = 0

                if consecutive_silence >= silence_chunks_needed:
                    log.debug(
                        "VAD(energy): silence after {n} chunks.",
                        n=len(all_chunks),
                    )
                    break
            else:
                time.sleep(0.01)
                continue
            break

        if not all_chunks:
            return None

        result = np.concatenate(all_chunks)
        if len(result) < self._min_speech_samples:
            log.debug("VAD(energy): capture too short, discarding.")
            return None

        return result

    # ── Helper ────────────────────────────────────────────────────────────

    def _load_webrtcvad(self):
        """Try to import webrtcvad. Returns None if not installed."""
        try:
            import webrtcvad  # type: ignore[import]
            vad = webrtcvad.Vad(self._aggressiveness)
            log.debug("VAD: webrtcvad loaded (aggressiveness={a}).", a=self._aggressiveness)
            return vad
        except ImportError:
            log.debug(
                "VAD: webrtcvad not installed — using energy fallback. "
                "Install with: pip install webrtcvad-wheels"
            )
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("VAD: webrtcvad failed to load ({exc}) — using energy fallback.", exc=exc)
            return None
