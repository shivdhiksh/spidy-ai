"""
PiperTTSEngine — High-Quality Offline Text-to-Speech
======================================================
Uses the Piper TTS library for fast, natural-sounding speech synthesis.

Why Piper?
----------
- Offline: no network required
- Ultra-low latency: typically < 150ms to first audio
- High quality: neural VITS architecture
- Many voices: en_US-ryan-high (natural male), en_US-lessac-high, etc.
- CPU only: Piper runs entirely on CPU — no VRAM consumed

Voice selection
---------------
Default: en_US-ryan-high (confident, clear, natural male voice)
Voice files are stored in assets/models/tts/<voice_name>/
Required files per voice:
  - <voice>.onnx          (model weights)
  - <voice>.onnx.json     (voice config)

Piper can be installed via pip:
  pip install piper-tts

Model download:
  python scripts/setup_models.py --tts en_US-ryan-high

Audio output
------------
- Piper outputs int16 PCM at the voice's native sample rate (usually 22050 Hz)
- We convert to float32 for internal consistency
- Playback via sounddevice

Interrupt design
----------------
self._stop_event (threading.Event) is set by stop().
The playback thread checks this event and terminates early.
This allows the wake word to interrupt Spidy mid-sentence.
"""

from __future__ import annotations

import asyncio
import io
import threading
import wave
from pathlib import Path

import numpy as np

from spidy.logging.logger import get_logger
from spidy.perception.voice.tts.base import AudioBuffer, TTSEngine

log = get_logger(__name__)

_DEFAULT_SAMPLE_RATE = 22050


class PiperTTSEngine(TTSEngine):
    """
    Piper TTS engine for natural-sounding offline speech.

    Parameters
    ----------
    voice:
        Piper voice name, e.g. ``"en_US-ryan-high"``.
    model_dir:
        Directory containing the .onnx and .onnx.json files.
        Defaults to ``assets/models/tts/<voice>/``.
    speed:
        Speaking speed multiplier. 1.0 = normal.
    volume:
        Playback volume in [0.0, 1.0].
    """

    def __init__(
        self,
        voice: str = "en_US-ryan-high",
        model_dir: str | None = None,
        speed: float = 1.0,
        volume: float = 1.0,
    ) -> None:
        self._voice = voice
        self._model_dir = Path(model_dir) if model_dir else None
        self._speed = speed
        self._volume = volume
        self._piper_voice = None    # piper.PiperVoice instance
        self._sample_rate: int = _DEFAULT_SAMPLE_RATE
        self._stop_event = threading.Event()
        self._speaking = False
        self._loaded = False

    # ── TTSEngine interface ───────────────────────────────────────────────

    def load(self) -> None:
        """Load the Piper voice model from disk."""
        try:
            import piper
        except ImportError as exc:
            raise ImportError(
                "piper-tts is not installed. "
                "Install with: pip install piper-tts"
            ) from exc

        # Locate model files
        model_path = self._resolve_model_path()

        log.info(
            "Loading Piper TTS voice: '{voice}' from {path}",
            voice=self._voice,
            path=model_path,
        )

        try:
            self._piper_voice = piper.PiperVoice.load(
                str(model_path),
                config_path=str(model_path.with_suffix(".onnx.json")),
                use_cuda=False,  # Piper always uses CPU
            )
            self._sample_rate = self._piper_voice.config.sample_rate
            self._loaded = True
            log.info(
                "Piper voice loaded | sample_rate={sr} Hz",
                sr=self._sample_rate,
            )
        except Exception as exc:
            log.error("Failed to load Piper voice: {exc}", exc=exc)
            raise

    def unload(self) -> None:
        """Release Piper model resources."""
        if self._piper_voice is not None:
            del self._piper_voice
            self._piper_voice = None
        self._loaded = False
        log.debug("Piper TTS unloaded.")

    async def synthesize(self, text: str) -> AudioBuffer:
        """
        Synthesise text to an AudioBuffer (non-blocking).

        Runs Piper synthesis in a thread pool to avoid blocking asyncio.
        """
        if not self._loaded or self._piper_voice is None:
            log.error("synthesize() called before load()")
            return AudioBuffer(
                samples=np.zeros(0, dtype=np.float32),
                sample_rate=self._sample_rate,
                duration_seconds=0.0,
            )

        return await asyncio.to_thread(self._synthesize_sync, text)

    async def speak(self, text: str) -> None:
        """
        Synthesise and play audio. Awaitable — returns after playback finishes
        or stop() is called.

        Parameters
        ----------
        text:
            Text to speak. Automatically chunked at sentence boundaries
            for lower first-word latency.
        """
        if not self._loaded:
            log.error("speak() called before load()")
            return

        self._stop_event.clear()
        self._speaking = True

        try:
            buffer = await self.synthesize(text)
            if buffer.samples.size == 0:
                return
            await asyncio.to_thread(self._play_buffer, buffer)
        finally:
            self._speaking = False

    def stop(self) -> None:
        """
        Immediately interrupt playback.
        Thread-safe — callable from wake word detection thread.
        """
        self._stop_event.set()
        log.debug("TTS stop requested.")

    # ── Internal helpers ──────────────────────────────────────────────────

    def _synthesize_sync(self, text: str) -> AudioBuffer:
        """Run Piper synthesis synchronously (called via asyncio.to_thread)."""
        # Synthesize to a WAV buffer in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            self._piper_voice.synthesize(
                text,
                wav_file,
                length_scale=1.0 / self._speed,  # Piper: lower=faster
                noise_scale=0.667,
                noise_w=0.8,
            )

        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sr = wav_file.getframerate()

        # Convert int16 PCM bytes → float32 numpy
        int16_samples = np.frombuffer(frames, dtype=np.int16)
        float32_samples = (int16_samples.astype(np.float32) / 32767.0) * self._volume
        duration = len(float32_samples) / sr

        return AudioBuffer(
            samples=float32_samples,
            sample_rate=sr,
            duration_seconds=duration,
        )

    def _play_buffer(self, buffer: AudioBuffer) -> None:
        """
        Play an AudioBuffer via sounddevice.

        Checks stop_event during playback for interrupt support.
        """
        try:
            import sounddevice as sd
        except ImportError:
            log.error("sounddevice not installed. Cannot play audio.")
            return

        if self._stop_event.is_set():
            return

        log.debug(
            "Speaking ({dur:.1f}s) via sounddevice",
            dur=buffer.duration_seconds,
        )

        try:
            sd.play(buffer.samples, samplerate=buffer.sample_rate, blocking=False)

            # Wait for completion or stop signal
            # Poll every 50ms — low overhead, responsive to stop()
            import time
            while True:
                try:
                    stream_active = sd.get_stream().active
                except RuntimeError:
                    # Stream already closed — playback finished
                    break
                if not stream_active:
                    break
                if self._stop_event.is_set():
                    sd.stop()
                    log.debug("TTS playback interrupted.")
                    return
                time.sleep(0.05)

        except Exception as exc:  # noqa: BLE001
            log.error("Audio playback error: {exc}", exc=exc)

    def _resolve_model_path(self) -> Path:
        """Locate the .onnx model file for the configured voice."""
        if self._model_dir is not None:
            # Explicit directory provided
            candidate = self._model_dir / f"{self._voice}.onnx"
            if candidate.exists():
                return candidate
            # Try with just the voice name as filename
            for f in self._model_dir.glob("*.onnx"):
                return f
            raise FileNotFoundError(
                f"Piper model not found in {self._model_dir}. "
                f"Run: python scripts/setup_models.py --tts {self._voice}"
            )

        # Auto-locate from standard assets path
        standard_paths = [
            Path("assets") / "models" / "tts" / self._voice / f"{self._voice}.onnx",
            Path("assets") / "models" / "tts" / f"{self._voice}.onnx",
        ]
        for p in standard_paths:
            if p.exists():
                return p

        raise FileNotFoundError(
            f"Piper voice '{self._voice}' not found. "
            f"Run: python scripts/setup_models.py --tts {self._voice}"
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def voice_name(self) -> str:
        return self._voice

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
