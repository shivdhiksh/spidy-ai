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
import time
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
        self._stream = None          # persistent sounddevice OutputStream
        self._stream_lock = threading.Lock()

    # ── TTSEngine interface ───────────────────────────────────────────────

    def load(self) -> None:
        """Load the Piper voice model from disk and pre-warm the engine."""
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
            # Pre-warm Piper ONNX execution graph so user queries avoid the ~2.5s cold penalty
            try:
                self._synthesize_sync("ready")
                self._ensure_stream()
                log.info("Piper TTS engine pre-warmed and output stream initialized.")
            except Exception as warm_exc:
                log.debug("Piper pre-warm non-fatal warning: {exc}", exc=warm_exc)
        except Exception as exc:
            log.error("Failed to load Piper voice: {exc}", exc=exc)
            raise

    def unload(self) -> None:
        """Release Piper model and audio stream resources."""
        with self._stream_lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
        if self._piper_voice is not None:
            del self._piper_voice
            self._piper_voice = None
        self._loaded = False
        log.debug("Piper TTS unloaded.")

    def _ensure_stream(self) -> None:
        """Ensure persistent low-latency sounddevice OutputStream is active."""
        import sounddevice as sd
        with self._stream_lock:
            if self._stream is None or self._stream.closed:
                self._stream = sd.OutputStream(
                    samplerate=self._sample_rate,
                    channels=1,
                    dtype="float32",
                    latency="low",
                )
                self._stream.start()

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

        t0 = time.monotonic()
        buf = await asyncio.to_thread(self._synthesize_sync, text)
        elapsed = (time.monotonic() - t0) * 1000
        log.debug("[TTS] synth_complete: ms={ms:.0f}ms | dur={dur:.2f}s | text='{t}'",
                  ms=elapsed, dur=buf.duration_seconds, t=text[:40])
        return buf

    async def play_buffer(self, buffer: AudioBuffer) -> None:
        """
        Play a pre-synthesised AudioBuffer directly without re-synthesis.
        """
        if not self._loaded:
            log.error("play_buffer() called before load()")
            return

        if buffer.samples.size == 0 or self._stop_event.is_set():
            return

        self._stop_event.clear()
        self._speaking = True
        try:
            await asyncio.to_thread(self._play_buffer, buffer)
        finally:
            self._speaking = False

    async def speak(self, text: str) -> None:
        """
        Synthesise and play audio. Awaitable — returns after playback finishes
        or stop() is called.
        """
        if not self._loaded:
            log.error("speak() called before load()")
            return

        t_total_start = time.monotonic()
        self._stop_event.clear()
        self._speaking = True

        try:
            t0 = time.monotonic()
            buffer = await self.synthesize(text)
            syn_ms = (time.monotonic() - t0) * 1000

            if buffer.samples.size == 0 or self._stop_event.is_set():
                return

            t1 = time.monotonic()
            await asyncio.to_thread(self._play_buffer, buffer)
            play_ms = (time.monotonic() - t1) * 1000
            total_ms = (time.monotonic() - t_total_start) * 1000

            log.info(
                "[TTS] synth_complete={syn:.0f}ms | playback_complete={play:.0f}ms | "
                "audio_dur={dur:.0f}ms | total={tot:.0f}ms",
                syn=syn_ms,
                play=play_ms,
                dur=buffer.duration_seconds * 1000,
                tot=total_ms,
            )
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
        from piper.config import SynthesisConfig

        syn_config = SynthesisConfig(
            length_scale=1.0 / self._speed,
            noise_scale=0.667,
            noise_w_scale=0.8,
            volume=self._volume,
        )

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            self._piper_voice.synthesize_wav(text, wav_file, syn_config=syn_config)

        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sr = wav_file.getframerate()

        int16_samples = np.frombuffer(frames, dtype=np.int16)
        float32_samples = int16_samples.astype(np.float32) / 32767.0
        duration = len(float32_samples) / sr if sr > 0 else 0.0

        return AudioBuffer(
            samples=float32_samples,
            sample_rate=sr,
            duration_seconds=duration,
        )

    def _play_buffer(self, buffer: AudioBuffer) -> None:
        """
        Play an AudioBuffer via persistent sounddevice OutputStream.

        Writes in 1024-frame chunks, monitoring stop_event for instant cutoff.
        """
        try:
            import sounddevice as sd
        except ImportError:
            log.error("sounddevice not installed. Cannot play audio.")
            return

        if self._stop_event.is_set() or buffer.samples.size == 0:
            return

        try:
            self._ensure_stream()
            chunk_size = 1024
            samples = buffer.samples.reshape(-1, 1)
            total_samples = len(samples)
            offset = 0

            while offset < total_samples:
                if self._stop_event.is_set():
                    log.debug("TTS playback interrupted mid-buffer.")
                    return
                end = min(offset + chunk_size, total_samples)
                chunk = samples[offset:end]
                with self._stream_lock:
                    if self._stream and not self._stream.closed:
                        self._stream.write(chunk)
                    else:
                        break
                offset = end

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
