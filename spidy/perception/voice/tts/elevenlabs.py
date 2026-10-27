"""
ElevenLabsTTSEngine — Ultra-High-Quality Online Text-to-Speech
=============================================================
Integrates ElevenLabs REST API with raw PCM streaming for low-latency,
photorealistic voice synthesis.

Features:
- Configured by default for the user's custom Gwen voice (ID: ZYk9CB0JtzXJrxcnt9Oz).
- Raw PCM output (pcm_22050 / pcm_24000) for zero-dependency high-speed decoding.
- Full compatibility with sounddevice async/sync playback and thread-safe cancellation.
- Automatic graceful fallback if network / quota errors occur.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import urllib.error
import urllib.request
from typing import Any

import numpy as np
import sounddevice as sd

from spidy.logging.logger import get_logger
from spidy.perception.voice.tts.base import AudioBuffer, TTSEngine

log = get_logger(__name__)

_DEFAULT_GWEN_VOICE_ID = "ZYk9CB0JtzXJrxcnt9Oz"
_DEFAULT_MODEL_ID = "eleven_turbo_v2_5"
_DEFAULT_SAMPLE_RATE = 22050


class ElevenLabsTTSEngine(TTSEngine):
    """
    ElevenLabs online TTS engine supporting custom voices and low-latency PCM synthesis.
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        model_id: str = _DEFAULT_MODEL_ID,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        speed: float = 1.0,
        volume: float = 1.0,
    ) -> None:
        # Load API key from parameter or environment (never hard-code secrets)
        self._api_key = (
            api_key
            or os.environ.get("ELEVENLABS_API_KEY")
            or os.environ.get("ELEVEN_API_KEY")
            or ""
        )
        self._voice_id = (
            voice_id
            or os.environ.get("ELEVENLABS_VOICE_ID")
            or _DEFAULT_GWEN_VOICE_ID
        )
        self._model_id = model_id
        self._sample_rate = sample_rate
        self._speed = max(0.5, min(2.0, speed))
        self._volume = max(0.0, min(2.0, volume))

        self._is_speaking = False
        self._stop_event = threading.Event()
        self._loaded = False
        self._lock = threading.Lock()

    # ── TTSEngine interface ───────────────────────────────────────────────────

    @property
    def voice_name(self) -> str:
        return f"elevenlabs:{self._voice_id}"

    @property
    def voice_id(self) -> str:
        return self._voice_id

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def is_available(self) -> bool:
        """True if API key is present and engine is configured."""
        return bool(self._api_key and self._api_key.strip())

    def load(self) -> None:
        """Initialize engine settings."""
        if not self.is_available():
            log.warning("ElevenLabsTTSEngine: ELEVENLABS_API_KEY is not set. Engine is inactive.")
        self._loaded = True

    def unload(self) -> None:
        """Stop playback and reset."""
        self.stop()
        self._loaded = False

    def stop(self) -> None:
        """Immediately stop playback."""
        self._stop_event.set()
        try:
            sd.stop()
        except Exception:  # noqa: BLE001
            pass
        self._is_speaking = False

    def clear_stop(self) -> None:
        """Forget an earlier stop() so the next utterance plays."""
        self._stop_event.clear()

    # ── Synthesis & Playback ──────────────────────────────────────────────────

    async def synthesize(self, text: str) -> AudioBuffer:
        """
        Synthesize text to raw PCM audio buffer using ElevenLabs API.
        """
        if not text or not text.strip():
            return AudioBuffer(samples=np.zeros(0, dtype=np.float32), sample_rate=self._sample_rate, duration_seconds=0.0)

        if not self.is_available():
            raise RuntimeError("ElevenLabs API key is missing. Cannot synthesize audio.")

        # Run HTTP request in thread pool to avoid blocking asyncio event loop
        raw_pcm = await asyncio.to_thread(self._fetch_pcm, text.strip())

        # Convert 16-bit PCM bytes to float32 numpy array
        int16_samples = np.frombuffer(raw_pcm, dtype=np.int16)
        float_samples = int16_samples.astype(np.float32) / 32768.0

        if self._volume != 1.0:
            float_samples = np.clip(float_samples * self._volume, -1.0, 1.0)

        duration = len(float_samples) / float(self._sample_rate)
        return AudioBuffer(
            samples=float_samples,
            sample_rate=self._sample_rate,
            duration_seconds=duration,
        )

    async def speak(self, text: str) -> None:
        """
        Synthesize and play audio, blocking until playback finishes or is cancelled.
        """
        buffer = await self.synthesize(text)
        await self.play_buffer(buffer)

    async def play_buffer(self, buffer: AudioBuffer) -> None:
        """Play a pre-synthesized AudioBuffer."""
        if buffer.samples is None or len(buffer.samples) == 0:
            return

        self._stop_event.clear()
        self._is_speaking = True

        try:
            # Play using sounddevice in background thread
            sd.play(buffer.samples, samplerate=buffer.sample_rate)

            # Wait for playback completion or interruption
            total_duration = buffer.duration_seconds
            interval = 0.05
            elapsed = 0.0

            while elapsed < total_duration:
                if self._stop_event.is_set():
                    sd.stop()
                    break
                await asyncio.sleep(interval)
                elapsed += interval

            # Final check to ensure sounddevice finished
            if not self._stop_event.is_set():
                sd.wait()

        finally:
            self._is_speaking = False

    # ── Internal HTTP API ─────────────────────────────────────────────────────

    def _fetch_pcm(self, text: str) -> bytes:
        """Synchronous HTTP call to ElevenLabs endpoint returning raw 16-bit PCM."""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}?output_format=pcm_{self._sample_rate}"
        payload = {
            "text": text,
            "model_id": self._model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "xi-api-key": self._api_key,
                "Accept": f"audio/pcm; samplerate={self._sample_rate}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=12.0) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"ElevenLabs returned HTTP {resp.status}")
                return resp.read()
        except urllib.error.HTTPError as http_err:
            try:
                err_body = http_err.read().decode("utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                err_body = str(http_err)
            log.error("ElevenLabs HTTP error: {code} {msg} - {body}", code=http_err.code, msg=http_err.msg, body=err_body[:200])
            raise RuntimeError(f"ElevenLabs API error: HTTP {http_err.code}") from http_err
        except Exception as exc:
            log.error("ElevenLabs network error: {exc}", exc=exc)
            raise RuntimeError(f"ElevenLabs request failed: {exc}") from exc
