"""
TTSManager — Centralized Hybrid Text-to-Speech Architecture
===========================================================
Coordinates online (ElevenLabs Gwen voice) and offline (Piper) TTS engines.

Architecture:
             SPIDY
                |
             TTSManager
                |
        ┌───────┴────────┐
        │                │
     ONLINE           OFFLINE
        │                │
  ElevenLabs           Piper
        │
 Existing Gwen
   Voice ID:
ZYk9CB0JtzXJrxcnt9Oz

Guarantees:
1. Online ElevenLabs Gwen voice is used whenever available and connected.
2. If internet / ElevenLabs is offline or fails, automatically falls back to Piper.
3. Does not expose API keys or internal provider errors to the user.
4. Seamlessly routes `speak()`, `synthesize()`, `stop()`, and `play_buffer()`.
5. Pre-caching and wake acknowledgements operate seamlessly.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from spidy.logging.logger import get_logger
from spidy.perception.voice.tts.base import AudioBuffer, TTSEngine

if TYPE_CHECKING:
    from spidy.perception.voice.tts.elevenlabs import ElevenLabsTTSEngine
    from spidy.perception.voice.tts.piper import PiperTTSEngine

log = get_logger(__name__)


class TTSManager(TTSEngine):
    """
    Centralized TTS Manager providing online ElevenLabs synthesis with offline Piper fallback.
    """

    def __init__(
        self,
        primary_engine: "TTSEngine | None" = None,
        fallback_engine: "TTSEngine | None" = None,
        prefer_online: bool = True,
    ) -> None:
        self._primary = primary_engine
        self._fallback = fallback_engine
        self._prefer_online = prefer_online
        self._active_engine: TTSEngine | None = primary_engine or fallback_engine
        self._is_online_available = False

    # ── TTSEngine Interface ───────────────────────────────────────────────────

    @property
    def voice_name(self) -> str:
        if self._active_engine is not None:
            return self._active_engine.voice_name
        return "tts_manager:uninitialized"

    @property
    def is_speaking(self) -> bool:
        if self._primary and self._primary.is_speaking:
            return True
        if self._fallback and self._fallback.is_speaking:
            return True
        return False

    def get_active_engine(self) -> TTSEngine | None:
        """Return the currently active TTS engine."""
        return self._active_engine

    def load(self) -> None:
        """Load both engines if available and log startup diagnostic block."""
        if self._fallback is not None:
            try:
                self._fallback.load()
            except Exception as exc:  # noqa: BLE001
                log.warning("TTSManager: Fallback engine load error: {exc}", exc=exc)

        primary_voice_id = "ZYk9CB0JtzXJrxcnt9Oz"
        if self._primary is not None:
            try:
                self._primary.load()
                primary_voice_id = getattr(self._primary, "voice_id", primary_voice_id)
                if hasattr(self._primary, "is_available") and self._primary.is_available():
                    self._is_online_available = True
                    self._active_engine = self._primary
                else:
                    self._is_online_available = False
                    self._active_engine = self._fallback
            except Exception as exc:  # noqa: BLE001
                log.warning("TTSManager: Primary engine load error: {exc}", exc=exc)
                self._active_engine = self._fallback

        fallback_voice_name = getattr(self._fallback, "voice_name", "en_US-ryan-high")
        if fallback_voice_name.startswith("piper:"):
            fallback_voice_name = fallback_voice_name[len("piper:"):]

        log.info("[TTS] Manager initialized")
        log.info("[TTS] Primary provider: ElevenLabs")
        log.info("[TTS] Primary voice: Gwen")
        log.info("[TTS] Voice ID: {vid}", vid=primary_voice_id)
        log.info("[TTS] Fallback provider: Piper")
        log.info("[TTS] Fallback voice: {fv}", fv=fallback_voice_name)

    def unload(self) -> None:
        """Unload both engines."""
        if self._primary is not None:
            self._primary.unload()
        if self._fallback is not None:
            self._fallback.unload()

    def stop(self) -> None:
        """Stop playback across all active engines."""
        if self._primary is not None:
            self._primary.stop()
        if self._fallback is not None:
            self._fallback.stop()

    def clear_stop(self) -> None:
        """Forget an earlier stop() on every engine so the next utterance plays."""
        for engine in (self._primary, self._fallback):
            clear = getattr(engine, "clear_stop", None)
            if callable(clear):
                clear()

    def _synthesize_sync(self, text: str) -> AudioBuffer:
        """
        Synchronous synthesis helper for pre-caching wake acknowledgements.
        """
        if self._fallback is not None and hasattr(self._fallback, "_synthesize_sync"):
            return self._fallback._synthesize_sync(text)
        if self._primary is not None and hasattr(self._primary, "_fetch_pcm"):
            try:
                raw_pcm = self._primary._fetch_pcm(text)
                import numpy as np
                int16_samples = np.frombuffer(raw_pcm, dtype=np.int16)
                float_samples = int16_samples.astype(np.float32) / 32768.0
                return AudioBuffer(
                    samples=float_samples,
                    sample_rate=getattr(self._primary, "_sample_rate", 22050),
                    duration_seconds=len(float_samples) / float(getattr(self._primary, "_sample_rate", 22050)),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("TTSManager: Sync pre-cache error: {exc}", exc=exc)

        import numpy as np
        return AudioBuffer(samples=np.zeros(0, dtype=np.float32), sample_rate=22050, duration_seconds=0.0)

    # ── Synthesis & Speech Dispatch ───────────────────────────────────────────

    async def synthesize(self, text: str) -> AudioBuffer:
        """
        Synthesize text using the primary online engine, falling back to offline engine on failure.
        """
        if not text or not text.strip():
            import numpy as np
            return AudioBuffer(samples=np.zeros(0, dtype=np.float32), sample_rate=22050, duration_seconds=0.0)

        # 1. Attempt Primary Online Engine if available and preferred
        if self._prefer_online and self._primary is not None:
            is_ready = True
            if hasattr(self._primary, "is_available"):
                is_ready = self._primary.is_available()

            if is_ready:
                try:
                    log.info("[TTS] provider=elevenlabs voice=gwen")
                    buffer = await self._primary.synthesize(text)
                    self._active_engine = self._primary
                    return buffer
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "[TTS] ElevenLabs unavailable ({exc}) → fallback=piper",
                        exc=exc,
                    )

        # 2. Fallback to Offline Piper Engine
        if self._fallback is not None:
            log.info("[TTS] provider=piper voice={v}", v=self._fallback.voice_name)
            self._active_engine = self._fallback
            return await self._fallback.synthesize(text)

        raise RuntimeError("TTSManager: No available TTS engine could synthesize audio.")

    async def speak(self, text: str) -> None:
        """
        Speak text with seamless online -> offline fallback.
        """
        if not text or not text.strip():
            return

        # 1. Attempt Primary Online Engine
        if self._prefer_online and self._primary is not None:
            is_ready = True
            if hasattr(self._primary, "is_available"):
                is_ready = self._primary.is_available()

            if is_ready:
                try:
                    log.info("[TTS] provider=elevenlabs voice=gwen")
                    await self._primary.speak(text)
                    self._active_engine = self._primary
                    return
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "[TTS] ElevenLabs unavailable ({exc}) → fallback=piper",
                        exc=exc,
                    )

        # 2. Fallback to Offline Piper Engine
        if self._fallback is not None:
            log.info("[TTS] provider=piper voice={v}", v=self._fallback.voice_name)
            self._active_engine = self._fallback
            await self._fallback.speak(text)
            return

        log.warning("TTSManager: No TTS engine available to speak text.")

    async def play_buffer(self, buffer: AudioBuffer) -> None:
        """Play pre-synthesized audio buffer using active engine or fallback."""
        if self._active_engine is not None:
            await self._active_engine.play_buffer(buffer)
        elif self._fallback is not None:
            await self._fallback.play_buffer(buffer)
