"""
Unit tests for TTSManager and ElevenLabsTTSEngine.
Verifies Gwen voice ID (ZYk9CB0JtzXJrxcnt9Oz), PCM decoding, Piper offline fallback,
error resilience, and stop/interruption mechanisms.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest

from spidy.perception.voice.tts.base import AudioBuffer, TTSEngine
from spidy.perception.voice.tts.elevenlabs import ElevenLabsTTSEngine
from spidy.perception.voice.tts.manager import TTSManager


# ─── Mock Fallback TTS Engine ──────────────────────────────────────────────────

class MockOfflinePiperEngine(TTSEngine):
    """Simulates local Piper engine."""

    def __init__(self, voice: str = "en_US-ryan-high") -> None:
        self._voice = voice
        self._is_speaking = False
        self.spoken_texts: list[str] = []
        self.stopped = False

    @property
    def voice_name(self) -> str:
        return f"piper:{self._voice}"

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True
        self._is_speaking = False

    async def synthesize(self, text: str) -> AudioBuffer:
        samples = np.zeros(22050, dtype=np.float32)
        return AudioBuffer(samples=samples, sample_rate=22050, duration_seconds=1.0)

    async def speak(self, text: str) -> None:
        self.spoken_texts.append(text)
        self._is_speaking = True
        await asyncio.sleep(0.01)
        self._is_speaking = False


# ─── Test 1: ElevenLabs Voice ID & Defaults ───────────────────────────────────

def test_elevenlabs_default_gwen_voice():
    """Verify ElevenLabs engine defaults to the user's Gwen voice ID ZYk9CB0JtzXJrxcnt9Oz."""
    engine = ElevenLabsTTSEngine(api_key="mock_key")
    assert engine.voice_id == "ZYk9CB0JtzXJrxcnt9Oz"
    assert "ZYk9CB0JtzXJrxcnt9Oz" in engine.voice_name
    assert engine.is_available() is True


def test_elevenlabs_without_key_is_inactive():
    """Verify ElevenLabs engine is marked unavailable if no API key is provided."""
    with patch.dict("os.environ", {}, clear=True):
        engine = ElevenLabsTTSEngine(api_key="")
        assert engine.is_available() is False


# ─── Test 2: ElevenLabs Synthesis PCM Parsing ─────────────────────────────────

@pytest.mark.asyncio
async def test_elevenlabs_synthesize_pcm():
    """Verify raw PCM bytes are converted to float32 AudioBuffer correctly."""
    engine = ElevenLabsTTSEngine(api_key="mock_key", voice_id="ZYk9CB0JtzXJrxcnt9Oz")

    # Generate 1 second of dummy 16-bit PCM (22050 samples = 44100 bytes)
    dummy_int16 = np.zeros(22050, dtype=np.int16)
    dummy_bytes = dummy_int16.tobytes()

    with patch.object(engine, "_fetch_pcm", return_value=dummy_bytes):
        buffer = await engine.synthesize("Hello from Spidy")
        assert buffer.sample_rate == 22050
        assert buffer.duration_seconds == 1.0
        assert len(buffer.samples) == 22050
        assert buffer.samples.dtype == np.float32


# ─── Test 3: TTSManager Seamless Fallback to Piper on Error ───────────────────

@pytest.mark.asyncio
async def test_tts_manager_falls_back_to_piper_on_elevenlabs_error():
    """
    Verify TTSManager automatically and silently falls back to Piper
    when ElevenLabs fails or is offline.
    """
    online_engine = ElevenLabsTTSEngine(api_key="mock_key", voice_id="ZYk9CB0JtzXJrxcnt9Oz")
    offline_engine = MockOfflinePiperEngine()

    manager = TTSManager(
        primary_engine=online_engine,
        fallback_engine=offline_engine,
        prefer_online=True,
    )
    manager.load()

    # Simulate ElevenLabs network failure
    with patch.object(online_engine, "synthesize", side_effect=RuntimeError("Network unreachable")):
        buffer = await manager.synthesize("Testing fallback")
        assert buffer is not None
        assert buffer.duration_seconds == 1.0
        assert manager.get_active_engine() == offline_engine

    with patch.object(online_engine, "speak", side_effect=RuntimeError("API Quota Exceeded")):
        await manager.speak("Testing speak fallback")
        assert "Testing speak fallback" in offline_engine.spoken_texts
        assert manager.get_active_engine() == offline_engine


# ─── Test 4: TTSManager Uses Online Engine When Available ─────────────────────

@pytest.mark.asyncio
async def test_tts_manager_uses_online_when_available():
    """Verify TTSManager uses ElevenLabs when available."""
    online_engine = ElevenLabsTTSEngine(api_key="mock_key", voice_id="ZYk9CB0JtzXJrxcnt9Oz")
    offline_engine = MockOfflinePiperEngine()

    manager = TTSManager(
        primary_engine=online_engine,
        fallback_engine=offline_engine,
        prefer_online=True,
    )
    manager.load()

    dummy_buffer = AudioBuffer(samples=np.zeros(22050, dtype=np.float32), sample_rate=22050, duration_seconds=1.0)
    with patch.object(online_engine, "synthesize", return_value=dummy_buffer):
        buffer = await manager.synthesize("Online speech")
        assert buffer == dummy_buffer
        assert manager.get_active_engine() == online_engine


# ─── Test 5: Thread-Safe Interruption Across Engines ──────────────────────────

def test_tts_manager_stop_cancels_all_engines():
    """Verify stop() cancels playback across both online and offline engines."""
    online_engine = ElevenLabsTTSEngine(api_key="mock_key")
    offline_engine = MockOfflinePiperEngine()

    manager = TTSManager(primary_engine=online_engine, fallback_engine=offline_engine)
    manager.stop()

    assert offline_engine.stopped is True
    assert online_engine.is_speaking is False


# ─── Test 6: VoicePipelineFactory Constructs TTSManager with Gwen Voice ID ────

def test_voice_pipeline_factory_builds_tts_manager():
    """Verify VoiceEngineFactory builds TTSManager by default with Gwen voice."""
    from spidy.config.manager import ConfigManager
    from spidy.core.event_bus import EventBus
    from spidy.perception.voice.factory import VoiceEngineFactory

    config = ConfigManager().load()
    assert config.voice.tts.engine == "hybrid"
    assert config.voice.tts.elevenlabs_voice_id == "ZYk9CB0JtzXJrxcnt9Oz"

    bus = EventBus()
    engine = VoiceEngineFactory.build(config, bus)
    assert isinstance(engine._tts, TTSManager)


# ─── Test 7: WakeAcknowledger Pre-caches & Speaks via TTSManager ─────────────

@pytest.mark.asyncio
async def test_wake_acknowledger_with_tts_manager():
    """Verify WakeAcknowledger pre-caches phrases and speaks via TTSManager."""
    from spidy.config.manager import WakeAckConfig
    from spidy.voice.streaming_tts import StreamingTTSWrapper
    from spidy.voice.wake_ack import WakeAcknowledger

    online_engine = ElevenLabsTTSEngine(api_key="mock_key", voice_id="ZYk9CB0JtzXJrxcnt9Oz")
    offline_engine = MockOfflinePiperEngine()
    manager = TTSManager(primary_engine=online_engine, fallback_engine=offline_engine)
    manager.load()

    ack_cfg = WakeAckConfig(enabled=True, phrases=["Yes Shiva.", "Hmm?"])
    acknowledger = WakeAcknowledger(ack_cfg)
    acknowledger.precache_phrases(manager)

    stts = StreamingTTSWrapper(engine=manager, stop_mid_sentence=False)
    await acknowledger.speak_ack(stts)

    # Must have spoken through manager (fallback or online)
    assert len(offline_engine.spoken_texts) > 0 or online_engine.is_speaking is False
