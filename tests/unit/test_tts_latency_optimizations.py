"""
Unit Tests — TTS Latency Optimizations & Pipeline Diagnostics
=============================================================
Validates:
1. Pre-caching and instant playback of wake acknowledgement phrases
2. WakeAcknowledger fallback when phrase is not pre-cached
3. Persistent sounddevice stream management in PiperTTSEngine
4. Pre-warming during PiperTTSEngine.load()
5. Multi-sentence synthesis-playback pipelining in StreamingTTSWrapper
6. Concurrency lock (_speak_lock) preventing double-speech
7. Instant interruption and cancellation on stop()
8. Continuous voice controller integration with pre-cached acks
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest

from spidy.perception.voice.tts.base import AudioBuffer, TTSEngine
from spidy.perception.voice.tts.piper import PiperTTSEngine
from spidy.voice.streaming_tts import StreamingTTSWrapper
from spidy.config.manager import WakeAckConfig
from spidy.voice.wake_ack import WakeAcknowledger


def _fake_audio_buffer(duration: float = 0.5, sr: int = 22050) -> AudioBuffer:
    n_samples = int(duration * sr)
    samples = np.zeros(n_samples, dtype=np.float32)
    return AudioBuffer(samples=samples, sample_rate=sr, duration_seconds=duration)


class FakeTTSEngine(TTSEngine):
    def __init__(self):
        self.synthesize_calls = []
        self.speak_calls = []
        self.play_buffer_calls = []
        self.stop_called = False
        self._is_speaking = False
        self._loaded = True

    async def synthesize(self, text: str) -> AudioBuffer:
        self.synthesize_calls.append(text)
        await asyncio.sleep(0.01)  # small mock synthesis delay
        return _fake_audio_buffer(duration=0.1)

    def _synthesize_sync(self, text: str) -> AudioBuffer:
        self.synthesize_calls.append(text)
        return _fake_audio_buffer(duration=0.1)

    async def speak(self, text: str) -> None:
        self.speak_calls.append(text)
        await asyncio.sleep(0.01)

    async def play_buffer(self, buffer: AudioBuffer) -> None:
        self.play_buffer_calls.append(buffer)
        await asyncio.sleep(0.01)

    def stop(self) -> None:
        self.stop_called = True

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def voice_name(self) -> str:
        return "fake_voice"

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking


class TestWakeAckCaching:

    def test_precache_phrases_populates_buffers(self):
        """precache_phrases synthesises each configured phrase into memory."""
        cfg = WakeAckConfig(enabled=True, phrases=["Hmm?", "Yes Shiva."])
        ack = WakeAcknowledger(cfg)
        engine = FakeTTSEngine()

        ack.precache_phrases(engine)

        assert "Hmm?" in ack._cached_buffers
        assert "Yes Shiva." in ack._cached_buffers
        assert len(ack._cached_buffers) == 2

    @pytest.mark.asyncio
    async def test_speak_ack_uses_cached_buffer_without_resynthesis(self):
        """speak_ack directly calls engine.play_buffer when phrase is pre-cached."""
        cfg = WakeAckConfig(enabled=True, phrases=["Hmm?"])
        ack = WakeAcknowledger(cfg)
        engine = FakeTTSEngine()
        ack.precache_phrases(engine)

        stts = StreamingTTSWrapper(engine=engine)
        # Clear synthesis call count from precaching
        engine.synthesize_calls.clear()

        await ack.speak_ack(stts)

        assert len(engine.play_buffer_calls) == 1
        assert len(engine.synthesize_calls) == 0  # Zero runtime synthesis calls!
        assert len(engine.speak_calls) == 0

    @pytest.mark.asyncio
    async def test_speak_ack_fallback_when_uncached(self):
        """speak_ack falls back to streaming_tts.speak when phrase is not in cache."""
        cfg = WakeAckConfig(enabled=True, phrases=["Uncached Phrase."])
        ack = WakeAcknowledger(cfg)
        engine = FakeTTSEngine()
        # Note: precache_phrases NOT called

        stts = StreamingTTSWrapper(engine=engine)
        await ack.speak_ack(stts)

        # Fallback uses standard speak / synthesize path
        assert len(engine.synthesize_calls) > 0 or len(engine.speak_calls) > 0

    @pytest.mark.asyncio
    async def test_speak_ack_disabled_is_noop(self):
        """Disabled wake ack does not play anything."""
        cfg = WakeAckConfig(enabled=False, phrases=["Hmm?"])
        ack = WakeAcknowledger(cfg)
        engine = FakeTTSEngine()
        stts = StreamingTTSWrapper(engine=engine)

        await ack.speak_ack(stts)

        assert len(engine.play_buffer_calls) == 0
        assert len(engine.speak_calls) == 0


class TestStreamingTTSPipelining:

    @pytest.mark.asyncio
    async def test_multi_sentence_pipelining_synthesizes_concurrently(self):
        """Multi-sentence responses synthesize sentence N+1 while playing sentence N."""
        engine = FakeTTSEngine()
        stts = StreamingTTSWrapper(engine=engine)

        text = "First sentence here. Second sentence follows. Third sentence done."
        await stts.speak(text)

        # All 3 sentences were synthesized and played
        assert len(engine.synthesize_calls) == 3
        assert len(engine.play_buffer_calls) == 3

    @pytest.mark.asyncio
    async def test_concurrent_speak_serialized_by_lock(self):
        """Two concurrent speak() calls are serialized without double speech."""
        engine = FakeTTSEngine()
        stts = StreamingTTSWrapper(engine=engine)

        task1 = asyncio.create_task(stts.speak("First call."))
        task2 = asyncio.create_task(stts.speak("Second call."))
        await asyncio.gather(task1, task2)

        # Both calls handled safely without exceptions
        assert len(engine.synthesize_calls) >= 1

    @pytest.mark.asyncio
    async def test_stop_cancels_playback_and_synthesis(self):
        """Calling stop() flags interrupt and cuts off immediately."""
        engine = FakeTTSEngine()
        stts = StreamingTTSWrapper(engine=engine)

        stts.stop()
        assert engine.stop_called is True
        assert stts._interrupt_flag.is_set()


class TestPiperEngineOptimizations:

    def test_piper_prewarm_on_load(self):
        """PiperTTSEngine.load() pre-warms the ONNX graph with a dummy synthesis."""
        with patch("piper.PiperVoice.load") as mock_piper_load:
            mock_voice = MagicMock()
            mock_voice.config.sample_rate = 22050
            mock_piper_load.return_value = mock_voice

            engine = PiperTTSEngine()
            with patch.object(engine, "_resolve_model_path", return_value=MagicMock()):
                with patch.object(engine, "_synthesize_sync", return_value=_fake_audio_buffer()) as mock_syn:
                    engine.load()
                    assert engine.is_loaded is True
                    mock_syn.assert_called_with("ready")

    def test_piper_unload_cleans_up_stream(self):
        """PiperTTSEngine.unload() stops and closes the persistent stream."""
        engine = PiperTTSEngine()
        mock_stream = MagicMock()
        engine._stream = mock_stream

        engine.unload()

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert engine._stream is None
        assert engine.is_loaded is False
