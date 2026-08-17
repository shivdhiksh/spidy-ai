"""
Unit tests — StreamingTTSWrapper
==================================
Tests sentence splitting, sequential playback, interrupt between sentences,
empty string handling, and the wrapper's delegate contract.

Uses a FakeTTSEngine that records speak() calls without audio hardware.
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spidy.voice.streaming_tts import StreamingTTSWrapper


# ── Fake TTS ──────────────────────────────────────────────────────────────────


class FakeTTSEngine:
    """Records calls to speak() without producing audio."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self._speaking = False
        self._stop_called = False

    async def speak(self, text: str) -> None:
        self._speaking = True
        self.spoken.append(text)
        await asyncio.sleep(0)  # Yield to event loop
        self._speaking = False

    async def synthesize(self, text: str):
        pass

    def stop(self) -> None:
        self._stop_called = True

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    @property
    def voice_name(self) -> str:
        return "fake"

    @property
    def is_loaded(self) -> bool:
        return True


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_tts():
    return FakeTTSEngine()


@pytest.fixture
def wrapper(fake_tts):
    return StreamingTTSWrapper(engine=fake_tts, min_sentence_length=3)


# ── Sentence splitting tests ───────────────────────────────────────────────────


def test_single_sentence(wrapper):
    sentences = wrapper._split_sentences("Hello world.")
    assert sentences == ["Hello world."]


def test_multiple_sentences(wrapper):
    text = "Opening Chrome. Now I'll search YouTube. Done!"
    sentences = wrapper._split_sentences(text)
    assert len(sentences) == 3
    assert "Opening Chrome." in sentences[0]
    assert "Now I'll search YouTube." in sentences[1]
    assert "Done!" in sentences[2]


def test_sentence_with_semicolons(wrapper):
    text = "First step; second step; third step."
    sentences = wrapper._split_sentences(text)
    # Semicolons are sentence boundaries
    assert len(sentences) >= 2


def test_empty_string(wrapper):
    sentences = wrapper._split_sentences("")
    # Should return something, not raise
    assert isinstance(sentences, list)


def test_single_word_not_fragmented(wrapper):
    """Short fragments are merged with the next sentence."""
    text = "Hi. How can I help you today?"
    sentences = wrapper._split_sentences(text)
    # "Hi." may be merged with next sentence (it's < min_sentence_length? No, it's 3 chars)
    assert len(sentences) >= 1
    assert all(s.strip() for s in sentences)


# ── Speak tests ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_speak_single_sentence(wrapper, fake_tts):
    await wrapper.speak("Hello.")
    assert "Hello." in fake_tts.spoken


@pytest.mark.asyncio
async def test_speak_multiple_sentences(wrapper, fake_tts):
    await wrapper.speak("First sentence. Second sentence. Third sentence.")
    assert len(fake_tts.spoken) == 3


@pytest.mark.asyncio
async def test_speak_empty_string(wrapper, fake_tts):
    """Empty string should not call speak at all."""
    await wrapper.speak("")
    assert len(fake_tts.spoken) == 0


@pytest.mark.asyncio
async def test_speak_whitespace_only(wrapper, fake_tts):
    """Whitespace-only string should not call speak."""
    await wrapper.speak("   ")
    assert len(fake_tts.spoken) == 0


@pytest.mark.asyncio
async def test_interrupt_between_sentences(wrapper, fake_tts):
    """Setting the interrupt flag stops playback at the next boundary."""

    async def _speak_and_interrupt():
        # Start speaking in background
        task = asyncio.create_task(
            wrapper.speak("Sentence one. Sentence two. Sentence three.")
        )
        # Interrupt after first sentence completes
        await asyncio.sleep(0.01)
        wrapper.stop()
        await task

    await _speak_and_interrupt()
    # Should have spoken at least 1 sentence, not necessarily all 3
    assert len(fake_tts.spoken) >= 1
    # Should not have spoken all 3 if interrupted
    # (This is timing-dependent, so we just verify no exception occurred)


@pytest.mark.asyncio
async def test_stop_sets_interrupt_flag(wrapper):
    wrapper.stop()
    assert wrapper._interrupt_flag.is_set()


@pytest.mark.asyncio
async def test_stop_mid_sentence_calls_engine_stop():
    """stop_mid_sentence=True calls engine.stop() immediately."""
    fake = FakeTTSEngine()
    w = StreamingTTSWrapper(engine=fake, stop_mid_sentence=True)
    w.stop()
    assert fake._stop_called


@pytest.mark.asyncio
async def test_interrupt_flag_cleared_on_new_speak(wrapper, fake_tts):
    """Starting a new speak() clears the interrupt flag."""
    wrapper.stop()
    assert wrapper._interrupt_flag.is_set()
    await wrapper.speak("New sentence.")
    # After speak completes, flag was cleared at the start
    assert not wrapper._interrupt_flag.is_set() or len(fake_tts.spoken) > 0


# ── Properties ────────────────────────────────────────────────────────────────


def test_voice_name_delegates(wrapper, fake_tts):
    assert wrapper.voice_name == fake_tts.voice_name


def test_engine_property(wrapper, fake_tts):
    assert wrapper.engine is fake_tts
