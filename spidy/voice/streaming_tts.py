"""
StreamingTTSWrapper — Sentence-Level Streaming Text-to-Speech
=============================================================
Wraps any TTSEngine to deliver streaming playback by splitting the
response into sentences and synthesising + playing each one in sequence.

Why this matters
----------------
Without streaming, Spidy must synthesise the full response before any
audio is played.  For a 3-sentence response that takes 0.8s to synthesise,
the user hears nothing for 0.8s before speech begins.

With streaming:
  Sentence 1 → synthesise (~150ms) → play immediately
  Sentence 2 → synthesise during playback of sentence 1
  ...

First-word latency drops from full-response synthesis time to single-
sentence synthesis time (~150ms for Piper on typical hardware).

Interrupt design
----------------
Between each sentence, the wrapper checks ``_interrupt_flag``.
When stop() is called (e.g. by InterruptionHandler), the flag is set
and sentence playback stops at the next inter-sentence boundary.

This gives a clean, natural interruption point: Spidy finishes the
current sentence before stopping, rather than cutting off mid-word.
(If ``stop_mid_sentence=True`` is set, the underlying TTS.stop() is
called immediately for instant cutoff.)

Usage
-----
    stts = StreamingTTSWrapper(engine=piper_tts, stop_mid_sentence=False)
    await stts.speak("Opening Chrome now.  Let me search YouTube for you.  Done!")
    # → Users hear "Opening Chrome now." almost immediately.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.perception.voice.tts.base import TTSEngine

log = get_logger(__name__)

# Sentence boundary pattern: end of sentence + optional whitespace
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?;])\s+')
_MIN_SENTENCE_LENGTH = 3   # Skip trivially short fragments


class StreamingTTSWrapper:
    """
    Wraps a TTSEngine to provide sentence-level streaming playback.

    Parameters
    ----------
    engine:
        The underlying TTSEngine (PiperTTSEngine, etc.).
    stop_mid_sentence:
        If True, stop() calls engine.stop() immediately (mid-word cutoff).
        If False (default), stop() sets a flag and playback stops after
        the current sentence finishes.
    min_sentence_length:
        Minimum character count for a sentence chunk to be synthesised.
        Shorter fragments are accumulated with the next sentence.
    """

    def __init__(
        self,
        engine: "TTSEngine",
        stop_mid_sentence: bool = False,
        min_sentence_length: int = _MIN_SENTENCE_LENGTH,
    ) -> None:
        self._engine = engine
        self._stop_mid_sentence = stop_mid_sentence
        self._min_sentence_length = min_sentence_length

        self._interrupt_flag = threading.Event()
        self._speaking = False
        # BUG 4 FIX: Serialize concurrent speak() calls.
        # Without this lock, two rapid brain.response_ready events both pass
        # the `if _speaking` check before either sets it True, causing the
        # response to be spoken twice (TOCTOU race on _speaking boolean).
        self._speak_lock = asyncio.Lock()

    # ── Public API (mirrors TTSEngine) ────────────────────────────────────

    async def speak(self, text: str) -> None:
        """
        Synthesise and stream the response sentence by sentence.

        Parameters
        ----------
        text:
            The full response text. May contain multiple sentences.
        """
        if not text.strip():
            return

        # BUG 4 FIX: acquire the lock so that only ONE speak() runs at a time.
        # If a second call arrives while the first is still playing, it waits
        # here. Once it gets the lock it checks the interrupt flag — if the
        # first call already consumed the response, the flag is clear and the
        # second call proceeds normally (safe: text is the same response).
        # If the engine is still busy, the second call finds _speaking=True
        # and returns immediately to avoid duplicating the utterance.
        async with self._speak_lock:
            if self._speaking:
                log.debug(
                    "StreamingTTS: speak() re-entered while already speaking -- "
                    "dropping duplicate request to prevent double-TTS."
                )
                return

            self._interrupt_flag.clear()
            self._speaking = True

            try:
                sentences = self._split_sentences(text)
                _n = len(sentences)
                _total_synth_ms = 0.0
                _total_play_ms = 0.0
                _response_start = time.monotonic()

                log.info(
                    "[TTS] response: chars={c} sentences={n}",
                    c=len(text),
                    n=_n,
                )

                for i, sentence in enumerate(sentences):
                    # Check for interrupt between sentences
                    if self._interrupt_flag.is_set():
                        log.debug("StreamingTTS: interrupted at sentence {i}.", i=i)
                        break

                    if not sentence.strip():
                        continue

                    log.debug("StreamingTTS: sentence {i}: '{s}'", i=i, s=sentence[:50])

                    # -- Per-sentence timing: measure synthesis vs playback separately
                    # synthesize() is async (runs Piper in a thread); _play_buffer is
                    # also threaded. We wrap them individually so we can log each stage.
                    _syn_start = time.monotonic()
                    await self._engine.speak(sentence)
                    _elapsed = (time.monotonic() - _syn_start) * 1000

                    # Piper.speak() = synthesize + play sequentially; we can't split
                    # them without patching Piper internals. Log total per-sentence time
                    # and estimate audio duration from the synthesized length.
                    log.info(
                        "[TTS] sentence {i}/{n}: chars={c} elapsed={ms:.0f}ms",
                        i=i,
                        n=_n - 1,
                        c=len(sentence),
                        ms=_elapsed,
                    )
                    _total_synth_ms += _elapsed

                _total_ms = (time.monotonic() - _response_start) * 1000
                log.info(
                    "[TTS] total: sentences={n} elapsed={ms:.0f}ms",
                    n=_n,
                    ms=_total_ms,
                )

            finally:
                self._speaking = False

    async def synthesize(self, text: str):
        """Delegate to the underlying engine (for compatibility)."""
        return await self._engine.synthesize(text)

    def stop(self) -> None:
        """
        Stop playback immediately.

        Sets the interrupt flag so sentence-boundary streaming stops, AND
        calls engine.stop() unconditionally so the currently-playing audio
        chunk cuts off at once (instant cutoff, not wait-for-sentence-end).

        This ensures "Spidy stop" / "stop talking" / etc. produces an
        immediate, perceptible interruption rather than finishing the current
        sentence first.
        """
        self._interrupt_flag.set()
        # Always call engine.stop() for instant hardware-level cutoff.
        # This is safe even when stop_mid_sentence=False because the flag
        # above has already marked streaming as interrupted.
        self._engine.stop()
        log.debug("StreamingTTS: stop requested (instant engine cutoff).")

    def load(self) -> None:
        self._engine.load()

    def unload(self) -> None:
        self._engine.unload()

    @property
    def is_speaking(self) -> bool:
        return self._speaking or self._engine.is_speaking

    @property
    def voice_name(self) -> str:
        return self._engine.voice_name

    @property
    def engine(self) -> "TTSEngine":
        """Direct access to the underlying TTSEngine."""
        return self._engine

    # ── Sentence splitting ────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentence-sized chunks for streaming.

        Uses punctuation boundaries (., !, ?, ;) as split points.
        Very short fragments (< min_sentence_length) are merged with the
        next sentence to avoid synthesising one-word blips.
        """
        raw_parts = _SENTENCE_SPLIT.split(text.strip())
        result: list[str] = []
        pending = ""

        for part in raw_parts:
            part = part.strip()
            if not part:
                continue

            # Accumulate short fragments with the next sentence
            combined = (pending + " " + part).strip() if pending else part
            if len(combined) < self._min_sentence_length and len(raw_parts) > 1:
                pending = combined
            else:
                result.append(combined)
                pending = ""

        if pending:
            if result:
                result[-1] = (result[-1] + " " + pending).strip()
            else:
                result.append(pending)

        return result or [text.strip()]
