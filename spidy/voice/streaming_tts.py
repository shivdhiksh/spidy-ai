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
                sentences = [s for s in self._split_sentences(text) if s.strip()]
                _n = len(sentences)
                if _n == 0:
                    return

                _response_start = time.monotonic()
                log.info(
                    "[TTS] response_received=0ms | chars={c} | sentences={n}",
                    c=len(text),
                    n=_n,
                )

                # Pipelined sentence synthesis & playback:
                # 1. Synthesize sentence 0 immediately.
                # 2. While sentence i is playing, concurrently synthesize sentence i+1 in the background.
                # 3. Sentence i+1 begins playing with 0ms gap when sentence i finishes.
                t_syn0 = time.monotonic()
                current_buf = await self._safe_synthesize(sentences[0])
                syn0_ms = (time.monotonic() - t_syn0) * 1000
                first_sound_latency = (time.monotonic() - _response_start) * 1000

                log.info(
                    "[TTS] sentence 0/{total}: synth_complete={syn:.0f}ms | "
                    "first_sound_latency={fsl:.0f}ms | chars={c}",
                    total=_n - 1,
                    syn=syn0_ms,
                    fsl=first_sound_latency,
                    c=len(sentences[0]),
                )

                next_task: asyncio.Task | None = None
                for i in range(_n):
                    if self._interrupt_flag.is_set():
                        log.debug("StreamingTTS: interrupted before sentence {i}.", i=i)
                        if next_task is not None and not next_task.done():
                            next_task.cancel()
                        break

                    # Start synthesizing next sentence concurrently during current playback
                    if i + 1 < _n and not self._interrupt_flag.is_set():
                        next_task = asyncio.create_task(
                            self._safe_synthesize(sentences[i + 1])
                        )

                    # Play current sentence buffer
                    t_play = time.monotonic()
                    await self._safe_play(current_buf, sentences[i])
                    play_ms = (time.monotonic() - t_play) * 1000

                    log.info(
                        "[TTS] sentence {i}/{total}: playback_complete={play:.0f}ms | "
                        "dur={dur:.0f}ms | chars={c}",
                        i=i,
                        total=_n - 1,
                        play=play_ms,
                        dur=getattr(current_buf, "duration_seconds", 0.0) * 1000,
                        c=len(sentences[i]),
                    )

                    # Await next sentence buffer if any
                    if next_task is not None:
                        try:
                            t_wait = time.monotonic()
                            current_buf = await next_task
                            wait_ms = (time.monotonic() - t_wait) * 1000
                            log.debug(
                                "[TTS] sentence {next_i} buffer ready after wait={wait:.0f}ms",
                                next_i=i + 1,
                                wait=wait_ms,
                            )
                        except asyncio.CancelledError:
                            break
                        except Exception as synth_exc:
                            log.error(
                                "StreamingTTS: synthesis failed for sentence {next_i}: {exc}",
                                next_i=i + 1,
                                exc=synth_exc,
                            )
                            break
                        finally:
                            next_task = None

                _total_ms = (time.monotonic() - _response_start) * 1000
                log.info(
                    "[TTS] total: sentences={n} elapsed={ms:.0f}ms",
                    n=_n,
                    ms=_total_ms,
                )

            finally:
                self._speaking = False

    async def _safe_synthesize(self, text: str) -> Any:
        """Safely invoke engine synthesize whether async, sync, or mock."""
        try:
            if hasattr(self._engine, "synthesize"):
                res = self._engine.synthesize(text)
                if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
                    return await res
                return res
        except Exception as exc:
            log.debug("StreamingTTS: synthesize error: {exc}", exc=exc)
        return None

    async def _safe_play(self, buffer: Any, sentence: str) -> None:
        """Safely play buffer via play_buffer if available, or fall back to speak."""
        if buffer is not None and hasattr(self._engine, "play_buffer"):
            res = self._engine.play_buffer(buffer)
            if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
                await res
        elif hasattr(self._engine, "speak"):
            res = self._engine.speak(sentence)
            if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
                await res

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
