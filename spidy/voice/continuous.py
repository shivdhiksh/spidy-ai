"""
ContinuousVoiceController — M14 Voice Companion Coordinator
============================================================
The top-level coordinator for the Milestone 14 voice companion layer.

Responsibilities
----------------
1. Wrap the existing VoiceEngine (does NOT replace it)
2. Enable continuous conversation: session stays open between turns
3. Route transcripts through InterruptionHandler before Brain
4. Manage session lifecycle via VoiceSessionManager
5. Deliver responses via StreamingTTSWrapper
6. Subscribe to brain.response_ready and voice.* events

Architecture Integration
------------------------
                   ┌─────────────────────────┐
  VoiceEngine ────►│ ContinuousVoiceController│
  (unchanged)      │                         │
                   │  ┌─────────────────────┐│
                   │  │  VoiceSessionManager││
                   │  └─────────────────────┘│
                   │                         │
                   │  ┌─────────────────────┐│
                   │  │  InterruptionHandler ││
                   │  └─────────────────────┘│
                   │                         │
                   │  ┌─────────────────────┐│
                   │  │  StreamingTTSWrapper ││
                   │  └─────────────────────┘│
                   └────────────┬────────────┘
                                │
                            Brain.process()
                            Brain.run_goal()
                            (UNCHANGED)

Event subscriptions
-------------------
  voice.transcript    → check interrupts → Brain → StreamingTTS
  wake_word.detected  → VoiceSessionManager.activate()
  brain.response_ready → StreamingTTS.speak()
  voice.session_timeout → VoiceSessionManager.deactivate()
  system.shutting_down → stop()

Key design rule: ContinuousVoiceController never touches Brain internals.
It only calls brain.process(), brain.run_goal(), brain.cancel_goal().
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any

from spidy.brain.goal_intent_classifier import GoalIntentClassifier
from spidy.brain.intent_classifier import IntentClassifier
from spidy.logging.logger import get_logger
from spidy.voice.events import (
    VoiceInterruptEvent,
    VoicePausedEvent,
    VoiceResumedEvent,
    VoiceSessionStartedEvent,
)
from spidy.voice.barge_in import BargeInDetector
from spidy.voice.interruption import InterruptionHandler
from spidy.voice.session import VoiceSessionManager, VoiceSessionState
from spidy.voice.streaming_tts import StreamingTTSWrapper
from spidy.voice.transcript_guard import TranscriptQualityGuard
from spidy.voice.tts_text_filter import TTSTextFilter
from spidy.voice.wake_ack import WakeAcknowledger
from spidy.voice.wake_word_stripper import WakeWordStripper

if TYPE_CHECKING:
    from spidy.brain.brain import Brain
    from spidy.core.event_bus import EventBus
    from spidy.perception.voice.engine import VoiceEngine

log = get_logger(__name__)


class ContinuousVoiceController:
    """
    Continuous voice conversation coordinator for Milestone 14.

    Parameters
    ----------
    voice_engine:
        The existing VoiceEngine instance (hardware layer).
    brain:
        The Brain instance — all cognitive processing goes here.
    bus:
        Application EventBus.
    session_manager:
        VoiceSessionManager that tracks conversation state.
    streaming_tts:
        StreamingTTSWrapper wrapping the TTS engine.
    interruption_handler:
        InterruptionHandler for stop/cancel/pause/resume commands.
    continuous_mode:
        If True (default), session stays active between turns.
    measure_latency:
        If True, log end-to-end voice latency measurements.
    wake_acknowledger:
        Optional WakeAcknowledger.  When provided, Spidy speaks a short
        ack phrase (e.g. "Yes Shiva.") immediately after wake detection
        with the microphone muted, so the ack audio is never forwarded
        to STT or Brain.  If None, no ack is spoken.
    """

    def __init__(
        self,
        voice_engine: "VoiceEngine",
        brain: "Brain",
        bus: "EventBus",
        session_manager: VoiceSessionManager,
        streaming_tts: StreamingTTSWrapper,
        interruption_handler: InterruptionHandler,
        barge_in: BargeInDetector | None = None,
        tts_filter: TTSTextFilter | None = None,
        continuous_mode: bool = True,
        measure_latency: bool = True,
        wake_acknowledger: WakeAcknowledger | None = None,
    ) -> None:
        self._engine = voice_engine
        self._brain = brain
        self._bus = bus
        self._session = session_manager
        self._streaming_tts = streaming_tts
        self._interruption = interruption_handler
        self._barge_in = barge_in
        self._tts_filter = tts_filter or TTSTextFilter(enabled=False)  # disabled by default if not wired
        self._continuous_mode = continuous_mode
        self._measure_latency = measure_latency
        self._wake_acknowledger = wake_acknowledger

        # Intent classifiers for routing (same as REPL)
        self._intent_clf = IntentClassifier()
        self._goal_clf = GoalIntentClassifier()

        # Voice pipeline normalization and quality guards
        self._wake_stripper = WakeWordStripper()
        self._quality_guard = TranscriptQualityGuard()

        self._running = False
        self._speech_end_time: float | None = None  # For latency measurement

        # Transcript deduplication (Issue 4: prevents stale audio re-processed)
        self._last_transcript: str = ""
        self._last_transcript_time: float = 0.0
        self._dedup_window: float = 1.5          # configurable at construction time
        self._post_ack_flush_ms: int = 100       # configurable at construction time

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Subscribe to events and activate the continuous voice loop."""
        if self._running:
            return
        self._running = True

        # Subscribe to voice events
        self._bus.subscribe("voice.transcript", self._on_transcript)
        self._bus.subscribe("wake_word.detected", self._on_wake_word)
        self._bus.subscribe("voice.speaking_end", self._on_speaking_end)

        # Issue 2 fix: Take exclusive ownership of brain.response_ready for TTS.
        # VoiceEngine.start() already subscribed its own _on_brain_response handler
        # which calls self._tts.speak() directly.  CVC also subscribes here to use
        # StreamingTTSWrapper instead.  The EventBus calls ALL subscribers sequentially,
        # so WITHOUT unsubscribing the VoiceEngine handler, every response is spoken
        # TWICE — once by VoiceEngine (plain Piper) and once by CVC (StreamingTTS).
        # Fix: subscribe CVC's handler first, then remove VoiceEngine's handler so
        # only StreamingTTS owns speech delivery for the rest of the session.
        self._bus.subscribe("brain.response_ready", self._on_brain_response)
        self._bus.unsubscribe("brain.response_ready", self._engine._on_brain_response)
        log.info(
            "ContinuousVoiceController: took exclusive ownership of brain.response_ready "
            "(VoiceEngine direct-speak path disabled)."
        )

        # Session timeout → deactivate
        self._bus.subscribe("voice.session_timeout", self._on_session_timeout)

        # Shutdown
        self._bus.subscribe("system.shutting_down", self._on_shutdown)

        log.info(
            "ContinuousVoiceController started | continuous={c}",
            c=self._continuous_mode,
        )

    async def stop(self) -> None:
        """Unsubscribe and deactivate session."""
        if not self._running:
            return
        self._running = False

        self._bus.unsubscribe("voice.transcript", self._on_transcript)
        self._bus.unsubscribe("wake_word.detected", self._on_wake_word)
        self._bus.unsubscribe("voice.speaking_end", self._on_speaking_end)
        self._bus.unsubscribe("brain.response_ready", self._on_brain_response)
        self._bus.unsubscribe("voice.session_timeout", self._on_session_timeout)
        self._bus.unsubscribe("system.shutting_down", self._on_shutdown)

        await self._session.deactivate(reason="controller_stopped")
        log.info("ContinuousVoiceController stopped.")

    # ── Event Handlers ────────────────────────────────────────────────────

    async def _on_wake_word(self, event: Any) -> None:
        """
        Handle wake_word.detected.

        8-step sequence
        ---------------
        0. Guard: TTS currently speaking? → ignore (prevents duplication).
        1. Guard: session already active? → ignore (prevents duplicate sessions).
        2. Publish ui.state_change → "listening" immediately (overlay goes AWAKE).
        3. Open the voice session.
        4. Set CaptureMode.IDLE — mic OFF before ack TTS so ack audio is never
           captured by STT or the wake-word model.
        5. Speak the ack phrase ("Yes Shiva." / "Hmm?") if acknowledger is set.
        6. Reset OWW prediction buffer (discard stale scores from ack audio).
        7. Set CaptureMode.RECORDING — mic ON; user's command is next.
        """
        # Step 0: Belt-and-suspenders TTS guard — if Spidy is actively
        # speaking a response, a wake event must be silently dropped.
        # session.is_active handles this in most cases, but an explicit
        # TTS check prevents edge cases where session state transitions
        # faster than TTS finishes.
        if self._streaming_tts.is_speaking:
            log.info(
                "ContinuousVoiceController: wake word received while TTS is "
                "speaking — ignoring to prevent duplicate session."
            )
            return

        # Step 1: Session already active guard
        if self._session.is_active:
            log.info(
                "ContinuousVoiceController: wake word received while session "
                "already active (sid={sid}) — ignoring duplicate wake event.",
                sid=self._session.session_id,
            )
            return

        wake_word = getattr(event, "model_name", "hey_jarvis")

        # Step 2: overlay → AWAKE/LISTENING immediately (before ack audio)
        from spidy.ui.events import UIStateChangeEvent
        await self._bus.publish(UIStateChangeEvent(state="listening"))

        # Step 3: open session
        sid = await self._session.activate(wake_word=wake_word)
        log.info("Voice session opened: {sid}", sid=sid)

        # Steps 4-7: ack with mic muted
        if self._wake_acknowledger is not None:
            from spidy.perception.voice.audio_capture import CaptureMode
            capture = self._engine._capture_engine

            # Step 4: mic OFF — ack audio must not enter STT
            capture.set_mode(CaptureMode.IDLE)
            log.info(
                "ContinuousVoiceController: mic set IDLE before ack — "
                "ack audio isolated from STT pipeline."
            )

            try:
                # Step 5: speak ack ("Yes Shiva." / "Hmm?")
                await self._wake_acknowledger.speak_ack(self._streaming_tts)
            finally:
                # Step 6: flush OWW buffer so ack frames don't re-trigger
                self._engine._wake_model.reset_buffer()

                # Step 7: mic ON — user's command is now captured.
                # IMPORTANT: Before opening the mic, clear the capture buffer
                # to discard any Piper echo that accumulated during the ack.
                # Without this, the echo tail ("...Shiva") enters the next
                # STT window and produces "Yes, Shiva. <user command>" transcripts.
                if self._post_ack_flush_ms > 0:
                    await asyncio.sleep(self._post_ack_flush_ms / 1000)
                    import threading as _threading
                    with self._engine._buffer_lock:
                        self._engine._capture_buffer.clear()
                    log.info(
                        "ContinuousVoiceController: post-ack buffer flushed "
                        "({ms}ms settle) -- ack echo cleared.",
                        ms=self._post_ack_flush_ms,
                    )

                capture.set_mode(CaptureMode.RECORDING)
                log.info(
                    "ContinuousVoiceController: mic restored to RECORDING -- "
                    "ready for user command."
                )

    async def _on_transcript(self, event: Any) -> None:
        """
        Handle voice.transcript events.

        Full pipeline:
          0. Correlation ID (tx_id) for end-to-end log tracing
          1. Wake-prefix strip (e.g. "Hey Jarvis, open Edge" -> "open Edge")
          2. Transcript quality guard (reject garbled/noise-only STT output)
          2b. Deduplication guard (reject same transcript within dedup_window)
          3. Interrupt command detection  <- SAFETY: runs before Brain/LLM
          4. Session gate: DROP if no active session (no wake -> no session)
          5. Utterance recorded
          6. Intent classification
          7. Brain.process() or Brain.run_goal()
          8. Latency measurement logged
        """
        # -- Step 0: Correlation ID -------------------------------------------------
        tx_id = uuid.uuid4().hex[:8]
        _transcript_received = time.monotonic()

        text: str = getattr(event, "text", "").strip()
        if not text:
            return

        # -- Step 1: Strip wake-word prefix ----------------------------------------
        # e.g. "Hey Jarvis, open Edge." -> "open Edge."

        # Guard: reject transcripts that consist ENTIRELY of wake phrases
        # (e.g. "Hey Jarvis. Hey Jarvis.").  A single strip would leave
        # "Hey Jarvis." which is still wake-only and must NOT reach Brain.
        if self._wake_stripper.is_wake_only(text):
            log.debug(
                "[tx:{tx}] Wake-only transcript (repeated phrases) -- dropping: '{t}'",
                tx=tx_id, t=text[:60],
            )
            return

        stripped = self._wake_stripper.strip_wake_prefix(text)
        if stripped != text:
            log.info(
                "[tx:{tx}] Wake prefix stripped: '{orig}' -> '{clean}'",
                tx=tx_id, orig=text[:60], clean=stripped[:60],
            )
        text = stripped

        # If stripping consumed the entire utterance (only wake phrase spoken),
        # treat as a no-command event — do not forward to Brain.
        if not text:
            log.debug("[tx:{tx}] Wake phrase only -- no command, skipping Brain.", tx=tx_id)
            return

        # ── Step 2: Transcript quality guard ───────────────────────────────
        # Reject clearly garbled/noise-only transcripts before the LLM call.
        # Uses faster-whisper segment signals when available (fail-open otherwise).
        segment_signals = getattr(event, "segment_signals", None)
        if self._quality_guard.is_suspicious(text, segments=segment_signals):
            reason = self._quality_guard.suspicious_reason(text, segments=segment_signals)
            log.info(
                "[tx:{tx}] Transcript rejected by quality guard (reason={r}): '{t}'",
                tx=tx_id, r=reason, t=text[:60],
            )
            return

        # -- Step 2b: Transcript deduplication guard --------------------------------
        # Drops the same transcript if re-delivered within dedup_window seconds.
        # This catches stale capture-buffer audio re-transcribed after relisten,
        # WITHOUT suppressing two genuinely separate identical commands.
        if self._dedup_window > 0.0:
            _now = time.monotonic()
            _norm = text.lower().strip()
            if _norm == self._last_transcript and (_now - self._last_transcript_time) < self._dedup_window:
                log.info(
                    "[tx:{tx}] Duplicate transcript within {w:.1f}s -- dropping: '{t}'",
                    tx=tx_id, w=self._dedup_window, t=text[:60],
                )
                return
            self._last_transcript = _norm
            self._last_transcript_time = _now

        # ── STT timing ────────────────────────────────────────────────────
        if self._measure_latency and self._speech_end_time is not None:
            stt_latency = (time.monotonic() - self._speech_end_time) * 1000
            log.info("[tx:{tx}] [Voice latency] stt_to_brain_start={ms:.0f}ms", tx=tx_id, ms=stt_latency)

        log.info("[tx:{tx}] Transcript: '{t}'", tx=tx_id, t=text[:80])

        # ── Step 3: Interrupt detection ────────────────────────────────────
        # SAFETY: This runs BEFORE Brain, LLM, Memory, or any other component.
        # The interrupt handler is purely deterministic (lexical matching).
        # No NVIDIA, Ollama, Brain, Planner, Memory, or web-search involved.
        interrupt_cmd = self._interruption.detect(text)
        if interrupt_cmd is not None:
            response = await self._interruption.execute(interrupt_cmd, text)
            if interrupt_cmd.is_cancellation:
                log.info("[tx:{tx}] Stop command -- restoring overlay and re-listen.", tx=tx_id)
                from spidy.ui.events import UIStateChangeEvent
                await self._bus.publish(UIStateChangeEvent(state="listening"))
                if self._continuous_mode and self._session.is_active:
                    self._engine.signal_relisten()
            elif response:
                await self._streaming_tts.speak(response)
            return

        # -- Step 4: Session gate -- HARD DROP if no active session ----------------
        # A transcript MUST NOT open a session by itself. Only a real
        # wake_word.detected event may activate a VoiceSession.
        # Removes the old "text-mode test" fallback that allowed any raw
        # STT transcript to silently open a session with wake='(none)'.
        if not self._session.is_active:
            log.warning(
                "[tx:{tx}] Transcript while SLEEPING (no active session) -- "
                "dropping. Only a real wake event may open a session.",
                tx=tx_id,
            )
            return

        # ── Step 5: Record utterance ───────────────────────────────────────
        await self._session.record_utterance(text)

        # ── Step 6: Intent classification + Step 7: Brain routing ──────────
        _brain_start = time.monotonic()
        try:
            intent = await self._intent_clf.classify(text)
            _t_intent = (time.monotonic() - _brain_start) * 1000
            log.info("[tx:{tx}] [Voice latency] intent_classify={ms:.0f}ms", tx=tx_id, ms=_t_intent)
            if self._goal_clf.is_executable(intent):
                response = await self._brain.run_goal(text)
            else:
                response = await self._brain.process(text)
        except Exception as exc:  # noqa: BLE001
            log.error("[tx:{tx}] Brain error: {exc}", tx=tx_id, exc=exc)
            response = "I ran into a problem. Please try again."

        # -- Step 8: Latency measurement -------------------------------------------
        if self._measure_latency:
            _brain_total = (time.monotonic() - _brain_start) * 1000
            log.info("[tx:{tx}] [Voice latency] brain_roundtrip={ms:.0f}ms", tx=tx_id, ms=_brain_total)

        # Response delivery is handled by _on_brain_response (via event bus)

    async def _on_brain_response(self, event: Any) -> None:
        """
        Handle brain.response_ready → deliver via StreamingTTS.

        This handler intercepts the brain.response_ready event and delivers
        it through StreamingTTSWrapper instead of the plain VoiceEngine.speak().
        Both handlers receive the event; StreamingTTS takes priority for M14.
        """
        response_text: str = getattr(event, "response_text", "") or getattr(event, "text", "")
        if not response_text.strip():
            return

        if self._session.is_paused:
            log.debug("ContinuousVoiceController: session paused -- skipping TTS.")
            return

        # Apply TTS text filter — strip markdown/code for natural speech.
        # The original response_text is NOT modified; the UI sees full markdown.
        filter_result = self._tts_filter.filter(response_text)
        tts_text = filter_result.tts_text

        if filter_result.code_blocks_removed > 0 or filter_result.original_chars != filter_result.filtered_chars:
            log.info(
                "[TTS filter] original_chars={orig} filtered_chars={filt} "
                "code_blocks={cb}",
                orig=filter_result.original_chars,
                filt=filter_result.filtered_chars,
                cb=filter_result.code_blocks_removed,
            )

        if not tts_text.strip():
            log.debug("[TTS filter] filtered text is empty -- skipping TTS.")
            return

        await self._session.mark_speaking()

        # Switch mic to BARGE_IN mode so BargeInDetector can hear "Spidy stop".
        # BARGE_IN routes audio ONLY to barge_in.feed_chunk — the wake-word
        # model and main STT buffer receive nothing (no self-talk risk).
        # Falls back to IDLE if barge-in is not available (None / disabled).
        from spidy.perception.voice.audio_capture import CaptureMode
        capture = self._engine._capture_engine  # access engine's capture instance

        if self._barge_in is not None and self._barge_in.is_loaded:
            capture.set_mode(CaptureMode.BARGE_IN)
            self._barge_in.start(tts=self._streaming_tts)
            log.info(
                "[BargeIn] CaptureMode.BARGE_IN activated — "
                "BargeInDetector listening for stop phrase."
            )
        else:
            # No barge-in configured: fall back to IDLE (safe, mic muted)
            capture.set_mode(CaptureMode.IDLE)
            log.debug("[BargeIn] barge-in not available — mic IDLE during TTS.")

        _tts_start = time.monotonic()
        log.info("[Voice latency] tts_start")
        try:
            await self._streaming_tts.speak(tts_text)
        finally:
            # ── Always stop barge-in detection first ───────────────────────
            if self._barge_in is not None:
                self._barge_in.stop()

            _tts_ms = (time.monotonic() - _tts_start) * 1000
            log.info("[Voice latency] tts_complete={ms:.0f}ms", ms=_tts_ms)

            # ── Echo-settle delay ──────────────────────────────────────────────
            # After TTS finishes, the speaker hardware has a short echo tail
            # (~100–200ms). Restoring mic capture immediately can cause the
            # STT engine to transcribe Spidy's own TTS audio, producing
            # spurious transcripts and self-generated conversation (Issue 4).
            #
            # Strategy:
            #   1. First reset_buffer() call — discard TTS audio in OWW before IDLE
            #   2. Wait 150ms settle delay — let hardware echo decay
            #   3. Second reset_buffer() call — flush any echo captured during settle
            #   4. Restore CaptureMode.DETECTING — mic is now clean
            self._engine._wake_model.reset_buffer()

            _settle_start = time.monotonic()
            await asyncio.sleep(0.15)  # 150ms echo-settle
            _settle_ms = (time.monotonic() - _settle_start) * 1000
            log.info(
                "[Safety] TTS echo-settle complete ({ms:.0f}ms) — "
                "flushing OWW buffer and restoring capture.",
                ms=_settle_ms,
            )

            # Second buffer flush — discard any echo that leaked during settle
            self._engine._wake_model.reset_buffer()

            # Restore wake-word detection now that playback is complete.
            capture.set_mode(CaptureMode.DETECTING)
            log.info(
                "[Voice latency] microphone_restored — listening for next utterance."
            )

            if self._measure_latency and self._speech_end_time is not None:
                total_latency = (time.monotonic() - self._speech_end_time) * 1000
                log.info(
                    "[Voice latency] speech_end_to_tts_complete={ms:.0f}ms",
                    ms=total_latency,
                )
            await self._session.mark_response_complete()

            # ── Continuous conversation re-listen ──────────────────────────
            # After TTS finishes and the session is still open, signal the
            # VoiceEngine to immediately start a new listen cycle.  This is
            # the key mechanism for continuous conversation: the user can ask
            # a follow-up without repeating the wake word.
            #
            # We only signal when:
            #   1. continuous_mode is enabled (default: True)
            #   2. the session is still active (not timed out, not cancelled)
            #   3. the session is not paused (user said "wait")
            #
            # signal_relisten() itself is also guarded: if the engine is not
            # in SLEEPING state, the signal is silently dropped.
            if (
                self._continuous_mode
                and self._session.is_active
                and not self._session.is_paused
            ):
                self._engine.signal_relisten()
                log.info(
                    "ContinuousVoiceController: re-listen triggered "
                    "(session={sid}).",
                    sid=self._session.session_id,
                )
                # Overlay back to AWAKE/LISTENING after each response
                from spidy.ui.events import UIStateChangeEvent
                await self._bus.publish(UIStateChangeEvent(state="listening"))

    async def _on_speaking_end(self, event: Any) -> None:
        """Record when speech ends (for latency measurement)."""
        self._speech_end_time = time.monotonic()

    async def _on_session_timeout(self, event: Any) -> None:
        """Session timed out — already deactivated by VoiceSessionManager."""
        log.info("ContinuousVoiceController: session timed out.")
        # Overlay → SLEEPING (idle) when session expires
        from spidy.ui.events import UIStateChangeEvent
        await self._bus.publish(UIStateChangeEvent(state="idle"))

    async def _on_shutdown(self, event: Any) -> None:
        """Handle application shutdown."""
        await self.stop()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def session(self) -> VoiceSessionManager:
        return self._session

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_session_active(self) -> bool:
        return self._session.is_active
