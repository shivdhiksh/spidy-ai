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
from spidy.core.pipeline_tracer import PipelineTrace, PipelineTracer
from spidy.logging.logger import get_logger
from spidy.perception.voice.engine import VoiceState
from spidy.voice.events import (
    VoiceInterruptEvent,
    VoicePausedEvent,
    VoiceResumedEvent,
    VoiceSessionStartedEvent,
)
from spidy.voice.barge_in import BargeInDetector
from spidy.voice.interruption import InterruptionHandler
from spidy.voice.normalizer import TranscriptNormalizer
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
        ack_phrases = getattr(self._wake_acknowledger, "phrases", []) if self._wake_acknowledger else []
        self._wake_stripper = WakeWordStripper(ack_phrases=ack_phrases)
        self._normalizer = TranscriptNormalizer(ack_phrases=ack_phrases)
        self._quality_guard = TranscriptQualityGuard()

        self._running = False
        self._speech_end_time: float | None = None  # For latency measurement

        # Transcript deduplication (Issue 4: prevents stale audio re-processed)
        self._last_transcript: str = ""
        self._last_transcript_time: float = 0.0
        self._dedup_window: float = 1.5          # configurable at construction time
        self._post_ack_flush_ms: int = 350       # 350ms settle allows sound card DAC & room echo decay

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Subscribe to events and activate the continuous voice loop."""
        if self._running:
            return
        self._running = True

        # A chunk still in flight to the barge-in detector when the user interrupts goes to the recorder.
        if self._barge_in is not None and hasattr(self._engine, "_buffer_audio"):
            self._barge_in.set_handoff_sink(self._engine._buffer_audio)
        if self._barge_in is not None and hasattr(self._streaming_tts, "reference"):
            self._barge_in.attach_reference(self._streaming_tts.reference)
        capture = getattr(self._engine, "_capture_engine", None)
        ratio = getattr(capture, "barge_in_gain_ratio", None)
        if self._barge_in is not None and isinstance(ratio, (int, float)):
            self._barge_in.set_gain_ratio(ratio)

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
        if hasattr(self._engine, "_on_brain_response"):
            self._bus.unsubscribe("brain.response_ready", self._engine._on_brain_response)
        log.info(
            "ContinuousVoiceController: took exclusive ownership of brain.response_ready "
            "(VoiceEngine direct-speak path disabled)."
        )

        # Session timeout → deactivate
        self._bus.subscribe("voice.session_timeout", self._on_session_timeout)
        self._bus.subscribe("voice.session_ended", self._on_session_ended)

        # Shutdown
        self._bus.subscribe("system.shutting_down", self._on_shutdown)

        log.info(
            "ContinuousVoiceController started | continuous={c}",
            c=self._continuous_mode,
        )

    def halt(self) -> None:
        """Stop talking right now (thread-safe; used when Spidy is being closed)."""
        try:
            self._streaming_tts.stop()
        except Exception as exc:  # noqa: BLE001
            log.debug("ContinuousVoiceController.halt: tts stop failed: {exc}", exc=exc)
        if self._barge_in is not None:
            self._barge_in.stop()

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
        self._bus.unsubscribe("voice.session_ended", self._on_session_ended)
        self._bus.unsubscribe("system.shutting_down", self._on_shutdown)

        await self._session.deactivate(reason="controller_stopped")
        log.info("ContinuousVoiceController stopped.")

    async def _safe_session_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Helper to invoke VoiceSessionManager methods safely whether async or mock."""
        fn = getattr(self._session, method, None)
        if fn is not None:
            res = fn(*args, **kwargs)
            if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
                return await res
            return res
        return None

    async def _safe_engine_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Helper to invoke VoiceEngine methods safely whether async or mock."""
        fn = getattr(self._engine, method, None)
        if fn is not None:
            res = fn(*args, **kwargs)
            if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
                return await res
            return res
        return None

    # ── Event Handlers ────────────────────────────────────────────────────

    async def _on_wake_word(self, event: Any) -> None:
        """
        Handle wake_word.detected.

        8-step sequence
        ---------------
        0. Guard: TTS currently speaking or in flush? → ignore (prevents duplicate sessions / loop).
        1. Guard: session already active? → ignore (prevents duplicate sessions).
        2. Publish ui.state_change → "listening" immediately (overlay goes AWAKE).
        3. Open the voice session.
        4. Set CaptureMode.IDLE — mic OFF before ack TTS so ack audio is never
           captured by STT or the wake-word model.
        5. Speak the ack phrase ("Yes Shiva." / "Hmm?") if acknowledger is set.
        6. Reset OWW prediction buffer & flush post-ack audio.
        7. Set CaptureMode.RECORDING and signal engine relisten — mic ON; user's command is next.
        """
        # Step 0: Belt-and-suspenders TTS guard
        engine_audio_busy = (
            getattr(self._streaming_tts, "is_speaking", False) is True
            or (hasattr(self._engine, "is_speaking") and getattr(self._engine, "is_speaking", False) is True)
            or (hasattr(self._engine, "is_flushing") and getattr(self._engine, "is_flushing", False) is True)
            or getattr(self._session, "state", None) in (VoiceSessionState.SPEAKING, VoiceSessionState.AUDIO_FLUSH)
        )
        if engine_audio_busy:
            if hasattr(self._engine, "discard_pending_wake"):
                restore = VoiceState.SPEAKING if (hasattr(self._engine, "is_speaking") and self._engine.is_speaking) else VoiceState.AUDIO_FLUSH
                self._engine.discard_pending_wake(restore)
            log.info(
                "ContinuousVoiceController: wake word received while TTS is "
                "speaking/flushing — ignoring to prevent duplicate session."
            )
            return

        # A wake heard while the current utterance is still travelling through
        # STT/Brain is not a new session.  Crucially, acknowledge the rejected
        # event back to VoiceEngine so it cannot remain stuck in WAKING.
        if getattr(self._session, "state", None) == VoiceSessionState.PROCESSING:
            if hasattr(self._engine, "discard_pending_wake"):
                self._engine.discard_pending_wake(VoiceState.PROCESSING)
            log.info(
                "ContinuousVoiceController: wake word received while the "
                "current turn is processing — keeping the existing session."
            )
            return

        wake_word = getattr(event, "wake_phrase", None) or getattr(event, "model_name", "hey_jarvis")
        log.info("[WAKE] detected phrase='{phrase}'", phrase=wake_word)

        # Step 2: restore HUD overlay & transition to LISTENING immediately (before ack audio)
        from spidy.ui.events import UIStateChangeEvent, UIShowEvent
        await self._bus.publish(UIShowEvent(source="wake_word"))
        await self._bus.publish(UIStateChangeEvent(state="listening"))

        # Step 3: open or refresh session
        sid = await self._safe_session_call("activate", wake_word=wake_word)
        log.info("[SESSION] started | session_id={sid} | wake_phrase='{phrase}'", sid=sid, phrase=wake_word)

        # Steps 4-7: ack with mic muted
        if self._wake_acknowledger is not None and getattr(self._wake_acknowledger, "enabled", True) is not False:
            from spidy.perception.voice.audio_capture import CaptureMode
            capture = getattr(self._engine, "_capture_engine", None)

            # Step 4: mic OFF — ack audio must not enter STT
            if capture is not None:
                capture.set_mode(CaptureMode.IDLE)
            if hasattr(self._engine, "set_speaking"):
                self._engine.set_speaking(True)
            await self._safe_session_call("mark_speaking")
            log.info(
                "ContinuousVoiceController: mic set IDLE before ack — "
                "ack audio isolated from STT pipeline."
            )

            try:
                # Step 5: speak ack ("Yes Shiva." / "Hmm?")
                await self._wake_acknowledger.speak_ack(self._streaming_tts)
            finally:
                # Step 6: flush buffers and settle
                await self._safe_session_call("mark_audio_flush")
                if hasattr(self._engine, "_wake_model") and hasattr(self._engine._wake_model, "reset_buffer"):
                    self._engine._wake_model.reset_buffer()
                if hasattr(self._engine, "enter_audio_flush"):
                    await self._safe_engine_call("enter_audio_flush", settle_seconds=self._post_ack_flush_ms / 1000 if self._post_ack_flush_ms > 0 else 0.20)
                else:
                    if self._post_ack_flush_ms > 0:
                        await asyncio.sleep(self._post_ack_flush_ms / 1000)
                    if hasattr(self._engine, "_capture_buffer"):
                        with getattr(self._engine, "_buffer_lock", threading.Lock()):
                            self._engine._capture_buffer.clear()

                if hasattr(self._engine, "exit_audio_flush"):
                    self._engine.exit_audio_flush()
                await self._safe_session_call("mark_response_complete")

                # Step 7: mic ON — ready for user command
                if capture is not None:
                    capture.set_mode(CaptureMode.RECORDING)
                if hasattr(self._engine, "signal_relisten"):
                    self._engine.signal_relisten()
                log.info(
                    "ContinuousVoiceController: mic restored to RECORDING -- "
                    "ready for user command."
                )
        else:
            from spidy.perception.voice.audio_capture import CaptureMode
            capture = getattr(self._engine, "_capture_engine", None)
            if hasattr(self._engine, "exit_audio_flush"):
                self._engine.exit_audio_flush()
            if hasattr(self._engine, "_set_state"):
                self._engine._set_state(VoiceState.SLEEPING)
            if capture is not None:
                capture.set_mode(CaptureMode.RECORDING)
            if hasattr(self._engine, "signal_relisten"):
                self._engine.signal_relisten()

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
        # -- Step 0: Correlation ID & Pipeline Trace -------------------------------
        tx_id = uuid.uuid4().hex[:8]
        _transcript_received = time.monotonic()

        raw_text: str = getattr(event, "text", "").strip()
        if not raw_text:
            return

        # ── Step 0b: Strict Audio Ownership & Echo Gating Check ───────────
        tts_active = (
            getattr(event, "tts_active", False) is True
            or getattr(self._streaming_tts, "is_speaking", False) is True
            or (hasattr(self._engine, "is_speaking") and getattr(self._engine, "is_speaking", False) is True)
            or getattr(self._session, "state", None) in (VoiceSessionState.SPEAKING, VoiceSessionState.AUDIO_FLUSH)
        )
        flush_active = (
            getattr(event, "flush_active", False) is True
            or (hasattr(self._engine, "is_flushing") and getattr(self._engine, "is_flushing", False) is True)
            or getattr(self._session, "state", None) == VoiceSessionState.AUDIO_FLUSH
        )
        raw_source = getattr(event, "source", "user")
        source = raw_source if isinstance(raw_source, str) else "user"

        if tts_active or flush_active or source in ("tts_echo", "spidy_tts", "unknown"):
            log.info("[VOICE] transcript=rejected reason=tts_echo")
            log.info("[VOICE AUDIO] rejected source=tts_echo")
            session_state = getattr(self._session, "state", None)
            ss_name = session_state.name if hasattr(session_state, "name") else ("AWAKE" if getattr(self._session, "is_active", False) else "IDLE")
            cap_obj = getattr(getattr(self._engine, "_capture_engine", None), "mode", "unknown")
            cap_str = cap_obj.name if hasattr(cap_obj, "name") else str(cap_obj)
            log.info(
                "[VOICE FLOW] audio_source={src} capture_mode={cap} session_state={ss} "
                "tts_state={tts} stt_state={stt} wake_state={ws} transcript='{t}' "
                "origin={orig} accepted/rejected=rejected",
                src=str(source),
                cap=str(cap_str),
                ss=str(ss_name),
                tts=tts_active,
                stt="SPEAKING" if tts_active else ("AUDIO_FLUSH" if flush_active else "PROCESSING"),
                ws="ACTIVE" if getattr(self._session, "is_active", False) else "IDLE",
                t=raw_text[:40],
                orig=str(source),
            )
            return

        # -- Step 4: Session gate -- HARD DROP if no active session (with wake phrase recovery)
        if getattr(self._session, "is_active", False) is not True:
            clean_wake = raw_text.lower().strip().rstrip(".,!?")
            if clean_wake in (
                "wake up spidy", "wake up spidey",
                "hey spidy", "hey spidey",
                "ok spidy", "okay spidy", "spidy", "spidey",
            ):
                log.info(
                    "[WAKE] detected phrase='{phrase}' via deterministic speech recovery while session inactive",
                    phrase=clean_wake,
                )
                from spidy.perception.voice.engine import WakeWordDetectedEvent
                await self._bus.publish(
                    WakeWordDetectedEvent(
                        confidence=0.95,
                        model_name="speech_recovery",
                        wake_phrase=clean_wake.title(),
                    )
                )
                return

            log.warning(
                "[tx:{tx}] Transcript while SLEEPING (no active session) -- "
                "dropping. Only a real wake event may open a session.",
                tx=tx_id,
            )
            log.info("[VOICE AUDIO] rejected source=inactive_session")
            return

        trace = PipelineTracer.start_trace(raw_transcript=raw_text)
        trace.trace_id = tx_id

        # ── STT timing ────────────────────────────────────────────────────
        if self._measure_latency and self._speech_end_time is not None:
            stt_latency = (time.monotonic() - self._speech_end_time) * 1000
            trace.stt_ms = stt_latency
            log.info("[tx:{tx}] [Voice latency] stt_to_brain_start={ms:.0f}ms", tx=tx_id, ms=stt_latency)

        # -- Step 1: Normalize Transcript / ASR Recovery ---------------------------
        t_clean_start = time.monotonic()
        norm_result = self._normalizer.normalize(raw_text)
        text = norm_result.normalized_transcript
        trace.normalized_transcript = text
        trace.transcript_cleanup_ms = (time.monotonic() - t_clean_start) * 1000.0

        # If normalization or stripping consumed the entire utterance or utterance is just wake word, skip Brain
        if not text or text.lower().strip().rstrip(".,!?") in (
            "wake up spidy", "wake up spidey",
            "hey spidy", "hey spidey",
            "ok spidy", "okay spidy", "spidy", "spidey",
        ):
            log.debug("[tx:{tx}] Wake phrase/noise only -- skipping Brain.", tx=tx_id)
            return

        # ── Step 2: Transcript quality guard ───────────────────────────────
        # Reject clearly garbled/noise-only transcripts before the LLM call.
        segment_signals = getattr(event, "segment_signals", None)
        if (
            self._quality_guard.is_suspicious(raw_text, segments=segment_signals)
            or self._quality_guard.is_suspicious(text, segments=segment_signals)
        ):
            reason = self._quality_guard.suspicious_reason(raw_text, segments=segment_signals) or self._quality_guard.suspicious_reason(text, segments=segment_signals)
            log.info(
                "[tx:{tx}] Transcript rejected by quality guard (reason={r}): '{t}'",
                tx=tx_id, r=reason, t=raw_text[:60],
            )
            return

        # -- Step 2b: Transcript deduplication guard --------------------------------
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

        session_state = getattr(self._session, "state", None)
        ss_name = session_state.name if hasattr(session_state, "name") else ("AWAKE" if getattr(self._session, "is_active", False) else "IDLE")
        cap_obj = getattr(getattr(self._engine, "_capture_engine", None), "mode", "unknown")
        cap_str = cap_obj.name if hasattr(cap_obj, "name") else str(cap_obj)
        log.info("[tx:{tx}] Clean Transcript: '{t}'", tx=tx_id, t=text[:80])
        log.info("[VOICE AUDIO] accepted source=user")
        log.info(
            "[VOICE FLOW] audio_source=user capture_mode={cap} session_state={ss} "
            "tts_state=False stt_state=LISTENING wake_state={ws} transcript='{t}' "
            "origin=user accepted/rejected=accepted",
            cap=str(cap_str),
            ss=str(ss_name),
            ws="ACTIVE" if getattr(self._session, "is_active", False) else "IDLE",
            t=text[:40],
        )

        # ── Step 3: Interrupt detection ────────────────────────────────────
        interrupt_cmd = self._interruption.detect(text)
        if interrupt_cmd is not None:
            response = await self._interruption.execute(interrupt_cmd, text)
            if interrupt_cmd.is_cancellation:
                log.info("[tx:{tx}] Stop command -- restoring overlay and re-listen.", tx=tx_id)
                from spidy.ui.events import UIStateChangeEvent
                await self._bus.publish(UIStateChangeEvent(state="listening"))
                if self._continuous_mode and getattr(self._session, "is_active", False) is True:
                    if hasattr(self._engine, "signal_relisten"):
                        self._engine.signal_relisten()
            elif response:
                await self._streaming_tts.speak(response)
            trace_event = trace.finish()
            await self._bus.publish(trace_event)
            return

        # ── Step 5: Record utterance ───────────────────────────────────────
        await self._safe_session_call("record_utterance", text)

        # ── Step 6: Intent classification + Step 7: Brain routing ──────────
        _brain_start = time.monotonic()
        try:
            t_intent_start = time.monotonic()
            intent = await self._intent_clf.classify(text)
            _t_intent = (time.monotonic() - t_intent_start) * 1000
            trace.intent_ms = _t_intent
            log.info("[tx:{tx}] [Voice latency] intent_classify={ms:.0f}ms", tx=tx_id, ms=_t_intent)

            if self._goal_clf.is_executable(intent):
                response = await self._brain.run_goal(text)
            else:
                response = await self._brain.process(text)
        except Exception as exc:  # noqa: BLE001
            log.error("[tx:{tx}] Brain error: {exc}", tx=tx_id, exc=exc)
            response = "I ran into a problem. Please try again."
            from spidy.brain.events import BrainResponseReadyEvent
            await self._bus.publish(BrainResponseReadyEvent(
                session_id=getattr(self._session, "session_id", ""),
                response_text=response,
                decision_mode="error",
            ))

        # -- Step 8: Latency measurement & Trace Finalize --------------------------
        if self._measure_latency:
            _brain_total = (time.monotonic() - _brain_start) * 1000
            log.info("[tx:{tx}] [Voice latency] brain_roundtrip={ms:.0f}ms", tx=tx_id, ms=_brain_total)

        trace_event = trace.finish()
        await self._bus.publish(trace_event)

    async def _on_brain_response(self, event: Any) -> None:
        """
        Handle brain.response_ready → deliver via StreamingTTS.

        This handler intercepts the brain.response_ready event and delivers
        it through StreamingTTSWrapper instead of the plain VoiceEngine.speak().
        Both handlers receive the event; StreamingTTS takes priority for M14.
        """
        response_text: str = getattr(event, "response_text", "") or getattr(event, "text", "")
        if not response_text.strip():
            if hasattr(self._engine, "return_to_sleeping"):
                self._engine.return_to_sleeping()
            await self._safe_session_call("mark_response_complete")
            if self._continuous_mode and getattr(self._session, "is_active", False) is True:
                if hasattr(self._engine, "signal_relisten"):
                    self._engine.signal_relisten()
            return

        if getattr(self._session, "is_paused", False) is True:
            log.debug("ContinuousVoiceController: session paused -- skipping TTS.")
            if hasattr(self._engine, "return_to_sleeping"):
                self._engine.return_to_sleeping()
            await self._safe_session_call("mark_response_complete")
            return

        # Apply TTS text filter — strip markdown/code for natural speech.
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
            if hasattr(self._engine, "return_to_sleeping"):
                self._engine.return_to_sleeping()
            await self._safe_session_call("mark_response_complete")
            if self._continuous_mode and getattr(self._session, "is_active", False) is True:
                if hasattr(self._engine, "signal_relisten"):
                    self._engine.signal_relisten()
            return

        await self._safe_session_call("mark_speaking")
        if hasattr(self._engine, "set_speaking"):
            self._engine.set_speaking(True)

        # Switch mic to BARGE_IN mode so BargeInDetector can hear "Spidy stop".
        from spidy.perception.voice.audio_capture import CaptureMode
        capture = getattr(self._engine, "_capture_engine", None)

        if self._barge_in is not None and getattr(self._barge_in, "is_loaded", False) is True:
            if capture is not None:
                capture.set_mode(CaptureMode.BARGE_IN)
            self._barge_in.start(tts=self._streaming_tts, spoken_text=tts_text)
            log.info(
                "[BargeIn] CaptureMode.BARGE_IN activated — "
                "BargeInDetector listening for the user speaking over Spidy."
            )
        else:
            # No barge-in configured: fall back to IDLE (safe, mic muted)
            if capture is not None:
                capture.set_mode(CaptureMode.IDLE)
            log.debug("[BargeIn] barge-in not available — mic IDLE during TTS.")

        _tts_start = time.monotonic()
        log.info("[Voice latency] tts_start")
        try:
            await self._streaming_tts.speak(tts_text)
        finally:
            # ── The user talked over Spidy: skip the flush, record their new request ──
            interrupted = (
                self._barge_in is not None
                and getattr(self._barge_in, "interrupted", False) is True
            )
            if interrupted:
                self._resume_after_interrupt()

            # ── Always stop barge-in detection first ───────────────────────
            if self._barge_in is not None:
                self._barge_in.stop()

            _tts_ms = (time.monotonic() - _tts_start) * 1000
            log.info("[Voice latency] tts_complete={ms:.0f}ms", ms=_tts_ms)

            if interrupted:
                await self._after_interrupt()
            else:
                await self._finish_playback(capture)

    def _resume_after_interrupt(self) -> None:
        """
        The user spoke over Spidy and TTS has been cut off: record the rest of their request.

        Runs synchronously (no awaits) so the mic is back in a recording mode within milliseconds
        of the speech stopping; the audio barge-in collected so far is handed to the recorder.
        """
        log.info(
            "[BargeIn] user interrupted Spidy ('{t}') -- listening to the new request.",
            t=getattr(self._barge_in, "interrupt_text", "")[:60],
        )
        resume = getattr(self._engine, "resume_listening_after_interrupt", None)
        if callable(resume):
            resume(self._barge_in.consume_interrupt)
        elif hasattr(self._engine, "signal_relisten"):
            self._engine.signal_relisten()

    async def _after_interrupt(self) -> None:
        """Session bookkeeping once an interrupted utterance is over."""
        await self._safe_session_call("mark_response_complete")
        from spidy.ui.events import UIStateChangeEvent
        await self._bus.publish(UIStateChangeEvent(state="listening"))

    async def _finish_playback(self, capture: Any) -> None:
        """Normal end of an utterance: flush the echo, restore wake listening, re-listen if the session goes on."""
        from spidy.perception.voice.audio_capture import CaptureMode

        # ── Post-TTS Audio Flush & Echo Settle ─────────────────────────
        await self._safe_session_call("mark_audio_flush")
        if hasattr(self._engine, "enter_audio_flush"):
            await self._safe_engine_call("enter_audio_flush", settle_seconds=0.20)
        else:
            if hasattr(self._engine, "_wake_model") and hasattr(self._engine._wake_model, "reset_buffer"):
                self._engine._wake_model.reset_buffer()
            await asyncio.sleep(0.15)
            if hasattr(self._engine, "_capture_buffer"):
                with getattr(self._engine, "_buffer_lock", threading.Lock()):
                    self._engine._capture_buffer.clear()
            if hasattr(self._engine, "_wake_model") and hasattr(self._engine._wake_model, "reset_buffer"):
                self._engine._wake_model.reset_buffer()

        # Use the single recovery primitive so every normal response and
        # every session-end path has the same clean state: SLEEPING,
        # DETECTING, empty buffers, and reset wake-model history.
        if hasattr(self._engine, "return_to_sleeping"):
            self._engine.return_to_sleeping()
        else:
            if hasattr(self._engine, "exit_audio_flush"):
                self._engine.exit_audio_flush()
            if capture is not None:
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
        await self._safe_session_call("mark_response_complete")

        # ── Continuous conversation re-listen ──────────────────────────
        if (
            self._continuous_mode
            and getattr(self._session, "is_active", False) is True
            and not getattr(self._session, "is_paused", False)
        ):
            if hasattr(self._engine, "signal_relisten"):
                self._engine.signal_relisten()
            log.info(
                "ContinuousVoiceController: re-listen triggered "
                "(session={sid}).",
                sid=getattr(self._session, "session_id", ""),
            )
            from spidy.ui.events import UIStateChangeEvent
            await self._bus.publish(UIStateChangeEvent(state="listening"))

    async def _on_speaking_end(self, event: Any) -> None:
        """Record when speech ends (for latency measurement)."""
        self._speech_end_time = time.monotonic()

    async def _on_session_timeout(self, event: Any) -> None:
        """Session timed out — already deactivated by VoiceSessionManager. Restore wake listener."""
        log.info("ContinuousVoiceController: session timed out — restoring wake listener.")
        # 1. Return engine and capture to clean wake detection state
        if hasattr(self._engine, "return_to_sleeping"):
            self._engine.return_to_sleeping()
        else:
            if hasattr(self._engine, "exit_audio_flush"):
                self._engine.exit_audio_flush()
            if hasattr(self._engine, "_set_state"):
                from spidy.perception.voice.engine import VoiceState
                self._engine._set_state(VoiceState.SLEEPING)
            from spidy.perception.voice.audio_capture import CaptureMode
            capture = getattr(self._engine, "_capture_engine", None)
            if capture is not None:
                capture.set_mode(CaptureMode.DETECTING)
            if hasattr(self._engine, "_wake_model") and hasattr(self._engine._wake_model, "reset_buffer"):
                self._engine._wake_model.reset_buffer()
            if hasattr(self._engine, "_capture_buffer"):
                with getattr(self._engine, "_buffer_lock", threading.Lock()):
                    self._engine._capture_buffer.clear()

        # 2. Reset deduplication & transient state
        self._last_transcript = ""
        self._last_transcript_time = 0.0

        # 3. Overlay → SLEEPING (idle) when session expires
        from spidy.ui.events import UIStateChangeEvent
        await self._bus.publish(UIStateChangeEvent(state="idle"))

    async def _on_session_ended(self, event: Any) -> None:
        """Session ended (timeout, user stop, max turns, etc.) — ensure wake listener is restored."""
        reason = getattr(event, "reason", "unknown")
        log.info(
            "ContinuousVoiceController: session ended (reason={r}) — ensuring wake listener active.",
            r=reason,
        )
        if hasattr(self._engine, "return_to_sleeping"):
            self._engine.return_to_sleeping()
        else:
            if hasattr(self._engine, "exit_audio_flush"):
                self._engine.exit_audio_flush()
            if hasattr(self._engine, "_set_state"):
                from spidy.perception.voice.engine import VoiceState
                self._engine._set_state(VoiceState.SLEEPING)
            from spidy.perception.voice.audio_capture import CaptureMode
            capture = getattr(self._engine, "_capture_engine", None)
            if capture is not None:
                capture.set_mode(CaptureMode.DETECTING)
            if hasattr(self._engine, "_wake_model") and hasattr(self._engine._wake_model, "reset_buffer"):
                self._engine._wake_model.reset_buffer()
            if hasattr(self._engine, "_capture_buffer"):
                with getattr(self._engine, "_buffer_lock", threading.Lock()):
                    self._engine._capture_buffer.clear()

        self._last_transcript = ""
        self._last_transcript_time = 0.0

        if reason != "timeout":
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
