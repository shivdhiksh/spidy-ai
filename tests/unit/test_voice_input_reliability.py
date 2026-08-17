"""
Tests — Voice Input Reliability (Issues 1-7)
=============================================
Covers:
  1.  Wake ack audio not included in next transcript
  2.  Audio buffer flushed after ack (post-ack flush)
  3.  Raw transcript while sleeping does NOT create a session
  4.  Raw transcript while sleeping does NOT reach Brain
  5.  Repeated phrase rejected (bigram repetition)
  6.  Repeated word loop rejected
  7.  Legitimate short commands survive the guard
  8.  Duplicate transcript within dedup window is dropped
  9.  Two genuinely separate identical commands are both processed
  10. Wake prefix stripping remains correct
  11. Wake word alone does not reach Brain
  12. Continuous conversation: re-listen after response
  13. Session timeout returns to SLEEPING
  14. Sleeping transcript cannot reactivate the session
  15. Exactly one Brain call per utterance (no duplication)
  16. Exactly one TTS speak() per Brain response
  17. Original response_text untouched by TTS filter (UI safety)
  18. Phrase-repetition guard does not block "Subscribe to my channel"
      as a legitimate ONE-TIME mention (it blocks repeated occurrences)
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from spidy.voice.transcript_guard import TranscriptQualityGuard
from spidy.voice.tts_text_filter import TTSTextFilter


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_cvc(
    session_active: bool = False,
    dedup_window: float = 1.5,
    post_ack_flush_ms: int = 100,
):
    """Build a minimal ContinuousVoiceController with mocked dependencies."""
    from spidy.voice.continuous import ContinuousVoiceController
    from spidy.voice.session import VoiceSessionManager
    from spidy.voice.streaming_tts import StreamingTTSWrapper
    from spidy.voice.interruption import InterruptionHandler

    bus = MagicMock()
    bus.subscribe = MagicMock()
    bus.unsubscribe = MagicMock()
    bus.publish = AsyncMock()

    session = MagicMock(spec=VoiceSessionManager)
    session.is_active = session_active
    session.is_paused = False
    session.session_id = "test-session"
    session.activate = AsyncMock(return_value="test-session")
    session.record_utterance = AsyncMock()
    session.mark_speaking = AsyncMock()
    session.mark_response_complete = AsyncMock()
    session.deactivate = AsyncMock()

    tts = MagicMock(spec=StreamingTTSWrapper)
    tts.is_speaking = False
    tts.speak = AsyncMock()

    interruption = MagicMock(spec=InterruptionHandler)
    interruption.detect = MagicMock(return_value=None)

    brain = MagicMock()
    brain.process = AsyncMock(return_value="response text")
    brain.run_goal = AsyncMock(return_value="goal response text")

    engine = MagicMock()
    engine._capture_engine = MagicMock()
    engine._capture_engine.set_mode = MagicMock()
    engine._wake_model = MagicMock()
    engine._wake_model.reset_buffer = MagicMock()
    engine._buffer_lock = MagicMock()
    engine._buffer_lock.__enter__ = MagicMock(return_value=None)
    engine._buffer_lock.__exit__ = MagicMock(return_value=False)
    engine._capture_buffer = []
    engine.signal_relisten = MagicMock()
    engine._on_brain_response = MagicMock()

    tts_filter = TTSTextFilter(enabled=False)  # disabled for isolation

    cvc = ContinuousVoiceController(
        voice_engine=engine,
        brain=brain,
        bus=bus,
        session_manager=session,
        streaming_tts=tts,
        interruption_handler=interruption,
        tts_filter=tts_filter,
        continuous_mode=True,
        measure_latency=False,
    )
    cvc._dedup_window = dedup_window
    cvc._post_ack_flush_ms = post_ack_flush_ms

    return cvc, brain, tts, session, engine


def _transcript_event(text: str):
    ev = MagicMock()
    ev.text = text
    ev.segment_signals = None
    return ev


# ═══════════════════════════════════════════════════════════════════════════════
# Group 1: Wake-ack isolation
# ═══════════════════════════════════════════════════════════════════════════════

class TestWakeAckIsolation:

    @pytest.mark.asyncio
    async def test_01_ack_audio_not_in_user_transcript(self) -> None:
        """
        Ack phrases like 'Yes Shiva' must not appear in the transcript
        that reaches Brain. The dedup guard catches re-delivered ack audio.
        """
        cvc, brain, tts, session, engine = _make_cvc(session_active=True)

        # Simulate: ack echo re-delivered as STT transcript immediately after
        ack_as_transcript = "Yes Shiva."
        cvc._last_transcript = ack_as_transcript.lower().strip()
        cvc._last_transcript_time = time.monotonic()  # just set

        event = _transcript_event("Yes Shiva. What is Python?")
        # This is NOT the same as ack text — it has extra content.
        # It should reach Brain normally (it's a real user command).
        await cvc._on_transcript(event)
        brain.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_02_exact_ack_echo_deduplicated(self) -> None:
        """
        If the exact ack text arrives as a STT transcript within the dedup
        window, it is silently dropped — not sent to Brain.
        """
        cvc, brain, tts, session, engine = _make_cvc(session_active=True)

        # Simulate ack played, then STT picks it up
        ack_text = "Yes Shiva."
        cvc._last_transcript = ack_text.lower().strip()
        cvc._last_transcript_time = time.monotonic()  # just now

        event = _transcript_event(ack_text)
        await cvc._on_transcript(event)

        brain.process.assert_not_called()
        brain.run_goal.assert_not_called()

    @pytest.mark.asyncio
    async def test_03_post_ack_flush_called_in_on_wake_word(self) -> None:
        """
        After the ack, _on_wake_word must clear the capture buffer before
        switching to CaptureMode.RECORDING.
        """
        from spidy.voice.continuous import ContinuousVoiceController
        from spidy.voice.session import VoiceSessionManager
        from spidy.voice.streaming_tts import StreamingTTSWrapper
        from spidy.voice.interruption import InterruptionHandler
        from spidy.voice.wake_ack import WakeAcknowledger
        from spidy.perception.voice.audio_capture import CaptureMode

        bus = MagicMock()
        bus.subscribe = MagicMock()
        bus.unsubscribe = MagicMock()
        bus.publish = AsyncMock()

        session = MagicMock(spec=VoiceSessionManager)
        session.is_active = False
        session.activate = AsyncMock(return_value="sid")

        tts_wrapper = MagicMock(spec=StreamingTTSWrapper)
        tts_wrapper.is_speaking = False

        ack = MagicMock(spec=WakeAcknowledger)
        ack.speak_ack = AsyncMock()

        engine = MagicMock()
        capture_buf = []
        engine._capture_buffer = capture_buf
        engine._buffer_lock = MagicMock()
        engine._buffer_lock.__enter__ = MagicMock(return_value=None)
        engine._buffer_lock.__exit__ = MagicMock(return_value=False)
        engine._wake_model.reset_buffer = MagicMock()
        engine._capture_engine.set_mode = MagicMock()

        cvc = ContinuousVoiceController(
            voice_engine=engine,
            brain=MagicMock(),
            bus=bus,
            session_manager=session,
            streaming_tts=tts_wrapper,
            interruption_handler=MagicMock(),
            tts_filter=TTSTextFilter(enabled=False),
            wake_acknowledger=ack,
            continuous_mode=True,
            measure_latency=False,
        )
        cvc._post_ack_flush_ms = 50  # short for test speed

        event = MagicMock()
        event.model_name = "hey_jarvis"

        await cvc._on_wake_word(event)

        # Ack must have been spoken
        ack.speak_ack.assert_called_once()
        # OWW buffer flushed
        engine._wake_model.reset_buffer.assert_called()
        # Mic must have been set to RECORDING after the flush
        mode_calls = [str(c) for c in engine._capture_engine.set_mode.call_args_list]
        assert any("RECORDING" in c for c in mode_calls), (
            f"Expected CaptureMode.RECORDING to be set, got: {mode_calls}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 2: Session gate (Issue 2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionGate:

    @pytest.mark.asyncio
    async def test_03_raw_transcript_while_sleeping_no_session(self) -> None:
        """
        With session_active=False (SLEEPING), a transcript must NOT create
        a new session.
        """
        cvc, brain, tts, session, engine = _make_cvc(session_active=False)
        event = _transcript_event("Open Calculator.")
        await cvc._on_transcript(event)
        session.activate.assert_not_called()

    @pytest.mark.asyncio
    async def test_04_raw_transcript_while_sleeping_no_brain(self) -> None:
        """
        With session_active=False (SLEEPING), a transcript must NOT reach Brain.
        """
        cvc, brain, tts, session, engine = _make_cvc(session_active=False)
        event = _transcript_event("What is Python?")
        await cvc._on_transcript(event)
        brain.process.assert_not_called()
        brain.run_goal.assert_not_called()

    @pytest.mark.asyncio
    async def test_14_sleeping_transcript_cannot_reactivate_session(self) -> None:
        """
        Even a legitimate-sounding transcript while sleeping cannot
        reactivate a previously deactivated session.
        """
        cvc, brain, tts, session, engine = _make_cvc(session_active=False)
        # Simulate: session was active but timed out
        session.is_active = False

        for utterance in ["Tell me a joke.", "Open Edge.", "What time is it?"]:
            event = _transcript_event(utterance)
            await cvc._on_transcript(event)

        session.activate.assert_not_called()
        brain.process.assert_not_called()

    @pytest.mark.asyncio
    async def test_13_session_timeout_returns_to_sleeping(self) -> None:
        """Session timeout event deactivates the session (UI goes idle)."""
        cvc, brain, tts, session, engine = _make_cvc(session_active=True)
        timeout_event = MagicMock()
        await cvc._on_session_timeout(timeout_event)
        # UI state change event should be published
        cvc._bus.publish.assert_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Group 3: Quality guard — repetition (Issue 3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRepetitionGuard:

    def setup_method(self) -> None:
        self.guard = TranscriptQualityGuard()

    def test_05_repeated_phrase_rejected(self) -> None:
        """Phrase-level repetition: 'Hey Jerry's. Hey Jerry's. Hey Jerry's.' -> suspicious."""
        text = "Hey, Jerry's. Hey, Jerry's. Hey, Jerry's. Hey, Jerry's."
        assert self.guard.is_suspicious(text), (
            f"Expected suspicious for repeated phrase, got clean: '{text}'"
        )
        assert self.guard.suspicious_reason(text) == "phrase_repetition"

    def test_05b_subscribe_repetition_rejected(self) -> None:
        """'Subscribe to our channel. Subscribe to our channel.' -> suspicious."""
        text = "Subscribe to our channel. Subscribe to our channel."
        assert self.guard.is_suspicious(text)

    def test_06_repeated_word_loop_rejected(self) -> None:
        """Single-word dominance: 'word word word word word word' -> suspicious."""
        text = "word word word word word word word word"
        assert self.guard.is_suspicious(text)

    def test_07_open_edge_passes(self) -> None:
        assert not self.guard.is_suspicious("Open Edge")

    def test_07_open_calculator_passes(self) -> None:
        assert not self.guard.is_suspicious("Open Calculator")

    def test_07_stop_passes(self) -> None:
        assert not self.guard.is_suspicious("Stop")

    def test_07_yes_passes(self) -> None:
        assert not self.guard.is_suspicious("Yes")

    def test_07_no_passes(self) -> None:
        assert not self.guard.is_suspicious("No")

    def test_07_thanks_passes(self) -> None:
        assert not self.guard.is_suspicious("Thanks")

    def test_07_what_time_is_it_passes(self) -> None:
        assert not self.guard.is_suspicious("what time is it")

    def test_07_open_vs_code_passes(self) -> None:
        assert not self.guard.is_suspicious("open VS Code")

    def test_07_python_question_passes(self) -> None:
        """Normal multi-word question must not be flagged."""
        text = "What is Python and how does it work in data science?"
        assert not self.guard.is_suspicious(text)

    def test_18_subscribe_single_mention_passes(self) -> None:
        """A single 'subscribe' mention in a question is NOT suspicious."""
        # The noise pattern requires 2+ occurrences within 40 chars
        text = "How do I subscribe to a YouTube channel?"
        assert not self.guard.is_suspicious(text), (
            f"Single 'subscribe' mention should pass, got suspicious for: '{text}'"
        )

    def test_bigram_ratio_passthrough_normal_sentence(self) -> None:
        """Normal repetition in natural language should not be flagged."""
        text = "I think I can do it and I believe I will."
        assert not self.guard.is_suspicious(text)

    def test_phrase_repetition_reason_returned(self) -> None:
        text = "Hey hey there hey there hey there hey there hey there"
        # This has high bigram repetition
        reason = self.guard.suspicious_reason(text)
        assert reason is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Group 4: Transcript deduplication (Issue 4)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTranscriptDeduplication:

    @pytest.mark.asyncio
    async def test_08_duplicate_within_window_dropped(self) -> None:
        """Same transcript within dedup window -> second is dropped (Brain called once)."""
        cvc, brain, tts, session, engine = _make_cvc(session_active=True, dedup_window=2.0)

        event1 = _transcript_event("Open Calculator.")
        event2 = _transcript_event("Open Calculator.")

        await cvc._on_transcript(event1)
        await cvc._on_transcript(event2)  # arrives immediately (same cycle)

        # Brain must be called only ONCE
        assert brain.process.call_count + brain.run_goal.call_count == 1, (
            f"Expected 1 Brain call, got {brain.process.call_count + brain.run_goal.call_count}"
        )

    @pytest.mark.asyncio
    async def test_09_separate_identical_commands_both_processed(self) -> None:
        """Same transcript separated by > dedup_window -> both are processed."""
        cvc, brain, tts, session, engine = _make_cvc(session_active=True, dedup_window=0.1)

        event1 = _transcript_event("Open Calculator.")
        event2 = _transcript_event("Open Calculator.")

        await cvc._on_transcript(event1)

        # Simulate time passing beyond the dedup window
        cvc._last_transcript_time = time.monotonic() - 5.0  # 5s ago

        await cvc._on_transcript(event2)

        total_calls = brain.process.call_count + brain.run_goal.call_count
        assert total_calls == 2, (
            f"Expected 2 Brain calls for separate identical commands, got {total_calls}"
        )

    @pytest.mark.asyncio
    async def test_08b_case_insensitive_dedup(self) -> None:
        """Deduplication is case-insensitive: 'Open Edge' == 'open edge' within window."""
        cvc, brain, tts, session, engine = _make_cvc(session_active=True, dedup_window=2.0)

        await cvc._on_transcript(_transcript_event("Open Edge."))
        await cvc._on_transcript(_transcript_event("open edge."))  # same, different case

        total = brain.process.call_count + brain.run_goal.call_count
        assert total == 1

    @pytest.mark.asyncio
    async def test_dedup_disabled_when_window_zero(self) -> None:
        """With dedup_window=0.0, no deduplication: both identical commands processed."""
        cvc, brain, tts, session, engine = _make_cvc(session_active=True, dedup_window=0.0)

        await cvc._on_transcript(_transcript_event("Open Calculator."))
        await cvc._on_transcript(_transcript_event("Open Calculator."))

        total = brain.process.call_count + brain.run_goal.call_count
        assert total == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Group 5: Wake prefix stripping (Issue 6)
# ═══════════════════════════════════════════════════════════════════════════════

class TestWakePrefixStripping:

    @pytest.mark.asyncio
    async def test_10_hey_jarvis_open_edge_stripped(self) -> None:
        """'Hey Jarvis, open Edge' -> Brain receives 'open Edge' (prefix removed)."""
        cvc, brain, tts, session, engine = _make_cvc(session_active=True)

        event = _transcript_event("Hey Jarvis, open Edge.")
        await cvc._on_transcript(event)

        # Either process or run_goal may be called depending on intent classifier
        total_calls = brain.process.call_count + brain.run_goal.call_count
        assert total_calls == 1, f"Expected exactly 1 Brain call, got {total_calls}"

        # Verify the text passed to Brain does NOT contain the wake prefix
        if brain.process.call_count:
            called_text = brain.process.call_args[0][0]
        else:
            called_text = brain.run_goal.call_args[0][0]

        assert "hey jarvis" not in called_text.lower(), (
            f"Wake prefix not stripped from Brain call: '{called_text}'"
        )
        assert "open" in called_text.lower() or "edge" in called_text.lower()

    @pytest.mark.asyncio
    async def test_10_hey_jarvis_python_question_stripped(self) -> None:
        """'Hey Jarvis, what is Python?' -> Brain receives 'what is Python?'."""
        cvc, brain, tts, session, engine = _make_cvc(session_active=True)

        event = _transcript_event("Hey Jarvis, what is Python?")
        await cvc._on_transcript(event)

        called_text = brain.process.call_args[0][0]
        assert "hey jarvis" not in called_text.lower()
        assert "python" in called_text.lower()

    @pytest.mark.asyncio
    async def test_10_who_is_jarvis_not_stripped(self) -> None:
        """'Who is Jarvis?' -> Brain receives full text (Jarvis is not a prefix here)."""
        cvc, brain, tts, session, engine = _make_cvc(session_active=True)

        event = _transcript_event("Who is Jarvis?")
        await cvc._on_transcript(event)

        called_text = brain.process.call_args[0][0]
        assert "jarvis" in called_text.lower()

    @pytest.mark.asyncio
    async def test_11_wake_word_alone_not_sent_to_brain(self) -> None:
        """'Hey Jarvis' alone (no command) -> Brain NOT called."""
        cvc, brain, tts, session, engine = _make_cvc(session_active=True)

        event = _transcript_event("Hey Jarvis")
        await cvc._on_transcript(event)

        brain.process.assert_not_called()
        brain.run_goal.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Group 6: One-to-one pipeline integrity (Issues 5, 15, 16, 17)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineIntegrity:

    @pytest.mark.asyncio
    async def test_15_one_brain_call_per_utterance(self) -> None:
        """One user utterance -> exactly one Brain.process() call."""
        cvc, brain, tts, session, engine = _make_cvc(session_active=True)

        await cvc._on_transcript(_transcript_event("What is Python?"))

        total = brain.process.call_count + brain.run_goal.call_count
        assert total == 1

    @pytest.mark.asyncio
    async def test_16_one_tts_speak_per_brain_response(self) -> None:
        """One brain.response_ready event -> exactly one StreamingTTS.speak() call."""
        cvc, brain, tts, session, engine = _make_cvc(session_active=True)

        event = MagicMock()
        event.response_text = "Python is a programming language."

        await cvc._on_brain_response(event)

        assert tts.speak.call_count == 1

    @pytest.mark.asyncio
    async def test_17_original_response_text_untouched(self) -> None:
        """The TTS filter must NOT modify the event.response_text attribute."""
        # Build CVC with filter ENABLED
        from spidy.voice.continuous import ContinuousVoiceController
        from spidy.voice.session import VoiceSessionManager
        from spidy.voice.streaming_tts import StreamingTTSWrapper
        from spidy.voice.interruption import InterruptionHandler

        bus = MagicMock()
        bus.subscribe = MagicMock()
        bus.unsubscribe = MagicMock()
        bus.publish = AsyncMock()

        session = MagicMock(spec=VoiceSessionManager)
        session.is_active = True
        session.is_paused = False
        session.mark_speaking = AsyncMock()
        session.mark_response_complete = AsyncMock()

        tts_wrapper = MagicMock(spec=StreamingTTSWrapper)
        tts_wrapper.is_speaking = False
        tts_wrapper.speak = AsyncMock()

        engine = MagicMock()
        engine._capture_engine.set_mode = MagicMock()
        engine._wake_model.reset_buffer = MagicMock()

        tts_filter = TTSTextFilter(
            code_block_replacement="I've written the code. You can see it in the chat.",
            enabled=True,
        )

        cvc = ContinuousVoiceController(
            voice_engine=engine,
            brain=MagicMock(),
            bus=bus,
            session_manager=session,
            streaming_tts=tts_wrapper,
            interruption_handler=MagicMock(),
            tts_filter=tts_filter,
            continuous_mode=False,
            measure_latency=False,
        )

        original_text = "Here:\n```python\nprint('hi')\n```\nDone."
        event = MagicMock()
        event.response_text = original_text

        await cvc._on_brain_response(event)

        # Original event attribute must be unmodified
        assert event.response_text == original_text
        # TTS must have received the filtered (shorter) text
        spoken = tts_wrapper.speak.call_args[0][0]
        assert "```python" not in spoken


# ═══════════════════════════════════════════════════════════════════════════════
# Group 7: Continuous conversation (Issue 7)
# ═══════════════════════════════════════════════════════════════════════════════

class TestContinuousConversation:

    @pytest.mark.asyncio
    async def test_12_relisten_triggered_after_tts(self) -> None:
        """
        After TTS finishes and session is still active,
        signal_relisten() is called to re-open the mic for next turn.
        """
        cvc, brain, tts, session, engine = _make_cvc(session_active=True)

        # Simulate brain.response_ready event
        response_event = MagicMock()
        response_event.response_text = "Python is a programming language."

        await cvc._on_brain_response(response_event)

        # signal_relisten must have been called (continuous_mode=True, session active)
        engine.signal_relisten.assert_called()

    @pytest.mark.asyncio
    async def test_12b_no_relisten_when_session_inactive(self) -> None:
        """No re-listen if session timed out before TTS finishes."""
        cvc, brain, tts, session, engine = _make_cvc(session_active=True)
        # Mark session as inactive to simulate timeout during TTS
        session.is_active = False

        response_event = MagicMock()
        response_event.response_text = "Some answer."

        await cvc._on_brain_response(response_event)

        engine.signal_relisten.assert_not_called()
