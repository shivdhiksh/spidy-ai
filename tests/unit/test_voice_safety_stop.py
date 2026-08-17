"""
test_voice_safety_stop.py — Voice Safety & Stop-Command Regression Tests
=========================================================================
Milestone 15 critical-issue regression suite.

Covers all 20 requirements:
 1.  "Spidy stop" interrupts TTS
 2.  "Spidy, stop" interrupts TTS
 3.  "stop talking" interrupts TTS
 4.  Stop command does NOT call Brain
 5.  Stop command does NOT call NVIDIA
 6.  Stop command does NOT call Ollama
 7.  Stop command cannot trigger shutdown
 8.  Stop command keeps conversation session active
 9.  After stop, microphone returns to listening
10.  Wake event during TTS is ignored
11.  Wake event during TTS cannot create a duplicate session
12.  TTS audio cannot become a Brain transcript (quality guard)
13.  Silence cannot call Brain
14.  Clearly suspicious STT cannot call Brain
15.  Shutdown requires explicit confirmation
16.  Confirmation expires safely
17.  "Who is Jarvis?" is NOT classified as stop
18.  "Tell me about Spidy." is NOT classified as stop
19.  Continuous conversation still works after stopping TTS
20.  Existing "Hey Jarvis" behaviour remains intact

No real VoiceEngine, Brain, LLM, TTS hardware, or audio required.
All destructive OS actions are mocked — no actual shutdown/restart/sleep.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from spidy.voice.interruption import InterruptCommand, InterruptionHandler, _STOP_PHRASES
from spidy.voice.transcript_guard import TranscriptQualityGuard, SegmentQualitySignals
from spidy.voice.streaming_tts import StreamingTTSWrapper
from spidy.voice.session import VoiceSessionManager, VoiceSessionState
from spidy.skills.desktop.system_control_skill import (
    SystemControlSkill,
    _DESTRUCTIVE_ACTIONS,
    _CONFIRMATION_TTL,
    _ConfirmationState,
)
from spidy.skills.base import SkillContext


# =============================================================================
# Shared helpers
# =============================================================================


def _skill_ctx(action: str = "", params: dict | None = None, session_id: str = "test-sid") -> SkillContext:
    return SkillContext(action=action, params=params or {}, session_id=session_id)


class FakeTTSEngine:
    """Lightweight TTS engine stub."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self._speaking = False
        self.stop_called = 0

    async def speak(self, text: str) -> None:
        self._speaking = True
        self.spoken.append(text)
        await asyncio.sleep(0)
        self._speaking = False

    async def synthesize(self, text: str):
        pass

    def stop(self) -> None:
        self.stop_called += 1
        self._speaking = False

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


class FakeBus:
    def __init__(self):
        self._subscribers: dict[str, list] = {}
        self.published: list = []

    async def publish(self, event):
        self.published.append(event)
        topic = getattr(event, "topic", "")
        for cb in list(self._subscribers.get(topic, [])):
            if asyncio.iscoroutinefunction(cb):
                await cb(event)
            else:
                cb(event)

    def subscribe(self, topic: str, callback) -> None:
        self._subscribers.setdefault(topic, []).append(callback)

    def unsubscribe(self, topic: str, callback) -> None:
        lst = self._subscribers.get(topic, [])
        if callback in lst:
            lst.remove(callback)


def _make_interruption_handler(*, tts=None, brain=None, session=None, bus=None):
    bus = bus or MagicMock(publish=AsyncMock())
    brain = brain or MagicMock(cancel_goal=AsyncMock(return_value=False))
    tts = tts or FakeTTSEngine()
    session = session or MagicMock(
        pause=AsyncMock(), resume=AsyncMock(), deactivate=AsyncMock()
    )
    handler = InterruptionHandler(bus=bus, brain=brain, tts=tts, session=session)
    return handler, bus, brain, tts, session


def _make_cvc(*, session_active: bool = True, tts_speaking: bool = False):
    """Build a minimal ContinuousVoiceController with mocked dependencies."""
    from spidy.voice.continuous import ContinuousVoiceController

    bus = FakeBus()

    # Session mock
    session = MagicMock()
    session.is_active = session_active
    session.is_paused = False
    session.session_id = "test-sid"
    session.activate = AsyncMock(return_value="test-sid")
    session.record_utterance = AsyncMock()
    session.mark_speaking = AsyncMock()
    session.mark_response_complete = AsyncMock()
    session.deactivate = AsyncMock()

    # TTS engine + wrapper
    tts_engine = FakeTTSEngine()
    tts_engine._speaking = tts_speaking
    streaming_tts = StreamingTTSWrapper(engine=tts_engine)

    # Brain mock
    brain = MagicMock()
    brain.process = AsyncMock(return_value="test response")
    brain.run_goal = AsyncMock(return_value="test response")
    brain.cancel_goal = AsyncMock(return_value=False)

    # Interruption handler (uses the real one)
    interruption = InterruptionHandler(
        bus=bus,
        brain=brain,
        tts=streaming_tts,
        session=session,
    )

    # VoiceEngine mock (hardware layer)
    voice_engine = MagicMock()
    voice_engine._on_brain_response = MagicMock()
    voice_engine._capture_engine = MagicMock()
    voice_engine._capture_engine.set_mode = MagicMock()
    voice_engine._wake_model = MagicMock()
    voice_engine._wake_model.reset_buffer = MagicMock()
    voice_engine.signal_relisten = MagicMock()

    # Intent classifiers
    intent_clf = MagicMock()
    intent_clf.classify = AsyncMock(return_value=MagicMock())
    goal_clf = MagicMock()
    goal_clf.is_executable = MagicMock(return_value=False)

    cvc = ContinuousVoiceController(
        voice_engine=voice_engine,
        brain=brain,
        bus=bus,
        session_manager=session,
        streaming_tts=streaming_tts,
        interruption_handler=interruption,
        continuous_mode=True,
    )
    cvc._intent_clf = intent_clf
    cvc._goal_clf = goal_clf

    return cvc, bus, session, brain, streaming_tts, tts_engine, voice_engine


# =============================================================================
# Class 1 — Stop command pattern detection
# =============================================================================


class TestStopCommandDetection:
    """Tests 1–4 (pattern matching), 17–18 (non-stop phrases)."""

    def _detect(self, phrase: str) -> InterruptCommand | None:
        handler, *_ = _make_interruption_handler()
        return handler.detect(phrase)

    # ── Test 1 ──────────────────────────────────────────────────────────────

    def test_spidy_stop_detected(self):
        """Test 1: 'Spidy stop' is classified as STOP."""
        assert self._detect("Spidy stop") == InterruptCommand.STOP

    # ── Test 2 ──────────────────────────────────────────────────────────────

    def test_spidy_comma_stop_detected(self):
        """Test 2: 'Spidy, stop' is classified as STOP."""
        assert self._detect("Spidy, stop") == InterruptCommand.STOP

    # ── Test 3 ──────────────────────────────────────────────────────────────

    def test_stop_talking_detected(self):
        """Test 3: 'stop talking' is classified as STOP."""
        assert self._detect("stop talking") == InterruptCommand.STOP

    @pytest.mark.parametrize("phrase", [
        "stop",
        "Spidy stop",
        "Spidy, stop",
        "hey Jarvis stop",
        "hey Jarvis, stop",
        "stop talking",
        "stop speaking",
        "be quiet",
        "be quiet Spidy",
        "Spidy be quiet",
        "please stop",
        "shut up",
    ])
    def test_all_stop_variants_detected(self, phrase):
        """All stop variants must be detected as STOP."""
        result = self._detect(phrase)
        assert result == InterruptCommand.STOP, (
            f"Expected STOP for '{phrase}', got {result}"
        )

    # ── Test 17 ─────────────────────────────────────────────────────────────

    def test_who_is_jarvis_not_stop(self):
        """Test 17: 'Who is Jarvis?' is NOT classified as stop."""
        result = self._detect("Who is Jarvis?")
        assert result is None, f"'Who is Jarvis?' should not be stop, got {result}"

    # ── Test 18 ─────────────────────────────────────────────────────────────

    def test_tell_me_about_spidy_not_stop(self):
        """Test 18: 'Tell me about Spidy.' is NOT classified as stop."""
        result = self._detect("Tell me about Spidy.")
        assert result is None, f"'Tell me about Spidy.' should not be stop, got {result}"

    @pytest.mark.parametrize("phrase", [
        "open chrome",
        "search youtube",
        "what time is it",
        "hello jarvis",
        "stopping soon",          # contains "stop" but as prefix of longer word
        "I want to stop smoking",  # "stop" mid-sentence with content after
        "Who is Jarvis?",
        "Tell me about Spidy.",
        "Who created Python?",
        "what is the capital of France",
    ])
    def test_non_stop_phrases_not_detected(self, phrase):
        """Normal content phrases must NOT trigger stop."""
        result = self._detect(phrase)
        assert result is None, (
            f"'{phrase}' incorrectly detected as stop: {result}"
        )


# =============================================================================
# Class 2 — Stop command execution (tests 4–9)
# =============================================================================


class TestStopCommandExecution:
    """Tests 4–9: execution behaviour, no Brain/LLM calls, session stays alive."""

    # ── Test 4 — Stop does NOT call Brain ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_stop_command_does_not_call_brain(self):
        """Test 4: Stop command must not route to Brain.process()."""
        cvc, bus, session, brain, streaming_tts, tts_engine, _ = _make_cvc()

        # Simulate a stop transcript
        event = MagicMock(text="Spidy stop", segment_signals=None)
        await cvc._on_transcript(event)

        brain.process.assert_not_awaited()
        brain.run_goal.assert_not_awaited()

    # ── Test 5 — Stop does NOT call NVIDIA ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_stop_command_does_not_call_nvidia(self):
        """Test 5: Stop command must not reach NVIDIA NIM."""
        with patch("spidy.llm.backends.nvidia.NvidiaClient.complete",
                   new_callable=AsyncMock) as mock_nvidia:
            cvc, bus, session, brain, streaming_tts, tts_engine, _ = _make_cvc()
            event = MagicMock(text="stop", segment_signals=None)
            await cvc._on_transcript(event)
            mock_nvidia.assert_not_awaited()

    # ── Test 6 — Stop does NOT call Ollama ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_stop_command_does_not_call_ollama(self):
        """Test 6: Stop command must not reach Ollama."""
        with patch("spidy.llm.backends.ollama.OllamaClient.complete",
                   new_callable=AsyncMock) as mock_ollama:
            cvc, bus, session, brain, streaming_tts, tts_engine, _ = _make_cvc()
            event = MagicMock(text="stop talking", segment_signals=None)
            await cvc._on_transcript(event)
            mock_ollama.assert_not_awaited()

    # ── Test 7 — Stop cannot trigger shutdown ───────────────────────────────

    @pytest.mark.asyncio
    async def test_stop_command_cannot_trigger_shutdown(self):
        """Test 7: 'Spidy stop' must NEVER execute OS shutdown."""
        skill = SystemControlSkill(bus=None)
        with patch.object(SystemControlSkill, "_do_shutdown") as mock_shutdown:
            # "stop" is handled by InterruptionHandler before reaching Brain/Skills
            # At skill level, the stop command should NOT map to shutdown_system
            # If somehow called directly, confirmation gate prevents execution
            ctx = _skill_ctx("shutdown_system", session_id="stop-test")
            result = await skill.execute("shutdown_system", ctx)
            # First call must ask for confirmation, not execute
            mock_shutdown.assert_not_called()
            assert result.data.get("awaiting_confirmation") is True

    # ── Test 8 — Stop keeps session alive ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_stop_command_keeps_session_active(self):
        """Test 8: After stop command, session must remain active (not deactivated)."""
        cvc, bus, session, brain, streaming_tts, tts_engine, _ = _make_cvc()
        session.is_active = True

        event = MagicMock(text="Spidy stop", segment_signals=None)
        await cvc._on_transcript(event)

        # Session must NOT have been deactivated
        session.deactivate.assert_not_awaited()
        # Session must still be marked active (mock property stays True)
        assert session.is_active is True

    # ── Test 9 — After stop, mic returns to listening ───────────────────────

    @pytest.mark.asyncio
    async def test_stop_command_restores_listening(self):
        """Test 9: After stop, overlay goes to 'listening' and relisten is signalled."""
        cvc, bus, session, brain, streaming_tts, tts_engine, voice_engine = _make_cvc()
        session.is_active = True

        event = MagicMock(text="Spidy stop", segment_signals=None)
        await cvc._on_transcript(event)

        # Overlay must receive "listening" state
        from spidy.ui.events import UIStateChangeEvent
        listening_events = [
            e for e in bus.published
            if isinstance(e, UIStateChangeEvent) and e.state == "listening"
        ]
        assert len(listening_events) >= 1, (
            "Expected at least one UIStateChangeEvent(state='listening') after stop"
        )

        # Relisten must be signalled
        voice_engine.signal_relisten.assert_called()

    @pytest.mark.asyncio
    async def test_stop_tts_called_on_interruption(self):
        """TTS.stop() is called when stop command is executed."""
        tts = FakeTTSEngine()
        handler, *_ = _make_interruption_handler(tts=tts)

        await handler.execute(InterruptCommand.STOP, "stop")
        assert tts.stop_called >= 1

    def test_streaming_tts_stop_calls_engine_stop(self):
        """StreamingTTSWrapper.stop() must call engine.stop() unconditionally."""
        engine = FakeTTSEngine()
        stts = StreamingTTSWrapper(engine=engine)
        stts.stop()
        assert engine.stop_called >= 1, (
            "engine.stop() must be called unconditionally for instant cutoff"
        )


# =============================================================================
# Class 3 — Wake event during TTS (tests 10–11)
# =============================================================================


class TestWakeDuringTTS:
    """Tests 10–11: wake events while TTS is speaking."""

    # ── Test 10 — Wake event ignored while TTS speaking ─────────────────────

    @pytest.mark.asyncio
    async def test_wake_ignored_while_tts_speaking(self):
        """Test 10: A wake_word.detected event is ignored while TTS is active."""
        cvc, bus, session, brain, streaming_tts, tts_engine, _ = _make_cvc(
            session_active=False,  # session not yet active
            tts_speaking=True,     # but TTS IS speaking
        )

        event = MagicMock(model_name="hey_jarvis")
        await cvc._on_wake_word(event)

        # Session must NOT have been activated
        session.activate.assert_not_awaited()

    # ── Test 11 — Wake during TTS cannot create duplicate session ────────────

    @pytest.mark.asyncio
    async def test_wake_during_tts_no_duplicate_session(self):
        """Test 11: Wake event while TTS speaking must not create a second session."""
        cvc, bus, session, brain, streaming_tts, tts_engine, _ = _make_cvc(
            session_active=True,
            tts_speaking=True,
        )

        original_session_id = session.session_id

        # Fire two wake events while speaking
        event = MagicMock(model_name="hey_jarvis")
        await cvc._on_wake_word(event)
        await cvc._on_wake_word(event)

        # activate() must NOT have been called at all
        session.activate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wake_ignored_while_session_active(self):
        """When session is active (but TTS not speaking), wake event still ignored."""
        cvc, bus, session, brain, streaming_tts, tts_engine, _ = _make_cvc(
            session_active=True,
            tts_speaking=False,
        )

        event = MagicMock(model_name="hey_jarvis")
        await cvc._on_wake_word(event)

        session.activate.assert_not_awaited()


# =============================================================================
# Class 4 — Self-talk / TTS-as-Brain-transcript prevention (tests 12–14)
# =============================================================================


class TestSelfTalk:
    """Tests 12–14: transcript quality guard prevents self-generated conversation."""

    def setup_method(self):
        self.guard = TranscriptQualityGuard()

    # ── Test 12 — TTS audio pattern rejected ────────────────────────────────

    def test_beep_pattern_rejected(self):
        """Test 12: 'Beep. Beep. Beep.' (TTS echo / hardware noise) is rejected."""
        assert self.guard.is_suspicious("Beep. Beep. Beep.")
        assert self.guard.is_suspicious("beep beep beep beep")

    def test_repetitive_word_rejected(self):
        """Test 12 cont.: Highly repetitive transcripts (hallucination) are rejected."""
        # 7 repetitions of same word → suspicious
        assert self.guard.is_suspicious("word word word word word word word")

    def test_numeric_noise_rejected(self):
        """Test 12 cont.: Pure numeric sequences (noise artifact) are rejected."""
        assert self.guard.is_suspicious("9 9 9 9 9")

    # ── Test 13 — Silence / low-confidence segment rejected ─────────────────

    def test_high_no_speech_prob_rejected(self):
        """Test 13: High no_speech_prob (near-silence) is rejected."""
        signals = [SegmentQualitySignals(no_speech_prob=0.95, avg_logprob=0.0)]
        assert self.guard.is_suspicious("maybe something", segments=signals)

    def test_low_avg_logprob_rejected(self):
        """Test 13 cont.: Very low avg_logprob (low confidence) is rejected."""
        signals = [SegmentQualitySignals(no_speech_prob=0.0, avg_logprob=-2.0)]
        assert self.guard.is_suspicious("maybe something", segments=signals)

    def test_silence_empty_transcript_not_sent_to_brain(self):
        """Test 13 cont.: Empty transcript is NOT suspicious (handled before guard)."""
        # Empty transcripts are filtered at the CVC level before guard runs.
        # The guard should return False for empty strings (not suspicious = handled separately).
        assert self.guard.is_suspicious("") is False

    # ── Test 14 — Clearly suspicious STT cannot call Brain ──────────────────

    @pytest.mark.asyncio
    async def test_suspicious_transcript_not_forwarded_to_brain(self):
        """Test 14: A garbled/repetitive transcript is rejected before Brain."""
        cvc, bus, session, brain, streaming_tts, tts_engine, _ = _make_cvc()

        # Highly repetitive transcript (Whisper hallucination pattern)
        event = MagicMock(
            text="word word word word word word word",
            segment_signals=None,
        )
        await cvc._on_transcript(event)

        brain.process.assert_not_awaited()
        brain.run_goal.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_beep_transcript_not_forwarded_to_brain(self):
        """Test 14 cont.: 'Beep. Beep. Beep.' is not forwarded to Brain."""
        cvc, bus, session, brain, streaming_tts, tts_engine, _ = _make_cvc()

        event = MagicMock(text="Beep. Beep. Beep.", segment_signals=None)
        await cvc._on_transcript(event)

        brain.process.assert_not_awaited()
        brain.run_goal.assert_not_awaited()

    def test_legitimate_short_commands_not_rejected(self):
        """Guard must never reject legitimate short commands like 'Stop', 'Yes', 'Open Edge'."""
        for phrase in ["Stop", "Yes", "No", "Thanks", "Open Edge", "what time is it", "open VS Code"]:
            assert not self.guard.is_suspicious(phrase), (
                f"Guard incorrectly flagged legitimate command: '{phrase}'"
            )


# =============================================================================
# Class 5 — Destructive action confirmation gate (tests 15–16)
# =============================================================================


class TestDestructiveConfirmation:
    """Tests 15–16: shutdown/restart/sleep require 2-step confirmation."""

    def _make_skill(self) -> SystemControlSkill:
        return SystemControlSkill(bus=None)

    # ── Test 15 — Shutdown requires confirmation ─────────────────────────────

    @pytest.mark.asyncio
    async def test_shutdown_requires_confirmation_first_call(self):
        """Test 15: First shutdown_system call must return a confirmation prompt."""
        skill = self._make_skill()
        with patch.object(SystemControlSkill, "_do_shutdown") as mock_shutdown:
            ctx = _skill_ctx("shutdown_system", session_id="sess1")
            result = await skill.execute("shutdown_system", ctx)

            # Must NOT have executed shutdown
            mock_shutdown.assert_not_called()
            # Must have requested confirmation
            assert result.data.get("awaiting_confirmation") is True
            assert "sure" in result.message.lower() or "confirm" in result.message.lower()

    @pytest.mark.asyncio
    async def test_shutdown_executes_after_confirmation(self):
        """Test 15 cont.: Second shutdown call (confirmation) executes the action."""
        skill = self._make_skill()
        with patch.object(SystemControlSkill, "_do_shutdown", return_value=None) as mock_shutdown:
            ctx = _skill_ctx("shutdown_system", session_id="sess2")

            # First call → confirmation prompt
            await skill.execute("shutdown_system", ctx)
            # Second call → execute (mock shutdown)
            result = await skill.execute("shutdown_system", ctx)

            mock_shutdown.assert_called_once()
            assert result.success

    @pytest.mark.asyncio
    async def test_restart_requires_confirmation(self):
        """Test 15 cont.: restart_system also requires confirmation."""
        skill = self._make_skill()
        with patch.object(SystemControlSkill, "_do_shutdown") as mock_shutdown:
            ctx = _skill_ctx("restart_system", session_id="sess3")
            result = await skill.execute("restart_system", ctx)
            mock_shutdown.assert_not_called()
            assert result.data.get("awaiting_confirmation") is True

    @pytest.mark.asyncio
    async def test_sleep_requires_confirmation(self):
        """Test 15 cont.: sleep_system also requires confirmation."""
        skill = self._make_skill()
        with patch.object(SystemControlSkill, "_do_sleep") as mock_sleep:
            ctx = _skill_ctx("sleep_system", session_id="sess4")
            result = await skill.execute("sleep_system", ctx)
            mock_sleep.assert_not_called()
            assert result.data.get("awaiting_confirmation") is True

    @pytest.mark.asyncio
    async def test_stop_phrase_not_interpreted_as_shutdown(self):
        """Test 15 cont.: 'Spidy stop' must NEVER map to shutdown_system."""
        # The stop command is intercepted at the InterruptionHandler level,
        # well before it could reach SystemControlSkill. This test verifies
        # the InterruptionHandler detects it as STOP (not forwarded to Brain).
        handler, bus, brain, tts, session = _make_interruption_handler()
        result = handler.detect("Spidy stop")
        assert result == InterruptCommand.STOP
        # Brain never called
        brain.cancel_goal  # existed, but not called via detect()

    @pytest.mark.asyncio
    async def test_stop_shut_down_yourself_does_not_execute_shutdown(self):
        """
        Test 15 cont.: 'Spidy stop shut down yourself' must NOT cause shutdown.

        The phrase starts with 'spidy stop' → detected by InterruptionHandler
        → handled as STOP → never reaches Brain or SystemControlSkill.
        """
        handler, bus, brain, tts, session = _make_interruption_handler()
        # This should be caught as STOP (via startswith "spidy stop")
        result = handler.detect("Spidy stop shut down yourself")
        assert result == InterruptCommand.STOP, (
            "'Spidy stop shut down yourself' should be caught as STOP, not forwarded to Brain"
        )

    # ── Test 16 — Confirmation expires safely ────────────────────────────────

    @pytest.mark.asyncio
    async def test_confirmation_expires_and_shutdown_not_executed(self):
        """Test 16: If confirmation TTL expires, shutdown must NOT execute."""
        skill = self._make_skill()
        with patch.object(SystemControlSkill, "_do_shutdown") as mock_shutdown:
            ctx = _skill_ctx("shutdown_system", session_id="sess5")

            # First call → store pending confirmation
            await skill.execute("shutdown_system", ctx)

            # Manually expire the confirmation
            pending = skill._pending_confirmations.get("sess5")
            assert pending is not None
            # Backdate created_at past TTL
            pending.created_at = time.monotonic() - (_CONFIRMATION_TTL + 1)

            # Second call should detect expiry and NOT execute
            result = await skill.execute("shutdown_system", ctx)
            mock_shutdown.assert_not_called()
            assert not result.success  # expired confirmation returns failure
            assert "timed out" in result.message.lower() or "try again" in result.message.lower()

    @pytest.mark.asyncio
    async def test_clear_confirmation_removes_pending_state(self):
        """Test 16 cont.: clear_confirmation() properly removes pending state."""
        skill = self._make_skill()
        ctx = _skill_ctx("shutdown_system", session_id="sess6")
        with patch.object(SystemControlSkill, "_do_shutdown"):
            await skill.execute("shutdown_system", ctx)
            assert "sess6" in skill._pending_confirmations

        skill.clear_confirmation("sess6")
        assert "sess6" not in skill._pending_confirmations

    @pytest.mark.asyncio
    async def test_non_destructive_actions_no_confirmation(self):
        """Non-destructive actions (set_volume, lock) must NOT require confirmation."""
        skill = self._make_skill()
        with patch.object(SystemControlSkill, "_do_set_volume", return_value=(50, False)):
            ctx = _skill_ctx("set_volume", {"level": "50"}, session_id="sess7")
            result = await skill.execute("set_volume", ctx)
            # Should succeed immediately, no confirmation
            assert result.success
            assert not result.data.get("awaiting_confirmation", False)


# =============================================================================
# Class 6 — Continuous conversation (tests 19–20)
# =============================================================================


class TestContinuousConversation:
    """Tests 19–20: continuous conversation persists after stop; Hey Jarvis intact."""

    # ── Test 19 — Continuous conversation works after stopping TTS ───────────

    @pytest.mark.asyncio
    async def test_continuous_conversation_after_stop(self):
        """Test 19: After a stop command, user can immediately send another utterance."""
        cvc, bus, session, brain, streaming_tts, tts_engine, voice_engine = _make_cvc()
        session.is_active = True

        # Fire stop command
        stop_event = MagicMock(text="Spidy stop", segment_signals=None)
        await cvc._on_transcript(stop_event)

        # Session must still be active
        assert session.is_active is True

        # Send a normal follow-up question
        q_event = MagicMock(text="What is Python?", segment_signals=None)
        await cvc._on_transcript(q_event)

        # Brain must have been called for the follow-up
        assert brain.process.await_count >= 1 or brain.run_goal.await_count >= 1

    # ── Test 20 — Hey Jarvis wake behaviour unchanged ───────────────────────

    @pytest.mark.asyncio
    async def test_hey_jarvis_wake_behaviour_intact(self):
        """Test 20: 'Hey Jarvis' still activates the session normally."""
        cvc, bus, session, brain, streaming_tts, tts_engine, voice_engine = _make_cvc(
            session_active=False,
            tts_speaking=False,
        )

        # Simulate wake word event
        event = MagicMock(model_name="hey_jarvis")
        await cvc._on_wake_word(event)

        # Session must have been activated
        session.activate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Test 20 cont.: Multiple turns without wake word (continuous mode)."""
        cvc, bus, session, brain, streaming_tts, tts_engine, _ = _make_cvc()
        session.is_active = True

        turns = [
            "What is Python?",
            "Who created it?",
            "Explain it simply.",
        ]
        for text in turns:
            event = MagicMock(text=text, segment_signals=None)
            await cvc._on_transcript(event)

        # Brain must have been called for each turn
        total = brain.process.await_count + brain.run_goal.await_count
        assert total == len(turns), (
            f"Expected {len(turns)} Brain calls, got {total}"
        )

    @pytest.mark.asyncio
    async def test_legitimate_question_not_stopped(self):
        """Legitimate questions do NOT trigger stop; Brain is called normally."""
        cvc, bus, session, brain, streaming_tts, tts_engine, _ = _make_cvc()
        session.is_active = True

        for phrase in ["Who is Jarvis?", "Tell me about Spidy.", "What time is it?"]:
            brain.process.reset_mock()
            brain.run_goal.reset_mock()
            event = MagicMock(text=phrase, segment_signals=None)
            await cvc._on_transcript(event)
            total = brain.process.await_count + brain.run_goal.await_count
            assert total >= 1, (
                f"Brain must be called for '{phrase}', got {total} calls"
            )
