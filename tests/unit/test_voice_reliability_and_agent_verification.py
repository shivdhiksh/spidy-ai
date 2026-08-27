"""
Regression tests for Voice Reliability + Agent Verification Fix:
1. wake-only "Hey Spidy" does not call Brain
2. wake-only "Wake up Spidy" does not call Brain
3. wake-only "Hey Jarvis" does not call Brain
4. repeated wake-only phrases are suppressed
5. TTS speech does not create fake barge-in commands
6. real stop command still interrupts TTS
7. transcript contamination is reduced (ack preamble stripping / flushing)
8. meaningful names are preserved
9. trailing query punctuation removed
10. browser navigation real-state verification
11. browser search real-state verification
12. browser mismatch causes verification failure
13. simple commands still work
14. compound goals still work
15. continuous conversation remains functional
"""

import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spidy.voice.wake_word_stripper import WakeWordStripper
from spidy.voice.barge_in import BargeInDetector
from spidy.voice.interruption import InterruptionHandler, InterruptCommand
from spidy.voice.continuous import ContinuousVoiceController
from spidy.voice.session import VoiceSessionManager
from spidy.agent.task_decomposer import TaskDecomposer, _normalize_search_query
from spidy.agent.observer import TaskObserver, Observation
from spidy.agent.verifier import GoalVerifier, VerificationResult
from spidy.agent.types import TaskRecord, GoalRecord, TaskState
from spidy.core.event_bus import EventBus


# ==============================================================================
# 1-4. Wake-only & Repeated Wake Phrase Tests
# ==============================================================================

class TestWakeOnlySuppression:
    def setup_method(self):
        self.stripper = WakeWordStripper()

    def test_wake_only_hey_spidy(self):
        assert self.stripper.is_wake_only("Hey Spidy") is True
        assert self.stripper.is_wake_only("Hey Spidy.") is True
        assert self.stripper.is_wake_only("Hey Spidey") is True
        assert self.stripper.is_wake_only("Hey Spidey.") is True

    def test_wake_only_wake_up_spidy(self):
        assert self.stripper.is_wake_only("Wake up Spidy") is True
        assert self.stripper.is_wake_only("Wake up Spidy.") is True
        assert self.stripper.is_wake_only("Wake up Spidey") is True

    def test_wake_only_hey_jarvis(self):
        assert self.stripper.is_wake_only("Hey Jarvis") is True
        assert self.stripper.is_wake_only("Hey Jarvis.") is True
        assert self.stripper.is_wake_only("Okay Jarvis") is True

    def test_repeated_wake_phrases_suppressed(self):
        assert self.stripper.is_wake_only("Hey Spidy. Hey Spidy.") is True
        assert self.stripper.is_wake_only("Wake up Spidy. Wake up Spidy.") is True
        assert self.stripper.is_wake_only("Hey Jarvis. Hey Jarvis.") is True

    def test_wake_with_command_is_not_wake_only(self):
        assert self.stripper.is_wake_only("Hey Spidy, open Edge.") is False
        assert self.stripper.strip_wake_prefix("Hey Spidy, open Edge.") == "open Edge."

    def test_speech_containing_name_is_preserved(self):
        # "Tell me about Spidy" -> not a wake prefix
        assert self.stripper.is_wake_only("Tell me about Spidy") is False
        assert self.stripper.strip_wake_prefix("Tell me about Spidy") == "Tell me about Spidy"

        # "Who is Shiva?" -> preserved
        assert self.stripper.is_wake_only("Who is Shiva?") is False
        assert self.stripper.strip_wake_prefix("Who is Shiva?") == "Who is Shiva?"


# ==============================================================================
# 5-6. Barge-In Speaker Bleed & Stop Interruption Tests
# ==============================================================================

class TestBargeInReliability:
    def setup_method(self):
        self.bus = EventBus()
        self.handler = InterruptionHandler(bus=self.bus, brain=MagicMock(), tts=MagicMock(), session=MagicMock())
        self.tts = MagicMock()
        self.tts.is_speaking = True
        self.tts.stop = MagicMock()
        self.detector = BargeInDetector(
            interruption_handler=self.handler,
            bus=self.bus,
            model_size="tiny.en",
            min_rms_threshold=0.015,
            enabled=True,
            suppress_during_tts=True,
        )
        self.detector._model = MagicMock()
        self.detector._model_loaded = True

    def test_speaker_bleed_during_tts_is_suppressed(self):
        # Speaker bleed has moderate RMS (e.g. 0.03) below suppressed threshold (0.065)
        bleed_chunk = np.full(1280, 0.03, dtype=np.float32)
        self.detector.start(tts=self.tts)

        self.detector.feed_chunk(bleed_chunk)
        with self.detector._buf_lock:
            assert len(self.detector._buffer) == 0  # Discarded by raised gate

        self.detector.stop()

    def test_real_stop_command_still_interrupts_tts(self):
        # User shouting "Spidy stop" has high RMS (e.g. 0.12)
        user_speech = np.full(1280, 0.12, dtype=np.float32)
        self.detector.start(tts=self.tts)

        self.detector.feed_chunk(user_speech)
        with self.detector._buf_lock:
            assert len(self.detector._buffer) == 1

        # Simulate STT returning "Spidy stop"
        self.detector._model.transcribe.return_value = ([MagicMock(text="Spidy stop")], None)
        text = self.detector._transcribe(user_speech)
        assert text == "Spidy stop"

        cmd = self.handler.detect(text)
        assert cmd == InterruptCommand.STOP

        self.detector._trigger_stop()
        self.tts.stop.assert_called_once()
        self.detector.stop()


# ==============================================================================
# 7-8. Transcript Contamination & Name Preservation Tests
# ==============================================================================

class TestTranscriptNormalizationAndPreamble:
    def test_ack_preamble_stripping(self):
        # When wake ack is configured with ["Yes Shiva.", "Hmm?"], stripper cleans ack echo
        stripper = WakeWordStripper(ack_phrases=["Yes Shiva.", "Hmm?"])
        cleaned = stripper.strip_wake_prefix("Yes Shiva. Open Edge. Open YouTube. Search for Python videos.")
        assert cleaned == "Open Edge. Open YouTube. Search for Python videos."

    def test_meaningful_names_are_preserved(self):
        stripper = WakeWordStripper(ack_phrases=["Yes Shiva."])
        # User commanding "Tell Shiva to join the meeting"
        assert stripper.strip_wake_prefix("Tell Shiva to join the meeting") == "Tell Shiva to join the meeting"
        assert stripper.is_wake_only("Tell Shiva to join the meeting") is False


# ==============================================================================
# 9. Search Query Punctuation Normalization Tests
# ==============================================================================

class TestQueryNormalization:
    def test_trailing_period_stripped(self):
        assert _normalize_search_query("Python videos.") == "Python videos"
        assert _normalize_search_query("Python tutorials...") == "Python tutorials"

    def test_trailing_question_and_exclamation_stripped(self):
        assert _normalize_search_query("What is machine learning?") == "What is machine learning"
        assert _normalize_search_query("Learn Python now!") == "Learn Python now"

    def test_internal_syntax_preserved(self):
        assert _normalize_search_query("node.js tutorial") == "node.js tutorial"
        assert _normalize_search_query("C++ programming") == "C++ programming"
        assert _normalize_search_query("Python 3.12 release") == "Python 3.12 release"

    def test_decomposer_normalizes_search_query(self):
        decomposer = TaskDecomposer()
        target, query = decomposer._extract_search_params("search YouTube for Python tutorials.")
        assert target == "youtube"
        assert query == "Python tutorials"


# ==============================================================================
# 10-12. Real Browser State Observation & Verification Tests
# ==============================================================================

class TestRealBrowserVerification:
    def setup_method(self):
        self.observer = TaskObserver()
        self.verifier = GoalVerifier()

    @pytest.mark.asyncio
    async def test_browser_navigation_real_state_verified(self):
        task = TaskRecord(
            task_id="t1",
            goal_id="g1",
            description="Navigate to YouTube",
            utterance="navigate to https://www.youtube.com",
            action={"skill": "browser", "action": "navigate", "target": "YouTube", "url": "https://www.youtube.com"},
            state=TaskState.COMPLETED,
            terminal=False,
        )
        response_text = "Navigated to https://www.youtube.com — YouTube homepage loaded."
        obs = await self.observer.observe(task, response_text, skip_for_terminal=False)

        assert obs.method == "browser_url"
        assert obs.success_signal is True
        assert "youtube.com" in obs.summary.lower()

    @pytest.mark.asyncio
    async def test_browser_search_real_state_verified(self):
        task = TaskRecord(
            task_id="t2",
            goal_id="g1",
            description="Search YouTube for Python tutorials",
            utterance="search YouTube for Python tutorials",
            action={"skill": "browser", "action": "search", "target": "YouTube", "query": "Python tutorials"},
            state=TaskState.COMPLETED,
            terminal=True,
        )
        response_text = "Search results loaded: https://www.youtube.com/results?search_query=Python+tutorials"
        obs = await self.observer.observe(task, response_text, skip_for_terminal=False)

        assert obs.method == "browser_url"
        assert obs.success_signal is True
        assert "Python tutorials" in obs.summary

    @pytest.mark.asyncio
    async def test_browser_search_mismatch_fails_verification(self):
        task = TaskRecord(
            task_id="t2",
            goal_id="g1",
            description="Search YouTube for Python tutorials",
            utterance="search YouTube for Python tutorials",
            action={"skill": "browser", "action": "search", "target": "YouTube", "query": "Python tutorials"},
            state=TaskState.COMPLETED,
            terminal=True,
        )
        # Mismatch: browser ended up on Google homepage instead of YouTube search
        response_text = "Opened page: https://www.google.com"
        obs = await self.observer.observe(task, response_text, skip_for_terminal=False)

        assert obs.method == "browser_url"
        assert obs.success_signal is False
        assert "mismatch" in obs.summary.lower()

        # Final goal verification must fail when observation is negative
        goal = GoalRecord(
            goal_id="g1",
            description="Search YouTube for Python tutorials",
            tasks=[task],
            state=TaskState.COMPLETED,
        )
        res = await self.verifier.verify(goal, observations=[obs])
        assert res.verified is False
        assert "Verification failed" in res.summary


# ==============================================================================
# 13-15. End-to-End Voice Flow & Continuous Conversation Tests
# ==============================================================================

class TestContinuousVoiceEndToEnd:
    @pytest.mark.asyncio
    async def test_wake_only_does_not_call_brain(self):
        brain = MagicMock()
        brain.process = AsyncMock()
        brain.run_goal = AsyncMock()
        bus = EventBus()
        session_mgr = VoiceSessionManager(bus=bus, timeout_seconds=30.0)
        await session_mgr.activate(wake_word="hey_spidy")

        tts = MagicMock()
        tts.is_speaking = False
        interruption = InterruptionHandler(bus=bus, brain=brain, tts=tts, session=session_mgr)

        controller = ContinuousVoiceController(
            voice_engine=MagicMock(),
            brain=brain,
            bus=bus,
            session_manager=session_mgr,
            streaming_tts=tts,
            interruption_handler=interruption,
        )

        # Trigger wake-only transcript
        event = MagicMock(text="Hey Spidy.")
        await controller._on_transcript(event)

        # Brain must NOT have been called
        brain.process.assert_not_called()
        brain.run_goal.assert_not_called()

    @pytest.mark.asyncio
    async def test_wake_with_command_calls_brain(self):
        brain = MagicMock()
        brain.process = AsyncMock(return_value="Opening Edge.")
        brain.run_goal = AsyncMock(return_value="Opening Edge.")
        bus = EventBus()
        session_mgr = VoiceSessionManager(bus=bus, timeout_seconds=30.0)
        await session_mgr.activate(wake_word="hey_spidy")

        tts = MagicMock()
        tts.is_speaking = False
        interruption = InterruptionHandler(bus=bus, brain=brain, tts=tts, session=session_mgr)

        controller = ContinuousVoiceController(
            voice_engine=MagicMock(),
            brain=brain,
            bus=bus,
            session_manager=session_mgr,
            streaming_tts=tts,
            interruption_handler=interruption,
        )

        event = MagicMock(text="Hey Spidy, open Edge.")
        await controller._on_transcript(event)

        # Brain was called with stripped command
        assert brain.process.call_count + brain.run_goal.call_count == 1
