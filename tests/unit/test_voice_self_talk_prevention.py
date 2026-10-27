"""
test_voice_self_talk_prevention.py — Dedicated Voice Self-Talk & Audio Ownership Suite
=====================================================================================
Tests the complete end-to-end audio ownership, TTS playback gating, barge-in isolation,
post-TTS audio flush, and transcript provenance filtering architecture.

Guarantees:
  1. SPIDY never treats its own TTS output as user input.
  2. Infinite feedback loops (TTS -> Mic -> STT -> Brain -> TTS) are impossible.
  3. Wake acknowledgement speech is isolated and never enters STT / Brain.
  4. Transcripts during TTS or AUDIO_FLUSH are strictly rejected.
  5. Legitimate barge-in ("Spidy stop") stops TTS cleanly.
  6. Multi-turn user conversations progress seamlessly.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, call, patch
import numpy as np
import pytest

from spidy.perception.voice.audio_capture import AudioOwnership, CaptureMode
from spidy.perception.voice.engine import VoiceState, VoiceTranscriptEvent
from spidy.voice.session import VoiceSessionManager, VoiceSessionState
from spidy.voice.continuous import ContinuousVoiceController


def _make_controller(*, session_active: bool = False, continuous_mode: bool = True):
    """Construct a ContinuousVoiceController with mocked underlying components."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    bus.subscribe = MagicMock()
    bus.unsubscribe = MagicMock()

    session_mgr = VoiceSessionManager(bus=bus, timeout_seconds=60.0, continuous_mode=continuous_mode)
    if session_active:
        session_mgr._state = VoiceSessionState.AWAKE
        session_mgr._session_id = "test-sess-1"

    capture_engine = MagicMock()
    capture_engine.set_mode = MagicMock()
    capture_engine.mode = CaptureMode.DETECTING

    wake_model = MagicMock()
    wake_model.reset_buffer = MagicMock()
    wake_model.model_name = "hey_jarvis"

    voice_engine = MagicMock()
    voice_engine._capture_engine = capture_engine
    voice_engine._wake_model = wake_model
    voice_engine._capture_buffer = []
    voice_engine._buffer_lock = MagicMock()
    voice_engine._buffer_lock.__enter__ = MagicMock(return_value=None)
    voice_engine._buffer_lock.__exit__ = MagicMock(return_value=None)
    voice_engine.signal_relisten = MagicMock()
    voice_engine._state = VoiceState.SLEEPING
    voice_engine.is_speaking = False
    voice_engine.is_flushing = False
    voice_engine.set_speaking = MagicMock()
    voice_engine.enter_audio_flush = AsyncMock()
    voice_engine.exit_audio_flush = MagicMock()

    streaming_tts = MagicMock()
    streaming_tts.speak = AsyncMock()
    streaming_tts.stop = MagicMock()
    streaming_tts.is_speaking = False

    brain = MagicMock()
    brain.process = AsyncMock(return_value="I am ready to help.")
    brain.run_goal = AsyncMock(return_value="Goal completed.")

    interruption = MagicMock()
    interruption.detect = MagicMock(return_value=None)
    interruption.execute = AsyncMock(return_value=None)

    wake_ack = MagicMock()
    wake_ack.enabled = True
    wake_ack.speak_ack = AsyncMock()

    controller = ContinuousVoiceController(
        voice_engine=voice_engine,
        brain=brain,
        bus=bus,
        session_manager=session_mgr,
        streaming_tts=streaming_tts,
        interruption_handler=interruption,
        continuous_mode=continuous_mode,
        measure_latency=False,
        wake_acknowledger=wake_ack,
    )

    return controller, {
        "bus": bus,
        "session": session_mgr,
        "capture": capture_engine,
        "wake_model": wake_model,
        "engine": voice_engine,
        "tts": streaming_tts,
        "brain": brain,
        "interruption": interruption,
        "wake_ack": wake_ack,
    }


# ===========================================================================
# Test 1: TTS Transcript Rejected During Speaking State
# ===========================================================================
@pytest.mark.asyncio
async def test_tts_transcript_rejected_during_speaking():
    ctrl, mocks = _make_controller(session_active=True)

    # Put session into SPEAKING state
    await mocks["session"].mark_speaking()
    assert mocks["session"].state == VoiceSessionState.SPEAKING

    # Attempt to send an STT transcript while speaking
    evt = VoiceTranscriptEvent(
        text="I am opening Edge for you now",
        confidence=0.95,
        source="tts_echo",
        tts_active=True,
    )
    await ctrl._on_transcript(evt)

    # Verify Brain was NEVER called
    mocks["brain"].process.assert_not_called()
    mocks["brain"].run_goal.assert_not_called()


# ===========================================================================
# Test 2: Transcript Arriving During AUDIO_FLUSH is Rejected
# ===========================================================================
@pytest.mark.asyncio
async def test_audio_flush_transcript_rejected():
    ctrl, mocks = _make_controller(session_active=True)

    # Put session into SPEAKING then AUDIO_FLUSH
    await mocks["session"].mark_speaking()
    await mocks["session"].mark_audio_flush()
    assert mocks["session"].state == VoiceSessionState.AUDIO_FLUSH

    evt = VoiceTranscriptEvent(
        text="opening Edge",
        confidence=0.90,
        source="user",
        flush_active=True,
    )
    await ctrl._on_transcript(evt)

    # Verify Brain was NEVER called
    mocks["brain"].process.assert_not_called()
    mocks["brain"].run_goal.assert_not_called()


# ===========================================================================
# Test 3: Non-user Source Transcripts Rejected
# ===========================================================================
@pytest.mark.asyncio
async def test_non_user_source_transcript_rejected():
    ctrl, mocks = _make_controller(session_active=True)
    assert mocks["session"].state == VoiceSessionState.AWAKE

    evt = VoiceTranscriptEvent(
        text="some stray audio",
        confidence=0.88,
        source="spidy_tts",
        tts_active=False,
        flush_active=False,
    )
    await ctrl._on_transcript(evt)

    mocks["brain"].process.assert_not_called()
    mocks["brain"].run_goal.assert_not_called()


# ===========================================================================
# Test 4: Self-Talk Infinite Loop Simulation (0 Brain calls, 0 loops)
# ===========================================================================
@pytest.mark.asyncio
async def test_self_talk_infinite_loop_prevention():
    ctrl, mocks = _make_controller(session_active=True)

    # User says "open Edge"
    user_evt = VoiceTranscriptEvent(text="open Edge", confidence=0.98, source="user")
    await ctrl._on_transcript(user_evt)

    # Brain is called once for user input
    assert mocks["brain"].process.call_count == 1 or mocks["brain"].run_goal.call_count == 1

    # Brain outputs response
    brain_resp_evt = MagicMock()
    brain_resp_evt.response_text = "I have opened Microsoft Edge for you."
    
    # Brain response triggers TTS and state transitions
    await ctrl._on_brain_response(brain_resp_evt)
    mocks["tts"].speak.assert_awaited_with("I have opened Microsoft Edge for you.")

    # Now simulate the physical loop: microphone picks up TTS and Whisper generates transcript
    # with Spidy's own words while TTS or flush is running
    echo_evt = VoiceTranscriptEvent(
        text="I have opened Microsoft Edge for you",
        confidence=0.99,
        source="tts_echo",
        tts_active=True,
    )
    await ctrl._on_transcript(echo_evt)

    # Brain call count MUST STILL BE 1 (not 2)
    total_brain_calls = mocks["brain"].process.call_count + mocks["brain"].run_goal.call_count
    assert total_brain_calls == 1, f"Brain was called {total_brain_calls} times! Self-talk loop occurred!"


# ===========================================================================
# Test 5: Wake Ack Isolation (Mic IDLE during ack, buffer cleared)
# ===========================================================================
@pytest.mark.asyncio
async def test_wake_ack_isolation():
    ctrl, mocks = _make_controller(session_active=False)

    wake_evt = MagicMock()
    wake_evt.wake_phrase = "hey jarvis"
    wake_evt.model_name = "hey_jarvis"

    await ctrl._on_wake_word(wake_evt)

    # Ack was spoken
    mocks["wake_ack"].speak_ack.assert_awaited_once()

    # Session is now AWAKE
    assert mocks["session"].state == VoiceSessionState.AWAKE

    # Capture engine mode was set to IDLE then RECORDING
    mode_calls = [c[0][0] for c in mocks["capture"].set_mode.call_args_list]
    assert CaptureMode.IDLE in mode_calls
    assert mode_calls[-1] == CaptureMode.RECORDING

    # Audio flush was entered
    mocks["engine"].enter_audio_flush.assert_awaited_once()


# ===========================================================================
# Test 6: Active Session Wake Word Suppressed
# ===========================================================================
@pytest.mark.asyncio
async def test_active_session_wake_word_suppressed():
    ctrl, mocks = _make_controller(session_active=True)

    # Put session in SPEAKING state
    await mocks["session"].mark_speaking()

    wake_evt = MagicMock()
    wake_evt.wake_phrase = "hey jarvis"
    await ctrl._on_wake_word(wake_evt)

    # Wake ack was NOT triggered
    mocks["wake_ack"].speak_ack.assert_not_called()


# ===========================================================================
# Test 7: Legitimate Barge-In Stop Works Cleanly
# ===========================================================================
@pytest.mark.asyncio
async def test_legitimate_barge_in_stop():
    ctrl, mocks = _make_controller(session_active=True)

    # Interruption mock detects "stop"
    interrupt_cmd = MagicMock()
    interrupt_cmd.is_cancellation = True
    mocks["interruption"].detect = MagicMock(return_value=interrupt_cmd)

    stop_evt = VoiceTranscriptEvent(text="stop", confidence=0.99, source="user")
    await ctrl._on_transcript(stop_evt)

    mocks["interruption"].execute.assert_awaited_once()
    # Brain is not invoked for cancellation command
    mocks["brain"].process.assert_not_called()
    mocks["brain"].run_goal.assert_not_called()


# ===========================================================================
# Test 8: Multi-Turn Clean Conversation
# ===========================================================================
@pytest.mark.asyncio
async def test_multi_turn_clean_conversation():
    ctrl, mocks = _make_controller(session_active=False)

    # Turn 1: Wake word
    wake_evt = MagicMock()
    wake_evt.wake_phrase = "hey jarvis"
    await ctrl._on_wake_word(wake_evt)
    assert mocks["session"].state == VoiceSessionState.AWAKE

    # Turn 1: User says "what time is it"
    t1_evt = VoiceTranscriptEvent(text="what time is it", confidence=0.95, source="user")
    await ctrl._on_transcript(t1_evt)
    assert mocks["session"].state == VoiceSessionState.PROCESSING

    # Turn 1: Brain responds
    b1_resp = MagicMock()
    b1_resp.response_text = "It is 3 PM."
    await ctrl._on_brain_response(b1_resp)
    assert mocks["session"].state == VoiceSessionState.AWAKE

    # Turn 2: Follow-up user command without wake word
    t2_evt = VoiceTranscriptEvent(text="and what is the date", confidence=0.95, source="user")
    await ctrl._on_transcript(t2_evt)
    assert mocks["session"].state == VoiceSessionState.PROCESSING

    # Turn 2: Brain responds
    b2_resp = MagicMock()
    b2_resp.response_text = "Today is Sunday."
    await ctrl._on_brain_response(b2_resp)
    assert mocks["session"].state == VoiceSessionState.AWAKE

    # Verify 2 user turns were executed cleanly
    total_turns = mocks["session"].turn_count
    assert total_turns == 2
