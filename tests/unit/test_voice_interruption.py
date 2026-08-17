"""
Unit tests — InterruptionHandler
==================================
Tests keyword detection, command dispatch, TTS stop, goal cancel,
pause/resume flows, and non-interrupt pass-through.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spidy.voice.interruption import InterruptionHandler, InterruptCommand


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_bus():
    bus = MagicMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def mock_brain():
    brain = MagicMock()
    brain.cancel_goal = AsyncMock(return_value=True)
    return brain


@pytest.fixture
def mock_tts():
    tts = MagicMock()
    tts.stop = MagicMock()
    return tts


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.deactivate = AsyncMock()
    session.pause = AsyncMock()
    session.resume = AsyncMock()
    return session


@pytest.fixture
def handler(mock_bus, mock_brain, mock_tts, mock_session):
    return InterruptionHandler(
        bus=mock_bus,
        brain=mock_brain,
        tts=mock_tts,
        session=mock_session,
    )


# ── Keyword detection tests ────────────────────────────────────────────────────


@pytest.mark.parametrize("phrase,expected_cmd", [
    ("stop", InterruptCommand.STOP),
    ("cancel", InterruptCommand.CANCEL),
    ("cancel that", InterruptCommand.CANCEL),
    ("never mind", InterruptCommand.NEVER_MIND),
    ("nevermind", InterruptCommand.NEVER_MIND),
    ("stop that", InterruptCommand.STOP),
    ("wait", InterruptCommand.WAIT),
    ("wait a sec", InterruptCommand.WAIT),
    ("hold on", InterruptCommand.WAIT),
    ("pause", InterruptCommand.PAUSE),
    ("resume", InterruptCommand.RESUME),
    ("continue", InterruptCommand.CONTINUE),
    ("go ahead", InterruptCommand.RESUME),
    ("carry on", InterruptCommand.RESUME),
])
def test_detect_known_keywords(handler, phrase, expected_cmd):
    result = handler.detect(phrase)
    assert result == expected_cmd, f"Expected {expected_cmd} for '{phrase}', got {result}"


@pytest.mark.parametrize("phrase", [
    "open chrome",
    "search youtube",
    "create a folder",
    "what time is it",
    "hello spidy",
    "stopping soon",   # contains "stop" but not as a phrase
])
def test_detect_non_interrupt_phrases(handler, phrase):
    result = handler.detect(phrase)
    # Only exact / boundary matches should trigger
    if phrase == "stopping soon":
        assert result is None  # "stopping" ≠ "stop"
    else:
        assert result is None


# ── Execute: cancellation ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_stop_calls_tts_stop(handler, mock_tts):
    await handler.execute(InterruptCommand.STOP, "stop")
    mock_tts.stop.assert_called_once()


@pytest.mark.asyncio
async def test_execute_stop_cancels_goal(handler, mock_brain):
    await handler.execute(InterruptCommand.STOP, "stop")
    mock_brain.cancel_goal.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_stop_does_not_deactivate_session(handler, mock_session):
    """
    After a stop command the session must stay ACTIVE so the user can
    continue speaking without repeating the wake word.

    Milestone 15 safety fix: _handle_cancellation no longer calls
    session.deactivate(). The overlay/relisten state is restored by
    ContinuousVoiceController, not the handler.
    """
    await handler.execute(InterruptCommand.STOP, "stop")
    mock_session.deactivate.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_stop_publishes_event(handler, mock_bus):
    await handler.execute(InterruptCommand.STOP, "stop")
    from spidy.voice.events import VoiceInterruptEvent
    event = mock_bus.publish.call_args[0][0]
    assert isinstance(event, VoiceInterruptEvent)
    assert event.command == "stop"


@pytest.mark.asyncio
async def test_execute_cancel_response(handler):
    """
    Cancellation now returns an empty string.

    Milestone 15 safety fix: _handle_cancellation returns '' so that
    ContinuousVoiceController (not the handler) decides when and how
    to restore overlay state and re-listen. Emitting a TTS ack on stop
    would interfere with the immediate microphone restore.
    """
    response = await handler.execute(InterruptCommand.CANCEL, "cancel")
    assert isinstance(response, str)  # must be a string (empty is correct)


@pytest.mark.asyncio
async def test_execute_stop_no_active_goal(handler, mock_brain):
    """
    When no goal is running, the handler still stops TTS cleanly.

    Milestone 15 safety fix: returns '' regardless of whether a goal
    was cancelled (CVC handles the response feedback, not this handler).
    """
    mock_brain.cancel_goal = AsyncMock(return_value=False)
    response = await handler.execute(InterruptCommand.STOP, "stop")
    assert isinstance(response, str)  # empty string is the correct response


# ── Execute: pause / resume ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_pause_calls_session_pause(handler, mock_session, mock_tts):
    await handler.execute(InterruptCommand.PAUSE, "pause")
    mock_session.pause.assert_awaited_once()
    mock_tts.stop.assert_called_once()  # TTS stopped during pause


@pytest.mark.asyncio
async def test_execute_resume_calls_session_resume(handler, mock_session):
    response = await handler.execute(InterruptCommand.RESUME, "resume")
    mock_session.resume.assert_awaited_once()
    assert "listen" in response.lower()


# ── InterruptCommand properties ────────────────────────────────────────────────


def test_is_cancellation():
    assert InterruptCommand.STOP.is_cancellation
    assert InterruptCommand.CANCEL.is_cancellation
    assert InterruptCommand.NEVER_MIND.is_cancellation
    assert not InterruptCommand.PAUSE.is_cancellation
    assert not InterruptCommand.RESUME.is_cancellation


def test_is_pause():
    assert InterruptCommand.WAIT.is_pause
    assert InterruptCommand.PAUSE.is_pause
    assert not InterruptCommand.STOP.is_pause


def test_is_resume():
    assert InterruptCommand.RESUME.is_resume
    assert InterruptCommand.CONTINUE.is_resume
    assert not InterruptCommand.STOP.is_resume
