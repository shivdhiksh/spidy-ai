"""
Unit tests — VoiceSessionManager
=================================
Tests the session state machine, timeout logic, turn counting,
pause/resume, and deactivation.

All tests use a real asyncio event loop.  No real audio hardware.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from spidy.voice.session import VoiceSessionManager, VoiceSessionState


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_bus():
    bus = MagicMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def manager(mock_bus):
    return VoiceSessionManager(
        bus=mock_bus,
        timeout_seconds=60.0,
        max_turns=50,
        continuous_mode=True,
    )


# ── State machine tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_initial_state(manager):
    """Session starts in IDLE state."""
    assert manager.state == VoiceSessionState.IDLE
    assert not manager.is_active
    assert manager.turn_count == 0


@pytest.mark.asyncio
async def test_activate_transitions_to_awake(manager):
    sid = await manager.activate(wake_word="hey spidy")
    assert manager.state == VoiceSessionState.AWAKE
    assert manager.is_active
    assert len(sid) > 0


@pytest.mark.asyncio
async def test_activate_publishes_session_started(manager, mock_bus):
    await manager.activate(wake_word="hey spidy")
    mock_bus.publish.assert_called_once()
    event = mock_bus.publish.call_args[0][0]
    from spidy.voice.events import VoiceSessionStartedEvent
    assert isinstance(event, VoiceSessionStartedEvent)
    assert event.wake_word == "hey spidy"


@pytest.mark.asyncio
async def test_record_utterance_increments_turn_count(manager):
    await manager.activate()
    await manager.record_utterance("Open Chrome")
    assert manager.turn_count == 1
    assert manager.state == VoiceSessionState.PROCESSING
    assert manager.current_goal == "Open Chrome"


@pytest.mark.asyncio
async def test_multiple_utterances(manager):
    await manager.activate()
    await manager.record_utterance("Open Chrome")
    await manager.mark_response_complete()
    await manager.record_utterance("Search YouTube")
    await manager.mark_response_complete()
    assert manager.turn_count == 2


@pytest.mark.asyncio
async def test_mark_speaking(manager):
    await manager.activate()
    await manager.record_utterance("hello")
    await manager.mark_speaking()
    assert manager.state == VoiceSessionState.SPEAKING


@pytest.mark.asyncio
async def test_continuous_mode_stays_awake(manager):
    """In continuous mode, session stays AWAKE after response."""
    await manager.activate()
    await manager.record_utterance("hello")
    await manager.mark_speaking()
    await manager.mark_response_complete()
    assert manager.state == VoiceSessionState.AWAKE
    assert manager.is_active


@pytest.mark.asyncio
async def test_non_continuous_mode_deactivates(mock_bus):
    """Non-continuous mode: session ends after one response."""
    mgr = VoiceSessionManager(
        bus=mock_bus,
        timeout_seconds=60.0,
        continuous_mode=False,
    )
    await mgr.activate()
    await mgr.record_utterance("hello")
    await mgr.mark_speaking()
    await mgr.mark_response_complete()
    assert mgr.state == VoiceSessionState.IDLE


@pytest.mark.asyncio
async def test_pause_and_resume(manager):
    await manager.activate()
    await manager.pause()
    assert manager.state == VoiceSessionState.PAUSED
    assert manager.is_paused

    await manager.resume()
    assert manager.state == VoiceSessionState.AWAKE
    assert not manager.is_paused


@pytest.mark.asyncio
async def test_deactivate_from_active(manager, mock_bus):
    await manager.activate()
    mock_bus.publish.reset_mock()
    await manager.deactivate(reason="user_stopped")

    assert manager.state == VoiceSessionState.IDLE
    assert not manager.is_active

    from spidy.voice.events import VoiceSessionEndedEvent
    event = mock_bus.publish.call_args[0][0]
    assert isinstance(event, VoiceSessionEndedEvent)
    assert event.reason == "user_stopped"


@pytest.mark.asyncio
async def test_deactivate_idle_is_noop(manager, mock_bus):
    """Deactivating an already-idle session does nothing."""
    await manager.deactivate(reason="nothing")
    mock_bus.publish.assert_not_called()


@pytest.mark.asyncio
async def test_max_turns_deactivates(mock_bus):
    """Session automatically deactivates when max_turns is reached."""
    mgr = VoiceSessionManager(bus=mock_bus, max_turns=2)
    await mgr.activate()
    await mgr.record_utterance("turn 1")
    await mgr.record_utterance("turn 2")

    from spidy.voice.events import VoiceSessionEndedEvent
    published_events = [c[0][0] for c in mock_bus.publish.call_args_list]
    assert any(isinstance(e, VoiceSessionEndedEvent) and e.reason == "max_turns"
               for e in published_events)


@pytest.mark.asyncio
async def test_activate_while_active_resets_timeout(manager):
    """Calling activate() while already active reuses the session."""
    sid1 = await manager.activate()
    sid2 = await manager.activate()
    assert sid1 == sid2  # Same session


@pytest.mark.asyncio
async def test_timeout_setter(manager):
    manager.timeout_seconds = 30.0
    assert manager.timeout_seconds == 30.0
    # Minimum enforced
    manager.timeout_seconds = 1.0
    assert manager.timeout_seconds == 5.0
