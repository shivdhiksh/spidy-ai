"""
Unit tests for spidy.ui.brain_bridge.BrainUIBridge
====================================================
Verifies that brain.* events are correctly translated into ui.* events
via the EventBus, without modifying Brain logic.

All tests use a real EventBus (no mocks) and real event dataclasses
to keep the tests meaningful and coupled to the actual event contracts.
"""

from __future__ import annotations

import pytest

from spidy.brain.events import BrainProcessingStartedEvent, BrainResponseReadyEvent
from spidy.core.event_bus import EventBus
from spidy.ui.brain_bridge import BrainUIBridge
from spidy.ui.events import UIMessageEvent, UIStateChangeEvent


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_bus_and_bridge() -> tuple[EventBus, BrainUIBridge]:
    bus = EventBus()
    bridge = BrainUIBridge(bus=bus)
    bridge.start()
    return bus, bridge


# ─── Lifecycle ────────────────────────────────────────────────────────────────


class TestBrainUIBridgeLifecycle:
    """start() / stop() guard behaviour."""

    def test_double_start_is_idempotent(self) -> None:
        bus = EventBus()
        bridge = BrainUIBridge(bus=bus)
        bridge.start()
        bridge.start()  # Should NOT raise or double-subscribe
        # Only 1 subscription per topic (EventBus deduplicates)
        assert len(bus._subscribers["brain.processing_started"]) == 1
        assert len(bus._subscribers["brain.response_ready"]) == 1

    def test_stop_without_start_is_safe(self) -> None:
        bus = EventBus()
        bridge = BrainUIBridge(bus=bus)
        bridge.stop()  # Must not raise

    def test_stop_unsubscribes(self) -> None:
        bus, bridge = _make_bus_and_bridge()
        bridge.stop()
        assert len(bus._subscribers["brain.processing_started"]) == 0
        assert len(bus._subscribers["brain.response_ready"]) == 0


# ─── brain.processing_started → ui.message + ui.state_change ─────────────────


class TestProcessingStartedTranslation:
    """brain.processing_started must produce a user bubble and 'thinking' state."""

    @pytest.mark.asyncio
    async def test_user_message_published(self) -> None:
        bus, _ = _make_bus_and_bridge()
        received: list[UIMessageEvent] = []

        async def capture(e: UIMessageEvent) -> None:
            received.append(e)

        bus.subscribe("ui.message", capture)

        await bus.publish(BrainProcessingStartedEvent(
            session_id="s1", utterance="Hello Spidy"
        ))

        assert len(received) == 1
        assert received[0].role == "user"
        assert received[0].text == "Hello Spidy"
        assert received[0].session_id == "s1"

    @pytest.mark.asyncio
    async def test_thinking_state_published(self) -> None:
        bus, _ = _make_bus_and_bridge()
        states: list[str] = []

        async def capture(e: UIStateChangeEvent) -> None:
            states.append(e.state)

        bus.subscribe("ui.state_change", capture)

        await bus.publish(BrainProcessingStartedEvent(
            session_id="s1", utterance="What time is it?"
        ))

        assert "thinking" in states

    @pytest.mark.asyncio
    async def test_empty_utterance_skips_message_bubble(self) -> None:
        """An empty utterance must not produce a message bubble, but state still changes."""
        bus, _ = _make_bus_and_bridge()
        messages: list[UIMessageEvent] = []
        states: list[str] = []

        async def cap_msg(e: UIMessageEvent) -> None:
            messages.append(e)

        async def cap_state(e: UIStateChangeEvent) -> None:
            states.append(e.state)

        bus.subscribe("ui.message", cap_msg)
        bus.subscribe("ui.state_change", cap_state)

        await bus.publish(BrainProcessingStartedEvent(session_id="s1", utterance=""))

        assert messages == []          # No bubble for empty utterance
        assert "thinking" in states    # State still transitions


# ─── brain.response_ready → ui.message + ui.state_change ─────────────────────


class TestResponseReadyTranslation:
    """brain.response_ready must produce an assistant bubble and 'idle' state."""

    @pytest.mark.asyncio
    async def test_assistant_message_published(self) -> None:
        bus, _ = _make_bus_and_bridge()
        received: list[UIMessageEvent] = []

        async def capture(e: UIMessageEvent) -> None:
            received.append(e)

        bus.subscribe("ui.message", capture)

        await bus.publish(BrainResponseReadyEvent(
            session_id="s1",
            response_text="I am Spidy, your AI companion!",
            decision_mode="llm",
        ))

        assert len(received) == 1
        assert received[0].role == "assistant"
        assert received[0].text == "I am Spidy, your AI companion!"
        assert received[0].session_id == "s1"

    @pytest.mark.asyncio
    async def test_idle_state_published_after_response(self) -> None:
        bus, _ = _make_bus_and_bridge()
        states: list[str] = []

        async def capture(e: UIStateChangeEvent) -> None:
            states.append(e.state)

        bus.subscribe("ui.state_change", capture)

        await bus.publish(BrainResponseReadyEvent(
            session_id="s1",
            response_text="Sure thing!",
            decision_mode="skill",
        ))

        assert "idle" in states

    @pytest.mark.asyncio
    async def test_empty_response_skips_message_bubble_but_returns_idle(self) -> None:
        """An empty response (no-op turn) must not produce a bubble but must idle the state."""
        bus, _ = _make_bus_and_bridge()
        messages: list[UIMessageEvent] = []
        states: list[str] = []

        async def cap_msg(e: UIMessageEvent) -> None:
            messages.append(e)

        async def cap_state(e: UIStateChangeEvent) -> None:
            states.append(e.state)

        bus.subscribe("ui.message", cap_msg)
        bus.subscribe("ui.state_change", cap_state)

        await bus.publish(BrainResponseReadyEvent(
            session_id="s1", response_text="", decision_mode="noop"
        ))

        assert messages == []       # No empty bubble
        assert "idle" in states     # Overlay still returns to idle

    @pytest.mark.asyncio
    async def test_full_turn_sequence(self) -> None:
        """A complete user-utterance + response cycle produces correct event sequence."""
        bus, _ = _make_bus_and_bridge()
        messages: list[UIMessageEvent] = []
        states: list[str] = []

        async def cap_msg(e: UIMessageEvent) -> None:
            messages.append(e)

        async def cap_state(e: UIStateChangeEvent) -> None:
            states.append(e.state)

        bus.subscribe("ui.message", cap_msg)
        bus.subscribe("ui.state_change", cap_state)

        # Simulate: user speaks → brain starts → brain responds
        await bus.publish(BrainProcessingStartedEvent(
            session_id="s1", utterance="Tell me a joke"
        ))
        await bus.publish(BrainResponseReadyEvent(
            session_id="s1", response_text="Why did the AI cross the road?",
            decision_mode="llm",
        ))

        # Two messages: user, then assistant
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].text == "Tell me a joke"
        assert messages[1].role == "assistant"
        assert messages[1].text == "Why did the AI cross the road?"

        # State sequence: thinking → idle
        assert states[0] == "thinking"
        assert states[-1] == "idle"
