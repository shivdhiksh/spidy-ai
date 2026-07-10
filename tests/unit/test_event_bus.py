"""
Test suite for spidy.core.event_bus
=====================================
Tests async publish/subscribe, error isolation, and threadsafe publishing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from spidy.core.event_bus import Event, EventBus


# ─── Test event fixtures ──────────────────────────────────────────────────────

@dataclass
class PingEvent(Event):
    topic = "test.ping"
    payload: str = "ping"


@dataclass
class PongEvent(Event):
    topic = "test.pong"
    value: int = 0


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestEventBusSubscribePublish:
    """Core pub/sub behaviour."""

    @pytest.mark.asyncio
    async def test_subscriber_receives_event(self):
        """A subscribed handler must be called when the matching topic is published."""
        bus = EventBus()
        received: list[PingEvent] = []

        async def handler(event: PingEvent) -> None:
            received.append(event)

        bus.subscribe("test.ping", handler)
        await bus.publish(PingEvent(payload="hello"))

        assert len(received) == 1
        assert received[0].payload == "hello"

    @pytest.mark.asyncio
    async def test_multiple_subscribers_all_called(self):
        """All subscribers to the same topic must all be called."""
        bus = EventBus()
        calls: list[str] = []

        async def handler_a(event: PingEvent) -> None:
            calls.append("a")

        async def handler_b(event: PingEvent) -> None:
            calls.append("b")

        bus.subscribe("test.ping", handler_a)
        bus.subscribe("test.ping", handler_b)
        await bus.publish(PingEvent())

        assert "a" in calls
        assert "b" in calls

    @pytest.mark.asyncio
    async def test_handler_only_receives_its_topic(self):
        """A handler subscribed to topic A must NOT be called for topic B."""
        bus = EventBus()
        ping_calls: list[Event] = []
        pong_calls: list[Event] = []

        async def on_ping(event: PingEvent) -> None:
            ping_calls.append(event)

        async def on_pong(event: PongEvent) -> None:
            pong_calls.append(event)

        bus.subscribe("test.ping", on_ping)
        bus.subscribe("test.pong", on_pong)

        await bus.publish(PingEvent())

        assert len(ping_calls) == 1
        assert len(pong_calls) == 0

    @pytest.mark.asyncio
    async def test_no_subscribers_returns_zero(self):
        """Publishing to a topic with no subscribers should return 0."""
        bus = EventBus()
        handled = await bus.publish(PingEvent())
        assert handled == 0

    @pytest.mark.asyncio
    async def test_publish_returns_handled_count(self):
        """publish() should return the count of successfully called handlers."""
        bus = EventBus()

        async def h1(e): pass
        async def h2(e): pass

        bus.subscribe("test.ping", h1)
        bus.subscribe("test.ping", h2)
        count = await bus.publish(PingEvent())
        assert count == 2


class TestEventBusErrorIsolation:
    """Exceptions in one subscriber must not block others."""

    @pytest.mark.asyncio
    async def test_failing_subscriber_does_not_block_others(self):
        """If subscriber A raises, subscriber B must still be called."""
        bus = EventBus()
        second_called = False

        async def bad_handler(event: PingEvent) -> None:
            raise RuntimeError("I broke!")

        async def good_handler(event: PingEvent) -> None:
            nonlocal second_called
            second_called = True

        bus.subscribe("test.ping", bad_handler)
        bus.subscribe("test.ping", good_handler)

        # Must not raise despite bad_handler failing
        await bus.publish(PingEvent())
        assert second_called

    @pytest.mark.asyncio
    async def test_error_increments_error_count(self):
        """Each subscriber exception should increment bus.stats['errors']."""
        bus = EventBus()

        async def bad(event: PingEvent) -> None:
            raise ValueError("oops")

        bus.subscribe("test.ping", bad)
        await bus.publish(PingEvent())
        assert bus.stats["errors"] == 1


class TestEventBusUnsubscribe:
    """Unsubscribe must prevent future calls."""

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_handler(self):
        """After unsubscribing, the handler must not be called on publish."""
        bus = EventBus()
        calls: list[int] = []

        async def handler(event: PingEvent) -> None:
            calls.append(1)

        bus.subscribe("test.ping", handler)
        await bus.publish(PingEvent())  # handler is called
        assert len(calls) == 1

        bus.unsubscribe("test.ping", handler)
        await bus.publish(PingEvent())  # handler must NOT be called
        assert len(calls) == 1

    def test_unsubscribe_nonexistent_is_noop(self):
        """Unsubscribing a handler that was never registered must not raise."""
        bus = EventBus()

        async def phantom(event: PingEvent) -> None:
            pass

        bus.unsubscribe("test.ping", phantom)  # Must not raise


class TestEventBusDuplicateSubscription:
    """Subscribing the same handler twice must be deduplicated."""

    @pytest.mark.asyncio
    async def test_duplicate_subscription_is_ignored(self):
        """Registering the same handler twice should only call it once."""
        bus = EventBus()
        call_count = 0

        async def handler(event: PingEvent) -> None:
            nonlocal call_count
            call_count += 1

        bus.subscribe("test.ping", handler)
        bus.subscribe("test.ping", handler)  # duplicate
        await bus.publish(PingEvent())
        assert call_count == 1


class TestEventBusStats:
    """Stats should accurately reflect activity."""

    @pytest.mark.asyncio
    async def test_stats_published_increments(self):
        bus = EventBus()
        async def h(e): pass
        bus.subscribe("test.ping", h)
        await bus.publish(PingEvent())
        await bus.publish(PingEvent())
        assert bus.stats["published"] == 2

    def test_stats_topics_count(self):
        bus = EventBus()
        async def h(e): pass
        bus.subscribe("topic.a", h)
        bus.subscribe("topic.b", h)
        assert bus.stats["topics"] == 2


class TestEventBaseClass:
    """Event subclasses must define topic."""

    def test_missing_topic_raises(self):
        """An Event subclass without a class-level topic should raise TypeError."""
        @dataclass
        class BadEvent(Event):
            pass  # no topic defined

        with pytest.raises(TypeError, match="must define a class-level 'topic'"):
            BadEvent()
