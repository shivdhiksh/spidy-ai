"""
Spidy Event Bus
===============
Async pub/sub event system that decouples all modules.

Architecture
------------
Every module communicates through events — they never call each other
directly. This gives us:

- Loose coupling: the VoiceEngine doesn't know the UI exists
- Easy testing: inject fake events in tests
- Plugin safety: plugins subscribe to events, can't call internals
- Observability: log every event for debugging

Design
------
- Events are typed dataclasses for clarity and IDE support.
- The bus runs inside the asyncio event loop.
- Multiple subscribers per topic are supported.
- Subscribers run sequentially in subscription order (simple, predictable).
- Unhandled exceptions in a subscriber are logged and don't crash the bus.

Usage
-----
    from spidy.core.event_bus import EventBus, Event

    bus = EventBus()

    # Define an event
    @dataclass
    class WakeWordDetectedEvent(Event):
        topic = "wake_word.detected"
        confidence: float

    # Subscribe
    async def on_wake(event: WakeWordDetectedEvent):
        print(f"Wake word heard! Confidence: {event.confidence}")

    bus.subscribe("wake_word.detected", on_wake)

    # Publish
    await bus.publish(WakeWordDetectedEvent(confidence=0.95))
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from spidy.logging.logger import get_logger

log = get_logger(__name__)

# ─── Event base class ─────────────────────────────────────────────────────────


@dataclass
class Event:
    """
    Base class for all Spidy events.

    Subclasses must define a class-level ``topic`` string that acts as
    the routing key. Subscribers bind to this topic string.

    Example
    -------
        @dataclass
        class UserSpoke(Event):
            topic = "voice.user_spoke"
            transcript: str
            confidence: float
    """

    topic: str = field(init=False, default="event.base")

    def __post_init__(self) -> None:
        # Ensure subclasses define their own topic at class level
        if type(self).topic == "event.base" and type(self) is not Event:
            raise TypeError(
                f"{type(self).__name__} must define a class-level 'topic' attribute."
            )


# ─── Built-in system events ───────────────────────────────────────────────────


@dataclass
class SpidyStartedEvent(Event):
    """Published once when SpidyCore finishes initialisation."""
    topic = "system.started"


@dataclass
class SpidyShuttingDownEvent(Event):
    """Published when the application begins graceful shutdown."""
    topic = "system.shutting_down"


@dataclass
class ErrorEvent(Event):
    """Published when a module encounters an unrecoverable error."""
    topic = "system.error"
    source: str = ""       # module that raised the error
    message: str = ""
    exc: Exception | None = None


# ─── Handler type alias ───────────────────────────────────────────────────────

# A handler is an async callable that accepts one Event argument.
EventHandler = Callable[[Any], Awaitable[None]]


# ─── EventBus ─────────────────────────────────────────────────────────────────


class EventBus:
    """
    Async publish/subscribe event bus.

    Lifecycle
    ---------
    Create one instance in SpidyCore and pass it to all modules that need it.
    The bus does not own any background tasks; it dispatches to subscribers
    inline within the asyncio event loop.

    Thread Safety
    -------------
    This bus is designed for single-threaded asyncio use. For events from
    background threads (e.g. the audio capture thread), use
    ``publish_threadsafe()`` which routes through ``call_soon_threadsafe``.
    """

    def __init__(self) -> None:
        # topic → list of async handlers
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        # Counts for observability
        self._published: int = 0
        self._errors: int = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """
        Register an async handler for the given topic.

        Parameters
        ----------
        topic:
            The event topic string to listen for.
        handler:
            An ``async def`` callable that accepts a single Event argument.

        Notes
        -----
        Subscribing the same handler twice to the same topic is a no-op
        (deduplicated).
        """
        if handler in self._subscribers[topic]:
            log.warning(
                "Handler {h} already subscribed to '{topic}' — skipping duplicate.",
                h=handler.__qualname__,
                topic=topic,
            )
            return

        self._subscribers[topic].append(handler)
        log.debug(
            "Subscribed {h} → '{topic}'",
            h=handler.__qualname__,
            topic=topic,
        )

    def unsubscribe(self, topic: str, handler: EventHandler) -> None:
        """
        Remove a previously registered handler.

        Silently ignores if handler was not registered.
        """
        try:
            self._subscribers[topic].remove(handler)
            log.debug(
                "Unsubscribed {h} from '{topic}'",
                h=handler.__qualname__,
                topic=topic,
            )
        except ValueError:
            pass

    async def publish(self, event: Event) -> int:
        """
        Publish an event and await all subscribers.

        Subscribers are called sequentially. Exceptions in one subscriber
        do NOT propagate to others — they are logged and counted.

        Parameters
        ----------
        event:
            The event instance to publish. Must be an Event subclass.

        Returns
        -------
        int
            Number of subscribers that handled the event.
        """
        topic = event.topic
        handlers = self._subscribers.get(topic, [])

        log.debug(
            "Publishing '{topic}' → {n} subscriber(s)",
            topic=topic,
            n=len(handlers),
        )

        self._published += 1
        handled = 0

        for handler in handlers:
            try:
                await handler(event)
                handled += 1
            except Exception as exc:  # noqa: BLE001
                self._errors += 1
                log.exception(
                    "Subscriber {h} raised an exception handling '{topic}': {exc}",
                    h=handler.__qualname__,
                    topic=topic,
                    exc=exc,
                )

        return handled

    def publish_threadsafe(self, event: Event) -> None:
        """
        Schedule an event publication from a non-asyncio thread.

        Use this when publishing from background threads (e.g. the audio
        capture thread, OpenCV loop, etc.).

        Parameters
        ----------
        event:
            The event to publish. Will be scheduled on the event loop.

        Raises
        ------
        RuntimeError
            If no event loop has been registered via ``set_loop()``.
        """
        if self._loop is None:
            raise RuntimeError(
                "Call EventBus.set_loop(loop) from the main async context "
                "before publishing events from threads."
            )
        self._loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(self.publish(event), loop=self._loop)
        )

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the asyncio event loop for threadsafe publishing."""
        self._loop = loop

    @property
    def stats(self) -> dict[str, int]:
        """Return simple observability stats."""
        return {
            "published": self._published,
            "errors": self._errors,
            "topics": len(self._subscribers),
            "total_handlers": sum(len(h) for h in self._subscribers.values()),
        }

    def __repr__(self) -> str:
        return (
            f"EventBus(topics={len(self._subscribers)}, "
            f"published={self._published}, errors={self._errors})"
        )
