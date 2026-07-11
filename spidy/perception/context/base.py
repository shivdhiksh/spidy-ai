"""
BaseObserver — Abstract Contract for All Context Observers
============================================================
Every observer in Milestone 2 implements this interface.

Design
------
- Each observer is an independent, self-contained polling unit
- Observers run as asyncio Tasks (non-blocking)
- They publish events to the EventBus — never return data directly
- Each has a configurable poll_interval
- All handle graceful start/stop

Thread model
------------
- Observers run entirely in the asyncio event loop by default
- Blocking OS calls (win32gui, psutil disk I/O) are wrapped in
  asyncio.to_thread() to prevent stalling the event loop
- Observers that use external file-system watchers (watchdog)
  bridge callbacks via EventBus.publish_threadsafe()

Error handling
--------------
- Errors in poll() are caught and logged — never crash the observer
- If an observer fails N consecutive times it backs off exponentially
- After max_consecutive_errors it self-suspends and publishes a
  context.observer_error event
"""

from __future__ import annotations

import asyncio
import abc
from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)

_MAX_CONSECUTIVE_ERRORS = 10
_BACKOFF_BASE = 1.5
_MAX_BACKOFF_SECONDS = 30.0


class BaseObserver(abc.ABC):
    """
    Abstract base class for all Context Observers.

    Subclasses must implement:
      - poll() — one observation cycle
      - name — a unique, human-readable identifier

    Parameters
    ----------
    bus:
        The application EventBus for publishing events.
    poll_interval:
        How often to call poll(), in seconds.
    """

    def __init__(self, bus: "EventBus", poll_interval: float = 1.0) -> None:
        self._bus = bus
        self._poll_interval = poll_interval
        self._running = False
        self._task: asyncio.Task | None = None
        self._consecutive_errors = 0

    # ── Abstract interface ─────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique observer name (e.g. 'active_window')."""

    @abc.abstractmethod
    async def poll(self) -> None:
        """
        Perform one observation cycle.

        Called repeatedly at poll_interval.
        Must not raise — catch and handle all exceptions internally.
        Must not block — use asyncio.to_thread() for blocking I/O.
        """

    # ── Optional lifecycle hooks ───────────────────────────────────────────

    async def on_start(self) -> None:
        """Called once before the polling loop begins. Override if needed."""

    async def on_stop(self) -> None:
        """Called once after the polling loop ends. Override if needed."""

    # ── Public API ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the observer polling loop as an asyncio Task."""
        if self._running:
            log.debug("{name} observer already running.", name=self.name)
            return

        self._running = True
        await self.on_start()
        self._task = asyncio.create_task(
            self._loop(),
            name=f"observer-{self.name}",
        )
        log.debug("Observer started: {name} | interval={i}s",
                  name=self.name, i=self._poll_interval)

    async def stop(self) -> None:
        """Stop the observer and wait for the task to finish."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.on_stop()
        log.debug("Observer stopped: {name}", name=self.name)

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Internal polling loop ──────────────────────────────────────────────

    async def _loop(self) -> None:
        """Main poll loop with error isolation and exponential backoff."""
        backoff = self._poll_interval

        while self._running:
            try:
                await self.poll()
                self._consecutive_errors = 0
                backoff = self._poll_interval
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                self._consecutive_errors += 1
                log.warning(
                    "Observer '{name}' poll error #{n}: {exc}",
                    name=self.name,
                    n=self._consecutive_errors,
                    exc=exc,
                )

                if self._consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    log.error(
                        "Observer '{name}' exceeded {max} consecutive errors. "
                        "Self-suspending.",
                        name=self.name,
                        max=_MAX_CONSECUTIVE_ERRORS,
                    )
                    await self._publish_error(str(exc))
                    break

                # Exponential backoff
                backoff = min(backoff * _BACKOFF_BASE, _MAX_BACKOFF_SECONDS)

            await asyncio.sleep(backoff)

    async def _publish_error(self, message: str) -> None:
        """Publish a context.observer_error event."""
        from spidy.perception.context.events import ObserverErrorEvent
        await self._bus.publish(ObserverErrorEvent(
            observer_name=self.name,
            message=message,
        ))
