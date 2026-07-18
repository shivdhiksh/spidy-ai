"""
BrainUIBridge — Brain -> UI event translator
============================================
Subscribes to Brain events on the EventBus and re-publishes them as UI
events so the overlay chat view stays in sync with the conversation.

Why this module exists
----------------------
The Brain publishes events in its own namespace (``brain.*``).
The overlay subscribes exclusively to the UI namespace (``ui.*``).
Neither module knows about the other -- that is the correct design.
This bridge is the only place that couples the two namespaces, and it
lives in ``spidy/ui/`` to make that dependency direction explicit:
the UI layer depends on the Brain layer, not the other way around.

Event mapping
-------------
    brain.processing_started
        -> ui.message        (role="user",      text=event.utterance)
        -> ui.state_change   (state="thinking")

    brain.response_ready
        -> ui.message        (role="assistant", text=event.response_text)  [if non-empty]
        -> ui.state_change   (state="idle")

Usage
-----
    bridge = BrainUIBridge(bus=event_bus)
    bridge.start()   # subscribe

    # ... later ...
    bridge.stop()    # unsubscribe
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)


class BrainUIBridge:
    """
    Translates ``brain.*`` EventBus events into ``ui.*`` events.

    This is deliberately a one-way adapter: it reads Brain events and
    writes UI events. It never reads UI events or writes Brain events.

    Parameters
    ----------
    bus:
        The shared application EventBus instance.
    """

    def __init__(self, bus: "EventBus") -> None:
        self._bus = bus
        self._running = False

    # -- Lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Subscribe to brain events."""
        if self._running:
            return

        self._bus.subscribe("brain.processing_started", self._on_processing_started)
        self._bus.subscribe("brain.response_ready", self._on_response_ready)
        self._running = True
        log.debug("BrainUIBridge started -- subscribing to brain.* events.")

    def stop(self) -> None:
        """Unsubscribe from brain events."""
        if not self._running:
            return

        self._bus.unsubscribe("brain.processing_started", self._on_processing_started)
        self._bus.unsubscribe("brain.response_ready", self._on_response_ready)
        self._running = False
        log.debug("BrainUIBridge stopped.")

    # -- Handlers -----------------------------------------------------------

    async def _on_processing_started(self, event: Any) -> None:
        """
        brain.processing_started -> ui.message (user) + ui.state_change (thinking).

        Adds the user utterance as a "user" bubble and activates the
        thinking spinner so the user knows the Brain is working.
        """
        from spidy.ui.events import UIMessageEvent, UIStateChangeEvent

        utterance: str = getattr(event, "utterance", "")
        session_id: str = getattr(event, "session_id", "")

        if utterance:
            await self._bus.publish(
                UIMessageEvent(role="user", text=utterance, session_id=session_id)
            )
            log.debug(
                "BrainUIBridge: user message forwarded to UI ({n} chars).",
                n=len(utterance),
            )

        # Always transition to "thinking" while the Brain works
        await self._bus.publish(UIStateChangeEvent(state="thinking"))

    async def _on_response_ready(self, event: Any) -> None:
        """
        brain.response_ready -> ui.message (assistant) + ui.state_change (idle).

        Adds the assistant reply as an "assistant" bubble and returns the
        overlay to the idle state. Empty responses (no-op turns) are
        silently skipped -- no empty bubble is shown.
        """
        from spidy.ui.events import UIMessageEvent, UIStateChangeEvent

        response_text: str = getattr(event, "response_text", "")
        session_id: str = getattr(event, "session_id", "")

        if response_text:
            await self._bus.publish(
                UIMessageEvent(
                    role="assistant", text=response_text, session_id=session_id
                )
            )
            log.debug(
                "BrainUIBridge: assistant message forwarded to UI ({n} chars).",
                n=len(response_text),
            )
        else:
            log.debug("BrainUIBridge: empty response -- no bubble emitted.")

        # Always return to idle after a response (even a silent one)
        await self._bus.publish(UIStateChangeEvent(state="idle"))
