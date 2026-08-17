"""
VoiceSessionManager — Continuous Conversation State Machine
============================================================
Manages the lifecycle of a continuous voice conversation session.

After wake word detection, the session opens.  Each user utterance
resets the inactivity timer.  If no speech arrives within
``timeout_seconds``, the session closes automatically (Spidy returns
to sleep / wake-word-listening mode).

Session State Machine
---------------------

  ┌──────────┐   activate()   ┌──────────┐
  │   IDLE   │ ─────────────► │  AWAKE   │
  └──────────┘                └────┬─────┘
       ▲                           │ record_utterance()
       │                           ▼
  deactivate()             ┌──────────────┐
  or timeout               │  PROCESSING  │ ← Brain pipeline runs here
       ▲                   └──────┬───────┘
       │                          │ response ready
       │                          ▼
  ┌────┴──────┐             ┌──────────────┐
  │  PAUSED   │◄─ pause() ─ │   SPEAKING   │
  └───────────┘             └──────────────┘
       │
       └─ resume() ─► AWAKE (reset timer)

After each speaking cycle, if continuous_mode=True the session stays
AWAKE and listens for the next utterance without requiring the wake word.

Thread Safety
-------------
State is protected by an asyncio.Lock.  All public methods are async-safe
and may be called from multiple coroutines.

Usage
-----
    mgr = VoiceSessionManager(bus=bus, timeout_seconds=60.0)
    await mgr.activate(wake_word="hey spidy")
    await mgr.record_utterance("Open Chrome")
    # ... Brain processes ...
    # If no speech for 60 seconds:
    await mgr.deactivate(reason="timeout")
"""

from __future__ import annotations

import asyncio
import uuid
from enum import Enum, auto
from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger
from spidy.voice.events import (
    VoiceSessionEndedEvent,
    VoiceSessionStartedEvent,
    VoiceSessionTimeoutEvent,
)

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)


class VoiceSessionState(Enum):
    IDLE = auto()
    AWAKE = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    PAUSED = auto()


class VoiceSessionManager:
    """
    Manages a continuous voice conversation session.

    Parameters
    ----------
    bus:
        Application EventBus — session lifecycle events are published here.
    timeout_seconds:
        Seconds of inactivity before the session ends automatically.
        Default: 60.0 s.
    max_turns:
        Maximum utterances per session before forcing a reset.
        Default: 50.
    continuous_mode:
        When True, the session stays AWAKE after each response, allowing
        follow-up commands without repeating the wake word.
    """

    def __init__(
        self,
        bus: "EventBus",
        timeout_seconds: float = 60.0,
        max_turns: int = 50,
        continuous_mode: bool = True,
    ) -> None:
        self._bus = bus
        self._timeout_seconds = timeout_seconds
        self._max_turns = max_turns
        self._continuous_mode = continuous_mode

        self._state = VoiceSessionState.IDLE
        self._state_lock = asyncio.Lock()

        self._session_id: str = ""
        self._turn_count: int = 0
        self._last_wake_word: str = ""

        # Current goal being pursued (populated by voice controller)
        self._current_goal: str = ""

        # Timeout timer handle
        self._timeout_task: asyncio.Task | None = None

    # ── Primary API ───────────────────────────────────────────────────────

    async def activate(self, wake_word: str = "") -> str:
        """
        Open a new voice session after wake word detection.

        Returns
        -------
        str
            The new session ID.
        """
        async with self._state_lock:
            if self._state not in (VoiceSessionState.IDLE,):
                # Already active — reset timer instead of creating new session
                log.debug("VoiceSession: already active, resetting timeout.")
                self._reset_timeout()
                return self._session_id

            self._session_id = str(uuid.uuid4())[:8]
            self._turn_count = 0
            self._last_wake_word = wake_word
            self._current_goal = ""
            self._state = VoiceSessionState.AWAKE

        log.info(
            "VoiceSession started: id={sid} | wake='{ww}'",
            sid=self._session_id,
            ww=wake_word or "(none)",
        )

        await self._bus.publish(VoiceSessionStartedEvent(
            session_id=self._session_id,
            wake_word=wake_word,
        ))

        self._reset_timeout()
        return self._session_id

    async def record_utterance(self, text: str) -> None:
        """
        Record that the user spoke. Resets the inactivity timeout.

        Parameters
        ----------
        text:
            The transcribed utterance (used for goal tracking).
        """
        async with self._state_lock:
            if self._state == VoiceSessionState.IDLE:
                return
            self._turn_count += 1
            self._current_goal = text
            self._state = VoiceSessionState.PROCESSING

        self._reset_timeout()

        if self._turn_count >= self._max_turns:
            log.info("VoiceSession: max turns reached ({n}).", n=self._max_turns)
            await self.deactivate(reason="max_turns")

    async def mark_speaking(self) -> None:
        """Transition to SPEAKING state while TTS plays."""
        async with self._state_lock:
            if self._state == VoiceSessionState.PROCESSING:
                self._state = VoiceSessionState.SPEAKING

    async def mark_response_complete(self) -> None:
        """
        Called when TTS finishes speaking.

        If continuous_mode is on, returns to AWAKE (listen for next
        utterance without needing wake word).
        Otherwise, deactivates.
        """
        async with self._state_lock:
            if self._state == VoiceSessionState.SPEAKING:
                if self._continuous_mode:
                    self._state = VoiceSessionState.AWAKE
                else:
                    self._state = VoiceSessionState.IDLE
                    return   # Will deactivate below

        if not self._continuous_mode:
            await self.deactivate(reason="single_turn")

    async def pause(self) -> None:
        """Pause the session (user said 'wait' or 'pause')."""
        async with self._state_lock:
            if self._state in (VoiceSessionState.AWAKE, VoiceSessionState.SPEAKING):
                self._state = VoiceSessionState.PAUSED
                self._cancel_timeout()
        log.info("VoiceSession paused: id={sid}", sid=self._session_id)

    async def resume(self) -> None:
        """Resume from paused state."""
        async with self._state_lock:
            if self._state == VoiceSessionState.PAUSED:
                self._state = VoiceSessionState.AWAKE
        self._reset_timeout()
        log.info("VoiceSession resumed: id={sid}", sid=self._session_id)

    async def deactivate(self, reason: str = "user_stopped") -> None:
        """
        End the voice session.

        Parameters
        ----------
        reason:
            Why the session ended: "timeout" | "cancelled" | "user_stopped"
            | "max_turns" | "single_turn"
        """
        async with self._state_lock:
            if self._state == VoiceSessionState.IDLE:
                return
            sid = self._session_id
            turns = self._turn_count
            self._state = VoiceSessionState.IDLE
            self._current_goal = ""

        self._cancel_timeout()

        log.info(
            "VoiceSession ended: id={sid} | turns={t} | reason={r}",
            sid=sid,
            t=turns,
            r=reason,
        )

        await self._bus.publish(VoiceSessionEndedEvent(
            session_id=sid,
            turn_count=turns,
            reason=reason,
        ))

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def state(self) -> VoiceSessionState:
        return self._state

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_active(self) -> bool:
        return self._state != VoiceSessionState.IDLE

    @property
    def is_paused(self) -> bool:
        return self._state == VoiceSessionState.PAUSED

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def current_goal(self) -> str:
        return self._current_goal

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    @timeout_seconds.setter
    def timeout_seconds(self, value: float) -> None:
        self._timeout_seconds = max(5.0, value)

    @property
    def continuous_mode(self) -> bool:
        return self._continuous_mode

    @continuous_mode.setter
    def continuous_mode(self, value: bool) -> None:
        self._continuous_mode = value

    # ── Timeout management ────────────────────────────────────────────────

    def _reset_timeout(self) -> None:
        """Cancel any existing timeout and start a fresh one."""
        self._cancel_timeout()
        try:
            loop = asyncio.get_running_loop()
            self._timeout_task = loop.create_task(
                self._timeout_watcher(),
                name="voice_session_timeout",
            )
        except RuntimeError:
            # No running loop (unit tests may call from sync context)
            pass

    def _cancel_timeout(self) -> None:
        """Cancel the inactivity timeout task if running."""
        if self._timeout_task is not None and not self._timeout_task.done():
            self._timeout_task.cancel()
        self._timeout_task = None

    async def _timeout_watcher(self) -> None:
        """Async task: fire timeout after inactivity."""
        try:
            await asyncio.sleep(self._timeout_seconds)
        except asyncio.CancelledError:
            return

        if self._state in (VoiceSessionState.AWAKE,):
            log.info(
                "VoiceSession: timeout after {t}s of inactivity.",
                t=self._timeout_seconds,
            )
            await self._bus.publish(VoiceSessionTimeoutEvent(
                session_id=self._session_id,
                timeout_seconds=self._timeout_seconds,
            ))
            await self.deactivate(reason="timeout")
