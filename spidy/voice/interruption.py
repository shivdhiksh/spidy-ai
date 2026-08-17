"""
InterruptionHandler — Voice Interrupt Command Processor
========================================================
Detects and handles voice interrupt commands before routing
utterances to the Brain.

Supported interrupt commands (Part 7)
--------------------------------------
  stop / cancel / never mind  → Cancel active goal + stop TTS
  wait / pause               → Pause voice session (keep awake)
  resume / continue          → Resume paused session

Design
------
Detection is purely lexical — no LLM call required.
The handler checks the transcript against a configurable keyword set.
Matched phrases are intercepted and never forwarded to Brain.process().

This keeps interruption response time well under 100ms.

Usage
-----
    handler = InterruptionHandler(bus=bus, brain=brain, tts=tts, session=mgr)

    command = handler.detect(transcript)
    if command:
        await handler.execute(command, transcript)
        # Do NOT forward to Brain — return early
    else:
        response = await brain.process(transcript)
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger
from spidy.voice.events import VoiceInterruptEvent

if TYPE_CHECKING:
    from spidy.brain.brain import Brain
    from spidy.core.event_bus import EventBus
    from spidy.perception.voice.tts.base import TTSEngine
    from spidy.voice.session import VoiceSessionManager

log = get_logger(__name__)


class InterruptCommand(Enum):
    STOP = auto()
    CANCEL = auto()
    NEVER_MIND = auto()
    WAIT = auto()
    PAUSE = auto()
    RESUME = auto()
    CONTINUE = auto()

    @property
    def is_cancellation(self) -> bool:
        """True if this command should cancel the active goal."""
        return self in (
            InterruptCommand.STOP,
            InterruptCommand.CANCEL,
            InterruptCommand.NEVER_MIND,
        )

    @property
    def is_pause(self) -> bool:
        return self in (InterruptCommand.WAIT, InterruptCommand.PAUSE)

    @property
    def is_resume(self) -> bool:
        return self in (InterruptCommand.RESUME, InterruptCommand.CONTINUE)


# ---------------------------------------------------------------------------
# Stop-command phrases — expanded set for reliable TTS interruption.
#
# These are evaluated FIRST (before general interrupt patterns) so that
# wake-name-prefixed variants like "Spidy stop", "Spidy, stop" are caught
# immediately without falling through to the LLM / Brain.
#
# Safety rule: generic words like "stop" only trigger when they appear at
# a clear phrase boundary — never inside a longer content phrase such as
# "I want to stop smoking" (which has additional words after "stop").
# Ordering: longest / most-specific phrases first.
# ---------------------------------------------------------------------------
_STOP_PHRASES: tuple[str, ...] = (
    # Wake-name prefixed variants (always high priority)
    "spidy stop",
    "spidy, stop",
    "hey spidy stop",
    "hey spidy, stop",
    "jarvis stop",
    "jarvis, stop",
    "hey jarvis stop",
    "hey jarvis, stop",
    # Descriptive stop variants
    "stop talking",
    "stop speaking",
    "stop now",
    "stop that",
    "please stop",
    "please be quiet",
    "be quiet spidy",
    "be quiet jarvis",
    "be quiet",
    "quiet please",
    "spidy be quiet",
    "jarvis be quiet",
    "shut up",
    "that's enough",
    "thats enough",
    "enough",
    # Bare stop (matched as whole phrase or at word boundary)
    "stop",
)

# General interrupt patterns (pause / resume / cancel)
# These extend the stop set for pause/resume flows.
_INTERRUPT_PATTERNS: list[tuple[str, InterruptCommand]] = [
    ("never mind",    InterruptCommand.NEVER_MIND),
    ("nevermind",     InterruptCommand.NEVER_MIND),
    ("cancel that",   InterruptCommand.CANCEL),
    ("cancel",        InterruptCommand.CANCEL),
    ("stop that",     InterruptCommand.STOP),
    ("stop",          InterruptCommand.STOP),
    ("wait a moment", InterruptCommand.WAIT),
    ("wait a sec",    InterruptCommand.WAIT),
    ("hold on",       InterruptCommand.WAIT),
    ("wait",          InterruptCommand.WAIT),
    ("pause",         InterruptCommand.PAUSE),
    ("resume",        InterruptCommand.RESUME),
    ("continue",      InterruptCommand.CONTINUE),
    ("go ahead",      InterruptCommand.RESUME),
    ("carry on",      InterruptCommand.RESUME),
]


class InterruptionHandler:
    """
    Detects and executes voice interrupt commands.

    Parameters
    ----------
    bus:
        EventBus — publishes VoiceInterruptEvent on detection.
    brain:
        Brain instance — cancel_goal() called on stop/cancel.
    tts:
        TTSEngine — stop() called immediately on cancellation.
    session:
        VoiceSessionManager — pause/resume state transitions.
    extra_keywords:
        Additional interrupt keyword mappings beyond the defaults.
    """

    def __init__(
        self,
        bus: "EventBus",
        brain: "Brain",
        tts: "TTSEngine",
        session: "VoiceSessionManager",
        extra_keywords: list[tuple[str, InterruptCommand]] | None = None,
    ) -> None:
        self._bus = bus
        self._brain = brain
        self._tts = tts
        self._session = session

        # Build pattern list (longest-match first)
        self._patterns = list(_INTERRUPT_PATTERNS)
        if extra_keywords:
            self._patterns = extra_keywords + self._patterns

    def detect(self, transcript: str) -> InterruptCommand | None:
        """
        Check a transcript for interrupt keywords.

        Detection order
        ---------------
        1. Check against the expanded ``_STOP_PHRASES`` set — handles all
           wake-name-prefixed and descriptive variants ("Spidy stop",
           "stop talking", "be quiet Spidy", etc.).
        2. Fall through to the general ``_INTERRUPT_PATTERNS`` for any
           remaining pause/resume keywords not covered by step 1.

        Parameters
        ----------
        transcript:
            The raw user utterance (after wake-prefix stripping).

        Returns
        -------
        InterruptCommand or None
            The detected command, or None if no interrupt was found.
        """
        normalised = transcript.strip().lower()
        # Strip leading/trailing punctuation for cleaner boundary matching
        normalised_clean = normalised.strip(".,!?;: ")

        # ── Step 1: High-priority stop-phrase set ──────────────────────────
        for phrase in _STOP_PHRASES:
            if (
                normalised_clean == phrase
                or normalised_clean.startswith(phrase + " ")
                or normalised_clean.endswith(" " + phrase)
            ):
                log.debug(
                    "InterruptionHandler: stop phrase matched '{phrase}' in '{text}'",
                    phrase=phrase,
                    text=transcript[:60],
                )
                return InterruptCommand.STOP

        # ── Step 2: General interrupt patterns (pause / resume / cancel) ───
        for keyword, command in self._patterns:
            if (
                normalised_clean == keyword
                or normalised_clean.startswith(keyword + " ")
                or normalised_clean.endswith(" " + keyword)
            ):
                log.debug(
                    "InterruptionHandler: detected '{cmd}' in '{text}'",
                    cmd=command.name,
                    text=transcript[:60],
                )
                return command
        return None

    async def execute(self, command: InterruptCommand, utterance: str = "") -> str:
        """
        Execute an interrupt command.

        Parameters
        ----------
        command:
            The detected interrupt command.
        utterance:
            The original user phrase (for event logging).

        Returns
        -------
        str
            A response to speak back to the user (may be empty).
        """
        # Publish interrupt event
        await self._bus.publish(VoiceInterruptEvent(
            command=command.name.lower(),
            utterance=utterance,
        ))

        if command.is_cancellation:
            return await self._handle_cancellation(command, utterance)
        elif command.is_pause:
            return await self._handle_pause()
        elif command.is_resume:
            return await self._handle_resume()

        return ""

    # ── Private handlers ──────────────────────────────────────────────────

    async def _handle_cancellation(
        self,
        command: InterruptCommand,
        utterance: str,
    ) -> str:
        """
        Stop TTS + cancel active goal.

        SAFETY DESIGN: The session is intentionally NOT deactivated here.
        After a stop command, Spidy remains running with the session AWAKE
        so the user can immediately continue speaking without repeating the
        wake word.  The overlay transitions to AWAKE·LISTENING via the
        UIStateChangeEvent published by ContinuousVoiceController after
        this handler returns.

        TTS is stopped via both the interrupt flag (clean sentence-boundary
        stop) AND a direct engine.stop() call for truly instant cutoff.
        """
        # 1. Stop TTS immediately — interrupt flag + direct engine stop
        #    so cutoff is instant regardless of sentence boundaries.
        self._tts.stop()
        log.info(
            "InterruptionHandler: TTS stopped by '{cmd}'.",
            cmd=command.name,
        )

        # 2. Cancel any active autonomous goal (non-fatal if none running)
        cancelled = await self._brain.cancel_goal()

        # 3. Session intentionally kept ACTIVE — user can continue speaking.
        #    ContinuousVoiceController restores overlay and re-listen state.
        if cancelled:
            log.info("InterruptionHandler: active goal cancelled.")
        else:
            log.debug("InterruptionHandler: no active goal to cancel.")

        # Return empty string — CVC will handle overlay + re-listen.
        return ""

    async def _handle_pause(self) -> str:
        """Pause the voice session."""
        self._tts.stop()
        await self._session.pause()
        return "Sure, I'll wait. Just say 'resume' when you're ready."

    async def _handle_resume(self) -> str:
        """Resume a paused session."""
        await self._session.resume()
        return "I'm listening again."
