"""
ConversationManager — Rolling Context Window & Session Lifecycle
================================================================
Manages the conversation turns for a single user session.

Responsibilities
----------------
- Append turns (user / assistant / system) to a rolling window
- Enforce a maximum window size (evict oldest turns when full)
- Build an LLM-ready message list from the current window
- Track session start/end and publish lifecycle events

Design
------
- One ConversationManager instance per session (Brain creates a new one
  per ``start_session()`` call)
- Turns are stored as ``ConversationTurn`` objects (immutable)
- The window is a plain deque — O(1) append and eviction
- Thread-safe: the Brain is single-threaded async; no lock needed

Lifelong Companion notes
------------------------
In future milestones, ``add_turn()`` will also call
``MemoryInterface.store()`` to persist episodic memories.
The method signature is already designed for this integration.
"""

from __future__ import annotations

import uuid
from collections import deque
from typing import TYPE_CHECKING

from spidy.brain.types import ConversationTurn, Intent, TurnRole
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)

_DEFAULT_MAX_TURNS = 20
_SYSTEM_PROMPT = (
    "You are Spidy, a helpful, thoughtful, and friendly AI desktop companion. "
    "You help the user with tasks on their Windows computer. "
    "Keep responses concise and natural. "
    "If you don't know something, say so honestly."
)


class ConversationManager:
    """
    Manages the rolling conversation window for an active session.

    Parameters
    ----------
    bus:
        Application EventBus for publishing lifecycle events.
    max_turns:
        Maximum number of turns to keep in the window (oldest evicted first).
    system_prompt:
        The system message prepended to every LLM context window.
    """

    def __init__(
        self,
        bus: "EventBus",
        max_turns: int = _DEFAULT_MAX_TURNS,
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> None:
        self._bus = bus
        self._max_turns = max_turns
        self._system_prompt = system_prompt
        self._turns: deque[ConversationTurn] = deque()
        self._session_id: str = ""
        self._active = False

    # ── Session lifecycle ──────────────────────────────────────────────────

    async def start_session(self, session_id: str | None = None) -> str:
        """
        Begin a new conversation session.

        Parameters
        ----------
        session_id:
            Explicit session ID. If None, a UUID is generated.

        Returns
        -------
        str
            The session ID.
        """
        self._session_id = session_id or str(uuid.uuid4())
        self._turns.clear()
        self._active = True

        from spidy.brain.events import BrainSessionStartedEvent
        await self._bus.publish(BrainSessionStartedEvent(session_id=self._session_id))
        log.debug("Session started: {sid}", sid=self._session_id)
        return self._session_id

    async def end_session(self) -> int:
        """
        End the active session.

        Returns
        -------
        int
            Number of turns that occurred in this session.
        """
        turn_count = len(self._turns)
        self._active = False

        from spidy.brain.events import BrainSessionEndedEvent
        await self._bus.publish(BrainSessionEndedEvent(
            session_id=self._session_id,
            turn_count=turn_count,
        ))
        log.debug(
            "Session ended: {sid} | {n} turns",
            sid=self._session_id,
            n=turn_count,
        )
        return turn_count

    # ── Turn management ────────────────────────────────────────────────────

    def add_turn(
        self,
        role: TurnRole,
        text: str,
        intent: Intent | None = None,
    ) -> ConversationTurn:
        """
        Append a turn to the conversation window.

        If the window is at capacity, the oldest non-system turn is evicted.

        Parameters
        ----------
        role:
            Who is speaking (USER, ASSISTANT, or SYSTEM).
        text:
            The turn text content.
        intent:
            The Intent associated with user turns (optional).

        Returns
        -------
        ConversationTurn
            The newly created turn object.
        """
        turn = ConversationTurn(
            role=role,
            text=text,
            intent=intent,
            session_id=self._session_id,
        )

        self._turns.append(turn)

        # Evict oldest turn if over capacity
        while len(self._turns) > self._max_turns:
            self._turns.popleft()

        log.debug(
            "[{role}] {preview}",
            role=role.value,
            preview=text[:80] + ("..." if len(text) > 80 else ""),
        )
        return turn

    def get_context(self) -> list[ConversationTurn]:
        """Return the current turns in order (oldest first)."""
        return list(self._turns)

    def get_llm_messages(
        self,
        include_system: bool = True,
    ) -> list[dict[str, str]]:
        """
        Build an LLM-ready message list from the current window.

        Parameters
        ----------
        include_system:
            If True, prepends the system prompt as the first message.

        Returns
        -------
        list[dict]
            Messages in ``{"role": str, "content": str}`` format.
        """
        messages: list[dict[str, str]] = []
        if include_system and self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(t.to_llm_message() for t in self._turns)
        return messages

    def get_summary(self) -> str:
        """
        One-line summary of the recent conversation.

        Used by the DecisionEngine for quick context checks.
        """
        if not self._turns:
            return ""

        recent = list(self._turns)[-3:]
        parts: list[str] = []
        for t in recent:
            prefix = "User" if t.role == TurnRole.USER else "Spidy"
            parts.append(f"{prefix}: {t.text[:50]}")
        return " | ".join(parts)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def last_user_utterance(self) -> str:
        """Return the most recent user utterance, or empty string."""
        for turn in reversed(list(self._turns)):
            if turn.role == TurnRole.USER:
                return turn.text
        return ""
