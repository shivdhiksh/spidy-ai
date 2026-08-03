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
    "You are Spidy, an intelligent and friendly AI desktop companion — like JARVIS but warmer. "
    "You help the user with tasks on their Windows computer: opening apps, searching the web, "
    "managing files, setting timers, controlling system settings, and answering questions. "
    "Keep responses concise and natural — no bullet points unless asked. "
    "When you complete an action, confirm it briefly ('I've opened VS Code'). "
    "When you can't do something, say so honestly and suggest an alternative. "
    "Remember what the user has asked for earlier in the conversation and use that context. "
    "If the user says 'it' or 'that', figure out what they mean from the conversation. "
    "Never sound robotic. Be warm, helpful, and human."
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
        # V2: Track the last executed action and result for context resolution
        self._last_action: str = ""
        self._last_result_message: str = ""

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
        Summary of the recent conversation for Planner context injection.

        Includes up to the last 8 turns with full text (up to 120 chars each).
        Used by the DecisionEngine for quick context checks.
        """
        if not self._turns:
            return ""

        recent = list(self._turns)[-8:]
        parts: list[str] = []
        for t in recent:
            prefix = "User" if t.role == TurnRole.USER else "Spidy"
            parts.append(f"{prefix}: {t.text[:120]}")
        return " | ".join(parts)

    def get_entity_history(self, max_turns: int = 10) -> list["ConversationTurn"]:
        """
        Return the most recent N turns that have an Intent attached.

        Used by ContextResolver to extract named entities for anaphora resolution.

        Parameters
        ----------
        max_turns:
            Maximum number of recent turns to return.

        Returns
        -------
        list[ConversationTurn]
            Turns with intents, ordered oldest-first.
        """
        turns_with_intent = [
            t for t in self._turns
            if t.intent is not None
        ]
        return turns_with_intent[-max_turns:]

    def record_action(self, action: str, result_message: str) -> None:
        """
        Record the last executed action and its result for context resolution.

        Called by the Brain after each ToolRouter result so the ContextResolver
        can track what "it" or "that" refers to in subsequent turns.

        Parameters
        ----------
        action:
            The skill action that was executed (e.g. ``"launch_app"``)
        result_message:
            The response message produced by the skill.
        """
        self._last_action = action
        self._last_result_message = result_message

    def get_last_action(self) -> tuple[str, str]:
        """
        Return the last executed action and result message.

        Returns
        -------
        tuple[str, str]
            ``(action, result_message)`` — both empty strings if no action yet.
        """
        return self._last_action, self._last_result_message

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
