"""
Brain — Central Cognitive Kernel
==================================
The Brain is the top-level orchestrator of Spidy's intelligence layer.

It coordinates the full pipeline:
    utterance
      → IntentClassifier.classify()
      → DecisionEngine.decide()
      → Planner.plan()
      → ToolRouter.execute()
      → response_text

Architecture (Milestone 3)
--------------------------
    Brain
    ├── IntentClassifier      classify raw utterance → Intent
    ├── ConversationManager   rolling context window + session lifecycle
    ├── DecisionEngine        Intent + context → Decision (SKILL / LLM / CLARIFY)
    ├── Planner               Decision → Plan (ordered PlanSteps)
    └── ToolRouter            Plan → ToolResult(s) via Skill or LLM

Lifelong Companion extension points
------------------------------------
The Brain accepts three optional interface slots for future milestones:

    memory   (MemoryInterface)    — long-term episodic recall, M6
    knowledge (KnowledgeInterface) — document RAG, knowledge graph, M8+
    learning  (LearningInterface)  — preferences, habits, feedback, M9+

When these are None (the default), the Brain works purely from the
rolling conversation window. No features are broken — companion
capabilities simply don't activate.

Event subscription
------------------
Brain.start() subscribes to "voice.user_spoke" events on the EventBus.
When the VoiceEngine detects a command, it publishes that event; the Brain
processes it and publishes "brain.response_ready" for TTS + UI.

Usage
-----
    brain = Brain(
        bus=bus,
        config=settings.reasoning,
        skill_registry=registry,
        llm_client=LLMClientFactory.build(settings.reasoning),
    )
    await brain.start()
    # Brain is now listening on EventBus
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from spidy.brain.conversation_manager import ConversationManager
from spidy.brain.decision_engine import DecisionEngine
from spidy.brain.intent_classifier import IntentClassifier
from spidy.brain.planner import Planner
from spidy.brain.tool_router import ToolRouter
from spidy.brain.types import TurnRole
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.brain.interfaces import KnowledgeInterface, LearningInterface, MemoryInterface
    from spidy.config.manager import BrainConfig, ReasoningConfig
    from spidy.core.event_bus import EventBus
    from spidy.llm.client import BaseLLMClient
    from spidy.skills.registry import SkillRegistry

log = get_logger(__name__)

_UNKNOWN_SESSION = "default"
_SILENCE_RESPONSE = ""          # Empty → no TTS output for noop steps


class Brain:
    """
    Central cognitive kernel of Spidy.

    Parameters
    ----------
    bus:
        Application EventBus. Brain subscribes to voice events here.
    config:
        Reasoning / Brain configuration (from SpidyConfig.reasoning or .brain).
    skill_registry:
        All registered Skills. Passed to DecisionEngine + ToolRouter.
    llm_client:
        LLM backend for open-ended answers. May be None (LLM disabled).
    memory:
        Optional MemoryInterface (Milestone 6). None = short context only.
    knowledge:
        Optional KnowledgeInterface (Milestone 8+). None = no RAG.
    learning:
        Optional LearningInterface (Milestone 9+). None = no personalisation.
    user_name:
        The user's name for personalised responses.
    """

    def __init__(
        self,
        bus: "EventBus",
        config: "ReasoningConfig",
        skill_registry: "SkillRegistry",
        llm_client: "BaseLLMClient | None" = None,
        memory: "MemoryInterface | None" = None,
        knowledge: "KnowledgeInterface | None" = None,
        learning: "LearningInterface | None" = None,
        user_name: str = "User",
    ) -> None:
        self._bus = bus
        self._config = config
        self._registry = skill_registry
        self._llm = llm_client
        self._user_name = user_name

        # Companion interface slots (filled by future milestones)
        self._memory = memory
        self._knowledge = knowledge
        self._learning = learning

        # Internal components
        self._classifier = IntentClassifier()
        self._conversation = ConversationManager(bus=bus)
        self._decision_engine = DecisionEngine(
            skill_registry=skill_registry,
            memory=memory,
        )
        self._planner = Planner(knowledge=knowledge)
        self._router = ToolRouter(
            bus=bus,
            skill_registry=skill_registry,
            llm_client=llm_client,
            learning=learning,
        )

        self._running = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the Brain and subscribe to EventBus events."""
        if self._running:
            return

        self._running = True

        # Start a default session
        await self._conversation.start_session(session_id=_UNKNOWN_SESSION)

        # Subscribe to VoiceEngine utterance events
        self._bus.subscribe("voice.user_spoke", self._on_user_spoke)

        log.info(
            "Brain started | LLM={llm} | memory={mem} | knowledge={know} | learning={learn}",
            llm="enabled" if self._llm else "disabled",
            mem="enabled" if self._memory else "disabled",
            know="enabled" if self._knowledge else "disabled",
            learn="enabled" if self._learning else "disabled",
        )

    async def stop(self) -> None:
        """Stop the Brain, unsubscribe, and end the session."""
        if not self._running:
            return

        self._running = False
        self._bus.unsubscribe("voice.user_spoke", self._on_user_spoke)
        await self._conversation.end_session()
        log.info("Brain stopped.")

    # ── Primary processing method ──────────────────────────────────────────

    async def process(
        self,
        utterance: str,
        session_id: str | None = None,
    ) -> str:
        """
        Process a user utterance through the full Brain pipeline.

        Can be called directly (e.g. from tests or API) or triggered
        via the EventBus subscription on "voice.user_spoke".

        Parameters
        ----------
        utterance:
            The user's raw text input.
        session_id:
            Override session ID. Defaults to the current active session.

        Returns
        -------
        str
            The response text to be spoken (TTS) and displayed (UI).
            Empty string if there is nothing to say (e.g. noop).
        """
        from spidy.brain.events import (
            BrainProcessingStartedEvent,
            BrainResponseReadyEvent,
        )

        sid = session_id or self._conversation.session_id

        log.info(
            "Processing utterance: '{utt}' | session={sid}",
            utt=utterance[:80],
            sid=sid,
        )

        await self._bus.publish(BrainProcessingStartedEvent(
            session_id=sid,
            utterance=utterance,
        ))

        # 1. Add user turn to conversation window
        self._conversation.add_turn(TurnRole.USER, utterance)

        # 2. Classify intent
        intent = await self._classifier.classify(utterance)

        # 3. Record intent on the user turn (for memory indexing later)
        # Note: ConversationTurn is frozen — the intent was passed at add_turn.
        # For M3 we re-add the turn with intent; for M6 memory stores the turn.

        # 4. Decide
        decision = await self._decision_engine.decide(intent, session_id=sid)

        # 5. Plan
        plan = await self._planner.plan(
            decision=decision,
            session_id=sid,
            conversation_context=self._conversation.get_summary(),
        )

        # 6. Build LLM messages from conversation history
        llm_messages = self._conversation.get_llm_messages(include_system=True)
        # Import here to avoid circular at module level
        from spidy.llm.client import LLMMessage
        typed_messages = [
            LLMMessage(role=m["role"], content=m["content"])
            for m in llm_messages
        ]

        # 7. Execute plan
        results = await self._router.execute(
            plan=plan,
            session_id=sid,
            llm_messages=typed_messages,
            user_name=self._user_name,
        )

        # 8. Compose response text from results
        response_text = self._compose_response(results)

        # 9. Add assistant turn to conversation window
        if response_text:
            self._conversation.add_turn(TurnRole.ASSISTANT, response_text)

        # 10. Publish response event
        await self._bus.publish(BrainResponseReadyEvent(
            session_id=sid,
            response_text=response_text,
            decision_mode=decision.mode.value,
        ))

        log.info(
            "Response ready [{mode}]: '{resp}'",
            mode=decision.mode.value,
            resp=response_text[:80],
        )

        return response_text

    # ── EventBus handler ──────────────────────────────────────────────────

    async def _on_user_spoke(self, event: Any) -> None:
        """Handle voice.user_spoke events from the VoiceEngine."""
        utterance: str = getattr(event, "text", "") or getattr(event, "utterance", "")
        if utterance:
            await self.process(utterance)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def session_id(self) -> str:
        return self._conversation.session_id

    @property
    def turn_count(self) -> int:
        return self._conversation.turn_count

    @property
    def memory(self) -> "MemoryInterface | None":
        """The connected memory interface (None until M6)."""
        return self._memory

    @property
    def knowledge(self) -> "KnowledgeInterface | None":
        """The connected knowledge interface (None until M8+)."""
        return self._knowledge

    @property
    def learning(self) -> "LearningInterface | None":
        """The connected learning interface (None until M9+)."""
        return self._learning

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _compose_response(results: list) -> str:
        """
        Compose the final response text from a list of ToolResults.

        For M3 (single-step plans) this is just the first non-empty result.
        For future multi-step plans, this will aggregate and synthesise.
        """
        for result in results:
            if result.message:
                return result.message
        return ""
