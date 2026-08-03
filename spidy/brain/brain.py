"""
Brain — Central Cognitive Kernel
==================================
The Brain is the top-level orchestrator of Spidy's intelligence layer.

It coordinates the full pipeline:
    utterance
      → ContextResolver.resolve()        [V2: pronoun/reference resolution]
      → IntentClassifier.classify()
      → DecisionEngine.decide()
      → Planner.plan()
      → ToolRouter.execute()             [V2: retry, proactive, progress]
      → ResponseComposer.compose()       [V2: natural language responses]
      → response_text

Architecture (V2.0 — JARVIS Experience)
-----------------------------------------
    Brain
    ├── ContextResolver       resolve "it"/"that" → concrete entity [NEW V2]
    ├── IntentClassifier      classify raw utterance → Intent (compound-aware)
    ├── ConversationManager   rolling context window + session lifecycle
    ├── DecisionEngine        Intent + context → Decision (SKILL / LLM / CLARIFY)
    ├── Planner               Decision → Plan (multi-step capable) [UPGRADED V2]
    ├── ToolRouter            Plan → ToolResult(s) with retry + proactive [UPGRADED V2]
    ├── ResponseComposer      ToolResult(s) → natural language text [NEW V2]
    └── AutonomousAgent       Goal → Tasks → Execution loop [NEW M13]

Lifelong Companion extension points
------------------------------------
The Brain accepts three optional interface slots for future milestones:

    memory   (MemoryInterface)    — long-term episodic recall, M6
    knowledge (KnowledgeInterface) — document RAG, knowledge graph, M8+
    learning  (LearningInterface)  — preferences, habits, feedback, M9+

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

from spidy.brain.context_resolver import ContextResolver
from spidy.brain.conversation_manager import ConversationManager
from spidy.brain.decision_engine import DecisionEngine
from spidy.brain.intent_classifier import IntentClassifier
from spidy.brain.planner import Planner
from spidy.brain.response_composer import ResponseComposer
from spidy.brain.tool_router import ToolRouter
from spidy.brain.types import TurnRole
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.agent.agent import AutonomousAgent
    from spidy.brain.interfaces import KnowledgeInterface, LearningInterface, MemoryInterface, VisionInterface
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
        vision: "VisionInterface | None" = None,
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
        self._vision = vision

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

        # V2.0 new components
        self._context_resolver = ContextResolver()
        self._response_composer = ResponseComposer()

        # M13: Autonomous agent (lazily injected via attach_agent)
        self._agent: "AutonomousAgent | None" = None

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
            "Brain started | LLM={llm} | memory={mem} | knowledge={know} | learning={learn} | vision={vis}",
            llm="enabled" if self._llm else "disabled",
            mem="enabled" if self._memory else "disabled",
            know="enabled" if self._knowledge else "disabled",
            learn="enabled" if self._learning else "disabled",
            vis="enabled" if self._vision else "disabled",
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

        V2.0 pipeline:
        1. Context resolution ("it" → "VS Code")
        2. Intent classification (compound-aware)
        3. Decision engine
        4. Multi-step planning
        5. Memory recall
        6. Tool execution (with retry + proactive checks + progress events)
        7. Natural language response composition
        8. Memory storage

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
            BrainContextResolvedEvent,
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

        # ── Step 1: Context resolution ──────────────────────────────────────
        #
        # Resolve pronouns and references ("it", "that", "the file") against
        # the conversation history before anything else.
        entity_turns = self._conversation.get_entity_history(max_turns=10)
        resolution = self._context_resolver.resolve(utterance, entity_turns)
        resolved_utterance = resolution.utterance

        if resolution.was_resolved:
            log.debug(
                "ContextResolver: '{orig}' → '{res}'",
                orig=utterance[:60],
                res=resolved_utterance[:60],
            )
            await self._bus.publish(BrainContextResolvedEvent(
                session_id=sid,
                context_data={
                    "original": utterance,
                    "resolved": resolved_utterance,
                    "refs": [
                        {"original": r.original, "resolved": r.resolved}
                        for r in resolution.resolved_refs
                    ],
                },
            ))

        # ── Step 2: Classify intent first ───────────────────────────────────
        #
        # We classify on the RESOLVED utterance but store the ORIGINAL as the
        # conversation text (so conversation history reads naturally).
        intent = await self._classifier.classify(resolved_utterance)

        # ── Step 3: Add user turn with intent attached ───────────────────────
        #
        # Add once with intent so ContextResolver can see entities in future turns.
        self._conversation.add_turn(TurnRole.USER, utterance, intent=intent)

        # ── Step 4: Decide ──────────────────────────────────────────────────
        decision = await self._decision_engine.decide(intent, session_id=sid)

        # ── Step 6: Recall relevant memories ───────────────────────────────
        memory_context = ""
        if self._memory is not None:
            try:
                recalled = await self._memory.recall(utterance, session_id=sid, limit=3)
                if recalled:
                    memory_context = "\n".join(
                        f"[Memory] {m.get('content', '')[:200]}" for m in recalled
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("Brain: memory recall failed (non-fatal): {exc}", exc=exc)

        # ── Step 7: Plan ────────────────────────────────────────────────────
        plan = await self._planner.plan(
            decision=decision,
            session_id=sid,
            conversation_context=(
                self._conversation.get_summary()
                + (f"\n\nRelevant memories:\n{memory_context}" if memory_context else "")
            ),
        )

        # ── Step 8: Build LLM messages ──────────────────────────────────────
        llm_messages = self._conversation.get_llm_messages(include_system=True)
        from spidy.llm.client import LLMMessage
        typed_messages = [
            LLMMessage(role=m["role"], content=m["content"])
            for m in llm_messages
        ]

        # ── Step 9: Execute plan ────────────────────────────────────────────
        results = await self._router.execute(
            plan=plan,
            session_id=sid,
            llm_messages=typed_messages,
            user_name=self._user_name,
        )

        # ── Step 10: Compose response ───────────────────────────────────────
        #
        # V2: ResponseComposer handles multi-step aggregation + natural phrasing.
        primary_action = plan.first.action if plan.first else ""
        response_text = self._response_composer.compose(results, action=primary_action)

        # ── Step 11: Record action in conversation ──────────────────────────
        if results:
            best = next((r for r in results if r.success), results[0])
            self._conversation.record_action(best.action, best.message)

        # ── Step 12: Add assistant turn ─────────────────────────────────────
        if response_text:
            self._conversation.add_turn(TurnRole.ASSISTANT, response_text)

        # ── Step 13: Store interaction in memory ────────────────────────────
        if self._memory is not None and response_text:
            try:
                await self._memory.store_interaction(
                    utterance=utterance,
                    response=response_text,
                    session_id=sid,
                    intent=intent.category.value if hasattr(intent, "category") and hasattr(intent.category, "value") else str(getattr(intent, "category", "")),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Brain: memory store failed (non-fatal): {exc}", exc=exc)

        # ── Step 14: Publish response event ─────────────────────────────────
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

    @property
    def vision(self) -> "VisionInterface | None":
        """The connected vision interface (None until M9)."""
        return self._vision

    # ── M13: Autonomous Agent ──────────────────────────────────────────────

    def attach_agent(self, agent: "AutonomousAgent") -> None:
        """
        Attach an AutonomousAgent to this Brain instance.

        Called by SpidyCore (or tests) after both Brain and AutonomousAgent
        are constructed. This avoids circular imports at construction time.

        Parameters
        ----------
        agent:
            A fully-constructed AutonomousAgent that uses this Brain.
        """
        self._agent = agent
        log.debug("Brain: AutonomousAgent attached.")

    async def run_goal(self, goal: str, session_id: str | None = None) -> str:
        """
        Execute a high-level goal autonomously (Milestone 13).

        Delegates to the attached AutonomousAgent. If no agent is attached,
        falls back to Brain.process() for simple single-step goals.

        Parameters
        ----------
        goal:
            The user's high-level goal description
            (e.g. "Create a Flask project").
        session_id:
            Optional override for the session ID.

        Returns
        -------
        str
            Natural language response describing what was accomplished.
        """
        if self._agent is not None:
            return await self._agent.run_goal(goal, session_id=session_id)

        # Graceful fallback when no agent is attached:
        # treat the goal as a single Brain.process() call
        log.debug(
            "Brain.run_goal: no AutonomousAgent attached — falling back to process()"
        )
        return await self.process(goal, session_id=session_id)

    async def cancel_goal(self) -> bool:
        """
        Cancel the currently running autonomous goal.

        Returns
        -------
        bool
            True if a goal was cancelled, False if none was running.
        """
        if self._agent is not None:
            return await self._agent.cancel_current_goal()
        return False

    @property
    def autonomous_agent(self) -> "AutonomousAgent | None":
        """The attached AutonomousAgent (None until attach_agent() is called)."""
        return self._agent

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _compose_response(results: list) -> str:
        """
        Legacy single-result composer (kept for backward compatibility).

        V2 uses ResponseComposer.compose() directly in process().
        This method is retained so existing tests that call it still pass.
        """
        for result in results:
            if result.message:
                return result.message
        return ""
