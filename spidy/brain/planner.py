"""
Planner — Decision → Plan Conversion
======================================
Converts a ``Decision`` into an executable ``Plan``.

The Plan is an ordered list of ``PlanStep`` objects.
In Milestone 3, plans are always single-step (one action per turn).
Future milestones will enable multi-step planning (e.g. "search + summarise + email").

Plan step types
---------------
``skill``   : Dispatch to the SkillRegistry for execution
``llm``     : Call the LLM directly with conversation context
``clarify`` : Return a clarification question to the user
``noop``    : Do nothing (rejected / unsupported intent)

Param extraction
----------------
For SKILL decisions, the Planner extracts parameters from the Intent's
entities and maps them to the skill's expected parameter schema.

Currently this is a simple entity-to-param mapping (entity.name → param key).
Future versions will use the skill's ``ParamSchema`` declarations for
typed extraction and validation.

Lifelong Companion notes
------------------------
Future versions will call ``KnowledgeInterface.search_rag()`` to inject
retrieved context into the ``prompt_context`` of LLM steps. The
``knowledge`` parameter is accepted but unused in M3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spidy.brain.types import Decision, DecisionMode, Plan, PlanStep
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.brain.interfaces import KnowledgeInterface

log = get_logger(__name__)


class Planner:
    """
    Converts a Decision into an executable Plan.

    Parameters
    ----------
    knowledge:
        Optional knowledge interface for RAG context injection
        (unused in M3; accepted for forward compatibility).
    """

    def __init__(
        self,
        knowledge: "KnowledgeInterface | None" = None,
    ) -> None:
        self._knowledge = knowledge  # Unused in M3 — reserved for M8+

    async def plan(
        self,
        decision: Decision,
        session_id: str = "",
        conversation_context: str = "",
    ) -> Plan:
        """
        Generate a Plan from a Decision.

        Parameters
        ----------
        decision:
            The Decision produced by the DecisionEngine.
        session_id:
            Current session ID (attached to the Plan for tracing).
        conversation_context:
            Optional summary of recent conversation turns
            (injected into LLM step prompt context).

        Returns
        -------
        Plan
            An ordered sequence of PlanStep objects to execute.
        """
        step = self._make_step(decision, conversation_context)

        log.debug(
            "Plan created: [{step_type}] action='{action}' | decision={mode}",
            step_type=step.step_type,
            action=step.action or "(none)",
            mode=decision.mode.value,
        )

        return Plan(
            steps=(step,),
            session_id=session_id,
            decision=decision,
        )

    # ── Internal ──────────────────────────────────────────────────────────

    def _make_step(self, decision: Decision, conversation_context: str) -> PlanStep:
        """Map a Decision mode to a PlanStep."""

        if decision.mode == DecisionMode.SKILL:
            return self._skill_step(decision)

        if decision.mode == DecisionMode.LLM_DIRECT:
            return self._llm_step(decision, conversation_context)

        if decision.mode == DecisionMode.CLARIFY:
            return PlanStep(
                step_type="clarify",
                action="clarify",
                prompt_context=decision.clarification_question,
            )

        # REJECT or unknown
        return PlanStep(step_type="noop")

    def _skill_step(self, decision: Decision) -> PlanStep:
        """Build a skill PlanStep, mapping intent entities to params."""
        intent = decision.intent
        params: dict = {
            entity.name: entity.value
            for entity in intent.entities
        }
        return PlanStep(
            step_type="skill",
            action=intent.action,
            params=params,
        )

    def _llm_step(self, decision: Decision, context: str) -> PlanStep:
        """Build an LLM PlanStep with conversation context."""
        return PlanStep(
            step_type="llm",
            action="direct",
            prompt_context=context,
        )
