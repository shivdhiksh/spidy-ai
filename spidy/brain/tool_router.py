"""
ToolRouter — Plan Execution Engine
=====================================
Executes each ``PlanStep`` in a ``Plan`` by routing to the correct handler.

Routing rules
-------------
step_type == "skill"   : Look up skill in SkillRegistry → call skill.execute()
step_type == "llm"     : Call BaseLLMClient.complete() with conversation context
step_type == "clarify" : Return the clarification question as a ToolResult
step_type == "noop"    : Return an empty successful ToolResult

Error handling
--------------
- Skill exceptions are caught and returned as ToolResult.fail()
- LLM failures (response.success=False) are surfaced as ToolResult.fail()
- Neither case propagates an exception to the Brain

Event publishing
----------------
Emits BrainToolCalledEvent before each step and BrainToolResultEvent after.

Lifelong Companion notes
------------------------
In future milestones the ToolRouter will also call
``LearningInterface.record_feedback()`` to collect implicit signals
from every tool execution. The ``learning`` parameter is accepted but
unused in M3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spidy.brain.types import Plan, PlanStep, ToolResult
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.brain.interfaces import LearningInterface
    from spidy.core.event_bus import EventBus
    from spidy.llm.client import BaseLLMClient, LLMMessage
    from spidy.skills.base import SkillContext
    from spidy.skills.registry import SkillRegistry

log = get_logger(__name__)

_FALLBACK_RESPONSE = (
    "I'm not able to help with that right now. "
    "Could you try rephrasing, or ask me something else?"
)


class ToolRouter:
    """
    Executes a Plan by routing each step to the correct handler.

    Parameters
    ----------
    bus:
        Application EventBus for publishing tool events.
    skill_registry:
        Registry of all loaded Skills.
    llm_client:
        LLM backend for direct-answer steps. May be None (LLM disabled).
    learning:
        Optional learning interface for feedback collection
        (unused in M3; accepted for forward compatibility).
    """

    def __init__(
        self,
        bus: "EventBus",
        skill_registry: "SkillRegistry",
        llm_client: "BaseLLMClient | None" = None,
        learning: "LearningInterface | None" = None,
    ) -> None:
        self._bus = bus
        self._registry = skill_registry
        self._llm = llm_client
        self._learning = learning  # Reserved for M9+

    async def execute(
        self,
        plan: Plan,
        session_id: str = "",
        llm_messages: list["LLMMessage"] | None = None,
        user_name: str = "User",
    ) -> list[ToolResult]:
        """
        Execute all steps in a Plan and return results.

        Parameters
        ----------
        plan:
            The Plan to execute.
        session_id:
            Session ID for event publishing.
        llm_messages:
            Full conversation history in LLM format (used by llm steps).
        user_name:
            For personalisation in skill contexts.

        Returns
        -------
        list[ToolResult]
            One ToolResult per PlanStep.
        """
        results: list[ToolResult] = []

        for step in plan.steps:
            result = await self._execute_step(
                step=step,
                session_id=session_id,
                llm_messages=llm_messages or [],
                user_name=user_name,
            )
            results.append(result)

        return results

    # ── Step dispatch ──────────────────────────────────────────────────────

    async def _execute_step(
        self,
        step: PlanStep,
        session_id: str,
        llm_messages: list["LLMMessage"],
        user_name: str,
    ) -> ToolResult:
        """Route a single step to the correct handler."""
        from spidy.brain.events import BrainToolCalledEvent, BrainToolResultEvent

        # Publish tool-called event
        await self._bus.publish(BrainToolCalledEvent(
            session_id=session_id,
            action=step.action,
            step_type=step.step_type,
        ))

        if step.step_type == "skill":
            result = await self._run_skill(step, session_id, user_name)
        elif step.step_type == "llm":
            result = await self._run_llm(step, llm_messages)
        elif step.step_type == "clarify":
            result = ToolResult.ok(
                message=step.prompt_context or "Could you clarify what you mean?",
                action="clarify",
                step_type="clarify",
            )
        else:  # noop
            result = ToolResult.ok(
                message="",
                action="noop",
                step_type="noop",
            )

        # Publish tool-result event
        await self._bus.publish(BrainToolResultEvent(
            session_id=session_id,
            action=result.action,
            success=result.success,
            message=result.message,
        ))

        return result

    async def _run_skill(
        self,
        step: PlanStep,
        session_id: str,
        user_name: str,
    ) -> ToolResult:
        """Dispatch a skill step to the SkillRegistry."""
        from spidy.skills.base import SkillContext

        skill = self._registry.find_skill_for_action(step.action)
        if skill is None:
            return ToolResult.fail(
                message=_FALLBACK_RESPONSE,
                action=step.action,
                error=f"No skill registered for action '{step.action}'.",
                step_type="skill",
            )

        ctx = SkillContext(
            action=step.action,
            params=dict(step.params),
            session_id=session_id,
            user_name=user_name,
        )

        try:
            skill_result = await skill.execute(step.action, ctx)
            log.debug(
                "Skill '{name}' executed '{action}' → success={ok}",
                name=skill.name,
                action=step.action,
                ok=skill_result.success,
            )
            return ToolResult(
                success=skill_result.success,
                message=skill_result.message,
                action=step.action,
                data=skill_result.data,
                error=str(skill_result.error) if skill_result.error else "",
                step_type="skill",
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Skill '{name}' raised unexpectedly: {exc}",
                name=skill.name,
                exc=exc,
            )
            return ToolResult.fail(
                message="Something went wrong while running that skill.",
                action=step.action,
                error=str(exc),
                step_type="skill",
            )

    async def _run_llm(
        self,
        step: PlanStep,
        llm_messages: list["LLMMessage"],
    ) -> ToolResult:
        """Call the LLM with the current conversation context."""
        if self._llm is None:
            return ToolResult.ok(
                message=_FALLBACK_RESPONSE,
                action="llm",
                step_type="llm",
            )

        try:
            response = await self._llm.complete(llm_messages)
            if not response.success:
                log.warning(
                    "LLM call failed: {err}",
                    err=response.error_message,
                )
                return ToolResult.ok(
                    message=_FALLBACK_RESPONSE,
                    action="llm",
                    step_type="llm",
                )
            return ToolResult.ok(
                message=response.text.strip(),
                action="llm",
                data={"model": response.model, "usage": response.usage},
                step_type="llm",
            )
        except Exception as exc:  # noqa: BLE001
            log.error("ToolRouter LLM error: {exc}", exc=exc)
            return ToolResult.ok(
                message=_FALLBACK_RESPONSE,
                action="llm",
                step_type="llm",
            )
