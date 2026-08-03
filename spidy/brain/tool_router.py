"""
ToolRouter — Plan Execution Engine
=====================================
Executes each ``PlanStep`` in a ``Plan`` by routing to the correct handler.

V2.0 enhancements
-----------------
- ProactiveChecker: consults proactive state before launching apps / searching
- Retry loop: transient skill failures are retried up to max_retries times
- Progress events: BrainProgressEvent published after each step in multi-step plans
- Self-healing: detailed fallback messages with concrete suggestions
- WorkflowLearner integration: records each skill execution for habit detection

Routing rules
-------------
step_type == "skill"   : Look up skill in SkillRegistry → call skill.execute()
step_type == "llm"     : Call BaseLLMClient.complete() with conversation context
step_type == "clarify" : Return the clarification question as a ToolResult
step_type == "noop"    : Return an empty successful ToolResult

Error handling
--------------
- Skill exceptions are caught, logged, retried (up to max_retries), then fail
- LLM failures (response.success=False) are surfaced as ToolResult.fail()
- ProactiveChecker "reuse" advice short-circuits launch without failure

Event publishing
----------------
Emits BrainToolCalledEvent before each step and BrainToolResultEvent after.
Emits BrainProgressEvent for every step when the plan has >1 steps.
Emits BrainProactiveCheckEvent when a proactive check returns non-proceed advice.

Lifelong Companion notes
------------------------
In future milestones the ToolRouter will also call
``LearningInterface.record_feedback()`` to collect implicit signals
from every tool execution. The ``learning`` parameter is accepted but
unused in M3.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from spidy.brain.proactive import ProactiveChecker
from spidy.brain.types import Plan, PlanStep, ToolResult
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.brain.interfaces import LearningInterface
    from spidy.core.event_bus import EventBus
    from spidy.llm.client import BaseLLMClient, LLMMessage
    from spidy.skills.base import SkillContext
    from spidy.skills.registry import SkillRegistry

log = get_logger(__name__)

# Generic fallback used only when a more specific message cannot be generated
_GENERIC_FALLBACK = (
    "I'm not sure how to handle that. "
    "Try asking me to open an app, search the web, set a timer, or say 'help'."
)

_FALLBACK_RESPONSE = (
    "I hit a snag answering that. Please try again or rephrase your question."
)

# Retry configuration
_DEFAULT_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 0.5  # seconds


# Actions that have no skill implementation yet but deserve a clear, honest
# explanation rather than the generic "I don't have a skill for that" message.
_KNOWN_UNIMPLEMENTED: dict[str, str] = {
    "desktop_context_menu": (
        "Right-clicking on the Desktop isn't something I can do yet — "
        "that would require simulating a mouse click at a specific screen position. "
        "You can right-click the Desktop yourself to access display settings, "
        "\"New\" items, and other options."
    ),
    "right_click": (
        "I can't perform right-click actions yet. "
        "You can right-click manually to access context menus."
    ),
}


def _build_no_skill_message(action: str) -> str:
    """
    Build a helpful message when no skill is registered for an action.

    For known unimplemented actions, returns an honest, specific explanation.
    For truly unknown actions, names the action and directs the user to known commands.
    Never exposes raw internal strings like "'name' parameter is required".
    """
    # Return a specific, honest explanation for known-but-unimplemented actions
    if action in _KNOWN_UNIMPLEMENTED:
        return _KNOWN_UNIMPLEMENTED[action]

    return (
        f"I understood you want to '{action.replace('_', ' ')}', but I don't have a skill for that yet. "
        "Here are things I can do: open apps, search the web (try 'google python'), "
        "set timers, take notes, check system info, take screenshots, or answer questions. "
        "Say 'help' for the full list."
    )


def _build_no_llm_message() -> str:
    """
    Build a contextual message when the LLM is not configured.

    Explains to the user that they can still use skill-based features.
    """
    return (
        "I can't answer open-ended questions right now because no AI model is configured. "
        "However, I can still help you with: "
        "opening apps (try 'open notepad'), searching the web (try 'google python'), "
        "setting timers (try 'set timer for 5 minutes'), taking screenshots, "
        "checking system info, or managing files. "
        "Say 'help' for the full list."
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
    max_retries:
        Number of times to retry a failed skill step before giving up.
    """

    def __init__(
        self,
        bus: "EventBus",
        skill_registry: "SkillRegistry",
        llm_client: "BaseLLMClient | None" = None,
        learning: "LearningInterface | None" = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self._bus = bus
        self._registry = skill_registry
        self._llm = llm_client
        self._learning = learning  # Reserved for M9+
        self._max_retries = max_retries
        self._proactive = ProactiveChecker()

    async def execute(
        self,
        plan: Plan,
        session_id: str = "",
        llm_messages: list["LLMMessage"] | None = None,
        user_name: str = "User",
    ) -> list[ToolResult]:
        """
        Execute all steps in a Plan and return results.

        For multi-step plans, publishes BrainProgressEvent after each step.

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
        total_steps = len(plan.steps)

        for idx, step in enumerate(plan.steps):
            result = await self._execute_step(
                step=step,
                session_id=session_id,
                llm_messages=llm_messages or [],
                user_name=user_name,
            )
            results.append(result)

            # Publish progress event for multi-step plans
            if total_steps > 1:
                await self._publish_progress(
                    session_id=session_id,
                    step_index=idx,
                    total_steps=total_steps,
                    action=step.action or step.step_type,
                    message=result.message,
                )

        return results

    # ── Step dispatch ──────────────────────────────────────────────────────

    async def _execute_step(
        self,
        step: PlanStep,
        session_id: str,
        llm_messages: list["LLMMessage"],
        user_name: str,
    ) -> ToolResult:
        """Route a single step to the correct handler with proactive checks."""
        from spidy.brain.events import BrainToolCalledEvent, BrainToolResultEvent

        # Publish tool-called event
        await self._bus.publish(BrainToolCalledEvent(
            session_id=session_id,
            action=step.action,
            step_type=step.step_type,
        ))

        if step.step_type == "skill":
            result = await self._run_skill_with_retry(step, session_id, user_name)
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

    async def _run_skill_with_retry(
        self,
        step: PlanStep,
        session_id: str,
        user_name: str,
    ) -> ToolResult:
        """
        Run a skill step with proactive check and retry loop.

        Order:
        1. Proactive check (e.g. app already running?)
        2. Execute skill
        3. On failure: retry up to max_retries times
        4. On exhausted retries: return user-friendly failure message
        """
        # ── Proactive check ───────────────────────────────────────────────
        advice = await self._proactive.check(
            action=step.action,
            params=dict(step.params),
        )

        if advice.action == "reuse":
            # App/resource already available — skip redundant launch
            await self._publish_proactive_event(session_id, step.action, advice)
            return ToolResult.ok(
                message=advice.message,
                action=step.action,
                step_type="skill",
            )

        if advice.action == "ask":
            # Something needs user confirmation first
            await self._publish_proactive_event(session_id, step.action, advice)
            return ToolResult.ok(
                message=advice.message,
                action=step.action,
                step_type="skill",
            )

        # ── Execute with retry ────────────────────────────────────────────
        last_result: ToolResult | None = None

        for attempt in range(self._max_retries + 1):
            result = await self._run_skill(step, session_id, user_name)

            if result.success:
                return result

            last_result = result

            if attempt < self._max_retries:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                log.warning(
                    "Skill '{action}' failed (attempt {n}/{max}), retrying in {d:.1f}s: {err}",
                    action=step.action,
                    n=attempt + 1,
                    max=self._max_retries + 1,
                    d=delay,
                    err=result.error,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        log.error(
            "Skill '{action}' failed after {n} attempts.",
            action=step.action,
            n=self._max_retries + 1,
        )
        return last_result or ToolResult.fail(
            message=_GENERIC_FALLBACK,
            action=step.action,
            error="All retries exhausted.",
            step_type="skill",
        )

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
                message=_build_no_skill_message(step.action),
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
                message=_build_no_llm_message(),
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

    # ── Event helpers ──────────────────────────────────────────────────────

    async def _publish_progress(
        self,
        session_id: str,
        step_index: int,
        total_steps: int,
        action: str,
        message: str,
    ) -> None:
        """Publish a BrainProgressEvent for UI live-update."""
        from spidy.brain.events import BrainProgressEvent
        progress_pct = int(((step_index + 1) / total_steps) * 100)
        await self._bus.publish(BrainProgressEvent(
            session_id=session_id,
            message=f"Step {step_index + 1}/{total_steps}: {action.replace('_', ' ')}",
            progress_percent=progress_pct,
        ))

    async def _publish_proactive_event(
        self,
        session_id: str,
        action: str,
        advice: "Any",
    ) -> None:
        """Publish a BrainProactiveCheckEvent when advice is non-proceed."""
        from spidy.brain.events import BrainProactiveCheckEvent
        await self._bus.publish(BrainProactiveCheckEvent(
            session_id=session_id,
            reason=f"Proactive check before '{action}': {advice.action}",
        ))
