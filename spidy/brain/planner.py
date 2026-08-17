"""
Planner — Decision → Plan Conversion
======================================
Converts a ``Decision`` into an executable ``Plan``.

The Plan is an ordered list of ``PlanStep`` objects.

V2.0: Multi-step plans
-----------------------
The Planner now supports compound intents (action="compound"), producing
ordered Plans with multiple PlanSteps. This enables:
  - "Create a folder, open it in VS Code, create main.py" → 3 steps
  - "Open Edge and search for Python, then open Notepad" → 2 steps

For single intents, behaviour is unchanged (one-step Plan).

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

Lifelong Companion notes
------------------------
Future versions will call ``KnowledgeInterface.search_rag()`` to inject
retrieved context into the ``prompt_context`` of LLM steps. The
``knowledge`` parameter is accepted but unused in M3.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from spidy.brain.types import Decision, DecisionMode, Entity, Intent, Plan, PlanStep
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.brain.interfaces import KnowledgeInterface

log = get_logger(__name__)

# ─── Intent → Skill Action Alias Table ───────────────────────────────────────
#
# Maps intent action names (from IntentClassifier) to the canonical action names
# registered by skills (in SkillRegistry). This decouples the user-facing intent
# vocabulary from the internal skill API.
#
# Format: intent_action → skill_action
#
_ACTION_ALIASES: dict[str, str] = {
    # SystemSkill stubs removed → real SystemControlSkill actions
    "lock_screen":   "lock_workstation",      # lock_screen intent → real lock
    "sleep_system":  "sleep_system",          # already correct
    "shutdown":      "shutdown_system",       # old shutdown intent → real shutdown
    "restart":       "restart_system",        # old restart intent → real restart

    # Web search — default engine is Google via BrowserSkill
    "search_web":    "search_google",         # generic web search → Google

    # Calculator synonyms
    "math":          "calculate",             # "do the math" → calculate skill

    # M13.2 — Desktop / OS alias fixes
    # "show desktop" and "go to desktop" → SystemControlSkill.show_desktop
    "show_desktop":  "show_desktop",
    # "right click on desktop" → no skill exists; ToolRouter will give honest message
    "desktop_context_menu": "desktop_context_menu",
    # "open terminal" intent → launch_app (entity extractor sets name=terminal)
    "open_terminal": "launch_app",

    # M13.3 — Window control (BUG 3 FIX)
    # minimize_window and maximize_window route to AppSkill
    "minimize_window": "minimize_window",
    "maximize_window": "maximize_window",
}



class Planner:
    """
    Converts a Decision into an executable Plan.

    V2.0: Supports multi-step Plans from compound intents.

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

        For compound intents (action="compound"), produces a multi-step Plan.
        For all other intents, produces a single-step Plan (unchanged behaviour).

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
        # Multi-step: compound intent → multiple PlanSteps
        if decision.intent.action == "compound":
            steps = await self._plan_compound(decision, conversation_context)
            log.debug(
                "Compound Plan created: {n} steps | session={sid}",
                n=len(steps),
                sid=session_id,
            )
        else:
            step = self._make_step(decision, conversation_context)
            steps = (step,)
            log.debug(
                "Plan created: [{step_type}] action='{action}' | decision={mode}",
                step_type=step.step_type,
                action=step.action or "(none)",
                mode=decision.mode.value,
            )

        return Plan(
            steps=steps,
            session_id=session_id,
            decision=decision,
        )

    # ── Compound planning ──────────────────────────────────────────────────

    async def _plan_compound(
        self,
        decision: Decision,
        conversation_context: str,
    ) -> tuple[PlanStep, ...]:
        """
        Build a multi-step Plan from a compound intent.

        Parses the sub-intents encoded in the compound Intent's entities,
        then maps each sub-intent to a PlanStep.
        """
        steps: list[PlanStep] = []

        # Decode ordered sub-intents from entities (step_0, step_1, ...)
        sub_intents_data: list[dict] = []
        for entity in decision.intent.entities:
            if entity.name.startswith("step_"):
                try:
                    data = json.loads(entity.value)
                    sub_intents_data.append(data)
                except (json.JSONDecodeError, ValueError):
                    log.warning(
                        "Planner: failed to parse compound step entity: {val}",
                        val=entity.value[:80],
                    )

        if not sub_intents_data:
            # Fallback: treat as LLM step
            return (self._llm_step(decision, conversation_context),)

        for i, step_data in enumerate(sub_intents_data):
            sub_action = step_data.get("action", "chat")
            sub_entities_raw = step_data.get("entities", [])
            sub_confidence = step_data.get("confidence", 0.75)
            sub_utterance = step_data.get("utterance", "")

            # Reconstruct a minimal Intent for the sub-step
            sub_entities = tuple(
                Entity(name=e["name"], value=e["value"])
                for e in sub_entities_raw
                if isinstance(e, dict)
            )
            sub_intent = Intent(
                action=sub_action,
                entities=sub_entities,
                confidence=sub_confidence,
                raw_utterance=sub_utterance,
                source="compound_sub",
            )

            # Build the step
            step = self._make_step_from_intent(
                sub_intent,
                conversation_context=conversation_context,
            )
            steps.append(step)
            log.debug(
                "  Compound step {i}: [{type}] action='{action}'",
                i=i,
                type=step.step_type,
                action=step.action or "(none)",
            )

        return tuple(steps)

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

    def _make_step_from_intent(
        self,
        intent: Intent,
        conversation_context: str = "",
    ) -> PlanStep:
        """
        Build a PlanStep directly from a sub-Intent (used in compound plans).

        Mirrors _make_step() but takes an Intent directly instead of a Decision.
        """
        from spidy.brain.types import DecisionMode

        # Skill step: map action to canonical name
        action = _ACTION_ALIASES.get(intent.action, intent.action)

        if intent.action in ("chat", "llm"):
            return PlanStep(
                step_type="llm",
                action="direct",
                prompt_context=conversation_context,
            )

        if intent.action in ("", "noop"):
            return PlanStep(step_type="noop")

        # Default: skill step
        params: dict = {
            entity.name: entity.value
            for entity in intent.entities
        }
        return PlanStep(
            step_type="skill",
            action=action,
            params=params,
        )

    def _skill_step(self, decision: Decision) -> PlanStep:
        """Build a skill PlanStep, mapping intent entities to params."""
        intent = decision.intent
        params: dict = {
            entity.name: entity.value
            for entity in intent.entities
        }
        # Map intent action names to canonical skill action names.
        # This allows the intent vocabulary to differ from the skill API.
        action = _ACTION_ALIASES.get(intent.action, intent.action)
        return PlanStep(
            step_type="skill",
            action=action,
            params=params,
        )

    def _llm_step(self, decision: Decision, context: str) -> PlanStep:
        """Build an LLM PlanStep with conversation context."""
        return PlanStep(
            step_type="llm",
            action="direct",
            prompt_context=context,
        )
