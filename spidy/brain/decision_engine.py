"""
DecisionEngine — Intent → Decision Mapping
==========================================
Given a classified Intent and the current conversation context,
the DecisionEngine decides *how* the Brain will respond.

Decision modes
--------------
SKILL       : The intent maps to a registered Skill. Dispatch to ToolRouter.
LLM_DIRECT  : No matching Skill — answer directly via the LLM.
CLARIFY     : Intent confidence too low — ask the user for clarification.
REJECT      : The Brain cannot or should not fulfil this request.

Decision logic
--------------
1. If confidence < min_confidence → CLARIFY
2. If action maps to a registered Skill → SKILL
3. If action == "chat" → LLM_DIRECT
4. Otherwise → LLM_DIRECT (Brain will try to answer with its LLM)

Special cases
-------------
- ``shutdown`` intent → REJECT with a shutdown rationale
  (SpidyCore handles the actual shutdown via event)
- Empty utterance / empty intent → REJECT

Lifelong Companion notes
------------------------
Future versions will call ``MemoryInterface.recall()`` to enrich the
decision with long-term context before choosing a mode.
The ``memory`` parameter is already accepted but unused in M3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spidy.brain.types import Decision, DecisionMode, Intent
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.brain.interfaces import MemoryInterface
    from spidy.skills.registry import SkillRegistry

log = get_logger(__name__)

# Actions that should never be dispatched to LLM_DIRECT
# NOTE: 'shutdown_system' and 'restart_system' are real SystemControlSkill actions.
# The old 'shutdown' intent (brain-stop signal) is mapped to 'shutdown_system' by
# the Planner's alias table. Removing it from REJECT allows the skill to execute.
_REJECT_ACTIONS: frozenset[str] = frozenset()  # No hard rejects — all route to skill or LLM

# Minimum confidence to act without asking for clarification
_DEFAULT_MIN_CONFIDENCE = 0.6


class DecisionEngine:
    """
    Decides how the Brain will handle an Intent.

    Parameters
    ----------
    skill_registry:
        The application skill registry. Used to check if an intent
        has a registered skill handler.
    min_confidence:
        Below this threshold, the engine emits a CLARIFY decision.
    memory:
        Optional memory interface for context enrichment (unused in M3;
        accepted here for forward compatibility).
    """

    def __init__(
        self,
        skill_registry: "SkillRegistry",
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        memory: "MemoryInterface | None" = None,
    ) -> None:
        self._registry = skill_registry
        self._min_confidence = min_confidence
        self._memory = memory   # Unused in M3 — reserved for M6

    async def decide(self, intent: Intent, session_id: str = "") -> Decision:
        """
        Evaluate an Intent and return a Decision.

        Parameters
        ----------
        intent:
            The classified intent from IntentClassifier.
        session_id:
            Current session ID (for future memory recall).

        Returns
        -------
        Decision
            What the Brain will do and why.
        """
        # Edge case: empty utterance
        if not intent.action or not intent.raw_utterance.strip():
            return Decision(
                mode=DecisionMode.REJECT,
                intent=intent,
                rationale="Empty utterance — nothing to act on.",
            )

        # Shutdown intent — reject and let SpidyCore handle via event
        if intent.action in _REJECT_ACTIONS:
            return Decision(
                mode=DecisionMode.REJECT,
                intent=intent,
                rationale=f"Action '{intent.action}' is handled by the system, not the Brain.",
            )

        # Low confidence — ask for clarification
        if intent.confidence < self._min_confidence:
            return Decision(
                mode=DecisionMode.CLARIFY,
                intent=intent,
                rationale=f"Low confidence ({intent.confidence:.0%}) — intent unclear.",
                clarification_question=(
                    "I'm not sure I understood that. Could you rephrase or be more specific?"
                ),
            )

        # Check if a skill handles this action
        skill = self._registry.find_skill_for_action(intent.action)
        if skill is not None:
            log.debug(
                "Intent '{a}' → SKILL '{s}'",
                a=intent.action,
                s=skill.name,
            )
            return Decision(
                mode=DecisionMode.SKILL,
                intent=intent,
                skill_name=skill.name,
                rationale=f"Skill '{skill.name}' handles '{intent.action}'.",
            )

        # chat or unknown action → LLM_DIRECT
        log.debug(
            "Intent '{a}' → LLM_DIRECT (no matching skill)",
            a=intent.action,
        )
        return Decision(
            mode=DecisionMode.LLM_DIRECT,
            intent=intent,
            rationale=(
                "No registered skill for this action. "
                "Brain will answer directly via LLM."
            ),
        )
