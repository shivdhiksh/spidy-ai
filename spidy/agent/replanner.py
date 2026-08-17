"""
Replanner — Goal Recovery and Plan Revision
============================================
When a task fails with evidence that the environment changed unexpectedly
(soft failure), the Replanner generates a revised plan for the remaining
steps rather than simply aborting.

Replanning strategies
---------------------
1. LLM-based (when llm_client is available):
   Sends the LLM a structured prompt with:
   - The original goal
   - The failed step + observation summary
   - The hint from TaskEvaluator
   - The list of remaining steps (as context)
   The LLM returns a revised JSON task list for the remaining steps only.

2. Skip-and-continue fallback (always available):
   Mark the failed task as SKIPPED with reason "replanned: skipped",
   then continue with the original remaining tasks unchanged.
   This is the conservative safe default when LLM replanning fails or
   is unavailable.

Safety constraints
------------------
- max_replan_attempts per goal (default 2) prevents infinite loops.
- Replanning never increases the total task count beyond max_tasks.
- Replanned tasks inherit the same goal_id as the original tasks.
- Replanning is NEVER triggered by deterministic failures
  (TaskEvaluator.should_replan=False for those).

Usage
-----
    replanner = Replanner(max_attempts=2)
    revised = await replanner.replan(
        goal_description="...",
        goal_id="...",
        failed_task=task,
        observation=obs,
        remaining_tasks=tasks[idx+1:],
        replan_hint="Try a browser approach instead.",
        llm_client=llm_client,
    )
    # revised: list[TaskRecord] — replacement for remaining_tasks
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from spidy.agent.types import TaskRecord, TaskState
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.agent.observer import Observation
    from spidy.llm.client import BaseLLMClient

log = get_logger(__name__)

_MAX_TASKS = 8   # Safety cap on replanned task list

_REPLAN_SYSTEM_PROMPT = """\
You are a recovery planner for an AI desktop agent. A step in a multi-step
goal has failed and you must revise the remaining plan.

Rules:
- Return ONLY a valid JSON array, no prose, no markdown fences.
- Each item must have: "description" (short name) and "utterance" (command to send).
- Optional: "terminal": true for the last step.
- Produce the MINIMUM number of steps needed to recover and achieve the goal.
- Do NOT repeat steps that already succeeded.
- If recovery is impossible, return an empty array [].
"""

_REPLAN_USER_TEMPLATE = """\
Original goal: "{goal}"

Failed step: "{failed_desc}"
Utterance tried: "{failed_utt}"
Observation: "{observation}"
Hint: "{hint}"

Remaining original steps (for reference):
{remaining}

Provide a revised plan for the remaining steps to still achieve the original goal.
"""


class Replanner:
    """
    Generates a revised task plan when a step fails unexpectedly.

    Parameters
    ----------
    max_attempts:
        Maximum number of replan cycles per goal. Prevents infinite loops.
    max_tasks:
        Maximum number of tasks in a replanned list (safety cap).
    """

    def __init__(
        self,
        max_attempts: int = 2,
        max_tasks: int = _MAX_TASKS,
    ) -> None:
        self._max_attempts = max_attempts
        self._max_tasks = max_tasks
        # replan_count is tracked per goal_id
        self._replan_counts: dict[str, int] = {}

    def can_replan(self, goal_id: str) -> bool:
        """True if this goal has not exhausted its replan budget."""
        return self._replan_counts.get(goal_id, 0) < self._max_attempts

    def reset(self, goal_id: str) -> None:
        """Reset replan counter when a goal completes/fails/cancels."""
        self._replan_counts.pop(goal_id, None)

    async def replan(
        self,
        goal_description: str,
        goal_id: str,
        failed_task: TaskRecord,
        observation: "Observation",
        remaining_tasks: list[TaskRecord],
        replan_hint: str = "",
        llm_client: "BaseLLMClient | None" = None,
    ) -> list[TaskRecord]:
        """
        Generate a revised plan for the remaining tasks.

        Parameters
        ----------
        goal_description:
            The original high-level goal.
        goal_id:
            Parent goal ID (for tracking replan attempts).
        failed_task:
            The TaskRecord that failed.
        observation:
            The Observation after the failed task.
        remaining_tasks:
            The tasks that haven't been executed yet.
        replan_hint:
            Optional hint from TaskEvaluator (e.g. "try a browser approach").
        llm_client:
            Optional LLM client for intelligent replanning.

        Returns
        -------
        list[TaskRecord]
            Revised task list for the remaining steps.
            Returns the original remaining_tasks unchanged on failure.
        """
        # ── Guard: check replan budget ─────────────────────────────────────
        count = self._replan_counts.get(goal_id, 0)
        if count >= self._max_attempts:
            log.warning(
                "[AGENT] Replanner: max replan attempts ({n}) reached for goal {gid} -- aborting.",
                n=self._max_attempts,
                gid=goal_id[:8],
            )
            return []  # Signal abort

        self._replan_counts[goal_id] = count + 1

        log.info(
            "[AGENT] Replanning: attempt {n}/{max} for goal '{desc}' "
            "after failed task '{task}' (hint: '{hint}')",
            n=count + 1,
            max=self._max_attempts,
            desc=goal_description[:60],
            task=failed_task.description[:60],
            hint=replan_hint[:60],
        )

        # ── Try LLM-based replanning first ─────────────────────────────────
        if llm_client is not None:
            revised = await self._replan_with_llm(
                goal_description=goal_description,
                goal_id=goal_id,
                failed_task=failed_task,
                observation=observation,
                remaining_tasks=remaining_tasks,
                replan_hint=replan_hint,
                llm_client=llm_client,
            )
            if revised is not None:
                log.info(
                    "[AGENT] Replanner: LLM produced {n} revised tasks.",
                    n=len(revised),
                )
                return revised[:self._max_tasks]

        # ── Fallback: skip failed task, continue with remaining ─────────────
        log.info(
            "[AGENT] Replanner: LLM unavailable or failed — using skip-and-continue fallback."
        )
        return list(remaining_tasks)  # Keep original remaining tasks unchanged

    # ── LLM Replanning ─────────────────────────────────────────────────────

    async def _replan_with_llm(
        self,
        goal_description: str,
        goal_id: str,
        failed_task: TaskRecord,
        observation: "Observation",
        remaining_tasks: list[TaskRecord],
        replan_hint: str,
        llm_client: "BaseLLMClient",
    ) -> list[TaskRecord] | None:
        """Ask the LLM for a revised plan. Returns None on failure."""
        from spidy.llm.client import LLMMessage

        remaining_str = "\n".join(
            f"  {i+1}. {t.description}: {t.utterance}"
            for i, t in enumerate(remaining_tasks)
        ) or "  (none)"

        user_content = _REPLAN_USER_TEMPLATE.format(
            goal=goal_description,
            failed_desc=failed_task.description,
            failed_utt=failed_task.utterance,
            observation=observation.summary[:200] if observation.summary else "No observation.",
            hint=replan_hint or "None",
            remaining=remaining_str,
        )

        messages = [
            LLMMessage(role="system", content=_REPLAN_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_content),
        ]

        try:
            response = await llm_client.complete(messages)
            if not response.success or not response.text:
                log.debug("[AGENT] Replanner: LLM response unsuccessful.")
                return None
            return self._parse_llm_response(response.text, goal_id)

        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[AGENT] Replanner: LLM call failed (non-fatal): {exc}", exc=exc
            )
            return None

    def _parse_llm_response(
        self,
        text: str,
        goal_id: str,
    ) -> list[TaskRecord] | None:
        """Parse the LLM's JSON array of revised tasks."""
        cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        cleaned = cleaned.rstrip("```").strip()

        array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not array_match:
            log.debug("[AGENT] Replanner: no JSON array found in LLM response.")
            return None

        try:
            data = json.loads(array_match.group(0))
            if not isinstance(data, list):
                return None
            if not data:
                # LLM says recovery is impossible
                log.info("[AGENT] Replanner: LLM indicated recovery is impossible (empty array).")
                return []

            tasks: list[TaskRecord] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                desc = str(item.get("description", "")).strip()
                utt = str(item.get("utterance", "")).strip()
                if not desc or not utt:
                    continue
                terminal_raw = item.get("terminal", False)
                terminal = bool(terminal_raw) if isinstance(terminal_raw, bool) else str(terminal_raw).lower() == "true"
                tasks.append(TaskRecord(
                    goal_id=goal_id,
                    description=desc,
                    utterance=utt,
                    state=TaskState.PENDING,
                    terminal=terminal,
                    extra={"replanned": True},
                ))

            return tasks if tasks else None

        except (json.JSONDecodeError, ValueError) as exc:
            log.debug("[AGENT] Replanner: JSON parse error: {exc}", exc=exc)
            return None
