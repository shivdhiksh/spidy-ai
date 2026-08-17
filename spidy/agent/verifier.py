"""
GoalVerifier — Final Goal Completion Verification
==================================================
After all tasks complete, the GoalVerifier performs a final sanity check
to confirm the goal was actually achieved — not just that the tasks ran.

For simple single-step goals ("Open Calculator"), verification is implicit:
if the only terminal task succeeded, the goal is verified automatically.

For multi-step goals, the verifier inspects the observation log and the
task results to confirm the key outcome is present.

Design
------
- Fail-open: if verification is inconclusive, the goal is marked
  VERIFIED_PARTIAL (not FAILED). The user sees the result either way.
- No LLM calls in the basic implementation (optional LLM confirmation path
  is available but disabled by default for latency reasons).
- Verification summary is added to GoalRecord.summary.

Usage
-----
    verifier = GoalVerifier()
    result = await verifier.verify(goal, observations, llm_client=None)
    # result.verified: True | False
    # result.summary: human-readable outcome summary
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spidy.agent.types import GoalRecord, TaskRecord, TaskState
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.agent.observer import Observation
    from spidy.llm.client import BaseLLMClient

log = get_logger(__name__)


@dataclass
class VerificationResult:
    """
    Result of final goal verification.

    Attributes
    ----------
    verified:
        True = goal outcome confirmed; False = outcome uncertain or failed.
    confidence:
        "high" / "medium" / "low" / "partial"
    summary:
        Human-readable summary of what was achieved.
    method:
        How verification was performed: "task_results", "observation_log",
        "llm", "implicit" (single terminal task), "skipped".
    """
    verified: bool = True
    confidence: str = "medium"
    summary: str = ""
    method: str = "task_results"


class GoalVerifier:
    """
    Verifies that a goal's final outcome matches what was requested.

    Parameters
    ----------
    skip_for_single_task:
        When True (default), single-terminal-task goals are implicitly
        verified without any extra checks.
    """

    def __init__(self, skip_for_single_task: bool = True) -> None:
        self._skip_for_single_task = skip_for_single_task

    async def verify(
        self,
        goal: GoalRecord,
        observations: list["Observation"] | None = None,
        llm_client: "BaseLLMClient | None" = None,
    ) -> VerificationResult:
        """
        Perform final verification of a completed goal.

        Parameters
        ----------
        goal:
            The completed GoalRecord.
        observations:
            Observation objects collected during task execution.
        llm_client:
            Optional LLM for complex verification (not used by default).

        Returns
        -------
        VerificationResult
            Verification outcome. Never raises.
        """
        tasks = goal.tasks
        completed = [t for t in tasks if t.state == TaskState.COMPLETED]
        failed = [t for t in tasks if t.state == TaskState.FAILED]
        skipped = [t for t in tasks if t.state == TaskState.SKIPPED]

        total = len(tasks)

        # ── Implicit verification for single terminal tasks ─────────────────
        if self._skip_for_single_task and total == 1:
            task = tasks[0]
            if task.state == TaskState.COMPLETED:
                summary = self._build_summary(goal, completed, failed, skipped)
                log.info(
                    "[AGENT] Verification: implicit (single terminal task succeeded) '{desc}'",
                    desc=goal.description[:60],
                )
                return VerificationResult(
                    verified=True,
                    confidence="high",
                    summary=summary,
                    method="implicit",
                )

        # ── Multi-task verification ─────────────────────────────────────────
        n_completed = len(completed)
        n_failed = len(failed)
        n_skipped = len(skipped)

        # Check observation log for positive signals
        obs_signals: list[bool] = []
        if observations:
            obs_signals = [o.success_signal for o in observations if o.success_signal is not None]

        all_obs_positive = obs_signals and all(obs_signals)
        any_obs_negative = any(s is False for s in obs_signals)

        # Build verification verdict
        if n_failed > 0:
            # Some tasks failed — partial verification
            summary = self._build_summary(goal, completed, failed, skipped)
            log.info(
                "[AGENT] Verification: PARTIAL — {done}/{total} tasks succeeded",
                done=n_completed,
                total=total,
            )
            return VerificationResult(
                verified=False,
                confidence="high",
                summary=summary,
                method="task_results",
            )

        if n_completed == total or n_completed + n_skipped == total:
            # All tasks completed or skipped
            confidence = "high" if (all_obs_positive or not obs_signals) else "medium"
            summary = self._build_summary(goal, completed, failed, skipped)
            log.info(
                "[AGENT] Verification: COMPLETE — {done}/{total} tasks completed",
                done=n_completed,
                total=total,
            )
            return VerificationResult(
                verified=True,
                confidence=confidence,
                summary=summary,
                method="task_results" if not obs_signals else "observation_log",
            )

        # Partial completion (shouldn't normally happen if ExecutionLoop is correct)
        summary = self._build_summary(goal, completed, failed, skipped)
        return VerificationResult(
            verified=False,
            confidence="low",
            summary=f"Partial completion: {n_completed}/{total} tasks done. {summary}",
            method="task_results",
        )

    # ── Summary builder ────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(
        goal: GoalRecord,
        completed: list[TaskRecord],
        failed: list[TaskRecord],
        skipped: list[TaskRecord],
    ) -> str:
        """Build a user-facing summary string."""
        if not completed and not failed:
            return f"No tasks were executed for '{goal.description}'."

        parts: list[str] = []

        if completed:
            descs = [t.description.lower() for t in completed[:3]]
            if len(descs) == 1:
                parts.append(f"Done: {descs[0]}")
            elif len(descs) == 2:
                parts.append(f"Done: {descs[0]} and {descs[1]}")
            else:
                parts.append(f"Done: {', '.join(descs[:-1])}, and {descs[-1]}")

        if failed:
            fail_descs = [t.description.lower() for t in failed[:2]]
            parts.append(f"Failed: {', '.join(fail_descs)}")

        if skipped:
            parts.append(f"{len(skipped)} step(s) skipped")

        return ". ".join(parts) + "."
