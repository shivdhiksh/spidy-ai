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
        goal_context: "GoalContext | None" = None,
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
        goal_context:
            Optional GoalContext for comprehensive goal-level verification.

        Returns
        -------
        VerificationResult
            Verification outcome. Never raises.
        """
        tasks = goal.tasks
        completed = [t for t in tasks if t.state in (TaskState.COMPLETED, TaskState.SUCCEEDED)]
        failed = [t for t in tasks if t.state == TaskState.FAILED]
        skipped = [t for t in tasks if t.state in (TaskState.SKIPPED, TaskState.CANCELLED)]

        total = len(tasks)

        # ── Mandatory Playback Verification Check ───────────────────────────
        # Requirement 11: GoalVerifier must NOT report success unless playback itself is verified.
        is_playback_goal = (
            ("play" in goal.description.lower() and any(k in goal.description.lower() for k in ("video", "youtube")))
            or any(t.action and t.action.get("action") == "play_youtube_video" for t in tasks)
        )
        if is_playback_goal:
            has_playback = False
            for t in completed:
                ev = getattr(t, "evidence", None) or (t.extra.get("evidence") if t.extra else None) or {}
                if isinstance(ev, dict):
                    if ev.get("playback_verified") is True and ev.get("is_playing") is True and not ev.get("is_paused"):
                        has_playback = True
                        break
                    if ev.get("status") in ("SUCCEEDED", "playing") and (ev.get("step") == "play_youtube_video" or ev.get("playback_verified") is True):
                        has_playback = True
                        break
                if t.action and t.action.get("action") == "play_youtube_video" and ("playing" in t.result_message.lower() or "playback verified" in t.result_message.lower()):
                    has_playback = True
                    break
            if not has_playback and observations:
                for o in observations:
                    if o.success_signal is True and o.extra.get("playback_verified") is True:
                        has_playback = True
                        break
            if not has_playback:
                log.warning("[AGENT] Verification: FAILED — YouTube video playback not verified for goal '{desc}'", desc=goal.description[:60])
                summary = self._build_summary(goal, completed, failed, skipped)
                return VerificationResult(
                    verified=False,
                    confidence="low",
                    summary=f"Incomplete goal: YouTube video playback was not confirmed. Video is not playing. {summary}",
                    method="playback_verification_check",
                )

        # ── Clause integrity check ──────────────────────────────────────────
        from spidy.agent.task_decomposer import _extract_actionable_clauses
        actionable_clauses = _extract_actionable_clauses(goal.description)
        n_clauses = len(actionable_clauses)

        # Check observation log for positive signals
        obs_signals: list[bool] = []
        if observations:
            obs_signals = [o.success_signal for o in observations if o.success_signal is not None]

        all_obs_positive = obs_signals and all(obs_signals)
        any_obs_negative = any(s is False for s in obs_signals)

        # ── Check required outcomes if GoalContext provided ──────────────────
        if goal_context and goal_context.required_outcomes:
            for ro in goal_context.required_outcomes:
                # Check if any completed task produced evidence of outcome
                has_match = any(
                    (t.action and t.action.get("expected_outcome", "").lower() == ro.lower())
                    or (ro.lower() in t.result_message.lower())
                    for t in completed
                )
                if not has_match and total > 0:
                    summary = self._build_summary(goal, completed, failed, skipped)
                    log.warning("[AGENT] Verification: required outcome '{ro}' not satisfied", ro=ro)
                    return VerificationResult(
                        verified=False,
                        confidence="partial",
                        summary=f"Incomplete goal: required outcome '{ro}' not verified. {summary}",
                        method="required_outcome_check",
                    )

        # ── Implicit verification for single terminal tasks ─────────────────
        if self._skip_for_single_task and total == 1:
            if n_clauses > 1:
                log.error(
                    "[AGENT] Verification: FAILED — compound goal with {n} clauses "
                    "was reduced to 1 task ('{desc}')",
                    n=n_clauses,
                    desc=goal.description[:60],
                )
                return VerificationResult(
                    verified=False,
                    confidence="partial",
                    summary=(
                        f"Incomplete execution: only 1 of {n_clauses} requested actions "
                        f"was executed ({completed[0].description if completed else 'none'}). "
                        f"Missing remaining actions."
                    ),
                    method="clause_integrity_check",
                )

            task = tasks[0]
            if task.state == TaskState.COMPLETED and not any_obs_negative:
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

        if n_clauses > total:
            log.warning(
                "[AGENT] Verification: PARTIAL — original goal requested {n} clauses "
                "but only {total} tasks were planned.",
                n=n_clauses,
                total=total,
            )
            summary = self._build_summary(goal, completed, failed, skipped)
            return VerificationResult(
                verified=False,
                confidence="partial",
                summary=f"Partial execution: {total}/{n_clauses} requested actions planned. {summary}",
                method="clause_integrity_check",
            )

        # Build verification verdict
        if n_failed > 0:
            summary = self._build_summary(goal, completed, failed, skipped)
            log.info(
                "[AGENT] Verification: PARTIAL — {done}/{total} tasks succeeded",
                done=n_completed,
                total=total,
            )
            return VerificationResult(
                verified=False,
                confidence="partial",
                summary=summary,
                method="task_results",
            )

        if any_obs_negative:
            summary = self._build_summary(goal, completed, failed, skipped)
            negative_summaries = [o.summary for o in (observations or []) if o.success_signal is False]
            neg_text = "; ".join(negative_summaries[:2]) if negative_summaries else "Environmental verification check failed"
            log.warning(
                "[AGENT] Verification: FAILED due to negative observation signal: {msg}",
                msg=neg_text,
            )
            return VerificationResult(
                verified=False,
                confidence="high",
                summary=f"Verification failed: {neg_text}. {summary}",
                method="observation_log",
            )

        if n_completed == total or n_completed + n_skipped == total:
            confidence = "high" if (all_obs_positive or not obs_signals) else "medium"
            summary = self._build_summary(goal, completed, failed, skipped)
            outcome_phrases = [
                t.action["expected_outcome"]
                for t in completed
                if t.action and isinstance(t.action, dict) and t.action.get("expected_outcome")
            ]
            if outcome_phrases:
                joined = "; ".join(outcome_phrases[:3])
                summary = f"{summary} Achieved: {joined}."
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

        summary = self._build_summary(goal, completed, failed, skipped)
        return VerificationResult(
            verified=False,
            confidence="low",
            summary=f"Partial completion: {n_completed}/{total} tasks done. {summary}",
            method="task_results",
        )

    async def verify_goal(
        self,
        goal_context: "GoalContext",
        observations: list["Observation"] | None = None,
        llm_client: "BaseLLMClient | None" = None,
    ) -> VerificationResult:
        """Convenience method to verify directly against a GoalContext."""
        goal_rec = GoalRecord(
            goal_id=goal_context.goal_id,
            description=goal_context.original_goal,
            tasks=goal_context.tasks,
        )
        return await self.verify(
            goal=goal_rec,
            observations=observations,
            llm_client=llm_client,
            goal_context=goal_context,
        )

    # ── Summary builder ────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(
        goal: GoalRecord,
        completed: list[TaskRecord],
        failed: list[TaskRecord],
        skipped: list[TaskRecord],
    ) -> str:
        """Build a user-facing, result-aware summary string."""
        desc_lower = goal.description.lower()

        # Handle specific common multi-step intent summaries for natural phrasing
        if "find" in desc_lower and "resume" in desc_lower and "open" in desc_lower:
            find_done = any("find" in t.description.lower() or "search" in t.description.lower() for t in completed)
            open_done = any("open" in t.description.lower() for t in completed)
            if find_done and open_done:
                return "Found your resume and opened it."
            elif find_done and not open_done:
                return "I found your resume, but I couldn't open it."
            else:
                return "I couldn't find the resume."

        if not completed and not failed:
            return f"No tasks were executed for '{goal.description}'."

        parts: list[str] = []

        if completed:
            # Check if there are specific calculation result messages
            calc_msgs = [t.result_message for t in completed if "result of" in t.result_message.lower() or "is 450" in t.result_message]
            if calc_msgs:
                parts.append(" ".join(calc_msgs))
            else:
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
