"""
AgentTaskLogger — Structured [AGENT] Log Helper
================================================
Thin helper that emits structured [AGENT]-prefixed log lines at every
lifecycle milestone of autonomous goal execution.

All log lines follow this format:
    [AGENT] <milestone>: <details>

Examples
--------
    [AGENT] Goal created: 'Open Edge and search Python' | id=abc12345
    [AGENT] Plan generated: 5 tasks
    [AGENT] Task 1/5 started: 'Open Edge browser'
    [AGENT] Task 1/5 success: 'Open Edge browser'
    [AGENT] Task 2/5 failed: 'Search YouTube' | reason=App not found
    [AGENT] Observation: method=app_state signal=True
    [AGENT] Evaluation: success=True confidence=high
    [AGENT] Replanning: attempt 1/2
    [AGENT] Goal complete: 'Open Edge and search Python' | 2/2 tasks
    [AGENT] Goal failed: 'Upload to Instagram' | error=Permission denied
    [AGENT] Goal cancelled: 'Open Edge' | 0/1 tasks completed
    [AGENT] Verification: verified=True confidence=high

Design
------
- Zero dependencies beyond the logger — safe to import anywhere
- All methods are synchronous (log output only)
- Secrets are never included in log output
- Methods never raise
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.agent.evaluator import TaskOutcome
    from spidy.agent.observer import Observation
    from spidy.agent.types import GoalRecord, TaskRecord
    from spidy.agent.verifier import VerificationResult

log = get_logger(__name__)

# Tag prefix — all agent lifecycle log lines use this prefix
_TAG = "[AGENT]"


class AgentTaskLogger:
    """
    Emits structured [AGENT] log lines at every goal lifecycle stage.

    This class has no state — all methods are effectively static helpers
    wrapped in a class for injection/mocking in tests.
    """

    # ── Goal lifecycle ─────────────────────────────────────────────────────

    def goal_created(self, goal: "GoalRecord") -> None:
        """Emitted when a new goal is registered."""
        log.info(
            "{tag} Goal created: '{desc}' | id={gid}",
            tag=_TAG,
            desc=goal.description[:80],
            gid=goal.goal_id[:8],
        )

    def plan_generated(self, goal: "GoalRecord", tasks: "list[TaskRecord]") -> None:
        """Emitted after TaskDecomposer produces the task list."""
        log.info(
            "{tag} Plan generated: {n} task(s) for goal '{desc}'",
            tag=_TAG,
            n=len(tasks),
            desc=goal.description[:60],
        )
        for i, task in enumerate(tasks):
            log.info(
                "  {tag}   Task {i}/{n}: '{desc}'",
                tag=_TAG,
                i=i + 1,
                n=len(tasks),
                desc=task.description[:60],
            )

    def goal_complete(self, goal: "GoalRecord") -> None:
        """Emitted when all tasks complete successfully."""
        log.info(
            "{tag} Goal complete: '{desc}' | {done}/{total} tasks",
            tag=_TAG,
            desc=goal.description[:80],
            done=goal.completed_task_count,
            total=goal.task_count,
        )

    def goal_failed(self, goal: "GoalRecord", error: str) -> None:
        """Emitted when the goal fails permanently."""
        log.warning(
            "{tag} Goal failed: '{desc}' | error={err}",
            tag=_TAG,
            desc=goal.description[:80],
            err=error[:120],
        )

    def goal_cancelled(self, goal: "GoalRecord") -> None:
        """Emitted when the goal is cancelled by the user."""
        log.info(
            "{tag} Goal cancelled: '{desc}' | {done}/{total} completed",
            tag=_TAG,
            desc=goal.description[:80],
            done=goal.completed_task_count,
            total=goal.task_count,
        )

    # ── Task lifecycle ─────────────────────────────────────────────────────

    def task_started(
        self,
        task: "TaskRecord",
        task_index: int,
        total: int,
        authority_label: str = "T0/SAFE",
    ) -> None:
        """Emitted when a task begins executing."""
        log.info(
            "{tag} Task {i}/{n} started: '{desc}' | auth={auth} | attempt={a}",
            tag=_TAG,
            i=task_index + 1,
            n=total,
            desc=task.description[:60],
            auth=authority_label,
            a=task.attempt,
        )

    def task_success(
        self,
        task: "TaskRecord",
        task_index: int,
        total: int,
    ) -> None:
        """Emitted when a task completes successfully."""
        log.info(
            "{tag} Task {i}/{n} success: '{desc}'",
            tag=_TAG,
            i=task_index + 1,
            n=total,
            desc=task.description[:60],
        )

    def task_failed(
        self,
        task: "TaskRecord",
        task_index: int,
        total: int,
        reason: str,
        will_retry: bool = False,
        will_replan: bool = False,
    ) -> None:
        """Emitted when a task fails."""
        action = "retrying" if will_retry else ("replanning" if will_replan else "aborting")
        log.warning(
            "{tag} Task {i}/{n} failed: '{desc}' | reason={reason} | action={action}",
            tag=_TAG,
            i=task_index + 1,
            n=total,
            desc=task.description[:60],
            reason=reason[:80],
            action=action,
        )

    def task_skipped(self, task: "TaskRecord", task_index: int, total: int, reason: str) -> None:
        """Emitted when a task is intentionally skipped."""
        log.info(
            "{tag} Task {i}/{n} skipped: '{desc}' | reason={reason}",
            tag=_TAG,
            i=task_index + 1,
            n=total,
            desc=task.description[:60],
            reason=reason[:60],
        )

    def confirmation_required(self, task: "TaskRecord", authority_label: str) -> None:
        """Emitted when a task requires user confirmation before executing."""
        log.warning(
            "{tag} Confirmation required: '{desc}' | authority={auth}",
            tag=_TAG,
            desc=task.description[:60],
            auth=authority_label,
        )

    # ── Observation / Evaluation / Replanning / Verification ───────────────

    def observation(self, task: "TaskRecord", obs: "Observation") -> None:
        """Emitted after TaskObserver.observe()."""
        log.info(
            "{tag} Observation: task='{desc}' method={m} signal={s}",
            tag=_TAG,
            desc=task.description[:60],
            m=obs.method,
            s=obs.success_signal,
        )

    def evaluation(self, task: "TaskRecord", outcome: "TaskOutcome") -> None:
        """Emitted after TaskEvaluator.evaluate()."""
        log.info(
            "{tag} Evaluation: task='{desc}' success={s} confidence={c} replan={r}",
            tag=_TAG,
            desc=task.description[:60],
            s=outcome.success,
            c=outcome.confidence.value,
            r=outcome.should_replan,
        )

    def replanning(self, attempt: int, max_attempts: int, reason: str) -> None:
        """Emitted when replanning is triggered."""
        log.info(
            "{tag} Replanning: attempt {n}/{max} | reason={reason}",
            tag=_TAG,
            n=attempt,
            max=max_attempts,
            reason=reason[:80],
        )

    def verification(self, result: "VerificationResult") -> None:
        """Emitted after GoalVerifier.verify()."""
        log.info(
            "{tag} Verification: verified={v} confidence={c} method={m} summary='{s}'",
            tag=_TAG,
            v=result.verified,
            c=result.confidence,
            m=result.method,
            s=result.summary[:80],
        )
