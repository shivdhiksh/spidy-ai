"""
TaskEvaluator — Structured Task Outcome Evaluation
===================================================
Combines the TaskObserver's environmental observation with the
ReflectionEngine's lexical analysis to produce a structured TaskOutcome.

TaskOutcome replaces the raw (ReflectionDecision, reason) pair from
the earlier architecture. It carries richer information:

  - success        → did the task accomplish its goal?
  - confidence     → how sure are we? (HIGH / MEDIUM / LOW / UNKNOWN)
  - reason         → human-readable explanation
  - should_replan  → True if the remaining plan should be revised
  - replan_hint    → suggestion for the Replanner ("try browser instead")

Design
------
- Fully synchronous — no I/O, no LLM calls
- Deterministic: same inputs always produce the same outcome
- Conservative: defaults to CONTINUE on ambiguous evidence
- Deterministic failures (permission denied, no skill) never trigger replanning

Usage
-----
    evaluator = TaskEvaluator()
    outcome = evaluator.evaluate(
        task=task,
        observation=obs,
        reflection_decision=decision,
        reflection_reason=reason,
    )
    if outcome.should_replan:
        new_tasks = await replanner.replan(...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from spidy.agent.types import Confidence, ReflectionDecision, TaskErrorCode, TaskOutcome, TaskRecord
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.agent.observer import Observation

log = get_logger(__name__)


def classify_task_error(error_text: str) -> tuple[TaskErrorCode, str]:
    """Classify failure text into a structured TaskErrorCode and recovery strategy."""
    text = (error_text or "").lower()
    if any(k in text for k in ("app not found", "cannot find app", "executable not found", "not installed")):
        return TaskErrorCode.APP_NOT_FOUND, "check_installed_apps"
    if any(k in text for k in ("window not found", "no running window", "window matching")):
        return TaskErrorCode.WINDOW_NOT_FOUND, "enumerate_windows_and_fuzzy_match"
    if any(k in text for k in ("focus lost", "could not bring to foreground", "focus policy")):
        return TaskErrorCode.FOCUS_LOST, "refocus_target_window"
    if any(k in text for k in ("control not found", "element not found", "uiautomation")):
        return TaskErrorCode.CONTROL_NOT_FOUND, "fallback_to_coordinates_or_vision"
    if any(k in text for k in ("file not found", "no such file", "could not find file", "no matching files")):
        return TaskErrorCode.FILE_NOT_FOUND, "broaden_safe_search_roots"
    if any(k in text for k in ("permission denied", "access denied", "administrator")):
        return TaskErrorCode.PERMISSION_DENIED, "request_user_authority"
    if any(k in text for k in ("browser failure", "navigation timed out", "page crashed", "playwright")):
        return TaskErrorCode.BROWSER_FAILURE, "restart_browser_session"
    if any(k in text for k in ("network failure", "connection refused", "dns", "unreachable")):
        return TaskErrorCode.NETWORK_FAILURE, "check_network_and_retry"
    if any(k in text for k in ("save failed", "cannot save", "disk full")):
        return TaskErrorCode.SAVE_FAILED, "verify_write_path_and_retry"
    if any(k in text for k in ("nameerror", "syntaxerror", "typeerror", "traceback", "indexerror", "keyerror")):
        return TaskErrorCode.CODE_ERROR, "trigger_code_repair_engine"
    if any(k in text for k in ("test failed", "assertionerror")):
        return TaskErrorCode.TEST_FAILED, "diagnose_test_traceback"
    if any(k in text for k in ("screen unreadable", "ocr failed", "vision error")):
        return TaskErrorCode.SCREEN_UNREADABLE, "capture_high_dpi_region"
    if any(k in text for k in ("clarification required", "ambiguous", "multiple matches", "which one")):
        return TaskErrorCode.AMBIGUOUS_INTENT, "ask_user_clarification"
    if "timeout" in text:
        return TaskErrorCode.TIMEOUT, "increase_deadline"
    return TaskErrorCode.UNKNOWN, "general_reflection"


# ── Deterministic Failure Patterns ────────────────────────────────────────────

# These failures cannot be fixed by replanning — abort immediately
_DETERMINISTIC_FAILURE_PHRASES: list[str] = [
    "permission denied",
    "no skill",
    "i don't have a skill",
    "i'm not able to",
    "app not found",
    "not installed",
    "command not found",
    "access denied",
    "not available",
    "i cannot",
    "i can't perform",
    "cannot open",
    "no such file",
    "file not found",
]


def _is_deterministic_failure(response_text: str) -> bool:
    """Return True if the failure pattern is deterministic (no point replanning)."""
    text = response_text.lower()
    return any(p in text for p in _DETERMINISTIC_FAILURE_PHRASES)


class TaskEvaluator:
    """
    Combines observation + reflection to produce a structured TaskOutcome.

    Parameters
    ----------
    replan_on_soft_failures:
        When True (default), soft failures trigger replanning when confidence
        is MEDIUM or higher. Set False to disable replanning.
    """

    def __init__(self, replan_on_soft_failures: bool = True) -> None:
        self._replan_on_soft_failures = replan_on_soft_failures

    def evaluate(
        self,
        task: TaskRecord,
        observation: "Observation",
        reflection_decision: ReflectionDecision,
        reflection_reason: str,
        response_text: str = "",
    ) -> TaskOutcome:
        """
        Evaluate the outcome of a completed task.

        Parameters
        ----------
        task:
            The TaskRecord (includes attempt count, utterance, extra).
        observation:
            The Observation from TaskObserver.observe().
        reflection_decision:
            The decision from ReflectionEngine.reflect().
        reflection_reason:
            The reason string from ReflectionEngine.reflect().
        response_text:
            Brain's raw response text (for deterministic failure check).

        Returns
        -------
        TaskOutcome
            Structured evaluation result.
        """
        # ── Case 1: Clear success from reflection ──────────────────────────
        if reflection_decision == ReflectionDecision.CONTINUE:
            # Physical desktop operations are not complete merely because a
            # skill/LLM claimed success.  A required verification must be a
            # positive environmental observation, never response text or an
            # unknown state.
            physical_actions = {
                "open_app", "open_file_in_app", "launch_app", "focus",
                "create_file", "write_file", "save_file", "save",
                "open_file", "open_folder", "navigate", "search",
            }
            action_name = (
                (task.action or {}).get("action", "").lower()
                if isinstance(task.action, dict) else ""
            )
            requires_authoritative_observation = (
                task.verification_required and action_name in physical_actions
            )
            if requires_authoritative_observation and (
                observation.success_signal is not True
                or observation.method in ("response_text", "skipped")
            ):
                error_text = observation.summary or response_text
                error_code, recovery = classify_task_error(error_text)
                return TaskOutcome(
                    success=False,
                    confidence=Confidence.HIGH if observation.success_signal is False else Confidence.MEDIUM,
                    reason=f"Required environmental verification did not pass: {error_text}",
                    error_code=error_code,
                    recovery_strategy=recovery,
                    should_replan=observation.success_signal is False and self._replan_on_soft_failures,
                    replan_hint="Obtain an authoritative filesystem or window observation before continuing.",
                    reflection_decision=ReflectionDecision.ALTERNATIVE,
                )
            # Verify with observation if available
            if observation.success_signal is True:
                return TaskOutcome(
                    success=True,
                    confidence=Confidence.HIGH,
                    reason="Reflection and observation both indicate success.",
                    should_replan=False,
                    reflection_decision=reflection_decision,
                )
            if observation.success_signal is None:
                # Observation inconclusive — trust reflection
                return TaskOutcome(
                    success=True,
                    confidence=Confidence.MEDIUM,
                    reason=f"Reflection indicates success (observation inconclusive). {reflection_reason}",
                    should_replan=False,
                    reflection_decision=reflection_decision,
                )
            # Observation disagrees with reflection → downgrade confidence
            # but still mark success (reflection is primary signal)
            return TaskOutcome(
                success=True,
                confidence=Confidence.LOW,
                reason=(
                    f"Reflection indicates success but observation is uncertain "
                    f"({observation.summary[:60]}). {reflection_reason}"
                ),
                should_replan=False,
                reflection_decision=reflection_decision,
            )

        # ── Case 2: Clarification needed ──────────────────────────────────
        if reflection_decision == ReflectionDecision.ASK_USER:
            return TaskOutcome(
                success=False,
                confidence=Confidence.HIGH,
                reason="Brain requested user clarification.",
                should_replan=False,  # User must answer, not replanner
                reflection_decision=reflection_decision,
            )

        # ── Case 3: Retry decision ─────────────────────────────────────────
        if reflection_decision == ReflectionDecision.RETRY:
            return TaskOutcome(
                success=False,
                confidence=Confidence.MEDIUM,
                reason=reflection_reason,
                should_replan=False,  # Retry first, replan only after exhaustion
                reflection_decision=reflection_decision,
            )

        # ── Case 4: Alternative utterance ─────────────────────────────────
        if reflection_decision == ReflectionDecision.ALTERNATIVE:
            # Check for deterministic failure before deciding on replan
            if _is_deterministic_failure(response_text):
                return TaskOutcome(
                    success=False,
                    confidence=Confidence.HIGH,
                    reason=f"Deterministic failure — cannot recover by replanning. {reflection_reason}",
                    should_replan=False,
                    reflection_decision=reflection_decision,
                )
            return TaskOutcome(
                success=False,
                confidence=Confidence.MEDIUM,
                reason=reflection_reason,
                should_replan=self._replan_on_soft_failures,
                replan_hint="Try a different approach or tool for this step.",
                reflection_decision=reflection_decision,
            )

        # ── Case 5: Abort ─────────────────────────────────────────────────
        if reflection_decision == ReflectionDecision.ABORT:
            if _is_deterministic_failure(response_text):
                return TaskOutcome(
                    success=False,
                    confidence=Confidence.HIGH,
                    reason=f"Deterministic failure — aborting without replanning. {reflection_reason}",
                    should_replan=False,
                    reflection_decision=reflection_decision,
                )
            # Soft abort — maybe the environment changed; offer replan
            if observation.success_signal is False:
                return TaskOutcome(
                    success=False,
                    confidence=Confidence.HIGH,
                    reason=f"Observation confirms failure. {reflection_reason}",
                    should_replan=self._replan_on_soft_failures,
                    replan_hint="Environment state changed. Revise the remaining steps.",
                    reflection_decision=reflection_decision,
                )
            return TaskOutcome(
                success=False,
                confidence=Confidence.MEDIUM,
                reason=reflection_reason,
                should_replan=self._replan_on_soft_failures,
                replan_hint="Revise the approach for remaining steps.",
                reflection_decision=reflection_decision,
            )

        # ── Fallback ────────────────────────────────────────────────────────
        return TaskOutcome(
            success=True,
            confidence=Confidence.LOW,
            reason=f"Unknown reflection decision '{reflection_decision}'. Proceeding cautiously.",
            should_replan=False,
            reflection_decision=reflection_decision,
        )

    def log_outcome(self, task: TaskRecord, outcome: TaskOutcome) -> None:
        """Emit a structured [AGENT] Evaluation log line."""
        log.info(
            "[AGENT] Evaluation: task='{desc}' success={s} confidence={c} replan={r} reason='{reason}'",
            desc=task.description[:60],
            s=outcome.success,
            c=outcome.confidence.value,
            r=outcome.should_replan,
            reason=outcome.reason[:80],
        )
