"""
ExecutionLoop -- Autonomous Agent Execution Engine (Milestones 16 + 18)
========================================================================
The core loop that drives autonomous goal execution.

Pipeline (per-goal)
--------------------
  tasks = TaskDecomposer.decompose(goal)
  for task in tasks:
      if ctx.cancelled: break

      1. TaskAuthorityChecker.check(task)  [M16]
         -> CONFIRM/CRITICAL -> publish AgentConfirmationRequiredEvent, skip
         -> SAFE/AWARE       -> proceed

      2a. IF task.action (structured contract) [M18]:
          -> StructuredRouter.execute(action, task, session_id)
          -> ctx.add_result(task.task_id, result)
      2b. ELSE:
          -> Brain.process(task.utterance)   [original path]

      3. TaskObserver.observe(task, response_text)  [M16]

      4. ReflectionEngine.reflect(task, response_text, success)

      5. TaskEvaluator.evaluate(observation, reflection_decision)  [M16]
         -> outcome.success -> mark COMPLETED
         -> outcome.should_replan -> Replanner.replan() -> revise remaining
         -> otherwise -> RETRY / ABORT as before

  GoalVerifier.verify(goal, observations)  [M16]

Design
------
- Observer, Evaluator, Replanner, AuthorityChecker, StructuredRouter are
  all optional (None-safe). When not configured, the loop behaves
  identically to M13.
- Cancellation is checked at the top of every task iteration AND after each
  execution call (long-running tasks respect mid-task cancellation).
- All [AGENT] structured log tags are emitted via AgentTaskLogger.
- Self-recovery: ALTERNATIVE tries a rephrased utterance before replanning.
- Natural progress events are published via ProgressTracker.

Usage
-----
    loop = ExecutionLoop(
        brain=brain,
        goal_manager=goal_manager,
        reflection=reflection_engine,
        tracker=progress_tracker,
        bus=bus,
        observer=observer,
        evaluator=evaluator,
        replanner=replanner,
        authority=authority_checker,
        task_log=task_logger,
        router=structured_router,   # M18
    )
    completed_goal = await loop.run(goal, tasks, ctx)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from spidy.agent.reflection_engine import ReflectionEngine
from spidy.agent.types import (
    ExecutionContext,
    GoalRecord,
    GoalState,
    ReflectionDecision,
    TaskRecord,
    TaskState,
)
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.agent.authority import AuthorityLevel, TaskAuthorityChecker
    from spidy.agent.evaluator import TaskEvaluator
    from spidy.agent.goal_manager import GoalManager
    from spidy.agent.observer import Observation, TaskObserver
    from spidy.agent.progress_tracker import ProgressTracker
    from spidy.agent.replanner import Replanner
    from spidy.agent.structured_router import StructuredRouter  # M18
    from spidy.agent.task_logger import AgentTaskLogger
    from spidy.brain.brain import Brain
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)

# Delay between task executions to avoid overwhelming OS (seconds)
_INTER_TASK_DELAY: float = 0.3


class ExecutionLoop:
    """
    Drives autonomous goal execution task by task.

    Parameters
    ----------
    brain:
        The Brain instance for processing each task utterance.
    goal_manager:
        GoalManager to update task and goal state after each step.
    reflection:
        ReflectionEngine for evaluating task outcomes.
    tracker:
        ProgressTracker for publishing progress events.
    bus:
        EventBus for publishing task lifecycle events.
    observer:
        Optional TaskObserver for environment inspection (M16).
    evaluator:
        Optional TaskEvaluator for structured outcome evaluation (M16).
    replanner:
        Optional Replanner for plan revision on soft failures (M16).
    authority:
        Optional TaskAuthorityChecker to gate destructive tasks (M16).
    task_log:
        Optional AgentTaskLogger for structured [AGENT] log lines (M16).
    inter_task_delay:
        Seconds to wait between task executions.
    """

    def __init__(
        self,
        brain: "Brain",
        goal_manager: "GoalManager",
        reflection: ReflectionEngine,
        tracker: "ProgressTracker",
        bus: "EventBus",
        observer: "TaskObserver | None" = None,
        evaluator: "TaskEvaluator | None" = None,
        replanner: "Replanner | None" = None,
        authority: "TaskAuthorityChecker | None" = None,
        task_log: "AgentTaskLogger | None" = None,
        router: "StructuredRouter | None" = None,  # M18
        inter_task_delay: float = _INTER_TASK_DELAY,
    ) -> None:
        self._brain = brain
        self._goal_manager = goal_manager
        self._reflection = reflection
        self._tracker = tracker
        self._bus = bus
        self._observer = observer
        self._evaluator = evaluator
        self._replanner = replanner
        self._authority = authority
        self._task_log = task_log
        self._router = router  # M18
        self._inter_task_delay = inter_task_delay

    # ── Public API ─────────────────────────────────────────────────────────

    async def run(
        self,
        goal: GoalRecord,
        tasks: list[TaskRecord],
        ctx: ExecutionContext,
    ) -> tuple[GoalRecord, list["Observation"]]:
        """
        Execute all tasks for a goal and return (final GoalRecord, observations).

        Parameters
        ----------
        goal:
            The GoalRecord being executed (state=EXECUTING).
        tasks:
            Ordered list of TaskRecords to execute.
        ctx:
            ExecutionContext carrying the cancellation flag and config.

        Returns
        -------
        tuple[GoalRecord, list[Observation]]
            The updated GoalRecord in a terminal state, plus all observations
            collected during execution (for final verification).
        """
        total = len(tasks)
        self._tracker.start(
            goal_id=goal.goal_id,
            total_tasks=total,
            session_id=ctx.session_id,
        )

        log.info(
            "[AGENT] ExecutionLoop: starting | goal='{desc}' | {n} tasks",
            desc=goal.description[:80],
            n=total,
        )

        observations: list["Observation"] = []

        # tasks is a mutable reference — replanning can replace remaining entries
        task_list = list(tasks)
        idx = 0

        while idx < len(task_list):
            task = task_list[idx]

            # ── Cancellation check ─────────────────────────────────────────
            if ctx.cancelled:
                log.info("[AGENT] ExecutionLoop: cancelled before task {i}", i=idx)
                return await self._goal_manager.cancel_goal(goal.goal_id), observations

            # ── Execute task with full pipeline ────────────────────────────
            updated_task, obs, replanned_remaining = await self._execute_task(
                task=task,
                task_index=idx,
                total_tasks=len(task_list),
                ctx=ctx,
                goal_description=goal.description,
                remaining_tasks=task_list[idx + 1:],
            )

            if obs is not None:
                observations.append(obs)

            # ── Handle replanning ──────────────────────────────────────────
            if replanned_remaining is not None:
                if not replanned_remaining:
                    # Replanner says recovery is impossible → abort
                    goal = await self._goal_manager.update_task(goal.goal_id, updated_task)
                    error_msg = f"Task '{updated_task.description}' failed and recovery is impossible."
                    return await self._goal_manager.fail_goal(goal.goal_id, error=error_msg), observations

                # Replace tail of task_list with replanned tasks
                task_list = task_list[:idx + 1] + replanned_remaining
                log.info(
                    "[AGENT] ExecutionLoop: task list revised — now {n} total tasks",
                    n=len(task_list),
                )

            # ── Update task in goal record ─────────────────────────────────
            goal = await self._goal_manager.update_task(goal.goal_id, updated_task)

            # ── Abort on unrecoverable failure ─────────────────────────────
            if updated_task.state == TaskState.FAILED:
                await self._tracker.publish_failed(updated_task.error)
                error_msg = (
                    f"Task '{updated_task.description}' failed: "
                    f"{updated_task.error or updated_task.result_message}"
                )
                return await self._goal_manager.fail_goal(goal.goal_id, error=error_msg), observations

            # ── Early-exit: terminal task succeeded — goal is achieved ──────
            if updated_task.terminal and updated_task.state == TaskState.COMPLETED:
                remaining = task_list[idx + 1:]
                if remaining:
                    log.info(
                        "[AGENT] ExecutionLoop: terminal task succeeded — "
                        "skipping {n} remaining task(s) and completing goal.",
                        n=len(remaining),
                    )
                    for remaining_task in remaining:
                        skipped = remaining_task.mark_skipped(
                            "Goal achieved by a prior terminal task."
                        )
                        goal = await self._goal_manager.update_task(
                            goal.goal_id, skipped
                        )
                break  # exit the task loop — goal completion follows below

            # ── Brief pause between tasks ──────────────────────────────────
            if idx < len(task_list) - 1 and self._inter_task_delay > 0:
                await asyncio.sleep(self._inter_task_delay)

            idx += 1

        # ── All tasks done ─────────────────────────────────────────────────
        await self._tracker.publish_completed()
        summary = self._build_completion_summary(goal)
        log.info(
            "[AGENT] ExecutionLoop: goal complete | '{desc}'",
            desc=goal.description[:80],
        )
        return await self._goal_manager.complete_goal(goal.goal_id, summary=summary), observations

    # ── Internal task execution ────────────────────────────────────────────

    async def _execute_task(
        self,
        task: TaskRecord,
        task_index: int,
        total_tasks: int,
        ctx: ExecutionContext,
        goal_description: str = "",
        remaining_tasks: list[TaskRecord] | None = None,
    ) -> tuple[TaskRecord, "Observation | None", list[TaskRecord] | None]:
        """
        Execute a single task with reflection-driven retry and optional
        observation/evaluation/replanning.

        Returns
        -------
        tuple[TaskRecord, Observation | None, list[TaskRecord] | None]
            - updated TaskRecord in a terminal state
            - Observation (or None if skipped)
            - replanned remaining tasks (None = no replanning, [] = abort)
        """
        from spidy.agent.observer import Observation as ObsType

        max_attempts = ctx.max_task_retries + 1
        observation: "Observation | None" = None

        # ── Authority check BEFORE execution (M16) ─────────────────────────
        authority_label = "T0/SAFE"
        if self._authority is not None:
            authority_label = self._authority.level_label(task)
            if self._authority.requires_confirmation(task):
                if self._task_log:
                    self._task_log.confirmation_required(task, authority_label)
                await self._publish_confirmation_required(task, authority_label)
                # Skip task — user must re-issue after confirming
                skipped = task.mark_skipped(
                    f"Requires confirmation (authority={authority_label}). "
                    "Please confirm and reissue the command."
                )
                if self._task_log:
                    self._task_log.task_skipped(task, task_index, total_tasks,
                                                f"Confirmation required: {authority_label}")
                return skipped, None, None

        for attempt in range(1, max_attempts + 1):
            if ctx.cancelled:
                return task.mark_skipped("Cancelled by user."), None, None

            # Mark as running
            running_task = task.mark_running()

            # ── [AGENT] Task started log ───────────────────────────────────
            if self._task_log:
                self._task_log.task_started(running_task, task_index, total_tasks, authority_label)

            # Publish task started event
            await self._publish_task_started(
                task=running_task,
                task_index=task_index,
                total_tasks=total_tasks,
                authority_level=authority_label,
            )
            await self._tracker.advance(
                task_index=task_index,
                task_description=task.description,
            )

            log.info(
                "[AGENT] ExecutionLoop: task {i}/{n} attempt {a}: '{desc}'",
                i=task_index + 1,
                n=total_tasks,
                a=attempt,
                desc=task.description[:80],
            )

            # -- Structured action path vs Brain.process() fallback ----------
            if task.action and self._router is not None:
                # Inject input_from result and template variables if declared
                resolved_action = self._inject_step_result(task.action, ctx, running_task.task_id)
                # Log structured action details
                self._log_structured_action(running_task, resolved_action)
                response_text, result_success = await self._execute_structured(
                    resolved_action, running_task, ctx.session_id
                )
                # Store rich result for potential later input_from references
                structured_data: dict[str, Any] = {
                    "result": response_text,
                    "text": response_text,
                    "success": result_success,
                }
                if "http" in response_text:
                    import re
                    m_url = re.search(r'https?://[^\s]+', response_text)
                    if m_url:
                        structured_data["url"] = m_url.group(0).rstrip(".,;)'\"")
                if "title" in response_text.lower():
                    import re
                    m_title = re.search(r'title:?\s*["\']?([^"\',\n]+)', response_text, re.IGNORECASE)
                    if m_title:
                        structured_data["title"] = m_title.group(1).strip()

                ctx.add_structured_result(running_task.task_id, structured_data, step_index=task_index)
            else:
                # -- Original Brain.process() path (preserved) -----------------
                utterance = running_task.utterance
                response_text, result_success = await self._call_brain(utterance, ctx.session_id)
                ctx.add_structured_result(running_task.task_id, response_text, step_index=task_index)

            # Check cancellation after Brain returns (long-running tasks)
            if ctx.cancelled:
                return task.mark_skipped("Cancelled mid-task."), None, None

            # ── TaskObserver (M16) ─────────────────────────────────────────
            observation = None
            if self._observer is not None:
                observation = await self._observer.observe(
                    task=running_task,
                    response_text=response_text,
                    skip_for_terminal=task.terminal and attempt == 1,
                )
                if self._task_log:
                    self._task_log.observation(running_task, observation)
                await self._publish_observation(running_task, observation)

            # ── ReflectionEngine ───────────────────────────────────────────
            running_task_with_attempt = TaskRecord(
                task_id=task.task_id,
                goal_id=task.goal_id,
                description=task.description,
                utterance=running_task.utterance,   # always safe (no local var dependency)
                action=task.action,                  # M18: preserve action
                input_from=task.input_from,          # M18: preserve input_from
                state=TaskState.RUNNING,
                attempt=attempt,
                extra=task.extra,
                started_at=running_task.started_at,
            )
            decision, reason = self._reflection.reflect(
                task=running_task_with_attempt,
                response_text=response_text,
                result_success=result_success,
            )

            log.debug(
                "[AGENT] ReflectionEngine: [{decision}] {reason}",
                decision=decision.value,
                reason=reason,
            )

            # Publish reflection event
            await self._publish_reflection(
                task=running_task,
                decision=decision,
                reason=reason,
            )

            # ── TaskEvaluator (M16) ────────────────────────────────────────
            if self._evaluator is not None and observation is not None:
                outcome = self._evaluator.evaluate(
                    task=running_task_with_attempt,
                    observation=observation,
                    reflection_decision=decision,
                    reflection_reason=reason,
                    response_text=response_text,
                )
                self._evaluator.log_outcome(running_task_with_attempt, outcome)
                if self._task_log:
                    self._task_log.evaluation(running_task_with_attempt, outcome)
                await self._publish_evaluation(running_task_with_attempt, outcome)

                # ── Replanning path (M16) ──────────────────────────────────
                if outcome.should_replan and self._replanner is not None:
                    can = self._replanner.can_replan(task.goal_id)
                    if can:
                        if self._task_log:
                            self._task_log.replanning(
                                attempt=self._replanner._replan_counts.get(task.goal_id, 0) + 1,
                                max_attempts=self._replanner._max_attempts,
                                reason=outcome.replan_hint or reason,
                            )
                        await self._publish_replanning(running_task, outcome)
                        revised = await self._replanner.replan(
                            goal_description=goal_description,
                            goal_id=task.goal_id,
                            failed_task=running_task_with_attempt,
                            observation=observation,
                            remaining_tasks=remaining_tasks or [],
                            replan_hint=outcome.replan_hint,
                            llm_client=self._brain._llm,  # noqa: SLF001 — needed for replanning
                        )
                        failed_task = running_task_with_attempt.mark_failed(
                            error=reason, message=response_text
                        )
                        await self._publish_task_failed(task=failed_task, will_retry=False)
                        return failed_task, observation, revised

                # Use evaluator's decision as canonical
                decision = outcome.reflection_decision

            # ── Act on decision ────────────────────────────────────────────
            if decision == ReflectionDecision.CONTINUE:
                completed = running_task_with_attempt.mark_completed(response_text)
                if self._task_log:
                    self._task_log.task_success(completed, task_index, total_tasks)
                await self._publish_task_completed(completed, task_index, total_tasks)
                await self._tracker.task_done(task_index, task.description)
                return completed, observation, None

            elif decision == ReflectionDecision.RETRY:
                if attempt < max_attempts:
                    await self._publish_task_failed(
                        task=running_task_with_attempt,
                        will_retry=True,
                    )
                    if self._task_log:
                        self._task_log.task_failed(
                            running_task_with_attempt, task_index, total_tasks,
                            reason, will_retry=True,
                        )
                    log.debug("[AGENT] Retrying task in 1s...")
                    await asyncio.sleep(1.0)
                    task = running_task_with_attempt.mark_failed(
                        error=f"Attempt {attempt} failed",
                        message=response_text,
                    )
                    task = TaskRecord(
                        task_id=task.task_id,
                        goal_id=task.goal_id,
                        description=task.description,
                        utterance=task.utterance,
                        state=TaskState.PENDING,
                        attempt=attempt,
                        extra=task.extra,
                    )
                    continue  # retry

            elif decision == ReflectionDecision.ALTERNATIVE:
                alt_utterance = self._reflection.build_alternative_utterance(
                    running_task_with_attempt
                )
                log.info(
                    "[AGENT] ExecutionLoop: trying alternative: '{alt}'",
                    alt=alt_utterance[:80],
                )
                alt_response, alt_success = await self._call_brain(alt_utterance, ctx.session_id)
                if alt_success or (alt_response and len(alt_response) > 8):
                    completed = running_task_with_attempt.mark_completed(alt_response)
                    if self._task_log:
                        self._task_log.task_success(completed, task_index, total_tasks)
                    await self._publish_task_completed(completed, task_index, total_tasks)
                    await self._tracker.task_done(task_index, task.description)
                    return completed, observation, None
                # Alternative also failed — fall through to ABORT

            elif decision == ReflectionDecision.ASK_USER:
                await self._publish_clarification(running_task, ctx)
                skipped = running_task_with_attempt.mark_skipped("Awaiting user clarification.")
                if self._task_log:
                    self._task_log.task_skipped(skipped, task_index, total_tasks,
                                                "Awaiting user clarification")
                return skipped, observation, None

            # ── ABORT or all retries exhausted ─────────────────────────────
            failed = running_task_with_attempt.mark_failed(
                error=reason,
                message=response_text,
            )
            if self._task_log:
                self._task_log.task_failed(failed, task_index, total_tasks, reason)
            await self._publish_task_failed(task=failed, will_retry=False)
            return failed, observation, None

        # Exhausted all attempts without return — mark failed
        final_failed = task.mark_failed(error=f"All {max_attempts} attempts exhausted.")
        if self._task_log:
            self._task_log.task_failed(final_failed, task_index, total_tasks,
                                       f"All {max_attempts} attempts exhausted.")
        return final_failed, observation, None

    # ── Brain integration ──────────────────────────────────────────────────

    async def _call_brain(
        self,
        utterance: str,
        session_id: str,
    ) -> tuple[str, bool]:
        """
        Call Brain.process() and return (response_text, success_bool).

        Never raises -- all errors are caught and surfaced as failure tuples.
        """
        try:
            response_text = await self._brain.process(utterance, session_id=session_id)
            success = bool(response_text and len(response_text.strip()) > 4)
            return response_text, success
        except Exception as exc:  # noqa: BLE001
            log.error("[AGENT] ExecutionLoop: Brain.process() raised: {exc}", exc=exc)
            return f"An error occurred: {exc}", False

    # -- M18: Structured execution helpers ------------------------------------

    @staticmethod
    def _inject_step_result(
        action: dict,
        ctx: ExecutionContext,
        current_task_id: str = "",
    ) -> dict:
        """
        Resolve input_from and template variables (${task_id.field}, ${prev.field}, etc.)
        in a structured action by injecting prior task results into action fields.
        """
        resolved = dict(action)
        input_from = action.get("input_from", "")

        # 1. Resolve explicit input_from if declared
        prior_result = ""
        if input_from:
            if input_from in ctx.step_results:
                stored = ctx.step_results[input_from]
                if isinstance(stored, dict):
                    prior_result = str(stored.get("result") or stored.get("text") or stored.get("url") or "")
                else:
                    prior_result = str(stored)
            elif input_from.startswith("<input_from:") or "${" in input_from:
                res = ctx.resolve_variable(input_from, current_task_id)
                prior_result = str(res) if res != input_from else ""
            else:
                prior_result = ""

        skill = (resolved.get("skill") or "").lower()
        act = (resolved.get("action") or "").lower()

        # 2. Interpolate template expressions across string fields
        for field_key in ("text", "query", "target", "url", "input", "expression"):
            val = resolved.get(field_key)
            if isinstance(val, str):
                resolved_val = ctx.resolve_variable(val, current_task_id)
                if isinstance(resolved_val, str) and resolved_val != val:
                    resolved[field_key] = resolved_val

        # 3. If input_from was declared, map it to the primary field if unset or generic
        if prior_result and input_from:
            if skill == "file" and act in ("open_file", "open", "read_file", "read"):
                if not resolved.get("target"):
                    resolved["target"] = prior_result
            elif skill == "browser" and act in ("navigate", "search"):
                if not resolved.get("url") and ("http://" in prior_result or "https://" in prior_result):
                    resolved["url"] = prior_result
                elif not resolved.get("query"):
                    resolved["query"] = prior_result
            elif skill == "desktop" and act in ("type_text", "type", "keyboard_type", "write"):
                if not resolved.get("text") or resolved.get("text") in ("${prev}.text", "the result", "the search result"):
                    resolved["text"] = prior_result
            else:
                if not resolved.get("input"):
                    resolved["input"] = prior_result

        return resolved

    async def _execute_structured(
        self,
        action: dict,
        task: TaskRecord,
        session_id: str,
    ) -> tuple[str, bool]:
        """
        Execute a structured action via StructuredRouter.

        Falls back gracefully to Brain.process(utterance) if the router
        is unavailable or the action is invalid.
        """
        if self._router is None:
            return await self._call_brain(task.utterance, session_id)
        try:
            return await self._router.execute(action, task, session_id)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "[AGENT] ExecutionLoop: StructuredRouter.execute raised: %s", exc
            )
            return await self._call_brain(task.utterance, session_id)

    @staticmethod
    def _log_structured_action(task: TaskRecord, action: dict) -> None:
        """Emit a structured [AGENT] log line for the action contract."""
        parts = []
        for key in ("skill", "action", "target", "query", "url"):
            val = action.get(key, "")
            if val:
                parts.append(f"{key}={val!r}")
        eo = action.get("expected_outcome", "")
        summary = " ".join(parts)
        log.info(
            "[AGENT] Structured task: %s | expected_outcome='%s'",
            summary,
            eo[:80] if eo else "(none)",
        )

    # ── Event publishers ───────────────────────────────────────────────────

    async def _publish_task_started(
        self,
        task: TaskRecord,
        task_index: int,
        total_tasks: int,
        authority_level: str = "safe",
    ) -> None:
        from spidy.agent.events import TaskStartedEvent
        try:
            await self._bus.publish(TaskStartedEvent(
                goal_id=task.goal_id,
                task_id=task.task_id,
                description=task.description,
                task_index=task_index,
                total_tasks=total_tasks,
                attempt=task.attempt,
                authority_level=authority_level,
                session_id="",
            ))
        except Exception:  # noqa: BLE001
            pass

    async def _publish_task_completed(
        self,
        task: TaskRecord,
        task_index: int,
        total_tasks: int,
    ) -> None:
        from spidy.agent.events import TaskCompletedEvent
        try:
            await self._bus.publish(TaskCompletedEvent(
                goal_id=task.goal_id,
                task_id=task.task_id,
                description=task.description,
                result_message=task.result_message,
                task_index=task_index,
                total_tasks=total_tasks,
            ))
        except Exception:  # noqa: BLE001
            pass

    async def _publish_task_failed(
        self,
        task: TaskRecord,
        will_retry: bool,
    ) -> None:
        from spidy.agent.events import TaskFailedEvent
        try:
            await self._bus.publish(TaskFailedEvent(
                goal_id=task.goal_id,
                task_id=task.task_id,
                description=task.description,
                error=task.error,
                attempt=task.attempt,
                will_retry=will_retry,
            ))
        except Exception:  # noqa: BLE001
            pass

    async def _publish_reflection(
        self,
        task: TaskRecord,
        decision: ReflectionDecision,
        reason: str,
    ) -> None:
        from spidy.agent.events import AgentReflectionEvent
        try:
            await self._bus.publish(AgentReflectionEvent(
                goal_id=task.goal_id,
                task_id=task.task_id,
                decision=decision.value,
                reason=reason,
            ))
        except Exception:  # noqa: BLE001
            pass

    async def _publish_clarification(
        self,
        task: TaskRecord,
        ctx: ExecutionContext,
    ) -> None:
        from spidy.agent.events import AgentClarificationEvent
        try:
            await self._bus.publish(AgentClarificationEvent(
                goal_id=task.goal_id,
                task_id=task.task_id,
                question=task.result_message or "Could you clarify what you need?",
                session_id=ctx.session_id,
            ))
        except Exception:  # noqa: BLE001
            pass

    async def _publish_observation(
        self,
        task: TaskRecord,
        observation: "Observation",
    ) -> None:
        from spidy.agent.events import AgentObservationEvent
        try:
            await self._bus.publish(AgentObservationEvent(
                goal_id=task.goal_id,
                task_id=task.task_id,
                task_description=task.description,
                method=observation.method,
                summary=observation.summary,
                success_signal=observation.success_signal,
            ))
        except Exception:  # noqa: BLE001
            pass

    async def _publish_evaluation(
        self,
        task: TaskRecord,
        outcome: "object",
    ) -> None:
        from spidy.agent.events import AgentEvaluationEvent
        try:
            await self._bus.publish(AgentEvaluationEvent(
                goal_id=task.goal_id,
                task_id=task.task_id,
                task_description=task.description,
                success=getattr(outcome, "success", False),
                confidence=getattr(outcome, "confidence", type("C", (), {"value": "unknown"})()).value
                           if hasattr(getattr(outcome, "confidence", None), "value")
                           else str(getattr(outcome, "confidence", "unknown")),
                should_replan=getattr(outcome, "should_replan", False),
                reason=getattr(outcome, "reason", ""),
            ))
        except Exception:  # noqa: BLE001
            pass

    async def _publish_replanning(
        self,
        task: TaskRecord,
        outcome: "object",
    ) -> None:
        from spidy.agent.events import AgentReplanningEvent
        try:
            replanner = self._replanner
            current_count = replanner._replan_counts.get(task.goal_id, 0) + 1 if replanner else 1
            max_att = replanner._max_attempts if replanner else 2
            await self._bus.publish(AgentReplanningEvent(
                goal_id=task.goal_id,
                failed_task_id=task.task_id,
                failed_task_description=task.description,
                attempt=current_count,
                max_attempts=max_att,
                reason=getattr(outcome, "replan_hint", "") or getattr(outcome, "reason", ""),
            ))
        except Exception:  # noqa: BLE001
            pass

    async def _publish_confirmation_required(
        self,
        task: TaskRecord,
        authority_label: str,
    ) -> None:
        from spidy.agent.events import AgentConfirmationRequiredEvent
        try:
            await self._bus.publish(AgentConfirmationRequiredEvent(
                goal_id=task.goal_id,
                task_id=task.task_id,
                task_description=task.description,
                utterance=task.utterance,
                authority_level=authority_label,
                prompt=(
                    f"I need your permission to: {task.description}. "
                    f"This is a {authority_label} action. Do you confirm?"
                ),
            ))
        except Exception:  # noqa: BLE001
            pass

    # ── Response composition ───────────────────────────────────────────────

    @staticmethod
    def _build_completion_summary(goal: GoalRecord) -> str:
        """
        Build a natural-language summary of the completed goal.

        Gathers the result messages of all completed tasks and composes
        a brief, friendly summary.
        """
        completed = [t for t in goal.tasks if t.state == TaskState.COMPLETED]
        skipped = [t for t in goal.tasks if t.state == TaskState.SKIPPED]

        if not completed:
            return f"I've finished working on '{goal.description}', though some steps were skipped."

        task_count = len(completed)
        descriptions = [t.description.lower() for t in completed[:3]]

        if task_count == 1:
            return f"Done! I've completed: {descriptions[0]}."

        if task_count == 2:
            return f"All done! I've {descriptions[0]} and {descriptions[1]}."

        # 3+ tasks
        listed = ", ".join(descriptions[:-1])
        last = descriptions[-1]
        summary = f"All done! I've {listed}, and {last}."

        if skipped:
            summary += f" ({len(skipped)} step(s) were skipped or required clarification.)"

        return summary
