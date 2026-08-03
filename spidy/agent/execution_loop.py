"""
ExecutionLoop — Autonomous Agent Execution Engine (Milestone 13)
================================================================
The core loop that drives autonomous goal execution.

Pipeline (per-goal)
--------------------
  tasks = TaskDecomposer.decompose(goal)
  for task in tasks:
      if ctx.cancelled: break
      ProgressTracker.advance(task)
      response = Brain.process(task.utterance)
      decision, reason = ReflectionEngine.reflect(task, response)
      match decision:
          CONTINUE    → mark task complete, next
          RETRY       → retry task (up to max_retries)
          ALTERNATIVE → try alternative utterance, then CONTINUE or ABORT
          ASK_USER    → publish clarification event, pause (best-effort)
          ABORT       → fail the goal

Design
------
- The loop calls Brain.process() for each task — all Brain capabilities
  (skills, LLM, retry, proactive checks) remain fully active.
- Each task result goes through ReflectionEngine before proceeding.
- Cancellation is checked at the top of every task iteration.
- Self-recovery: ALTERNATIVE tries a rephrased utterance before giving up.
- Natural progress events are published via ProgressTracker.

Usage
-----
    loop = ExecutionLoop(
        brain=brain,
        reflection=reflection_engine,
        tracker=progress_tracker,
        bus=bus,
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
    from spidy.agent.goal_manager import GoalManager
    from spidy.agent.progress_tracker import ProgressTracker
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
        inter_task_delay: float = _INTER_TASK_DELAY,
    ) -> None:
        self._brain = brain
        self._goal_manager = goal_manager
        self._reflection = reflection
        self._tracker = tracker
        self._bus = bus
        self._inter_task_delay = inter_task_delay

    # ── Public API ─────────────────────────────────────────────────────────

    async def run(
        self,
        goal: GoalRecord,
        tasks: list[TaskRecord],
        ctx: ExecutionContext,
    ) -> GoalRecord:
        """
        Execute all tasks for a goal and return the final GoalRecord.

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
        GoalRecord
            The updated GoalRecord in a terminal state.
        """
        total = len(tasks)
        self._tracker.start(
            goal_id=goal.goal_id,
            total_tasks=total,
            session_id=ctx.session_id,
        )

        log.info(
            "ExecutionLoop: starting | goal='{desc}' | {n} tasks",
            desc=goal.description[:80],
            n=total,
        )

        for idx, task in enumerate(tasks):
            # ── Cancellation check ─────────────────────────────────────────
            if ctx.cancelled:
                log.info("ExecutionLoop: cancelled before task {i}", i=idx)
                return await self._goal_manager.cancel_goal(goal.goal_id)

            # ── Execute task with reflection ───────────────────────────────
            updated_task = await self._execute_task(
                task=task,
                task_index=idx,
                total_tasks=total,
                ctx=ctx,
            )

            # ── Update task in goal record ─────────────────────────────────
            goal = await self._goal_manager.update_task(goal.goal_id, updated_task)

            # ── Abort on unrecoverable failure ─────────────────────────────
            if updated_task.state == TaskState.FAILED:
                await self._tracker.publish_failed(updated_task.error)
                error_msg = (
                    f"Task '{updated_task.description}' failed: {updated_task.error or updated_task.result_message}"
                )
                return await self._goal_manager.fail_goal(goal.goal_id, error=error_msg)

            # ── Brief pause between tasks ──────────────────────────────────
            if idx < total - 1 and self._inter_task_delay > 0:
                await asyncio.sleep(self._inter_task_delay)

        # ── All tasks done ─────────────────────────────────────────────────
        await self._tracker.publish_completed()
        summary = self._build_completion_summary(goal)
        log.info(
            "ExecutionLoop: goal complete | '{desc}'",
            desc=goal.description[:80],
        )
        return await self._goal_manager.complete_goal(goal.goal_id, summary=summary)

    # ── Internal task execution ────────────────────────────────────────────

    async def _execute_task(
        self,
        task: TaskRecord,
        task_index: int,
        total_tasks: int,
        ctx: ExecutionContext,
    ) -> TaskRecord:
        """
        Execute a single task with reflection-driven retry.

        Returns the TaskRecord in a terminal state (COMPLETED, FAILED, or SKIPPED).
        """
        max_attempts = ctx.max_task_retries + 1

        for attempt in range(1, max_attempts + 1):
            if ctx.cancelled:
                return task.mark_skipped("Cancelled by user.")

            # Mark as running
            running_task = task.mark_running()

            # Publish task started event
            await self._publish_task_started(
                task=running_task,
                task_index=task_index,
                total_tasks=total_tasks,
            )
            await self._tracker.advance(
                task_index=task_index,
                task_description=task.description,
            )

            log.info(
                "ExecutionLoop: task {i}/{n} attempt {a}: '{desc}'",
                i=task_index + 1,
                n=total_tasks,
                a=attempt,
                desc=task.description[:80],
            )

            # ── Call Brain.process() ───────────────────────────────────────
            utterance = running_task.utterance
            response_text, result_success = await self._call_brain(utterance, ctx.session_id)

            # ── ReflectionEngine evaluation ────────────────────────────────
            running_task_with_attempt = TaskRecord(
                task_id=task.task_id,
                goal_id=task.goal_id,
                description=task.description,
                utterance=utterance,
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
                "ReflectionEngine: [{decision}] {reason}",
                decision=decision.value,
                reason=reason,
            )

            # Publish reflection event
            await self._publish_reflection(
                task=running_task,
                decision=decision,
                reason=reason,
            )

            # ── Act on decision ────────────────────────────────────────────
            if decision == ReflectionDecision.CONTINUE:
                completed = running_task_with_attempt.mark_completed(response_text)
                await self._publish_task_completed(completed, task_index, total_tasks)
                await self._tracker.task_done(task_index, task.description)
                return completed

            elif decision == ReflectionDecision.RETRY:
                if attempt < max_attempts:
                    await self._publish_task_failed(
                        task=running_task_with_attempt,
                        will_retry=True,
                    )
                    log.debug("Retrying task in 1s...")
                    await asyncio.sleep(1.0)
                    # Update task with failed attempt for next loop
                    task = running_task_with_attempt.mark_failed(
                        error=f"Attempt {attempt} failed",
                        message=response_text,
                    )
                    # Reset to pending-like state for retry
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
                    "ExecutionLoop: trying alternative: '{alt}'",
                    alt=alt_utterance[:80],
                )
                alt_response, alt_success = await self._call_brain(alt_utterance, ctx.session_id)
                if alt_success or (alt_response and len(alt_response) > 8):
                    completed = running_task_with_attempt.mark_completed(alt_response)
                    await self._publish_task_completed(completed, task_index, total_tasks)
                    await self._tracker.task_done(task_index, task.description)
                    return completed
                # Alternative also failed — fall through to ABORT

            elif decision == ReflectionDecision.ASK_USER:
                # Publish clarification event and skip (best-effort; can't block loop)
                await self._publish_clarification(running_task, ctx)
                # Treat as skipped (user can follow up with a new command)
                return running_task_with_attempt.mark_skipped("Awaiting user clarification.")

            # ── ABORT or all retries exhausted ─────────────────────────────
            failed = running_task_with_attempt.mark_failed(
                error=reason,
                message=response_text,
            )
            await self._publish_task_failed(task=failed, will_retry=False)
            return failed

        # Exhausted all attempts without return — mark failed
        return task.mark_failed(error=f"All {max_attempts} attempts exhausted.")

    # ── Brain integration ──────────────────────────────────────────────────

    async def _call_brain(
        self,
        utterance: str,
        session_id: str,
    ) -> tuple[str, bool]:
        """
        Call Brain.process() and return (response_text, success_bool).

        Never raises — all errors are caught and surfaced as failure tuples.
        """
        try:
            response_text = await self._brain.process(utterance, session_id=session_id)
            # Determine success heuristically from response
            success = bool(response_text and len(response_text.strip()) > 4)
            return response_text, success
        except Exception as exc:  # noqa: BLE001
            log.error("ExecutionLoop: Brain.process() raised: {exc}", exc=exc)
            return f"An error occurred: {exc}", False

    # ── Event publishers ───────────────────────────────────────────────────

    async def _publish_task_started(
        self,
        task: TaskRecord,
        task_index: int,
        total_tasks: int,
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
