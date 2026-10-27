"""
AutonomousAgent — Goal-Oriented AI Agent (Milestone 16 upgrade)
===============================================================
The top-level façade for Spidy's autonomous agent capabilities.

The AutonomousAgent wires together all M13/M16 components and exposes a
simple API:

    response = await agent.run_goal("Create a Flask project")
    cancelled = await agent.cancel_current_goal()

Internally, it orchestrates:
  1. GoalManager       — goal lifecycle (create → plan → execute → complete)
  2. TaskDecomposer    — break goal into ordered tasks
  3. ExecutionLoop     — run each task via Brain.process()
     3a. TaskAuthorityChecker — gate destructive tasks before execution [M16]
     3b. TaskObserver         — inspect environment after each task [M16]
     3c. ReflectionEngine     — lexical success/failure evaluation
     3d. TaskEvaluator        — structured outcome with confidence [M16]
     3e. Replanner            — revise plan on soft failures [M16]
  4. ProgressTracker   — publish real-time progress events
  5. GoalVerifier      — confirm final outcome after completion [M16]
  6. AgentTaskLogger   — emit [AGENT] structured log lines [M16]

Context awareness
-----------------
The agent uses all available Brain capabilities through Brain.process():
  - Conversation history (ConversationManager)
  - Memory Engine (if configured)
  - Knowledge Engine (if configured)
  - Vision Engine (if configured)
  - Skills (all registered skills)
  - LLM (for open-ended steps)

No duplication: skill execution, retry, and proactive checks remain
entirely inside the Brain pipeline.

Long-task support
-----------------
- Each task publishes AgentProgressEvent (picked up by UI)
- Cancellation via cancel_current_goal() sets a flag checked every iteration
- Users can interrupt by cancelling via voice/UI while the loop runs

Usage
-----
    agent = AutonomousAgent(
        brain=brain,
        bus=bus,
        llm_client=llm_client,  # optional — for LLM-based decomposition
    )

    # Start a goal (non-blocking if run in asyncio task)
    response = await agent.run_goal("Create a Flask project")
    # → "All done! I've created the project folder, installed Flask,
    #    and opened it in VS Code."

    # Cancel a running goal
    await agent.cancel_current_goal()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spidy.agent.authority import TaskAuthorityChecker
from spidy.agent.evaluator import TaskEvaluator
from spidy.agent.execution_loop import ExecutionLoop
from spidy.agent.goal_manager import GoalManager
from spidy.agent.observer import TaskObserver
from spidy.agent.progress_tracker import ProgressTracker
from spidy.agent.reflection_engine import ReflectionEngine
from spidy.agent.replanner import Replanner
from spidy.agent.structured_router import StructuredRouter  # M18
from spidy.agent.task_decomposer import TaskDecomposer
from spidy.agent.task_logger import AgentTaskLogger
from spidy.agent.types import ExecutionContext, GoalContext, GoalState, TaskState
from spidy.agent.verifier import GoalVerifier
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.agent.types import GoalRecord
    from spidy.brain.brain import Brain
    from spidy.core.event_bus import EventBus
    from spidy.llm.client import BaseLLMClient

log = get_logger(__name__)

# Default configuration
_DEFAULT_MAX_TASK_RETRIES = 2


class AutonomousAgent:
    """
    Goal-oriented autonomous AI agent for Spidy.

    Parameters
    ----------
    brain:
        The Brain instance — all task execution goes through Brain.process().
    bus:
        Application EventBus for publishing agent lifecycle events.
    llm_client:
        Optional LLM client for intelligent task decomposition + replanning.
        Falls back to heuristic decomposition when None.
    max_task_retries:
        Maximum retry attempts per task before moving on or aborting.
    inter_task_delay:
        Seconds to pause between tasks (avoids overwhelming OS).
    observation_enabled:
        Enable TaskObserver for environment inspection after each task.
        Default True — disable only in unit tests or low-resource scenarios.
    replan_enabled:
        Enable Replanner for plan revision on soft failures.
        Default True.
    verify_enabled:
        Enable GoalVerifier for final goal verification.
        Default True.
    vision_enabled:
        Enable screenshot-based visual observation (requires VisionInterface).
        Default False (performance-sensitive).
    """

    def __init__(
        self,
        brain: "Brain",
        bus: "EventBus",
        llm_client: "BaseLLMClient | None" = None,
        max_task_retries: int = _DEFAULT_MAX_TASK_RETRIES,
        inter_task_delay: float = 0.3,
        observation_enabled: bool = True,
        replan_enabled: bool = True,
        verify_enabled: bool = True,
        vision_enabled: bool = False,
        goal_manager: "GoalManager | None" = None,
        decomposer: "TaskDecomposer | None" = None,
        router: "StructuredRouter | None" = None,
    ) -> None:
        self._brain = brain
        self._bus = bus
        self._llm_client = llm_client
        self._max_task_retries = max_task_retries
        self._verify_enabled = verify_enabled

        # ── M13 components (preserved) ─────────────────────────────────────
        self._goal_manager = goal_manager or GoalManager(bus=bus)
        self._decomposer = decomposer or TaskDecomposer()
        self._reflection = ReflectionEngine(max_retries=max_task_retries)
        self._tracker = ProgressTracker(bus=bus)

        # ── M16 components (new) ───────────────────────────────────────────
        self._task_log = AgentTaskLogger()

        vision = getattr(brain, "_vision", None)
        observer = TaskObserver(
            vision=vision,
            vision_enabled=vision_enabled,
        ) if observation_enabled else None

        evaluator = TaskEvaluator(
            replan_on_soft_failures=replan_enabled,
        ) if observation_enabled else None

        replanner = Replanner(
            max_attempts=2,
        ) if replan_enabled else None

        authority = TaskAuthorityChecker()

        self._observer = observer
        self._evaluator = evaluator
        self._replanner = replanner
        self._authority = authority
        self._verifier = GoalVerifier() if verify_enabled else None

        # M18: Structured action router
        router = router or StructuredRouter(brain=brain)

        # Wire all components into ExecutionLoop
        self._loop = ExecutionLoop(
            brain=brain,
            goal_manager=self._goal_manager,
            reflection=self._reflection,
            tracker=self._tracker,
            bus=bus,
            observer=observer,
            evaluator=evaluator,
            replanner=replanner,
            authority=authority,
            task_log=self._task_log,
            router=router,  # M18
            inter_task_delay=inter_task_delay,
        )

        # Active execution context (for cancellation)
        self._active_ctx: ExecutionContext | None = None
        self._waiting_ctx: ExecutionContext | None = None

    # ── Primary API ────────────────────────────────────────────────────────

    async def run_goal(
        self,
        goal_description: str,
        session_id: str | None = None,
    ) -> str:
        """
        Execute an autonomous multi-step goal.

        Pipeline:
          create goal → [AGENT] log → decompose → execute tasks
          → observe after each → evaluate → replan if needed
          → verify completion → build response

        Parameters
        ----------
        goal_description:
            The high-level user goal (e.g. "Create a Flask project").
        session_id:
            Brain session ID. Defaults to the Brain's current session.

        Returns
        -------
        str
            A natural language response describing what was accomplished.
            Never raises — errors are surfaced in the response text.
        """
        sid = session_id or self._brain.session_id

        log.info(
            "[AGENT] AutonomousAgent: run_goal '{desc}' | session={sid}",
            desc=goal_description[:80],
            sid=sid,
        )

        # Publish brain-level goal event for backward-compat subscribers
        await self._publish_brain_goal_started(goal_description, sid)

        try:
            import time
            t0 = time.monotonic()

            # 1. Create goal record
            goal = await self._goal_manager.create_goal(
                description=goal_description,
                session_id=sid,
            )
            self._task_log.goal_created(goal)

            # 2. Transition to PLANNING
            goal = await self._goal_manager.begin_planning(goal.goal_id)
            await self._tracker.publish_planning()

            # 3. Decompose goal into tasks
            t_plan_start = time.monotonic()
            tasks = await self._decomposer.decompose(
                goal_description=goal_description,
                goal_id=goal.goal_id,
                llm_client=self._llm_client,
                session_id=sid,
            )
            planning_ms = (time.monotonic() - t_plan_start) * 1000.0
            self._task_log.plan_generated(goal, tasks)

            # 4. Initialize GoalContext with complete persistent state
            goal_context = GoalContext(
                goal_id=goal.goal_id,
                original_goal=goal_description,
                normalized_goal=goal_description.strip(),
                original_transcript=goal_description,
                normalized_transcript=goal_description.strip(),
                tasks=tasks,
                metrics={"planning_ms": planning_ms},
            )

            # 5. Transition to EXECUTING
            goal = await self._goal_manager.begin_execution(
                goal_id=goal.goal_id,
                tasks=tasks,
            )

            # 6. Build execution context
            ctx = ExecutionContext(
                goal=goal,
                session_id=sid,
                max_task_retries=self._max_task_retries,
                goal_context=goal_context,
            )
            self._active_ctx = ctx

            # 7. Run execution loop (now returns observations too)
            t_exec_start = time.monotonic()
            final_goal, observations = await self._loop.run(
                goal=goal,
                tasks=tasks,
                ctx=ctx,
            )
            goal_context.metrics["execution_ms"] = (time.monotonic() - t_exec_start) * 1000.0

            if final_goal.state == GoalState.WAITING_FOR_USER:
                self._waiting_ctx = ctx
                self._active_ctx = None
                return self._compose_response(final_goal)

            self._waiting_ctx = None
            self._active_ctx = None

            # 8. Final verification (M16)
            if self._verifier is not None and final_goal.state == GoalState.COMPLETED:
                t_ver_start = time.monotonic()
                verification = await self._verifier.verify(
                    goal=final_goal,
                    observations=observations,
                    llm_client=self._llm_client,
                    goal_context=goal_context,
                )
                goal_context.metrics["verification_ms"] = (time.monotonic() - t_ver_start) * 1000.0
                goal_context.verification_state = "verified" if verification.verified else "partial"
                self._task_log.verification(verification)
                await self._publish_goal_verified(final_goal, verification)

                # If verification returned a detailed summary, record it on goal
                if verification.summary:
                    final_goal = final_goal.with_state(final_goal.state, summary=verification.summary)

                # On goal completion, reset replanner counter
                if self._replanner is not None:
                    self._replanner.reset(final_goal.goal_id)

            goal_context.metrics["total_ms"] = (time.monotonic() - t0) * 1000.0

            # Log goal lifecycle event
            if final_goal.state == GoalState.COMPLETED:
                self._task_log.goal_complete(final_goal)
            elif final_goal.state == GoalState.FAILED:
                self._task_log.goal_failed(final_goal, final_goal.error or "unknown")
            elif final_goal.state == GoalState.CANCELLED:
                self._task_log.goal_cancelled(final_goal)

            # 9. Build response
            return self._compose_response(final_goal)

        except RuntimeError as exc:
            # Another goal is active, or state machine error
            log.error("[AGENT] AutonomousAgent: {exc}", exc=exc)
            return str(exc)

        except Exception as exc:  # noqa: BLE001
            log.error("[AGENT] AutonomousAgent: unexpected error: {exc}", exc=exc)
            return (
                f"I ran into an unexpected problem while working on '{goal_description}'. "
                f"Details: {exc}"
            )

    def has_waiting_goal(self) -> bool:
        """True if a goal is currently paused waiting for user clarification."""
        return (
            self._waiting_ctx is not None
            or (self.active_goal is not None and self.active_goal.state == GoalState.WAITING_FOR_USER)
        )

    async def resume_with_user_input(self, user_input: str) -> str:
        """
        Resume a goal that is paused in WAITING_FOR_USER using the provided user answer.
        """
        if not self.has_waiting_goal():
            return "No goal is currently waiting for clarification."

        ctx = self._waiting_ctx
        goal = self.active_goal
        if ctx is None or goal is None:
            return "No active goal context found to resume."

        waiting_task_id = ctx.waiting_task_id
        log.info(
            "[AGENT] Resuming goal '{desc}' with user answer: '{ans}'",
            desc=goal.description[:60],
            ans=user_input[:60],
        )

        # Bind the user's input to the waiting task in context
        ctx.add_structured_result(
            waiting_task_id,
            {"result": user_input, "text": user_input, "query": user_input},
        )
        if waiting_task_id not in ctx.step_results:
            ctx.step_results[waiting_task_id] = user_input

        # Mark the waiting task as completed with user's answer
        completed_task = None
        new_tasks = []
        resume_index = 0
        for i, t in enumerate(goal.tasks):
            if t.task_id == waiting_task_id or t.state == TaskState.WAITING_FOR_USER:
                completed_task = t.mark_completed(user_input)
                new_tasks.append(completed_task)
                resume_index = i + 1
            else:
                new_tasks.append(t)

        if completed_task is not None:
            goal = await self._goal_manager.update_task(goal.goal_id, completed_task)
        goal = await self._goal_manager.resume_goal(goal.goal_id)

        self._waiting_ctx = None
        self._active_ctx = ctx

        # If all tasks are now done:
        if resume_index >= len(new_tasks):
            summary = self._loop._build_completion_summary(goal)
            completed_goal = await self._goal_manager.complete_goal(goal.goal_id, summary=summary)
            self._active_ctx = None
            return self._compose_response(completed_goal)

        # Continue execution with remaining tasks
        final_goal, observations = await self._loop.run(
            goal=goal,
            tasks=new_tasks,
            ctx=ctx,
        )
        if final_goal.state == GoalState.WAITING_FOR_USER:
            self._waiting_ctx = ctx
            self._active_ctx = None
            return self._compose_response(final_goal)

        self._active_ctx = None
        return self._compose_response(final_goal)

    async def cancel_current_goal(self) -> bool:
        """
        Cancel the currently running goal.

        Returns
        -------
        bool
            True if a goal was cancelled, False if no goal was active.
        """
        if self._active_ctx is None and self._waiting_ctx is None:
            log.debug("[AGENT] AutonomousAgent: cancel requested but no active goal.")
            return False

        ctx = self._active_ctx or self._waiting_ctx
        if ctx:
            ctx.cancel()  # Signal the execution loop to stop

        goal = self._goal_manager.active_goal
        if goal:
            await self._goal_manager.cancel_goal(goal.goal_id)
            log.info(
                "[AGENT] AutonomousAgent: goal cancelled: '{desc}'",
                desc=goal.description[:80],
            )
            await self._publish_brain_goal_cancelled(goal.description, goal.session_id)
            if self._replanner is not None:
                self._replanner.reset(goal.goal_id)

        self._active_ctx = None
        self._waiting_ctx = None
        return True

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def active_goal(self) -> "GoalRecord | None":
        """The currently active GoalRecord, or None."""
        return self._goal_manager.active_goal

    @property
    def history(self) -> "list[GoalRecord]":
        """All completed/failed/cancelled goals (most recent last)."""
        return self._goal_manager.get_history()

    @property
    def is_running(self) -> bool:
        """True if a goal is currently being executed."""
        return self._active_ctx is not None

    @property
    def goal_manager(self) -> GoalManager:
        """Direct access to the GoalManager (for testing/introspection)."""
        return self._goal_manager

    # ── Response composition ───────────────────────────────────────────────

    def _compose_response(self, goal: "GoalRecord") -> str:
        """
        Build the final user-facing response from the completed goal.

        Uses the goal's summary if available, or constructs a response
        from the terminal state and task results.
        """
        if goal.state == GoalState.WAITING_FOR_USER:
            for t in goal.tasks:
                if t.state == TaskState.WAITING_FOR_USER:
                    return t.result_message or "Could you please clarify how you'd like to proceed?"
            return "Could you please clarify how you'd like to proceed?"

        if goal.state == GoalState.COMPLETED:
            if goal.summary:
                return goal.summary
            if goal.tasks and len(goal.tasks) > 0:
                last_msg = goal.tasks[-1].result_message
                if last_msg:
                    return last_msg
            return (
                f"I've finished working on '{goal.description}'. "
                f"All {goal.completed_task_count} steps completed successfully."
            )

        if goal.state == GoalState.CANCELLED:
            done = goal.completed_task_count
            total = goal.task_count
            return (
                f"I've stopped working on '{goal.description}'. "
                f"Completed {done} of {total} steps before cancellation."
            )

        if goal.state == GoalState.FAILED:
            error = goal.error or "an unexpected problem"
            done = goal.completed_task_count
            total = goal.task_count
            return (
                f"I wasn't able to complete '{goal.description}'. "
                f"I completed {done} of {total} steps before running into {error}. "
                "You might want to try a simpler command or check if the required apps are available."
            )

        # Fallback (shouldn't happen in normal flow)
        return f"I finished working on '{goal.description}'."

    # ── Event publishing ───────────────────────────────────────────────────

    async def _publish_brain_goal_started(
        self,
        description: str,
        session_id: str,
    ) -> None:
        """Publish BrainGoalStartedEvent for backward-compat subscribers."""
        try:
            from spidy.brain.events import BrainGoalStartedEvent
            await self._bus.publish(BrainGoalStartedEvent(
                session_id=session_id,
                goal_description=description,
            ))
        except Exception:  # noqa: BLE001
            pass

    async def _publish_brain_goal_cancelled(
        self,
        description: str,
        session_id: str,
    ) -> None:
        """Publish BrainGoalCancelledEvent for backward-compat subscribers."""
        try:
            from spidy.brain.events import BrainGoalCancelledEvent
            await self._bus.publish(BrainGoalCancelledEvent(
                session_id=session_id,
                goal_description=description,
            ))
        except Exception:  # noqa: BLE001
            pass

    async def _publish_goal_verified(
        self,
        goal: "GoalRecord",
        result: "object",
    ) -> None:
        """Publish GoalVerifiedEvent after verification."""
        try:
            from spidy.agent.events import GoalVerifiedEvent
            await self._bus.publish(GoalVerifiedEvent(
                goal_id=goal.goal_id,
                description=goal.description,
                verified=getattr(result, "verified", True),
                confidence=getattr(result, "confidence", "medium"),
                summary=getattr(result, "summary", ""),
                method=getattr(result, "method", "task_results"),
                session_id=goal.session_id,
            ))
        except Exception:  # noqa: BLE001
            pass
