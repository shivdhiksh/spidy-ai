"""
AutonomousAgent — Goal-Oriented AI Agent (Milestone 13)
========================================================
The top-level façade for Spidy's autonomous agent capabilities.

The AutonomousAgent wires together all M13 components and exposes a
simple API:

    response = await agent.run_goal("Create a Flask project")
    cancelled = await agent.cancel_current_goal()

Internally, it orchestrates:
  1. GoalManager     — goal lifecycle (create → plan → execute → complete)
  2. TaskDecomposer  — break goal into ordered tasks
  3. ExecutionLoop   — run each task via Brain.process()
  4. ReflectionEngine — evaluate each result and guide next step
  5. ProgressTracker — publish real-time progress events
  6. ResponseComposer — build the natural language final response

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

from spidy.agent.execution_loop import ExecutionLoop
from spidy.agent.goal_manager import GoalManager
from spidy.agent.progress_tracker import ProgressTracker
from spidy.agent.reflection_engine import ReflectionEngine
from spidy.agent.task_decomposer import TaskDecomposer
from spidy.agent.types import ExecutionContext, GoalState
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
        Optional LLM client for intelligent task decomposition.
        Falls back to heuristic decomposition when None.
    max_task_retries:
        Maximum retry attempts per task before moving on or aborting.
    inter_task_delay:
        Seconds to pause between tasks (avoids overwhelming OS).
    """

    def __init__(
        self,
        brain: "Brain",
        bus: "EventBus",
        llm_client: "BaseLLMClient | None" = None,
        max_task_retries: int = _DEFAULT_MAX_TASK_RETRIES,
        inter_task_delay: float = 0.3,
    ) -> None:
        self._brain = brain
        self._bus = bus
        self._llm_client = llm_client
        self._max_task_retries = max_task_retries

        # Instantiate all M13 components
        self._goal_manager = GoalManager(bus=bus)
        self._decomposer = TaskDecomposer()
        self._reflection = ReflectionEngine(max_retries=max_task_retries)
        self._tracker = ProgressTracker(bus=bus)
        self._loop = ExecutionLoop(
            brain=brain,
            goal_manager=self._goal_manager,
            reflection=self._reflection,
            tracker=self._tracker,
            bus=bus,
            inter_task_delay=inter_task_delay,
        )

        # Active execution context (for cancellation)
        self._active_ctx: ExecutionContext | None = None

    # ── Primary API ────────────────────────────────────────────────────────

    async def run_goal(
        self,
        goal_description: str,
        session_id: str | None = None,
    ) -> str:
        """
        Execute an autonomous multi-step goal.

        This is the primary entry point for M13 autonomous execution.

        Pipeline:
          create goal → publish planning event → decompose → execute tasks
          → reflect after each → build completion response

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
            "AutonomousAgent: run_goal '{desc}' | session={sid}",
            desc=goal_description[:80],
            sid=sid,
        )

        # Publish brain-level goal event for backward-compat subscribers
        await self._publish_brain_goal_started(goal_description, sid)

        try:
            # 1. Create goal record
            goal = await self._goal_manager.create_goal(
                description=goal_description,
                session_id=sid,
            )

            # 2. Transition to PLANNING
            goal = await self._goal_manager.begin_planning(goal.goal_id)
            await self._tracker.publish_planning()

            # 3. Decompose goal into tasks
            tasks = await self._decomposer.decompose(
                goal_description=goal_description,
                goal_id=goal.goal_id,
                llm_client=self._llm_client,
                session_id=sid,
            )

            # 4. Transition to EXECUTING
            goal = await self._goal_manager.begin_execution(
                goal_id=goal.goal_id,
                tasks=tasks,
            )

            # 5. Build execution context
            ctx = ExecutionContext(
                goal=goal,
                session_id=sid,
                max_task_retries=self._max_task_retries,
            )
            self._active_ctx = ctx

            # 6. Run execution loop
            final_goal = await self._loop.run(
                goal=goal,
                tasks=tasks,
                ctx=ctx,
            )
            self._active_ctx = None

            # 7. Build response
            return self._compose_response(final_goal)

        except RuntimeError as exc:
            # Another goal is active, or state machine error
            log.error("AutonomousAgent: {exc}", exc=exc)
            return str(exc)

        except Exception as exc:  # noqa: BLE001
            log.error("AutonomousAgent: unexpected error: {exc}", exc=exc)
            return (
                f"I ran into an unexpected problem while working on '{goal_description}'. "
                f"Details: {exc}"
            )

    async def cancel_current_goal(self) -> bool:
        """
        Cancel the currently running goal.

        Returns
        -------
        bool
            True if a goal was cancelled, False if no goal was active.
        """
        if self._active_ctx is None:
            log.debug("AutonomousAgent: cancel requested but no active goal.")
            return False

        ctx = self._active_ctx
        ctx.cancel()  # Signal the execution loop to stop

        goal = self._goal_manager.active_goal
        if goal:
            await self._goal_manager.cancel_goal(goal.goal_id)
            log.info(
                "AutonomousAgent: goal cancelled: '{desc}'",
                desc=goal.description[:80],
            )
            await self._publish_brain_goal_cancelled(goal.description, goal.session_id)

        self._active_ctx = None
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
        if goal.state == GoalState.COMPLETED:
            if goal.summary:
                return goal.summary
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
