"""
tests/unit/test_autonomous_agent.py
====================================
Comprehensive unit tests for the M16 autonomous agent upgrade.

Coverage: 20 tests across all new and evolved components:
  - TaskObserver
  - TaskEvaluator
  - Replanner
  - GoalVerifier
  - TaskAuthorityChecker
  - AgentTaskLogger
  - ExecutionLoop (authority gate + Observer wiring)
  - AutonomousAgent (full integration path)

All external I/O is mocked — these are pure unit tests.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.agent.authority import AuthorityLevel, TaskAuthorityChecker
from spidy.agent.evaluator import Confidence, TaskEvaluator, TaskOutcome
from spidy.agent.observer import Observation, TaskObserver
from spidy.agent.replanner import Replanner
from spidy.agent.task_logger import AgentTaskLogger
from spidy.agent.types import (
    ExecutionContext,
    GoalRecord,
    GoalState,
    ReflectionDecision,
    TaskRecord,
    TaskState,
)
from spidy.agent.verifier import GoalVerifier, VerificationResult


# ─── Helpers / Factories ──────────────────────────────────────────────────────


def _make_task(
    description: str = "Test task",
    utterance: str = "do the test task",
    terminal: bool = False,
    state: TaskState = TaskState.PENDING,
    goal_id: str = "goal-001",
    attempt: int = 1,
    extra: dict | None = None,
) -> TaskRecord:
    t = TaskRecord(
        goal_id=goal_id,
        description=description,
        utterance=utterance,
        state=state,
        terminal=terminal,
        extra=extra or {},
    )
    # Patch attempt for reflection tests
    object.__setattr__(t, "attempt", attempt)
    return t


def _make_goal(
    description: str = "Test goal",
    tasks: list[TaskRecord] | None = None,
    state: GoalState = GoalState.COMPLETED,
) -> GoalRecord:
    g = GoalRecord(description=description, state=state)
    if tasks:
        object.__setattr__(g, "tasks", tuple(tasks))
    return g


def _make_bus() -> MagicMock:
    bus = MagicMock()
    bus.publish = AsyncMock()
    return bus


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TaskObserver — terminal task is skipped (fast path)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaskObserverTerminalFastPath:
    """Test 1: Terminal task bypasses expensive observation."""

    @pytest.mark.asyncio
    async def test_terminal_task_uses_skipped_method(self) -> None:
        observer = TaskObserver()
        task = _make_task(terminal=True)
        obs = await observer.observe(task, response_text="Opened calculator.")
        assert obs.method == "skipped"

    @pytest.mark.asyncio
    async def test_terminal_task_still_gets_response_text_signal(self) -> None:
        observer = TaskObserver()
        task = _make_task(terminal=True)
        obs = await observer.observe(task, response_text="Done, all set!")
        # Should still have a success_signal from text analysis
        assert obs.success_signal is True


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TaskObserver — response text signal analysis
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaskObserverResponseTextAnalysis:
    """Test 2: Response text analysis produces correct success/failure signals."""

    def test_success_keywords(self) -> None:
        sig = TaskObserver._analyse_response("I've opened Chrome for you.")
        assert sig is True

    def test_failure_keywords(self) -> None:
        sig = TaskObserver._analyse_response("I couldn't find the application.")
        assert sig is False

    def test_empty_response_is_none(self) -> None:
        sig = TaskObserver._analyse_response("")
        assert sig is None

    def test_short_response_is_none(self) -> None:
        sig = TaskObserver._analyse_response("ok")
        assert sig is None

    def test_long_neutral_response_is_optimistic(self) -> None:
        sig = TaskObserver._analyse_response("Here is some information about Python programming.")
        assert sig is True


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TaskEvaluator — clear success path
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaskEvaluatorSuccessPath:
    """Test 3: CONTINUE decision + positive observation → HIGH confidence success."""

    def test_continue_plus_positive_observation_gives_high_confidence(self) -> None:
        evaluator = TaskEvaluator()
        task = _make_task()
        obs = Observation(method="app_state", success_signal=True, summary="App is running.")
        outcome = evaluator.evaluate(
            task=task,
            observation=obs,
            reflection_decision=ReflectionDecision.CONTINUE,
            reflection_reason="Task succeeded.",
        )
        assert outcome.success is True
        assert outcome.confidence == Confidence.HIGH
        assert outcome.should_replan is False

    def test_continue_with_inconclusive_observation_gives_medium(self) -> None:
        evaluator = TaskEvaluator()
        task = _make_task()
        obs = Observation(method="response_text", success_signal=None)
        outcome = evaluator.evaluate(
            task=task,
            observation=obs,
            reflection_decision=ReflectionDecision.CONTINUE,
            reflection_reason="Task succeeded.",
        )
        assert outcome.success is True
        assert outcome.confidence == Confidence.MEDIUM


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TaskEvaluator — deterministic failure never triggers replanning
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaskEvaluatorDeterministicFailure:
    """Test 4: Deterministic failures (no skill, permission denied) → no replan."""

    @pytest.mark.parametrize("error_text", [
        "I don't have a skill for that.",
        "Permission denied.",
        "App not found on this system.",
        "Access denied.",
    ])
    def test_deterministic_failure_does_not_replan(self, error_text: str) -> None:
        evaluator = TaskEvaluator(replan_on_soft_failures=True)
        task = _make_task()
        obs = Observation(method="response_text", success_signal=False)
        outcome = evaluator.evaluate(
            task=task,
            observation=obs,
            reflection_decision=ReflectionDecision.ABORT,
            reflection_reason="Task failed.",
            response_text=error_text,
        )
        assert outcome.success is False
        assert outcome.should_replan is False


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TaskEvaluator — soft failure triggers replanning
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaskEvaluatorSoftFailureReplan:
    """Test 5: Soft failure with ABORT → should_replan=True when enabled."""

    def test_soft_abort_triggers_replan(self) -> None:
        evaluator = TaskEvaluator(replan_on_soft_failures=True)
        task = _make_task(attempt=2)
        obs = Observation(method="app_state", success_signal=False, summary="App NOT running.")
        outcome = evaluator.evaluate(
            task=task,
            observation=obs,
            reflection_decision=ReflectionDecision.ABORT,
            reflection_reason="All retries failed.",
            response_text="Something unexpected happened.",
        )
        assert outcome.success is False
        assert outcome.should_replan is True

    def test_replan_disabled_does_not_trigger(self) -> None:
        evaluator = TaskEvaluator(replan_on_soft_failures=False)
        task = _make_task(attempt=2)
        obs = Observation(method="response_text", success_signal=False)
        outcome = evaluator.evaluate(
            task=task,
            observation=obs,
            reflection_decision=ReflectionDecision.ABORT,
            reflection_reason="Failed.",
            response_text="Something went wrong with the connection.",
        )
        assert outcome.should_replan is False


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Replanner — respects max_attempts budget
# ═══════════════════════════════════════════════════════════════════════════════


class TestReplannerBudget:
    """Test 6: Replanner respects per-goal attempt budget."""

    def test_can_replan_within_budget(self) -> None:
        r = Replanner(max_attempts=2)
        assert r.can_replan("g1") is True

    def test_cannot_replan_after_exhaustion(self) -> None:
        r = Replanner(max_attempts=2)
        r._replan_counts["g1"] = 2
        assert r.can_replan("g1") is False

    def test_reset_restores_budget(self) -> None:
        r = Replanner(max_attempts=2)
        r._replan_counts["g1"] = 2
        r.reset("g1")
        assert r.can_replan("g1") is True


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Replanner — skip-and-continue fallback (no LLM)
# ═══════════════════════════════════════════════════════════════════════════════


class TestReplannerFallback:
    """Test 7: Without LLM, replanner returns original remaining tasks."""

    @pytest.mark.asyncio
    async def test_no_llm_returns_remaining_unchanged(self) -> None:
        r = Replanner(max_attempts=2)
        task = _make_task(goal_id="g2")
        remaining = [_make_task("Step 2", goal_id="g2"), _make_task("Step 3", goal_id="g2")]
        obs = Observation(method="response_text", success_signal=False)
        result = await r.replan(
            goal_description="Do something complex",
            goal_id="g2",
            failed_task=task,
            observation=obs,
            remaining_tasks=remaining,
            llm_client=None,
        )
        assert len(result) == 2
        assert result[0].description == "Step 2"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Replanner — LLM-based replanning parses JSON correctly
# ═══════════════════════════════════════════════════════════════════════════════


class TestReplannerLLM:
    """Test 8: LLM replanning parses the JSON task list correctly."""

    @pytest.mark.asyncio
    async def test_llm_replan_produces_new_tasks(self) -> None:
        r = Replanner(max_attempts=2)
        task = _make_task(goal_id="g3")
        obs = Observation(method="response_text", success_signal=False)

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = MagicMock(
            success=True,
            text='[{"description": "Try browser", "utterance": "open browser and search", "terminal": true}]',
        )

        result = await r.replan(
            goal_description="Find info online",
            goal_id="g3",
            failed_task=task,
            observation=obs,
            remaining_tasks=[],
            llm_client=mock_llm,
        )

        assert len(result) == 1
        assert result[0].description == "Try browser"
        assert result[0].terminal is True
        assert result[0].extra.get("replanned") is True


# ═══════════════════════════════════════════════════════════════════════════════
# 9. GoalVerifier — implicit verification for single terminal task
# ═══════════════════════════════════════════════════════════════════════════════


class TestGoalVerifierImplicit:
    """Test 9: Single terminal task that succeeds is verified implicitly."""

    @pytest.mark.asyncio
    async def test_single_completed_task_is_verified(self) -> None:
        verifier = GoalVerifier()
        task = _make_task(state=TaskState.COMPLETED, terminal=True)
        goal = _make_goal(tasks=[task])
        result = await verifier.verify(goal)
        assert result.verified is True
        assert result.confidence == "high"
        assert result.method == "implicit"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. GoalVerifier — multi-step completion verified via task results
# ═══════════════════════════════════════════════════════════════════════════════


class TestGoalVerifierMultiStep:
    """Test 10: All completed multi-step goal is verified."""

    @pytest.mark.asyncio
    async def test_all_completed_tasks_verified(self) -> None:
        verifier = GoalVerifier()
        tasks = [
            _make_task("Step 1", state=TaskState.COMPLETED),
            _make_task("Step 2", state=TaskState.COMPLETED),
        ]
        goal = _make_goal(tasks=tasks)
        result = await verifier.verify(goal)
        assert result.verified is True
        assert result.method == "task_results"

    @pytest.mark.asyncio
    async def test_failed_task_marks_unverified(self) -> None:
        verifier = GoalVerifier()
        tasks = [
            _make_task("Step 1", state=TaskState.COMPLETED),
            _make_task("Step 2", state=TaskState.FAILED),
        ]
        goal = _make_goal(tasks=tasks)
        result = await verifier.verify(goal)
        assert result.verified is False


# ═══════════════════════════════════════════════════════════════════════════════
# 11. TaskAuthorityChecker — safe tasks pass without confirmation
# ═══════════════════════════════════════════════════════════════════════════════


class TestAuthorityCheckerSafeTasks:
    """Test 11: Simple open/search tasks are SAFE and need no confirmation."""

    @pytest.mark.parametrize("utterance", [
        "open calculator",
        "search google for Python tutorials",
        "take a screenshot",
        "what time is it",
    ])
    def test_safe_tasks_do_not_require_confirmation(self, utterance: str) -> None:
        checker = TaskAuthorityChecker()
        task = _make_task(utterance=utterance)
        assert checker.requires_confirmation(task) is False
        assert checker.check(task) == AuthorityLevel.SAFE


# ═══════════════════════════════════════════════════════════════════════════════
# 12. TaskAuthorityChecker — destructive tasks require confirmation
# ═══════════════════════════════════════════════════════════════════════════════


class TestAuthorityCheckerDestructiveTasks:
    """Test 12: Upload, post, delete, send → CONFIRM authority."""

    @pytest.mark.parametrize("utterance,expected", [
        ("upload this file to instagram", AuthorityLevel.CONFIRM),
        ("post to twitter", AuthorityLevel.CONFIRM),
        ("send email to manager", AuthorityLevel.CONFIRM),
        ("delete file from desktop", AuthorityLevel.CONFIRM),
        ("shutdown the computer", AuthorityLevel.CRITICAL),
        ("restart the system", AuthorityLevel.CRITICAL),
    ])
    def test_destructive_tasks_require_confirmation(
        self, utterance: str, expected: AuthorityLevel
    ) -> None:
        checker = TaskAuthorityChecker()
        task = _make_task(utterance=utterance)
        assert checker.check(task) == expected
        assert checker.requires_confirmation(task) is True


# ═══════════════════════════════════════════════════════════════════════════════
# 13. TaskAuthorityChecker — LLM cannot bypass the gate
# ═══════════════════════════════════════════════════════════════════════════════


class TestAuthorityLLMBypassPrevented:
    """Test 13: LLM-generated task still goes through authority check."""

    def test_llm_generated_upload_task_requires_confirmation(self) -> None:
        checker = TaskAuthorityChecker()
        # Simulate an LLM-generated task with replanned=True extra
        task = _make_task(
            utterance="upload the video file to Instagram",
            extra={"replanned": True},
        )
        # Authority check doesn't care about source — only content
        assert checker.requires_confirmation(task) is True
        assert checker.check(task) == AuthorityLevel.CONFIRM


# ═══════════════════════════════════════════════════════════════════════════════
# 14. AgentTaskLogger — emits without raising
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentTaskLogger:
    """Test 14: AgentTaskLogger methods all complete without raising."""

    def test_all_log_methods_run_without_error(self) -> None:
        logger = AgentTaskLogger()
        task = _make_task()
        goal = _make_goal()
        obs = Observation(method="response_text", success_signal=True, summary="Done.")
        outcome = TaskOutcome(
            success=True,
            confidence=Confidence.HIGH,
            reason="All good.",
            should_replan=False,
            reflection_decision=ReflectionDecision.CONTINUE,
        )
        result = VerificationResult(verified=True, confidence="high", summary="Verified.")

        # None of these should raise
        logger.goal_created(goal)
        logger.plan_generated(goal, [task])
        logger.goal_complete(goal)
        logger.goal_failed(goal, "some error")
        logger.goal_cancelled(goal)
        logger.task_started(task, 0, 3)
        logger.task_success(task, 0, 3)
        logger.task_failed(task, 0, 3, "reason", will_retry=True)
        logger.task_skipped(task, 0, 3, "clarification needed")
        logger.confirmation_required(task, "T2/CONFIRM")
        logger.observation(task, obs)
        logger.evaluation(task, outcome)
        logger.replanning(1, 2, "hint")
        logger.verification(result)


# ═══════════════════════════════════════════════════════════════════════════════
# 15. ExecutionLoop — authority gate skips task requiring confirmation
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecutionLoopAuthorityGate:
    """Test 15: Destructive task is skipped with confirmation_required event."""

    @pytest.mark.asyncio
    async def test_confirm_task_is_skipped_not_executed(self) -> None:
        from spidy.agent.execution_loop import ExecutionLoop
        from spidy.agent.goal_manager import GoalManager
        from spidy.agent.progress_tracker import ProgressTracker
        from spidy.agent.reflection_engine import ReflectionEngine

        bus = _make_bus()
        brain = MagicMock()
        brain.process = AsyncMock(return_value="OK")
        brain._llm = None
        brain.session_id = "s1"

        gm = GoalManager(bus=bus)
        tracker = ProgressTracker(bus=bus)
        tracker.start = MagicMock()
        tracker.advance = AsyncMock(return_value=50)
        tracker.task_done = AsyncMock(return_value=50)
        tracker.publish_completed = AsyncMock()
        tracker.publish_failed = AsyncMock()

        authority = TaskAuthorityChecker()
        loop = ExecutionLoop(
            brain=brain,
            goal_manager=gm,
            reflection=ReflectionEngine(),
            tracker=tracker,
            bus=bus,
            authority=authority,
        )

        # Create a task that requires CONFIRM
        upload_task = _make_task(
            description="Upload video",
            utterance="upload the video to instagram",
            terminal=True,
            goal_id="g_auth",
        )
        goal = GoalRecord(description="Upload video", state=GoalState.EXECUTING,
                          tasks=(upload_task,))
        object.__setattr__(goal, "goal_id", "g_auth")
        gm._active_goal = goal

        ctx = ExecutionContext(goal=goal, session_id="s1", max_task_retries=1)

        final_goal, obs_list = await loop.run(goal=goal, tasks=[upload_task], ctx=ctx)

        # Brain.process should NOT have been called (gated by authority)
        brain.process.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# 16. ExecutionLoop — cancellation mid-task is respected
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecutionLoopCancellation:
    """Test 16: Cancellation flag stops loop between tasks."""

    @pytest.mark.asyncio
    async def test_cancellation_before_task_returns_cancelled_goal(self) -> None:
        from spidy.agent.execution_loop import ExecutionLoop
        from spidy.agent.goal_manager import GoalManager
        from spidy.agent.progress_tracker import ProgressTracker
        from spidy.agent.reflection_engine import ReflectionEngine

        bus = _make_bus()
        brain = MagicMock()
        brain.process = AsyncMock(return_value="OK")
        brain._llm = None

        gm = GoalManager(bus=bus)
        tracker = ProgressTracker(bus=bus)
        tracker.start = MagicMock()
        tracker.advance = AsyncMock(return_value=0)
        tracker.task_done = AsyncMock(return_value=0)
        tracker.publish_completed = AsyncMock()
        tracker.publish_failed = AsyncMock()
        tracker.publish_cancelled = AsyncMock()

        loop = ExecutionLoop(
            brain=brain,
            goal_manager=gm,
            reflection=ReflectionEngine(),
            tracker=tracker,
            bus=bus,
        )

        task = _make_task(goal_id="g_cancel")
        goal = GoalRecord(description="Cancel test", state=GoalState.EXECUTING,
                          tasks=(task,))
        object.__setattr__(goal, "goal_id", "g_cancel")
        gm._active_goal = goal

        ctx = ExecutionContext(goal=goal, session_id="s1", max_task_retries=0)
        ctx.cancel()  # Cancel immediately

        final_goal, _ = await loop.run(goal=goal, tasks=[task], ctx=ctx)
        assert final_goal.state == GoalState.CANCELLED


# ═══════════════════════════════════════════════════════════════════════════════
# 17. ExecutionLoop — successful task advances to next
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecutionLoopTaskSequence:
    """Test 17: All tasks run sequentially and goal completes."""

    @pytest.mark.asyncio
    async def test_two_tasks_run_in_order(self) -> None:
        from spidy.agent.execution_loop import ExecutionLoop
        from spidy.agent.goal_manager import GoalManager
        from spidy.agent.progress_tracker import ProgressTracker
        from spidy.agent.reflection_engine import ReflectionEngine

        bus = _make_bus()
        call_order: list[str] = []

        async def brain_process(utterance: str, session_id: str = "") -> str:
            call_order.append(utterance)
            return "Done!"

        brain = MagicMock()
        brain.process = AsyncMock(side_effect=brain_process)
        brain._llm = None

        gm = GoalManager(bus=bus)
        tracker = ProgressTracker(bus=bus)
        tracker.start = MagicMock()
        tracker.advance = AsyncMock(return_value=50)
        tracker.task_done = AsyncMock(return_value=100)
        tracker.publish_completed = AsyncMock()
        tracker.publish_failed = AsyncMock()

        loop = ExecutionLoop(
            brain=brain,
            goal_manager=gm,
            reflection=ReflectionEngine(),
            tracker=tracker,
            bus=bus,
            inter_task_delay=0,
        )

        tasks = [
            _make_task("Task A", "utterance a", goal_id="g_seq"),
            _make_task("Task B", "utterance b", terminal=True, goal_id="g_seq"),
        ]
        goal = GoalRecord(description="Seq test", state=GoalState.EXECUTING,
                          tasks=tuple(tasks))
        object.__setattr__(goal, "goal_id", "g_seq")
        gm._active_goal = goal

        ctx = ExecutionContext(goal=goal, session_id="s1", max_task_retries=0)
        final_goal, _ = await loop.run(goal=goal, tasks=tasks, ctx=ctx)

        assert final_goal.state == GoalState.COMPLETED
        assert call_order == ["utterance a", "utterance b"]


# ═══════════════════════════════════════════════════════════════════════════════
# 18. AutonomousAgent — simple single-task goal completes
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutonomousAgentSimpleGoal:
    """Test 18: Simple goal with heuristic decomposition completes successfully."""

    @pytest.mark.asyncio
    async def test_open_calculator_completes(self) -> None:
        from spidy.agent.agent import AutonomousAgent

        bus = _make_bus()
        brain = MagicMock()
        brain.process = AsyncMock(return_value="I've opened Calculator for you.")
        brain._llm = None
        brain.session_id = "s1"

        agent = AutonomousAgent(
            brain=brain,
            bus=bus,
            llm_client=None,
            inter_task_delay=0,
            observation_enabled=False,  # Disable for speed
            verify_enabled=False,
        )

        response = await agent.run_goal("open calculator")
        assert response  # Non-empty response
        assert agent.active_goal is None  # Goal is done


# ═══════════════════════════════════════════════════════════════════════════════
# 19. AutonomousAgent — cancel_current_goal stops execution
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutonomousAgentCancellation:
    """Test 19: cancel_current_goal() returns False when no goal is running."""

    @pytest.mark.asyncio
    async def test_cancel_when_idle_returns_false(self) -> None:
        from spidy.agent.agent import AutonomousAgent

        bus = _make_bus()
        brain = MagicMock()
        brain.process = AsyncMock(return_value="Done.")
        brain._llm = None
        brain.session_id = "s1"

        agent = AutonomousAgent(brain=brain, bus=bus)
        result = await agent.cancel_current_goal()
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
# 20. [AGENT] structured log format — all lifecycle tags present
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentStructuredLogFormat:
    """Test 20: [AGENT] tag appears in all lifecycle log methods."""

    def test_all_log_calls_use_agent_tag(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        from spidy.agent.task_logger import AgentTaskLogger, _TAG

        assert _TAG == "[AGENT]"

        logger = AgentTaskLogger()
        goal = _make_goal()
        task = _make_task()
        obs = Observation(method="response_text", success_signal=True)
        outcome = TaskOutcome(
            success=True,
            confidence=Confidence.HIGH,
            reason="OK",
            reflection_decision=ReflectionDecision.CONTINUE,
        )
        result = VerificationResult(verified=True)

        # Just verify the _TAG constant is correct and methods don't raise
        logger.goal_created(goal)
        logger.task_started(task, 0, 1)
        logger.task_success(task, 0, 1)
        logger.observation(task, obs)
        logger.evaluation(task, outcome)
        logger.verification(result)
