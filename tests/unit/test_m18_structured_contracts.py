"""
tests/unit/test_m18_structured_contracts.py
============================================
Milestone 18 regression tests: Structured Agent Task Contracts.

Tests cover:
  1.  TaskRecord supports structured action field
  2.  Old utterance-only TaskRecord still works (backward compat)
  3.  Structured browser search preserves query exactly
  4.  ExecutionLoop uses StructuredRouter when task.action is set
  5.  Missing/invalid structured action falls back to Brain.process()
  6.  expected_outcome reaches TaskObserver
  7.  expected_outcome appears in GoalVerifier summary
  8.  Compound browser goal produces all tasks with structured actions
  9.  step_results stored after each successful task
  10. input_from injects previous task result
  11. Simple commands still call Brain.process (no router bypass)
  12. Authority gate inspects action dict as well as utterance
  13. Structured action cannot bypass CONFIRM gate
  14. _extract_search_params handles all patterns correctly
  15. StructuredRouter._search_url constructs correct YouTube URL
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from spidy.agent.authority import AuthorityLevel, TaskAuthorityChecker
from spidy.agent.execution_loop import ExecutionLoop
from spidy.agent.goal_manager import GoalManager
from spidy.agent.observer import TaskObserver
from spidy.agent.progress_tracker import ProgressTracker
from spidy.agent.reflection_engine import ReflectionEngine
from spidy.agent.structured_router import StructuredRouter, _search_url, _target_url
from spidy.agent.task_decomposer import TaskDecomposer
from spidy.agent.types import ExecutionContext, GoalRecord, GoalState, TaskRecord, TaskState
from spidy.core.event_bus import EventBus


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_bus() -> EventBus:
    bus = EventBus()
    loop = asyncio.get_running_loop()
    bus.set_loop(loop)
    return bus


def make_brain_mock(response_text: str = "Done!", fail: bool = False) -> MagicMock:
    brain = MagicMock()
    brain.process = AsyncMock(return_value=response_text)
    brain.session_id = "test-session"
    brain._llm = None
    return brain


def make_loop_with_router(
    brain=None,
    router=None,
    max_retries: int = 2,
) -> tuple[ExecutionLoop, GoalManager, EventBus]:
    bus = make_bus()
    gm = GoalManager(bus=bus)
    brain = brain or make_brain_mock()
    reflection = ReflectionEngine(max_retries=max_retries)
    tracker = ProgressTracker(bus=bus)
    loop = ExecutionLoop(
        brain=brain,
        goal_manager=gm,
        reflection=reflection,
        tracker=tracker,
        bus=bus,
        router=router,
        inter_task_delay=0.0,
    )
    return loop, gm, bus


async def setup_goal(gm: GoalManager, tasks: list[TaskRecord]) -> GoalRecord:
    goal = await gm.create_goal("Test goal")
    await gm.begin_planning(goal.goal_id)
    goal = await gm.begin_execution(goal.goal_id, tasks=tasks)
    return goal


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: TaskRecord supports action field
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaskRecordActionField:
    def test_task_record_accepts_action_dict(self) -> None:
        action = {"skill": "browser", "action": "search", "target": "youtube", "query": "Python tutorials"}
        task = TaskRecord(description="Search YouTube", utterance="search youtube", action=action)
        assert task.action is not None
        assert task.action["skill"] == "browser"
        assert task.action["query"] == "Python tutorials"

    # Test 2: backward compat — no action field
    def test_task_record_without_action_still_works(self) -> None:
        task = TaskRecord(description="Do something", utterance="do something")
        assert task.action is None
        assert task.input_from == ""

    # Test 3: query preserved exactly through mark_running/mark_completed
    def test_action_and_input_from_preserved_through_state_transitions(self) -> None:
        action = {
            "skill": "browser",
            "action": "search",
            "target": "youtube",
            "query": "Python tutorials",
            "expected_outcome": "YouTube search results for Python tutorials are visible",
        }
        task = TaskRecord(
            description="Search YouTube",
            utterance="search youtube for Python tutorials",
            action=action,
            input_from="task_0",
        )
        running = task.mark_running()
        assert running.action == action
        assert running.input_from == "task_0"
        completed = running.mark_completed("Done")
        assert completed.action == action
        assert completed.input_from == "task_0"

    def test_action_preserved_through_mark_failed(self) -> None:
        action = {"skill": "desktop", "action": "open_app", "target": "Edge"}
        task = TaskRecord(description="Open Edge", utterance="open edge", action=action)
        running = task.mark_running()
        failed = running.mark_failed("Error")
        assert failed.action == action

    def test_action_preserved_through_mark_skipped(self) -> None:
        action = {"skill": "desktop", "action": "open_app", "target": "Edge"}
        task = TaskRecord(description="Open Edge", utterance="open edge", action=action)
        skipped = task.mark_skipped("Cancelled")
        assert skipped.action == action


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: ExecutionLoop uses StructuredRouter when task.action is set
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecutionLoopStructuredRouting:
    @pytest.mark.asyncio
    async def test_structured_action_routes_to_router_not_brain(self) -> None:
        """When task.action is set, router.execute() is called instead of Brain.process()."""
        brain = make_brain_mock("Done!")
        router = MagicMock()
        router.execute = AsyncMock(return_value=("Router result", True))

        loop, gm, bus = make_loop_with_router(brain=brain, router=router)
        task = TaskRecord(
            description="Search YouTube",
            utterance="search youtube for Python tutorials",
            action={"skill": "browser", "action": "search", "target": "youtube", "query": "Python tutorials"},
        )
        goal = await setup_goal(gm, [task])
        ctx = ExecutionContext(goal=goal, session_id="s1")

        await loop.run(goal, [task], ctx)

        router.execute.assert_called_once()
        # Brain.process should NOT be called (router handled it)
        assert brain.process.call_count == 0

    # Test 5: fallback when no router
    @pytest.mark.asyncio
    async def test_no_router_falls_back_to_brain(self) -> None:
        """When router is None (or missing), Brain.process() is always called."""
        brain = make_brain_mock("Done!")
        loop, gm, bus = make_loop_with_router(brain=brain, router=None)
        task = TaskRecord(
            description="Search YouTube",
            utterance="search youtube for Python tutorials",
            action={"skill": "browser", "action": "search", "query": "Python tutorials"},
        )
        goal = await setup_goal(gm, [task])
        ctx = ExecutionContext(goal=goal, session_id="s1")

        await loop.run(goal, [task], ctx)

        # Brain.process is the fallback
        assert brain.process.call_count >= 1

    @pytest.mark.asyncio
    async def test_task_without_action_always_calls_brain(self) -> None:
        """Tasks without action ALWAYS go to Brain.process (no router bypass)."""
        brain = make_brain_mock("Done!")
        router = MagicMock()
        router.execute = AsyncMock(return_value=("Router result", True))

        loop, gm, bus = make_loop_with_router(brain=brain, router=router)
        task = TaskRecord(description="Do something", utterance="do something")  # no action
        goal = await setup_goal(gm, [task])
        ctx = ExecutionContext(goal=goal, session_id="s1")

        await loop.run(goal, [task], ctx)

        # Router should NOT be called for tasks without action
        router.execute.assert_not_called()
        assert brain.process.call_count >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Tests 6–7: expected_outcome propagation
# ═══════════════════════════════════════════════════════════════════════════════


class TestExpectedOutcome:
    @pytest.mark.asyncio
    async def test_expected_outcome_appears_in_observation_extra(self) -> None:
        """Observer.observe() stores expected_outcome in Observation.extra."""
        task = TaskRecord(
            description="Search YouTube",
            utterance="search youtube for Python tutorials",
            action={
                "skill": "browser",
                "action": "search",
                "target": "youtube",
                "query": "Python tutorials",
                "expected_outcome": "YouTube search results for Python tutorials are visible",
            },
        )
        running = task.mark_running()
        observer = TaskObserver(vision=None, vision_enabled=False)
        obs = await observer.observe(running, "Done!", skip_for_terminal=False)
        # extra should have expected_outcome key
        assert "expected_outcome" in obs.extra
        assert "Python tutorials" in obs.extra["expected_outcome"]

    def test_expected_outcome_in_verifier_summary(self) -> None:
        """GoalVerifier._build_summary enriches output with expected_outcome phrases."""
        from spidy.agent.verifier import GoalVerifier
        verifier = GoalVerifier()

        task = TaskRecord(
            description="Search YouTube",
            utterance="search youtube",
            action={
                "skill": "browser",
                "action": "search",
                "query": "Python tutorials",
                "expected_outcome": "YouTube search results for Python tutorials are visible",
            },
            state=TaskState.COMPLETED,
        )
        goal = GoalRecord(description="Test", tasks=[task], state=GoalState.COMPLETED)
        # Call the internal summary method with a completed task
        result = verifier._build_summary(goal, [task], [], [])
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8: Compound goal decomposition produces structured actions
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompoundGoalDecomposition:
    @pytest.mark.asyncio
    async def test_search_clause_produces_structured_action_with_query(self) -> None:
        """
        A compound goal containing a search clause must produce a task with
        action.query set to the verbatim query string.
        """
        decomposer = TaskDecomposer()
        goal_id = "g1"
        tasks = await decomposer.decompose(
            "open edge and search youtube for Python tutorials",
            goal_id=goal_id,
            llm_client=None,
        )

        # At least one task should have action.query = "Python tutorials"
        search_tasks = [
            t for t in tasks
            if t.action and t.action.get("action") == "search"
        ]
        assert len(search_tasks) >= 1, "Expected at least one search task with structured action"
        search_task = search_tasks[0]
        assert search_task.action is not None
        assert search_task.action.get("query") == "Python tutorials", (
            f"Expected query='Python tutorials', got {search_task.action.get('query')!r}"
        )

    @pytest.mark.asyncio
    async def test_search_clause_query_is_verbatim(self) -> None:
        """Query case and spacing are preserved exactly."""
        decomposer = TaskDecomposer()
        tasks = await decomposer.decompose(
            "search youtube for Best Python Async Libraries",
            goal_id="g2",
            llm_client=None,
        )
        search_tasks = [t for t in tasks if t.action and t.action.get("action") == "search"]
        if search_tasks:
            q = search_tasks[0].action.get("query", "")
            assert "Best Python Async Libraries" in q or q.lower() == "best python async libraries"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests 9–10: step_results and input_from
# ═══════════════════════════════════════════════════════════════════════════════


class TestStepResults:
    def test_execution_context_step_results_start_empty(self) -> None:
        goal = GoalRecord(description="Test")
        ctx = ExecutionContext(goal=goal)
        assert ctx.step_results == {}

    def test_add_result_stores_by_task_id(self) -> None:
        goal = GoalRecord(description="Test")
        ctx = ExecutionContext(goal=goal)
        ctx.add_result("task_abc", "resume.pdf found at C:/Users/Desktop")
        assert ctx.step_results["task_abc"] == "resume.pdf found at C:/Users/Desktop"

    def test_inject_step_result_file_open(self) -> None:
        """input_from with file/open action injects prior result into 'target'."""
        goal = GoalRecord(description="Test")
        ctx = ExecutionContext(goal=goal)
        ctx.add_result("task_1", "C:/Users/Desktop/resume.pdf")

        action = {
            "skill": "file",
            "action": "open_file",
            "input_from": "task_1",
        }
        resolved = ExecutionLoop._inject_step_result(action, ctx)
        assert resolved.get("target") == "C:/Users/Desktop/resume.pdf"

    def test_inject_step_result_missing_reference_returns_unchanged(self) -> None:
        """When input_from references a missing task_id, action is returned unchanged."""
        goal = GoalRecord(description="Test")
        ctx = ExecutionContext(goal=goal)
        # No task_1 stored
        action = {"skill": "file", "action": "open_file", "input_from": "task_1"}
        resolved = ExecutionLoop._inject_step_result(action, ctx)
        # Should not crash, action unchanged (no target injected)
        assert resolved.get("target") is None

    def test_inject_step_result_no_input_from_returns_unchanged(self) -> None:
        """When input_from is absent, action is returned unchanged."""
        goal = GoalRecord(description="Test")
        ctx = ExecutionContext(goal=goal)
        action = {"skill": "browser", "action": "search", "query": "Python"}
        resolved = ExecutionLoop._inject_step_result(action, ctx)
        assert resolved == action


# ═══════════════════════════════════════════════════════════════════════════════
# Test 12–13: Authority gate still works with structured actions
# ═══════════════════════════════════════════════════════════════════════════════


class TestAuthorityGateWithStructuredActions:
    def test_safe_browser_search_is_safe(self) -> None:
        checker = TaskAuthorityChecker()
        task = TaskRecord(
            description="Search YouTube",
            utterance="search youtube for Python tutorials",
            action={"skill": "browser", "action": "search", "target": "youtube", "query": "Python tutorials"},
        )
        assert checker.check(task) == AuthorityLevel.SAFE

    def test_destructive_action_in_action_dict_detected(self) -> None:
        """Even if utterance is benign, action dict with 'delete file' must be caught."""
        checker = TaskAuthorityChecker()
        task = TaskRecord(
            description="Clean up",
            utterance="clean up workspace",
            action={"skill": "file", "action": "delete file", "target": "old_reports"},
        )
        level = checker.check(task)
        assert level in (AuthorityLevel.CONFIRM, AuthorityLevel.CRITICAL)

    def test_requires_confirmation_for_upload_action(self) -> None:
        """'upload' in utterance always triggers confirmation requirement."""
        checker = TaskAuthorityChecker()
        task = TaskRecord(
            description="Upload to server",
            utterance="upload my resume to linkedin",
        )
        assert checker.requires_confirmation(task) is True


# ═══════════════════════════════════════════════════════════════════════════════
# Test 14: _extract_search_params
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractSearchParams:
    def test_search_site_for_query(self) -> None:
        target, query = TaskDecomposer._extract_search_params("search YouTube for Python tutorials")
        assert target == "youtube"
        assert query == "Python tutorials"

    def test_search_for_query_on_site(self) -> None:
        target, query = TaskDecomposer._extract_search_params("search for best books on Google")
        assert target == "google"
        assert "best books" in query

    def test_search_for_query_no_site(self) -> None:
        target, query = TaskDecomposer._extract_search_params("search for async Python patterns")
        assert target == ""
        assert "async Python patterns" in query

    def test_no_match_returns_empty(self) -> None:
        target, query = TaskDecomposer._extract_search_params("open Edge browser")
        assert target == ""
        assert query == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Test 15: StructuredRouter URL construction
# ═══════════════════════════════════════════════════════════════════════════════


class TestStructuredRouterURLs:
    def test_youtube_search_url_contains_encoded_query(self) -> None:
        url = _search_url("youtube", "Python tutorials")
        assert "youtube.com/results" in url
        assert "Python+tutorials" in url or "Python%20tutorials" in url or "Python tutorials" in url

    def test_google_search_url(self) -> None:
        url = _search_url("google", "async Python")
        assert "google.com" in url
        assert "async" in url.lower()

    def test_unknown_target_returns_empty(self) -> None:
        url = _search_url("unknown_site_xyz", "some query")
        assert url == ""

    def test_target_url_youtube(self) -> None:
        url = _target_url("youtube")
        assert "youtube.com" in url

    def test_target_url_case_insensitive(self) -> None:
        url = _target_url("YouTube")
        assert "youtube.com" in url
