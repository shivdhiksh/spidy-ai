"""
Unit Tests: Compound Goal Integrity & OpenRouter Fallback Model
================================================================
Covers:
1. Clause extraction and noise tolerance (_extract_actionable_clauses).
2. Decomposition of compound goals with conversational preambles ("That's Shiva...").
3. Structured action contracts and terminal flag invariants (only the final task is terminal).
4. Single-action fast path integrity ("Open Edge" -> 1 task).
5. GoalVerifier clause integrity invariant (rejecting false completion when clauses dropped).
6. OpenRouter fallback model configuration and failover.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spidy.agent.task_decomposer import (
    TaskDecomposer,
    _extract_actionable_clauses,
    _split_compound_goal,
)
from spidy.agent.types import GoalRecord, GoalState, TaskRecord, TaskState
from spidy.agent.verifier import GoalVerifier, VerificationResult
from spidy.config.manager import LLMProviderConfig, MultiLLMConfig
from spidy.llm.client import LLMClientFactory, LLMMessage, LLMResponse
from spidy.llm.router import LLMRouter


# ─── 1. Clause Extraction & Noise Tolerance ───────────────────────────────────


class TestClauseExtraction:
    """Test extracting actionable clauses with preamble/filler tolerance."""

    def test_preamble_with_three_clauses(self):
        utterance = "That's Shiva. Open Edge. Open YouTube. And search for Python tutorials."
        clauses = _extract_actionable_clauses(utterance)
        assert len(clauses) == 3
        assert clauses[0].lower() == "open edge"
        assert clauses[1].lower() == "open youtube"
        assert "search" in clauses[2].lower()
        assert "python tutorials" in clauses[2].lower()

    def test_standard_compound_goal(self):
        utterance = "Open Edge, open YouTube, and search for Python tutorials."
        clauses = _extract_actionable_clauses(utterance)
        assert len(clauses) == 3
        assert clauses[0].lower() == "open edge"
        assert clauses[1].lower() == "open youtube"
        assert "search" in clauses[2].lower()

    def test_two_clause_conjunction(self):
        utterance = "Open Edge and search Python"
        clauses = _extract_actionable_clauses(utterance)
        assert len(clauses) == 2
        assert clauses[0].lower() == "open edge"
        assert clauses[1].lower() == "search python"

    def test_single_action_goal(self):
        utterance = "Open Edge"
        clauses = _extract_actionable_clauses(utterance)
        assert len(clauses) == 1
        assert clauses[0] == "Open Edge"
        assert _split_compound_goal(utterance) is None

    def test_non_action_query(self):
        utterance = "What is Python?"
        clauses = _extract_actionable_clauses(utterance)
        assert len(clauses) == 0
        assert _split_compound_goal(utterance) is None


# ─── 2. TaskDecomposer Compound Goal Planning ─────────────────────────────────


class TestTaskDecomposerCompoundGoals:
    """Test that compound goals produce all structured tasks with proper terminal flags."""

    @pytest.mark.asyncio
    async def test_decompose_compound_with_noise_prefix(self):
        decomposer = TaskDecomposer()
        goal = "That's Shiva. Open Edge. Open YouTube. And search for Python tutorials."
        tasks = await decomposer.decompose(goal, goal_id="g100")

        assert len(tasks) == 3, f"Expected 3 tasks, got {len(tasks)}"

        # Task 1: Open Edge
        t1 = tasks[0]
        assert "edge" in t1.description.lower()
        assert t1.action is not None
        assert t1.action.get("skill") == "desktop"
        assert t1.action.get("action") == "open_app"
        assert t1.terminal is False

        # Task 2: Navigate YouTube
        t2 = tasks[1]
        assert "youtube" in t2.description.lower()
        assert t2.action is not None
        assert t2.action.get("skill") == "browser"
        assert t2.action.get("action") == "navigate"
        assert "youtube.com" in t2.action.get("url", "")
        assert t2.terminal is False

        # Task 3: Search YouTube for Python tutorials
        t3 = tasks[2]
        assert "search" in t3.description.lower()
        assert t3.action is not None
        assert t3.action.get("skill") == "browser"
        assert t3.action.get("action") == "search"
        assert t3.action.get("query") == "Python tutorials"
        assert t3.action.get("target") == "youtube"
        assert t3.terminal is True  # Only final task is terminal

    @pytest.mark.asyncio
    async def test_decompose_single_action_fast_path(self):
        decomposer = TaskDecomposer()
        goal = "Open Edge"
        tasks = await decomposer.decompose(goal, goal_id="g101")

        assert len(tasks) == 1
        t1 = tasks[0]
        assert "edge" in t1.description.lower()
        assert t1.terminal is True

    @pytest.mark.asyncio
    async def test_decompose_open_notepad(self):
        decomposer = TaskDecomposer()
        goal = "Open Notepad"
        tasks = await decomposer.decompose(goal, goal_id="g102")

        assert len(tasks) == 1
        t1 = tasks[0]
        assert "notepad" in t1.description.lower()
        assert t1.terminal is True


# ─── 3. GoalVerifier Clause Integrity ─────────────────────────────────────────


class TestGoalVerifierIntegrity:
    """Test that GoalVerifier enforces clause completeness for compound goals."""

    @pytest.mark.asyncio
    async def test_single_task_on_compound_goal_fails_verification(self):
        verifier = GoalVerifier(skip_for_single_task=True)
        # Original goal had 3 clauses
        compound_desc = "That's Shiva. Open Edge. Open YouTube. And search for Python tutorials."
        # But only 1 task was planned/executed
        completed_task = TaskRecord(
            task_id="t1",
            goal_id="g1",
            description="Open Edge",
            utterance="open edge",
            state=TaskState.COMPLETED,
            terminal=True,
        )
        goal = GoalRecord(
            goal_id="g1",
            description=compound_desc,
            state=GoalState.EXECUTING,
            tasks=[completed_task],
            session_id="s1",
        )

        result = await verifier.verify(goal=goal)
        assert result.verified is False
        assert result.confidence == "partial"
        assert result.method == "clause_integrity_check"
        assert "Incomplete execution" in result.summary

    @pytest.mark.asyncio
    async def test_all_tasks_completed_on_compound_goal_passes_verification(self):
        verifier = GoalVerifier(skip_for_single_task=True)
        compound_desc = "Open Edge, open YouTube, and search for Python tutorials."
        t1 = TaskRecord(task_id="t1", goal_id="g1", description="Open Edge", utterance="open edge", state=TaskState.COMPLETED, terminal=False)
        t2 = TaskRecord(task_id="t2", goal_id="g1", description="Navigate YouTube", utterance="navigate to youtube", state=TaskState.COMPLETED, terminal=False)
        t3 = TaskRecord(task_id="t3", goal_id="g1", description="Search Python tutorials", utterance="search Python tutorials", state=TaskState.COMPLETED, terminal=True)

        goal = GoalRecord(
            goal_id="g1",
            description=compound_desc,
            state=GoalState.EXECUTING,
            tasks=[t1, t2, t3],
            session_id="s1",
        )

        result = await verifier.verify(goal=goal)
        assert result.verified is True
        assert result.confidence in ("high", "medium")

    @pytest.mark.asyncio
    async def test_single_clause_goal_passes_implicit_verification(self):
        verifier = GoalVerifier(skip_for_single_task=True)
        single_desc = "Open Edge"
        t1 = TaskRecord(task_id="t1", goal_id="g1", description="Open Edge", utterance="open edge", state=TaskState.COMPLETED, terminal=True)
        goal = GoalRecord(
            goal_id="g1",
            description=single_desc,
            state=GoalState.EXECUTING,
            tasks=[t1],
            session_id="s1",
        )

        result = await verifier.verify(goal=goal)
        assert result.verified is True
        assert result.method == "implicit"


# ─── 4. OpenRouter Fallback Model Configuration ───────────────────────────────


class TestOpenRouterFallbackConfiguration:
    """Test OpenRouter fallback model default and failover behaviour."""

    def test_default_model_is_nemotron_free(self):
        from spidy.llm.backends.openrouter import OpenRouterClient, _DEFAULT_MODEL
        assert _DEFAULT_MODEL == "nvidia/nemotron-3-ultra-550b-a55b:free"
        client = OpenRouterClient(api_key="test-key")
        assert client._model == "nvidia/nemotron-3-ultra-550b-a55b:free"

    @pytest.mark.asyncio
    async def test_router_falls_back_to_openrouter_on_primary_failure(self):
        primary_mock = AsyncMock()
        primary_mock.name = "nvidia"
        primary_mock.complete.return_value = LLMResponse.failure("503 Service Unavailable")

        fallback_mock = AsyncMock()
        fallback_mock.name = "openrouter"
        fallback_mock.complete.return_value = LLMResponse(
            text="Hello from OpenRouter fallback!",
            success=True,
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
        )

        router = LLMRouter(
            providers=[primary_mock, fallback_mock],
            auto_fallback=True,
        )

        response = await router.complete([LLMMessage(role="user", content="Hello")])
        assert response.success is True
        assert "OpenRouter fallback" in response.text
        assert primary_mock.complete.call_count == 1
        assert fallback_mock.complete.call_count == 1
