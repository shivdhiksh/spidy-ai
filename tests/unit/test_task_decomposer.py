"""
tests/unit/test_task_decomposer.py — Unit tests for TaskDecomposer
===================================================================
Tests heuristic decomposition, LLM-based decomposition (mocked),
JSON parsing, and edge cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from spidy.agent.task_decomposer import TaskDecomposer
from spidy.agent.types import TaskState


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_decomposer(**kwargs) -> TaskDecomposer:
    return TaskDecomposer(**kwargs)


def make_mock_llm(response_text: str, success: bool = True) -> MagicMock:
    """Create a mock LLM client that returns the given text."""
    llm = MagicMock()
    mock_response = MagicMock()
    mock_response.success = success
    mock_response.text = response_text
    llm.complete = AsyncMock(return_value=mock_response)
    return llm


# ═══════════════════════════════════════════════════════════════════════════════
# Heuristic decomposition
# ═══════════════════════════════════════════════════════════════════════════════


class TestHeuristicDecomposition:
    @pytest.mark.asyncio
    async def test_flask_project_decomposed(self) -> None:
        d = make_decomposer()
        tasks = await d.decompose("Create a Flask project")
        assert len(tasks) >= 3
        descriptions = [t.description.lower() for t in tasks]
        assert any("flask" in d or "folder" in d or "terminal" in d for d in descriptions)

    @pytest.mark.asyncio
    async def test_react_project_decomposed(self) -> None:
        d = make_decomposer()
        tasks = await d.decompose("Create a React app")
        assert len(tasks) >= 2
        descriptions = [t.description.lower() for t in tasks]
        assert any("react" in d or "terminal" in d for d in descriptions)

    @pytest.mark.asyncio
    async def test_python_project_decomposed(self) -> None:
        d = make_decomposer()
        tasks = await d.decompose("Create a Python project")
        assert len(tasks) >= 3

    @pytest.mark.asyncio
    async def test_youtube_decomposed(self) -> None:
        d = make_decomposer()
        tasks = await d.decompose("Search YouTube for Python tutorials")
        assert len(tasks) >= 2
        descriptions = [t.description.lower() for t in tasks]
        assert any("youtube" in d or "browser" in d for d in descriptions)

    @pytest.mark.asyncio
    async def test_vs_code_decomposed(self) -> None:
        d = make_decomposer()
        tasks = await d.decompose("Open VS Code and create main.py")
        assert len(tasks) >= 2

    @pytest.mark.asyncio
    async def test_all_tasks_pending(self) -> None:
        d = make_decomposer()
        tasks = await d.decompose("Create a Flask project")
        assert all(t.state == TaskState.PENDING for t in tasks)

    @pytest.mark.asyncio
    async def test_all_tasks_have_utterances(self) -> None:
        d = make_decomposer()
        tasks = await d.decompose("Create a Flask project")
        assert all(t.utterance.strip() for t in tasks)

    @pytest.mark.asyncio
    async def test_all_tasks_have_descriptions(self) -> None:
        d = make_decomposer()
        tasks = await d.decompose("Create a Flask project")
        assert all(t.description.strip() for t in tasks)

    @pytest.mark.asyncio
    async def test_unknown_goal_becomes_single_task(self) -> None:
        d = make_decomposer()
        tasks = await d.decompose("Do something completely unrecognised XYZ123")
        assert len(tasks) == 1
        assert tasks[0].utterance == "Do something completely unrecognised XYZ123"

    @pytest.mark.asyncio
    async def test_max_tasks_cap_respected(self) -> None:
        d = make_decomposer(max_tasks=3)
        # Flask project normally generates > 3 tasks
        tasks = await d.decompose("Create a Flask project")
        assert len(tasks) <= 3

    @pytest.mark.asyncio
    async def test_goal_id_attached_to_tasks(self) -> None:
        d = make_decomposer()
        tasks = await d.decompose("Create a Flask project", goal_id="goal-123")
        assert all(t.goal_id == "goal-123" for t in tasks)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM decomposition
# ═══════════════════════════════════════════════════════════════════════════════


class TestLLMDecomposition:
    @pytest.mark.asyncio
    async def test_llm_json_parsed_correctly(self) -> None:
        llm_json = """[
          {"description": "Open terminal", "utterance": "open a terminal"},
          {"description": "Install deps", "utterance": "install dependencies"}
        ]"""
        llm = make_mock_llm(llm_json)
        d = make_decomposer()
        tasks = await d.decompose("Set up project", llm_client=llm)
        assert len(tasks) == 2
        assert tasks[0].description == "Open terminal"
        assert tasks[0].utterance == "open a terminal"
        assert tasks[1].description == "Install deps"

    @pytest.mark.asyncio
    async def test_llm_with_markdown_fences_parsed(self) -> None:
        llm_json = """```json
[
  {"description": "Step 1", "utterance": "do step one"},
  {"description": "Step 2", "utterance": "do step two"}
]
```"""
        llm = make_mock_llm(llm_json)
        d = make_decomposer()
        tasks = await d.decompose("Goal", llm_client=llm)
        assert len(tasks) == 2
        assert tasks[0].description == "Step 1"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_heuristic(self) -> None:
        llm = make_mock_llm("", success=False)
        d = make_decomposer()
        tasks = await d.decompose("Create a Flask project", llm_client=llm)
        # Should fall back to heuristic
        assert len(tasks) >= 3

    @pytest.mark.asyncio
    async def test_llm_invalid_json_falls_back_to_heuristic(self) -> None:
        llm = make_mock_llm("This is not JSON at all")
        d = make_decomposer()
        tasks = await d.decompose("Create a Flask project", llm_client=llm)
        # Heuristic fallback
        assert len(tasks) >= 3

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back_to_heuristic(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM offline"))
        d = make_decomposer()
        tasks = await d.decompose("Create a Flask project", llm_client=llm)
        assert len(tasks) >= 3

    @pytest.mark.asyncio
    async def test_llm_empty_array_falls_back(self) -> None:
        llm = make_mock_llm("[]")
        d = make_decomposer()
        tasks = await d.decompose("Create a Flask project", llm_client=llm)
        assert len(tasks) >= 3

    @pytest.mark.asyncio
    async def test_llm_skips_invalid_items(self) -> None:
        llm_json = """[
          {"description": "Good step", "utterance": "do the thing"},
          {"missing_fields": true},
          {"description": "", "utterance": "empty desc"}
        ]"""
        llm = make_mock_llm(llm_json)
        d = make_decomposer()
        tasks = await d.decompose("Goal", llm_client=llm)
        # Only valid items should pass
        assert len(tasks) >= 1
        assert tasks[0].description == "Good step"
