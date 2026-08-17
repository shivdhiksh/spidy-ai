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

# ═══════════════════════════════════════════════════════════════════════════════
# Terminal flag — heuristic decomposition
# ═══════════════════════════════════════════════════════════════════════════════


class TestTerminalFlag:
    """Verify that the terminal flag is set correctly by the decomposer."""

    @pytest.mark.asyncio
    async def test_single_app_launch_is_terminal(self) -> None:
        """Single-step heuristic patterns must produce exactly one terminal task."""
        d = make_decomposer()
        for goal in [
            "open notepad",
            "open calculator",
            "open edge",
            "launch edge",
            "open chrome",
            "open vs code",
            "open discord",
        ]:
            tasks = await d.decompose(goal)
            assert len(tasks) == 1, f"Expected 1 task for {goal!r}, got {len(tasks)}"
            assert tasks[0].terminal, f"Task for {goal!r} should be terminal=True"

    @pytest.mark.asyncio
    async def test_multi_step_heuristic_last_task_is_terminal(self) -> None:
        """For multi-step heuristic patterns, only the LAST task is terminal."""
        d = make_decomposer()
        tasks = await d.decompose("Create a Flask project")
        assert len(tasks) > 1
        assert tasks[-1].terminal, "Last task should be terminal=True"
        for t in tasks[:-1]:
            assert not t.terminal, f"Non-final task {t.description!r} should not be terminal"

    @pytest.mark.asyncio
    async def test_unknown_passthrough_is_terminal(self) -> None:
        """Unknown goals produce a single-task passthrough that is terminal."""
        d = make_decomposer()
        tasks = await d.decompose("Completely unknown goal XYZ987")
        assert len(tasks) == 1
        assert tasks[0].terminal

    @pytest.mark.asyncio
    async def test_terminal_survives_state_transitions(self) -> None:
        """terminal flag is preserved across all state transitions."""
        d = make_decomposer()
        t = (await d.decompose("open notepad"))[0]
        assert t.terminal
        assert t.mark_running().terminal
        assert t.mark_running().mark_completed("done").terminal
        assert t.mark_skipped("skip").terminal


# ═══════════════════════════════════════════════════════════════════════════════
# Heuristic-first — LLM is bypassed for known patterns
# ═══════════════════════════════════════════════════════════════════════════════


class TestHeuristicFirst:
    """Verify the heuristic takes precedence over LLM for known goals."""

    @pytest.mark.asyncio
    async def test_app_launch_bypasses_llm(self) -> None:
        """'open edge' must produce 1 heuristic task even when LLM is available."""
        llm = make_mock_llm(
            '[{"description":"Find Edge","utterance":"find edge on desktop"},'
            '{"description":"Double-click","utterance":"double click edge"},'
            '{"description":"Minimise","utterance":"minimise other windows"}]'
        )
        d = make_decomposer()
        tasks = await d.decompose("open edge", llm_client=llm)
        assert len(tasks) == 1, (
            f"Expected 1 heuristic task for 'open edge', got {len(tasks)}"
        )
        assert tasks[0].terminal
        llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_notepad_bypasses_llm(self) -> None:
        llm = make_mock_llm('[{"description":"Find Notepad","utterance":"find notepad"}]')
        d = make_decomposer()
        tasks = await d.decompose("open notepad", llm_client=llm)
        assert len(tasks) == 1
        llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_heuristic_goal_uses_llm(self) -> None:
        """Goals not in the heuristic table must consult the LLM."""
        llm = make_mock_llm(
            '[{"description":"Step 1","utterance":"do step one"},'
            '{"description":"Step 2","utterance":"do step two"}]'
        )
        d = make_decomposer()
        tasks = await d.decompose(
            "Reconfigure my entire dev environment from scratch",
            llm_client=llm,
        )
        llm.complete.assert_called_once()
        assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_flask_heuristic_beats_llm(self) -> None:
        """Flask project is in the heuristic table — LLM must be bypassed."""
        llm = make_mock_llm('[{"description":"LLM step","utterance":"llm step"}]')
        d = make_decomposer()
        tasks = await d.decompose("Create a Flask project", llm_client=llm)
        llm.complete.assert_not_called()
        assert len(tasks) >= 3


# ═══════════════════════════════════════════════════════════════════════════════
# LLM terminal field parsing
# ═══════════════════════════════════════════════════════════════════════════════


class TestLLMTerminalParsing:
    """Verify the LLM terminal=true JSON field is parsed correctly."""

    @pytest.mark.asyncio
    async def test_llm_terminal_true_parsed(self) -> None:
        llm = make_mock_llm(
            '[{"description":"Do A","utterance":"do a","terminal":false},'
            '{"description":"Do B","utterance":"do b","terminal":true}]'
        )
        d = make_decomposer()
        tasks = await d.decompose("XYZ non-heuristic goal entirely new", llm_client=llm)
        assert len(tasks) == 2
        assert not tasks[0].terminal
        assert tasks[1].terminal

    @pytest.mark.asyncio
    async def test_llm_missing_terminal_defaults_false(self) -> None:
        """Old LLM responses without 'terminal' key default to terminal=False."""
        llm = make_mock_llm(
            '[{"description":"Step A","utterance":"do a"},'
            '{"description":"Step B","utterance":"do b"}]'
        )
        d = make_decomposer()
        tasks = await d.decompose("XYZ non-heuristic goal entirely new", llm_client=llm)
        assert not tasks[0].terminal
        assert not tasks[1].terminal
