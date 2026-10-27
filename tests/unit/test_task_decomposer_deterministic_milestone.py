"""
Unit Tests: TaskDecomposer Deterministic Desktop Planning & Tool Selection
==========================================================================
Verifies that common actionable Windows computer use requests decompose
directly into structured deterministic action contracts without generating
spurious ask_user clarification tasks or routing through general LLMs.
"""

import pytest
from spidy.agent.task_decomposer import TaskDecomposer
from spidy.agent.types import TaskState


@pytest.fixture
def decomposer() -> TaskDecomposer:
    return TaskDecomposer()


class TestDeterministicDecomposition:
    """Test deterministic decomposition of common computer use requests."""

    @pytest.mark.asyncio
    async def test_find_resume_and_open_it(self, decomposer: TaskDecomposer):
        """'Find my resume and open it' must decompose into file search + open."""
        tasks = await decomposer.decompose("Find my resume and open it.", "goal-1")
        assert tasks is not None
        assert len(tasks) == 2

        # Step 1: search for resume
        t1 = tasks[0]
        assert t1.action is not None
        assert t1.action.get("skill") == "file"
        assert t1.action.get("action") == "search_files"
        assert "resume" in str(t1.action.get("query")).lower()
        assert not t1.terminal

        # Step 2: open the found file
        t2 = tasks[1]
        assert t2.action is not None
        assert t2.action.get("skill") == "file"
        assert t2.action.get("action") == "open_file"
        assert t2.action.get("target") in ("${prev.path}", "${step_1.path}", "<input_from:prev.path>") or t2.input_from != ""
        assert t2.terminal

    @pytest.mark.asyncio
    async def test_find_my_resume_single(self, decomposer: TaskDecomposer):
        """'Find my resume' must be a single deterministic file search task."""
        tasks = await decomposer.decompose("Find my resume", "goal-2")
        assert tasks is not None
        assert len(tasks) == 1
        t = tasks[0]
        assert t.action is not None
        assert t.action.get("skill") == "file"
        assert t.action.get("action") == "search_files"
        assert "resume" in str(t.action.get("query")).lower()
        assert t.terminal

    @pytest.mark.asyncio
    async def test_find_latest_pdf_in_downloads(self, decomposer: TaskDecomposer):
        """'Find the latest PDF in Downloads' must match file search."""
        tasks = await decomposer.decompose("Find the latest PDF in Downloads", "goal-3")
        assert tasks is not None
        assert len(tasks) == 1
        t = tasks[0]
        assert t.action is not None
        assert t.action.get("skill") == "file"
        assert t.action.get("action") == "search_files"
        assert t.terminal

    @pytest.mark.asyncio
    async def test_calculate_expression(self, decomposer: TaskDecomposer):
        """'Calculate 25 * 18' decomposes to desktop calculate."""
        tasks = await decomposer.decompose("calculate 25 * 18", "goal-4")
        assert tasks is not None
        assert len(tasks) == 1
        t = tasks[0]
        assert t.action is not None
        assert t.action.get("skill") == "desktop"
        assert t.action.get("action") == "calculate"
        assert "25 * 18" in t.action.get("query")
        assert t.terminal

    @pytest.mark.asyncio
    async def test_screen_understanding(self, decomposer: TaskDecomposer):
        """'What is on my screen' decomposes to vision ask_screen."""
        tasks = await decomposer.decompose("what is on my screen", "goal-5")
        assert tasks is not None
        assert len(tasks) == 1
        t = tasks[0]
        assert t.action is not None
        assert t.action.get("skill") == "vision"
        assert t.action.get("action") == "ask_screen"
        assert t.terminal

    @pytest.mark.asyncio
    async def test_code_error_repair(self, decomposer: TaskDecomposer):
        """'Fix the error on my screen' decomposes to code repair_code."""
        tasks = await decomposer.decompose("fix the error on my screen", "goal-6")
        assert tasks is not None
        assert len(tasks) == 1
        t = tasks[0]
        assert t.action is not None
        assert t.action.get("skill") == "code"
        assert t.action.get("action") == "repair_code"
        assert t.terminal

    @pytest.mark.asyncio
    async def test_ambiguous_search_triggers_ask_user(self, decomposer: TaskDecomposer):
        """'Search for something' is genuinely ambiguous and MUST trigger dialog.ask_user."""
        tasks = await decomposer.decompose("search for something", "goal-7")
        assert tasks is not None
        assert len(tasks) >= 1
        t = tasks[0]
        assert t.action is not None
        assert t.action.get("skill") == "dialog"
        assert t.action.get("action") == "ask_user"
        assert "search for" in (t.action.get("prompt") or t.utterance).lower()
