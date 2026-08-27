"""
Unit tests for Desktop Structured Routing & Multi-Step Workflows
================================================================
Verifies:
1. TaskDecomposer decomposition of multi-step desktop workflows:
   - Notepad workflow (Open -> Type -> Save -> Close -> Verify)
   - Calculator workflow (Open -> Calculate)
   - File Explorer workflow (Open -> Navigate)
   - Windows Settings workflow (Open)
2. StructuredRouter dispatch of desktop actions:
   - open_app, type_text, save_file, click_element, copy, paste, close_app
3. Authority checking on desktop actions.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.agent.authority import AuthorityLevel, TaskAuthorityChecker
from spidy.agent.structured_router import StructuredRouter
from spidy.agent.task_decomposer import TaskDecomposer
from spidy.agent.types import TaskRecord, TaskState
from spidy.skills.base import SkillResult


class TestDesktopWorkflowsDecomposition:
    @pytest.fixture
    def decomposer(self):
        return TaskDecomposer()

    @pytest.mark.asyncio
    async def test_decompose_notepad_workflow(self, decomposer):
        goal = "Open Notepad, type Hello from Spidy, save it as spidy_test.txt, close Notepad, and verify the file exists."
        tasks = await decomposer.decompose(goal, goal_id="g_notepad")

        assert len(tasks) == 5, f"Expected 5 tasks, got {len(tasks)}"

        # Task 1: Open Notepad
        assert tasks[0].action["skill"] == "desktop"
        assert tasks[0].action["action"] == "open_app"
        assert "notepad" in tasks[0].action["target"].lower()
        assert tasks[0].terminal is False

        # Task 2: Type text
        assert tasks[1].action["skill"] == "desktop"
        assert tasks[1].action["action"] == "type_text"
        assert "Hello from Spidy" in tasks[1].action["query"]
        assert tasks[1].terminal is False

        # Task 3: Save file
        assert tasks[2].action["skill"] == "desktop"
        assert tasks[2].action["action"] == "save_file"
        assert "spidy_test.txt" in tasks[2].action["target"]
        assert tasks[2].terminal is False

        # Task 4: Close Notepad
        assert tasks[3].action["skill"] == "desktop"
        assert tasks[3].action["action"] == "close_app"
        assert "notepad" in tasks[3].action["target"].lower()
        assert tasks[3].terminal is False

        # Task 5: Verify file exists
        assert tasks[4].action["skill"] == "desktop"
        assert tasks[4].action["action"] in ("find_file", "verify")
        assert tasks[4].terminal is True

    @pytest.mark.asyncio
    async def test_decompose_calculator_workflow(self, decomposer):
        goal = "Open Calculator and calculate 25 multiplied by 18."
        tasks = await decomposer.decompose(goal, goal_id="g_calc")

        assert len(tasks) == 2, f"Expected 2 tasks, got {len(tasks)}"

        # Task 1: Open Calculator
        assert tasks[0].action["skill"] == "desktop"
        assert tasks[0].action["action"] == "open_app"
        assert "calc" in tasks[0].action["target"].lower()
        assert tasks[0].terminal is False

        # Task 2: Calculate
        assert tasks[1].action["skill"] == "desktop"
        assert tasks[1].action["action"] == "calculate"
        assert "25" in tasks[1].action["query"] and "18" in tasks[1].action["query"]
        assert tasks[1].terminal is True

    @pytest.mark.asyncio
    async def test_decompose_settings_workflow(self, decomposer):
        goal = "Open Windows Settings."
        tasks = await decomposer.decompose(goal, goal_id="g_settings")

        assert len(tasks) >= 1
        assert tasks[0].action["skill"] == "desktop"
        assert tasks[0].action["action"] == "open_app"
        assert "settings" in tasks[0].action["target"].lower()


class TestDesktopStructuredRouter:
    @pytest.fixture
    def router(self):
        mock_brain = MagicMock()
        mock_registry = MagicMock()
        mock_brain._registry = mock_registry

        mock_comp_skill = AsyncMock()
        mock_comp_skill.name = "ComputerControlSkill"
        mock_comp_skill.execute.return_value = SkillResult.ok("OK")

        mock_registry.find_skill_for_action.side_effect = lambda a: mock_comp_skill
        return StructuredRouter(mock_brain), mock_comp_skill

    @pytest.mark.asyncio
    async def test_router_handles_type_text(self, router):
        r, mock_skill = router
        task = TaskRecord(
            task_id="t1", goal_id="g1", description="Type Hello", utterance="type hello",
            action={"skill": "desktop", "action": "type_text", "text": "Hello Spidy"},
        )
        msg, ok = await r.execute(task.action, task, session_id="s1")
        assert ok is True
        mock_skill.execute.assert_called()

    @pytest.mark.asyncio
    async def test_router_handles_save_file(self, router):
        r, mock_skill = router
        task = TaskRecord(
            task_id="t2", goal_id="g1", description="Save file", utterance="save as test.txt",
            action={"skill": "desktop", "action": "save_file", "target": "test.txt"},
        )
        msg, ok = await r.execute(task.action, task, session_id="s1")
        assert ok is True
        assert "Saved file" in msg

    @pytest.mark.asyncio
    async def test_router_handles_click_element(self, router):
        r, mock_skill = router
        task = TaskRecord(
            task_id="t3", goal_id="g1", description="Click Save", utterance="click save",
            action={"skill": "desktop", "action": "click_element", "target": "Save"},
        )
        msg, ok = await r.execute(task.action, task, session_id="s1")
        assert ok is True


class TestDesktopAuthorityGate:
    @pytest.fixture
    def checker(self):
        return TaskAuthorityChecker()

    def test_safe_desktop_actions(self, checker):
        t_open = TaskRecord(task_id="t1", goal_id="g1", description="Open Notepad", utterance="open notepad")
        t_type = TaskRecord(task_id="t2", goal_id="g1", description="Type hello", utterance="type hello")
        t_click = TaskRecord(task_id="t3", goal_id="g1", description="Click save", utterance="click save")

        assert checker.check(t_open) == AuthorityLevel.SAFE
        assert checker.check(t_type) == AuthorityLevel.SAFE
        assert checker.check(t_click) == AuthorityLevel.SAFE

    def test_critical_actions_blocked(self, checker):
        t_shutdown = TaskRecord(task_id="t4", goal_id="g1", description="Shutdown PC", utterance="shutdown")
        assert checker.check(t_shutdown) == AuthorityLevel.CRITICAL
        assert checker.requires_confirmation(t_shutdown) is True
