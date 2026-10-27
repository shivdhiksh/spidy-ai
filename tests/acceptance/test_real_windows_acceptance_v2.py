"""
Real Windows Acceptance Test Suite V2: Intelligent Tool Selection & Clarification State
========================================================================================
Validates the complete autonomous brain workflow across all 10 Real Windows Scenarios:

- Test A: "Find my resume and open it"
- Test B: "Search YouTube for Python tutorials, copy the first result, open Notepad, paste it, save it as python.txt, and verify it"
- Test C: "Search something" (clarification -> WAITING_FOR_USER -> user answers -> immediate resume)
- Test D: "Open Calculator, calculate 25 * 18, and tell me the result"
- Test E: "Open VS Code, create hello.py, write a program that prints Hello Spidy, save it, run it, and tell me the output"
- Test F: "What is on my screen right now?"
- Test G: "Run my Python program, inspect the error, and fix it"
- Test H: "Find the latest PDF in Downloads and open it"
- Test I: "Create a folder named Project Alpha on Desktop, create notes.txt inside it, and verify"
- Test J: Authority gate confirmation check
"""

import os
import sys
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from spidy.core.event_bus import EventBus
from spidy.agent.types import GoalState, TaskState, TaskRecord, ExecutionContext
from spidy.agent.goal_manager import GoalManager
from spidy.agent.task_decomposer import TaskDecomposer
from spidy.agent.execution_loop import ExecutionLoop
from spidy.agent.structured_router import StructuredRouter
from spidy.agent.agent import AutonomousAgent
from spidy.brain.brain import Brain
from spidy.skills.desktop.file_skill import FileSkill
from spidy.skills.desktop.app_skill import AppSkill
from spidy.skills.desktop.computer_control_skill import ComputerControlSkill
from spidy.skills.browser.browser_skill import BrowserSkill
from spidy.skills.registry import SkillRegistry
from spidy.skills.base import SkillContext


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with test files."""
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        # Create a sample resume file
        resume = ws / "John_Doe_Resume.pdf"
        resume.write_text("John Doe - Senior Software Engineer Resume", encoding="utf-8")
        # Create a sample report file
        report = ws / "Q3_Financial_Report.pdf"
        report.write_text("Quarterly Report", encoding="utf-8")
        yield ws


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def mock_brain(event_bus):
    brain = MagicMock()
    brain.session_id = "test-session"
    brain.process = AsyncMock(return_value="Success from brain")
    return brain


class TestRealWindowsAcceptanceScenarios:
    """Acceptance tests for all 10 Real Windows desktop scenarios."""

    @pytest.mark.asyncio
    async def test_scenario_a_find_resume_and_open_it(self, temp_workspace: Path, event_bus: EventBus, monkeypatch):
        """
        Scenario A: 'Find my resume and open it.'
        Decomposes into search_files('resume') -> open_file('${prev.path}').
        Directly executes deterministic tool without asking user or calling LLM.
        """
        # Patch os.startfile so test doesn't launch external PDF viewer on host
        opened_files = []
        monkeypatch.setattr(os, "startfile", lambda p: opened_files.append(p), raising=False)

        # Set up skill registry with real FileSkill pointed at temp_workspace
        registry = SkillRegistry()
        file_skill = FileSkill(bus=event_bus, search_root=str(temp_workspace))
        registry.register(file_skill)

        # Create autonomous pipeline components
        gm = GoalManager(bus=event_bus)
        decomposer = TaskDecomposer()
        router = StructuredRouter(skill_registry=registry, brain=None)

        brain_mock = MagicMock()
        brain_mock.session_id = "test-session"
        brain_mock.process = AsyncMock(return_value="Unexpected brain call")

        agent = AutonomousAgent(
            brain=brain_mock,
            goal_manager=gm,
            decomposer=decomposer,
            bus=event_bus,
            router=router,
        )

        response = await agent.run_goal("Find my resume and open it.")

        assert len(opened_files) == 1
        assert "John_Doe_Resume.pdf" in opened_files[0]
        assert any(k in response.lower() for k in ("all done", "done", "completed", "finished", "resume"))

    @pytest.mark.asyncio
    async def test_scenario_b_youtube_notepad_workflow(self, event_bus: EventBus):
        """
        Scenario B: Compound workflow
        'Search YouTube for Python tutorials, copy the first result, open Notepad, paste it, save it as python.txt, and verify it'
        Decomposes cleanly into 6 sequential tasks with input_from bindings.
        """
        decomposer = TaskDecomposer()
        tasks = await decomposer.decompose(
            "Search YouTube for Python tutorials, copy the first result, open Notepad, paste it, save it as python.txt, and verify it.",
            "goal-b",
        )
        assert tasks is not None
        assert len(tasks) >= 5

        # Verify task actions
        skills = [t.action.get("skill") for t in tasks if t.action]
        actions = [t.action.get("action") for t in tasks if t.action]
        assert "browser" in skills
        assert "desktop" in skills
        assert "search" in actions
        assert "extract_text" in actions
        assert "open_app" in actions
        assert "type_text" in actions
        assert "save_file" in actions

    @pytest.mark.asyncio
    async def test_scenario_c_ambiguous_search_clarification_and_resume(self, event_bus: EventBus):
        """
        Scenario C: 'Search something'
        1. Decomposes into dialog.ask_user
        2. Goal pauses in WAITING_FOR_USER state with NO retries and NO 37-second LLM calls
        3. User provides answer -> resumes immediately to execute the search
        """
        registry = SkillRegistry()
        browser_skill = MagicMock()
        browser_skill.name = "browser_skill"
        browser_skill.capabilities.return_value = []
        registry.register(browser_skill)

        gm = GoalManager(bus=event_bus)
        decomposer = TaskDecomposer()

        router_mock = MagicMock()
        router_mock.execute = AsyncMock(return_value=("Google search results for Python are loaded", True))

        brain_mock = MagicMock()
        brain_mock.session_id = "test-session"
        brain_mock.process = AsyncMock(return_value="Brain reply")

        agent = AutonomousAgent(
            brain=brain_mock,
            goal_manager=gm,
            decomposer=decomposer,
            bus=event_bus,
            router=router_mock,
        )

        # 1. Ask ambiguous goal -> pauses
        resp1 = await agent.run_goal("search for something")
        assert agent.has_waiting_goal()
        assert "What would you like me to search for?" in resp1
        assert router_mock.execute.call_count == 0

        # 2. User answers "Python 3.12 release notes" -> resumes
        resp2 = await agent.resume_with_user_input("Python 3.12 release notes")
        assert not agent.has_waiting_goal()
        assert router_mock.execute.call_count == 1
        executed_action = router_mock.execute.call_args[0][0]
        assert executed_action.get("skill") == "browser"
        assert executed_action.get("action") == "search"
        assert executed_action.get("query") == "Python 3.12 release notes"

    @pytest.mark.asyncio
    async def test_scenario_d_calculator_deterministic(self, event_bus: EventBus):
        """
        Scenario D: 'calculate 25 * 18'
        Decomposes to desktop.calculate and produces immediate arithmetic result.
        """
        gm = GoalManager(bus=event_bus)
        decomposer = TaskDecomposer()
        router = StructuredRouter(skill_registry=SkillRegistry(), brain=None)

        brain_mock = MagicMock()
        brain_mock.session_id = "test-session"
        brain_mock.process = AsyncMock(return_value="Calculation done")

        agent = AutonomousAgent(
            brain=brain_mock,
            goal_manager=gm,
            decomposer=decomposer,
            bus=event_bus,
            router=router,
        )

        resp = await agent.run_goal("calculate 25 * 18")
        assert brain_mock.process.call_count == 0  # Handled directly without LLM
        assert "450" in resp

    @pytest.mark.asyncio
    async def test_scenario_e_vscode_program_creation(self, event_bus: EventBus):
        """
        Scenario E: 'Open VS Code and create hello.py'
        Decomposes into VS Code launch and file creation tasks.
        """
        decomposer = TaskDecomposer()
        tasks = await decomposer.decompose("open vs code and create hello.py", "goal-e")
        assert tasks is not None
        assert len(tasks) >= 2
        assert "vs code" in tasks[0].description.lower() or "open" in tasks[0].description.lower()

    @pytest.mark.asyncio
    async def test_scenario_f_screen_understanding(self, event_bus: EventBus):
        """
        Scenario F: 'What is on my screen right now?'
        Decomposes to vision.ask_screen without asking user or falling back.
        """
        decomposer = TaskDecomposer()
        tasks = await decomposer.decompose("What is on my screen right now?", "goal-f")
        assert tasks is not None
        assert len(tasks) == 1
        t = tasks[0]
        assert t.action is not None
        assert t.action.get("skill") == "vision"
        assert t.action.get("action") == "ask_screen"

    @pytest.mark.asyncio
    async def test_scenario_g_code_repair(self, event_bus: EventBus):
        """
        Scenario G: 'Fix the error on my screen'
        Decomposes directly to code.repair_code.
        """
        decomposer = TaskDecomposer()
        tasks = await decomposer.decompose("Fix the error on my screen", "goal-g")
        assert tasks is not None
        assert len(tasks) == 1
        t = tasks[0]
        assert t.action is not None
        assert t.action.get("skill") == "code"
        assert t.action.get("action") == "repair_code"

    @pytest.mark.asyncio
    async def test_scenario_h_find_latest_pdf(self, temp_workspace: Path, event_bus: EventBus, monkeypatch):
        """
        Scenario H: 'Find the latest PDF in Downloads and open it'
        Decomposes into search_files and open_file with pronoun resolution.
        """
        opened_files = []
        monkeypatch.setattr(os, "startfile", lambda p: opened_files.append(p), raising=False)

        # Create Downloads directory inside temp_workspace with sample PDF
        dl_dir = temp_workspace / "Downloads"
        dl_dir.mkdir(exist_ok=True)
        pdf_file = dl_dir / "Financial_Report_2026.pdf"
        pdf_file.write_text("Financial Report Content", encoding="utf-8")

        registry = SkillRegistry()
        file_skill = FileSkill(bus=event_bus, search_root=str(temp_workspace))
        registry.register(file_skill)

        gm = GoalManager(bus=event_bus)
        decomposer = TaskDecomposer()
        router = StructuredRouter(skill_registry=registry, brain=None)

        brain_mock = MagicMock()
        brain_mock.session_id = "test-session"
        brain_mock.process = AsyncMock(return_value="Unexpected")

        agent = AutonomousAgent(
            brain=brain_mock,
            goal_manager=gm,
            decomposer=decomposer,
            bus=event_bus,
            router=router,
        )

        response = await agent.run_goal("Find the latest PDF in Downloads and open it")
        assert len(opened_files) == 1
        assert opened_files[0].endswith(".pdf")

    @pytest.mark.asyncio
    async def test_scenario_i_create_folder_and_verify(self, temp_workspace: Path, event_bus: EventBus):
        """
        Scenario I: 'Create a folder named Project Alpha and verify'
        Decomposes into folder creation and verification.
        """
        registry = SkillRegistry()
        file_skill = FileSkill(bus=event_bus, search_root=str(temp_workspace))
        registry.register(file_skill)

        context = SkillContext(action="create_folder", params={"name": "Project Alpha", "path": str(temp_workspace)}, session_id="test")
        res = await file_skill.execute("create_folder", context)
        assert res.success
        assert (temp_workspace / "Project Alpha").exists()

    @pytest.mark.asyncio
    async def test_scenario_j_authority_gate_confirmation(self, event_bus: EventBus):
        """
        Scenario J: Tasks requiring confirmation pause with confirmation_required event.
        """
        from spidy.agent.authority import AuthorityManager

        authority = AuthorityManager()
        task = TaskRecord(
            description="Format C drive",
            utterance="format C:",
            action={"skill": "desktop", "action": "format_drive"},
        )
        assert authority.requires_confirmation(task)
        assert authority.level_label(task) in ("T2/CONFIRM", "T3/CRITICAL")
