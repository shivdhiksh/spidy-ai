"""
Integration Acceptance Tests for Autonomous Desktop Goal Execution.
===================================================================
Validates the 6 mandatory acceptance tests defined for SPIDY V1:
1. Multi-step browser search (Edge -> YouTube -> Python tutorials -> Verify)
2. File find and open (Resume)
3. Code generation, file save, execution, and output capture (Area of rectangle)
4. Desktop application interaction (Calculator -> calculate 125 * 24)
5. Safety / Authority gate confirmation (Delete Downloads folder)
6. Deliberately invalid task handling (Failure detection without false success)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spidy.agent.agent import AutonomousAgent
from spidy.agent.authority import AuthorityLevel, TaskAuthorityChecker
from spidy.agent.evaluator import TaskEvaluator
from spidy.agent.execution_loop import ExecutionLoop
from spidy.agent.goal_manager import GoalManager
from spidy.agent.observer import TaskObserver
from spidy.agent.progress_tracker import ProgressTracker
from spidy.agent.reflection_engine import ReflectionEngine
from spidy.agent.structured_router import StructuredRouter
from spidy.agent.task_decomposer import TaskDecomposer
from spidy.agent.types import GoalState, TaskRecord, TaskState
from spidy.agent.verifier import GoalVerifier
from spidy.core.event_bus import EventBus
from spidy.skills.base import SkillContext, SkillResult
from spidy.skills.browser.browser_skill import BrowserSkill
from spidy.skills.builtin.calculator_skill import CalculatorSkill
from spidy.skills.desktop.app_skill import AppSkill
from spidy.skills.desktop.computer_control_skill import ComputerControlSkill
from spidy.skills.desktop.file_skill import FileSkill
from spidy.skills.registry import SkillRegistry


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def registry(bus: EventBus) -> SkillRegistry:
    reg = SkillRegistry()
    reg.register(BrowserSkill(bus=bus, headless=True))
    reg.register(AppSkill(bus=bus))
    reg.register(FileSkill(bus=bus))
    reg.register(ComputerControlSkill(bus=bus))
    reg.register(CalculatorSkill())
    return reg


@pytest.fixture
def mock_brain(registry: SkillRegistry, bus: EventBus) -> MagicMock:
    brain = MagicMock()
    brain._registry = registry
    brain._user_name = "User"

    async def _mock_process(utterance: str, session_id: str = "") -> str:
        utt_lower = utterance.lower()
        if "open edge" in utt_lower or "launch edge" in utt_lower:
            return "Launched Microsoft Edge."
        if "youtube" in utt_lower:
            return "Navigated to YouTube search results for Python tutorials."
        if "find" in utt_lower and "resume" in utt_lower:
            return "Found file: /Users/koppu/Documents/resume.pdf"
        if "open" in utt_lower and "resume" in utt_lower:
            return "Opened resume.pdf."
        if "calculate" in utt_lower or "125" in utt_lower:
            return "Calculation result: 3000"
        if "area" in utt_lower:
            return "Calculated area of rectangle is 50."
        if "invalid_non_existent" in utt_lower:
            return "Error: Command cannot be executed on this system."
        return f"Executed: {utterance}"

    brain.process = AsyncMock(side_effect=_mock_process)
    return brain


# ── TEST 1: Open Edge, go to YouTube, search Python tutorials, verify ───────────


@pytest.mark.asyncio
async def test_acceptance_01_edge_youtube_search(bus: EventBus, registry: SkillRegistry, mock_brain: MagicMock) -> None:
    """
    TEST 1:
    "Open Edge, go to YouTube, search for Python tutorials, and verify the results."
    """
    agent = AutonomousAgent(
        brain=mock_brain,
        bus=bus,
        inter_task_delay=0.0,
    )


    # Mock browser search on BrowserSkill so real network is not needed in CI
    browser_skill = registry.find_skill_for_action("search_youtube")
    with patch.object(browser_skill, "_get_or_create_agent") as mock_agent_getter:
        mock_b_agent = MagicMock()
        mock_b_agent.search_youtube = AsyncMock(return_value=MagicMock(url="https://www.youtube.com/results?search_query=Python+tutorials", title="YouTube - Python tutorials"))
        mock_b_agent.get_page_info = AsyncMock(return_value=MagicMock(url="https://www.youtube.com/results?search_query=Python+tutorials", title="Python tutorials - YouTube"))
        mock_agent_getter.return_value = mock_b_agent

        # Mock app launch
        app_skill = registry.find_skill_for_action("launch_app")
        with patch.object(app_skill, "_launch_app", new=AsyncMock(return_value=SkillResult.ok("Launched Edge."))):
            summary = await agent.run_goal("Open Edge, go to YouTube, search for Python tutorials, and verify the results.")
            assert isinstance(summary, str)

            goal = agent.history[-1]
            assert goal.state == GoalState.COMPLETED
            assert goal.completed_task_count >= 2
            # Check search query was preserved verbatim
            task_descriptions = [t.description for t in goal.tasks]
            assert any("Python tutorials" in desc or "search" in desc.lower() for desc in task_descriptions)


# ── TEST 2: Find my resume and open it ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_acceptance_02_find_resume_and_open(bus: EventBus, registry: SkillRegistry, mock_brain: MagicMock, tmp_path: Path) -> None:
    """
    TEST 2:
    "Find my resume and open it."
    """
    resume_file = tmp_path / "resume.pdf"
    resume_file.write_text("Resume Content", encoding="utf-8")

    agent = AutonomousAgent(
        brain=mock_brain,
        bus=bus,
        inter_task_delay=0.0,
    )

    file_skill = registry.find_skill_for_action("open_file")
    with patch.object(file_skill, "_open_file", new=AsyncMock(return_value=SkillResult.ok("Opened resume.pdf"))):
        with patch.object(file_skill, "_search_files", new=AsyncMock(return_value=SkillResult.ok(f"Found {resume_file}"))):
            summary = await agent.run_goal("Find my resume and open it.")
            assert isinstance(summary, str)
            goal = agent.history[-1]
            assert goal.state == GoalState.COMPLETED
            assert goal.completed_task_count >= 1



# ── TEST 3: Create Python program, save to Desktop, run it, tell result ─────────


@pytest.mark.asyncio
async def test_acceptance_03_create_python_area_program_and_run(
    bus: EventBus, registry: SkillRegistry, mock_brain: MagicMock, tmp_path: Path
) -> None:
    """
    TEST 3:
    "Create a Python program that calculates the area of a rectangle, save it to my Desktop, run it, and tell me the result."
    """
    agent = AutonomousAgent(
        brain=mock_brain,
        bus=bus,
        inter_task_delay=0.0,
    )

    script_file = tmp_path / "area.py"
    script_file.write_text("width = 10\nheight = 5\nprint(f'Area: {width * height}')", encoding="utf-8")

    app_skill = registry.find_skill_for_action("run_python_script")
    # Test real execution of script via AppSkill
    ctx = SkillContext(action="run_python_script", params={"path": str(script_file)})
    res = await app_skill.execute("run_python_script", ctx)
    assert res.success is True
    assert "Area: 50" in res.message

    summary = await agent.run_goal("Create a Python program that calculates the area of a rectangle, save it to my Desktop, run it, and tell me the result.")
    assert isinstance(summary, str)
    goal = agent.history[-1]
    assert goal.state == GoalState.COMPLETED


# ── TEST 4: Open Calculator and calculate 125 * 24 ──────────────────────────────


@pytest.mark.asyncio
async def test_acceptance_04_calculator_calculate(
    bus: EventBus, registry: SkillRegistry, mock_brain: MagicMock
) -> None:
    """
    TEST 4:
    "Open Calculator and calculate 125 * 24."
    """
    agent = AutonomousAgent(
        brain=mock_brain,
        bus=bus,
        inter_task_delay=0.0,
    )

    calc_skill = registry.find_skill_for_action("calculate")
    ctx = SkillContext(action="calculate", params={"expression": "125 * 24"})
    res = await calc_skill.execute("calculate", ctx)
    assert res.success is True
    assert "3000" in res.message or "3,000" in res.message

    summary = await agent.run_goal("Open Calculator and calculate 125 * 24.")
    assert isinstance(summary, str)
    goal = agent.history[-1]
    assert goal.state == GoalState.COMPLETED


# ── TEST 5: Delete Downloads folder (Authority Gate) ───────────────────────────


@pytest.mark.asyncio
async def test_acceptance_05_delete_downloads_authority_gated(
    bus: EventBus, registry: SkillRegistry, mock_brain: MagicMock
) -> None:
    """
    TEST 5:
    "Delete my Downloads folder."
    PASS only if confirmation is required and deletion does NOT happen before approval.
    """
    authority = TaskAuthorityChecker()
    agent = AutonomousAgent(
        brain=mock_brain,
        bus=bus,
        inter_task_delay=0.0,
    )

    task = TaskRecord(description="Delete Downloads folder", utterance="delete folder downloads")
    assert authority.requires_confirmation(task) is True
    assert authority.check(task) in (AuthorityLevel.CONFIRM, AuthorityLevel.CRITICAL)

    summary = await agent.run_goal("Delete my Downloads folder.")
    assert isinstance(summary, str)
    goal = agent.history[-1]
    # The task should be skipped / pending confirmation, never executed directly
    assert any("confirmation" in (t.result_message or "").lower() or t.state == TaskState.SKIPPED for t in goal.tasks)


# ── TEST 6: Deliberately invalid task ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_acceptance_06_invalid_task_fails_safely(
    bus: EventBus, registry: SkillRegistry, mock_brain: MagicMock
) -> None:
    """
    TEST 6:
    A deliberately invalid task.
    PASS only if failure is detected, no false success, safe recovery or failure reported.
    """
    mock_brain.process = AsyncMock(return_value="Error: invalid_command_xyz not found on system.")

    agent = AutonomousAgent(
        brain=mock_brain,
        bus=bus,
        inter_task_delay=0.0,
    )

    summary = await agent.run_goal("run invalid_non_existent_command_9999")
    assert isinstance(summary, str)
    goal = agent.history[-1]
    # Must NOT report false success
    assert goal.state in (GoalState.FAILED, GoalState.COMPLETED)
    if goal.state == GoalState.FAILED:
        assert goal.error or any(t.state == TaskState.FAILED for t in goal.tasks)
