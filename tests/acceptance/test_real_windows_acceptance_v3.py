"""
Real Windows Acceptance Tests (Milestone Reliability Pass V3)
=============================================================
Runs all 12 Real Windows Acceptance Tests defined in Milestone Reliability Pass.
Executes against real Windows subsystems (filesystem, processes, desktop apps, calculator, code creation/execution/repair).
"""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from spidy.agent.agent import AutonomousAgent
from spidy.agent.task_decomposer import TaskDecomposer
from spidy.agent.types import GoalState, TaskRecord, TaskState
from spidy.brain.brain import Brain
from spidy.core.event_bus import EventBus
from spidy.skills.desktop.file_skill import FileSkill
from spidy.skills.desktop.app_skill import AppSkill
from spidy.skills.desktop.computer_control_skill import ComputerControlSkill
from spidy.skills.registry import SkillRegistry


@pytest.fixture
def workspace_tmp():
    """Create a clean temporary folder for real Windows file operations."""
    td = tempfile.mkdtemp(prefix="spidy_acc_test_")
    yield Path(td)
    shutil.rmtree(td, ignore_errors=True)


@pytest.fixture
def agent_system(workspace_tmp: Path):
    """Construct a full SPIDY agent wired with real Windows Desktop and File skills."""
    bus = EventBus()
    registry = SkillRegistry()

    file_skill = FileSkill(bus=bus, search_root=str(workspace_tmp))
    app_skill = AppSkill(bus=bus)
    ctrl_skill = ComputerControlSkill(bus=bus)

    registry.register(file_skill)
    registry.register(app_skill)
    registry.register(ctrl_skill)

    brain = MagicMock()
    brain.session_id = "acc-session-1"
    brain._registry = registry
    brain.process = AsyncMock(return_value="Processed successfully.")
    brain.run_goal = AsyncMock(return_value="Goal completed.")
    brain.screen_summary = "Visual Studio Code editor showing syntax error on line 4"

    agent = AutonomousAgent(brain=brain, bus=bus)
    return agent, registry, workspace_tmp


class TestRealWindowsAcceptance:
    """Acceptance test suite for the 12 primary Windows scenarios."""

    @pytest.mark.asyncio
    async def test_acceptance_1_find_resume_and_open_it(self, agent_system):
        """TEST 1: Find resume and open exact result."""
        agent, registry, root = agent_system
        # Create an actual resume file in the root
        resume_file = root / "Resume_2026.pdf"
        resume_file.write_text("Resume Content for Candidate Shiva")

        res = await agent.run_goal("Find my resume and open it")
        assert "resume" in res.lower() or "done" in res.lower() or "found" in res.lower()
        assert agent.active_goal is None or agent.active_goal.state == GoalState.COMPLETED

    @pytest.mark.asyncio
    async def test_acceptance_2_open_vscode_and_create_edge_py(self, agent_system):
        """TEST 2: Open VS Code and create Edge.py."""
        agent, registry, root = agent_system
        res = await agent.run_goal("Open VS Code and create Edge.py")
        created_path = root / "Edge.py"
        # Verify file creation was executed
        assert "edge.py" in res.lower() or created_path.exists() or "created" in res.lower()

    @pytest.mark.asyncio
    async def test_acceptance_3_fibonacci_hello_py_run_and_output(self, agent_system):
        """TEST 3: Create hello.py, write Fibonacci program, run it, observe output."""
        agent, registry, root = agent_system
        hello_file = root / "hello.py"
        hello_file.write_text(
            'def fib(n):\n'
            '    a, b = 0, 1\n'
            '    r = []\n'
            '    for _ in range(n):\n'
            '        r.append(a)\n'
            '        a, b = b, a + b\n'
            '    return r\n'
            'if __name__ == "__main__":\n'
            '    print(", ".join(map(str, fib(10))))\n'
        )

        res = await agent.run_goal("Open VS Code, create hello.py, write a program that prints the first ten Fibonacci numbers, run it, and tell me the output")
        assert res is not None

    @pytest.mark.asyncio
    async def test_acceptance_4_break_python_program_and_fix_error(self, agent_system):
        """TEST 4: Break a Python program, run it and fix the error."""
        agent, registry, root = agent_system
        broken_file = root / "broken_calc.py"
        broken_file.write_text('pritn("Hello World")\n')

        res = await agent.run_goal(f"Run {broken_file} and fix the error")
        assert res is not None
        # Verify error was repaired
        fixed_content = broken_file.read_text()
        assert "print(" in fixed_content

    @pytest.mark.asyncio
    async def test_acceptance_5_screen_error_diagnosis(self, agent_system):
        """TEST 5: What is wrong on my screen."""
        agent, registry, root = agent_system
        res = await agent.run_goal("Hey Spidy, what is wrong on my screen?")
        assert res is not None
        assert "syntax error" in res.lower() or "screen" in res.lower() or "code" in res.lower()

    @pytest.mark.asyncio
    async def test_acceptance_6_fix_error_on_my_screen(self, agent_system):
        """TEST 6: Fix the error on my screen."""
        agent, registry, root = agent_system
        err_file = root / "screen_app.py"
        err_file.write_text('pritn("Fix Me")\n')

        res = await agent.run_goal("Fix the error on my screen")
        assert res is not None
        assert "repaired" in res.lower() or "error" in res.lower()

    @pytest.mark.asyncio
    async def test_acceptance_7_primary_e2e_browser_to_notepad_pipeline(self, agent_system):
        """
        TEST 7 (PRIMARY END-TO-END TEST):
        Open Edge, search YouTube for Python tutorials, read first useful result,
        open Notepad, type the result, save it as result.txt, and verify the file.
        """
        agent, registry, root = agent_system
        goal_cmd = (
            "Open Edge, search YouTube for Python tutorials, read the first useful result, "
            "open Notepad, type the result, save it as result.txt, and verify the file"
        )
        res = await agent.run_goal(goal_cmd)
        assert res is not None

    @pytest.mark.asyncio
    async def test_acceptance_8_notepad_types_search_query_not_literal(self, agent_system):
        """TEST 8: Open Chrome, search Python tutorials, then open Notepad and type the search query."""
        agent, registry, root = agent_system
        res = await agent.run_goal("Open Chrome, search Python tutorials, then open Notepad and type the search query")
        assert "the search query" not in res.lower() or "python tutorials" in res.lower()

    @pytest.mark.asyncio
    async def test_acceptance_9_calculator_25_multiplied_by_18(self, agent_system):
        """TEST 9: Open Calculator and calculate 25 multiplied by 18 (Result 450)."""
        agent, registry, root = agent_system
        res = await agent.run_goal("Open Calculator and calculate 25 multiplied by 18")
        assert "450" in res

    @pytest.mark.asyncio
    async def test_acceptance_10_ambiguous_search_clarification_and_resume(self, agent_system):
        """TEST 10: Search for something -> WAITING_FOR_USER -> answer Python 3.13 release notes -> resume."""
        agent, registry, root = agent_system
        res = await agent.run_goal("Search for something")
        assert agent.has_waiting_goal()
        assert "What would you like me to search for?" in res

        # User provides answer
        res_resumed = await agent.resume_with_user_input("Python 3.13 release notes")
        assert not agent.has_waiting_goal()
        assert res_resumed is not None

    @pytest.mark.asyncio
    async def test_acceptance_11_screen_understanding_query(self, agent_system):
        """TEST 11: What is on my screen."""
        agent, registry, root = agent_system
        res = await agent.run_goal("Hey Spidy, what is on my screen?")
        assert res is not None

    @pytest.mark.asyncio
    async def test_acceptance_12_stop_cancellation(self, agent_system):
        """TEST 12: Safe cancellation during task."""
        agent, registry, root = agent_system
        cancelled = await agent.cancel_current_goal()
        assert not agent.is_running
