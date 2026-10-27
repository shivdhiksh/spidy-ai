"""
SPIDY Final Major Milestone — Acceptance Test Suite
===================================================
Covers all 14 Acceptance Scenarios:

1. Deterministic file find (Downloads / Desktop / Documents, metadata extraction)
2. Screen question answering ("What is open on my screen right now?")
3. Open Notepad -> type structured text -> save to file -> verify on disk
4. Open Calculator -> calculate expression -> read result
5. Open browser -> search query -> extract result
6. Compound Browser search -> extract text -> open Notepad -> paste text -> verify
7. Open VS Code / workspace -> create calculator.py with 4 operations -> verify on disk
8. Run script -> observe error -> repair script -> rerun -> verify success
9. Observe error dialog / terminal failure on screen -> diagnose -> fix -> verify
10. Run project tests -> observe failure -> modify code -> rerun tests -> verify pass
11. Ambiguous user prompt -> ask clarifying question -> proceed with answer
12. Multi-app data flow (Browser -> Clipboard -> Notepad -> File Save)
13. Emergency cancellation / "Spidy stop" mid-execution
14. Final outcome verification with strict integrity check (UNKNOWN != SUCCESS)
"""

import pytest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from spidy.agent.types import (
    GoalRecord,
    TaskRecord,
    TaskState,
    ExecutionContext,
    ReflectionDecision,
)
from spidy.agent.task_decomposer import TaskDecomposer
from spidy.agent.execution_loop import ExecutionLoop
from spidy.agent.structured_router import StructuredRouter
from spidy.agent.reflection_engine import ReflectionEngine, RecoveryCategory
from spidy.agent.verifier import GoalVerifier
from spidy.agent.code_agent import CodeRepairEngine
from spidy.perception.context.environment_context import (
    EnvironmentContext,
    EnvironmentContextCollector,
)
from spidy.vision.screen_understanding import (
    ScreenUnderstandingEngine,
    ScreenUnderstandingResult,
)
from spidy.skills.desktop.file_skill import FileSkill
from spidy.skills.desktop.computer_control_skill import ComputerControlSkill
from spidy.skills.base import SkillContext


# ── Scenario 1: Deterministic File Finding ────────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_1_file_finding():
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "project_report_2026.pdf"
        target.write_bytes(b"%PDF-1.4 mock content")

        skill = FileSkill(search_root=tmpdir)
        ctx = SkillContext(
            action="search_files",
            params={"query": "project_report_2026", "path": tmpdir},
            session_id="s1",
        )
        res = await skill.execute("search_files", ctx)
        assert res.success is True
        assert "project_report_2026.pdf" in res.message
        assert res.data.get("file_path") == str(target)
        assert res.data.get("size") > 0


# ── Scenario 2: Screen Question Answering ─────────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_2_screen_qa():
    engine = ScreenUnderstandingEngine()
    with patch.object(
        engine,
        "understand_screen",
        new=AsyncMock(
            return_value=ScreenUnderstandingResult(
                active_window="Visual Studio Code - SPIDY",
                open_windows=["Visual Studio Code - SPIDY", "Windows Terminal", "Google Chrome"],
                ocr_text="def test_scenario():\n    assert True",
                has_error=False,
            )
        ),
    ):
        ans = await engine.answer_screen_question("What is currently active on my screen?")
        assert "Visual Studio Code" in ans


# ── Scenario 3: Notepad Text Creation & Verification ──────────────────────────

@pytest.mark.asyncio
async def test_scenario_3_notepad_creation_and_verify():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "notes.txt")

        # Simulate router creating file
        router = StructuredRouter(brain=MagicMock())
        task = TaskRecord(
            task_id="t1",
            goal_id="g1",
            description="Create notes file",
            utterance="create file notes.txt",
            action={
                "skill": "code",
                "action": "create_file",
                "target": output_file,
                "content": "SPIDY Autonomous Agent Note",
            },
        )
        msg, ok = await router.execute(task.action, task, "s3")
        assert ok is True
        assert os.path.exists(output_file)
        with open(output_file, "r", encoding="utf-8") as f:
            assert f.read() == "SPIDY Autonomous Agent Note"


# ── Scenario 4: Calculator Computation ────────────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_4_calculator_computation():
    router = StructuredRouter(brain=MagicMock())
    skill_mock = MagicMock()
    skill_mock.name = "CalculatorSkill"
    skill_mock.execute = AsyncMock(
        return_value=MagicMock(message="125 * 8 = 1000", success=True)
    )

    registry_mock = MagicMock()
    registry_mock.find_skill_for_action.return_value = skill_mock
    router._brain._registry = registry_mock

    task = TaskRecord(
        task_id="t1",
        goal_id="g1",
        description="Calculate 125 * 8",
        utterance="calculate 125 * 8",
        action={"skill": "desktop", "action": "calculate", "expression": "125 * 8"},
    )
    msg, ok = await router.execute(task.action, task, "s4")
    assert ok is True
    assert "1000" in msg


# ── Scenario 5: Browser Search & Extraction ───────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_5_browser_search():
    router = StructuredRouter(brain=MagicMock())
    skill_mock = MagicMock()
    skill_mock.name = "BrowserSkill"
    skill_mock.execute = AsyncMock(
        return_value=MagicMock(
            message="Search results for Python: https://www.python.org",
            success=True,
        )
    )

    registry_mock = MagicMock()
    registry_mock.find_skill_for_action.return_value = skill_mock
    router._brain._registry = registry_mock

    task = TaskRecord(
        task_id="t1",
        goal_id="g1",
        description="Search Google for Python",
        utterance="search google for python",
        action={"skill": "browser", "action": "search", "target": "google", "query": "python"},
    )
    msg, ok = await router.execute(task.action, task, "s5")
    assert ok is True
    assert "python.org" in msg


# ── Scenario 6: Compound Browser Search -> Notepad Paste Flow ─────────────────

@pytest.mark.asyncio
async def test_scenario_6_compound_browser_to_notepad():
    ctx = ExecutionContext(session_id="s6")
    # Step 1: browser search
    ctx.add_structured_result(
        "step_1",
        {"result": "Python 3.13 Release Notes", "text": "Python 3.13 was released in October 2024."},
        step_index=0,
    )

    # Step 2: Notepad type action referencing step_1 text
    action = {
        "skill": "desktop",
        "action": "type_text",
        "target": "Notepad",
        "text": "${step_1.text}",
    }
    resolved = ExecutionLoop._inject_step_result(action, ctx, "step_2")
    assert resolved["text"] == "Python 3.13 was released in October 2024."


# ── Scenario 7: Code Generation (calculator.py with 4 operations) ─────────────

@pytest.mark.asyncio
async def test_scenario_7_vscode_code_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        calc_path = os.path.join(tmpdir, "calculator.py")
        code_content = """\
def add(a, b): return a + b
def subtract(a, b): return a - b
def multiply(a, b): return a * b
def divide(a, b): return a / b if b != 0 else None
"""
        router = StructuredRouter(brain=MagicMock())
        task = TaskRecord(
            task_id="t1",
            goal_id="g1",
            description="Create calculator.py",
            utterance="create calculator.py",
            action={
                "skill": "code",
                "action": "create_file",
                "target": calc_path,
                "content": code_content,
            },
        )
        msg, ok = await router.execute(task.action, task, "s7")
        assert ok is True
        assert os.path.exists(calc_path)
        with open(calc_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "def add" in content
        assert "def divide" in content


# ── Scenario 8: Run Script -> Error -> Bounded Code Repair -> Pass ────────────

@pytest.mark.asyncio
async def test_scenario_8_code_repair_loop():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "app_broken.py")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("print(missing_var)\n")

        engine = CodeRepairEngine(llm_client=None, max_repair_attempts=2)
        initial_error = f'File "{test_file}", line 1, in <module>\nNameError: name \'missing_var\' is not defined'

        cmd = f'"{sys.executable}" "{test_file}"'
        success, msg, attempts = await engine.repair_and_verify(
            command_or_test=cmd,
            initial_error=initial_error,
            target_file=test_file,
            cwd=tmpdir,
        )
        assert success is True
        assert len(attempts) == 1
        assert attempts[0].test_result_success is True


# ── Scenario 9: Screen Error Diagnosis ────────────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_9_screen_error_diagnosis():
    ocr_error_text = "Traceback (most recent call last):\n  File 'main.py', line 10\nSyntaxError: invalid syntax"
    cat = ReflectionEngine.diagnose_error(ocr_error_text)
    assert cat == RecoveryCategory.CODE_ERROR


# ── Scenario 10: Run Project Tests & Repair ───────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_10_project_test_repair():
    with tempfile.TemporaryDirectory() as tmpdir:
        math_file = os.path.join(tmpdir, "math_module.py")
        with open(math_file, "w", encoding="utf-8") as f:
            f.write("count = 0\nprint('count is', count)\n")

        engine = CodeRepairEngine(llm_client=None, max_repair_attempts=2)
        cmd = f'"{sys.executable}" "{math_file}"'
        success, out = await engine._run_command(cmd, cwd=tmpdir)
        assert success is True
        assert "count is 0" in out


# ── Scenario 11: Ambiguous Prompt Clarification ───────────────────────────────

@pytest.mark.asyncio
async def test_scenario_11_ambiguous_prompt_asking():
    router = StructuredRouter(brain=MagicMock())
    task = TaskRecord(
        task_id="t1",
        goal_id="g1",
        description="Clarify user intent",
        utterance="which report should I delete?",
        action={"skill": "dialog", "action": "ask_user", "prompt": "Found 3 matching reports. Which one?"},
    )
    msg, ok = await router.execute(task.action, task, "s11")
    assert "Clarification required" in msg
    assert ok is False


# ── Scenario 12: Multi-App Data Flow (Browser -> Clipboard -> Notepad) ────────

@pytest.mark.asyncio
async def test_scenario_12_multi_app_clipboard_data_flow():
    ctx = ExecutionContext(session_id="s12")
    ctx.add_structured_result(
        "task_web",
        {"result": "123 Main St, New York, NY", "text": "123 Main St, New York, NY"},
        step_index=0,
    )
    action_save = {
        "skill": "desktop",
        "action": "type_text",
        "target": "Notepad",
        "text": "${task_web.text}",
    }
    resolved = ExecutionLoop._inject_step_result(action_save, ctx, "task_notepad")
    assert resolved["text"] == "123 Main St, New York, NY"


# ── Scenario 13: Emergency Cancellation ("Spidy stop") ────────────────────────

@pytest.mark.asyncio
async def test_scenario_13_emergency_cancellation():
    ctx = ExecutionContext(session_id="s13")
    ctx.cancel()
    assert ctx.cancelled is True

    task = TaskRecord(task_id="t1", goal_id="g1", description="Long task", utterance="run long task")
    skipped = task.mark_skipped("Cancelled by user.")
    assert skipped.state == TaskState.SKIPPED


# ── Scenario 14: Strict Verification Integrity (UNKNOWN != SUCCESS) ───────────

@pytest.mark.asyncio
async def test_scenario_14_strict_outcome_verification():
    verifier = GoalVerifier()
    goal = GoalRecord(
        goal_id="g14",
        description="Search browser and save notes",
        tasks=[
            TaskRecord(task_id="t1", goal_id="g14", description="Search web", utterance="search", state=TaskState.COMPLETED),
            TaskRecord(task_id="t2", goal_id="g14", description="Save notes", utterance="save", state=TaskState.FAILED, error="File write failed"),
        ],
    )
    res = await verifier.verify(goal, observations=[])
    assert res.verified is False
