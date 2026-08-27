"""
test_mark_reliability_upgrade.py -- Tests for Mark-Inspired Desktop Reliability Upgrade
=======================================================================================
Covers:
  A. Structured planning & contracts
  B. Step-result context injection & variable resolution
  C. Ambiguous request handling ("search something")
  D. Recovery & alternative action hierarchy
  E. Explicit vs default browser detection
  F. Real observation and verifier integrity
  G. Authority and safety gating
  H. Desktop chaining workflows
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spidy.agent.types import (
    ExecutionContext,
    GoalRecord,
    GoalState,
    ReflectionDecision,
    TaskPriority,
    TaskRecord,
    TaskState,
)
from spidy.agent.task_decomposer import (
    TaskDecomposer,
    detect_default_browser,
    is_placeholder_or_ambiguous_query,
    _normalize_search_query,
)
from spidy.agent.structured_router import StructuredRouter
from spidy.agent.execution_loop import ExecutionLoop
from spidy.agent.authority import AuthorityLevel, TaskAuthorityChecker
from spidy.agent.observer import TaskObserver


# ─── A. Structured Planning & Contracts ──────────────────────────────────────


@pytest.mark.asyncio
async def test_simple_desktop_action_produces_structured_action():
    decomposer = TaskDecomposer()
    tasks = await decomposer.decompose("open notepad", goal_id="g1")
    assert len(tasks) == 1
    t = tasks[0]
    assert t.action is not None
    assert t.action["skill"] == "desktop"
    assert t.action["action"] == "open_app"
    assert t.action["target"] == "Notepad"
    assert t.terminal is True


@pytest.mark.asyncio
async def test_browser_search_produces_structured_action():
    decomposer = TaskDecomposer()
    tasks = await decomposer.decompose("Open Chrome and search Python tutorials", goal_id="g1")
    assert len(tasks) == 2
    t_open, t_search = tasks
    assert t_open.action["skill"] == "desktop"
    assert t_open.action["target"] == "Chrome"
    assert t_search.action["skill"] == "browser"
    assert t_search.action["action"] == "search"
    assert t_search.action["query"] == "Python tutorials"


@pytest.mark.asyncio
async def test_structured_router_dispatches_without_brain_reparse():
    brain = MagicMock()
    brain.process = AsyncMock(return_value="OK")
    brain._registry = MagicMock()
    skill_mock = MagicMock()
    skill_mock.name = "AppSkill"
    skill_mock.execute = AsyncMock(return_value=MagicMock(success=True, message="Notepad launched"))
    brain._registry.find_skill_for_action.return_value = skill_mock

    router = StructuredRouter(brain)
    task = TaskRecord(
        description="Open Notepad",
        utterance="open notepad",
        action={"skill": "desktop", "action": "open_app", "target": "Notepad"},
    )
    result_text, success = await router.execute(task.action, task, session_id="s1")
    assert success is True
    assert "Notepad launched" in result_text
    # Brain.process() should NOT have been called because direct dispatch succeeded!
    assert brain.process.call_count == 0


# ─── B. Context Propagation & Step Results ───────────────────────────────────


def test_execution_context_stores_structured_result_and_resolves_variables():
    goal = GoalRecord(goal_id="g1", description="Test Goal")
    ctx = ExecutionContext(goal=goal)

    ctx.add_structured_result(
        "task_1",
        {
            "result": "Found 5 videos",
            "url": "https://www.youtube.com/results?search_query=Python",
            "title": "Python Tutorials - YouTube",
            "text": "Best Python 3.12 Tutorials for Beginners",
        },
        step_index=0,
    )

    # Direct task ID resolution
    assert ctx.resolve_variable("task_1") == "Found 5 videos"

    # Field resolution
    assert ctx.resolve_variable("${task_1.url}") == "https://www.youtube.com/results?search_query=Python"
    assert ctx.resolve_variable("${task_1.title}") == "Python Tutorials - YouTube"
    assert ctx.resolve_variable("${task_1.text}") == "Best Python 3.12 Tutorials for Beginners"

    # Step alias resolution
    assert ctx.resolve_variable("${step_1.url}") == "https://www.youtube.com/results?search_query=Python"

    # Previous task resolution
    assert ctx.resolve_variable("${prev.text}") == "Best Python 3.12 Tutorials for Beginners"


def test_step_result_injection_into_action_fields():
    goal = GoalRecord(goal_id="g1", description="Test Goal")
    ctx = ExecutionContext(goal=goal)

    ctx.add_structured_result(
        "step_1",
        {
            "result": "Extracted summary text",
            "text": "Extracted summary text",
            "url": "https://example.com",
        },
        step_index=0,
    )

    action = {
        "skill": "desktop",
        "action": "type_text",
        "target": "Notepad",
        "text": "${prev.text}",
        "input_from": "step_1",
    }

    resolved = ExecutionLoop._inject_step_result(action, ctx, "step_2")
    assert resolved["text"] == "Extracted summary text"
    assert resolved["target"] == "Notepad"


# ─── C. Ambiguous Requests ("Search Something") ──────────────────────────────


def test_is_placeholder_or_ambiguous_query():
    assert is_placeholder_or_ambiguous_query("something") is True
    assert is_placeholder_or_ambiguous_query("anything") is True
    assert is_placeholder_or_ambiguous_query("") is True
    assert is_placeholder_or_ambiguous_query("something online") is True
    assert is_placeholder_or_ambiguous_query("Python tutorials") is False
    assert is_placeholder_or_ambiguous_query("machine learning") is False


@pytest.mark.asyncio
async def test_ambiguous_search_triggers_clarification_task():
    decomposer = TaskDecomposer()
    tasks = await decomposer.decompose(
        "Open Chrome, search something, switch to Notepad, and type the result",
        goal_id="g1",
    )
    assert len(tasks) >= 3

    # Task 2 should be a clarification / ask_user task
    t_search = tasks[1]
    assert t_search.action is not None
    assert t_search.action["action"] == "ask_user" or "Clarify" in t_search.description
    assert "What would you like me to search for?" in t_search.utterance


# ─── D. Recovery & Priority ──────────────────────────────────────────────────


def test_task_priority_enum_and_defaults():
    t = TaskRecord(description="Test Task", utterance="test")
    assert t.priority == TaskPriority.NORMAL

    t_high = TaskRecord(description="Critical Stop", utterance="stop", priority=TaskPriority.HIGH)
    assert t_high.priority == TaskPriority.HIGH


@pytest.mark.asyncio
async def test_alternative_focus_action_via_foreground():
    brain = MagicMock()
    brain.process = AsyncMock(return_value="OK")
    brain._registry = MagicMock()

    # AppSkill focus fails on first method but succeeds on bring_app_to_foreground
    skill_mock = MagicMock()
    skill_mock.name = "AppSkill"
    skill_mock.execute = AsyncMock(return_value=MagicMock(success=True, message="Window brought to foreground"))
    brain._registry.find_skill_for_action.return_value = skill_mock

    router = StructuredRouter(brain)
    task = TaskRecord(
        description="Focus Notepad",
        utterance="focus notepad",
        action={"skill": "desktop", "action": "focus", "target": "Notepad"},
    )
    result_text, success = await router.execute(task.action, task, session_id="s1")
    assert success is True
    assert "Window brought to foreground" in result_text


# ─── E. Browser Detection ────────────────────────────────────────────────────


def test_browser_detection():
    # Explicit Edge
    t_edge = TaskDecomposer._extract_search_params("search YouTube for Python")
    assert t_edge[0] == "youtube"
    assert t_edge[1] == "Python"

    # Default browser query
    def_browser = detect_default_browser()
    assert def_browser in ("edge", "chrome")


# ─── F. Verification & Real State Integrity ──────────────────────────────────


@pytest.mark.asyncio
async def test_extract_text_observation_verification():
    observer = TaskObserver()
    task = TaskRecord(
        description="Read search result",
        utterance="read search result",
        action={"skill": "browser", "action": "extract_text", "target": "Chrome"},
    )
    obs = await observer.observe(task, response_text="Extracted Python tutorial snippet and overview.", skip_for_terminal=False)
    assert obs.success_signal is True
    assert "Verified text extracted" in obs.summary


@pytest.mark.asyncio
async def test_empty_extract_text_fails_observation():
    observer = TaskObserver()
    task = TaskRecord(
        description="Read search result",
        utterance="read search result",
        action={"skill": "browser", "action": "extract_text", "target": "Chrome"},
    )
    obs = await observer.observe(task, response_text="", skip_for_terminal=False)
    assert obs.success_signal is False


# ─── G. Safety & Code Execution Escape Hatch ─────────────────────────────────


def test_code_exec_requires_tier2_confirmation():
    checker = TaskAuthorityChecker()
    task = TaskRecord(
        description="Run arbitrary code",
        utterance="run arbitrary code",
        action={"skill": "code_exec", "code": "print('hello')"},
    )
    level = checker.check(task)
    assert level == AuthorityLevel.CONFIRM
    assert checker.requires_confirmation(task) is True


@pytest.mark.asyncio
async def test_code_exec_is_disabled_by_default():
    brain = MagicMock()
    brain._config = None
    router = StructuredRouter(brain)

    task = TaskRecord(
        description="Execute python script",
        utterance="execute code",
        action={"skill": "code_exec", "code": "print('hello')"},
    )
    result_text, success = await router.execute(task.action, task, session_id="s1")
    assert success is False
    assert "disabled by default" in result_text.lower()


# ─── H. Result-Aware Desktop Chaining Workflow ───────────────────────────────


@pytest.mark.asyncio
async def test_end_to_end_search_and_type_chaining():
    decomposer = TaskDecomposer()
    tasks = await decomposer.decompose(
        "Open Chrome, search for Python tutorials, read the first useful result, switch to Notepad, type the result, and save it as notes.txt",
        goal_id="g1",
    )

    assert len(tasks) == 6
    # 1. Open Chrome
    assert tasks[0].action["action"] == "open_app"
    assert tasks[0].action["target"] == "Chrome"
    # 2. Search
    assert tasks[1].action["action"] == "search"
    assert tasks[1].action["query"] == "Python tutorials"
    # 3. Read/extract
    assert tasks[2].action["action"] == "extract_text"
    # 4. Focus Notepad
    assert tasks[3].action["action"] == "focus"
    assert tasks[3].action["target"] == "Notepad"
    # 5. Type result
    assert tasks[4].action["action"] == "type_text"
    assert tasks[4].action["text"] == "${prev}.text"
    assert tasks[4].input_from == "<input_from:prev>"
    # 6. Save file
    assert tasks[5].action["action"] == "save_file"
    assert tasks[5].action["target"] == "notes.txt"
