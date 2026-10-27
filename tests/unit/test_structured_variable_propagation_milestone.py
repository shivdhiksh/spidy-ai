"""
Tests for Structured Variable Propagation & Reference Resolution (Phase D)
"""

import pytest
from spidy.agent.types import ExecutionContext
from spidy.agent.execution_loop import ExecutionLoop


def test_resolve_variable_all_formats():
    ctx = ExecutionContext(session_id="test_session")
    ctx.add_structured_result(
        "task_1",
        {
            "result": "https://python.org",
            "url": "https://python.org",
            "title": "Welcome to Python.org",
            "text": "Python is a programming language.",
        },
        step_index=0,
    )
    ctx.add_structured_result(
        "task_2",
        {
            "result": "C:\\Users\\test\\sample.txt",
            "file_path": "C:\\Users\\test\\sample.txt",
            "path": "C:\\Users\\test\\sample.txt",
        },
        step_index=1,
    )

    # 1. ${task_id.field}
    assert ctx.resolve_variable("${task_1.url}") == "https://python.org"
    assert ctx.resolve_variable("${task_1.title}") == "Welcome to Python.org"

    # 2. ${prev.field} from task_2
    assert ctx.resolve_variable("${prev.file_path}", current_task_id="task_3") == "C:\\Users\\test\\sample.txt"

    # 3. <input_from:task_id>
    assert ctx.resolve_variable("<input_from:task_1>") == "https://python.org"

    # 4. <input_from:step_1>
    assert ctx.resolve_variable("<input_from:step_1.url>") == "https://python.org"

    # 5. Global <task_result> alias
    assert ctx.resolve_variable("<task_result>") == "C:\\Users\\test\\sample.txt"


def test_inject_step_result_compound_flow():
    ctx = ExecutionContext(session_id="test_session")
    ctx.add_structured_result(
        "step_1",
        {"result": "https://example.com", "url": "https://example.com", "text": "Example Domain Content"},
        step_index=0,
    )

    # Notepad type action referencing step_1 text
    action = {
        "skill": "desktop",
        "action": "type_text",
        "target": "Notepad",
        "text": "${step_1.text}",
    }

    resolved = ExecutionLoop._inject_step_result(action, ctx, "step_2")
    assert resolved["text"] == "Example Domain Content"
