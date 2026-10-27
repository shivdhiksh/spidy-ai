"""
Unit tests for Mark XXXIX OR Structured Tool Dispatch, ToolRegistry,
ToolExecutor, Context Chaining, and Goal Deduplication.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from spidy.brain.context_resolver import ContextResolver
from spidy.brain.planner import Planner
from spidy.brain.tool_executor import ToolExecutor
from spidy.brain.types import (
    ConversationTurn,
    Decision,
    DecisionMode,
    Entity,
    Intent,
    TurnRole,
)
from spidy.core.event_bus import EventBus
from spidy.llm.client import BaseLLMClient, LLMMessage, LLMResponse
from spidy.skills.base import BaseSkill, SkillCapability, SkillContext, SkillResult
from spidy.skills.desktop.file_skill import FileSkill
from spidy.skills.registry import SkillRegistry
from spidy.skills.tool_registry import GLOBAL_TOOL_REGISTRY, ToolDefinition, ToolRegistry


# ─── Mock LLM Client for Tool Testing ─────────────────────────────────────────

class MockToolLoopLLM(BaseLLMClient):
    """Simulates an LLM that outputs a tool call on turn 1, then a final response on turn 2."""
    def __init__(self, tool_call_json: str, final_answer: str) -> None:
        self.tool_call_json = tool_call_json
        self.final_answer = final_answer
        self.calls: list[list[LLMMessage]] = []

    async def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            return LLMResponse(text=self.tool_call_json, success=True)
        return LLMResponse(text=self.final_answer, success=True)

    async def stream(self, messages: list[LLMMessage], **kwargs):
        yield self.final_answer

    async def close(self) -> None:
        pass


# ─── Test 1: Tool Registry & Schema Export ─────────────────────────────────────

def test_tool_registry_schemas():
    """Verify tool registry exports valid OpenAI function schemas and maps aliases."""
    tools = GLOBAL_TOOL_REGISTRY.get_all_tools()
    assert len(tools) >= 15

    search_tool = GLOBAL_TOOL_REGISTRY.get_tool("file.search")
    assert search_tool is not None
    assert search_tool.skill_action == "search_files"

    # Test alias lookup
    assert GLOBAL_TOOL_REGISTRY.get_tool("search_files") == search_tool
    assert GLOBAL_TOOL_REGISTRY.get_tool("find_file") == search_tool

    # Test OpenAI schema structure
    schema = search_tool.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "file.search"
    assert "query" in schema["function"]["parameters"]["properties"]
    assert "query" in schema["function"]["parameters"]["required"]


# ─── Test 2: Parameter Validation & Type Coercion ──────────────────────────────

def test_tool_parameter_validation():
    """Verify tool parameters are validated and coerced correctly."""
    tool_def = GLOBAL_TOOL_REGISTRY.get_tool("file.search")
    assert tool_def is not None

    # Valid args
    coerced, errors = GLOBAL_TOOL_REGISTRY.validate_parameters(tool_def, {"query": "resume", "max_results": "5"})
    assert not errors
    assert coerced["query"] == "resume"
    assert coerced["max_results"] == 5

    # Missing required arg
    coerced, errors = GLOBAL_TOOL_REGISTRY.validate_parameters(tool_def, {})
    assert len(errors) > 0
    assert "Missing required parameter 'query'" in errors[0]


# ─── Test 3: Tool Execution & Verification ────────────────────────────────────

@pytest.mark.asyncio
async def test_tool_executor_runs_real_action(tmp_path):
    """Verify ToolExecutor runs registered skill and returns verified SkillResult."""
    registry = SkillRegistry()
    file_skill = FileSkill(search_root=str(tmp_path))
    registry.register(file_skill)

    executor = ToolExecutor(skill_registry=registry)

    # 1. Execute file.create
    res = await executor.execute_tool(
        tool_name="file.create",
        args={"path": "test_output.txt", "content": "Hello Spidy from ToolExecutor"},
    )
    assert res.success is True
    target_file = tmp_path / "test_output.txt"
    assert target_file.exists()
    assert target_file.read_text(encoding="utf-8") == "Hello Spidy from ToolExecutor"

    # 2. Execute file.read
    read_res = await executor.execute_tool(
        tool_name="file.read",
        args={"path": str(target_file)},
    )
    assert read_res.success is True
    assert "Hello Spidy from ToolExecutor" in read_res.data["content"]


# ─── Test 4: Mark-Style Model-Tool Loop ───────────────────────────────────────

@pytest.mark.asyncio
async def test_mark_tool_loop_executes_action_and_produces_natural_response(tmp_path):
    """
    Test Mark XXXIX OR flow:
    LLM outputs tool call JSON -> ToolExecutor executes tool on disk ->
    Tool result fed to LLM -> LLM returns final natural language answer.
    """
    registry = SkillRegistry()
    file_skill = FileSkill(search_root=str(tmp_path))
    registry.register(file_skill)

    # Create target file for search
    resume_file = tmp_path / "Shiva_Resume.pdf"
    resume_file.write_bytes(b"%PDF-1.4 dummy content")

    executor = ToolExecutor(skill_registry=registry)

    tool_call_json = '{\n  "tool": "file.search",\n  "args": {"query": "resume"}\n}'
    final_answer = "I found your resume at Shiva_Resume.pdf."

    mock_llm = MockToolLoopLLM(tool_call_json=tool_call_json, final_answer=final_answer)

    messages = [LLMMessage(role="user", content="Find my resume")]
    final_text, executed_tools = await executor.run_tool_loop(mock_llm, messages)

    assert len(executed_tools) == 1
    assert executed_tools[0]["tool"] == "file.search"
    assert executed_tools[0]["success"] is True
    assert final_text == "I found your resume at Shiva_Resume.pdf."

    # Verify no raw JSON in final response
    assert not final_text.startswith("{")


# ─── Test 5: Fallback Parser Rejects Unknown Tools ────────────────────────────

@pytest.mark.asyncio
async def test_tool_executor_rejects_unknown_tool():
    """Verify ToolExecutor safely rejects unregistered tools."""
    registry = SkillRegistry()
    executor = ToolExecutor(skill_registry=registry)

    res = await executor.execute_tool(
        tool_name="unregistered_malicious_tool",
        args={"foo": "bar"},
    )
    assert res.success is False
    assert "Unknown tool" in res.message


# ─── Test 6: Goal Deduplication ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_planner_and_decomposer_deduplicate_idempotent_steps():
    """Verify 'Open Calculator. Open Calculator.' collapses into 1 step."""
    from spidy.agent.task_decomposer import TaskDecomposer

    decomposer = TaskDecomposer()
    tasks = await decomposer.decompose("Open Calculator. Open Calculator.", goal_id="g_dedup")

    # Must collapse to 1 task
    assert len(tasks) == 1
    assert tasks[0].action["action"] == "open_app"
    assert tasks[0].action["target"] == "Calculator"


# ─── Test 7: Multi-Turn Context Resolution Chaining ───────────────────────────

def test_context_resolver_chains_assistant_results():
    """
    Test turn context resolution:
    Turn 1 (USER): Find my resume
    Turn 1 (ASSISTANT): Found 'Shiva_Resume.pdf' at C:\\Users\\Shiva\\Desktop\\Shiva_Resume.pdf
    Turn 2 (USER): Open it
    -> Resolves to 'Open Shiva_Resume.pdf' or full path.
    """
    resolver = ContextResolver()

    turn1_user = ConversationTurn(
        role=TurnRole.USER,
        text="Find my resume",
        intent=Intent(action="search_files", entities=(Entity("query", "resume"),)),
    )
    turn1_assistant = ConversationTurn(
        role=TurnRole.ASSISTANT,
        text="Found 'Shiva_Resume.pdf' at C:\\Users\\Shiva\\Desktop\\Shiva_Resume.pdf",
    )

    history = [turn1_user, turn1_assistant]

    res = resolver.resolve("Open it", history)
    assert res.was_resolved is True
    assert "Shiva_Resume.pdf" in res.utterance or "C:\\Users\\Shiva\\Desktop\\Shiva_Resume.pdf" in res.utterance
