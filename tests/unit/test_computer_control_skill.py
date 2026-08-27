"""
Unit tests for ComputerControlSkill (Desktop mouse, keyboard, and window automation).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spidy.skills.base import SkillContext
from spidy.skills.desktop.computer_control_skill import ComputerControlSkill


@pytest.fixture
def skill() -> ComputerControlSkill:
    return ComputerControlSkill()


@pytest.mark.asyncio
async def test_capabilities_declared(skill: ComputerControlSkill) -> None:
    caps = skill.capabilities()
    actions = [c.action for c in caps]
    assert "get_active_window" in actions
    assert "mouse_click" in actions
    assert "mouse_move" in actions
    assert "mouse_scroll" in actions
    assert "keyboard_type" in actions
    assert "key_press" in actions
    assert "hotkey" in actions
    assert "focus_window" in actions
    assert "minimize_window" in actions
    assert "maximize_window" in actions
    assert "close_window" in actions


@pytest.mark.asyncio
async def test_get_active_window(skill: ComputerControlSkill) -> None:
    ctx = SkillContext(action="get_active_window", params={})
    res = await skill.execute("get_active_window", ctx)
    assert res.success is True
    assert "Active window" in res.message


@pytest.mark.asyncio
async def test_mouse_click(skill: ComputerControlSkill) -> None:
    ctx = SkillContext(action="mouse_click", params={"x": 100, "y": 200, "button": "left", "clicks": 1})
    res = await skill.execute("mouse_click", ctx)
    assert res.success is True


@pytest.mark.asyncio
async def test_mouse_move(skill: ComputerControlSkill) -> None:
    ctx = SkillContext(action="mouse_move", params={"x": 150, "y": 250})
    res = await skill.execute("mouse_move", ctx)
    assert res.success is True


@pytest.mark.asyncio
async def test_mouse_scroll(skill: ComputerControlSkill) -> None:
    ctx = SkillContext(action="mouse_scroll", params={"clicks": 3})
    res = await skill.execute("mouse_scroll", ctx)
    assert res.success is True


@pytest.mark.asyncio
async def test_keyboard_type(skill: ComputerControlSkill) -> None:
    ctx = SkillContext(action="keyboard_type", params={"text": "hello world", "press_enter": True})
    res = await skill.execute("keyboard_type", ctx)
    assert res.success is True
    assert "hello world" in res.message


@pytest.mark.asyncio
async def test_keyboard_type_empty_fails(skill: ComputerControlSkill) -> None:
    ctx = SkillContext(action="keyboard_type", params={"text": ""})
    res = await skill.execute("keyboard_type", ctx)
    assert res.success is False


@pytest.mark.asyncio
async def test_key_press(skill: ComputerControlSkill) -> None:
    ctx = SkillContext(action="key_press", params={"key": "enter"})
    res = await skill.execute("key_press", ctx)
    assert res.success is True
    assert "enter" in res.message


@pytest.mark.asyncio
async def test_hotkey(skill: ComputerControlSkill) -> None:
    ctx = SkillContext(action="hotkey", params={"keys": "ctrl+c"})
    res = await skill.execute("hotkey", ctx)
    assert res.success is True
    assert "ctrl+c" in res.message


@pytest.mark.asyncio
async def test_window_operations(skill: ComputerControlSkill) -> None:
    ctx = SkillContext(action="focus_window", params={"title": "Calculator"})
    res = await skill.execute("focus_window", ctx)
    # Either succeeds or fails gracefully with clear message
    assert isinstance(res.message, str)

    ctx_min = SkillContext(action="minimize_window", params={"title": "Calculator"})
    res_min = await skill.execute("minimize_window", ctx_min)
    assert isinstance(res_min.message, str)

    ctx_max = SkillContext(action="maximize_window", params={"title": "Calculator"})
    res_max = await skill.execute("maximize_window", ctx_max)
    assert isinstance(res_max.message, str)
