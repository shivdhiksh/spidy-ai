"""
Tests for Generic Computer Actions and DesktopStateLock (Phase C)
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from spidy.desktop.controller import DesktopComputerController, DesktopStateLock
from spidy.skills.desktop.computer_control_skill import ComputerControlSkill
from spidy.skills.base import SkillContext


@pytest.mark.asyncio
async def test_desktop_state_lock():
    async with DesktopStateLock.acquire():
        assert DesktopStateLock.is_locked() is True
    assert DesktopStateLock.is_locked() is False


@pytest.mark.asyncio
async def test_computer_controller_window_and_typing_mocked():
    ctrl = DesktopComputerController()
    with patch.object(ctrl, "find_window", return_value={"hwnd": 1234, "title": "Notepad"}):
        with patch.object(ctrl, "is_window_focused", return_value=True):
            with patch.object(ctrl, "type_text", return_value=True):
                ok = ctrl.safe_type_text("hello", target_window="Notepad")
                assert ok is True


@pytest.mark.asyncio
async def test_computer_control_skill_new_actions():
    skill = ComputerControlSkill()
    caps = {cap.action for cap in skill.capabilities()}
    assert "get_active_window" in caps
    assert "keyboard_type" in caps
    assert "mouse_click" in caps

    # Test select all
    with patch.object(skill._controller, "select_all", return_value=True):
        ctx = SkillContext(action="select_all", params={}, session_id="test")
        res = await skill.execute("select_all", ctx)
        assert res.success is True

    # Test undo
    with patch.object(skill._controller, "undo", return_value=True):
        ctx = SkillContext(action="undo", params={}, session_id="test")
        res = await skill.execute("undo", ctx)
        assert res.success is True

    # Test drag_and_drop
    with patch.object(skill._controller, "drag_and_drop", return_value=True):
        ctx = SkillContext(action="drag_and_drop", params={"x1": 10, "y1": 20, "x2": 50, "y2": 60}, session_id="test")
        res = await skill.execute("drag_and_drop", ctx)
        assert res.success is True
