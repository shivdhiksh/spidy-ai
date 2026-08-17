"""
Tests for Milestone 6 — AppSkill
==================================
Unit tests for all AppSkill actions:
- detect_running_apps
- launch_app
- bring_app_to_foreground
- close_app
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.skills.base import SkillContext, SkillResult
from spidy.skills.desktop.app_skill import AppSkill, _APP_ALIASES


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ctx(action: str = "", params: dict | None = None, session_id: str = "test") -> SkillContext:
    return SkillContext(action=action, params=params or {}, session_id=session_id)


def _make_skill(bus=None) -> AppSkill:
    return AppSkill(bus=bus)


# Mock psutil process
def _mock_proc(name: str, pid: int, status: str = "running"):
    p = MagicMock()
    p.info = {"name": name, "pid": pid, "status": status}
    p.name.return_value = name
    p.pid = pid
    return p


# ──────────────────────────────────────────────────────────────────────────────
# 1. Capability declarations
# ──────────────────────────────────────────────────────────────────────────────


class TestAppSkillCapabilities:
    def test_all_actions_declared(self):
        skill = _make_skill()
        actions = set(skill.capability_names())
        expected = {
            "detect_running_apps", "launch_app",
            "bring_app_to_foreground", "close_app",
            "minimize_window", "maximize_window",
        }
        assert expected == actions

    def test_detect_is_t0(self):
        skill = _make_skill()
        for cap in skill.capabilities():
            if cap.action == "detect_running_apps":
                assert cap.permission_tier == "T0"

    def test_launch_and_foreground_are_t1(self):
        skill = _make_skill()
        for cap in skill.capabilities():
            if cap.action in ("launch_app", "bring_app_to_foreground"):
                assert cap.permission_tier == "T1", f"{cap.action} should be T1"

    def test_close_is_t2(self):
        skill = _make_skill()
        for cap in skill.capabilities():
            if cap.action == "close_app":
                assert cap.permission_tier == "T2"

    def test_skill_name_and_version(self):
        assert AppSkill.name == "app_skill"
        assert AppSkill.version == "1.0.0"

    def test_all_actions_supported(self):
        skill = _make_skill()
        for action in ("detect_running_apps", "launch_app",
                       "bring_app_to_foreground", "close_app"):
            assert skill.supports(action)


# ──────────────────────────────────────────────────────────────────────────────
# 2. detect_running_apps
# ──────────────────────────────────────────────────────────────────────────────


class TestDetectRunningApps:
    @pytest.mark.asyncio
    async def test_detect_returns_process_list(self):
        mock_procs = [
            _mock_proc("chrome.exe", 1001),
            _mock_proc("notepad.exe", 1002),
            _mock_proc("explorer.exe", 1003),
        ]
        with patch("psutil.process_iter", return_value=mock_procs):
            skill = _make_skill()
            result = await skill.execute("detect_running_apps", _ctx("detect_running_apps"))

        assert result.success
        assert len(result.data["apps"]) == 3
        names = [a["name"] for a in result.data["apps"]]
        assert "chrome.exe" in names

    @pytest.mark.asyncio
    async def test_detect_with_name_filter(self):
        mock_procs = [
            _mock_proc("chrome.exe", 1001),
            _mock_proc("notepad.exe", 1002),
        ]
        with patch("psutil.process_iter", return_value=mock_procs):
            skill = _make_skill()
            ctx = _ctx("detect_running_apps", {"filter": "chrome"})
            result = await skill.execute("detect_running_apps", ctx)

        assert result.success
        assert all("chrome" in a["name"].lower() for a in result.data["apps"])

    @pytest.mark.asyncio
    async def test_detect_empty_result(self):
        with patch("psutil.process_iter", return_value=[]):
            skill = _make_skill()
            result = await skill.execute("detect_running_apps", _ctx("detect_running_apps"))
        assert result.success
        assert result.data["apps"] == []

    @pytest.mark.asyncio
    async def test_detect_publishes_event(self):
        bus = MagicMock()
        bus.publish = AsyncMock()
        skill = _make_skill(bus=bus)
        with patch("psutil.process_iter", return_value=[_mock_proc("test.exe", 100)]):
            await skill.execute("detect_running_apps", _ctx("detect_running_apps"))
        bus.publish.assert_called_once()
        from spidy.skills.desktop.events import RunningAppsResultEvent
        assert isinstance(bus.publish.call_args[0][0], RunningAppsResultEvent)

    @pytest.mark.asyncio
    async def test_detect_handles_psutil_not_installed(self):
        skill = _make_skill()
        with patch.dict("sys.modules", {"psutil": None}):
            # Without psutil, should return empty gracefully
            with patch("spidy.skills.desktop.app_skill.AppSkill._get_running_processes",
                       return_value=[]):
                result = await skill.execute("detect_running_apps", _ctx("detect_running_apps"))
        assert result.success


# ──────────────────────────────────────────────────────────────────────────────
# 3. launch_app
# ──────────────────────────────────────────────────────────────────────────────


class TestLaunchApp:
    @pytest.mark.asyncio
    async def test_launch_by_alias(self):
        mock_proc = MagicMock()
        mock_proc.pid = 9999
        mock_proc.poll.return_value = None  # Still running after 350ms settle
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            skill = _make_skill()
            ctx = _ctx("launch_app", {"name": "notepad"})
            result = await skill.execute("launch_app", ctx)

        assert result.success
        assert result.data["pid"] == 9999
        # Should have resolved alias
        called_cmd = mock_popen.call_args[0][0]
        assert "notepad.exe" in called_cmd

    @pytest.mark.asyncio
    async def test_launch_with_args(self):
        mock_proc = MagicMock()
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None  # Still running
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            skill = _make_skill()
            ctx = _ctx("launch_app", {"name": "notepad", "args": "C:/test.txt"})
            result = await skill.execute("launch_app", ctx)

        assert result.success
        called_cmd = mock_popen.call_args[0][0]
        assert "C:/test.txt" in called_cmd

    @pytest.mark.asyncio
    async def test_launch_app_not_found(self):
        with patch("subprocess.Popen", side_effect=FileNotFoundError("not found")):
            skill = _make_skill()
            ctx = _ctx("launch_app", {"name": "nonexistent_app_xyz"})
            result = await skill.execute("launch_app", ctx)
        assert not result.success

    @pytest.mark.asyncio
    async def test_launch_requires_name(self):
        skill = _make_skill()
        result = await skill.execute("launch_app", _ctx("launch_app", {}))
        assert not result.success
        assert "name" in result.message.lower()

    @pytest.mark.asyncio
    async def test_launch_publishes_event(self):
        bus = MagicMock()
        bus.publish = AsyncMock()
        skill = _make_skill(bus=bus)
        mock_proc = MagicMock()
        mock_proc.pid = 5555
        mock_proc.poll.return_value = None  # Still running
        with patch("subprocess.Popen", return_value=mock_proc):
            ctx = _ctx("launch_app", {"name": "notepad"})
            await skill.execute("launch_app", ctx)
        bus.publish.assert_called_once()
        from spidy.skills.desktop.events import AppLaunchedEvent
        assert isinstance(bus.publish.call_args[0][0], AppLaunchedEvent)

    def test_app_aliases_populated(self):
        assert "notepad" in _APP_ALIASES
        assert "chrome" in _APP_ALIASES
        assert "calculator" in _APP_ALIASES
        assert _APP_ALIASES["notepad"] == "notepad.exe"
        assert _APP_ALIASES["calculator"] == "calc.exe"


# ──────────────────────────────────────────────────────────────────────────────
# 4. bring_app_to_foreground
# ──────────────────────────────────────────────────────────────────────────────


class TestBringToForeground:
    @pytest.mark.asyncio
    async def test_foreground_success(self):
        skill = _make_skill()
        with patch.object(AppSkill, "_find_window_by_name", return_value=(12345, "Notepad")):
            with patch.object(AppSkill, "_set_foreground", return_value=True):
                ctx = _ctx("bring_app_to_foreground", {"name": "notepad"})
                result = await skill.execute("bring_app_to_foreground", ctx)
        assert result.success
        assert "Notepad" in result.message

    @pytest.mark.asyncio
    async def test_foreground_app_not_found(self):
        skill = _make_skill()
        with patch.object(AppSkill, "_find_window_by_name", return_value=(0, "")):
            ctx = _ctx("bring_app_to_foreground", {"name": "nonexistent_xyz"})
            result = await skill.execute("bring_app_to_foreground", ctx)
        assert not result.success
        assert "no running window" in result.message.lower()

    @pytest.mark.asyncio
    async def test_foreground_window_focus_fails(self):
        skill = _make_skill()
        with patch.object(AppSkill, "_find_window_by_name", return_value=(99, "App")):
            with patch.object(AppSkill, "_set_foreground", return_value=False):
                ctx = _ctx("bring_app_to_foreground", {"name": "app"})
                result = await skill.execute("bring_app_to_foreground", ctx)
        assert not result.success

    @pytest.mark.asyncio
    async def test_foreground_requires_name(self):
        skill = _make_skill()
        result = await skill.execute("bring_app_to_foreground", _ctx("bring_app_to_foreground", {}))
        assert not result.success

    @pytest.mark.asyncio
    async def test_foreground_publishes_event(self):
        bus = MagicMock()
        bus.publish = AsyncMock()
        skill = _make_skill(bus=bus)
        with patch.object(AppSkill, "_find_window_by_name", return_value=(42, "Chrome")):
            with patch.object(AppSkill, "_set_foreground", return_value=True):
                ctx = _ctx("bring_app_to_foreground", {"name": "chrome"})
                await skill.execute("bring_app_to_foreground", ctx)
        bus.publish.assert_called_once()
        from spidy.skills.desktop.events import AppForegroundedEvent
        assert isinstance(bus.publish.call_args[0][0], AppForegroundedEvent)


# ──────────────────────────────────────────────────────────────────────────────
# 5. close_app
# ──────────────────────────────────────────────────────────────────────────────


class TestCloseApp:
    def _make_psutil_proc(self, name: str = "notepad.exe", pid: int = 1001):
        proc = MagicMock()
        proc.info = {"name": name, "pid": pid}
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        return proc

    @pytest.mark.asyncio
    async def test_close_graceful(self):
        import psutil

        skill = _make_skill()
        proc_info = [{"name": "notepad.exe", "pid": 1001}]
        mock_proc = MagicMock()
        mock_proc.terminate = MagicMock()

        with patch.object(AppSkill, "_find_processes_by_name", return_value=proc_info):
            with patch("psutil.Process", return_value=mock_proc):
                ctx = _ctx("close_app", {"name": "notepad"})  # force defaults to False
                result = await skill.execute("close_app", ctx)

        assert result.success
        mock_proc.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_forced(self):
        skill = _make_skill()
        proc_info = [{"name": "chrome.exe", "pid": 2002}]
        mock_proc = MagicMock()
        mock_proc.kill = MagicMock()

        with patch.object(AppSkill, "_find_processes_by_name", return_value=proc_info):
            with patch("psutil.Process", return_value=mock_proc):
                ctx = _ctx("close_app", {"name": "chrome", "force": True})
                result = await skill.execute("close_app", ctx)

        assert result.success
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_app_not_found(self):
        skill = _make_skill()
        with patch.object(AppSkill, "_find_processes_by_name", return_value=[]):
            ctx = _ctx("close_app", {"name": "nonexistent_xyz"})
            result = await skill.execute("close_app", ctx)
        assert not result.success
        assert "no running process" in result.message.lower()

    @pytest.mark.asyncio
    async def test_close_requires_name(self):
        skill = _make_skill()
        result = await skill.execute("close_app", _ctx("close_app", {}))
        assert not result.success

    @pytest.mark.asyncio
    async def test_close_publishes_event(self):
        bus = MagicMock()
        bus.publish = AsyncMock()
        skill = _make_skill(bus=bus)
        proc_info = [{"name": "notepad.exe", "pid": 100}]
        mock_proc = MagicMock()
        mock_proc.terminate = MagicMock()

        with patch.object(AppSkill, "_find_processes_by_name", return_value=proc_info):
            with patch("psutil.Process", return_value=mock_proc):
                ctx = _ctx("close_app", {"name": "notepad"})
                await skill.execute("close_app", ctx)

        bus.publish.assert_called_once()
        from spidy.skills.desktop.events import AppClosedEvent
        assert isinstance(bus.publish.call_args[0][0], AppClosedEvent)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Unknown action
# ──────────────────────────────────────────────────────────────────────────────


class TestUnknownAction:
    @pytest.mark.asyncio
    async def test_unknown_action_fails_gracefully(self):
        skill = _make_skill()
        result = await skill.execute("do_something_weird", _ctx("do_something_weird"))
        assert not result.success
        assert "unknown action" in result.message.lower()


# ──────────────────────────────────────────────────────────────────────────────
# 7. Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


class TestAppSkillInternals:
    def test_get_running_processes_filters_by_name(self):
        mock_procs = [
            _mock_proc("chrome.exe", 1),
            _mock_proc("notepad.exe", 2),
            _mock_proc("chrome_helper.exe", 3),
        ]
        with patch("psutil.process_iter", return_value=mock_procs):
            result = AppSkill._get_running_processes("chrome", limit=10)
        assert len(result) == 2
        assert all("chrome" in r["name"].lower() for r in result)

    def test_get_running_processes_respects_limit(self):
        mock_procs = [_mock_proc(f"app{i}.exe", i) for i in range(50)]
        with patch("psutil.process_iter", return_value=mock_procs):
            result = AppSkill._get_running_processes("", limit=5)
        assert len(result) <= 5

    def test_find_processes_by_name_uses_alias(self):
        mock_procs = [
            _mock_proc("notepad.exe", 1),
            _mock_proc("chrome.exe", 2),
        ]
        with patch("psutil.process_iter", return_value=mock_procs):
            result = AppSkill._find_processes_by_name("notepad")
        assert len(result) == 1
        assert result[0]["name"] == "notepad.exe"
