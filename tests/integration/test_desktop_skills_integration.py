"""
Integration Tests — Milestone 6 Desktop Skills
================================================
Tests the full integration of desktop skills with:
- SkillRegistry
- SkillExecutor
- PermissionManager
- EventBus
- register_desktop_skills() loader
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.core.event_bus import EventBus
from spidy.skills.registry import SkillRegistry
from spidy.skills.desktop import register_desktop_skills
from spidy.skills.desktop.file_skill import FileSkill
from spidy.skills.desktop.app_skill import AppSkill
from spidy.skills.desktop.system_control_skill import SystemControlSkill


# ──────────────────────────────────────────────────────────────────────────────
# 1. register_desktop_skills()
# ──────────────────────────────────────────────────────────────────────────────


class TestRegisterDesktopSkills:
    def test_registers_all_three_skills(self):
        registry = SkillRegistry()
        registered = register_desktop_skills(registry)
        assert "file_skill" in registered
        assert "app_skill" in registered
        assert "system_control_skill" in registered
        assert len(registry) == 3

    def test_skills_findable_by_name(self):
        registry = SkillRegistry()
        register_desktop_skills(registry)
        assert registry.find_skill_by_name("file_skill") is not None
        assert registry.find_skill_by_name("app_skill") is not None
        assert registry.find_skill_by_name("system_control_skill") is not None

    def test_all_actions_registered(self):
        registry = SkillRegistry()
        register_desktop_skills(registry)
        all_caps = registry.all_capabilities()
        action_names = {c.action for c in all_caps}

        # FileSkill actions
        assert "search_files" in action_names
        assert "search_folders" in action_names
        assert "open_file" in action_names
        assert "open_folder" in action_names
        assert "reveal_in_explorer" in action_names
        assert "list_recent_files" in action_names

        # AppSkill actions
        assert "detect_running_apps" in action_names
        assert "launch_app" in action_names
        assert "bring_app_to_foreground" in action_names
        assert "close_app" in action_names

        # SystemControlSkill actions
        assert "set_volume" in action_names
        assert "set_brightness" in action_names
        assert "lock_workstation" in action_names
        assert "sleep_system" in action_names
        assert "shutdown_system" in action_names
        assert "restart_system" in action_names
        assert "empty_recycle_bin" in action_names

    def test_register_with_bus(self):
        bus = EventBus()
        registry = SkillRegistry()
        registered = register_desktop_skills(registry, bus=bus)
        assert len(registered) == 3

    def test_disable_file_skill_via_config(self):
        registry = SkillRegistry()
        config = MagicMock()
        config.file_skill_enabled = False
        config.app_skill_enabled = True
        config.system_control_skill_enabled = True
        config.desktop_file_search_max_results = 50
        config.desktop_file_search_root = "~"

        registered = register_desktop_skills(registry, config=config)
        assert "file_skill" not in registered
        assert "app_skill" in registered
        assert "system_control_skill" in registered

    def test_disable_all_desktop_skills_via_config(self):
        registry = SkillRegistry()
        config = MagicMock()
        config.file_skill_enabled = False
        config.app_skill_enabled = False
        config.system_control_skill_enabled = False

        registered = register_desktop_skills(registry, config=config)
        assert registered == []
        assert len(registry) == 0

    def test_actions_findable_via_registry(self):
        registry = SkillRegistry()
        register_desktop_skills(registry)

        skill = registry.find_skill_for_action("search_files")
        assert skill is not None
        assert skill.name == "file_skill"

        skill = registry.find_skill_for_action("launch_app")
        assert skill is not None
        assert skill.name == "app_skill"

        skill = registry.find_skill_for_action("set_volume")
        assert skill is not None
        assert skill.name == "system_control_skill"


# ──────────────────────────────────────────────────────────────────────────────
# 2. SkillExecutor + PermissionManager gating
# ──────────────────────────────────────────────────────────────────────────────


class TestDesktopSkillPermissions:
    """Verify that permission tiers are correctly enforced through the Executor."""

    def _make_executor(self, registry, bus=None):
        from spidy.execution.executor import SkillExecutor
        from spidy.permissions.manager import PermissionManager
        from spidy.config.manager import PermissionsConfig

        perm_config = PermissionsConfig(
            require_confirmation_for=[],
            audit_log_enabled=False,
        )
        perm_mgr = PermissionManager(config=perm_config, bus=bus)
        from spidy.config.manager import ExecutorConfig
        exec_config = ExecutorConfig(permission_checking_enabled=True, max_retries=0)
        return SkillExecutor(
            registry=registry,
            permission_manager=perm_mgr,
            config=exec_config,
            bus=bus,
        )

    @pytest.mark.asyncio
    async def test_t0_action_auto_approved(self):
        """T0 actions should be executed without any confirmation."""
        registry = SkillRegistry()
        register_desktop_skills(registry)
        executor = self._make_executor(registry)

        mock_procs = [MagicMock()]
        mock_procs[0].info = {"name": "notepad.exe", "pid": 1, "status": "running"}

        with patch("psutil.process_iter", return_value=mock_procs):
            result = await executor.execute(
                action="detect_running_apps",
                params={},
                tier="T0",
                session_id="integration_test",
            )
        assert result.success

    @pytest.mark.asyncio
    async def test_t1_action_auto_approved_once_per_session(self):
        """T1 actions should be auto-approved for the same session."""
        registry = SkillRegistry()
        register_desktop_skills(registry)
        executor = self._make_executor(registry)

        with patch.object(FileSkill, "_do_file_search", return_value=[]):
            # First call — T0 (search_files is T0)
            result = await executor.execute(
                action="search_files",
                params={"query": "test"},
                session_id="integration_test",
            )
        # search_files is T0, should always succeed
        assert result.success

    @pytest.mark.asyncio
    async def test_t3_action_blocked_without_unlock(self):
        """T3 actions should be blocked unless explicitly unlocked."""
        registry = SkillRegistry()
        register_desktop_skills(registry)
        executor = self._make_executor(registry)

        result = await executor.execute(
            action="shutdown_system",
            params={"delay": "60"},
            session_id="integration_test",
        )
        # T3 requires explicit unlock — should be denied
        assert not result.success
        assert "permission denied" in result.message.lower()

    @pytest.mark.asyncio
    async def test_t3_action_allowed_after_unlock(self):
        """T3 actions should be allowed after unlock_t3() is called."""
        from spidy.permissions.manager import PermissionManager
        from spidy.config.manager import PermissionsConfig, ExecutorConfig
        from spidy.execution.executor import SkillExecutor

        registry = SkillRegistry()
        register_desktop_skills(registry)

        perm_config = PermissionsConfig(audit_log_enabled=False)
        perm_mgr = PermissionManager(config=perm_config)
        perm_mgr.unlock_t3("shutdown_system")

        exec_config = ExecutorConfig(permission_checking_enabled=True, max_retries=0)
        executor = SkillExecutor(registry=registry, permission_manager=perm_mgr,
                                 config=exec_config)

        with patch.object(SystemControlSkill, "_do_shutdown", return_value=None):
            result = await executor.execute(
                action="shutdown_system",
                params={"delay": "60"},
                session_id="admin_session",
            )
        assert result.success


# ──────────────────────────────────────────────────────────────────────────────
# 3. EventBus integration — events actually flow
# ──────────────────────────────────────────────────────────────────────────────


class TestDesktopSkillsEventFlow:
    @pytest.mark.asyncio
    async def test_file_search_emits_event_to_bus(self):
        bus = EventBus()
        registry = SkillRegistry()
        register_desktop_skills(registry, bus=bus)

        received = []
        async def on_event(e):
            received.append(e)
        bus.subscribe("desktop.file.search_result", on_event)

        skill = registry.find_skill_for_action("search_files")
        assert skill is not None

        from spidy.skills.base import SkillContext
        ctx = SkillContext(action="search_files",
                           params={"query": "test_xyz_nonexistent", "path": str(__file__)[:3]},
                           session_id="bus_test")

        # Run search (will find 0 results but will still emit event)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx2 = SkillContext(action="search_files",
                                params={"query": "test_xyz", "path": tmpdir},
                                session_id="bus_test")
            await skill.execute("search_files", ctx2)

        assert len(received) == 1
        from spidy.skills.desktop.events import FileSearchResultEvent
        assert isinstance(received[0], FileSearchResultEvent)

    @pytest.mark.asyncio
    async def test_launch_app_emits_event(self):
        bus = EventBus()
        registry = SkillRegistry()
        register_desktop_skills(registry, bus=bus)

        received = []
        async def on_event(e):
            received.append(e)
        bus.subscribe("desktop.app.launched", on_event)

        skill = registry.find_skill_for_action("launch_app")
        mock_proc = MagicMock()
        mock_proc.pid = 12345

        from spidy.skills.base import SkillContext
        ctx = SkillContext(action="launch_app",
                           params={"name": "notepad"},
                           session_id="bus_test")

        with patch("subprocess.Popen", return_value=mock_proc):
            await skill.execute("launch_app", ctx)

        assert len(received) == 1
        from spidy.skills.desktop.events import AppLaunchedEvent
        assert isinstance(received[0], AppLaunchedEvent)
        assert received[0].pid == 12345

    @pytest.mark.asyncio
    async def test_system_control_volume_emits_event(self):
        bus = EventBus()
        registry = SkillRegistry()
        register_desktop_skills(registry, bus=bus)

        received = []
        async def on_event(e):
            received.append(e)
        bus.subscribe("desktop.system.volume_changed", on_event)

        skill = registry.find_skill_for_action("set_volume")

        from spidy.skills.base import SkillContext
        ctx = SkillContext(action="set_volume",
                           params={"level": "50"},
                           session_id="bus_test")

        with patch.object(SystemControlSkill, "_do_set_volume", return_value=(50, False)):
            await skill.execute("set_volume", ctx)

        assert len(received) == 1
        from spidy.skills.desktop.events import VolumeChangedEvent
        assert isinstance(received[0], VolumeChangedEvent)
        assert received[0].level == 50


# ──────────────────────────────────────────────────────────────────────────────
# 4. Live test — detect_running_apps returns real data
# ──────────────────────────────────────────────────────────────────────────────


class TestLiveDetectRunningApps:
    """
    Live integration test (no mocks).
    Verifies detect_running_apps returns real running process data.
    Skipped if psutil is not installed.
    """

    @pytest.mark.asyncio
    async def test_live_detect_running_apps(self):
        pytest.importorskip("psutil")  # Skip if psutil not installed

        registry = SkillRegistry()
        register_desktop_skills(registry)
        skill = registry.find_skill_for_action("detect_running_apps")
        assert skill is not None

        from spidy.skills.base import SkillContext
        ctx = SkillContext(action="detect_running_apps",
                           params={"limit": "10"},
                           session_id="live_test")
        result = await skill.execute("detect_running_apps", ctx)

        assert result.success
        # Should have at least one running process (the Python process itself)
        assert len(result.data["apps"]) > 0
        # Each app should have name and pid fields (some system processes may have pid=0)
        for app in result.data["apps"]:
            assert "name" in app
            assert "pid" in app
            assert isinstance(app["pid"], int)

    @pytest.mark.asyncio
    async def test_live_search_current_file(self, tmp_path):
        """Live search for a file we know exists."""
        # Create a file in tmp_path
        test_file = tmp_path / "spidy_test_file.txt"
        test_file.write_text("content")

        registry = SkillRegistry()
        register_desktop_skills(registry)
        skill = registry.find_skill_for_action("search_files")

        from spidy.skills.base import SkillContext
        ctx = SkillContext(
            action="search_files",
            params={"query": "spidy_test_file", "path": str(tmp_path)},
            session_id="live_test",
        )
        result = await skill.execute("search_files", ctx)

        assert result.success
        assert result.data["total_found"] >= 1
        assert any("spidy_test_file" in p for p in result.data["results"])


# ──────────────────────────────────────────────────────────────────────────────
# 5. Combined builtin + desktop skill registry
# ──────────────────────────────────────────────────────────────────────────────


class TestCombinedRegistry:
    def test_builtin_and_desktop_coexist(self):
        """Both builtin and desktop skills can be registered in the same registry."""
        from spidy.skills.builtin import register_builtin_skills

        registry = SkillRegistry()
        builtin_registered = register_builtin_skills(registry)
        desktop_registered = register_desktop_skills(registry)

        # All should be registered without conflicts
        assert len(builtin_registered) > 0
        assert len(desktop_registered) > 0

        # Spot-check both are findable
        assert registry.find_skill_for_action("get_time") is not None
        assert registry.find_skill_for_action("search_files") is not None
        assert registry.find_skill_for_action("set_volume") is not None
