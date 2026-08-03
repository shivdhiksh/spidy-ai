"""
Integration Tests — Milestone 7 Browser Skills
================================================
Tests the full integration of BrowserSkill with:
- SkillRegistry
- SkillExecutor
- PermissionManager
- EventBus
- register_browser_skills() loader
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.core.event_bus import EventBus
from spidy.skills.registry import SkillRegistry
from spidy.skills.browser import register_browser_skills
from spidy.skills.browser.browser_skill import BrowserSkill
from spidy.browser.types import DownloadResult, PageInfo, TabInfo


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_mock_agent(running: bool = True):
    agent = MagicMock()
    agent.is_running = running
    agent.start = AsyncMock(side_effect=lambda: setattr(agent, "is_running", True))
    agent.stop = AsyncMock()
    agent.open_url = AsyncMock(return_value=PageInfo(title="T", url="https://t.com", tab_id=0))
    agent.close_tab = AsyncMock(return_value=True)
    agent.get_current_page_info = AsyncMock(return_value=PageInfo(title="G", url="https://google.com"))
    agent.get_all_tabs = AsyncMock(return_value=[TabInfo(tab_id=0, title="G", url="https://g.com", is_active=True)])
    agent.navigate_back = AsyncMock(return_value=PageInfo(title="Back", url="https://back.com"))
    agent.navigate_forward = AsyncMock(return_value=PageInfo(title="Fwd", url="https://fwd.com"))
    agent.refresh = AsyncMock(return_value=PageInfo(title="R", url="https://r.com"))
    agent.search_google = AsyncMock(return_value=PageInfo(title="GS", url="https://www.google.com/search?q=x"))
    agent.search_youtube = AsyncMock(return_value=PageInfo(title="YT", url="https://www.youtube.com/results?search_query=x"))
    agent.read_page_text = AsyncMock(return_value="Integration test page content.")
    agent.download_file = AsyncMock(return_value=DownloadResult(
        filename="int.pdf", path="/tmp/int.pdf", size_bytes=512, success=True
    ))
    agent.get_browser_history = AsyncMock(return_value=[])
    return agent


def _make_executor(registry, bus=None):
    from spidy.execution.executor import SkillExecutor
    from spidy.permissions.manager import PermissionManager
    from spidy.config.manager import PermissionsConfig, ExecutorConfig

    perm_config = PermissionsConfig(
        require_confirmation_for=[],
        audit_log_enabled=False,
    )
    perm_mgr = PermissionManager(config=perm_config, bus=bus)
    exec_config = ExecutorConfig(permission_checking_enabled=True, max_retries=0)
    return SkillExecutor(
        registry=registry,
        permission_manager=perm_mgr,
        config=exec_config,
        bus=bus,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. register_browser_skills()
# ──────────────────────────────────────────────────────────────────────────────


class TestRegisterBrowserSkills:
    def test_registers_browser_skill(self):
        registry = SkillRegistry()
        registered = register_browser_skills(registry)
        assert "browser_skill" in registered
        assert len(registry) == 1

    def test_skill_findable_by_name(self):
        registry = SkillRegistry()
        register_browser_skills(registry)
        skill = registry.find_skill_by_name("browser_skill")
        assert skill is not None

    def test_all_browser_actions_registered(self):
        """BrowserSkill currently provides 15 actions (14 original + open_browser_and_search)."""
        registry = SkillRegistry()
        register_browser_skills(registry)
        all_caps = registry.all_capabilities()
        action_names = {c.action for c in all_caps}

        # T0
        assert "get_page_info" in action_names
        assert "list_tabs" in action_names
        assert "read_page" in action_names

        # T1
        assert "open_browser" in action_names
        assert "open_url" in action_names
        assert "open_new_tab" in action_names
        assert "navigate_back" in action_names
        assert "navigate_forward" in action_names
        assert "refresh_page" in action_names
        assert "search_google" in action_names
        assert "search_youtube" in action_names
        assert "open_browser_and_search" in action_names  # compound action added in M6

        # T2
        assert "close_browser" in action_names
        assert "close_tab" in action_names
        assert "download_file" in action_names

        assert len(action_names) >= 15  # future actions will only expand this

    def test_actions_findable_via_registry(self):
        registry = SkillRegistry()
        register_browser_skills(registry)

        skill = registry.find_skill_for_action("search_google")
        assert skill is not None
        assert skill.name == "browser_skill"

        skill = registry.find_skill_for_action("download_file")
        assert skill is not None
        assert skill.name == "browser_skill"

    def test_register_with_bus(self):
        bus = EventBus()
        registry = SkillRegistry()
        registered = register_browser_skills(registry, bus=bus)
        assert "browser_skill" in registered

    def test_disable_browser_skill_via_config(self):
        registry = SkillRegistry()
        config = MagicMock()
        config.browser_skill_enabled = False
        registered = register_browser_skills(registry, config=config)
        assert registered == []
        assert len(registry) == 0

    def test_all_config_fields_forwarded(self):
        registry = SkillRegistry()
        config = MagicMock()
        config.browser_skill_enabled = True
        config.browser_type = "firefox"
        config.browser_headless = True
        config.browser_download_dir = "/custom"
        config.browser_connect_to_existing = False
        config.browser_cdp_endpoint = "http://localhost:9223"
        config.browser_page_load_timeout_ms = 15_000
        config.browser_navigation_timeout_ms = 5_000
        config.browser_read_page_max_chars = 2000
        config.browser_history_limit = 10

        register_browser_skills(registry, config=config)
        skill = registry.find_skill_by_name("browser_skill")
        assert skill is not None
        assert skill._browser_type == "firefox"
        assert skill._headless is True
        assert skill._download_dir == "/custom"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Permission gating via SkillExecutor
# ──────────────────────────────────────────────────────────────────────────────


class TestBrowserSkillPermissions:
    @pytest.mark.asyncio
    async def test_t0_action_auto_approved(self):
        """T0 actions should be executed without any confirmation."""
        registry = SkillRegistry()
        register_browser_skills(registry)
        executor = _make_executor(registry)

        skill = registry.find_skill_by_name("browser_skill")
        agent = _make_mock_agent(running=False)
        skill._agent = agent

        result = await executor.execute(
            action="get_page_info",
            params={},
            session_id="integration_test",
        )
        assert result.success  # "not open" message, but succeeds

    @pytest.mark.asyncio
    async def test_t1_open_url_approved(self):
        """T1 actions should be auto-approved for the same session."""
        registry = SkillRegistry()
        register_browser_skills(registry)
        executor = _make_executor(registry)

        skill = registry.find_skill_by_name("browser_skill")
        agent = _make_mock_agent(running=True)
        skill._agent = agent

        result = await executor.execute(
            action="open_url",
            params={"url": "https://google.com"},
            session_id="integration_test",
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_t2_download_requires_confirmation(self):
        """T2 actions: with no bus and default timeout, T2 times out and is denied."""
        from spidy.config.manager import PermissionsConfig, ExecutorConfig
        from spidy.permissions.manager import PermissionManager
        from spidy.execution.executor import SkillExecutor

        perm_config = PermissionsConfig(
            require_confirmation_for=[],
            audit_log_enabled=False,
            t2_confirmation_timeout_seconds=0.01,  # tiny timeout → auto-deny
        )
        perm_mgr = PermissionManager(config=perm_config, bus=None)
        exec_config = ExecutorConfig(permission_checking_enabled=True, max_retries=0)

        registry = SkillRegistry()
        register_browser_skills(registry)
        executor = SkillExecutor(
            registry=registry,
            permission_manager=perm_mgr,
            config=exec_config,
            bus=None,
        )

        result = await executor.execute(
            action="download_file",
            params={"url": "https://a.com/f.pdf"},
            session_id="integration_test",
        )
        # T2 should be denied (timeout → deny when no bus)
        assert not result.success

    @pytest.mark.asyncio
    async def test_t2_search_google_not_t2(self):
        """search_google is T1, should NOT be blocked."""
        registry = SkillRegistry()
        register_browser_skills(registry)
        executor = _make_executor(registry)

        skill = registry.find_skill_by_name("browser_skill")
        agent = _make_mock_agent(running=True)
        skill._agent = agent

        result = await executor.execute(
            action="search_google",
            params={"query": "integration test"},
            session_id="integration_test",
        )
        assert result.success


# ──────────────────────────────────────────────────────────────────────────────
# 3. Full pipeline with EventBus
# ──────────────────────────────────────────────────────────────────────────────


class TestBrowserSkillFullPipeline:
    @pytest.mark.asyncio
    async def test_search_google_publishes_event_through_executor(self):
        bus = EventBus()
        received = []

        async def on_search(evt):
            received.append(evt)

        bus.subscribe("browser.search_results", on_search)

        registry = SkillRegistry()
        register_browser_skills(registry, bus=bus)
        executor = _make_executor(registry, bus=bus)

        skill = registry.find_skill_by_name("browser_skill")
        agent = _make_mock_agent(running=True)
        skill._agent = agent

        await executor.execute(
            action="search_google",
            params={"query": "full pipeline test"},
            session_id="pipeline_test",
        )
        assert len(received) == 1
        assert received[0].engine == "google"

    @pytest.mark.asyncio
    async def test_list_tabs_returns_structured_data(self):
        registry = SkillRegistry()
        register_browser_skills(registry)
        executor = _make_executor(registry)

        skill = registry.find_skill_by_name("browser_skill")
        agent = _make_mock_agent(running=True)
        skill._agent = agent

        result = await executor.execute(
            action="list_tabs",
            params={},
            session_id="pipeline_test",
        )
        assert result.success
        assert "tabs" in result.data

    @pytest.mark.asyncio
    async def test_navigate_back_and_forward_pipeline(self):
        registry = SkillRegistry()
        register_browser_skills(registry)
        executor = _make_executor(registry)

        skill = registry.find_skill_by_name("browser_skill")
        agent = _make_mock_agent(running=True)
        skill._agent = agent

        back = await executor.execute("navigate_back", {}, session_id="s")
        forward = await executor.execute("navigate_forward", {}, session_id="s")
        assert back.success
        assert forward.success
