"""
Tests for Milestone 7 — BrowserSkill
========================================
Unit tests for all 14 BrowserSkill capabilities:
- Capability declarations (count, tiers, names)
- T0: get_page_info, list_tabs, read_page
- T1: open_browser, open_url, open_new_tab, navigate_back,
      navigate_forward, refresh_page, search_google, search_youtube
- T2: close_browser, close_tab, download_file
- EventBus integration
- Error handling
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.core.event_bus import EventBus
from spidy.skills.base import SkillContext, SkillResult
from spidy.skills.browser.browser_skill import BrowserSkill
from spidy.browser.types import DownloadResult, HistoryEntry, PageInfo, TabInfo


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ctx(action: str = "", params: dict | None = None, session_id: str = "test") -> SkillContext:
    return SkillContext(action=action, params=params or {}, session_id=session_id)


def _make_skill(bus=None) -> BrowserSkill:
    return BrowserSkill(bus=bus)


def _make_mock_agent(running: bool = True):
    """Build a MagicMock that simulates a running BrowserAgent."""
    agent = MagicMock()
    agent.is_running = running
    agent.start = AsyncMock()
    agent.stop = AsyncMock()
    agent.open_url = AsyncMock(return_value=PageInfo(title="Test", url="https://test.com", tab_id=0, load_time_ms=100))
    agent.close_tab = AsyncMock(return_value=True)
    agent.get_current_page_info = AsyncMock(return_value=PageInfo(title="Google", url="https://google.com"))
    agent.get_all_tabs = AsyncMock(return_value=[
        TabInfo(tab_id=0, title="Google", url="https://google.com", is_active=True)
    ])
    agent.navigate_back = AsyncMock(return_value=PageInfo(title="Back", url="https://back.com"))
    agent.navigate_forward = AsyncMock(return_value=PageInfo(title="Fwd", url="https://fwd.com"))
    agent.refresh = AsyncMock(return_value=PageInfo(title="Same", url="https://same.com"))
    agent.search_google = AsyncMock(return_value=PageInfo(
        title="Google Search", url="https://www.google.com/search?q=test"
    ))
    agent.search_youtube = AsyncMock(return_value=PageInfo(
        title="YouTube Search", url="https://www.youtube.com/results?search_query=test"
    ))
    agent.read_page_text = AsyncMock(return_value="Hello world page content")
    agent.download_file = AsyncMock(return_value=DownloadResult(
        filename="file.pdf", path="/tmp/file.pdf", size_bytes=2048, success=True
    ))
    agent.get_browser_history = AsyncMock(return_value=[
        HistoryEntry(title="Example", url="https://example.com", visit_time="2026-07-11T10:00:00")
    ])
    return agent


# ──────────────────────────────────────────────────────────────────────────────
# 1. Capability declarations
# ──────────────────────────────────────────────────────────────────────────────


class TestBrowserSkillCapabilities:
    def test_fourteen_capabilities_declared(self):
        skill = _make_skill()
        assert len(skill.capabilities()) == 14

    def test_all_expected_actions_present(self):
        skill = _make_skill()
        actions = set(skill.capability_names())
        expected = {
            "get_page_info", "list_tabs", "read_page",
            "open_browser", "open_url", "open_new_tab",
            "navigate_back", "navigate_forward", "refresh_page",
            "search_google", "search_youtube",
            "close_browser", "close_tab", "download_file",
        }
        assert expected == actions

    def test_t0_actions_are_read_only(self):
        skill = _make_skill()
        t0_expected = {"get_page_info", "list_tabs", "read_page"}
        t0_actual = {c.action for c in skill.capabilities() if c.permission_tier == "T0"}
        assert t0_expected == t0_actual

    def test_t1_actions(self):
        skill = _make_skill()
        t1_expected = {
            "open_browser", "open_url", "open_new_tab",
            "navigate_back", "navigate_forward", "refresh_page",
            "search_google", "search_youtube",
        }
        t1_actual = {c.action for c in skill.capabilities() if c.permission_tier == "T1"}
        assert t1_expected == t1_actual

    def test_t2_actions(self):
        skill = _make_skill()
        t2_expected = {"close_browser", "close_tab", "download_file"}
        t2_actual = {c.action for c in skill.capabilities() if c.permission_tier == "T2"}
        assert t2_expected == t2_actual

    def test_no_t3_actions(self):
        skill = _make_skill()
        t3 = [c for c in skill.capabilities() if c.permission_tier == "T3"]
        assert t3 == []

    def test_name_and_version(self):
        assert BrowserSkill.name == "browser_skill"
        assert BrowserSkill.version == "1.0.0"

    def test_all_actions_supported(self):
        skill = _make_skill()
        for action in skill.capability_names():
            assert skill.supports(action)

    def test_unknown_action_not_supported(self):
        skill = _make_skill()
        assert not skill.supports("fly_to_moon")

    def test_all_capabilities_have_descriptions(self):
        skill = _make_skill()
        for cap in skill.capabilities():
            assert cap.description, f"'{cap.action}' has no description"

    def test_all_capabilities_have_examples(self):
        skill = _make_skill()
        for cap in skill.capabilities():
            assert cap.examples, f"'{cap.action}' has no examples"


# ──────────────────────────────────────────────────────────────────────────────
# 2. T0 — Read-only actions
# ──────────────────────────────────────────────────────────────────────────────


class TestGetPageInfo:
    @pytest.mark.asyncio
    async def test_browser_not_running_returns_ok_with_message(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=False)
        skill._agent = agent
        result = await skill.execute("get_page_info", _ctx())
        assert result.success
        assert "not open" in result.message.lower()

    @pytest.mark.asyncio
    async def test_browser_running_returns_page_info(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("get_page_info", _ctx())
        assert result.success
        assert "Google" in result.message or "google" in result.message
        agent.get_current_page_info.assert_called_once()


class TestListTabs:
    @pytest.mark.asyncio
    async def test_browser_not_running(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=False)
        skill._agent = agent
        result = await skill.execute("list_tabs", _ctx())
        assert result.success
        assert "not open" in result.message.lower()

    @pytest.mark.asyncio
    async def test_returns_tabs(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("list_tabs", _ctx())
        assert result.success
        assert "tabs" in result.data
        assert len(result.data["tabs"]) == 1

    @pytest.mark.asyncio
    async def test_empty_tabs_returns_ok(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        agent.get_all_tabs = AsyncMock(return_value=[])
        skill._agent = agent
        result = await skill.execute("list_tabs", _ctx())
        assert result.success
        assert "No open tabs" in result.message


class TestReadPage:
    @pytest.mark.asyncio
    async def test_browser_not_running_returns_fail(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=False)
        skill._agent = agent
        result = await skill.execute("read_page", _ctx())
        assert not result.success

    @pytest.mark.asyncio
    async def test_returns_text(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("read_page", _ctx())
        assert result.success
        assert "Hello world" in result.message
        assert "text" in result.data

    @pytest.mark.asyncio
    async def test_respects_max_chars_param(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        agent.read_page_text = AsyncMock(return_value="Short text")
        skill._agent = agent
        result = await skill.execute("read_page", _ctx(params={"max_chars": 100}))
        assert result.success
        agent.read_page_text.assert_called_with(max_chars=100)


# ──────────────────────────────────────────────────────────────────────────────
# 3. T1 — Reversible actions
# ──────────────────────────────────────────────────────────────────────────────


class TestOpenBrowser:
    @pytest.mark.asyncio
    async def test_launches_browser_when_not_running(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=False)
        skill._agent = agent
        # After start(), is_running becomes True
        async def fake_start():
            agent.is_running = True
        agent.start.side_effect = fake_start
        result = await skill.execute("open_browser", _ctx())
        assert result.success
        agent.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_already_running_returns_ok(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("open_browser", _ctx())
        assert result.success
        assert "already" in result.message.lower() or result.success

    @pytest.mark.asyncio
    async def test_open_browser_with_url(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("open_browser", _ctx(params={"url": "https://google.com"}))
        assert result.success
        agent.open_url.assert_called_once()


class TestOpenUrl:
    @pytest.mark.asyncio
    async def test_missing_url_returns_fail(self):
        skill = _make_skill()
        result = await skill.execute("open_url", _ctx(params={}))
        assert not result.success

    @pytest.mark.asyncio
    async def test_opens_url(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("open_url", _ctx(params={"url": "https://github.com"}))
        assert result.success
        agent.open_url.assert_called_once_with("https://github.com", new_tab=False)

    @pytest.mark.asyncio
    async def test_page_info_in_data(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("open_url", _ctx(params={"url": "https://test.com"}))
        assert result.data is not None
        assert "url" in result.data


class TestOpenNewTab:
    @pytest.mark.asyncio
    async def test_missing_url_returns_fail(self):
        skill = _make_skill()
        result = await skill.execute("open_new_tab", _ctx(params={}))
        assert not result.success

    @pytest.mark.asyncio
    async def test_opens_in_new_tab(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("open_new_tab", _ctx(params={"url": "https://github.com"}))
        assert result.success
        agent.open_url.assert_called_once_with("https://github.com", new_tab=True)


class TestNavigateBack:
    @pytest.mark.asyncio
    async def test_browser_not_running_returns_fail(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=False)
        skill._agent = agent
        result = await skill.execute("navigate_back", _ctx())
        assert not result.success

    @pytest.mark.asyncio
    async def test_calls_navigate_back(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("navigate_back", _ctx())
        assert result.success
        agent.navigate_back.assert_called_once()


class TestNavigateForward:
    @pytest.mark.asyncio
    async def test_calls_navigate_forward(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("navigate_forward", _ctx())
        assert result.success
        agent.navigate_forward.assert_called_once()


class TestRefreshPage:
    @pytest.mark.asyncio
    async def test_calls_refresh(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("refresh_page", _ctx())
        assert result.success
        agent.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_browser_not_running_returns_fail(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=False)
        skill._agent = agent
        result = await skill.execute("refresh_page", _ctx())
        assert not result.success


class TestSearchGoogle:
    @pytest.mark.asyncio
    async def test_missing_query_returns_fail(self):
        skill = _make_skill()
        result = await skill.execute("search_google", _ctx(params={}))
        assert not result.success

    @pytest.mark.asyncio
    async def test_searches_google(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("search_google", _ctx(params={"query": "python"}))
        assert result.success
        agent.search_google.assert_called_once_with("python", new_tab=False)

    @pytest.mark.asyncio
    async def test_search_google_data_contains_engine(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("search_google", _ctx(params={"query": "cats"}))
        assert result.data["engine"] == "google"
        assert result.data["query"] == "cats"

    @pytest.mark.asyncio
    async def test_search_google_new_tab(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        await skill.execute("search_google", _ctx(params={"query": "x", "new_tab": True}))
        agent.search_google.assert_called_once_with("x", new_tab=True)


class TestSearchYouTube:
    @pytest.mark.asyncio
    async def test_missing_query_returns_fail(self):
        skill = _make_skill()
        result = await skill.execute("search_youtube", _ctx(params={}))
        assert not result.success

    @pytest.mark.asyncio
    async def test_searches_youtube(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("search_youtube", _ctx(params={"query": "lofi"}))
        assert result.success
        agent.search_youtube.assert_called_once_with("lofi", new_tab=False)

    @pytest.mark.asyncio
    async def test_search_youtube_data_contains_engine(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("search_youtube", _ctx(params={"query": "jazz"}))
        assert result.data["engine"] == "youtube"


# ──────────────────────────────────────────────────────────────────────────────
# 4. T2 — Disruptive actions
# ──────────────────────────────────────────────────────────────────────────────


class TestCloseBrowser:
    @pytest.mark.asyncio
    async def test_not_running_returns_ok(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=False)
        skill._agent = agent
        result = await skill.execute("close_browser", _ctx())
        assert result.success

    @pytest.mark.asyncio
    async def test_running_stops_browser(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("close_browser", _ctx())
        assert result.success
        agent.stop.assert_called_once()
        assert result.data["running"] is False


class TestCloseTab:
    @pytest.mark.asyncio
    async def test_missing_tab_id_returns_fail(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("close_tab", _ctx(params={}))
        assert not result.success

    @pytest.mark.asyncio
    async def test_closes_tab(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("close_tab", _ctx(params={"tab_id": 1}))
        assert result.success
        agent.close_tab.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_invalid_tab_returns_fail(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        agent.close_tab = AsyncMock(return_value=False)
        skill._agent = agent
        result = await skill.execute("close_tab", _ctx(params={"tab_id": 99}))
        assert not result.success

    @pytest.mark.asyncio
    async def test_browser_not_running_returns_fail(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=False)
        skill._agent = agent
        result = await skill.execute("close_tab", _ctx(params={"tab_id": 0}))
        assert not result.success


class TestDownloadFile:
    @pytest.mark.asyncio
    async def test_missing_url_returns_fail(self):
        skill = _make_skill()
        result = await skill.execute("download_file", _ctx(params={}))
        assert not result.success

    @pytest.mark.asyncio
    async def test_successful_download(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        result = await skill.execute("download_file", _ctx(params={"url": "https://a.com/f.pdf"}))
        assert result.success
        assert "file.pdf" in result.message
        assert result.data["success"] is True

    @pytest.mark.asyncio
    async def test_failed_download_returns_fail(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        agent.download_file = AsyncMock(return_value=DownloadResult(
            success=False, error="404 Not Found"
        ))
        skill._agent = agent
        result = await skill.execute("download_file", _ctx(params={"url": "https://a.com/bad"}))
        assert not result.success
        assert "404" in result.message

    @pytest.mark.asyncio
    async def test_respects_dest_dir_param(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        await skill.execute("download_file", _ctx(params={
            "url": "https://a.com/f.pdf",
            "dest_dir": "/custom/dir",
        }))
        agent.download_file.assert_called_once_with(
            "https://a.com/f.pdf", dest_dir="/custom/dir"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Unknown action
# ──────────────────────────────────────────────────────────────────────────────


class TestUnknownAction:
    @pytest.mark.asyncio
    async def test_unknown_action_returns_fail(self):
        skill = _make_skill()
        result = await skill.execute("fly_to_moon", _ctx())
        assert not result.success
        assert "unknown action" in result.message.lower()


# ──────────────────────────────────────────────────────────────────────────────
# 6. EventBus integration
# ──────────────────────────────────────────────────────────────────────────────


class TestBrowserSkillEventBus:
    @pytest.mark.asyncio
    async def test_page_navigated_event_published(self):
        bus = EventBus()
        received = []

        async def on_nav(evt):
            received.append(evt)

        bus.subscribe("browser.page_navigated", on_nav)
        skill = _make_skill(bus=bus)
        agent = _make_mock_agent(running=True)
        skill._agent = agent

        await skill.execute("open_url", _ctx(params={"url": "https://google.com"}))
        assert len(received) == 1
        assert received[0].url == "https://test.com"  # from mock PageInfo

    @pytest.mark.asyncio
    async def test_search_results_event_published(self):
        bus = EventBus()
        received = []

        async def on_search(evt):
            received.append(evt)

        bus.subscribe("browser.search_results", on_search)
        skill = _make_skill(bus=bus)
        agent = _make_mock_agent(running=True)
        skill._agent = agent

        await skill.execute("search_google", _ctx(params={"query": "test"}))
        assert len(received) == 1
        assert received[0].engine == "google"

    @pytest.mark.asyncio
    async def test_download_events_published(self):
        bus = EventBus()
        started = []
        completed = []

        async def on_started(evt):
            started.append(evt)

        async def on_completed(evt):
            completed.append(evt)

        bus.subscribe("browser.download_started", on_started)
        bus.subscribe("browser.download_completed", on_completed)
        skill = _make_skill(bus=bus)
        agent = _make_mock_agent(running=True)
        skill._agent = agent

        await skill.execute("download_file", _ctx(params={"url": "https://a.com/f.pdf"}))
        assert len(started) == 1
        assert len(completed) == 1
        assert completed[0].success is True

    @pytest.mark.asyncio
    async def test_nav_back_event_published(self):
        bus = EventBus()
        received = []

        async def handler(evt):
            received.append(evt)

        bus.subscribe("browser.nav_back", handler)
        skill = _make_skill(bus=bus)
        agent = _make_mock_agent(running=True)
        skill._agent = agent

        await skill.execute("navigate_back", _ctx())
        assert len(received) == 1


# ──────────────────────────────────────────────────────────────────────────────
# 7. On-unload lifecycle
# ──────────────────────────────────────────────────────────────────────────────


class TestOnUnload:
    @pytest.mark.asyncio
    async def test_on_unload_stops_running_agent(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=True)
        skill._agent = agent
        await skill.on_unload()
        agent.stop.assert_called_once()
        assert skill._agent is None

    @pytest.mark.asyncio
    async def test_on_unload_no_agent_is_noop(self):
        skill = _make_skill()
        assert skill._agent is None
        await skill.on_unload()  # Should not raise

    @pytest.mark.asyncio
    async def test_on_unload_not_running_skips_stop(self):
        skill = _make_skill()
        agent = _make_mock_agent(running=False)
        skill._agent = agent
        await skill.on_unload()
        agent.stop.assert_not_called()
