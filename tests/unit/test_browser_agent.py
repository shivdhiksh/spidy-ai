"""
Tests for Milestone 7 — BrowserAgent
========================================
Unit tests for BrowserAgent lifecycle, action dispatch, and error handling.
The PlaywrightBackend is replaced with a FakeBackend (implements BrowserBackend)
so no real browser is started during tests.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spidy.browser.agent import BrowserAgent
from spidy.browser.backends.base import BrowserBackend
from spidy.browser.types import DownloadResult, HistoryEntry, PageInfo, TabInfo


# ──────────────────────────────────────────────────────────────────────────────
# Fake backend
# ──────────────────────────────────────────────────────────────────────────────


class FakeBackend(BrowserBackend):
    """In-memory backend for testing. Never touches Playwright."""

    def __init__(self) -> None:
        self._running = False
        self._pages: list[dict] = [
            {"title": "New Tab", "url": "about:blank", "tab_id": 0}
        ]
        self._active_idx = 0

    async def start(self, **kwargs) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def open_url(self, url: str, new_tab: bool = False) -> PageInfo:
        if new_tab:
            self._pages.append({"title": "New Tab", "url": url, "tab_id": len(self._pages)})
            self._active_idx = len(self._pages) - 1
        else:
            self._pages[self._active_idx]["url"] = url
            self._pages[self._active_idx]["title"] = f"Page: {url}"
        p = self._pages[self._active_idx]
        return PageInfo(title=p["title"], url=p["url"], tab_id=p["tab_id"], load_time_ms=10)

    async def close_tab(self, tab_id: int) -> bool:
        if tab_id >= len(self._pages):
            return False
        self._pages.pop(tab_id)
        self._active_idx = max(0, min(self._active_idx, len(self._pages) - 1))
        return True

    async def get_current_page_info(self) -> PageInfo:
        if not self._pages:
            return PageInfo()
        p = self._pages[self._active_idx]
        return PageInfo(title=p["title"], url=p["url"], tab_id=self._active_idx)

    async def get_all_tabs(self) -> list[TabInfo]:
        return [
            TabInfo(tab_id=i, title=p["title"], url=p["url"],
                    is_active=(i == self._active_idx))
            for i, p in enumerate(self._pages)
        ]

    async def navigate_back(self) -> PageInfo:
        self._pages[self._active_idx]["url"] = "https://back.example.com"
        self._pages[self._active_idx]["title"] = "Back Page"
        return PageInfo(title="Back Page", url="https://back.example.com", tab_id=self._active_idx)

    async def navigate_forward(self) -> PageInfo:
        self._pages[self._active_idx]["url"] = "https://forward.example.com"
        self._pages[self._active_idx]["title"] = "Forward Page"
        return PageInfo(title="Forward Page", url="https://forward.example.com",
                        tab_id=self._active_idx)

    async def refresh(self) -> PageInfo:
        p = self._pages[self._active_idx]
        return PageInfo(title=p["title"], url=p["url"], tab_id=self._active_idx)

    async def read_page_text(self, max_chars: int = 5000) -> str:
        return "Fake page content from FakeBackend."

    async def download_file(self, url: str, dest_dir: str) -> DownloadResult:
        return DownloadResult(
            filename="fake_file.zip",
            path=f"{dest_dir}/fake_file.zip",
            size_bytes=1024,
            success=True,
        )

    async def get_browser_history(self, limit: int = 20) -> list[HistoryEntry]:
        return [
            HistoryEntry(title="Example", url="https://example.com",
                         visit_time="2026-07-11T10:00:00")
        ]


def _make_agent(**kwargs) -> BrowserAgent:
    backend = FakeBackend()
    return BrowserAgent(backend=backend, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Lifecycle
# ──────────────────────────────────────────────────────────────────────────────


class TestBrowserAgentLifecycle:
    @pytest.mark.asyncio
    async def test_initial_state_not_running(self):
        agent = _make_agent()
        assert agent.is_running is False

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        agent = _make_agent()
        await agent.start()
        assert agent.is_running is True

    @pytest.mark.asyncio
    async def test_stop_clears_running(self):
        agent = _make_agent()
        await agent.start()
        await agent.stop()
        assert agent.is_running is False

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        agent = _make_agent()
        await agent.start()
        await agent.start()  # Should not raise
        assert agent.is_running is True

    @pytest.mark.asyncio
    async def test_stop_when_not_running_is_noop(self):
        agent = _make_agent()
        await agent.stop()  # Should not raise
        assert agent.is_running is False


# ──────────────────────────────────────────────────────────────────────────────
# 2. Auto-start
# ──────────────────────────────────────────────────────────────────────────────


class TestBrowserAgentAutoStart:
    @pytest.mark.asyncio
    async def test_open_url_auto_starts(self):
        agent = _make_agent()
        assert not agent.is_running
        info = await agent.open_url("https://google.com")
        assert agent.is_running
        assert "google" in info.url

    @pytest.mark.asyncio
    async def test_get_current_page_info_auto_starts(self):
        agent = _make_agent()
        info = await agent.get_current_page_info()
        assert agent.is_running
        assert isinstance(info, PageInfo)

    @pytest.mark.asyncio
    async def test_get_all_tabs_auto_starts(self):
        agent = _make_agent()
        tabs = await agent.get_all_tabs()
        assert agent.is_running
        assert isinstance(tabs, list)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Navigation
# ──────────────────────────────────────────────────────────────────────────────


class TestBrowserAgentNavigation:
    @pytest.mark.asyncio
    async def test_open_url_returns_page_info(self):
        agent = _make_agent()
        info = await agent.open_url("https://python.org")
        assert isinstance(info, PageInfo)
        assert "python" in info.url

    @pytest.mark.asyncio
    async def test_open_url_new_tab(self):
        agent = _make_agent()
        await agent.start()
        info = await agent.open_url("https://github.com", new_tab=True)
        assert isinstance(info, PageInfo)
        tabs = await agent.get_all_tabs()
        assert len(tabs) == 2

    @pytest.mark.asyncio
    async def test_close_tab_returns_true_on_success(self):
        agent = _make_agent()
        await agent.start()
        await agent.open_url("https://tab1.com", new_tab=True)
        result = await agent.close_tab(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_close_tab_returns_false_for_invalid_id(self):
        agent = _make_agent()
        await agent.start()
        result = await agent.close_tab(99)
        assert result is False

    @pytest.mark.asyncio
    async def test_navigate_back(self):
        agent = _make_agent()
        info = await agent.navigate_back()
        assert isinstance(info, PageInfo)
        assert info.url  # non-empty

    @pytest.mark.asyncio
    async def test_navigate_forward(self):
        agent = _make_agent()
        info = await agent.navigate_forward()
        assert isinstance(info, PageInfo)

    @pytest.mark.asyncio
    async def test_refresh(self):
        agent = _make_agent()
        await agent.open_url("https://refresh.com")
        info = await agent.refresh()
        assert isinstance(info, PageInfo)

    @pytest.mark.asyncio
    async def test_get_all_tabs_returns_list(self):
        agent = _make_agent()
        await agent.start()
        tabs = await agent.get_all_tabs()
        assert isinstance(tabs, list)
        assert all(isinstance(t, TabInfo) for t in tabs)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Search
# ──────────────────────────────────────────────────────────────────────────────


class TestBrowserAgentSearch:
    @pytest.mark.asyncio
    async def test_search_google_navigates_to_google_url(self):
        agent = _make_agent()
        info = await agent.search_google("python tutorials")
        assert "google.com" in info.url
        assert "python+tutorials" in info.url or "python%20tutorials" in info.url or "python" in info.url

    @pytest.mark.asyncio
    async def test_search_youtube_navigates_to_youtube_url(self):
        agent = _make_agent()
        info = await agent.search_youtube("lofi music")
        assert "youtube.com" in info.url
        assert "lofi" in info.url or "lofi+music" in info.url

    @pytest.mark.asyncio
    async def test_search_google_new_tab(self):
        agent = _make_agent()
        await agent.start()
        info = await agent.search_google("cats", new_tab=True)
        tabs = await agent.get_all_tabs()
        assert len(tabs) == 2


# ──────────────────────────────────────────────────────────────────────────────
# 5. Content / Downloads / History
# ──────────────────────────────────────────────────────────────────────────────


class TestBrowserAgentContent:
    @pytest.mark.asyncio
    async def test_read_page_text_returns_string(self):
        agent = _make_agent()
        await agent.start()
        text = await agent.read_page_text()
        assert isinstance(text, str)
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_read_page_text_respects_max_chars(self):
        agent = _make_agent(read_page_max_chars=10)
        await agent.start()
        # BrowserAgent passes max_chars to backend.read_page_text().
        # FakeBackend doesn't actually truncate (truncation is PlaywrightBackend's job),
        # so we verify the correct argument was forwarded.
        original_fn = agent._backend.read_page_text
        calls = []

        async def capture(*args, **kwargs):
            calls.append(kwargs.get("max_chars", args[0] if args else None))
            return await original_fn(*args, **kwargs)

        agent._backend.read_page_text = capture
        await agent.read_page_text(max_chars=10)
        assert calls == [10]

    @pytest.mark.asyncio
    async def test_download_file_success(self):
        agent = _make_agent()
        result = await agent.download_file("https://example.com/file.zip", dest_dir="/tmp")
        assert isinstance(result, DownloadResult)
        assert result.success is True
        assert result.filename == "fake_file.zip"

    @pytest.mark.asyncio
    async def test_download_file_uses_default_dir(self):
        agent = _make_agent(download_dir="/my/downloads")
        result = await agent.download_file("https://example.com/f.zip")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_get_browser_history(self):
        agent = _make_agent()
        history = await agent.get_browser_history(limit=5)
        assert isinstance(history, list)
        if history:
            assert isinstance(history[0], HistoryEntry)
