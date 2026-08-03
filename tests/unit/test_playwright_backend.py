"""
Tests for PlaywrightBackend stale-session recovery
====================================================
Covers the three-level healing strategy added to fix:
    Page.goto: Target page, context or browser has been closed

All Playwright objects are mocked — no real browser is launched.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.browser.backends.playwright_backend import PlaywrightBackend
from spidy.browser.types import PageInfo


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_page(closed: bool = False, url: str = "https://example.com") -> MagicMock:
    page = MagicMock()
    page.is_closed.return_value = closed
    page.url = url
    page.goto = AsyncMock()
    page.title = AsyncMock(return_value="Test Page")
    page.inner_text = AsyncMock(return_value="Hello world")
    page.go_back = AsyncMock()
    page.go_forward = AsyncMock()
    page.reload = AsyncMock()
    page.close = AsyncMock()
    return page


def _make_context(pages: list) -> MagicMock:
    ctx = MagicMock()
    ctx.pages = pages
    ctx.new_page = AsyncMock(side_effect=lambda: _make_page())
    return ctx


def _make_browser(connected: bool = True) -> MagicMock:
    browser = MagicMock()
    browser.is_connected.return_value = connected
    browser.close = AsyncMock()
    browser.contexts = []
    return browser


def _running_backend(
    page: MagicMock | None = None,
    context: MagicMock | None = None,
    browser: MagicMock | None = None,
    connected: bool = True,
) -> PlaywrightBackend:
    p = page or _make_page()
    ctx = context or _make_context([p])
    b = browser or _make_browser(connected=connected)
    b.contexts = [ctx]
    backend = PlaywrightBackend()
    backend._running = True
    backend._browser = b
    backend._context = ctx
    backend._pages = [p]
    backend._active_page_idx = 0
    backend._last_start_kwargs = {
        "browser_type": "chromium",
        "headless": False,
        "download_dir": "~",
        "cdp_endpoint": "http://localhost:9222",
        "connect_to_existing": False,
    }
    return backend


# ─────────────────────────────────────────────────────────────────────────────
# 1. is_running structural health check
# ─────────────────────────────────────────────────────────────────────────────


class TestIsRunningStructural:
    def test_false_when_not_started(self):
        assert PlaywrightBackend().is_running is False

    def test_true_when_browser_connected(self):
        assert _running_backend(connected=True).is_running is True

    def test_false_when_browser_disconnected(self):
        assert _running_backend(connected=False).is_running is False

    def test_clears_running_flag_after_disconnect(self):
        b = _running_backend(connected=False)
        _ = b.is_running
        assert b._running is False

    def test_true_when_no_browser_object(self):
        b = PlaywrightBackend()
        b._running = True
        b._browser = None
        assert b.is_running is True

    def test_false_when_is_connected_raises(self):
        browser = MagicMock()
        browser.is_connected.side_effect = Exception("pipe broken")
        assert _running_backend(browser=browser).is_running is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. _is_page_healthy
# ─────────────────────────────────────────────────────────────────────────────


class TestIsPageHealthy:
    def test_open_page_is_healthy(self):
        b = PlaywrightBackend()
        assert b._is_page_healthy(_make_page(closed=False)) is True

    def test_closed_page_is_not_healthy(self):
        b = PlaywrightBackend()
        assert b._is_page_healthy(_make_page(closed=True)) is False

    def test_raises_returns_false(self):
        b = PlaywrightBackend()
        p = MagicMock()
        p.is_closed.side_effect = Exception("Target closed")
        assert b._is_page_healthy(p) is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. _heal() — teardown + relaunch
# ─────────────────────────────────────────────────────────────────────────────


class TestHeal:
    @pytest.mark.asyncio
    async def test_heal_clears_stale_handles(self):
        backend = _running_backend()
        fresh_page = _make_page()
        fresh_ctx = _make_context([fresh_page])
        fresh_browser = _make_browser()

        async def fake_start(**kwargs):
            backend._browser = fresh_browser
            backend._context = fresh_ctx
            backend._pages = [fresh_page]
            backend._active_page_idx = 0
            backend._running = True

        backend.start = fake_start
        await backend._heal()
        assert backend._browser is fresh_browser
        assert backend._running is True

    @pytest.mark.asyncio
    async def test_heal_calls_start_with_stored_kwargs(self):
        backend = _running_backend()
        called_with: dict = {}

        async def spy_start(**kwargs):
            called_with.update(kwargs)
            backend._running = True
            backend._pages = [_make_page()]
            backend._active_page_idx = 0

        backend.start = spy_start
        await backend._heal()
        assert called_with.get("browser_type") == "chromium"
        assert called_with.get("headless") is False
        assert called_with.get("connect_to_existing") is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. _get_or_heal_page() — three-level recovery
# ─────────────────────────────────────────────────────────────────────────────


class TestGetOrHealPage:
    @pytest.mark.asyncio
    async def test_level1_returns_healthy_page(self):
        page = _make_page(closed=False)
        backend = _running_backend(page=page)
        assert await backend._get_or_heal_page() is page

    @pytest.mark.asyncio
    async def test_level2_opens_new_page_when_active_is_closed(self):
        stale = _make_page(closed=True)
        fresh = _make_page(closed=False)
        ctx = _make_context([stale])
        ctx.new_page = AsyncMock(return_value=fresh)
        backend = _running_backend(page=stale, context=ctx)
        result = await backend._get_or_heal_page()
        assert result is fresh
        ctx.new_page.assert_called_once()

    @pytest.mark.asyncio
    async def test_level3_full_heal_when_context_is_dead(self):
        stale = _make_page(closed=True)
        ctx = _make_context([stale])
        ctx.new_page = AsyncMock(side_effect=Exception("context closed"))
        fresh = _make_page(closed=False)
        heal_called = []

        backend = _running_backend(page=stale, context=ctx)

        async def fake_heal():
            heal_called.append(True)
            backend._pages = [fresh]
            backend._active_page_idx = 0
            backend._running = True

        backend._heal = fake_heal
        result = await backend._get_or_heal_page()
        assert heal_called
        assert result is fresh

    @pytest.mark.asyncio
    async def test_level1_refreshes_page_list_from_context(self):
        old = _make_page(closed=False, url="https://old.com")
        new = _make_page(closed=False, url="https://new.com")
        ctx = MagicMock()
        ctx.pages = [new]
        backend = _running_backend(page=old, context=ctx)
        assert await backend._get_or_heal_page() is new


# ─────────────────────────────────────────────────────────────────────────────
# 5. open_url — end-to-end stale-session simulation
# ─────────────────────────────────────────────────────────────────────────────


class TestOpenUrlStalePage:
    @pytest.mark.asyncio
    async def test_second_open_url_heals_after_external_close(self):
        first_page = _make_page(closed=False, url="https://first.com")
        fresh_page = _make_page(closed=False, url="https://google.com/search?q=dinosaurs")
        ctx = _make_context([first_page])
        backend = _running_backend(page=first_page, context=ctx)

        # First navigation succeeds
        await backend.open_url("https://first.com")

        # External close — page goes stale
        first_page.is_closed.return_value = True
        ctx.pages = [first_page]
        ctx.new_page = AsyncMock(return_value=fresh_page)

        # Second navigation must heal automatically
        info = await backend.open_url("https://google.com/search?q=dinosaurs")
        assert info is not None
        ctx.new_page.assert_called_once()

    @pytest.mark.asyncio
    async def test_open_url_does_not_raise_on_closed_page(self):
        stale = _make_page(closed=True)
        fresh = _make_page(closed=False)
        ctx = _make_context([stale])
        ctx.new_page = AsyncMock(return_value=fresh)
        backend = _running_backend(page=stale, context=ctx)
        result = await backend.open_url("https://example.com")
        assert isinstance(result, PageInfo)


# ─────────────────────────────────────────────────────────────────────────────
# 6. start() stores _last_start_kwargs
# ─────────────────────────────────────────────────────────────────────────────


class TestStartKwargsPersistence:
    @pytest.mark.asyncio
    async def test_start_stores_kwargs(self):
        backend = PlaywrightBackend()

        with (
            patch(
                "spidy.browser.backends.playwright_backend._PLAYWRIGHT_AVAILABLE", True
            ),
            patch(
                "spidy.browser.backends.playwright_backend.async_playwright"
            ) as mock_pw,
        ):
            pw_instance = AsyncMock()
            mock_pw.return_value.start = AsyncMock(return_value=pw_instance)
            chromium = MagicMock()
            pw_instance.chromium = chromium
            browser = _make_browser()
            chromium.connect_over_cdp = AsyncMock(side_effect=Exception("no CDP"))
            chromium.launch = AsyncMock(return_value=browser)
            ctx = _make_context([_make_page()])
            browser.new_context = AsyncMock(return_value=ctx)

            await backend.start(
                browser_type="firefox",
                headless=True,
                download_dir="/tmp",
                connect_to_existing=False,
                cdp_endpoint="",
            )

        assert backend._last_start_kwargs["browser_type"] == "firefox"
        assert backend._last_start_kwargs["headless"] is True
        assert backend._last_start_kwargs["connect_to_existing"] is False
