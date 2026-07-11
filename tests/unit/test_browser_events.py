"""
Tests for Milestone 7 — Browser Events
========================================
Unit tests for all 12 EventBus event types in the browser.* namespace:
- BrowserLaunchedEvent
- BrowserClosedEvent
- PageNavigatedEvent
- TabClosedEvent
- PageReadEvent
- SearchResultsEvent
- DownloadStartedEvent
- DownloadCompletedEvent
- NavigationBackEvent
- NavigationForwardEvent
- PageRefreshedEvent
- BrowserErrorEvent
"""

from __future__ import annotations

import asyncio

import pytest

from spidy.core.event_bus import Event, EventBus
from spidy.skills.browser.events import (
    BrowserClosedEvent,
    BrowserErrorEvent,
    BrowserLaunchedEvent,
    DownloadCompletedEvent,
    DownloadStartedEvent,
    NavigationBackEvent,
    NavigationForwardEvent,
    PageNavigatedEvent,
    PageReadEvent,
    PageRefreshedEvent,
    SearchResultsEvent,
    TabClosedEvent,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Event subclass registration
# ──────────────────────────────────────────────────────────────────────────────


class TestEventInheritance:
    def test_all_events_are_event_subclasses(self):
        events = [
            BrowserLaunchedEvent, BrowserClosedEvent, PageNavigatedEvent,
            TabClosedEvent, PageReadEvent, SearchResultsEvent,
            DownloadStartedEvent, DownloadCompletedEvent,
            NavigationBackEvent, NavigationForwardEvent,
            PageRefreshedEvent, BrowserErrorEvent,
        ]
        for evt_cls in events:
            assert issubclass(evt_cls, Event), f"{evt_cls.__name__} must subclass Event"

    def test_all_events_have_unique_topics(self):
        events = [
            BrowserLaunchedEvent(), BrowserClosedEvent(), PageNavigatedEvent(),
            TabClosedEvent(), PageReadEvent(), SearchResultsEvent(),
            DownloadStartedEvent(), DownloadCompletedEvent(),
            NavigationBackEvent(), NavigationForwardEvent(),
            PageRefreshedEvent(), BrowserErrorEvent(),
        ]
        topics = [e.topic for e in events]
        assert len(topics) == len(set(topics)), "Duplicate topics found"

    def test_all_topics_start_with_browser(self):
        events = [
            BrowserLaunchedEvent(), BrowserClosedEvent(), PageNavigatedEvent(),
            TabClosedEvent(), PageReadEvent(), SearchResultsEvent(),
            DownloadStartedEvent(), DownloadCompletedEvent(),
            NavigationBackEvent(), NavigationForwardEvent(),
            PageRefreshedEvent(), BrowserErrorEvent(),
        ]
        for evt in events:
            assert evt.topic.startswith("browser."), (
                f"{type(evt).__name__}.topic='{evt.topic}' must start with 'browser.'"
            )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Lifecycle events
# ──────────────────────────────────────────────────────────────────────────────


class TestBrowserLaunchedEvent:
    def test_topic(self):
        assert BrowserLaunchedEvent().topic == "browser.launched"

    def test_defaults(self):
        evt = BrowserLaunchedEvent()
        assert evt.browser_type == "chromium"
        assert evt.headless is False
        assert evt.session_id == ""

    def test_custom_values(self):
        evt = BrowserLaunchedEvent(browser_type="firefox", headless=True, session_id="s1")
        assert evt.browser_type == "firefox"
        assert evt.headless is True
        assert evt.session_id == "s1"


class TestBrowserClosedEvent:
    def test_topic(self):
        assert BrowserClosedEvent().topic == "browser.closed"

    def test_session_id(self):
        evt = BrowserClosedEvent(session_id="abc")
        assert evt.session_id == "abc"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Navigation events
# ──────────────────────────────────────────────────────────────────────────────


class TestPageNavigatedEvent:
    def test_topic(self):
        assert PageNavigatedEvent().topic == "browser.page_navigated"

    def test_defaults(self):
        evt = PageNavigatedEvent()
        assert evt.title == ""
        assert evt.url == ""
        assert evt.tab_id == 0
        assert evt.new_tab is False
        assert evt.load_time_ms == 0
        assert evt.session_id == ""

    def test_with_values(self):
        evt = PageNavigatedEvent(
            title="Google", url="https://google.com",
            tab_id=1, new_tab=True, load_time_ms=250, session_id="s1",
        )
        assert evt.title == "Google"
        assert evt.url == "https://google.com"
        assert evt.tab_id == 1
        assert evt.new_tab is True
        assert evt.load_time_ms == 250


class TestTabClosedEvent:
    def test_topic(self):
        assert TabClosedEvent().topic == "browser.tab_closed"

    def test_tab_id(self):
        evt = TabClosedEvent(tab_id=3, session_id="s2")
        assert evt.tab_id == 3
        assert evt.session_id == "s2"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Content events
# ──────────────────────────────────────────────────────────────────────────────


class TestPageReadEvent:
    def test_topic(self):
        assert PageReadEvent().topic == "browser.page_read"

    def test_char_count(self):
        evt = PageReadEvent(url="https://a.com", title="A", char_count=3000, session_id="s")
        assert evt.char_count == 3000
        assert evt.url == "https://a.com"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Search events
# ──────────────────────────────────────────────────────────────────────────────


class TestSearchResultsEvent:
    def test_topic(self):
        assert SearchResultsEvent().topic == "browser.search_results"

    def test_defaults(self):
        evt = SearchResultsEvent()
        assert evt.engine == "google"
        assert evt.query == ""
        assert evt.results_url == ""

    def test_youtube_search(self):
        evt = SearchResultsEvent(engine="youtube", query="lofi music",
                                  results_url="https://www.youtube.com/results?search_query=lofi+music")
        assert evt.engine == "youtube"
        assert evt.query == "lofi music"


# ──────────────────────────────────────────────────────────────────────────────
# 6. Download events
# ──────────────────────────────────────────────────────────────────────────────


class TestDownloadStartedEvent:
    def test_topic(self):
        assert DownloadStartedEvent().topic == "browser.download_started"

    def test_url_and_dir(self):
        evt = DownloadStartedEvent(url="https://a.com/file.zip", dest_dir="/tmp", session_id="s")
        assert evt.url == "https://a.com/file.zip"
        assert evt.dest_dir == "/tmp"


class TestDownloadCompletedEvent:
    def test_topic(self):
        assert DownloadCompletedEvent().topic == "browser.download_completed"

    def test_success(self):
        evt = DownloadCompletedEvent(
            filename="file.zip", path="/tmp/file.zip",
            size_bytes=10240, success=True, error=""
        )
        assert evt.success is True
        assert evt.filename == "file.zip"
        assert evt.size_bytes == 10240

    def test_failure(self):
        evt = DownloadCompletedEvent(success=False, error="Timeout")
        assert evt.success is False
        assert evt.error == "Timeout"


# ──────────────────────────────────────────────────────────────────────────────
# 7. Navigation history events
# ──────────────────────────────────────────────────────────────────────────────


class TestNavigationBackEvent:
    def test_topic(self):
        assert NavigationBackEvent().topic == "browser.nav_back"

    def test_fields(self):
        evt = NavigationBackEvent(title="Prev", url="https://prev.com", tab_id=0, session_id="s")
        assert evt.title == "Prev"
        assert evt.url == "https://prev.com"


class TestNavigationForwardEvent:
    def test_topic(self):
        assert NavigationForwardEvent().topic == "browser.nav_forward"

    def test_fields(self):
        evt = NavigationForwardEvent(title="Next", url="https://next.com", tab_id=1)
        assert evt.title == "Next"


class TestPageRefreshedEvent:
    def test_topic(self):
        assert PageRefreshedEvent().topic == "browser.page_refreshed"

    def test_fields(self):
        evt = PageRefreshedEvent(title="Refreshed", url="https://r.com", tab_id=0)
        assert evt.title == "Refreshed"


# ──────────────────────────────────────────────────────────────────────────────
# 8. Error event
# ──────────────────────────────────────────────────────────────────────────────


class TestBrowserErrorEvent:
    def test_topic(self):
        assert BrowserErrorEvent().topic == "browser.error"

    def test_fields(self):
        evt = BrowserErrorEvent(action="open_url", error="Connection refused", session_id="s")
        assert evt.action == "open_url"
        assert "refused" in evt.error
        assert evt.session_id == "s"


# ──────────────────────────────────────────────────────────────────────────────
# 9. EventBus round-trip
# ──────────────────────────────────────────────────────────────────────────────


class TestEventBusRoundtrip:
    @pytest.mark.asyncio
    async def test_page_navigated_event_roundtrip(self):
        bus = EventBus()
        received: list[PageNavigatedEvent] = []

        async def handler(evt: PageNavigatedEvent) -> None:
            received.append(evt)

        bus.subscribe("browser.page_navigated", handler)
        evt = PageNavigatedEvent(title="Google", url="https://google.com", session_id="t")
        await bus.publish(evt)

        assert len(received) == 1
        assert received[0].url == "https://google.com"

    @pytest.mark.asyncio
    async def test_download_completed_roundtrip(self):
        bus = EventBus()
        received: list[DownloadCompletedEvent] = []

        async def handler(evt: DownloadCompletedEvent) -> None:
            received.append(evt)

        bus.subscribe("browser.download_completed", handler)
        await bus.publish(DownloadCompletedEvent(filename="f.pdf", success=True))

        assert received[0].filename == "f.pdf"

    @pytest.mark.asyncio
    async def test_all_12_events_publishable(self):
        """All events should be publishable without errors."""
        bus = EventBus()
        events = [
            BrowserLaunchedEvent(), BrowserClosedEvent(), PageNavigatedEvent(),
            TabClosedEvent(), PageReadEvent(), SearchResultsEvent(),
            DownloadStartedEvent(), DownloadCompletedEvent(),
            NavigationBackEvent(), NavigationForwardEvent(),
            PageRefreshedEvent(), BrowserErrorEvent(),
        ]
        for evt in events:
            count = await bus.publish(evt)
            assert count == 0  # No subscribers, but no errors either
