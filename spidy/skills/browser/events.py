"""
Browser Skill Events — EventBus Events for the Browser Agent
============================================================
Topic namespace: ``browser.*``

All 12 event types are frozen dataclasses. The Brain subscribes to these
events to track browser state. The UI can subscribe to show the user
what URL is open, what was downloaded, etc.

Topics
------
browser.launched          — browser session started
browser.closed            — browser session stopped
browser.page_navigated    — URL changed (open_url, search, etc.)
browser.tab_closed        — a tab was closed
browser.page_read         — visible text was extracted from page
browser.search_results    — Google or YouTube search completed
browser.download_started  — a download was initiated
browser.download_completed — a download finished
browser.nav_back          — browser navigated backward
browser.nav_forward       — browser navigated forward
browser.page_refreshed    — current page was reloaded
browser.error             — any browser-layer error occurred
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spidy.core.event_bus import Event


# ─── Lifecycle ────────────────────────────────────────────────────────────────


@dataclass
class BrowserLaunchedEvent(Event):
    """Published when a browser session starts successfully."""
    topic = "browser.launched"
    browser_type: str = "chromium"     # chromium | firefox | webkit
    headless: bool = False
    session_id: str = ""


@dataclass
class BrowserClosedEvent(Event):
    """Published when the browser session is closed."""
    topic = "browser.closed"
    session_id: str = ""


# ─── Navigation ───────────────────────────────────────────────────────────────


@dataclass
class PageNavigatedEvent(Event):
    """
    Published after any navigation that changes the active page URL.

    Covers: open_url, open_new_tab, search_google, search_youtube.
    """
    topic = "browser.page_navigated"
    title: str = ""
    url: str = ""
    tab_id: int = 0
    new_tab: bool = False
    load_time_ms: int = 0
    session_id: str = ""


@dataclass
class TabClosedEvent(Event):
    """Published when a tab is closed."""
    topic = "browser.tab_closed"
    tab_id: int = 0
    session_id: str = ""


# ─── Content ─────────────────────────────────────────────────────────────────


@dataclass
class PageReadEvent(Event):
    """Published after visible text is extracted from the current page."""
    topic = "browser.page_read"
    url: str = ""
    title: str = ""
    char_count: int = 0
    session_id: str = ""


# ─── Search ───────────────────────────────────────────────────────────────────


@dataclass
class SearchResultsEvent(Event):
    """
    Published after a Google or YouTube search is initiated.

    Note: this event reflects the navigation to the search results page,
    not individual parsed results. Use PageReadEvent after to get content.
    """
    topic = "browser.search_results"
    engine: str = "google"             # "google" | "youtube"
    query: str = ""
    results_url: str = ""
    session_id: str = ""


# ─── Downloads ────────────────────────────────────────────────────────────────


@dataclass
class DownloadStartedEvent(Event):
    """Published when a file download is initiated."""
    topic = "browser.download_started"
    url: str = ""
    dest_dir: str = ""
    session_id: str = ""


@dataclass
class DownloadCompletedEvent(Event):
    """Published when a file download finishes (success or failure)."""
    topic = "browser.download_completed"
    filename: str = ""
    path: str = ""
    size_bytes: int = 0
    success: bool = False
    error: str = ""
    session_id: str = ""


# ─── Navigation History ───────────────────────────────────────────────────────


@dataclass
class NavigationBackEvent(Event):
    """Published after navigating backward in browser history."""
    topic = "browser.nav_back"
    title: str = ""
    url: str = ""
    tab_id: int = 0
    session_id: str = ""


@dataclass
class NavigationForwardEvent(Event):
    """Published after navigating forward in browser history."""
    topic = "browser.nav_forward"
    title: str = ""
    url: str = ""
    tab_id: int = 0
    session_id: str = ""


@dataclass
class PageRefreshedEvent(Event):
    """Published after the current page is reloaded."""
    topic = "browser.page_refreshed"
    title: str = ""
    url: str = ""
    tab_id: int = 0
    session_id: str = ""


# ─── Error ───────────────────────────────────────────────────────────────────


@dataclass
class BrowserErrorEvent(Event):
    """Published when a browser-layer error occurs."""
    topic = "browser.error"
    action: str = ""
    error: str = ""
    session_id: str = ""
