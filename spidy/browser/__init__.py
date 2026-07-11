"""
Browser Package — Public API
"""
from spidy.browser.agent import BrowserAgent
from spidy.browser.backends.base import BrowserBackend
from spidy.browser.backends.playwright_backend import PlaywrightBackend
from spidy.browser.types import (
    DownloadResult,
    HistoryEntry,
    PageInfo,
    SearchResult,
    TabInfo,
)

__all__ = [
    "BrowserAgent",
    "BrowserBackend",
    "PlaywrightBackend",
    "PageInfo",
    "TabInfo",
    "DownloadResult",
    "HistoryEntry",
    "SearchResult",
]
