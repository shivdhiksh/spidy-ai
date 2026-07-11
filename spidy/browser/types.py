"""
Browser Agent Types
===================
Shared immutable data structures used by BrowserAgent, BrowserBackend,
and BrowserSkill.

All types are frozen dataclasses for safety and hashability.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ─── Page / Tab Info ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PageInfo:
    """
    Information about the currently active browser page.

    Attributes
    ----------
    title:
        The page's ``<title>`` element text.
    url:
        The full URL of the page.
    tab_id:
        Internal tab identifier (context index in Playwright).
    load_time_ms:
        Approximate page load time in milliseconds (0 if not measured).
    """
    title: str = ""
    url: str = ""
    tab_id: int = 0
    load_time_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "tab_id": self.tab_id,
            "load_time_ms": self.load_time_ms,
        }


@dataclass(frozen=True)
class TabInfo:
    """
    Lightweight descriptor for an open browser tab.

    Attributes
    ----------
    tab_id:
        Index of the browser context/page (0-based).
    title:
        Page title.
    url:
        Current URL.
    is_active:
        True if this is the tab most recently interacted with.
    """
    tab_id: int = 0
    title: str = ""
    url: str = ""
    is_active: bool = False

    def to_dict(self) -> dict:
        return {
            "tab_id": self.tab_id,
            "title": self.title,
            "url": self.url,
            "is_active": self.is_active,
        }


# ─── Download ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DownloadResult:
    """
    Result of a file download operation.

    Attributes
    ----------
    filename:
        The name of the downloaded file.
    path:
        Full path to the downloaded file on disk.
    size_bytes:
        File size in bytes (0 if unknown).
    success:
        True if the download completed successfully.
    error:
        Error message if download failed, empty string otherwise.
    """
    filename: str = ""
    path: str = ""
    size_bytes: int = 0
    success: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "success": self.success,
            "error": self.error,
        }


# ─── History ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HistoryEntry:
    """
    A single entry from the browser's navigation history.

    Attributes
    ----------
    title:
        Page title at the time of the visit.
    url:
        The visited URL.
    visit_time:
        ISO-8601 timestamp of the visit, empty if unknown.
    """
    title: str = ""
    url: str = ""
    visit_time: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "visit_time": self.visit_time,
        }


# ─── Search ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SearchResult:
    """
    A single web search result (reserved for future result-scraping).

    Currently unused; search actions navigate to the search results page
    and return a ``PageInfo`` rather than parsed results.
    """
    title: str = ""
    url: str = ""
    snippet: str = ""

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}
