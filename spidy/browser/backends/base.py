"""
BrowserBackend — Abstract Base Class
=====================================
Defines the stable contract that any browser automation backend must implement.

Current implementations
-----------------------
- PlaywrightBackend  (spidy/browser/backends/playwright_backend.py)

Future implementations
----------------------
- SeleniumBackend  (M8+, if needed)
- CDPDirectBackend (raw Chrome DevTools Protocol, M8+)

Design
------
BrowserAgent holds a reference to one BrowserBackend instance.
BrowserSkill never imports this module — it only knows about BrowserAgent.
This keeps the dependency chain:

    BrowserSkill → BrowserAgent → BrowserBackend ← PlaywrightBackend
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from spidy.browser.types import DownloadResult, HistoryEntry, PageInfo, TabInfo


class BrowserBackend(ABC):
    """
    Abstract contract for browser automation backends.

    All methods are async. Implementations must not raise to the caller
    except for ``BrowserBackendError`` subclasses; all other exceptions
    should be caught internally and returned through the result types.
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────

    @abstractmethod
    async def start(
        self,
        browser_type: str = "chromium",
        headless: bool = False,
        download_dir: str = "~",
        cdp_endpoint: str = "",
        connect_to_existing: bool = True,
    ) -> None:
        """
        Start the browser session.

        Parameters
        ----------
        browser_type:
            ``"chromium"`` | ``"firefox"`` | ``"webkit"``
        headless:
            Run the browser without a visible window.
        download_dir:
            Default directory for downloaded files.
        cdp_endpoint:
            Chrome DevTools Protocol endpoint URL for attaching to an
            existing browser session (e.g. ``"http://localhost:9222"``).
        connect_to_existing:
            If True, try to attach to ``cdp_endpoint`` first; fall back
            to launching a new browser if the connection fails.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Close the browser and release all resources."""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Return True if the browser session is active."""

    @property
    def active_browser_type(self) -> str:
        """Canonical name of the active browser (e.g. 'edge', 'chrome', 'firefox', 'chromium')."""
        return ""

    @property
    def active_channel(self) -> str:
        """Playwright launch channel if used (e.g. 'msedge', 'chrome')."""
        return ""

    # ── Navigation ────────────────────────────────────────────────────────

    @abstractmethod
    async def open_url(self, url: str, new_tab: bool = False) -> PageInfo:
        """
        Navigate to a URL.

        Parameters
        ----------
        url:
            The URL to open. Must include the scheme (``https://``).
        new_tab:
            If True, open the URL in a new tab; otherwise reuse the
            current active page.

        Returns
        -------
        PageInfo
            Info about the resulting page after navigation.
        """

    @abstractmethod
    async def close_tab(self, tab_id: int) -> bool:
        """
        Close the tab with the given ID.

        Returns
        -------
        bool
            True if the tab was found and closed; False otherwise.
        """

    @abstractmethod
    async def get_current_page_info(self) -> PageInfo:
        """Return info about the currently active page."""

    @abstractmethod
    async def get_all_tabs(self) -> list[TabInfo]:
        """Return a list of all open tabs."""

    @abstractmethod
    async def navigate_back(self) -> PageInfo:
        """Navigate back in the current tab's history."""

    @abstractmethod
    async def navigate_forward(self) -> PageInfo:
        """Navigate forward in the current tab's history."""

    @abstractmethod
    async def refresh(self) -> PageInfo:
        """Reload the current page."""

    # ── Content ───────────────────────────────────────────────────────────

    @abstractmethod
    async def read_page_text(self, max_chars: int = 5000) -> str:
        """
        Extract the visible text content of the current page.

        Parameters
        ----------
        max_chars:
            Maximum number of characters to return.

        Returns
        -------
        str
            The extracted text, truncated to ``max_chars``.
        """

    # ── Downloads ─────────────────────────────────────────────────────────

    @abstractmethod
    async def download_file(self, url: str, dest_dir: str) -> DownloadResult:
        """
        Download a file from a URL.

        Parameters
        ----------
        url:
            The direct download URL.
        dest_dir:
            Directory to save the file. Will be created if it does not exist.

        Returns
        -------
        DownloadResult
            Result of the download, including path and success status.
        """

    # ── History ───────────────────────────────────────────────────────────

    @abstractmethod
    async def get_browser_history(self, limit: int = 20) -> list[HistoryEntry]:
        """
        Retrieve recent browser history entries.

        Implementation note: Chrome history is stored in a SQLite database
        that is locked while Chrome is running. Backends may return an empty
        list if history is unavailable in the current mode.

        Returns
        -------
        list[HistoryEntry]
            Most recent entries first, up to ``limit``.
        """
