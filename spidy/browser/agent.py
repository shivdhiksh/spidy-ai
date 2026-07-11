"""
BrowserAgent — Browser Session Orchestrator
============================================
Owns the lifecycle of a single browser automation session and exposes a
clean async API that BrowserSkill calls.

Architecture
------------
BrowserSkill
    └── BrowserAgent  ← this module
            └── BrowserBackend ABC
                    └── PlaywrightBackend (default)

BrowserAgent is the single point of contact between the Skill layer and the
browser automation layer. It:
  - Manages start/stop lifecycle
  - Delegates all browser operations to the injected BrowserBackend
  - Builds search/navigation URLs
  - Ensures the backend is running before delegating (auto-start)
  - Returns plain Python types (PageInfo, TabInfo, DownloadResult) —
    no Playwright types leak upward

Design decisions
----------------
1. **Backend injection** — the default backend is PlaywrightBackend, but any
   BrowserBackend subclass can be injected at construction (useful in tests).

2. **Auto-start** — if a method is called before start(), BrowserAgent starts
   the backend automatically using the stored config. This avoids requiring
   the skill to call start() explicitly before every action.

3. **Search URLs** — Google and YouTube searches open the search results page
   in the browser (no scraping). The user sees the results; they can then use
   read_page or scroll manually.

4. **Error containment** — all operations return SkillResult-ready tuples or
   typed objects. Exceptions are caught and logged; callers receive meaningful
   error info rather than a traceback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spidy.browser.backends.playwright_backend import PlaywrightBackend
from spidy.browser.types import DownloadResult, HistoryEntry, PageInfo, TabInfo
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.browser.backends.base import BrowserBackend

log = get_logger(__name__)

# Search URL templates
_GOOGLE_SEARCH_URL = "https://www.google.com/search?q={query}"
_YOUTUBE_SEARCH_URL = "https://www.youtube.com/results?search_query={query}"


class BrowserAgent:
    """
    Orchestrates browser sessions through a pluggable BrowserBackend.

    Parameters
    ----------
    backend:
        BrowserBackend implementation. Defaults to PlaywrightBackend.
    browser_type:
        ``"chromium"`` | ``"firefox"`` | ``"webkit"`` (default: ``"chromium"``).
    headless:
        Run browser without visible window. Default: False (user sees it).
    download_dir:
        Default destination directory for downloads. Default: home dir.
    connect_to_existing:
        Try to attach to an existing Chrome CDP session first.
    cdp_endpoint:
        Chrome DevTools endpoint URL for CDP attach.
    page_load_timeout_ms:
        Page load timeout in milliseconds.
    navigation_timeout_ms:
        Navigation operation timeout in milliseconds.
    read_page_max_chars:
        Maximum characters returned by read_page_text().
    """

    def __init__(
        self,
        backend: "BrowserBackend | None" = None,
        browser_type: str = "chromium",
        headless: bool = False,
        download_dir: str = "~",
        connect_to_existing: bool = True,
        cdp_endpoint: str = "http://localhost:9222",
        page_load_timeout_ms: int = 30_000,
        navigation_timeout_ms: int = 10_000,
        read_page_max_chars: int = 5000,
    ) -> None:
        self._backend: BrowserBackend = backend or PlaywrightBackend(
            page_load_timeout_ms=page_load_timeout_ms,
            navigation_timeout_ms=navigation_timeout_ms,
        )
        self._browser_type = browser_type
        self._headless = headless
        self._download_dir = download_dir
        self._connect_to_existing = connect_to_existing
        self._cdp_endpoint = cdp_endpoint
        self._read_page_max_chars = read_page_max_chars

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Start the browser session.

        Safe to call multiple times — no-op if already running.
        """
        if self._backend.is_running:
            return
        log.info(
            "BrowserAgent.start(): launching {bt} (headless={h})",
            bt=self._browser_type,
            h=self._headless,
        )
        await self._backend.start(
            browser_type=self._browser_type,
            headless=self._headless,
            download_dir=self._download_dir,
            cdp_endpoint=self._cdp_endpoint,
            connect_to_existing=self._connect_to_existing,
        )

    async def stop(self) -> None:
        """Stop the browser session and release all resources."""
        if not self._backend.is_running:
            return
        log.info("BrowserAgent.stop(): closing browser.")
        await self._backend.stop()

    @property
    def is_running(self) -> bool:
        """Return True if the browser session is active."""
        return self._backend.is_running

    # ── Auto-start helper ─────────────────────────────────────────────────

    async def _ensure_running(self) -> None:
        """Start the backend if it is not already running."""
        if not self._backend.is_running:
            await self.start()

    # ── Navigation ────────────────────────────────────────────────────────

    async def open_url(self, url: str, new_tab: bool = False) -> PageInfo:
        """
        Open a URL in the current or a new tab.

        Parameters
        ----------
        url:
            Target URL. Scheme (``https://``) is added if missing.
        new_tab:
            Open in a new tab if True.

        Returns
        -------
        PageInfo
            Info about the resulting page.
        """
        await self._ensure_running()
        log.info("BrowserAgent.open_url: '{url}' (new_tab={nt})", url=url, nt=new_tab)
        return await self._backend.open_url(url, new_tab=new_tab)

    async def close_tab(self, tab_id: int) -> bool:
        """Close the tab with the given ID. Returns True on success."""
        await self._ensure_running()
        closed = await self._backend.close_tab(tab_id)
        log.info("BrowserAgent.close_tab({tab_id}): success={ok}", tab_id=tab_id, ok=closed)
        return closed

    async def get_current_page_info(self) -> PageInfo:
        """Return info about the currently active page."""
        await self._ensure_running()
        return await self._backend.get_current_page_info()

    async def get_all_tabs(self) -> list[TabInfo]:
        """Return a list of all open tabs."""
        await self._ensure_running()
        return await self._backend.get_all_tabs()

    async def navigate_back(self) -> PageInfo:
        """Navigate back in the current tab's history."""
        await self._ensure_running()
        log.info("BrowserAgent.navigate_back()")
        return await self._backend.navigate_back()

    async def navigate_forward(self) -> PageInfo:
        """Navigate forward in the current tab's history."""
        await self._ensure_running()
        log.info("BrowserAgent.navigate_forward()")
        return await self._backend.navigate_forward()

    async def refresh(self) -> PageInfo:
        """Reload the current page."""
        await self._ensure_running()
        log.info("BrowserAgent.refresh()")
        return await self._backend.refresh()

    # ── Search ────────────────────────────────────────────────────────────

    async def search_google(self, query: str, new_tab: bool = False) -> PageInfo:
        """
        Open Google search results for ``query``.

        Parameters
        ----------
        query:
            The search query string.
        new_tab:
            Open in a new tab.

        Returns
        -------
        PageInfo
            Info about the Google search results page.
        """
        from urllib.parse import quote_plus
        url = _GOOGLE_SEARCH_URL.format(query=quote_plus(query))
        log.info("BrowserAgent.search_google: '{q}'", q=query)
        return await self.open_url(url, new_tab=new_tab)

    async def search_youtube(self, query: str, new_tab: bool = False) -> PageInfo:
        """
        Open YouTube search results for ``query``.

        Parameters
        ----------
        query:
            The search query string.
        new_tab:
            Open in a new tab.

        Returns
        -------
        PageInfo
            Info about the YouTube search results page.
        """
        from urllib.parse import quote_plus
        url = _YOUTUBE_SEARCH_URL.format(query=quote_plus(query))
        log.info("BrowserAgent.search_youtube: '{q}'", q=query)
        return await self.open_url(url, new_tab=new_tab)

    # ── Content ───────────────────────────────────────────────────────────

    async def read_page_text(self, max_chars: int | None = None) -> str:
        """
        Extract visible text from the current page.

        Parameters
        ----------
        max_chars:
            Override the default character limit.

        Returns
        -------
        str
            The visible page text, truncated if necessary.
        """
        await self._ensure_running()
        limit = max_chars if max_chars is not None else self._read_page_max_chars
        return await self._backend.read_page_text(max_chars=limit)

    # ── Downloads ─────────────────────────────────────────────────────────

    async def download_file(
        self, url: str, dest_dir: str | None = None
    ) -> DownloadResult:
        """
        Download a file from a URL.

        Parameters
        ----------
        url:
            The direct download URL.
        dest_dir:
            Destination directory. Falls back to the configured download_dir.

        Returns
        -------
        DownloadResult
            Result including path and success status.
        """
        await self._ensure_running()
        directory = dest_dir or self._download_dir
        log.info(
            "BrowserAgent.download_file: '{url}' → {dir}",
            url=url, dir=directory,
        )
        return await self._backend.download_file(url, directory)

    # ── History ───────────────────────────────────────────────────────────

    async def get_browser_history(self, limit: int = 20) -> list[HistoryEntry]:
        """
        Retrieve recent browser history entries.

        Returns an empty list if history is unavailable (browser running,
        locked DB, or non-Chromium browser).

        Parameters
        ----------
        limit:
            Maximum number of entries to return.

        Returns
        -------
        list[HistoryEntry]
            Most recent entries first.
        """
        await self._ensure_running()
        return await self._backend.get_browser_history(limit=limit)
