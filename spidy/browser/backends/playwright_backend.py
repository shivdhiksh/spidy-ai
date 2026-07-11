"""
PlaywrightBackend — Playwright-based Browser Automation
=======================================================
Implements the BrowserBackend ABC using Microsoft Playwright's async API.

Capabilities
------------
- Chromium (default), Firefox, WebKit browser types
- CDP attach: reuse an existing Chrome session started with
  --remote-debugging-port=9222 (no new window for the user)
- Headed or headless mode (headless=False by default)
- Download handling with configurable destination directory
- Multi-tab management via Playwright Page list

Graceful degradation
--------------------
If ``playwright`` is not installed, ``import`` raises ``ImportError`` which is
caught by ``BrowserAgent.start()`` and returned as a ``SkillResult.fail()``.
This follows the same pattern as M6 optional Windows dependencies (pycaw, winshell).

Thread Safety
-------------
Playwright's async API must run on the asyncio event loop.
All methods are ``async def`` — no ``asyncio.to_thread`` needed.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from spidy.browser.backends.base import BrowserBackend
from spidy.browser.types import DownloadResult, HistoryEntry, PageInfo, TabInfo
from spidy.logging.logger import get_logger

log = get_logger(__name__)

# Playwright is optional — guarded at import time
try:
    from playwright.async_api import (
        Browser,
        BrowserContext,
        Page,
        Playwright,
        async_playwright,
    )
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False
    Browser = object  # type: ignore[assignment,misc]
    BrowserContext = object  # type: ignore[assignment,misc]
    Page = object  # type: ignore[assignment,misc]
    Playwright = object  # type: ignore[assignment,misc]


class PlaywrightBackend(BrowserBackend):
    """
    Playwright-based implementation of BrowserBackend.

    Parameters
    ----------
    page_load_timeout_ms:
        Maximum time in ms to wait for a page to load.
    navigation_timeout_ms:
        Maximum time in ms for navigation operations (back/forward/refresh).
    """

    def __init__(
        self,
        page_load_timeout_ms: int = 30_000,
        navigation_timeout_ms: int = 10_000,
    ) -> None:
        self._page_load_timeout = page_load_timeout_ms
        self._nav_timeout = navigation_timeout_ms
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._pages: list[Page] = []
        self._active_page_idx: int = 0
        self._download_dir: str = str(Path.home())
        self._running: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(
        self,
        browser_type: str = "chromium",
        headless: bool = False,
        download_dir: str = "~",
        cdp_endpoint: str = "http://localhost:9222",
        connect_to_existing: bool = True,
    ) -> None:
        if not _PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Playwright is not installed. Run: py -m playwright install chromium"
            )
        if self._running:
            log.debug("PlaywrightBackend.start() called when already running — no-op.")
            return

        self._download_dir = str(Path(download_dir).expanduser())
        os.makedirs(self._download_dir, exist_ok=True)

        self._playwright = await async_playwright().start()

        # Try attaching to an existing session first
        if connect_to_existing and cdp_endpoint and browser_type == "chromium":
            attached = await self._try_cdp_attach(cdp_endpoint)
            if attached:
                self._running = True
                log.info("PlaywrightBackend: attached to existing Chrome session.")
                return

        # Launch a new browser
        browser_launcher = {
            "chromium": self._playwright.chromium,
            "firefox": self._playwright.firefox,
            "webkit": self._playwright.webkit,
        }.get(browser_type, self._playwright.chromium)

        self._browser = await browser_launcher.launch(headless=headless)
        self._context = await self._browser.new_context(
            accept_downloads=True,
        )
        page = await self._context.new_page()
        self._pages = [page]
        self._active_page_idx = 0
        self._running = True
        log.info(
            "PlaywrightBackend: launched {bt} browser (headless={h}).",
            bt=browser_type, h=headless,
        )

    async def _try_cdp_attach(self, endpoint: str) -> bool:
        """Try to connect to an existing Chrome via CDP. Return True on success."""
        try:
            assert self._playwright is not None
            self._browser = await self._playwright.chromium.connect_over_cdp(endpoint)
            contexts = self._browser.contexts
            if contexts:
                self._context = contexts[0]
                pages = self._context.pages
                if pages:
                    self._pages = list(pages)
                    self._active_page_idx = 0
                    return True
            # Context/page empty — fall through to launching fresh
            await self._browser.close()
            self._browser = None
        except Exception as exc:  # noqa: BLE001
            log.debug("CDP attach failed ({exc}) — will launch new browser.", exc=exc)
        return False

    async def stop(self) -> None:
        if not self._running:
            return
        try:
            if self._browser is not None:
                await self._browser.close()
            if self._playwright is not None:
                await self._playwright.stop()
        except Exception as exc:  # noqa: BLE001
            log.warning("PlaywrightBackend.stop() error (ignored): {exc}", exc=exc)
        finally:
            self._browser = None
            self._context = None
            self._pages = []
            self._playwright = None
            self._running = False
            log.info("PlaywrightBackend: stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Active page helper ────────────────────────────────────────────────

    def _active_page(self) -> Page:
        """Return the currently active page. Raises if not running."""
        if not self._running or not self._pages:
            raise RuntimeError("Browser is not running. Call BrowserAgent.start() first.")
        # Refresh from context in case Playwright closed/added pages
        if self._context is not None:
            ctx_pages = list(self._context.pages)
            if ctx_pages:
                self._pages = ctx_pages
                self._active_page_idx = min(
                    self._active_page_idx, len(self._pages) - 1
                )
        return self._pages[self._active_page_idx]

    async def _page_info(self, page: Page, tab_id: int = 0) -> PageInfo:
        """Build a PageInfo from a Playwright Page object."""
        try:
            title = await page.title()
        except Exception:  # noqa: BLE001
            title = ""
        return PageInfo(title=title, url=page.url, tab_id=tab_id)

    # ── Navigation ────────────────────────────────────────────────────────

    async def open_url(self, url: str, new_tab: bool = False) -> PageInfo:
        if not url.startswith(("http://", "https://", "file://")):
            url = "https://" + url

        t0 = time.monotonic()
        if new_tab and self._context is not None:
            page = await self._context.new_page()
            self._pages.append(page)
            self._active_page_idx = len(self._pages) - 1
        else:
            page = self._active_page()

        try:
            await page.goto(url, timeout=self._page_load_timeout, wait_until="domcontentloaded")
        except Exception as exc:  # noqa: BLE001
            log.warning("open_url: navigation error for '{url}': {exc}", url=url, exc=exc)

        load_ms = int((time.monotonic() - t0) * 1000)
        title = ""
        try:
            title = await page.title()
        except Exception:  # noqa: BLE001
            pass
        return PageInfo(title=title, url=page.url, tab_id=self._active_page_idx, load_time_ms=load_ms)

    async def close_tab(self, tab_id: int) -> bool:
        if not self._pages or tab_id >= len(self._pages):
            return False
        try:
            page = self._pages[tab_id]
            await page.close()
            self._pages.pop(tab_id)
            # Adjust active index
            if self._pages:
                self._active_page_idx = max(0, min(self._active_page_idx, len(self._pages) - 1))
            else:
                self._active_page_idx = 0
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("close_tab({tab_id}) error: {exc}", tab_id=tab_id, exc=exc)
            return False

    async def get_current_page_info(self) -> PageInfo:
        page = self._active_page()
        return await self._page_info(page, tab_id=self._active_page_idx)

    async def get_all_tabs(self) -> list[TabInfo]:
        tabs: list[TabInfo] = []
        for idx, page in enumerate(self._pages):
            title = ""
            try:
                title = await page.title()
            except Exception:  # noqa: BLE001
                pass
            tabs.append(TabInfo(
                tab_id=idx,
                title=title,
                url=page.url,
                is_active=(idx == self._active_page_idx),
            ))
        return tabs

    async def navigate_back(self) -> PageInfo:
        page = self._active_page()
        try:
            await page.go_back(timeout=self._nav_timeout, wait_until="domcontentloaded")
        except Exception as exc:  # noqa: BLE001
            log.debug("navigate_back error: {exc}", exc=exc)
        return await self._page_info(page, tab_id=self._active_page_idx)

    async def navigate_forward(self) -> PageInfo:
        page = self._active_page()
        try:
            await page.go_forward(timeout=self._nav_timeout, wait_until="domcontentloaded")
        except Exception as exc:  # noqa: BLE001
            log.debug("navigate_forward error: {exc}", exc=exc)
        return await self._page_info(page, tab_id=self._active_page_idx)

    async def refresh(self) -> PageInfo:
        page = self._active_page()
        try:
            await page.reload(timeout=self._nav_timeout, wait_until="domcontentloaded")
        except Exception as exc:  # noqa: BLE001
            log.debug("refresh error: {exc}", exc=exc)
        return await self._page_info(page, tab_id=self._active_page_idx)

    # ── Content ───────────────────────────────────────────────────────────

    async def read_page_text(self, max_chars: int = 5000) -> str:
        page = self._active_page()
        try:
            text = await page.inner_text("body")
            # Collapse excessive whitespace
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            text = "\n".join(lines)
            if len(text) > max_chars:
                text = text[:max_chars] + "\n[...truncated]"
            return text
        except Exception as exc:  # noqa: BLE001
            log.warning("read_page_text error: {exc}", exc=exc)
            return ""

    # ── Downloads ─────────────────────────────────────────────────────────

    async def download_file(self, url: str, dest_dir: str) -> DownloadResult:
        dest = Path(dest_dir).expanduser()
        dest.mkdir(parents=True, exist_ok=True)

        page = self._active_page()
        try:
            async with page.expect_download(timeout=60_000) as dl_info:
                # Navigate to the download URL — Playwright intercepts the download
                await page.goto(url, timeout=self._page_load_timeout)
            download = await dl_info.value
            filename = download.suggested_filename or _filename_from_url(url)
            save_path = dest / filename
            await download.save_as(str(save_path))
            size = save_path.stat().st_size if save_path.exists() else 0
            return DownloadResult(
                filename=filename,
                path=str(save_path),
                size_bytes=size,
                success=True,
            )
        except asyncio.TimeoutError:
            return DownloadResult(filename="", path="", success=False,
                                  error="Download timed out.")
        except Exception as exc:  # noqa: BLE001
            return DownloadResult(filename="", path="", success=False, error=str(exc))

    # ── History ───────────────────────────────────────────────────────────

    async def get_browser_history(self, limit: int = 20) -> list[HistoryEntry]:
        """
        Attempt to retrieve Chrome history from the user profile SQLite DB.

        This only works for Chromium/Chrome when the database is not locked.
        Returns empty list on failure (browser running → DB locked).
        """
        try:
            import sqlite3
            chrome_history_path = _find_chrome_history()
            if not chrome_history_path or not chrome_history_path.exists():
                return []

            # Copy to temp file to avoid lock issues
            import shutil, tempfile
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            shutil.copy2(chrome_history_path, tmp_path)

            entries: list[HistoryEntry] = []
            conn = sqlite3.connect(str(tmp_path))
            try:
                cursor = conn.execute(
                    "SELECT title, url, last_visit_time FROM urls "
                    "ORDER BY last_visit_time DESC LIMIT ?",
                    (limit,)
                )
                for row in cursor.fetchall():
                    title, url, ts = row
                    # Chrome timestamp: microseconds since 1601-01-01
                    # Convert to ISO string (approximate)
                    from datetime import datetime, timedelta
                    chrome_epoch = datetime(1601, 1, 1)
                    visit_dt = chrome_epoch + timedelta(microseconds=ts)
                    entries.append(HistoryEntry(
                        title=title or "",
                        url=url or "",
                        visit_time=visit_dt.isoformat(),
                    ))
            finally:
                conn.close()
                tmp_path.unlink(missing_ok=True)
            return entries
        except Exception as exc:  # noqa: BLE001
            log.debug("get_browser_history failed: {exc}", exc=exc)
            return []


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _filename_from_url(url: str) -> str:
    """Derive a safe filename from a URL path."""
    try:
        path = urlparse(url).path
        name = path.rstrip("/").split("/")[-1]
        return name if name else "download"
    except Exception:  # noqa: BLE001
        return "download"


def _find_chrome_history() -> Path | None:
    """Locate the Chrome History SQLite file on the current platform."""
    import platform
    system = platform.system()
    if system == "Windows":
        local_app = os.environ.get("LOCALAPPDATA", "")
        candidates = [
            Path(local_app) / "Google" / "Chrome" / "User Data" / "Default" / "History",
            Path(local_app) / "Microsoft" / "Edge" / "User Data" / "Default" / "History",
        ]
    elif system == "Darwin":
        home = Path.home()
        candidates = [
            home / "Library" / "Application Support" / "Google" / "Chrome" / "Default" / "History",
        ]
    else:  # Linux
        home = Path.home()
        candidates = [
            home / ".config" / "google-chrome" / "Default" / "History",
            home / ".config" / "chromium" / "Default" / "History",
        ]

    for p in candidates:
        if p.exists():
            return p
    return None
