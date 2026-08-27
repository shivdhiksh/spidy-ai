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
from spidy.browser.types import (
    BROWSER_PLAYWRIGHT_CHANNELS,
    BROWSER_PROCESS_MAP,
    DownloadResult,
    HistoryEntry,
    PageInfo,
    TabInfo,
    normalize_browser_target,
)
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
        self._active_browser_type: str = ""
        self._active_channel: str = ""
        # Stored so _heal() can relaunch with identical configuration
        self._last_start_kwargs: dict = {}

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

        norm_type = normalize_browser_target(browser_type, default="chromium")

        # If already running with the exact same browser type, no-op
        if self.is_running:
            if self._active_browser_type == norm_type:
                log.debug("PlaywrightBackend.start() called when already running {bt} — reusing session.", bt=norm_type)
                return
            log.info("PlaywrightBackend: switching active browser from {old} to {new}.", old=self._active_browser_type, new=norm_type)
            await self.stop()

        self._active_browser_type = norm_type
        self._active_channel = BROWSER_PLAYWRIGHT_CHANNELS.get(norm_type, "")

        # Persist so _heal() can relaunch with identical configuration
        self._last_start_kwargs = {
            "browser_type": norm_type,
            "headless": headless,
            "download_dir": download_dir,
            "cdp_endpoint": cdp_endpoint,
            "connect_to_existing": connect_to_existing,
        }

        self._download_dir = str(Path(download_dir).expanduser())
        os.makedirs(self._download_dir, exist_ok=True)

        self._playwright = await async_playwright().start()

        # Try attaching to an existing session first
        if connect_to_existing and cdp_endpoint and norm_type in ("chromium", "chrome"):
            attached = await self._try_cdp_attach(cdp_endpoint)
            if attached:
                self._running = True
                log.info("PlaywrightBackend: attached to existing Chrome/CDP session.")
                return

        # Launch the target browser
        log.info(
            "[AGENT] Browser target={bt} (channel={ch}, headless={h})",
            bt=norm_type,
            ch=self._active_channel or "default",
            h=headless,
        )

        try:
            if norm_type == "edge":
                self._browser = await self._playwright.chromium.launch(
                    channel="msedge",
                    headless=headless,
                )
            elif norm_type == "chrome":
                self._browser = await self._playwright.chromium.launch(
                    channel="chrome",
                    headless=headless,
                )
            elif norm_type == "firefox":
                self._browser = await self._playwright.firefox.launch(
                    headless=headless,
                )
            elif norm_type == "webkit":
                self._browser = await self._playwright.webkit.launch(
                    headless=headless,
                )
            else:
                self._browser = await self._playwright.chromium.launch(
                    headless=headless,
                )
        except Exception as exc:
            log.error(
                "[AGENT] Failed to launch requested browser '{bt}': {exc}",
                bt=norm_type, exc=exc,
            )
            if norm_type == "edge":
                raise RuntimeError(
                    f"Microsoft Edge could not be launched or controlled: {exc}"
                ) from exc
            elif norm_type == "chrome":
                raise RuntimeError(
                    f"Google Chrome could not be launched or controlled: {exc}"
                ) from exc
            else:
                raise RuntimeError(
                    f"Browser '{norm_type}' could not be launched: {exc}"
                ) from exc

        self._context = await self._browser.new_context(
            accept_downloads=True,
        )
        page = await self._context.new_page()
        self._pages = [page]
        self._active_page_idx = 0
        self._running = True
        log.info(
            "PlaywrightBackend: launched {bt} browser (channel={ch}, headless={h}).",
            bt=norm_type, ch=self._active_channel or "default", h=headless,
        )

    @property
    def active_browser_type(self) -> str:
        """Canonical name of active browser (e.g. 'edge', 'chrome', 'firefox', 'chromium')."""
        return self._active_browser_type

    @property
    def active_channel(self) -> str:
        """Playwright launch channel if used (e.g. 'msedge', 'chrome')."""
        return self._active_channel

    async def verify_process_identity(self) -> tuple[bool, str]:
        """
        Verify that the active browser process genuinely matches the requested browser target.
        Checks both Playwright context/UA and OS process.
        """
        if not self.is_running:
            return False, "Browser is not running"

        expected_type = self._active_browser_type
        if not expected_type:
            return True, "No specific browser target recorded"

        # 1. Page User-Agent inspection
        ua = ""
        try:
            page = self._active_page()
            ua = await page.evaluate("navigator.userAgent")
        except Exception as exc:
            log.debug("verify_process_identity: evaluate UA failed: {exc}", exc=exc)

        # 2. Check expected markers
        if expected_type == "edge":
            if ua and "Edg/" not in ua:
                return False, f"Browser identity mismatch: expected Edge, but User-Agent is '{ua}' (not Microsoft Edge)"
        elif expected_type == "chrome":
            if ua and ("Chrome/" not in ua or "Edg/" in ua):
                return False, f"Browser identity mismatch: expected Chrome, but User-Agent is '{ua}'"
        elif expected_type == "firefox":
            if ua and "Firefox/" not in ua:
                return False, f"Browser identity mismatch: expected Firefox, but User-Agent is '{ua}'"

        # 3. Process check on Windows
        proc_name = BROWSER_PROCESS_MAP.get(expected_type)
        if proc_name and os.name == "nt":
            import subprocess
            try:
                res = subprocess.run(
                    ["tasklist", "/FI", f"IMAGENAME eq {proc_name}", "/NH"],
                    capture_output=True, text=True, timeout=3,
                )
                if proc_name.lower() not in res.stdout.lower():
                    return False, f"Expected browser process '{proc_name}' not found in running processes"
            except Exception as exc:
                log.debug("verify_process_identity: tasklist check error: {exc}", exc=exc)

        return True, f"Verified genuine {expected_type} session (channel={self._active_channel})"

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
        """Return True only when the browser process is still alive.

        Uses a structural check (``browser.is_connected()``) rather than
        trusting the internal ``_running`` flag, which can become stale when
        the user closes the browser window externally or a CDP session dies.
        """
        if not self._running:
            return False
        # For a launched browser, verify the process is still connected.
        if self._browser is not None:
            try:
                if not self._browser.is_connected():
                    # Process died externally — mark stale so _ensure_running
                    # triggers a full relaunch on the next action.
                    self._running = False
                    return False
            except Exception:  # noqa: BLE001
                self._running = False
                return False
        return True

    # ── Active page helpers ───────────────────────────────────────────────

    def _active_page(self) -> Page:
        """Return the currently active page (synchronous, no healing).

        Used only for safe read-only methods that don't call ``goto()``.
        Raises ``RuntimeError`` if not running.
        """
        if not self._running or not self._pages:
            raise RuntimeError("Browser is not running. Call BrowserAgent.start() first.")
        # Refresh from context in case Playwright closed/added pages
        if self._context is not None:
            try:
                ctx_pages = list(self._context.pages)
                if ctx_pages:
                    self._pages = ctx_pages
                    self._active_page_idx = min(
                        self._active_page_idx, len(self._pages) - 1
                    )
            except Exception:  # noqa: BLE001
                pass  # context may be closed; callers handle stale pages
        return self._pages[self._active_page_idx]

    def _is_page_healthy(self, page: "Page") -> bool:
        """Return True when *page* is still open and usable."""
        try:
            return not page.is_closed()
        except Exception:  # noqa: BLE001
            return False

    async def _heal(self) -> None:
        """Tear down stale handles and relaunch the browser.

        Called when the session is detected to be dead mid-operation.
        Uses ``_last_start_kwargs`` so the new session matches the original
        configuration (browser type, headless flag, CDP endpoint, etc.).
        """
        log.warning(
            "PlaywrightBackend._heal(): stale session detected — relaunching browser."
        )
        # Reset all handles without calling stop() — they may already be dead.
        self._browser = None
        self._context = None
        self._pages = []
        self._playwright = None
        self._running = False
        kwargs = self._last_start_kwargs or {}
        await self.start(**kwargs)

    async def _get_or_heal_page(self) -> "Page":
        """Return a healthy, usable Playwright Page — healing automatically if needed.

        Three-level recovery strategy
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        1. **Fast path** — refresh ``_pages`` from the live context and return
           the active page if it is healthy.
        2. **Page closed** — the page was closed externally but the context is
           still alive: open a fresh page in the existing context.
        3. **Context / browser dead** — perform a full ``_heal()`` (clears all
           stale handles, relaunches the browser) and return the new page.
        """
        # ── Level 1: refresh page list from the live context ──────────────
        if self._context is not None:
            try:
                ctx_pages = list(self._context.pages)
                if ctx_pages:
                    self._pages = ctx_pages
                    self._active_page_idx = min(
                        self._active_page_idx, len(self._pages) - 1
                    )
            except Exception:  # noqa: BLE001
                pass  # context closed — will be caught below

        if self._pages:
            page = self._pages[self._active_page_idx]
            if self._is_page_healthy(page):
                return page
            log.warning(
                "PlaywrightBackend._get_or_heal_page(): active page is closed "
                "(tab_id={idx}).",
                idx=self._active_page_idx,
            )

        # ── Level 2: open a fresh page in the existing context ────────────
        if self._context is not None:
            try:
                page = await self._context.new_page()
                self._pages = [page]
                self._active_page_idx = 0
                log.info(
                    "PlaywrightBackend._get_or_heal_page(): "
                    "recovered — new page opened in existing context."
                )
                return page
            except Exception:  # noqa: BLE001
                log.warning(
                    "PlaywrightBackend._get_or_heal_page(): "
                    "context is dead — performing full heal."
                )

        # ── Level 3: full heal (relaunch browser) ─────────────────────────
        await self._heal()
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
            try:
                page = await self._context.new_page()
            except Exception:  # context dead — heal and open in fresh context
                await self._heal()
                page = await self._context.new_page()  # type: ignore[union-attr]
            self._pages.append(page)
            self._active_page_idx = len(self._pages) - 1
        else:
            page = await self._get_or_heal_page()

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
        page = await self._get_or_heal_page()
        try:
            await page.go_back(timeout=self._nav_timeout, wait_until="domcontentloaded")
        except Exception as exc:  # noqa: BLE001
            log.debug("navigate_back error: {exc}", exc=exc)
        return await self._page_info(page, tab_id=self._active_page_idx)

    async def navigate_forward(self) -> PageInfo:
        page = await self._get_or_heal_page()
        try:
            await page.go_forward(timeout=self._nav_timeout, wait_until="domcontentloaded")
        except Exception as exc:  # noqa: BLE001
            log.debug("navigate_forward error: {exc}", exc=exc)
        return await self._page_info(page, tab_id=self._active_page_idx)

    async def refresh(self) -> PageInfo:
        page = await self._get_or_heal_page()
        try:
            await page.reload(timeout=self._nav_timeout, wait_until="domcontentloaded")
        except Exception as exc:  # noqa: BLE001
            log.debug("refresh error: {exc}", exc=exc)
        return await self._page_info(page, tab_id=self._active_page_idx)

    # ── Content ───────────────────────────────────────────────────────────

    async def read_page_text(self, max_chars: int = 5000) -> str:
        page = await self._get_or_heal_page()
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

        page = await self._get_or_heal_page()
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
