"""
BrowserSkill — Browser Agent Skill
====================================
Provides 14 browser automation capabilities through the existing
SkillRegistry / SkillExecutor / PermissionManager pipeline.

Permission Tiers
----------------
T0 — Read-only (no system changes):
    get_page_info, list_tabs, read_page

T1 — Reversible, low-risk actions:
    open_browser, open_url, open_new_tab, navigate_back,
    navigate_forward, refresh_page, search_google, search_youtube

T2 — Potentially disruptive / writes to disk:
    close_browser, close_tab, download_file

Architecture
------------
BrowserSkill owns ONE BrowserAgent instance (lazy-init on first call).
The agent is created the first time any T1/T2 action runs; T0 actions
on a non-running browser return a meaningful "not running" message.

All browser output flows through the EventBus (``browser.*`` topics).
The Brain never imports Playwright or BrowserAgent directly.

``on_unload()`` is called at application shutdown and stops the browser.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger
from spidy.skills.base import (
    BaseSkill,
    ParamSchema,
    SkillCapability,
    SkillContext,
    SkillResult,
)

if TYPE_CHECKING:
    from spidy.browser.agent import BrowserAgent
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)


class BrowserSkill(BaseSkill):
    """
    Browser automation skill — the only skill that touches BrowserAgent.

    Parameters
    ----------
    bus:
        EventBus for publishing browser events. May be None.
    browser_type:
        Browser engine: ``"chromium"`` | ``"firefox"`` | ``"webkit"``.
    headless:
        Run without a visible window.
    download_dir:
        Default download destination.
    connect_to_existing:
        Try CDP attach before launching a new browser.
    cdp_endpoint:
        Chrome DevTools Protocol endpoint for CDP attach.
    page_load_timeout_ms:
        Page load timeout in milliseconds.
    navigation_timeout_ms:
        Navigation operation timeout in milliseconds.
    read_page_max_chars:
        Maximum characters for read_page action.
    history_limit:
        Maximum history entries for get_browser_history.
    """

    name = "browser_skill"
    version = "1.0.0"

    def __init__(
        self,
        bus: "EventBus | None" = None,
        browser_type: str = "chromium",
        headless: bool = False,
        download_dir: str = "~",
        connect_to_existing: bool = True,
        cdp_endpoint: str = "http://localhost:9222",
        page_load_timeout_ms: int = 30_000,
        navigation_timeout_ms: int = 10_000,
        read_page_max_chars: int = 5000,
        history_limit: int = 20,
    ) -> None:
        self._bus = bus
        self._browser_type = browser_type
        self._headless = headless
        self._download_dir = download_dir
        self._connect_to_existing = connect_to_existing
        self._cdp_endpoint = cdp_endpoint
        self._page_load_timeout_ms = page_load_timeout_ms
        self._navigation_timeout_ms = navigation_timeout_ms
        self._read_page_max_chars = read_page_max_chars
        self._history_limit = history_limit
        self._agent: "BrowserAgent | None" = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def _get_or_create_agent(self) -> "BrowserAgent":
        """Lazily create the BrowserAgent on first use."""
        if self._agent is None:
            from spidy.browser.agent import BrowserAgent
            self._agent = BrowserAgent(
                browser_type=self._browser_type,
                headless=self._headless,
                download_dir=self._download_dir,
                connect_to_existing=self._connect_to_existing,
                cdp_endpoint=self._cdp_endpoint,
                page_load_timeout_ms=self._page_load_timeout_ms,
                navigation_timeout_ms=self._navigation_timeout_ms,
                read_page_max_chars=self._read_page_max_chars,
            )
        return self._agent

    async def on_unload(self) -> None:
        """Stop the browser gracefully when the skill is unregistered."""
        if self._agent is not None and self._agent.is_running:
            log.info("BrowserSkill.on_unload(): stopping browser agent.")
            await self._agent.stop()
            self._agent = None

    # ── Capabilities ──────────────────────────────────────────────────────

    def capabilities(self) -> list[SkillCapability]:
        return [
            # ── T0 — Read-only ─────────────────────────────────────────
            SkillCapability(
                action="get_page_info",
                description=(
                    "Get the title and URL of the currently active browser page."
                ),
                permission_tier="T0",
                params=[],
                examples=[
                    "what page is open", "what website am I on",
                    "what's the current URL", "read the page title",
                ],
            ),
            SkillCapability(
                action="list_tabs",
                description="List all currently open browser tabs with their titles and URLs.",
                permission_tier="T0",
                params=[],
                examples=[
                    "show all tabs", "what tabs are open",
                    "list open browser tabs",
                ],
            ),
            SkillCapability(
                action="read_page",
                description=(
                    "Extract and return the visible text content of the current browser page."
                ),
                permission_tier="T0",
                params=[
                    ParamSchema("max_chars", "int", required=False, default=5000,
                                description="Maximum characters to return."),
                ],
                examples=[
                    "read the page", "what does this page say",
                    "extract text from page", "summarise this webpage",
                ],
            ),
            # ── T1 — Reversible actions ────────────────────────────────
            SkillCapability(
                action="open_browser",
                description="Launch the browser or ensure it is open.",
                permission_tier="T1",
                params=[
                    ParamSchema("url", "string", required=False,
                                description="Optional URL to open immediately after launching."),
                ],
                examples=[
                    "open browser", "launch Chrome", "start the browser",
                    "open browser and go to google",
                ],
            ),
            SkillCapability(
                action="open_url",
                description="Open a URL in the current browser tab.",
                permission_tier="T1",
                params=[
                    ParamSchema("url", "string", required=True,
                                description="The URL to navigate to."),
                ],
                examples=[
                    "open google.com", "go to youtube.com",
                    "navigate to https://github.com",
                    "open the BBC website",
                ],
            ),
            SkillCapability(
                action="open_new_tab",
                description="Open a URL in a new browser tab.",
                permission_tier="T1",
                params=[
                    ParamSchema("url", "string", required=True,
                                description="The URL to open in a new tab."),
                ],
                examples=[
                    "open in new tab", "new tab google.com",
                    "open github in a new tab",
                ],
            ),
            SkillCapability(
                action="navigate_back",
                description="Navigate back to the previous page in browser history.",
                permission_tier="T1",
                params=[],
                examples=[
                    "go back", "previous page", "browser back",
                    "navigate back",
                ],
            ),
            SkillCapability(
                action="navigate_forward",
                description="Navigate forward in browser history.",
                permission_tier="T1",
                params=[],
                examples=[
                    "go forward", "next page", "browser forward",
                ],
            ),
            SkillCapability(
                action="refresh_page",
                description="Reload the current browser page.",
                permission_tier="T1",
                params=[],
                examples=[
                    "refresh", "reload page", "refresh the browser",
                    "F5", "reload this page",
                ],
            ),
            SkillCapability(
                action="search_google",
                description="Search Google for a query and open the search results page.",
                permission_tier="T1",
                params=[
                    ParamSchema("query", "string", required=True,
                                description="The search query."),
                    ParamSchema("new_tab", "bool", required=False, default=False,
                                description="Open search in a new tab."),
                ],
                examples=[
                    "search google for Python tutorials",
                    "google the weather in London",
                    "look up quantum computing on Google",
                    "search for best pizza recipes",
                ],
            ),
            SkillCapability(
                action="search_youtube",
                description="Search YouTube for a query and open the search results page.",
                permission_tier="T1",
                params=[
                    ParamSchema("query", "string", required=True,
                                description="The YouTube search query."),
                    ParamSchema("new_tab", "bool", required=False, default=False,
                                description="Open search in a new tab."),
                ],
                examples=[
                    "search youtube for lofi music",
                    "find a tutorial on youtube",
                    "youtube search for cooking videos",
                    "play lofi on youtube",
                ],
            ),
            # ── T2 — Potentially disruptive ────────────────────────────
            SkillCapability(
                action="close_browser",
                description="Close the browser session and all open tabs.",
                permission_tier="T2",
                params=[],
                examples=[
                    "close the browser", "shut down Chrome",
                    "close all browser tabs",
                ],
            ),
            SkillCapability(
                action="close_tab",
                description="Close a specific browser tab by its ID.",
                permission_tier="T2",
                params=[
                    ParamSchema("tab_id", "int", required=True,
                                description="Tab ID to close (from list_tabs)."),
                ],
                examples=[
                    "close tab 0", "close this tab", "close the second tab",
                ],
            ),
            SkillCapability(
                action="download_file",
                description="Download a file from a URL to the local filesystem.",
                permission_tier="T2",
                params=[
                    ParamSchema("url", "string", required=True,
                                description="Direct download URL."),
                    ParamSchema("dest_dir", "path", required=False,
                                description="Destination directory. Defaults to Downloads."),
                ],
                examples=[
                    "download this file", "save this PDF",
                    "download the file at https://example.com/report.pdf",
                ],
            ),
        ]

    # ── Dispatch ──────────────────────────────────────────────────────────

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        dispatch = {
            "get_page_info":      self._get_page_info,
            "list_tabs":          self._list_tabs,
            "read_page":          self._read_page,
            "open_browser":       self._open_browser,
            "open_url":           self._open_url,
            "open_new_tab":       self._open_new_tab,
            "navigate_back":      self._navigate_back,
            "navigate_forward":   self._navigate_forward,
            "refresh_page":       self._refresh_page,
            "search_google":      self._search_google,
            "search_youtube":     self._search_youtube,
            "close_browser":      self._close_browser,
            "close_tab":          self._close_tab,
            "download_file":      self._download_file,
        }
        handler = dispatch.get(action)
        if handler is None:
            return SkillResult.fail(f"BrowserSkill: unknown action '{action}'.")
        try:
            return await handler(context)
        except Exception as exc:  # noqa: BLE001
            log.error("BrowserSkill.execute('{action}'): {exc}", action=action, exc=exc)
            await self._emit_error(action, str(exc), context.session_id)
            return SkillResult.fail(
                f"Browser error during '{action}': {exc}", error=exc
            )

    # ── T0 Actions ────────────────────────────────────────────────────────

    async def _get_page_info(self, context: SkillContext) -> SkillResult:
        agent = self._get_or_create_agent()
        if not agent.is_running:
            return SkillResult.ok(
                "Browser is not open. Say 'open browser' to start it.",
                data={"running": False},
            )
        info = await agent.get_current_page_info()
        return SkillResult.ok(
            f"Current page: \"{info.title}\" — {info.url}",
            data=info.to_dict(),
            action_taken="get_page_info",
        )

    async def _list_tabs(self, context: SkillContext) -> SkillResult:
        agent = self._get_or_create_agent()
        if not agent.is_running:
            return SkillResult.ok(
                "Browser is not open.",
                data={"tabs": [], "running": False},
            )
        tabs = await agent.get_all_tabs()
        if not tabs:
            return SkillResult.ok("No open tabs.", data={"tabs": []})
        lines = [
            f"  Tab {t.tab_id}: \"{t.title}\" — {t.url}"
            + (" [active]" if t.is_active else "")
            for t in tabs
        ]
        return SkillResult.ok(
            "Open tabs:\n" + "\n".join(lines),
            data={"tabs": [t.to_dict() for t in tabs]},
            action_taken="list_tabs",
        )

    async def _read_page(self, context: SkillContext) -> SkillResult:
        agent = self._get_or_create_agent()
        if not agent.is_running:
            return SkillResult.fail("Browser is not open. Open a URL first.")
        max_chars = int(context.get("max_chars") or self._read_page_max_chars)
        text = await agent.read_page_text(max_chars=max_chars)
        if not text.strip():
            return SkillResult.ok("The page appears to be empty or unreadable.", data={"text": ""})
        info = await agent.get_current_page_info()
        await self._emit(
            "browser.page_read",
            lambda: __import__(
                "spidy.skills.browser.events", fromlist=["PageReadEvent"]
            ).PageReadEvent(
                url=info.url,
                title=info.title,
                char_count=len(text),
                session_id=context.session_id,
            ),
        )
        return SkillResult.ok(
            f"Page text ({len(text)} chars):\n{text}",
            data={"text": text, "url": info.url, "title": info.title},
            action_taken="read_page",
        )

    # ── T1 Actions ────────────────────────────────────────────────────────

    async def _open_browser(self, context: SkillContext) -> SkillResult:
        agent = self._get_or_create_agent()
        if agent.is_running:
            # Just open an optional URL
            url = context.get("url", "")
            if url:
                return await self._do_open_url(agent, url, new_tab=False, context=context)
            return SkillResult.ok("Browser is already open.", data={"running": True})

        await agent.start()
        await self._emit_event_obj(
            __import__(
                "spidy.skills.browser.events", fromlist=["BrowserLaunchedEvent"]
            ).BrowserLaunchedEvent(
                browser_type=self._browser_type,
                headless=self._headless,
                session_id=context.session_id,
            )
        )

        url = context.get("url", "")
        if url:
            return await self._do_open_url(agent, url, new_tab=False, context=context)
        return SkillResult.ok(
            f"Browser launched ({self._browser_type}).",
            data={"browser_type": self._browser_type, "running": True},
            action_taken="open_browser",
        )

    async def _open_url(self, context: SkillContext) -> SkillResult:
        url = context.get("url", "")
        if not url:
            return SkillResult.fail("BrowserSkill.open_url: 'url' parameter is required.")
        agent = self._get_or_create_agent()
        return await self._do_open_url(agent, url, new_tab=False, context=context)

    async def _open_new_tab(self, context: SkillContext) -> SkillResult:
        url = context.get("url", "")
        if not url:
            return SkillResult.fail("BrowserSkill.open_new_tab: 'url' parameter is required.")
        agent = self._get_or_create_agent()
        return await self._do_open_url(agent, url, new_tab=True, context=context)

    async def _do_open_url(
        self, agent: "BrowserAgent", url: str, new_tab: bool, context: SkillContext
    ) -> SkillResult:
        info = await agent.open_url(url, new_tab=new_tab)
        from spidy.skills.browser.events import PageNavigatedEvent
        await self._emit_event_obj(PageNavigatedEvent(
            title=info.title,
            url=info.url,
            tab_id=info.tab_id,
            new_tab=new_tab,
            load_time_ms=info.load_time_ms,
            session_id=context.session_id,
        ))
        tab_word = "new tab" if new_tab else "tab"
        return SkillResult.ok(
            f"Opened \"{info.title}\" in {tab_word}. URL: {info.url}",
            data=info.to_dict(),
            action_taken="open_url",
        )

    async def _navigate_back(self, context: SkillContext) -> SkillResult:
        agent = self._get_or_create_agent()
        if not agent.is_running:
            return SkillResult.fail("Browser is not open.")
        info = await agent.navigate_back()
        from spidy.skills.browser.events import NavigationBackEvent
        await self._emit_event_obj(NavigationBackEvent(
            title=info.title, url=info.url, tab_id=info.tab_id,
            session_id=context.session_id,
        ))
        return SkillResult.ok(
            f"Navigated back to \"{info.title}\" — {info.url}",
            data=info.to_dict(),
            action_taken="navigate_back",
        )

    async def _navigate_forward(self, context: SkillContext) -> SkillResult:
        agent = self._get_or_create_agent()
        if not agent.is_running:
            return SkillResult.fail("Browser is not open.")
        info = await agent.navigate_forward()
        from spidy.skills.browser.events import NavigationForwardEvent
        await self._emit_event_obj(NavigationForwardEvent(
            title=info.title, url=info.url, tab_id=info.tab_id,
            session_id=context.session_id,
        ))
        return SkillResult.ok(
            f"Navigated forward to \"{info.title}\" — {info.url}",
            data=info.to_dict(),
            action_taken="navigate_forward",
        )

    async def _refresh_page(self, context: SkillContext) -> SkillResult:
        agent = self._get_or_create_agent()
        if not agent.is_running:
            return SkillResult.fail("Browser is not open.")
        info = await agent.refresh()
        from spidy.skills.browser.events import PageRefreshedEvent
        await self._emit_event_obj(PageRefreshedEvent(
            title=info.title, url=info.url, tab_id=info.tab_id,
            session_id=context.session_id,
        ))
        return SkillResult.ok(
            f"Page refreshed: \"{info.title}\" — {info.url}",
            data=info.to_dict(),
            action_taken="refresh_page",
        )

    async def _search_google(self, context: SkillContext) -> SkillResult:
        query = context.get("query", "")
        if not query:
            return SkillResult.fail("BrowserSkill.search_google: 'query' parameter is required.")
        new_tab = bool(context.get("new_tab", False))
        agent = self._get_or_create_agent()
        info = await agent.search_google(query, new_tab=new_tab)
        from spidy.skills.browser.events import SearchResultsEvent, PageNavigatedEvent
        await self._emit_event_obj(SearchResultsEvent(
            engine="google", query=query, results_url=info.url,
            session_id=context.session_id,
        ))
        await self._emit_event_obj(PageNavigatedEvent(
            title=info.title, url=info.url, tab_id=info.tab_id,
            new_tab=new_tab, load_time_ms=info.load_time_ms,
            session_id=context.session_id,
        ))
        return SkillResult.ok(
            f"Searched Google for \"{query}\". Page: {info.url}",
            data={"query": query, "engine": "google", "page": info.to_dict()},
            action_taken="search_google",
        )

    async def _search_youtube(self, context: SkillContext) -> SkillResult:
        query = context.get("query", "")
        if not query:
            return SkillResult.fail("BrowserSkill.search_youtube: 'query' parameter is required.")
        new_tab = bool(context.get("new_tab", False))
        agent = self._get_or_create_agent()
        info = await agent.search_youtube(query, new_tab=new_tab)
        from spidy.skills.browser.events import SearchResultsEvent, PageNavigatedEvent
        await self._emit_event_obj(SearchResultsEvent(
            engine="youtube", query=query, results_url=info.url,
            session_id=context.session_id,
        ))
        await self._emit_event_obj(PageNavigatedEvent(
            title=info.title, url=info.url, tab_id=info.tab_id,
            new_tab=new_tab, load_time_ms=info.load_time_ms,
            session_id=context.session_id,
        ))
        return SkillResult.ok(
            f"Searched YouTube for \"{query}\". Page: {info.url}",
            data={"query": query, "engine": "youtube", "page": info.to_dict()},
            action_taken="search_youtube",
        )

    # ── T2 Actions ────────────────────────────────────────────────────────

    async def _close_browser(self, context: SkillContext) -> SkillResult:
        agent = self._get_or_create_agent()
        if not agent.is_running:
            return SkillResult.ok("Browser is not open.")
        await agent.stop()
        from spidy.skills.browser.events import BrowserClosedEvent
        await self._emit_event_obj(BrowserClosedEvent(session_id=context.session_id))
        return SkillResult.ok(
            "Browser closed.",
            data={"running": False},
            action_taken="close_browser",
        )

    async def _close_tab(self, context: SkillContext) -> SkillResult:
        tab_id_raw = context.get("tab_id")
        if tab_id_raw is None:
            return SkillResult.fail("BrowserSkill.close_tab: 'tab_id' parameter is required.")
        tab_id = int(tab_id_raw)
        agent = self._get_or_create_agent()
        if not agent.is_running:
            return SkillResult.fail("Browser is not open.")
        closed = await agent.close_tab(tab_id)
        if closed:
            from spidy.skills.browser.events import TabClosedEvent
            await self._emit_event_obj(TabClosedEvent(
                tab_id=tab_id, session_id=context.session_id
            ))
            return SkillResult.ok(
                f"Tab {tab_id} closed.",
                data={"tab_id": tab_id, "closed": True},
                action_taken="close_tab",
            )
        return SkillResult.fail(f"Tab {tab_id} not found.")

    async def _download_file(self, context: SkillContext) -> SkillResult:
        url = context.get("url", "")
        if not url:
            return SkillResult.fail("BrowserSkill.download_file: 'url' parameter is required.")
        dest_dir = context.get("dest_dir") or self._download_dir
        agent = self._get_or_create_agent()

        from spidy.skills.browser.events import DownloadStartedEvent, DownloadCompletedEvent
        await self._emit_event_obj(DownloadStartedEvent(
            url=url, dest_dir=str(dest_dir), session_id=context.session_id
        ))

        result = await agent.download_file(url, dest_dir=str(dest_dir))

        await self._emit_event_obj(DownloadCompletedEvent(
            filename=result.filename,
            path=result.path,
            size_bytes=result.size_bytes,
            success=result.success,
            error=result.error,
            session_id=context.session_id,
        ))

        if result.success:
            return SkillResult.ok(
                f"Downloaded \"{result.filename}\" ({result.size_bytes:,} bytes) → {result.path}",
                data=result.to_dict(),
                action_taken="download_file",
            )
        return SkillResult.fail(
            f"Download failed: {result.error}",
            error=RuntimeError(result.error),
        )

    # ── EventBus Helpers ──────────────────────────────────────────────────

    async def _emit_event_obj(self, event: object) -> None:
        """Publish a pre-built Event object to the bus."""
        if self._bus is not None:
            try:
                from spidy.core.event_bus import Event
                if isinstance(event, Event):
                    await self._bus.publish(event)
            except Exception as exc:  # noqa: BLE001
                log.warning("BrowserSkill: event publish failed: {exc}", exc=exc)

    async def _emit_error(self, action: str, error: str, session_id: str) -> None:
        """Publish a BrowserErrorEvent to the bus."""
        if self._bus is not None:
            try:
                from spidy.skills.browser.events import BrowserErrorEvent
                await self._bus.publish(BrowserErrorEvent(
                    action=action, error=error, session_id=session_id
                ))
            except Exception as exc:  # noqa: BLE001
                log.warning("BrowserSkill._emit_error: {exc}", exc=exc)

    async def _emit(self, topic: str, build_event) -> None:
        """Publish an event built by a factory callable."""
        if self._bus is not None:
            try:
                await self._bus.publish(build_event())
            except Exception as exc:  # noqa: BLE001
                log.warning("BrowserSkill._emit({topic}): {exc}", topic=topic, exc=exc)
