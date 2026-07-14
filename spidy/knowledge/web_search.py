"""
Knowledge Engine — Web Search Engine
======================================
Optional web search integration for knowledge retrieval.

Design
------
- Feature-flagged: disabled by default (``knowledge.web_search.enabled: false``).
- Provider abstraction: DuckDuckGoProvider is the default (no API key required).
- Graceful degrade: if ``duckduckgo-search`` (ddgs) is not installed,
  ``is_available`` returns False and search returns an empty list.
- Privacy: no search history is stored; results are kept only in the
  current working context and discarded after the session.
- Results are returned as KnowledgeChunks with ``source_type="web_search"``.

Dependencies (optional)
-----------------------
    duckduckgo-search>=6.0    (add to pyproject.toml[knowledge])
    Imported as: from duckduckgo_search import DDGS
"""

from __future__ import annotations

import asyncio
from typing import Any

from spidy.knowledge.types import KnowledgeChunk
from spidy.logging.logger import get_logger

log = get_logger(__name__)


# ─── Base Provider ────────────────────────────────────────────────────────────


class BaseSearchProvider:
    """Abstract base for web search providers."""

    provider_name: str = "unknown"

    @property
    def is_available(self) -> bool:
        """True if required deps are installed."""
        return False

    def search_sync(
        self,
        query: str,
        max_results: int = 5,
        safe_search: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Run a web search and return raw result dicts.

        Returns
        -------
        list[dict]
            Each dict has ``{"title": str, "href": str, "body": str}``.
        """
        return []


# ─── DuckDuckGo Provider ──────────────────────────────────────────────────────


class DuckDuckGoProvider(BaseSearchProvider):
    """
    DuckDuckGo web search via the ``duckduckgo-search`` library (ddgs).

    No API key required. Privacy-respecting by design.

    Install: ``pip install duckduckgo-search``
    """

    provider_name = "duckduckgo"

    @property
    def is_available(self) -> bool:
        try:
            from duckduckgo_search import DDGS  # noqa: F401
            return True
        except ImportError:
            return False

    def search_sync(
        self,
        query: str,
        max_results: int = 5,
        safe_search: bool = True,
    ) -> list[dict[str, Any]]:
        if not self.is_available:
            return []
        try:
            from duckduckgo_search import DDGS
            safesearch = "moderate" if safe_search else "off"
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        query,
                        max_results=max_results,
                        safesearch=safesearch,
                    )
                )
            return results
        except Exception as exc:
            log.warning(
                "DuckDuckGoProvider: search failed: {exc}", exc=exc
            )
            return []


# ─── WebSearchEngine ─────────────────────────────────────────────────────────


class WebSearchEngine:
    """
    Feature-flagged web search for knowledge augmentation.

    Wraps a ``BaseSearchProvider`` and converts raw results to
    ``KnowledgeChunk`` objects for uniform handling.

    Parameters
    ----------
    provider:
        Search provider instance. Defaults to ``DuckDuckGoProvider``.
    enabled:
        Master feature flag. When False, all searches return empty results
        immediately without touching the network.
    max_results:
        Default maximum number of results per query.
    safe_search:
        Whether to apply safe-search filtering.
    """

    def __init__(
        self,
        provider: BaseSearchProvider | None = None,
        enabled: bool = False,
        max_results: int = 5,
        safe_search: bool = True,
    ) -> None:
        self._provider = provider or DuckDuckGoProvider()
        self._enabled = enabled
        self._max_results = max_results
        self._safe_search = safe_search

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """
        True if web search is both enabled AND the provider dep is installed.
        """
        return self._enabled and self._provider.is_available

    @property
    def is_enabled(self) -> bool:
        """True if the feature flag is set (regardless of dep availability)."""
        return self._enabled

    @property
    def provider_name(self) -> str:
        return self._provider.provider_name

    # ── Search ────────────────────────────────────────────────────────────

    def search_sync(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[KnowledgeChunk]:
        """
        Run a web search synchronously and return KnowledgeChunks.

        Returns an empty list if not available or disabled.
        """
        if not self.is_available:
            return []

        n = max_results or self._max_results
        raw_results = self._provider.search_sync(
            query=query,
            max_results=n,
            safe_search=self._safe_search,
        )

        return [
            self._result_to_chunk(r, idx, len(raw_results))
            for idx, r in enumerate(raw_results)
            if r.get("body")
        ]

    async def search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[KnowledgeChunk]:
        """
        Async version of ``search_sync`` — runs in thread executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.search_sync, query, max_results
        )

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _result_to_chunk(
        result: dict[str, Any],
        index: int,
        total: int,
    ) -> KnowledgeChunk:
        """Convert a raw search result dict to a KnowledgeChunk."""
        title = result.get("title", "Web Result")
        url = result.get("href", result.get("url", ""))
        body = result.get("body", "")

        # Score placeholder — web results don't have similarity scores
        score = max(0.0, 1.0 - (index * 0.1))  # simple rank-decay

        return KnowledgeChunk.create(
            content=f"{title}\n\n{body}".strip(),
            source=url,
            source_type="web_search",
            chunk_index=index,
            total_chunks=total,
            metadata={
                "title": title,
                "url": url,
                "provider": "duckduckgo",
            },
            tags=["web_search"],
            score=score,
        )
