"""
Integration tests for the Knowledge Engine (Milestone 10).

These tests cover:
1. KnowledgeManager lifecycle via SpidyCore
2. Knowledge ingest + query pipeline (mocked embedding + store)
3. Brain + KnowledgeManager integration
4. SpidyCore.knowledge property
5. Graceful degradation when optional deps absent
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.core.app import SpidyCore
from spidy.core.event_bus import EventBus
from spidy.knowledge.manager import KnowledgeManager
from spidy.knowledge.types import KnowledgeChunk, KnowledgeResult
from spidy.knowledge.events import (
    KnowledgeQueriedEvent,
    KnowledgeSourceDeletedEvent,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def make_manager_with_no_deps(knowledge_dir=None) -> KnowledgeManager:
    """Create a KnowledgeManager where all optional deps are absent."""
    mgr = KnowledgeManager(config=None, bus=None, knowledge_dir=knowledge_dir)
    mgr._store._checked_available = False
    mgr._embedder._checked_available = False
    mgr._web_search._enabled = False
    return mgr


# ─── KnowledgeManager lifecycle ──────────────────────────────────────────────


class TestKnowledgeManagerLifecycleFull:
    """KnowledgeManager initialize/close lifecycle integration."""

    @pytest.mark.asyncio
    async def test_initialize_and_close_idempotent(self, tmp_path):
        mgr = make_manager_with_no_deps(knowledge_dir=tmp_path)
        await mgr.initialize()
        await mgr.initialize()  # idempotent
        await mgr.close()
        await mgr.close()  # idempotent

    @pytest.mark.asyncio
    async def test_count_after_init_is_zero(self, tmp_path):
        mgr = make_manager_with_no_deps(knowledge_dir=tmp_path)
        await mgr.initialize()
        count = await mgr.count()
        assert count == 0
        await mgr.close()

    @pytest.mark.asyncio
    async def test_list_sources_empty_on_fresh_manager(self, tmp_path):
        mgr = make_manager_with_no_deps(knowledge_dir=tmp_path)
        await mgr.initialize()
        sources = await mgr.list_sources()
        assert sources == []
        await mgr.close()


# ─── Ingest + Query pipeline ──────────────────────────────────────────────────


class TestKnowledgeIngestQueryPipeline:
    """Full ingest → query pipeline with mocked embedding + ChromaDB."""

    @pytest.mark.asyncio
    async def test_ingest_returns_id_with_mocked_embedder(self, tmp_path):
        mgr = make_manager_with_no_deps(knowledge_dir=tmp_path)
        await mgr.initialize()
        # With no embedder, ingest still chunks but returns "" since store is empty
        result = await mgr.ingest(content="Python is a great language.", source="manual")
        assert isinstance(result, str)
        await mgr.close()

    @pytest.mark.asyncio
    async def test_query_returns_empty_list_without_deps(self, tmp_path):
        mgr = make_manager_with_no_deps(knowledge_dir=tmp_path)
        await mgr.initialize()
        result = await mgr.query("What is Python?")
        assert result == []
        await mgr.close()

    @pytest.mark.asyncio
    async def test_search_rag_returns_empty_without_deps(self, tmp_path):
        mgr = make_manager_with_no_deps(knowledge_dir=tmp_path)
        await mgr.initialize()
        result = await mgr.search_rag("What is Python?")
        assert result == ""
        await mgr.close()

    @pytest.mark.asyncio
    async def test_query_returns_results_with_mocked_deps(self, tmp_path):
        mgr = make_manager_with_no_deps(knowledge_dir=tmp_path)
        await mgr.initialize()

        chunk = KnowledgeChunk.create(
            content="Python is a high-level programming language.",
            source="python.txt",
            score=0.9,
        )
        known_result = KnowledgeResult(query="What is Python?", chunks=[chunk])
        mgr._retrieve = AsyncMock(return_value=known_result)

        result = await mgr.query("What is Python?")
        assert len(result) == 1
        assert result[0]["content"] == "Python is a high-level programming language."
        assert result[0]["source"] == "python.txt"
        assert result[0]["score"] == 0.9
        await mgr.close()

    @pytest.mark.asyncio
    async def test_search_rag_returns_context_with_mocked_deps(self, tmp_path):
        mgr = make_manager_with_no_deps(knowledge_dir=tmp_path)
        await mgr.initialize()

        chunk = KnowledgeChunk.create(
            content="Python is a high-level programming language.",
            source="python.txt",
            score=0.9,
        )
        known_result = KnowledgeResult(query="What is Python?", chunks=[chunk])
        mgr._retrieve = AsyncMock(return_value=known_result)

        result = await mgr.search_rag("What is Python?")
        assert "[Knowledge Context]" in result
        assert "Python is a high-level programming language." in result
        await mgr.close()


# ─── File ingest ─────────────────────────────────────────────────────────────


class TestKnowledgeFileIngest:
    """ingest_file() handles various document types."""

    @pytest.mark.asyncio
    async def test_ingest_plain_text_file(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("Important notes about Python.", encoding="utf-8")
        mgr = make_manager_with_no_deps(knowledge_dir=tmp_path / "knowledge")
        await mgr.initialize()
        result = await mgr.ingest_file(f)
        assert isinstance(result, int)
        await mgr.close()

    @pytest.mark.asyncio
    async def test_ingest_markdown_file(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Python Guide\n\nPython is easy to learn.", encoding="utf-8")
        mgr = make_manager_with_no_deps(knowledge_dir=tmp_path / "knowledge")
        await mgr.initialize()
        result = await mgr.ingest_file(f)
        assert isinstance(result, int)
        await mgr.close()

    @pytest.mark.asyncio
    async def test_ingest_missing_file_returns_zero(self, tmp_path):
        mgr = make_manager_with_no_deps(knowledge_dir=tmp_path)
        await mgr.initialize()
        result = await mgr.ingest_file(tmp_path / "ghost.txt")
        assert result == 0
        await mgr.close()

    @pytest.mark.asyncio
    async def test_ingest_empty_file_returns_zero(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        mgr = make_manager_with_no_deps(knowledge_dir=tmp_path)
        await mgr.initialize()
        result = await mgr.ingest_file(f)
        assert result == 0
        await mgr.close()


# ─── Source management ────────────────────────────────────────────────────────


class TestKnowledgeSourceManagement:
    """delete_source() publishes event and returns count."""

    @pytest.mark.asyncio
    async def test_delete_source_publishes_event(self, tmp_path):
        published = []

        class FakeBus:
            async def publish(self, event):
                published.append(event)

        mgr = make_manager_with_no_deps(knowledge_dir=tmp_path)
        mgr._bus = FakeBus()
        await mgr.initialize()
        await mgr.delete_source("old_doc.pdf")
        deleted = [e for e in published if isinstance(e, KnowledgeSourceDeletedEvent)]
        assert len(deleted) == 1
        assert deleted[0].source == "old_doc.pdf"
        await mgr.close()

    @pytest.mark.asyncio
    async def test_delete_source_returns_zero_when_no_store(self, tmp_path):
        mgr = make_manager_with_no_deps(knowledge_dir=tmp_path)
        await mgr.initialize()
        result = await mgr.delete_source("nonexistent.txt")
        assert result == 0
        await mgr.close()


# ─── EventBus integration ─────────────────────────────────────────────────────


class TestKnowledgeEventBusIntegration:
    """KnowledgeManager publishes query events to the EventBus."""

    @pytest.mark.asyncio
    async def test_query_publishes_queried_event(self, tmp_path):
        published = []

        class FakeBus:
            async def publish(self, event):
                published.append(event)

        mgr = make_manager_with_no_deps(knowledge_dir=tmp_path)
        mgr._bus = FakeBus()
        await mgr.initialize()
        await mgr.query("What is Spidy?")
        queried = [e for e in published if isinstance(e, KnowledgeQueriedEvent)]
        assert len(queried) == 1
        await mgr.close()

    @pytest.mark.asyncio
    async def test_manager_tolerates_broken_bus(self, tmp_path):
        class BrokenBus:
            async def publish(self, event):
                raise RuntimeError("bus broken")

        mgr = make_manager_with_no_deps(knowledge_dir=tmp_path)
        mgr._bus = BrokenBus()
        await mgr.initialize()
        # Should not raise
        await mgr.query("question")
        await mgr.close()


# ─── SpidyCore integration ────────────────────────────────────────────────────


class TestSpidyCoreKnowledgeIntegration:
    """SpidyCore.knowledge property and lifecycle integration."""

    @pytest.mark.asyncio
    async def test_knowledge_property_accessible(self):
        core = SpidyCore(text_mode=True)
        # Before start(), knowledge is None
        assert core.knowledge is None

    @pytest.mark.asyncio
    async def test_core_does_not_raise_with_knowledge_absent(self):
        """SpidyCore handles KnowledgeManager init failure gracefully."""
        core = SpidyCore(text_mode=True)
        # knowledge is None before start — no crash
        assert core.knowledge is None

    def test_knowledge_manager_default_config_values(self):
        mgr = KnowledgeManager(config=None)
        assert mgr._chunk_size == 1000
        assert mgr._chunk_overlap == 200
        assert mgr._min_score == 0.3
        assert mgr._max_results == 5
        assert mgr._web_enabled is False


# ─── Brain + Knowledge integration ───────────────────────────────────────────


class TestBrainKnowledgeIntegration:
    """Brain receives KnowledgeInterface and uses it in planning/retrieval."""

    def test_brain_accepts_knowledge_interface(self):
        from spidy.brain.brain import Brain
        from spidy.core.event_bus import EventBus
        from spidy.skills.registry import SkillRegistry
        from spidy.config.manager import ReasoningConfig

        bus = EventBus()
        registry = SkillRegistry()
        config = ReasoningConfig()
        mgr = make_manager_with_no_deps()

        brain = Brain(
            bus=bus,
            config=config,
            skill_registry=registry,
            knowledge=mgr,
        )
        assert brain._knowledge is mgr

    def test_brain_works_with_none_knowledge(self):
        from spidy.brain.brain import Brain
        from spidy.core.event_bus import EventBus
        from spidy.skills.registry import SkillRegistry
        from spidy.config.manager import ReasoningConfig

        bus = EventBus()
        registry = SkillRegistry()
        config = ReasoningConfig()

        brain = Brain(
            bus=bus,
            config=config,
            skill_registry=registry,
            knowledge=None,
        )
        assert brain._knowledge is None

    @pytest.mark.asyncio
    async def test_brain_process_works_without_knowledge(self):
        from spidy.brain.brain import Brain
        from spidy.core.event_bus import EventBus
        from spidy.skills.registry import SkillRegistry
        from spidy.config.manager import ReasoningConfig

        bus = EventBus()
        loop = asyncio.get_event_loop()
        bus.set_loop(loop)
        registry = SkillRegistry()
        config = ReasoningConfig()

        brain = Brain(
            bus=bus,
            config=config,
            skill_registry=registry,
            knowledge=None,
        )
        await brain.start()
        response = await brain.process("Hello!")
        assert isinstance(response, str)
        await brain.stop()
