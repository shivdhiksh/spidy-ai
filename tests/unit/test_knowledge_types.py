"""
Unit tests for spidy.knowledge.types
Tests KnowledgeChunk, KnowledgeResult, and SourceType.
"""

from __future__ import annotations

import pytest

from spidy.knowledge.types import KnowledgeChunk, KnowledgeResult, SourceType


# ─── KnowledgeChunk ───────────────────────────────────────────────────────────


class TestKnowledgeChunkConstruction:
    """KnowledgeChunk can be constructed with required and optional fields."""

    def test_minimal_construction(self):
        chunk = KnowledgeChunk(
            chunk_id="abc",
            content="Hello world",
            source="test.txt",
        )
        assert chunk.chunk_id == "abc"
        assert chunk.content == "Hello world"
        assert chunk.source == "test.txt"
        assert chunk.source_type == "unknown"
        assert chunk.chunk_index == 0
        assert chunk.total_chunks == 1
        assert chunk.metadata == {}
        assert chunk.score == 0.0
        assert chunk.tags == []

    def test_full_construction(self):
        chunk = KnowledgeChunk(
            chunk_id="xyz",
            content="Some knowledge",
            source="doc.pdf",
            source_type="pdf",
            chunk_index=2,
            total_chunks=5,
            metadata={"page": 3, "author": "Alice"},
            score=0.87,
            tags=["science", "research"],
        )
        assert chunk.source_type == "pdf"
        assert chunk.chunk_index == 2
        assert chunk.total_chunks == 5
        assert chunk.metadata["page"] == 3
        assert chunk.score == 0.87
        assert "science" in chunk.tags

    def test_frozen_immutability(self):
        chunk = KnowledgeChunk(chunk_id="x", content="hi", source="s")
        with pytest.raises((AttributeError, TypeError)):
            chunk.content = "changed"  # type: ignore[misc]

    def test_score_clamped_above_1(self):
        chunk = KnowledgeChunk(chunk_id="x", content="hi", source="s", score=1.5)
        assert chunk.score == 1.0

    def test_score_clamped_below_0(self):
        chunk = KnowledgeChunk(chunk_id="x", content="hi", source="s", score=-0.5)
        assert chunk.score == 0.0

    def test_score_exact_boundaries(self):
        chunk_zero = KnowledgeChunk(chunk_id="x", content="hi", source="s", score=0.0)
        chunk_one = KnowledgeChunk(chunk_id="y", content="hi", source="s", score=1.0)
        assert chunk_zero.score == 0.0
        assert chunk_one.score == 1.0


class TestKnowledgeChunkProperties:
    """KnowledgeChunk properties work correctly."""

    def test_content_preview_short(self):
        chunk = KnowledgeChunk(chunk_id="x", content="Short text", source="s")
        assert chunk.content_preview == "Short text"

    def test_content_preview_truncates_at_80(self):
        long_text = "A" * 100
        chunk = KnowledgeChunk(chunk_id="x", content=long_text, source="s")
        assert len(chunk.content_preview) == 80
        assert chunk.content_preview == "A" * 80

    def test_content_preview_empty(self):
        chunk = KnowledgeChunk(chunk_id="x", content="", source="s")
        assert chunk.content_preview == ""


class TestKnowledgeChunkToDict:
    """KnowledgeChunk.to_dict() serialises correctly."""

    def test_to_dict_contains_all_fields(self):
        chunk = KnowledgeChunk(
            chunk_id="abc",
            content="hello",
            source="file.md",
            source_type="markdown",
            chunk_index=1,
            total_chunks=3,
            metadata={"title": "Test"},
            score=0.75,
            tags=["t1"],
        )
        d = chunk.to_dict()
        assert d["chunk_id"] == "abc"
        assert d["content"] == "hello"
        assert d["source"] == "file.md"
        assert d["source_type"] == "markdown"
        assert d["chunk_index"] == 1
        assert d["total_chunks"] == 3
        assert d["metadata"] == {"title": "Test"}
        assert d["score"] == 0.75
        assert d["tags"] == ["t1"]

    def test_to_dict_metadata_is_copy(self):
        meta = {"key": "value"}
        chunk = KnowledgeChunk(chunk_id="x", content="c", source="s", metadata=meta)
        d = chunk.to_dict()
        d["metadata"]["new_key"] = "new"
        assert "new_key" not in chunk.metadata

    def test_to_dict_tags_is_copy(self):
        chunk = KnowledgeChunk(chunk_id="x", content="c", source="s", tags=["a"])
        d = chunk.to_dict()
        d["tags"].append("b")
        assert "b" not in chunk.tags


class TestKnowledgeChunkCreate:
    """KnowledgeChunk.create() factory method."""

    def test_create_generates_unique_ids(self):
        c1 = KnowledgeChunk.create(content="text", source="s")
        c2 = KnowledgeChunk.create(content="text", source="s")
        assert c1.chunk_id != c2.chunk_id

    def test_create_defaults(self):
        chunk = KnowledgeChunk.create(content="test content", source="test.txt")
        assert chunk.content == "test content"
        assert chunk.source == "test.txt"
        assert chunk.source_type == "text"
        assert chunk.chunk_index == 0
        assert chunk.total_chunks == 1
        assert chunk.metadata == {}
        assert chunk.tags == []
        assert chunk.score == 0.0

    def test_create_with_all_params(self):
        chunk = KnowledgeChunk.create(
            content="chunk text",
            source="doc.pdf",
            source_type="pdf",
            chunk_index=3,
            total_chunks=10,
            metadata={"author": "Bob"},
            tags=["pdf", "research"],
            score=0.9,
        )
        assert chunk.source_type == "pdf"
        assert chunk.chunk_index == 3
        assert chunk.total_chunks == 10
        assert chunk.metadata["author"] == "Bob"
        assert "pdf" in chunk.tags
        assert chunk.score == 0.9

    def test_create_chunk_id_is_uuid_format(self):
        import uuid
        chunk = KnowledgeChunk.create(content="text", source="s")
        uuid.UUID(chunk.chunk_id)


# ─── KnowledgeResult ─────────────────────────────────────────────────────────


class TestKnowledgeResultConstruction:
    """KnowledgeResult can be constructed correctly."""

    def test_empty_result(self):
        result = KnowledgeResult(query="what is X?", chunks=[])
        assert result.query == "what is X?"
        assert result.chunks == []
        assert result.used_web_search is False
        assert result.total_found == 0

    def test_with_chunks(self):
        chunks = [
            KnowledgeChunk.create(content="answer", source="doc.txt", score=0.8),
            KnowledgeChunk.create(content="more", source="doc2.txt", score=0.6),
        ]
        result = KnowledgeResult(
            query="test query",
            chunks=chunks,
            used_web_search=True,
            total_found=10,
        )
        assert len(result.chunks) == 2
        assert result.used_web_search is True
        assert result.total_found == 10

    def test_frozen_immutability(self):
        result = KnowledgeResult(query="q", chunks=[])
        with pytest.raises((AttributeError, TypeError)):
            result.query = "changed"  # type: ignore[misc]


class TestKnowledgeResultProperties:
    """KnowledgeResult properties compute correctly."""

    def test_is_empty_true_when_no_chunks(self):
        result = KnowledgeResult(query="q", chunks=[])
        assert result.is_empty is True

    def test_is_empty_false_when_has_chunks(self):
        chunk = KnowledgeChunk.create(content="text", source="s")
        result = KnowledgeResult(query="q", chunks=[chunk])
        assert result.is_empty is False

    def test_best_score_empty(self):
        result = KnowledgeResult(query="q", chunks=[])
        assert result.best_score == 0.0

    def test_best_score_single_chunk(self):
        chunk = KnowledgeChunk.create(content="text", source="s", score=0.75)
        result = KnowledgeResult(query="q", chunks=[chunk])
        assert result.best_score == 0.75

    def test_best_score_multiple_chunks(self):
        chunks = [
            KnowledgeChunk.create(content="a", source="s", score=0.5),
            KnowledgeChunk.create(content="b", source="s", score=0.9),
            KnowledgeChunk.create(content="c", source="s", score=0.7),
        ]
        result = KnowledgeResult(query="q", chunks=chunks)
        assert result.best_score == 0.9


class TestKnowledgeResultAsRagContext:
    """KnowledgeResult.as_rag_context() formats correctly for LLM injection."""

    def test_empty_result_returns_empty_string(self):
        result = KnowledgeResult(query="q", chunks=[])
        assert result.as_rag_context() == ""

    def test_single_chunk_basic_format(self):
        chunk = KnowledgeChunk.create(
            content="Python is a programming language.",
            source="python.txt",
            score=0.8,
        )
        result = KnowledgeResult(query="what is Python?", chunks=[chunk])
        context = result.as_rag_context()
        assert "[Knowledge Context]" in context
        assert "Python is a programming language." in context
        assert "--- Source 1 ---" in context

    def test_includes_source_attribution(self):
        chunk = KnowledgeChunk.create(
            content="Some text",
            source="myfile.txt",
            score=0.7,
        )
        result = KnowledgeResult(query="q", chunks=[chunk])
        context = result.as_rag_context(include_sources=True)
        assert "myfile.txt" in context

    def test_excludes_source_when_disabled(self):
        chunk = KnowledgeChunk.create(
            content="Some text",
            source="myfile.txt",
            score=0.7,
        )
        result = KnowledgeResult(query="q", chunks=[chunk])
        context = result.as_rag_context(include_sources=False)
        assert "Some text" in context

    def test_max_chunks_limits_output(self):
        chunks = [
            KnowledgeChunk.create(content=f"Chunk {i}", source="s", score=0.8)
            for i in range(5)
        ]
        result = KnowledgeResult(query="q", chunks=chunks)
        context_all = result.as_rag_context()
        context_limited = result.as_rag_context(max_chunks=2)
        assert context_all.count("--- Source") == 5
        assert context_limited.count("--- Source") == 2

    def test_web_search_annotation_present(self):
        chunk = KnowledgeChunk.create(content="web result", source="https://example.com")
        result = KnowledgeResult(query="q", chunks=[chunk], used_web_search=True)
        context = result.as_rag_context()
        assert "web search" in context.lower()

    def test_no_web_search_annotation_when_false(self):
        chunk = KnowledgeChunk.create(content="local result", source="local.txt")
        result = KnowledgeResult(query="q", chunks=[chunk], used_web_search=False)
        context = result.as_rag_context()
        assert "web search" not in context.lower()

    def test_web_source_type_shows_url(self):
        chunk = KnowledgeChunk(
            chunk_id="web1",
            content="web content",
            source="https://example.com/page",
            source_type="web_search",
        )
        result = KnowledgeResult(query="q", chunks=[chunk])
        context = result.as_rag_context(include_sources=True)
        assert "https://example.com/page" in context

    def test_title_in_metadata_used_as_source_label(self):
        chunk = KnowledgeChunk.create(
            content="content",
            source="file.pdf",
            metadata={"title": "My Document"},
        )
        result = KnowledgeResult(query="q", chunks=[chunk])
        context = result.as_rag_context(include_sources=True)
        assert "My Document" in context
