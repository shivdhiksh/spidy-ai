"""
Unit tests for spidy.knowledge.chunker
Tests the Chunker class — text splitting with configurable size and overlap.
"""

from __future__ import annotations

import pytest

from spidy.knowledge.chunker import Chunker
from spidy.knowledge.types import KnowledgeChunk


# ─── Construction ─────────────────────────────────────────────────────────────


class TestChunkerConstruction:
    """Chunker validates constructor arguments."""

    def test_default_construction(self):
        c = Chunker()
        assert c.chunk_size == 1000
        assert c.chunk_overlap == 200

    def test_custom_construction(self):
        c = Chunker(chunk_size=500, chunk_overlap=100)
        assert c.chunk_size == 500
        assert c.chunk_overlap == 100

    def test_min_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be at least"):
            Chunker(chunk_size=10)

    def test_max_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be at most"):
            Chunker(chunk_size=9000)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap must be"):
            Chunker(chunk_size=200, chunk_overlap=-1)

    def test_overlap_equal_to_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than"):
            Chunker(chunk_size=200, chunk_overlap=200)

    def test_overlap_greater_than_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than"):
            Chunker(chunk_size=200, chunk_overlap=300)

    def test_zero_overlap_is_valid(self):
        c = Chunker(chunk_size=100, chunk_overlap=0)
        assert c.chunk_overlap == 0

    def test_min_valid_chunk_size(self):
        c = Chunker(chunk_size=50, chunk_overlap=0)
        assert c.chunk_size == 50

    def test_max_valid_chunk_size(self):
        c = Chunker(chunk_size=8000, chunk_overlap=100)
        assert c.chunk_size == 8000


# ─── chunk() method ──────────────────────────────────────────────────────────


class TestChunkerChunk:
    """Chunker.chunk() produces correct KnowledgeChunks."""

    def test_empty_text_returns_empty(self):
        c = Chunker(chunk_size=100, chunk_overlap=0)
        result = c.chunk(text="", source="s")
        assert result == []

    def test_whitespace_only_returns_empty(self):
        c = Chunker(chunk_size=100, chunk_overlap=0)
        result = c.chunk(text="   \n\t  ", source="s")
        assert result == []

    def test_short_text_single_chunk(self):
        c = Chunker(chunk_size=1000, chunk_overlap=0)
        text = "This is a short text."
        chunks = c.chunk(text=text, source="test.txt")
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].source == "test.txt"
        assert chunks[0].source_type == "text"

    def test_chunk_returns_knowledge_chunk_instances(self):
        c = Chunker(chunk_size=100, chunk_overlap=0)
        chunks = c.chunk(text="Hello world!", source="s")
        assert all(isinstance(ch, KnowledgeChunk) for ch in chunks)

    def test_chunk_ids_are_deterministic(self):
        """Same source → same chunk IDs on two calls (deterministic hash)."""
        c = Chunker(chunk_size=100, chunk_overlap=0)
        text = "Consistent text for determinism testing."
        chunks1 = c.chunk(text=text, source="same_source.txt")
        chunks2 = c.chunk(text=text, source="same_source.txt")
        ids1 = [ch.chunk_id for ch in chunks1]
        ids2 = [ch.chunk_id for ch in chunks2]
        assert ids1 == ids2

    def test_chunk_ids_differ_for_different_sources(self):
        c = Chunker(chunk_size=100, chunk_overlap=0)
        text = "Same text."
        chunks1 = c.chunk(text=text, source="source_a.txt")
        chunks2 = c.chunk(text=text, source="source_b.txt")
        # IDs are hash of source, so same index → different IDs
        assert chunks1[0].chunk_id != chunks2[0].chunk_id

    def test_source_type_preserved(self):
        c = Chunker(chunk_size=100, chunk_overlap=0)
        chunks = c.chunk(text="text", source="doc.pdf", source_type="pdf")
        assert chunks[0].source_type == "pdf"

    def test_metadata_attached_to_all_chunks(self):
        c = Chunker(chunk_size=100, chunk_overlap=0)
        # Create text long enough for multiple chunks
        text = "X" * 250
        meta = {"author": "Alice", "date": "2024"}
        chunks = c.chunk(text=text, source="s", metadata=meta)
        for chunk in chunks:
            assert chunk.metadata["author"] == "Alice"
            assert chunk.metadata["date"] == "2024"

    def test_tags_attached_to_all_chunks(self):
        c = Chunker(chunk_size=100, chunk_overlap=0)
        text = "X" * 250
        tags = ["science", "research"]
        chunks = c.chunk(text=text, source="s", tags=tags)
        for chunk in chunks:
            assert "science" in chunk.tags
            assert "research" in chunk.tags

    def test_chunk_index_sequential(self):
        c = Chunker(chunk_size=100, chunk_overlap=0)
        text = "X" * 300
        chunks = c.chunk(text=text, source="s")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_total_chunks_consistent(self):
        c = Chunker(chunk_size=100, chunk_overlap=0)
        text = "X" * 300
        chunks = c.chunk(text=text, source="s")
        total = len(chunks)
        for chunk in chunks:
            assert chunk.total_chunks == total

    def test_long_text_produces_multiple_chunks(self):
        c = Chunker(chunk_size=100, chunk_overlap=0)
        text = "A" * 500
        chunks = c.chunk(text=text, source="s")
        assert len(chunks) > 1

    def test_overlap_creates_shared_content(self):
        """Adjacent chunks with overlap should share content at boundaries."""
        c = Chunker(chunk_size=100, chunk_overlap=50)
        text = "A" * 200
        chunks = c.chunk(text=text, source="s")
        # With chunk_size=100, overlap=50, stride=50:
        # chunk0 = text[0:100], chunk1 = text[50:150], etc.
        assert len(chunks) >= 3

    def test_score_zero_on_fresh_chunks(self):
        c = Chunker(chunk_size=100, chunk_overlap=0)
        chunks = c.chunk(text="hello world", source="s")
        assert all(ch.score == 0.0 for ch in chunks)

    def test_metadata_includes_chunk_index(self):
        c = Chunker(chunk_size=100, chunk_overlap=0)
        text = "X" * 300
        chunks = c.chunk(text=text, source="s")
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i


# ─── chunk_pages() method ────────────────────────────────────────────────────


class TestChunkerChunkPages:
    """Chunker.chunk_pages() processes multi-page documents."""

    def test_empty_pages_list(self):
        c = Chunker(chunk_size=100, chunk_overlap=0)
        result = c.chunk_pages(pages=[], source="doc.pdf")
        assert result == []

    def test_single_page(self):
        c = Chunker(chunk_size=1000, chunk_overlap=0)
        pages = ["Page one content with sufficient text."]
        chunks = c.chunk_pages(pages=pages, source="doc.pdf")
        assert len(chunks) >= 1
        assert chunks[0].source == "doc.pdf"
        assert chunks[0].source_type == "pdf"

    def test_multiple_pages_all_processed(self):
        c = Chunker(chunk_size=1000, chunk_overlap=0)
        pages = ["Page 1 content", "Page 2 content", "Page 3 content"]
        chunks = c.chunk_pages(pages=pages, source="doc.pdf")
        assert len(chunks) == 3  # Each page is one chunk (within size limit)

    def test_empty_pages_skipped(self):
        c = Chunker(chunk_size=1000, chunk_overlap=0)
        pages = ["Page 1", "", "   ", "Page 4"]
        chunks = c.chunk_pages(pages=pages, source="doc.pdf")
        # Only non-empty pages produce chunks
        assert len(chunks) == 2

    def test_page_number_in_metadata(self):
        c = Chunker(chunk_size=1000, chunk_overlap=0)
        pages = ["Page one.", "Page two.", "Page three."]
        chunks = c.chunk_pages(pages=pages, source="doc.pdf")
        page_nums = [ch.metadata.get("page") for ch in chunks]
        assert 1 in page_nums
        assert 2 in page_nums
        assert 3 in page_nums

    def test_source_type_pdf_default(self):
        c = Chunker(chunk_size=1000, chunk_overlap=0)
        pages = ["Some text"]
        chunks = c.chunk_pages(pages=pages, source="doc.pdf")
        assert chunks[0].source_type == "pdf"

    def test_custom_source_type(self):
        c = Chunker(chunk_size=1000, chunk_overlap=0)
        pages = ["Some text"]
        chunks = c.chunk_pages(pages=pages, source="doc.docx", source_type="docx")
        assert chunks[0].source_type == "docx"

    def test_metadata_merges_with_page_meta(self):
        c = Chunker(chunk_size=1000, chunk_overlap=0)
        pages = ["Content"]
        chunks = c.chunk_pages(
            pages=pages,
            source="doc.pdf",
            metadata={"author": "Alice"},
        )
        assert chunks[0].metadata.get("author") == "Alice"
        assert chunks[0].metadata.get("page") == 1


# ─── _split() helper ─────────────────────────────────────────────────────────


class TestChunkerSplit:
    """Chunker._split() internal splitting logic."""

    def test_exact_size_text(self):
        c = Chunker(chunk_size=100, chunk_overlap=0)
        text = "A" * 100
        splits = c._split(text)
        assert len(splits) == 1
        assert splits[0] == "A" * 100

    def test_double_size_text_two_chunks(self):
        c = Chunker(chunk_size=100, chunk_overlap=0)
        text = "A" * 200
        splits = c._split(text)
        assert len(splits) == 2

    def test_strips_whitespace_from_chunks(self):
        c = Chunker(chunk_size=100, chunk_overlap=0)
        text = "  content  " + "x" * 89
        splits = c._split(text)
        assert splits[0] == splits[0].strip()


# ─── _hash_source() ──────────────────────────────────────────────────────────


class TestChunkerHashSource:
    """Chunker._hash_source() produces deterministic hashes."""

    def test_same_source_same_hash(self):
        h1 = Chunker._hash_source("test.txt")
        h2 = Chunker._hash_source("test.txt")
        assert h1 == h2

    def test_different_sources_different_hash(self):
        h1 = Chunker._hash_source("source_a.txt")
        h2 = Chunker._hash_source("source_b.txt")
        assert h1 != h2

    def test_hash_is_12_chars(self):
        h = Chunker._hash_source("any_source_string")
        assert len(h) == 12

    def test_hash_is_hex(self):
        h = Chunker._hash_source("any_source_string")
        int(h, 16)  # Should not raise
