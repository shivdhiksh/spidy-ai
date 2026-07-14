"""
Unit tests for spidy.knowledge.ingestor
Tests DocumentIngestor and all format-specific sub-ingestors.
Uses tmp_path and in-memory text files — no external deps required.
"""

from __future__ import annotations

import pytest

from spidy.knowledge.ingestor import (
    DocumentIngestor,
    MarkdownIngestor,
    PlainTextIngestor,
    PDFIngestor,
    DocxIngestor,
)


# ─── PlainTextIngestor ────────────────────────────────────────────────────────


class TestPlainTextIngestor:
    """PlainTextIngestor reads plain text files (no deps required)."""

    def test_reads_simple_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        ingestor = PlainTextIngestor()
        result = ingestor.read(f)
        assert result == ["Hello, world!"]

    def test_returns_empty_for_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        ingestor = PlainTextIngestor()
        result = ingestor.read(f)
        assert result == []

    def test_returns_empty_for_whitespace_only_file(self, tmp_path):
        f = tmp_path / "ws.txt"
        f.write_text("   \n\t  ", encoding="utf-8")
        ingestor = PlainTextIngestor()
        result = ingestor.read(f)
        assert result == []

    def test_strips_surrounding_whitespace(self, tmp_path):
        f = tmp_path / "ws.txt"
        f.write_text("  content  \n", encoding="utf-8")
        ingestor = PlainTextIngestor()
        result = ingestor.read(f)
        assert result == ["content"]

    def test_returns_empty_for_missing_file(self, tmp_path):
        ingestor = PlainTextIngestor()
        result = ingestor.read(tmp_path / "nonexistent.txt")
        assert result == []

    def test_reads_multiline_text(self, tmp_path):
        f = tmp_path / "multi.txt"
        content = "Line 1\nLine 2\nLine 3"
        f.write_text(content, encoding="utf-8")
        ingestor = PlainTextIngestor()
        result = ingestor.read(f)
        assert len(result) == 1
        assert "Line 1" in result[0]
        assert "Line 3" in result[0]

    def test_is_available_always_true(self):
        ingestor = PlainTextIngestor()
        assert ingestor.is_available is True

    def test_format_name(self):
        assert PlainTextIngestor.format_name == "text"


# ─── MarkdownIngestor ─────────────────────────────────────────────────────────


class TestMarkdownIngestor:
    """MarkdownIngestor strips Markdown syntax (no deps required)."""

    def test_reads_plain_text_markdown(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("Hello world.", encoding="utf-8")
        ingestor = MarkdownIngestor()
        result = ingestor.read(f)
        assert len(result) == 1
        assert "Hello world." in result[0]

    def test_strips_headings(self, tmp_path):
        f = tmp_path / "h.md"
        f.write_text("# Title\n\nSome content.", encoding="utf-8")
        ingestor = MarkdownIngestor()
        result = ingestor.read(f)
        assert len(result) == 1
        assert "Title" in result[0]
        assert result[0].lstrip().startswith("Title") or "Title" in result[0]
        assert "#" not in result[0]

    def test_strips_bold(self, tmp_path):
        f = tmp_path / "bold.md"
        f.write_text("This is **bold** text.", encoding="utf-8")
        ingestor = MarkdownIngestor()
        result = ingestor.read(f)
        assert "**" not in result[0]
        assert "bold" in result[0]

    def test_strips_italic(self, tmp_path):
        f = tmp_path / "italic.md"
        f.write_text("This is *italic* text.", encoding="utf-8")
        ingestor = MarkdownIngestor()
        result = ingestor.read(f)
        assert "*italic*" not in result[0]
        assert "italic" in result[0]

    def test_strips_links(self, tmp_path):
        f = tmp_path / "link.md"
        f.write_text("Check [this link](https://example.com) here.", encoding="utf-8")
        ingestor = MarkdownIngestor()
        result = ingestor.read(f)
        assert "https://example.com" not in result[0]
        assert "this link" in result[0]

    def test_strips_code_fences(self, tmp_path):
        f = tmp_path / "code.md"
        content = "Text before.\n```python\nprint('hello')\n```\nText after."
        f.write_text(content, encoding="utf-8")
        ingestor = MarkdownIngestor()
        result = ingestor.read(f)
        assert "```" not in result[0]
        assert "Text before." in result[0]
        assert "Text after." in result[0]

    def test_returns_empty_for_empty_file(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text("", encoding="utf-8")
        ingestor = MarkdownIngestor()
        result = ingestor.read(f)
        assert result == []

    def test_returns_empty_for_missing_file(self, tmp_path):
        ingestor = MarkdownIngestor()
        result = ingestor.read(tmp_path / "nonexistent.md")
        assert result == []

    def test_is_available_always_true(self):
        ingestor = MarkdownIngestor()
        assert ingestor.is_available is True

    def test_format_name(self):
        assert MarkdownIngestor.format_name == "markdown"


class TestMarkdownStripMarkdown:
    """MarkdownIngestor._strip_markdown() static method coverage."""

    def test_strips_inline_code(self):
        result = MarkdownIngestor._strip_markdown("Use `print()` function.")
        # The inline-code regex `[^`]+` removes the backtick-delimited token.
        # The surrounding text remains.
        assert "`" not in result
        assert "Use" in result
        assert "function." in result

    def test_strips_image_syntax(self):
        result = MarkdownIngestor._strip_markdown("![alt text](image.png) below")
        assert "![" not in result
        assert "below" in result

    def test_strips_horizontal_rules(self):
        result = MarkdownIngestor._strip_markdown("Section\n---\nContent")
        assert "---" not in result

    def test_collapses_multiple_blank_lines(self):
        result = MarkdownIngestor._strip_markdown("Line 1\n\n\n\nLine 2")
        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in result


# ─── PDFIngestor ─────────────────────────────────────────────────────────────


class TestPDFIngestor:
    """PDFIngestor degrades gracefully when pypdf is absent/unavailable."""

    def test_format_name(self):
        assert PDFIngestor.format_name == "pdf"

    def test_returns_empty_if_unavailable(self, tmp_path, monkeypatch):
        """If pypdf is not installed, read() returns empty list."""
        ingestor = PDFIngestor()
        # Force unavailable
        monkeypatch.setattr(type(ingestor), "is_available",
                            property(lambda self: False))
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"not a real pdf")
        result = ingestor.read(fake_pdf)
        assert result == []

    def test_returns_empty_for_corrupt_file(self, tmp_path):
        """Invalid PDF bytes → graceful return of empty list."""
        ingestor = PDFIngestor()
        if not ingestor.is_available:
            pytest.skip("pypdf not installed")
        fake = tmp_path / "corrupt.pdf"
        fake.write_bytes(b"not a pdf")
        result = ingestor.read(fake)
        assert result == []


# ─── DocxIngestor ─────────────────────────────────────────────────────────────


class TestDocxIngestor:
    """DocxIngestor degrades gracefully when python-docx is absent."""

    def test_format_name(self):
        assert DocxIngestor.format_name == "docx"

    def test_returns_empty_if_unavailable(self, tmp_path, monkeypatch):
        ingestor = DocxIngestor()
        monkeypatch.setattr(type(ingestor), "is_available",
                            property(lambda self: False))
        fake = tmp_path / "fake.docx"
        fake.write_bytes(b"not a real docx")
        result = ingestor.read(fake)
        assert result == []

    def test_returns_empty_for_corrupt_file(self, tmp_path):
        ingestor = DocxIngestor()
        if not ingestor.is_available:
            pytest.skip("python-docx not installed")
        fake = tmp_path / "corrupt.docx"
        fake.write_bytes(b"not a docx")
        result = ingestor.read(fake)
        assert result == []


# ─── DocumentIngestor (unified) ───────────────────────────────────────────────


class TestDocumentIngestor:
    """DocumentIngestor unified dispatch and metadata methods."""

    def test_reads_plain_text(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("My notes here.", encoding="utf-8")
        ingestor = DocumentIngestor()
        result = ingestor.read(f)
        assert len(result) == 1
        assert "My notes" in result[0]

    def test_reads_markdown(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Hello\n\nWorld.", encoding="utf-8")
        ingestor = DocumentIngestor()
        result = ingestor.read(f)
        assert len(result) == 1
        assert "Hello" in result[0]

    def test_reads_markdown_extension_alias(self, tmp_path):
        f = tmp_path / "readme.markdown"
        f.write_text("# Doc\n\nContent.", encoding="utf-8")
        ingestor = DocumentIngestor()
        result = ingestor.read(f)
        assert len(result) >= 1

    def test_returns_empty_for_missing_file(self, tmp_path):
        ingestor = DocumentIngestor()
        result = ingestor.read(tmp_path / "ghost.txt")
        assert result == []

    def test_returns_empty_for_directory(self, tmp_path):
        ingestor = DocumentIngestor()
        result = ingestor.read(tmp_path)
        assert result == []

    def test_unknown_extension_uses_plaintext(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1,col2\nval1,val2", encoding="utf-8")
        ingestor = DocumentIngestor()
        result = ingestor.read(f)
        assert len(result) == 1

    def test_get_source_type_pdf(self):
        ingestor = DocumentIngestor()
        assert ingestor.get_source_type("report.pdf") == "pdf"

    def test_get_source_type_docx(self):
        ingestor = DocumentIngestor()
        assert ingestor.get_source_type("doc.docx") == "docx"

    def test_get_source_type_doc(self):
        ingestor = DocumentIngestor()
        assert ingestor.get_source_type("doc.doc") == "docx"

    def test_get_source_type_markdown(self):
        ingestor = DocumentIngestor()
        assert ingestor.get_source_type("readme.md") == "markdown"

    def test_get_source_type_markdown_long(self):
        ingestor = DocumentIngestor()
        assert ingestor.get_source_type("readme.markdown") == "markdown"

    def test_get_source_type_text_fallback(self):
        ingestor = DocumentIngestor()
        assert ingestor.get_source_type("data.csv") == "text"

    def test_get_metadata_for_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content", encoding="utf-8")
        ingestor = DocumentIngestor()
        meta = ingestor.get_metadata(f)
        assert meta["filename"] == "test.txt"
        assert meta["extension"] == ".txt"
        assert meta["size_bytes"] > 0

    def test_get_metadata_for_missing_file(self, tmp_path):
        ingestor = DocumentIngestor()
        meta = ingestor.get_metadata(tmp_path / "ghost.txt")
        assert meta["size_bytes"] == 0

    def test_supported_extensions_includes_common_types(self):
        ingestor = DocumentIngestor()
        exts = ingestor.supported_extensions
        assert ".pdf" in exts
        assert ".docx" in exts
        assert ".md" in exts
        assert ".txt" in exts

    def test_is_supported_for_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content", encoding="utf-8")
        ingestor = DocumentIngestor()
        assert ingestor.is_supported(f) is True

    def test_is_supported_false_for_missing(self, tmp_path):
        ingestor = DocumentIngestor()
        assert ingestor.is_supported(tmp_path / "ghost.txt") is False

    def test_string_path_accepted(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("String path test", encoding="utf-8")
        ingestor = DocumentIngestor()
        result = ingestor.read(str(f))
        assert len(result) == 1
