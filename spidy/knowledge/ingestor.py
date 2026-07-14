"""
Knowledge Engine — Document Ingestor
======================================
Reads source documents and returns raw text (one string per page/section).

Architecture
------------
``DocumentIngestor`` is the unified entry point.  It dispatches to a
format-specific ingestor based on file extension:

    .pdf              → PDFIngestor      (pypdf, optional)
    .docx             → DocxIngestor     (python-docx, optional)
    .md / .markdown   → MarkdownIngestor (stdlib re only)
    anything else     → PlainTextIngestor

Design principles
-----------------
- All format-specific deps are optional: if absent, the ingestor logs a
  warning and returns an empty list.  The caller (KnowledgeManager) is
  responsible for logging the skip.
- Only local file paths are accepted.  URL / web ingestion is reserved for
  M11 Browser Agent integration.
- All ingestors return ``list[str]`` — one string per logical page or section.
  The Chunker then splits these further into overlapping chunks.
- No I/O inside the Chunker; no parsing inside the Manager.  Separation of
  concerns is strict.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from spidy.logging.logger import get_logger

log = get_logger(__name__)


# ─── Base Ingestor ────────────────────────────────────────────────────────────


class BaseIngestor:
    """
    Abstract base for format-specific document ingestors.

    Subclasses override ``read(path)`` and ``is_available``.
    """

    #: Human-readable name for logging
    format_name: str = "unknown"

    @property
    def is_available(self) -> bool:
        """True if the required optional dependency is installed."""
        return True

    def read(self, path: Path) -> list[str]:
        """
        Read the document at ``path`` and return a list of text pages/sections.

        Returns an empty list on any error (always non-fatal).
        """
        raise NotImplementedError


# ─── PDF Ingestor ─────────────────────────────────────────────────────────────


class PDFIngestor(BaseIngestor):
    """
    PDF text extraction via ``pypdf``.

    ``pypdf`` is already in ``pyproject.toml[skills]``.
    Returns one string per PDF page.
    """

    format_name = "pdf"

    @property
    def is_available(self) -> bool:
        try:
            import pypdf  # noqa: F401
            return True
        except ImportError:
            return False

    def read(self, path: Path) -> list[str]:
        if not self.is_available:
            log.warning(
                "PDFIngestor: pypdf not installed. "
                "Install with: pip install pypdf"
            )
            return []
        try:
            import pypdf

            pages: list[str] = []
            with pypdf.PdfReader(str(path)) as reader:
                for page in reader.pages:
                    text = page.extract_text() or ""
                    pages.append(text)
            log.debug(
                "PDFIngestor: '{path}' → {n} pages",
                path=path.name,
                n=len(pages),
            )
            return pages
        except Exception as exc:
            log.warning(
                "PDFIngestor: failed to read '{path}': {exc}",
                path=path,
                exc=exc,
            )
            return []


# ─── DOCX Ingestor ────────────────────────────────────────────────────────────


class DocxIngestor(BaseIngestor):
    """
    Word document text extraction via ``python-docx``.

    ``python-docx`` is already in ``pyproject.toml[skills]``.
    Returns one string per paragraph (non-empty paragraphs only).
    Grouped into a single page-equivalent section for the chunker.
    """

    format_name = "docx"

    @property
    def is_available(self) -> bool:
        try:
            import docx  # noqa: F401
            return True
        except ImportError:
            return False

    def read(self, path: Path) -> list[str]:
        if not self.is_available:
            log.warning(
                "DocxIngestor: python-docx not installed. "
                "Install with: pip install python-docx"
            )
            return []
        try:
            import docx

            doc = docx.Document(str(path))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            # Group into sections of ~50 paragraphs each to mimic pages
            section_size = 50
            sections: list[str] = []
            for i in range(0, max(1, len(paragraphs)), section_size):
                section_text = "\n\n".join(paragraphs[i : i + section_size])
                if section_text.strip():
                    sections.append(section_text)
            log.debug(
                "DocxIngestor: '{path}' → {n} sections ({p} paragraphs)",
                path=path.name,
                n=len(sections),
                p=len(paragraphs),
            )
            return sections
        except Exception as exc:
            log.warning(
                "DocxIngestor: failed to read '{path}': {exc}",
                path=path,
                exc=exc,
            )
            return []


# ─── Markdown Ingestor ────────────────────────────────────────────────────────


class MarkdownIngestor(BaseIngestor):
    """
    Markdown text extraction — zero external dependencies.

    Strips Markdown syntax (headings, bold, italic, links, code fences,
    inline code) and returns the resulting plain text as one section.
    Preserves paragraph structure (double newlines).
    """

    format_name = "markdown"

    def read(self, path: Path) -> list[str]:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            cleaned = self._strip_markdown(raw)
            if not cleaned.strip():
                return []
            log.debug("MarkdownIngestor: '{path}' read OK", path=path.name)
            return [cleaned]
        except Exception as exc:
            log.warning(
                "MarkdownIngestor: failed to read '{path}': {exc}",
                path=path,
                exc=exc,
            )
            return []

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove the most common Markdown syntax, preserving readable text."""
        # Remove code fences (``` ... ```)
        text = re.sub(r"```[\s\S]*?```", "", text)
        # Remove inline code
        text = re.sub(r"`[^`]+`", "", text)
        # Remove headings (#, ##, etc.) — keep the text
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove bold/italic (**text**, *text*, __text__, _text_)
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
        text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
        # Remove links [text](url) → text
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        # Remove image syntax ![alt](url)
        text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
        # Remove horizontal rules
        text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# ─── Plain Text Ingestor ──────────────────────────────────────────────────────


class PlainTextIngestor(BaseIngestor):
    """
    Plain text fallback — reads any UTF-8 text file as a single section.
    """

    format_name = "text"

    def read(self, path: Path) -> list[str]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                return []
            log.debug("PlainTextIngestor: '{path}' read OK", path=path.name)
            return [text]
        except Exception as exc:
            log.warning(
                "PlainTextIngestor: failed to read '{path}': {exc}",
                path=path,
                exc=exc,
            )
            return []


# ─── Document Ingestor (unified entry point) ─────────────────────────────────


class DocumentIngestor:
    """
    Unified document ingestor — dispatches to format-specific readers.

    Usage
    -----
        ingestor = DocumentIngestor()
        pages = ingestor.read(Path("report.pdf"))
        # pages is list[str], one entry per page/section
        metadata = ingestor.get_metadata(Path("report.pdf"))

    Supported formats
    -----------------
    - .pdf   — requires pypdf
    - .docx  — requires python-docx
    - .md / .markdown — stdlib only
    - .txt and any other extension — stdlib only (plain text)

    URL ingestion is not supported here; pass content as text via
    ``KnowledgeManager.ingest()`` for URL-sourced content.
    """

    # Extension → ingestor mapping
    _INGESTOR_MAP: dict[str, BaseIngestor] = {}

    def __init__(self) -> None:
        self._pdf = PDFIngestor()
        self._docx = DocxIngestor()
        self._markdown = MarkdownIngestor()
        self._plaintext = PlainTextIngestor()

        self._map: dict[str, BaseIngestor] = {
            ".pdf": self._pdf,
            ".docx": self._docx,
            ".doc": self._docx,   # basic support
            ".md": self._markdown,
            ".markdown": self._markdown,
        }

    def read(self, path: Path | str) -> list[str]:
        """
        Read a document and return raw text sections.

        Parameters
        ----------
        path:
            Path to the local document file.

        Returns
        -------
        list[str]
            One string per logical page or section.
            Empty list on failure or unsupported type with missing dep.
        """
        path = Path(path)
        if not path.exists():
            log.warning("DocumentIngestor: file not found: '{path}'", path=path)
            return []
        if not path.is_file():
            log.warning("DocumentIngestor: not a file: '{path}'", path=path)
            return []

        ext = path.suffix.lower()
        ingestor = self._map.get(ext, self._plaintext)

        log.debug(
            "DocumentIngestor: reading '{path}' as {fmt}",
            path=path.name,
            fmt=ingestor.format_name,
        )
        return ingestor.read(path)

    def get_source_type(self, path: Path | str) -> str:
        """Return the source type string for a given file path."""
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return "pdf"
        if ext in (".docx", ".doc"):
            return "docx"
        if ext in (".md", ".markdown"):
            return "markdown"
        return "text"

    def get_metadata(self, path: Path | str) -> dict[str, Any]:
        """
        Extract basic file metadata without fully reading the document.

        Returns
        -------
        dict
            ``{"filename": str, "extension": str, "size_bytes": int}``
        """
        path = Path(path)
        meta: dict[str, Any] = {
            "filename": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
        return meta

    @property
    def supported_extensions(self) -> list[str]:
        """All explicitly supported file extensions (fallback to plain text for others)."""
        return list(self._map.keys()) + [".txt"]

    def is_supported(self, path: Path | str) -> bool:
        """True if the file can be ingested (always True for plain text fallback)."""
        return Path(path).is_file()
