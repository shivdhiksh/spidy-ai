"""
Brain Companion Interfaces — Forward-Declared Contracts
========================================================
These three ABCs define the extension points for Spidy's
Lifelong AI Companion capabilities.

Status: INTERFACE CONTRACTS ONLY (Milestone 3)
-----------------------------------------------
All methods raise ``NotImplementedError``. Future milestones replace
``NotImplementedError`` with real implementations by subclassing:

  MemoryInterface   — Milestone 6: Memory Engine
                      Short-term buffer, long-term episodic memory,
                      semantic (vector) search, preference storage.

  KnowledgeInterface — Milestone 8+: Knowledge System
                       Personal knowledge graph, RAG over documents,
                       PDF / Markdown ingestion, web-sourced facts.

  LearningInterface  — Milestone 9+: Continuous Learning Engine
                       User preference learning, habit modelling,
                       workflow detection, feedback integration.

Design
------
The Brain accepts any concrete implementation of these interfaces at
construction time. Passing ``None`` (or omitting) causes companion
features to degrade gracefully:

  Memory=None     → context window only (no long-term recall)
  Knowledge=None  → no document / RAG answers
  Learning=None   → no personalisation or habit suggestions

This separation of interface from implementation is critical:
  1. Future implementations slot in without touching the Brain
  2. Test doubles are trivial to write (extend the ABC, implement stubs)
  3. The companion roadmap is visible in the source itself
  4. The Brain can use these interfaces from day one via duck typing
"""

from __future__ import annotations

import abc
from typing import Any


# ─── Memory Interface ─────────────────────────────────────────────────────────


class MemoryInterface(abc.ABC):
    """
    Interface for Spidy's memory system.

    Future implementation (Milestone 6) will provide:
    - Short-term: rolling conversation buffer backed by SQLite
    - Long-term: episodic memories with timestamps and tags
    - Semantic: vector-similarity search via ChromaDB

    The Brain calls this interface but never knows the backing store.
    """

    @abc.abstractmethod
    async def store(
        self,
        content: str,
        session_id: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a memory entry and return its ID.

        Parameters
        ----------
        content:
            Text content to remember.
        session_id:
            Conversation session this memory belongs to.
        tags:
            Optional categorisation tags (e.g. ``["work", "code"]``).
        metadata:
            Arbitrary key-value metadata attached to the memory.

        Returns
        -------
        str
            The memory ID (for later retrieval or deletion).
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def recall(
        self,
        query: str,
        session_id: str = "",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories relevant to a query.

        Parameters
        ----------
        query:
            Natural language query (e.g. "what did the user say about Python?").
        session_id:
            If provided, scope the search to this session.
        limit:
            Maximum number of memories to return.

        Returns
        -------
        list[dict]
            Memory records, each with at minimum
            ``{"id": str, "content": str, "score": float}``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search across all stored memories.

        Returns
        -------
        list[dict]
            Ranked memory records.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def clear(self, session_id: str = "") -> int:
        """
        Clear memories for a session (or all if session_id is empty).

        Returns
        -------
        int
            Number of memories cleared.
        """
        raise NotImplementedError


# ─── Knowledge Interface ──────────────────────────────────────────────────────


class KnowledgeInterface(abc.ABC):
    """
    Interface for Spidy's personal knowledge retrieval system.

    Future implementation (Milestone 8+) will provide:
    - Personal knowledge graph (RDF / property graph)
    - RAG over user documents (PDF, Markdown, DOCX, web pages)
    - Web search with source attribution (optional, user-controlled)
    - Document ingestion pipeline (chunking, embedding, indexing)

    The Brain calls this to augment LLM responses with grounded facts.
    """

    @abc.abstractmethod
    async def query(
        self,
        question: str,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Query the knowledge base for relevant information.

        Parameters
        ----------
        question:
            Natural language question.
        max_results:
            Maximum number of knowledge chunks to return.

        Returns
        -------
        list[dict]
            Knowledge chunks with at minimum
            ``{"content": str, "source": str, "score": float}``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def ingest(
        self,
        content: str,
        source: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add content to the knowledge base.

        Parameters
        ----------
        content:
            Text content to ingest.
        source:
            Human-readable source identifier (e.g. file path, URL).
        tags:
            Optional categorisation tags.
        metadata:
            Arbitrary metadata (author, date, doc_type, etc.).

        Returns
        -------
        str
            The document / chunk ID.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def search_rag(
        self,
        query: str,
        limit: int = 5,
    ) -> str:
        """
        Retrieve and format knowledge chunks for RAG prompt injection.

        Returns a formatted string ready to be injected into an LLM prompt
        as retrieved context. Empty string if no relevant chunks are found.

        Returns
        -------
        str
            Formatted context block (may be empty).
        """
        raise NotImplementedError


# ─── Learning Interface ───────────────────────────────────────────────────────


class LearningInterface(abc.ABC):
    """
    Interface for Spidy's continuous learning and personalisation system.

    Future implementation (Milestone 9+) will provide:
    - User preference learning from explicit + implicit signals
    - Habit and workflow detection over time
    - Personalised response style adaptation
    - Privacy-preserving local-only learning (no data leaves the device)

    The Brain calls this to enrich decisions with learned user context
    and to record feedback signals for continuous improvement.
    """

    @abc.abstractmethod
    async def record_feedback(
        self,
        session_id: str,
        utterance: str,
        response: str,
        rating: float,          # -1.0 (negative) to +1.0 (positive); 0.0 neutral
        tags: list[str] | None = None,
    ) -> None:
        """
        Record user feedback on a Brain response.

        Used to fine-tune preferences and improve future responses.
        Called by the Brain after every interaction where feedback is available.

        Parameters
        ----------
        rating:
            -1.0 = explicitly negative (user said "no", "wrong", "stop")
             0.0 = neutral / no feedback
            +1.0 = explicitly positive (user said "yes", "thanks", "perfect")
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def get_preference(
        self,
        category: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a learned user preference.

        Parameters
        ----------
        category:
            Preference key (e.g. ``"response_style"``, ``"tts_speed"``,
            ``"preferred_browser"``, ``"working_hours"``).

        Returns
        -------
        Any
            The learned preference value, or ``default`` if not established.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def get_habit(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Check if a learned habit matches the current context.

        Parameters
        ----------
        context:
            Current signals: time, active_app, day_of_week, recent_actions, etc.

        Returns
        -------
        dict | None
            A habit record if one matches, else None.
            Format: ``{"action": str, "confidence": float, "description": str}``
        """
        raise NotImplementedError


# ─── Vision Interface ─────────────────────────────────────────────────────────


class VisionInterface(abc.ABC):
    """
    Interface for Spidy's Vision & Screen Understanding system.

    Implementation (Milestone 9): VisionManager in ``spidy.vision``.

    Provides:
    - Screenshot capture (full screen, active window, region, multi-monitor)
    - OCR text extraction from screenshots
    - Screen state analysis (active app, visible windows, UI regions)
    - Human-readable screen description for LLM context injection

    The Brain calls this interface but never knows the backing engines
    (mss, easyocr, opencv). Passing ``None`` causes vision features to
    degrade gracefully — all Brain core functions continue working.
    """

    @abc.abstractmethod
    async def capture(
        self,
        source: str = "fullscreen",
        **kwargs: Any,
    ) -> Any:
        """
        Take a screenshot.

        Parameters
        ----------
        source:
            ``"fullscreen"`` | ``"window"`` | ``"region"``
        **kwargs:
            Engine-specific parameters (e.g. monitor_index, x, y, width, height).

        Returns
        -------
        ScreenshotResult
            PNG bytes and metadata. Empty result on failure.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def read_screen_text(self) -> Any:
        """
        Capture the screen and extract all visible text via OCR.

        Returns
        -------
        OCRResult
            Extracted text with confidence scores.
            Empty result when OCR deps absent.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def analyze_screen(self) -> Any:
        """
        Perform a full analysis of the current screen state.

        Returns
        -------
        ScreenAnalysis
            Active application, visible windows, detected UI regions.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def describe_screen(self) -> str:
        """
        Return a human-readable description of the current screen state.

        Combines screen analysis and OCR text into one string suitable
        for injection into the LLM context window.

        Returns
        -------
        str
            Multi-line description. Non-empty even on degraded operation.
        """
        raise NotImplementedError
