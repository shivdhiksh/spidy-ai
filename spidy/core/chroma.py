"""
ChromaClientRegistry — Shared ChromaDB Client Cache
===================================================
Provides shared PersistentClient and EphemeralClient instances keyed
by persist path to avoid redundant client initializations and SQLite handles.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from spidy.logging.logger import get_logger

log = get_logger(__name__)


class ChromaClientRegistry:
    """
    Registry for ChromaDB clients to avoid duplicate client initialization.
    """

    _clients: dict[str, Any] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_client(cls, persist_dir: str | Path | None) -> Any:
        """
        Get or create a ChromaDB client for the given persist directory.

        Parameters
        ----------
        persist_dir:
            Directory path for persistent storage, or None / ':memory:' for EphemeralClient.

        Returns
        -------
        chromadb.ClientAPI
            The shared ChromaDB client instance.
        """
        import chromadb

        is_ephemeral = (persist_dir is None or str(persist_dir) == ":memory:")
        if is_ephemeral:
            return chromadb.EphemeralClient()

        key = str(Path(persist_dir).resolve())
        with cls._lock:
            if key not in cls._clients:
                path_obj = Path(persist_dir)
                path_obj.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(path_obj.resolve()))
                cls._clients[key] = client
            return cls._clients[key]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cache."""
        with cls._lock:
            cls._clients.clear()
