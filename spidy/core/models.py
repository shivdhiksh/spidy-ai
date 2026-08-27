"""
EmbeddingModelRegistry — Shared SentenceTransformer Model Cache
================================================================
Ensures only ONE SentenceTransformer instance per model name/device
is loaded and shared across SPIDY components (SemanticMemory, EmbeddingEngine).

Thread-safe, lazy initialization, and prevents duplicate memory allocations.
"""

from __future__ import annotations

import threading
from typing import Any

from spidy.logging.logger import get_logger

log = get_logger(__name__)


class EmbeddingModelRegistry:
    """
    Singleton registry caching SentenceTransformer model instances.
    """

    _instances: dict[tuple[str, str], Any] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_model(
        cls,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> Any:
        """
        Get or create a shared SentenceTransformer model instance.

        Parameters
        ----------
        model_name:
            HuggingFace / SentenceTransformer model identifier.
        device:
            Device to place the model on ("cpu", "cuda", "auto").

        Returns
        -------
        SentenceTransformer
            The shared model instance.
        """
        # Resolve 'auto' to 'cpu' when torch is CPU build or to match best practice
        resolved_device = device
        if resolved_device == "auto":
            try:
                import torch
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                resolved_device = "cpu"

        key = (model_name, resolved_device)
        with cls._lock:
            if key not in cls._instances:
                from sentence_transformers import SentenceTransformer
                log.info(
                    "[EMBEDDING] model={model} | device={device} | shared_instance=true",
                    model=model_name,
                    device=resolved_device,
                )
                model = SentenceTransformer(model_name, device=resolved_device)
                cls._instances[key] = model
            return cls._instances[key]

    @classmethod
    def has_model(cls, model_name: str, device: str = "cpu") -> bool:
        """Return True if model instance is currently cached."""
        key = (model_name, device)
        with cls._lock:
            return key in cls._instances

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached model instances (useful for testing)."""
        with cls._lock:
            cls._instances.clear()
