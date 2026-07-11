"""
Vision Events — EventBus Events for the Vision Engine
======================================================
Topic namespace: ``vision.*``

All event types are frozen dataclasses that extend ``Event``.
The ``VisionManager`` publishes these; any module may subscribe.

Topics
------
vision.capture_started      — a capture operation has begun
vision.capture_completed    — a capture operation finished successfully
vision.ocr_completed        — OCR was performed on a screenshot
vision.analysis_completed   — screen analysis was performed
vision.error                — any vision-layer error occurred
"""

from __future__ import annotations

from dataclasses import dataclass

from spidy.core.event_bus import Event


# ─── Vision Lifecycle Events ──────────────────────────────────────────────────


@dataclass
class VisionCaptureStartedEvent(Event):
    """Published when a screenshot capture is initiated."""

    topic = "vision.capture_started"
    source: str = "fullscreen"      # "fullscreen" | "window" | "region"
    monitor_index: int = 0


@dataclass
class VisionCaptureCompletedEvent(Event):
    """Published when a screenshot capture completes successfully."""

    topic = "vision.capture_completed"
    source: str = "fullscreen"      # "fullscreen" | "window" | "region"
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    duration_ms: float = 0.0        # Capture wall-clock time in milliseconds
    monitor_index: int = 0


@dataclass
class VisionOCRCompletedEvent(Event):
    """Published when OCR is performed on a screenshot or image."""

    topic = "vision.ocr_completed"
    text_length: int = 0            # Length of extracted text
    confidence: float = 0.0         # Mean confidence across all blocks
    block_count: int = 0            # Number of text blocks detected
    language: str = "en"
    available: bool = True          # False when OCR deps absent


@dataclass
class VisionAnalysisCompletedEvent(Event):
    """Published when a full screen analysis is performed."""

    topic = "vision.analysis_completed"
    active_app: str = ""
    active_window: str = ""
    window_count: int = 0
    region_count: int = 0
    available: bool = True


@dataclass
class VisionErrorEvent(Event):
    """Published when any vision-layer error occurs."""

    topic = "vision.error"
    operation: str = ""     # "capture" | "ocr" | "analyze" | "describe"
    error: str = ""
    source: str = ""        # Engine that failed
