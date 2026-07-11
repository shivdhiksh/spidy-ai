"""
spidy.vision — Vision & Screen Understanding Engine
=====================================================
Milestone 9: Vision Engine

Public API
----------
    from spidy.vision import VisionManager
    from spidy.vision.types import ScreenshotResult, OCRResult, ScreenAnalysis
    from spidy.vision.events import VisionCaptureCompletedEvent

Architecture
------------
    VisionManager  (implements VisionInterface)
        ├── ScreenshotEngine   ultra-fast mss-based capture
        ├── OCREngine          easyocr text extraction (lazy init)
        └── ScreenAnalyzer     win32gui + opencv region detection

All optional dependencies (mss, easyocr, opencv-python) degrade
gracefully when not installed.
"""

from spidy.vision.events import (
    VisionAnalysisCompletedEvent,
    VisionCaptureCompletedEvent,
    VisionCaptureStartedEvent,
    VisionErrorEvent,
    VisionOCRCompletedEvent,
)
from spidy.vision.manager import VisionManager
from spidy.vision.ocr import OCREngine
from spidy.vision.screen_analyzer import ScreenAnalyzer
from spidy.vision.screenshot import ScreenshotEngine
from spidy.vision.types import (
    MonitorInfo,
    OCRBlock,
    OCRResult,
    ScreenAnalysis,
    ScreenshotResult,
    UIRegion,
    VisionDependencyError,
)

__all__ = [
    # Manager
    "VisionManager",
    # Engines
    "ScreenshotEngine",
    "OCREngine",
    "ScreenAnalyzer",
    # Types
    "ScreenshotResult",
    "OCRResult",
    "OCRBlock",
    "ScreenAnalysis",
    "UIRegion",
    "MonitorInfo",
    "VisionDependencyError",
    # Events
    "VisionCaptureStartedEvent",
    "VisionCaptureCompletedEvent",
    "VisionOCRCompletedEvent",
    "VisionAnalysisCompletedEvent",
    "VisionErrorEvent",
]
