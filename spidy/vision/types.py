"""
Vision Types — Data Structures for the Vision Engine
=====================================================
All types are frozen dataclasses for immutability and thread safety.

Types
-----
ScreenshotResult   — output of ScreenshotEngine captures
OCRBlock           — a single detected text block with bounding box
OCRResult          — full OCR output with blocks and confidence
UIRegion           — a detected UI region in the screenshot
ScreenAnalysis     — full screen analysis result
MonitorInfo        — information about a physical/virtual monitor
VisionDependencyError — raised when a required optional dep is absent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ─── Exception ────────────────────────────────────────────────────────────────


class VisionDependencyError(RuntimeError):
    """
    Raised when a Vision Engine operation requires an optional dependency
    that is not installed.

    Callers should catch this and degrade gracefully rather than crash.
    The ``VisionManager`` always catches this internally.
    """


# ─── Monitor Info ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MonitorInfo:
    """Information about a single physical or virtual monitor."""

    index: int = 0              # 0-based index
    width: int = 0              # resolution width in pixels
    height: int = 0             # resolution height in pixels
    x: int = 0                  # screen origin X
    y: int = 0                  # screen origin Y
    is_primary: bool = True     # True for the primary display

    def __str__(self) -> str:
        tag = " [primary]" if self.is_primary else ""
        return (
            f"Monitor[{self.index}] {self.width}×{self.height} "
            f"at ({self.x},{self.y}){tag}"
        )


# ─── Screenshot Result ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScreenshotResult:
    """
    Output of a screenshot capture operation.

    ``image_data`` is raw PNG bytes ready for OCR, display, or saving.
    """

    image_data: bytes = b""         # Raw PNG bytes
    width: int = 0                  # Image width in pixels
    height: int = 0                 # Image height in pixels
    monitor_index: int = 0          # Which monitor was captured
    region: Optional[tuple] = None  # (x, y, w, h) if a region was captured
    timestamp: float = 0.0          # Unix timestamp of capture
    source: str = "fullscreen"      # "fullscreen" | "window" | "region"

    @property
    def size_bytes(self) -> int:
        return len(self.image_data)

    @property
    def is_empty(self) -> bool:
        return len(self.image_data) == 0

    def __str__(self) -> str:
        return (
            f"Screenshot({self.source} {self.width}×{self.height} "
            f"monitor={self.monitor_index} {self.size_bytes}B)"
        )


# ─── OCR Result ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OCRBlock:
    """
    A single text block detected by the OCR engine.

    ``bbox`` is (x, y, width, height) in pixels relative to the image.
    """

    text: str = ""
    confidence: float = 0.0
    bbox: tuple = field(default_factory=lambda: (0, 0, 0, 0))  # (x, y, w, h)

    def __str__(self) -> str:
        return f"OCRBlock('{self.text[:40]}' conf={self.confidence:.2f})"


@dataclass(frozen=True)
class OCRResult:
    """
    Full output of an OCR operation on a screenshot or image file.

    ``blocks`` contains individual detected text regions.
    ``text`` is the concatenation of all block texts, suitable for
    injection into LLM context.
    """

    text: str = ""                              # Full extracted text
    confidence: float = 0.0                    # Mean confidence across blocks
    blocks: tuple = field(default_factory=tuple)  # tuple[OCRBlock, ...]
    language: str = "en"                        # Detected / configured language
    available: bool = True                      # False if OCR deps absent

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def __str__(self) -> str:
        return (
            f"OCRResult(blocks={self.block_count} conf={self.confidence:.2f} "
            f"chars={len(self.text)})"
        )


# ─── Screen Analysis ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UIRegion:
    """
    A detected UI region (button, panel, text area, etc.) in a screenshot.
    Detected via basic contour analysis or object detection.
    """

    name: str = "unknown"           # e.g. "window", "button", "text_field"
    bbox: tuple = field(default_factory=lambda: (0, 0, 0, 0))  # (x, y, w, h)
    confidence: float = 0.0

    def __str__(self) -> str:
        return f"UIRegion('{self.name}' bbox={self.bbox} conf={self.confidence:.2f})"


@dataclass(frozen=True)
class ScreenAnalysis:
    """
    Full analysis of the current screen state.

    Produced by ``ScreenAnalyzer.analyze_screen()``.
    """

    active_app: str = ""                                    # e.g. "Code.exe"
    active_window: str = ""                                 # window title
    visible_windows: tuple = field(default_factory=tuple)   # tuple[str, ...]
    ui_regions: tuple = field(default_factory=tuple)        # tuple[UIRegion, ...]
    timestamp: float = 0.0                                  # Unix timestamp

    @property
    def window_count(self) -> int:
        return len(self.visible_windows)

    @property
    def region_count(self) -> int:
        return len(self.ui_regions)

    def to_description(self) -> str:
        """Human-readable description for LLM context injection."""
        parts = []
        if self.active_app:
            parts.append(f"Active application: {self.active_app}")
        if self.active_window:
            parts.append(f"Active window: {self.active_window}")
        if self.visible_windows:
            parts.append(
                f"Visible windows ({self.window_count}): "
                + ", ".join(str(w) for w in list(self.visible_windows)[:5])
            )
        if self.ui_regions:
            parts.append(f"UI regions detected: {self.region_count}")
        return "\n".join(parts) if parts else "Screen analysis unavailable."

    def __str__(self) -> str:
        return (
            f"ScreenAnalysis(app='{self.active_app}' "
            f"windows={self.window_count} regions={self.region_count})"
        )
