"""
Vision Manager — Unified Vision API
=====================================
``VisionManager`` is the single entry point for all vision operations.
It implements the ``VisionInterface`` contract and wraps all three
vision engines behind one clean async API.

Architecture
------------
    VisionManager  (VisionInterface)
        ├── ScreenshotEngine   ultra-fast mss-based capture
        ├── OCREngine          easyocr text extraction (lazy init)
        └── ScreenAnalyzer     win32gui + opencv region detection

All Brain interactions go through VisionManager.
The Brain never touches mss, easyocr, or opencv directly.

EventBus Integration
--------------------
Every public operation publishes a corresponding ``vision.*`` event:
    capture()          → VisionCaptureStartedEvent + VisionCaptureCompletedEvent
    read_screen_text() → VisionOCRCompletedEvent
    analyze_screen()   → VisionAnalysisCompletedEvent
    (errors)           → VisionErrorEvent

The EventBus is optional — if no bus is passed, events are silently skipped.
This keeps unit tests simple (no bus required).

Brain API
---------
The Brain's convenience method is ``describe_screen()``, which returns a
human-readable summary of the current screen state for LLM context injection.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from spidy.brain.interfaces import VisionInterface
from spidy.logging.logger import get_logger
from spidy.vision.events import (
    VisionAnalysisCompletedEvent,
    VisionCaptureCompletedEvent,
    VisionCaptureStartedEvent,
    VisionErrorEvent,
    VisionOCRCompletedEvent,
)
from spidy.vision.ocr import OCREngine
from spidy.vision.screen_analyzer import ScreenAnalyzer
from spidy.vision.screenshot import ScreenshotEngine
from spidy.vision.types import OCRResult, ScreenAnalysis, ScreenshotResult, VisionDependencyError

if TYPE_CHECKING:
    from spidy.config.manager import VisionConfig
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)


class VisionManager(VisionInterface):
    """
    Unified vision API that coordinates all three vision engines.

    Parameters
    ----------
    config:
        ``VisionConfig`` from ``SpidyConfig.vision``. May be None for defaults.
    bus:
        Optional EventBus for publishing vision events.
        When None, events are silently skipped.
    """

    def __init__(
        self,
        config: "VisionConfig | None" = None,
        bus: "EventBus | None" = None,
    ) -> None:
        self._config = config
        self._bus = bus
        self._initialized = False

        # Extract sub-config values (with defaults if config is None)
        screenshot_enabled = True
        ocr_enabled = True
        analysis_enabled = True
        default_monitor = 0
        ocr_language = "en"
        ocr_confidence = 0.5
        ocr_gpu = False

        if config is not None:
            screenshot_enabled = config.enable_screenshot
            ocr_enabled = config.enable_ocr
            analysis_enabled = config.enable_screen_analysis
            default_monitor = config.screenshot.default_monitor
            ocr_language = config.ocr.language
            ocr_confidence = config.ocr.confidence_threshold
            ocr_gpu = config.ocr.gpu

        # Lazily constructed — only built if enabled
        self._screenshot_engine: ScreenshotEngine | None = None
        self._ocr_engine: OCREngine | None = None
        self._screen_analyzer: ScreenAnalyzer | None = None

        self._screenshot_enabled = screenshot_enabled
        self._ocr_enabled = ocr_enabled
        self._analysis_enabled = analysis_enabled
        self._default_monitor = default_monitor
        self._ocr_language = ocr_language
        self._ocr_confidence = ocr_confidence
        self._ocr_gpu = ocr_gpu

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Initialize the VisionManager and its sub-engines.

        Sub-engines are created here (lazy for OCR model loading).
        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._initialized:
            return

        if self._screenshot_enabled:
            self._screenshot_engine = ScreenshotEngine(
                default_monitor=self._default_monitor,
            )
            log.debug(
                "ScreenshotEngine ready (available={avail})",
                avail=self._screenshot_engine.is_available,
            )

        if self._ocr_enabled:
            self._ocr_engine = OCREngine(
                language=self._ocr_language,
                confidence_threshold=self._ocr_confidence,
                gpu=self._ocr_gpu,
            )
            log.debug(
                "OCREngine ready (available={avail})",
                avail=self._ocr_engine.is_available,
            )

        if self._analysis_enabled:
            self._screen_analyzer = ScreenAnalyzer()
            log.debug(
                "ScreenAnalyzer ready (available={avail})",
                avail=self._screen_analyzer.is_available,
            )

        self._initialized = True
        log.info(
            "VisionManager initialized | screenshot={ss} ocr={ocr} analysis={an}",
            ss=self._screenshot_enabled,
            ocr=self._ocr_enabled,
            an=self._analysis_enabled,
        )

    async def close(self) -> None:
        """Release any held resources (currently no persistent handles)."""
        self._initialized = False
        log.debug("VisionManager closed.")

    # ── Engine Properties ─────────────────────────────────────────────────────

    @property
    def screenshot_engine(self) -> ScreenshotEngine | None:
        return self._screenshot_engine

    @property
    def ocr_engine(self) -> OCREngine | None:
        return self._ocr_engine

    @property
    def screen_analyzer(self) -> ScreenAnalyzer | None:
        return self._screen_analyzer

    # ── VisionInterface Implementation ────────────────────────────────────────

    async def capture(
        self,
        source: str = "fullscreen",
        monitor_index: int | None = None,
        x: int = 0,
        y: int = 0,
        width: int = 0,
        height: int = 0,
        **kwargs: Any,
    ) -> ScreenshotResult:
        """
        Take a screenshot.

        Parameters
        ----------
        source:
            ``"fullscreen"`` | ``"window"`` | ``"region"``
        monitor_index:
            Monitor to capture (fullscreen mode only).
        x, y, width, height:
            Region coordinates (region mode only).

        Returns
        -------
        ScreenshotResult
            PNG bytes + metadata. Returns an empty result on failure.
        """
        await self._ensure_initialized()

        if self._screenshot_engine is None:
            log.warning("ScreenshotEngine not enabled — returning empty result.")
            return ScreenshotResult(source=source)

        # Publish start event
        await self._publish(VisionCaptureStartedEvent(
            source=source,
            monitor_index=monitor_index or self._default_monitor,
        ))

        t_start = time.perf_counter()

        try:
            # Run sync capture in thread executor (mss may block briefly)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._do_capture,
                source,
                monitor_index,
                x, y, width, height,
            )
        except VisionDependencyError as exc:
            log.warning("Vision capture failed (deps): {exc}", exc=exc)
            await self._publish(VisionErrorEvent(
                operation="capture",
                error=str(exc),
                source=source,
            ))
            return ScreenshotResult(source=source)
        except Exception as exc:
            log.error("Vision capture failed: {exc}", exc=exc)
            await self._publish(VisionErrorEvent(
                operation="capture",
                error=str(exc),
                source=source,
            ))
            return ScreenshotResult(source=source)

        duration_ms = (time.perf_counter() - t_start) * 1000

        await self._publish(VisionCaptureCompletedEvent(
            source=result.source,
            width=result.width,
            height=result.height,
            size_bytes=result.size_bytes,
            duration_ms=round(duration_ms, 2),
            monitor_index=result.monitor_index,
        ))

        return result

    def _do_capture(
        self,
        source: str,
        monitor_index: int | None,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> ScreenshotResult:
        """Synchronous capture dispatch (runs in thread executor)."""
        engine = self._screenshot_engine
        assert engine is not None

        if source == "region" and width > 0 and height > 0:
            return engine.capture_region(x, y, width, height, monitor_index or 0)
        elif source == "window":
            return engine.capture_active_window()
        else:
            return engine.capture_fullscreen(monitor_index)

    async def read_screen_text(self) -> OCRResult:
        """
        Capture the screen and extract all visible text via OCR.

        Returns
        -------
        OCRResult
            Extracted text with confidence. Returns empty result on failure
            or when OCR deps are absent.
        """
        await self._ensure_initialized()

        if self._ocr_engine is None:
            log.warning("OCREngine not enabled — returning empty OCRResult.")
            return OCRResult(available=False)

        # First capture the screen
        screenshot = await self.capture(source="fullscreen")
        if screenshot.is_empty:
            return OCRResult(available=False)

        try:
            # Run sync OCR in thread executor (may take several seconds)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._ocr_engine.read_image,
                screenshot.image_data,
            )
        except Exception as exc:
            log.error("Vision OCR failed: {exc}", exc=exc)
            await self._publish(VisionErrorEvent(
                operation="ocr",
                error=str(exc),
                source="ocr_engine",
            ))
            return OCRResult(available=False)

        await self._publish(VisionOCRCompletedEvent(
            text_length=len(result.text),
            confidence=round(result.confidence, 4),
            block_count=result.block_count,
            language=result.language,
            available=result.available,
        ))

        return result

    async def analyze_screen(self) -> ScreenAnalysis:
        """
        Perform a full analysis of the current screen state.

        Detects the active application, visible windows, and (if opencv
        is available) UI regions from a screenshot.

        Returns
        -------
        ScreenAnalysis
            Immutable analysis dataclass.
        """
        await self._ensure_initialized()

        if self._screen_analyzer is None:
            log.warning("ScreenAnalyzer not enabled — returning empty ScreenAnalysis.")
            return ScreenAnalysis(timestamp=time.time())

        # Optionally grab a screenshot for UI region detection
        screenshot: ScreenshotResult | None = None
        if self._screenshot_engine is not None and self._screenshot_engine.is_available:
            try:
                screenshot = await self.capture(source="fullscreen")
            except Exception:
                pass  # Region detection will be skipped

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._screen_analyzer.analyze_screen,
                screenshot,
            )
        except Exception as exc:
            log.error("Vision screen analysis failed: {exc}", exc=exc)
            await self._publish(VisionErrorEvent(
                operation="analyze",
                error=str(exc),
                source="screen_analyzer",
            ))
            return ScreenAnalysis(timestamp=time.time())

        await self._publish(VisionAnalysisCompletedEvent(
            active_app=result.active_app,
            active_window=result.active_window,
            window_count=result.window_count,
            region_count=result.region_count,
            available=self._screen_analyzer.is_available,
        ))

        return result

    async def describe_screen(self) -> str:
        """
        Return a human-readable description of the current screen state.

        This is the primary method for Brain context injection.
        Combines screen analysis and OCR text into one LLM-ready string.

        Returns
        -------
        str
            Multi-line description, or a fallback message on failure.
        """
        await self._ensure_initialized()

        parts: list[str] = []

        try:
            analysis = await self.analyze_screen()
            description = analysis.to_description()
            if description:
                parts.append(description)
        except Exception as exc:
            log.warning("describe_screen: analysis failed: {exc}", exc=exc)

        try:
            ocr_result = await self.read_screen_text()
            if ocr_result.text.strip():
                preview = ocr_result.text[:500]
                parts.append(f"Visible text (OCR):\n{preview}")
        except Exception as exc:
            log.warning("describe_screen: OCR failed: {exc}", exc=exc)

        if not parts:
            return "Screen state could not be determined."

        return "\n\n".join(parts)

    # ── Private Helpers ───────────────────────────────────────────────────────

    async def _ensure_initialized(self) -> None:
        """Auto-initialize on first call if not already done."""
        if not self._initialized:
            await self.initialize()

    async def _publish(self, event: Any) -> None:
        """Publish an event to the bus, silently ignoring if no bus."""
        if self._bus is not None:
            try:
                await self._bus.publish(event)
            except Exception as exc:
                log.debug("VisionManager: event publish failed: {exc}", exc=exc)
