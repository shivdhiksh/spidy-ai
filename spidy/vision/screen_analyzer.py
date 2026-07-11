"""
Screen Analyzer — Desktop State Detection
==========================================
Detects the active application, visible windows, and UI regions.

Capabilities
------------
- ``get_active_application()``  — foreground process name (e.g. "Code.exe")
- ``get_active_window_title()`` — title of the foreground window
- ``get_visible_windows()``     — list of all visible window titles
- ``detect_ui_regions()``       — basic contour-based region detection (opencv)
- ``analyze_screen()``          — full ScreenAnalysis dataclass

Graceful Degradation
--------------------
- ``win32gui`` (already a Spidy dep from M6) used for window enumeration.
  If somehow absent, psutil-based fallback is used.
- ``opencv-python`` is optional — when absent, ``detect_ui_regions()``
  returns an empty list; ``analyze_screen()`` still succeeds.
- ``is_available`` is True as long as either psutil or win32gui is present.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger
from spidy.vision.types import ScreenAnalysis, ScreenshotResult, UIRegion

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# ── Lazy optional imports ─────────────────────────────────────────────────────

try:
    import win32gui as _win32gui
    import win32process as _win32process
    _WIN32_AVAILABLE = True
except ImportError:
    _WIN32_AVAILABLE = False
    _win32gui = None     # type: ignore[assignment]
    _win32process = None  # type: ignore[assignment]

try:
    import psutil as _psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    _psutil = None  # type: ignore[assignment]

try:
    import cv2 as _cv2
    import numpy as _np
    _OPENCV_AVAILABLE = True
except ImportError:
    _OPENCV_AVAILABLE = False
    _cv2 = None  # type: ignore[assignment]
    _np = None   # type: ignore[assignment]


class ScreenAnalyzer:
    """
    Desktop state detector: active app, visible windows, UI regions.

    Parameters
    ----------
    min_region_area:
        Minimum pixel area for a contour to be counted as a UI region.
    max_regions:
        Maximum number of UI regions to return per analysis.
    """

    def __init__(
        self,
        min_region_area: int = 1000,
        max_regions: int = 20,
    ) -> None:
        self._min_region_area = min_region_area
        self._max_regions = max_regions

    # ── Availability ──────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True when at least psutil or win32gui is available."""
        return _WIN32_AVAILABLE or _PSUTIL_AVAILABLE

    @property
    def ocv_available(self) -> bool:
        """True when opencv is available for UI region detection."""
        return _OPENCV_AVAILABLE

    # ── Active Application ────────────────────────────────────────────────────

    def get_active_application(self) -> str:
        """
        Return the process name of the foreground application.

        Uses ``win32gui`` + ``win32process`` when available, falls back to
        ``psutil`` (which can only see the process list, not foreground state).

        Returns
        -------
        str
            Process name (e.g. ``"Code.exe"``) or ``""`` if unavailable.
        """
        if _WIN32_AVAILABLE:
            try:
                hwnd = _win32gui.GetForegroundWindow()
                if hwnd == 0:
                    return ""
                _, pid = _win32process.GetWindowThreadProcessId(hwnd)
                if pid == 0:
                    return ""
                if _PSUTIL_AVAILABLE:
                    proc = _psutil.Process(pid)
                    return proc.name()
                return str(pid)
            except Exception as exc:
                log.debug("get_active_application() win32 failed: {exc}", exc=exc)

        # psutil fallback: return the highest CPU process (rough heuristic)
        if _PSUTIL_AVAILABLE:
            try:
                procs = sorted(
                    _psutil.process_iter(["name", "cpu_percent"]),
                    key=lambda p: p.info.get("cpu_percent", 0.0) or 0.0,
                    reverse=True,
                )
                for p in procs[:5]:
                    name = p.info.get("name", "") or ""
                    if name and name not in ("System", "Registry", "Idle"):
                        return name
            except Exception as exc:
                log.debug("get_active_application() psutil fallback failed: {exc}", exc=exc)

        return ""

    def get_active_window_title(self) -> str:
        """
        Return the title of the foreground window.

        Returns
        -------
        str
            Window title string, or ``""`` if unavailable.
        """
        if not _WIN32_AVAILABLE:
            return ""
        try:
            hwnd = _win32gui.GetForegroundWindow()
            if hwnd == 0:
                return ""
            return _win32gui.GetWindowText(hwnd) or ""
        except Exception as exc:
            log.debug("get_active_window_title() failed: {exc}", exc=exc)
            return ""

    # ── Window Enumeration ────────────────────────────────────────────────────

    def get_visible_windows(self) -> list[str]:
        """
        Return titles of all currently visible top-level windows.

        Filters out windows with empty titles and common system windows.
        Windows are sorted alphabetically for deterministic output.

        Returns
        -------
        list[str]
            Window title strings. Empty list when win32gui unavailable.
        """
        if not _WIN32_AVAILABLE:
            return []

        windows: list[str] = []
        _SKIP = {"", "Program Manager", "Windows Shell Experience Host"}

        def _enum_handler(hwnd: int, _: None) -> bool:
            try:
                if not _win32gui.IsWindowVisible(hwnd):
                    return True
                title = _win32gui.GetWindowText(hwnd)
                if title and title not in _SKIP:
                    windows.append(title)
            except Exception:
                pass
            return True

        try:
            _win32gui.EnumWindows(_enum_handler, None)
        except Exception as exc:
            log.debug("get_visible_windows() EnumWindows failed: {exc}", exc=exc)

        return sorted(set(windows))

    # ── UI Region Detection ───────────────────────────────────────────────────

    def detect_ui_regions(self, image_data: bytes) -> list[UIRegion]:
        """
        Detect rectangular UI regions in a screenshot using contour analysis.

        Uses OpenCV Canny edge detection + contour finding. When OpenCV is
        not installed, returns an empty list (graceful degrade).

        Parameters
        ----------
        image_data:
            Raw PNG bytes from ScreenshotEngine.

        Returns
        -------
        list[UIRegion]
            Up to ``self._max_regions`` detected regions, sorted by area desc.
        """
        if not _OPENCV_AVAILABLE:
            log.debug("opencv not available — detect_ui_regions() returns empty.")
            return []

        try:
            # Decode PNG bytes → numpy array
            buf = _np.frombuffer(image_data, dtype=_np.uint8)  # type: ignore[union-attr]
            img = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)  # type: ignore[union-attr]
            if img is None:
                return []

            # Convert to grayscale → Gaussian blur → Canny edges
            gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)  # type: ignore[union-attr]
            blurred = _cv2.GaussianBlur(gray, (5, 5), 0)  # type: ignore[union-attr]
            edges = _cv2.Canny(blurred, 50, 150)  # type: ignore[union-attr]

            # Find contours
            contours, _ = _cv2.findContours(  # type: ignore[union-attr]
                edges, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE  # type: ignore[union-attr]
            )

            regions: list[UIRegion] = []
            for cnt in contours:
                area = _cv2.contourArea(cnt)  # type: ignore[union-attr]
                if area < self._min_region_area:
                    continue
                x, y, w, h = _cv2.boundingRect(cnt)  # type: ignore[union-attr]
                # Rough confidence: larger regions = higher confidence (normalised)
                confidence = min(1.0, area / (img.shape[0] * img.shape[1]))
                regions.append(UIRegion(
                    name="region",
                    bbox=(x, y, w, h),
                    confidence=round(confidence, 4),
                ))

            # Sort by area descending, cap at max_regions
            regions.sort(key=lambda r: r.bbox[2] * r.bbox[3], reverse=True)
            return regions[: self._max_regions]

        except Exception as exc:
            log.warning("detect_ui_regions() failed: {exc}", exc=exc)
            return []

    # ── Full Analysis ─────────────────────────────────────────────────────────

    def analyze_screen(
        self,
        screenshot: ScreenshotResult | None = None,
    ) -> ScreenAnalysis:
        """
        Perform a full analysis of the current screen state.

        Combines window detection, active app detection, and optionally
        UI region detection (if a screenshot is provided and opencv is present).

        Parameters
        ----------
        screenshot:
            Optional screenshot to detect UI regions from.
            If None, UI region detection is skipped.

        Returns
        -------
        ScreenAnalysis
            Immutable analysis dataclass.
        """
        active_app = self.get_active_application()
        active_window = self.get_active_window_title()
        visible_windows = self.get_visible_windows()

        ui_regions: list[UIRegion] = []
        if screenshot is not None and not screenshot.is_empty:
            ui_regions = self.detect_ui_regions(screenshot.image_data)

        analysis = ScreenAnalysis(
            active_app=active_app,
            active_window=active_window,
            visible_windows=tuple(visible_windows),
            ui_regions=tuple(ui_regions),
            timestamp=time.time(),
        )

        log.debug(
            "Screen analyzed | app={app} windows={wc} regions={rc}",
            app=active_app,
            wc=len(visible_windows),
            rc=len(ui_regions),
        )

        return analysis
