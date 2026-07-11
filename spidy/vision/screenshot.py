"""
Screenshot Engine — Screen Capture Backend
==========================================
Provides fast, multi-monitor screenshot capture using ``mss``.

Supported capture modes
-----------------------
- Full screen (one monitor by index)
- Active window (win32gui region crop)
- Arbitrary region (x, y, width, height)
- Monitor enumeration

Graceful Degradation
--------------------
``mss`` is an optional dependency. When absent:
  - ``is_available`` returns ``False``
  - All capture methods raise ``VisionDependencyError``
  - ``get_monitors()`` returns a single synthetic MonitorInfo

The ``VisionManager`` catches ``VisionDependencyError`` and never crashes.

Usage (when deps available)
---------------------------
    engine = ScreenshotEngine()
    if engine.is_available:
        result = engine.capture_fullscreen(monitor_index=0)
        print(result)   # Screenshot(fullscreen 1920×1080 monitor=0 ...)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger
from spidy.vision.types import MonitorInfo, ScreenshotResult, VisionDependencyError

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# ── Lazy optional imports ─────────────────────────────────────────────────────
# We try to import mss at module load. If absent, _MSS_AVAILABLE is False
# and all capture methods raise VisionDependencyError.

try:
    import mss as _mss_module
    import mss.tools as _mss_tools
    _MSS_AVAILABLE = True
except ImportError:
    _MSS_AVAILABLE = False
    _mss_module = None   # type: ignore[assignment]
    _mss_tools = None    # type: ignore[assignment]

# win32gui for active-window capture (already a dep from M6)
try:
    import win32gui as _win32gui
    import win32con as _win32con
    _WIN32_AVAILABLE = True
except ImportError:
    _WIN32_AVAILABLE = False
    _win32gui = None   # type: ignore[assignment]
    _win32con = None   # type: ignore[assignment]


class ScreenshotEngine:
    """
    Ultra-fast screen capture engine backed by ``mss``.

    All methods are synchronous — they should be called from a thread
    executor (e.g. ``asyncio.to_thread``) when used inside async code,
    or directly in the ``VisionManager`` which wraps them.

    Parameters
    ----------
    default_monitor:
        Default monitor index for ``capture_fullscreen()``. 0 = primary.
    capture_format:
        Image output format. "PNG" is the default and is lossless.
    """

    def __init__(
        self,
        default_monitor: int = 0,
        capture_format: str = "PNG",
    ) -> None:
        self._default_monitor = default_monitor
        self._capture_format = capture_format.upper()

    # ── Availability ──────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True when ``mss`` is installed and screenshots can be taken."""
        return _MSS_AVAILABLE

    # ── Monitor Enumeration ───────────────────────────────────────────────────

    def get_monitors(self) -> list[MonitorInfo]:
        """
        Return information about all available monitors.

        When ``mss`` is not installed, returns a single synthetic
        MonitorInfo representing a generic 1920×1080 primary display.

        Returns
        -------
        list[MonitorInfo]
            One entry per physical/virtual monitor.
        """
        if not _MSS_AVAILABLE:
            log.debug("mss not available — returning synthetic monitor info.")
            return [MonitorInfo(index=0, width=1920, height=1080, x=0, y=0, is_primary=True)]

        monitors: list[MonitorInfo] = []
        try:
            with _mss_module.mss() as sct:
                # mss.monitors[0] is the combined virtual screen; [1:] are real monitors
                real_monitors = sct.monitors[1:]
                for idx, mon in enumerate(real_monitors):
                    monitors.append(MonitorInfo(
                        index=idx,
                        width=mon["width"],
                        height=mon["height"],
                        x=mon["left"],
                        y=mon["top"],
                        is_primary=(idx == 0),
                    ))
        except Exception as exc:
            log.warning("get_monitors() failed: {exc}", exc=exc)
            monitors = [MonitorInfo(index=0, width=1920, height=1080, x=0, y=0, is_primary=True)]

        return monitors

    # ── Capture Methods ───────────────────────────────────────────────────────

    def capture_fullscreen(self, monitor_index: int | None = None) -> ScreenshotResult:
        """
        Capture the full contents of one monitor.

        Parameters
        ----------
        monitor_index:
            0-based monitor index. Defaults to ``self._default_monitor``.

        Returns
        -------
        ScreenshotResult
            PNG bytes, dimensions, and metadata.

        Raises
        ------
        VisionDependencyError
            If ``mss`` is not installed.
        """
        if not _MSS_AVAILABLE:
            raise VisionDependencyError(
                "mss is not installed. Install with: pip install mss"
            )

        idx = monitor_index if monitor_index is not None else self._default_monitor
        t_start = time.perf_counter()

        try:
            with _mss_module.mss() as sct:
                # mss.monitors[0] is virtual, real monitors start at [1]
                real_monitors = sct.monitors
                if len(real_monitors) < 2:
                    monitor_spec = real_monitors[0]
                else:
                    safe_idx = min(idx + 1, len(real_monitors) - 1)
                    monitor_spec = real_monitors[safe_idx]

                raw = sct.grab(monitor_spec)
                png_bytes = _mss_tools.to_png(raw.rgb, raw.size)

        except Exception as exc:
            log.error("capture_fullscreen() failed: {exc}", exc=exc)
            raise VisionDependencyError(f"Screenshot failed: {exc}") from exc

        duration = (time.perf_counter() - t_start) * 1000
        log.debug(
            "Full screen captured | monitor={idx} {w}×{h} in {ms:.1f}ms",
            idx=idx,
            w=raw.width,
            h=raw.height,
            ms=duration,
        )

        return ScreenshotResult(
            image_data=png_bytes,
            width=raw.width,
            height=raw.height,
            monitor_index=idx,
            region=None,
            timestamp=time.time(),
            source="fullscreen",
        )

    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        monitor_index: int = 0,
    ) -> ScreenshotResult:
        """
        Capture a rectangular region of the screen.

        Parameters
        ----------
        x, y:
            Top-left corner in screen coordinates.
        width, height:
            Dimensions of the region in pixels.
        monitor_index:
            Which monitor the region belongs to (metadata only).

        Raises
        ------
        VisionDependencyError
            If ``mss`` is not installed.
        """
        if not _MSS_AVAILABLE:
            raise VisionDependencyError(
                "mss is not installed. Install with: pip install mss"
            )

        t_start = time.perf_counter()
        region = {"left": x, "top": y, "width": width, "height": height}

        try:
            with _mss_module.mss() as sct:
                raw = sct.grab(region)
                png_bytes = _mss_tools.to_png(raw.rgb, raw.size)
        except Exception as exc:
            log.error("capture_region() failed: {exc}", exc=exc)
            raise VisionDependencyError(f"Region screenshot failed: {exc}") from exc

        duration = (time.perf_counter() - t_start) * 1000
        log.debug(
            "Region captured | ({x},{y}) {w}×{h} in {ms:.1f}ms",
            x=x, y=y, w=width, h=height, ms=duration,
        )

        return ScreenshotResult(
            image_data=png_bytes,
            width=raw.width,
            height=raw.height,
            monitor_index=monitor_index,
            region=(x, y, width, height),
            timestamp=time.time(),
            source="region",
        )

    def capture_active_window(self) -> ScreenshotResult:
        """
        Capture the currently focused window's screen region.

        Uses ``win32gui`` to get the window rect, then captures that region
        via ``mss``. Falls back to full-screen capture when ``win32gui``
        is not available.

        Raises
        ------
        VisionDependencyError
            If ``mss`` is not installed.
        """
        if not _MSS_AVAILABLE:
            raise VisionDependencyError(
                "mss is not installed. Install with: pip install mss"
            )

        if _WIN32_AVAILABLE:
            try:
                hwnd = _win32gui.GetForegroundWindow()
                rect = _win32gui.GetWindowRect(hwnd)
                x, y, x2, y2 = rect
                w = max(1, x2 - x)
                h = max(1, y2 - y)

                t_start = time.perf_counter()
                region = {"left": x, "top": y, "width": w, "height": h}
                with _mss_module.mss() as sct:
                    raw = sct.grab(region)
                    png_bytes = _mss_tools.to_png(raw.rgb, raw.size)

                duration = (time.perf_counter() - t_start) * 1000
                log.debug(
                    "Active window captured | ({x},{y}) {w}×{h} in {ms:.1f}ms",
                    x=x, y=y, w=w, h=h, ms=duration,
                )

                return ScreenshotResult(
                    image_data=png_bytes,
                    width=raw.width,
                    height=raw.height,
                    monitor_index=0,
                    region=(x, y, w, h),
                    timestamp=time.time(),
                    source="window",
                )

            except Exception as exc:
                log.warning(
                    "Active window capture failed, falling back to fullscreen: {exc}",
                    exc=exc,
                )

        # Fallback: full-screen capture
        log.debug("Using fullscreen fallback for active-window capture.")
        result = self.capture_fullscreen()
        # Return with source="window" so callers know the intent
        return ScreenshotResult(
            image_data=result.image_data,
            width=result.width,
            height=result.height,
            monitor_index=result.monitor_index,
            region=result.region,
            timestamp=result.timestamp,
            source="window",
        )
