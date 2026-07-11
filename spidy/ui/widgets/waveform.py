"""
WaveformWidget — Animated audio waveform bars
==============================================
Renders 20 animated vertical bars that visualise audio amplitude.

Two modes
---------
Simulated   No real data available. Bars animate with a flowing sine
            wave to indicate "listening" state visually.
Data-driven Real amplitude data (e.g. from microphone FFT) supplied
            via set_amplitudes(). Bars smoothly interpolate to new values.

Animation
---------
Driven by a 50ms QTimer (20 fps). Each bar uses exponential smoothing
toward its target value so rapid amplitude changes look fluid, not jittery.
"""

from __future__ import annotations

import math
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath
from PySide6.QtWidgets import QWidget

_BAR_COUNT = 20
_FRAME_MS = 50         # ~20 fps
_SMOOTH = 0.25         # Exponential smoothing factor (0 = no smooth, 1 = snap)


class WaveformWidget(QWidget):
    """
    Animated waveform visualiser for the Spidy overlay.

    Usage
    -----
        waveform = WaveformWidget(parent)
        waveform.set_active(True)            # Start animation (simulated)
        waveform.set_amplitudes([0.2, 0.8, 0.5, ...])  # Feed real data
        waveform.set_active(False)           # Stop and collapse bars
        waveform.apply_theme(theme)          # Update colors
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(48)
        self.setMinimumWidth(160)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._bars: list[float] = [0.0] * _BAR_COUNT
        self._targets: list[float] = [0.0] * _BAR_COUNT
        self._simulated = True          # Use sine wave when no real data
        self._active = False
        self._bar_color = QColor("#6C63FF")

        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_MS)
        self._timer.timeout.connect(self._tick)

    # ── Public API ────────────────────────────────────────────────────────

    def set_active(self, active: bool) -> None:
        """Start or stop the animation."""
        self._active = active
        if active:
            self._timer.start()
        else:
            self._timer.stop()
            # Smoothly collapse bars to zero
            self._targets = [0.0] * _BAR_COUNT
            self.update()

    def set_amplitudes(self, amplitudes: list[float]) -> None:
        """
        Feed real audio amplitude data.

        Parameters
        ----------
        amplitudes:
            List of floats in [0.0, 1.0].  Length can vary; will be
            resampled to fit _BAR_COUNT bars.
        """
        if not amplitudes:
            self._simulated = True
            return
        self._simulated = False
        # Resample to _BAR_COUNT
        n = len(amplitudes)
        resampled = [
            amplitudes[int(i * n / _BAR_COUNT)]
            for i in range(_BAR_COUNT)
        ]
        self._targets = [max(0.0, min(1.0, v)) for v in resampled]

    def apply_theme(self, theme) -> None:
        """Update bar color from a Theme object."""
        self._bar_color = QColor(theme.colors.waveform_bar)
        self.update()

    # ── Qt overrides ──────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        slot_w = w / _BAR_COUNT
        bar_w = max(2.0, slot_w * 0.65)
        gap = (slot_w - bar_w) / 2

        for i, amp in enumerate(self._bars):
            bar_h = max(4.0, amp * (h - 8))
            x = i * slot_w + gap
            y = (h - bar_h) / 2

            # Colour intensity follows amplitude
            color = QColor(self._bar_color)
            color.setAlphaF(0.25 + amp * 0.75)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)

            r = min(bar_w / 2, 4.0)
            painter.drawRoundedRect(
                QRectF(x, y, bar_w, bar_h), r, r
            )

    # ── Animation ─────────────────────────────────────────────────────────

    def _tick(self) -> None:
        t = time.monotonic()
        if self._simulated and self._active:
            # Generate sine wave pattern
            for i in range(_BAR_COUNT):
                phase = t * 4.0 + i * 0.45
                self._targets[i] = 0.25 + 0.55 * (math.sin(phase) * 0.5 + 0.5)

        # Exponential smoothing toward targets
        changed = False
        for i in range(_BAR_COUNT):
            diff = self._targets[i] - self._bars[i]
            if abs(diff) > 0.001:
                self._bars[i] += diff * _SMOOTH
                changed = True

        if changed or self._active:
            self.update()
