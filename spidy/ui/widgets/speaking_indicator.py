"""
SpeakingIndicator — Three animated pulsing dots
=================================================
Classic "Spidy is typing/speaking" indicator:
three dots that animate with staggered phase offsets
to give a smooth wave-like pulsing effect.

Used
----
- During THINKING state: "Processing…" label + dots
- During SPEAKING state: "Speaking…" label + dots
"""

from __future__ import annotations

import math
import time

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QPainter
from PySide6.QtWidgets import QWidget

_FRAME_MS = 40   # ~25 fps
_DOT_COUNT = 3
_DOT_RADIUS = 5.0
_DOT_SPACING = 18.0


class SpeakingIndicator(QWidget):
    """
    Three-dot pulsing animation widget.

    The dots use different phase offsets so they appear to "wave",
    giving Spidy a lively, attentive appearance.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(24)
        self.setMinimumWidth(60)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._phase = 0.0
        self._color = QColor("#6C63FF")

        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_MS)
        self._timer.timeout.connect(self._tick)

    # ── Public API ────────────────────────────────────────────────────────

    def set_active(self, active: bool) -> None:
        if active and not self._timer.isActive():
            self._timer.start()
        elif not active:
            self._timer.stop()
            self._phase = 0.0
        self.update()

    def apply_theme(self, theme) -> None:
        """Update dot color from a Theme object."""
        self._color = QColor(theme.colors.thinking_dot)
        self.update()

    # ── Qt overrides ──────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        w = self.width()
        h = self.height()
        cx = w / 2.0
        cy = h / 2.0

        total_w = (_DOT_COUNT - 1) * _DOT_SPACING
        start_x = cx - total_w / 2.0

        for i in range(_DOT_COUNT):
            phase_offset = i * (math.pi * 2 / 3)
            amp = math.sin(self._phase + phase_offset)
            # Scale from 0.4 to 1.0
            scale = 0.4 + 0.6 * (amp * 0.5 + 0.5)

            dot_r = _DOT_RADIUS * scale
            # Vertical offset for the wave effect
            vert = -4.0 * math.sin(self._phase + phase_offset)

            color = QColor(self._color)
            color.setAlphaF(0.35 + 0.65 * scale)
            painter.setBrush(QBrush(color))

            x = start_x + i * _DOT_SPACING
            painter.drawEllipse(QPointF(x, cy + vert), dot_r, dot_r)

        painter.end()

    # ── Animation ─────────────────────────────────────────────────────────

    def _tick(self) -> None:
        self._phase = (self._phase + 0.15) % (math.pi * 2)
        self.update()
