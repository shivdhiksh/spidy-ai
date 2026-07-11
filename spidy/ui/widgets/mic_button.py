"""
MicrophoneButton — Animated microphone button
==============================================
A round button that:
- Shows purple when idle (clickable)
- Pulses red with an expanding ring when listening
- Dims and shows a spinner when thinking
- Is disabled (grey) when Spidy is speaking

Signal
------
clicked  Emitted when the user clicks the button.
"""

from __future__ import annotations

import math
import time

from PySide6.QtCore import (
    QPointF, QRectF, Qt, QTimer, Signal
)
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPainterPath, QPen
)
from PySide6.QtWidgets import QWidget

_FRAME_MS = 40  # ~25 fps for smooth pulse


class MicrophoneButton(QWidget):
    """
    Animated microphone button for the Spidy overlay.

    Appearance per state
    --------------------
    idle        Purple circle + white mic icon
    listening   Red circle + pulsing outer ring + white mic icon
    thinking    Muted grey circle + rotating arc spinner
    speaking    Muted dark circle + speaker wave icon
    disabled    Flat grey circle (non-interactive)
    """

    clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(56, 56)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._state = "idle"
        self._phase = 0.0
        self._spinner_angle = 0.0

        # Theme-driven colors (defaults = dark theme)
        self._color_idle = QColor("#6C63FF")
        self._color_listening = QColor("#FF4D6A")
        self._color_thinking = QColor("#444466")
        self._color_speaking = QColor("#444466")
        self._color_disabled = QColor("#2A2B45")
        self._color_icon = QColor("#FFFFFF")

        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_MS)
        self._timer.timeout.connect(self._tick)

    # ── Public API ────────────────────────────────────────────────────────

    def set_state(self, state: str) -> None:
        """
        Update the button visual state.

        Parameters
        ----------
        state : str
            One of: "idle" | "listening" | "thinking" | "speaking" | "disabled"
        """
        self._state = state
        is_animated = state in ("listening", "thinking")
        if is_animated and not self._timer.isActive():
            self._timer.start()
        elif not is_animated:
            self._timer.stop()
            self._phase = 0.0
            self._spinner_angle = 0.0
        self.setCursor(
            Qt.CursorShape.PointingHandCursor
            if state not in ("disabled", "speaking", "thinking")
            else Qt.CursorShape.ForbiddenCursor
        )
        self.update()

    def apply_theme(self, theme) -> None:
        """Update colors from a Theme object."""
        self._color_idle = QColor(theme.colors.mic_idle)
        self._color_listening = QColor(theme.colors.mic_listening)
        self._color_icon = QColor(theme.colors.text_on_accent)
        self.update()

    # ── Qt overrides ──────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        r = 20.0

        # ── Pulsing ring (listening state) ──────────────────────────────
        if self._state == "listening":
            pulse_r = r + 6.0 + 6.0 * math.sin(self._phase)
            ring_alpha = 0.4 * abs(math.sin(self._phase))
            ring_color = QColor(self._color_listening)
            ring_color.setAlphaF(ring_alpha)
            painter.setBrush(QBrush(ring_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(cx, cy), pulse_r, pulse_r)

        # ── Main circle ────────────────────────────────────────────────
        if self._state == "idle":
            fill = self._color_idle
        elif self._state == "listening":
            fill = self._color_listening
        elif self._state == "thinking":
            fill = self._color_thinking
        elif self._state == "speaking":
            fill = self._color_speaking
        else:
            fill = self._color_disabled

        painter.setBrush(QBrush(fill))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), r, r)

        # ── Icon ───────────────────────────────────────────────────────
        if self._state in ("idle", "listening"):
            self._draw_mic(painter, cx, cy)
        elif self._state == "thinking":
            self._draw_spinner(painter, cx, cy, r)
        elif self._state == "speaking":
            self._draw_speaker(painter, cx, cy)

    def mousePressEvent(self, event) -> None:
        if self._state in ("idle", "wake_ready"):
            self.clicked.emit()

    # ── Internal drawing ──────────────────────────────────────────────────

    def _draw_mic(self, painter: QPainter, cx: float, cy: float) -> None:
        pen = QPen(self._color_icon, 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(QBrush(self._color_icon))

        # Mic capsule
        mic_rect = QRectF(cx - 5.5, cy - 10.5, 11, 14)
        painter.drawRoundedRect(mic_rect, 5.5, 5.5)

        # Arc (mic stand)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        arc_rect = QRectF(cx - 9.5, cy - 1.0, 19, 16)
        painter.drawArc(arc_rect, 0, 180 * 16)

        # Stand
        painter.drawLine(int(cx), int(cy + 15), int(cx), int(cy + 19))
        painter.drawLine(int(cx - 4), int(cy + 19), int(cx + 4), int(cy + 19))

    def _draw_spinner(
        self, painter: QPainter, cx: float, cy: float, r: float
    ) -> None:
        """Rotating arc spinner for thinking state."""
        arc_r = r - 6
        pen = QPen(QColor(108, 99, 255, 200), 3.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        rect = QRectF(cx - arc_r, cy - arc_r, arc_r * 2, arc_r * 2)
        start_angle = int(self._spinner_angle * 16)
        span_angle = 100 * 16  # 100-degree arc
        painter.drawArc(rect, start_angle, span_angle)

    def _draw_speaker(self, painter: QPainter, cx: float, cy: float) -> None:
        """Speaker waves icon for speaking state."""
        pen = QPen(self._color_icon, 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Speaker body (simple path)
        body = QPainterPath()
        body.moveTo(cx - 7, cy - 5)
        body.lineTo(cx - 2, cy - 5)
        body.lineTo(cx + 4, cy - 10)
        body.lineTo(cx + 4, cy + 10)
        body.lineTo(cx - 2, cy + 5)
        body.lineTo(cx - 7, cy + 5)
        body.closeSubpath()
        painter.fillPath(body, QBrush(self._color_icon))

        # Sound waves
        for i, arc_r in enumerate([10.0, 14.0]):
            wave_pen = QPen(self._color_icon, 1.5 + i * 0.5)
            wave_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(wave_pen)
            rect = QRectF(cx + 2, cy - arc_r / 2, arc_r, arc_r)
            painter.drawArc(rect, -30 * 16, 60 * 16)

    # ── Animation ─────────────────────────────────────────────────────────

    def _tick(self) -> None:
        self._phase += 0.18
        self._spinner_angle = (self._spinner_angle + 8.0) % 360.0
        self.update()
