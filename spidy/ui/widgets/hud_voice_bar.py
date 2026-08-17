"""
hud_voice_bar.py -- HUDVoiceBar: Futuristic voice visualizer
=============================================================
Bottom center element -- reacts to amplitude.

IDLE:     near-static
LISTENING: cyan bars react to mic amplitude
SPEAKING: teal bars react to TTS playback

20 fps when active, 5 fps when idle.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

if TYPE_CHECKING:
    from spidy.ui.themes.base import Theme


_NUM_BARS = 24
_FPS_ACTIVE = 50  # ms -> 20fps
_FPS_IDLE   = 200 # ms -> 5fps


class HUDVoiceBar(QWidget):
    """Futuristic bar visualizer for voice activity."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._state     = "idle"
        self._amplitude = 0.0
        self._bars      = [0.0] * _NUM_BARS
        self._phase     = 0.0

        self._c_primary   = QColor("#00E5FF")
        self._c_secondary = QColor("#00ACC1")
        self._c_dim       = QColor("#004D5E")

        self._timer = QTimer(self)
        self._timer.setInterval(_FPS_IDLE)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_state(self, state: str) -> None:
        self._state = state
        ms = _FPS_ACTIVE if state in ("listening", "speaking") else _FPS_IDLE
        if self._timer.interval() != ms:
            self._timer.setInterval(ms)
        self.update()

    def set_amplitude(self, amp: float) -> None:
        self._amplitude = max(0.0, min(1.0, amp))

    def apply_theme(self, theme: "Theme") -> None:
        c = theme.colors
        self._c_primary   = QColor(c.hud_primary)
        self._c_secondary = QColor(c.hud_secondary)
        self._c_dim       = QColor(c.hud_dim)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        bar_w = w / _NUM_BARS * 0.6
        gap   = w / _NUM_BARS * 0.4
        max_h = h * 0.85
        base_y = h * 0.92

        for i, val in enumerate(self._bars):
            bar_h = max(2, val * max_h)
            x = (w / _NUM_BARS) * i + gap * 0.5
            y = base_y - bar_h

            # Gradient per bar
            grad = QLinearGradient(x, y + bar_h, x, y)
            base_col = self._c_primary if self._state == "listening" else self._c_secondary
            top_col = QColor(base_col)
            top_col.setAlphaF(0.90 * val + 0.10)
            bot_col = QColor(base_col)
            bot_col.setAlphaF(0.20)
            grad.setColorAt(0, bot_col)
            grad.setColorAt(1, top_col)

            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(x, y, bar_w, bar_h), 1.5, 1.5)

            # Top glow dot
            if val > 0.3:
                dot = QColor(base_col)
                dot.setAlphaF(val * 0.8)
                painter.setBrush(QBrush(dot))
                painter.drawEllipse(QRectF(x + bar_w * 0.2, y - 1.5, bar_w * 0.6, 3))

        # Center line
        line_col = QColor(self._c_dim)
        line_col.setAlphaF(0.35)
        painter.setPen(QPen(line_col, 0.8))
        painter.drawLine(QRectF(0, base_y, w, 0).topLeft(), QRectF(0, base_y, w, 0).topRight())
        painter.end()

    def _tick(self) -> None:
        state = self._state
        amp = self._amplitude
        self._phase += 0.12

        for i in range(_NUM_BARS):
            pos = i / _NUM_BARS
            if state == "listening":
                wave = math.sin(self._phase + pos * math.pi * 2) * 0.3
                base = amp * (0.5 + random.gauss(0, 0.15))
                self._bars[i] = max(0.05, min(1.0, base + wave))
            elif state == "speaking":
                wave = math.sin(self._phase * 1.3 + pos * math.pi * 3) * 0.25
                base = amp * (0.4 + random.gauss(0, 0.10))
                self._bars[i] = max(0.05, min(1.0, base + wave))
            else:
                # Idle: gentle sine
                target = 0.05 + 0.03 * math.sin(self._phase * 0.3 + pos * math.pi)
                self._bars[i] = self._bars[i] * 0.85 + target * 0.15

        self.update()
