"""
SpidyOrb — Central animated Spidy visual element
=================================================
The orb is the primary visual focus of the Spidy overlay.
It renders entirely with QPainter so it works without any image files.

When the final Spidy logo is available, drop it into:
    assets/icons/spidy_logo.png   (128x128 or larger PNG)
The orb will automatically use it as the centerpiece.

Visual design
-------------
Three layered rings create depth:
  - Outer ring: slow breathing glow (always present, intensity varies by state)
  - Middle ring: frosted glass circle (the body of the orb)
  - Inner ring: tight accent ring whose color changes per state
  - Centre: logo or lightning symbol

State mappings
--------------
IDLE        Slow purple breathe (3s cycle), dim glow
WAKE_READY  Slightly brighter purple, faster breathe
LISTENING   Green outer glow, moderate pulse
THINKING    Blue rotating arc on outer ring
SPEAKING    Warm amber pulse
WORKING     Gold outer ring, faster independent breathe
ERROR       Red flash, then returns to dim
"""

from __future__ import annotations

import math
import os
import time
from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush, QColor, QFont,
    QPainter, QPen, QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from spidy.ui.themes.base import Theme

_IDLE_FRAME_MS    = 80
_ACTIVE_FRAME_MS  = 40
_THINKING_FRAME_MS = 50

_LOGO_PATHS = [
    os.path.join("assets", "icons", "spidy_logo.png"),
    os.path.join("assets", "icons", "spidy_logo.svg"),
]


class SpidyOrb(QWidget):
    """
    Animated central orb for the Spidy overlay.

    Public API
    ----------
    set_state(state: str)       -- Drive animations from UIState value string
    apply_theme(theme: Theme)   -- Update colors when theme changes
    set_amplitude(amp: float)   -- Feed speaking amplitude 0-1 for reactive glow
    """

    _ORB_SIZE = 120

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(200, 200)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._state     = "idle"
        self._phase     = 0.0
        self._spin      = 0.0
        self._amplitude = 0.0
        self._error_flash = 0

        self._color_idle      = QColor("#6C63FF")
        self._color_listening = QColor("#00C853")
        self._color_thinking  = QColor("#00D4FF")
        self._color_speaking  = QColor("#FFB800")
        self._color_working   = QColor("#E8A020")
        self._color_error     = QColor("#FF4D6A")
        self._color_body      = QColor("#12131F")
        self._color_ring      = QColor("#2A2B45")
        self._color_text      = QColor("#E8E9F0")

        self._logo: QPixmap | None = self._load_logo()

        self._timer = QTimer(self)
        self._timer.setInterval(_IDLE_FRAME_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_state(self, state: str) -> None:
        self._state = state
        self._set_timer_speed(state)
        if state == "error":
            self._error_flash = 6
        elif state != "error":
            self._error_flash = 0
        self.update()

    def apply_theme(self, theme: "Theme") -> None:
        c = theme.colors
        self._color_idle      = QColor(c.accent_primary)
        self._color_listening = QColor(c.accent_success)
        self._color_thinking  = QColor(c.accent_secondary)
        self._color_speaking  = QColor(c.accent_warning)
        self._color_working   = QColor(c.working_glow)
        self._color_error     = QColor(c.accent_error)
        self._color_body      = QColor(c.bg_secondary)
        self._color_ring      = QColor(c.border)
        self._color_text      = QColor(c.text_primary)
        self.update()

    def set_amplitude(self, amp: float) -> None:
        self._amplitude = max(0.0, min(1.0, amp))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx = self.width()  / 2.0
        cy = self.height() / 2.0
        r  = self._ORB_SIZE / 2.0

        accent, glow_bonus, inner_alpha = self._state_visuals()

        # Layer 1: Outer glow ring
        glow_r = r + 14 + glow_bonus
        grad = QRadialGradient(QPointF(cx, cy), glow_r)
        glow_col = QColor(accent)
        glow_col.setAlphaF(0.22)
        grad.setColorAt(0.0, glow_col)
        transparent = QColor(accent)
        transparent.setAlphaF(0.0)
        grad.setColorAt(1.0, transparent)
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), glow_r, glow_r)

        # Layer 2: Thinking spinner arc
        if self._state == "thinking":
            self._draw_thinking_arc(painter, cx, cy, r)

        # Layer 3: Working orbit ring
        if self._state == "working":
            self._draw_working_orbit(painter, cx, cy, r)

        # Layer 4: Glass body
        body_col = QColor(self._color_body)
        body_col.setAlphaF(0.88)
        painter.setBrush(QBrush(body_col))
        ring_col = QColor(accent)
        ring_col.setAlphaF(inner_alpha)
        painter.setPen(QPen(ring_col, 2.5))
        painter.drawEllipse(QPointF(cx, cy), r, r)

        # Layer 5: Inner accent ring
        inner_r = r - 6
        ring2 = QColor(accent)
        ring2.setAlphaF(inner_alpha * 0.5)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(ring2, 1.0))
        painter.drawEllipse(QPointF(cx, cy), inner_r, inner_r)

        # Layer 6: Centre content
        self._draw_centre(painter, cx, cy, r)

        painter.end()

    def _state_visuals(self) -> tuple:
        if self._error_flash > 0:
            return self._color_error, 8.0, 0.9

        breath = math.sin(self._phase) * 0.5 + 0.5

        if self._state == "idle":
            return self._color_idle, breath * 4.0, 0.25 + breath * 0.25
        elif self._state == "wake_ready":
            return self._color_idle, breath * 6.0, 0.45 + breath * 0.25
        elif self._state == "listening":
            return self._color_listening, breath * 8.0, 0.55 + breath * 0.35
        elif self._state == "thinking":
            return self._color_thinking, 4.0, 0.55
        elif self._state == "speaking":
            amp_bonus = self._amplitude * 10.0
            return self._color_speaking, breath * 4.0 + amp_bonus, 0.45 + self._amplitude * 0.45
        elif self._state == "working":
            return self._color_working, breath * 7.0, 0.60 + breath * 0.25
        else:
            return self._color_idle, 0.0, 0.20

    def _draw_thinking_arc(self, painter: QPainter, cx: float, cy: float, r: float) -> None:
        arc_r = r + 7
        pen = QPen(self._color_thinking, 3.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        rect = QRectF(cx - arc_r, cy - arc_r, arc_r * 2, arc_r * 2)
        painter.drawArc(rect, int(self._spin * 16), 90 * 16)

    def _draw_working_orbit(self, painter: QPainter, cx: float, cy: float, r: float) -> None:
        arc_r = r + 8
        pen = QPen(self._color_working, 2.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        rect = QRectF(cx - arc_r, cy - arc_r, arc_r * 2, arc_r * 2)
        for offset in (0, 180):
            painter.drawArc(rect, int((self._spin + offset) * 16), 60 * 16)

    def _draw_centre(self, painter: QPainter, cx: float, cy: float, r: float) -> None:
        icon_size = int(r * 0.90)
        if self._logo is not None:
            scaled = self._logo.scaled(
                icon_size, icon_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(int(cx - scaled.width() / 2), int(cy - scaled.height() / 2), scaled)
        else:
            font = QFont("Segoe UI", int(r * 0.55), QFont.Weight.Bold)
            painter.setFont(font)
            painter.setPen(QPen(self._color_text))
            painter.drawText(QRectF(cx - r, cy - r, r * 2, r * 2), Qt.AlignmentFlag.AlignCenter, "\u26a1")

    def _set_timer_speed(self, state: str) -> None:
        if state in ("idle", "wake_ready", "error"):
            ms = _IDLE_FRAME_MS
        elif state == "thinking":
            ms = _THINKING_FRAME_MS
        else:
            ms = _ACTIVE_FRAME_MS
        if self._timer.interval() != ms:
            self._timer.setInterval(ms)

    def _tick(self) -> None:
        if self._state in ("idle", "wake_ready"):
            self._phase += 0.035
        elif self._state == "listening":
            self._phase += 0.07
        elif self._state in ("speaking", "working"):
            self._phase += 0.09
        elif self._state == "thinking":
            self._spin = (self._spin + 6.0) % 360.0
            self._phase += 0.04
        else:
            self._phase += 0.035
        self._phase %= math.pi * 2
        if self._error_flash > 0:
            self._error_flash -= 1
        self.update()

    @staticmethod
    def _load_logo() -> "QPixmap | None":
        for path in _LOGO_PATHS:
            if os.path.exists(path):
                pix = QPixmap(path)
                if not pix.isNull():
                    return pix
        return None
