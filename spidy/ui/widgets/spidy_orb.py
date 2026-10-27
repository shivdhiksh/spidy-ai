"""
SpidyOrb — Spider-Verse Tactical Status Orb
===========================================
The central animated companion orb for the Spidy HUD overlay.
Renders with high-precision QPainter vector graphics (zero external asset dependency).

Visual design
-------------
1. Outer glow ring:
   - IDLE:      Gentle dark-red / navy breathing pulse (2.8s cycle, low amplitude).
   - LISTENING: 10% scale-up bloom, vibrant Spider-red (#E62429) pulse ring.
   - THINKING:  Spider-sense cyan (#3FA9F5) + red glow with rotating radial web-spinner.
   - SPEAKING:  Amplitude-reactive red/navy ripple.
   - WORKING:   Tactical red orbit with targeting pings.
2. Web-spinner thinking animation:
   - 8 radiating geometric web hairlines with connecting radial thread arcs.
3. Centerpiece:
   - Stylized Spider-Sense geometric targeting reticle or custom logo if provided.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush, QColor, QFont,
    QPainter, QPainterPath, QPen, QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from spidy.ui.themes.base import Theme

_IDLE_FRAME_MS     = 80
_ACTIVE_FRAME_MS   = 35
_THINKING_FRAME_MS = 40

_LOGO_PATHS = [
    os.path.join("assets", "icons", "spidy_logo.png"),
    os.path.join("assets", "icons", "spidy_logo.svg"),
]


class SpidyOrb(QWidget):
    """
    Spider-Verse animated central status orb.

    Public API
    ----------
    set_state(state: str)       -- Drive animations from UIState value string
    apply_theme(theme: Theme)   -- Update colors when theme changes
    set_amplitude(amp: float)   -- Feed speaking/mic amplitude 0-1 for reactive glow
    """

    _ORB_SIZE = 120

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(200, 200)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._state       = "idle"
        self._phase       = 0.0
        self._spin        = 0.0
        self._scale_anim  = 1.0
        self._amplitude   = 0.0
        self._error_flash = 0

        # Spider-Verse default palette
        self._color_idle      = QColor("#E62429")  # Spider-red
        self._color_listening = QColor("#E62429")  # Vibrant red pulse
        self._color_thinking  = QColor("#3FA9F5")  # Spider-sense cyan
        self._color_speaking  = QColor("#E62429")  # Red voice pulse
        self._color_working   = QColor("#E62429")  # Tactical red
        self._color_error     = QColor("#FF4D4F")  # Danger
        self._color_navy      = QColor("#1B2A4A")  # Navy web-blue
        self._color_body      = QColor("#12141C")  # Glass panel base
        self._color_ring      = QColor(255, 255, 255, 18)  # Web hairline
        self._color_text      = QColor("#F5F5F7")

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
        self._color_listening = QColor(c.accent_primary)
        self._color_thinking  = QColor(getattr(c, "accent_tertiary", "#3FA9F5"))
        self._color_speaking  = QColor(c.accent_primary)
        self._color_working   = QColor(c.accent_primary)
        self._color_error     = QColor(c.accent_error)
        self._color_navy      = QColor(c.accent_secondary)
        self._color_body      = QColor(c.bg_secondary)
        self._color_text      = QColor(c.text_primary)
        self.update()

    def set_amplitude(self, amp: float) -> None:
        self._amplitude = max(0.0, min(1.0, amp))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx = self.width() / 2.0
        cy = self.height() / 2.0
        
        # Scale bloom when active
        target_scale = 1.10 if self._state in ("listening", "speaking") else 1.0
        self._scale_anim += (target_scale - self._scale_anim) * 0.25
        
        r = (self._ORB_SIZE / 2.0) * self._scale_anim

        accent, glow_bonus, inner_alpha = self._state_visuals()

        # ── Layer 1: Outer Spider-Sense Glow Bloom ────────────────────────────
        glow_r = r + 16 + glow_bonus
        grad = QRadialGradient(QPointF(cx, cy), glow_r)
        glow_col = QColor(accent)
        glow_col.setAlphaF(0.24 if self._state in ("listening", "speaking") else 0.12)
        grad.setColorAt(0.0, glow_col)
        transparent = QColor(accent)
        transparent.setAlphaF(0.0)
        grad.setColorAt(1.0, transparent)
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), glow_r, glow_r)

        # ── Layer 2: Thinking Web-Spinner / Working Orbit ─────────────────────
        if self._state == "thinking":
            self._draw_web_spinner(painter, cx, cy, r)
        elif self._state == "working":
            self._draw_working_reticle(painter, cx, cy, r)

        # ── Layer 3: Tactical Corner Ticks / Reticle Ring ─────────────────────
        self._draw_reticle_ticks(painter, cx, cy, r, accent, inner_alpha)

        # ── Layer 4: Glassmorphic Void Body ──────────────────────────────────
        body_col = QColor(self._color_body)
        body_col.setAlphaF(0.92)
        painter.setBrush(QBrush(body_col))
        ring_col = QColor(accent)
        ring_col.setAlphaF(inner_alpha)
        painter.setPen(QPen(ring_col, 2.0))
        painter.drawEllipse(QPointF(cx, cy), r, r)

        # ── Layer 5: Inner Web-Hairline Ring ─────────────────────────────────
        inner_r = r - 6
        ring2 = QColor(self._color_navy)
        ring2.setAlphaF(0.7)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(ring2, 1.0))
        painter.drawEllipse(QPointF(cx, cy), inner_r, inner_r)

        # ── Layer 6: Centerpiece (Spider-Sense Eye Reticle or Logo) ───────────
        self._draw_centre(painter, cx, cy, r, accent)

        painter.end()

    def _state_visuals(self) -> tuple:
        if self._error_flash > 0:
            return self._color_error, 10.0, 0.95

        breath = math.sin(self._phase) * 0.5 + 0.5

        if self._state == "idle":
            return self._color_idle, breath * 4.0, 0.25 + breath * 0.20
        elif self._state == "wake_ready":
            return self._color_idle, breath * 8.0, 0.60 + breath * 0.30
        elif self._state == "listening":
            amp_boost = self._amplitude * 12.0
            return self._color_listening, breath * 6.0 + amp_boost + 6.0, 0.75 + self._amplitude * 0.25
        elif self._state == "thinking":
            return self._color_thinking, 6.0, 0.65
        elif self._state == "speaking":
            amp_bonus = self._amplitude * 14.0
            return self._color_speaking, breath * 4.0 + amp_bonus, 0.55 + self._amplitude * 0.45
        elif self._state == "working":
            return self._color_working, breath * 8.0, 0.70 + breath * 0.25
        else:
            return self._color_idle, 0.0, 0.20

    def _draw_web_spinner(self, painter: QPainter, cx: float, cy: float, r: float) -> None:
        """Draw 8 rotating radial web hairlines with connecting thread arcs."""
        spoke_r = r + 12
        pen = QPen(self._color_thinking, 1.2)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Draw 8 radial spokes
        for i in range(8):
            angle_rad = math.radians(self._spin + i * 45)
            x2 = cx + spoke_r * math.cos(angle_rad)
            y2 = cy + spoke_r * math.sin(angle_rad)
            pen.setColor(QColor(63, 169, 245, 180 if i % 2 == 0 else 80))
            painter.setPen(pen)
            painter.drawLine(QPointF(cx, cy), QPointF(x2, y2))

        # Concentric connecting web arcs
        arc_rect = QRectF(cx - spoke_r * 0.75, cy - spoke_r * 0.75, spoke_r * 1.5, spoke_r * 1.5)
        pen.setColor(QColor(63, 169, 245, 120))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.drawArc(arc_rect, int(self._spin * 16), 120 * 16)
        painter.drawArc(arc_rect, int((self._spin + 180) * 16), 120 * 16)

    def _draw_working_reticle(self, painter: QPainter, cx: float, cy: float, r: float) -> None:
        """Tactical rotating radar reticle with corner brackets."""
        arc_r = r + 10
        pen = QPen(self._color_working, 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        rect = QRectF(cx - arc_r, cy - arc_r, arc_r * 2, arc_r * 2)
        for offset in (0, 120, 240):
            painter.drawArc(rect, int((self._spin + offset) * 16), 40 * 16)

    def _draw_reticle_ticks(self, painter: QPainter, cx: float, cy: float, r: float, accent: QColor, alpha: float) -> None:
        """Draw fine corner bracket reticles around the orb."""
        tick_r1 = r + 3
        tick_r2 = r + 7
        pen = QPen(accent, 1.2)
        for i in range(12):
            angle = math.radians(i * 30)
            x1 = cx + tick_r1 * math.cos(angle)
            y1 = cy + tick_r1 * math.sin(angle)
            x2 = cx + tick_r2 * math.cos(angle)
            y2 = cy + tick_r2 * math.sin(angle)
            pen_color = QColor(accent)
            pen_color.setAlphaF(alpha * (0.8 if i % 3 == 0 else 0.3))
            pen.setColor(pen_color)
            painter.setPen(pen)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    def _draw_centre(self, painter: QPainter, cx: float, cy: float, r: float, accent: QColor) -> None:
        """Render logo or stylized geometric Spider-Eye reticle."""
        icon_size = int(r * 0.90)
        if self._logo is not None:
            scaled = self._logo.scaled(
                icon_size, icon_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(int(cx - scaled.width() / 2), int(cy - scaled.height() / 2), scaled)
        else:
            # Stylized geometric Spider-Sense Eye Mask emblem
            path = QPainterPath()
            w = r * 0.42
            h = r * 0.58
            
            # Left eye polygon
            left_eye = QPainterPath()
            left_eye.moveTo(cx - 3, cy - h * 0.6)
            left_eye.lineTo(cx - w, cy - h * 0.1)
            left_eye.lineTo(cx - w * 0.85, cy + h * 0.5)
            left_eye.lineTo(cx - 3, cy + h * 0.15)
            left_eye.closeSubpath()

            # Right eye polygon
            right_eye = QPainterPath()
            right_eye.moveTo(cx + 3, cy - h * 0.6)
            right_eye.lineTo(cx + w, cy - h * 0.1)
            right_eye.lineTo(cx + w * 0.85, cy + h * 0.5)
            right_eye.lineTo(cx + 3, cy + h * 0.15)
            right_eye.closeSubpath()

            # Fill eyes with white and stroke with accent red
            eye_fill = QColor("#F5F5F7")
            painter.setBrush(QBrush(eye_fill))
            painter.setPen(QPen(accent, 2.0))
            painter.drawPath(left_eye)
            painter.drawPath(right_eye)

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
            self._phase += 0.08
        elif self._state in ("speaking", "working"):
            self._phase += 0.09
            self._spin = (self._spin + 3.0) % 360.0
        elif self._state == "thinking":
            self._spin = (self._spin + 7.5) % 360.0
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
