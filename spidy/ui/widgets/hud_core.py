"""
hud_core.py -- SpidyCoreWidget: Large futuristic central AI core
================================================================
The primary visual element of the Spidy HUD.

Draws entirely with QPainter -- no external images required at runtime.
A logo slot is provided: drop assets/icons/spidy_logo.png to override
the rendered core symbol.

Visual layers (back to front)
------------------------------
1. Deep background circle (near-black radial gradient)
2. Outer decorative ring + 72 tick marks
3. Outer rotating arc (state-dependent speed)
4. Second ring + radial markers
5. Inner metric ring (gauge arc showing CPU% or amplitude)
6. Scanning arc (THINKING / WORKING states)
7. Inner tight ring
8. Nucleus radial-gradient glow (color = state)
9. HUD cross-hair lines (thin technical)
10. State text (compact)
11. Logo slot / core symbol

Performance
-----------
- IDLE:    12 fps (80ms timer)  -- slow breathe only
- ACTIVE:  25 fps (40ms timer)  -- listening/speaking
- WORKING: 25 fps               -- orbit arcs
- THINKING: 20 fps (50ms)       -- scanning sweep
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPainterPath,
    QPen, QPixmap, QRadialGradient,
)
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from spidy.ui.themes.base import Theme

# Frame rates
_FPS_IDLE    = 80   # ms -- 12.5fps
_FPS_ACTIVE  = 40   # ms -- 25fps
_FPS_THINK   = 50   # ms -- 20fps

# Asset paths (checked on first paint)
_LOGO_PATHS = [
    os.path.join("assets", "icons", "spidy_logo.png"),
    os.path.join("assets", "icons", "spidy_logo.svg"),
]

# Ring radii as fractions of half-widget-size
_R_OUTER_DECO  = 0.92   # outermost decorative ring
_R_OUTER_ARC   = 0.88   # rotating arc
_R_MID_RING    = 0.78   # second ring
_R_METRIC      = 0.68   # gauge arc
_R_SCAN        = 0.74   # scanning arc
_R_INNER       = 0.58   # inner ring
_R_NUCLEUS     = 0.42   # glowing nucleus
_R_LOGO        = 0.30   # logo / symbol area


class SpidyCoreWidget(QWidget):
    """
    Large central futuristic AI core.

    Public API
    ----------
    set_state(state: str)           Drive animations from UIState value
    apply_theme(theme: Theme)       Update all colors
    set_amplitude(amp: float)       0-1 from waveform (listening/speaking)
    set_metric(value: float)        0-100 for inner gauge (CPU%)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._state       = "idle"
        self._phase       = 0.0       # breathe phase (0..2pi)
        self._arc_angle   = 0.0       # outer arc rotation (degrees)
        self._scan_angle  = 0.0       # scanning arc angle
        self._orbit_angle = 0.0       # working orbit arcs
        self._amplitude   = 0.0       # speaking/listening amplitude 0-1
        self._metric      = 0.0       # inner gauge 0-100

        # Boot animation
        self._boot_scale  = 1.0       # 0.0..1.0 for expand-in

        # Colors (defaults = dark HUD theme)
        self._c_primary   = QColor("#00E5FF")  # luminous cyan
        self._c_secondary = QColor("#00ACC1")  # teal
        self._c_dim       = QColor("#004D5E")  # dim
        self._c_grid      = QColor("#060D1A")  # near-black
        self._c_text      = QColor("#B2EBF2")  # soft cyan-white
        self._c_working   = QColor("#E8A020")  # amber
        self._c_error     = QColor("#FF2444")  # red
        self._c_nucleus   = QColor("#00E5FF")  # changes per state

        self._logo: QPixmap | None = self._try_load_logo()

        self._timer = QTimer(self)
        self._timer.setInterval(_FPS_IDLE)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ---- Public API ---------------------------------------------------------

    def set_state(self, state: str) -> None:
        self._state = state
        self._update_timer_speed()
        self.update()

    def apply_theme(self, theme: "Theme") -> None:
        c = theme.colors
        self._c_primary   = QColor(c.hud_primary)
        self._c_secondary = QColor(c.hud_secondary)
        self._c_dim       = QColor(c.hud_dim)
        self._c_grid      = QColor(c.bg_primary)
        self._c_text      = QColor(c.hud_text)
        self._c_working   = QColor(c.working_glow)
        self._c_error     = QColor(c.accent_error)
        self.update()

    def set_amplitude(self, amp: float) -> None:
        self._amplitude = max(0.0, min(1.0, amp))

    def set_metric(self, value: float) -> None:
        self._metric = max(0.0, min(100.0, value))

    def set_boot_scale(self, scale: float) -> None:
        self._boot_scale = max(0.0, min(1.0, scale))
        self.update()

    # ---- Painting -----------------------------------------------------------

    def paintEvent(self, event) -> None:
        if self._boot_scale < 0.01:
            return
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx = self.width()  / 2.0
        cy = self.height() / 2.0
        # `half` always uses the full available radius.
        # Boot-scale is handled ONLY via the painter transform below,
        # preventing the previous double-scaling artifact.
        half = min(cx, cy)

        if self._boot_scale < 1.0:
            painter.translate(cx, cy)
            painter.scale(self._boot_scale, self._boot_scale)
            painter.translate(-cx, -cy)

        accent, nucleus_color, glow_alpha = self._state_palette()

        # Layer 1: deep background
        self._draw_bg(painter, cx, cy, half, nucleus_color, glow_alpha)

        # Layer 2: outer decorative ring + tick marks
        self._draw_outer_deco(painter, cx, cy, half, accent)

        # Layer 3: rotating outer arc
        self._draw_outer_arc(painter, cx, cy, half, accent)

        # Layer 4: second ring + radial markers
        self._draw_mid_ring(painter, cx, cy, half, accent)

        # Layer 5: inner metric gauge
        self._draw_metric_ring(painter, cx, cy, half, accent)

        # Layer 6: scanning arc (THINKING / WORKING)
        if self._state in ("thinking", "working"):
            self._draw_scan_arc(painter, cx, cy, half, accent)

        # Layer 7: inner tight ring
        self._draw_inner_ring(painter, cx, cy, half, accent)

        # Layer 8: nucleus glow
        self._draw_nucleus(painter, cx, cy, half, nucleus_color, glow_alpha)

        # Layer 9: HUD cross-hair lines
        self._draw_crosshairs(painter, cx, cy, half, accent)

        # Layer 10: state text
        self._draw_state_text(painter, cx, cy, half)

        # Layer 11: logo / core symbol
        self._draw_logo(painter, cx, cy, half)

        painter.end()

    # ---- Internal drawing helpers -------------------------------------------

    def _draw_bg(self, p: QPainter, cx, cy, half, nucleus_color, glow_alpha):
        r = half * _R_OUTER_DECO + 4
        grad = QRadialGradient(QPointF(cx, cy), r)
        nucleus_dim = QColor(nucleus_color)
        nucleus_dim.setAlphaF(glow_alpha * 0.18)
        grad.setColorAt(0.0, nucleus_dim)
        dark = QColor(self._c_grid)
        dark.setAlphaF(0.92)
        grad.setColorAt(0.55, dark)
        outer = QColor(self._c_grid)
        outer.setAlphaF(0.0)
        grad.setColorAt(1.0, outer)
        p.setBrush(QBrush(grad))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), r, r)

    def _draw_outer_deco(self, p: QPainter, cx, cy, half, accent):
        r = half * _R_OUTER_DECO
        ring_col = QColor(accent)
        ring_col.setAlphaF(0.50)
        p.setPen(QPen(ring_col, 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)

        # 72 tick marks
        tick_pen_major = QPen(QColor(accent), 1.5)
        tick_pen_minor = QPen(QColor(accent), 0.8)
        tick_minor = QColor(accent)
        tick_minor.setAlphaF(0.35)
        tick_pen_minor = QPen(tick_minor, 0.7)
        for i in range(72):
            angle = math.radians(i * 5)
            ca, sa = math.cos(angle), math.sin(angle)
            is_major = (i % 9 == 0)
            outer_r = r
            inner_r = r - (5 if is_major else 2.5)
            x1 = cx + ca * outer_r
            y1 = cy + sa * outer_r
            x2 = cx + ca * inner_r
            y2 = cy + sa * inner_r
            p.setPen(tick_pen_major if is_major else tick_pen_minor)
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    def _draw_outer_arc(self, p: QPainter, cx, cy, half, accent):
        r = half * _R_OUTER_ARC
        rect = QRectF(cx - r, cy - r, r * 2, r * 2)
        arc_pen = QPen(QColor(accent), 2.5)
        arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(arc_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        start = int(self._arc_angle * 16)
        # Arc length varies by state
        if self._state == "working":
            span = int(150 * 16)
        elif self._state == "thinking":
            span = int(100 * 16)
        elif self._state == "listening":
            span = int(200 * 16)
        else:
            span = int(90 * 16)
        p.drawArc(rect, start, span)

        # Opposing short arc
        opposing_col = QColor(accent)
        opposing_col.setAlphaF(0.4)
        opp_pen = QPen(opposing_col, 1.5)
        opp_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(opp_pen)
        p.drawArc(rect, start + 180 * 16, 40 * 16)

    def _draw_mid_ring(self, p: QPainter, cx, cy, half, accent):
        r = half * _R_MID_RING
        ring_col = QColor(accent)
        ring_col.setAlphaF(0.30)
        p.setPen(QPen(ring_col, 0.8))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)

        # 12 radial markers at cardinal + ordinal positions
        marker_col = QColor(accent)
        marker_col.setAlphaF(0.55)
        p.setPen(QPen(marker_col, 1.2))
        for i in range(12):
            angle = math.radians(i * 30 + self._arc_angle * 0.1)
            outer_r = half * _R_OUTER_ARC - 2
            inner_r = half * _R_MID_RING + 2
            p.drawLine(
                QPointF(cx + math.cos(angle) * inner_r, cy + math.sin(angle) * inner_r),
                QPointF(cx + math.cos(angle) * outer_r, cy + math.sin(angle) * outer_r),
            )

    def _draw_metric_ring(self, p: QPainter, cx, cy, half, accent):
        r = half * _R_METRIC
        rect = QRectF(cx - r, cy - r, r * 2, r * 2)

        # Background arc (dim)
        bg_col = QColor(accent)
        bg_col.setAlphaF(0.12)
        p.setPen(QPen(bg_col, 2.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)

        # Gauge fill
        value = self._amplitude * 100 if self._state in ("listening", "speaking") else self._metric
        if value > 0:
            gauge_col = QColor(accent)
            gauge_col.setAlphaF(0.70)
            gauge_pen = QPen(gauge_col, 2.5)
            gauge_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(gauge_pen)
            span = int((value / 100.0) * 270 * 16)
            p.drawArc(rect, 135 * 16, span)

    def _draw_scan_arc(self, p: QPainter, cx, cy, half, accent):
        r = half * _R_SCAN
        rect = QRectF(cx - r, cy - r, r * 2, r * 2)
        scan_col = QColor(accent)
        scan_col.setAlphaF(0.75)
        scan_pen = QPen(scan_col, 2.0)
        scan_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(scan_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(rect, int(self._scan_angle * 16), 60 * 16)

        # Faint trail
        trail_col = QColor(accent)
        trail_col.setAlphaF(0.20)
        trail_pen = QPen(trail_col, 1.0)
        p.setPen(trail_pen)
        p.drawArc(rect, int((self._scan_angle - 40) * 16), 40 * 16)

    def _draw_inner_ring(self, p: QPainter, cx, cy, half, accent):
        r = half * _R_INNER
        ring_col = QColor(accent)
        ring_col.setAlphaF(0.55)
        p.setPen(QPen(ring_col, 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)

    def _draw_nucleus(self, p: QPainter, cx, cy, half, nucleus_color, glow_alpha):
        r = half * _R_NUCLEUS
        breath = math.sin(self._phase) * 0.5 + 0.5  # 0..1
        glow_r = r * (1.0 + 0.25 * breath)

        grad = QRadialGradient(QPointF(cx, cy), glow_r)
        core_bright = QColor(nucleus_color)
        core_bright.setAlphaF(glow_alpha)
        grad.setColorAt(0.0, core_bright)
        core_mid = QColor(nucleus_color)
        core_mid.setAlphaF(glow_alpha * 0.55)
        grad.setColorAt(0.45, core_mid)
        core_dim = QColor(nucleus_color)
        core_dim.setAlphaF(0.0)
        grad.setColorAt(1.0, core_dim)
        p.setBrush(QBrush(grad))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), glow_r, glow_r)

        # Solid inner nucleus
        solid = QColor(nucleus_color)
        solid.setAlphaF(min(1.0, glow_alpha + 0.1))
        p.setBrush(QBrush(solid))
        p.drawEllipse(QPointF(cx, cy), r * 0.28, r * 0.28)

    def _draw_crosshairs(self, p: QPainter, cx, cy, half, accent):
        line_col = QColor(accent)
        line_col.setAlphaF(0.18)
        p.setPen(QPen(line_col, 0.8))
        inner_r = half * _R_INNER
        outer_r = half * _R_OUTER_DECO

        for angle_deg in (0, 45, 90, 135):
            angle = math.radians(angle_deg)
            ca, sa = math.cos(angle), math.sin(angle)
            p.drawLine(
                QPointF(cx + ca * (-outer_r), cy + sa * (-outer_r)),
                QPointF(cx + ca * (-inner_r), cy + sa * (-inner_r)),
            )
            p.drawLine(
                QPointF(cx + ca * inner_r, cy + sa * inner_r),
                QPointF(cx + ca * outer_r, cy + sa * outer_r),
            )

    def _draw_state_text(self, p: QPainter, cx, cy, half):
        label_map = {
            "idle":       "STANDBY",
            "wake_ready": "AWAKE",
            "listening":  "LISTENING",
            "thinking":   "PROCESSING",
            "speaking":   "SPEAKING",
            "working":    "AUTONOMOUS",
            "error":      "ERROR",
        }
        label = label_map.get(self._state, "")
        text_col = QColor(self._c_text)
        text_col.setAlphaF(0.70)
        p.setPen(QPen(text_col))
        # State text font — capped 7pt min, 14pt max
        font = QFont("Consolas", max(7, min(14, int(half * 0.07))))
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        p.setFont(font)
        text_y = cy + half * (_R_NUCLEUS + 0.10)
        rect = QRectF(cx - half * 0.5, text_y, half, half * 0.15)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    def _draw_logo(self, p: QPainter, cx, cy, half):
        logo_r = half * _R_LOGO
        if self._logo is not None:
            size = int(logo_r * 1.8)
            scaled = self._logo.scaled(
                size, size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            p.drawPixmap(int(cx - scaled.width() / 2), int(cy - scaled.height() / 2), scaled)
        else:
            # High-tech geometric cyber emblem (vector-drawn)
            emblem_pen = QPen(QColor(self._c_nucleus), 1.8)
            emblem_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            emblem_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(emblem_pen)
            p.setBrush(Qt.BrushStyle.NoBrush)

            # Central diamond core
            d_size = logo_r * 0.45
            diamond = QPainterPath()
            diamond.moveTo(cx, cy - d_size)
            diamond.lineTo(cx + d_size * 0.8, cy)
            diamond.lineTo(cx, cy + d_size)
            diamond.lineTo(cx - d_size * 0.8, cy)
            diamond.closeSubpath()
            p.drawPath(diamond)

            # Angular legs / cyber lines
            leg_pen = QPen(QColor(self._c_nucleus), 1.4)
            p.setPen(leg_pen)
            s_leg = logo_r * 0.95
            # Top-left leg
            p.drawLine(QPointF(cx - d_size * 0.6, cy - d_size * 0.4), QPointF(cx - s_leg * 0.85, cy - s_leg * 0.65))
            p.drawLine(QPointF(cx - s_leg * 0.85, cy - s_leg * 0.65), QPointF(cx - s_leg, cy - s_leg * 0.25))
            # Top-right leg
            p.drawLine(QPointF(cx + d_size * 0.6, cy - d_size * 0.4), QPointF(cx + s_leg * 0.85, cy - s_leg * 0.65))
            p.drawLine(QPointF(cx + s_leg * 0.85, cy - s_leg * 0.65), QPointF(cx + s_leg, cy - s_leg * 0.25))
            # Bottom-left leg
            p.drawLine(QPointF(cx - d_size * 0.6, cy + d_size * 0.4), QPointF(cx - s_leg * 0.85, cy + s_leg * 0.65))
            p.drawLine(QPointF(cx - s_leg * 0.85, cy + s_leg * 0.65), QPointF(cx - s_leg, cy + s_leg * 0.35))
            # Bottom-right leg
            p.drawLine(QPointF(cx + d_size * 0.6, cy + d_size * 0.4), QPointF(cx + s_leg * 0.85, cy + s_leg * 0.65))
            p.drawLine(QPointF(cx + s_leg * 0.85, cy + s_leg * 0.65), QPointF(cx + s_leg, cy + s_leg * 0.35))

            # Inner 'S' monogram
            font = QFont("Consolas", max(9, min(20, int(logo_r * 0.55))), QFont.Weight.Bold)
            p.setFont(font)
            col = QColor(self._c_text)
            col.setAlphaF(0.95)
            p.setPen(QPen(col))
            p.drawText(QRectF(cx - d_size, cy - d_size, d_size * 2, d_size * 2), Qt.AlignmentFlag.AlignCenter, "S")

            # Technical branding below core
            font_small = QFont("Consolas", max(6, min(10, int(logo_r * 0.26))), QFont.Weight.DemiBold)
            font_small.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
            p.setFont(font_small)
            small_col = QColor(self._c_text)
            small_col.setAlphaF(0.65)
            p.setPen(QPen(small_col))
            p.drawText(
                QRectF(cx - logo_r, cy + logo_r * 0.85, logo_r * 2, logo_r * 0.4),
                Qt.AlignmentFlag.AlignCenter, "SPIDY CORE",
            )

    # ---- State palette -------------------------------------------------------

    def _state_palette(self) -> tuple:
        """Return (accent_color, nucleus_color, glow_alpha) for current state."""
        breath = math.sin(self._phase) * 0.5 + 0.5

        if self._state == "idle":
            return self._c_dim, self._c_secondary, 0.25 + breath * 0.20

        elif self._state == "wake_ready":
            return self._c_secondary, self._c_primary, 0.40 + breath * 0.25

        elif self._state == "listening":
            amp_glow = self._amplitude * 0.45
            return self._c_primary, self._c_primary, 0.55 + amp_glow + breath * 0.15

        elif self._state == "thinking":
            return self._c_primary, self._c_secondary, 0.55 + breath * 0.10

        elif self._state == "speaking":
            amp_glow = self._amplitude * 0.40
            return self._c_primary, self._c_primary, 0.60 + amp_glow

        elif self._state == "working":
            amber_col = QColor(self._c_working)
            return self._c_working, amber_col, 0.65 + breath * 0.20

        elif self._state == "error":
            return self._c_error, self._c_error, 0.80 + breath * 0.20

        return self._c_dim, self._c_secondary, 0.20

    # ---- Animation timer ----------------------------------------------------

    def _update_timer_speed(self) -> None:
        ms = {
            "idle":       _FPS_IDLE,
            "wake_ready": _FPS_IDLE,
            "error":      _FPS_IDLE,
            "thinking":   _FPS_THINK,
        }.get(self._state, _FPS_ACTIVE)
        if self._timer.interval() != ms:
            self._timer.setInterval(ms)

    def _tick(self) -> None:
        state = self._state
        if state in ("idle", "wake_ready"):
            self._phase     += 0.035
            self._arc_angle  = (self._arc_angle + 0.15) % 360
        elif state == "listening":
            self._phase     += 0.07
            self._arc_angle  = (self._arc_angle + 1.2) % 360
        elif state == "thinking":
            self._phase     += 0.04
            self._arc_angle  = (self._arc_angle + 2.0) % 360
            self._scan_angle = (self._scan_angle + 4.5) % 360
        elif state == "speaking":
            self._phase     += 0.09
            self._arc_angle  = (self._arc_angle + 1.5) % 360
        elif state == "working":
            self._phase     += 0.09
            self._arc_angle  = (self._arc_angle + 2.5) % 360
            self._scan_angle = (self._scan_angle + 6.0) % 360
        elif state == "error":
            self._phase     += 0.12
            self._arc_angle  = (self._arc_angle + 3.0) % 360
        else:
            self._phase += 0.03

        self._phase %= math.pi * 2
        self.update()

    @staticmethod
    def _try_load_logo() -> "QPixmap | None":
        for path in _LOGO_PATHS:
            if os.path.exists(path):
                from PySide6.QtGui import QPixmap as _QP
                pix = _QP(path)
                if not pix.isNull():
                    return pix
        return None
