"""
The dashboard face: a rotating sphere of light dots inside HUD rings (tick ring, dotted ring, dashed arcs,
red accent arcs) over a faint grid. Drawn live from a few thousand points, so it can react to Spidy's state.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF, QRadialGradient

CYAN = (64, 204, 232)
RED = (222, 44, 56)
AMBER = (255, 184, 48)
_BINS = 6                       # depth layers: far dots are small and dim, near dots big and bright


def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def _c(rgb, a: float) -> QColor:
    return QColor(rgb[0], rgb[1], rgb[2], max(0, min(255, int(a))))


def _hsv(h: float, s: float, v: float, a: float) -> QColor:
    c = QColor.fromHsvF(h % 1.0, s, v)
    c.setAlpha(max(0, min(255, int(a))))
    return c


class DotBlob:
    """
    The Cosmic Violet face: a clean 3D ball of glowing dots that turns slowly and *waves with the voice*.

    ``level`` (0..1, live loudness of Spidy's or the user's voice) pushes travelling bulges across the
    surface, so the shape squashes and swells in time with each word and settles back to a calm ball
    in silence. Same ``paint`` signature as :class:`DotSphere`, so it is a drop-in replacement.
    """

    HUE_TOP, HUE_BOTTOM = 0.58, 0.73          # blue at the top fading to violet at the bottom
    _BANDS = 4                                  # colour bands (top -> bottom) x depth bins

    def __init__(self, points: int = 2400) -> None:
        self.pts = _fibonacci_sphere(points)

    def paint(self, p: QPainter, cx: float, cy: float, radius: float, phase: float, level: float,
              tilt: tuple[float, float] = (0.0, 0.0), state: str = "idle", k: float = 1.0,
              rings: bool = False, grid_rect: QRectF | None = None) -> None:
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        h0, h1 = self.HUE_TOP, self.HUE_BOTTOM
        if state == "error":
            h0, h1 = 0.08, 0.02                               # amber -> red
        elif state in ("thinking", "working"):
            shift = 0.04 * math.sin(phase * 2.2)              # gentle colour breathing while it works
            h0, h1 = h0 + 0.06 + shift, h1 + 0.08 + shift
        self._glow(p, cx, cy, radius, level, h1)
        self._dots(p, cx, cy, radius, phase, level, tilt, h0, h1, k, state)
        p.restore()

    @staticmethod
    def _glow(p: QPainter, cx: float, cy: float, radius: float, level: float, hue: float) -> None:
        g = QRadialGradient(QPointF(cx, cy), radius * 1.75)
        g.setColorAt(0.0, _hsv(hue, 0.75, 0.9, 70 + 70 * level))
        g.setColorAt(0.45, _hsv(hue - 0.06, 0.8, 0.8, 26 + 34 * level))
        g.setColorAt(1.0, _hsv(hue, 0.8, 0.6, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(g))
        p.drawEllipse(QPointF(cx, cy), radius * 1.75, radius * 1.75)

    def _dots(self, p, cx, cy, radius, phase, level, tilt, h0, h1, k, state) -> None:
        x, y, z = self.pts[:, 0], self.pts[:, 1], self.pts[:, 2]
        # voice: smooth bulges travelling round the ball; a slow breath when quiet
        wave = (0.10 * np.sin(3 * x + phase * 2.1) * np.cos(2 * y - phase * 1.3)
                + 0.07 * np.sin(5 * z + 4 * y + phase * 3.0)
                + 0.05 * np.sin(7 * x - 3 * z - phase * 4.2))
        breath = 0.015 * math.sin(phase * 0.8)
        if state in ("thinking", "working"):
            breath += 0.03 * math.sin(phase * 3.0)
        r = 1.0 + breath + level * 1.05 * wave
        x, y, z = x * r, y * r, z * r
        yaw = phase * (0.22 + 0.35 * level) + math.radians(tilt[0]) * 2.5
        pitch = 0.35 + math.radians(tilt[1]) * 2.5
        cyw, syw, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        x1, z1 = x * cyw + z * syw, -x * syw + z * cyw
        y2, z2 = y * cp - z1 * sp, y * sp + z1 * cp
        persp = 1.0 / (1.0 - 0.2 * z2)
        sx, sy = cx + x1 * persp * radius, cy + y2 * persp * radius
        depth = np.clip((z2 + 1.25) / 2.5, 0, 0.999)
        dbin = (depth * _BINS).astype(int)
        band = np.clip(((y2 + 1.1) / 2.2 * self._BANDS).astype(int), 0, self._BANDS - 1)
        bright = 0.9 + 0.25 * level
        pen = QPen()
        pen.setCapStyle(Qt.PenCapStyle.SquareCap)
        for b in range(_BINS):
            t = (b + 0.5) / _BINS
            pen.setWidthF((1.0 + 2.6 * t) * k)
            for c in range(self._BANDS):
                sel = (dbin == b) & (band == c)
                if not sel.any():
                    continue
                hue = h0 + (h1 - h0) * (c + 0.5) / self._BANDS
                pen.setColor(_hsv(hue, 0.72 - 0.3 * t, 1.0, (70 + 185 * t ** 1.2) * bright))
                p.setPen(pen)
                p.drawPoints(QPolygonF([QPointF(a, d) for a, d in zip(sx[sel].tolist(), sy[sel].tolist())]))


class DotSphere:
    """Paints the sphere and its rings; holds no widget state (the caller supplies phase, level and tilt)."""

    def __init__(self, points: int = 1500) -> None:
        self.pts = _fibonacci_sphere(points)

    def paint(self, p: QPainter, cx: float, cy: float, radius: float, phase: float, level: float,
              tilt: tuple[float, float] = (0.0, 0.0), state: str = "idle", k: float = 1.0,
              rings: bool = True, grid_rect: QRectF | None = None) -> None:
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        accent = AMBER if state == "error" else CYAN
        if grid_rect is not None:
            self._grid(p, grid_rect, k)
        self._glow(p, cx, cy, radius, level, accent)
        self._dots(p, cx, cy, radius, phase, level, tilt, accent, k)
        if rings:
            self._rings(p, cx, cy, radius, phase, level, state, accent, k)
        p.restore()

    # -- layers
    @staticmethod
    def _grid(p: QPainter, r: QRectF, k: float) -> None:
        p.setPen(QPen(_c(CYAN, 11), 1.0))
        step = 60 * k
        x = r.left() + (r.width() / 2) % step
        while x < r.right():
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            x += step
        y = r.top() + (r.height() / 2) % step
        while y < r.bottom():
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            y += step

    @staticmethod
    def _glow(p: QPainter, cx: float, cy: float, radius: float, level: float, accent) -> None:
        g = QRadialGradient(QPointF(cx, cy), radius * 1.05)
        g.setColorAt(0.0, _c(accent, 70 + 90 * level))
        g.setColorAt(0.6, _c(accent, 22 + 30 * level))
        g.setColorAt(1.0, _c(accent, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(g))
        p.drawEllipse(QPointF(cx, cy), radius * 1.05, radius * 1.05)

    def _dots(self, p, cx, cy, radius, phase, level, tilt, accent, k) -> None:
        yaw = phase * (0.30 + 0.5 * level) + math.radians(tilt[0]) * 2.5
        pitch = 0.32 + math.radians(tilt[1]) * 2.5
        cy_, sy_, cx_, sx_ = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        x, y, z = self.pts[:, 0], self.pts[:, 1], self.pts[:, 2]
        x1, z1 = x * cy_ + z * sy_, -x * sy_ + z * cy_            # spin around the vertical axis
        y2, z2 = y * cx_ - z1 * sx_, y * sx_ + z1 * cx_           # tilt toward the viewer
        persp = 1.0 / (1.0 - 0.18 * z2)
        sx, sy = cx + x1 * persp * radius, cy + y2 * persp * radius
        depth = np.clip((z2 + 1) * 0.5, 0, 0.999)
        bins = (depth * _BINS).astype(int)
        base_a = 0.55 + 0.45 * level
        pen = QPen()
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        for b in range(_BINS):
            sel = bins == b
            if not sel.any():
                continue
            t = (b + 0.5) / _BINS
            pen.setColor(_c(accent, (45 + 200 * t) * base_a))
            pen.setWidthF((1.1 + 2.9 * t) * k)
            p.setPen(pen)
            p.drawPoints(QPolygonF([QPointF(a, c) for a, c in zip(sx[sel].tolist(), sy[sel].tolist())]))

    @staticmethod
    def _rings(p, cx, cy, radius, phase, level, state, accent, k) -> None:
        spin = 0.5 + 3.0 * level + (1.2 if state in ("thinking", "working") else 0.0)
        centre = QPointF(cx, cy)
        # tick ring
        r_out, tick = radius * 1.46, 7 * k
        p.setPen(QPen(_c(accent, 150), 1.3 * k, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for i in range(120):
            a = math.radians(i * 3 + phase * 2.0)
            c, s = math.cos(a), math.sin(a)
            long_tick = 1.5 if i % 5 == 0 else 1.0
            p.drawLine(QPointF(cx + c * (r_out - tick * long_tick), cy + s * (r_out - tick * long_tick)),
                       QPointF(cx + c * r_out, cy + s * r_out))
        # dotted ring
        p.setPen(QPen(_c(accent, 70), 1.6 * k, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        r_dot = radius * 1.34
        p.drawPoints(QPolygonF([QPointF(cx + math.cos(math.radians(a)) * r_dot, cy + math.sin(math.radians(a)) * r_dot)
                                for a in range(0, 360, 2)]))

        def arcs(r_rel: float, base: float, spans, width: float, color, alpha: int, direction: int) -> None:
            r = radius * r_rel
            p.setPen(QPen(_c(color, alpha), width * k, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
            p.setBrush(Qt.BrushStyle.NoBrush)
            rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
            for start, span in spans:
                a = (base * direction + start) % 360
                p.drawArc(rect, int(a * 16), int(span * 16))

        arcs(1.27, phase * 9 * spin, ((8, 22), (70, 12), (130, 30), (215, 10), (255, 26), (320, 14)), 2.0, accent, 215, 1)
        arcs(1.22, phase * 6 * spin, ((30, 9), (160, 16), (280, 8)), 1.4, RED if state != "error" else AMBER, 190, -1)
        arcs(1.12, phase * 11 * spin, ((100, 50), (260, 34)), 1.2, RED if state != "error" else AMBER, 130, 1)
        arcs(1.06, phase * 4 * spin, ((0, 120), (180, 90)), 0.9, accent, 70, -1)
