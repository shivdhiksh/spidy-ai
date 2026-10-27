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
