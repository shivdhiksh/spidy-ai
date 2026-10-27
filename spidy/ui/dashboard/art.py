"""
Artwork for the dashboard (all drawn with QPainter, no image files):

- ``draw_head``         the 3D spider face used as the hero (glossy visor eyes, web, rim light)
- ``draw_spider_line``  thin line-art spider used as the small logo
- ``draw_card``         subtle glass card (hairline border, soft highlight)
- ``make_background``   near-black backdrop with a soft red glow and fine film grain
"""

from __future__ import annotations

import math
import random

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPainterPathStroker, QPen, QPixmap,
    QRadialGradient, QTransform,
)

from spidy.ui.dashboard import palette as pal


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    return QColor(int(a.red() + (b.red() - a.red()) * t), int(a.green() + (b.green() - a.green()) * t),
                  int(a.blue() + (b.blue() - a.blue()) * t), int(a.alpha() + (b.alpha() - a.alpha()) * t))


# ── Spider face ─────────────────────────────────────────────────────────────

# right half of the visor outline: cubic segments (control1, control2, end), starting at (0, -196)
_HEAD_SEGS = [((180, -212), (296, -170), (330, -98)),
              ((360, -42), (364, 30), (336, 92)),
              ((312, 144), (256, 184), (196, 208)),
              ((126, 234), (60, 240), (0, 240))]


def _head_path() -> QPainterPath:
    """Symmetric closed outline: down the right side, back up the mirrored left side."""
    path = QPainterPath(QPointF(0, -212))
    for c1, c2, end in _HEAD_SEGS:
        path.cubicTo(QPointF(*c1), QPointF(*c2), QPointF(*end))
    starts = [(0, -212)] + [seg[2] for seg in _HEAD_SEGS[:-1]]
    for (c1, c2, _end), start in zip(reversed(_HEAD_SEGS), reversed(starts)):
        path.cubicTo(QPointF(-c2[0], c2[1]), QPointF(-c1[0], c1[1]), QPointF(-start[0], start[1]))
    path.closeSubpath()
    return path

def _eye_path(sign: int) -> QPainterPath:
    """One angular visor eye (sign +1 = right). Outer corner high, inner corner low, like a spider's gaze."""
    def P(x, y): return QPointF(sign * x, y)
    path = QPainterPath(P(40, 52))
    path.cubicTo(P(86, -2), P(210, -58), P(312, -66))
    path.cubicTo(P(318, -30), P(288, 28), P(236, 60))
    path.cubicTo(P(190, 88), P(120, 100), P(78, 104))
    path.cubicTo(P(56, 100), P(44, 78), P(40, 52))
    path.closeSubpath()
    return path


def head_red(mood: str) -> QColor:
    return pal.RED_BRIGHT if mood != "error" else QColor(255, 40, 40)


def draw_head_glow(p: QPainter, red: QColor) -> None:
    """Red atmosphere behind the face (a flat ellipse so it fades out inside the widget)."""
    p.save()
    p.scale(1.0, 0.5)
    g = QRadialGradient(QPointF(0, 40), 440)
    g.setColorAt(0.0, pal.with_alpha(red, 86))
    g.setColorAt(0.55, pal.with_alpha(red, 28))
    g.setColorAt(1.0, pal.with_alpha(red, 0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(g))
    p.drawEllipse(QPointF(0, 40), 440, 440)
    p.restore()


def draw_head_body(p: QPainter, red: QColor) -> None:
    """Everything static: glossy shell, web, seam, eye bezels, mandibles, bevel and rim light."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    head = _head_path()
    body = QLinearGradient(0, -212, 0, 240)
    body.setColorAt(0.0, QColor(30, 31, 37))
    body.setColorAt(0.45, QColor(12, 12, 16))
    body.setColorAt(1.0, QColor(3, 3, 5))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(body))
    p.drawPath(head)

    p.save()
    p.setClipPath(head)
    sheen = QRadialGradient(QPointF(-150, -140), 360)
    sheen.setColorAt(0.0, QColor(255, 255, 255, 70))
    sheen.setColorAt(0.5, QColor(255, 255, 255, 14))
    sheen.setColorAt(1.0, QColor(255, 255, 255, 0))
    p.setBrush(QBrush(sheen))
    p.drawRect(QRectF(-380, -230, 760, 480))
    p.setBrush(Qt.BrushStyle.NoBrush)
    centre = QPointF(0, -34)
    for k in range(-5, 6):                                   # web: radial threads
        ang = math.radians(-90 + k * 15)
        p.setPen(QPen(QColor(255, 255, 255, 34 if k % 2 else 52), 1.1))
        p.drawLine(centre, QPointF(centre.x() + 420 * math.cos(ang), centre.y() + 420 * math.sin(ang)))
    for r in (70, 130, 200, 275):                            # web: sagging cross threads
        pts = [QPointF(centre.x() + r * math.cos(math.radians(-90 + k * 15)),
                       centre.y() + r * math.sin(math.radians(-90 + k * 15))) for k in range(-5, 6)]
        for a, b in zip(pts, pts[1:]):
            seg = QPainterPath(a)
            seg.quadTo(QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2 + r * 0.035), b)
            p.setPen(QPen(QColor(255, 255, 255, 40), 1.0))
            p.drawPath(seg)
    vol = QLinearGradient(0, -20, 0, 240)
    vol.setColorAt(0.0, QColor(0, 0, 0, 0))
    vol.setColorAt(1.0, QColor(0, 0, 0, 190))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(vol))
    p.drawRect(QRectF(-380, -20, 760, 270))
    p.restore()

    seam = QPainterPath(QPointF(0, -212))
    seam.cubicTo(QPointF(4, -90), QPointF(10, 40), QPointF(0, 240))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor(0, 0, 0, 200), 5))
    p.drawPath(seam)
    p.setPen(QPen(pal.with_alpha(red, 120), 1.2))
    p.drawPath(seam.translated(2.5, 0))

    for sign in (-1, 1):                                     # eye cavities with metal bezels
        eye = _eye_path(sign)
        stroker = QPainterPathStroker()
        stroker.setWidth(20)
        stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        bz = QLinearGradient(sign * 40, -70, sign * 300, 110)
        bz.setColorAt(0.0, QColor(58, 60, 68))
        bz.setColorAt(0.5, QColor(16, 16, 20))
        bz.setColorAt(1.0, QColor(34, 35, 40))
        p.setPen(QPen(pal.with_alpha(red, 190), 2.4))
        p.setBrush(QBrush(bz))
        p.drawPath(stroker.createStroke(eye))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(8, 8, 10)))
        p.drawPath(eye)

    for sign in (-1, 1):                                     # mandibles
        blade = QPainterPath(QPointF(sign * 36, 236))
        blade.lineTo(QPointF(sign * 92, 232))
        blade.lineTo(QPointF(sign * 70, 284))
        blade.lineTo(QPointF(sign * 44, 258))
        blade.closeSubpath()
        bl = QLinearGradient(sign * 36, 236, sign * 92, 284)
        bl.setColorAt(0.0, QColor(46, 47, 54))
        bl.setColorAt(1.0, QColor(6, 6, 9))
        p.setPen(QPen(pal.with_alpha(red, 180), 1.6))
        p.setBrush(QBrush(bl))
        p.drawPath(blade)

    rim = QLinearGradient(-360, 0, 360, 0)
    rim.setColorAt(0.0, pal.with_alpha(red, 240))
    rim.setColorAt(0.28, pal.with_alpha(red, 40))
    rim.setColorAt(0.72, pal.with_alpha(red, 40))
    rim.setColorAt(1.0, pal.with_alpha(red, 240))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(pal.with_alpha(red, 60), 9))
    p.drawPath(head)
    p.setPen(QPen(QBrush(rim), 2.6))
    p.drawPath(head)
    hl = QLinearGradient(-300, -190, 300, 200)
    hl.setColorAt(0.0, QColor(255, 255, 255, 120))
    hl.setColorAt(0.4, QColor(255, 255, 255, 0))
    p.setPen(QPen(QBrush(hl), 1.4))
    p.drawPath(QTransform().scale(0.965, 0.965).map(head))
    p.restore()


def draw_head_lenses(p: QPainter, level: float) -> None:
    """Both glossy lenses at one brightness (0..1). The widget blends a dim and a bright copy."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    bright = 0.30 + 0.70 * max(0.0, min(1.0, level))
    for sign in (-1, 1):
        eye = _eye_path(sign)
        lens = QRadialGradient(QPointF(sign * 150, 6), 215)
        lens.setColorAt(0.0, _mix(QColor(255, 255, 255), QColor(0, 0, 0), 1 - bright))
        lens.setColorAt(0.34, _mix(QColor(214, 217, 224), QColor(0, 0, 0), 1 - bright))
        lens.setColorAt(0.72, _mix(QColor(110, 114, 124), QColor(0, 0, 0), 1 - bright * 0.8))
        lens.setColorAt(1.0, QColor(22, 23, 28))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(lens))
        p.drawPath(eye)
        p.save()
        p.setClipPath(eye)
        streak = QLinearGradient(sign * 90, -60, sign * 260, 20)
        streak.setColorAt(0.0, QColor(255, 255, 255, int(150 * bright)))
        streak.setColorAt(1.0, QColor(255, 255, 255, 0))
        arc = QPainterPath(QPointF(sign * 70, 18))
        arc.cubicTo(QPointF(sign * 130, -34), QPointF(sign * 220, -58), QPointF(sign * 300, -50))
        p.setPen(QPen(QBrush(streak), 9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(arc)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(255, 255, 255, int(220 * bright))))
        p.drawEllipse(QPointF(sign * 112, 20), 12, 6)
        shade = QLinearGradient(0, 20, 0, 104)
        shade.setColorAt(0.0, QColor(0, 0, 0, 0))
        shade.setColorAt(1.0, QColor(0, 0, 0, 120))
        p.setBrush(QBrush(shade))
        p.drawRect(QRectF(-380, 20, 760, 100))
        p.restore()
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 46), 1.2))
        p.drawPath(eye)
    p.restore()


def draw_head_tint(p: QPainter, red: QColor) -> None:
    """Both eyes filled with red (the widget fades this in while Spidy is thinking or working)."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(pal.with_alpha(red, 255)))
    for sign in (-1, 1):
        p.drawPath(_eye_path(sign))
    p.restore()


def draw_head(p: QPainter, cx: float, cy: float, scale: float, level: float = 0.5, mood: str = "idle",
              tilt: tuple[float, float] = (0.0, 0.0), phase: float = 0.0) -> None:
    """Draw the whole face directly (used by tests and one-off renders; the widget caches the layers)."""
    p.save()
    t = QTransform()
    t.translate(cx, cy)
    t.rotate(tilt[0], Qt.Axis.YAxis)
    t.rotate(tilt[1], Qt.Axis.XAxis)
    t.scale(scale, scale)
    p.setTransform(t, True)
    red = head_red(mood)
    draw_head_glow(p, red)
    draw_head_body(p, red)
    draw_head_lenses(p, level)
    if mood in ("thinking", "working"):
        p.setOpacity(0.2 + 0.2 * math.sin(phase * 3) ** 2)
        draw_head_tint(p, red)
    p.restore()


# ── Line-art spider (small logo) ────────────────────────────────────────────
_LINE_LEGS = [[(10, -10), (34, -54), (62, -72), (82, -50)], [(12, -4), (50, -28), (92, -34), (108, -8)],
              [(12, 4), (50, 8), (92, 18), (108, 48)], [(10, 10), (38, 38), (66, 66), (78, 98)]]


def draw_spider_line(p: QPainter, cx: float, cy: float, scale: float, color: QColor, width: float = 1.6) -> None:
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.translate(cx, cy)
    p.scale(scale, scale)
    pen = QPen(color, width / scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for sign in (1, -1):
        for leg in _LINE_LEGS:
            path = QPainterPath(QPointF(sign * leg[0][0], leg[0][1]))
            for x, y in leg[1:]:
                path.lineTo(sign * x, y)
            p.drawPath(path)
    p.setBrush(QBrush(color))
    p.drawEllipse(QPointF(0, 50), 17, 28)
    p.drawEllipse(QPointF(0, 6), 11, 11)
    p.drawEllipse(QPointF(0, -14), 8, 8)
    p.restore()


# ── Cards ───────────────────────────────────────────────────────────────────

def draw_card(p: QPainter, rect: QRectF, radius: float = 20.0, strength: float = 1.0,
              border: QColor | None = None) -> None:
    """Dark glass card: faint gradient fill, hairline border and a soft top highlight."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    fill = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    fill.setColorAt(0.0, QColor(255, 255, 255, int(15 * strength)))
    fill.setColorAt(1.0, QColor(255, 255, 255, int(4 * strength)))
    p.setBrush(QBrush(fill))
    p.setPen(QPen(border or QColor(255, 255, 255, 30), 1.0))
    p.drawRoundedRect(rect, radius, radius)
    hi = QLinearGradient(rect.topLeft(), rect.topRight())
    hi.setColorAt(0.0, QColor(255, 255, 255, 0))
    hi.setColorAt(0.5, QColor(255, 255, 255, int(50 * strength)))
    hi.setColorAt(1.0, QColor(255, 255, 255, 0))
    p.setPen(QPen(QBrush(hi), 1.0))
    p.drawLine(QPointF(rect.left() + radius, rect.top() + 0.5), QPointF(rect.right() - radius, rect.top() + 0.5))
    p.restore()


# ── Background ──────────────────────────────────────────────────────────────

def make_background(w: int, h: int, dpr: float = 1.0, k: float = 1.0) -> QPixmap:
    """Near-black backdrop, a faint red glow behind the hero, film grain and the rounded outer frame."""
    pm = QPixmap(max(1, int(w * dpr)), max(1, int(h * dpr)))
    pm.setDevicePixelRatio(dpr)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.fillRect(0, 0, w, h, QColor(4, 4, 6))

    glow = QRadialGradient(QPointF(w * 0.47, h * 0.34), w * 0.42)
    glow.setColorAt(0.0, QColor(170, 20, 30, 46))
    glow.setColorAt(0.6, QColor(120, 14, 22, 14))
    glow.setColorAt(1.0, QColor(120, 14, 22, 0))
    p.fillRect(0, 0, w, h, QBrush(glow))
    floor = QRadialGradient(QPointF(w * 0.6, h * 1.02), w * 0.5)
    floor.setColorAt(0.0, QColor(255, 255, 255, 10))
    floor.setColorAt(1.0, QColor(255, 255, 255, 0))
    p.fillRect(0, 0, w, h, QBrush(floor))

    rnd = random.Random(21)
    p.setPen(Qt.PenStyle.NoPen)
    for _ in range(int(w * h / 900)):
        x, y = rnd.uniform(0, w), rnd.uniform(0, h)
        p.setBrush(QBrush(QColor(255, 255, 255, rnd.randint(3, 12))))
        p.drawRect(QRectF(x, y, 1, 1))

    vg = QRadialGradient(QPointF(w / 2, h / 2), math.hypot(w, h) * 0.56)
    vg.setColorAt(0.6, QColor(0, 0, 0, 0))
    vg.setColorAt(1.0, QColor(0, 0, 0, 190))
    p.fillRect(0, 0, w, h, QBrush(vg))

    inset = 24 * k
    frame = QRectF(inset, inset, w - 2 * inset, h - 2 * inset)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor(255, 255, 255, 34), 1.0))
    p.drawRoundedRect(frame, 40 * k, 40 * k)
    p.setPen(QPen(QColor(255, 255, 255, 12), 1.0))
    p.drawRoundedRect(frame.adjusted(5, 5, -5, -5), 36 * k, 36 * k)
    p.end()
    return pm

