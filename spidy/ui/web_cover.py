"""
Cobwebs after a long break
==========================
If the laptop sat unused for IDLE_SECS (no keyboard / mouse, or it was asleep / lid closed that
long), the whole screen gets spun over with old, dusty cobwebs -- dense tangled sheets, tattered
corner webs and long sagging threads, each with a soft blurry glow -- and the desktop pet sleeps in
a hammock up in the corner.

Move the cursor through them: every thread the cursor gets close to snaps and drops away. When the
last one is gone the cover hides and the pet wakes up. The window is click-through, so the
computer stays fully usable while the webs are up.
"""

from __future__ import annotations

import ctypes
import math
import random
import time

from PySide6.QtCore import QObject, QPointF, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QGuiApplication, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

IDLE_SECS = 3600.0          # one hour away
CUT_RADIUS = 38.0           # how close the cursor must come to a thread to snap it
GROW_SECS = 5.0             # the webs are spun in over this long when they first appear
FALL_SECS = 0.45


def idle_seconds() -> float:
    """Seconds since the last keyboard / mouse input (Windows); 0 elsewhere."""
    try:
        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(info)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return 0.0
        return max(0.0, (ctypes.windll.kernel32.GetTickCount() - info.dwTime) / 1000.0)
    except Exception:  # noqa: BLE001
        return 0.0


def _seg_dist(p: QPointF, a: QPointF, b: QPointF) -> float:
    dx, dy = b.x() - a.x(), b.y() - a.y()
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / L))
    return math.hypot(p.x() - (a.x() + t * dx), p.y() - (a.y() + t * dy))


def _strand(a: QPointF, b: QPointF, rng: random.Random, sag: float = 0.12, order: float = 0.0) -> dict:
    """A slightly sagging, curved silk thread from a to b."""
    mx, my = (a.x() + b.x()) / 2, (a.y() + b.y()) / 2
    L = math.hypot(b.x() - a.x(), b.y() - a.y())
    nx, ny = (-(b.y() - a.y()) / L, (b.x() - a.x()) / L) if L else (0.0, 0.0)
    bend = rng.uniform(-sag, sag) * L
    c = QPointF(mx + nx * bend, my + ny * bend + abs(bend) * 0.5)          # gravity pulls the middle down
    mid = QPointF(0.25 * a.x() + 0.5 * c.x() + 0.25 * b.x(), 0.25 * a.y() + 0.5 * c.y() + 0.25 * b.y())
    return {"a": a, "b": b, "c": c, "m": mid, "order": order,
            "w": rng.uniform(0.35, 1.1), "lum": rng.uniform(0.25, 1.0), "cut": None}


def _tangle(c: QPointF, rx: float, ry: float, n: int, rng: random.Random, order: float, out: list) -> None:
    """A messy cobweb sheet: random knots joined to their nearest neighbours by curved threads."""
    knots = [QPointF(c.x() + rng.gauss(0, rx), c.y() + rng.gauss(0, ry)) for _ in range(n)]
    for i, k in enumerate(knots):
        near = sorted(range(n), key=lambda j: (knots[j].x() - k.x()) ** 2 + (knots[j].y() - k.y()) ** 2)[1:5]
        for j in near:
            if j > i or rng.random() < 0.3:
                out.append(_strand(k, knots[j], rng, 0.09, order + rng.uniform(0, 0.5)))
    for _ in range(n // 5):                                             # long threads out of the sheet
        k = rng.choice(knots)
        ang = rng.uniform(0, math.tau)
        far = QPointF(k.x() + math.cos(ang) * rx * 2.2, k.y() + math.sin(ang) * ry * 2.2)
        out.append(_strand(k, far, rng, 0.1, order + rng.uniform(0, 0.4)))


def build_webs(area: QRect, rng: random.Random | None = None) -> list[dict]:
    """Threads for a screen: dense tangled sheets, tattered corner webs and long sagging threads."""
    rng = rng or random.Random()
    segs: list[dict] = []
    L, T, R, B = area.left(), area.top(), area.right(), area.bottom()
    W, H = area.width(), area.height()
    short = min(W, H)

    def orb(c: QPointF, r: float, a0: float, a1: float, spokes: int, rings: int, order0: float) -> None:
        angs = [a0 + (a1 - a0) * i / (spokes - 1) + rng.uniform(-0.06, 0.06) for i in range(spokes)]
        ends = [QPointF(c.x() + math.cos(a) * r * rng.uniform(0.85, 1.1),
                        c.y() + math.sin(a) * r * rng.uniform(0.85, 1.1)) for a in angs]
        for e in ends:
            segs.append(_strand(QPointF(c), e, rng, 0.03, order0 + rng.uniform(0, 0.15)))
        for k in range(1, rings + 1):
            f = (k / (rings + 1)) ** 0.85
            pts = [QPointF(c.x() + (e.x() - c.x()) * f * rng.uniform(0.93, 1.05),
                           c.y() + (e.y() - c.y()) * f * rng.uniform(0.93, 1.05)) for e in ends]
            for a, b in zip(pts, pts[1:]):
                if rng.random() < 0.9:                                   # old webs have gaps
                    segs.append(_strand(a, b, rng, 0.2, order0 + 0.15 + 0.35 * k / rings))

    orb(QPointF(L, T), short * 0.40, 0.0, math.pi / 2, 10, 10, 0.0)
    orb(QPointF(R, T), short * 0.36, math.pi / 2, math.pi, 9, 9, 0.05)
    orb(QPointF(L, B), short * 0.30, -math.pi / 2, 0.0, 8, 8, 0.1)
    orb(QPointF(R, B), short * 0.34, math.pi, 1.5 * math.pi, 9, 9, 0.12)
    # dense tangled sheets, heaviest along a diagonal band like old dusty webbing
    for i in range(10):
        f = (i + rng.uniform(0.1, 0.9)) / 10
        c = QPointF(L + W * f, T + H * (0.12 + 0.76 * (1 - f)) + rng.uniform(-H * 0.1, H * 0.1))
        _tangle(c, W * rng.uniform(0.05, 0.08), H * rng.uniform(0.07, 0.11), rng.randint(110, 150),
                rng, 0.1 + 0.04 * i, segs)
    for _ in range(6):                                                   # lighter patches elsewhere
        c = QPointF(rng.uniform(L, R), rng.uniform(T, B))
        _tangle(c, W * 0.04, H * 0.05, rng.randint(35, 55), rng, 0.3, segs)

    def on(side: str) -> QPointF:
        return {"top": QPointF(rng.uniform(L, R), T), "bottom": QPointF(rng.uniform(L, R), B),
                "left": QPointF(L, rng.uniform(T, B)), "right": QPointF(R, rng.uniform(T, B))}[side]

    for _ in range(10):                                                  # long sagging threads
        a, b = (on(x) for x in rng.sample(["top", "left", "right", "bottom"], 2))
        n = 5
        pts = [QPointF(a.x() + (b.x() - a.x()) * t,
                       a.y() + (b.y() - a.y()) * t + math.sin(math.pi * t) * rng.uniform(20, 90))
               for t in (i / n for i in range(n + 1))]
        o = rng.uniform(0.0, 0.4)
        for p, q in zip(pts, pts[1:]):
            segs.append(_strand(p, q, rng, 0.05, o))
    return segs


class WebCover(QWidget):
    """Full-screen, click-through cobweb layer that the cursor sweeps away."""

    cleared = Signal()

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool | Qt.WindowType.NoDropShadowWindowHint
                            | Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.segs: list[dict] = []
        self.shown_at = 0.0
        self._cache: QImage | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._step)
        self._now = time.monotonic

    @property
    def active(self) -> bool:
        return self.isVisible() and bool(self.segs)

    def cover(self) -> None:
        screen = QGuiApplication.primaryScreen()
        area = screen.geometry() if screen else QRect(0, 0, 1920, 1080)
        self.setGeometry(area)
        self.segs = build_webs(QRect(0, 0, area.width(), area.height()))
        self.shown_at = self._now()
        self._cache = None
        self.show()
        self._timer.start()

    def sweep(self, cursor: QPointF) -> int:
        """Snap every grown thread near the cursor (local coords). Returns how many snapped."""
        now = self._now()
        grown = (now - self.shown_at) / GROW_SECS
        n = 0
        for s in self.segs:
            if s["cut"] is None and s["order"] <= grown and \
                    min(_seg_dist(cursor, s["a"], s["m"]), _seg_dist(cursor, s["m"], s["b"])) < CUT_RADIUS:
                s["cut"] = now
                n += 1
        return n

    def _step(self) -> None:
        now = self._now()
        cur = QPointF(QCursor.pos()) - QPointF(self.x(), self.y())
        snapped = self.sweep(cur)
        before = len(self.segs)
        self.segs = [s for s in self.segs if s["cut"] is None or now - s["cut"] < FALL_SECS]
        if not self.segs:
            self._timer.stop()
            self.hide()
            self.cleared.emit()
            return
        if (now - self.shown_at < GROW_SECS + 0.3 or snapped or len(self.segs) != before
                or any(s["cut"] is not None for s in self.segs)):
            self._cache = None                                        # something changed: redraw
            self.update()

    def _draw(self, p: QPainter, now: float) -> None:
        grown = (now - self.shown_at) / GROW_SECS
        glow, line = QPen(), QPen()
        glow.setCapStyle(Qt.PenCapStyle.RoundCap)
        line.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for s in self.segs:
            if s["order"] > grown:
                continue
            a, b, c = s["a"], s["b"], s["c"]
            alpha = 1.0
            if s["cut"] is not None:                                   # snapped: droops down and fades
                k = (now - s["cut"]) / FALL_SECS
                alpha = max(0.0, 1 - k)
                drop = 60 * k * k
                a, b, c = (QPointF(q.x(), q.y() + drop * f) for q, f in ((a, 1.0), (b, 1.3), (c, 1.6)))
            fresh = min(1.0, (grown - s["order"]) * 4)
            path = QPainterPath(a)
            path.quadTo(c, b)
            lum = s["lum"] * alpha * fresh
            line.setColor(QColor(242, 246, 255, int(205 * lum)))
            line.setWidthF(s["w"])
            p.setPen(line)
            p.drawPath(path)

    def paintEvent(self, _e) -> None:
        if self._cache is None or self._cache.size() != self.size():
            img = QImage(self.size(), QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(Qt.GlobalColor.transparent)
            q = QPainter(img)
            q.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._draw(q, self._now())
            q.end()
            # soft blur: a downscaled + upscaled copy underneath gives the hazy, glowing look
            small = img.scaled(max(1, img.width() // 6), max(1, img.height() // 6),
                               Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            haze = small.scaled(img.size(), Qt.AspectRatioMode.IgnoreAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
            out = QImage(self.size(), QImage.Format.Format_ARGB32_Premultiplied)
            out.fill(Qt.GlobalColor.transparent)
            q = QPainter(out)
            q.setOpacity(1.0)
            q.drawImage(0, 0, haze)
            q.drawImage(0, 0, haze)
            q.setOpacity(0.9)
            q.drawImage(0, 0, img)
            q.end()
            self._cache = out
        p = QPainter(self)
        p.drawImage(0, 0, self._cache)
        p.end()


class LongAwayWatcher(QObject):
    """Shows the cobwebs (and puts the pet to sleep in its hammock) after an hour away."""

    def __init__(self, pet=None, idle_secs: float = IDLE_SECS, parent=None) -> None:
        super().__init__(parent)
        self.pet = pet
        self.idle_secs = idle_secs
        self.cover = WebCover()
        self.cover.cleared.connect(self._on_cleared)
        self._last_poll = time.time()
        self._timer = QTimer(self)
        self._timer.setInterval(20_000)
        self._timer.timeout.connect(self.poll)

    def start(self) -> None:
        self._last_poll = time.time()
        self._timer.start()

    def poll(self, idle: float | None = None, now: float | None = None) -> None:
        now = time.time() if now is None else now
        gap = now - self._last_poll                       # big gap = the laptop was asleep / lid shut
        self._last_poll = now
        idle = idle_seconds() if idle is None else idle
        if self.cover.active:
            return
        if idle >= self.idle_secs or gap >= self.idle_secs:
            self.trigger()

    def trigger(self) -> None:
        self.cover.cover()
        if self.pet is not None:
            try:
                self.pet.sleep_in_hammock()
                self.pet.raise_()
            except Exception:  # noqa: BLE001
                pass

    def _on_cleared(self) -> None:
        if self.pet is not None:
            try:
                self.pet.wake_from_hammock()
            except Exception:  # noqa: BLE001
                pass
