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

import numpy as np

from PySide6.QtCore import QObject, QPointF, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QGuiApplication, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

IDLE_SECS = 3600.0          # one hour away
CUT_RADIUS = 60.0           # how close the cursor must come to a thread to snap it
GROW_SECS = 5.0             # the webs are spun in over this long when they first appear
FALL_SECS = 0.9             # a snapped thread tumbles down for this long
GRAVITY = 900.0
CHAIN_HOPS = 3              # a snap pulls down threads tied to it, this many knots deep
LAST_BITS = 0.12            # when only this fraction is left, the rest falls on its own


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

    def clamp(q: QPointF) -> QPointF:                    # every thread stays where the cursor can reach it
        return QPointF(min(max(q.x(), L), R), min(max(q.y(), T), B))
    for sg in segs:
        for key in ("a", "b", "c", "m"):
            sg[key] = clamp(sg[key])
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
        self._built_at, self._stale = 0.0, True
        self._erase, self._last_snap = [], 0.0
        self._rng = random.Random()
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._step)
        self._now = time.monotonic

    @property
    def active(self) -> bool:
        return self.isVisible() and any(not s.get("gone") for s in self.segs)

    def cover(self) -> None:
        screen = QGuiApplication.primaryScreen()
        area = screen.geometry() if screen else QRect(0, 0, 1920, 1080)
        self.setGeometry(area)
        self.load(build_webs(QRect(0, 0, area.width(), area.height())))
        self.show()
        self._timer.start()

    def load(self, segs: list[dict]) -> None:
        """Take a set of threads and work out which ones are tied together at a knot."""
        self.segs = segs
        self.shown_at = self._now()
        # flat arrays so the cursor check runs in numpy (thousands of threads, every frame)
        self._P = np.array([[sg["a"].x(), sg["a"].y(), sg["m"].x(), sg["m"].y(), sg["b"].x(), sg["b"].y()]
                            for sg in segs], dtype=np.float32).reshape(-1, 6)
        self._order = np.array([sg["order"] for sg in segs], dtype=np.float32)
        self._alive = np.ones(len(segs), dtype=bool)
        self._cache, self._built_at, self._stale = None, 0.0, True
        self._erase, self._last_snap = [], 0.0
        grid: dict[tuple[int, int], list[int]] = {}
        for i, sg in enumerate(segs):
            sg.update(cut=None, pend=None, gone=False, nb=[], idx=i)
            for q in (sg["a"], sg["b"]):
                if not self._on_edge(q):
                    grid.setdefault((int(q.x() // 6), int(q.y() // 6)), []).append(i)
        for i, sg in enumerate(segs):
            nb = set()
            for q in (sg["a"], sg["b"]):
                if self._on_edge(q):
                    continue
                cx, cy = int(q.x() // 6), int(q.y() // 6)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nb.update(grid.get((cx + dx, cy + dy), ()))
            nb.discard(i)
            sg["nb"] = list(nb)

    def _on_edge(self, q: QPointF) -> bool:
        return q.x() <= 3 or q.y() <= 3 or q.x() >= self.width() - 4 or q.y() >= self.height() - 4

    def _snap(self, sg: dict, now: float, hop: int = 0) -> None:
        if sg["cut"] is not None:
            return
        r = self._rng
        if "idx" in sg:
            self._alive[sg["idx"]] = False
        sg.update(cut=now, vx=r.uniform(-60, 60), vy=r.uniform(-50, 10), spin=r.uniform(-2.8, 2.8),
                  sway=r.uniform(0, math.tau))
        self._erase.append(sg)
        self._last_snap = now
        self._stale = True
        if hop < CHAIN_HOPS:                         # threads tied to this one lose their hold too
            for j in sg["nb"]:
                o = self.segs[j]
                if o["cut"] is None and o["pend"] is None and r.random() < 0.6:
                    o["pend"] = (now + r.uniform(0.04, 0.12), hop + 1)

    def sweep(self, cursor: QPointF) -> int:
        """Snap every grown thread near the cursor (local coords). Returns how many snapped."""
        now = self._now()
        grown = (now - self.shown_at) / GROW_SECS
        if not len(self.segs):
            return 0
        P, cx, cy = self._P, cursor.x(), cursor.y()

        def dist(ax, ay, bx, by):
            dx, dy = bx - ax, by - ay
            L = dx * dx + dy * dy
            t = np.clip(((cx - ax) * dx + (cy - ay) * dy) / np.maximum(L, 1e-6), 0.0, 1.0)
            return np.hypot(cx - (ax + t * dx), cy - (ay + t * dy))
        d = np.minimum(dist(P[:, 0], P[:, 1], P[:, 2], P[:, 3]), dist(P[:, 2], P[:, 3], P[:, 4], P[:, 5]))
        mask = self._alive & (self._order <= grown) & (d < CUT_RADIUS)
        hits = [self.segs[i] for i in np.nonzero(mask)[0]]
        for sg in hits:
            self._snap(sg, now)
        return len(hits)

    def _step(self) -> None:
        now = self._now()
        cur = QPointF(QCursor.pos()) - QPointF(self.x(), self.y())
        last, self._last_cur = getattr(self, "_last_cur", None), cur
        if last is not None and (cur - last).manhattanLength() > 3 and now - self.shown_at > GROW_SECS:
            self.sweep(cur)
        standing = 0
        for sg in self.segs:
            if sg["cut"] is None:
                if sg["pend"] is not None and now >= sg["pend"][0]:
                    self._snap(sg, now, sg["pend"][1])
                else:
                    standing += 1
            elif not sg["gone"] and now - sg["cut"] >= FALL_SECS:
                sg["gone"] = True
        grown = now - self.shown_at >= GROW_SECS
        if grown and 0 < standing <= len(self.segs) * LAST_BITS:          # the last scraps let go
            for sg in self.segs:
                if sg["cut"] is None and sg["pend"] is None:
                    sg["pend"] = (now + self._rng.uniform(0.0, 0.6), CHAIN_HOPS)
        if not any(not sg["gone"] for sg in self.segs):
            self._timer.stop()
            self.hide()
            self.cleared.emit()
            return
        if not grown and now - self._built_at > 0.15:                      # still being spun in
            self._cache = None
        elif self._erase and self._cache is not None:                       # cheap: rub out just the snapped threads
            q = QPainter(self._cache)
            q.setRenderHint(QPainter.RenderHint.Antialiasing)
            q.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            pen = QPen(Qt.GlobalColor.black)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            q.setBrush(Qt.BrushStyle.NoBrush)
            for sg in self._erase:
                path = QPainterPath(sg["a"])
                path.quadTo(sg["c"], sg["b"])
                pen.setWidthF(sg["w"] + 9)                                   # also clears its blurry halo
                q.setPen(pen)
                q.drawPath(path)
            q.end()
        elif grown and self._stale and now - self._last_snap > 0.8:         # calm again: tidy full redraw
            self._cache = None
        self._erase = []
        self.update()

    def _fallen(self, sg: dict, now: float):
        """Where a snapped thread is now: limp, tumbling and drifting down."""
        t = now - sg["cut"]
        k = min(1.0, t / FALL_SECS)
        m = sg["m"]
        ang = sg["spin"] * t
        ca, sa = math.cos(ang), math.sin(ang)
        dx = sg["vx"] * t + 18 * math.sin(sg["sway"] + t * 5) * k          # floats side to side
        dy = sg["vy"] * t + 0.5 * GRAVITY * t * t
        shrink = 1 - 0.45 * k                                              # curls up as it falls

        def move(q: QPointF, limp: float = 0.0) -> QPointF:
            rx, ry = (q.x() - m.x()) * shrink, (q.y() - m.y()) * shrink + limp
            return QPointF(m.x() + rx * ca - ry * sa + dx, m.y() + rx * sa + ry * ca + dy)
        return move(sg["a"]), move(sg["b"]), move(sg["c"], 40 * k), max(0.0, 1 - k ** 1.6)

    def _draw_still(self, p: QPainter, now: float) -> None:
        grown = (now - self.shown_at) / GROW_SECS
        line = QPen()
        line.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for sg in self.segs:
            if sg["cut"] is not None or sg["order"] > grown:
                continue
            path = QPainterPath(sg["a"])
            path.quadTo(sg["c"], sg["b"])
            lum = sg["lum"] * min(1.0, (grown - sg["order"]) * 4)
            line.setColor(QColor(242, 246, 255, int(205 * lum)))
            line.setWidthF(sg["w"])
            p.setPen(line)
            p.drawPath(path)

    def _build_cache(self, now: float) -> None:
        img = QImage(self.size(), QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(Qt.GlobalColor.transparent)
        q = QPainter(img)
        q.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_still(q, now)
        q.end()
        # soft blur: a downscaled + upscaled copy underneath gives the hazy, glowing look
        small = img.scaled(max(1, img.width() // 6), max(1, img.height() // 6),
                           Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        haze = small.scaled(img.size(), Qt.AspectRatioMode.IgnoreAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        out = QImage(self.size(), QImage.Format.Format_ARGB32_Premultiplied)
        out.fill(Qt.GlobalColor.transparent)
        q = QPainter(out)
        q.drawImage(0, 0, haze)
        q.drawImage(0, 0, haze)
        q.setOpacity(0.9)
        q.drawImage(0, 0, img)
        q.end()
        self._cache, self._built_at, self._stale = out, now, False
        self._erase = []

    def paintEvent(self, _e) -> None:
        now = self._now()
        if self._cache is None or self._cache.size() != self.size():
            self._build_cache(now)
        p = QPainter(self)
        p.drawImage(0, 0, self._cache)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(Qt.BrushStyle.NoBrush)
        line = QPen()
        falling = [sg for sg in self.segs if sg["cut"] is not None and not sg["gone"]]
        if len(falling) > 380:                                             # keep it smooth: newest snaps only
            falling.sort(key=lambda sg: sg["cut"], reverse=True)
            for sg in falling[380:]:
                sg["gone"] = True
            falling = falling[:380]
        for sg in falling:                                                 # the threads falling right now
            a, b, c, alpha = self._fallen(sg, now)
            if alpha * sg["lum"] < 0.04:
                continue
            path = QPainterPath(a)
            path.quadTo(c, b)
            line.setColor(QColor(245, 248, 255, int(235 * alpha * sg["lum"])))
            line.setWidthF(sg["w"] * 1.3)
            p.setPen(line)
            p.drawPath(path)
        p.end()


def screen_locked() -> bool:
    """True while Windows shows the lock screen (the input desktop can't be switched to)."""
    try:
        u = ctypes.windll.user32
        h = u.OpenInputDesktop(0, False, 0x0100)          # DESKTOP_SWITCHDESKTOP
        if not h:
            return True
        ok = u.SwitchDesktop(h)
        u.CloseDesktop(h)
        return not ok
    except Exception:  # noqa: BLE001
        return False


class LongAwayWatcher(QObject):
    """Cobwebs + hammock nap at start-up and after an hour away; the door intro after an unlock."""

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
        self._was_locked = False
        self._lock_timer = QTimer(self)
        self._lock_timer.setInterval(1500)
        self._lock_timer.timeout.connect(self.check_lock)

    def start(self, boot: bool = True) -> None:
        self._last_poll = time.time()
        self._timer.start()
        self._lock_timer.start()
        if boot:                                          # laptop just switched on: webs + hammock
            QTimer.singleShot(1200, self.trigger)

    def check_lock(self, locked: bool | None = None, now: float | None = None) -> None:
        locked = screen_locked() if locked is None else locked
        now = time.time() if now is None else now
        gap = now - getattr(self, "_last_lock_poll", now)    # big gap: the laptop slept, screen just woke
        self._last_lock_poll = now
        if locked:
            self._was_locked = True
            return
        woke = gap > 30.0
        if self._was_locked or woke:
            self._was_locked = False
            if gap >= self.idle_secs:
                return                                    # gone an hour+: poll() spins the webs instead
            if self.pet is not None and not self.cover.active:
                try:
                    self.pet.play_door_intro()
                except Exception:  # noqa: BLE001
                    pass

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
