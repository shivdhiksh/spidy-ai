"""
Pet -> HUD morph
================
When the HUD opens while the desktop pet is on screen (pet clicked, "Hey Spidy", "Wake up Spidy"),
the pet bursts into holographic violet dots that roll round the screen in a ring and spiral into the HUD's
dot blob, and the HUD fades in underneath them. The whole thing takes TOTAL (1.3 s).

Timeline (seconds)
    0.00-0.30  dissolve: the pet's pixels become glowing dots that drift apart
    0.30-0.85  flight: the dots stream along curved paths toward the blob (staggered)
    0.85       the HUD starts its own fade-in
    0.85-1.30  swirl: the dots settle onto a spinning sphere where the blob is, then fade out
"""

from __future__ import annotations

import math
import random

from PySide6.QtCore import QObject, QPointF, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QRadialGradient
from PySide6.QtWidgets import QWidget

TOTAL = 1.80
DISSOLVE = 0.15          # burst: the pet is gone almost at once
GHOST = 0.06             # how long the pet picture lingers while it bursts
RING_IN = 0.45           # dots fly out onto a big ring around the screen
SPIRAL_AT = 0.95         # ring starts spiralling into the UI
FLIGHT_END = 1.40        # every dot has reached the blob sphere
HUD_AT = 1.20
MAX_DOTS = 520
_VIOLET = (168, 120, 255)
_CYAN = (120, 220, 255)


def _ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def sample_dots(img: QImage, origin: QPointF, step: int = 4, limit: int = MAX_DOTS, rng=None) -> list[dict]:
    """Opaque pixels of the pet's picture -> dots (screen position + the pixel's colour)."""
    rng = rng or random.Random(7)
    pts = []
    for y in range(0, img.height(), step):
        for x in range(0, img.width(), step):
            c = img.pixelColor(x, y)
            if c.alpha() > 90:
                pts.append((origin.x() + x, origin.y() + y, c))
    if len(pts) > limit:
        pts = rng.sample(pts, limit)
    dots = []
    for x, y, c in pts:
        ang = rng.uniform(0, math.tau)
        dots.append({
            "sx": x, "sy": y, "c": (c.red(), c.green(), c.blue()),
            "drift": (math.cos(ang) * rng.uniform(4, 22), math.sin(ang) * rng.uniform(4, 22)),
            "delay": rng.uniform(0.0, 0.18),
            "bend": rng.uniform(-0.45, 0.45),
            "theta": rng.uniform(0, math.tau), "phi": math.acos(rng.uniform(-1, 1)),
            "size": rng.uniform(1.6, 3.2),
        })
    return dots


def dot_state(d: dict, t: float, target: QPointF, radius: float, ring=None):
    """Where one dot is at time t: (x, y, alpha 0..1, colour mix 0=pet colour .. 1=hologram).

    ring = (centre QPointF, radius): the big circle the dots roll around before flowing into the UI."""
    sx, sy = d["sx"], d["sy"]
    dx, dy = d["drift"]
    k = _ease(t / DISSOLVE)
    px, py = sx + dx * k, sy + dy * k                       # end of the dissolve
    # final spot on the spinning sphere
    spin = d["theta"] + max(0.0, t - FLIGHT_END) * 5.0
    ex = target.x() + radius * math.sin(d["phi"]) * math.cos(spin)
    ey = target.y() + radius * math.cos(d["phi"])
    if t <= DISSOLVE:
        return px, py, 1.0, k
    rc, rr = ring if ring is not None else (target, radius * 2.5)
    ang = d["theta"] + 6.0 * (t - DISSOLVE)                  # rolling round the ring
    # 1) burst position -> onto the ring
    f1 = _ease((t - DISSOLVE - d["delay"]) / RING_IN)
    rx, ry = rc.x() + rr * math.cos(ang), rc.y() + rr * math.sin(ang)
    x, y = px + (rx - px) * f1, py + (ry - py) * f1
    # 2) ring spirals in: centre slides to the blob, radius shrinks, then lands on the sphere
    f2 = _ease((t - SPIRAL_AT - d["delay"] * 0.5) / max(0.05, FLIGHT_END - SPIRAL_AT - d["delay"] * 0.5))
    if f2 > 0:
        cx = rc.x() + (target.x() - rc.x()) * f2
        cy = rc.y() + (target.y() - rc.y()) * f2
        r2 = rr + (radius - rr) * f2
        sx2, sy2 = cx + r2 * math.cos(ang), cy + r2 * math.sin(ang)
        w = f2 * f2
        x, y = sx2 + (ex - sx2) * w, sy2 + (ey - sy2) * w
    alpha = 1.0 if t < FLIGHT_END else max(0.0, 1.0 - (t - FLIGHT_END) / (TOTAL - FLIGHT_END))
    return x, y, alpha, 1.0


class MorphOverlay(QWidget):
    """Transparent, click-through window that paints the flying dots."""

    def __init__(self, dots: list[dict] | None = None, target: QPointF | None = None, radius: float = 100.0,
                 ghost: QImage | None = None, ghost_at: QPointF | None = None, on_hud=None, on_done=None,
                 ring=None, reusable: bool = False) -> None:
        super().__init__(None)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool | Qt.WindowType.NoDropShadowWindowHint
                            | Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.reusable = reusable
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._step)
        import time
        self._clock = time.monotonic
        self._t0 = 0.0
        self.dots, self.t = [], 0.0
        if dots:
            self.load(dots, target, radius, ghost, ghost_at, on_hud, on_done, ring)

    def warm_up(self) -> None:
        """Create the native window now, so the first morph starts on the very next frame."""
        self.setGeometry(-10, -10, 2, 2)
        self.winId()

    def load(self, dots, target, radius, ghost, ghost_at, on_hud, on_done, ring=None) -> None:
        self.dots, self.target, self.radius = dots, target, radius
        scr = QGuiApplication.screenAt(target.toPoint()) or QGuiApplication.primaryScreen()
        if ring is None and scr is not None:
            g = scr.geometry()
            ring = (QPointF(g.center()), 0.36 * min(g.width(), g.height()))
        self.ring = ring
        self.ghost, self.ghost_at = ghost, ghost_at
        self._on_hud, self._on_done = on_hud, on_done
        self._hud_fired = False
        self.t = 0.0
        # cover everything between the pet and the HUD
        xs = [d["sx"] for d in dots] + [target.x() - radius * 1.6, target.x() + radius * 1.6]
        ys = [d["sy"] for d in dots] + [target.y() - radius * 1.6, target.y() + radius * 1.6]
        box = QRect(int(min(xs)) - 260, int(min(ys)) - 260, int(max(xs) - min(xs)) + 520,
                    int(max(ys) - min(ys)) + 520)
        virt = QRect()
        for s in QGuiApplication.screens():
            virt = virt.united(s.geometry())
        if self.ring is not None:
            c, r = self.ring
            box = box.united(QRect(int(c.x() - r) - 40, int(c.y() - r) - 40, int(2 * r) + 80, int(2 * r) + 80))
        self.setGeometry(box.intersected(virt) if not virt.isNull() else box)

    def run(self) -> None:
        self._t0 = self._clock()
        self.t = 0.0
        self.show()
        self.raise_()
        self.repaint()                        # first frame right now, not on the next timer tick
        self._timer.start()

    def _step(self) -> None:
        self.t = self._clock() - self._t0
        if not self._hud_fired and self.t >= HUD_AT:
            self._hud_fired = True
            self._on_hud()
            self.raise_()                     # stay above the HUD while the dots settle
        if self.t >= TOTAL:
            self._timer.stop()
            self.hide()
            self._on_done()
            if not self.reusable:
                self.deleteLater()
            return
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        off = QPointF(self.x(), self.y())
        t = self.t
        # the pet itself fades out as it breaks apart
        if self.ghost is not None and t < GHOST:
            p.setOpacity(1.0 - t / GHOST)
            p.drawImage(self.ghost_at - off, self.ghost)
            p.setOpacity(1.0)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        tgt = self.target - off
        # glow core where the blob forms
        if t > DISSOLVE:
            g = min(1.0, (t - DISSOLVE) / 0.4) * (1.0 if t < FLIGHT_END else
                                                   max(0.0, 1 - (t - FLIGHT_END) / (TOTAL - FLIGHT_END)))
            rg = QRadialGradient(tgt, self.radius * 1.5)
            rg.setColorAt(0.0, QColor(150, 100, 255, int(110 * g)))
            rg.setColorAt(1.0, QColor(60, 20, 140, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(rg)
            p.drawEllipse(tgt, self.radius * 1.5, self.radius * 1.5)
        p.setPen(Qt.PenStyle.NoPen)
        for i, d in enumerate(self.dots):
            x, y, a, mix = dot_state(d, t, self.target, self.radius, self.ring)
            if a <= 0.01:
                continue
            hue = _CYAN if i % 5 == 0 else _VIOLET
            r, g_, b = (int(c0 + (c1 - c0) * mix) for c0, c1 in zip(d["c"], hue))
            pos = QPointF(x, y) - off
            s = d["size"] * (1.0 + 0.6 * mix)
            halo = QRadialGradient(pos, s * 3)
            halo.setColorAt(0.0, QColor(r, g_, b, int(255 * a)))
            halo.setColorAt(0.35, QColor(r, g_, b, int(120 * a)))
            halo.setColorAt(1.0, QColor(r, g_, b, 0))
            p.setBrush(halo)
            p.drawEllipse(pos, s * 3, s * 3)
        p.end()


class HudMorph(QObject):
    """Opens the HUD; when the pet is on screen, opens it through the dots morph instead."""

    def __init__(self, overlay, pet_getter, parent=None) -> None:
        super().__init__(parent)
        self._overlay = overlay
        self._pet = pet_getter
        self._running: MorphOverlay | None = None
        self._view = MorphOverlay(reusable=True)
        self._view.warm_up()

    def _hud_on_screen(self) -> bool:
        ov = self._overlay
        return ov.isVisible() and not ov.isMinimized() and ov.windowOpacity() > 0.05

    def _blob_target(self) -> tuple[QPointF, float]:
        ov = self._overlay
        hero = getattr(ov, "hero", None)
        pos = getattr(type(ov), "_session_pos", None)
        top_left = pos if pos is not None else QPointF(ov.pos())
        if hero is not None and hero.width() > 0:
            c = hero.mapTo(ov, hero.rect().center())
            r = min(hero.width(), hero.height()) * 0.33
            return QPointF(top_left.x() + c.x(), top_left.y() + c.y()), max(40.0, r)
        return QPointF(top_left.x() + ov.width() / 2, top_left.y() + ov.height() / 2), 110.0

    def restore(self) -> None:
        pet = self._pet()
        if self._running is not None:
            return                                          # already morphing: the HUD is on its way
        if pet is None or not pet.isVisible() or self._hud_on_screen():
            self._overlay.restore_hud()
            return
        try:
            pet.away = True                                 # no popping back while the dots fly
            pet.setWindowOpacity(0.0)                       # gone this very instant
            img = pet.grab().toImage()
            dpr = img.devicePixelRatio() or 1.0
            if dpr != 1.0:
                img = img.scaled(int(img.width() / dpr), int(img.height() / dpr))
                img.setDevicePixelRatio(1.0)
            origin = QPointF(pet.pos())
            dots = sample_dots(img, origin)
            pet._hide_now()
            self._overlay.ensure_valid_geometry()
            target, radius = self._blob_target()
        except Exception:  # noqa: BLE001 -- never let the effect block opening Spidy
            pet.away = False
            pet._hide_now()
            self._overlay.restore_hud()
            return
        if not dots:
            pet.away = False
            self._overlay.restore_hud()
            return

        def done() -> None:
            self._running = None
            pet.away = False
            pet._ui_was_visible = True
            pet._sync_with_ui()

        self._running = self._view
        self._view.load(dots, target, radius, None, origin, self._overlay.restore_hud, done)
        self._view.run()
