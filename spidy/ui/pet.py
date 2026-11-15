"""
Desktop pet: a tiny chibi Spidey who lives on the desktop while the HUD is closed.

He walks along the bottom of the screen, stops to look around, blinks, waves and now and then drops
down on a web line and swings. Click him and the full Spidy UI opens; say "Hey Spidy" and he hops up
and "turns into" the HUD (he hides as it fades in) and comes back when the HUD is closed or minimised.
Drag him to move him; right-click for a small menu.

Everything is drawn with QPainter (original art, no image files). The animation timer only runs while
he is visible, so a hidden pet costs nothing.
"""

from __future__ import annotations

import math
import random

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QMenu, QWidget

RED = QColor(214, 35, 43)
RED_DARK = QColor(140, 16, 24)
BLUE = QColor(36, 82, 190)
BLUE_DARK = QColor(20, 46, 120)
INK = QColor(18, 12, 14)
WHITE = QColor(250, 250, 255)

W, H = 120, 132            # widget size in pixels (the drawing is designed for a 100 x 110 box)


class DesktopPet(QWidget):
    clicked = Signal()                     # open the full Spidy UI

    def __init__(self, ui_visible=None, parent=None) -> None:
        """``ui_visible``: callable returning True while the HUD is on screen (the pet hides then)."""
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(W, H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Spidy — click to open")
        self._ui_visible = ui_visible or (lambda: False)
        self.enabled = True
        self.t = 0.0
        self.state = "idle"                 # idle | walk | wave | swing | hop | look
        self.state_left = 2.0
        self.facing = 1                     # 1 = right, -1 = left
        self.target_x = 0.0
        self.blink = 0.0
        self.swing_len = 0.0
        self.web_top = 0.0
        self.hop = 0.0                      # 0..1 transform-into-HUD animation
        self._press: QPointF | None = None
        self._dragged = False
        self._rng = random.Random()
        self._anim = QTimer(self)
        self._anim.setInterval(33)          # ~30 fps while visible
        self._anim.timeout.connect(self._tick)
        self._watch = QTimer(self)          # follows the HUD: show when it is gone, hide when it is up
        self._watch.setInterval(400)
        self._watch.timeout.connect(self._sync_with_ui)
        self._ui_was_visible = True
        self._place_on_floor()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        self._watch.start()
        self._sync_with_ui()

    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        if not on:
            self._hide_now()
        self._sync_with_ui()

    def _sync_with_ui(self) -> None:
        ui = bool(self._ui_visible())
        if ui and not self._ui_was_visible and self.isVisible() and self.state != "hop":
            self._set_state("hop", 0.45)        # "Hey Spidy": hop up and turn into the HUD
        elif ui and self.isVisible() and self.state != "hop":
            self._hide_now()
        elif not ui and self.enabled and not self.isVisible():
            self._show_now()
        self._ui_was_visible = ui

    def _show_now(self) -> None:
        self.hop = 0.0
        self._place_on_floor()
        self._set_state("idle", 1.5)
        self.setWindowOpacity(1.0)
        self.show()
        self.raise_()
        self._anim.start()

    def _hide_now(self) -> None:
        self._anim.stop()
        self.hide()

    # ── geometry ───────────────────────────────────────────────────────────
    def _area(self):
        screen = self.screen() or QGuiApplication.primaryScreen()
        return screen.availableGeometry()

    def _floor_y(self) -> int:
        return self._area().bottom() - H + 6

    def _place_on_floor(self) -> None:
        a = self._area()
        x = a.right() - W - 160 if self.x() <= 0 else max(a.left(), min(self.x(), a.right() - W))
        self.move(int(x), self._floor_y())

    # ── behaviour ──────────────────────────────────────────────────────────
    def _set_state(self, state: str, seconds: float) -> None:
        self.state, self.state_left = state, seconds
        if state == "walk":
            a = self._area()
            self.target_x = self._rng.uniform(a.left() + 10, a.right() - W - 10)
            self.facing = 1 if self.target_x > self.x() else -1
        elif state == "swing":
            self.web_top = self._area().top()
            self.swing_len = 0.0

    def _choose_next(self) -> None:
        r = self._rng.random()
        if r < 0.45:
            self._set_state("walk", 12.0)
        elif r < 0.62:
            self._set_state("idle", self._rng.uniform(2.0, 4.5))
        elif r < 0.76:
            self._set_state("look", 2.4)
        elif r < 0.88:
            self._set_state("wave", 2.2)
        else:
            self._set_state("swing", 6.0)

    def _tick(self) -> None:
        dt = 0.033
        self.t += dt
        self.state_left -= dt
        # blink every few seconds
        self.blink = max(0.0, self.blink - dt * 6)
        if self._rng.random() < dt / 3.5:
            self.blink = 1.0
        if self._press is not None and self._dragged:
            self.update()
            return
        if self.state == "walk":
            step = 70 * dt * self.facing
            nx = self.x() + step
            if (self.facing > 0 and nx >= self.target_x) or (self.facing < 0 and nx <= self.target_x):
                nx = self.target_x
                self.state_left = 0
            self.move(int(nx), self._floor_y())
        elif self.state == "swing":
            # drop on a web line from the top of the screen, swing, climb back down to the floor
            phase = 6.0 - self.state_left
            floor = self._floor_y()
            span = floor - self.web_top
            depth = min(1.0, phase / 1.2) if phase < 4.6 else max(0.0, 1 - (phase - 4.6) / 1.4)
            hang_y = self.web_top + 40 + depth * span * 0.55
            sway = math.sin(phase * 2.4) * 46 * min(1.0, phase / 1.5)
            self.swing_len = hang_y - self.web_top
            self._swing_anchor_x = getattr(self, "_swing_anchor_x", self.x() + W / 2)
            if phase < dt * 1.5:
                self._swing_anchor_x = self.x() + W / 2
            self.move(int(self._swing_anchor_x - W / 2 + sway), int(hang_y if phase < 4.6 else
                                                                      hang_y + (floor - hang_y) * (1 - depth)))
            if self.state_left <= 0:
                self.move(self.x(), floor)
        elif self.state == "hop":
            self.hop = min(1.0, self.hop + dt / 0.45)
            if self.hop >= 1.0:
                self._hide_now()
                return
        if self.state_left <= 0 and self.state != "hop":
            self._choose_next()
        self.update()

    # ── input ──────────────────────────────────────────────────────────────
    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._press = e.globalPosition() - QPointF(self.pos())
            self._dragged = False

    def mouseMoveEvent(self, e) -> None:
        if self._press is not None and e.buttons() & Qt.MouseButton.LeftButton:
            p = e.globalPosition() - self._press
            if (p - QPointF(self.pos())).manhattanLength() > 4:
                self._dragged = True
            self.move(p.toPoint())

    def mouseReleaseEvent(self, e) -> None:
        if e.button() != Qt.MouseButton.LeftButton:
            return
        dragged, self._press = self._dragged, None
        if dragged:                         # dropped: fall back onto the floor and carry on
            self.move(self.x(), self._floor_y())
            self._set_state("idle", 1.5)
            return
        self._set_state("hop", 0.45)
        self.clicked.emit()

    def contextMenuEvent(self, e) -> None:
        m = QMenu(self)
        m.addAction("Open Spidy", self.clicked.emit)
        m.addAction("Wave!", lambda: self._set_state("wave", 2.2))
        m.addAction("Swing on a web", lambda: self._set_state("swing", 6.0))
        m.addSeparator()
        m.addAction("Hide the pet", lambda: self.set_enabled(False))
        m.exec(e.globalPos())

    # ── drawing ────────────────────────────────────────────────────────────
    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.state == "swing" and self.swing_len > 0:      # the web line up to the top of the screen
            p.setPen(QPen(QColor(235, 235, 245, 220), 1.6))
            p.drawLine(QPointF(W / 2, 0), QPointF(W / 2, 30))
        p.translate(W / 2, H - 8)
        # hop: jump up, grow and fade into the HUD
        if self.state == "hop":
            k = self.hop
            p.translate(0, -60 * math.sin(k * math.pi / 2))
            s = 1 + 0.6 * k
            p.scale(s, s)
            p.setOpacity(max(0.0, 1 - k ** 2))
        p.scale(self.facing * 1.0, 1.0)
        draw_spidey(p, self.t, self.state, self.blink)
        p.end()


def draw_spidey(p: QPainter, t: float, state: str, blink: float) -> None:
    """Chibi Spidey with his feet at (0, 0), about 100 x 110 px, facing right."""
    walk = state == "walk"
    bob = abs(math.sin(t * 9)) * 3 if walk else math.sin(t * 2) * 1.2
    leg = math.sin(t * 9) * 16 if walk else 0.0
    swing = state == "swing"
    p.save()
    # shadow
    if not swing:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 60)))
        p.drawEllipse(QPointF(0, 2), 22, 5)
    p.translate(0, -bob)
    outline = QPen(INK, 2.4)
    outline.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

    def limb(x0, y0, x1, y1, col, width):
        p.setPen(QPen(INK, width + 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(x0, y0), QPointF(x1, y1))
        p.setPen(QPen(col, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

    # legs (blue with red boots)
    for side, swing_a in ((-1, leg), (1, -leg)):
        hx, hy = side * 7, -22
        fx = hx + math.sin(math.radians(swing_a)) * 14 + (side * 3 if swing else 0)
        fy = -3 - (6 if swing else 0)
        limb(hx, hy, fx, fy, BLUE, 8)
        limb(fx - 1, fy - 1, fx + 4, fy, RED, 7)
    # body
    body = QPainterPath()
    body.addRoundedRect(QRectF(-15, -46, 30, 28), 11, 11)
    p.setPen(outline)
    p.setBrush(QBrush(BLUE))
    p.drawPath(body)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(RED))
    p.drawRoundedRect(QRectF(-9, -45, 18, 25), 8, 8)            # red chest panel
    p.setPen(QPen(INK, 1.4))                                     # tiny spider emblem
    p.drawLine(QPointF(0, -38), QPointF(0, -30))
    for dy in (-36, -33):
        p.drawLine(QPointF(-4, dy - 2), QPointF(4, dy + 2))
        p.drawLine(QPointF(4, dy - 2), QPointF(-4, dy + 2))
    # arms
    if state == "wave":
        wave = math.sin(t * 12) * 18
        limb(12, -42, 26, -60 + wave * 0.3, RED, 7)
        limb(-12, -40, -18, -26, RED, 7)
    elif swing:
        pass                                                     # drawn after the head (hands on the web)
    else:
        arm = -leg * 0.8
        limb(12, -42, 16 + math.sin(math.radians(arm)) * 8, -26, RED, 7)
        limb(-12, -42, -16 - math.sin(math.radians(arm)) * 8, -26, RED, 7)
    # head
    look = math.sin(t * 2.2) * 5 if state == "look" else 0.0
    p.translate(look, 0)
    head = QRectF(-33, -104, 66, 60)
    p.setPen(QPen(INK, 2.8))
    p.setBrush(QBrush(RED))
    p.drawEllipse(head)
    # web pattern on the mask
    p.save()
    clip = QPainterPath()
    clip.addEllipse(head)
    p.setClipPath(clip)
    p.setPen(QPen(QColor(95, 8, 14, 200), 1.3))
    p.setBrush(Qt.BrushStyle.NoBrush)
    c = QPointF(0, -70)
    for i in range(10):
        a = math.radians(i * 36 + 10)
        p.drawLine(c, QPointF(c.x() + math.cos(a) * 60, c.y() + math.sin(a) * 60))
    for r in (12, 22, 33):
        p.drawEllipse(c, r * 1.15, r)
    p.restore()
    # eyes: big white lenses with thick black frames
    open_ = max(0.08, 1 - blink)
    for side in (-1, 1):
        # classic mask lens: a rounded teardrop, wide at the outer top, narrowing to the nose
        p.save()
        p.translate(side * 13, -74)
        p.scale(side, open_)
        eye = QPainterPath()
        eye.moveTo(-8, 5)                                   # inner corner, near the nose
        eye.quadTo(-6, -9, 6, -13)                          # up and out
        eye.quadTo(17, -14, 15, -2)                         # rounded outer top
        eye.quadTo(12, 10, -8, 5)                           # back along the bottom
        p.setPen(QPen(INK, 3.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(QBrush(WHITE))
        p.drawPath(eye)
        p.restore()
    if swing:                                               # hands up on the web, in front of the head
        p.translate(-look, 0)
        limb(10, -46, 4, -112, RED, 7)
        limb(-10, -46, -4, -112, RED, 7)
    p.restore()
