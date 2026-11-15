"""
Desktop pet: a tiny chibi Spidey who lives on the desktop while the HUD is closed.

* Walks along the taskbar with a real gait (hips, knees, elbows; bob, lean, arms swinging opposite),
  stops, looks around, blinks and waves.
* Shoots a web to almost anywhere -- the top edge, a top corner, the left or right wall -- zips up and
  swings on it like a pendulum, sometimes hanging upside down, then lowers himself back down.
* Click him and the full Spidy UI opens; say "Hey Spidy" and he hops up and turns into the HUD.
  Drag him around; right-click for a small menu.

The window grows to cover the web line while he hangs; its transparent pixels let clicks through.
All art is drawn with QPainter (original drawing, no image files). The timer runs only while visible.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QMenu, QWidget

RED = QColor(214, 35, 43)
RED_SHADE = QColor(168, 22, 32)
BLUE = QColor(38, 86, 196)
BLUE_SHADE = QColor(26, 60, 150)
INK = QColor(18, 12, 14)
WHITE = QColor(250, 250, 255)
WEB = QColor(246, 248, 255)

W, H = 130, 140                 # widget size while on the floor
FOOT = QPointF(W / 2, H - 10)   # where his feet stand inside that widget
GRAVITY = 1400.0                # px/s^2 for the web swing


# ── pose & drawing ───────────────────────────────────────────────────────────

@dataclass
class Pose:
    """Joint angles in degrees. 0 = limb pointing straight down; + = toward the direction he faces."""
    bob: float = 0.0
    lean: float = 0.0
    head_tilt: float = 0.0
    turn: float = 0.0           # 0 = facing you, 1 = three-quarter view toward where he walks
    blink: float = 0.0
    arm_n: tuple[float, float] = (8.0, 10.0)     # near arm: (shoulder, elbow bend)
    arm_f: tuple[float, float] = (-8.0, 10.0)    # far arm
    leg_n: tuple[float, float] = (0.0, 0.0)      # near leg: (hip, knee bend)
    leg_f: tuple[float, float] = (0.0, 0.0)
    reach_n: float = 1.0        # near-arm length multiplier (stretches up past the big head to grip a web)
    reach_f: float = 1.0        # far-arm length multiplier
    phone: bool = False         # holding a tiny phone in the near hand (lounging in the hammock)
    sense: float = 0.0          # 0..1 spidey-sense squiggles around the head
    extras: dict = field(default_factory=dict)


def _pt(o: QPointF, length: float, angle_deg: float) -> QPointF:
    a = math.radians(angle_deg)
    return QPointF(o.x() + math.sin(a) * length, o.y() + math.cos(a) * length)


def _limb(p: QPainter, pts: list[QPointF], color: QColor, width: float) -> None:
    path = QPainterPath(pts[0])
    for q in pts[1:]:                       # straight segments with round joints = a bent elbow/knee
        path.lineTo(q)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(INK, width + 3.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.drawPath(path)
    p.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.drawPath(path)


def _leg(p: QPainter, hip: QPointF, hip_a: float, knee: float, near: bool) -> QPointF:
    k = _pt(hip, 12, hip_a)
    f = _pt(k, 12, hip_a - knee)
    _limb(p, [hip, k, f], BLUE if near else BLUE_SHADE, 8.5)
    boot_tip = _pt(f, 6, hip_a - knee + 90)                       # boot points forward
    _limb(p, [_pt(f, 2.5, hip_a - knee + 180), boot_tip], RED if near else RED_SHADE, 7.5)
    return f


def _arm(p: QPainter, sh: QPointF, sh_a: float, elbow: float, near: bool, reach: float = 1.0) -> QPointF:
    e = _pt(sh, 10.5 * reach, sh_a)
    h = _pt(e, 10 * reach, sh_a + elbow)
    _limb(p, [sh, e, h], RED if near else RED_SHADE, 7.5)
    p.setPen(QPen(INK, 2.2))
    p.setBrush(QBrush(RED if near else RED_SHADE))
    p.drawEllipse(h, 4.6, 4.6)                                      # glove
    return h


def _phone(p: QPainter, hand: QPointF) -> None:
    """A tiny phone held in the hand, screen glowing."""
    p.save()
    p.translate(hand.x() + 2, hand.y() - 9)
    p.rotate(-12)
    p.setPen(QPen(INK, 1.6))
    p.setBrush(QBrush(QColor(30, 32, 40)))
    p.drawRoundedRect(QRectF(-4.5, -8, 9, 15), 2, 2)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(120, 200, 255, 230)))
    p.drawRoundedRect(QRectF(-3.2, -6.6, 6.4, 11.4), 1.2, 1.2)
    p.restore()


def _spidey_sense(p: QPainter, head: QPointF, k: float) -> None:
    """Yellow wavy squiggles around the head: his spidey-sense is tingling."""
    p.save()
    p.setOpacity(min(1.0, k))
    for i in range(7):
        a = math.radians(-160 + i * 47)
        r0 = 38 + 3 * math.sin(i * 1.7)
        base = QPointF(head.x() + math.cos(a) * r0, head.y() + math.sin(a) * r0)
        path = QPainterPath(base)
        for j in range(1, 4):
            along = QPointF(math.cos(a) * 4.5 * j, math.sin(a) * 4.5 * j)
            side = 3.2 * (1 if j % 2 else -1)
            path.lineTo(QPointF(base.x() + along.x() - math.sin(a) * side, base.y() + along.y() + math.cos(a) * side))
        p.setPen(QPen(QColor(255, 222, 40), 3.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
    p.restore()


def draw_spidey_pose(p: QPainter, pose: Pose) -> dict[str, QPointF]:
    """Chibi Spidey, feet near (0, 0), facing +x. Returns key points (hands, feet, hip) in local coords."""
    out: dict[str, QPointF] = {}
    p.save()
    p.translate(0, -pose.bob)
    p.rotate(pose.lean)
    hip = QPointF(0, -24)
    sh_y = -45
    depth = 4 * pose.turn                                           # far limbs shift back in 3/4 view
    # far side first (behind the body)
    out["foot_f"] = _leg(p, QPointF(hip.x() - 5 + depth * 0.4, hip.y()), *pose.leg_f, near=False)
    out["hand_f"] = _arm(p, QPointF(-12 + depth, sh_y), *pose.arm_f, near=False, reach=pose.reach_f)
    # torso: blue sides, red chest, belt
    body = QPainterPath()
    body.moveTo(-14, -48)
    body.cubicTo(-18, -34, -14, -22, -9, -20)
    body.lineTo(9, -20)
    body.cubicTo(14, -22, 18, -34, 14, -48)
    body.cubicTo(8, -52, -8, -52, -14, -48)
    p.setPen(QPen(INK, 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(QBrush(BLUE))
    p.drawPath(body)
    chest = QPainterPath()
    chest.moveTo(-9 + depth * 0.5, -49)
    chest.cubicTo(-11 + depth * 0.5, -36, -7, -27, 0 + depth * 0.4, -24)
    chest.cubicTo(7, -27, 11 + depth * 0.3, -36, 9 + depth * 0.5, -49)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(RED))
    p.drawPath(chest)
    p.setPen(QPen(INK, 1.3))
    ex = depth * 0.45                                               # little spider emblem
    p.drawLine(QPointF(ex, -42), QPointF(ex, -33))
    for dy in (-40, -36):
        p.drawLine(QPointF(ex - 4, dy - 2), QPointF(ex + 4, dy + 2))
        p.drawLine(QPointF(ex + 4, dy - 2), QPointF(ex - 4, dy + 2))
    # near leg + near arm
    out["foot_n"] = _leg(p, QPointF(hip.x() + 5 + depth * 0.2, hip.y()), *pose.leg_n, near=True)
    out["hip"] = hip
    gripping = pose.reach_n > 1.5            # arm stretched up to a web: it goes behind the head, past the ear
    if gripping:
        out["hand_n"] = _arm(p, QPointF(12 + depth * 0.3, sh_y), *pose.arm_n, near=True, reach=pose.reach_n)
    # head
    p.save()
    p.translate(depth * 0.6, -74)
    p.rotate(pose.head_tilt)
    head = QRectF(-31, -29, 62, 58)
    p.setPen(QPen(INK, 2.8))
    p.setBrush(QBrush(RED))
    p.drawEllipse(head)
    clip = QPainterPath()
    clip.addEllipse(head)
    p.save()
    p.setClipPath(clip)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor(95, 8, 14, 190), 1.2))
    c = QPointF(4 * pose.turn, 4)
    for i in range(10):
        a = math.radians(i * 36 + 10)
        p.drawLine(c, QPointF(c.x() + math.cos(a) * 60, c.y() + math.sin(a) * 60))
    for r in (11, 21, 32):
        p.drawEllipse(c, r * 1.15, r)
    p.restore()
    open_ = max(0.08, 1 - pose.blink)
    for side in (-1, 1):
        far = side < 0
        p.save()
        shift = 8 * pose.turn
        p.translate(side * (13 - (3 * pose.turn if far else 0)) + shift, 0)
        sc = (0.78 if far and pose.turn > 0.3 else 1.0)
        p.scale(side * sc, open_)
        eye = QPainterPath()
        eye.moveTo(-8, 5)
        eye.quadTo(-6, -9, 6, -13)
        eye.quadTo(17, -14, 15, -2)
        eye.quadTo(12, 10, -8, 5)
        p.setPen(QPen(INK, 3.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(QBrush(WHITE))
        p.drawPath(eye)
        p.restore()
    p.restore()
    if not gripping:
        out["hand_n"] = _arm(p, QPointF(12 + depth * 0.3, sh_y), *pose.arm_n, near=True, reach=pose.reach_n)
    if pose.phone:
        _phone(p, out["hand_n"])
    if pose.sense > 0:
        _spidey_sense(p, QPointF(depth * 0.6, -74), pose.sense)
    p.restore()
    # map points through bob/lean so callers can attach the web exactly to a hand
    lean = math.radians(pose.lean)
    for k, v in list(out.items()):
        x, y = v.x(), v.y()
        out[k] = QPointF(x * math.cos(lean) - y * math.sin(lean), x * math.sin(lean) + y * math.cos(lean) - pose.bob)
    return out


def walk_pose(phase: float, blink: float = 0.0) -> Pose:
    s = math.sin(phase)
    return Pose(
        bob=abs(math.cos(phase)) * 2.6, lean=4.0, turn=1.0, blink=blink,
        head_tilt=math.sin(phase * 2) * 1.5,
        leg_n=(26 * s, 8 + max(0.0, 34 * math.sin(phase + math.pi / 2))),
        leg_f=(-26 * s, 8 + max(0.0, 34 * math.sin(phase - math.pi / 2))),
        arm_n=(-24 * s, 28), arm_f=(24 * s, 28),
    )


def idle_pose(t: float, blink: float, look: float = 0.0) -> Pose:
    br = math.sin(t * 2.0)
    return Pose(bob=br * 0.8, head_tilt=look * 8, turn=max(0.0, look) * 0.6, blink=blink,
                arm_n=(10 + br * 2, 14), arm_f=(-10 - br * 2, 14), leg_n=(4, 2), leg_f=(-4, 2))


def wave_pose(t: float, blink: float) -> Pose:
    w = math.sin(t * 11)
    return Pose(bob=math.sin(t * 2) * 0.6, head_tilt=-6, blink=blink,
                arm_n=(150 + w * 18, -40 + w * 20), arm_f=(-12, 16), leg_n=(5, 2), leg_f=(-5, 2))


def hang_pose(t: float, blink: float, swing_v: float) -> Pose:
    """Holding the web overhead with the near hand; legs dangle and trail the swing."""
    trail = max(-30.0, min(30.0, -swing_v * 12))
    return Pose(turn=0.4, blink=blink, head_tilt=-trail * 0.15,
                arm_n=(146, 24), reach_n=2.8,                 # up along the side of the head, hand above it
                arm_f=(-30 + math.sin(t * 3) * 10, 30),
                leg_n=(trail + 6 + math.sin(t * 2.6) * 6, 18), leg_f=(trail - 4 + math.sin(t * 2.6 + 1) * 6, 26))


def upside_pose(t: float, blink: float, wave: bool = False) -> Pose:
    """Classic upside-down hang (drawn rotated 180 degrees by the caller): the web is stuck between his
    feet, knees splayed out wide in a diamond, hands gripping the web up at his feet -- or the front
    hand waving down past his head while the other one holds on."""
    sway = math.sin(t * 2.2)
    grip_n = (-9, 4)
    near = (180 + math.sin(t * 10) * 18, -35 + math.sin(t * 10) * 20) if wave else grip_n
    return Pose(turn=0.1, blink=blink, head_tilt=sway * 3,
                leg_n=(70, 120), leg_f=(-70, -120),                     # knees out, feet together
                arm_n=near, reach_n=1.0 if wave else 1.6,
                arm_f=(9, -4), reach_f=1.6)

def lounge_pose(t: float, blink: float) -> Pose:
    """Lying back in a web hammock (drawn rotated by the caller): one knee up, the other leg crossed,
    near hand holding a phone in front of his face, far hand behind his head."""
    scroll = math.sin(t * 1.3)
    return Pose(turn=0.3, blink=blink, head_tilt=-6,
                leg_n=(-40, -75), leg_f=(-8, -6),                         # knee up, other leg stretched out
                arm_n=(-120 + scroll * 4, 70), arm_f=(-178, -40), reach_f=1.25, phone=True)


def shoot_pose(blink: float, aim_deg: float) -> Pose:
    return Pose(turn=0.6, blink=blink, lean=-4, arm_n=(aim_deg, 0), arm_f=(-14, 20), leg_n=(14, 6), leg_f=(-12, 10))


def draw_spidey(p: QPainter, t: float, state: str, blink: float) -> None:
    """Compatibility helper: draw one representative pose for ``state``."""
    if state == "walk":
        pose = walk_pose(t * 9, blink)
    elif state == "wave":
        pose = wave_pose(t, blink)
    elif state in ("swing", "hang"):
        pose = hang_pose(t, blink, math.sin(t * 2))
    elif state == "look":
        pose = idle_pose(t, blink, look=math.sin(t * 2))
    else:
        pose = idle_pose(t, blink)
    draw_spidey_pose(p, pose)


def draw_web(p: QPainter, a: QPointF, b: QPointF, alpha: float = 1.0) -> None:
    """A clearly visible web line from a to b, with a little splat where it sticks at a."""
    p.save()
    p.setOpacity(alpha)
    p.setPen(QPen(QColor(20, 20, 30, 160), 4.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawLine(a, b)
    p.setPen(QPen(WEB, 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawLine(a, b)
    # twisty strand texture: small alternating wraps along the line
    length = math.hypot(b.x() - a.x(), b.y() - a.y())
    if length > 20:
        ux, uy = (b.x() - a.x()) / length, (b.y() - a.y()) / length
        p.setPen(QPen(WEB, 1.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        n = int(length // 11)
        for i in range(1, n):
            cx, cy = a.x() + ux * i * 11, a.y() + uy * i * 11
            s = 2.6 if i % 2 else -2.6
            p.drawLine(QPointF(cx - uy * s - ux * 2, cy + ux * s - uy * 2), QPointF(cx + uy * s + ux * 2, cy - ux * s + uy * 2))
    p.setPen(QPen(WEB, 1.3))
    for i in range(6):                                            # splat
        ang = math.radians(i * 60 + 15)
        p.drawLine(a, QPointF(a.x() + math.cos(ang) * 7, a.y() + math.sin(ang) * 7))
    p.restore()


# ── the pet window ───────────────────────────────────────────────────────────

class DesktopPet(QWidget):
    clicked = Signal()                     # open the full Spidy UI

    def __init__(self, ui_visible=None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Spidy — click to open")
        self._ui_visible = ui_visible or (lambda: False)
        self._rng = random.Random()
        self.enabled = True
        self.t = 0.0
        self.state = "idle"        # idle | walk | look | wave | shoot | swing | lower | hammock | hop
        self.state_left = 2.0
        self.facing = 1
        self.target_x = 0.0
        self.blink = 0.0
        self.hop = 0.0
        self.walk_phase = 0.0
        # world (screen) positions: feet point while on the floor; web anchor + pendulum while hanging
        self.fx, self.fy = 0.0, 0.0
        self.anchor = QPointF()
        self.rope = 0.0
        self.rope_target = 0.0
        self.theta = 0.0
        self.omega = 0.0
        self.upside = False
        self.hang_wave = False
        self.shoot_k = 0.0
        self.sense = 0.0                   # spidey-sense squiggles (hover, surprise)
        self.after_shoot = ""               # "hammock": the web shot is for stringing a hammock
        self.hammock: tuple[QPointF, QPointF] | None = None
        self.hammock_k = 0.0
        self._press: QPointF | None = None
        self._dragged = False
        self._anim = QTimer(self)
        self._anim.setInterval(33)
        self._anim.timeout.connect(self._tick)
        self._watch = QTimer(self)
        self._watch.setInterval(400)
        self._watch.timeout.connect(self._sync_with_ui)
        self._ui_was_visible = True
        self.resize(W, H)
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
            self._set_state("hop", 0.45)
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

    def _floor(self) -> float:
        return float(self._area().bottom() - 4)

    def _place_on_floor(self) -> None:
        a = self._area()
        if self.fx <= 0:
            self.fx = a.right() - 220
        self.fx = max(a.left() + 40, min(self.fx, a.right() - 40))
        self.fy = self._floor()
        self._fit_floor()

    def _fit_floor(self) -> None:
        self.setGeometry(QRect(int(self.fx - FOOT.x()), int(self.fy - FOOT.y()), W, H))

    def _body_world(self) -> QPointF:
        """Hand (or feet when upside down) position on the rope, in screen coordinates."""
        return QPointF(self.anchor.x() + math.sin(self.theta) * self.rope,
                       self.anchor.y() + math.cos(self.theta) * self.rope)

    def _fit_hanging(self) -> None:
        b = self._body_world()
        pad = 95
        left = min(self.anchor.x(), b.x()) - pad
        top = min(self.anchor.y(), b.y()) - 20
        right = max(self.anchor.x(), b.x()) + pad
        bottom = max(self.anchor.y(), b.y()) + 115
        self.setGeometry(QRect(int(left), int(top), int(right - left), int(bottom - top)))

    # ── behaviour ──────────────────────────────────────────────────────────
    def _set_state(self, state: str, seconds: float) -> None:
        self.state, self.state_left = state, seconds
        if state == "walk":
            a = self._area()
            self.target_x = self._rng.uniform(a.left() + 60, a.right() - 60)
            self.facing = 1 if self.target_x > self.fx else -1
        elif state == "shoot":
            if self.after_shoot == "hammock" and self.hammock is not None:
                self.anchor = QPointF(self.hammock[0])            # first strand goes to the hammock's top end
            else:
                self._pick_anchor()
            self.shoot_k = 0.0
            self.facing = 1 if self.anchor.x() >= self.fx else -1
        elif state == "hop":
            self.hop = 0.0
        elif state == "hammock":
            self.hammock_k = 0.0

    def _pick_anchor(self) -> None:
        """Somewhere to stick the web: the top edge, a top corner, or a side wall."""
        a = self._area()
        where = self._rng.choice(["top", "top", "top", "corner_l", "corner_r", "wall_l", "wall_r"])
        if where == "top":
            x = max(a.left() + 40, min(a.right() - 40, self.fx + self._rng.uniform(-260, 260)))
            self.anchor = QPointF(x, a.top() + 2)
        elif where == "corner_l":
            self.anchor = QPointF(a.left() + 3, a.top() + 3)
        elif where == "corner_r":
            self.anchor = QPointF(a.right() - 3, a.top() + 3)
        elif where == "wall_l":
            self.anchor = QPointF(a.left() + 2, a.top() + a.height() * self._rng.uniform(0.25, 0.55))
        else:
            self.anchor = QPointF(a.right() - 2, a.top() + a.height() * self._rng.uniform(0.25, 0.55))
        self.upside = where == "top" and self._rng.random() < 0.45
        self.hang_wave = self._rng.random() < 0.5

    def start_hammock(self) -> None:
        """String a web hammock diagonally across a top corner and lounge in it for a while."""
        a = self._area()
        right = self.fx > (a.left() + a.right()) / 2 if self._rng.random() < 0.8 else self._rng.random() < 0.5
        span = min(330.0, a.width() * 0.3)
        if right:
            top_end = QPointF(a.right() - span, a.top() + 2)
            wall_end = QPointF(a.right() - 2, a.top() + span * 0.62)
        else:
            top_end = QPointF(a.left() + span, a.top() + 2)
            wall_end = QPointF(a.left() + 2, a.top() + span * 0.62)
        self.hammock = (top_end, wall_end)
        self.after_shoot = "hammock"
        self._set_state("shoot", 0.45)

    def _choose_next(self) -> None:
        r = self._rng.random()
        if r < 0.40:
            self._set_state("walk", 14.0)
        elif r < 0.52:
            self._set_state("idle", self._rng.uniform(2.0, 4.0))
        elif r < 0.62:
            self._set_state("look", 2.6)
        elif r < 0.71:
            self._set_state("wave", 2.4)
        elif r < 0.85:
            self._set_state("shoot", 0.45)
        else:
            self.start_hammock()

    # hammock geometry: a sagging band between the two ends
    def _hammock_point(self, s: float, offset: float = 0.0) -> QPointF:
        a, b = self.hammock
        sag = 58 + 4 * math.sin(self.t * 1.2)                          # gentle sway
        mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2 + sag * 2)
        x = (1 - s) ** 2 * a.x() + 2 * (1 - s) * s * mid.x() + s ** 2 * b.x()
        y = (1 - s) ** 2 * a.y() + 2 * (1 - s) * s * mid.y() + s ** 2 * b.y() + offset
        return QPointF(x, y)

    def _fit_hammock(self) -> None:
        a, b = self.hammock
        low = self._hammock_point(0.5)
        left = min(a.x(), b.x()) - 70
        right = max(a.x(), b.x()) + 70
        top = min(a.y(), b.y()) - 10
        bottom = max(low.y(), b.y()) + 90
        self.setGeometry(QRect(int(left), int(top), int(right - left), int(bottom - top)))

    def _start_swing(self) -> None:
        """Web stuck: zip up -- the rope starts at the current distance and reels in."""
        dx, dy = self.fx - self.anchor.x(), (self.fy - 70) - self.anchor.y()
        self.rope = max(40.0, math.hypot(dx, dy))
        self.theta = math.atan2(dx, dy)
        self.omega = 0.0
        a = self._area()
        self.rope_target = self._rng.uniform(110, min(320, a.height() * 0.45))
        self._set_state("swing", self._rng.uniform(5.5, 8.5))

    def _tick(self) -> None:
        dt = 0.033
        self.t += dt
        self.state_left -= dt
        self.blink = max(0.0, self.blink - dt * 6)
        if self._rng.random() < dt / 3.5:
            self.blink = 1.0
        self.sense = max(0.0, self.sense - dt * 0.7)
        if self._press is not None and self._dragged:
            self.update()
            return
        st = self.state
        if st == "walk":
            speed = 62.0
            self.walk_phase += dt * speed / 9.0            # stride matched to speed: no foot sliding
            nx = self.fx + speed * dt * self.facing
            if (self.facing > 0 and nx >= self.target_x) or (self.facing < 0 and nx <= self.target_x):
                nx, self.state_left = self.target_x, 0
            self.fx, self.fy = nx, self._floor()
            self._fit_floor()
        elif st == "shoot":
            self.shoot_k = min(1.0, self.shoot_k + dt / 0.35)
            self._fit_shoot()
            if self.shoot_k >= 1.0:
                if self.after_shoot == "hammock":
                    self.after_shoot = ""
                    self._set_state("hammock", self._rng.uniform(10.0, 16.0))
                    self._fit_hammock()
                else:
                    self._start_swing()
        elif st == "hammock":
            self.hammock_k = min(1.0, self.hammock_k + dt / 0.6)       # net weaves in, he settles
            self._fit_hammock()
            if self.state_left <= 0:                                    # climb out: drop down on a web
                spot = self._hammock_point(0.5)
                self.anchor = QPointF(spot.x(), spot.y() - 4)
                self.rope, self.theta, self.omega, self.upside = 20.0, 0.0, 0.0, False
                self.hammock = None
                self._set_state("lower", 99.0)
        elif st in ("swing", "lower"):
            if st == "swing":
                self.rope += (self.rope_target - self.rope) * min(1.0, dt * 2.2)        # reel in
                if self._rng.random() < dt * 0.25:
                    self.omega += self._rng.choice([-1, 1]) * 0.9                       # a little kick
            else:                                                                      # lower to the floor
                self.rope += 160 * dt
            alpha = -(GRAVITY / max(self.rope, 30)) * math.sin(self.theta) - 0.55 * self.omega
            self.omega += alpha * dt
            self.theta += self.omega * dt
            b = self._body_world()
            a = self._area()
            if b.x() < a.left() + 30 or b.x() > a.right() - 30:                         # don't swing through walls
                self.theta -= self.omega * dt
                self.omega *= -0.4
                b = self._body_world()
            if st == "lower" and b.y() + 70 >= self._floor():
                self.fx, self.fy = b.x(), self._floor()
                self.upside = False
                self._fit_floor()
                self._set_state("idle", 1.2)
            else:
                self._fit_hanging()
                if st == "swing" and self.state_left <= 0:
                    self._set_state("lower", 99.0)
        elif st == "hop":
            self.hop = min(1.0, self.hop + dt / 0.45)
            if self.hop >= 1.0:
                self._hide_now()
                return
        if self.state_left <= 0 and self.state in ("idle", "walk", "look", "wave"):
            self._choose_next()
        self.update()

    def _fit_shoot(self) -> None:
        a = self.anchor
        left = min(a.x(), self.fx) - 90
        right = max(a.x(), self.fx) + 90
        top = min(a.y(), self.fy - H) - 10
        self.setGeometry(QRect(int(left), int(top), int(right - left), int(self.fy + 12 - top)))

    # ── input ──────────────────────────────────────────────────────────────
    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._press = e.globalPosition()
            self._dragged = False

    def mouseMoveEvent(self, e) -> None:
        if self._press is not None and e.buttons() & Qt.MouseButton.LeftButton:
            g = e.globalPosition()
            if (g - self._press).manhattanLength() > 5:
                self._dragged = True
            if self._dragged:
                if self.state in ("swing", "lower", "shoot", "hammock"):
                    self.state, self.hammock, self.after_shoot = "idle", None, ""
                self.fx, self.fy = g.x(), g.y() + 40
                self.setGeometry(QRect(int(self.fx - FOOT.x()), int(self.fy - FOOT.y()), W, H))

    def mouseReleaseEvent(self, e) -> None:
        if e.button() != Qt.MouseButton.LeftButton:
            return
        dragged, self._press = self._dragged, None
        if dragged:
            self.fy = self._floor()
            self._fit_floor()
            self._set_state("idle", 1.5)
            return
        self._set_state("hop", 0.45)
        self.clicked.emit()

    def contextMenuEvent(self, e) -> None:
        m = QMenu(self)
        m.addAction("Open Spidy", self.clicked.emit)
        m.addAction("Wave!", lambda: self._set_state("wave", 2.4))
        m.addAction("Swing on a web", lambda: self._set_state("shoot", 0.45))
        m.addAction("Relax in a web hammock", self.start_hammock)
        m.addSeparator()
        m.addAction("Hide the pet", lambda: self.set_enabled(False))
        m.exec(e.globalPos())

    # ── drawing ────────────────────────────────────────────────────────────
    def _local(self, world: QPointF) -> QPointF:
        return QPointF(world.x() - self.x(), world.y() - self.y())

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        st = self.state
        if st in ("swing", "lower"):
            self._paint_hanging(p)
        elif st == "hammock":
            self._paint_hammock(p)
        elif st == "shoot":
            self._paint_shoot(p)
        else:
            self._paint_floor(p)
        p.end()

    def _paint_floor(self, p: QPainter) -> None:
        foot = self._local(QPointF(self.fx, self.fy))
        # soft shadow
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 55)))
        p.drawEllipse(QPointF(foot.x(), foot.y() + 2), 20, 4.5)
        p.translate(foot)
        if self.state == "hop":
            k = self.hop
            p.translate(0, -60 * math.sin(k * math.pi / 2))
            s = 1 + 0.6 * k
            p.scale(s, s)
            p.setOpacity(max(0.0, 1 - k ** 2))
        p.scale(self.facing, 1)
        if self.state == "walk":
            pose = walk_pose(self.walk_phase, self.blink)
        elif self.state == "wave":
            pose = wave_pose(self.t, self.blink)
        elif self.state == "look":
            pose = idle_pose(self.t, self.blink, look=math.sin(self.t * 2.2))
        else:
            pose = idle_pose(self.t, self.blink)
        pose.sense = self.sense
        draw_spidey_pose(p, pose)

    def _paint_hammock(self, p: QPainter) -> None:
        """A woven web hammock across the corner, Spidey lying back in it scrolling his phone."""
        if self.hammock is None:
            return
        k = self.hammock_k
        a, b = self.hammock
        n = 22
        top = [self._local(self._hammock_point(i / n)) for i in range(n + 1)]
        bot = [self._local(self._hammock_point(i / n, offset=26 * math.sin(math.pi * i / n))) for i in range(n + 1)]
        shown = max(2, int((n + 1) * min(1.0, k * 1.4)))
        p.save()
        for pts, w in ((top, 2.2), (bot, 1.8)):                          # two rims
            path = QPainterPath(pts[0])
            for q in pts[1:shown]:
                path.lineTo(q)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(20, 20, 30, 140), w + 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPath(path)
            p.setPen(QPen(WEB, w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPath(path)
        p.setPen(QPen(QColor(246, 248, 255, 210), 1.1))                  # diamond mesh
        for i in range(min(shown, n)):
            p.drawLine(top[i], bot[i + 1])
            p.drawLine(bot[i], top[i + 1])
        draw_web(p, self._local(a), top[0])
        draw_web(p, self._local(b), top[-1])
        p.restore()
        if k < 0.35:
            return
        # lie back in the sag of the hammock: hips in the dip, head toward the higher end
        s = 0.5
        here = self._hammock_point(s, offset=13)
        ahead, behind = self._hammock_point(s - 0.06), self._hammock_point(s + 0.06)
        dx, dy = ahead.x() - behind.x(), ahead.y() - behind.y()       # along the net, toward the top end
        L = math.hypot(dx, dy) or 1.0
        tx, ty = dx / L, dy / L
        nx, ny = (-ty, tx) if tx > 0 else (ty, -tx)                    # normal pointing up, out of the net
        if ny > 0:
            nx, ny = -nx, -ny
        hips = self._local(QPointF(here.x() + nx * 10, here.y() + ny * 10))
        ang = math.degrees(math.atan2(tx, -ty))                         # body's "up" (head) along the net
        p.save()
        p.setOpacity(min(1.0, (k - 0.35) / 0.4))
        p.translate(hips)
        p.rotate(ang)
        side = 1 if (nx * math.cos(math.radians(ang)) + ny * math.sin(math.radians(ang))) < 0 else -1
        p.scale(side, 1)                                                # chest faces up, away from the net
        p.translate(0, 24)                                              # put his hips at the origin
        pose = lounge_pose(self.t, self.blink)
        pose.sense = self.sense
        draw_spidey_pose(p, pose)
        p.restore()
    def enterEvent(self, _e) -> None:
        self.sense = 1.0                                                # spidey-sense tingles when you come close

    def _paint_shoot(self, p: QPainter) -> None:
        foot = self._local(QPointF(self.fx, self.fy))
        anchor = self._local(self.anchor)
        dx, dy = (anchor.x() - foot.x()) * self.facing, anchor.y() - (foot.y() - 60)
        aim = math.degrees(math.atan2(dx, dy))                        # 180 = straight up
        p.save()
        p.translate(foot)
        p.scale(self.facing, 1)
        pts = draw_spidey_pose(p, shoot_pose(self.blink, aim))
        p.restore()
        hand = QPointF(foot.x() + pts["hand_n"].x() * self.facing, foot.y() + pts["hand_n"].y())
        tip = QPointF(hand.x() + (anchor.x() - hand.x()) * self.shoot_k, hand.y() + (anchor.y() - hand.y()) * self.shoot_k)
        draw_web(p, tip, hand)

    def _paint_hanging(self, p: QPainter) -> None:
        anchor = self._local(self.anchor)
        grip = self._local(self._body_world())
        draw_web(p, anchor, grip)
        p.save()
        p.translate(grip)
        p.rotate(-math.degrees(self.theta))                           # hang along the rope
        if self.upside:
            # classic upside-down hang: web on his feet, knees out, hands gripping the web (or one waving)
            p.rotate(180)
            p.scale(self.facing, 1)
            pose = upside_pose(self.t, self.blink, wave=self.hang_wave)
            pose.sense = self.sense
        else:
            # one hand holds the web overhead, the other relaxes, legs trail the swing
            p.scale(self.facing, 1)
            pose = hang_pose(self.t, self.blink, self.omega)
            pose.sense = self.sense
        probe = draw_spidey_pose(_NullPainter(), pose)                 # where does the gripping part end up?
        if self.upside:
            fn, ff = probe["foot_n"], probe["foot_f"]
            attach = QPointF((fn.x() + ff.x()) / 2, (fn.y() + ff.y()) / 2)   # between his feet
        else:
            attach = probe["hand_n"]
        p.translate(-attach.x(), -attach.y())                          # ...put exactly that on the rope end
        draw_spidey_pose(p, pose)
        p.restore()


class _NullPainter:
    """Swallows drawing calls so draw_spidey_pose can be used to measure joint positions."""

    def __getattr__(self, _name):
        return lambda *a, **k: None
