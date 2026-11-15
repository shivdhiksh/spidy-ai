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
from PySide6.QtGui import (QBrush, QColor, QGuiApplication, QLinearGradient, QPainter, QPainterPath, QPen,
                           QRadialGradient)
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QMenu, QWidget

from spidy.ui.pet_sprites import SpriteBank

RED = QColor(214, 35, 43)
RED_SHADE = QColor(168, 22, 32)
BLUE = QColor(38, 86, 196)
BLUE_SHADE = QColor(26, 60, 150)
INK = QColor(18, 12, 14)
WHITE = QColor(250, 250, 255)
WEB = QColor(246, 248, 255)
# Miles-style street outfit (an original chibi in that style)
SUIT = QColor(28, 28, 38)          # black suit (mask, legs)
SUIT_RED = QColor(214, 34, 44)     # red web lines / accents
HOOD = QColor(200, 30, 38)         # red hood
JACKET = QColor(62, 74, 78)        # dark grey-teal hoodie jacket
SHORTS = QColor(66, 56, 52)        # dark shorts
SNEAKER = QColor(214, 34, 44)      # red sneakers with white soles

W, H = 200, 225                 # widget size while on the floor (room for bubbles and spidey-sense)
FOOT = QPointF(W / 2, H - 12)   # where his feet stand inside that widget
SPRITE_SCALE = 0.76             # 3D sprite size on screen (x depth: nearer = bigger)
ROAM = 150                      # depth of the floor band above the taskbar he wanders around in
THROW_GRAVITY = 2100.0          # px/s^2 when thrown or falling
GRAVITY = 1400.0                # px/s^2 for the web swing
INTRO_WAKE, INTRO_WAVE, INTRO_STAND, INTRO_END = 1.8, 2.3, 5.2, 6.1   # wake-up intro timeline (seconds)


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
    """A lit 3D tube: soft dark rim, shaded side, base colour and a specular streak (light from top-left)."""
    path = QPainterPath(pts[0])
    for q in pts[1:]:                       # straight segments with round joints = a bent elbow/knee
        path.lineTo(q)
    p.setBrush(Qt.BrushStyle.NoBrush)

    def stroke(col: QColor, w: float, dx: float = 0.0, dy: float = 0.0) -> None:
        p.setPen(QPen(col, w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(path.translated(dx, dy))

    rim = QColor(color.darker(320))
    rim.setAlpha(225)
    stroke(rim, width + 2.2)                               # silhouette
    stroke(color.darker(155), width)                       # shadow side
    stroke(color, width * 0.72, -0.9, -0.9)                # lit body
    hi = QColor(color.lighter(165))
    hi.setAlpha(190)
    stroke(hi, width * 0.24, -1.9, -2.0)                   # specular streak


def _ball(p: QPainter, c: QPointF, r: float, color: QColor) -> None:
    """A shiny shaded sphere (gloves, boots, joints)."""
    g = QRadialGradient(QPointF(c.x() - r * 0.35, c.y() - r * 0.4), r * 1.35)
    g.setColorAt(0.0, color.lighter(170))
    g.setColorAt(0.35, color)
    g.setColorAt(1.0, color.darker(210))
    rim = QColor(color.darker(320))
    rim.setAlpha(225)
    p.setPen(QPen(rim, 1.4))
    p.setBrush(QBrush(g))
    p.drawEllipse(c, r, r)

def _leg(p: QPainter, hip: QPointF, hip_a: float, knee: float, near: bool) -> QPointF:
    k = _pt(hip, 12, hip_a)
    f = _pt(k, 12, hip_a - knee)
    _limb(p, [k, f], SUIT if near else SUIT.darker(140), 7.6)                  # black suit shin
    _limb(p, [hip, k], SHORTS if near else SHORTS.darker(135), 10.5)           # baggy shorts
    a = hip_a - knee
    heel, toe = _pt(f, 3.5, a + 180 + 90 * 0), _pt(f, 7.5, a + 90)
    _limb(p, [_pt(f, 3.0, a - 90), toe], QColor(245, 246, 250), 8.6)          # chunky white sole
    _limb(p, [_pt(f, 2.0, a - 90), _pt(f, 5.5, a + 90)], SNEAKER if near else SNEAKER.darker(140), 6.4)
    del heel
    return f


def _arm(p: QPainter, sh: QPointF, sh_a: float, elbow: float, near: bool, reach: float = 1.0) -> QPointF:
    e = _pt(sh, 10.5 * reach, sh_a)
    h = _pt(e, 10 * reach, sh_a + elbow)
    _limb(p, [sh, e, h], JACKET if near else JACKET.darker(140), 8.4)          # hoodie sleeve
    _ball(p, h, 4.8, SUIT_RED if near else SUIT_RED.darker(140))               # glove
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


def _draw_head(p: QPainter, pose: Pose, depth: float) -> None:
    """The big masked head as a lit 3D sphere: shading + specular shine + rim light, the mask's web
    pattern projected onto the sphere around the face, and glossy lenses that sit on the curve."""
    R, RY = 31.0, 29.0
    yaw = math.radians(34 * pose.turn)                 # head turns toward where he is going
    pitch = math.radians(-8)                           # face looks slightly down
    p.save()
    p.translate(depth * 0.6, -74)
    p.rotate(pose.head_tilt)
    head = QRectF(-R, -RY, 2 * R, 2 * RY)
    # red hood behind the head, with a soft peak at the back
    hood = QPainterPath()
    hood.addEllipse(QRectF(-R - 8, -RY - 11, 2 * R + 16, 2 * RY + 17))
    peak = QPainterPath()
    peak.moveTo(-R * 0.75, -RY * 0.55)
    peak.quadTo(-R * 1.05, -RY * 1.35, -R * 0.35, -RY * 1.32)
    peak.lineTo(R * 0.1, -RY * 0.9)
    peak.closeSubpath()
    hood = hood.united(peak)
    cut = QPainterPath()
    cut.addRect(QRectF(-R - 20, -RY - 30, 2 * R + 40, RY + 30 + RY * 0.62))   # open below the cheeks
    hood = hood.intersected(cut)
    hg = QRadialGradient(QPointF(-R * 0.45, -RY * 0.7), R * 1.9)
    hg.setColorAt(0.0, HOOD.lighter(150))
    hg.setColorAt(0.4, HOOD)
    hg.setColorAt(1.0, HOOD.darker(250))
    p.setPen(QPen(QColor(70, 6, 10, 235), 1.8))
    p.setBrush(QBrush(hg))
    p.drawPath(hood)
    # black mask as a lit sphere (cool highlight)
    g = QRadialGradient(QPointF(-R * 0.38, -RY * 0.42), R * 1.55)
    g.setColorAt(0.0, QColor(110, 116, 148))
    g.setColorAt(0.28, SUIT.lighter(160))
    g.setColorAt(0.65, SUIT)
    g.setColorAt(1.0, QColor(6, 6, 10))
    p.setPen(QPen(QColor(8, 8, 12, 235), 1.6))
    p.setBrush(QBrush(g))
    p.drawEllipse(head)

    # face frame: front pole f, right u, up v (rotated by yaw/pitch)
    def rot(x, y, z):
        x, z = x * math.cos(yaw) + z * math.sin(yaw), -x * math.sin(yaw) + z * math.cos(yaw)
        y, z = y * math.cos(pitch) - z * math.sin(pitch), y * math.sin(pitch) + z * math.cos(pitch)
        return x, y, z

    def on_sphere(rho, phi):
        # rho: angle from the face centre; phi: direction around it (0 = right, 90 = up)
        x = math.sin(rho) * math.cos(phi)
        y = -math.sin(rho) * math.sin(phi)
        z = math.cos(rho)
        return rot(x, y, z)

    clip = QPainterPath()
    clip.addEllipse(head)
    p.save()
    p.setClipPath(clip)
    p.setBrush(Qt.BrushStyle.NoBrush)

    def polyline(pts3):
        for (x0, y0, z0), (x1, y1, z1) in zip(pts3, pts3[1:]):
            vis = min(z0, z1)
            if vis <= 0.02:
                continue
            col = QColor(220, 36, 46, int(55 + 150 * min(1.0, vis * 1.6)))   # red web lines, fade toward the edge
            p.setPen(QPen(col, 0.8 + 0.6 * vis))
            p.drawLine(QPointF(x0 * R, y0 * RY), QPointF(x1 * R, y1 * RY))

    for i in range(12):                                 # radial strands from the face centre
        phi = math.radians(i * 30 + 15)
        polyline([on_sphere(math.radians(r), phi) for r in range(0, 100, 6)])
    for rho in (16, 32, 50, 70, 90):                    # concentric rings
        polyline([on_sphere(math.radians(rho), math.radians(a)) for a in range(0, 366, 8)])
    # rim light on the shadow side + soft specular shine
    rim = QRadialGradient(QPointF(R * 0.15, RY * 0.15), R * 1.05)
    rim.setColorAt(0.86, QColor(140, 160, 230, 0))
    rim.setColorAt(0.97, QColor(150, 170, 240, 90))
    rim.setColorAt(1.0, QColor(150, 170, 240, 0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(rim))
    p.drawEllipse(head)
    shine = QRadialGradient(QPointF(-R * 0.42, -RY * 0.5), R * 0.42)
    shine.setColorAt(0.0, QColor(220, 228, 255, 120))
    shine.setColorAt(1.0, QColor(255, 255, 255, 0))
    p.setBrush(QBrush(shine))
    p.drawEllipse(QPointF(-R * 0.42, -RY * 0.5), R * 0.45, RY * 0.36)
    p.restore()

    # glossy lenses sitting on the sphere, foreshortened as the head turns (clipped: they wrap round it)
    p.setClipPath(clip)
    open_ = max(0.08, 1 - pose.blink)
    for side in (-1, 1):
        cx, cy, cz = on_sphere(math.radians(33), math.radians(90 - side * 62))
        if cz <= 0.05:
            continue
        p.save()
        p.translate(cx * R, cy * RY)
        squash = max(0.35, min(1.0, cz * 1.15))           # seen at an angle -> narrower
        p.scale(side * squash, open_)
        eye = QPainterPath()
        eye.moveTo(-8, 5)
        eye.quadTo(-6, -9, 6, -13)
        eye.quadTo(17, -14, 15, -2)
        eye.quadTo(12, 10, -8, 5)
        p.setPen(QPen(QColor(10, 8, 12), 4.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin))
        lens = QLinearGradient(QPointF(0, -13), QPointF(0, 8))
        lens.setColorAt(0.0, QColor(255, 255, 255))
        lens.setColorAt(0.7, QColor(226, 232, 245))
        lens.setColorAt(1.0, QColor(178, 188, 212))
        p.setBrush(QBrush(lens))
        p.drawPath(eye)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(255, 255, 255, 235)))
        p.drawEllipse(QPointF(3, -7), 2.4, 1.6)          # tiny reflection
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(SUIT_RED, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(eye)                                    # red rim inside the black frame
        p.restore()
    p.setClipping(False)
    # the hood's front edge framing the face (over the top and down the sides)
    p.setBrush(Qt.BrushStyle.NoBrush)
    edge = QRectF(-R - 3, -RY - 4, 2 * R + 6, 2 * RY + 6)
    p.setPen(QPen(QColor(70, 6, 10, 235), 9.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawArc(edge, -12 * 16, 204 * 16)
    p.setPen(QPen(HOOD, 7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawArc(edge, -12 * 16, 204 * 16)
    p.setPen(QPen(HOOD.lighter(160), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawArc(edge.adjusted(-1.5, -1.5, 1.5, 1.5), 95 * 16, 70 * 16)   # highlight on the hood rim
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
    torso = QLinearGradient(QPointF(-17, 0), QPointF(17, 0))       # a lit cylinder
    torso.setColorAt(0.0, JACKET.darker(115))
    torso.setColorAt(0.32, JACKET.lighter(140))
    torso.setColorAt(0.6, JACKET)
    torso.setColorAt(1.0, JACKET.darker(230))
    p.setPen(QPen(QColor(10, 18, 50, 225), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(QBrush(torso))
    p.drawPath(body)
    chest = QPainterPath()
    chest.moveTo(-9 + depth * 0.5, -49)
    chest.cubicTo(-11 + depth * 0.5, -36, -7, -27, 0 + depth * 0.4, -24)
    chest.cubicTo(7, -27, 11 + depth * 0.3, -36, 9 + depth * 0.5, -49)
    chest_g = QLinearGradient(QPointF(-11, -49), QPointF(11, -24))
    chest_g.setColorAt(0.0, SUIT.lighter(170))               # the black suit shirt in the open jacket
    chest_g.setColorAt(0.45, SUIT)
    chest_g.setColorAt(1.0, SUIT.darker(160))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(chest_g))
    p.drawPath(chest)
    p.setPen(QPen(SUIT_RED, 1.4))
    ex = depth * 0.45                                               # red spider emblem on the black shirt
    p.drawLine(QPointF(ex, -42), QPointF(ex, -33))
    for dy in (-40, -36):
        p.drawLine(QPointF(ex - 4, dy - 2), QPointF(ex + 4, dy + 2))
        p.drawLine(QPointF(ex + 4, dy - 2), QPointF(ex - 4, dy + 2))
    p.setPen(QPen(QColor(20, 26, 30, 200), 1.3))                    # open jacket edges + hood drawstrings
    p.drawLine(QPointF(-9 + depth * 0.5, -49), QPointF(-6, -24))
    p.drawLine(QPointF(9 + depth * 0.5, -49), QPointF(6, -24))
    p.setPen(QPen(HOOD, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawLine(QPointF(-4 + depth * 0.4, -48), QPointF(-6 + depth * 0.4, -35))
    p.drawLine(QPointF(4 + depth * 0.4, -48), QPointF(5 + depth * 0.4, -37))
    # near leg + near arm
    out["foot_n"] = _leg(p, QPointF(hip.x() + 5 + depth * 0.2, hip.y()), *pose.leg_n, near=True)
    out["hip"] = hip
    gripping = pose.reach_n > 1.5            # arm stretched up to a web: it goes behind the head, past the ear
    if gripping:
        out["hand_n"] = _arm(p, QPointF(12 + depth * 0.3, sh_y), *pose.arm_n, near=True, reach=pose.reach_n)
    _draw_head(p, pose, depth)
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


def _draw_bed(p: QPainter, floor: QPointF, mat_y: float) -> None:
    """A cozy little 3D-shaded bed on the taskbar: wooden frame, mattress and a pillow (head end on the left)."""
    x0, x1 = floor.x() - 128, floor.x() + 34
    wood = QColor(150, 96, 58)
    frame = QLinearGradient(QPointF(0, mat_y), QPointF(0, floor.y()))
    frame.setColorAt(0.0, wood.lighter(125))
    frame.setColorAt(1.0, wood.darker(160))
    p.setPen(QPen(QColor(60, 34, 18, 230), 1.4))
    p.setBrush(QBrush(frame))
    p.drawRoundedRect(QRectF(x0, mat_y + 6, x1 - x0, floor.y() - mat_y - 6), 4, 4)        # base
    p.drawRoundedRect(QRectF(x0 - 6, mat_y - 34, 12, floor.y() - mat_y + 34), 4, 4)       # headboard
    p.drawRoundedRect(QRectF(x1 - 6, mat_y - 12, 12, floor.y() - mat_y + 12), 4, 4)       # footboard
    mat = QLinearGradient(QPointF(0, mat_y - 4), QPointF(0, mat_y + 10))
    mat.setColorAt(0.0, QColor(250, 250, 255))
    mat.setColorAt(1.0, QColor(196, 204, 222))
    p.setBrush(QBrush(mat))
    p.setPen(QPen(QColor(120, 128, 150, 200), 1.2))
    p.drawRoundedRect(QRectF(x0 + 4, mat_y - 4, x1 - x0 - 8, 14), 6, 6)
    pil = QRadialGradient(QPointF(x0 + 30, mat_y - 14), 26)
    pil.setColorAt(0.0, QColor(255, 255, 255))
    pil.setColorAt(1.0, QColor(200, 208, 230))
    p.setBrush(QBrush(pil))
    p.drawRoundedRect(QRectF(x0 + 8, mat_y - 22, 42, 18), 9, 9)                          # pillow


def _draw_blanket(p: QPainter) -> None:
    """A soft blue blanket over his body (in the body's own coordinates, so it lies with him)."""
    g = QLinearGradient(QPointF(-22, 0), QPointF(22, 0))
    g.setColorAt(0.0, QColor(70, 120, 220))
    g.setColorAt(0.4, QColor(120, 170, 250))
    g.setColorAt(1.0, QColor(40, 78, 170))
    path = QPainterPath()
    path.moveTo(-23, -50)
    path.cubicTo(-10, -53, 10, -53, 23, -50)
    path.lineTo(24, 6)
    path.cubicTo(8, 9, -8, 9, -24, 6)
    path.closeSubpath()
    p.setPen(QPen(QColor(20, 40, 100, 220), 1.4))
    p.setBrush(QBrush(g))
    p.drawPath(path)
    p.setPen(QPen(QColor(255, 255, 255, 90), 1.0))                  # quilt stitching
    for y in (-36, -20, -4):
        p.drawLine(QPointF(-20, y), QPointF(20, y))


def _draw_bubble(p: QPainter, at: QPointF, text: str, k: float) -> None:
    """A comic speech bubble whose tail points down-left at ``at``."""
    p.save()
    p.setOpacity(k)
    f = p.font()
    f.setPixelSize(15)
    f.setBold(True)
    p.setFont(f)
    w = p.fontMetrics().horizontalAdvance(text) + 22
    h = 30
    r = QRectF(at.x(), at.y() - h - 10, w, h)
    tail = QPainterPath()
    tail.moveTo(r.left() + 10, r.bottom() - 2)
    tail.lineTo(at.x() - 4, at.y() + 4)
    tail.lineTo(r.left() + 24, r.bottom() - 2)
    body = QPainterPath()
    body.addRoundedRect(r, 13, 13)
    body = body.united(tail)
    p.setPen(QPen(QColor(20, 20, 30), 2.0))
    p.setBrush(QBrush(QColor(255, 255, 255)))
    p.drawPath(body)
    p.setPen(QColor(200, 20, 30))
    p.drawText(r, int(Qt.AlignmentFlag.AlignCenter), text)
    p.restore()


# ── the pet window ───────────────────────────────────────────────────────────

class DesktopPet(QWidget):
    clicked = Signal()                     # open the full Spidy UI
    greet = Signal(str)                    # the wake-up greeting (the app has Spidy say it out loud)

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
        self.intro_t = 0.0                  # wake-up intro after Spidy starts (bed -> wave -> stand)
        self._greeted = False
        self.greeting = "Hello boss!"
        self.hammock: tuple[QPointF, QPointF] | None = None
        self.hammock_k = 0.0
        self._press: QPointF | None = None
        self._dragged = False
        # 3D sprites + physics
        self.bank = SpriteBank()
        self.yaw = 0.0                      # which way he faces (0 = toward you, 90 = right, 180 = away)
        self.target_y = 0.0
        self.vx = self.vy = 0.0             # thrown / falling velocity
        self.spin = 0.0
        self.peak_speed = 0.0
        self.bubble, self.bubble_t = "", 0.0
        self.cut_pieces: list[dict] = []    # snapped web ends recoiling after a cut
        self._drag_samples: list[tuple[float, QPointF]] = []
        self._last_cursor: QPointF | None = None
        self.jump = None                    # (start, end, t) while hopping onto / off a box
        self.perch_box = -1                 # which toy box he is sitting on (-1 = floor)
        self.boxes = PetBoxes()             # the toy boxes in his corner
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
        if not self._greeted:                    # first appearance since Spidy started (every boot/login)
            self._greeted = True
            self.intro_t = 0.0
            self.state, self.state_left = "intro", 99.0
            self._fit_intro()
        else:
            self._set_state("idle", 1.5)
        self.setWindowOpacity(1.0)
        self.boxes.place(self._area())
        self.boxes.show()
        self.show()
        self.raise_()                                   # always in front of his boxes
        self._anim.start()

    def _hide_now(self) -> None:
        self._anim.stop()
        self.hide()
        self.boxes.hide()

    # ── geometry ───────────────────────────────────────────────────────────
    def _area(self):
        screen = self.screen() or QGuiApplication.primaryScreen()
        return screen.availableGeometry()

    def _floor(self) -> float:
        return float(self._area().bottom() - 4)

    def _place_on_floor(self) -> None:
        a = self._area()
        if self.fx <= 0:
            self.fx = a.right() - 300
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
            if self._rng.random() < 0.78:                        # mostly pottering about in his corner
                x0, x1 = self.boxes.play_area(a)
                self.target_x = self._rng.uniform(x0, x1)
                self.target_y = self._floor() - self._rng.uniform(0, 55)
            else:                                               # sometimes a wander across the screen
                self.target_x = self._rng.uniform(a.left() + 60, a.right() - 60)
                self.target_y = self._floor() - self._rng.uniform(0, ROAM) if self.bank.ok else self._floor()
            self.perch_box = -1
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
        if self.perch_box >= 0:                                 # sitting on a box: wave, look, or hop down
            r = self._rng.random()
            if r < 0.35:
                self._set_state("wave", 2.4)
            elif r < 0.6:
                self._set_state("look", 2.6)
            else:
                x0, x1 = self.boxes.play_area(self._area())
                self.start_jump(QPointF(self._rng.uniform(x0, x1), self._floor() - self._rng.uniform(0, 40)), -1)
            return
        if False:                                               # tiny toys: he plays beside them, no climbing
            i = self._rng.randrange(len(self.boxes.tops(self._area())))
            self.start_jump(self.boxes.tops(self._area())[i], i)
            return
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
            speed = 62.0 * self._depth()
            self.walk_phase += dt * speed / 9.0            # stride matched to speed: no foot sliding
            dx, dy = self.target_x - self.fx, self.target_y - self.fy
            dist = math.hypot(dx, dy)
            if dist < speed * dt or dist < 1:
                self.fx, self.fy, self.state_left = self.target_x, self.target_y, 0
            else:
                self.fx += dx / dist * speed * dt
                self.fy += dy / dist * speed * dt
                self.yaw = math.degrees(math.atan2(dx, dy))           # toward you = 0, right = 90, away = 180
                self.facing = 1 if dx >= 0 else -1
            self._fit_floor()
        elif st == "held":
            pass
        elif st == "jump":
            start, end, k = self.jump
            k = min(1.0, k + dt / 0.55)
            self.jump = (start, end, k)
            self.fx = start.x() + (end.x() - start.x()) * k
            self.fy = start.y() + (end.y() - start.y()) * k - 70 * math.sin(math.pi * k)
            self._fit_fly()
            if k >= 1.0:
                self.fx, self.fy = end.x(), end.y()
                self._fit_floor()
                self._set_state("idle", self._rng.uniform(2.5, 5.0))
        elif st == "fly":
            self._fly(dt)
        elif st == "dizzy":
            if self.state_left <= 0:
                self._set_state("idle", 1.0)
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
        elif st == "intro":
            before = self.intro_t
            self.intro_t += dt
            if before < INTRO_WAVE <= self.intro_t:
                self.greet.emit(self.greeting)                       # "Hello boss!" out loud
            if self.intro_t >= INTRO_END:
                self.fx, self.fy = self._intro_stand_x(), self._floor()
                self._fit_floor()
                self._set_state("idle", 1.0)
            else:
                self._fit_intro()
        elif st == "hop":
            self.hop = min(1.0, self.hop + dt / 0.45)
            if self.hop >= 1.0:
                self._hide_now()
                return
        self._check_web_cut()
        self.bubble_t = max(0.0, self.bubble_t - dt)
        for piece in self.cut_pieces:
            piece["age"] += dt
        self.cut_pieces = [pc for pc in self.cut_pieces if pc["age"] < 0.7]
        if self.state_left <= 0 and self.state in ("idle", "walk", "look", "wave"):
            self._choose_next()
        self.update()

    # ── wake-up intro: asleep in bed -> wakes, waves "Hello boss!" -> stands up ───
    def _intro_stand_x(self) -> float:
        return self.fx + 34

    def _fit_intro(self) -> None:
        self.setGeometry(QRect(int(self.fx - 170), int(self.fy - 250), 330, 262))

    def _paint_intro(self, p: QPainter) -> None:
        t = self.intro_t
        floor = self._local(QPointF(self.fx, self.fy))
        bed_fade = 1.0 if t < INTRO_STAND else max(0.0, 1 - (t - INTRO_STAND) / 0.7)
        mat_y = floor.y() - 26                                         # top of the mattress
        feet_lying = QPointF(floor.x() + 20, mat_y - 16)               # lying ON the mattress, head on the pillow
        stand = min(1.0, max(0.0, (t - INTRO_STAND) / 0.6))           # 0 lying -> 1 standing
        ease = stand * stand * (3 - 2 * stand)
        if bed_fade > 0:
            p.save()
            p.setOpacity(bed_fade)
            _draw_bed(p, floor, mat_y)
            p.restore()
        # body: rotate from lying (head on the pillow, to the left) to standing beside the bed
        feet = QPointF(feet_lying.x() + (34 - 20) * ease, feet_lying.y() + (floor.y() - feet_lying.y()) * ease)
        ang = -90 * (1 - ease)
        waving = INTRO_WAVE <= t < INTRO_STAND
        asleep = t < INTRO_WAKE
        if waving:
            pose = wave_pose(t, self.blink)
            w = math.sin(t * 9)
            pose.arm_n = (92 + w * 26, -28 + w * 22)                  # straight up out of the blanket (he's lying)
        else:
            pose = idle_pose(t, 1.0 if asleep else self.blink)
            if t < INTRO_STAND:
                pose.arm_n, pose.arm_f = (6, 10), (-6, 10)            # tucked under the blanket
        pose.sense = 0.0
        p.save()
        p.translate(feet)
        p.rotate(ang)
        draw_spidey_pose(p, pose)
        if t < INTRO_STAND + 0.25:                                      # the blanket, thrown off as he gets up
            p.save()
            p.setOpacity(max(0.0, 1 - max(0.0, t - INTRO_STAND) / 0.25))
            _draw_blanket(p)
            p.restore()
        if waving:                                                      # the waving arm comes out over the blanket
            _arm(p, QPointF(12, -45), *pose.arm_n, near=True, reach=1.45)
        p.restore()
        a = math.radians(ang)                                           # head = feet + (0, -74) rotated by ang
        hx, hy = feet.x() + 74 * math.sin(a), feet.y() - 74 * math.cos(a)
        if asleep:                                                      # Z z z
            for i in range(3):
                k = ((t * 0.8 + i / 3) % 1.0)
                p.save()
                p.setOpacity(1 - k)
                f = p.font()
                f.setPixelSize(int(11 + 6 * k))
                f.setBold(True)
                p.setFont(f)
                p.setPen(QColor(235, 240, 255))
                p.drawText(QPointF(hx + 18 + 14 * k, hy - 26 - 30 * k), "Z")
                p.restore()
        if INTRO_WAVE <= t < INTRO_END - 0.2:
            _draw_bubble(p, QPointF(hx + 22, hy - 34), self.greeting, min(1.0, (t - INTRO_WAVE) / 0.25))

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
                self.grab_to(g)

    def grab_to(self, g: QPointF) -> None:
        """Being carried by the hood: he dangles under the cursor, flailing with a surprised face."""
        import time as _time
        if self.state != "held":
            self.hammock, self.after_shoot = None, ""
            self.state, self.state_left = "held", 99.0
            self.bubble, self.bubble_t = "Hey!", 0.9
        self.fx, self.fy = g.x(), g.y() + 95 * self._depth()
        now = _time.monotonic()
        self._drag_samples = [s for s in self._drag_samples if now - s[0] < 0.12] + [(now, QPointF(g))]
        self._fit_fly()

    def throw_release(self, g: QPointF | None = None) -> None:
        """Let go: he flies off with the cursor's speed and direction (or just drops)."""
        import time as _time
        now = _time.monotonic()
        vx = vy = 0.0
        samples = [s for s in self._drag_samples if now - s[0] < 0.12]
        if len(samples) >= 2 and samples[-1][0] > samples[0][0]:
            dt = samples[-1][0] - samples[0][0]
            vx = (samples[-1][1].x() - samples[0][1].x()) / dt
            vy = (samples[-1][1].y() - samples[0][1].y()) / dt
        self._drag_samples = []
        speed = math.hypot(vx, vy)
        if speed > 2600:
            vx, vy = vx / speed * 2600, vy / speed * 2600
        self.start_fly(vx, vy, "Whoa!" if speed > 350 else "")

    def mouseReleaseEvent(self, e) -> None:
        if e.button() != Qt.MouseButton.LeftButton:
            return
        dragged, self._press = self._dragged, None
        if dragged:
            self.throw_release(e.globalPosition())
            return
        self._set_state("hop", 0.45)
        self.clicked.emit()

    # ── his play corner ───────────────────────────────────────────────────
    def _near_corner(self) -> bool:
        x0, x1 = self.boxes.play_area(self._area())
        return x0 - 40 <= self.fx <= x1 + 40 and self.fy > self._floor() - 70

    def start_jump(self, to: QPointF, box: int) -> None:
        """Hop in an arc onto a toy box (box >= 0) or back down to the floor (box = -1)."""
        self.jump = (QPointF(self.fx, self.fy), QPointF(to), 0.0)
        self.perch_box = box
        self.facing = 1 if to.x() >= self.fx else -1
        self.yaw = 90.0 * self.facing
        self.state, self.state_left = "jump", 99.0

    # ── physics: thrown / falling ──────────────────────────────────────────
    def _depth(self) -> float:
        """Perspective: 1.0 at the front of the floor band (near the taskbar), smaller further back."""
        fl = self._floor()
        k = 1.0 - min(1.0, max(0.0, (fl - self.fy) / ROAM))
        return 0.84 + 0.16 * k

    def start_fly(self, vx: float, vy: float, shout: str = "") -> None:
        self.perch_box = -1
        self.vx, self.vy = vx, vy
        self.spin = 0.0
        self.peak_speed = math.hypot(vx, vy)
        self.hammock, self.after_shoot, self.upside = None, "", False
        self.state, self.state_left = "fly", 99.0
        if shout:
            self.bubble, self.bubble_t = shout, 1.0
        self._fit_fly()

    def _fly(self, dt: float) -> None:
        a = self._area()
        fl = self._floor()
        self.vy += THROW_GRAVITY * dt
        self.fx += self.vx * dt
        self.fy += self.vy * dt
        self.spin += self.vx * dt * 0.55 + (60 if abs(self.vx) < 40 and self.vy < 0 else 0) * dt
        self.peak_speed = max(self.peak_speed, math.hypot(self.vx, self.vy))
        if self.fx < a.left() + 30:                                     # bounce off the screen edges
            self.fx, self.vx = a.left() + 30, abs(self.vx) * 0.55
        elif self.fx > a.right() - 30:
            self.fx, self.vx = a.right() - 30, -abs(self.vx) * 0.55
        if self.fy - 130 < a.top():
            self.fy, self.vy = a.top() + 130, abs(self.vy) * 0.4
        if self.fy >= fl:                                               # hit the floor: bounce, then land
            self.fy = fl
            if self.vy > 260:
                self.vy = -self.vy * 0.38
                self.vx *= 0.7
                self.spin *= 0.5
            else:
                self.vx = self.vy = 0.0
                self.spin = 0.0
                hard = self.peak_speed > 520 or (self.peak_speed > 220 and self._rng.random() < 0.55)
                self.peak_speed = 0.0
                if hard:
                    self._set_state("dizzy", 2.6)
                else:
                    self._set_state("idle", 1.0)
                self._fit_floor()
                return
        self._fit_fly()

    def _fit_fly(self) -> None:
        cx, cy = self.fx, self.fy - 60
        rect = QRect(int(cx - 140), int(cy - 150), 280, 300)
        for pc in self.cut_pieces:
            for pt in (pc["a"], pc["b"]):
                rect = rect.united(QRect(int(pt.x()) - 12, int(pt.y()) - 12, 24, 24))
        self.setGeometry(rect)

    # ── swiping across a web snaps it ─────────────────────────────────────
    def _web_segments(self) -> list[tuple[QPointF, QPointF]]:
        if self.state in ("swing", "lower"):
            return [(self.anchor, self._body_world())]
        if self.state == "hammock" and self.hammock is not None and self.hammock_k > 0.5:
            pts = [self._hammock_point(i / 10) for i in range(11)]
            segs = list(zip(pts, pts[1:]))
            return segs + [(self.hammock[0], pts[0]), (self.hammock[1], pts[-1])]
        return []

    @staticmethod
    def _seg_cross(p1, p2, q1, q2) -> QPointF | None:
        d = (p2.x() - p1.x()) * (q2.y() - q1.y()) - (p2.y() - p1.y()) * (q2.x() - q1.x())
        if abs(d) < 1e-9:
            return None
        t = ((q1.x() - p1.x()) * (q2.y() - q1.y()) - (q1.y() - p1.y()) * (q2.x() - q1.x())) / d
        u = ((q1.x() - p1.x()) * (p2.y() - p1.y()) - (q1.y() - p1.y()) * (p2.x() - p1.x())) / d
        if 0 <= t <= 1 and 0 <= u <= 1:
            return QPointF(p1.x() + t * (p2.x() - p1.x()), p1.y() + t * (p2.y() - p1.y()))
        return None

    def _check_web_cut(self, cursor: QPointF | None = None) -> bool:
        cur = QPointF(cursor) if cursor is not None else QPointF(QCursor.pos())
        prev, self._last_cursor = self._last_cursor, cur
        if prev is None or (cur - prev).manhattanLength() < 3 or self._press is not None:
            return False
        for a, b in self._web_segments():
            hit = self._seg_cross(prev, cur, a, b)
            if hit is not None:
                self.cut_web(hit, a, b)
                return True
        return False

    def cut_web(self, at: QPointF, a: QPointF, b: QPointF) -> None:
        """The web snaps where the cursor crossed it: both ends recoil and Spidey drops with an "Eek!"."""
        self.cut_pieces = [{"a": QPointF(a), "b": QPointF(at), "age": 0.0, "fall": False},
                           {"a": QPointF(at), "b": QPointF(b), "age": 0.0, "fall": True}]
        if self.state in ("swing", "lower"):
            v = self.omega * self.rope                                   # keep the swing's momentum
            vx, vy = math.cos(self.theta) * v, -math.sin(self.theta) * v
            g = self._body_world()
            self.fx, self.fy = g.x(), g.y() + 120 * self._depth()
        else:                                                           # out of the hammock
            spot = self._hammock_point(0.5)
            vx, vy = 0.0, -160.0
            self.fx, self.fy = spot.x(), spot.y() + 60
        self.start_fly(vx, vy, "Eek!")

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
        self._paint_cut_pieces(p)
        if self.bank.ok:
            self._paint_3d(p)
            p.end()
            return
        if st in ("swing", "lower"):
            self._paint_hanging(p)
        elif st == "hammock":
            self._paint_hammock(p)
        elif st == "intro":
            self._paint_intro(p)
        elif st == "shoot":
            self._paint_shoot(p)
        else:
            self._paint_floor(p)
        p.end()

    def _paint_floor(self, p: QPainter) -> None:
        foot = self._local(QPointF(self.fx, self.fy))
        # soft contact shadow (blurred, darker in the middle)
        sh = QRadialGradient(QPointF(foot.x(), foot.y() + 2), 24)
        sh.setColorAt(0.0, QColor(0, 0, 0, 95))
        sh.setColorAt(0.55, QColor(0, 0, 0, 40))
        sh.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(sh))
        p.save()
        p.translate(foot.x(), foot.y() + 2)
        p.scale(1.0, 0.24)
        p.drawEllipse(QPointF(0, 0), 24, 24)
        p.restore()
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


# ── 3D sprite painting ─────────────────────────────────────────────────────────
def _paint_3d(self, p: QPainter) -> None:
    st, b = self.state, self.bank
    S = SPRITE_SCALE * self._depth()
    loc = self._local
    head = None
    if st in ("swing", "lower"):
        anchor, grip = loc(self.anchor), loc(self._body_world())
        draw_web(p, anchor, grip)
        rot = -math.degrees(self.theta)
        if self.upside:
            pts = b.blit(p, "upside", 0, int(self.t * 3), "feet", grip, S, rot + 180)
        else:
            pts = b.blit(p, "hang", 25 * self.facing, int(self.t * 3), "hand_r", grip, S, rot)
        head = pts["head"]
    elif st == "shoot":
        feet = loc(QPointF(self.fx, self.fy))
        _shadow(p, feet, S)
        pts = b.blit(p, "shoot", 30 * self.facing, 0, "feet", feet, S)
        hand, anchor = pts["hand_r"], loc(self.anchor)
        tip = QPointF(hand.x() + (anchor.x() - hand.x()) * self.shoot_k, hand.y() + (anchor.y() - hand.y()) * self.shoot_k)
        draw_web(p, tip, hand)
        head = pts["head"]
    elif st == "hammock":
        self._paint_hammock_net(p)
        if self.hammock is not None and self.hammock_k >= 0.35:
            a, bb = self.hammock
            here = self._hammock_point(0.5, offset=6)
            ahead, behind = self._hammock_point(0.44), self._hammock_point(0.56)
            ang = math.degrees(math.atan2(ahead.x() - behind.x(), -(ahead.y() - behind.y())))
            side = 1 if a.x() < bb.x() else -1
            p.save()
            p.setOpacity(min(1.0, (self.hammock_k - 0.35) / 0.4))
            pts = b.blit(p, "lounge", 70 * side, int(self.t * 0.8), "hip_c", loc(here), S, ang)
            p.restore()
            head = pts["head"]
    elif st == "intro":
        head = self._paint_intro_3d(p, S)
    elif st == "held":
        at = loc(QPointF(self.fx, self.fy - 95 * self._depth()))
        pts = b.blit(p, "tumble", 0, int(self.t * 8), "head", QPointF(at.x(), at.y() + 34 * S),
                     S, math.sin(self.t * 9) * 10)
        head = pts["head"]
    elif st == "jump":
        start, end, k = self.jump
        feet = loc(QPointF(self.fx, self.fy))
        pts = b.blit(p, "walk", 90 * self.facing, 2, "feet", feet, S, 8 * self.facing * math.sin(math.pi * k))
        head = pts["head"]
    elif st == "fly":
        hip = loc(QPointF(self.fx, self.fy - 62 * S / SPRITE_SCALE))
        pts = b.blit(p, "tumble", 0, int(self.t * 10), "hip_c", hip, S, self.spin)
        head = pts["head"]
    elif st == "dizzy":
        feet = loc(QPointF(self.fx, self.fy))
        _shadow(p, feet, S)
        pts = b.blit(p, "dizzy", 0, int(self.t * 2), "feet", feet, S)
        head = pts["head"]
        _stars(p, head, self.t, S)
    else:                                                              # on the floor
        feet = loc(QPointF(self.fx, self.fy))
        _shadow(p, feet, S)
        opacity, scale, lift = 1.0, S, 0.0
        if st == "hop":
            k = self.hop
            lift, scale, opacity = 60 * math.sin(k * math.pi / 2), S * (1 + 0.6 * k), max(0.0, 1 - k ** 2)
        at = QPointF(feet.x(), feet.y() - lift)
        if st == "walk":
            idx = int(self.walk_phase / (2 * math.pi) * b.frames("walk"))
            pts = b.blit(p, "walk", self.yaw, idx, "feet", at, scale, 0, opacity)
        elif st == "wave":
            pts = b.blit(p, "wave", 15 * self.facing, int(self.t * 9), "feet", at, scale, 0, opacity)
        else:
            yaw = max(-90.0, min(90.0, self.yaw if abs(self.yaw) <= 100 else 0.0))
            if st == "look":
                yaw = 50 * math.sin(self.t * 1.8)
            pts = b.blit(p, "idle", yaw, int(self.t * 2.5), "feet", at, scale, 0, opacity)
        head = pts["head"]
    if head is not None:
        if self.sense > 0:
            _spidey_sense(p, head, self.sense)
        if self.bubble_t > 0 and self.bubble:
            _draw_bubble(p, QPointF(head.x() + 26 * S, head.y() - 40 * S), self.bubble, min(1.0, self.bubble_t * 4))


def _paint_intro_3d(self, p: QPainter, S: float):
    """Asleep in bed (Zzz) -> sits up and waves "Hello boss!" -> stands up beside the bed."""
    t = self.intro_t
    b = self.bank
    floor = self._local(QPointF(self.fx, self.fy))
    bed_fade = 1.0 if t < INTRO_STAND else max(0.0, 1 - (t - INTRO_STAND) / 0.7)
    mat_y = floor.y() - 26
    if bed_fade > 0:
        p.save()
        p.setOpacity(bed_fade)
        _draw_bed(p, floor, mat_y)
        p.restore()
    hips_bed = QPointF(floor.x() - 30, mat_y - 14)
    stand = min(1.0, max(0.0, (t - INTRO_STAND) / 0.6))
    ease = stand * stand * (3 - 2 * stand)
    if t < INTRO_WAVE:                                                 # lying asleep under the blanket
        pts = b.blit(p, "sleep", 0, 0, "hip_c", hips_bed, S, -90)
        if t < INTRO_STAND:
            _blanket_over(p, pts, S)
        head = pts["head"]
        if t < INTRO_WAKE:
            _zzz(p, head, t)
        return head
    if t < INTRO_STAND:                                                # sits up, waves from bed
        pts = b.blit(p, "wave", 0, int(t * 9), "hip_c", hips_bed, S, -32)
        _blanket_over(p, pts, S, legs_only=True)
    else:                                                              # gets up, bed fades
        feet_end = QPointF(floor.x() + 34, floor.y())
        hips_end = QPointF(feet_end.x(), feet_end.y() - 58 * S / SPRITE_SCALE)
        at = QPointF(hips_bed.x() + (hips_end.x() - hips_bed.x()) * ease, hips_bed.y() + (hips_end.y() - hips_bed.y()) * ease)
        pts = b.blit(p, "idle", 0, 0, "hip_c", at, S, -32 * (1 - ease))
    head = pts["head"]
    if INTRO_WAVE <= t < INTRO_END - 0.2:
        _draw_bubble(p, QPointF(head.x() + 22, head.y() - 40), self.greeting, min(1.0, (t - INTRO_WAVE) / 0.25))
    return head


def _paint_cut_pieces(self, p: QPainter) -> None:
    """Snapped web ends: the top piece springs back to its anchor, the bottom one falls and fades."""
    for pc in self.cut_pieces:
        k = min(1.0, pc["age"] / 0.5)
        a, b = self._local(pc["a"]), self._local(pc["b"])
        if pc["fall"]:
            a = QPointF(a.x() + (b.x() - a.x()) * k, a.y() + (b.y() - a.y()) * k + 40 * k)
        else:
            b = QPointF(b.x() + (a.x() - b.x()) * k, b.y() + (a.y() - b.y()) * k)
        draw_web(p, a, b, alpha=max(0.0, 1 - k))


def _paint_hammock_net(self, p: QPainter) -> None:
    if self.hammock is None:
        return
    k = self.hammock_k
    a, b = self.hammock
    n = 22
    top = [self._local(self._hammock_point(i / n)) for i in range(n + 1)]
    bot = [self._local(self._hammock_point(i / n, offset=26 * math.sin(math.pi * i / n))) for i in range(n + 1)]
    shown = max(2, int((n + 1) * min(1.0, k * 1.4)))
    p.save()
    for pts, w in ((top, 2.2), (bot, 1.8)):
        path = QPainterPath(pts[0])
        for q in pts[1:shown]:
            path.lineTo(q)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(20, 20, 30, 140), w + 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(path)
        p.setPen(QPen(WEB, w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(path)
    p.setPen(QPen(QColor(246, 248, 255, 210), 1.1))
    for i in range(min(shown, n)):
        p.drawLine(top[i], bot[i + 1])
        p.drawLine(bot[i], top[i + 1])
    draw_web(p, self._local(a), top[0])
    draw_web(p, self._local(b), top[-1])
    p.restore()


DesktopPet._paint_3d = _paint_3d
DesktopPet._paint_intro_3d = _paint_intro_3d
DesktopPet._paint_cut_pieces = _paint_cut_pieces
DesktopPet._paint_hammock_net = _paint_hammock_net


def _shadow(p: QPainter, feet: QPointF, S: float) -> None:
    sh = QRadialGradient(QPointF(0, 0), 30)
    sh.setColorAt(0.0, QColor(0, 0, 0, 100))
    sh.setColorAt(0.55, QColor(0, 0, 0, 42))
    sh.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.save()
    p.translate(feet.x(), feet.y() + 1)
    p.scale(S / SPRITE_SCALE, 0.24 * S / SPRITE_SCALE)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(sh))
    p.drawEllipse(QPointF(0, 0), 30, 30)
    p.restore()


def _stars(p: QPainter, head: QPointF, t: float, S: float) -> None:
    """Little yellow stars circling his head after a hard landing."""
    p.save()
    for i in range(4):
        a = t * 3.2 + i * math.pi / 2
        c = QPointF(head.x() + math.cos(a) * 34 * S / SPRITE_SCALE, head.y() - 30 * S / SPRITE_SCALE + math.sin(a) * 9)
        star = QPainterPath()
        for k in range(10):
            r = 6.5 if k % 2 == 0 else 2.8
            ang = -math.pi / 2 + k * math.pi / 5
            q = QPointF(c.x() + math.cos(ang) * r, c.y() + math.sin(ang) * r)
            star.moveTo(q) if k == 0 else star.lineTo(q)
        star.closeSubpath()
        p.setPen(QPen(QColor(120, 80, 0), 1.0))
        p.setBrush(QBrush(QColor(255, 215, 40)))
        p.drawPath(star)
    p.restore()


def _zzz(p: QPainter, head: QPointF, t: float) -> None:
    for i in range(3):
        k = ((t * 0.8 + i / 3) % 1.0)
        p.save()
        p.setOpacity(1 - k)
        f = p.font()
        f.setPixelSize(int(11 + 6 * k))
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(235, 240, 255))
        p.drawText(QPointF(head.x() + 18 + 14 * k, head.y() - 26 - 30 * k), "Z")
        p.restore()


def _blanket_over(p: QPainter, pts: dict, S: float, legs_only: bool = False) -> None:
    """A soft blue blanket from his feet up to his chest (or just over his legs while sitting up)."""
    feet, hip = pts["feet"], pts["hip_c"]
    top = hip if legs_only else QPointF(hip.x() + (pts["head"].x() - hip.x()) * 0.45, hip.y() + (pts["head"].y() - hip.y()) * 0.45)
    dx, dy = top.x() - feet.x(), top.y() - feet.y()
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L, dx / L
    w = 30 * S / SPRITE_SCALE
    poly = QPainterPath(QPointF(feet.x() + nx * w - dx / L * 8, feet.y() + ny * w - dy / L * 8))
    poly.lineTo(QPointF(top.x() + nx * w, top.y() + ny * w))
    poly.quadTo(QPointF(top.x() + dx / L * 6, top.y() + dy / L * 6), QPointF(top.x() - nx * w, top.y() - ny * w))
    poly.lineTo(QPointF(feet.x() - nx * w - dx / L * 8, feet.y() - ny * w - dy / L * 8))
    poly.closeSubpath()
    g = QLinearGradient(QPointF(feet.x() + nx * w, feet.y() + ny * w), QPointF(feet.x() - nx * w, feet.y() - ny * w))
    g.setColorAt(0.0, QColor(70, 120, 220))
    g.setColorAt(0.45, QColor(122, 172, 250))
    g.setColorAt(1.0, QColor(40, 78, 170))
    p.setPen(QPen(QColor(20, 40, 100, 220), 1.4))
    p.setBrush(QBrush(g))
    p.drawPath(poly)


class PetBoxes(QWidget):
    """A little stack of 3D-shaded toy boxes in the bottom-right corner: his play spot."""

    BOXES = [  # (x offset from the right edge, width, height, kind) -- left to right
        (236, 18, 15, "toy"), (196, 24, 20, "cardboard"), (196 - 3, 17, 13, "cardboard_top"), (150, 16, 16, "blue")]

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._area = None

    def _rects(self, area) -> list[tuple[QRectF, str]]:
        floor = area.bottom() - 4
        out = []
        for dx, w, h, kind in self.BOXES:
            x = area.right() - dx
            if kind == "cardboard_top":                       # stacked on the big cardboard box
                base = out[1][0]
                out.append((QRectF(base.left() + 10, base.top() - h, w, h), kind))
            else:
                out.append((QRectF(x, floor - h, w, h), kind))
        return out

    def tops(self, area) -> list[QPointF]:
        """Where his feet go when he sits on each box."""
        return [QPointF(r.center().x(), r.top() + 3) for r, _ in self._rects(area)]

    def play_area(self, area) -> tuple[float, float]:
        return float(area.right() - 360), float(area.right() - 40)

    def place(self, area) -> None:
        self._area = area
        self.setGeometry(QRect(int(area.right() - 330), int(area.bottom() - 200), 320, 200))
        self.update()

    def paintEvent(self, _e) -> None:
        if self._area is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.translate(-self.x(), -self.y())
        for r, kind in self._rects(self._area):
            _draw_box(p, r, kind)
        p.end()


def _draw_box(p: QPainter, r: QRectF, kind: str) -> None:
    """A 3D-shaded box: front face, lit top face and darker side, with a little decoration."""
    base = {"toy": QColor(205, 36, 46), "blue": QColor(46, 92, 190)}.get(kind, QColor(196, 150, 96))
    d = r.height() * 0.28                                              # depth of the top/side faces
    # soft contact shadow
    sh = QRadialGradient(QPointF(r.center().x(), r.bottom()), r.width() * 0.75)
    sh.setColorAt(0.0, QColor(0, 0, 0, 70))
    sh.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(sh))
    p.drawEllipse(QPointF(r.center().x() + d * 0.4, r.bottom() + 1), r.width() * 0.75, 6)
    outline = QPen(base.darker(260), 1.3)
    top = QPainterPath()
    top.moveTo(r.topLeft())
    top.lineTo(QPointF(r.left() + d, r.top() - d * 0.6))
    top.lineTo(QPointF(r.right() + d, r.top() - d * 0.6))
    top.lineTo(r.topRight())
    top.closeSubpath()
    side = QPainterPath()
    side.moveTo(r.topRight())
    side.lineTo(QPointF(r.right() + d, r.top() - d * 0.6))
    side.lineTo(QPointF(r.right() + d, r.bottom() - d * 0.6))
    side.lineTo(r.bottomRight())
    side.closeSubpath()
    front = QLinearGradient(r.topLeft(), r.bottomLeft())
    front.setColorAt(0.0, base.lighter(118))
    front.setColorAt(1.0, base.darker(125))
    p.setPen(outline)
    p.setBrush(QBrush(front))
    p.drawRoundedRect(r, 3, 3)
    p.setBrush(QBrush(base.lighter(145)))
    p.drawPath(top)
    p.setBrush(QBrush(base.darker(165)))
    p.drawPath(side)
    if kind.startswith("cardboard"):                                  # packing tape
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(222, 196, 140, 230)))
        p.drawRect(QRectF(r.center().x() - 6, r.top(), 12, r.height() * 0.45))
        tape = QPainterPath()
        tape.moveTo(QPointF(r.center().x() - 6, r.top()))
        tape.lineTo(QPointF(r.center().x() - 6 + d, r.top() - d * 0.6))
        tape.lineTo(QPointF(r.center().x() + 6 + d, r.top() - d * 0.6))
        tape.lineTo(QPointF(r.center().x() + 6, r.top()))
        p.drawPath(tape)
    elif kind == "toy":                                               # spider logo on the red block
        c = r.center()
        p.setPen(QPen(QColor(20, 16, 24), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(QBrush(QColor(20, 16, 24)))
        p.drawEllipse(c, 4.5, 6)
        for k in (-1, 1):
            for dy in (-4, 0, 4):
                p.drawLine(QPointF(c.x() + k * 3, c.y() + dy), QPointF(c.x() + k * 11, c.y() + dy + (dy or -2) * 0.8))
    else:                                                             # a white star on the blue box
        c = r.center()
        star = QPainterPath()
        for k in range(10):
            rr = 12 if k % 2 == 0 else 5
            ang = -math.pi / 2 + k * math.pi / 5
            q = QPointF(c.x() + math.cos(ang) * rr, c.y() + math.sin(ang) * rr)
            star.moveTo(q) if k == 0 else star.lineTo(q)
        star.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(250, 250, 255)))
        p.drawPath(star)


class _NullPainter:
    """Swallows drawing calls so draw_spidey_pose can be used to measure joint positions."""

    def __getattr__(self, _name):
        return lambda *a, **k: None
