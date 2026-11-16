"""
Render the desktop pet as a real 3D character (signed-distance-field ray marching in numpy).

The chibi Spidey is built from smooth 3D shapes -- a big sphere head inside a red hood, an organic
torso/hips/limbs blended together like a vinyl toy, chunky sneakers -- and lit with a key light,
fill, rim light, ambient occlusion and a specular highlight. The mask's red web lines and the
red-rimmed white eyes are painted onto the 3D head surface.

Output: assets/pet/<anim>_<view>_<frame>.png sprites (RGBA, transparent background) plus
assets/pet/meta.json with each frame's hand / foot / head positions so webs attach exactly.

    python scripts/render_pet_sprites.py            # render everything (a few minutes, once)
    python scripts/render_pet_sprites.py --preview  # a quick sheet of a few frames
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "pet")

# ── colours (linear-ish 0..1) ──────────────────────────────────────────────────
C = {
    "suit": (0.035, 0.045, 0.10), "web": (0.30, 0.38, 0.58), "hood": (0.66, 0.07, 0.09),
    "hood_in": (0.20, 0.02, 0.03), "jacket": (0.085, 0.14, 0.13), "shirt": (0.03, 0.035, 0.07),
    "shorts": (0.19, 0.14, 0.12), "shin": (0.035, 0.045, 0.10), "sneaker": (0.80, 0.07, 0.10),
    "sole": (0.95, 0.95, 0.97), "glove": (0.78, 0.07, 0.10), "eye": (0.97, 0.97, 1.0),
    "eye_rim": (0.85, 0.08, 0.12), "eye_frame": (0.01, 0.01, 0.03), "string": (0.82, 0.10, 0.13),
    "phone": (0.05, 0.05, 0.07), "screen": (0.45, 0.78, 1.0), "spider_red": (0.85, 0.08, 0.12),
    "can": (0.34, 0.60, 0.58), "can_cap": (0.70, 0.72, 0.75), "fingers": (0.03, 0.03, 0.06),
}
MAT = {name: i for i, name in enumerate(["head", "hood", "torso", "hips", "thigh", "shin", "shoe", "arm",
                                         "glove", "string", "phone", "can"])}


# ── pose ───────────────────────────────────────────────────────────────────────
@dataclass
class Pose:
    """Angles in degrees. Legs: hip swing (+ forward), knee bend. Arms: swing (+ forward), raise
    (sideways, 0 = down, 90 = straight out, 180 = up), elbow bend."""
    hip_l: float = 0.0
    knee_l: float = 0.0
    hip_r: float = 0.0
    knee_r: float = 0.0
    arm_l: tuple[float, float, float] = (0.0, 12.0, 15.0)
    arm_r: tuple[float, float, float] = (0.0, 12.0, 15.0)
    bob: float = 0.0
    lean: float = 0.0           # forward lean of the body (deg)
    head_yaw: float = 0.0
    head_tilt: float = 0.0
    phone: bool = False
    leg_spread: float = 0.0     # extra sideways splay (deg) -- for the upside-down hang
    can: bool = False           # (unused) spray can prop
    head_pitch: float = 0.0     # + looks up, - looks down
    expr: str = "normal"        # normal | surprised (big round eyes) | dizzy (spiral eyes)


def _rot_x(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0], v[1] * c - v[2] * s, v[1] * s + v[2] * c])


def _limb_dir_leg(hip_deg, side, spread):
    a = math.radians(hip_deg)
    d = np.array([0.0, -math.cos(a), math.sin(a)])
    if spread:
        b = math.radians(spread) * side
        d = np.array([math.sin(b) * math.cos(a), -math.cos(b) * math.cos(a), math.sin(a)])
    return d / np.linalg.norm(d)


def _arm_dirs(swing, raise_, elbow, side):
    s, r = math.radians(swing), math.radians(raise_)
    up = np.array([side * math.sin(r), -math.cos(r), 0.0])            # raised sideways
    up = _rot_x(up, -s)                                                # then swung forward
    up /= np.linalg.norm(up)
    # elbow bends the forearm forward / inward
    e = math.radians(elbow)
    fwd = np.array([0.0, 0.0, 1.0]) - up * up[2]
    if np.linalg.norm(fwd) < 1e-6:
        fwd = np.array([-side, 0.0, 0.0])
    fwd /= np.linalg.norm(fwd)
    fore = up * math.cos(e) + fwd * math.sin(e)
    return up, fore / np.linalg.norm(fore)


def build_scene(pose: Pose) -> dict:
    """Joint positions (character space: x right, y up, z forward; feet at y = 0)."""
    lift = pose.bob
    hip_c = np.array([0.0, 0.56 + lift, 0.0])                         # legs long enough to see the suit
    joints = {}
    for side, name, hip_a, knee in ((-1, "l", pose.hip_l, pose.knee_l), (1, "r", pose.hip_r, pose.knee_r)):
        hip = hip_c + np.array([0.13 * side, 0.0, 0.0])
        d1 = _limb_dir_leg(hip_a, side, pose.leg_spread)
        knee_p = hip + d1 * 0.20
        d2 = _limb_dir_leg(hip_a - knee, side, pose.leg_spread * 0.4)
        ankle = knee_p + d2 * 0.19
        joints[f"hip_{name}"], joints[f"knee_{name}"], joints[f"ankle_{name}"] = hip, knee_p, ankle
        joints[f"foot_dir_{name}"] = _rot_x(np.array([0, 0, 1.0]), -math.radians(hip_a - knee))
    lean = math.radians(pose.lean)
    chest = hip_c + np.array([0.0, 0.31 * math.cos(lean), 0.31 * math.sin(lean)])   # short compact body
    joints["chest"] = chest
    joints["hip_c"] = hip_c
    for side, name, arm in ((-1, "l", pose.arm_l), (1, "r", pose.arm_r)):
        sh = chest + np.array([0.27 * side, -0.03, 0.0])
        up, fore = _arm_dirs(*arm, side)
        el = sh + up * 0.16                                           # short chubby arms
        hand = el + fore * 0.15
        joints[f"sh_{name}"], joints[f"el_{name}"], joints[f"hand_{name}"] = sh, el, hand
    joints["head"] = chest + np.array([0.0, 0.60, 0.03])
    return joints


# ── signed distance functions ─────────────────────────────────────────────────
def sd_sphere(p, c, r):
    return np.linalg.norm(p - c, axis=-1) - r


def sd_ellipsoid(p, c, r):
    q = (p - c) / r
    k0 = np.linalg.norm(q, axis=-1)
    k1 = np.linalg.norm(q / r, axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)


def sd_capsule(p, a, b, r, r2=None):
    pa, ba = p - a, b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0.0, 1.0)
    rr = r if r2 is None else r + (r2 - r) * h
    return np.linalg.norm(pa - h[:, None] * ba, axis=-1) - rr


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


class Character:
    def __init__(self, pose: Pose):
        self.pose = pose
        self.j = build_scene(pose)
        j = self.j
        self.head_c = j["head"]
        hy, ht = math.radians(pose.head_yaw), math.radians(pose.head_tilt)
        hp = math.radians(pose.head_pitch)
        f = np.array([math.sin(hy) * math.cos(hp), math.sin(hp) - 0.16, math.cos(hy) * math.cos(hp)])
        f = _rot_x(f, 0) / np.linalg.norm(f)
        right = np.array([math.cos(hy), 0.0, -math.sin(hy)])
        up = np.cross(f, right)
        up /= np.linalg.norm(up)
        # head tilt rotates right/up around the face axis
        c, s = math.cos(ht), math.sin(ht)
        self.face = (f, right * c + up * s, -right * s + up * c)

    def _to_head_local(self, p):
        """Undo the head's yaw / pitch around the head centre (vectorised)."""
        hy, hp = math.radians(self.pose.head_yaw), math.radians(self.pose.head_pitch)
        if hy == 0 and hp == 0:
            return p
        q = p - self.head_c
        c, s_ = math.cos(-hy), math.sin(-hy)
        x = q[:, 0] * c + q[:, 2] * s_
        z = -q[:, 0] * s_ + q[:, 2] * c
        y = q[:, 1]
        c2, s2 = math.cos(hp), math.sin(hp)                  # undo pitch (look up = positive)
        y2 = y * c2 - z * s2
        z2 = y * s2 + z * c2
        return np.stack([x, y2, z2], axis=1) + self.head_c

    def parts(self, p):
        """Distance of every part: list of (material, distance)."""
        j, P = self.j, self.pose
        hc = self.head_c
        out = []
        p_world = p
        p = self._to_head_local(p)                     # head + hood turn together as one piece
        # head: round skull + soft cheeks narrowing to a small chin
        skull = sd_ellipsoid(p, hc, np.array([0.58, 0.53, 0.545]))
        cheeks = sd_ellipsoid(p, hc + np.array([0, -0.17, 0.07]), np.array([0.45, 0.35, 0.43]))
        out.append(("head", smin(skull, cheeks, 0.12)))
        # hood: fabric shell that frames the face with a thick rolled rim, drapes down the back of the
        # neck onto the shoulders, and ends in a soft tip flopping backwards
        f = np.array([0.0, -0.16, 1.0]) / np.linalg.norm([0.0, -0.16, 1.0])
        rel = p - hc
        shell = np.maximum(sd_ellipsoid(p, hc + np.array([0, 0.04, -0.06]), np.array([0.665, 0.645, 0.64])),
                           -sd_ellipsoid(p, hc + np.array([0, 0.02, -0.03]), np.array([0.62, 0.60, 0.61])))
        drape = np.maximum(sd_ellipsoid(p, hc + np.array([0, -0.40, -0.20]), np.array([0.46, 0.34, 0.30])),
                           -sd_ellipsoid(p, hc + np.array([0, -0.40, -0.12]), np.array([0.41, 0.30, 0.27])))
        shell = smin(shell, drape, 0.08)
        opening = sd_ellipsoid(p, hc + f * 0.52 + np.array([0, -0.08, 0]), np.array([0.50, 0.56, 0.50]))
        shell = np.maximum(shell, -opening)                                    # face opening
        shell = np.maximum(shell, (hc[1] - 0.20) - p[:, 1] - np.clip(-rel[:, 2] - 0.05, 0, None) * 3.0)
        # rolled rim around the opening (an elliptical tube)
        q = p - (hc + f * 0.40 + np.array([0, -0.06, 0]))
        a_, b_ = 0.50, 0.55
        ell = (np.sqrt((q[:, 0] / a_) ** 2 + (q[:, 1] / b_) ** 2) - 1.0) * 0.52
        rim = np.sqrt(ell ** 2 + (q[:, 2] * 0.9) ** 2) - 0.06
        rim = np.maximum(rim, -(q[:, 1] + 0.40))                                # only around the top and sides
        tip = sd_capsule(p, hc + np.array([0, 0.48, -0.28]), hc + np.array([0, 0.56, -0.66]), 0.15, 0.045)
        out.append(("hood", smin(smin(shell, rim, 0.05), tip, 0.12)))
        p = p_world
        # body: a short round hoodie with a little belly, blended with chubby limbs like a vinyl toy
        chest, hip_c = j["chest"], j["hip_c"]
        torso = sd_capsule(p, hip_c + np.array([0, 0.10, 0.0]), chest + np.array([0, 0.0, 0]), 0.265, 0.25)
        belly = sd_ellipsoid(p, hip_c + np.array([0, 0.12, 0.045]), np.array([0.265, 0.20, 0.23]))
        out.append(("torso", smin(torso, belly, 0.08)))
        out.append(("hips", sd_ellipsoid(p, hip_c + np.array([0, -0.03, 0]), np.array([0.30, 0.16, 0.23]))))
        for s in "lr":
            mid = j[f"hip_{s}"] + (j[f"knee_{s}"] - j[f"hip_{s}"]) * 0.55
            out.append(("thigh", sd_capsule(p, j[f"hip_{s}"], mid, 0.135, 0.13)))                 # baggy shorts
            out.append(("shin", sd_capsule(p, mid, j[f"knee_{s}"], 0.105, 0.10)))                 # navy suit leg
            out.append(("shin", sd_capsule(p, j[f"knee_{s}"], j[f"ankle_{s}"], 0.10, 0.085)))
            fd = j[f"foot_dir_{s}"]
            heel = j[f"ankle_{s}"] + np.array([0, -0.065, 0]) - fd * 0.07
            toe = j[f"ankle_{s}"] + fd * 0.18 + np.array([0, -0.07, 0])
            out.append(("shoe", smin(sd_capsule(p, heel, toe, 0.125, 0.12),
                                     sd_capsule(p, j[f"ankle_{s}"] + np.array([0, -0.04, -0.01]),
                                                j[f"ankle_{s}"] + np.array([0, 0.07, -0.01]), 0.105), 0.05)))   # chunky high-tops
            out.append(("arm", sd_capsule(p, j[f"sh_{s}"], j[f"el_{s}"], 0.105, 0.10)))           # puffy sleeves
            out.append(("arm", sd_capsule(p, j[f"el_{s}"], j[f"hand_{s}"], 0.10, 0.105)))         # wide cuffs
            out.append(("glove", sd_sphere(p, j[f"hand_{s}"], 0.115)))                            # round mitten hands
            # hood drawstrings hanging on the chest
            sx = 0.07 if s == "r" else -0.07
            top = chest + np.array([sx, 0.08, 0.24])
            out.append(("string", sd_capsule(p, top, top + np.array([sx * 0.3, -0.19, 0.05]), 0.02)))
        if P.can:                                                          # spray can in the left hand
            h = j["hand_l"]
            base = h + np.array([0.02, -0.02, 0.10])
            out.append(("can", sd_capsule(p, base, base + np.array([0, 0.24, 0]), 0.075)))
        if P.phone:
            h = j["hand_r"]
            out.append(("phone", np.maximum(np.abs(p - (h + np.array([-0.02, 0.10, 0.06]))) -
                                            np.array([0.07, 0.11, 0.015]), 0).max(axis=-1) - 0.008))
        return out

    def sdf(self, p, want_material=False):
        parts = self.parts(p)
        body_names = {"torso", "hips", "thigh", "shin", "arm"}
        body = None
        others = []
        for name, d in parts:
            if name in body_names:
                body = d if body is None else smin(body, d, 0.045)
            else:
                others.append(d)
        d = body
        for o in others:
            d = np.minimum(d, o)
        if not want_material:
            return d
        names = [n for n, _ in parts]
        stack = np.stack([dd for _, dd in parts], axis=0)
        idx = np.argmin(stack, axis=0)
        return d, np.array([MAT[names[i]] for i in range(len(names))])[idx]


# ── surface colour (texturing) ────────────────────────────────────────────────
def head_color(ch: Character, p):
    d = p - ch.head_c
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    f, u, v = ch.face
    cz = np.clip(d @ f, -1, 1)
    x, y = d @ u, d @ v
    theta = np.arccos(cz)
    r = np.tan(theta / 2)                                    # stereographic radius around the face centre
    phi = np.arctan2(y, x)
    sx, sy = r * np.cos(phi), r * np.sin(phi)
    col = np.tile(np.array(C["suit"]), (len(p), 1))
    # red web: concentric rings + radial strands (constant width on the surface)
    ring = np.abs(r / 0.17 - np.round(r / 0.17)) * 0.17 < 0.0065 * (1 + r)
    sector = (2 * np.pi) / 14
    rad = np.abs(((phi + sector / 2) % sector) - sector / 2) * np.maximum(r, 0.02) < 0.0068
    web = (ring | rad) & (theta < 2.5)
    col[web] = C["web"]
    # eyes: big angular lenses -- white, red rim, black frame
    expr = ch.pose.expr
    for side in (-1, 1):
        ex, ey = sx - side * 0.235, sy - 0.07
        a = math.radians(24) * side
        rx, ry = ex * math.cos(a) + ey * math.sin(a), -ex * math.sin(a) + ey * math.cos(a)
        a_r, b_r = 0.215, 0.145
        e = (np.abs(rx / a_r) ** 1.45 + np.abs(ry / b_r) ** 1.45) ** (1 / 1.45)     # pointed, angular lens
        e = np.maximum(e, (-ry + rx * side * -0.5) / 0.115)                         # sharp inner-bottom cut
        if expr in ("surprised", "dizzy"):                                          # big round shocked eyes
            e = np.sqrt((ex / 0.20) ** 2 + ((ey - 0.02) / 0.19) ** 2)
        front = cz > 0.15
        col[(e < 1.42) & front] = C["eye_frame"]
        col[(e < 1.18) & front] = C["eye_rim"]
        col[(e < 1.0) & front] = C["eye"]
        if expr == "angry":
            brow = (ry > 0.03 + rx * side * 0.6) & front & (e < 1.42)
            col[brow & (e < 1.18)] = C["eye_frame"]
        if expr == "closed":                                                       # asleep: eyes shut
            lid = (e < 1.42) & front
            col[lid] = C["suit"]
            col[lid & (np.abs(ry + 0.02) < 0.018)] = C["eye_rim"]
        if expr == "dizzy":                                                         # @_@ spiral eyes
            rr = np.sqrt(ex ** 2 + (ey - 0.02) ** 2)
            ang = np.arctan2(ey - 0.02, ex) * side
            spiral = np.abs(((rr / 0.19) * 3.0 - ang / (2 * np.pi)) % 1.0 - 0.5) > 0.36
            col[(e < 0.92) & front & spiral] = C["suit"]
    return col


def torso_color(ch: Character, p):
    jacket = np.array(C["jacket"])
    col = np.tile(jacket, (len(p), 1))
    chest, hip_c = ch.j["chest"], ch.j["hip_c"]
    rel = p - hip_c
    front = rel[:, 2] > 0.10
    h = p[:, 1] - hip_c[1]
    # ribbed hem band at the bottom of the hoodie
    hem = h < 0.035
    col[hem] = jacket * 0.72
    rib = hem & (np.abs(np.sin(np.arctan2(rel[:, 0], rel[:, 2]) * 28)) > 0.75)
    col[rib] = jacket * 0.6
    # kangaroo pocket on the belly (with a darker stitched edge)
    pocket = front & (h > 0.045) & (h < 0.19) & (np.abs(rel[:, 0]) < 0.16 + (0.19 - h) * 0.25)
    col[pocket] = jacket * 0.86
    edge = front & (np.abs(h - 0.19) < 0.012) & (np.abs(rel[:, 0]) < 0.16)
    col[edge] = jacket * 0.55
    # open zip above the pocket shows the black suit with a red spider
    t = np.clip((h - 0.19) / max(chest[1] - hip_c[1] - 0.19, 1e-3), 0, 1)
    opening = front & (h > 0.19) & (np.abs(rel[:, 0]) < 0.045 + 0.075 * t)
    col[opening] = C["shirt"]
    zip_ = front & (h > 0.19) & (np.abs(np.abs(rel[:, 0]) - (0.045 + 0.075 * t)) < 0.01)
    col[zip_] = (0.55, 0.58, 0.6)
    sp = p - (hip_c + np.array([0, 0.28, 0]))
    spider = opening & (((np.abs(sp[:, 0]) < 0.011) & (np.abs(sp[:, 1]) < 0.05)) |
                        ((np.abs(np.abs(sp[:, 0]) - np.abs(sp[:, 1]) * 1.1) < 0.011) & (np.abs(sp[:, 1]) < 0.045)))
    col[spider] = C["spider_red"]
    return col


def arm_color(ch: Character, p):
    jacket = np.array(C["jacket"])
    col = np.tile(jacket, (len(p), 1))
    near_hand = np.minimum(np.linalg.norm(p - ch.j["hand_l"], axis=-1), np.linalg.norm(p - ch.j["hand_r"], axis=-1))
    cuff = near_hand < 0.155                                                  # ribbed sleeve cuffs
    col[cuff] = jacket * 0.7
    ribs = cuff & (np.abs(np.sin(near_hand * 260)) > 0.8)
    col[ribs] = jacket * 0.55
    return col


def shorts_color(ch: Character, p):
    shorts = np.array(C["shorts"])
    col = np.tile(shorts, (len(p), 1))
    near_knee = np.minimum(np.linalg.norm(p - ch.j["knee_l"], axis=-1), np.linalg.norm(p - ch.j["knee_r"], axis=-1))
    col[near_knee < 0.14] = shorts * 1.35                                    # rolled-up cuffs
    col[(near_knee > 0.13) & (near_knee < 0.15)] = shorts * 0.7
    return col


def shoe_color(ch: Character, p):
    """White high-tops with red panels, a red collar and red laces on a thick white sole."""
    white = np.array(C["sole"])
    red = np.array(C["sneaker"])
    col = np.tile(white, (len(p), 1))
    for s in "lr":
        a = ch.j[f"ankle_{s}"]
        fd = ch.j[f"foot_dir_{s}"]
        near = np.linalg.norm(p - a, axis=-1) < 0.35
        rel = p - a
        along = rel @ fd
        side = np.abs(rel @ np.cross(fd, [0, 1.0, 0]))
        panel = near & (along < 0.06) & (rel[:, 1] > -0.10)                     # red heel / side panel
        col[panel] = red
        col[near & (rel[:, 1] > 0.035)] = red                                    # red collar at the top
        swoosh = near & (np.abs(rel[:, 1] + 0.055 - along * 0.25) < 0.018) & (along > 0.0) & (along < 0.15) & (side > 0.06)
        col[swoosh] = red                                                        # side stripe
        laces = near & (rel[:, 1] > -0.03) & (along > 0.02) & (along < 0.14) & (side < 0.05) & \
            (np.abs(np.sin(along * 70)) > 0.5)
        col[laces] = red
        col[near & (rel[:, 1] < -0.115)] = white                                # thick white sole
    return col


def can_color(ch: Character, p):
    col = np.tile(np.array(C["can"]), (len(p), 1))
    h = ch.j["hand_l"] + np.array([0.02, -0.02, 0.10])
    rel = p - h
    col[rel[:, 1] > 0.20] = C["can_cap"]                                         # grey cap + nozzle
    col[rel[:, 1] < 0.02] = np.array(C["can"]) * 0.6                             # darker base ring
    logo = (np.abs(rel[:, 1] - 0.11) < 0.055) & (rel[:, 2] > 0.03) & (np.abs(rel[:, 0]) < 0.05)
    col[logo] = (0.93, 0.93, 0.95)                                               # white label
    spider = logo & ((np.abs(rel[:, 0]) < 0.008) | (np.abs(np.abs(rel[:, 0]) - np.abs(rel[:, 1] - 0.11)) < 0.008))
    col[spider] = C["spider_red"]
    return col

def shade(ch: Character, p, n, mat, view_dir):
    base = np.zeros((len(p), 3))
    for name, i in MAT.items():
        m = mat == i
        if not m.any():
            continue
        if name == "head":
            base[m] = head_color(ch, p[m])
        elif name == "hood":
            rel_h = p[m] - ch.head_c
            inner = (rel_h @ ch.face[0] > 0.22) & (np.linalg.norm(rel_h, axis=-1) < 0.615)   # inside only; rim stays bright
            c = np.tile(np.array(C["hood"]), (int(m.sum()), 1))
            c[inner] = C["hood_in"]
            base[m] = c
        elif name == "torso":
            base[m] = torso_color(ch, p[m])
        elif name == "hips" or name == "thigh":
            base[m] = shorts_color(ch, p[m])
        elif name == "shin":
            base[m] = C["shin"]
        elif name == "shoe":
            base[m] = shoe_color(ch, p[m])
        elif name == "arm":
            base[m] = arm_color(ch, p[m])
        elif name == "glove":
            base[m] = C["glove"]
        elif name == "string":
            base[m] = C["string"]
        elif name == "can":
            base[m] = can_color(ch, p[m])
        elif name == "phone":
            facing = n[m] @ np.array([0, 0, 1.0]) > 0.7
            c = np.tile(np.array(C["phone"]), (int(m.sum()), 1))
            c[facing] = C["screen"]
            base[m] = c
    key = np.array([-0.55, 0.75, 0.45]); key /= np.linalg.norm(key)
    fill = np.array([0.6, 0.2, 0.5]); fill /= np.linalg.norm(fill)
    back = np.array([0.3, 0.4, -0.85]); back /= np.linalg.norm(back)
    ndl = np.clip(n @ key, 0, 1)
    wrap = np.clip((n @ key + 0.35) / 1.35, 0, 1)
    fil = np.clip(n @ fill, 0, 1)
    # ambient occlusion
    ao = np.ones(len(p))
    for k in range(1, 6):
        h = 0.035 * k
        ao -= (h - ch.sdf(p + n * h)) * (0.9 / k)
    ao = np.clip(ao, 0.25, 1.0)
    v = -view_dir
    hvec = key + v
    hvec /= np.linalg.norm(hvec)
    gloss = np.full(len(p), 32.0)
    gloss[(mat == MAT["head"]) | (mat == MAT["glove"]) | (mat == MAT["shoe"]) | (mat == MAT["hood"])] = 60.0
    spec = np.clip(n @ hvec, 0, 1) ** gloss[:, None].ravel()
    spec_amt = np.where((mat == MAT["head"]) | (mat == MAT["shoe"]) | (mat == MAT["glove"]), 0.55, 0.22)
    rim = np.clip(1 - np.clip(n @ v, 0, 1), 0, 1) ** 3 * np.clip(n @ back + 0.4, 0, 1)
    light = 0.30 * ao + 0.85 * wrap * (0.55 + 0.45 * ndl) * ao + 0.18 * fil * ao
    col = base * light[:, None] + (spec * spec_amt)[:, None] + (rim * 0.30)[:, None] * np.array([0.65, 0.75, 1.0])
    return np.clip(col, 0, 1)


# ── camera + ray marching ─────────────────────────────────────────────────────
def render(pose: Pose, yaw_deg: float, size=(150, 190), ss=2, pitch_deg=10.0, world_h=2.62, extra=None):
    """Render RGBA (uint8) with the character turned by ``yaw_deg`` (0 = facing you, 90 = facing right)."""
    W, H = size[0] * ss, size[1] * ss
    ch = Character(pose)
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    # camera basis in character space: camera looks at the character from direction "back" (toward -view)
    cam_back = np.array([math.sin(-yaw) * math.cos(pitch), math.sin(pitch), math.cos(-yaw) * math.cos(pitch)])
    right = np.array([math.cos(-yaw), 0.0, -math.sin(-yaw)])
    up = np.cross(cam_back, right)
    up /= np.linalg.norm(up)
    scale = world_h / H
    xs = (np.arange(W) - W / 2 + 0.5) * scale
    ys = (H * 0.95 - np.arange(H) - 0.5) * scale - 0.02           # feet near the bottom
    gx, gy = np.meshgrid(xs, ys)
    centre = np.array([0.0, 0.0, 0.0])
    origins = centre + gx.reshape(-1, 1) * right + gy.reshape(-1, 1) * up + cam_back * 4.0
    view = -cam_back
    n_rays = origins.shape[0]
    t = np.zeros(n_rays)
    hit = np.zeros(n_rays, bool)
    active = np.arange(n_rays)
    for _ in range(90):
        if active.size == 0:
            break
        pts = origins[active] + view * t[active, None]
        d = ch.sdf(pts)
        done = d < 0.0015
        hit[active[done]] = True
        t[active] += d
        keep = (~done) & (t[active] < 8.5)
        active = active[keep]
    rgba = np.zeros((n_rays, 4))
    hi = np.where(hit)[0]
    if hi.size:
        p = origins[hi] + view * t[hi, None]
        e = 0.0015
        n = np.stack([ch.sdf(p + np.array([e, 0, 0])) - ch.sdf(p - np.array([e, 0, 0])),
                      ch.sdf(p + np.array([0, e, 0])) - ch.sdf(p - np.array([0, e, 0])),
                      ch.sdf(p + np.array([0, 0, e])) - ch.sdf(p - np.array([0, 0, e]))], axis=1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        _, mat = ch.sdf(p, want_material=True)
        col = shade(ch, p, n, mat, view)
        rgba[hi, :3] = col ** (1 / 1.15)
        rgba[hi, 3] = 1.0
    img = rgba.reshape(H, W, 4)
    img = img.reshape(size[1], ss, size[0], ss, 4).mean(axis=(1, 3))       # supersampled anti-aliasing
    # premultiplied -> straight alpha
    a = img[..., 3:4]
    img[..., :3] = np.where(a > 0, img[..., :3] / np.maximum(a, 1e-6), 0)
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    def project(pt):
        rel = pt - centre
        px = (rel @ right) / scale / ss + size[0] / 2
        py = (H * 0.95 - 0.02 / scale - (rel @ up) / scale) / ss
        return [round(float(px), 1), round(float(py), 1)]

    pts = {k: project(ch.j[k]) for k in ("hand_l", "hand_r", "ankle_l", "ankle_r", "head", "hip_c")}
    pts["feet"] = project((ch.j["ankle_l"] + ch.j["ankle_r"]) / 2 + np.array([0, -0.12, 0]))
    return out, pts


def save_png(path, rgba):
    from PySide6.QtGui import QImage
    h, w = rgba.shape[:2]
    img = QImage(np.ascontiguousarray(rgba).data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
    img.save(path)


# ── animations ────────────────────────────────────────────────────────────────
def walk(i, n):
    ph = 2 * math.pi * i / n
    s = math.sin(ph)
    return Pose(hip_l=28 * s, knee_l=10 + max(0, 42 * math.sin(ph + math.pi / 2)),
                hip_r=-28 * s, knee_r=10 + max(0, 42 * math.sin(ph - math.pi / 2)),
                arm_l=(-30 * s, 14, 25), arm_r=(30 * s, 14, 25), bob=abs(math.cos(ph)) * 0.035, lean=5)


def idle(i, n):
    b = math.sin(2 * math.pi * i / n)
    return Pose(arm_l=(4, 14 + b * 3, 18), arm_r=(4, 14 + b * 3, 18), bob=b * 0.01, head_tilt=b * 3)


def wave(i, n):
    w = math.sin(2 * math.pi * i / n)
    return Pose(arm_r=(28, 118 + w * 16, 55 + w * 30), arm_l=(4, 14, 18), head_tilt=-6, head_yaw=8)


def hang(i, n):                     # right hand holding the web overhead, legs dangling
    w = math.sin(2 * math.pi * i / n)
    return Pose(arm_r=(5, 172, 8), arm_l=(10 + w * 8, 30, 35), hip_l=8 + w * 8, knee_l=25,
                hip_r=-4 + w * 8, knee_r=35, head_tilt=-4)


def upside(i, n):                   # rendered upright; the pet rotates it 180 degrees: web at the feet
    w = math.sin(2 * math.pi * i / n)
    return Pose(hip_l=-10, knee_l=95, hip_r=-10, knee_r=95, leg_spread=38,
                arm_l=(-10, 168, 10 + w * 6), arm_r=(-10, 168, 10 - w * 6), head_tilt=w * 4)


def shoot(i, n):
    return Pose(arm_r=(60, 100, -10), arm_l=(10, 25, 30), hip_l=12, knee_l=10, hip_r=-8, knee_r=12, lean=-3)   # arm out, away from the face


def lounge(i, n):                   # rendered upright; rotated to lie in the hammock / bed
    s = math.sin(2 * math.pi * i / n)
    return Pose(hip_l=70, knee_l=110, hip_r=20, knee_r=10, arm_r=(70, 25, 95 + s * 4),
                arm_l=(-20, 165, 120), phone=True, head_tilt=-8)


def hamsleep(i, n):                 # asleep in the hammock: arms folded on the chest, eyes shut
    b = math.sin(2 * math.pi * i / n)
    return Pose(hip_l=70, knee_l=110, hip_r=20, knee_r=10, arm_r=(40, 120, 150 + b * 3),
                arm_l=(-30, 120, 150), head_tilt=-14 + b * 2, expr="closed")


def hero(i, n):
    return Pose(hip_l=-30, knee_l=70, hip_r=70, knee_r=105, arm_r=(-10, 150, 40), arm_l=(55, 40, 95),
                can=True, lean=8, head_tilt=-8, head_yaw=-10)


def spray(i, n):                     # standing, showing off the spray can (idle variety)
    b = math.sin(2 * math.pi * i / n)
    return Pose(arm_l=(55, 30, 95 + b * 4), arm_r=(4, 14, 18), can=True, head_yaw=-12, head_tilt=-4 + b * 2)


def tumble(i, n):                    # thrown / falling: flailing limbs, shocked face (the pet spins it)
    w = math.sin(2 * math.pi * i / n)
    return Pose(hip_l=40 + w * 25, knee_l=70, hip_r=-20 - w * 25, knee_r=60, arm_l=(10, 140 + w * 25, 30),
                arm_r=(-10, 150 - w * 25, 30), expr="surprised", head_tilt=w * 6)


def dizzy(i, n):                     # sat down hard after a fall: legs out, wobbling head, spiral eyes
    w = math.sin(2 * math.pi * i / n)
    return Pose(hip_l=88, knee_l=12, hip_r=84, knee_r=18, bob=-0.30, arm_l=(-20, 35, 20), arm_r=(-20, 35, 20),
                expr="dizzy", head_tilt=w * 8)


def sit(i, n):                       # sitting calmly on the floor, hands resting, slow breathing
    b = math.sin(2 * math.pi * i / n)
    return Pose(hip_l=86, knee_l=40, hip_r=82, knee_r=46, bob=-0.30, arm_l=(30, 22, 40), arm_r=(30, 22, 40),
                head_tilt=b * 3)


def nap(i, n):                       # dozing while sitting: eyes closed, head nodding
    b = math.sin(2 * math.pi * i / n)
    return Pose(hip_l=86, knee_l=40, hip_r=82, knee_r=46, bob=-0.30, arm_l=(30, 22, 40), arm_r=(30, 22, 40),
                head_tilt=8 + b * 5, expr="closed")


GAZE_YAWS = (0, 30, 60)
GAZE_PITCHES = (28, 0, -24)


def gaze(i, n):                      # standing, head turned toward the cursor (yaw x pitch grid)
    yaw, pitch = GAZE_YAWS[i // 3], GAZE_PITCHES[i % 3]
    return Pose(arm_l=(4, 14, 18), arm_r=(4, 14, 18), head_yaw=yaw, head_pitch=pitch)


def angry(i, n):                     # web torn: fists clenched down, stomping, angry eyes
    w = math.sin(2 * math.pi * i / n)
    return Pose(arm_l=(-15, 28, 75), arm_r=(-15, 28, 75), hip_l=6 + w * 14, knee_l=10 + max(0, w) * 30,
                hip_r=-4, knee_r=6, lean=6, head_pitch=-6, head_tilt=w * 4, expr="angry")


def sitgaze(i, n):                   # sitting calmly, head turned toward the cursor / an app
    yaw, pitch = GAZE_YAWS[i // 3], GAZE_PITCHES[i % 3]
    return Pose(hip_l=86, knee_l=40, hip_r=82, knee_r=46, bob=-0.30, arm_l=(30, 22, 40), arm_r=(30, 22, 40),
                head_yaw=yaw, head_pitch=pitch)


def sleep(i, n):
    return Pose(arm_l=(0, 8, 10), arm_r=(0, 8, 10), head_tilt=6)


ANIMS = {
    # name: (pose fn, frames, views)
    "walk": (walk, 10, [0, 45, 90, 135, 180]),
    "idle": (idle, 4, [0, 45, 90]),
    "wave": (wave, 6, [0, 30]),
    "hang": (hang, 4, [0, 30]),
    "upside": (upside, 4, [0]),
    "shoot": (shoot, 1, [30, 0]),
    "lounge": (lounge, 2, [70]),
    "sleep": (sleep, 1, [0]),
    "tumble": (lambda i, n: tumble(i, n), 4, [0]),
    "dizzy": (lambda i, n: dizzy(i, n), 2, [0]),
    "sit": (sit, 2, [0]),
    "angry": (angry, 2, [0]),
    "gaze": (gaze, 9, [0]),
    "sitgaze": (sitgaze, 9, [0]),
    "nap": (nap, 2, [0]),
    "hamsleep": (hamsleep, 2, [70]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--only", default="", help="comma list: render just these, merge into meta.json")
    args = ap.parse_args()
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication(sys.argv[:1] + ["-platform", "offscreen"])
    os.makedirs(OUT, exist_ok=True)
    if args.preview:
        frames = [("idle", idle(0, 4), 0), ("walk", walk(2, 10), 45), ("walk", walk(7, 10), 90),
                  ("wave", wave(1, 6), 0), ("walk", walk(4, 10), 180)]
        imgs = []
        t0 = time.time()
        for name, pose, yaw in frames:
            img, _ = render(pose, yaw, size=(220, 280), ss=args.ss)
            imgs.append(img)
        sheet = np.concatenate(imgs, axis=1)
        bg = np.zeros_like(sheet)
        bg[..., :3] = (214, 220, 230)
        bg[..., 3] = 255
        a = sheet[..., 3:4] / 255.0
        bg[..., :3] = (sheet[..., :3] * a + bg[..., :3] * (1 - a)).astype(np.uint8)
        save_png(os.path.join(OUT, "_preview.png"), bg)
        print(f"preview done in {time.time() - t0:.1f}s", flush=True)
        return
    meta = {"size": [150, 190], "anims": {}}
    only = [x for x in args.only.split(",") if x]
    if only and os.path.exists(os.path.join(OUT, "meta.json")):
        with open(os.path.join(OUT, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
    t0 = time.time()
    for name, (fn, n, views) in ANIMS.items():
        if only and name not in only:
            continue
        meta["anims"][name] = {"frames": n, "views": views}
        for v in views:
            for i in range(n):
                img, pts = render(fn(i, n), v, ss=args.ss)
                fname = f"{name}_{v}_{i}.png"
                save_png(os.path.join(OUT, fname), img)
                meta["anims"][name].setdefault("points", {}).setdefault(str(v), []).append(pts)
        print(f"{name}: {n} frames x {len(views)} views ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    print("done", flush=True)


if __name__ == "__main__":
    main()
