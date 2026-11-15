"""
3D sprite frames for the desktop pet (rendered by scripts/render_pet_sprites.py into assets/pet/).

Each animation was rendered from a few camera angles ("views", degrees: 0 = facing you, 90 = facing
right, 180 = back). Angles in between, or on the other side, pick the nearest rendered view and mirror
it. meta.json stores where the hands / feet / head / hips are in every frame so webs, speech bubbles
and the blanket line up exactly with the 3D body.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainter, QPixmap

from spidy.logging.logger import get_logger

log = get_logger(__name__)

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "pet"


class SpriteBank:
    def __init__(self, folder: Path = ASSETS) -> None:
        self.folder = folder
        self.meta: dict = {}
        self._cache: dict[str, QPixmap] = {}
        try:
            self.meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 -- no sprites: the pet falls back to vector drawing
            log.info("[PET] no 3D sprites ({e}); using the drawn pet.", e=exc)

    @property
    def ok(self) -> bool:
        return bool(self.meta.get("anims"))

    def has(self, anim: str) -> bool:
        return anim in self.meta.get("anims", {})

    def frames(self, anim: str) -> int:
        return int(self.meta["anims"][anim]["frames"])

    def _pick_view(self, anim: str, yaw: float) -> tuple[int, bool]:
        views = self.meta["anims"][anim]["views"]
        yaw = ((yaw + 180) % 360) - 180
        best, best_d, mirror = views[0], 1e9, False
        for v in views:
            for cand, mir in ((v, False), (-v, True)):
                if mir and v in (0, 180):
                    continue
                d = abs(((yaw - cand + 180) % 360) - 180)
                if d < best_d:
                    best, best_d, mirror = v, d, mir
        return best, mirror

    def frame(self, anim: str, yaw: float, idx: int) -> tuple[QPixmap, dict[str, list[float]], bool]:
        view, mirror = self._pick_view(anim, yaw)
        n = self.frames(anim)
        idx %= n
        key = f"{anim}_{view}_{idx}"
        pm = self._cache.get(key)
        if pm is None:
            pm = QPixmap(str(self.folder / f"{key}.png"))
            self._cache[key] = pm
        pts = self.meta["anims"][anim]["points"][str(view)][idx]
        return pm, pts, mirror

    def blit(self, p: QPainter, anim: str, yaw: float, idx: int, anchor: str, at: QPointF,
             scale: float, rot: float = 0.0, opacity: float = 1.0, mirror_x: bool = False) -> dict[str, QPointF]:
        """Draw a frame with its ``anchor`` point placed at ``at``; returns every key point on screen."""
        pm, pts, mirror = self.frame(anim, yaw, idx)
        mirror = mirror ^ mirror_x
        ax, ay = pts[anchor]
        sx = -scale if mirror else scale
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setOpacity(p.opacity() * opacity)
        p.translate(at)
        p.rotate(rot)
        p.scale(sx, scale)
        p.translate(-ax, -ay)
        p.drawPixmap(0, 0, pm)
        p.restore()
        c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        out = {}
        for k, (x, y) in pts.items():
            dx, dy = (x - ax) * sx, (y - ay) * scale
            out[k] = QPointF(at.x() + dx * c - dy * s, at.y() + dx * s + dy * c)
        return out

    def height(self, anim: str = "idle") -> float:
        """Feet-to-head distance of the idle pose in sprite pixels (for sizing)."""
        pts = self.meta["anims"][anim]["points"]["0"][0]
        return pts["feet"][1] - pts["head"][1]
