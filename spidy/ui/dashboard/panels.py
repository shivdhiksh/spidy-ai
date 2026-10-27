"""The dashboard panels in the minimal "obsidian" style. Each paints itself in reference pixels."""

from __future__ import annotations

import datetime
import math
from collections import deque

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import QLineEdit

from spidy.ui.dashboard import art, palette as pal
from spidy.ui.dashboard.base import RefWidget, text
from spidy.ui.dashboard.icons import draw_icon
from spidy.ui.dashboard.sphere import DotSphere

LEFT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
CENTER = Qt.AlignmentFlag.AlignCenter
LIGHT = QFont.Weight.Light
MEDIUM = QFont.Weight.Medium


SPHERE = DotSphere(1000)
MINI_SPHERE = DotSphere(420)


def caps(p: QPainter, rect: QRectF, s: str, px: float, color: QColor, spacing: float = 2.4,
         align=LEFT, weight=MEDIUM) -> None:
    text(p, rect, s.upper(), pal.font(px, weight, spacing), color, align)


# ── Left rail ───────────────────────────────────────────────────────────────

NAV_ITEMS = [
    ("Home", "home"), ("Chat", "chat"), ("Voice", "mic"), ("Browser", "globe"), ("Files", "folder"),
    ("Applications", "apps"), ("Automation", "automation"), ("Code Assistant", "code"),
    ("Screen Vision", "vision"), ("Memory", "memory"), ("Settings", "settings"),
]
NAV_LABELS = {"Applications": "Apps", "Automation": "Automate", "Code Assistant": "Code", "Screen Vision": "Vision"}


class NavRail(RefWidget):
    navigate = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(240, 880, parent)
        self.current, self._hover = "Home", -1

    def _row(self, i: int) -> QRectF:
        return QRectF(8, 150 + i * 62, 224, 54)

    def mouseMoveEvent(self, e) -> None:
        pos = self.ref_pos(e)
        idx = next((i for i in range(len(NAV_ITEMS)) if self._row(i).contains(pos)), -1)
        if idx != self._hover:
            self._hover = idx
            self.setCursor(Qt.CursorShape.PointingHandCursor if idx >= 0 else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, _e) -> None:
        self._hover = -1
        self.update()

    def mousePressEvent(self, e) -> None:
        pos = self.ref_pos(e)
        for i, (name, _icon) in enumerate(NAV_ITEMS):
            if self._row(i).contains(pos):
                self.current = name
                self.navigate.emit(name)
                self.update()
                return

    def paintEvent(self, _e) -> None:
        p = self.begin()
        MINI_SPHERE.paint(p, 62, 68, 24, phase=0.9, level=0.45, rings=False)
        p.setPen(QPen(QColor(64, 204, 232, 150), 1.2))
        p.drawEllipse(QPointF(62, 68), 33, 33)
        p.setPen(QPen(pal.HAIR, 1))
        p.drawLine(QPointF(28, 128), QPointF(212, 128))
        for i, (name, icon) in enumerate(NAV_ITEMS):
            row = self._row(i)
            sel, hov = name == self.current, i == self._hover
            col = pal.RED_BRIGHT if sel else (pal.TEXT if hov else pal.MUTED)
            if hov and not sel:
                p.setBrush(QBrush(QColor(255, 255, 255, 10)))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(row, 12, 12)
            draw_icon(p, icon, QRectF(30, row.top() + 15, 24, 24), col, 1.5)
            caps(p, QRectF(74, row.top(), 130, row.height()), NAV_LABELS.get(name, name), 13, col, 2.6)
            if sel:
                p.setBrush(QBrush(pal.RED_BRIGHT))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QPointF(214, row.center().y()), 3.4, 3.4)


# ── Header ──────────────────────────────────────────────────────────────────

class HeaderBar(RefWidget):
    """Title, live status line, weather and clock."""

    def __init__(self, city: str = "", parent=None) -> None:
        super().__init__(1130, 100, parent)
        self.message = "Ready for your command…"
        self.city = city
        self.weather: tuple[str, str] | None = None
        self._t = QTimer(self)
        self._t.setInterval(5000)
        self._t.timeout.connect(self.update)
        self._t.start()

    def set_message(self, msg: str) -> None:
        self.message = msg
        self.update()

    def set_weather(self, data: tuple[str, str] | None) -> None:
        if data:
            self.weather = data
            self.update()

    def paintEvent(self, _e) -> None:
        p = self.begin()
        text(p, QRectF(0, 0, 560, 50), "SPIDY", pal.font(34, LIGHT, 24), pal.TEXT, LEFT)
        caps(p, QRectF(2, 52, 640, 22), "Your desktop companion. Ask. Act. Done.", 11.5, pal.MUTED, 3.0)
        p.setBrush(QBrush(pal.RED_BRIGHT))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(6, 87), 3.6, 3.6)
        text(p, QRectF(20, 76, 700, 24), self.message, pal.font(15), pal.SOFT, LEFT)
        now = datetime.datetime.now()
        text(p, QRectF(800, 0, 330, 44), now.strftime("%I:%M %p"), pal.font(30, LIGHT, 1.0), pal.TEXT, RIGHT)
        p.setBrush(QBrush(pal.RED_BRIGHT))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(852, 22), 3.4, 3.4)
        caps(p, QRectF(862, 8, 110, 28), "Live", 11.5, pal.SOFT, 3.0)
        if self.weather:
            text(p, QRectF(700, 48, 430, 24), f"{self.weather[0]}  ·  {self.weather[1]}  ·  {self.city}",
                 pal.font(13.5), pal.MUTED, RIGHT)
        else:
            text(p, QRectF(700, 48, 430, 24), now.strftime("%A, %B %d"), pal.font(13.5), pal.MUTED, RIGHT)


class WindowButtons(RefWidget):
    minimize = Signal()
    close = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(64, 26, parent)
        self._hover = -1

    def mouseMoveEvent(self, e) -> None:
        self._hover = 0 if self.ref_pos(e).x() < 32 else 1
        self.update()

    def leaveEvent(self, _e) -> None:
        self._hover = -1
        self.update()

    def mousePressEvent(self, e) -> None:
        (self.minimize if self.ref_pos(e).x() < 32 else self.close).emit()

    def paintEvent(self, _e) -> None:
        p = self.begin()
        for i, name in enumerate(("minimize", "close")):
            col = pal.RED_BRIGHT if (i == 1 and self._hover == 1) else (pal.TEXT if self._hover == i else pal.FAINT)
            draw_icon(p, name, QRectF(i * 32 + 4, 3, 20, 20), col, 1.8)


# ── Hero: the 3D spider face ────────────────────────────────────────────────

STATE_TEXT = {
    "idle": ("Hey Spidy", "Say “Hey Spidy”"),
    "wake_ready": ("Hey Spidy", "Say “Hey Spidy”"),
    "listening": ("Hey Spidy", "I'm listening…"),
    "thinking": ("Spidy", "Thinking…"),
    "speaking": ("Spidy", "Speaking…"),
    "working": ("Spidy", "Working on it…"),
    "error": ("Spidy", "Something went wrong"),
}
_BASE_LEVEL = {"idle": 0.34, "wake_ready": 0.36, "listening": 0.86, "thinking": 0.62, "speaking": 0.8,
               "working": 0.7, "error": 0.9}


class HeroHead(RefWidget):
    """
    Glossy spider face. Eyes brighten with Spidy's state/voice level; the head tilts toward the cursor.

    The expensive vector art is rendered once into cached layers; each frame only blends the dim and
    bright lens layers (and applies a perspective tilt while the mouse moves), so idle cost stays low.
    """

    HEAD_Y = 177.0
    HEAD_SCALE = 0.78

    def __init__(self, parent=None) -> None:
        super().__init__(860, 410, parent)
        self.state = "idle"
        self.level = 0.0                  # live voice amplitude (0..1)
        self._phase = 0.0
        self._tilt = [0.0, 0.0]
        self._target = [0.0, 0.0]
        self._layers: dict[tuple, dict[str, QPixmap]] = {}
        self._t = QTimer(self)
        self._t.setInterval(90)
        self._t.timeout.connect(self._tick)

    def set_state(self, state: str) -> None:
        self.state = state if state in _BASE_LEVEL else "idle"
        self.update()

    def set_voice_level(self, level: float) -> None:
        self.level = max(0.0, min(1.0, level))

    def set_animating(self, on: bool) -> None:
        self._t.start() if on else self._t.stop()

    def mouseMoveEvent(self, e) -> None:
        pos = self.ref_pos(e)
        self._target = [(pos.x() / self.ref_w - 0.5) * 2 * 9.0, -(pos.y() / self.ref_h - 0.5) * 2 * 6.0]

    def leaveEvent(self, _e) -> None:
        self._target = [0.0, 0.0]

    def _busy(self) -> bool:
        moving = any(abs(self._target[i] - self._tilt[i]) > 0.05 for i in (0, 1))
        return moving or self.state in ("listening", "speaking", "thinking", "working", "error")

    def _tick(self) -> None:
        self._phase += 0.07 if self._busy() else 0.1
        for i in (0, 1):
            self._tilt[i] += (self._target[i] - self._tilt[i]) * 0.14
        self._t.setInterval(40 if self._busy() else 100)   # ~25 fps when active, ~10 fps when idle
        self.update()

    def current_level(self) -> float:
        base = _BASE_LEVEL.get(self.state, 0.34)
        wobble = 0.08 * math.sin(self._phase * 1.4)
        if self.state == "thinking":
            wobble = 0.22 * math.sin(self._phase * 3.0)
        return max(0.05, min(1.0, base + wobble + 0.3 * self.level * (self.state in ("listening", "speaking"))))

    def _get_layers(self) -> dict[str, QPixmap]:
        dpr = self.devicePixelRatioF()
        key = (round(self.k, 3), round(dpr, 2), self.state == "error")
        if key not in self._layers:
            self._layers.clear()
            red = art.head_red("error" if key[2] else "idle")
            size_w, size_h = int(self.ref_w * self.k * dpr) + 1, int(self.ref_h * self.k * dpr) + 1

            def layer(draw) -> QPixmap:
                pm = QPixmap(size_w, size_h)
                pm.setDevicePixelRatio(dpr)
                pm.fill(Qt.GlobalColor.transparent)
                q = QPainter(pm)
                q.setRenderHint(QPainter.RenderHint.Antialiasing)
                q.scale(self.k, self.k)
                q.translate(self.ref_w / 2, self.HEAD_Y)
                q.scale(self.HEAD_SCALE, self.HEAD_SCALE)
                draw(q)
                q.end()
                return pm

            self._layers[key] = {
                "glow": layer(lambda q: art.draw_head_glow(q, red)),
                "body": layer(lambda q: art.draw_head_body(q, red)),
                "dim": layer(lambda q: art.draw_head_lenses(q, 0.0)),
                "bright": layer(lambda q: art.draw_head_lenses(q, 1.0)),
                "tint": layer(lambda q: art.draw_head_tint(q, red)),
            }
        return self._layers[key]

    SPHERE_R = 134.0          # radius of the dot sphere in reference pixels (rings reach ~1.46x)

    def _paint_sphere(self) -> None:
        k = self.k
        p = QPainter(self)
        SPHERE.paint(p, self.ref_w / 2 * k, self.ref_h / 2 * k, self.SPHERE_R * k, self._phase, self.current_level(),
                     tilt=(self._tilt[0], self._tilt[1]), state=self.state, k=k,
                     grid_rect=QRectF(0, 0, self.ref_w * k, self.ref_h * k))

    def paintEvent(self, _e) -> None:
        self._paint_sphere()
        return
        layers = self._get_layers()
        level = self.current_level()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if abs(self._tilt[0]) > 0.05 or abs(self._tilt[1]) > 0.05:      # perspective tilt around the head centre
            cx, cy = self.ref_w / 2 * self.k, self.HEAD_Y * self.k
            t = QTransform()
            t.translate(cx, cy)
            t.rotate(self._tilt[0], Qt.Axis.YAxis)
            t.rotate(self._tilt[1], Qt.Axis.XAxis)
            t.translate(-cx, -cy)
            p.setTransform(t)
        p.setOpacity(0.55 + 0.45 * level)
        p.drawPixmap(0, 0, layers["glow"])
        p.setOpacity(1.0)
        p.drawPixmap(0, 0, layers["body"])
        p.drawPixmap(0, 0, layers["dim"])
        p.setOpacity(max(0.0, min(1.0, (level - 0.30) / 0.70)))
        p.drawPixmap(0, 0, layers["bright"])
        if self.state in ("thinking", "working"):
            p.setOpacity(0.2 + 0.2 * math.sin(self._phase * 3) ** 2)
            p.drawPixmap(0, 0, layers["tint"])


# ── Voice controls (waveform, status, mic, modes) ───────────────────────────

class VoiceControls(RefWidget):
    mic_clicked = Signal()
    mode_changed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(840, 200, parent)
        self.state, self.mode = "idle", "voice"
        self.level = 0.0
        self._target = 0.0
        self._phase = 0.0
        self._hover = ""
        self._t = QTimer(self)
        self._t.setInterval(45)
        self._t.timeout.connect(self._tick)

    def _active(self) -> bool:
        return self.state in ("listening", "speaking", "thinking", "working")

    def set_state(self, state: str) -> None:
        self.state = state if state in STATE_TEXT else "idle"
        if self._active() and self.isVisible():
            self._t.start()
        elif not self._active():
            self._target = 0.0
        self.update()

    def set_level(self, level: float) -> None:
        self._target = max(0.0, min(1.0, level))

    def set_animating(self, on: bool) -> None:
        if on and self._active():
            self._t.start()
        elif not on:
            self._t.stop()

    def _tick(self) -> None:
        self._phase += 0.35
        if self.state == "listening":
            self._target = max(self._target * 0.9, 0.25 + 0.15 * math.sin(self._phase * 0.5))
        elif self.state in ("speaking", "working", "thinking"):
            self._target = 0.35 + 0.25 * abs(math.sin(self._phase * 0.4))
        self.level += (self._target - self.level) * 0.35
        if not self._active() and self.level < 0.01:
            self._t.stop()
        self.update()

    def _mic_rect(self) -> QRectF:
        return QRectF(420 - 32, 160 - 32, 64, 64)

    def _pills(self) -> dict[str, QRectF]:
        return {"voice": QRectF(200, 138, 160, 44), "text": QRectF(480, 138, 160, 44)}

    def mouseMoveEvent(self, e) -> None:
        pos = self.ref_pos(e)
        hover = "mic" if self._mic_rect().contains(pos) else next(
            (k for k, r in self._pills().items() if r.contains(pos)), "")
        if hover != self._hover:
            self._hover = hover
            self.setCursor(Qt.CursorShape.PointingHandCursor if hover else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, _e) -> None:
        self._hover = ""
        self.update()

    def mousePressEvent(self, e) -> None:
        pos = self.ref_pos(e)
        if self._mic_rect().contains(pos):
            self.mic_clicked.emit()
            return
        for key, r in self._pills().items():
            if r.contains(pos):
                self.mode = key
                self.mode_changed.emit(key)
                self.update()

    def paintEvent(self, _e) -> None:
        p = self.begin()
        small, big = STATE_TEXT[self.state]
        # waveform: thin red bars; a quiet dotted line when idle
        n, x0, x1, cy = 56, 150.0, 690.0, 24.0
        for i in range(n):
            t = i / (n - 1)
            x = x0 + (x1 - x0) * t
            if self.mode == "text":
                break
            if self._active():
                env = math.exp(-((t - 0.5) ** 2) / 0.07)
                wob = abs(0.55 + 0.45 * math.sin(self._phase * 1.3 + i * 0.9) * math.cos(self._phase * 0.7 + i * 0.37))
                amp = 1.5 + 20 * self.level * env * wob
                p.setPen(QPen(pal.with_alpha(pal.RED_BRIGHT, 230), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            else:
                amp = 1.2
                p.setPen(QPen(pal.with_alpha(pal.FAINT, 160), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(x, cy - amp), QPointF(x, cy + amp))
        caps(p, QRectF(0, 56, 840, 20), small, 12, pal.MUTED, 4.0, CENTER)
        text(p, QRectF(0, 76, 840, 44), big, pal.font(30, LIGHT), pal.TEXT, CENTER)

        for key, r in self._pills().items():
            sel, hov = self.mode == key, self._hover == key
            p.setBrush(QBrush(QColor(255, 255, 255, 18 if sel else (10 if hov else 0))))
            p.setPen(QPen(pal.RED_BRIGHT if sel else pal.HAIR, 1.2))
            p.drawRoundedRect(r, 22, 22)
            draw_icon(p, "mic" if key == "voice" else "chat", QRectF(r.left() + 18, r.center().y() - 10, 20, 20),
                      pal.TEXT if sel else pal.MUTED, 1.5)
            caps(p, QRectF(r.left() + 46, r.top(), r.width() - 50, r.height()),
                 "Voice mode" if key == "voice" else "Text mode", 12, pal.TEXT if sel else pal.MUTED, 2.2)

        c = QPointF(420, 160)
        on = self.state == "listening"
        p.setBrush(QBrush(QColor(255, 255, 255, 14 if self._hover == "mic" else 8)))
        p.setPen(QPen(pal.RED_BRIGHT if on else pal.HAIR_STRONG, 2.0 if on else 1.4))
        p.drawEllipse(c, 30, 30)
        if on:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(pal.with_alpha(pal.RED_BRIGHT, 70), 1.2))
            p.drawEllipse(c, 38 + 3 * math.sin(self._phase), 38 + 3 * math.sin(self._phase))
        draw_icon(p, "mic", QRectF(c.x() - 15, c.y() - 15, 30, 30), pal.TEXT if on else pal.SOFT, 1.7)


class CommandInput(QLineEdit):
    """Text-mode command line (replaces the waveform)."""

    submitted = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrame(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setPlaceholderText("Type a command and press Enter")
        self.returnPressed.connect(self._send)
        self.hide()

    def apply_scale(self, k: float) -> None:
        self.setStyleSheet(
            f"QLineEdit {{ background: rgba(255,255,255,0.05); color: #eceef4; font-family: 'Segoe UI'; "
            f"font-size: {max(10, int(19 * k))}px; border: 1px solid rgba(255,255,255,0.18); "
            f"border-radius: {max(6, int(20 * k))}px; padding: 0 {int(18 * k)}px; selection-background-color: #e2242c; }}"
            f"QLineEdit:focus {{ border: 1px solid #ff4a52; }}")

    def _send(self) -> None:
        msg = self.text().strip()
        if msg:
            self.submitted.emit(msg)
            self.clear()


# ── Dock row ────────────────────────────────────────────────────────────────

DOCK = [("globe", "Open Browser", "open chrome"), ("code", "Open VS Code", "open vs code"),
        ("folder", "Open Files", "open file explorer"), ("monitor", "System Info", "show system info"),
        ("screenshot", "Screenshot", "take a screenshot"), ("automation", "Automation", "__page__:Automation")]
DOCK_X = [15 + i * 135 for i in range(6)]


class DockRow(RefWidget):
    command = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(840, 74, parent)
        self._hover = -1

    def _slot(self, i: int) -> QRectF:
        return QRectF(DOCK_X[i], 2, 120, 70)

    def mouseMoveEvent(self, e) -> None:
        pos = self.ref_pos(e)
        idx = next((i for i in range(6) if self._slot(i).contains(pos)), -1)
        if idx != self._hover:
            self._hover = idx
            self.setCursor(Qt.CursorShape.PointingHandCursor if idx >= 0 else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, _e) -> None:
        self._hover = -1
        self.update()

    def mousePressEvent(self, e) -> None:
        pos = self.ref_pos(e)
        for i in range(6):
            if self._slot(i).contains(pos):
                self.command.emit(DOCK[i][2])

    def paintEvent(self, _e) -> None:
        p = self.begin()
        for i, (icon, label, _cmd) in enumerate(DOCK):
            r = self._slot(i)
            hov = i == self._hover
            if hov:
                p.setBrush(QBrush(QColor(255, 255, 255, 12)))
                p.setPen(QPen(pal.HAIR, 1))
                p.drawRoundedRect(r, 14, 14)
            col = pal.TEXT if hov else pal.SOFT
            draw_icon(p, icon, QRectF(r.center().x() - 14, r.top() + 9, 28, 28), col, 1.5)
            caps(p, QRectF(r.left(), r.top() + 42, r.width(), 20), label, 11, pal.TEXT if hov else pal.MUTED, 1.8, CENTER)


# ── Suggestions / task list ─────────────────────────────────────────────────

SUGGESTIONS = [
    ("nodes", "Open YouTube and play a video", "open youtube and play a video"),
    ("doc_plus", "Create a Python program", "__page__:Code Assistant"),
    ("search", "Find files on my computer", "find files on my computer"),
    ("code", "Open VS Code", "open vs code"),
    ("chart", "Show me system performance", "show system info"),
    ("bolt", "Automate a task for me", "__page__:Automation"),
]


class SuggestionList(RefWidget):
    command = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(310, 330, parent)
        self._hover = -1
        self.goal = ""
        self.steps: list[tuple[str, str]] = []

    def set_task(self, goal: str, steps: list[tuple[str, str]]) -> None:
        self.goal, self.steps = goal, steps
        self.update()

    def clear_task(self) -> None:
        self.goal, self.steps = "", []
        self.update()

    def _row(self, i: int) -> QRectF:
        return QRectF(0, 40 + i * 48, 310, 46)

    def mouseMoveEvent(self, e) -> None:
        if self.goal:
            return
        pos = self.ref_pos(e)
        idx = next((i for i in range(len(SUGGESTIONS)) if self._row(i).contains(pos)), -1)
        if idx != self._hover:
            self._hover = idx
            self.setCursor(Qt.CursorShape.PointingHandCursor if idx >= 0 else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, _e) -> None:
        self._hover = -1
        self.update()

    def mousePressEvent(self, e) -> None:
        if self.goal:
            return
        pos = self.ref_pos(e)
        for i, (_i, _l, cmd) in enumerate(SUGGESTIONS):
            if self._row(i).contains(pos):
                self.command.emit(cmd)

    def paintEvent(self, _e) -> None:
        p = self.begin()
        if self.goal:
            caps(p, QRectF(0, 0, 310, 26), "Working on it", 12, pal.RED_BRIGHT, 3.2)
            text(p, QRectF(0, 30, 310, 30), self.goal, pal.font(15), pal.SOFT, LEFT)
            for i, (desc, status) in enumerate(self.steps[:6]):
                y = 74 + i * 40
                col = {"done": pal.GREEN, "running": pal.RED_BRIGHT, "failed": pal.RED}.get(status, pal.FAINT)
                p.setBrush(QBrush(col) if status != "pending" else Qt.BrushStyle.NoBrush)
                p.setPen(QPen(col, 1.6))
                p.drawEllipse(QPointF(8, y + 14), 5, 5)
                text(p, QRectF(28, y, 282, 28), desc, pal.font(15), pal.TEXT if status != "pending" else pal.MUTED, LEFT)
            return
        caps(p, QRectF(0, 0, 310, 26), "How can I help", 12, pal.MUTED, 3.2)
        for i, (icon, label, _cmd) in enumerate(SUGGESTIONS):
            row = self._row(i)
            hov = i == self._hover
            p.setPen(QPen(pal.HAIR, 1))
            p.drawLine(QPointF(0, row.top()), QPointF(310, row.top()))
            draw_icon(p, icon, QRectF(2, row.top() + 12, 22, 22), pal.RED_BRIGHT if hov else pal.MUTED, 1.5)
            text(p, QRectF(40, row.top(), 236, row.height()), label, pal.font(15), pal.TEXT if hov else pal.SOFT, LEFT)
            draw_icon(p, "chevron", QRectF(282, row.top() + 13, 20, 20), pal.RED_BRIGHT if hov else pal.FAINT, 1.6)
        p.setPen(QPen(pal.HAIR, 1))
        p.drawLine(QPointF(0, self._row(len(SUGGESTIONS) - 1).bottom()), QPointF(310, self._row(len(SUGGESTIONS) - 1).bottom()))


# ── Quick actions ───────────────────────────────────────────────────────────

QUICK = [("power", "Shutdown", "shut down the computer", pal.RED_BRIGHT),
         ("restart", "Restart", "restart the computer", pal.SOFT),
         ("lock", "Lock", "lock my computer", pal.SOFT),
         ("moon", "Sleep", "put my computer to sleep", pal.SOFT)]


class QuickRow(RefWidget):
    command = Signal(str, str)   # command text, human label (for the confirmation)

    def __init__(self, parent=None) -> None:
        super().__init__(310, 150, parent)
        self._hover = -1

    def _rect(self, i: int) -> QRectF:
        return QRectF(i * 80, 36, 70, 100)

    def mouseMoveEvent(self, e) -> None:
        pos = self.ref_pos(e)
        idx = next((i for i in range(4) if self._rect(i).contains(pos)), -1)
        if idx != self._hover:
            self._hover = idx
            self.setCursor(Qt.CursorShape.PointingHandCursor if idx >= 0 else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, _e) -> None:
        self._hover = -1
        self.update()

    def mousePressEvent(self, e) -> None:
        pos = self.ref_pos(e)
        for i, (_ic, label, cmd, _c) in enumerate(QUICK):
            if self._rect(i).contains(pos):
                self.command.emit(cmd, label)

    def paintEvent(self, _e) -> None:
        p = self.begin()
        caps(p, QRectF(0, 0, 310, 26), "Quick actions", 12, pal.MUTED, 3.2)
        for i, (icon, label, _cmd, col) in enumerate(QUICK):
            r = self._rect(i)
            hov = i == self._hover
            c = QPointF(r.center().x(), r.top() + 32)
            p.setBrush(QBrush(QColor(255, 255, 255, 14 if hov else 6)))
            p.setPen(QPen(pal.with_alpha(col, 200) if hov else pal.HAIR_STRONG, 1.3))
            p.drawEllipse(c, 28, 28)
            draw_icon(p, icon, QRectF(c.x() - 13, c.y() - 13, 26, 26), col if hov or col == pal.RED_BRIGHT else pal.SOFT, 1.6)
            caps(p, QRectF(r.left() - 6, r.top() + 70, r.width() + 12, 22), label, 10.5, pal.TEXT if hov else pal.MUTED, 1.6, CENTER)


# ── Recent activity ─────────────────────────────────────────────────────────

class RecentCard(RefWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(310, 150, parent)
        self.items: deque[tuple[str, str]] = deque(maxlen=3)

    def add(self, role: str, msg: str) -> None:
        self.items.append((role, " ".join(msg.split())))
        self.update()

    def paintEvent(self, _e) -> None:
        p = self.begin()
        caps(p, QRectF(0, 0, 310, 26), "Recent", 12, pal.MUTED, 3.2)
        if not self.items:
            text(p, QRectF(0, 38, 310, 30), "Nothing yet. Say “Hey Spidy”.", pal.font(14), pal.FAINT, LEFT)
        for i, (role, msg) in enumerate(reversed(self.items)):
            y = 36 + i * 38
            user = role.lower() in ("user", "you")
            p.setBrush(QBrush(pal.SOFT if user else pal.RED_BRIGHT))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(5, y + 14), 3, 3)
            text(p, QRectF(18, y, 292, 28), msg, pal.font(14.5), pal.SOFT if user else pal.TEXT, LEFT)


# ── Stats row ───────────────────────────────────────────────────────────────

class StatsRow(RefWidget):
    """CPU / GPU / RAM / VRAM with live sparklines, plus network state."""

    COLS = [("cpu", "CPU"), ("gpu", "GPU"), ("ram", "RAM"), ("vram", "VRAM")]

    def __init__(self, parent=None) -> None:
        super().__init__(1190, 140, parent)
        self.vals: dict[str, float | None] = {k: None for k, _ in self.COLS}
        self.hist: dict[str, deque] = {k: deque(maxlen=40) for k, _ in self.COLS}
        self.specs: dict = {}
        self.online, self.host = True, ""
        self._phase = 0.0
        self._t = QTimer(self)
        self._t.setInterval(100)
        self._t.timeout.connect(self._tick)

    def set_animating(self, on: bool) -> None:
        self._t.start() if on else self._t.stop()

    def _tick(self) -> None:
        self._phase = (self._phase + 5.0) % 360
        self.update(QRectF(952 * self.k, 0, 238 * self.k, 140 * self.k).toAlignedRect())

    def set_data(self, d: dict) -> None:
        for key, _ in self.COLS:
            v = d.get(key)
            self.vals[key] = v
            if v is not None:
                self.hist[key].append(float(v))
        self.specs = d.get("specs") or self.specs
        self.online, self.host = bool(d.get("online", True)), d.get("host", self.host)
        self.update()

    def _spec(self, key: str) -> str:
        s = self.specs
        if key == "cpu":
            return " ".join(x for x in s.get("cpu", ("", "")) if x)
        if key == "gpu":
            return s.get("gpu", ("", ""))[0]
        if key == "ram":
            return " ".join(x for x in s.get("ram", ("", "")) if x)
        return s.get("gpu", ("", ""))[1]

    def paintEvent(self, _e) -> None:
        p = self.begin()
        col_w = 238.0
        for i, (key, label) in enumerate(self.COLS):
            x = i * col_w
            if i:
                p.setPen(QPen(pal.HAIR, 1))
                p.drawLine(QPointF(x, 6), QPointF(x, 130))
            v = self.vals[key]
            caps(p, QRectF(x + 22, 6, 190, 22), label, 12, pal.MUTED, 3.0)
            hot = v is not None and v >= 85
            value = "—" if v is None else f"{v:.0f}"
            vf = pal.font(52, LIGHT)
            text(p, QRectF(x + 22, 26, 130, 62), value, vf, pal.RED_BRIGHT if hot else pal.TEXT, LEFT)
            if v is not None:
                w = QFontMetricsF(vf).horizontalAdvance(value)
                text(p, QRectF(x + 26 + w, 44, 40, 30), "%", pal.font(20, LIGHT), pal.MUTED, LEFT)
            data = list(self.hist[key])
            if len(data) >= 2:
                left, top, w_, h_ = x + 22, 94, 190.0, 22.0
                lo, hi = min(data), max(data)
                span = max(hi - lo, 10.0)              # auto-scale so small movements are visible
                pts = [QPointF(left + w_ * j / (len(data) - 1), top + h_ * (1 - (d - lo) / span)) for j, d in enumerate(data)]
                area = QPainterPath(QPointF(pts[0].x(), top + h_))
                for pt in pts:
                    area.lineTo(pt)
                area.lineTo(pts[-1].x(), top + h_)
                area.closeSubpath()
                grad = QLinearGradient(0, top, 0, top + h_)
                grad.setColorAt(0, pal.with_alpha(pal.RED_BRIGHT, 70))
                grad.setColorAt(1, pal.with_alpha(pal.RED_BRIGHT, 0))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(grad))
                p.drawPath(area)
                line = QPainterPath(pts[0])
                for pt in pts[1:]:
                    line.lineTo(pt)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(pal.RED_BRIGHT, 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                p.drawPath(line)
            text(p, QRectF(x + 22, 118, 200, 20), self._spec(key), pal.font(12.5), pal.FAINT, LEFT)

        # network: radar
        x = 4 * col_w
        p.setPen(QPen(pal.HAIR, 1))
        p.drawLine(QPointF(x, 6), QPointF(x, 130))
        caps(p, QRectF(x + 22, 6, 190, 22), "Network", 12, pal.MUTED, 3.0)
        text(p, QRectF(x + 22, 34, 120, 40), "Online" if self.online else "Offline", pal.font(24, LIGHT),
             pal.TEXT if self.online else pal.RED_BRIGHT, LEFT)
        text(p, QRectF(x + 22, 76, 130, 22), self.host, pal.font(12.5), pal.FAINT, LEFT)
        c = QPointF(x + 176, 68)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for r, a in ((46, 70), (30, 90), (14, 110)):
            p.setPen(QPen(pal.with_alpha(pal.SOFT, a // 2), 1))
            p.drawEllipse(c, r, r)
        p.setPen(QPen(pal.HAIR, 1))
        p.drawLine(QPointF(c.x() - 50, c.y()), QPointF(c.x() + 50, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - 50), QPointF(c.x(), c.y() + 50))
        ang = math.radians(self._phase)
        p.setPen(QPen(pal.with_alpha(pal.RED_BRIGHT, 140), 1.2))
        p.drawLine(c, QPointF(c.x() + 46 * math.cos(ang), c.y() + 46 * math.sin(ang)))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(pal.RED_BRIGHT if self.online else pal.FAINT))
        p.drawEllipse(QPointF(c.x() + 16, c.y() - 12), 3.2, 3.2)
