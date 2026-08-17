"""
overlay.py -- Spidy HUD: Full-screen futuristic desktop interface (M17.2)
=========================================================================
Completely replaces the 800x600 split-column layout.

Architecture
-----------
- OverlayWindow          top-level frameless window (responsive HUD)
  ├─ _HUDHeader          top bar: date/time | Spidy status | sys status
  ├─ _HUDLeftPanel       system telemetry (CPU/RAM/GPU/etc.)
  ├─ SpidyCoreWidget     LARGE central AI core (the primary focus)
  ├─ _HUDRightPanel      task console + notifications
  ├─ HUDChatOverlay      floating conversation cards (over center)
  ├─ HUDConfirmation     amber confirmation overlay (over center)
  └─ _HUDFooter          [controls] | voice visualizer | status

The class surface (public methods/signals) is kept compatible with the
previous OverlayWindow so that app.py needs minimal changes.
UISignalBridge is kept in this module for backward compatibility.

UISignalBridge
--------------
Thread-safe: the async EventBus calls request_* from background threads.
Qt signals marshal the calls onto the Qt main thread.
"""

from __future__ import annotations

import ctypes
import datetime
import logging
import math
import platform
import sys
from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QEasingCurve, QPointF, QPropertyAnimation, QRectF, QSequentialAnimationGroup,
    Qt, QTimer, Signal,
)
from PySide6.QtGui import (
    QBrush, QColor, QFont, QGuiApplication, QLinearGradient,
    QPainter, QPainterPath, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QHBoxLayout, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

from spidy.ui.widgets.hud_chat        import HUDChatOverlay
from spidy.ui.widgets.hud_confirmation import HUDConfirmation
from spidy.ui.widgets.hud_core        import SpidyCoreWidget
from spidy.ui.widgets.hud_task_console import HUDTaskConsole, TaskStep
from spidy.ui.widgets.hud_telemetry   import TelemetryPanel
from spidy.ui.widgets.hud_voice_bar   import HUDVoiceBar

if TYPE_CHECKING:
    from spidy.ui.themes.base import Theme

log = logging.getLogger(__name__)


# ── UISignalBridge ─────────────────────────────────────────────────────────

class UISignalBridge(QWidget):
    """
    Thread-safe bridge from async EventBus → Qt main thread.

    All ``request_*`` methods are safe to call from any thread.
    """

    # Original signals (kept for backward compat with app.py)
    state_change_requested   = Signal(str)
    message_received         = Signal(str, str)       # role, text
    notification_requested   = Signal(str, str, str)  # title, msg, level
    waveform_data_received   = Signal(object)         # list[float]
    theme_change_requested   = Signal(object)         # Theme
    task_progress_received   = Signal(str, object)    # goal, steps
    confirmation_show        = Signal(str, str, str, str)  # tid, gid, desc, prompt
    confirmation_hide        = Signal()

    def request_state_change(self, state: str) -> None:
        self.state_change_requested.emit(state)

    def request_message(self, role: str, text: str) -> None:
        self.message_received.emit(role, text)

    def request_notification(self, title: str, msg: str, level: str = "info") -> None:
        self.notification_requested.emit(title, msg, level)

    def request_waveform(self, amplitudes: list) -> None:
        self.waveform_data_received.emit(amplitudes)

    def request_task_progress(self, goal: str, steps: object) -> None:
        self.task_progress_received.emit(goal, steps)

    def request_confirmation_show(
        self, task_id: str, goal_id: str, description: str, prompt: str
    ) -> None:
        self.confirmation_show.emit(task_id, goal_id, description, prompt)

    def request_confirmation_hide(self) -> None:
        self.confirmation_hide.emit()


# ── HUD sub-panels ─────────────────────────────────────────────────────────

class _HUDInfoWidget(QWidget):
    """Tiny glass HUD module with title + value lines."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._title = title
        self._lines: list[tuple[str, str]] = []   # (label, value)
        self._c_primary  = QColor("#00E5FF")
        self._c_dim      = QColor("#004D5E")
        self._c_text     = QColor("#B2EBF2")
        self._c_bg       = QColor("#0A1929")

    def set_lines(self, lines: list[tuple[str, str]]) -> None:
        self._lines = list(lines)
        self.update()

    def apply_theme(self, theme: "Theme") -> None:
        c = theme.colors
        self._c_primary  = QColor(c.hud_primary)
        self._c_dim      = QColor(c.hud_dim)
        self._c_text     = QColor(c.hud_text)
        self._c_bg       = QColor(c.hud_grid)
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        if not p.isActive(): return
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        pad = 6

        # Glass bg
        bg = QColor(self._c_bg)
        bg.setAlphaF(0.72)
        border = QColor(self._c_primary)
        border.setAlphaF(0.35)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, w, h), 5, 5)
        p.setBrush(QBrush(bg))
        p.setPen(QPen(border, 0.8))
        p.drawPath(path)

        # Title — capped: 7pt min, 10pt max
        title_col = QColor(self._c_primary)
        title_col.setAlphaF(0.75)
        p.setPen(QPen(title_col))
        font_t = QFont("Consolas", max(7, min(10, int(h * 0.11))), QFont.Weight.Bold)
        font_t.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        p.setFont(font_t)
        p.drawText(QRectF(pad, pad, w - pad * 2, h * 0.22), Qt.AlignmentFlag.AlignLeft, self._title)

        # Lines
        if not self._lines:
            p.end(); return
        row_h = (h * 0.72) / len(self._lines)
        y = h * 0.26
        # Value font — capped: 7pt min, 11pt max
        font_v = QFont("Consolas", max(7, min(11, int(row_h * 0.45))))
        p.setFont(font_v)
        for label, val in self._lines:
            lbl_col = QColor(self._c_text)
            lbl_col.setAlphaF(0.55)
            p.setPen(QPen(lbl_col))
            p.drawText(QRectF(pad, y, w * 0.52 - pad, row_h), Qt.AlignmentFlag.AlignVCenter, label)
            val_col = QColor(self._c_primary)
            val_col.setAlphaF(0.80)
            p.setPen(QPen(val_col))
            p.drawText(QRectF(w * 0.52, y, w * 0.46, row_h),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, val)
            y += row_h
        p.end()


class _HUDHeader(QWidget):
    """Top bar: left datetime | center status | right sys info."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedHeight(52)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(8)

        self._tl = _HUDInfoWidget("SPIDY SYSTEM")
        self._tc = _HUDInfoWidget("SPIDY INTELLIGENCE")
        self._tr = _HUDInfoWidget("SPIDY NETWORK")

        for w in (self._tl, self._tc, self._tr):
            layout.addWidget(w, 1)

        self._state_label = self._tc  # kept for test compat

        # Clock timer
        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._update_clock)
        self._clock.start()
        self._update_clock()

    def set_state(self, state: str) -> None:
        _label_map = {
            "idle":      "SLEEPING",
            "wake_ready":"AWAKE",
            "listening": "LISTENING",
            "thinking":  "PROCESSING",
            "speaking":  "SPEAKING",
            "working":   "WORKING",
            "error":     "ERROR",
        }
        txt = _label_map.get(state, state.upper())
        self._tc.set_lines([("STATUS", txt), ("CORE", "ONLINE")])

    def apply_theme(self, theme: "Theme") -> None:
        for w in (self._tl, self._tc, self._tr):
            w.apply_theme(theme)

    def _update_clock(self) -> None:
        now = datetime.datetime.now()
        self._tl.set_lines([
            ("DATE", now.strftime("%d %b %Y")),
            ("TIME", now.strftime("%H:%M:%S")),
        ])
        self._tr.set_lines([
            ("SYS",  "ONLINE"),
            ("NET",  "ACTIVE"),
        ])


class _HUDLeftPanel(QWidget):
    """Left side: system telemetry."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        # Width is set dynamically by OverlayWindow._compute_side_widths()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._tele = TelemetryPanel(["cpu", "ram", "gpu", "vram", "disk"])
        layout.addWidget(self._tele, 3)

        self._mic_info = _HUDInfoWidget("MIC STATUS")
        self._mic_info.set_lines([("MIC", "READY"), ("WAKE", "HOT")])
        layout.addWidget(self._mic_info, 1)

    def apply_theme(self, theme: "Theme") -> None:
        self._tele.apply_theme(theme)
        self._mic_info.apply_theme(theme)

    def set_mic_state(self, listening: bool) -> None:
        self._mic_info.set_lines([
            ("MIC", "ACTIVE" if listening else "READY"),
            ("WAKE", "HOT"),
        ])


class _HUDRightPanel(QWidget):
    """Right side: task console + status widgets."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        # Width is set dynamically by OverlayWindow._compute_side_widths()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._agent_info = _HUDInfoWidget("SPIDY AGENT")
        self._agent_info.set_lines([("STATUS", "IDLE"), ("TASKS", "0")])
        layout.addWidget(self._agent_info, 1)

        self._task_console = HUDTaskConsole()
        layout.addWidget(self._task_console, 3)

        self._app_info = _HUDInfoWidget("ACTIVE APP")
        self._app_info.set_lines([("APP", "—"), ("WIN", "—")])
        layout.addWidget(self._app_info, 1)

    def apply_theme(self, theme: "Theme") -> None:
        self._agent_info.apply_theme(theme)
        self._task_console.apply_theme(theme)
        self._app_info.apply_theme(theme)

    def show_task_console(self, visible: bool) -> None:
        self._task_console.setVisible(visible)

    def update_task(self, goal: str, steps: list) -> None:
        self._task_console.update_progress(goal, steps)
        self._agent_info.set_lines([("STATUS", "WORKING"), ("GOAL", goal[:18])])


class _HUDFooter(QWidget):
    """Bottom bar: controls | voice visualizer | status."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedHeight(64)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Left controls
        self._ctrl_panel = _HUDInfoWidget("CONTROLS")
        self._ctrl_panel.set_lines([("HOTKEY", "Ctrl+Space")])
        self._ctrl_panel.setFixedWidth(130)
        layout.addWidget(self._ctrl_panel)

        # Voice visualizer (center)
        self._voice_bar = HUDVoiceBar()
        layout.addWidget(self._voice_bar, 3)

        # Right status
        self._status_panel = _HUDInfoWidget("SPIDY MEMORY")
        self._status_panel.set_lines([("SESSION", "ACTIVE"), ("MEM", "—")])
        self._status_panel.setFixedWidth(140)
        layout.addWidget(self._status_panel)

    def apply_theme(self, theme: "Theme") -> None:
        self._ctrl_panel.apply_theme(theme)
        self._status_panel.apply_theme(theme)
        self._voice_bar.apply_theme(theme)

    @property
    def voice_bar(self) -> HUDVoiceBar:
        return self._voice_bar


# ── OverlayWindow ─────────────────────────────────────────────────────────

def _hud_size() -> tuple[int, int]:
    """Compute responsive HUD dimensions based on primary screen."""
    screens = QGuiApplication.screens()
    screen = screens[0] if screens else None
    if screen:
        geom = screen.availableGeometry()
        sw, sh = geom.width(), geom.height()
    else:
        sw, sh = 1920, 1080
    w = max(1100, min(1400, int(sw * 0.82)))
    h = max(650,  min(850,  int(sh * 0.78)))
    return w, h


def _apply_acrylic(hwnd: int, color_hex: str) -> None:
    """Apply Windows acrylic/blur behind the window (best-effort)."""
    if platform.system() != "Windows":
        return
    try:
        from ctypes import windll, byref, c_int
        from ctypes.wintypes import DWORD

        class ACCENT_POLICY(ctypes.Structure):
            _fields_ = [
                ("AccentState",   c_int),
                ("AccentFlags",   c_int),
                ("GradientColor", DWORD),
                ("AnimationId",   c_int),
            ]

        class WINDOWCOMPOSITIONATTRIBDATA(ctypes.Structure):
            _fields_ = [
                ("Attribute",  c_int),
                ("Data",       ctypes.POINTER(ACCENT_POLICY)),
                ("SizeOfData", ctypes.c_size_t),
            ]

        col = QColor(color_hex)
        r, g, b, a = col.red(), col.green(), col.blue(), int(0.88 * 255)
        gradient_color = (a << 24) | (b << 16) | (g << 8) | r

        accent = ACCENT_POLICY(3, 0, gradient_color, 0)  # 3 = ACCENT_ENABLE_ACRYLICBLURBEHIND
        data = WINDOWCOMPOSITIONATTRIBDATA(19, byref(accent), ctypes.sizeof(accent))
        windll.user32.SetWindowCompositionAttribute(hwnd, byref(data))
    except Exception:
        pass  # not critical


class OverlayWindow(QWidget):
    """
    Spidy HUD Overlay (Milestone 17.2).

    Futuristic full-HUD command center with:
    - Responsive sizing (75-90% of screen)
    - Central AI core as the dominant visual
    - Surrounding telemetry modules
    - Floating conversation cards
    - Futuristic confirmation panel
    - Boot/close animation sequence

    Public API (backward-compatible with M17.1)
    -------------------------------------------
    set_state(state: str)
    add_message(role: str, text: str)
    update_waveform(amplitudes: list[float])
    show_notification(title, msg, level)
    update_task_progress(goal, steps)
    show_confirmation(task_id, goal_id, description, prompt)
    hide_confirmation()
    apply_theme(theme)
    show_animated()
    hide_animated()

    Signals
    -------
    mic_button_clicked
    overlay_closed
    """

    mic_button_clicked = Signal()
    overlay_closed     = Signal()

    def __init__(
        self,
        theme: "Theme",
        width: int = 0,       # 0 = auto-responsive
        height: int = 0,
        edge: str = "center",
        always_on_top: bool = True,
        animate: bool = True,
    ) -> None:
        super().__init__(None)

        # Responsive size
        auto_w, auto_h = _hud_size()
        self._hud_w = width  if width  > 0 else auto_w
        self._hud_h = height if height > 0 else auto_h
        self._animate = animate

        # Window flags: frameless, on top, translucent
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setFixedSize(self._hud_w, self._hud_h)

        # Center on screen
        screens = QGuiApplication.screens()
        if screens:
            sg = screens[0].availableGeometry()
            self.move(
                sg.x() + (sg.width()  - self._hud_w) // 2,
                sg.y() + (sg.height() - self._hud_h) // 2,
            )

        # Window opacity: start invisible for boot animation.
        # NOTE: We intentionally do NOT use QGraphicsOpacityEffect here.
        # QGraphicsOpacityEffect allocates an intermediate offscreen buffer with
        # its own internal QPainter.  When our custom paintEvent() then creates
        # a second QPainter on the same paint device, Qt prints:
        #   "QPainter::begin: A paint device can only be painted by one painter"
        # Using setWindowOpacity() avoids this conflict entirely.
        self.setWindowOpacity(0.0)

        # Build layout
        self._theme = theme
        self._build_layout()
        self._apply_theme_internal(theme)

        # Boot scale tracking (for core)
        self._boot_phase = 0
        self._boot_timer = QTimer(self)
        self._boot_timer.setInterval(16)  # ~60fps for animation
        self._boot_timer.timeout.connect(self._boot_tick)

        # Try acrylic
        QTimer.singleShot(100, self._setup_acrylic)

    # ---- Build Layout -------------------------------------------------------

    def _build_layout(self) -> None:
        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(6)

        # Header
        self._header = _HUDHeader()
        main.addWidget(self._header, 0)

        # Middle row: left | center | right
        middle = QHBoxLayout()
        middle.setSpacing(8)
        middle.setContentsMargins(0, 0, 0, 0)

        self._left_panel = _HUDLeftPanel()
        middle.addWidget(self._left_panel, 0)

        # Center: core + chat overlay stacked
        self._center_stack = QWidget()
        self._center_stack.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._center_stack.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        middle.addWidget(self._center_stack, 1)

        self._right_panel = _HUDRightPanel()
        middle.addWidget(self._right_panel, 0)

        # Set initial panel widths based on window size
        lw, rw = self._compute_side_widths(self._hud_w)
        self._left_panel.setFixedWidth(lw)
        self._right_panel.setFixedWidth(rw)

        main.addLayout(middle, 1)

        # Footer
        self._footer = _HUDFooter()
        main.addWidget(self._footer, 0)

        # Core widget inside center stack (resized in resizeEvent)
        self._core = SpidyCoreWidget(self._center_stack)

        # Chat overlay (positioned over lower center)
        self._chat_overlay = HUDChatOverlay(self._center_stack)

        # Confirmation overlay (centered over center stack)
        self._confirmation_card = HUDConfirmation(self._center_stack)
        self._confirmation_card.hide()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Update side panel widths proportionally when window is resized
        lw, rw = self._compute_side_widths(self.width())
        self._left_panel.setFixedWidth(lw)
        self._right_panel.setFixedWidth(rw)
        self._layout_center_children()

    @staticmethod
    def _compute_side_widths(total_w: int) -> tuple[int, int]:
        """Compute left/right panel widths proportional to window width."""
        left_w  = max(152, min(210, int(total_w * 0.128)))
        right_w = max(172, min(230, int(total_w * 0.145)))
        return left_w, right_w

    def _layout_center_children(self) -> None:
        s = self._center_stack
        w, h = s.width(), s.height()

        # Core: square, taking 80% of center height centered
        core_size = min(w, int(h * 0.88))
        cx = (w - core_size) // 2
        cy = (h - core_size) // 2
        self._core.setGeometry(cx, cy, core_size, core_size)

        # Chat: lower 35% of center
        chat_h = int(h * 0.35)
        self._chat_overlay.setGeometry(0, h - chat_h, w, chat_h)

        # Confirmation: middle 60% centered
        conf_w = int(w * 0.70)
        conf_h = int(h * 0.52)
        self._confirmation_card.setGeometry(
            (w - conf_w) // 2, (h - conf_h) // 2, conf_w, conf_h
        )

    # ---- Public API ---------------------------------------------------------

    def set_state(self, state: str) -> None:
        self._core.set_state(state)
        self._header.set_state(state)
        self._footer.voice_bar.set_state(state)
        self._left_panel.set_mic_state(state == "listening")

        # Task console visibility
        show_tasks = state == "working"
        self._right_panel.show_task_console(show_tasks)

    def add_message(self, role: str, text: str) -> None:
        self._chat_overlay.add_message(role, text)

    def update_waveform(self, amplitudes) -> None:
        if amplitudes:
            amp = max(amplitudes) if hasattr(amplitudes, "__iter__") else float(amplitudes)
        else:
            amp = 0.0
        self._core.set_amplitude(amp)
        self._footer.voice_bar.set_amplitude(amp)

    def show_notification(self, title: str, msg: str, level: str = "info") -> None:
        # Notifications handled by NotificationManager in app.py
        pass

    def update_task_progress(self, goal: str, steps) -> None:
        step_objs: list[TaskStep] = []
        for s in (steps or []):
            if isinstance(s, dict):
                step_objs.append(TaskStep(s.get("description", ""), s.get("status", "pending")))
            elif hasattr(s, "description"):
                step_objs.append(s)
            elif isinstance(s, (list, tuple)) and len(s) == 2:
                step_objs.append(TaskStep(s[0], s[1]))
        self._right_panel.update_task(goal, step_objs)

    def show_confirmation(
        self, task_id: str, goal_id: str, description: str, prompt: str = ""
    ) -> None:
        self._confirmation_card.show_confirmation(task_id, goal_id, description, prompt)

    def hide_confirmation(self) -> None:
        self._confirmation_card.hide_confirmation()

    def apply_theme(self, theme: "Theme") -> None:
        self._theme = theme
        self._apply_theme_internal(theme)
        self.update()

    # ---- Animation ----------------------------------------------------------

    def show_animated(self) -> None:
        if not self._animate:
            self.setWindowOpacity(1.0)
            self.show()
            return
        self.setWindowOpacity(0.0)
        self.show()
        self._boot_phase = 0
        self._core.set_boot_scale(0.05)
        self._boot_timer.start()

    def hide_animated(self) -> None:
        if not self._animate:
            self.hide()
            self.overlay_closed.emit()
            return
        anim = QPropertyAnimation(self, b"windowOpacity", self)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setDuration(400)
        anim.setEasingCurve(QEasingCurve.Type.InCubic)
        anim.finished.connect(self._on_hide_done)
        anim.start()

    def _boot_tick(self) -> None:
        """Drives the multi-stage boot animation (~60fps via 16ms timer)."""
        self._boot_phase += 1
        p = self._boot_phase

        if p <= 15:  # fade in window (0–240ms)
            self.setWindowOpacity(p / 15.0 * 0.85)
        elif p <= 30:  # core expands (240–480ms)
            scale = (p - 15) / 15.0
            ease = 1 - (1 - scale) ** 3  # ease-out cubic
            self._core.set_boot_scale(ease)
            self.setWindowOpacity(0.85 + 0.15 * ease)
        elif p <= 40:  # settle
            self._core.set_boot_scale(1.0)
            self.setWindowOpacity(1.0)
            self._boot_timer.stop()
            log.debug("HUD boot animation complete")

    def _on_hide_done(self) -> None:
        self.hide()
        self.overlay_closed.emit()

    # ---- Painting (HUD frame) -----------------------------------------------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        if not p.isActive(): return
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()

        # Deep dark background
        bg = QColor("#060D1A")
        bg.setAlphaF(0.94)
        p.setBrush(QBrush(bg))
        p.setPen(Qt.PenStyle.NoPen)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, w, h), 12, 12)
        p.drawPath(path)

        # Glowing cyan edge
        for i, alpha in enumerate([0.55, 0.30, 0.12]):
            pen_col = QColor("#00E5FF")
            pen_col.setAlphaF(alpha)
            p.setPen(QPen(pen_col, 1.0 - i * 0.25))
            p.setBrush(Qt.BrushStyle.NoBrush)
            inset = i
            p.drawRoundedRect(QRectF(inset, inset, w - inset*2, h - inset*2), 12 - inset, 12 - inset)

        # Subtle grid lines (horizontal)
        grid_col = QColor("#00E5FF")
        grid_col.setAlphaF(0.04)
        p.setPen(QPen(grid_col, 0.5))
        for gy in range(0, h, 30):
            p.drawLine(QPointF(8, gy), QPointF(w - 8, gy))

        p.end()

    # ---- Internal -----------------------------------------------------------

    def _apply_theme_internal(self, theme: "Theme") -> None:
        self._core.apply_theme(theme)
        self._header.apply_theme(theme)
        self._left_panel.apply_theme(theme)
        self._right_panel.apply_theme(theme)
        self._footer.apply_theme(theme)
        self._chat_overlay.apply_theme(theme)
        self._confirmation_card.apply_theme(theme)

    def _setup_acrylic(self) -> None:
        try:
            hwnd = int(self.winId())
            _apply_acrylic(hwnd, "#06091A")
        except Exception:
            pass

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.hide_animated()
        super().keyPressEvent(event)
