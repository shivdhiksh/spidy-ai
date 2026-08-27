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

    # Thread-safe HUD visibility control signals (dispatched to Qt main thread)
    show_hud_requested       = Signal()
    hide_hud_requested       = Signal()
    toggle_hud_requested     = Signal()

    def request_show_hud(self) -> None:
        self.show_hud_requested.emit()

    def request_hide_hud(self) -> None:
        self.hide_hud_requested.emit()

    def request_toggle_hud(self) -> None:
        self.toggle_hud_requested.emit()

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
    """Top bar: left datetime | center status | right sys info + window controls + dragging."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedHeight(52)
        self._drag_pos = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(8)

        self._tl = _HUDInfoWidget("SPIDY SYSTEM")
        self._tc = _HUDInfoWidget("SPIDY INTELLIGENCE")
        self._tr = _HUDInfoWidget("SPIDY NETWORK")

        for w in (self._tl, self._tc, self._tr):
            layout.addWidget(w, 1)

        # Window controls (minimize & close buttons)
        self._btn_min = QPushButton("—")
        self._btn_min.setFixedSize(28, 28)
        self._btn_min.setStyleSheet("""
            QPushButton {
                background: rgba(10, 25, 41, 0.7);
                color: #80DEEA;
                border: 1px solid rgba(0, 229, 255, 0.4);
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background: rgba(0, 229, 255, 0.2);
                color: #FFFFFF;
            }
        """)
        self._btn_min.clicked.connect(self._on_minimize)
        layout.addWidget(self._btn_min)

        self._btn_close = QPushButton("✕")
        self._btn_close.setFixedSize(28, 28)
        self._btn_close.setStyleSheet("""
            QPushButton {
                background: rgba(10, 25, 41, 0.7);
                color: #FF8A80;
                border: 1px solid rgba(255, 36, 68, 0.4);
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background: rgba(255, 36, 68, 0.3);
                color: #FFFFFF;
            }
        """)
        self._btn_close.clicked.connect(self._on_close)
        layout.addWidget(self._btn_close)

        self._state_label = self._tc  # kept for test compat

        # Clock timer
        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._update_clock)
        self._clock.start()
        self._update_clock()

    def _on_minimize(self) -> None:
        win = self.window()
        if win:
            win.showMinimized()

    def _on_close(self) -> None:
        win = self.window()
        if win:
            OverlayWindow._session_pos = win.pos()
            if hasattr(win, "hide_animated"):
                win.hide_animated()
            else:
                win.hide()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            win = self.window()
            if win:
                self._drag_pos = event.globalPosition().toPoint() - win.pos()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton and self._drag_pos is not None:
            win = self.window()
            if win:
                new_pos = event.globalPosition().toPoint() - self._drag_pos
                win.move(new_pos)
                OverlayWindow._session_pos = new_pos
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)

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

def _hud_size(width_hint: int = 0, height_hint: int = 0) -> tuple[int, int]:
    """
    Compute responsive wide HUD dimensions based on primary screen geometry.
    Priority: WIDTH > HEIGHT (horizontal futuristic command center).
    """
    if width_hint > 0 and height_hint > 0:
        return width_hint, height_hint

    screens = QGuiApplication.screens()
    screen = screens[0] if screens else None
    if screen and screen.availableGeometry().width() > 800:
        geom = screen.availableGeometry()
        sw, sh = geom.width(), geom.height()
    else:
        sw, sh = 1920, 1080

    # Minimum desktop safe boundaries (adapt for small screens)
    min_safe_w = min(1050, max(750, sw - 40))
    min_safe_h = min(580,  max(480, sh - 60))

    # Standard wide horizontal target:
    # At 1920x1080 -> w ≈ 1550, h ≈ 780
    # At 1536x864  -> w ≈ 1350, h ≈ 700
    # At 1280x720  -> w ≈ 1150, h ≈ 600
    # At 2560x1440 -> w ≈ 1950, h ≈ 1000
    w = max(min_safe_w, min(1980, int(sw * 0.82)))
    h = max(min_safe_h, min(1080, int(sh * 0.78)))

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
    Spidy HUD Overlay.

    Futuristic wide desktop command center:
    - Wide horizontal layout (WIDTH > HEIGHT)
    - Central large AI core with multi-ring animation
    - Full-width scrollable conversation area
    - Real hardware telemetry (CPU, RAM, GPU, VRAM, Disk)
    - Autonomous agent task console
    - Floating & draggable window with persistent position
    - Clean hide/show/toggle lifecycle
    """

    mic_button_clicked = Signal()
    overlay_closed     = Signal()

    _session_pos = None  # Persistent position across toggles in same session

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

        # Responsive wide size
        auto_w, auto_h = _hud_size(width, height)
        self._hud_w = auto_w
        self._hud_h = auto_h
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

        # Position on screen (use saved session position if user moved window)
        if OverlayWindow._session_pos is not None:
            self.move(OverlayWindow._session_pos)
        else:
            screens = QGuiApplication.screens()
            if screens:
                sg = screens[0].availableGeometry()
                self.move(
                    sg.x() + (sg.width()  - self._hud_w) // 2,
                    sg.y() + (sg.height() - self._hud_h) // 2,
                )

        # Window opacity: start invisible for boot animation
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

        # Log diagnostics
        self._log_diagnostics()

    def _log_diagnostics(self) -> None:
        screens = QGuiApplication.screens()
        screen = screens[0] if screens else None
        sw = screen.geometry().width() if screen else 0
        sh = screen.geometry().height() if screen else 0
        aw = screen.availableGeometry().width() if screen else 0
        ah = screen.availableGeometry().height() if screen else 0
        dpr = screen.devicePixelRatio() if screen else 1.0
        log.info(
            f"[HUD DIAGNOSTICS] screen={sw}x{sh} available={aw}x{ah} dpr={dpr} | "
            f"requested_size={self._hud_w}x{self._hud_h} actual_size={self.width()}x{self.height()} "
            f"pos=({self.x()},{self.y()})"
        )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._log_diagnostics()

    # ---- Build Layout -------------------------------------------------------

    def _build_layout(self) -> None:
        main = QVBoxLayout(self)
        main.setContentsMargins(10, 8, 10, 8)
        main.setSpacing(8)

        # 1. Header (SPIDY SYSTEM | SPIDY INTELLIGENCE | SPIDY NETWORK | — X)
        self._header = _HUDHeader()
        main.addWidget(self._header, 0)

        # 2. Upper row: TELEMETRY | SPIDY CORE | SPIDY AGENT
        self._upper_row = QHBoxLayout()
        self._upper_row.setSpacing(10)
        self._upper_row.setContentsMargins(0, 0, 0, 0)

        self._left_panel = _HUDLeftPanel()
        self._upper_row.addWidget(self._left_panel, 0)

        # Central core container
        self._core_container = QWidget()
        self._core_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._core_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._center_stack = self._core_container  # Alias for backward compatibility

        core_box = QHBoxLayout(self._core_container)
        core_box.setContentsMargins(0, 0, 0, 0)
        self._core = SpidyCoreWidget(self._core_container)
        core_box.addWidget(self._core, 0, Qt.AlignmentFlag.AlignCenter)
        self._upper_row.addWidget(self._core_container, 1)

        self._right_panel = _HUDRightPanel()
        self._upper_row.addWidget(self._right_panel, 0)

        main.addLayout(self._upper_row, 0)

        # 3. Middle row: CONVERSATION (Full horizontal width)
        self._chat_container = QWidget()
        self._chat_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._chat_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        chat_box = QVBoxLayout(self._chat_container)
        chat_box.setContentsMargins(0, 0, 0, 0)
        self._chat_overlay = HUDChatOverlay(self._chat_container)
        chat_box.addWidget(self._chat_overlay)
        main.addWidget(self._chat_container, 1)

        # 4. Footer (CONTROLS | VOICE BAR | SPIDY MEMORY)
        self._footer = _HUDFooter()
        main.addWidget(self._footer, 0)

        # Confirmation card (overlaid on top of center stack when active)
        self._confirmation_card = HUDConfirmation(self._center_stack)
        self._confirmation_card.hide()

        # Apply initial sizes
        self._update_layout_dimensions()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_layout_dimensions()

    def moveEvent(self, event) -> None:
        super().moveEvent(event)
        OverlayWindow._session_pos = self.pos()

    def _update_layout_dimensions(self) -> None:
        lw, rw = self._compute_side_widths(self.width())
        self._left_panel.setFixedWidth(lw)
        self._right_panel.setFixedWidth(rw)

        # Upper row height is ~36-40% of total height
        upper_h = max(190, min(360, int(self.height() * 0.38)))
        self._left_panel.setFixedHeight(upper_h)
        self._right_panel.setFixedHeight(upper_h)
        self._core_container.setFixedHeight(upper_h)

        # Large core sizing
        avail_core_w = self.width() - lw - rw - 36
        core_size = max(170, min(upper_h - 8, avail_core_w))
        self._core.setFixedSize(core_size, core_size)

        # Confirmation card geometry
        if self._confirmation_card.isVisible():
            cw = self._center_stack.width()
            ch = self._center_stack.height()
            conf_w = max(240, int(cw * 0.85))
            conf_h = max(160, int(ch * 0.75))
            self._confirmation_card.setGeometry(
                (cw - conf_w) // 2,
                (ch - conf_h) // 2,
                conf_w,
                conf_h,
            )

    @staticmethod
    def _compute_side_widths(total_w: int) -> tuple[int, int]:
        """Compute left/right panel widths proportional to window width."""
        left_w  = max(152, min(210, int(total_w * 0.128)))
        right_w = max(172, min(230, int(total_w * 0.145)))
        return left_w, right_w

    # ---- Public API ---------------------------------------------------------

    def show_hud(self) -> None:
        """Public API: show the HUD overlay with animation."""
        self.show_animated()

    def hide_hud(self) -> None:
        """Public API: hide the HUD overlay."""
        self.hide_animated()

    def toggle_hud(self) -> None:
        """Public API: toggle HUD visibility."""
        if self.isVisible() and self.windowOpacity() > 0.05:
            self.hide_animated()
        else:
            self.show_animated()

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
        cw = self._center_stack.width()
        ch = self._center_stack.height()
        conf_w = max(240, int(cw * 0.85))
        conf_h = max(160, int(ch * 0.75))
        self._confirmation_card.setGeometry(
            (cw - conf_w) // 2,
            (ch - conf_h) // 2,
            conf_w,
            conf_h,
        )
        self._confirmation_card.show_confirmation(task_id, goal_id, description, prompt)
        self._confirmation_card.show()
        self._confirmation_card.raise_()

    def hide_confirmation(self) -> None:
        self._confirmation_card.hide_confirmation()
        self._confirmation_card.hide()

    def apply_theme(self, theme: "Theme") -> None:
        self._theme = theme
        self._apply_theme_internal(theme)
        self.update()

    # ---- Animation ----------------------------------------------------------

    def show_animated(self) -> None:
        # If already fully visible and booted, raise and bring to front without re-zeroing opacity
        if self.isVisible() and self.windowOpacity() >= 0.95:
            self.raise_()
            self.activateWindow()
            log.debug("[HUD LIFECYCLE] show_animated called while already visible — raised window")
            return

        if OverlayWindow._session_pos is not None:
            self.move(OverlayWindow._session_pos)

        if not self._animate:
            self.setWindowOpacity(1.0)
            self.show()
            self.raise_()
            self.activateWindow()
            return

        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()
        self.activateWindow()
        self._boot_phase = 0
        self._core.set_boot_scale(0.05)
        self._boot_timer.start()
        log.info("[HUD LIFECYCLE] show_animated started boot animation on Qt thread")

    def hide_animated(self) -> None:
        # Save position before hiding
        OverlayWindow._session_pos = self.pos()
        if not self._animate:
            self.hide()
            self.overlay_closed.emit()
            return
        anim = QPropertyAnimation(self, b"windowOpacity", self)
        anim.setStartValue(self.windowOpacity())
        anim.setEndValue(0.0)
        anim.setDuration(300)
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
