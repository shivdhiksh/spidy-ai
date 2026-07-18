"""
OverlayWindow — Spidy's glassmorphism desktop companion
========================================================

Architecture
------------
OverlayWindow is a frameless, always-on-top QWidget that floats
over all other windows.  It is the only module that directly
creates Qt widgets; all other modules interact with it through:

    1.  UISignalBridge signals  (thread-safe Qt→Qt)
    2.  EventBus events (via SpidyApp)

Layout
------
    ┌─────────────────────────────────────────┐
    │  HEADER   [⚡ Spidy]  [state]  [− ×]    │  ← Draggable
    ├─────────────────────────────────────────┤
    │                                          │
    │  CHAT VIEW  (scrollable bubbles)         │
    │                                          │
    ├─────────────────────────────────────────┤
    │  WAVEFORM  ████ ████ ████ (listening)   │  ← Hidden when idle
    │  INDICATOR ●  ●  ●   (thinking/speaking)│
    ├─────────────────────────────────────────┤
    │  BOTTOM   [🎤 MicButton]  [state label] │
    └─────────────────────────────────────────┘

State → widget visibility mapping
----------------------------------
State         Waveform   Indicator   MicState   Label
-----------   --------   ---------   --------   --------
IDLE          hidden     hidden      idle       ""
WAKE_READY    hidden     hidden      idle       "Say 'Hey Spidy'…"
LISTENING     visible    hidden      listening  "Listening…"
THINKING      hidden     visible     thinking   "Thinking…"
SPEAKING      hidden     visible     speaking   "Speaking…"
ERROR         hidden     hidden      disabled   "Error"

Glassmorphism
-------------
Achieved via:
  - Qt.WA_TranslucentBackground on the window
  - Custom paintEvent drawing a rounded-rect with semi-transparent fill
  - Windows acrylic blur via SetWindowCompositionAttribute (optional,
    falls back gracefully on non-Windows or unsupported builds)

Wake-word readiness
-------------------
show_for_wake_word() is a no-arg slot ready to be connected to the
VoiceEngine's wake-word detection signal in a future milestone.
When called it will slide the overlay in from the configured edge.
"""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QEasingCurve, QObject, QPoint, QPropertyAnimation, QRect,
    Qt, QTimer, Signal, Slot,
)
from PySide6.QtGui import (
    QBrush, QColor, QFont, QGuiApplication,
    QPainter, QPainterPath, QPen,
    QMouseEvent, QCloseEvent,
)
from PySide6.QtWidgets import (
    QFrame, QGraphicsOpacityEffect,
    QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

if TYPE_CHECKING:
    from spidy.ui.themes.base import Theme

from spidy.ui.notifications import NotificationManager
from spidy.ui.state import UIState
from spidy.ui.widgets.chat_view import ChatMessage, ChatView
from spidy.ui.widgets.mic_button import MicrophoneButton
from spidy.ui.widgets.speaking_indicator import SpeakingIndicator
from spidy.ui.widgets.waveform import WaveformWidget

# ─── Constants ───────────────────────────────────────────────────────────────

_FADE_MS = 250
_SLIDE_MS = 320
_EDGE_GLOW_MS = 40   # Edge glow animation timer


# ─── UISignalBridge ──────────────────────────────────────────────────────────

class UISignalBridge(QObject):
    """
    Thread-safe bridge between async EventBus handlers and Qt slots.

    When an EventBus subscriber (running in asyncio or a background thread)
    needs to update the Qt UI, it calls methods on this bridge.
    PySide6 automatically queues cross-thread signal emissions, making
    the bridge safe to use from any thread.
    """

    state_change_requested = Signal(str)       # UIState value string
    message_received = Signal(str, str)        # role, text
    notification_requested = Signal(str, str, str, int)  # title, body, level, dur_ms
    waveform_data_received = Signal(list)      # list[float]
    theme_change_requested = Signal(str)       # theme name

    def request_state_change(self, state: str) -> None:
        self.state_change_requested.emit(state)

    def request_message(self, role: str, text: str) -> None:
        self.message_received.emit(role, text)

    def request_notification(
        self, title: str, body: str, level: str = "info", duration_ms: int = 4000
    ) -> None:
        self.notification_requested.emit(title, body, level, duration_ms)

    def request_waveform(self, amplitudes: list) -> None:
        self.waveform_data_received.emit(amplitudes)

    def request_theme_change(self, name: str) -> None:
        self.theme_change_requested.emit(name)


# ─── HeaderBar ───────────────────────────────────────────────────────────────

class _HeaderBar(QWidget):
    """Draggable header bar with title, state indicator, and window controls."""

    minimize_clicked = Signal()
    close_clicked = Signal()

    def __init__(self, theme: "Theme", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(48)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._drag_pos: QPoint | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 12, 0)
        layout.setSpacing(8)

        # Logo + title
        logo = QLabel("⚡")
        logo.setFont(QFont(theme.fonts.family_primary, 16))
        layout.addWidget(logo)

        self._title = QLabel("Spidy")
        self._title.setFont(
            QFont(theme.fonts.family_primary, theme.fonts.size_lg, QFont.Weight.DemiBold)
        )
        self._title.setStyleSheet(f"color: {theme.colors.text_primary};")
        layout.addWidget(self._title)

        layout.addStretch()

        # State label
        self._state_label = QLabel("")
        self._state_label.setFont(
            QFont(theme.fonts.family_primary, theme.fonts.size_sm)
        )
        self._state_label.setStyleSheet(f"color: {theme.colors.text_muted};")
        layout.addWidget(self._state_label)

        # Window controls
        for symbol, sig in [("−", self.minimize_clicked), ("×", self.close_clicked)]:
            btn = QPushButton(symbol)
            btn.setFixedSize(28, 28)
            btn.clicked.connect(sig.emit)
            btn.setStyleSheet(
                f"QPushButton {{"
                f"  background: transparent;"
                f"  color: {theme.colors.text_muted};"
                f"  border: none;"
                f"  border-radius: 6px;"
                f"  font-size: 16px;"
                f"}}"
                f"QPushButton:hover {{"
                f"  background: {theme.colors.bg_tertiary};"
                f"  color: {theme.colors.text_primary};"
                f"}}"
            )
            layout.addWidget(btn)

    def set_state_label(self, text: str) -> None:
        self._state_label.setText(text)

    def apply_theme(self, theme: "Theme") -> None:
        self._title.setStyleSheet(f"color: {theme.colors.text_primary};")
        self._state_label.setStyleSheet(f"color: {theme.colors.text_muted};")

    # Drag to reposition
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.window().frameGeometry().topLeft()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_pos and event.buttons() == Qt.MouseButton.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_pos = None


# ─── OverlayWindow ────────────────────────────────────────────────────────────

class OverlayWindow(QWidget):
    """
    Spidy's glassmorphism floating overlay window.

    Signals
    -------
    mic_button_clicked   User clicked the microphone button
    settings_requested   User pressed settings button (future)
    overlay_closed       Window close button pressed
    """

    mic_button_clicked = Signal()
    settings_requested = Signal()
    overlay_closed = Signal()

    def __init__(
        self,
        theme: "Theme",
        *,
        width: int = 400,
        height: int = 580,
        edge: str = "top-right",
        always_on_top: bool = True,
        animate: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._theme = theme
        self._edge = edge
        self._animate = animate
        self._state = UIState.IDLE
        self._glow_phase = 0.0
        self._is_wake_ready = False

        # ── Window flags ──────────────────────────────────────────────
        flags = (
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool              # No taskbar entry
        )
        if always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowTitle("Spidy")
        self.setFixedSize(width, height)

        # ── Animations ────────────────────────────────────────────────
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_anim.setDuration(_FADE_MS)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._slide_anim = QPropertyAnimation(self, b"pos")
        self._slide_anim.setDuration(_SLIDE_MS)
        self._slide_anim.setEasingCurve(QEasingCurve.Type.OutBack)

        # Edge glow animation (wake-ready state)
        self._glow_timer = QTimer(self)
        self._glow_timer.setInterval(_EDGE_GLOW_MS)
        self._glow_timer.timeout.connect(self._tick_glow)

        # ── Build UI ──────────────────────────────────────────────────
        self._build_ui(theme)
        self._apply_windows_blur()

        # Position on screen
        self._reposition()

    # ── Public API ────────────────────────────────────────────────────────

    @Slot()
    def show_animated(self) -> None:
        """Slide and fade the overlay into view."""
        if self._animate:
            screen = QGuiApplication.primaryScreen().geometry()
            end_pos = self._calc_position(screen)
            start_pos = self._calc_start_pos(screen, end_pos)

            # Start from off-screen
            self.move(start_pos)
            self.show()

            self._slide_anim.setStartValue(start_pos)
            self._slide_anim.setEndValue(end_pos)
            self._slide_anim.start()

            self._fade_anim.setStartValue(0.0)
            self._fade_anim.setEndValue(1.0)
            self._fade_anim.start()
        else:
            self._reposition()
            self.show()
            self._opacity_effect.setOpacity(1.0)

    @Slot()
    def hide_animated(self) -> None:
        """Fade the overlay out."""
        if self._animate:
            self._fade_anim.setStartValue(1.0)
            self._fade_anim.setEndValue(0.0)
            # Disconnect any previous finished→hide connection before adding a
            # new one.  Without this, rapid repeated calls to hide_animated()
            # stack up connections and call hide() multiple times, producing
            # "Painter not active" warnings on the second (no-op) call.
            try:
                self._fade_anim.finished.disconnect(self.hide)
            except RuntimeError:
                pass  # Not connected — that's fine
            self._fade_anim.finished.connect(self.hide)
            self._fade_anim.start()
        else:
            self.hide()

    @Slot()
    def show_for_wake_word(self) -> None:
        """
        Show the overlay when a wake word is detected.

        This slot is prepared for future integration with the VoiceEngine.
        When the wake word "Hey Spidy" is detected, connect:
            voice_engine.wake_word_detected → overlay.show_for_wake_word
        """
        self.show_animated()
        self.set_state(UIState.WAKE_READY)

    @Slot(str)
    def set_state(self, state: UIState | str) -> None:
        """Transition the overlay to a new visual state."""
        if isinstance(state, str):
            try:
                state = UIState(state)
            except ValueError:
                return

        self._state = state
        self._update_state_widgets(state)

    @Slot(str, str)
    def add_message(self, role: str, text: str) -> None:
        """Add a message bubble to the chat view."""
        msg = ChatMessage(role=role, text=text)
        self._chat_view.add_message(msg)

    @Slot(str, str, str, int)
    def show_notification(
        self, title: str, body: str, level: str = "info", duration_ms: int = 4000
    ) -> None:
        """Show a toast notification."""
        self._notif_mgr.show(title, body, level, duration_ms)

    @Slot(list)
    def update_waveform(self, amplitudes: list) -> None:
        """Feed real-time audio amplitude data to the waveform widget."""
        self._waveform.set_amplitudes(amplitudes)

    @Slot(str)
    def apply_theme(self, theme: "Theme") -> None:
        """Re-style all widgets with the new theme."""
        self._theme = theme
        self._header.apply_theme(theme)
        self._chat_view.apply_theme(theme)
        self._waveform.apply_theme(theme)
        self._mic_button.apply_theme(theme)
        self._indicator.apply_theme(theme)
        self._notif_mgr.apply_theme(theme)
        self.update()

    # ── UI building ───────────────────────────────────────────────────────

    def _build_ui(self, theme: "Theme") -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────
        self._header = _HeaderBar(theme, self)
        self._header.minimize_clicked.connect(self.hide_animated)
        self._header.close_clicked.connect(self._on_close_requested)
        root.addWidget(self._header)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {theme.colors.border};")
        sep.setFixedHeight(1)
        root.addWidget(sep)

        # ── Chat area ─────────────────────────────────────────────────
        self._chat_view = ChatView(self)
        self._chat_view.apply_theme(theme)
        root.addWidget(self._chat_view, stretch=1)

        # ── Waveform (visible only during LISTENING) ──────────────────
        self._waveform = WaveformWidget(self)
        self._waveform.apply_theme(theme)
        self._waveform.setFixedHeight(56)
        self._waveform.setVisible(False)
        root.addWidget(self._waveform)

        # ── Speaking/thinking indicator ───────────────────────────────
        self._indicator = SpeakingIndicator(self)
        self._indicator.apply_theme(theme)
        self._indicator.setVisible(False)
        root.addWidget(self._indicator)

        # ── Bottom bar ────────────────────────────────────────────────
        bottom = QWidget()
        bottom.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        bottom.setFixedHeight(72)
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(8, 8, 8, 8)
        bottom_layout.setSpacing(12)

        self._mic_button = MicrophoneButton(self)
        self._mic_button.apply_theme(theme)
        self._mic_button.clicked.connect(self.mic_button_clicked.emit)
        bottom_layout.addWidget(self._mic_button)

        self._status_label = QLabel("Ready")
        self._status_label.setFont(
            QFont(theme.fonts.family_primary, theme.fonts.size_sm)
        )
        self._status_label.setStyleSheet(f"color: {theme.colors.text_muted};")
        self._status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        bottom_layout.addWidget(self._status_label)

        root.addWidget(bottom)

        # Notification manager
        self._notif_mgr = NotificationManager(
            self,
            bg_color=theme.colors.bg_secondary,
            text_color=theme.colors.text_primary,
            font_family=theme.fonts.family_primary,
        )

    def _update_state_widgets(self, state: UIState) -> None:
        """Show/hide/animate widgets according to the new state."""
        # Waveform
        show_waveform = state == UIState.LISTENING
        self._waveform.setVisible(show_waveform)
        self._waveform.set_active(show_waveform)

        # Speaking/thinking indicator
        show_indicator = state in (UIState.THINKING, UIState.SPEAKING)
        self._indicator.setVisible(show_indicator)
        self._indicator.set_active(show_indicator)

        # Mic button state
        mic_state_map = {
            UIState.IDLE: "idle",
            UIState.WAKE_READY: "idle",
            UIState.LISTENING: "listening",
            UIState.THINKING: "thinking",
            UIState.SPEAKING: "speaking",
            UIState.ERROR: "disabled",
        }
        self._mic_button.set_state(mic_state_map.get(state, "idle"))

        # Status label
        label_map = {
            UIState.IDLE: "Ready",
            UIState.WAKE_READY: "Say 'Hey Spidy'…",
            UIState.LISTENING: "Listening…",
            UIState.THINKING: "Thinking…",
            UIState.SPEAKING: "Speaking…",
            UIState.ERROR: "Something went wrong",
        }
        self._status_label.setText(label_map.get(state, ""))
        self._header.set_state_label(
            "" if state == UIState.IDLE else label_map.get(state, "")
        )

        # Edge glow (wake-ready)
        self._is_wake_ready = state == UIState.WAKE_READY
        if self._is_wake_ready and not self._glow_timer.isActive():
            self._glow_timer.start()
        elif not self._is_wake_ready:
            self._glow_timer.stop()
            self._glow_phase = 0.0

        self.update()

    # ── Painting ──────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        r = self._theme.geometry.border_radius
        bg = QColor(self._theme.colors.bg_primary)
        bg.setAlphaF(self._theme.overlay_opacity)
        border = QColor(self._theme.colors.border)

        # Background
        path = QPainterPath()
        path.addRoundedRect(0, 0, w, h, r, r)
        painter.fillPath(path, QBrush(bg))

        # Border
        painter.setPen(QPen(border, self._theme.geometry.border_width))
        painter.drawPath(path)

        # Wake-ready edge glow
        if self._is_wake_ready:
            glow_color = QColor(self._theme.colors.wake_glow)
            glow_alpha = 0.15 + 0.35 * math.sin(self._glow_phase)
            glow_color.setAlphaF(max(0.0, min(1.0, glow_alpha)))
            glow_pen = QPen(glow_color, 3)
            painter.setPen(glow_pen)
            painter.drawPath(path)

        painter.end()

    # ── Positioning ───────────────────────────────────────────────────────

    def _reposition(self) -> None:
        screen = QGuiApplication.primaryScreen().geometry()
        pos = self._calc_position(screen)
        self.move(pos)

    def _calc_position(self, screen: QRect) -> QPoint:
        margin = self._theme.geometry.margin_edge
        w, h = self.width(), self.height()
        positions = {
            "top-right":    QPoint(screen.right() - w - margin, screen.top() + margin + 36),
            "top-left":     QPoint(screen.left() + margin, screen.top() + margin + 36),
            "bottom-right": QPoint(screen.right() - w - margin, screen.bottom() - h - margin - 36),
            "bottom-left":  QPoint(screen.left() + margin, screen.bottom() - h - margin - 36),
        }
        return positions.get(self._edge, positions["top-right"])

    def _calc_start_pos(self, screen: QRect, end_pos: QPoint) -> QPoint:
        """Off-screen start position for slide animation."""
        w = self.width()
        if "right" in self._edge:
            return QPoint(screen.right() + 20, end_pos.y())
        return QPoint(screen.left() - w - 20, end_pos.y())

    # ── Windows acrylic blur ──────────────────────────────────────────────

    def _apply_windows_blur(self) -> None:
        """Apply real acrylic blur on Windows 10/11."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import windll, c_int, byref, sizeof

            class _ACCENT_POLICY(ctypes.Structure):
                _fields_ = [
                    ("AccentState", c_int),
                    ("AccentFlags", c_int),
                    ("GradientColor", c_int),
                    ("AnimationId", c_int),
                ]

            class _WCAD(ctypes.Structure):
                _fields_ = [
                    ("Attribute", c_int),
                    ("Data", ctypes.POINTER(c_int)),
                    ("SizeOfData", ctypes.c_size_t),
                ]

            ACCENT_ENABLE_ACRYLICBLURBEHIND = 4
            WCA_ACCENT_POLICY = 19

            accent = _ACCENT_POLICY()
            accent.AccentState = ACCENT_ENABLE_ACRYLICBLURBEHIND
            # ABGR tint: 80% transparent very dark blue
            accent.GradientColor = 0xCC0D0E1A

            data = _WCAD()
            data.Attribute = WCA_ACCENT_POLICY
            data.Data = ctypes.cast(byref(accent), ctypes.POINTER(c_int))
            data.SizeOfData = ctypes.sizeof(accent)

            windll.user32.SetWindowCompositionAttribute(
                int(self.winId()), byref(data)
            )
        except Exception:
            pass  # Silently ignore on unsupported configs

    # ── Internal ──────────────────────────────────────────────────────────

    def _tick_glow(self) -> None:
        self._glow_phase = (self._glow_phase + 0.12) % (math.pi * 2)
        self.update()

    def _on_close_requested(self) -> None:
        self.hide_animated()
        self.overlay_closed.emit()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.overlay_closed.emit()
        event.accept()
