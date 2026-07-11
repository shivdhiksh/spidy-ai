"""
NotificationManager — Toast notification system
================================================
Displays transient pop-up toasts above (or near) the overlay window.
Each toast appears with a slide-down animation, displays for a
configurable duration, then fades out automatically.

Design
------
- Stacks vertically (newest on top)
- Each level has a distinct color (info/success/warning/error)
- Persistent toasts (duration_ms=0) stay until explicitly dismissed
- Thread-safe: call ``request_notify()`` from any thread via EventBus

Usage
-----
    mgr = NotificationManager(parent_widget)
    mgr.show_notification("Ready", "Spidy is listening", level="info")
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from PySide6.QtCore import (
    QEasingCurve, QPoint, QPropertyAnimation, QRect, QTimer,
    Qt, Signal
)
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPainterPath,
)
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect, QHBoxLayout, QLabel,
    QVBoxLayout, QWidget,
)


@dataclass
class _ToastRequest:
    title: str
    body: str
    level: str          # "info" | "success" | "warning" | "error"
    duration_ms: int    # 0 = persistent


# ─── ToastWidget ─────────────────────────────────────────────────────────────

class ToastWidget(QWidget):
    """Single notification toast popup."""

    dismissed = Signal()

    _LEVEL_COLORS = {
        "info":    "#00D4FF",
        "success": "#00E96A",
        "warning": "#FFB800",
        "error":   "#FF4D6A",
    }
    _ICONS = {
        "info": "ℹ",
        "success": "✓",
        "warning": "⚠",
        "error": "✕",
    }

    def __init__(
        self,
        title: str,
        body: str,
        level: str = "info",
        duration_ms: int = 4000,
        bg_color: str = "#12131F",
        text_color: str = "#E8E9F0",
        font_family: str = "Segoe UI",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._level = level
        self._bg = QColor(bg_color)
        self._accent = QColor(self._LEVEL_COLORS.get(level, "#6C63FF"))

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedWidth(320)

        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        # Accent stripe + icon
        icon_label = QLabel(self._ICONS.get(level, "•"))
        icon_label.setFont(QFont(font_family, 16))
        icon_label.setStyleSheet(f"color: {self._accent.name()};")
        icon_label.setFixedWidth(24)
        layout.addWidget(icon_label)

        # Text
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setFont(QFont(font_family, 13, QFont.Weight.DemiBold))
        title_label.setStyleSheet(f"color: {text_color};")
        text_layout.addWidget(title_label)

        if body:
            body_label = QLabel(body)
            body_label.setFont(QFont(font_family, 11))
            body_label.setStyleSheet(f"color: {text_color}; opacity: 0.8;")
            body_label.setWordWrap(True)
            text_layout.addWidget(body_label)

        layout.addLayout(text_layout)
        layout.addStretch()
        self.adjustSize()

        # Opacity effect for fade
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        # Fade in
        self._fade_in = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_in.setDuration(220)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)

        # Fade out
        self._fade_out = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_out.setDuration(350)
        self._fade_out.setStartValue(1.0)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setEasingCurve(QEasingCurve.Type.InCubic)
        self._fade_out.finished.connect(self._on_fade_out_done)

        # Auto-dismiss timer
        if duration_ms > 0:
            self._dismiss_timer = QTimer(self)
            self._dismiss_timer.setSingleShot(True)
            self._dismiss_timer.setInterval(duration_ms)
            self._dismiss_timer.timeout.connect(self.dismiss)

    def show_animated(self) -> None:
        self.show()
        self._fade_in.start()
        if hasattr(self, "_dismiss_timer"):
            self._dismiss_timer.start()

    def dismiss(self) -> None:
        if hasattr(self, "_dismiss_timer"):
            self._dismiss_timer.stop()
        self._fade_out.start()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        path = QPainterPath()
        path.addRoundedRect(0, 0, w, h, 12, 12)
        painter.fillPath(path, QBrush(self._bg))

        # Left accent stripe
        stripe = QPainterPath()
        stripe.addRoundedRect(0, 0, 4, h, 2, 2)
        painter.fillPath(stripe, QBrush(self._accent))

        # Subtle border
        border_color = QColor(self._accent)
        border_color.setAlphaF(0.3)
        painter.setPen(border_color)
        painter.drawPath(path)

    def _on_fade_out_done(self) -> None:
        self.hide()
        self.dismissed.emit()


# ─── NotificationManager ────────────────────────────────────────────────────

class NotificationManager:
    """
    Manages a queue of toast notifications.

    Creates ToastWidget instances, stacks them near the top of the
    screen (same corner as the overlay), and auto-dismisses them.

    Parameters
    ----------
    parent:
        Parent widget (the overlay window). Toasts are positioned
        relative to the parent's screen position.
    max_stack:
        Maximum number of simultaneous toasts before older ones are
        forced-dismissed to make room.
    """

    def __init__(
        self,
        parent: QWidget,
        max_stack: int = 4,
        bg_color: str = "#12131F",
        text_color: str = "#E8E9F0",
        font_family: str = "Segoe UI",
    ) -> None:
        self._parent = parent
        self._max_stack = max_stack
        self._bg_color = bg_color
        self._text_color = text_color
        self._font_family = font_family
        self._toasts: list[ToastWidget] = []

    def show(
        self,
        title: str,
        body: str = "",
        level: str = "info",
        duration_ms: int = 4000,
    ) -> None:
        """Display a toast notification."""
        # Enforce max stack
        while len(self._toasts) >= self._max_stack:
            oldest = self._toasts.pop(0)
            oldest.dismiss()

        toast = ToastWidget(
            title=title,
            body=body,
            level=level,
            duration_ms=duration_ms,
            bg_color=self._bg_color,
            text_color=self._text_color,
            font_family=self._font_family,
        )
        toast.dismissed.connect(lambda t=toast: self._on_dismissed(t))
        self._toasts.append(toast)

        self._restack()
        toast.show_animated()

    def apply_theme(self, theme) -> None:
        """Update colors for future toasts."""
        self._bg_color = theme.colors.bg_secondary
        self._text_color = theme.colors.text_primary
        self._font_family = theme.fonts.family_primary

    def dismiss_all(self) -> None:
        """Dismiss all active toasts."""
        for toast in list(self._toasts):
            toast.dismiss()

    # ── Internal ──────────────────────────────────────────────────────────

    def _on_dismissed(self, toast: ToastWidget) -> None:
        if toast in self._toasts:
            self._toasts.remove(toast)
        toast.deleteLater()
        self._restack()

    def _restack(self) -> None:
        """Reposition toasts in a vertical stack above the parent."""
        if not self._parent or not self._parent.isVisible():
            return

        parent_rect = self._parent.geometry()
        # Start from the top of the parent window
        y = parent_rect.top() - 8
        x = parent_rect.right() - 328  # 320 width + 8 margin

        for toast in reversed(self._toasts):
            toast_h = toast.height() + 8
            y -= toast_h
            toast.move(x, y)
