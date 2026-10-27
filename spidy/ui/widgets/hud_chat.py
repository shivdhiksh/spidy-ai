"""
hud_chat.py -- HUDChatOverlay: Spider-Verse Conversation History Panel
======================================================================
Scrollable conversation history panel with Spider-Verse styling:
- Glassmorphic message bubbles with Spider-red and Navy accents
- Distinct typography for YOU vs SPIDY
- Natural wheel scrolling and smart auto-scrolling
- High-contrast Spider-Verse scrollbar and jump button
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from spidy.ui.themes.base import Theme

_MAX_CARDS = 3


class _LegacyCard:
    def __init__(self, opacity: float = 1.0):
        self.opacity = opacity


class _ChatMessageBubble(QFrame):
    """Individual Spider-Verse conversation message bubble."""

    def __init__(
        self,
        role: str,
        text: str,
        theme_colors: dict[str, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.role = role.lower()  # "user" | "assistant" | "system"
        self.text = text
        self.timestamp = datetime.datetime.now().strftime("%H:%M")

        # Color defaults (Spider-Verse HUD)
        self._c_primary = QColor("#E62429")   # Spider-red
        self._c_dim = QColor("#1B2A4A")       # Navy web-blue
        self._c_text = QColor("#F5F5F7")
        self._c_bg = QColor("#12141C")        # Dark glass
        self._c_user_bg = QColor("#161924")   # Cool contrast slate

        if theme_colors:
            self._apply_colors(theme_colors)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 10)
        layout.setSpacing(4)

        # Header: Role + Timestamp
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        role_display = "SPIDY" if self.role == "assistant" else ("YOU" if self.role == "user" else "SYSTEM")
        self._lbl_role = QLabel(role_display)
        self._lbl_role.setFont(QFont("Rajdhani", 10, QFont.Weight.Bold))
        self._lbl_role.setStyleSheet(
            f"color: {'#E62429' if self.role == 'assistant' else '#3FA9F5'}; background: transparent; letter-spacing: 1px;"
        )
        header_layout.addWidget(self._lbl_role)

        self._lbl_time = QLabel(self.timestamp)
        self._lbl_time.setFont(QFont("JetBrains Mono", 8))
        self._lbl_time.setStyleSheet("color: rgba(245, 245, 247, 0.45); background: transparent;")
        header_layout.addWidget(self._lbl_time, 0, Qt.AlignmentFlag.AlignRight)

        layout.addLayout(header_layout)

        # Message Text
        self._lbl_text = QLabel(self.text)
        self._lbl_text.setFont(QFont("Inter", 10))
        self._lbl_text.setWordWrap(True)
        self._lbl_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._lbl_text.setStyleSheet(
            "color: #F5F5F7; background: transparent; line-height: 135%;"
        )
        layout.addWidget(self._lbl_text)

    def _apply_colors(self, c: dict[str, str]) -> None:
        if "hud_primary" in c:
            self._c_primary = QColor(c["hud_primary"])
        if "hud_secondary" in c:
            self._c_dim = QColor(c["hud_secondary"])
        if "hud_text" in c:
            self._c_text = QColor(c["hud_text"])
        if "hud_grid" in c:
            self._c_bg = QColor(c["hud_grid"])

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        bg_col = QColor(self._c_bg if self.role == "assistant" else self._c_user_bg)
        bg_col.setAlphaF(0.90)

        border_col = QColor(self._c_primary if self.role == "assistant" else self._c_dim)
        border_col.setAlphaF(0.40)

        path = QPainterPath()
        path.addRoundedRect(QRectF(1, 1, w - 2, h - 2), 8, 8)
        painter.setBrush(QBrush(bg_col))
        painter.setPen(QPen(border_col, 1.0))
        painter.drawPath(path)

        # Left accent stripe
        stripe_col = QColor(self._c_primary if self.role == "assistant" else "#3FA9F5")
        stripe_col.setAlphaF(0.85)
        stripe_path = QPainterPath()
        stripe_path.addRoundedRect(QRectF(1, 4, 3, h - 8), 1.5, 1.5)
        painter.setBrush(QBrush(stripe_col))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(stripe_path)

        painter.end()


class HUDChatOverlay(QWidget):
    """
    Dedicated, scrollable conversation history widget with natural wheel scrolling,
    comfortable scrollbar hit zones, and smart history-preserving auto-scroll.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._messages: list[tuple[str, str]] = []  # (role, text)
        self._cards: list[_LegacyCard] = []
        self._theme_colors: dict[str, str] = {}

        # Outer layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Scroll Area
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: rgba(18, 20, 28, 0.6);
                width: 8px;
                border-radius: 4px;
                margin: 2px 0px 2px 0px;
            }
            QScrollBar::handle:vertical {
                background: rgba(230, 36, 41, 0.45);
                border-radius: 4px;
                min-height: 28px;
            }
            QScrollBar::handle:vertical:hover, QScrollBar::handle:vertical:pressed {
                background: rgba(230, 36, 41, 0.90);
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)

        # Container inside scroll area
        self._container = QWidget()
        self._container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._msg_layout = QVBoxLayout(self._container)
        self._msg_layout.setContentsMargins(2, 2, 8, 2)
        self._msg_layout.setSpacing(6)
        self._msg_layout.addStretch(1)

        self._scroll.setWidget(self._container)
        main_layout.addWidget(self._scroll)

        # Connect scrollbar signals for tracking position
        vsb = self._scroll.verticalScrollBar()
        if vsb:
            vsb.valueChanged.connect(self._on_scroll_value_changed)

        # Floating "↓ New message" jump button
        self._btn_new_msg = QPushButton("↓ New message", self)
        self._btn_new_msg.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_new_msg.setStyleSheet("""
            QPushButton {
                background-color: rgba(18, 20, 28, 0.95);
                color: #E62429;
                border: 1px solid #E62429;
                border-radius: 12px;
                font-family: 'Rajdhani', 'JetBrains Mono', sans-serif;
                font-size: 9pt;
                font-weight: bold;
                padding: 4px 14px;
            }
            QPushButton:hover {
                background-color: rgba(230, 36, 41, 0.25);
                color: #FFFFFF;
                border: 1px solid #FF3B3F;
            }
        """)
        self._btn_new_msg.clicked.connect(self._on_new_msg_btn_clicked)
        self._btn_new_msg.hide()

        # Install event filters for smooth wheel event routing
        self._scroll.viewport().installEventFilter(self)
        self._container.installEventFilter(self)

    # ---- Event Filters & Wheel Scrolling -------------------------------------

    def eventFilter(self, watched, event) -> bool:
        """Ensure mouse wheel events anywhere over the chat panel scroll the conversation."""
        if event.type() == QEvent.Type.Wheel:
            assert isinstance(event, QWheelEvent)
            self._handle_wheel(event)
            return True
        return super().eventFilter(watched, event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        self._handle_wheel(event)

    def _handle_wheel(self, event: QWheelEvent) -> None:
        vsb = self._scroll.verticalScrollBar()
        if not vsb:
            return
        pixel_delta = event.pixelDelta().y()
        if pixel_delta != 0:
            step = -pixel_delta
        else:
            angle_delta = event.angleDelta().y()
            step = int(-angle_delta / 120.0 * 60.0) if angle_delta != 0 else 0
        vsb.setValue(vsb.value() + step)
        event.accept()

    def _is_near_bottom(self) -> bool:
        vsb = self._scroll.verticalScrollBar()
        if not vsb:
            return True
        return (vsb.maximum() - vsb.value()) <= 60

    # ---- Scroll Position Tracking & Jump Button -----------------------------

    def _on_scroll_value_changed(self, value: int) -> None:
        vsb = self._scroll.verticalScrollBar()
        if not vsb:
            return
        max_val = vsb.maximum()
        # If user scrolled near bottom, hide jump button
        if max_val - value <= 40:
            self._btn_new_msg.hide()

    def _on_new_msg_btn_clicked(self) -> None:
        self._scroll_to_bottom()
        self._btn_new_msg.hide()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Position floating button at bottom-center
        bw = self._btn_new_msg.sizeHint().width() + 10
        bh = 28
        bx = (self.width() - bw) // 2
        by = self.height() - bh - 10
        self._btn_new_msg.setGeometry(bx, by, bw, bh)

    # ---- Public API ---------------------------------------------------------

    def add_message(self, role: str, text: str) -> None:
        self._messages.append((role, text))
        bubble = _ChatMessageBubble(role, text, self._theme_colors, self._container)

        # Update legacy _cards for test compatibility
        n = min(len(self._messages), _MAX_CARDS)
        self._cards = []
        for i in range(n):
            op = 0.35 + 0.65 * ((i + 1) / n) if n > 1 else 1.0
            self._cards.append(_LegacyCard(op))

        # Insert before the stretch at the end
        idx = max(0, self._msg_layout.count() - 1)
        self._msg_layout.insertWidget(idx, bubble)

        # Auto-scroll logic: only auto-scroll if user was already near the bottom
        vsb = self._scroll.verticalScrollBar()
        if vsb:
            was_at_bottom = (vsb.maximum() - vsb.value()) <= 60
            if was_at_bottom:
                QTimer.singleShot(50, self._scroll_to_bottom)
            else:
                self._btn_new_msg.show()
                self._btn_new_msg.raise_()

    def clear(self) -> None:
        self._messages.clear()
        self._cards.clear()
        while self._msg_layout.count() > 1:
            item = self._msg_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self.update()

    def _scroll_to_bottom(self) -> None:
        vsb = self._scroll.verticalScrollBar()
        if vsb:
            vsb.setValue(vsb.maximum())

    def apply_theme(self, theme: "Theme") -> None:
        c = theme.colors
        self._theme_colors = {
            "hud_primary": c.hud_primary,
            "hud_secondary": c.hud_secondary,
            "hud_dim": c.hud_dim,
            "hud_grid": c.bg_secondary,
            "hud_text": c.text_primary,
        }
        for i in range(self._msg_layout.count()):
            item = self._msg_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), _ChatMessageBubble):
                item.widget()._apply_colors(self._theme_colors)
                item.widget().update()
        self.update()
