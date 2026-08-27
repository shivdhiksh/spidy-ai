"""
hud_chat.py -- HUDChatOverlay: Scrollable Conversation History Panel
====================================================================
Provides a dedicated, readable conversation history region for the HUD:
- Full multi-turn conversation history without artificial truncation
- Distinct visual appearance for user ("YOU") vs assistant ("SPIDY")
- Proper word wrapping and selectable text
- Natural mouse-wheel and trackpad scrolling across the entire panel
- Visible, high-contrast, comfortable scrollbar hit zone
- Smart auto-scrolling (preserves scroll position when reading history)
- Unobtrusive "↓ New message" jump indicator when scrolled up
- Glassmorphism aesthetic matching the futuristic HUD theme
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
    """Individual conversation message bubble."""

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

        # Color defaults
        self._c_primary = QColor("#00E5FF")
        self._c_dim = QColor("#004D5E")
        self._c_text = QColor("#E0F7FA")
        self._c_bg = QColor("#0A1929")
        self._c_user_bg = QColor("#102A43")

        if theme_colors:
            self._apply_colors(theme_colors)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(4)

        # Header: Role + Timestamp
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        role_display = "SPIDY" if self.role == "assistant" else ("YOU" if self.role == "user" else "SYSTEM")
        self._lbl_role = QLabel(role_display)
        self._lbl_role.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        self._lbl_role.setStyleSheet(
            f"color: {'#00E5FF' if self.role == 'assistant' else '#80DEEA'}; background: transparent;"
        )
        header_layout.addWidget(self._lbl_role)

        self._lbl_time = QLabel(self.timestamp)
        self._lbl_time.setFont(QFont("Consolas", 8))
        self._lbl_time.setStyleSheet("color: rgba(178, 235, 242, 0.5); background: transparent;")
        header_layout.addWidget(self._lbl_time, 0, Qt.AlignmentFlag.AlignRight)

        layout.addLayout(header_layout)

        # Message Text
        self._lbl_text = QLabel(self.text)
        self._lbl_text.setFont(QFont("Segoe UI", 10))
        self._lbl_text.setWordWrap(True)
        self._lbl_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._lbl_text.setStyleSheet(
            "color: #E0F7FA; background: transparent; line-height: 130%;"
        )
        layout.addWidget(self._lbl_text)

    def _apply_colors(self, c: dict[str, str]) -> None:
        if "hud_primary" in c:
            self._c_primary = QColor(c["hud_primary"])
        if "hud_dim" in c:
            self._c_dim = QColor(c["hud_dim"])
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
        bg_col.setAlphaF(0.85)

        border_col = QColor(self._c_primary if self.role == "assistant" else self._c_dim)
        border_col.setAlphaF(0.55 if self.role == "assistant" else 0.35)

        path = QPainterPath()
        path.addRoundedRect(QRectF(1, 1, w - 2, h - 2), 6, 6)
        painter.setBrush(QBrush(bg_col))
        painter.setPen(QPen(border_col, 1.0))
        painter.drawPath(path)
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
                background: rgba(10, 25, 41, 0.5);
                width: 10px;
                border-radius: 5px;
                margin: 2px 0px 2px 0px;
            }
            QScrollBar::handle:vertical {
                background: rgba(0, 229, 255, 0.45);
                border-radius: 5px;
                min-height: 28px;
            }
            QScrollBar::handle:vertical:hover, QScrollBar::handle:vertical:pressed {
                background: rgba(0, 229, 255, 0.85);
                border-radius: 5px;
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
                background-color: rgba(10, 25, 41, 0.94);
                color: #00E5FF;
                border: 1px solid #00E5FF;
                border-radius: 12px;
                font-family: "Segoe UI", "Consolas", sans-serif;
                font-size: 9pt;
                font-weight: bold;
                padding: 4px 14px;
            }
            QPushButton:hover {
                background-color: rgba(0, 229, 255, 0.25);
                color: #FFFFFF;
                border: 1px solid #80DEEA;
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
        """Direct wheel handler for the chat overlay widget."""
        self._handle_wheel(event)

    def _handle_wheel(self, event: QWheelEvent) -> None:
        """Process wheel / touchpad gestures smoothly."""
        vsb = self._scroll.verticalScrollBar()
        if not vsb:
            return

        p_delta = event.pixelDelta().y()
        a_delta = event.angleDelta().y()

        if p_delta != 0:
            # Precision trackpad scrolling
            vsb.setValue(vsb.value() - p_delta)
            event.accept()
        elif a_delta != 0:
            # Standard mouse wheel scrolling (~120 delta units per notch)
            step = int((a_delta / 120.0) * max(28, vsb.singleStep() * 3))
            vsb.setValue(vsb.value() - step)
            event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_new_msg_btn()

    def _reposition_new_msg_btn(self) -> None:
        if self._btn_new_msg.isVisible():
            btn_w = self._btn_new_msg.sizeHint().width() + 10
            btn_h = 28
            self._btn_new_msg.setGeometry(
                (self.width() - btn_w) // 2,
                self.height() - btn_h - 10,
                btn_w,
                btn_h,
            )
            self._btn_new_msg.raise_()

    # ---- Public API ---------------------------------------------------------

    def add_message(self, role: str, text: str) -> None:
        """Add a new conversation message, auto-scrolling if already at bottom."""
        if not text.strip():
            return

        was_at_bottom = self._is_near_bottom()

        self._messages.append((role, text))
        self._cards.append(_LegacyCard(opacity=1.0))
        if len(self._cards) > _MAX_CARDS:
            self._cards.pop(0)

        n = len(self._cards)
        for i, c in enumerate(self._cards):
            c.opacity = 0.3 + 0.7 * ((i + 1) / n)

        bubble = _ChatMessageBubble(role=role, text=text, theme_colors=self._theme_colors)
        bubble.installEventFilter(self)
        for child in bubble.findChildren(QWidget):
            child.installEventFilter(self)

        # Insert before stretch item at the end
        self._msg_layout.insertWidget(self._msg_layout.count() - 1, bubble)

        if was_at_bottom:
            QTimer.singleShot(20, self._scroll_to_bottom)
            self._btn_new_msg.hide()
        else:
            # User was reading older messages: do NOT disrupt view, show indicator
            self._btn_new_msg.show()
            self._reposition_new_msg_btn()

    def clear(self) -> None:
        """Clear conversation history."""
        self._messages.clear()
        self._cards.clear()
        self._btn_new_msg.hide()
        while self._msg_layout.count() > 1:
            item = self._msg_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def apply_theme(self, theme: "Theme") -> None:
        """Apply colors from Theme."""
        c = theme.colors
        self._theme_colors = {
            "hud_primary": c.hud_primary,
            "hud_secondary": c.hud_secondary,
            "hud_dim": c.hud_dim,
            "hud_text": c.hud_text,
            "hud_grid": c.hud_grid,
        }

    # ---- Internal -----------------------------------------------------------

    def _is_near_bottom(self) -> bool:
        """Return True if scroll position is within 40px of the bottom."""
        vsb = self._scroll.verticalScrollBar()
        if not vsb:
            return True
        if vsb.maximum() <= 0:
            return True
        return (vsb.maximum() - vsb.value()) <= 40

    def _scroll_to_bottom(self) -> None:
        vsb = self._scroll.verticalScrollBar()
        if vsb:
            vsb.setValue(vsb.maximum())
            self._btn_new_msg.hide()

    def _on_scroll_value_changed(self, value: int) -> None:
        if self._is_near_bottom():
            self._btn_new_msg.hide()

    def _on_new_msg_btn_clicked(self) -> None:
        self._scroll_to_bottom()
