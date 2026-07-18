"""
ChatView — Scrollable conversation display
==========================================
Displays conversation bubbles in a scrollable widget.
User messages appear right-aligned with a subtle background;
Spidy messages appear left-aligned with the accent color.

Bubbles use rounded corners, auto-wrapping text, and
smooth fade-in animations when new messages arrive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPainterPath
)
from PySide6.QtWidgets import (
    QFrame, QLabel, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)


# ─── Chat message model ───────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    role: Literal["user", "assistant", "system"]
    text: str
    session_id: str = ""


# ─── ChatBubble widget ────────────────────────────────────────────────────────

class ChatBubble(QFrame):
    """
    A single message bubble.

    Parameters
    ----------
    message:
        The chat message to render.
    bg_color:
        Bubble background color.
    text_color:
        Text color.
    align_right:
        If True, bubble is pinned right (user). Left = Spidy.
    font:
        QFont to use.
    """

    def __init__(
        self,
        message: ChatMessage,
        bg_color: QColor,
        text_color: QColor,
        align_right: bool,
        font: QFont,
        border_radius: int = 14,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._bg = bg_color
        self._radius = border_radius
        self._align_right = align_right

        self.setObjectName("ChatBubble")
        # Do NOT set WA_TranslucentBackground here: combining it with
        # QGraphicsOpacityEffect causes "QPainter::begin" / "Painter not
        # active" / "Unbalanced save/restore" Qt warnings because the effect
        # renders into an offscreen pixmap and then begins a second painter
        # on top of the translucent device. The custom paintEvent fills the
        # rounded rect background directly, so transparency is not needed.

        # Role label
        role_text = "You" if message.role == "user" else "Spidy"
        role_label = QLabel(role_text, self)
        role_font = QFont(font.family(), max(8, font.pointSize() - 2))
        role_label.setFont(role_font)
        role_label.setStyleSheet(
            f"color: {text_color.name()}; opacity: 0.6;"
        )

        # Message text label
        label = QLabel(message.text, self)
        label.setFont(font)
        label.setWordWrap(True)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        label.setStyleSheet(f"color: {text_color.name()};")
        label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        label.setMaximumWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)
        layout.addWidget(role_label)
        layout.addWidget(label)

        # NOTE: We do NOT create an internal 'outer' wrapper widget here.
        # Alignment (left/right) is handled by the wrapper created in _make_bubble.
        # Creating an orphan wrapper here caused the ChatBubble to be GC'd by Qt
        # since the orphan became the de-facto owner and was deleted on scope exit.

        # No fade-in delay: the bubble is visible immediately when added to the
        # layout. A delayed opacity=0 start caused messages to appear "late" —
        # the widget existed in the layout but was invisible until the timer
        # fired 50ms later and the 250ms animation completed.

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if not painter.isActive():
            return  # Guard: skip if painter failed to initialise
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(
            0, 0, self.width(), self.height(),
            self._radius, self._radius
        )
        painter.fillPath(path, QBrush(self._bg))
        painter.end()  # Explicit end prevents Unbalanced save/restore warnings


# ─── ChatView ─────────────────────────────────────────────────────────────────

class ChatView(QScrollArea):
    """
    Scrollable chat view that holds a list of ChatBubble widgets.

    Features
    --------
    - Auto-scrolls to bottom on new messages
    - Smooth fade-in animation per bubble
    - Max-message cap with oldest-first removal to prevent memory leak
    - Theme-aware (call apply_theme() to re-style all bubbles)
    """

    _MAX_MESSAGES = 80

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.setObjectName("ChatView")
        self.setStyleSheet(
            "#ChatView { background: transparent; border: none; }"
        )

        self._container = QWidget()
        self._container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(8, 12, 8, 12)
        self._layout.setSpacing(10)
        self._layout.addStretch()
        self.setWidget(self._container)

        self._messages: list[ChatMessage] = []
        self._bubbles: list[QWidget] = []

        # Theme defaults (dark)
        self._theme = None

    # ── Public API ────────────────────────────────────────────────────────

    def add_message(self, message: ChatMessage) -> None:
        """Add a new message bubble and scroll to the bottom."""
        self._messages.append(message)

        # Enforce max cap
        while len(self._bubbles) >= self._MAX_MESSAGES:
            oldest = self._bubbles.pop(0)
            self._layout.removeWidget(oldest)
            oldest.deleteLater()
            if self._messages:
                self._messages.pop(0)

        bubble = self._make_bubble(message)
        # Insert before the trailing stretch
        self._layout.insertWidget(self._layout.count() - 1, bubble)
        self._bubbles.append(bubble)

        # Auto-scroll to bottom
        QTimer.singleShot(100, self._scroll_to_bottom)

    def clear(self) -> None:
        """Remove all messages."""
        for b in self._bubbles:
            self._layout.removeWidget(b)
            b.deleteLater()
        self._bubbles.clear()
        self._messages.clear()

    def apply_theme(self, theme) -> None:
        """Store theme for future bubbles and update scrollbar styling."""
        self._theme = theme
        c = theme.colors
        self.setStyleSheet(
            f"#ChatView {{ background: transparent; border: none; }}"
            f"QScrollBar:vertical {{ background: {c.bg_secondary}; width: 6px; }}"
            f"QScrollBar::handle:vertical {{ background: {c.scrollbar}; border-radius: 3px; }}"
            f"QScrollBar::handle:vertical:hover {{ background: {c.scrollbar_hover}; }}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}"
        )

    # ── Internal ──────────────────────────────────────────────────────────

    def _make_bubble(self, msg: ChatMessage) -> QWidget:
        is_user = msg.role == "user"

        if self._theme:
            c = self._theme.colors
            bg = QColor(c.bubble_user if is_user else c.bubble_assistant)
            fg = QColor(c.bubble_user_text if is_user else c.bubble_assistant_text)
            font = QFont(self._theme.fonts.family_primary, self._theme.fonts.size_base)
            radius = self._theme.geometry.bubble_radius
        else:
            bg = QColor("#1E1F35" if is_user else "#252640")
            fg = QColor("#C8CAE0" if is_user else "#E0E2F5")
            font = QFont("Segoe UI", 12)
            radius = 14

        # Create wrapper first so it can serve as the Qt parent of ChatBubble.
        # Without an explicit parent, Python's GC can collect the ChatBubble
        # C++ object while the layout still holds a raw pointer to it.
        wrapper = QWidget(self._container)
        wrapper.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        bubble = ChatBubble(
            message=msg,
            bg_color=bg,
            text_color=fg,
            align_right=is_user,
            font=font,
            border_radius=radius,
            parent=wrapper,  # Owned by wrapper → safe from GC
        )
        bubble.setMaximumWidth(340)

        wl = QVBoxLayout(wrapper)
        wl.setContentsMargins(0, 0, 0, 0)
        alignment = Qt.AlignmentFlag.AlignRight if is_user else Qt.AlignmentFlag.AlignLeft
        wl.setAlignment(alignment)
        wl.addWidget(bubble)
        return wrapper


    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())
