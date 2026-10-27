"""
ConfirmationCard -- Spider-Verse Interactive Action Authorization Overlay
==========================================================================
Appears when a task requires user approval before AutonomousAgent proceeds.
High-contrast tactical card with Spider-red border and explicit ALLOW/DENY buttons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPainterPath, QPen,
)
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

if TYPE_CHECKING:
    from spidy.ui.themes.base import Theme


class ConfirmationCard(QWidget):
    """
    Glass confirmation card overlaid on the content area.

    Signals
    -------
    confirmed(task_id: str, goal_id: str)
    cancelled(task_id: str, goal_id: str)
    """

    confirmed = Signal(str, str)   # task_id, goal_id
    cancelled = Signal(str, str)   # task_id, goal_id

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setVisible(False)

        self._task_id = ""
        self._goal_id = ""

        self._bg_color     = QColor("#12141C")
        self._border_color = QColor("#E62429")
        self._radius       = 14

        # Layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(0)

        inner = QVBoxLayout()
        inner.setSpacing(12)

        # Header label
        self._header = QLabel("⚠  ACTION AUTHORIZATION")
        self._header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._header.setStyleSheet(
            "color: #E62429; font-family: 'Rajdhani', sans-serif; font-size: 13px; font-weight: bold; letter-spacing: 2px;"
        )
        inner.addWidget(self._header)

        # Task description
        self._desc_label = QLabel("")
        self._desc_label.setWordWrap(True)
        self._desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._desc_label.setStyleSheet(
            "color: #F5F5F7; font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 600;"
        )
        inner.addWidget(self._desc_label)

        # Prompt / detail text
        self._prompt_label = QLabel("")
        self._prompt_label.setWordWrap(True)
        self._prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._prompt_label.setStyleSheet(
            "color: #9A9AA5; font-family: 'JetBrains Mono', monospace; font-size: 11px;"
        )
        inner.addWidget(self._prompt_label)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(14)

        self._btn_cancel = QPushButton("DENY")
        self._btn_cancel.setFixedHeight(36)
        self._btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_cancel.clicked.connect(self._on_cancel)
        self._style_button(self._btn_cancel, primary=False)
        btn_row.addWidget(self._btn_cancel)

        self._btn_confirm = QPushButton("AUTHORIZE")
        self._btn_confirm.setFixedHeight(36)
        self._btn_confirm.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_confirm.clicked.connect(self._on_confirm)
        self._style_button(self._btn_confirm, primary=True)
        btn_row.addWidget(self._btn_confirm)

        inner.addLayout(btn_row)
        outer.addLayout(inner)

    def show_confirmation(
        self,
        task_id: str,
        goal_id: str,
        description: str,
        prompt: str = "",
    ) -> None:
        """Populate card content and make it visible."""
        self._task_id = task_id
        self._goal_id = goal_id
        self._desc_label.setText(description)
        self._prompt_label.setText(prompt or "Autonomous action requires explicit user permission.")
        self._prompt_label.setVisible(True)
        self.setVisible(True)
        self.raise_()

    def hide_confirmation(self) -> None:
        """Hide the card (call after user responds or goal cancelled)."""
        self.setVisible(False)
        self._task_id = ""
        self._goal_id = ""

    def apply_theme(self, theme: "Theme") -> None:
        c = theme.colors
        self._bg_color     = QColor(c.bg_secondary)
        self._border_color = QColor(c.accent_primary)
        self._header.setStyleSheet(
            f"color: {c.accent_primary}; font-family: 'Rajdhani', sans-serif; font-size: {theme.fonts.size_sm}px;"
            " font-weight: bold; letter-spacing: 2px;"
        )
        self._desc_label.setStyleSheet(
            f"color: {c.text_primary}; font-family: 'Inter', sans-serif; font-size: {theme.fonts.size_md}px; font-weight: 600;"
        )
        self._prompt_label.setStyleSheet(
            f"color: {c.text_secondary}; font-family: 'JetBrains Mono', monospace; font-size: {theme.fonts.size_xs}px;"
        )
        self._style_button(self._btn_cancel, primary=False, theme=theme)
        self._style_button(self._btn_confirm, primary=True, theme=theme)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), self._radius, self._radius)

        bg = QColor(self._bg_color)
        bg.setAlphaF(0.95)
        painter.fillPath(path, QBrush(bg))

        border = QColor(self._border_color)
        border.setAlphaF(0.85)
        painter.setPen(QPen(border, 1.8))
        painter.drawPath(path)

        painter.end()

    def _on_confirm(self) -> None:
        self.confirmed.emit(self._task_id, self._goal_id)
        self.hide_confirmation()

    def _on_cancel(self) -> None:
        self.cancelled.emit(self._task_id, self._goal_id)
        self.hide_confirmation()

    def _style_button(
        self,
        btn: QPushButton,
        *,
        primary: bool,
        theme: "Theme | None" = None,
    ) -> None:
        if primary:
            color = "#2ECC71"  # Tactical green for authorize
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {color}22;
                    color: {color};
                    border: 1.5px solid {color};
                    border-radius: 6px;
                    font-family: 'JetBrains Mono', Consolas, monospace;
                    font-size: 11px;
                    font-weight: bold;
                    letter-spacing: 1px;
                }}
                QPushButton:hover {{
                    background: {color}44;
                    color: #FFFFFF;
                }}
            """)
        else:
            color = "#FF4D4F"  # Red for deny
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {color}22;
                    color: {color};
                    border: 1.5px solid {color};
                    border-radius: 6px;
                    font-family: 'JetBrains Mono', Consolas, monospace;
                    font-size: 11px;
                    font-weight: bold;
                    letter-spacing: 1px;
                }}
                QPushButton:hover {{
                    background: {color}44;
                    color: #FFFFFF;
                }}
            """)
