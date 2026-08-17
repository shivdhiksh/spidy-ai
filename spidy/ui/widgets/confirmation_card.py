"""
ConfirmationCard -- Interactive confirmation overlay
====================================================
Appears when a task requires user approval before the AutonomousAgent
can proceed.  Floats over the content area with a clear glass card.

Signals
-------
confirmed(task_id: str, goal_id: str)  -- User pressed CONFIRM
cancelled(task_id: str, goal_id: str)  -- User pressed CANCEL

Design
------
- Frosted glass card with amber border (high-contrast -- clearly important)
- Task description + optional prompt text
- Two full-width buttons: CANCEL (secondary) and CONFIRM (primary)
- Does NOT bypass authority logic -- it only publishes events
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

    Usage
    -----
        card.show_confirmation(
            task_id="t1",
            goal_id="g1",
            description="Upload photo to Instagram",
            prompt="This action requires your confirmation."
        )
        card.confirmed.connect(on_confirmed)
        card.cancelled.connect(on_cancelled)
    """

    confirmed = Signal(str, str)   # task_id, goal_id
    cancelled = Signal(str, str)   # task_id, goal_id

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setVisible(False)

        self._task_id = ""
        self._goal_id = ""

        self._bg_color     = QColor("#1A1B2E")
        self._border_color = QColor("#E8A020")
        self._radius       = 16

        # Layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(0)

        inner = QVBoxLayout()
        inner.setSpacing(12)

        # Header label
        self._header = QLabel("\u26a0  CONFIRMATION REQUIRED")
        self._header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._header.setStyleSheet(
            "color: #E8A020; font-size: 11px; letter-spacing: 2px;"
        )
        inner.addWidget(self._header)

        # Task description
        self._desc_label = QLabel("")
        self._desc_label.setWordWrap(True)
        self._desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._desc_label.setStyleSheet(
            "color: #E8E9F0; font-size: 14px;"
        )
        inner.addWidget(self._desc_label)

        # Prompt / detail text
        self._prompt_label = QLabel("")
        self._prompt_label.setWordWrap(True)
        self._prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._prompt_label.setStyleSheet(
            "color: #8A8BA0; font-size: 12px;"
        )
        inner.addWidget(self._prompt_label)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self._btn_cancel = QPushButton("CANCEL")
        self._btn_cancel.setFixedHeight(40)
        self._btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_cancel.clicked.connect(self._on_cancel)
        self._style_button(self._btn_cancel, primary=False)
        btn_row.addWidget(self._btn_cancel)

        self._btn_confirm = QPushButton("CONFIRM")
        self._btn_confirm.setFixedHeight(40)
        self._btn_confirm.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_confirm.clicked.connect(self._on_confirm)
        self._style_button(self._btn_confirm, primary=True)
        btn_row.addWidget(self._btn_confirm)

        inner.addLayout(btn_row)
        outer.addLayout(inner)

    # ---- Public API ----------------------------------------------------------

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
        self._prompt_label.setText(prompt)
        self._prompt_label.setVisible(bool(prompt))
        self.setVisible(True)
        self.raise_()

    def hide_confirmation(self) -> None:
        """Hide the card (call after user responds or goal cancelled)."""
        self.setVisible(False)
        self._task_id = ""
        self._goal_id = ""

    def apply_theme(self, theme: "Theme") -> None:
        c = theme.colors
        self._bg_color     = QColor(c.bg_primary)
        self._border_color = QColor(c.working_glow)
        self._header.setStyleSheet(
            f"color: {c.working_glow}; font-size: {theme.fonts.size_xs}px;"
            " letter-spacing: 2px;"
        )
        self._desc_label.setStyleSheet(
            f"color: {c.text_primary}; font-size: {theme.fonts.size_md}px;"
        )
        self._prompt_label.setStyleSheet(
            f"color: {c.text_muted}; font-size: {theme.fonts.size_sm}px;"
        )
        self._style_button(self._btn_cancel, primary=False, theme=theme)
        self._style_button(self._btn_confirm, primary=True, theme=theme)
        self.update()

    # ---- Painting ------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), self._radius, self._radius)

        bg = QColor(self._bg_color)
        bg.setAlphaF(0.96)
        painter.fillPath(path, QBrush(bg))

        border = QColor(self._border_color)
        border.setAlphaF(0.70)
        painter.setPen(QPen(border, 1.5))
        painter.drawPath(path)

        painter.end()

    # ---- Internal ------------------------------------------------------------

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
        if theme:
            c = theme.colors
            if primary:
                bg = c.working_glow
                fg = "#0D0E1A"
                hover_bg = c.accent_primary
            else:
                bg = "transparent"
                fg = c.text_muted
                hover_bg = c.bg_tertiary
        else:
            if primary:
                bg = "#E8A020"
                fg = "#0D0E1A"
                hover_bg = "#6C63FF"
            else:
                bg = "transparent"
                fg = "#6B6D88"
                hover_bg = "#1A1B2E"

        btn.setStyleSheet(
            f"QPushButton {{"
            f"  background: {bg};"
            f"  color: {fg};"
            f"  border: 1px solid {'#E8A020' if primary else '#2A2B45'};"
            f"  border-radius: 8px;"
            f"  font-size: 12px;"
            f"  letter-spacing: 1px;"
            f"  font-weight: bold;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background: {hover_bg};"
            f"  color: {'#FFFFFF' if primary else '#E8E9F0'};"
            f"}}"
        )
