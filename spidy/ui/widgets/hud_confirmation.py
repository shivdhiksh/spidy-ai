"""
hud_confirmation.py -- HUDConfirmation: Futuristic action authorization overlay
================================================================================
Amber ⚠ ACTION AUTHORIZATION panel.
Shown when autonomous agent needs user approval.
Completely replaces the previous Windows-looking dialog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPainterPath
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect, QPushButton, QVBoxLayout, QWidget,
)

if TYPE_CHECKING:
    from spidy.ui.themes.base import Theme


class HUDConfirmation(QWidget):
    """
    Futuristic HUD confirmation overlay.

    Signals
    -------
    confirmed(task_id: str, goal_id: str)
    cancelled(task_id: str, goal_id: str)
    """

    confirmed = Signal(str, str)
    cancelled = Signal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._task_id = ""
        self._goal_id = ""
        self._description = ""
        self._prompt = ""

        self._c_amber    = QColor("#E8A020")
        self._c_primary  = QColor("#00E5FF")
        self._c_text     = QColor("#B2EBF2")
        self._c_bg       = QColor("#0D0D12")
        self._c_error    = QColor("#FF2444")

        # Opacity
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity_effect)

        # Buttons (will be positioned in resizeEvent)
        self._btn_confirm = QPushButton("CONFIRM", self)
        self._btn_cancel  = QPushButton("CANCEL",  self)
        self._style_buttons()
        self._btn_confirm.clicked.connect(self._on_confirm)
        self._btn_cancel.clicked.connect(self._on_cancel)

        self.hide()

    # ---- Public API ---------------------------------------------------------

    def show_confirmation(
        self,
        task_id: str,
        goal_id: str,
        description: str,
        prompt: str = "",
    ) -> None:
        self._task_id     = task_id
        self._goal_id     = goal_id
        self._description = description
        self._prompt      = prompt or "This action requires your authorization."
        self._layout_buttons()
        self.show()
        self.raise_()
        self.update()

    def hide_confirmation(self) -> None:
        self.hide()

    def apply_theme(self, theme: "Theme") -> None:
        c = theme.colors
        self._c_amber   = QColor(c.working_glow)
        self._c_primary = QColor(c.hud_primary)
        self._c_text    = QColor(c.hud_text)
        self._c_bg      = QColor(c.hud_grid)
        self._c_error   = QColor(c.accent_error)
        self._style_buttons()
        self.update()

    # ---- Painting -----------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        pad = 16

        # Dark glass background with amber border
        bg = QColor(self._c_bg)
        bg.setAlphaF(0.93)
        border_col = QColor(self._c_amber)
        border_col.setAlphaF(0.85)

        path = QPainterPath()
        path.addRoundedRect(QRectF(2, 2, w - 4, h - 4), 10, 10)
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border_col, 2.0))
        painter.drawPath(path)

        # Amber top accent bar
        accent_bar = QPainterPath()
        accent_bar.addRoundedRect(QRectF(2, 2, w - 4, 4), 2, 2)
        painter.setBrush(QBrush(border_col))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(accent_bar)

        # ⚠ ACTION AUTHORIZATION
        font_hdr = QFont("Consolas", max(9, min(14, int(h * 0.10))), QFont.Weight.Bold)
        font_hdr.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        painter.setFont(font_hdr)
        hdr_col = QColor(self._c_amber)
        hdr_col.setAlphaF(0.90)
        painter.setPen(QPen(hdr_col))
        painter.drawText(
            QRectF(pad, pad + 6, w - pad * 2, h * 0.20),
            Qt.AlignmentFlag.AlignCenter,
            "\u26a0  ACTION AUTHORIZATION",
        )

        # Divider
        div_col = QColor(self._c_amber)
        div_col.setAlphaF(0.30)
        painter.setPen(QPen(div_col, 0.8))
        y_div = h * 0.28
        painter.drawLine(QRectF(pad, y_div, w - pad * 2, 0).topLeft(),
                         QRectF(pad, y_div, w - pad * 2, 0).topRight())

        # Description
        font_desc = QFont("Segoe UI", max(8, min(12, int(h * 0.085))))
        font_desc.setBold(True)
        painter.setFont(font_desc)
        text_col = QColor(self._c_text)
        text_col.setAlphaF(0.90)
        painter.setPen(QPen(text_col))
        painter.drawText(
            QRectF(pad, h * 0.30, w - pad * 2, h * 0.24),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            self._description,
        )

        # Prompt
        font_prompt = QFont("Segoe UI", max(7, min(10, int(h * 0.07))))
        painter.setFont(font_prompt)
        dim_col = QColor(self._c_text)
        dim_col.setAlphaF(0.55)
        painter.setPen(QPen(dim_col))
        painter.drawText(
            QRectF(pad, h * 0.54, w - pad * 2, h * 0.15),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            self._prompt,
        )

        painter.end()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._layout_buttons()

    # ---- Internal -----------------------------------------------------------

    def _layout_buttons(self) -> None:
        w, h = self.width(), self.height()
        btn_w = min(120, int(w * 0.28))
        btn_h = max(28, int(h * 0.14))
        gap = 12
        total = btn_w * 2 + gap
        x_left = (w - total) // 2
        y_btn = int(h * 0.70)
        self._btn_cancel.setGeometry(x_left, y_btn, btn_w, btn_h)
        self._btn_confirm.setGeometry(x_left + btn_w + gap, y_btn, btn_w, btn_h)

    def _style_buttons(self) -> None:
        amber = self._c_amber.name()
        primary = self._c_primary.name()
        bg = self._c_bg.name()
        text = self._c_text.name()
        self._btn_confirm.setStyleSheet(f"""
            QPushButton {{
                background: {amber}33;
                border: 2px solid {amber};
                color: {amber};
                font-family: Consolas;
                font-size: 10px;
                font-weight: bold;
                border-radius: 4px;
                letter-spacing: 2px;
            }}
            QPushButton:hover {{ background: {amber}55; }}
            QPushButton:pressed {{ background: {amber}88; }}
        """)
        self._btn_cancel.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {primary}66;
                color: {text};
                font-family: Consolas;
                font-size: 10px;
                border-radius: 4px;
                letter-spacing: 2px;
            }}
            QPushButton:hover {{ border-color: {primary}; }}
        """)

    def _on_confirm(self) -> None:
        self.confirmed.emit(self._task_id, self._goal_id)
        self.hide()

    def _on_cancel(self) -> None:
        self.cancelled.emit(self._task_id, self._goal_id)
        self.hide()

