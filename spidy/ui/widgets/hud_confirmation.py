"""
hud_confirmation.py -- HUDConfirmation: Spider-Verse Tactical Action Authorization
===================================================================================
Safety-critical action authorization modal.
High-contrast glass card with Spider-red accent framing and unmistakable Allow/Deny chip controls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from spidy.ui.themes.base import Theme


class HUDConfirmation(QWidget):
    """
    Spider-Verse HUD action authorization overlay.

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

        # Spider-Verse colors
        self._c_red      = QColor("#E62429")
        self._c_success  = QColor("#2ECC71")
        self._c_text     = QColor("#F5F5F7")
        self._c_bg       = QColor("#12141C")
        self._c_error    = QColor("#FF4D4F")

        # High-contrast action buttons
        self._btn_confirm = QPushButton("AUTHORIZE", self)
        self._btn_cancel  = QPushButton("DENY",      self)
        self._style_buttons()
        self._btn_confirm.clicked.connect(self._on_confirm)
        self._btn_cancel.clicked.connect(self._on_cancel)

        self.hide()

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
        self._prompt      = prompt or "Autonomous action requires explicit user permission."
        self._layout_buttons()
        self.show()
        self.raise_()
        self.update()

    def hide_confirmation(self) -> None:
        self.hide()

    def apply_theme(self, theme: "Theme") -> None:
        c = theme.colors
        self._c_red     = QColor(c.accent_primary)
        self._c_success = QColor(c.accent_success)
        self._c_text    = QColor(c.text_primary)
        self._c_bg      = QColor(c.bg_secondary)
        self._c_error   = QColor(c.accent_error)
        self._style_buttons()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        pad = 16

        # Dark glass background with Spider-red border
        bg = QColor(self._c_bg)
        bg.setAlphaF(0.95)
        border_col = QColor(self._c_red)
        border_col.setAlphaF(0.90)

        path = QPainterPath()
        path.addRoundedRect(QRectF(2, 2, w - 4, h - 4), 10, 10)
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border_col, 2.0))
        painter.drawPath(path)

        # Red top accent bar
        accent_bar = QPainterPath()
        accent_bar.addRoundedRect(QRectF(2, 2, w - 4, 4), 2, 2)
        painter.setBrush(QBrush(border_col))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(accent_bar)

        # Corner bracket reticles
        blen = 12
        p_pen = QPen(border_col, 2.0)
        painter.setPen(p_pen)
        # Top-left
        painter.drawLine(QPointF(6, 6), QPointF(6 + blen, 6))
        painter.drawLine(QPointF(6, 6), QPointF(6, 6 + blen))
        # Top-right
        painter.drawLine(QPointF(w - 6, 6), QPointF(w - 6 - blen, 6))
        painter.drawLine(QPointF(w - 6, 6), QPointF(w - 6, 6 + blen))
        # Bottom-left
        painter.drawLine(QPointF(6, h - 6), QPointF(6 + blen, h - 6))
        painter.drawLine(QPointF(6, h - 6), QPointF(6, h - 6 - blen))
        # Bottom-right
        painter.drawLine(QPointF(w - 6, h - 6), QPointF(w - 6 - blen, h - 6))
        painter.drawLine(QPointF(w - 6, h - 6), QPointF(w - 6, h - 6 - blen))

        # Title: ⚠ ACTION AUTHORIZATION
        font_hdr = QFont("Rajdhani", max(9, min(14, int(h * 0.10))), QFont.Weight.Bold)
        font_hdr.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        painter.setFont(font_hdr)
        painter.setPen(QPen(self._c_red))
        painter.drawText(
            QRectF(pad, pad + 6, w - pad * 2, h * 0.20),
            Qt.AlignmentFlag.AlignCenter,
            "⚠  ACTION AUTHORIZATION",
        )

        # Divider
        div_col = QColor(255, 255, 255, 25)
        painter.setPen(QPen(div_col, 1.0))
        y_div = h * 0.28
        painter.drawLine(QPointF(pad, y_div), QPointF(w - pad, y_div))

        # Description
        font_desc = QFont("Inter", max(8, min(12, int(h * 0.085))), QFont.Weight.DemiBold)
        painter.setFont(font_desc)
        painter.setPen(QPen(self._c_text))
        painter.drawText(
            QRectF(pad, h * 0.30, w - pad * 2, h * 0.24),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            self._description,
        )

        # Prompt
        font_prompt = QFont("JetBrains Mono", max(7, min(10, int(h * 0.07))))
        painter.setFont(font_prompt)
        dim_col = QColor(self._c_text)
        dim_col.setAlphaF(0.65)
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

    def _layout_buttons(self) -> None:
        w, h = self.width(), self.height()
        btn_w = min(125, int(w * 0.30))
        btn_h = max(30, int(h * 0.15))
        gap = 14
        total = btn_w * 2 + gap
        x_left = (w - total) // 2
        y_btn = int(h * 0.72)
        self._btn_cancel.setGeometry(x_left, y_btn, btn_w, btn_h)
        self._btn_confirm.setGeometry(x_left + btn_w + gap, y_btn, btn_w, btn_h)

    def _style_buttons(self) -> None:
        red = self._c_red.name()
        succ = self._c_success.name()
        self._btn_confirm.setStyleSheet(f"""
            QPushButton {{
                background: {succ}22;
                border: 1.5px solid {succ};
                color: {succ};
                font-family: 'JetBrains Mono', Consolas, monospace;
                font-size: 11px;
                font-weight: bold;
                border-radius: 6px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: {succ}44;
                color: #FFFFFF;
            }}
            QPushButton:pressed {{
                background: {succ}88;
            }}
        """)
        self._btn_cancel.setStyleSheet(f"""
            QPushButton {{
                background: {red}22;
                border: 1.5px solid {red};
                color: {red};
                font-family: 'JetBrains Mono', Consolas, monospace;
                font-size: 11px;
                font-weight: bold;
                border-radius: 6px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: {red}44;
                color: #FFFFFF;
            }}
            QPushButton:pressed {{
                background: {red}88;
            }}
        """)

    def _on_confirm(self) -> None:
        self.confirmed.emit(self._task_id, self._goal_id)
        self.hide()

    def _on_cancel(self) -> None:
        self.cancelled.emit(self._task_id, self._goal_id)
        self.hide()
