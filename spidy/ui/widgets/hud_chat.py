"""
hud_chat.py -- HUDChatOverlay: Floating compact conversation cards
==================================================================
Replaces the permanent left-column chat panel with floating glass cards.

At most 3 message pairs visible.
Oldest card fades to 30%, middle to 65%, newest at 100%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPainterPath
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from spidy.ui.themes.base import Theme

_MAX_CARDS = 3   # max visible cards


@dataclass
class _Card:
    role: str   # "user" | "assistant"
    text: str
    opacity: float = 1.0


class HUDChatOverlay(QWidget):
    """Floating compact conversation cards."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._cards: list[_Card] = []
        self._messages: list[tuple[str, str]] = []  # (role, text)

        self._c_primary   = QColor("#00E5FF")
        self._c_secondary = QColor("#00ACC1")
        self._c_dim       = QColor("#004D5E")
        self._c_text      = QColor("#B2EBF2")
        self._c_bg        = QColor("#0A1929")
        self._c_user      = QColor("#1A2C3D")

    # ---- Public API ---------------------------------------------------------

    def add_message(self, role: str, text: str) -> None:
        self._messages.append((role, text))
        self._rebuild_cards()
        self.update()

    def clear(self) -> None:
        self._messages.clear()
        self._cards.clear()
        self.update()

    def apply_theme(self, theme: "Theme") -> None:
        c = theme.colors
        self._c_primary   = QColor(c.hud_primary)
        self._c_secondary = QColor(c.hud_secondary)
        self._c_dim       = QColor(c.hud_dim)
        self._c_text      = QColor(c.hud_text)
        self._c_bg        = QColor(c.hud_grid)
        self.update()

    # ---- Painting -----------------------------------------------------------

    def paintEvent(self, event) -> None:
        if not self._cards:
            return
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        card_h = min(60, self.height() // max(1, len(self._cards)))
        padding = 8
        y = self.height() - card_h * len(self._cards) - padding

        for card in self._cards:
            self._draw_card(painter, card, padding, y, w - padding * 2, card_h)
            y += card_h + 4

        painter.end()

    def _draw_card(self, p: QPainter, card: _Card, x, y, w, h) -> None:
        alpha = card.opacity

        # Glass background
        bg = QColor(self._c_bg)
        bg.setAlphaF(alpha * 0.80)
        border_col = QColor(
            self._c_primary if card.role == "assistant" else self._c_dim
        )
        border_col.setAlphaF(alpha * 0.60)

        path = QPainterPath()
        path.addRoundedRect(QRectF(x, y, w, h), 6, 6)
        p.setBrush(QBrush(bg))
        p.setPen(QPen(border_col, 0.8))
        p.drawPath(path)

        # Role label
        role_col = QColor(self._c_primary if card.role == "assistant" else self._c_dim)
        role_col.setAlphaF(alpha * 0.90)
        p.setPen(QPen(role_col))
        font_label = QFont("Consolas", max(7, min(10, int(h * 0.20))))
        font_label.setBold(True)
        p.setFont(font_label)
        role_txt = "SPIDY" if card.role == "assistant" else "YOU"
        p.drawText(QRectF(x + 8, y + 3, 60, h * 0.4), Qt.AlignmentFlag.AlignLeft, role_txt)

        # Message text
        text_col = QColor(self._c_text)
        text_col.setAlphaF(alpha * 0.85)
        p.setPen(QPen(text_col))
        font_text = QFont("Segoe UI", max(7, min(11, int(h * 0.25))))
        p.setFont(font_text)
        p.drawText(
            QRectF(x + 8, y + h * 0.4, w - 16, h * 0.55),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            card.text[:120] + ("…" if len(card.text) > 120 else ""),
        )

    # ---- Internal -----------------------------------------------------------

    def _rebuild_cards(self) -> None:
        recent = self._messages[-_MAX_CARDS:]
        self._cards = []
        n = len(recent)
        opacities = {1: [1.0], 2: [0.60, 1.0], 3: [0.30, 0.65, 1.0]}
        ops = opacities.get(n, [1.0] * n)
        for i, (role, text) in enumerate(recent):
            self._cards.append(_Card(role=role, text=text, opacity=ops[i]))

