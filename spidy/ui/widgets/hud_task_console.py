"""
hud_task_console.py -- HUDTaskConsole: Autonomous agent task display
====================================================================
Shows current goal and per-step progress.
Visible only when UIState.WORKING.
Styled as a futuristic data console with amber/cyan accents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPainterPath
from PySide6.QtWidgets import QSizePolicy, QWidget

if TYPE_CHECKING:
    from spidy.ui.themes.base import Theme


@dataclass
class TaskStep:
    """Single agent task step for display."""
    description: str
    status: str  # "done" | "running" | "pending" | "failed"

    @property
    def icon(self) -> str:
        return {"done": "\u2713", "running": "\u27f3", "pending": "\u25cb", "failed": "\u2717"}.get(
            self.status, "\u25cb"
        )

    @property
    def color_hint(self) -> str:
        return {"done": "primary", "running": "amber", "pending": "dim", "failed": "error"}.get(
            self.status, "dim"
        )


class HUDTaskConsole(QWidget):
    """Futuristic task progress console."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._goal  = ""
        self._steps: list[TaskStep] = []

        self._c_primary   = QColor("#00E5FF")
        self._c_secondary = QColor("#00ACC1")
        self._c_dim       = QColor("#004D5E")
        self._c_text      = QColor("#B2EBF2")
        self._c_bg        = QColor("#0A1929")
        self._c_amber     = QColor("#E8A020")
        self._c_error     = QColor("#FF2444")

    def update_progress(self, goal: str, steps: list[TaskStep]) -> None:
        self._goal  = goal
        self._steps = list(steps)
        self.update()

    def clear(self) -> None:
        self._goal  = ""
        self._steps = []
        self.update()

    def apply_theme(self, theme: "Theme") -> None:
        c = theme.colors
        self._c_primary   = QColor(c.hud_primary)
        self._c_secondary = QColor(c.hud_secondary)
        self._c_dim       = QColor(c.hud_dim)
        self._c_text      = QColor(c.hud_text)
        self._c_bg        = QColor(c.hud_grid)
        self._c_amber     = QColor(c.working_glow)
        self._c_error     = QColor(c.accent_error)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        pad = 8

        # Glass background
        bg = QColor(self._c_bg)
        bg.setAlphaF(0.70)
        border = QColor(self._c_amber)
        border.setAlphaF(0.55)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, w, h), 8, 8)
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 1.0))
        painter.drawPath(path)

        # Header
        header_col = QColor(self._c_amber)
        header_col.setAlphaF(0.85)
        painter.setPen(QPen(header_col))
        font_hdr = QFont("Consolas", max(7, min(10, int(h * 0.075))), QFont.Weight.Bold)
        font_hdr.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        painter.setFont(font_hdr)
        painter.drawText(QRectF(pad, pad, w - pad * 2, h * 0.15), Qt.AlignmentFlag.AlignLeft, "AUTONOMOUS TASK")

        # Goal
        y_goal = h * 0.18
        font_goal = QFont("Segoe UI", max(7, min(9, int(h * 0.065))))
        painter.setFont(font_goal)
        dim_col = QColor(self._c_text)
        dim_col.setAlphaF(0.65)
        painter.setPen(QPen(dim_col))
        painter.drawText(
            QRectF(pad, y_goal, w - pad * 2, h * 0.14),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            self._goal[:80] + ("…" if len(self._goal) > 80 else ""),
        )

        # Divider
        divider_col = QColor(self._c_amber)
        divider_col.setAlphaF(0.30)
        painter.setPen(QPen(divider_col, 0.8))
        y_div = h * 0.34
        painter.drawLine(QRectF(pad, y_div, w - pad * 2, 0).topLeft(),
                         QRectF(pad, y_div, w - pad * 2, 0).topRight())

        # Steps
        visible = self._steps[-6:]  # max 6 steps
        if not visible:
            painter.end()
            return

        step_h = (h * 0.60) / max(1, len(visible))
        y_step = h * 0.37
        font_step = QFont("Consolas", max(7, min(11, int(step_h * 0.38))))
        painter.setFont(font_step)

        for step in visible:
            c_map = {
                "primary": self._c_primary,
                "amber":   self._c_amber,
                "dim":     self._c_dim,
                "error":   self._c_error,
            }
            col = QColor(c_map.get(step.color_hint, self._c_dim))
            col.setAlphaF(0.85)
            painter.setPen(QPen(col))
            painter.drawText(
                QRectF(pad, y_step, w - pad * 2, step_h),
                Qt.AlignmentFlag.AlignVCenter,
                f"{step.icon}  {step.description[:35]}",
            )
            y_step += step_h

        painter.end()

