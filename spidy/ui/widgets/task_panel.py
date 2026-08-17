"""
TaskPanel -- Autonomous task progress display
=============================================
Shows the current goal and step-by-step task progress when the
AutonomousAgent is executing.  Only visible in UIState.WORKING.

Visual design
-------------
- Translucent glass sub-panel with rounded corners
- Header: goal description (truncated if too long)
- Scrollable step list with status icons:
    checkmark  completed
    spinning   in progress
    circle     pending
    x          failed
- Compact, minimal text -- the orb is the visual focus
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPainterPath, QPen,
)
from PySide6.QtWidgets import (
    QFrame, QLabel, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

if TYPE_CHECKING:
    from spidy.ui.themes.base import Theme


# ── Step status ────────────────────────────────────────────────────────────────

@dataclass
class TaskStep:
    """A single step in a goal's task list."""
    description: str
    status: Literal["pending", "running", "done", "failed"] = "pending"

    @property
    def icon(self) -> str:
        return {
            "pending": "\u25cb",     # ○
            "running": "\u27f3",     # ⟳
            "done":    "\u2713",     # ✓
            "failed":  "\u2717",     # ✗
        }.get(self.status, "\u25cb")

    @property
    def icon_color(self) -> str:
        return {
            "pending": "#6B6D88",
            "running": "#00D4FF",
            "done":    "#00E96A",
            "failed":  "#FF4D6A",
        }.get(self.status, "#6B6D88")


# ── StepRow widget ─────────────────────────────────────────────────────────────

class _StepRow(QWidget):
    """Single step row: [icon] [description]"""

    def __init__(self, step: TaskStep, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._step = step
        self._spin_phase = 0.0

        layout_h = QVBoxLayout(self)
        layout_h.setContentsMargins(4, 2, 4, 2)
        layout_h.setSpacing(0)

        from PySide6.QtWidgets import QHBoxLayout
        row = QHBoxLayout()
        row.setSpacing(8)

        self._icon_label = QLabel(step.icon)
        self._icon_label.setFixedWidth(20)
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_label.setStyleSheet(f"color: {step.icon_color}; font-size: 13px;")
        row.addWidget(self._icon_label)

        self._desc_label = QLabel(step.description)
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet("color: #B0B2C8; font-size: 12px;")
        self._desc_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        row.addWidget(self._desc_label)
        layout_h.addLayout(row)

    def update_step(self, step: TaskStep) -> None:
        self._step = step
        self._icon_label.setText(step.icon)
        self._icon_label.setStyleSheet(f"color: {step.icon_color}; font-size: 13px;")
        self._desc_label.setText(step.description)

    def apply_theme(self, theme: "Theme") -> None:
        c = theme.colors
        color = self._step.icon_color
        self._icon_label.setStyleSheet(f"color: {color}; font-size: {theme.fonts.size_sm}px;")
        self._desc_label.setStyleSheet(
            f"color: {c.text_secondary}; font-size: {theme.fonts.size_sm}px;"
        )


# ── TaskPanel ──────────────────────────────────────────────────────────────────

class TaskPanel(QWidget):
    """
    Task progress panel for the WORKING state.

    Usage
    -----
        panel.update_progress(
            goal="Open Edge and search YouTube",
            steps=[
                TaskStep("Open Edge",            "done"),
                TaskStep("Navigate to YouTube",  "done"),
                TaskStep("Search for query",     "running"),
                TaskStep("Verify result",        "pending"),
            ]
        )
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumWidth(160)

        self._bg_color     = QColor("#12131F")
        self._border_color = QColor("#E8A020")   # gold for WORKING
        self._radius       = 12

        # Build layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Inner content widget (provides a margin inside the painted border)
        self._content = QWidget(self)
        self._content.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        inner = QVBoxLayout(self._content)
        inner.setContentsMargins(12, 10, 12, 10)
        inner.setSpacing(6)

        # Goal header
        self._goal_label = QLabel("GOAL")
        self._goal_label.setStyleSheet(
            "color: #E8A020; font-size: 10px; letter-spacing: 1px;"
        )
        inner.addWidget(self._goal_label)

        self._goal_desc = QLabel("")
        self._goal_desc.setWordWrap(True)
        self._goal_desc.setStyleSheet("color: #E8E9F0; font-size: 12px;")
        inner.addWidget(self._goal_desc)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2A2B45;")
        sep.setFixedHeight(1)
        inner.addWidget(sep)

        # Scrollable step list
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("background: transparent; border: none;")
        self._scroll.setMaximumHeight(160)

        self._step_container = QWidget()
        self._step_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._step_layout = QVBoxLayout(self._step_container)
        self._step_layout.setContentsMargins(0, 0, 0, 0)
        self._step_layout.setSpacing(4)
        self._step_layout.addStretch()
        self._scroll.setWidget(self._step_container)
        inner.addWidget(self._scroll, 1)

        outer.addWidget(self._content)

        self._rows: list[_StepRow] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def update_progress(self, goal: str, steps: list[TaskStep]) -> None:
        """Refresh the goal description and task step list."""
        self._goal_desc.setText(goal)

        # Remove old rows
        for row in self._rows:
            self._step_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

        # Insert new rows before the trailing stretch
        for step in steps:
            row = _StepRow(step, self._step_container)
            # Insert before stretch
            self._step_layout.insertWidget(self._step_layout.count() - 1, row)
            self._rows.append(row)

    def apply_theme(self, theme: "Theme") -> None:
        c = theme.colors
        self._bg_color     = QColor(c.bg_secondary)
        self._border_color = QColor(c.working_glow)
        self._goal_label.setStyleSheet(
            f"color: {c.working_glow}; font-size: {theme.fonts.size_xs}px;"
            " letter-spacing: 1px;"
        )
        self._goal_desc.setStyleSheet(
            f"color: {c.text_primary}; font-size: {theme.fonts.size_sm}px;"
        )
        for row in self._rows:
            row.apply_theme(theme)
        self.update()

    # ── Painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), self._radius, self._radius)

        bg = QColor(self._bg_color)
        bg.setAlphaF(0.75)
        painter.fillPath(path, QBrush(bg))

        border = QColor(self._border_color)
        border.setAlphaF(0.45)
        painter.setPen(QPen(border, 1.0))
        painter.drawPath(path)

        painter.end()
