"""Colours and font helpers for the Spidy dashboard (near-black, hairlines, one red accent)."""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont

BG = QColor(4, 4, 6)
RED = QColor(226, 36, 44)
RED_BRIGHT = QColor(255, 74, 82)
TEXT = QColor(236, 238, 244)
SOFT = QColor(190, 194, 206)
MUTED = QColor(124, 128, 140)
FAINT = QColor(80, 83, 94)
HAIR = QColor(255, 255, 255, 30)
HAIR_STRONG = QColor(255, 255, 255, 60)
GREEN = QColor(70, 214, 140)
AMBER = QColor(255, 176, 64)

FONT_FAMILY = "Segoe UI"


def font(px: float, weight: QFont.Weight = QFont.Weight.Normal, spacing: float = 0.0) -> QFont:
    """A pixel-sized font so text scales with the window like the rest of the layout."""
    f = QFont(FONT_FAMILY)
    f.setPixelSize(max(8, int(round(px))))
    f.setWeight(weight)
    if spacing:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return f


def with_alpha(color: QColor, alpha: int) -> QColor:
    c = QColor(color)
    c.setAlpha(max(0, min(255, alpha)))
    return c
