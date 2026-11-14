"""Colours and font helpers for the Spidy dashboard -- the "Cosmic Violet" theme: deep indigo-black,
hairlines, one violet accent. (The accent keeps its historical names RED / RED_BRIGHT.)"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont

BG = QColor(7, 5, 17)
RED = QColor(150, 115, 255)          # accent (violet)
RED_BRIGHT = QColor(200, 175, 255)   # bright accent
ACCENT_HEX, ACCENT_BRIGHT_HEX, ACCENT_RGB = "#9673ff", "#c8afff", "150,115,255"
DANGER = QColor(255, 74, 82)         # real warnings stay red (shutdown, failures)
GLOW = QColor(95, 45, 190)           # background glow behind the blob
GLOW_2 = QColor(30, 70, 200)         # second, bluer glow
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
