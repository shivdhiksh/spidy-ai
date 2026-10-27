"""Shared base widget: everything is designed in reference pixels (1536 x 1024) and scaled."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter
from PySide6.QtWidgets import QWidget

from spidy.ui.dashboard import palette as pal

REF_W, REF_H = 1536.0, 1024.0


def text(p: QPainter, rect: QRectF, s: str, font: QFont, color: QColor,
         align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter) -> None:
    p.setFont(font)
    p.setPen(color)
    if not (align & Qt.AlignmentFlag.AlignHCenter) or rect.width() < 400:
        s = QFontMetricsF(font).elidedText(s, Qt.TextElideMode.ElideRight, max(10.0, rect.width()))
    p.drawText(rect, int(align), s)


class RefWidget(QWidget):
    """A transparent widget whose paint code uses reference pixels; ``k`` is the window scale."""

    def __init__(self, ref_w: float, ref_h: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ref_w, self.ref_h = ref_w, ref_h
        self.k = 1.0
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)

    def set_k(self, k: float) -> None:
        self.k = k
        self.update()

    def ref_pos(self, event) -> QPointF:
        pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
        return QPointF(pos.x() / self.k, pos.y() / self.k)

    def begin(self) -> QPainter:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        p.scale(self.k, self.k)
        return p

    @staticmethod
    def f(px: float, weight: QFont.Weight = QFont.Weight.Normal, spacing: float = 0.0) -> QFont:
        return pal.font(px, weight, spacing)
