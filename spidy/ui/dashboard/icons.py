"""
Outline icon set drawn as small SVGs (24x24 grid), rendered with QtSvg and cached.

One consistent stroke style for the whole UI. ``icon_pixmap(name, color, px)`` returns a
QPixmap; ``draw_icon`` paints one centred in a rectangle.
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_STROKE = 1.7

_ICONS: dict[str, str] = {
    "home": '<path d="M3 11.5 12 3.5l9 8"/><path d="M5.5 10v10h4.5v-6h4v6h4.5V10"/>',
    "chat": '<rect x="3" y="4" width="18" height="13" rx="2.5"/><path d="M8 17v4l4-4"/>',
    "mic": '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3M9 21h6"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18"/>',
    "folder": '<path d="M3 7.5h6l2 2.2h10V19H3z"/>',
    "apps": '<rect x="4" y="4" width="6.5" height="6.5" rx="1.5"/><rect x="13.5" y="4" width="6.5" height="6.5" rx="1.5"/><rect x="4" y="13.5" width="6.5" height="6.5" rx="1.5"/><rect x="13.5" y="13.5" width="6.5" height="6.5" rx="1.5"/>',
    "automation": '<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="8"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3"/>',
    "code": '<path d="m8 7-5 5 5 5M16 7l5 5-5 5M14 5l-4 14"/>',
    "vision": '<path d="M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4"/><circle cx="12" cy="12" r="3.5"/>',
    "memory": '<ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3"/>',
    "knowledge": '<path d="M4 5.5C6.5 4 9.5 4 12 5.5 14.5 4 17.5 4 20 5.5V19c-2.5-1.5-5.5-1.5-8 0-2.5-1.5-5.5-1.5-8 0z"/><path d="M12 5.5V19"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="7.2"/><path d="M12 2.5v2.3M12 19.2v2.3M2.5 12h2.3M19.2 12h2.3M5.3 5.3l1.6 1.6M17.1 17.1l1.6 1.6M18.7 5.3l-1.6 1.6M6.9 17.1l-1.6 1.6"/>',
    "nodes": '<circle cx="6" cy="6" r="2.2"/><circle cx="18" cy="6" r="2.2"/><circle cx="12" cy="18" r="2.2"/><path d="m8 7 3 9M16 7l-3 9M8.2 6h7.6"/>',
    "doc_plus": '<path d="M6 3h8l5 5v13H6z"/><path d="M14 3v5h5M12 11v6M9 14h6"/>',
    "search": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 5 5"/>',
    "chart": '<path d="M4 20V10M10 20V4M16 20v-8M22 20H2"/>',
    "bolt": '<path d="M13 2 5 13.5h6L10 22l9-12h-6.2z"/>',
    "chevron": '<path d="m9 6 6 6-6 6"/>',
    "monitor": '<rect x="3" y="4" width="18" height="12" rx="1.5"/><path d="M8 20h8M12 16v4"/>',
    "screenshot": '<path d="M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4"/><circle cx="12" cy="12" r="3.2"/>',
    "power": '<path d="M12 3v9"/><path d="M7 6.2a8 8 0 1 0 10 0"/>',
    "restart": '<path d="M20 12a8 8 0 1 1-2.6-5.9"/><path d="M20.5 3.5v4.6h-4.6"/>',
    "lock": '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/><path d="M12 15v2.5"/>',
    "moon": '<path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="1.5"/><rect x="9.5" y="9.5" width="5" height="5"/><path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3"/>',
    "gpu": '<rect x="2.5" y="6.5" width="19" height="10" rx="1.5"/><circle cx="9" cy="11.5" r="3"/><path d="M15 9.5h4M15 12h4M15 14.5h3M6 16.5v3M10 16.5v3M14 16.5v3"/>',
    "ram": '<rect x="2.5" y="8" width="19" height="8" rx="1.2"/><path d="M6 11v2M10 11v2M14 11v2M18 11v2M5 16v3M9 16v3M15 16v3M19 16v3"/>',
    "ssd": '<rect x="2.5" y="7.5" width="19" height="9" rx="2"/><rect x="6" y="10.5" width="6" height="3" rx=".6"/><circle cx="17.5" cy="12" r="1"/>',
    "wifi": '<path d="M2.5 9a14 14 0 0 1 19 0M5.5 12.5a9.5 9.5 0 0 1 13 0M8.7 16a5 5 0 0 1 6.6 0"/><circle cx="12" cy="19.2" r="1"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2.5M12 19v2.5M2.5 12H5M19 12h2.5M5.3 5.3l1.8 1.8M16.9 16.9l1.8 1.8M18.7 5.3l-1.8 1.8M7.1 16.9l-1.8 1.8"/>',
    "cloud": '<path d="M7 18.5a4.5 4.5 0 0 1-.6-8.96A6 6 0 0 1 18 11a3.8 3.8 0 0 1 .5 7.5z"/>',
    "minimize": '<path d="M6 12h12"/>',
    "close": '<path d="m6 6 12 12M18 6 6 18"/>',
    "send": '<path d="M4 12 20 4l-5 16-3.5-6.5z"/><path d="m11.5 13.5 3-3.5"/>',
    "check": '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
}

_cache: dict[tuple, QPixmap] = {}


def icon_names() -> list[str]:
    return sorted(_ICONS)


def icon_pixmap(name: str, color: QColor, px: int, stroke: float = _STROKE, dpr: float = 1.0) -> QPixmap:
    """Render an icon to a pixmap of ``px`` logical pixels (cached)."""
    key = (name, color.rgba(), px, round(stroke, 2), round(dpr, 2))
    cached = _cache.get(key)
    if cached is not None:
        return cached
    body = _ICONS.get(name, _ICONS["bolt"])
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color.name()}" stroke-opacity="{color.alphaF():.3f}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )
    size = max(1, int(round(px * dpr)))
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(p, QRectF(0, 0, size, size))
    p.end()
    pm.setDevicePixelRatio(dpr)
    _cache[key] = pm
    return pm


def draw_icon(painter: QPainter, name: str, rect: QRectF, color: QColor, stroke: float = _STROKE) -> None:
    """Paint an icon centred in ``rect`` (as large as fits)."""
    side = max(8, int(min(rect.width(), rect.height())))
    dpr = painter.device().devicePixelRatioF() if hasattr(painter.device(), "devicePixelRatioF") else 1.0
    pm = icon_pixmap(name, color, side, stroke, dpr)
    painter.drawPixmap(
        int(rect.x() + (rect.width() - side) / 2),
        int(rect.y() + (rect.height() - side) / 2),
        pm,
    )
