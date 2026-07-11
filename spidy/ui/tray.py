"""
SystemTrayManager — Windows system tray integration
=====================================================
Creates a QSystemTrayIcon with Spidy's icon and a context menu.

Tray icon interactions
-----------------------
Left-click          Toggle overlay visibility
Right-click         Open context menu

Context menu items
------------------
Show / Hide         Toggle overlay
─────────────
🌙 Dark mode        Switch to dark theme
☀  Light mode       Switch to light theme
─────────────
⚙  Settings         Open settings panel
─────────────
✕  Exit Spidy       Quit the application

The tray manager communicates through Qt signals only —
it never imports Brain or Voice modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import (
    QAction, QColor, QIcon, QImage, QPainter, QPainterPath, QBrush
)
from PySide6.QtWidgets import QMenu, QSystemTrayIcon, QApplication

if TYPE_CHECKING:
    from spidy.ui.themes.base import Theme


def _make_spidy_icon(size: int = 32, color: str = "#6C63FF") -> QIcon:
    """
    Generate a vector Spidy tray icon on the fly using QPainter.

    Draws a rounded-square background with a bold 'S' lettermark.
    No file assets required.
    """
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)

    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Rounded-square background
    path = QPainterPath()
    path.addRoundedRect(0, 0, size, size, size * 0.3, size * 0.3)
    painter.fillPath(path, QBrush(QColor(color)))

    # 'S' lettermark
    from PySide6.QtGui import QFont
    font = QFont("Segoe UI", int(size * 0.45), QFont.Weight.Bold)
    painter.setFont(font)
    painter.setPen(QColor("#FFFFFF"))
    painter.drawText(img.rect(), Qt.AlignmentFlag.AlignCenter, "S")
    painter.end()

    return QIcon(img)


class SystemTrayManager(QObject):
    """
    Manages the Spidy system tray icon and context menu.

    Signals
    -------
    show_overlay_requested      User clicked tray icon or "Show" menu item
    hide_overlay_requested      User clicked "Hide" menu item
    theme_change_requested      str — "dark" or "light"
    settings_requested          User clicked "Settings"
    quit_requested              User clicked "Exit Spidy"
    """

    show_overlay_requested = Signal()
    hide_overlay_requested = Signal()
    theme_change_requested = Signal(str)
    settings_requested = Signal()
    quit_requested = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._overlay_visible = True
        self._tray: QSystemTrayIcon | None = None
        self._icon_color = "#6C63FF"

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Create and show the system tray icon.

        Returns
        -------
        bool
            False if the system does not support tray icons.
        """
        if not QSystemTrayIcon.isSystemTrayAvailable():
            from spidy.logging.logger import get_logger
            get_logger(__name__).warning(
                "System tray not available on this platform."
            )
            return False

        self._tray = QSystemTrayIcon(self)
        self._tray.setIcon(_make_spidy_icon(32, self._icon_color))
        self._tray.setToolTip("Spidy — AI Companion  |  Ctrl+Space to toggle")
        self._tray.activated.connect(self._on_activated)

        self._tray.setContextMenu(self._build_menu())
        self._tray.show()
        return True

    def stop(self) -> None:
        """Remove the tray icon."""
        if self._tray:
            self._tray.hide()
            self._tray = None

    def set_overlay_visible(self, visible: bool) -> None:
        """Update the tray's knowledge of overlay visibility."""
        self._overlay_visible = visible
        self._update_menu()

    def show_message(
        self,
        title: str,
        message: str,
        level: str = "info",
        duration_ms: int = 3000,
    ) -> None:
        """Display a native OS balloon notification from the tray icon."""
        if self._tray is None:
            return
        icon_map = {
            "info": QSystemTrayIcon.MessageIcon.Information,
            "success": QSystemTrayIcon.MessageIcon.Information,
            "warning": QSystemTrayIcon.MessageIcon.Warning,
            "error": QSystemTrayIcon.MessageIcon.Critical,
        }
        self._tray.showMessage(
            title, message,
            icon_map.get(level, QSystemTrayIcon.MessageIcon.Information),
            duration_ms,
        )

    def apply_theme(self, theme: "Theme") -> None:
        """Update tray icon color to match theme accent."""
        self._icon_color = theme.colors.tray_icon_bg
        if self._tray:
            self._tray.setIcon(_make_spidy_icon(32, self._icon_color))

    # ── Internal ──────────────────────────────────────────────────────────

    def _build_menu(self) -> QMenu:
        menu = QMenu()
        menu.setStyleSheet(
            "QMenu {"
            "  background-color: #12131F;"
            "  color: #E8E9F0;"
            "  border: 1px solid #2A2B45;"
            "  border-radius: 8px;"
            "  padding: 4px;"
            "}"
            "QMenu::item { padding: 6px 20px; border-radius: 4px; }"
            "QMenu::item:selected { background-color: #1A1B2E; }"
            "QMenu::separator { height: 1px; background: #2A2B45; margin: 4px 10px; }"
        )

        # Toggle visibility
        self._toggle_action = QAction(
            "Hide Spidy" if self._overlay_visible else "Show Spidy",
            menu
        )
        self._toggle_action.triggered.connect(self._on_toggle)
        menu.addAction(self._toggle_action)

        menu.addSeparator()

        # Theme
        dark_action = QAction("🌙  Dark mode", menu)
        dark_action.triggered.connect(lambda: self.theme_change_requested.emit("dark"))
        menu.addAction(dark_action)

        light_action = QAction("☀  Light mode", menu)
        light_action.triggered.connect(lambda: self.theme_change_requested.emit("light"))
        menu.addAction(light_action)

        menu.addSeparator()

        settings_action = QAction("⚙  Settings", menu)
        settings_action.triggered.connect(self.settings_requested.emit)
        menu.addAction(settings_action)

        menu.addSeparator()

        exit_action = QAction("✕  Exit Spidy", menu)
        exit_action.triggered.connect(self.quit_requested.emit)
        menu.addAction(exit_action)

        return menu

    def _update_menu(self) -> None:
        if self._tray is None:
            return
        menu = self._build_menu()
        self._tray.setContextMenu(menu)

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            # Left-click: toggle
            self._on_toggle()

    def _on_toggle(self) -> None:
        if self._overlay_visible:
            self.hide_overlay_requested.emit()
        else:
            self.show_overlay_requested.emit()
