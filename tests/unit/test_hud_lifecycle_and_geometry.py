"""
test_hud_lifecycle_and_geometry.py -- Unit tests for HUD lifecycle, minimize/restore, and geometry retention.
==========================================================================================================
Verifies:
1. minimize_hud() hides window without destroying it or quitting Spidy.
2. minimize_hud() preserves valid dimensions (width >= 750, height >= 480).
3. Windows showMinimized() collapse bug is prevented: HUD never collapses to tiny rectangle.
4. restore_hud() restores window to full normal size (>= 750x480) and clears minimized state.
5. Cycling minimize_hud() and restore_hud() 10+ times retains normal geometry without collapse.
6. Close button click triggers minimize_hud() without terminating the application.
7. UISignalBridge signals invoke appropriate slots safely.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication

from spidy.ui.overlay import OverlayWindow, MIN_WIDTH, MIN_HEIGHT
from spidy.ui.themes import build_default_theme_manager


@pytest.fixture(scope="session")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


@pytest.fixture
def theme():
    mgr = build_default_theme_manager()
    return mgr.current


def test_hud_constants_and_initial_geometry(qapp, theme):
    """Test that OverlayWindow initializes with width >= MIN_WIDTH and height >= MIN_HEIGHT."""
    OverlayWindow._session_pos = None
    win = OverlayWindow(theme=theme, width=900, height=550, animate=False)

    assert win.MIN_WIDTH == 750
    assert win.MIN_HEIGHT == 480
    assert win._normal_width >= 750
    assert win._normal_height >= 480
    assert win.width() >= 750
    assert win.height() >= 480
    win.close()


def test_minimize_hud_hides_window_and_preserves_geometry(qapp, theme):
    """Test minimize_hud() hides the window, marks _is_hidden, and does NOT collapse dimensions."""
    win = OverlayWindow(theme=theme, width=1000, height=600, animate=False)
    win.show_hud()
    qapp.processEvents()

    assert win.isVisible() is True
    assert win._is_hidden is False
    assert win.width() == 1000
    assert win.height() == 600

    # Minimize HUD
    win.minimize_hud()
    qapp.processEvents()

    # Window must be hidden, not destroyed
    assert win.isVisible() is False
    assert win._is_hidden is True
    # Saved geometry must be valid
    assert win._normal_width == 1000
    assert win._normal_height == 600

    win.close()


def test_restore_hud_recovers_full_size_and_visibility(qapp, theme):
    """Test restore_hud() brings the window back with full normal dimensions."""
    win = OverlayWindow(theme=theme, width=950, height=580, animate=False)
    win.show_hud()
    qapp.processEvents()

    win.minimize_hud()
    qapp.processEvents()
    assert win.isVisible() is False

    # Simulate corrupted tiny geometry (the Windows DWM minimize bug: 160x28)
    win.setFixedSize(160, 28)
    assert win.width() == 160
    assert win.height() == 28

    # Restore HUD - must enforce MIN_WIDTH and MIN_HEIGHT
    win.restore_hud()
    qapp.processEvents()

    assert win.isVisible() is True
    assert win._is_hidden is False
    assert win.width() >= MIN_WIDTH
    assert win.height() >= MIN_HEIGHT
    assert win.width() == 950
    assert win.height() == 580

    win.close()


def test_clamp_geometry_prevents_tiny_dimensions(qapp, theme):
    """Test ensure_valid_geometry() clamps corrupt width/height up to minimums."""
    win = OverlayWindow(theme=theme, width=900, height=550, animate=False)
    win._normal_width = 100
    win._normal_height = 50

    win.ensure_valid_geometry()

    assert win._normal_width >= MIN_WIDTH
    assert win._normal_height >= MIN_HEIGHT
    assert win.width() >= MIN_WIDTH
    assert win.height() >= MIN_HEIGHT

    win.close()


def test_ten_minimize_restore_cycles_retain_geometry(qapp, theme):
    """Test that cycling minimize and restore 10+ times maintains exact normal dimensions."""
    target_w = 1100
    target_h = 650
    win = OverlayWindow(theme=theme, width=target_w, height=target_h, animate=False)
    win.show_hud()
    qapp.processEvents()

    for cycle in range(12):
        win.minimize_hud()
        qapp.processEvents()
        assert win.isVisible() is False
        assert win._normal_width == target_w
        assert win._normal_height == target_h

        win.restore_hud()
        qapp.processEvents()
        assert win.isVisible() is True
        assert win.width() == target_w
        assert win.height() == target_h

    win.close()


def test_close_button_triggers_minimize_hud_not_exit(qapp, theme):
    """Clicking the close button hides/minimizes HUD, keeping window valid and reusable."""
    win = OverlayWindow(theme=theme, width=900, height=550, animate=False)
    win.show_hud()
    qapp.processEvents()
    assert win.isVisible() is True

    # Simulate close button click
    win._header._btn_close.click()
    qapp.processEvents()

    # Must be hidden, not destroyed
    assert win.isVisible() is False
    assert win._is_hidden is True
    assert win._normal_width == 900
    assert win._normal_height == 550

    # Must be able to restore again
    win.restore_hud()
    qapp.processEvents()
    assert win.isVisible() is True
    assert win.width() == 900
    assert win.height() == 550

    win.close()


def test_minimize_button_triggers_minimize_hud(qapp, theme):
    """Clicking the minimize button hides/minimizes HUD, keeping window valid and reusable."""
    win = OverlayWindow(theme=theme, width=900, height=550, animate=False)
    win.show_hud()
    qapp.processEvents()
    assert win.isVisible() is True

    # Simulate minimize button click
    win._header._btn_min.click()
    qapp.processEvents()

    # Must be hidden
    assert win.isVisible() is False
    assert win._is_hidden is True

    # Restore again
    win.restore_hud()
    qapp.processEvents()
    assert win.isVisible() is True
    assert win.width() == 900
    assert win.height() == 550

    win.close()


def test_ui_signal_bridge_restore_and_minimize(qapp, theme):
    """Test UISignalBridge slots for restore_hud and minimize_hud operate properly."""
    from spidy.ui.overlay import UISignalBridge

    bridge = UISignalBridge()
    win = OverlayWindow(theme=theme, width=850, height=500, animate=False)
    bridge.restore_hud_requested.connect(win.restore_hud)
    bridge.minimize_hud_requested.connect(win.minimize_hud)

    # Minimize via bridge
    bridge.minimize_hud_requested.emit()
    qapp.processEvents()
    assert win.isVisible() is False

    # Restore via bridge
    bridge.restore_hud_requested.emit()
    qapp.processEvents()
    assert win.isVisible() is True
    assert win.width() == 850
    assert win.height() == 500

    win.close()
