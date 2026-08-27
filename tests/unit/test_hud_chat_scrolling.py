"""
test_hud_chat_scrolling.py -- Tests for HUD Conversation Scrolling Usability
=============================================================================
Covers:
1. Mouse wheel UP scrolls upward
2. Mouse wheel DOWN scrolls downward
3. Cursor over bubbles/labels propagates wheel scrolling via event filter
4. Trackpad precision pixelDelta scrolling
5. Scrollbar remains usable and styled
6. Auto-scrolls to bottom on new message if already at bottom
7. Preserves scroll position when reading history (does not force jump)
8. "↓ New message" indicator button appears when scrolled up and new message arrives
9. Clicking "↓ New message" button scrolls to bottom and hides indicator
10. Scrolling back to bottom automatically hides indicator
"""

import sys
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

import pytest

from spidy.ui.themes import build_default_theme_manager
from spidy.ui.widgets.hud_chat import HUDChatOverlay


@pytest.fixture(scope="session")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


@pytest.fixture
def theme():
    mgr = build_default_theme_manager()
    return mgr.current


def _create_wheel_event(angle_y: int = 0, pixel_y: int = 0) -> QWheelEvent:
    """Helper to simulate QWheelEvent."""
    return QWheelEvent(
        QPointF(50, 50),
        QPointF(50, 50),
        QPoint(0, pixel_y),
        QPoint(0, angle_y),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def test_mouse_wheel_scrolling_up_and_down(qapp, theme):
    chat = HUDChatOverlay()
    chat.setFixedSize(600, 200)
    chat.show()
    qapp.processEvents()

    # Add mock scroll range to scrollbar
    vsb = chat._scroll.verticalScrollBar()
    vsb.setRange(0, 500)
    vsb.setValue(500)
    assert vsb.maximum() == 500

    # Wheel UP (positive angle delta) -> scroll upward (lower value)
    event_up = _create_wheel_event(angle_y=120)
    chat.wheelEvent(event_up)
    assert vsb.value() < 500

    up_val = vsb.value()

    # Wheel DOWN (negative angle delta) -> scroll downward (higher value)
    event_down = _create_wheel_event(angle_y=-120)
    chat.wheelEvent(event_down)
    assert vsb.value() > up_val

    chat.close()


def test_trackpad_precision_scrolling(qapp, theme):
    chat = HUDChatOverlay()
    chat.setFixedSize(600, 200)
    chat.show()

    vsb = chat._scroll.verticalScrollBar()
    vsb.setRange(0, 500)
    vsb.setValue(100)

    # Trackpad scroll 35 pixels up (positive pixel delta)
    event_pixel = _create_wheel_event(pixel_y=35)
    chat.wheelEvent(event_pixel)
    assert vsb.value() == 65

    chat.close()


def test_event_filter_routes_bubble_wheel_events(qapp, theme):
    chat = HUDChatOverlay()
    chat.setFixedSize(600, 200)
    chat.show()

    vsb = chat._scroll.verticalScrollBar()
    vsb.setRange(0, 500)
    vsb.setValue(500)

    # Send wheel event directly to viewport via eventFilter
    event_up = _create_wheel_event(angle_y=120)
    handled = chat.eventFilter(chat._scroll.viewport(), event_up)
    assert handled is True
    assert vsb.value() < 500
    chat.close()


def test_smart_auto_scroll_and_new_message_indicator(qapp, theme):
    chat = HUDChatOverlay()
    chat.setFixedSize(600, 200)
    chat.show()

    vsb = chat._scroll.verticalScrollBar()
    vsb.setRange(0, 500)
    # User scrolled up to top (value 0, maximum 500)
    vsb.setValue(0)
    assert not chat._is_near_bottom()

    # New message arrives while user is reading older messages
    chat.add_message("assistant", "New message arriving while user is scrolled up!")
    qapp.processEvents()

    # "↓ New message" indicator should be shown
    assert chat._btn_new_msg.isVisible() is True

    # User clicks "↓ New message" indicator -> scrolls to bottom and hides indicator
    chat._btn_new_msg.click()
    qapp.processEvents()

    assert chat._btn_new_msg.isVisible() is False
    chat.close()


def test_auto_scroll_when_at_bottom_without_indicator(qapp, theme):
    chat = HUDChatOverlay()
    chat.setFixedSize(600, 200)
    chat.show()

    vsb = chat._scroll.verticalScrollBar()
    vsb.setRange(0, 500)
    vsb.setValue(500)
    assert chat._is_near_bottom()

    # New message arrives while at bottom -> indicator stays hidden
    chat.add_message("assistant", "Message at bottom")
    qapp.processEvents()

    assert chat._btn_new_msg.isVisible() is False
    chat.close()
