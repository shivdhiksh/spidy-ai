"""
tests/unit/test_ui_hud_responsive.py
=====================================
Regression tests for HUD responsive layout (M17.2 fix).
"""

from __future__ import annotations

import sys
import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    app.processEvents()
    return app


@pytest.fixture()
def dark_theme():
    from spidy.ui.themes.dark import DARK_THEME
    return DARK_THEME


def _make_win(dark_theme, width=1280, height=720):
    """Create a hidden HUD at the given explicit size."""
    from spidy.ui.overlay import OverlayWindow
    win = OverlayWindow(dark_theme, animate=False, width=width, height=height)
    # show_animated sets windowOpacity(1.0) and shows the window
    win.show_animated()
    return win


# 1. Window sizes ------------------------------------------------------------

class TestHUDSizes:
    def test_default_hud_1280x720(self, qapp, dark_theme):
        win = _make_win(dark_theme, 1280, 720)
        assert win.width() == 1280 and win.height() == 720
        win.close()

    def test_smaller_hud_1100x650(self, qapp, dark_theme):
        win = _make_win(dark_theme, 1100, 650)
        assert win.width() == 1100 and win.height() == 650
        win.close()

    def test_larger_hud_1920x1080(self, qapp, dark_theme):
        win = _make_win(dark_theme, 1920, 1080)
        assert win.width() == 1920 and win.height() == 1080
        win.close()


# 2. Panel bounds -----------------------------------------------------------

class TestPanelBounds:
    def _within(self, panel, overlay):
        og = overlay.rect()
        pg = panel.geometry()
        assert pg.left() >= 0,               f"left overflow: {pg.left()}"
        assert pg.top()  >= 0,               f"top overflow: {pg.top()}"
        # tolerance of 2px for layout rounding
        assert pg.right()  <= og.width()  + 2, f"right  {pg.right()} > {og.width()}"
        assert pg.bottom() <= og.height() + 2, f"bottom {pg.bottom()} > {og.height()}"

    def test_left_panel_1280(self, qapp, dark_theme):
        win = _make_win(dark_theme, 1280, 720); qapp.processEvents()
        self._within(win._left_panel, win); win.close()

    def test_right_panel_1280(self, qapp, dark_theme):
        win = _make_win(dark_theme, 1280, 720); qapp.processEvents()
        self._within(win._right_panel, win); win.close()

    def test_left_panel_1920(self, qapp, dark_theme):
        win = _make_win(dark_theme, 1920, 1080); qapp.processEvents()
        self._within(win._left_panel, win); win.close()

    def test_right_panel_1920(self, qapp, dark_theme):
        win = _make_win(dark_theme, 1920, 1080); qapp.processEvents()
        self._within(win._right_panel, win); win.close()


# 3. Proportional widths ----------------------------------------------------

class TestPanelProportions:
    def test_panels_wider_at_1920_than_1100(self, qapp, dark_theme):
        ws = _make_win(dark_theme, 1100, 650)
        wl = _make_win(dark_theme, 1920, 1080)
        qapp.processEvents()
        assert wl._left_panel.width()  >= ws._left_panel.width()
        assert wl._right_panel.width() >= ws._right_panel.width()
        ws.close(); wl.close()

    def test_compute_side_widths_min(self, qapp):
        from spidy.ui.overlay import OverlayWindow
        lw, rw = OverlayWindow._compute_side_widths(1100)
        assert lw >= 152 and rw >= 172

    def test_compute_side_widths_max(self, qapp):
        from spidy.ui.overlay import OverlayWindow
        lw, rw = OverlayWindow._compute_side_widths(2560)
        assert lw <= 210 and rw <= 230

    def test_compute_proportional_growth(self, qapp):
        from spidy.ui.overlay import OverlayWindow
        lw1, rw1 = OverlayWindow._compute_side_widths(1366)
        lw2, rw2 = OverlayWindow._compute_side_widths(1920)
        assert lw2 > lw1 and rw2 > rw1


# 4. No QPainter conflict ---------------------------------------------------

class TestNoQPainterConflict:
    def test_no_graphics_effect(self, qapp, dark_theme):
        """Overlay must NOT use QGraphicsOpacityEffect (causes QPainter conflicts)."""
        win = _make_win(dark_theme, 1280, 720)
        assert win.graphicsEffect() is None, (
            "QGraphicsOpacityEffect found — causes QPainter conflicts. "
            "Use setWindowOpacity() instead."
        )
        win.close()

    def test_opacity_is_1_when_animate_false(self, qapp, dark_theme):
        # show_animated(animate=False) calls setWindowOpacity(1.0) before show()
        win = _make_win(dark_theme, 1280, 720)
        assert win.windowOpacity() == pytest.approx(1.0, abs=0.01)
        win.close()


# 5. Core centered ----------------------------------------------------------

class TestCoreGeometry:
    def test_core_is_square(self, qapp, dark_theme):
        win = _make_win(dark_theme, 1280, 720); qapp.processEvents()
        c = win._core
        assert c.width() > 0, "Core must have non-zero width"
        assert c.width() == c.height(), f"Core must be square, got {c.width()}x{c.height()}"
        win.close()

    def test_core_centered_horizontally(self, qapp, dark_theme):
        win = _make_win(dark_theme, 1280, 720); qapp.processEvents()
        s = win._center_stack
        c = win._core
        assert c.width() > 0, "Core must be laid out before testing centering"
        core_cx  = c.x() + c.width()  / 2
        stack_cx = s.width() / 2
        assert abs(core_cx - stack_cx) <= 2, (
            f"Core center {core_cx:.0f} not near stack center {stack_cx:.0f}"
        )
        win.close()


# 6. Voice bar & task console -----------------------------------------------

class TestSubWidgetBounds:
    def test_voice_bar_inside_footer(self, qapp, dark_theme):
        win = _make_win(dark_theme, 1280, 720); qapp.processEvents()
        assert win._footer.voice_bar.width() <= win._footer.width() + 2
        win.close()

    def test_task_console_is_inside_right_panel_when_shown(self, qapp, dark_theme):
        """HUDTaskConsole must not overflow the right panel width."""
        from spidy.ui.widgets.hud_task_console import TaskStep
        win = _make_win(dark_theme, 1280, 720)
        win.set_state("working")
        win.update_task_progress("Goal", [TaskStep("Step A", "done")])
        qapp.processEvents()
        rp = win._right_panel
        tc = rp._task_console
        # The task console sizeHint should not exceed the panel width
        assert tc.sizeHint().width() <= rp.width() + 2 or tc.width() <= rp.width() + 2
        win.close()


# 7. Confirmation card -------------------------------------------------------

class TestConfirmationCardBounds:
    def test_within_center_stack(self, qapp, dark_theme):
        win = _make_win(dark_theme, 1280, 720)
        win.show_confirmation("t1", "g1", "Upload video to YouTube", "Confirm?")
        qapp.processEvents()
        s = win._center_stack
        c = win._confirmation_card
        assert c.x() >= 0 and c.y() >= 0
        assert c.x() + c.width()  <= s.width()  + 2
        assert c.y() + c.height() <= s.height() + 2
        win.close()


# 8. State transitions -------------------------------------------------------

class TestStateTransitionsResponsive:
    def test_all_states_1366x768(self, qapp, dark_theme):
        win = _make_win(dark_theme, 1366, 768)
        for s in ("idle","wake_ready","listening","thinking","speaking","working","error","idle"):
            win.set_state(s)
        qapp.processEvents(); win.close()

    def test_all_states_1920x1080(self, qapp, dark_theme):
        win = _make_win(dark_theme, 1920, 1080)
        for s in ("idle","wake_ready","listening","thinking","speaking","working","error","idle"):
            win.set_state(s)
        qapp.processEvents(); win.close()


# 9. Long text safety --------------------------------------------------------

class TestLongTextSafety:
    def test_long_message_overlay_stays_valid(self, qapp, dark_theme):
        """Adding very long messages must not break the chat overlay geometry."""
        win = _make_win(dark_theme, 1280, 720)
        win.add_message("user", "A" * 300)
        win.add_message("assistant", "B" * 500)
        win.add_message("user", "C" * 200)
        qapp.processEvents()
        co = win._chat_overlay
        # Overlay itself must have valid size
        assert co.width()  > 0
        assert co.height() > 0
        # Stored cards (dataclass) count should be <= _MAX_CARDS
        from spidy.ui.widgets.hud_chat import _MAX_CARDS
        assert len(co._cards) <= _MAX_CARDS
        win.close()
