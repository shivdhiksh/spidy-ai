import math
import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from spidy.ui import morph  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _img():
    img = QImage(40, 40, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    for y in range(10, 30):
        for x in range(10, 30):
            img.setPixelColor(x, y, QColor(220, 30, 40))
    return img


def test_dots_start_on_the_pet_and_end_on_the_blob_sphere(app):
    dots = morph.sample_dots(_img(), QPointF(100, 100))
    assert 0 < len(dots) <= morph.MAX_DOTS
    target, r = QPointF(900, 400), 120.0
    for d in dots:
        x, y, a, _ = morph.dot_state(d, 0.0, target, r)
        assert (x, y) == (d["sx"], d["sy"]) and a == 1.0
        x, y, _, _ = morph.dot_state(d, morph.FLIGHT_END, target, r)
        assert math.hypot(x - target.x(), y - target.y()) <= r + 1e-6
        assert morph.dot_state(d, morph.TOTAL, target, r)[2] == 0.0


def test_whole_morph_opens_the_hud_and_finishes_under_2_seconds(app):
    calls = {}
    dots = morph.sample_dots(_img(), QPointF(100, 100))
    t0 = time.monotonic()
    ov = morph.MorphOverlay(dots, QPointF(600, 400), 100.0, _img(), QPointF(100, 100),
                            lambda: calls.setdefault("hud", time.monotonic() - t0),
                            lambda: calls.setdefault("done", time.monotonic() - t0))
    ov.run()
    while "done" not in calls and time.monotonic() - t0 < 3:
        app.processEvents()
        time.sleep(0.005)
    assert "hud" in calls and calls["hud"] < 1.5
    assert calls["done"] < 2.0
    assert morph.TOTAL < 2.0
