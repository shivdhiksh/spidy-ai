import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPointF, QRect  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from spidy.ui import web_cover as wc  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _Pet:
    def __init__(self):
        self.calls = []

    def sleep_in_hammock(self):
        self.calls.append("sleep")

    def wake_from_hammock(self):
        self.calls.append("wake")

    def raise_(self):
        pass


def test_an_hour_idle_or_asleep_spins_webs_and_puts_the_pet_in_the_hammock(app):
    pet = _Pet()
    w = wc.LongAwayWatcher(pet)
    w.poll(idle=120, now=w._last_poll + 20)
    assert not w.cover.active
    w.poll(idle=0, now=w._last_poll + 3700)          # laptop slept / lid closed for over an hour
    assert w.cover.active and pet.calls == ["sleep"]
    w.cover.hide()


def test_cursor_sweeps_every_web_away_then_the_pet_wakes(app):
    pet = _Pet()
    w = wc.LongAwayWatcher(pet)
    w.trigger()
    c = w.cover
    c.shown_at -= wc.GROW_SECS + 1                   # fully grown
    g = c.geometry()
    for y in range(0, g.height() + 40, 40):
        for x in range(0, g.width() + 40, 40):
            c.sweep(QPointF(x, y))
    assert all(s["cut"] is not None for s in c.segs)
    for s in c.segs:
        s["cut"] -= wc.FALL_SECS + 1
    c._step()
    assert not c.isVisible() and pet.calls[-1] == "wake"


def test_webs_cover_corners_and_middle():
    segs = wc.build_webs(QRect(0, 0, 1920, 1080))
    assert len(segs) > 300


def test_one_swipe_brings_down_a_whole_tangle(app):
    import random
    c = wc.WebCover()
    c.setGeometry(0, 0, 1600, 900)
    c.load(wc.build_webs(QRect(0, 0, 1600, 900), random.Random(1)))
    c.shown_at -= wc.GROW_SECS + 1
    for x in range(0, 1600, 30):                     # one swipe straight across the middle
        c.sweep(QPointF(x, 450))
    direct = sum(s["cut"] is not None for s in c.segs)
    for _ in range(wc.CHAIN_HOPS + 1):               # let the chain reaction play out
        for s in c.segs:
            if s["cut"] is None and s["pend"] is not None:
                c._snap(s, c._now(), s["pend"][1])
    chained = sum(s["cut"] is not None for s in c.segs)
    assert chained > direct * 2
