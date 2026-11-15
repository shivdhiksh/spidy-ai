"""The desktop Spidey: visible while the HUD is closed, opens Spidy on click, hops into the HUD on wake."""

from __future__ import annotations

from unittest.mock import MagicMock

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from spidy.ui.pet import DesktopPet, draw_spidey  # noqa: E402


def _pet(ui=False):
    state = {"ui": ui}
    pet = DesktopPet(ui_visible=lambda: state["ui"])
    return pet, state


def test_every_pose_draws():
    img = QImage(200, 200, QImage.Format.Format_ARGB32)
    for st in ("idle", "walk", "wave", "look", "swing", "hop"):
        img.fill(0)
        p = QPainter(img)
        p.translate(100, 180)
        draw_spidey(p, 0.4, st, 0.0)
        p.end()
        assert img.pixelColor(100, 120).alpha() > 0          # something was drawn


def test_pet_shows_while_the_hud_is_closed_and_hides_when_it_opens():
    pet, s = _pet(ui=False)
    pet.start()
    assert pet.isVisible()
    s["ui"] = True                       # "Hey Spidy": the HUD opens
    pet._sync_with_ui()
    assert pet.state == "hop"            # he hops up and turns into the HUD
    for _ in range(30):
        pet._tick()
    assert not pet.isVisible()
    s["ui"] = False                      # HUD closed again
    pet._sync_with_ui()
    assert pet.isVisible()
    pet._watch.stop(); pet._hide_now()


def test_clicking_the_pet_opens_spidy():
    pet, _ = _pet()
    pet.start()
    opened = MagicMock()
    pet.clicked.connect(opened)

    class _E:
        def __init__(self, pos): self._p = pos
        def button(self): return Qt.MouseButton.LeftButton
        def buttons(self): return Qt.MouseButton.LeftButton
        def globalPosition(self): return self._p

    at = QPointF(pet.pos()) + QPointF(40, 40)
    pet.mousePressEvent(_E(at))
    pet.mouseReleaseEvent(_E(at))
    opened.assert_called_once()
    pet._watch.stop(); pet._hide_now()


def test_disabled_pet_stays_hidden():
    pet, _ = _pet()
    pet.start()
    pet.set_enabled(False)
    assert not pet.isVisible()
    pet._watch.stop()


def test_web_swing_from_anywhere_lands_back_on_the_floor():
    """Top edge, corners, side walls: shoot a web, zip up, swing, lower down -- always back on the taskbar."""
    pet, _ = _pet()
    pet.start()
    floor = pet._floor()
    anchors = set()
    for seed in range(10):
        pet._rng.seed(seed)
        pet._set_state("shoot", 0.45)
        anchors.add((round(pet.anchor.x()) < 50, round(pet.anchor.y()) < 10))
        seen = set()
        for _ in range(900):
            pet._tick()
            seen.add(pet.state)
            if pet.state == "idle" and "lower" in seen:
                break
        assert {"shoot", "swing", "lower"} <= seen
        assert pet.state == "idle" and abs(pet.fy - floor) < 2
    assert len(anchors) >= 2                          # webs went to different kinds of places
    pet._watch.stop(); pet._hide_now()


def test_web_line_is_drawn_while_hanging():
    pet, _ = _pet()
    pet.start()
    pet._rng.seed(1)
    pet._set_state("shoot", 0.45)
    for _ in range(40):
        pet._tick()
    assert pet.state == "swing"
    img = pet.grab().toImage()
    a = pet._local(pet.anchor)
    near_anchor = [img.pixelColor(int(a.x()) + dx, int(a.y()) + dy) for dx in (-1, 0, 1) for dy in (2, 4, 6)
                   if 0 <= int(a.x()) + dx < img.width() and 0 <= int(a.y()) + dy < img.height()]
    assert any(c.alpha() > 0 and c.lightness() > 150 for c in near_anchor)     # white web visible at the anchor
    pet._watch.stop(); pet._hide_now()


def test_hammock_lounge_then_back_down():
    """He strings a web hammock across a top corner, lounges with his phone, then drops back to the floor."""
    pet, _ = _pet()
    pet.start()
    floor = pet._floor()
    pet._rng.seed(2)
    pet.start_hammock()
    seen = set()
    for _ in range(1200):
        pet._tick()
        seen.add(pet.state)
        if pet.state == "hammock":
            assert pet.hammock is not None and pet.grab().width() > 200      # net drawn across the corner
        if pet.state == "idle" and "lower" in seen:
            break
    assert {"shoot", "hammock", "lower"} <= seen
    assert pet.state == "idle" and abs(pet.fy - floor) < 2 and pet.hammock is None
    pet._watch.stop(); pet._hide_now()


def test_upside_down_and_spidey_sense_poses_draw():
    from spidy.ui.pet import upside_pose, draw_spidey_pose
    img = QImage(220, 220, QImage.Format.Format_ARGB32)
    for wave in (False, True):
        img.fill(0)
        p = QPainter(img)
        p.translate(110, 30)
        p.rotate(180)
        pose = upside_pose(0.5, 0.0, wave=wave)
        pose.sense = 1.0
        draw_spidey_pose(p, pose)
        p.end()
        yellow = sum(1 for x in range(0, 220, 3) for y in range(0, 220, 3)
                     if (c := img.pixelColor(x, y)).alpha() > 0 and c.red() > 200 and c.green() > 180 and c.blue() < 120)
        assert yellow > 3                               # the spidey-sense squiggles are there


def test_wakes_up_in_bed_waves_hello_boss_then_stands():
    """First appearance after Spidy starts (every boot/login): asleep in bed -> wave + 'Hello boss!' -> stand up."""
    pet, _ = _pet()
    said = []
    pet.greet.connect(said.append)
    pet.start()
    assert pet.state == "intro"
    for _ in range(int(7 / 0.033)):
        pet._tick()
    assert said == ["Hello boss!"]                    # greeted exactly once
    assert pet.state not in ("intro",) and abs(pet.fy - pet._floor()) < 2
    pet._hide_now()
    pet._sync_with_ui()                              # shown again later: no second greeting
    assert pet.state != "intro"
    pet._watch.stop(); pet._hide_now()


def test_pet_walks_around():
    pet, _ = _pet()
    pet.start()
    pet._set_state("walk", 5.0)
    pet.target_x = pet.x() + 200 * pet.facing
    x0 = pet.x()
    for _ in range(20):
        pet._tick()
    assert pet.x() != x0
    pet._watch.stop(); pet._hide_now()
