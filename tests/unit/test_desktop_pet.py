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
