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

def test_thrown_spidey_flies_bounces_and_lands_dizzy():
    """Grab him and fling him: he flies that way with a 'Whoa!', bounces off the floor, then sits dizzy."""
    pet, _ = _pet()
    pet.start()
    pet._set_state("idle", 5)
    floor = pet._floor()
    pet.fx, pet.fy = pet._area().center().x(), floor - 300
    pet.start_fly(300.0, -900.0, "Whoa!")
    x0 = pet.fx
    seen = set()
    for _ in range(400):
        pet._tick()
        seen.add(pet.state)
        if pet.state != "fly":
            break
    assert pet.fx > x0 + 100                                           # went the way he was thrown
    assert pet.state in ("dizzy", "splat") and abs(pet.fy - floor) < 1  # hard landing: sits dizzy or lands flat
    for _ in range(220):
        pet._tick()
    assert pet.state not in ("dizzy", "fly", "splat", "getup")                            # shakes it off
    pet._watch.stop(); pet._hide_now()


def test_drag_then_release_throws_in_the_drag_direction():
    from PySide6.QtCore import QPointF
    pet, _ = _pet()
    pet.start()
    start = QPointF(500, 400)
    import time as _t
    for i in range(6):
        pet.grab_to(QPointF(start.x() - 25 * i, start.y() - 10 * i))   # flick up and to the left
        _t.sleep(0.012)
    assert pet.state == "held"
    pet.throw_release()
    assert pet.state == "fly" and pet.vx < -200 and pet.vy < 0
    pet._watch.stop(); pet._hide_now()


def test_swiping_across_his_web_cuts_it_and_he_falls():
    from PySide6.QtCore import QPointF
    pet, _ = _pet()
    pet.start()
    pet._rng.seed(3)
    pet._set_state("shoot", 0.45)
    for _ in range(40):
        pet._tick()
    assert pet.state == "swing"
    a, g = pet.anchor, pet._body_world()
    mid = QPointF((a.x() + g.x()) / 2, (a.y() + g.y()) / 2)
    pet._last_cursor = QPointF(mid.x() - 40, mid.y())                   # cursor sweeps across the web line
    assert pet._check_web_cut(QPointF(mid.x() + 40, mid.y() + 2))
    assert pet.state == "fly" and pet.bubble == "Eek!" and len(pet.cut_pieces) == 2
    for _ in range(400):
        pet._tick()
        if pet.state not in ("fly",):
            break
    assert abs(pet.fy - pet._floor()) < 1                               # landed
    pet._watch.stop(); pet._hide_now()


def test_sprite_bank_picks_and_mirrors_views():
    from spidy.ui.pet_sprites import SpriteBank
    bank = SpriteBank()
    if not bank.ok:
        return                                                          # sprites not rendered on this machine
    assert bank._pick_view("walk", 90) == (90, False)
    assert bank._pick_view("walk", -90) == (90, True)                  # facing left = mirrored right view
    assert bank._pick_view("walk", 170)[0] == 180
    pm, pts, _ = bank.frame("walk", 0, 3)
    assert not pm.isNull() and {"hand_r", "feet", "head", "hip_c"} <= set(pts)

def test_plays_in_his_corner_and_hops_onto_the_boxes():
    from PySide6.QtCore import QPointF
    pet, _ = _pet()
    pet.start()
    a = pet._area()
    x0, x1 = pet.boxes.play_area(a)
    pet._rng.seed(5)
    inside = 0
    for _ in range(40):                                  # most walks stay in the corner
        pet.perch_box = -1
        pet._set_state("walk", 5)
        inside += x0 <= pet.target_x <= x1
    assert inside >= 25
    top = pet.boxes.tops(a)[2]
    pet.start_jump(top, 2)
    for _ in range(30):
        pet._tick()
    assert pet.perch_box == 2 and abs(pet.fx - top.x()) < 1 and abs(pet.fy - top.y()) < 1
    pet._hide_now()
    assert not pet.boxes.isVisible()                     # boxes go away with him
    pet._watch.stop()

def test_lands_flat_when_tipped_over_then_gets_up():
    pet, _ = _pet()
    pet.start()
    pet._set_state("idle", 5)
    pet.fx, pet.fy = pet._area().center().x(), pet._floor() - 5
    pet.start_fly(0.0, 50.0)
    pet.spin = 100.0                                    # tumbling face-first as he hits the ground
    seen = []
    for _ in range(200):
        pet._tick()
        if not seen or seen[-1] != pet.state:
            seen.append(pet.state)
        if pet.state == "idle" and "splat" in seen:
            break
    assert seen[:4] == ["fly", "splat", "getup", "dizzy"] or seen[1:4] == ["splat", "getup", "dizzy"]
    pet.grab()                                          # every state paints
    pet._watch.stop(); pet._hide_now()

def test_sit_calm_stays_put_and_hidden_pet_can_come_back():
    pet, _ = _pet()
    pet.start()
    pet.set_calm(True)
    x, y = pet.fx, pet.fy
    for _ in range(400):
        pet._tick()
    assert pet.state == "sit" and (pet.fx, pet.fy) == (x, y)          # sits in one place
    pet.grab()
    pet.set_calm(False)
    assert pet.state == "idle"
    pet.set_enabled(False)
    assert not pet.isVisible()
    pet.set_enabled(True)                                              # tray: "Show desktop pet"
    assert pet.isVisible()
    pet._watch.stop(); pet._hide_now()
