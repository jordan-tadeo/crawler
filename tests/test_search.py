"""Searching for a person nobody can see: autopilot/search.py.

Where the goal goes next with the person gone: down the bearing they were
last seen on, short of the wall; the edges of the map, reachable ones,
nearest first; a far free cell with no edge left. Targets rotate on arrival
and after a generous no-progress clock, and ride a reseed link. It only
ever picks a goal point.
"""

import math
import os

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler.autopilot import follow as fl  # noqa: E402
from crawler.autopilot import search as sr  # noqa: E402
from crawler.mapping import grid as gr  # noqa: E402

from test_goal_reseed import crop_at, link  # noqa: E402
from test_pathgoal import world  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A room x -2..6, y -3..3, seen only up to x = 3: the edge of the map is the
# line x = 3 across the room.
HALF_SEEN = world(known=(-2.0, -3.0, 3.0, 3.0))


def test_frontiers_are_the_reachable_edge_of_the_map_not_specks_or_cut_off_pockets():
    fs = sr.frontiers(HALF_SEEN, (0.0, 0.0))
    assert len(fs) == 1
    x, y, edge, d = fs[0]
    assert x == pytest.approx(2.95, abs=0.15) and abs(y) < 0.3
    assert edge > 3.0, "nearly the room's width, less the clearance off the side walls"
    # A speck of unknown in the middle of the seen part is no frontier.
    view = world(known=(-2.0, -3.0, 3.0, 3.0))
    view.codes[15:17, 30:32] = gr.UNKNOWN      # 0.2 m square near (-0.4, 0.1)
    assert len(sr.frontiers(view, (0.0, 0.0))) == 1
    # Walled off from the truck (a wall right across at x = 1.5): none.
    sealed = world([(1.45, -3.0, 1.55, 3.0)], known=(-2.0, -3.0, 3.0, 3.0))
    assert sr.frontiers(sealed, (0.0, 0.0)) == []


def test_along_carries_their_bearing_on_and_stops_short_of_a_wall():
    open_ = world()
    assert sr.along(open_, (0.0, 0.0), (1.0, 0.0)) == pytest.approx((3.0, 0.0), abs=0.01)
    assert sr.along(None, (0.0, 0.0), (0.0, 1.0)) == pytest.approx((0.0, 3.0))
    walled = world([(2.0, -3.0, 2.1, 3.0)])
    x, y = sr.along(walled, (0.0, 0.0), (1.0, 0.0))
    assert y == 0.0 and 1.0 <= x <= 2.0 - sr.CLEAR_M + 0.06


def test_wander_is_the_far_free_cell_away_from_what_was_tried():
    view = world()
    a = sr.wander(view, (0.0, 0.0), [])
    assert math.hypot(*a) > 4.0
    b = sr.wander(view, (0.0, 0.0), [a])
    assert math.hypot(b[0] - a[0], b[1] - a[1]) > 3.0


def search_on(view, hint=None, t=0.0):
    s = sr.Search(map_view=lambda: view)
    s.start(crop_at(t=t), hint)
    s.step(crop_at(t=t))
    return s


def test_the_search_goes_along_first_then_to_the_edge_of_the_map():
    s = search_on(HALF_SEEN, hint=((0.0, 0.0), (0.0, 1.0), 0))
    assert s.kind == "along"
    assert s.target[:2] == pytest.approx((0.0, 3.0 - 0.1 - sr.CLEAR_M), abs=0.12), \
        "up the room to its wall, short of it"
    s.step(crop_at(0.0, s.target[1] - 0.2, t=5.0))
    assert s.arrived == 1 and s.kind == "frontier"
    assert s.target[0] == pytest.approx(2.95, abs=0.15)


def test_a_target_is_given_twenty_seconds_without_progress_not_two():
    s = search_on(HALF_SEEN)
    first = s.target
    assert s.kind == "frontier"
    for t in np.arange(0.1, sr.NO_PROGRESS_S - 0.5, 0.1):
        s.step(crop_at(t=float(t)))
    assert s.target == first and s.skipped == 0, "a slow model is not failed early"
    for t in np.arange(sr.NO_PROGRESS_S - 0.5, sr.NO_PROGRESS_S + 1.0, 0.1):
        s.step(crop_at(t=float(t)))
    assert s.skipped == 1 and s.target != first
    assert math.hypot(s.target[0] - first[0], s.target[1] - first[1]) >= sr.TRIED_M, \
        "not the one it gave up on"
    # Progress resets the clock: 0.3 m closer every 15 s never skips.
    s2 = search_on(HALF_SEEN)
    x = 0.0
    for t in np.arange(0.1, 60.0, 0.1):
        if int(t * 10) % 150 == 0:
            x += 0.35
        s2.step(crop_at(x, t=float(t)))
    assert s2.skipped == 0


def test_no_map_and_no_hint_is_a_hold_said_once():
    said = []
    s = sr.Search(map_view=lambda: None, say=lambda *a: said.append(a[-1]))
    s.start(crop_at(t=0.0))
    for t in np.arange(0.0, 5.0, 0.1):
        s.step(crop_at(t=float(t)))
    assert s.target is None and s.kind == ""
    assert sum("nowhere to go" in x for x in said) == 1


def test_a_target_rides_a_reseed_link():
    s = search_on(HALF_SEEN, hint=((0.0, 0.0), (1.0, 0.0), 0))
    x, y, _ = s.target
    s.follow(link(1, 0, 1, (1.0, 0.0, math.pi / 2)))
    assert s.target == pytest.approx((1.0 - y, x, 1))


def test_following_on_with_a_map_searches_and_the_goal_is_the_search_target():
    g = fl.PersonGoal(fl.CameraGeometry(105.0, 72.5, 0.0, 45.0, 0.15), map_view=lambda: HALF_SEEN)
    g.command({"cmd": "on"})
    g.observe(crop_at(t=0.0))
    g.observe(crop_at(t=fl.WAIT_S + 0.1))
    assert g.state() == "searching" and not g.reached()
    assert g.goal == g.search.target and g.search.kind == "frontier"
    st = g.follow_status(crop_at(t=fl.WAIT_S + 0.1))
    assert st["state"] == "searching" and st["search_kind"] == "frontier"
    assert (st["goal_x"], st["goal_y"]) == pytest.approx(g.goal[:2])


def test_nothing_in_search_can_drive():
    text = open(os.path.join(ROOT, "crawler", "autopilot", "search.py")).read()
    for word in ("set_throttle", "set_steering", "vecon", "controller", "pigpio",
                 "ESC_GPIO_PIN", "return_neutral", "steering"):
        assert word not in text, word
