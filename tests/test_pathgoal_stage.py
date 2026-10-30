"""The bearing the goal model is fed is staged and turns no faster than a
rate (autopilot/pathgoal.py's "Staged"), and the car feeds the lookahead
the model's path goals were drawn at (policy_goal.lookahead_of).

The goal models steer the wrong way for a goal well round to the left, so a
bearing past STAGE_DEG goes in as STAGE_DEG on its side until the point
comes round; one BEHIND_DEG or more round goes in as it is, the turn-round
being the model's. A replan that moves the point round a corner comes in at
AIM_RATE_DPS, not in one decision; the truck's own turning is never held.
"""

import json
import math
import shutil
import sys
import os

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler.autopilot import pathgoal as pgo  # noqa: E402
from crawler.autopilot import policy_goal as pg  # noqa: E402

from test_goal_reseed import crop_at  # noqa: E402
from test_pathgoal import Source, planner, world  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))


def shaped(**kw):
    """A planner on an open room with staging and the rate limit on, as
    on the car unless kw says otherwise."""
    kw.setdefault("stage_deg", pgo.STAGE_DEG)
    kw.setdefault("aim_rate_dps", pgo.AIM_RATE_DPS)
    p, clock = planner(world(size=((-6.0, 6.0), (-6.0, 6.0))), **kw)
    return p, clock


def at(deg, d=1.0):
    """The map point d metres off a truck at the origin facing +x, deg CCW."""
    return (d * math.cos(math.radians(deg)), d * math.sin(math.radians(deg)), 0)


def bearing(g):
    return math.degrees(math.atan2(float(g[0, pg.SIN]), float(g[0, pg.COS])))


def feed(p, deg, t, d=1.0, yaw=0.0):
    """One decision with the (lone, in-sight) goal at deg off a truck at the
    origin facing yaw; the planner plans on the first."""
    src = Source(at(deg, d))
    c = crop_at(0.0, 0.0, yaw, t=t)
    p.encode(c, src)
    return p.encode(c, src), src, c


# --- staging -------------------------------------------------------------------------------

def test_stage_feeds_a_sharp_bearing_as_the_band_edge_and_a_goal_behind_as_it_is():
    deg = np.array([0.0, 30.0, 59.0, 61.0, 100.0, 149.0, 150.0, 170.0, 180.0])
    for side in (1.0, -1.0):
        got = np.degrees(pg.stage(np.radians(side * deg)))
        want = side * np.array([0.0, 30.0, 59.0, 60.0, 60.0, 60.0, 150.0, 170.0, 180.0])
        np.testing.assert_allclose(got, want, atol=1e-9)
    np.testing.assert_allclose(pg.stage(np.radians(100.0), None), np.radians(100.0))


@pytest.mark.parametrize("side", [1.0, -1.0], ids=["left", "right"])
def test_a_goal_round_a_hairpin_is_fed_at_the_band_edge_at_its_own_distance(side):
    p, _ = shaped()
    g, src, c = feed(p, side * 120.0, t=0.0, d=0.9)
    assert p.fed == "goal" and p.why == "in sight"
    raw = src.encode(c)
    assert bearing(g) == pytest.approx(side * pgo.STAGE_DEG)
    assert g[0, pg.PRESENT] == raw[0, pg.PRESENT] and g[0, pg.DIST] == raw[0, pg.DIST]
    # /path/status has the point the model got, so fed_goals.py records it.
    assert math.hypot(*p.aim) == pytest.approx(0.9)
    assert math.degrees(math.atan2(p.aim[1], p.aim[0])) == pytest.approx(side * pgo.STAGE_DEG)
    st = p.status()
    assert st["bearing_deg"] == pytest.approx(side * 60.0)
    assert st["raw_bearing_deg"] == pytest.approx(side * 120.0) and st["staged"] >= 1


def test_a_goal_inside_the_band_or_behind_goes_in_untouched():
    for deg in (40.0, -55.0, 170.0, -165.0):
        p, _ = shaped()
        g, src, c = feed(p, deg, t=0.0)
        np.testing.assert_array_equal(g, src.encode(c))
        assert p.staged == 0


# --- the rate limit ------------------------------------------------------------------------

def test_a_jump_of_the_point_comes_in_at_the_rate_and_gets_there():
    p, _ = shaped()
    feed(p, 15.0, t=0.0)
    step = pgo.AIM_RATE_DPS * 0.1
    got = [bearing(feed(p, 55.0, t=0.1 * k)[0]) for k in range(1, 6)]
    assert got[0] == pytest.approx(15.0 + step)
    assert all(b - a <= step + 1e-4 for a, b in zip([15.0] + got, got))
    assert got[-1] == pytest.approx(55.0), "there once the rate has had the time"


def test_the_trucks_own_turning_is_never_held():
    # The same point on the map, the truck turned 30 degrees in a decision:
    # the bearing fed turns the 30 with it, at once.
    p, _ = shaped()
    feed(p, 40.0, t=0.0, yaw=0.0)
    g, _, _ = feed(p, 40.0, t=0.1, yaw=math.radians(30.0))
    assert bearing(g) == pytest.approx(10.0)


def test_coming_round_from_a_goal_behind_turns_through_the_band_without_a_jump():
    p, _ = shaped()
    g, _, _ = feed(p, 170.0, t=0.0)
    assert bearing(g) == pytest.approx(170.0)
    got = [bearing(feed(p, 30.0, t=0.1 * k)[0]) for k in range(1, 15)]
    prev = 170.0
    for b in got:
        assert abs(b - prev) <= pgo.AIM_RATE_DPS * 0.1 + 1e-4
        prev = b
    assert got[-1] == pytest.approx(30.0)


def test_after_a_gap_the_point_is_fed_where_it_is():
    p, _ = shaped()
    feed(p, -55.0, t=0.0)
    gap = pgo.AIM_GAP_S + 0.05
    assert pgo.AIM_RATE_DPS * gap < 110.0, "more than the rate covers in the gap"
    g, _, _ = feed(p, 55.0, t=gap)
    assert bearing(g) == pytest.approx(55.0)


def test_no_goal_forgets_the_last_bearing():
    p, _ = shaped()
    feed(p, 15.0, t=0.0)
    p.encode(crop_at(0.0, 0.0, 0.0, t=0.1), Source(None))
    g, _, _ = feed(p, 55.0, t=0.2)
    assert bearing(g) == pytest.approx(55.0)


# --- the lookahead, one number for the trainer and the car ---------------------------------

def test_the_car_feeds_the_lookahead_the_trainer_wrote_in_the_sidecar():
    import policy_train as pt
    assert pgo.LOOKAHEAD_M == pg.PATH_M
    assert pg.lookahead_of({"goal_draw": pt.goal_draw_meta(2.0)}) == 2.0
    assert pg.lookahead_of({"goal_draw": pt.goal_draw_meta(None)}) == pg.PATH_M
    assert pg.lookahead_of({}) == pg.PATH_M and pg.lookahead_of(None) == pg.PATH_M


@pytest.mark.parametrize("path_m", [None, 2.0], ids=["hindsight", "path-2.0"])
def test_the_autopilot_sets_the_planners_lookahead_off_the_model(tmp_path, path_m):
    from test_goal_reseed import click
    from test_patrol import Events, pose
    from test_policy_goal import GOAL_MODEL, _pilot
    import policy_train as pt
    model = tmp_path / "policy-goal.onnx"
    shutil.copy(GOAL_MODEL, model)
    meta = json.loads(GOAL_MODEL.with_suffix(".json").read_text())
    meta["goal_draw"] = pt.goal_draw_meta(path_m)
    model.with_suffix(".json").write_text(json.dumps(meta))
    _, _, slam, pilot = _pilot(Events(pose(4.0, 0.0)), policy=model)
    slam.map = lambda: world(t=slam.c.t)
    pilot.path, _ = planner(None, lookahead_m=1.0)
    pilot.path.slam = slam
    assert pilot.resume() is True and pilot.driver.path is pilot.path
    assert pilot.path.lookahead_m == (pg.PATH_M if path_m is None else path_m)
    assert click
