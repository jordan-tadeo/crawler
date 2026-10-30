"""The point along a path round the walls that the goal model is fed in
place of its goal (autopilot/pathgoal.py).

The planner finds a path on the SLAM map's /map picture, inflated by the
truck's half-width, with unknown cells costed and not banned; the point fed
is the farthest one LOOKAHEAD_M along it that the truck can see, through
the same policy_goal.encode a click goes through, and the goal itself when
there is no path to use. These hold the path, the point, the fallbacks and
what they say, the thread, the wiring into PolicyDriver and the autopilot,
/path and /path/status, and a simulated patrol round a wall through the
real PolicyDriver with a stub model that only reads its goal input: round
the wall with the planner, stuck against it without.
"""

import json
import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler.autopilot import pathgoal as pgo  # noqa: E402
from crawler.autopilot import patrol as pt  # noqa: E402
from crawler.autopilot import policy_goal as pg  # noqa: E402
from crawler.mapping import grid as gr  # noqa: E402
from crawler.mapping import runner as rn  # noqa: E402
from crawler.nav import truck as tk  # noqa: E402
from crawler.telemetry import path_view as pv  # noqa: E402
from crawler.telemetry import telemetry as tm  # noqa: E402

from test_goal_reseed import crop_at  # noqa: E402
from test_lidar_view import decode  # noqa: E402

RES = 0.1


def world(walls=(), size=((-2.0, 6.0), (-3.0, 3.0)), known=None, t=0.0):
    """A runner.MapView: free inside `size` with a wall round its edge, and
    each (x0, y0, x1, y1) box in `walls` occupied. `known` (a box) limits
    what has been seen; past it is unknown."""
    (ax, bx), (ay, by) = size
    nx, ny = int(round((bx - ax) / RES)), int(round((by - ay) / RES))
    cx = ax + (np.arange(nx) + 0.5) * RES
    cy = ay + (np.arange(ny) + 0.5) * RES
    X, Y = np.meshgrid(cx, cy, indexing="ij")
    codes = np.full((nx, ny), gr.FREE, np.uint8)
    codes[[0, -1], :] = gr.OCCUPIED
    codes[:, [0, -1]] = gr.OCCUPIED
    for x0, y0, x1, y1 in walls:
        codes[(X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)] = gr.OCCUPIED
    if known is not None:
        x0, y0, x1, y1 = known
        codes[~((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1))] = gr.UNKNOWN
    return rn.MapView(t=t, version=1, x0=ax, y0=ay, res=RES, codes=codes)


# A wall across the way from (0, 0) to (4, 0), open round both ends.
WALL = (1.95, -1.5, 2.05, 1.5)


def wall_clearance(pts, wall=WALL):
    x0, y0, x1, y1 = wall
    dx = np.maximum(np.maximum(x0 - pts[:, 0], 0.0), pts[:, 0] - x1)
    dy = np.maximum(np.maximum(y0 - pts[:, 1], 0.0), pts[:, 1] - y1)
    return np.hypot(dx, dy)


def sees(cm, a, b, clear=None):
    """The line from a to b keeps `clear` (the plan's own clearance, its
    cost map's inscribed) off every wall, bar its first LOS_SKIP_M: checked
    here every centimetre off the distance field."""
    clear = cm.inscribed if clear is None else clear
    n = 400
    t = np.linspace(0.0, 1.0, n)
    x = a[0] + t * (b[0] - a[0])
    y = a[1] + t * (b[1] - a[1])
    far = np.hypot(x - a[0], y - a[1]) >= pgo.LOS_SKIP_M
    return bool((cm.dist_at(x[far], y[far]) >= clear - 1e-6).all())


class Source:
    """A goal source as PathPlanner reads one (pathgoal.GoalSource)."""

    def __init__(self, goal=None):
        self.goal = goal

    def encode(self, crop):
        return pg.encode(crop, self.goal)


class MapSlam:
    def __init__(self, view):
        self.view = view

    def map(self):
        return self.view


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def planner(view, **kw):
    clock = kw.pop("clock", Clock())
    # The point as it is picked: staging and the rate limit are off unless
    # asked for (their own tests are test_pathgoal_stage.py's).
    kw.setdefault("stage_deg", None)
    kw.setdefault("aim_rate_dps", None)
    return pgo.PathPlanner(MapSlam(view), background=False, clock=clock, **kw), clock


# --- the path ---------------------------------------------------------------------------

def test_the_path_goes_round_a_wall_and_keeps_the_truck_off_it():
    pts, why, cm = pgo.find_path(world([WALL]), (0.0, 0.0), (4.0, 0.0))
    assert why == "" and pts is not None
    assert tuple(pts[0]) == (0.0, 0.0) and tuple(pts[-1]) == (4.0, 0.0)
    length = float(np.hypot(*np.diff(pts, axis=0).T).sum())
    assert length > 4.0 + 1.0, "round the end of the wall, not through it"
    # Past either end of it, and never closer than the inflated half-width
    # (cell centres to cell centres, so half a cell either way).
    assert np.abs(pts[:, 1]).max() > 1.5
    assert wall_clearance(pts[1:-1]).min() >= tk.HALF_WIDTH_M - RES / 2


def test_unknown_is_costed_not_banned_so_a_goal_past_the_map_gets_a_path():
    view = world(known=(-2.0, -3.0, 2.0, 3.0))
    pts, why, _ = pgo.find_path(view, (0.0, 0.0), (4.0, 0.0))
    assert why == "" and pts is not None, "out into what has not been seen"
    # A goal outside the picture altogether: the window is unknown there.
    pts, why, _ = pgo.find_path(world(size=((-2.0, 2.0), (-2.0, 2.0))), (0.0, 0.0), (5.0, 0.0))
    assert pts is None, "the picture's own border is a wall"
    pts, why, _ = pgo.find_path(world(size=((-2.0, 2.0), (-2.0, 2.0)), known=(-2.0, -2.0, 1.5, 2.0)),
                                (0.0, 0.0), (5.0, 0.0))
    assert why == "" and pts[-1][0] == 5.0
    # Unknown costs more than seen floor: round a wall, the short way over
    # the top is unknown and the long way under is seen, and it goes under.
    wall = (1.95, -1.0, 2.05, 0.6)
    view = world([wall])
    pts, _, _ = pgo.find_path(view, (0.0, 0.0), (4.0, 0.0))
    assert pts[:, 1].max() > 0.6, "all seen: over the top, the short way"
    codes = view.codes.copy()
    nx, ny = codes.shape
    xs = view.x0 + (np.arange(nx) + 0.5) * RES
    ys = view.y0 + (np.arange(ny) + 0.5) * RES
    top = np.zeros_like(codes, bool)
    top[np.ix_((xs > 0.8) & (xs < 3.2), ys > 0.4)] = True
    codes[top & (codes == gr.FREE)] = gr.UNKNOWN
    pts, _, _ = pgo.find_path(view._replace(codes=codes), (0.0, 0.0), (4.0, 0.0))
    assert pts[:, 1].min() < -1.0 and pts[:, 1].max() < 0.6, "under, through the seen"


def test_a_doorway_narrower_than_the_truck_is_not_a_way_through():
    def door(gap):
        h = gap / 2.0
        return [(1.95, -1.5, 2.05, -h), (1.95, h, 2.05, 1.5)]
    # The truck is 0.27 m wide.
    pts, _, _ = pgo.find_path(world(door(0.2)), (0.0, 0.0), (4.0, 0.0))
    assert np.abs(pts[:, 1]).max() > 1.5, "0.2 m: round the end, not through"
    pts, _, _ = pgo.find_path(world(door(0.7)), (0.0, 0.0), (4.0, 0.0))
    assert np.abs(pts[:, 1]).max() < 0.3, "a 0.7 m house door: through it"


def test_the_path_keeps_wide_of_a_corner_and_off_a_wall_it_runs_along():
    # Round the corner of a box in an open room, where the truck hooked a
    # wheel: the plan swings wide of it (0.28 m off it on the old 0.196 m
    # hard clearance and 0.6 m band).
    box = (1.0, -3.0, 2.0, 1.0)
    pts, why, cm = pgo.find_path(world([box]), (0.0, 0.0), (3.0, 2.0))
    assert why == "" and cm.inscribed == pytest.approx(tk.HALF_WIDTH_M + pgo.WIDE_MARGIN_M)
    assert wall_clearance(pts[1:-1], box).min() >= 0.5, "wide of the corner"
    # Along a wall, start and goal 0.4 m off it: out into the soft band on
    # the way (0.6 m on the old band), not hugging the wall.
    wall = (-2.0, -3.0, 6.0, -0.4)
    pts, _, _ = pgo.find_path(world([wall]), (0.0, 0.0), (4.0, 0.0))
    assert np.median(wall_clearance(pts[1:-1], wall)) >= 0.8, "off the wall it runs along"
    # With the old clearance and band, the corner is shaved: the wide is
    # what keeps the truck off it.
    pts, _, _ = pgo.find_path(world([box]), (0.0, 0.0), (3.0, 2.0),
                              margins=(pgo.NARROW_MARGIN_M,), soft_m=0.6, cost_weight=2.0)
    assert wall_clearance(pts[1:-1], box).min() < 0.5


def test_a_gap_too_tight_for_the_wide_clearance_is_still_planned_through_at_the_narrow():
    def door(gap):
        h = gap / 2.0
        return [(1.95, -3.0, 2.05, -h), (1.95, h, 2.05, 3.0)]
    # 0.7 m: wide. 0.4 m: no way at the wide, so made again at the truck's
    # own fit and through it; /path/status says which.
    for gap, clear in ((0.7, tk.HALF_WIDTH_M + pgo.WIDE_MARGIN_M),
                       (0.4, tk.HALF_WIDTH_M + pgo.NARROW_MARGIN_M)):
        pts, why, cm = pgo.find_path(world(door(gap)), (0.0, 0.0), (4.0, 0.0))
        assert why == "" and np.abs(pts[:, 1]).max() < 0.3, f"{gap} m: through it"
        assert cm.inscribed == pytest.approx(clear), f"{gap} m"
    p, _ = planner(world(door(0.4)))
    p.encode(crop_at(0.0, 0.0, 0.0), Source((4.0, 0.0, 0)))
    assert p.status()["clear_m"] == pytest.approx(tk.HALF_WIDTH_M + pgo.NARROW_MARGIN_M)


def test_a_goal_sealed_off_has_no_path_and_says_why():
    box = [(3.0, -1.0, 5.0, -0.9), (3.0, 0.9, 5.0, 1.0), (3.0, -1.0, 3.1, 1.0), (4.9, -1.0, 5.0, 1.0)]
    pts, why, _ = pgo.find_path(world(box), (0.0, 0.0), (4.0, 0.0))
    assert pts is None and why == "no way through"
    pts, why, _ = pgo.find_path(world([(3.0, -1.0, 5.0, 1.0)]), (0.0, 0.0), (4.0, 0.0))
    assert pts is None and why == "goal boxed in"


# --- the point fed ----------------------------------------------------------------------

def test_round_a_wall_the_model_is_fed_a_point_along_the_path_in_sight_not_the_goal():
    p, _ = planner(world([WALL]))
    src = Source((4.0, 0.0, 0))
    crop = crop_at(0.0, 0.0, 0.0)
    g = p.encode(crop, src)
    assert p.fed == "subgoal" and p.why == ""
    ax, ay = p.aim
    np.testing.assert_allclose(g, pg.encode(crop, (ax, ay, 0)))
    assert not np.allclose(g, src.encode(crop)), "not the straight-line bearing"
    d = math.hypot(ax, ay)
    assert 1.0 <= d <= pgo.LOOKAHEAD_M + 1e-6, "LOOKAHEAD_M along the path at most"
    assert sees(p._plan.cm, (0.0, 0.0), (ax, ay))
    assert abs(math.atan2(ay, ax)) > math.radians(15), "turned off the line to the goal"


def test_the_point_is_the_farthest_one_in_sight_so_a_corner_is_not_cut():
    # An L: the path turns 90 degrees round the corner of a wall 0.6 m on.
    # The plan now swings wide enough of the corner that LOOKAHEAD_M along
    # it is all in sight; 3 m along, the corner hides the rest.
    p, _ = planner(world([(0.6, -3.0, 0.7, 1.0)]), lookahead_m=3.0)
    src = Source((1.5, 0.0, 0))
    crop = crop_at(0.0, 1.2, 0.0)
    p.encode(crop, src)
    pts = p._plan.points
    assert p.fed == "subgoal"
    assert sees(p._plan.cm, (0.0, 1.2), p.aim), "the corner is not cut"
    # Farther along the path than every point it can see is one it cannot.
    k = int(np.argmin(np.hypot(pts[:, 0] - p.aim[0], pts[:, 1] - p.aim[1])))
    within = pts[k + 1:][p._plan.cum[k + 1:] <= p.lookahead_m]
    assert len(within) and all(not sees(p._plan.cm, (0.0, 1.2), q) for q in within)


def test_in_a_gap_one_cell_wide_at_the_wide_clearance_the_point_is_not_on_top_of_the_truck():
    # A corridor whose middle row of cells is the only one the wide
    # clearance keeps, the truck one cell off it (2026-10-30 p1200: fed a
    # point 0.26 to 0.54 m off for 40 s, 10 m of path to go).
    view = world([(1.0, -3.0, 5.0, -0.2), (1.0, 0.3, 5.0, 3.0)])
    p, _ = planner(view)
    crop = crop_at(1.5, -0.05, 0.0)
    p.encode(crop, Source((4.5, 0.05, 0)))
    cm = p._plan.cm
    assert p.fed == "subgoal"
    assert cm.inscribed == pytest.approx(tk.HALF_WIDTH_M + pgo.WIDE_MARGIN_M)
    ax, ay = p.aim
    assert math.hypot(ax - 1.5, ay + 0.05) >= pgo.LOS_NEAR_SHARE * pgo.LOOKAHEAD_M
    assert sees(cm, (1.5, -0.05), (ax, ay), pgo.LOS_CLEAR_M), "at the truck's own fit"
    # In the open the plan's own clearance still decides.
    p, _ = planner(world([WALL]))
    p.encode(crop_at(0.0, 0.0, 0.0), Source((4.0, 0.0, 0)))
    assert sees(p._plan.cm, (0.0, 0.0), p.aim)


def test_in_a_clear_line_and_close_the_model_is_fed_the_goal_itself_exactly():
    p, _ = planner(world([WALL]))
    src = Source((1.0, 0.5, 0))
    crop = crop_at(0.0, 0.0, 0.0)
    p.encode(crop, src)                       # the plan
    g = p.encode(crop, src)
    assert p.fed == "goal" and p.why == "in sight"
    np.testing.assert_array_equal(g, src.encode(crop))
    # The same along the path: the last stretch is the goal, not a point short of it.
    src.goal = (3.0, -2.0, 0)
    crop = crop_at(2.1, -2.0, 0.0)
    p.encode(crop, src)
    g = p.encode(crop, src)
    assert p.fed == "goal"
    np.testing.assert_array_equal(g, src.encode(crop))


@pytest.mark.parametrize("case", ["no map", "sealed", "other frame", "stale", "off path"])
def test_with_no_path_to_use_it_is_the_goal_itself_and_status_says_why(case):
    view = world([WALL])
    if case == "sealed":
        view = world([(3.0, -1.0, 5.0, 1.0)])
    p, clock = planner(None if case == "no map" else view)
    said = []
    p._say = lambda *a: said.append(a)
    src = Source((4.0, 0.0, 0))
    crop = crop_at(0.0, 0.0, 0.0)
    g = p.encode(crop, src)
    if case == "other frame":
        crop = crop_at(0.0, 0.0, 0.0, frame=1)
        g = p.encode(crop, src)
    if case == "stale":
        p.background = True               # nothing plans for it any more
        p._ensure_thread = lambda: None
        clock.t += pgo.STALE_S + 0.1
        g = p.encode(crop, src)
    if case == "off path":
        crop = crop_at(0.0, 2.6, 0.0)     # 1.1 m off a path that leaves at y = 0
        p.background = True
        p._ensure_thread = lambda: None
        g = p.encode(crop, src)
    np.testing.assert_array_equal(g, src.encode(crop))
    st = p.status()
    want = {"no map": "no map yet", "sealed": "goal boxed in",
            "other frame": "goal on another map", "stale": "plan stale",
            "off path": "off the path"}[case]
    assert st["why"] == want
    assert st["fed"] == ("none" if case == "other frame" else "goal")
    if case != "other frame":
        assert any(want in line[2] for line in said), "said in the journal"


def test_no_goal_is_no_goal_input_and_no_plan():
    p, _ = planner(world([WALL]))
    src = Source(None)
    src.encode = lambda crop: pg.missing()
    assert not p.encode(crop_at(0.0, 0.0, 0.0), src).any()
    assert p.fed == "none" and p.plans == 0 and p.marker() == {}


def test_it_replans_at_most_every_period_and_at_once_for_a_new_goal():
    p, clock = planner(world([WALL]))
    src = Source((4.0, 0.0, 0))
    p.encode(crop_at(0.0, 0.0), src)
    assert p.plans == 1
    p.encode(crop_at(0.5, 0.0), src)
    assert p.plans == 1, "moved, but not a period on"
    clock.t += pgo.PLAN_EVERY_S
    p.encode(crop_at(0.5, 0.0), src)
    assert p.plans == 2
    clock.t += pgo.PLAN_EVERY_S
    p.encode(crop_at(0.5, 0.0), src)
    assert p.plans == 2, "nothing moved, the map is the same"
    src.goal = (4.0, 1.0, 0)
    p.encode(crop_at(0.5, 0.0), src)
    assert p.plans == 3, "a new goal is planned at once"
    assert p._plan.goal[:2] == (4.0, 1.0)


def test_a_map_from_before_a_reseed_is_not_planned_on():
    view = world([WALL], t=5.0)
    p, _ = planner(view)
    src = Source((4.0, 0.0, 1))
    p.encode(crop_at(0.0, 0.0, frame=1, t=6.0), src)
    assert p.status()["why"] == "no map on this frame yet"
    p.slam.view = view._replace(t=6.5)
    p.encode(crop_at(0.0, 0.0, frame=1, t=7.0), src)
    assert p.fed == "subgoal"


class HeldSlam(MapSlam):
    """map() waits until let go: a plan cannot finish before the test says."""

    def __init__(self, view):
        super().__init__(view)
        self.entered = threading.Event()
        self.go = threading.Event()

    def map(self):
        self.entered.set()
        self.go.wait(10.0)
        return self.view


def test_the_plan_is_made_on_its_own_thread_and_update_returns_at_once():
    # Synchronised, not timed, so a loaded machine cannot fail it: the map
    # is held, so update() can only have returned without the plan if it
    # never waits on it; and the planning thread is seen inside it.
    slam = HeldSlam(world([WALL]))
    p = pgo.PathPlanner(slam, every_s=0.05)
    src = Source((4.0, 0.0, 0))
    try:
        p.update(crop_at(0.0, 0.0), src)
        assert p.plans == 0, "update() returned with the plan still being made"
        assert slam.entered.wait(10.0), "the plan is being made, elsewhere"
        assert p._thread is not None and p._thread.name == "path-planner"
        slam.go.set()
        deadline = time.monotonic() + 10.0
        while p.plans == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert p.plans >= 1
        p.encode(crop_at(0.0, 0.0), src)
        assert p.fed == "subgoal"
    finally:
        slam.go.set()
        p.stop()


# --- PolicyDriver and the autopilot -------------------------------------------------------

def test_the_driver_feeds_the_model_the_point_along_the_path_and_says_so():
    from test_policy_goal import GoalStub, PosedSlam, goal_driver
    from test_policy_map import FORWARD, T0
    d, model, slam, g = goal_driver(GoalStub(FORWARD), PosedSlam(0.0, 0.0, 0.0))
    g.set_goal(4.0, 0.0, 0)
    d.tick(T0 + 0.05)
    np.testing.assert_allclose(model.goals[-1], pg.encode(slam.crop(), g.goal))
    assert d.policy_fields()["policy_goal_fed"] == "goal"
    slam.map = lambda: world([WALL], t=T0)
    d.path, _ = planner(None, clock=Clock(T0))
    d.path.slam = slam
    d.tick(T0 + 0.15)
    assert d.path.fed == "subgoal"
    np.testing.assert_allclose(model.goals[-1], pg.encode(slam.crop(), d.path.aim + (0,)))
    f = d.policy_fields()
    assert f["policy_goal_present"] and f["policy_goal_fed"] == "subgoal"
    # The arrival hold reads the goal, never the point fed.
    assert not d.goal_reached and d.goal.goal == (4.0, 0.0, 0)
    from crawler.autopilot import footprint as fp
    assert fp.autopilot_fields(d)["policy_goal_fed"] == "subgoal"
    assert set(fp.AUTOPILOT_FIELDS) <= set(tm.JSON_SCHEMAS["footprint"]["schema"]["properties"])


def test_with_goal_the_autopilot_plans_in_any_mode_and_hands_its_planner_to_the_driver():
    from crawler import __main__ as mn
    from test_goal_reseed import click
    from test_patrol import Events, pose
    from test_policy_goal import _pilot
    assert mn.parse_args(["--goal"]).no_path is False
    assert mn.parse_args(["--goal", "--no-path"]).no_path is True
    events = Events(pose(4.0, 0.0))
    _, _, slam, pilot = _pilot(events)
    slam.map = lambda: world([WALL], t=slam.c.t)
    pilot.path, _ = planner(None)
    pilot.path.slam = slam
    pilot.goal_step()                     # manual: auto never started
    assert pilot.path.plans == 1, "the path is on the map before auto"
    assert pilot.resume() is True and pilot.driver.path is pilot.path
    assert click  # the helper the events are made of


# --- telemetry ------------------------------------------------------------------------------

def entities(update):
    return {e.id: e for e in decode(update).entities}


def test_path_status_fits_its_schema_and_path_draws_the_line_and_the_point():
    p, _ = planner(world([WALL]))
    p.encode(crop_at(0.0, 0.0), Source((4.0, 0.0, 0)))
    st = p.status()
    props = tm.JSON_SCHEMAS["path_status"]["schema"]["properties"]
    assert set(st) == set(props) and tm.JSON_TOPICS["path_status"] == "/path/status"
    for k, v in st.items():
        ok = {"string": str, "boolean": bool, "integer": int, "number": float}[props[k]["type"]]
        assert isinstance(v, ok), k
    assert st["path"] and st["length_m"] > 5.0 and st["plans"] == 1
    assert tm.NAV_CHANNELS["path"] == "/path"
    ents = entities(tm.Telemetry.lidar_view_message("path", p.marker(), 0))
    line = [(q.x, q.y) for q in ents[pv.LINE_ID].lines[0].points]
    assert line[0] == (0.0, 0.0) and line[-1] == (4.0, 0.0)
    disc = ents[pv.AIM_ID].cylinders[0].pose.position
    assert (disc.x, disc.y) == pytest.approx(p.aim)
    gone = decode(pv.scene({}, 0))
    assert not gone.entities and {x.id for x in gone.deletions} == {pv.LINE_ID, pv.AIM_ID}


def test_the_drive_layout_draws_the_path_and_shows_its_status():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = json.load(open(os.path.join(root, "deploy", "foxglove-drive.json")))["configById"]
    assert cfg["3D!slam"]["topics"]["/path"]["visible"]
    assert any(v.get("topicPath") == "/path/status" for v in cfg.values())


# --- a simulated patrol round a wall ----------------------------------------------------------

SIM_WAYPOINTS = [(4.0, 0.0), (0.0, 0.0)]


def simulate(seconds, path=True):
    """The truck as a unicycle under the real PolicyDriver at 10 Hz, on a
    two-point loop either side of WALL, the SLAM crop its true pose and the
    map the true walls. It cannot drive into a wall: a step that would put
    its middle within the half-width of one is not taken."""
    from scipy import ndimage
    from test_patrol import Events, GoalSteerStub, cmd, pose
    from test_policy_goal import PosedSlam
    from test_policy_map import T0, driver
    view = world([WALL])
    solid = view.codes == gr.OCCUPIED
    dist = ndimage.distance_transform_edt(~solid) * RES

    def clearance(x, y):
        i = int((x - view.x0) / RES)
        j = int((y - view.y0) / RES)
        return dist[i, j]

    slam = PosedSlam()
    model = GoalSteerStub()
    d = driver(model, slam)
    clock = Clock(T0)
    p = pt.Patrol(clock=clock)
    d.goal = p
    if path:
        d.path = pgo.PathPlanner(slam, background=False, clock=clock)
    slam.map = lambda: view._replace(t=slam.c.t)
    p.poll(Events(cmd("edit"), *[pose(*xy) for xy in SIM_WAYPOINTS], cmd("mode", mode="loop"),
                  cmd("start")), slam.crop())
    x, y, yaw = 0.0, 0.0, 0.0
    t, dt = T0, 0.1
    min_d = [math.inf] * len(SIM_WAYPOINTS)
    blocked = 0
    fed = {"subgoal": 0, "goal": 0, "none": 0}
    while t < T0 + seconds:
        t = round(t + dt, 6)
        clock.t = t
        slam.c = slam.c._replace(x=x, y=y, yaw=yaw, t=t)
        p.poll(Events(), slam.crop())
        d.tick(t)
        fed[d.policy_fields()["policy_goal_fed"]] += 1
        thr = d.controller.throttles[-1] if d.controller.throttles else 0.0
        front = d.controller.steerings[-1][0] if d.controller.steerings else 0.0
        v = 0.8 * thr
        yaw += -v * front * 1.6 * dt
        nx_, ny_ = x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt
        if clearance(nx_, ny_) >= tk.HALF_WIDTH_M:
            x, y = nx_, ny_
        else:
            blocked += 1
        for k, (wx, wy) in enumerate(SIM_WAYPOINTS):
            min_d[k] = min(min_d[k], math.hypot(x - wx, y - wy))
    return SimpleNamespace(patrol=p, laps=p.laps, skipped=p.skipped, min_d=min_d,
                           blocked=blocked, fed=fed, driver=d)


def test_a_simulated_patrol_goes_round_a_wall_with_the_path_and_sticks_on_it_without():
    with_path = simulate(180.0, path=True)
    without = simulate(180.0, path=False)
    for name, r in (("with the path", with_path), ("without", without)):
        print(f"\nsim {name}: {r.laps} laps in 180 s, closest pass per waypoint "
              f"{[round(v, 2) for v in r.min_d]} m, {r.skipped} skipped, "
              f"{r.blocked} steps against the wall, fed {r.fed}")
    assert with_path.min_d[0] <= pt.PASS_M and with_path.skipped == 0
    assert with_path.laps >= 2
    assert with_path.fed["subgoal"] > 0
    assert without.min_d[0] > 1.5, "never past the wall"
    assert without.skipped >= 1 and without.blocked > 0
