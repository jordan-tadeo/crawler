"""The navigator: cost map, planner, follower, frontiers, re-planning, the
Foxglove click, and the safety layers that stay underneath it.

Synthetic rooms only (tools/nav_world.room and hand-built grids). The house
maps the simulator runs on are not in the repo and nothing here reads them.
Every loop is bounded by a step count.
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

import crawler.__main__ as mn
from crawler.autopilot import driver as dv
from crawler.hardware import lidar as ld
from crawler.nav import click as ck
from crawler.nav import costmap as cmod
from crawler.nav import driver as nd
from crawler.nav import follower as fl
from crawler.nav import frontier as fr
from crawler.nav import mapview as mv
from crawler.nav import navigator as nv
from crawler.nav import planner as pl
from crawler.nav import truck as tk
from crawler.telemetry import telemetry as tm

import nav_world as nw

from conftest import FakeCamera, FakeController

RES = 0.05
T0 = 100.0


def view_of(occ, pose=(1.0, 1.0, 0.0), known=None, version=0, confidence=1.0):
    occ = np.asarray(occ, bool)
    if known is None:
        known = np.ones_like(occ)
    return mv.MapView(occ, known, RES, (0.0, 0.0), pose, confidence, version)


def walled(nx, ny):
    occ = np.zeros((nx, ny), bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return occ


# --- the cost map ------------------------------------------------------------

def test_an_obstacle_is_lethal_out_to_the_half_width_plus_the_margin():
    occ = np.zeros((81, 81), bool)
    occ[40, 40] = True
    cm = cmod.build(view_of(occ, (0.5, 0.5, 0.0)))
    assert cm.inscribed == pytest.approx(tk.HALF_WIDTH_M + cmod.MARGIN_M)
    # Along the row: the cell whose centre is just inside the band is lethal,
    # the first one past it is not.
    d = np.arange(81) - 40
    row_lethal = cm.lethal[:, 40]
    inside = np.abs(d) * RES < cm.inscribed
    assert np.array_equal(row_lethal, inside)
    assert not cm.lethal[40 + 5, 40]          # 0.25 m out: past 0.196
    assert cm.lethal[40 + 3, 40]              # 0.15 m out: inside the half-width


def test_the_cost_falls_away_beyond_the_lethal_band_and_is_gone_past_inflation():
    occ = np.zeros((81, 81), bool)
    occ[40, 40] = True
    cm = cmod.build(view_of(occ, (0.5, 0.5, 0.0)))
    row = cm.cost[40:, 40]
    out = int(math.ceil(cm.inscribed / RES))
    assert np.all(row[:out] == 1.0)
    tail = row[out:]
    assert np.all(np.diff(tail) <= 1e-6)              # monotone down
    assert tail[0] > 0.5
    assert row[int(math.ceil(cmod.INFLATION_M / RES)) + 1] == pytest.approx(0.0, abs=1e-6)


def test_unknown_floor_costs_but_is_not_lethal():
    occ = np.zeros((40, 40), bool)
    known = np.zeros_like(occ)
    known[:20] = True
    cm = cmod.build(view_of(occ, (0.5, 0.5, 0.0), known=known))
    assert not cm.lethal.any()
    assert cm.cost[30, 20] == pytest.approx(cmod.UNKNOWN_COST)
    assert cm.cost[5, 20] == 0.0


def test_an_extra_layer_is_solid_to_everything_downstream():
    occ = np.zeros((60, 60), bool)
    low = np.zeros_like(occ)
    low[30, 30] = True     # a kerb only the camera sees
    v = view_of(occ, (0.5, 0.5, 0.0))._replace(layers=(low,))
    cm = cmod.build(v)
    assert cm.lethal[30, 30] and cm.lethal[32, 30]


# --- the planner -------------------------------------------------------------

def wall_room():
    """6 x 5 m, a wall across x = 3 from y = 0 to 4, a 1 m gap at the top."""
    occ = walled(120, 100)
    occ[59:61, :80] = True
    return occ


def test_a_path_round_a_wall_keeps_the_margin_all_the_way():
    v = view_of(wall_room(), (1.0, 1.0, 0.0))
    cm = cmod.build(v)
    path = pl.plan(cm, (1.0, 1.0, 0.0), (5.0, 1.0))
    assert path is not None
    end = path.poses[-1]
    assert math.hypot(end[0] - 5.0, end[1] - 1.0) <= pl.GOAL_TOL_M + 1e-9
    assert pl.path_clearance(cm, path) >= cm.margin
    # It went through the gap, not through the wall.
    assert path.poses[:, 1].max() > 4.0


def test_a_turn_too_tight_for_one_sweep_is_planned_with_a_reverse():
    # A 1.2 m corridor, truck facing the dead end, goal behind it: there is
    # no room to turn round forward, so the plan has to back up.
    occ = np.ones((140, 60), bool)
    occ[2:138, 18:42] = False
    v = view_of(occ, (5.5, 1.5, 0.0))
    cm = cmod.build(v)
    path = pl.plan(cm, (5.5, 1.5, 0.0), (2.0, 1.5),
                   params=pl.Params(max_expansions=20000))
    assert path is not None
    assert (path.dirs < 0).any()
    assert pl.path_clearance(cm, path) >= cm.margin


def test_no_way_through_is_no_plan_and_the_search_is_bounded():
    occ = walled(120, 100)
    occ[59:61, :] = True             # the wall goes all the way across
    cm = cmod.build(view_of(occ, (1.0, 1.0, 0.0)))
    path = pl.plan(cm, (1.0, 1.0, 0.0), (5.0, 1.0),
                   params=pl.Params(max_expansions=3000))
    assert path is None


# --- the follower --------------------------------------------------------------

def straight_path(x0, x1, y, step=0.05):
    xs = np.arange(x0, x1 + 1e-9, step)
    poses = np.stack([xs, np.full_like(xs, y), np.zeros_like(xs)], 1)
    n = len(xs)
    return pl.Path(poses, np.ones(n, int), np.zeros(n), float(x1 - x0), 0.0, 0, 0)


def run_follower(truck, follower, decisions):
    """Drive a sim truck with the follower for at most `decisions` 10 Hz
    decisions; the cross-track error at each."""
    errs = []
    for _ in range(decisions):
        cmd = follower.step(truck.pose, abs(truck.v))
        if cmd.done:
            break
        out = nv.Output(cmd.direction, cmd.speed, cmd.curvature)
        throttle, front, _ = nd.request_for(out, dv.FORWARD_SPEED)
        for _ in range(5):
            truck.step(throttle, front)
        errs.append(abs(truck.y - 2.0))
    return errs


LAZY_AXLE = nw.Response(gain=0.6, sigma_fwd=0.0, sigma_rev=0.0, wander=0.0,
                        scrub_rate=0.0, speed_wobble=0.0, name="lazy")


def test_the_follower_converges_on_the_path_with_an_axle_at_60_percent():
    world = nw.room("box", (14.0, 4.0))
    truck = nw.Truck(world, (1.0, 2.5, 0.0), LAZY_AXLE, np.random.default_rng(0))
    follower = fl.Follower(straight_path(1.0, 12.5, 2.0))
    errs = run_follower(truck, follower, 200)
    assert errs[0] > 0.3
    assert max(errs[-10:]) < 0.05
    assert truck.collisions == 0
    # It learned that the steering falls short (the steering lag makes the
    # estimate read high of the true 0.6, but well under 1).
    assert 0.45 < follower.gain.gain < 0.9


def test_without_the_learned_gain_a_lazy_axle_still_converges():
    world = nw.room("box", (14.0, 4.0))
    truck = nw.Truck(world, (1.0, 2.5, 0.0), LAZY_AXLE, np.random.default_rng(0))
    errs = run_follower(truck, fl.Follower(straight_path(1.0, 12.5, 2.0),
                                           adaptive=False), 200)
    assert max(errs[-10:]) < 0.05
    assert truck.collisions == 0


def test_the_gain_estimate_learns_how_much_of_the_steering_reaches_the_ground():
    g = fl.GainEstimate()
    sent, driven, step = 1.0, 0.6, 0.05
    x = y = yaw = 0.0
    for _ in range(80):
        g.update((x, y, yaw))
        g.remember((x, y, yaw), sent, 1)
        th1 = yaw + driven * step
        x += (math.sin(th1) - math.sin(yaw)) / driven
        y -= (math.cos(th1) - math.cos(yaw)) / driven
        yaw = th1
    assert g.gain == pytest.approx(0.6, abs=0.01)


def test_the_steering_command_inverts_the_measured_curve():
    for k in (-1.2, -0.5, 0.0, 0.4, 1.2):
        front = tk.front_for_curvature(k)
        assert tk.curvature_for_front(front) == pytest.approx(k, abs=1e-3)
    # Left is a negative command, and straight is a little right trim.
    assert tk.front_for_curvature(1.0) < 0
    assert 0 < tk.front_for_curvature(0.0) < 0.03
    assert abs(tk.front_for_curvature(99.0)) == pytest.approx(tk.MAX_TURN)
    assert math.isnan(tk.front_for_curvature(float("nan")))


# --- frontiers -------------------------------------------------------------------

def test_the_frontier_picked_is_the_reachable_one_and_a_blacklisted_one_is_skipped():
    # A known 4 x 3 m room, open (unknown) through a 1 m gap at x = 4 and a
    # 1.5 m gap at y = 3. The truck is near the x gap.
    occ = np.zeros((120, 100), bool)
    known = np.zeros_like(occ)
    known[:81, :61] = True
    occ[80, :61] = True
    occ[:81, 60] = True
    occ[0, :61] = True
    occ[:81, 0] = True
    occ[80, 20:40] = False           # the 1 m gap east
    occ[20:50, 60] = False           # the 1.5 m gap north
    pose = (3.2, 1.5, 0.0)
    cm = cmod.build(view_of(occ, pose, known=known))
    found = fr.find(cm, pose)
    assert len(found) >= 2
    best = found[0]
    assert best.x > 3.5               # the east gap, the near one
    assert all(np.isfinite(f.distance_m) for f in found)
    again = fr.find(cm, pose, blacklist=[(best.x, best.y)])
    assert again and again[0].y > 2.5   # now the north gap


def test_a_frontier_the_truck_cannot_reach_is_never_picked():
    occ = np.zeros((100, 60), bool)
    known = np.zeros_like(occ)
    known[:60] = True
    occ[30, :] = True                 # a wall right across: the far side is sealed off
    pose = (0.6, 1.5, 0.0)
    cm = cmod.build(view_of(occ, pose, known=known))
    assert fr.frontier_mask(cm)[59, 30]   # there is a frontier, on the far side
    assert fr.find(cm, pose) == []


# --- re-planning -----------------------------------------------------------------

def test_a_path_that_becomes_blocked_is_planned_again_round_the_block():
    occ = walled(160, 80)            # 8 x 4 m, empty
    pose = (1.0, 2.0, 0.0)
    nav = nv.Navigator()
    nav.set_goal(7.0, 2.0)
    out = nav.tick(T0, view_of(occ, pose, version=1))
    assert out.direction == 1
    first = nav.path
    assert first is not None and abs(first.poses[:, 1] - 2.0).max() < 0.3
    # A box lands on the straight line, 2 m ahead.
    occ2 = occ.copy()
    occ2[58:66, 32:48] = True
    nav.tick(T0 + 0.6, view_of(occ2, pose, version=2))
    assert nav.path is not first
    cm = cmod.build(view_of(occ2, pose))
    assert pl.path_clearance(cm, nav.path) >= cm.margin
    assert abs(nav.path.poses[:, 1] - 2.0).max() > 0.5


def test_a_goal_with_no_way_to_it_is_given_up_after_a_bounded_number_of_tries():
    occ = walled(120, 100)
    occ[59:61, :] = True
    nav = nv.Navigator(nv.Params(planner=pl.Params(max_expansions=1500)))
    nav.set_goal(5.0, 1.0)
    for i in range(20):
        out = nav.tick(T0 + i, view_of(occ, (1.0, 1.0, 0.0), version=1))
        assert out.direction == 0
        if nav.status == nv.ST_FAILED:
            break
    assert nav.status == nv.ST_FAILED
    assert nav.mode == nv.IDLE


def test_no_progress_is_a_wait_then_the_back_out_then_it_gives_up():
    occ = walled(160, 80)
    nav = nv.Navigator()
    nav.set_goal(7.0, 2.0)
    pose = (1.0, 2.0, 0.0)
    recovers = 0
    first_wait = first_recover = None
    for i in range(400):            # 40 s of a truck that never moves
        now = T0 + 0.1 * i
        out = nav.tick(now, view_of(occ, pose, version=1))
        if nav.status == nv.ST_WAIT and first_wait is None:
            first_wait = now
        if out.recover:
            recovers += 1
            first_recover = now if first_recover is None else first_recover
            nav.recovered(now)
        if nav.status == nv.ST_FAILED:
            break
    assert first_wait is not None and first_recover is not None
    assert first_wait < first_recover
    assert recovers == nv.MAX_RECOVERS
    assert nav.status == nv.ST_FAILED


def test_low_confidence_drives_on_the_local_map_not_the_global_one():
    occ = walled(160, 80)
    nav = nv.Navigator()
    nav.set_goal(7.0, 2.0)
    local = mv.LocalMap()
    # One scan: a wall 1.5 m ahead, nothing else.
    ys = np.linspace(1.0, 3.0, 41)
    local.add(T0, (1.0, 2.0), np.stack([np.full_like(ys, 2.5), ys], 1))
    v = view_of(occ, (1.0, 2.0, 0.0), version=1, confidence=0.1)
    nav.tick(T0, v, local)
    assert nav.local
    assert nav.cm.shape == (160, 160)          # the 8 m local window
    v2 = v._replace(confidence=0.9)
    nav.tick(T0 + 1.0, v2, local)
    assert not nav.local


# --- the Foxglove click ----------------------------------------------------------

def pose_in_frame(x, y, yaw=0.0, frame="map"):
    return json.dumps({
        "timestamp": {"sec": 1, "nsec": 2}, "frame_id": frame,
        "pose": {"position": {"x": x, "y": y, "z": 0.0},
                 "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(yaw / 2),
                                 "w": math.cos(yaw / 2)}}}).encode()


def test_a_foxglove_pose_click_parses_to_a_map_goal():
    g = ck.parse("foxglove.PoseInFrame", pose_in_frame(2.5, -1.25, math.pi / 2),
                 "/move_base_simple/goal", now=5.0)
    assert (g.x, g.y, g.frame_id, g.topic) == (2.5, -1.25, "map", "/move_base_simple/goal")
    assert g.yaw == pytest.approx(math.pi / 2)
    assert ck.to_map(g, (9.0, 9.0, 1.0)) == pytest.approx((2.5, -1.25, math.pi / 2))


def test_ros_shaped_clicks_parse_too():
    pose = {"header": {"stamp": {"sec": 0, "nsec": 0}, "frame_id": "map"},
            "pose": {"position": {"x": 1, "y": 2, "z": 0},
                     "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}}}
    g = ck.parse("geometry_msgs/PoseStamped", json.dumps(pose).encode())
    assert (g.x, g.y, g.yaw) == (1.0, 2.0, 0.0)
    point = {"header": {"frame_id": "map"}, "point": {"x": 3, "y": 4, "z": 0}}
    g = ck.parse("geometry_msgs/msg/PointStamped", json.dumps(point).encode())
    assert (g.x, g.y, g.yaw) == (3.0, 4.0, None)


@pytest.mark.parametrize("schema,data", [
    ("foxglove.PoseInFrame", b"not json"),
    ("foxglove.PoseInFrame", b"[1, 2]"),
    ("foxglove.PoseInFrame", json.dumps({"frame_id": "map", "pose": {}}).encode()),
    ("foxglove.PoseInFrame", pose_in_frame(float("nan"), 1.0)),
    ("foxglove.SceneUpdate", pose_in_frame(1.0, 1.0)),
])
def test_anything_that_is_not_a_goal_is_none(schema, data):
    assert ck.parse(schema, data) is None


def test_a_click_relative_to_the_truck_lands_ahead_of_it_on_the_map():
    g = ck.parse("foxglove.PoseInFrame", pose_in_frame(1.0, 0.0, frame="base_link"))
    x, y, _ = ck.to_map(g, (2.0, 3.0, math.pi / 2))
    assert (x, y) == pytest.approx((2.0, 4.0))
    other = ck.parse("foxglove.PoseInFrame", pose_in_frame(1.0, 0.0, frame="odom"))
    assert ck.to_map(other, (0.0, 0.0, 0.0)) is None


def test_the_listener_keeps_the_newest_json_click_and_ignores_the_rest():
    goals = ck.ClickGoals()
    client = SimpleNamespace(id=1)
    goals.on_client_advertise(client, SimpleNamespace(
        id=7, topic="/move_base_simple/goal", schema_name="foxglove.PoseInFrame",
        encoding="json"))
    goals.on_client_advertise(client, SimpleNamespace(
        id=8, topic="/other", schema_name="foxglove.PoseInFrame", encoding="protobuf"))
    goals.on_message_data(client, 7, pose_in_frame(1.0, 1.0))
    goals.on_message_data(client, 7, pose_in_frame(2.0, 2.0))
    goals.on_message_data(client, 8, pose_in_frame(9.0, 9.0))     # not json
    goals.on_message_data(client, 99, pose_in_frame(9.0, 9.0))    # never advertised
    g = goals.take()
    assert (g.x, g.y) == (2.0, 2.0)
    assert goals.take() is None
    assert goals.received == 2 and goals.rejected == 1
    goals.on_client_unadvertise(client, 7)
    goals.on_message_data(client, 7, pose_in_frame(3.0, 3.0))
    assert goals.take() is None


def test_the_server_options_turn_on_client_publish_with_json_only():
    goals = ck.ClickGoals()
    opts = goals.server_options()
    from foxglove.websocket import Capability
    assert opts["capabilities"] == [Capability.ClientPublish]
    assert opts["supported_encodings"] == ["json"]
    assert opts["server_listener"] is goals


# --- footprint, turn heuristic, planning thread, look-ahead ---------------------

def circles_only(cm, x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return min(float(cm.dist_at(x + c * cx, y + s * cx)) - tk.CIRCLE_R_M
               for cx in tk.CIRCLE_X_M)


def test_the_footprint_is_the_rectangle_not_the_circles_past_the_bumper():
    occ = np.zeros((80, 80), bool)
    occ[40, :] = True                        # a wall across, from x = 2.0
    cm = cmod.build(view_of(occ, (1.0, 1.0, 0.0)))
    x = 2.0 - 0.05 - tk.FRONT_M               # the bumper 5 cm short of it
    assert circles_only(cm, x, 2.0, 0.0) < 0.0      # the circles say touching
    assert 0.0 < cm.footprint_clearance(x, 2.0, 0.0) <= 0.05 + RES


def test_the_footprint_is_the_rectangle_not_the_circles_beside_the_truck():
    occ = np.zeros((80, 80), bool)
    occ[:, 40] = True                        # a wall along, from y = 2.0
    cm = cmod.build(view_of(occ, (1.0, 1.0, 0.0)))
    y = 2.0 - 0.01 - tk.HALF_WIDTH_M          # the side 1 cm off it
    assert circles_only(cm, 2.0, y, 0.0) < 0.0
    assert 0.0 < cm.footprint_clearance(2.0, y, 0.0) <= 0.01 + RES


def test_something_inside_the_footprint_is_touching():
    occ = np.zeros((80, 80), bool)
    occ[40, 40] = True
    cm = cmod.build(view_of(occ, (1.0, 1.0, 0.0)))
    assert cm.footprint_clearance(2.0, 2.0, 0.3) < 0.0


def test_the_turn_distance_knows_a_goal_behind_or_beside_is_a_turn_away():
    R = 0.85
    ahead = pl.turn_distance(0.0, 0.0, 0.0, 2.0, 0.0, R)
    assert float(ahead) == pytest.approx(2.0, abs=1e-6)
    beside = pl.turn_distance(0.0, 0.0, 0.0, 0.0, 2 * R, R, reverse=False)
    assert float(beside) == pytest.approx(math.pi * R, abs=1e-6)
    # 1 m behind: backing up (at three times the cost) beats a lap.
    behind = pl.turn_distance(0.0, 0.0, 0.0, -1.0, 0.0, R, reverse_factor=3.0)
    assert float(behind) == pytest.approx(3.0, abs=1e-6)
    # Never less than the straight line.
    rng = np.random.default_rng(1)
    pts = rng.uniform(-3, 3, (200, 2))
    d = pl.turn_distance(0.0, 0.0, 0.0, pts[:, 0], pts[:, 1], R)
    assert (d >= np.hypot(pts[:, 0], pts[:, 1]) - 1e-9).all()


def test_a_turnaround_in_open_floor_is_found_without_flooding_the_search():
    occ = walled(120, 120)                    # 6 x 6 m, empty
    cm = cmod.build(view_of(occ, (3.0, 3.0, 0.0)))
    path = pl.plan(cm, (3.0, 3.0, 0.0), (3.0, 4.6))      # beside it, to the left
    assert path is not None
    assert path.expansions < 300
    assert pl.path_clearance(cm, path) >= cm.margin
    end = path.poses[-1]
    assert math.hypot(end[0] - 3.0, end[1] - 4.6) <= pl.GOAL_TOL_M + 1e-9


def test_the_turn_heuristic_steers_a_search_round_to_a_goal_behind():
    # 2.5 m wide, goal 2 m straight behind, the analytic shot off so only
    # the search is measured. The 2D distance alone says "go straight
    # back"; the turn distance knows that is a turn away, and it is the
    # difference between a few dozen expansions and a few hundred.
    occ = np.ones((140, 50), bool)
    occ[2:138, 2:48] = False
    cm = cmod.build(view_of(occ, (5.5, 1.25, 0.0)))
    path = pl.plan(cm, (5.5, 1.25, 0.0), (3.5, 1.25),
                   params=pl.Params(max_expansions=20000, shot=False))
    assert path is not None
    assert path.expansions < 120
    assert pl.path_clearance(cm, path) >= cm.margin


def test_a_plan_never_drives_forward_with_the_floor_veto_box_on_a_wall():
    # A wall 1.2 m ahead with a door in it 1.3 m to the left: the short way
    # noses into the wall to make the turn, which is where the camera would
    # hold the car. With the check on, the plan keeps the box off it, and
    # only the box: the last stretch to the goal is exempt, and so is the
    # start. (It is off by default; planner.PROBE_CLEAR_M says why.)
    occ = walled(140, 120)
    occ[44:46, :] = True
    occ[44:46, 47:65] = False
    start, goal = (1.0, 1.5, 0.0), (4.2, 2.8)
    cm = cmod.build(view_of(occ, start))
    path = pl.plan(cm, start, goal,
                   params=pl.Params(max_expansions=20000, floor_probe=True))
    assert path is not None
    p = path.poses
    m = ((path.dirs > 0)
         & (np.hypot(p[:, 0] - goal[0], p[:, 1] - goal[1]) > pl.PROBE_FREE_M)
         & (np.hypot(p[:, 0] - start[0], p[:, 1] - start[1]) > 0.3))
    assert m.any()
    assert cm.probe_clearance(p[m, 0], p[m, 1], p[m, 2]).min() >= pl.PROBE_CLEAR_M
    # And the box itself: on the wall it reads 0, a metre short of it more.
    assert cm.probe_clearance(1.6, 1.5, 0.0) == 0.0
    assert cm.probe_clearance(0.8, 1.5, 0.0) > 0.3


def test_the_plan_radius_is_most_of_the_measured_lock():
    r = 1.0 / (tk.MAX_CURVATURE * pl.PLAN_CURVATURE)
    assert tk.TURN_RADIUS_M == pytest.approx(0.75, abs=0.01)
    assert 0.8 <= r <= 0.9


class SlowPlans:
    """pl.plan, a fixed time slower."""

    def __init__(self, delay):
        self.delay = delay
        self.real = pl.plan

    def __call__(self, *a, **k):
        import time as _t
        _t.sleep(self.delay)
        return self.real(*a, **k)


def test_a_slow_plan_never_holds_up_a_decision_and_it_stands_still_meanwhile(monkeypatch):
    import time as _t
    monkeypatch.setattr(pl, "plan", SlowPlans(0.4))
    planner = nv.ThreadPlanner()
    try:
        nav = nv.Navigator(planner=planner)
        nav.set_goal(7.0, 2.0)
        v = view_of(walled(160, 80), (1.0, 2.0, 0.0), version=1)
        t0 = _t.perf_counter()
        out = nav.tick(T0, v)
        assert _t.perf_counter() - t0 < 0.2
        assert out.direction == 0
        moved = None
        for i in range(100):                  # at most 2 s
            _t.sleep(0.02)
            t1 = _t.perf_counter()
            out = nav.tick(T0 + 0.1 * (i + 1), v)
            assert _t.perf_counter() - t1 < 0.2
            if out.direction != 0:
                moved = i
                break
        assert moved is not None and moved >= 5   # not before the plan was in
        assert out.direction == 1
    finally:
        planner.close()


def test_a_plan_asked_for_before_a_new_goal_is_thrown_away():
    planner = nv.InlinePlanner(delay_scale=1e6)     # answers arrive much later
    nav = nv.Navigator(planner=planner)
    nav.set_goal(7.0, 2.0)
    v = view_of(walled(160, 80), (1.0, 2.0, 0.0), version=1)
    nav.tick(T0, v)
    assert planner.busy and nav.path is None
    nav.set_goal(1.0, 3.5)
    planner._pending = (T0, *planner._pending[1:])  # let it land now
    nav.tick(T0 + 0.1, v)
    assert nav.path is None


def test_a_command_that_would_run_into_something_is_a_stop():
    occ = walled(160, 80)
    occ[60:62, :] = True                       # a wall at x = 3.0
    nav = nv.Navigator()
    cm = cmod.build(view_of(occ, (2.0, 2.0, 0.0)))
    x = 3.0 - 0.3 - tk.FRONT_M                 # 30 cm short of it
    toward = fl.Command(1, 1.0, 0.0)
    away = fl.Command(-1, 1.0, 0.0)
    assert nav._forecast_hits(cm, (x, 2.0, 0.0), toward, 0.5)
    assert not nav._forecast_hits(cm, (x, 2.0, 0.0), away, 0.5)
    assert not nav._forecast_hits(cm, (x - 1.0, 2.0, 0.0), toward, 0.5)


def test_the_follower_aims_from_where_the_truck_will_be_not_where_it_was():
    f = fl.Follower(straight_path(0.0, 6.0, 0.0), delay_s=0.3)
    f._driving = (1, 1.0)                     # on a 1 m left circle now
    x, y, yaw = f.predict((1.0, 0.0, 0.0), 0.5)
    assert x == pytest.approx(1.0 + math.sin(0.15), abs=1e-6)
    assert y == pytest.approx(1.0 - math.cos(0.15), abs=1e-6)
    assert yaw == pytest.approx(0.15)
    now = fl.Follower(straight_path(0.0, 6.0, 0.0), delay_s=0.0)
    now._driving = (1, 1.0)
    assert now.predict((1.0, 0.0, 0.0), 0.5) == (1.0, 0.0, 0.0)


def opening_room():
    """4 x 5 m known, a 1.5 m opening in the east wall onto unknown."""
    occ = np.zeros((120, 100), bool)
    known = np.zeros_like(occ)
    known[:81, :] = True
    occ[:81, 0] = occ[:81, -1] = occ[0, :] = True
    occ[80, :] = True
    occ[80, 30:60] = False
    return occ, known


def test_a_frontier_goal_is_moved_off_the_wall_it_was_found_against():
    occ, known = opening_room()
    pose = (2.0, 0.6, 0.0)
    cm = cmod.build(view_of(occ, pose, known=known))
    found = fr.find(cm, pose)
    assert found
    ix, iy = cm.world_to_cell(found[0].x, found[0].y)
    assert cm.dist[ix, iy] >= 0.4             # not shaving the frame
    # A point in the opening's bottom corner, 0.25 m off the jamb, comes
    # out with room round it, and not further than SNAP_M.
    reach = pl.grid_distances(cm, [pose[:2]], known_only=True)
    x, y = fr.snap_clear(cm, reach, 4.025, 1.75)
    ix, iy = cm.world_to_cell(x, y)
    assert cm.dist[ix, iy] >= 0.45
    assert math.hypot(x - 4.025, y - 1.75) <= fr.SNAP_M + 1e-9


def test_a_blacklisted_frontier_comes_back_once_the_entry_expires():
    occ, known = opening_room()
    pose = (2.0, 2.0, 0.0)
    v = view_of(occ, pose, known=known, version=1)
    first = fr.find(cmod.build(v), pose)[0]
    nav = nv.Navigator()
    nav.rove()
    nav.blacklist = [(first.x, first.y, T0 - 1.0)]
    nav.tick(T0, v)
    assert nav.target is None or math.hypot(nav.target[0] - first.x,
                                            nav.target[1] - first.y) > fr.BLACKLIST_M
    nav = nv.Navigator()
    nav.rove()
    nav.blacklist = [(first.x, first.y, T0 - nv.BLACKLIST_S - 1.0)]
    nav.tick(T0, v)
    assert nav.target == (first.x, first.y)


# --- underneath it: the car's own safety layers ------------------------------------

class FixedMap:
    def __init__(self, view):
        self._view = view

    def view(self):
        return self._view


def clear_depth(mm=3000):
    return np.full((400, 640), mm, dtype=np.uint16)


def nav_driver(depth, occ=None, pose=(1.0, 2.0, 0.0), goal=(7.0, 2.0), lidar=None):
    occ = walled(160, 80) if occ is None else occ
    nav = nv.Navigator()
    nav.set_goal(*goal)
    return nd.NavDriver(FakeController(), FakeCamera(depth=depth),
                        FixedMap(view_of(occ, pose, version=1)), nav, lidar=lidar)


def test_the_navigators_forward_goes_out_capped_at_forward_speed():
    d = nav_driver(clear_depth())
    d.tick(T0)
    throttle = d.controller.throttles[-1]
    assert nd.MIN_FORWARD <= throttle <= d.forward_speed
    front, rear = d.controller.steerings[-1]
    assert rear == pytest.approx(-front)
    # A forward speed set under the floor the navigator would otherwise
    # lift a slow command to is still the cap.
    out = nv.Output(1, 1.0, 0.0, False, nv.ST_DRIVING, "follow", False, None)
    throttle, _, _ = nd.request_for(out, 0.2)
    assert 0.0 < throttle <= 0.2


def test_the_floor_still_holds_a_navigator_forward():
    depth = clear_depth()
    depth[:, 300:340] = 150          # inside REVERSE_FLOOR_MM, dead ahead
    d = nav_driver(depth)
    d.tick(T0)
    assert d.floor_veto.vetoes()
    assert d.controller.throttles[-1] == 0.0
    assert d.floor_held


def test_a_floor_held_that_long_is_the_back_out():
    depth = clear_depth()
    depth[:, 300:340] = 150
    d = nav_driver(depth)
    for i in range(int(nd.STANDOFF_S / 0.1) + 2):
        d.tick(T0 + 0.1 * i)
        if d._phase is not None:
            break
    assert d._phase is not None


class RearLidar:
    def __init__(self, mm):
        bins = np.zeros(ld.BINS, dtype=np.uint16)
        bins[int(round(180.0 / ld.BIN_DEG)) % ld.BINS] = mm
        self.scan = SimpleNamespace(bins=bins)

    def latest(self):
        return self.scan


def test_the_rear_stop_still_cuts_a_navigator_reverse():
    # Facing a wall 0.5 m ahead in a narrow dead end, goal behind: the plan
    # starts in reverse, and something is right behind the bumper.
    occ = np.ones((140, 60), bool)
    occ[2:138, 18:42] = False
    near = int(dv.REAR_SELF_MM + dv.REAR_STOP_CLEAR_MM - 40)
    d = nav_driver(clear_depth(), occ, pose=(6.3, 1.5, 0.0), goal=(2.0, 1.5),
                   lidar=RearLidar(near))
    d.nav.p = d.nav.p._replace(planner=pl.Params(max_expansions=20000))
    for i in range(20):
        d.tick(T0 + 0.1 * i)
        if d.rear_stop:
            break
    assert d.rear_stop
    assert d.controller.throttles[-1] == 0
    assert all(t >= 0 for t in d.controller.throttles)


def test_no_frame_is_neutral_under_the_navigator():
    d = nav_driver(None)
    d.tick(T0)
    assert d.controller.throttles[-1] == 0
    assert d.controller.steerings[-1] == (0, 0)


# --- off by default ------------------------------------------------------------------

class ReadyHub:
    from crawler.hardware import camera as _camera
    state = _camera.READY
    error = ""

    def view(self, name):
        return FakeCamera()


class Log:
    def __init__(self):
        self.lines = []

    def log(self, level, subject, message):
        self.lines.append((level, subject, message))


def test_nav_is_off_by_default_and_auto_is_the_hand_written_autopilot():
    args = mn.parse_args([])
    assert args.nav is False
    pilot = mn.AutoPilot(FakeController(), Log(), hub=ReadyHub())
    assert pilot.nav is False
    assert pilot.resume() is True
    assert type(pilot.driver) is dv.Driver
    assert tm.Telemetry().server_options == {}


def test_nav_with_no_map_is_auto_unavailable_never_a_blind_drive():
    args = mn.parse_args(["--nav"])
    log = Log()
    pilot = mn.AutoPilot(FakeController(), log, hub=ReadyHub(), nav=args.nav)
    assert pilot.resume() is False
    assert pilot.unavailable is True
    assert pilot.driver is None
    assert any("map" in m for _, _, m in log.lines)


def test_nav_with_a_map_builds_the_nav_driver_on_the_held_command():
    occ = walled(80, 80)
    pilot = mn.AutoPilot(FakeController(), Log(), hub=ReadyHub(), nav=True,
                         map_source=FixedMap(view_of(occ)))
    assert pilot.resume() is True
    assert isinstance(pilot.driver, nd.NavDriver)
    assert pilot.driver.controller is pilot.command
