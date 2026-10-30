"""Following a person: autopilot/follow.py.

The person the windshield camera sees, put on the SLAM map off the lidar (or
the OAK-D, or the box), is the goal the goal model drives to; within the
follow gap, or with nobody seen, the truck holds neutral. These hold the
geometry, the choice of who to follow, the lost hold, the gap, the Follow
buttons and their topic, /follow/status and /follow/marker, the wiring in
AutoPilot, and a simulated follow of a walking person through the real
PolicyDriver with a stub model that only reads its goal input.
"""

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler.autopilot import follow as fl  # noqa: E402
from crawler.autopilot import policy_goal as pg  # noqa: E402
from crawler.hardware import lidar as ld  # noqa: E402
from crawler.nav import click as ck  # noqa: E402
from crawler.telemetry import follow_view as fv  # noqa: E402
from crawler.telemetry import telemetry as tm  # noqa: E402
from crawler.vision import track as tk  # noqa: E402
from crawler.vision.runner import Sighting  # noqa: E402
from crawler.vision.track import Detection  # noqa: E402

from test_goal_reseed import crop_at, link  # noqa: E402
from test_lidar_view import decode  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAM = fl.CameraGeometry(105.0, 72.5, 0.0, 45.0, 0.15, 0.0, 0.22)


# --- fakes -----------------------------------------------------------------------------

def scan_of(points_xy, t=0.0):
    """A lidar.Scan with returns at these car-frame (x, y) metres."""
    xy = np.asarray(points_xy, float).reshape(-1, 2)
    b = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
    phi = np.mod(-90.0 - b, 360.0)               # lidar.car_bearing_deg inverted
    d = (np.hypot(xy[:, 0], xy[:, 1]) * 1000.0).astype(np.uint16)
    return ld.Scan(t, phi, d, np.zeros(len(d), np.uint8), 0.0, 1)


def wall_and(person_xy=(), t=0.0, wall_x=4.0, body=0.12):
    """A wall wall_x ahead, and a person-sized arc of returns at each point."""
    pts = [(wall_x, y) for y in np.linspace(-3, 3, 121)]
    for px, py in person_xy:
        for a in np.linspace(-math.pi / 2, math.pi / 2, 9):
            r = math.hypot(px, py)
            ux, uy = px / r, py / r
            pts.append((px - body * ux * math.cos(a) - body * uy * math.sin(a),
                        py - body * uy * math.cos(a) + body * ux * math.sin(a)))
    return scan_of(pts, t)


class Lidar:
    def __init__(self, scan=None):
        self.scan = scan

    def latest(self):
        return self.scan


class Sightings:
    def __init__(self):
        self.s = None
        self.seq = 0

    def see(self, people, t):
        self.seq += 1
        self.s = Sighting(self.seq, t, tuple(people), 0.0, 0.0)

    def sighting(self):
        return self.s


class Poses:
    """pose_at off a list of (t, x, y, yaw)."""

    def __init__(self):
        self.at = []

    def __call__(self, t):
        if not self.at:
            return None
        best = min(self.at, key=lambda p: abs(p[0] - t))
        return SimpleNamespace(x=best[1], y=best[2], yaw=best[3])


def u_for(bearing_deg, v=0.55, cam=CAM):
    """The frame column a car-frame bearing from the camera lands on (row v)."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if cam.ray(mid, v)[0] > bearing_deg:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def person_box(ax, ay, cam=CAM, w=0.08):
    """The box a person at car-frame (ax, ay) makes: its centre column off
    the camera's bearing to them, cut off top and bottom as they are close."""
    b = math.degrees(math.atan2(ay - cam.y_m, ax - cam.x_m))
    u = u_for(b)
    return Detection(0.8, u - w, 0.05, u + w, 0.995)


def goal_with(scan=None, **kw):
    s, poses = Sightings(), Poses()
    g = fl.PersonGoal(CAM, sightings=s, lidar=Lidar(scan), pose_at=poses, **kw)
    g.command({"cmd": "on"})
    return g, s, poses


# --- geometry --------------------------------------------------------------------------

def test_a_person_in_the_lidar_is_the_nearest_cluster_in_their_bearings_not_a_stray():
    scan = wall_and([(2.0, 0.5)])
    pts = fl.lidar_points(scan)
    b = math.degrees(math.atan2(0.5, 2.0 - CAM.x_m))
    r = fl.lidar_range(pts, CAM, b - 5, b + 5)
    assert r == pytest.approx(math.hypot(2.0 - CAM.x_m, 0.5), abs=0.15)
    # One stray return in front is not a person; the wall behind a gap is.
    stray = scan_of([(0.8, 0.0)] + [(4.0, y) for y in np.linspace(-0.3, 0.3, 21)])
    assert fl.lidar_range(fl.lidar_points(stray), CAM, -4, 4) == pytest.approx(3.85, abs=0.05)
    assert fl.lidar_range(fl.lidar_points(scan_of([(0.8, 0.0)])), CAM, -4, 4) is None


def test_a_box_gives_a_distance_off_its_foot_only():
    # A person 3 m off, on a level camera 0.22 m up, boxed only to the waist
    # (crouched, or half behind something): the foot on the floor, not the
    # box's height, which would put them 5 m off and the goal past them.
    # The recorded boxes at 1-1.5 m were mostly partial people like this,
    # and a height read put them 1.5-2.2x too far.
    cam = fl.CameraGeometry(105.0, 72.5, 0.0, 0.0, 0.0, 0.0, 0.22)
    waist = math.degrees(math.atan2(1.0 - 0.22, 3.0))
    foot = math.degrees(math.atan2(-0.22, 3.0))
    ty = math.tan(math.radians(72.5 / 2))
    v = lambda e: 0.5 - math.tan(math.radians(e)) / (2 * ty)   # noqa: E731
    partial = Detection(0.9, 0.45, v(waist), 0.55, v(foot))
    assert fl.box_range(partial, cam) == pytest.approx(3.0, rel=0.02)
    cut = Detection(0.9, 0.45, 0.0, 0.55, v(foot))
    assert fl.box_range(cut, cam) == pytest.approx(3.0, rel=0.02)
    # Foot cut off: nothing.
    assert fl.box_range(Detection(0.9, 0.4, 0.0, 0.6, 1.0), cam) is None
    # The windshield camera looks 45 up and sees no floor at all, so a box
    # alone never places anybody: the lidar or the OAK-D does.
    wind = fl.CameraGeometry(105.0, 72.5, 0.0, 45.0, 0.15, 0.0, 0.22)
    assert fl.box_range(Detection(0.9, 0.4, 0.3, 0.6, 0.97), wind) is None


def test_a_sighting_lands_on_the_map_off_the_pose_its_frame_was_taken_at():
    # The truck was at (1, 0) facing +y when the frame was read, and has
    # driven on to (1, 0.5) by the time the answer comes: the person goes
    # where they were seen from, not 0.5 m on.
    g, s, poses = goal_with(wall_and([(2.0, 0.0)]))
    poses.at = [(10.0, 1.0, 0.0, math.pi / 2), (10.4, 1.0, 0.5, math.pi / 2)]
    g.lidar.scan = g.lidar.scan._replace(t=10.0)
    s.see([person_box(2.0, 0.0)], 10.0)
    g.poll(crop_at(1.0, 0.5, math.pi / 2, t=10.4))
    x, y, frame = g.goal
    assert (x, y, frame) == pytest.approx((1.0, 2.0, 0), abs=0.12)
    assert g.source == "lidar"
    np.testing.assert_allclose(g.encode(crop_at(1.0, 0.5, math.pi / 2, t=10.4)),
                               pg.encode(crop_at(1.0, 0.5, math.pi / 2), g.goal))


def test_no_lidar_return_falls_back_to_the_oakd_then_nothing():
    calls = []
    g, s, poses = goal_with(scan_of([(4.0, 3.0)]),
                            oakd=lambda b, t: calls.append(b) or 2.5)
    poses.at = [(5.0, 0.0, 0.0, 0.0)]
    s.see([person_box(2.5, 0.0)], 5.0)
    g.poll(crop_at(t=5.0))
    assert g.source == "oakd" and calls
    assert g.goal[0] == pytest.approx(CAM.x_m + 2.5, abs=0.05)
    # Nothing at all (the box is cut off top and bottom): not placed.
    g2, s2, poses2 = goal_with(scan_of([(4.0, 3.0)]))
    poses2.at = [(5.0, 0.0, 0.0, 0.0)]
    s2.see([person_box(2.5, 0.0)], 5.0)
    g2.poll(crop_at(t=5.0))
    assert g2.goal is None and g2.unplaced == 1


# --- who, lost, the gap ------------------------------------------------------------------

def test_it_sticks_to_the_person_it_follows_when_somebody_else_is_nearer():
    g, s, poses = goal_with()
    poses.at = [(t, 0.0, 0.0, 0.0) for t in (1.0, 2.0)]
    g.lidar.scan = wall_and([(2.0, 1.0)], t=1.0)
    s.see([person_box(2.0, 1.0)], 1.0)
    g.poll(crop_at(t=1.0))
    first = g.goal
    assert first[:2] == pytest.approx((2.0, 1.0), abs=0.15)
    # A second person steps in nearer, on the other side; the first walks on a bit.
    g.lidar.scan = wall_and([(1.3, -0.8), (2.2, 1.1)], t=2.0)
    s.see([person_box(1.3, -0.8), person_box(2.2, 1.1)], 2.0)
    g.poll(crop_at(t=2.0))
    assert math.hypot(g.goal[0] - 2.2, g.goal[1] - 1.1) < 0.3
    # Only the other one in view: somebody else, the point stays.
    g.lidar.scan = wall_and([(1.3, -0.8)], t=2.5)
    s.see([person_box(1.3, -0.8)], 2.5)
    held = g.goal
    g.poll(crop_at(t=2.5))
    assert g.goal == held and g.ignored == 1


def test_nobody_for_lost_s_is_no_goal_and_a_hold_and_a_new_sighting_ends_it():
    g, s, poses = goal_with()
    assert g.reached() and g.state() == "searching", "on, nobody yet: hold"
    poses.at = [(t, 0.0, 0.0, 0.0) for t in np.arange(0.0, 10.0, 0.5)]
    g.lidar.scan = wall_and([(3.0, 0.0)], t=1.0)
    s.see([person_box(3.0, 0.0)], 1.0)
    g.poll(crop_at(t=1.0))
    assert g.state() == "following" and not g.reached()
    g.poll(crop_at(t=1.0 + fl.LOST_S - 0.1))
    assert g.goal is not None, "held a moment"
    g.poll(crop_at(t=1.0 + fl.LOST_S + 0.1))
    assert g.state() == "lost" and g.goal is None and g.reached()
    assert not g.encode(crop_at()).any(), "present 0"
    g.lidar.scan = wall_and([(3.0, 0.0)], t=6.0)
    s.see([person_box(3.0, 0.0)], 6.0)
    g.poll(crop_at(t=6.0))
    assert g.state() == "following" and not g.reached()


def test_the_gap_holds_inside_gap_m_and_lets_go_past_resume_m():
    g, s, poses = goal_with()
    g.point, g.seen_t = (2.0, 0.0, 0), 0.0
    for x, held in ((0.0, False), (0.7, False), (0.85, True), (0.6, True),
                    (0.45, True), (0.35, False)):
        g.observe(crop_at(x, t=0.1))
        assert g.reached() is held, x
    off = fl.PersonGoal(CAM)
    assert not off.reached(), "off: never a hold"


def test_a_followed_point_rides_a_reseed_link():
    g, s, poses = goal_with()
    g.point, g.seen_t = (2.0, 0.0, 0), 0.0
    g.observe(crop_at(frame=1, t=0.1, links=(link(1, 0, 1, (1.0, 0.0, math.pi / 2)),)))
    assert g.point == pytest.approx((1.0, 2.0, 1))


# --- the switch and its buttons ------------------------------------------------------------

class Events:
    def __init__(self, *events):
        self.events = list(events)

    def take_events(self):
        out, self.events = self.events, []
        return out


def test_the_switch_hands_the_goal_between_the_patrol_and_the_person():
    from crawler.autopilot import patrol as pt
    from test_goal_reseed import click
    p = pt.Patrol()
    g, s, poses = goal_with()
    g.command({"cmd": "off"})
    sw = fl.FollowSwitch(p, g)
    sw.poll(Events(("goal", click(3.0, 0.0))), crop_at())
    assert sw.use == "goal" and sw.goal == (3.0, 0.0, 0) and not sw.reached()
    sw.poll(Events(("follow", {"cmd": "on"}), ("goal", click(4.0, 0.0))), crop_at())
    assert sw.use == "follow" and sw.reached(), "on with nobody seen: hold"
    assert sw.goal is None, "following nobody: no goal, not the patrol's"
    assert p.goal == (4.0, 0.0, 0), "a click while following still goes to the patrol"
    assert not sw.encode(crop_at()).any()
    sw.poll(Events(("follow", {"cmd": "off"})), crop_at())
    assert sw.use == "goal" and sw.goal == (4.0, 0.0, 0)
    assert g.command({"cmd": "nope"}) is None


def test_a_followed_person_is_a_goal_source_the_planner_paths_round_a_wall_to():
    # pathgoal.GoalSource is goal + encode; the switch is what AutoPilot
    # holds, so the planner reads the person off it. Person 3.5 m ahead,
    # a wall between at 2 m: the model is fed a point along the way round,
    # not the person.
    from crawler.autopilot import patrol as pt
    from test_pathgoal import WALL, planner, world
    g, s, poses = goal_with()
    poses.at = [(1.0, 0.0, 0.0, 0.0)]
    g.lidar.scan = wall_and([(3.5, 0.0)], t=1.0)
    s.see([person_box(3.5, 0.0)], 1.0)
    sw = fl.FollowSwitch(pt.Patrol(), g)
    sw.poll(Events(), crop_at(t=1.0))
    assert sw.goal[:2] == pytest.approx((3.5, 0.0), abs=0.15)
    p, _ = planner(world([WALL], t=2.0))
    fed = p.encode(crop_at(t=1.0), sw)
    assert p.status()["path"] and p.fed == "subgoal"
    assert not np.allclose(fed, sw.encode(crop_at(t=1.0)))
    # Lost: no goal, nothing planned, and the switch's own none is fed.
    g.observe(crop_at(t=1.0 + fl.LOST_S + 0.5))
    assert sw.goal is None
    assert not p.encode(crop_at(t=1.0 + fl.LOST_S + 0.5), sw).any()


def test_the_listener_puts_follow_commands_in_the_queue():
    listener = ck.ClickGoals()
    client, chan = SimpleNamespace(id=1), SimpleNamespace(id=7, topic=ck.FOLLOW_TOPIC,
                                                          schema_name="x", encoding="json")
    listener.on_client_advertise(client, chan)
    listener.on_message_data(client, 7, b'{"cmd": "on"}')
    listener.on_message_data(client, 7, b'nope')
    assert listener.take_events() == [("follow", {"cmd": "on"})]
    assert listener.rejected == 1


def test_the_drive_layout_has_follow_on_and_off_and_draws_the_person():
    layout = json.load(open(os.path.join(ROOT, "deploy", "foxglove-drive.json")))
    cfg = layout["configById"]
    sent = set()
    for key, b in cfg.items():
        if key.startswith("Publish!") and b["topicName"] == ck.FOLLOW_TOPIC:
            msg = json.loads(b["value"])
            assert fl.PersonGoal(CAM).command(msg) is True, key
            sent.add(msg["cmd"])
    assert sent == {"on", "off"}
    assert cfg["3D!slam"]["topics"]["/follow/marker"]["visible"]
    assert cfg["RawMessages!follow"]["topicPath"] == "/follow/status"

    from test_foxglove_layouts import panels
    shown = panels(layout["layout"], cfg)
    assert {"Publish!follow_on", "Publish!follow_off", "RawMessages!follow"} <= set(shown)
    assert sorted(shown) == sorted(cfg), "every panel configured is laid out"


def test_follow_status_matches_its_schema_and_the_marker_draws():
    g, s, poses = goal_with()
    poses.at = [(1.0, 0.0, 0.0, 0.0)]
    g.lidar.scan = wall_and([(3.0, 0.0)], t=1.0)
    s.see([person_box(3.0, 0.0)], 1.0)
    g.poll(crop_at(t=1.0))
    st = g.follow_status(crop_at(t=1.0))
    assert set(st) == set(tm.JSON_SCHEMAS["follow_status"]["schema"]["properties"])
    json.dumps(st)
    assert st["state"] == "following" and st["goal_input"][pg.PRESENT] == 1.0
    assert set(g.status()) == set(tm.JSON_SCHEMAS["goal_status"]["schema"]["properties"])
    assert tm.JSON_TOPICS["follow_status"] == "/follow/status"
    assert tm.NAV_CHANNELS["follow_marker"] == "/follow/marker"
    m = g.marker()
    assert m["person"] == pytest.approx(g.point[:2])
    up = decode(tm.Telemetry.lidar_view_message("follow_marker", m, 1))
    assert [e.id for e in up.entities] == [fv.PERSON_ID, fv.GAP_ID]
    assert up.entities[0].cylinders[0].pose.position.x == pytest.approx(g.point[0])
    gone = decode(fv.scene({}, 1))
    assert not gone.entities and {d.id for d in gone.deletions} == {fv.PERSON_ID, fv.GAP_ID}


def test_with_goal_and_a_person_the_autopilot_goal_is_the_switch():
    from crawler.autopilot import patrol as pt
    from test_policy_goal import _pilot
    from crawler import __main__ as mn
    mn_, log, slam, pilot = _pilot(Events())
    assert isinstance(pilot.goal, pt.Patrol), "no person: the patrol, as before"
    person = fl.PersonGoal(CAM)
    from conftest import FakeController
    from test_policy import ReadyHub
    from test_policy_goal import GOAL_MODEL
    pilot = mn.AutoPilot(FakeController(), log, hub=ReadyHub(), slam=slam,
                         policy_path=str(GOAL_MODEL), clicks=Events(("follow", {"cmd": "on"})),
                         person=person)
    assert isinstance(pilot.goal, fl.FollowSwitch)
    pilot.goal_step()
    assert person.active and pilot.goal.reached()
    assert pilot.resume() is True and pilot.driver.goal is pilot.goal
    args = mn.parse_args(["--goal"])
    assert mn.follow_person(args, None, None, slam, None, log) is None, "no --pantilt"


def test_nothing_in_follow_can_drive():
    text = open(os.path.join(ROOT, "crawler", "autopilot", "follow.py")).read()
    for word in ("set_throttle", "set_steering", "vecon", "controller", "pigpio",
                 "ESC_GPIO_PIN", "return_neutral"):
        assert word not in text, word


# --- a simulated follow ----------------------------------------------------------------------

def simulate(seconds, walk, latency=0.4, every=0.7):
    """The truck as a unicycle under the real PolicyDriver at 10 Hz, behind
    a person walking walk(t) -> (x, y) on the map. Every `every` seconds the
    detector answers about a frame `latency` old: the person's box off the
    truck's pose then, and a lidar scan of them; the PersonGoal puts them on
    the map and the stub model is fed nothing but that goal."""
    from test_patrol import GoalSteerStub
    from test_policy_goal import PosedSlam
    from test_policy_map import T0, driver
    slam = PosedSlam()
    model = GoalSteerStub()
    d = driver(model, slam)
    sightings, poses, lidar = Sightings(), Poses(), Lidar()
    g = fl.PersonGoal(CAM, sightings=sightings, lidar=lidar, pose_at=poses)
    g.command({"cmd": "on"})
    d.goal = g
    x, y, yaw = 0.0, 0.0, 0.0
    t, dt = T0, 0.1
    next_sight = T0 + 0.5
    pending = []
    gaps, holds = [], 0
    while t < T0 + seconds:
        t = round(t + dt, 6)
        poses.at.append((t, x, y, yaw))
        px, py = walk(t - T0)
        c, s = math.cos(yaw), math.sin(yaw)
        ax, ay = c * (px - x) + s * (py - y), -s * (px - x) + c * (py - y)
        lidar.scan = wall_and([(ax, ay)], t=t, wall_x=9.0)
        g._keep_scan()
        if t >= next_sight:
            pending.append((t + latency, [person_box(ax, ay)] if abs(math.atan2(ay, ax - CAM.x_m)) < 1.2 else [], t))
            next_sight = t + every
        while pending and pending[0][0] <= t:
            _, people, frame_t = pending.pop(0)
            sightings.see(people, frame_t)
        slam.c = slam.c._replace(x=x, y=y, yaw=yaw, t=t)
        g.poll(slam.crop())
        d.tick(t)
        thr = d.controller.throttles[-1] if d.controller.throttles else 0.0
        front = d.controller.steerings[-1][0] if d.controller.steerings else 0.0
        v = 0.8 * thr
        yaw += -v * front * 1.6 * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        gaps.append(math.hypot(px - x, py - y))
        holds += int(d.goal_reached)
    return SimpleNamespace(goal=g, gaps=np.array(gaps), holds=holds, fed=model.fed,
                           truck=(x, y), driver=d)


WALK_M_S = 0.15


def walk_then_stop(t):
    """Starts 2.5 m ahead, walks a curve at WALK_M_S for 30 s, then stands.
    Slowly: auto's forward is 0.24, about 0.2 m/s, so anyone walking faster
    than that walks away from it whatever the goal says."""
    s = min(t, 30.0) * WALK_M_S
    return 2.5 + 3.0 * math.sin(s / 3.0), 3.0 - 3.0 * math.cos(s / 3.0)


def test_a_simulated_follow_keeps_the_gap_behind_a_walking_person_and_stops_when_they_do():
    r = simulate(60.0, walk_then_stop)
    after = r.gaps[50:]          # past the first 5 s, catching up
    print(f"\nsim follow: gap {after.min():.2f}-{after.max():.2f} m while walking, "
          f"{r.gaps[-1]:.2f} m standing, held {r.holds} of {len(r.gaps)} decisions, "
          f"{r.goal.placed} sightings placed, {r.goal.ignored} ignored")
    assert r.goal.placed > 40 and r.goal.ignored == 0
    # Never runs them over, never falls far behind.
    assert after.min() > 0.8
    assert after.max() < 3.0
    # Standing still: it closes to the gap and holds there.
    assert fl.GAP_M - 0.4 <= r.gaps[-1] <= fl.RESUME_M + 0.1
    assert r.driver.goal_reached and r.goal.state() == "gap"
    assert r.fed > 0


def test_a_simulated_follow_holds_when_the_person_walks_out_of_view():
    # Steps round behind the truck at 8 s: out of the camera, lost, and held.
    r = simulate(20.0, lambda t: (3.0, 0.0) if t < 8.0 else (-2.0, 0.0))
    assert r.goal.state() == "lost" and r.driver.goal_reached
    assert not r.goal.encode(r.driver.slam.crop()).any()
