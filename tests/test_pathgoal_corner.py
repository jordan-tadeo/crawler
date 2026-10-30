"""The point the goal model is fed carries on round a corner made of
waypoints before the first of them is reached (autopilot/pathgoal.py's
plan through the patrol's next waypoints, patrol.Patrol.ahead and passed).

Planned only to the current waypoint, the point 1.5 m along the path sits
on the waypoint for the last 1.5 m and then swings to the next one when it
is passed: on a hard left the truck comes in set up straight and has to
turn on the spot, or back up, at the waypoint. Planned through the next
ones too, it bends towards the next leg on the way in, and never jumps.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler.autopilot import follow as fl  # noqa: E402
from crawler.autopilot import pathgoal as pgo  # noqa: E402
from crawler.autopilot import patrol as pt  # noqa: E402

from test_goal_reseed import crop_at  # noqa: E402
from test_pathgoal import Clock, MapSlam, planner, world  # noqa: E402

ROOM = ((-2.0, 7.0), (-3.0, 6.0))
# The owner's two: a hard left made of four waypoints, and one made of one,
# each with a last waypoint up the next leg that the patrol ("once") stops at.
CURVE = [(3.0, 0.0), (3.7, 0.3), (4.0, 1.0), (4.0, 2.8)]
HARD_LEFT = [(3.0, 0.0), (3.0, 2.8)]
# The most the bearing to the fed point may turn from one decision to the
# next on the way round (10 Hz): about 18 planned through the next ones.
# Fed only the current waypoint it swings 27 degrees at each of the curve's
# and 69 at the single hard left when one is passed.
MAX_JUMP_DEG = 22.0


def patrol(points, mode="once", clock=None):
    p = pt.Patrol(clock=clock or Clock())
    for x, y in points:
        p.append(x, y, 0)
    p.mode = mode
    p.command({"cmd": "start"})
    return p


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# --- what the planner is told ---------------------------------------------------------------

def test_ahead_is_the_waypoints_after_the_current_one_in_the_order_they_are_driven():
    pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    p = patrol(pts, mode="loop")
    p.poll(None, crop_at(-1.0, 0.0))
    assert p.goal[:2] == (0.0, 0.0)
    assert [a[:2] for a in p.ahead] == [(1.0, 0.0), (2.0, 0.0), (0.0, 0.0)], "a loop wraps"
    p.index = 2
    assert [a[:2] for a in p.ahead] == [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    p.mode, p.index, p.direction = "patrol", 1, 1
    assert [a[:2] for a in p.ahead] == [(2.0, 0.0), (1.0, 0.0), (0.0, 0.0)], "a patrol turns"
    p.mode, p.index = "once", 1
    assert [a[:2] for a in p.ahead] == [(2.0, 0.0)]
    p.index = 2
    assert p.ahead == [], "the last stop of a once has nothing after it"
    p.dwell_s = 1.0
    p.index = 0
    assert p.ahead == [], "with a dwell every waypoint is a stop"
    p.dwell_s = 0.0
    p.command({"cmd": "pause"})
    assert p.ahead == []
    p.command({"cmd": "goal"})
    assert p.ahead == [], "a single click has nothing after it"


def test_following_a_person_has_nothing_ahead_and_never_passes_the_patrol_on():
    p = patrol(CURVE)
    p.poll(None, crop_at(0.0, 0.0))
    person = SimpleNamespace(active=True, goal=(5.0, 5.0, 0), reached=lambda: False)
    sw = fl.FollowSwitch(p, person)
    assert sw.ahead == [] and not sw.passed(p.goal, crop_at(2.9, 0.0))
    assert p.index == 0
    person.active = False
    assert [a[:2] for a in sw.ahead] == CURVE[1:]
    assert sw.passed(p.goal, crop_at(2.9, 0.0)) and p.index == 1


def test_the_plan_runs_through_the_next_waypoints_and_says_how_many():
    pl, _ = planner(world(size=ROOM))
    p = patrol(CURVE)
    c = crop_at(0.0, 0.0)
    p.poll(None, c)
    pl.encode(c, p)
    pl.encode(c, p)
    plan = pl._plan
    assert plan.route[0][:2] == CURVE[0]
    # On through the next until 1.5 m past the first: (3.7, 0.3) is 0.76 on,
    # (4.0, 1.0) 1.5 on.
    assert [r[:2] for r in plan.route] == CURVE[:3]
    for r, k in zip(plan.route, plan.vias):
        assert tuple(plan.points[k]) == pytest.approx(r[:2])
    assert pl.status()["route_n"] == 3


# --- the owner's case: bent before the waypoint, never a jump ----------------------------

@pytest.mark.parametrize("points", [CURVE, HARD_LEFT], ids=["curve", "hard-left"])
def test_coming_in_to_a_hard_left_the_point_fed_bends_left_before_the_waypoint(points):
    pl, _ = planner(world(size=ROOM))
    p = patrol(points)
    left = []
    for x in (2.0, 2.2, 2.4):
        c = crop_at(x, 0.0)
        p.poll(None, c)
        pl.encode(c, p)
        pl.encode(c, p)
        assert p.index == 0 and math.hypot(points[0][0] - x, points[0][1]) > pt.PASS_M
        assert pl.fed == "subgoal"
        ax, ay = pl.aim
        # Fed the waypoint itself it would be (3, 0) dead ahead; round the
        # corner it is up the next leg, left of the line in.
        assert ay >= 0.15 and ax >= points[0][0] - 0.1, (
            f"at x={x} the point is {pl.aim}, not round the corner")
        left.append(math.atan2(ay, ax - x))
    assert all(b >= a - 1e-9 for a, b in zip(left, left[1:])), "nearer, further round"


def sim(points, seconds=60.0, room=ROOM):
    """The truck as a unicycle at 10 Hz under the real PolicyDriver and a
    stub model steering at its goal input, the planner on the true map,
    along `points` once."""
    from test_patrol import GoalSteerStub
    from test_policy_goal import PosedSlam
    from test_policy_map import T0, driver
    view = world(size=room)
    slam = PosedSlam()
    slam.map = lambda: view._replace(t=slam.c.t)
    d = driver(GoalSteerStub(), slam)
    clock = Clock(T0)
    p = patrol(points, clock=clock)
    d.goal = p
    d.path = pgo.PathPlanner(slam, background=False, clock=clock)
    x, y, yaw, t, dt = 0.0, 0.0, 0.0, T0, 0.1
    bearings, track, jumps = [], [], []
    while t < T0 + seconds and not p.finished:
        t = round(t + dt, 6)
        clock.t = t
        slam.c = slam.c._replace(x=x, y=y, yaw=yaw, t=t)
        p.poll(None, slam.crop())
        d.tick(t)
        aim = d.path.aim
        if aim is not None:
            b = math.atan2(aim[1] - y, aim[0] - x)
            if bearings:
                jumps.append(abs(math.degrees(wrap(b - bearings[-1]))))
            bearings.append(b)
        thr = d.controller.throttles[-1] if d.controller.throttles else 0.0
        front = d.controller.steerings[-1][0] if d.controller.steerings else 0.0
        v = 0.8 * thr
        yaw += -v * front * 1.6 * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        track.append((x, y))
    return SimpleNamespace(patrol=p, jumps=np.array(jumps), track=np.array(track),
                           end=(x, y), planner=d.path, seconds=t - T0)


@pytest.mark.parametrize("points", [CURVE, HARD_LEFT], ids=["curve", "hard-left"])
def test_round_a_hard_left_the_point_fed_never_jumps_and_the_last_waypoint_stops(points):
    r = sim(points)
    print(f"\nsim {len(points)} waypoints: done {r.patrol.finished} in {r.seconds:.1f} s, "
          f"biggest turn of the point fed in one decision {r.jumps.max():.1f} deg, "
          f"farthest past x=3 {r.track[:, 0].max() - 3.0:.2f} m, "
          f"passed on the path {r.patrol.passed_on_path}")
    assert r.patrol.finished and r.patrol.skipped == 0
    last = points[-1]
    assert math.hypot(r.end[0] - last[0], r.end[1] - last[1]) <= r.patrol.arrive_m + 0.05
    assert r.jumps.max() <= MAX_JUMP_DEG


def test_a_corner_cut_wide_of_the_pass_radius_is_passed_on_the_path():
    # The truck has come round inside the waypoint and is on the next leg,
    # 0.6 m from it: passed, not driven back to.
    pl, _ = planner(world(size=ROOM))
    p = patrol(HARD_LEFT)
    c = crop_at(2.0, 0.0)
    p.poll(None, c)
    pl.encode(c, p)
    c = crop_at(2.75, 0.55, math.pi / 2)
    p.poll(None, c)
    assert p.index == 0 and p.last[0] > pt.PASS_M
    pl.encode(c, p)
    assert p.index == 1 and p.passed_on_path == 1 and pl.status()["passes"] == 1
    assert pl.aim[1] > 0.55, "fed on up the next leg, not back to the corner"


def test_a_waypoint_the_path_comes_back_from_is_not_passed_on_the_path():
    # There and back along the same line: by the end, 0.6 m short, the path
    # out and the path back are the same distance off, so no pass, and the
    # point fed is not folded back onto the truck.
    pl, _ = planner(world(size=ROOM))
    p = patrol([(3.0, 0.0), (0.0, 0.0)], mode="patrol")
    c = crop_at(1.0, 0.0)
    p.poll(None, c)
    pl.encode(c, p)
    c = crop_at(2.4, 0.0)
    p.poll(None, c)
    pl.encode(c, p)
    pl.encode(c, p)
    assert p.index == 0 and p.passed_on_path == 0
    assert math.hypot(pl.aim[0] - 2.4, pl.aim[1]) >= 0.5


def test_the_plan_through_the_next_waypoint_is_used_from_its_next_leg_until_the_new_one():
    # Passed on PASS_M; the plan for the next waypoint is not made yet (the
    # period has not run out), so the old one is used on from the waypoint,
    # not the next waypoint itself fed straight.
    clock = Clock()
    pl = pgo.PathPlanner(MapSlam(world(size=ROOM)), background=False, clock=clock,
                         every_s=10.0, stage_deg=None, aim_rate_dps=None)
    p = patrol(CURVE, clock=clock)
    c = crop_at(2.0, 0.0)
    p.poll(None, c)
    pl.encode(c, p)
    before = pl._plan
    c = crop_at(2.7, 0.1)
    p.poll(None, c)
    assert p.index == 1
    pl._step = lambda: None          # no new plan this decision
    pl.encode(c, p)
    assert pl._plan is before and pl.fed == "subgoal" and pl.why == ""
    k = int(np.argmin(np.hypot(*(before.points - np.asarray(pl.aim)).T)))
    assert k > before.vias[1], "on along the old plan past the new goal"


def test_a_single_goal_and_the_last_waypoint_are_fed_the_goal_itself_in_sight():
    pl, _ = planner(world(size=ROOM))
    p = patrol(HARD_LEFT)
    p.index = 1
    c = crop_at(3.0, 1.6, math.pi / 2)
    p.poll(None, c)
    pl.encode(c, p)
    assert pl.fed == "goal" and pl.why == "in sight" and pl.aim == HARD_LEFT[1]
    p.command({"cmd": "goal"})
    p.single.set_goal(3.0, 2.8, 0)
    pl.encode(c, p)
    assert pl.fed == "goal" and pl.why == "in sight"


def test_a_two_point_loop_on_one_plan_goes_on_to_the_second_visit_not_back_to_the_first():
    # The plan runs A, B, A: with B passed too before a new plan is in, the
    # goal A is the plan's third, ahead of the truck, not its first, behind.
    clock = Clock()
    pl = pgo.PathPlanner(MapSlam(world(size=ROOM)), background=False, clock=clock,
                         every_s=10.0)
    p = patrol([(1.0, 0.0), (1.8, 0.0)], mode="loop", clock=clock)
    c = crop_at(0.0, 0.0)
    p.poll(None, c)
    pl.encode(c, p)
    plan = pl._plan
    assert len(plan.route) == 3
    pl._step = lambda: None
    for x in (1.0, 1.8):
        c = crop_at(x, 0.0)
        p.poll(None, c)
        pl.encode(c, p)
    assert p.index == 0 and pl._plan is plan
    assert pl._at == (plan, 2)
