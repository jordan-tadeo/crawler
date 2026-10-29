"""Point navigation: click a spot, the car drives there, or as close as it can
get, and holds until there is another.

Two halves. The simulator (tools/nav_sim.py, the truck and lidar of
tools/nav_world.py) drives the navigator through whole gotos on synthetic
rooms, for what only a whole drive shows: that it arrives and then does not
move, that a goal it cannot reach ends at the nearest point it can, that a
lost map is a stop. And PointNavDriver on the fake hardware, for who has the
wheels: the free driver until a goal, the navigator after, my hand over both.
"""

import math

from crawler.autopilot import driver as dv
from crawler.nav import click as ck
from crawler.nav import driver as nd
from crawler.nav import navigator as nv
from crawler.nav import planner as pl
from crawler.nav import pointnav as pn

import nav_sim as ns
import nav_world as nw

from conftest import FakeCamera, FakeController
from test_nav import T0, clear_depth, view_of, walled


# --- whole drives, in the simulator ------------------------------------------------

START = (1.0, 2.5, 0.0)


def episode(world, goal, **kw):
    # Plans land at once and have no wall-clock budget, so the drive is the
    # same however busy this machine is; the batch runs at the Pi's latency.
    kw.setdefault("hold_s", 3.0)
    return ns.run_episode(world, "goto", START, goal, level="measured", seed=1,
                          slowdown=0, **kw)


def test_a_reachable_goal_is_arrived_at_and_then_held_without_moving():
    r = episode(nw.room("box", (6.0, 5.0)), (5.0, 2.0))
    assert r["outcome"] == "arrived", r["why"]
    assert r["final_err_m"] <= nv.ARRIVE_M
    assert r["collisions"] == 0
    # Three seconds held: not one decision with a throttle, and nothing moved
    # once the brake has stopped it.
    assert r["hold_drive"] == 0
    assert r["hold_travel_m"] < 0.02


def test_a_goal_inside_a_wall_ends_at_the_closest_point_it_can_reach():
    world = nw.room("wall", (6.0, 5.0))        # a wall across x = 3 m
    goal = (3.0, 1.5)                          # in it
    expect = ns.closest_reachable(world, START, goal)
    r = episode(world, goal, expect=expect)
    assert r["outcome"] == "closest", r["why"]
    assert r["closest_err_m"] <= ns.CLOSEST_OK_M
    assert r["hold_drive"] == 0 and r["collisions"] == 0


def test_a_goal_in_a_room_with_no_way_in_ends_outside_it_nearest_the_goal():
    world = nw.room("box", (6.0, 5.0))
    s = world.solid.copy()
    w = int(round(0.05 / world.res))
    # A shut 1.4 x 1.4 m room in the far corner.
    x0, y0 = int(4.3 / world.res), int(3.3 / world.res)
    s[x0:x0 + w, y0:] = True
    s[x0:, y0:y0 + w] = True
    world = nw.make_world(s, world.res, "shut")
    goal = (5.2, 4.2)
    expect = ns.closest_reachable(world, START, goal)
    assert math.hypot(expect[0] - goal[0], expect[1] - goal[1]) > 0.8
    r = episode(world, goal, expect=expect)
    assert r["outcome"] == "closest", r["why"]
    assert r["closest_err_m"] <= ns.CLOSEST_OK_M
    assert r["hold_drive"] == 0


def test_a_new_goal_while_holding_is_driven_to():
    r = episode(nw.room("box", (6.0, 5.0)), (5.0, 2.0),
                extra_goals=[(1.5, 4.0)])
    assert [h[0] for h in r["holds"]] == [nv.ST_ARRIVED, nv.ST_ARRIVED]
    assert r["outcome"] == "arrived"
    assert r["final_err_m"] <= nv.ARRIVE_M      # off the second goal
    assert r["hold_drive"] == 0


def test_a_slam_lost_mid_route_is_a_stop_until_it_is_back_then_it_arrives():
    r = episode(nw.room("box", (6.0, 5.0)), (5.0, 2.0), lost=(2.0, 5.0))
    assert r["lost_ticks"] >= 25
    assert r["lost_drive"] == 0
    assert r["outcome"] == "arrived", r["why"]


def test_the_closest_point_stays_put_across_plans_not_flipping_sides_of_the_wall():
    # A wall across x = 3 m with a gap at the top; the goal is in the wall,
    # and the nearest floor is as near on the far side as on this one.
    occ = walled(120, 100)
    occ[59:61, :82] = True
    nav = nv.Navigator(nv.Params(planner=pl.Params(max_expansions=4000)))
    nav.set_goal(3.0, 1.5)
    for i in range(10):
        nav.tick(T0 + i, view_of(occ, (1.0, 1.5, 0.0), version=1))
        if nav.path is not None:
            break
    first = nav.aim
    assert nav.closest and first[0] < 2.95
    # I drive it round to the gap, where the far side is the shorter drive,
    # and let go: the plan from there is still to the point it had.
    nav.hand(T0 + 20)
    for i in range(10):
        nav.tick(T0 + 21 + i, view_of(occ, (3.3, 4.6, -1.57), version=1))
        if nav.path is not None:
            break
    assert nav.path is not None
    assert nav.aim == first


def test_a_lost_map_drops_the_path_and_the_truck_is_never_driven_on_it_again():
    occ = walled(160, 80)
    nav = nv.Navigator()
    nav.set_goal(7.0, 2.0)
    out = nav.tick(T0, view_of(occ, (1.0, 2.0, 0.0), version=1))
    assert out.direction == 1 and nav.path is not None
    for i in range(3):
        out = nav.tick(T0 + 0.1 * (i + 1), None)
        assert out.direction == 0
        assert nav.path is None and nav.status == nv.ST_NO_MAP
    # Back, with the truck somewhere the old path never went: it moves only
    # on a plan from there.
    for i in range(20):
        out = nav.tick(T0 + 1.0 + 0.1 * i, view_of(occ, (2.5, 1.2, 0.6), version=1))
        if out.direction:
            break
    assert out.direction == 1
    assert math.hypot(*(nav.path.poses[0, :2] - (2.5, 1.2))) < 0.1


# --- who has the wheels -------------------------------------------------------------

class Goals:
    """click.ClickGoals' take(), off a list."""

    def __init__(self):
        self.queue = []

    def click(self, x, y):
        self.queue.append(ck.ClickGoal(x, y, None, "map", "/move_base_simple/goal", 0.0))

    def take(self):
        return self.queue.pop(0) if self.queue else None


class Map:
    def __init__(self, occ, pose):
        self.occ = occ
        self.pose = pose
        self.version = 1

    def view(self):
        return view_of(self.occ, self.pose, version=self.version)


def point_nav(pose=(1.0, 2.0, 0.0)):
    occ = walled(160, 80)                    # 8 x 4 m
    m = Map(occ, pose)
    camera = FakeCamera(depth=clear_depth())
    controller = FakeController()
    free = dv.Driver(controller, camera)
    nav = nd.NavDriver(controller, camera, m, nv.Navigator())
    goals = Goals()
    return pn.PointNavDriver(free, nav, goals), goals, m, controller


def test_before_any_goal_the_free_driver_drives_exactly_as_without_nav():
    d, goals, _, controller = point_nav()
    alone_controller = FakeController()
    alone = dv.Driver(alone_controller, FakeCamera(depth=clear_depth()))
    for i in range(30):
        d.tick(T0 + 0.1 * i)
        alone.tick(T0 + 0.1 * i)
    assert d.active is d.free and d.driving == "rules"
    assert d.nav.nav.mode == nv.IDLE
    assert controller.throttles == alone_controller.throttles
    assert controller.steerings == alone_controller.steerings
    assert max(controller.throttles) > 0.0
    assert d.policy_fields is None             # as the rules driver's own


def test_a_click_before_auto_started_is_dropped_and_the_next_one_goes():
    d, goals, _, controller = point_nav()
    goals.click(6.0, 2.0)
    d.tick(T0)
    assert d.active is d.free and d.goals_dropped == 1
    goals.click(6.0, 2.0)
    d.tick(T0 + 0.1)
    assert d.active is d.nav and d.goals_received == 1
    assert d.nav.nav.goal[:2] == (6.0, 2.0)
    assert d.nav.last_request[0] > 0.0
    assert d.policy_fields()["autopilot"] == "nav"


def test_my_hand_mid_route_drops_the_path_and_letting_go_plans_from_there():
    d, goals, m, controller = point_nav()
    d.tick(T0)
    goals.click(6.0, 2.0)
    d.tick(T0 + 0.1)
    n = d.nav.nav
    assert n.path is not None and d.nav.last_request[0] > 0.0
    # My hand on the steering: the navigator follows nothing and asks for
    # nothing, whatever my thumb does with the throttle.
    d.human_steering = True
    for i in range(5):
        d.tick(T0 + 0.2 + 0.1 * i)
        assert n.path is None and n.status == nv.ST_HAND
        assert d.nav.last_request == (0.0, 0.0, 0.0)
    assert d.nav_status()["state"] == pn.PAUSED
    # I put the truck somewhere else and let go: a fresh plan from there.
    m.pose = (3.0, 1.0, 0.5)
    d.human_steering = False
    d.tick(T0 + 1.0)
    assert n.path is not None
    assert math.hypot(*(n.path.poses[0, :2] - (3.0, 1.0))) < 0.1
    assert d.nav.last_request[0] > 0.0
    assert d.active is d.nav


def test_a_held_goal_stays_held_through_my_hand_and_the_same_click_is_ignored():
    d, goals, m, controller = point_nav(pose=(6.0, 2.0, 0.0))
    d.tick(T0)
    goals.click(6.1, 2.0)
    for i in range(5):
        d.tick(T0 + 0.1 * (i + 1))
    n = d.nav.nav
    assert n.status == nv.ST_ARRIVED and n.holding
    d.human_throttle = True
    d.tick(T0 + 1.0)
    d.human_throttle = False
    for i in range(10):
        d.tick(T0 + 1.1 + 0.1 * i)
        assert n.status == nv.ST_ARRIVED
        assert d.nav.last_request == (0.0, 0.0, 0.0)
    # The goal it is at, clicked again: nothing.
    goals.click(6.0, 2.05)
    d.tick(T0 + 3.0)
    assert d.goals_received == 1 and n.status == nv.ST_ARRIVED
    # Somewhere else: it goes.
    goals.click(2.0, 2.0)
    d.tick(T0 + 3.1)
    assert d.goals_received == 2
    assert n.goal[:2] == (2.0, 2.0) and not n.holding


def test_leaving_auto_forgets_the_goal_and_start_again_is_the_free_driver():
    d, goals, _, controller = point_nav()
    d.tick(T0)
    goals.click(6.0, 2.0)
    d.tick(T0 + 0.1)
    assert d.active is d.nav
    d.abort()
    assert d.active is d.free and d.nav.nav.mode == nv.IDLE
    d.tick(T0 + 1.0)
    assert d.active is d.free and d.driving == "rules"


def test_nav_status_and_path_are_what_foxglove_is_sent():
    d, goals, _, _ = point_nav()
    d.tick(T0)
    s = d.nav_status()
    assert s["state"] == pn.IDLE and s["driving"] == "rules"
    assert set(s) <= set(__import__("crawler.telemetry.telemetry", fromlist=["x"])
                         .JSON_SCHEMAS["nav_status"]["schema"]["properties"])
    assert d.nav_path() == {"path": [], "goal": None, "aim": None}
    goals.click(6.0, 2.0)
    d.tick(T0 + 0.1)
    s = d.nav_status()
    assert s["state"] == pn.DRIVING and s["has_goal"] and s["goal_x"] == 6.0
    p = d.nav_path()
    assert p["goal"] == (6.0, 2.0) and p["aim"] is None
    assert len(p["path"]) >= 2
    assert math.hypot(p["path"][-1][0] - 6.0, p["path"][-1][1] - 2.0) < 0.2


def test_the_path_goes_to_foxglove_as_a_scene_and_no_goal_clears_it():
    from crawler.telemetry import nav_view
    from crawler.telemetry import telemetry as tm
    from test_lidar_view import decode
    t = tm.Telemetry()
    on = decode(t.lidar_view_message("nav_path", {
        "path": [(1.0, 2.0), (2.0, 2.0), (3.0, 2.5)], "goal": (4.0, 2.5),
        "aim": (3.0, 2.5)}, 1_000_000_000))
    assert sorted(e.id for e in on.entities) == sorted(
        [nav_view.PATH_ID, nav_view.GOAL_ID, nav_view.AIM_ID])
    assert all(e.frame_id == "map" for e in on.entities)
    off = decode(t.lidar_view_message(
        "nav_path", {"path": [], "goal": None, "aim": None}, 2_000_000_000))
    assert not off.entities
    assert sorted(d.id for d in off.deletions) == sorted(
        [nav_view.PATH_ID, nav_view.GOAL_ID, nav_view.AIM_ID])
