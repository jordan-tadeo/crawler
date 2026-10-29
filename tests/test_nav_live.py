"""--slam --nav end to end: the live SLAM as the navigator's map, a Foxglove
click as its goal, the low-obstacle layer, and what main builds.

Fake SLAM and voxel runners on synthetic rooms, a real ClickGoals fed the
way the foxglove server feeds it, and the real AutoPilot, NavDriver and
planner thread. Every loop is bounded by a step count.
"""

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

import crawler.__main__ as mn
from crawler.hardware import lidar as ld
from crawler.mapping import grid as gr
from crawler.mapping import runner as rn
from crawler.mapping import voxel as vx
from crawler.nav import click as ck
from crawler.nav import driver as nd
from crawler.nav import live
from crawler.nav import navigator as nv
from crawler.nav import pointnav as pn

from conftest import FakeController
from test_nav import Log, ReadyHub, clear_depth, pose_in_frame, walled

RES = 0.05


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def result(t, x=1.0, y=2.0, yaw=0.0, state="tracking", matched=True,
           inliers=0.9, constraint=0.4):
    return SimpleNamespace(t=t, x=x, y=y, yaw=yaw, state=state,
                           matched=matched, inliers=inliers,
                           constraint=constraint)


def codes_of(occ):
    c = np.full(occ.shape, gr.FREE, np.uint8)
    c[occ] = gr.OCCUPIED
    return c


class FakeSlam:
    """SlamRunner's reads: stats(), latest(), nav_map(), mount."""

    def __init__(self, clock, occ, x0=0.0, y0=0.0):
        self.clock = clock
        self.mount = ld.MOUNT
        self.alive = True
        self.res = result(clock())
        self.nav = rn.MapView(t=clock(), version=1, x0=x0, y0=y0, res=RES,
                              codes=codes_of(occ))

    def stats(self):
        return {"alive": self.alive}

    def latest(self):
        return self.res

    def nav_map(self):
        return self.nav

    def fresh(self, **kw):
        """A new Result now, and the nav map redrawn now."""
        r = self.res
        base = dict(x=r.x, y=r.y, yaw=r.yaw, state="tracking", matched=True,
                    inliers=0.9, constraint=0.4)
        base.update(kw)
        self.res = result(self.clock(), **base)
        self.nav = self.nav._replace(t=self.clock())


class FakeVoxels:
    def __init__(self, low):
        self._low = low

    def low(self):
        return self._low


def low_view(codes, t, x0=0.0, y0=0.0):
    return vx.LowView(t, x0, y0, RES, np.asarray(codes, np.uint8))


# --- the constants agree with the mapper's ---------------------------------------

def test_the_codes_are_the_mappers_own():
    assert (live.UNKNOWN, live.FREE, live.OCCUPIED) == (gr.UNKNOWN, gr.FREE, gr.OCCUPIED)
    assert live.LOW_OBSTACLE == vx.LOW_OBSTACLE
    assert live.TRUST_MIN < nv.CONFIDENCE_MIN, "a band for the local map"


# --- SlamMap: a trusted pose or nothing --------------------------------------------

def test_a_tracking_pose_on_a_fresh_map_is_a_view_on_the_grids_own_cells():
    clk = Clock()
    occ = walled(160, 80)
    slam = FakeSlam(clk, occ, x0=-2.0, y0=-1.0)
    src = live.SlamMap(slam, clock=clk)
    v = src.view()
    assert v is not None and src.why == ""
    assert v.res == RES and v.origin == (-2.0, -1.0)
    assert (v.occupied == occ).all() and v.known.all()
    assert v.pose == (1.0, 2.0, 0.0)
    assert v.confidence == pytest.approx(rn.confidence(slam.res))
    assert v.layers == ()
    assert src.local() is src.local_map


@pytest.mark.parametrize("change, why", [
    (dict(state="lost"), "not tracking"),
    (dict(state="tilted"), "not tracking"),
    (dict(matched=False), "not tracking"),
    (dict(inliers=0.1), "confidence"),
    (dict(constraint=0.0), "confidence"),
])
def test_a_pose_it_would_not_trust_is_no_view(change, why):
    clk = Clock()
    slam = FakeSlam(clk, walled(80, 80))
    slam.fresh(**change)
    src = live.SlamMap(slam, clock=clk)
    assert src.view() is None
    assert src.why.startswith(why)


def test_a_stale_pose_a_stale_map_no_map_or_a_dead_mapper_is_no_view():
    clk = Clock()
    slam = FakeSlam(clk, walled(80, 80))
    src = live.SlamMap(slam, clock=clk)
    assert src.view() is not None
    clk.t += live.POSE_STALE_S + 0.05
    assert src.view() is None and src.why == "pose stale"
    slam.fresh()
    slam.nav = slam.nav._replace(t=clk.t - live.MAP_STALE_S - 0.1)
    assert src.view() is None and src.why == "map stale"
    slam.nav = None
    assert src.view() is None and src.why == "no map"
    slam.nav = rn.MapView(clk.t, 1, 0.0, 0.0, RES, codes_of(walled(80, 80)))
    slam.fresh()
    assert src.view() is not None
    slam.alive = False
    assert src.view() is None and src.why == "mapper not running"
    slam.latest = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    slam.alive = True
    assert src.view() is None and src.why.startswith("error")


# --- the low-obstacle layer: advisory, off by default ------------------------------

def test_the_low_layer_only_ever_adds_solid_cells_and_only_when_asked():
    clk = Clock()
    occ = walled(80, 80)
    occ[40, 40] = True                       # a lidar wall cell
    slam = FakeSlam(clk, occ)
    codes = np.full((40, 40), vx.LOW_UNSEEN, np.uint8)
    codes[:, :] = vx.LOW_FLOOR               # it says floor everywhere ...
    codes[5, 7] = vx.LOW_OBSTACLE            # ... but one shoe
    # The LowView starts 1 m, 0.5 m into the nav map: an integer offset.
    vox = FakeVoxels(low_view(codes, clk.t, x0=1.0, y0=0.5))

    off = live.SlamMap(slam, voxels=vox, clock=clk).view()
    assert off.layers == ()
    assert not off.solid()[20 + 5, 10 + 7]

    on = live.SlamMap(slam, voxels=vox, low_obstacles=True, clock=clk).view()
    solid = on.solid()
    assert solid[20 + 5, 10 + 7], "the shoe, on the same lattice"
    assert solid[40, 40], "its floor never clears a lidar cell"
    assert (solid == (occ | on.layers[0])).all()
    assert on.layers[0].sum() == 1

    vox._low = low_view(codes, clk.t - live.LOW_STALE_S - 0.1, x0=1.0, y0=0.5)
    assert live.SlamMap(slam, voxels=vox, low_obstacles=True,
                        clock=clk).view().layers == ()


def test_a_low_view_off_the_nav_map_or_at_another_resolution_is_no_layer():
    assert not live.low_layer(low_view(np.full((4, 4), 2), 0.0, x0=50.0), (10, 10),
                              0.0, 0.0, RES).any()
    lv = vx.LowView(0.0, 0.0, 0.0, 0.1, np.full((4, 4), 2, np.uint8))
    assert not live.low_layer(lv, (10, 10), 0.0, 0.0, RES).any()
    # Hanging off the low corner is clipped, not wrapped.
    out = live.low_layer(low_view(np.full((4, 4), 2), 0.0, x0=-0.1, y0=-0.1),
                         (10, 10), 0.0, 0.0, RES)
    assert out[:2, :2].all() and out.sum() == 4


# --- end to end: a click, a plan, a forward; a lost map, neutral --------------------

def advertise(goals):
    client = SimpleNamespace(id=3)
    goals.on_client_advertise(client, SimpleNamespace(
        id=1, topic="/move_base_simple/goal", schema_name="foxglove.PoseInFrame",
        encoding="json"))
    return client


def test_a_click_on_the_live_map_is_a_plan_and_a_forward_and_a_lost_map_is_neutral():
    clk = Clock()
    slam = FakeSlam(clk, walled(160, 80))          # 8 x 4 m, truck at (1, 2)
    goals = ck.ClickGoals()
    src = live.SlamMap(slam, clock=clk)
    pilot = mn.AutoPilot(FakeController(), Log(), hub=ReadyHub(), nav=True,
                         map_source=src, goals=goals)
    assert pilot.resume() is True
    d = pilot.driver
    assert isinstance(d, pn.PointNavDriver)
    assert isinstance(d.nav, nd.NavDriver)
    n = d.nav.nav
    d.camera.depth = clear_depth()
    try:
        # A click made before auto started is no goal.
        goals.on_message_data(advertise(goals), 1, pose_in_frame(3.0, 2.0))
        # No goal: the auto driver it has without --nav drives.
        d.tick(clk.t)
        assert d.active is d.free and d.driving == "rules"
        assert d.goals_dropped == 1 and d.goals_received == 0
        assert n.mode == nv.IDLE
        assert d.nav_status()["state"] == pn.IDLE

        goals.on_message_data(advertise(goals), 1, pose_in_frame(6.5, 2.0))
        moved = None
        for i in range(150):                          # at most about 3 s
            clk.t += 0.1
            slam.fresh()
            d.tick(clk.t)
            if d.last_request[0] > 0.0:
                moved = i
                break
            time.sleep(0.02)
        assert d.active is d.nav and d.goals_received == 1
        assert n.goal[:2] == (6.5, 2.0)
        assert moved is not None, n.status
        assert n.path is not None
        assert d.last_output.direction == 1
        assert pilot.command.throttle > 0.0
        status = d.nav_status()
        assert status["state"] == pn.DRIVING and status["driving"] == "nav"
        assert d.nav_path()["path"] and d.nav_path()["goal"] == (6.5, 2.0)

        # The mapper loses the room: the next decision is neutral, and it
        # stays there for as long as it is lost.
        for _ in range(5):
            clk.t += 0.1
            slam.fresh(state="lost")
            d.tick(clk.t)
            assert d.last_request[0] == 0.0
            assert pilot.command.throttle == 0.0
            assert n.status == nv.ST_NO_MAP and n.path is None
            assert d.nav_status()["state"] == pn.PAUSED
        # And a tracking pose that just stops coming is the same.
        slam.fresh()
        clk.t += 1.0
        d.tick(clk.t)
        assert pilot.command.throttle == 0.0 and src.why == "pose stale"
    finally:
        d.close()


def test_a_click_in_base_link_with_no_pose_to_place_it_by_is_dropped():
    clk = Clock()
    slam = FakeSlam(clk, walled(80, 80))
    slam.fresh(state="lost")
    goals = ck.ClickGoals()
    goals.on_message_data(advertise(goals), 1,
                          pose_in_frame(1.0, 0.0, frame="base_link"))
    d = nd.NavDriver(FakeController(), SimpleNamespace(
        get_depth_frame=clear_depth, get_imu_sample=lambda: None),
        live.SlamMap(slam, clock=clk), nv.Navigator(), goals=goals)
    d.tick(clk.t)
    assert d.goals_taken == 0 and d.nav.mode == nv.IDLE
    assert d.controller.throttles[-1] == 0.0


# --- what main builds -----------------------------------------------------------------

def test_main_builds_nothing_for_nav_without_the_flag():
    args = mn.parse_args([])
    assert args.nav is False and args.nav_low_obstacles is False
    assert mn.nav_goals(args, Log()) == (None, None)
    assert mn.nav_map_source(args, object(), None, None, Log()) is None
    assert rn.SlamRunner().nav_half_m is None
    assert rn.SlamRunner().nav_map() is None


def test_main_with_nav_lets_foxglove_publish_goals_and_maps_off_the_slam():
    from foxglove.websocket import Capability
    args = mn.parse_args(["--nav"])
    goals, opts = mn.nav_goals(args, Log())
    assert isinstance(goals, ck.ClickGoals)
    assert opts["server_listener"] is goals
    assert Capability.ClientPublish in opts["capabilities"]

    log = Log()
    assert mn.nav_map_source(args, None, None, None, log) is None
    assert any("--slam" in m for _, _, m in log.lines)

    slam = FakeSlam(Clock(), walled(40, 40))
    src = mn.nav_map_source(args, slam, None, None, Log())
    assert isinstance(src, live.SlamMap) and src.low_obstacles is False

    low = mn.parse_args(["--nav", "--nav-low-obstacles"])
    log = Log()
    assert mn.nav_map_source(low, slam, None, None, log).low_obstacles is False
    assert any("--voxel-map" in m for _, _, m in log.lines)
    assert mn.nav_map_source(low, slam, None, FakeVoxels(None), Log()).low_obstacles is True
