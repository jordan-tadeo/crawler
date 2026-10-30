"""A goal clicked on the SLAM map, as a policy input, and the arrival hold.

policy_goal.py is the one encoder the trainer and the car share: a point on
the map, put in the truck's frame off the pose the crop going into the same
model run was drawn at. These hold the encoder (bearing sign, mirror, other
map frames), the trainer's hindsight goals, egomap's map frame, the
ClickGoal the car keeps (arrival, the hold, a new goal), PolicyDriver
feeding it (only on the map model, only with the map), the sidecar
contract, and --goal in crawler/__main__.py.
"""

import json
import math
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler.autopilot import goal as gl  # noqa: E402
from crawler.autopilot import policy as pl  # noqa: E402
from crawler.autopilot import policy_goal as pg  # noqa: E402
from crawler.mapping import egomap as em  # noqa: E402
from crawler.mapping import grid as gr  # noqa: E402
from crawler.nav import click as ck  # noqa: E402
from crawler.telemetry import telemetry as tm  # noqa: E402
import map_crops as mc  # noqa: E402
import policy_train as pt  # noqa: E402

from test_policy_map import (FORWARD, T0, FakeSlam, MapStub,  # noqa: E402
                             _drive, _scans, fallback_driver, driver)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
GOAL_MODEL = FIXTURES / "policy-goal.onnx"   # untrained, profile + mode + lidar + map + goal


def crop_at(x=0.0, y=0.0, yaw=0.0, frame=0, present=True, t=T0):
    return em.Crop(t, present, np.full((em.SIZE, em.SIZE), gr.FREE, np.uint8),
                   np.zeros((em.SIZE, em.SIZE), np.uint8), x, y, yaw, frame)


# --- the encoder ---------------------------------------------------------------------

def test_the_goal_is_put_in_the_truck_frame_left_positive():
    """A truck at (1, 1) facing +y: (1, 3) is 2 m dead ahead, (0, 1) is 1 m
    on its left, (2, 1) on its right, (1, 0) behind."""
    c = crop_at(1.0, 1.0, math.pi / 2)
    ahead = pg.encode(c, (1.0, 3.0, 0))[0]
    np.testing.assert_allclose(ahead, [1.0, 2.0 / pg.FAR_M, 0.0, 1.0], atol=1e-6)
    left = pg.encode(c, (0.0, 1.0, 0))[0]
    np.testing.assert_allclose(left, [1.0, 1.0 / pg.FAR_M, 1.0, 0.0], atol=1e-6)
    right = pg.encode(c, (2.0, 1.0, 0))[0]
    assert right[pg.SIN] == pytest.approx(-1.0)
    behind = pg.encode(c, (1.0, 0.0, 0))[0]
    assert behind[pg.COS] == pytest.approx(-1.0)
    assert pg.encode(c, (1.0, 50.0, 0))[0, pg.DIST] == 1.0, "far is clipped to 1"
    d, b = pg.distance_bearing(c, (0.0, 1.0, 0))
    assert d == pytest.approx(1.0) and b == pytest.approx(math.pi / 2)


def test_no_goal_no_crop_a_lost_crop_or_another_map_is_no_goal():
    c = crop_at()
    assert pg.encode(c, (1.0, 0.0, 0))[0, pg.PRESENT] == 1.0
    for crop, goal in ((c, None), (None, (1.0, 0.0, 0)),
                       (c._replace(present=False), (1.0, 0.0, 0)),
                       (c, (1.0, 0.0, 1))):
        g = pg.encode(crop, goal)
        assert g.shape == (1, pg.SIZE) and not g.any()
    assert pg.distance_bearing(c, (1.0, 0.0, 1)) is None


def test_the_mirror_swaps_left_and_right():
    c = crop_at()
    left, right = pg.encode(c, (1.0, 1.0, 0)), pg.encode(c, (1.0, -1.0, 0))
    np.testing.assert_allclose(pg.mirror(left), right, atol=1e-6)
    assert not pg.mirror(pg.missing()).any()


# --- hindsight goals --------------------------------------------------------------------

def _poses(n=40, dt=0.5):
    """Two sessions of a truck driving +x at 1 m/s, facing +x; the second
    reseeds to frame 1 half way."""
    t = np.concatenate([np.arange(n) * dt, 50.0 + np.arange(n) * dt])
    sess = np.repeat([0, 1], n)
    x = np.concatenate([np.arange(n) * dt, np.arange(n) * dt])
    frame = np.concatenate([np.zeros(n, np.int64), np.repeat([0, 1], n // 2)])
    g = {"log_t": t, "x": x, "y": np.zeros(2 * n), "yaw": np.zeros(2 * n), "frame": frame,
         "present": np.ones(2 * n, bool), "sess": sess}
    g["key"] = pg.sort_key(sess, t)
    return g


def test_a_hindsight_goal_is_the_pose_k_seconds_on_in_the_ticks_frame():
    g = _poses()
    ahead, left, ok = pg.hindsight_goals([0, 4, 10], 5.0, g)
    assert ok.all()
    np.testing.assert_allclose(ahead, 5.0)
    np.testing.assert_allclose(left, 0.0)


def test_no_hindsight_goal_across_a_session_a_reseed_an_untrusted_pose_or_the_end():
    g = _poses()
    n = 40
    # Session 0's last rows: k on runs into session 1, never a goal there.
    _, _, ok = pg.hindsight_goals([n - 4], 5.0, g)
    assert not ok.any()
    # Session 1 reseeds at row n + 20: a goal on the other side is not one.
    _, _, ok = pg.hindsight_goals([n + 15, n + 5], 5.0, g)
    assert ok.tolist() == [False, True]
    # The end of the file.
    _, _, ok = pg.hindsight_goals([2 * n - 2], 5.0, g)
    assert not ok.any()
    # An untrusted pose at either end, and a tick with no crop.
    g["present"][10 + 10] = False
    _, _, ok = pg.hindsight_goals([10, 20, -1], 5.0, g)
    assert ok.tolist() == [False, False, False]


def test_the_trainer_feeds_no_goal_without_the_map_and_on_the_no_goal_share():
    g = _poses()
    data = {"goal_poses": g, "map_row": np.arange(len(g["log_t"]))}
    rng = np.random.default_rng(0)
    rows = np.arange(20)
    x = pt.goal_inputs(data, rows, np.ones(20, bool), rng, no_goal_share=0.0, k_s=5.0)
    assert x[:, pg.PRESENT].all()
    np.testing.assert_allclose(x[:, pg.DIST], 5.0 / pg.FAR_M, atol=1e-6)
    shown = np.ones(20, bool)
    shown[::2] = False
    x = pt.goal_inputs(data, rows, shown, rng, no_goal_share=0.0, k_s=5.0)
    assert x[:, pg.PRESENT].tolist() == shown.astype(float).tolist()
    big = np.arange(30)
    x = pt.goal_inputs(data, np.tile(big, 100) % 30, np.ones(3000, bool), rng,
                       no_goal_share=0.3, k_s=2.0)
    assert 0.25 < 1.0 - x[:, pg.PRESENT].mean() < 0.35
    # k drawn from the window when not given.
    x = pt.goal_inputs(data, np.full(500, 0), np.ones(500, bool), rng, no_goal_share=0.0)
    d = x[x[:, pg.PRESENT] > 0, pg.DIST] * pg.FAR_M
    assert len(d) and d.min() >= pg.NEAR_S - 0.51 and d.max() <= pg.FAR_S + 0.01
    assert not pt.goal_inputs({}, rows, np.ones(20, bool), rng).any()


def test_the_crops_carry_the_pose_and_frame_the_trainer_reads(tmp_path):
    n = 12
    d = _scans(n)
    out = mc.crops_of(d)
    for k in ("x", "y", "yaw", "frame"):
        assert len(out[k]) == n
    np.savez_compressed(tmp_path / "s1.npz", **out)
    data = {"meta": {"sessions": ["s0", "s1"]},
            "session": np.array([0, 1, 1]),
            "t_ns": (np.array([0.5, d["t"][2] + 0.01, d["t"][8] + 0.01]) * 1e9).astype(np.int64)}
    pt.load_maps(data, str(tmp_path))
    g = data["goal_poses"]
    np.testing.assert_array_equal(g["x"], out["x"])
    np.testing.assert_array_equal(g["frame"], out["frame"])
    assert g["present"].tolist() == out["present"].tolist()
    # A crops file from before the poses gives no goals.
    old = {k: v for k, v in out.items() if k not in ("x", "y", "yaw", "frame")}
    np.savez_compressed(tmp_path / "s1.npz", **old)
    pt.load_maps(data, str(tmp_path))
    assert not data["goal_poses"]["present"].any()


# --- egomap's map frame ---------------------------------------------------------------

def _res(**kw):
    from crawler.mapping.slam import Result
    base = dict(t=0.0, x=0.0, y=0.0, yaw=0.0, state="tracking", matched=True,
                written=True, why="", inliers=1.0, rms_m=0.0, constraint=0.4,
                returns=700, points=600, ground_dropped=0, tilt_deg=0.0, iters=3, ms=1.0)
    base.update(kw)
    return Result(**base)


def test_the_crop_says_its_pose_and_a_reseed_is_a_new_map_frame_a_merge_the_old_one():
    ego, g = em.EgoMap(), gr.Grid(size=256)
    c = ego.after_step(g, _res(t=0.0, x=1.5, y=-0.5, yaw=0.3))
    assert (c.x, c.y, c.yaw, c.frame) == (1.5, -0.5, 0.3, 0)
    assert ego.after_step(g, _res(t=0.1, reseed="lost")).frame == 1
    assert ego.after_step(g, _res(t=0.2)).frame == 1
    assert ego.after_step(g, _res(t=0.3, reseed="young")).frame == 2
    assert ego.after_step(g, _res(t=0.4, reseed="merged")).frame == 0, "the map shelved by lost"
    assert ego.after_step(g, _res(t=0.5, reseed="merged")).frame == 0, "nothing shelved: stays"
    assert ego.after_step(g, _res(t=0.6, reseed="lost")).frame == 3, "never a number reused"


# --- ClickGoal: the goal the car keeps, and the arrival hold ----------------------------

def click(x, y, frame_id="map"):
    return ck.ClickGoal(x, y, None, frame_id, "/move_base_simple/goal", 0.0)


class Clicks:
    def __init__(self, *goals):
        self.goals = list(goals)

    def take(self):
        return self.goals.pop(0) if self.goals else None


def test_a_click_is_placed_on_the_map_and_kept_until_replaced():
    said = []
    g = gl.ClickGoal(say=lambda *a: said.append(a))
    c = crop_at(1.0, 0.0, math.pi / 2)
    g.poll(Clicks(click(1.0, 2.0)), c)
    assert g.goal == (1.0, 2.0, 0) and not g.reached()
    st = g.status()
    assert st["goal_set"] and st["distance_m"] == pytest.approx(2.0) and st["on_map"]
    assert st["bearing_deg"] == pytest.approx(0.0, abs=1e-6)
    # A click in the truck's frame is put on the map off the crop's pose.
    g.poll(Clicks(click(1.0, 0.0, "base_link")), c)
    assert g.goal[0] == pytest.approx(1.0) and g.goal[1] == pytest.approx(1.0)
    # Nothing new: kept.
    for _ in range(5):
        g.poll(Clicks(), c)
    assert g.goal[1] == pytest.approx(1.0) and g.clicks == 2
    assert any("Goal set" in a[2] for a in said)


def test_a_click_with_no_map_or_in_a_frame_it_cannot_place_is_refused():
    g = gl.ClickGoal()
    g.poll(Clicks(click(1.0, 0.0)), None)
    g.poll(Clicks(click(1.0, 0.0, "odom")), crop_at())
    g.poll(Clicks(click(float("nan"), 0.0)), crop_at())
    assert g.goal is None and g.refused == 3 and not g.status()["goal_set"]
    assert not g.encode(crop_at()).any()


def test_arriving_holds_until_a_new_different_goal():
    said = []
    g = gl.ClickGoal(say=lambda *a: said.append(a))
    g.poll(Clicks(click(2.0, 0.0)), crop_at())
    g.observe(crop_at(2.0 - pg.ARRIVE_M - 0.02))
    assert not g.reached()
    g.observe(crop_at(2.0 - pg.ARRIVE_M + 0.02))
    assert g.reached() and g.status()["arrived"]
    assert any("Goal reached" in a[2] for a in said)
    # Rolling off it again does not let go.
    g.observe(crop_at(0.0))
    assert g.reached()
    # The same goal clicked again keeps the hold.
    g.poll(Clicks(click(2.1, 0.05)), crop_at(0.0))
    assert g.reached()
    # A different one lets go.
    g.poll(Clicks(click(4.0, 0.0)), crop_at(0.0))
    assert not g.reached() and g.goal == (4.0, 0.0, 0)
    assert any("arrival hold is off" in a[2] for a in said)


def test_a_goal_on_another_map_frame_is_kept_never_fed_and_never_arrives():
    g = gl.ClickGoal()
    g.poll(Clicks(click(1.0, 0.0)), crop_at(frame=0))
    on_other = crop_at(1.0, 0.0, frame=1)
    g.observe(on_other)
    assert not g.reached() and not g.status()["on_map"]
    assert not g.encode(on_other).any()
    assert g.status()["frame_now"] == 1 and g.status()["frame"] == 0
    # The map comes back (a merge): it is the goal again.
    g.observe(crop_at(0.0, 0.0, frame=0))
    assert g.encode(crop_at(frame=0))[0, pg.PRESENT] == 1.0


def test_the_status_fits_its_schema():
    props = tm.JSON_SCHEMAS["goal_status"]["schema"]["properties"]
    for g in (gl.ClickGoal(), gl.ClickGoal()):
        assert set(g.status()) == set(props)
    g.poll(Clicks(click(1.0, 0.0)), crop_at())
    assert set(g.status()) == set(props)
    assert tm.JSON_TOPICS["goal_status"] == "/goal/status"
    assert tm.NAV_CHANNELS["goal_marker"] == "/goal/marker"


# --- the driver feeds it ----------------------------------------------------------------

class GoalStub(MapStub):
    uses_goal = True

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.goals = []

    def run(self, x, prev, mode=None, imu=None, lidar=None, map=None, goal=None):
        self.goals.append(goal)
        return super().run(x, prev, mode, imu, lidar, map)


class PosedSlam(FakeSlam):
    def __init__(self, x=0.0, y=0.0, yaw=0.0, frame=0):
        super().__init__()
        self.c = self.c._replace(x=x, y=y, yaw=yaw, frame=frame)


def goal_driver(model=None, slam=None):
    model, slam = model or GoalStub(FORWARD), slam or PosedSlam()
    d = driver(model, slam)
    d.goal = g = gl.ClickGoal()
    return d, model, slam, g


def test_a_goal_model_is_fed_the_clicked_goal_off_the_crop_it_is_fed():
    d, model, slam, g = goal_driver(slam=PosedSlam(1.0, 0.0, math.pi / 2))
    d.tick(T0 + 0.05)
    assert not model.goals[-1].any(), "no click, no goal"
    g.set_goal(0.0, 0.0, 0)
    d.tick(T0 + 0.15)
    np.testing.assert_allclose(model.goals[-1], pg.encode(slam.crop(), g.goal))
    assert model.goals[-1][0, pg.SIN] == pytest.approx(1.0), "(0, 0) is on its left"
    assert d.policy_fields()["policy_goal_present"]
    # A crop gone stale: no map in, so no goal in.
    d.tick(T0 + 1.0)
    assert not model.goals[-1].any() and not d.policy_fields()["policy_goal_present"]
    assert d.controller.throttles[-1] > 0


def test_a_model_without_the_goal_input_is_never_fed_one():
    d, model, slam, g = goal_driver(model=MapStub(FORWARD))
    g.set_goal(2.0, 0.0, 0)
    d.tick(T0 + 0.05)
    assert not d.uses_goal and not d.policy_fields()["policy_goal_present"]


def test_the_fallback_is_never_fed_the_goal():
    d, primary, fallback, slam = fallback_driver(primary=GoalStub(FORWARD))
    d.goal = gl.ClickGoal()
    d.goal.set_goal(3.0, 0.0, 0)
    t = _drive(d, slam, T0, 0.3, True)
    assert primary.goals[-1][0, pg.PRESENT] == 1.0
    t = _drive(d, slam, t, pl.MAP_LOST_S + 0.5, False)
    assert d.policy_fields()["policy_fallback"] and fallback.fed
    assert not d.policy_fields()["policy_goal_present"]


def test_arriving_is_neutral_and_steers_nowhere_until_a_new_goal():
    d, model, slam, g = goal_driver(GoalStub(FORWARD, throttle=0.6, steer=0.5))
    g.set_goal(1.0, 0.0, 0)
    d.tick(T0 + 0.05)
    assert d.controller.throttles[-1] > 0
    slam.c = slam.c._replace(x=1.0 - pg.ARRIVE_M + 0.05, t=T0 + 0.1)
    d.tick(T0 + 0.15)
    assert d.controller.throttles[-1] == 0 and d.controller.steerings[-1] == (0, 0)
    assert d.last_decision.reason == "goal reached"
    assert d.policy_fields()["goal_reached"]
    # Still there, still held.
    slam.c = slam.c._replace(x=0.0, t=T0 + 0.2)
    d.tick(T0 + 0.25)
    assert d.controller.throttles[-1] == 0
    g.set_goal(4.0, 0.0, 0)
    slam.c = slam.c._replace(t=T0 + 0.3)
    d.tick(T0 + 0.35)
    assert d.controller.throttles[-1] > 0 and not d.policy_fields()["goal_reached"]


# --- the real graph and its contract ---------------------------------------------------

def test_the_goal_model_loads_and_its_goal_input_is_wired():
    model = pl.PolicyModel(str(GOAL_MODEL))
    assert model.uses_goal and model.uses_map and model.uses_lidar
    x = np.zeros((1, 3, 32), np.float32)
    prev = np.zeros((1, pl.PREV_SIZE), np.float32)
    mode = pl.encode_mode(False)
    m = np.zeros((1, em.CHANNELS, em.SIZE, em.SIZE), np.float32)
    m[0, em.PRESENT] = 1.0
    off = model.run(x, prev, mode, map=m)
    np.testing.assert_array_equal(off["steer"], model.run(x, prev, mode, map=m,
                                                          goal=pg.missing())["steer"])
    left = pg.encode(crop_at(), (1.0, 2.0, 0))
    on = model.run(x, prev, mode, map=m, goal=left)
    assert not np.allclose(on["steer"], off["steer"])
    assert not pl.PolicyModel(str(FIXTURES / "policy-map.onnx")).uses_goal


def test_a_goal_sidecar_this_car_cannot_encode_is_refused(tmp_path):
    meta = json.loads((FIXTURES / "policy-goal.json").read_text())
    for change in ({"goal_layout": "goal3_far4"}, {"uses_goal": False}, {"uses_map": False}):
        (tmp_path / "m.onnx").write_bytes(GOAL_MODEL.read_bytes())
        (tmp_path / "m.json").write_text(json.dumps(dict(meta, **change)))
        with pytest.raises(ValueError):
            pl.PolicyModel(str(tmp_path / "m.onnx"))
    # A sidecar that says goal over a graph without the input.
    (tmp_path / "m.onnx").write_bytes((FIXTURES / "policy-map.onnx").read_bytes())
    (tmp_path / "m.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        pl.PolicyModel(str(tmp_path / "m.onnx"))


# --- crawler/__main__.py: --goal --------------------------------------------------------

def _pilot(clicks, policy=GOAL_MODEL):
    from crawler import __main__ as mn
    from conftest import FakeController
    from test_policy import Log, ReadyHub
    log = Log()
    slam = PosedSlam()
    return mn, log, slam, mn.AutoPilot(FakeController(), log, hub=ReadyHub(), slam=slam,
                                       policy_path=str(policy), clicks=clicks)


def test_the_goal_flag_parses_and_will_not_go_with_nav():
    from crawler import __main__ as mn
    assert mn.parse_args(["--goal"]).goal is True
    assert mn.parse_args([]).goal is False
    with pytest.raises(SystemExit):
        mn.parse_args(["--goal", "--nav"])


def test_a_click_is_taken_in_manual_and_is_the_goal_auto_drives_to():
    clicks = Clicks(click(2.0, 0.0))
    mn, log, slam, pilot = _pilot(clicks)
    # The patrol, holding the single clicked goal until a patrol is edited.
    assert isinstance(pilot.goal.single, gl.ClickGoal) and pilot.goal.use == "goal"
    pilot.goal_step()            # manual: auto never started
    assert pilot.goal.goal == (2.0, 0.0, 0)
    assert pilot.goal_marker() == {"goal": (2.0, 0.0)}
    assert pilot.resume() is True
    assert pilot.driver.goal is pilot.goal and pilot.driver.uses_goal
    slam.c = slam.c._replace(frame=1)
    pilot.goal_step()
    assert pilot.goal_marker() == {}, "not on the map the truck is on"


def test_no_goal_flag_is_a_goal_nothing_sets():
    mn, log, slam, pilot = _pilot(None)
    assert type(pilot.goal) is gl.Goal
    pilot.goal_step()
    assert not pilot.goal.reached()


def test_a_click_for_a_model_without_the_goal_input_is_warned_about():
    mn, log, slam, pilot = _pilot(Clicks(), policy=FIXTURES / "policy-map.onnx")
    assert pilot.resume() is True
    assert any(subject == "Goal" and "takes no goal" in msg for _, subject, msg in log.lines)


def test_the_click_listener_answers_every_callback_the_sdk_makes_without_asking():
    """foxglove-sdk calls on_subscribe/on_unsubscribe on the listener for every
    channel a client opens or closes, and logs "Callback failed" for each one
    it lacks; the connection graph pair likewise. Every one is there, quiet."""
    fw = pytest.importorskip("foxglove.websocket")
    listener = ck.ClickGoals()
    for name in ("on_subscribe", "on_unsubscribe", "on_client_advertise",
                 "on_client_unadvertise", "on_message_data",
                 "on_connection_graph_subscribe", "on_connection_graph_unsubscribe"):
        assert hasattr(fw.ServerListener, name), name
        assert callable(getattr(listener, name, None)), name
    client, channel = SimpleNamespace(id=1), SimpleNamespace(id=2, topic="/x")
    assert listener.on_subscribe(client, channel) is None
    assert listener.on_unsubscribe(client, channel) is None
