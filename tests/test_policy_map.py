"""The SLAM map as a policy input, and the map model driving with the hand
rules off.

egomap.py turns the grid into the model's `map` input. These hold the
recording side (tools/map_crops.py over the scans, pair_ticks, the trainer's
map_inputs) to the car side (the runner's worker, SlamRunner.crop(),
egomap.encode, PolicyDriver) on the same scans; hold a map model to driving
on its own gear and steering with only the safety layer over it; and hold
every model before the map input to the rules it always had.
"""

import json
import math
import pathlib
import time

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler.autopilot import driver as dv  # noqa: E402
from crawler.autopilot import policy as pl  # noqa: E402
from crawler.hardware import lidar as ld  # noqa: E402
from crawler.mapping import egomap as em  # noqa: E402
from crawler.mapping import grid as gr  # noqa: E402
from crawler.mapping import runner as rn  # noqa: E402
import map_crops as mc  # noqa: E402
import policy_eval as pe  # noqa: E402
import policy_train as pt  # noqa: E402

from conftest import FakeCamera, FakeController  # noqa: E402
from test_mapping import FakeScan, PERIOD, scan_at  # noqa: E402

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
MAP_MODEL = FIXTURES / "policy-map.onnx"   # untrained, profile + mode + lidar + map
T0 = 100.0

FORWARD = (5.0, 0.0, 0.0)
REVERSE = (0.0, 5.0, 0.0)
STOP = (0.0, 0.0, 5.0)


# --- the crop ----------------------------------------------------------------------

def wall_grid(x_wall, half=1.0):
    g = gr.Grid(size=256)
    ys = np.linspace(-half, half, 81)
    for _ in range(3):
        g.integrate(0.0, 0.0, np.full(len(ys), x_wall), ys)
    return g


def rows_with(codes, what):
    return np.flatnonzero((codes == what).any(1))


def test_the_crop_is_heading_up_and_truck_centred():
    """A wall 1 m along +x: ahead of a truck facing +x, behind one facing -x,
    on the right of one facing +y. Row = ahead, column = left."""
    g = wall_grid(1.0)
    ego = em.EgoMap()
    S, r = em.SIZE, em.RES
    ahead, _ = ego.render(g, 0.0, 0.0, 0.0, 0.0)
    occ = rows_with(ahead, gr.OCCUPIED)
    assert len(occ) and all(abs((k + 0.5 - S / 2) * r - 1.0) <= r for k in occ)
    behind, _ = ego.render(g, 0.0, 0.0, math.pi, 0.0)
    occ = rows_with(behind, gr.OCCUPIED)
    assert len(occ) and all(abs((k + 0.5 - S / 2) * r + 1.0) <= r for k in occ)
    left, _ = ego.render(g, 0.0, 0.0, math.pi / 2, 0.0)
    cols = np.flatnonzero((left == gr.OCCUPIED).any(0))
    assert len(cols) and all((c + 0.5 - S / 2) * r < 0 for c in cols), "right is low columns"
    # The free space between the truck and the wall is free, not unknown.
    assert ahead[S // 2 + 3, S // 2] == gr.FREE


def test_a_thin_wall_survives_the_downsample():
    """One 5 cm cell of wall still makes its 10 cm pixel occupied."""
    g = gr.Grid(size=256)
    g.integrate(0.0, 0.0, np.array([1.02]), np.array([0.0]))
    codes, _ = em.EgoMap().render(g, 0.0, 0.0, 0.0, 0.0)
    assert (codes == gr.OCCUPIED).sum() == 1


def test_the_trail_fades_and_forgets():
    ego = em.EgoMap()
    g = gr.Grid(size=256)
    from crawler.mapping.slam import Result
    base = Result(t=0.0, x=0.0, y=0.0, yaw=0.0, state="tracking", matched=True,
                  written=True, why="", inliers=1.0, rms_m=0.0, constraint=0.4,
                  returns=700, points=600, ground_dropped=0, tilt_deg=0.0, iters=3, ms=1.0)
    ego.after_step(g, base._replace(t=0.0, x=-1.0))
    crop = ego.after_step(g, base._replace(t=30.0, x=0.0))
    S, r = em.SIZE, em.RES
    back = int(math.floor(-1.0 / r + S / 2))
    assert 120 <= crop.trail[back, S // 2] <= 135, "half way through TRAIL_S"
    assert crop.trail[S // 2, S // 2] == 255
    crop = ego.after_step(g, base._replace(t=em.TRAIL_S + 1.0, x=0.0))
    assert crop.trail[back, S // 2] == 0
    assert crop.present
    assert not ego.after_step(g, base._replace(t=62.0, state="lost")).present
    assert not ego.after_step(g, base._replace(t=62.1, matched=False)).present


def test_no_map_is_all_zeros_and_the_mirror_swaps_left_and_right():
    assert not em.missing().any()
    crop = em.Crop(10.0, True, np.full((em.SIZE, em.SIZE), gr.FREE, np.uint8),
                   np.zeros((em.SIZE, em.SIZE), np.uint8))
    assert em.encode(crop, 10.1)[0, em.PRESENT].all()
    assert not em.encode(crop, 10.0 + em.STALE_S + 0.01).any(), "stale"
    assert not em.encode(crop._replace(present=False), 10.1).any(), "not tracking"
    assert not em.encode(None, 10.1).any()
    c = crop.codes.copy()
    c[10, 3] = gr.OCCUPIED
    m = em.encode(crop._replace(codes=c), None)
    mm = em.mirror(m)
    assert mm[0, 0, 10, em.SIZE - 4] == 1.0 and mm[0, 0, 10, 3] == 0.0


# --- the recording and the car draw the same crop ----------------------------------

def _scans(n):
    t = np.array([(k + 1) * PERIOD for k in range(n)])
    ranges = np.array([scan_at(tk) for tk in t], np.float32)
    stamp = np.round(t * 1000).astype(np.int64)
    return dict(t=t, ranges=ranges, stamp=stamp, rpm=np.zeros(n), imu=np.zeros((0, 7)))


@pytest.mark.parametrize("mode", [rn.MODE_THREAD, rn.MODE_PROCESS])
def test_the_recording_and_the_car_draw_the_same_crop(monkeypatch, mode):
    monkeypatch.setattr(rn, "ranges_of", lambda scan, mount: scan.ranges)
    n = 30
    d = _scans(n)
    offline = mc.crops_of(d)
    r = rn.SlamRunner(mode=mode, map_every_s=0.0)
    r.start()
    live = []
    try:
        for k in range(n):
            r.on_scan(FakeScan(d["t"][k], d["ranges"][k], int(d["stamp"][k])))
            end = time.monotonic() + 5.0
            while (r.crop() is None or r.crop().t != offline["t"][k]) and time.monotonic() < end:
                time.sleep(0.002)
            live.append(r.crop())
    finally:
        r.close()
    assert offline["present"][-1]
    for k in (5, n // 2, n - 1):
        assert live[k] is not None and live[k].t == offline["t"][k]
        np.testing.assert_array_equal(live[k].codes, offline["codes"][k])
        np.testing.assert_array_equal(live[k].trail, offline["trail"][k])
        assert live[k].present == offline["present"][k]
    assert (offline["codes"][-1] == gr.OCCUPIED).any()


def test_a_tick_pairs_with_the_newest_crop_no_older_than_stale():
    log_t = np.array([1.0, 1.1, 1.2, 2.0])
    j = mc.pair_ticks(log_t, [0.9, 1.0, 1.15, 1.2 + em.STALE_S - 1e-6, 1.6, 2.05])
    assert j.tolist() == [-1, 0, 1, 2, -1, 3]


def test_the_trainer_and_the_eval_feed_the_paired_crops(tmp_path):
    n = 12
    d = _scans(n)
    out = mc.crops_of(d)
    np.savez_compressed(tmp_path / "s1.npz", **out)
    data = {"meta": {"sessions": ["s0", "s1"]},
            "session": np.array([0, 0, 1, 1, 1]),
            "t_ns": (np.array([0.5, 0.6, 0.05, d["t"][4] + 0.01, d["t"][-1] + 1.0]) * 1e9
                     ).astype(np.int64)}
    assert pt.load_maps(data, str(tmp_path)) == 1
    assert data["map_row"].tolist() == [-1, -1, -1, 4, -1]
    x = pt.map_inputs(data, np.arange(5))
    assert x.shape == (5, em.CHANNELS, em.SIZE, em.SIZE)
    assert not x[[0, 1, 2, 4]].any()
    np.testing.assert_array_equal(
        x[3:4], em.encode_many(out["codes"][4:5], out["trail"][4:5], [True]))
    np.testing.assert_array_equal(pe.map_input(data, 3), x[3:4])
    assert not pe.map_input(data, 0).any()
    bad = dict(out, layout=np.array("ego32_5cm_trail60"))
    np.savez_compressed(tmp_path / "s1.npz", **bad)
    with pytest.raises(ValueError):
        pt.load_maps(data, str(tmp_path))


# --- the model and its contract -----------------------------------------------------

def test_the_map_model_loads_and_its_branch_is_wired():
    model = pl.PolicyModel(str(MAP_MODEL))
    assert model.uses_map and model.uses_lidar and model.has_mode
    x = np.zeros((1, 3, 32), np.float32)
    prev = np.zeros((1, pl.PREV_SIZE), np.float32)
    mode = pl.encode_mode(False)
    off = model.run(x, prev, mode)
    np.testing.assert_array_equal(off["logits"], model.run(x, prev, mode, map=em.missing())["logits"])
    m = np.zeros((1, em.CHANNELS, em.SIZE, em.SIZE), np.float32)
    m[0, 0, 40:44, :] = 1.0
    m[0, em.PRESENT] = 1.0
    assert not np.allclose(model.run(x, prev, mode, map=m)["logits"], off["logits"])


def test_a_map_sidecar_this_car_cannot_draw_is_refused(tmp_path):
    meta = json.loads((FIXTURES / "policy-map.json").read_text())
    for change in ({"map_layout": "ego32_5cm_trail60"}, {"uses_lidar": False},
                   {"uses_map": False}):
        bad = dict(meta, **change)
        (tmp_path / "m.onnx").write_bytes(MAP_MODEL.read_bytes())
        (tmp_path / "m.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            pl.PolicyModel(str(tmp_path / "m.onnx"))


# --- driving with the rules off --------------------------------------------------------

class MapStub:
    """A map model that says what it is told and records what it is fed."""

    contract = pl.CONTRACT_DEPTH_SMALL
    representation = pl.PROFILE
    has_mode = False
    uses_prev = False
    uses_imu = False
    uses_lidar = True
    uses_map = True

    def __init__(self, logits=FORWARD, throttle=0.5, steer=0.25):
        self.say(logits, throttle, steer)
        self.fed = []

    def say(self, logits, throttle=None, steer=None):
        self.logits = logits
        if throttle is not None:
            self.throttle = throttle
        if steer is not None:
            self.steer = steer

    def run(self, x, prev, mode=None, imu=None, lidar=None, map=None):
        self.fed.append(map)
        return {"logits": np.array([self.logits], np.float32),
                "throttle": np.array([[self.throttle]], np.float32),
                "steer": np.array([[self.steer]], np.float32)}


class OldStub(MapStub):
    uses_map = False


class FakeSlam:
    def __init__(self, alive=True):
        self.alive = alive
        self.c = em.Crop(T0, True, np.full((em.SIZE, em.SIZE), gr.FREE, np.uint8),
                         np.zeros((em.SIZE, em.SIZE), np.uint8))

    def at(self, t, present=True):
        """The runner's newest crop is now one drawn at t."""
        self.c = self.c._replace(t=t, present=present)

    def crop(self):
        return self.c

    def stats(self):
        return {"alive": self.alive}


def clear(mm=3000):
    return np.full((400, 640), mm, dtype=np.uint16)


def reader(rear_mm=None):
    """A real LidarReader holding one fresh scan: nothing near, or a return
    rear_mm from the middle of the car dead behind."""
    clock = lambda: T0  # noqa: E731
    r = ld.LidarReader(clock=clock)
    bins = np.full(ld.BINS, 2500, np.uint16)
    if rear_mm is not None:
        bins[int(round(180.0 / ld.BIN_DEG)) % ld.BINS] = rear_mm
    r._scan = ld.Scan(t=T0, angles_deg=np.zeros(0), distances_mm=np.zeros(0, np.uint16),
                      intensities=np.zeros(0, np.uint8), speed_dps=3570.0, packets=40,
                      bins=bins)
    return r


def driver(model, slam=None, depth=None, lidar=None):
    return pl.PolicyDriver(FakeController(), FakeCamera(depth=clear() if depth is None else depth),
                           model, lidar=lidar if lidar is not None else reader(), slam=slam)


def test_a_map_model_is_refused_without_the_mapper():
    with pytest.raises(ValueError):
        driver(MapStub())
    assert driver(OldStub()).hand_rules, "an old model needs no mapper and keeps its rules"


def test_a_map_model_is_fed_the_runners_crop_and_neutral_with_the_mapper_down():
    model, slam = MapStub(), FakeSlam()
    d = driver(model, slam)
    assert not d.hand_rules
    d.tick(T0 + 0.05)
    assert model.fed[-1][0, em.PRESENT].all() and d.policy_fields()["policy_map_present"]
    assert d.policy_fields()["policy_hand_rules"] is False
    d.tick(T0 + 1.0)
    assert not model.fed[-1].any(), "stale: no map, and it still drives"
    assert d.controller.throttles[-1] > 0
    slam.alive = False
    d.tick(T0 + 1.1)
    assert d.controller.throttles[-1] == 0 and d.last_decision.reason == "mapper down"


@pytest.mark.parametrize("how", ["lost", "stale"])
def test_a_map_model_parks_when_the_map_is_lost_past_a_second(how):
    """Lost (a crop that is not tracking and matched) or no new crop at all:
    under MAP_LOST_S it drives on the no-map input, past it neutral with
    "map lost", and a trusted crop again drives."""
    model, slam = MapStub(FORWARD, steer=0.4), FakeSlam()
    d = driver(model, slam)
    slam.at(T0)
    d.tick(T0)
    assert d.controller.throttles[-1] > 0
    t = T0
    while t < T0 + pl.MAP_LOST_S - 0.15:
        t += 0.1
        if how == "lost":
            slam.at(t, present=False)
        d.tick(t)
        assert d.controller.throttles[-1] > 0, f"{t - T0:.1f} s in, still under the window"
    for _ in range(3):
        t += 0.1
        if how == "lost":
            slam.at(t, present=False)
        d.tick(t)
    assert d.controller.throttles[-1] == 0 and d.controller.steerings[-1] == (0, 0)
    assert d.last_decision.reason == "map lost"
    t += 0.1
    slam.at(t)
    d.tick(t)
    assert d.controller.throttles[-1] > 0, "a trusted crop drives again"


def _run(d, model, script, t0=T0):
    for k, logits in enumerate(script):
        model.say(logits)
        d.tick(t0 + k * 0.1)
    return d.controller.throttles


def test_the_map_model_reverses_on_its_own_steering_with_no_commit_or_refractory():
    model = MapStub(REVERSE, throttle=1.0, steer=-0.3)
    d = driver(model, FakeSlam())
    # A turning forward first: the old side rule would reverse to the other side.
    model.say(FORWARD, steer=0.9)
    d.tick(T0)
    model.say(REVERSE, steer=0.4)
    d.tick(T0 + 0.1)
    assert d.controller.throttles[-1] < 0
    assert d.controller.steerings[-1][0] == pytest.approx(0.4 * pl.STEER_SCALE)
    assert not d.reverse_commit
    # Forward the next tick is forward at once (no commit holds the reverse)...
    model.say(FORWARD, steer=0.0)
    d.tick(T0 + 0.2)
    assert d.controller.throttles[-1] > 0
    # ...and reverse straight after is reverse (no refractory).
    model.say(REVERSE, steer=-0.2)
    d.tick(T0 + 0.3)
    assert d.controller.throttles[-1] < 0 and not d.reverse_refractory
    assert d.controller.steerings[-1][0] == pytest.approx(-0.2 * pl.STEER_SCALE)


def test_the_old_model_keeps_the_commit_and_the_refractory():
    model = OldStub(REVERSE, throttle=1.0, steer=-0.3)
    d = driver(model)
    model.say(FORWARD, steer=0.9)
    d.tick(T0)
    model.say(REVERSE)
    d.tick(T0 + 0.1)
    assert d.reverse_commit
    assert d.controller.steerings[-1][0] == pytest.approx(-pl.REVERSE_STEER)
    model.say(FORWARD)
    d.tick(T0 + 0.2)
    assert d.controller.throttles[-1] < 0, "the commit holds the reverse"


def test_the_map_model_stops_when_it_says_stop_and_keeps_its_throttle_unscaled():
    model = MapStub(STOP)
    d = driver(model, FakeSlam())
    d.tick(T0)
    assert d.controller.throttles[-1] == 0 and not d.last_decision.stop_masked
    model.say((1.0, 0.0, 0.9), throttle=0.5)   # a forward it is barely sure of
    d.tick(T0 + 0.1)
    assert d.controller.throttles[-1] == pytest.approx(d.forward_speed * 0.5)
    old = OldStub(STOP)
    do = driver(old)
    do.tick(T0)
    assert do.last_decision.stop_masked and do.controller.throttles[-1] != 0


def test_the_safety_layer_still_cuts_a_map_models_throttle():
    # The rear stop: neutral, never the reverse, when the model's next best
    # is a stop.
    model = MapStub((0.0, 5.0, 1.0), throttle=1.0)
    near = int(dv.REAR_SELF_MM + dv.REAR_STOP_CLEAR_MM - 40)
    d = driver(model, FakeSlam(), lidar=reader(near))
    d.tick(T0)
    assert d.controller.throttles[-1] == 0 and d.controller.steerings[-1] == (0, 0)
    assert d.last_decision.reason == "rear stop"
    # The reverse speed cap the rear stop is set for.
    d = driver(MapStub(REVERSE, throttle=1.0), FakeSlam())
    d.tick(T0)
    assert d.controller.throttles[-1] == dv.ESCAPE_SPEED
    # The floor holds a forward into a wall at 0, and never backs out for it.
    wall = clear()
    wall[150:350, 220:420] = 300
    model = MapStub(FORWARD, steer=0.5)
    slam = FakeSlam()
    d = driver(model, slam, depth=wall)
    for k in range(int(pl.STANDOFF_S / 0.1) + 15):
        slam.at(T0 + k * 0.1)
        d.tick(T0 + k * 0.1)
    assert all(t == 0 for t in d.controller.throttles[-10:])
    assert not d.recovering(T0 + 3.0), "no standoff back-out"
    assert d.controller.steerings[-1][0] == pytest.approx(0.5 * pl.STEER_SCALE), \
        "the model keeps its steering under the hold"


def test_a_map_models_reverse_into_the_rear_stop_goes_forward_if_it_would():
    # 174641Z: reverse at p 0.66 into something behind, the front clear to
    # 800 mm, every tick for 9.5 s, each one cut to neutral. Forward was its
    # next best on every one of them.
    near = int(dv.REAR_SELF_MM + dv.REAR_STOP_CLEAR_MM - 40)
    slam = FakeSlam()
    d = driver(MapStub((1.0, 5.0, 0.0), throttle=0.8, steer=0.75), slam,
               lidar=reader(near))
    for k in range(20):
        slam.at(T0 + k * 0.1)
        d.tick(T0 + k * 0.1)
    assert all(t > 0 for t in d.controller.throttles[-20:]), "it drives on"
    dec = d.last_decision
    assert dec.action == pl.FORWARD and dec.reason == "model rear masked"
    assert dec.model_action == pl.REVERSE, "the row keeps what it asked"
    assert d.rear_stop
    assert d.controller.steerings[-1][0] == pytest.approx(0.75 * pl.STEER_SCALE)
    # And the floor still holds it, as any forward.
    wall = clear()
    wall[150:350, 220:420] = 300
    d = driver(MapStub((1.0, 5.0, 0.0)), FakeSlam(), depth=wall,
               lidar=reader(near))
    d.tick(T0)
    assert d.controller.throttles[-1] == 0
    assert d.last_decision.reason == "model floor"
    # Nothing behind: the reverse it asked for.
    d = driver(MapStub((1.0, 5.0, 0.0), throttle=1.0), FakeSlam())
    d.tick(T0)
    assert d.controller.throttles[-1] == dv.ESCAPE_SPEED


# --- a chassis left sitting tilted is not a lost map ----------------------------------

def _result(**kw):
    from crawler.mapping.slam import Result
    base = Result(t=0.0, x=0.0, y=0.0, yaw=0.0, state="tracking", matched=True,
                  written=True, why="", inliers=1.0, rms_m=0.0, constraint=0.4,
                  returns=700, points=600, ground_dropped=0, tilt_deg=0.0, iters=3, ms=1.0)
    return base._replace(**kw)


def test_a_crop_at_a_tilted_but_matched_pose_is_present():
    """Tilted is the write gate, not the pose: the scan still matched, so the
    crop drawn at it is one to drive on, off the map as it was last written.
    A tilted scan that did not match, lost and rolled are still no map."""
    g = wall_grid(1.0)
    ego = em.EgoMap()
    tilted = _result(t=1.0, state="tilted", written=False, why="tilted",
                     tilt_deg=7.5, roll_deg=7.5)
    crop = ego.after_step(g, tilted)
    assert crop.present
    assert em.encode(crop, 1.05)[0, em.PRESENT].all()
    assert not ego.after_step(g, tilted._replace(t=1.1, matched=False)).present
    assert not ego.after_step(g, tilted._replace(t=1.2, state="lost")).present
    assert not ego.after_step(g, tilted._replace(t=1.3, state="rolled", matched=False)).present


class LiveSlam:
    """The runner's face over a real Slam and EgoMap, stepped by the test."""

    def __init__(self):
        from crawler.mapping import slam as sl
        from crawler.mapping.imu import ImuTrack
        self.imu = ImuTrack()
        self.slam = sl.Slam(imu=self.imu)
        self.ego = em.EgoMap()
        self.c = None
        self.res = None

    def step(self, t, up):
        from test_mapping import drive
        self.res = drive(self.slam, 1, t0=t - PERIOD, imu=self.imu, up=up)[-1]
        self.c = self.ego.after_step(self.slam.grid, self.res)
        return self.res

    def crop(self):
        return self.c

    def latest(self):
        return self.res

    def stats(self):
        return {"alive": True}


def test_a_map_model_drives_on_through_a_chassis_left_tilted():
    """The drive of 2026-10-29 17:46Z: after a knock the truck sat rolled 6 to
    8 degrees, past the 5.5 degree roll gate, so every scan came back tilted
    (matched, not written), no crop was present, and a second later the car
    was parked on "map lost" until I took over. Measured there, 659 of 668
    tilted scans matched, median inliers 0.89. Here: a real Slam, level for
    2 s, then rolled 7.5 degrees and held there for 4 s. The pose keeps
    tracking, the model is fed the map every tick, and the car never stops."""
    from test_mapping import LEVEL, err_vs_truth, rolled_up
    live = LiveSlam()
    model = MapStub(FORWARD, steer=0.2)
    d = driver(model, live)
    t = 0.0
    for _ in range(20):
        t = round(t + PERIOD, 6)
        live.step(t, LEVEL)
        d.tick(t + 0.01)
    up = rolled_up(7.5)
    for _ in range(40):
        t = round(t + PERIOD, 6)
        res = live.step(t, up)
        assert res.state == "tilted" and res.matched and not res.written, \
            "the premise: past the roll gate, matched, not written"
        d.tick(t + 0.01)
        assert d.last_decision.reason == "model", f"{t:.1f} s: {d.last_decision.reason}"
        assert d.controller.throttles[-1] > 0
        assert model.fed[-1][0, em.PRESENT].all()
    pos_err, _ = err_vs_truth(res)
    assert pos_err < 0.05, "the pose kept tracking under the tilt"


def test_a_really_lost_map_still_parks_the_car():
    """The safety neutral stays for a SLAM that is genuinely lost."""
    from test_mapping import LEVEL
    live = LiveSlam()
    d = driver(MapStub(FORWARD), live)
    t = 0.0
    for _ in range(20):
        t = round(t + PERIOD, 6)
        live.step(t, LEVEL)
        d.tick(t + 0.01)
    assert d.controller.throttles[-1] > 0
    for _ in range(int(pl.MAP_LOST_S / PERIOD) + 3):
        t = round(t + PERIOD, 6)
        live.step(t, LEVEL)
        live.c = live.c._replace(present=False)   # what a lost Result draws
        d.tick(t + 0.01)
    assert d.controller.throttles[-1] == 0 and d.last_decision.reason == "map lost"


# --- every safety neutral in auto says why, once ------------------------------------

def _neutral_lines(out):
    return [ln for ln in out.splitlines() if ln.startswith("Auto ")]


def test_a_safety_neutral_in_auto_is_logged_with_its_reason_and_rate_limited(capsys):
    """A stop says why when it starts, again only every STOP_LOG_EVERY_S while
    it lasts, and how long it lasted when the car moves again: the journal
    of 10-29 had 15 takeovers and not one word on why the car had stopped."""
    model, slam = MapStub(FORWARD), FakeSlam()
    d = driver(model, slam)
    slam.at(T0)
    d.tick(T0)
    capsys.readouterr()
    t = T0
    n = int((pl.MAP_LOST_S + pl.STOP_LOG_EVERY_S + 0.5) / 0.1)
    for _ in range(n):
        t += 0.1
        slam.at(t, present=False)
        d.tick(t)
    lines = _neutral_lines(capsys.readouterr().out)
    assert len(lines) == 2, lines
    assert lines[0].startswith("Auto neutral - map lost")
    assert lines[1].startswith("Auto still neutral - map lost")
    t += 0.1
    slam.at(t)
    d.tick(t)
    lines = _neutral_lines(capsys.readouterr().out)
    assert len(lines) == 1 and lines[0].startswith("Auto driving again"), lines
    assert "map lost" in lines[0]


def test_a_flickering_stop_is_one_line_and_a_takeover_says_what_it_ended():
    said = []
    log = pl.StopLog(say=said.append)
    t = 0.0
    for _ in range(10):                      # stop, move, stop, move: 2 s
        log.stopped("map lost", t)
        log.moving(t + 0.1)
        t += 0.2
    assert said[0] == "Auto neutral - map lost"
    assert [s for s in said if s.startswith("Auto neutral")] == said[:1], said
    t = pl.STOP_LOG_EVERY_S + 1.0
    log.stopped("map lost", t, "slam tilted, pitch 0.2 roll 7.9")
    assert said[-1] == ("Auto neutral - map lost (slam tilted, pitch 0.2 roll 7.9), "
                        "9 more since the last line")
    log.taken(t + 2.5)
    assert said[-1] == "Auto takeover after 2.5 s of map lost"
    n = len(said)
    log.moving(t + 3.0)
    assert len(said) == n, "the takeover already ended it"


def test_a_stop_under_my_trigger_is_not_logged(capsys):
    model, slam = MapStub(FORWARD), FakeSlam()
    d = driver(model, slam)
    d.human_throttle = True
    t = T0
    for _ in range(int(pl.MAP_LOST_S / 0.1) + 5):
        t += 0.1
        slam.at(t, present=False)
        d.tick(t)
    assert d.last_decision.reason == "map lost"
    assert not _neutral_lines(capsys.readouterr().out)


def test_a_floor_blip_is_quiet_and_a_floor_hold_is_logged(capsys):
    wall = clear()
    wall[150:350, 220:420] = 300
    model, slam = MapStub(FORWARD), FakeSlam()
    d = driver(model, slam, depth=wall)
    t = T0
    for _ in range(3):
        slam.at(t)
        d.tick(t)
        t += 0.1
    assert d.last_decision.reason == "model floor"
    assert not _neutral_lines(capsys.readouterr().out), "a short hold says nothing"
    for _ in range(int(pl.FLOOR_LOG_AFTER_S / 0.1) + 2):
        slam.at(t)
        d.tick(t)
        t += 0.1
    lines = _neutral_lines(capsys.readouterr().out)
    assert len(lines) == 1 and lines[0].startswith("Auto neutral - model floor"), lines


# --- a lost map hands the wheels to the map-free fallback, not to neutral ----------

class FallbackStub(OldStub):
    label = "the fallback"
    chunk = 1


def fallback_driver(primary=None, fallback=None, slam=None, depth=None):
    primary = primary or MapStub(FORWARD, steer=0.25)
    fallback = fallback or FallbackStub(FORWARD, steer=-0.3)
    slam = slam or FakeSlam()
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear() if depth is None else depth),
                        primary, lidar=reader(), slam=slam, fallback=fallback)
    return d, primary, fallback, slam


def _drive(d, slam, t, seconds, present):
    """Tick every 0.1 s for `seconds`, the crop present or not."""
    for _ in range(int(round(seconds / 0.1))):
        t = round(t + 0.1, 6)
        slam.at(t, present=present)
        d.tick(t)
    return t


def test_a_lost_map_drives_on_the_fallback_and_comes_back_after_a_healthy_second():
    d, primary, fallback, slam = fallback_driver()
    t = _drive(d, slam, T0, 0.5, True)
    assert len(primary.fed) == 5 and not fallback.fed
    assert not d.policy_fields()["policy_fallback"]
    # Under MAP_LOST_S the map model rides the gap on its no-map input.
    t = _drive(d, slam, t, pl.MAP_LOST_S - 0.1, False)
    assert not fallback.fed, "inside the window it is still the map model"
    # Past it: the fallback, never neutral.
    t = _drive(d, slam, t, 0.5, False)
    assert fallback.fed and d.policy_fields()["policy_fallback"]
    assert d.controller.throttles[-1] > 0
    assert d.controller.steerings[-1][0] == pytest.approx(-0.3 * pl.STEER_SCALE)
    assert d.last_decision.reason != "map lost"
    # The map back, but not yet for a second: still the fallback.
    n = len(primary.fed)
    t = _drive(d, slam, t, pl.MAP_BACK_S - 0.2, True)
    assert len(primary.fed) == n, "hysteresis: a healthy map has to stay healthy first"
    # A blink restarts the second.
    t = _drive(d, slam, t, 0.1, False)
    t = _drive(d, slam, t, pl.MAP_BACK_S - 0.2, True)
    assert len(primary.fed) == n
    t = _drive(d, slam, t, 0.4, True)
    assert len(primary.fed) > n and not d.policy_fields()["policy_fallback"]
    assert d.controller.steerings[-1][0] == pytest.approx(0.25 * pl.STEER_SCALE)
    # One model a tick, never both.
    ticks = round((t - T0) / 0.1)
    assert len(primary.fed) + len(fallback.fed) == ticks


def test_the_fallback_is_never_fed_a_map_and_the_map_model_rides_short_gaps():
    d, primary, fallback, slam = fallback_driver()
    t = _drive(d, slam, T0, 0.3, True)
    t = _drive(d, slam, t, pl.MAP_LOST_S + 0.5, False)
    assert fallback.fed and all(m is None for m in fallback.fed)


def test_every_switch_resets_the_chunk_blend():
    d, primary, fallback, slam = fallback_driver()
    resets = {"map": 0, "fallback": 0}
    for name, slot in d._slots.items():
        real = slot["ensembler"].reset

        def counted(real=real, name=name):
            resets[name] += 1
            real()
        slot["ensembler"].reset = counted
    t = _drive(d, slam, T0, 0.3, True)
    before = dict(resets)
    t = _drive(d, slam, t, pl.MAP_LOST_S + 0.3, False)
    assert resets["fallback"] > before["fallback"], "switching to the fallback"
    before = dict(resets)
    t = _drive(d, slam, t, pl.MAP_BACK_S + 0.3, True)
    assert resets["map"] > before["map"], "switching back to the map model"


def test_the_mapper_down_goes_to_the_fallback_at_once():
    d, primary, fallback, slam = fallback_driver()
    t = _drive(d, slam, T0, 0.3, True)
    slam.alive = False
    t = _drive(d, slam, t, 0.1, True)
    assert fallback.fed and d.controller.throttles[-1] > 0


def test_bad_sensor_input_is_still_neutral_on_the_fallback():
    d, primary, fallback, slam = fallback_driver()
    t = _drive(d, slam, T0, 0.3, True)
    t = _drive(d, slam, t, pl.MAP_LOST_S + 0.3, False)
    assert d.policy_fields()["policy_fallback"]
    d.camera.depth = None
    t = round(t + 0.1, 6)
    d.tick(t)
    assert d.controller.throttles[-1] == 0 and d.last_decision.reason == "no frame"


def test_no_fallback_loaded_is_still_neutral_on_a_lost_map():
    model, slam = MapStub(FORWARD), FakeSlam()
    d = driver(model, slam)
    t = _drive(d, slam, T0, pl.MAP_LOST_S + 0.5, False)
    assert d.controller.throttles[-1] == 0 and d.last_decision.reason == "map lost"


def test_a_fallback_that_needs_the_map_is_refused():
    with pytest.raises(ValueError):
        fallback_driver(fallback=MapStub())


def test_each_switch_is_logged_with_its_reason(capsys):
    d, primary, fallback, slam = fallback_driver()
    t = _drive(d, slam, T0, 0.3, True)
    capsys.readouterr()
    t = _drive(d, slam, t, pl.MAP_LOST_S + 0.5, False)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("Auto ")]
    assert len(lines) == 1 and lines[0].startswith("Auto fallback - map lost"), lines
    assert "the fallback" in lines[0]
    t = _drive(d, slam, t, pl.MAP_BACK_S + 0.5, True)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("Auto ")]
    assert len(lines) == 1 and lines[0].startswith("Auto map back"), lines
    # A second round inside the rate limit says nothing; the next line
    # after it counts what went unsaid.
    t = _drive(d, slam, t, pl.MAP_LOST_S + 0.5, False)
    t = _drive(d, slam, t, pl.MAP_BACK_S + 0.5, True)
    assert not [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("Auto ")]
    t = _drive(d, slam, t, pl.STOP_LOG_EVERY_S, True)
    t = _drive(d, slam, t, pl.MAP_LOST_S + 0.5, False)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("Auto ")]
    assert len(lines) == 1 and "1 more like it" in lines[0], lines


def test_the_idle_model_s_imu_window_keeps_up():
    d, primary, fallback, slam = fallback_driver()
    pushes = {"map": 0, "fallback": 0}
    for name, slot in d._slots.items():
        real = slot["imu_features"].push

        def counted(row, now, real=real, name=name):
            pushes[name] += 1
            return real(row, now)
        slot["imu_features"].push = counted
    _drive(d, slam, T0, 0.5, True)
    assert pushes == {"map": 5, "fallback": 5}


# --- crawler/__main__.py: --fallback-policy ------------------------------------------

def _pilot(**kw):
    from crawler import __main__ as mn
    from test_policy import Log, ReadyHub
    log = Log()
    return mn, log, mn.AutoPilot(FakeController(), log, hub=ReadyHub(), slam=FakeSlam(),
                                 policy_path=str(MAP_MODEL), **kw)


def test_the_fallback_flag_loads_the_fallback_beside_the_map_model():
    smoke = FIXTURES / "policy-smoke.onnx"
    mn, log, pilot = _pilot(fallback_path=str(smoke))
    assert mn.parse_args(["--fallback-policy", str(smoke)]).fallback_policy == str(smoke)
    assert mn.parse_args([]).fallback_policy is None
    assert pilot.resume() is True
    assert pilot.driver.fallback is not None and pilot.driver.uses_map
    assert any("falls back to" in line[2] for line in log.lines)


def test_no_fallback_flag_is_the_map_model_alone():
    mn, log, pilot = _pilot()
    assert pilot.resume() is True
    assert pilot.driver.fallback is None


def test_a_fallback_that_will_not_load_is_auto_unavailable(tmp_path):
    mn, log, pilot = _pilot(fallback_path=str(tmp_path / "missing.onnx"))
    assert pilot.resume() is False and pilot.unavailable is True
