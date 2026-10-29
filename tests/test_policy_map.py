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
    # The rear stop: neutral, never the reverse.
    model = MapStub(REVERSE, throttle=1.0)
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
