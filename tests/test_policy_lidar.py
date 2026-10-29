"""The roof lidar as a policy input: one builder, the dataset and the car agree.

policy_lidar.py turns a scan into the model's `lidar` input. These hold the
recording side (the adapter's /lidar rows, scan_at, the dataset arrays, the
trainer's lidar_inputs) to the car side (LidarReader.latest(), live_bins,
encode, PolicyDriver) on the same scans, and hold the runtime to refusing a
lidar model it cannot feed.
"""

import json
import pathlib

import numpy as np
import pytest

from crawler.autopilot import driver as dv  # noqa: E402
from crawler.autopilot import footprint as fp  # noqa: E402
from crawler.autopilot import policy as pl  # noqa: E402
from crawler.autopilot import policy_lidar as pli  # noqa: E402
from crawler.hardware import lidar as ld  # noqa: E402
from crawler.telemetry import telemetry as tm  # noqa: E402
import models as md  # noqa: E402
import policy_adapters as pa  # noqa: E402
import policy_dataset as pd  # noqa: E402
import policy_eval as pe  # noqa: E402
import policy_train as pt  # noqa: E402

from conftest import FakeCamera, FakeController  # noqa: E402
from test_policy_tools import S, T0, footprint_session, row  # noqa: E402

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
SMOKE = FIXTURES / "policy-smoke.onnx"    # from before the lidar input
LIDAR_MODEL = FIXTURES / "policy-lidar.onnx"  # untrained, profile + mode + lidar
DEPTH = np.full((400, 640), 3000, np.uint16)


def bins_with(**at):
    """A scan: 1500 mm all round, bin i set to at[f"b{i}"]."""
    b = np.full(pli.BINS, 1500, np.uint16)
    for k, v in at.items():
        b[int(k[1:])] = v
    return b


def scan(t, bins):
    z = np.zeros(0)
    return ld.Scan(t, z, z.astype(np.uint16), z.astype(np.uint8), 0.0, 0, bins)


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


class LidarStub:
    """A model with the lidar input that records what it is fed."""

    contract = pl.CONTRACT_DEPTH_SMALL
    representation = pl.PROFILE
    has_mode = False
    uses_prev = False
    uses_imu = False
    uses_lidar = True

    def __init__(self):
        self.fed = []

    def run(self, x, prev, mode=None, imu=None, lidar=None):
        self.fed.append(None if lidar is None else lidar.copy())
        return {"logits": np.array([[5.0, 0.0, 0.0]], np.float32),
                "throttle": np.array([[0.5]], np.float32),
                "steer": np.array([[0.25]], np.float32)}


# --- the input itself --------------------------------------------------------------

def test_a_scan_is_scaled_clipped_and_no_return_is_not_touching():
    x = pli.encode(bins_with(b0=0, b1=400, b2=9000))
    assert x.shape == (1, pli.SIZE) and x.dtype == np.float32
    r, ret = x[0, pli.RANGES], x[0, pli.RETURNS]
    assert (r[0], ret[0]) == (0.0, 0.0)            # nothing came back
    assert r[1] == pytest.approx(0.1) and ret[1] == 1.0
    assert (r[2], ret[2]) == (1.0, 1.0)            # past CLIP_MM is CLIP_MM
    assert r[3] == pytest.approx(1500 / pli.CLIP_MM)
    assert x[0, pli.PRESENT] == 1.0


def test_no_scan_is_all_zeros_and_says_so():
    for missing in (None, np.zeros(36, np.uint16)):   # none, or another binning
        x = pli.encode(missing)
        assert not x.any() and x[0, pli.PRESENT] == 0.0
    # A scan that saw nothing anywhere is still a scan.
    blank = pli.encode(np.zeros(pli.BINS, np.uint16))
    assert blank[0, pli.PRESENT] == 1.0 and not blank[0, :pli.PRESENT].any()


def test_the_mirror_swaps_left_and_right_and_keeps_ahead_and_behind():
    x = pli.encode(bins_with(b0=100, b18=200, b36=300, b54=400, b1=500))
    m = pli.mirror(x)
    r = m[0, pli.RANGES] * pli.CLIP_MM
    assert r[0] == pytest.approx(100) and r[36] == pytest.approx(300)   # ahead, behind
    assert r[54] == pytest.approx(200) and r[18] == pytest.approx(400)  # left <-> right
    assert r[71] == pytest.approx(500)                                  # 5 deg left -> right
    assert m[0, pli.PRESENT] == 1.0
    np.testing.assert_array_equal(pli.mirror(m), x)


# --- pairing a tick with a scan ----------------------------------------------------

@pytest.mark.parametrize("age", [0.0, 0.1, 0.29, 0.31, 1.0])
def test_scan_at_is_latest_over_the_recorded_rows(age):
    """A recorded scan `age` seconds before the tick is there exactly when
    LidarReader.latest() would have handed it over."""
    clock = Clock(10.0)
    reader = ld.LidarReader(clock=clock)
    reader._scan = scan(10.0 - age, bins_with())
    live = reader.latest()
    i = pli.scan_at([round((10.0 - age) * S)], round(10.0 * S))
    assert (i == 0) == (live is not None)
    assert (i == 0) == (age <= ld.STALE_S)
    # And nothing at all before the first scan.
    assert pli.scan_at([round(11.0 * S)], round(10.0 * S)) == -1


def _drive(n=20):
    rows = [(k * 0.1, row(k, throttle=0.3, forward_limit=0.4, reverse_limit=-1.0))
            for k in range(n)]
    # Scans 50 ms after each tick, then a gap with nothing from 0.75 to 1.45 s,
    # so ticks 1.1 to 1.4 are past STALE_S, and every scan different.
    times = [k * 0.1 + 0.05 for k in range(n) if not 0.8 <= k * 0.1 + 0.05 <= 1.4]
    scans = [(t, bins_with(b0=300 + 37 * j, b18=(0 if j % 3 == 0 else 900 + j),
                           b40=4000 + 100 * j))
             for j, t in enumerate(times)]
    return rows, scans


def test_the_recording_and_the_car_build_the_same_lidar_input():
    """The same scans, once as /lidar rows through the adapter, scan_at, the
    dataset arrays and the trainer's lidar_inputs, once through a LidarReader
    and PolicyDriver live: the same input, tick by tick, stale ticks
    included. A drift here is a model running on input it was never shown."""
    rows, scans = _drive()
    session = footprint_session([(t, json.loads(json.dumps(r))) for t, r in rows])
    ad = pa.FootprintAdapter("s")
    for t, b in scans:
        ad.feed(pa.FOOTPRINT_TOPICS["lidar"], T0 + round(t * S),
                json.loads(json.dumps(ld.lidar_sample(scan(t, b), {}))))
    session = session._replace(lidar=ad.finish().lidar)
    ticks, _, _ = pd.align(session, rules=None)
    arrays = pd.to_arrays([(session, ticks)])
    recorded = pt.lidar_inputs(arrays)

    clock = Clock()
    reader = ld.LidarReader(clock=clock)
    model = LidarStub()
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=DEPTH), model, lidar=reader)
    present = []
    for t, _ in rows:
        now = T0 / S + t
        clock.now = now
        arrived = [(st, b) for st, b in scans if st <= t + 1e-9]
        if arrived:
            reader._scan = scan(T0 / S + arrived[-1][0], arrived[-1][1])
        d.tick(now)
        present.append(d.policy_fields()["policy_lidar_present"])
    live = np.concatenate(model.fed)

    assert recorded.shape == live.shape == (len(rows), pli.SIZE)
    np.testing.assert_allclose(live, recorded, rtol=0, atol=1e-6)
    # The cases are really in there: a stale stretch, and scans either side.
    stale = ~arrays["lidar_present"]
    assert list(np.flatnonzero(stale)) == [0, 11, 12, 13, 14]   # before the first scan, the gap
    assert present == list(arrays["lidar_present"])
    # Into the gap the last scan ages (150, 250 ms) and still counts.
    np.testing.assert_allclose(arrays["lidar_age_ms"][~stale],
                               [50] * 8 + [150, 250] + [50] * 5, atol=1)
    assert (arrays["lidar_age_ms"][stale] == -1).all()


def test_a_session_without_the_lidar_is_all_missing():
    rows, _ = _drive(5)
    session = footprint_session(rows)
    ticks, _, _ = pd.align(session, rules=None)
    arrays = pd.to_arrays([(session, ticks)])
    assert arrays["lidar_bins"].shape == (5, pli.BINS) and not arrays["lidar_bins"].any()
    assert not arrays["lidar_present"].any()
    assert not pt.lidar_inputs(arrays).any()


def test_a_lidar_row_in_another_binning_is_no_scan():
    ad = pa.FootprintAdapter("s")
    ad.feed(pa.FOOTPRINT_TOPICS["lidar"], T0, {"bins": [100] * 36})
    ad.feed(pa.FOOTPRINT_TOPICS["lidar"], T0 + 1, {"bins": [100] * 71 + [-5]})
    ad.feed(pa.FOOTPRINT_TOPICS["lidar"], T0 + 2, {"bins": [100] * 72})
    s = ad.finish()
    assert len(s.lidar) == 1 and ad.bad_lidar == 2


# --- the car: missing is missing ----------------------------------------------------

class Throws:
    def latest(self, *a):
        raise RuntimeError("serial went away")


@pytest.mark.parametrize("reader", [None, "stale", Throws()], ids=["no --lidar", "stale", "throws"])
def test_no_fresh_scan_feeds_no_scan_and_drives_on(reader):
    if reader == "stale":
        clock = Clock(100.0)
        reader = ld.LidarReader(clock=clock)
        reader._scan = scan(100.0 - ld.STALE_S - 0.01, bins_with())
    model = LidarStub()
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=DEPTH), model, lidar=reader)
    d.tick(100.0)
    assert d.last_decision.reason.startswith("model")
    np.testing.assert_array_equal(model.fed[0], pli.missing())
    assert d.policy_fields()["policy_lidar_present"] is False


def test_a_fresh_scan_goes_in_and_the_row_says_so():
    clock = Clock(100.0)
    reader = ld.LidarReader(clock=clock)
    reader._scan = scan(99.9, bins_with(b36=250))
    model = LidarStub()
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=DEPTH), model, lidar=reader)
    d.tick(100.0)
    np.testing.assert_array_equal(model.fed[0], pli.encode(bins_with(b36=250)))
    assert d.policy_fields()["policy_lidar_present"] is True
    assert fp.autopilot_fields(d)["policy_lidar_present"] is True


def test_a_model_without_the_lidar_is_never_fed_it():
    model = pl.PolicyModel(str(SMOKE))
    assert model.uses_lidar is False
    clock = Clock(100.0)
    reader = ld.LidarReader(clock=clock)
    reader._scan = scan(99.95, bins_with())
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=DEPTH), model, lidar=reader)
    d.tick(100.0)
    assert d.last_decision.reason.startswith("model")
    assert d.policy_fields()["policy_lidar_present"] is False


def test_the_lidar_model_loads_and_its_branch_is_wired():
    model = pl.PolicyModel(str(LIDAR_MODEL))
    assert model.uses_lidar and model.has_mode
    x = np.zeros((1, 3, 32), np.float32)
    prev = np.zeros((1, pl.PREV_SIZE), np.float32)
    off = model.run(x, prev, pl.encode_mode(False))
    np.testing.assert_array_equal(off["logits"],
                                  model.run(x, prev, pl.encode_mode(False),
                                            lidar=pli.missing())["logits"])
    on = model.run(x, prev, pl.encode_mode(False), lidar=pli.encode(bins_with(b36=200)))
    assert not np.allclose(on["logits"], off["logits"])
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=DEPTH), model)
    d.tick(100.0)
    assert d.last_decision.reason.startswith("model")


def test_telemetry_carries_policy_lidar_present():
    props = tm.JSON_SCHEMAS["footprint"]["schema"]["properties"]
    assert props["policy_lidar_present"]["type"] == "boolean"
    assert fp.AUTOPILOT_FIELDS["policy_lidar_present"] is False
    assert fp.autopilot_fields(None)["policy_lidar_present"] is False


# --- refusing what it cannot feed --------------------------------------------------

def _copy(tmp_path, onnx, meta):
    (tmp_path / "m.onnx").write_bytes(onnx.read_bytes())
    (tmp_path / "m.json").write_text(json.dumps(meta))
    return str(tmp_path / "m.onnx")


def test_a_lidar_sidecar_this_car_cannot_feed_is_refused(tmp_path):
    meta = json.loads(LIDAR_MODEL.with_suffix(".json").read_text())
    smoke = json.loads(SMOKE.with_suffix(".json").read_text())
    no_claim = {k: v for k, v in meta.items() if not k.startswith("lidar") and k != "uses_lidar"}
    with pytest.raises(ValueError, match="does not declare"):
        pl.PolicyModel(_copy(tmp_path, LIDAR_MODEL, no_claim))
    with pytest.raises(ValueError, match="no 'lidar' input"):
        pl.PolicyModel(_copy(tmp_path, SMOKE, {**smoke, **pli.meta()}))
    for key, bad in (("lidar_layout", "bins360_1deg"), ("lidar_size", 73),
                     ("lidar_clip_mm", 8000.0)):
        with pytest.raises(ValueError, match=key):
            pl.PolicyModel(_copy(tmp_path, LIDAR_MODEL, {**meta, key: bad}))


def test_a_graph_wanting_an_input_this_car_does_not_know_is_refused():
    with pytest.raises(ValueError, match="does not know how to feed"):
        pl.check_lidar({}, ["x", "prev", "mode", "radar"])
    assert pl.check_lidar({}, ["x", "prev", "mode"]) is False
    assert pl.check_lidar(pli.meta(), ["x", "prev", "mode", "lidar"]) is True


def test_the_lidar_is_a_contract_change():
    base = {"input_contract": "depth_small", "representation": "profile",
            "actions": list(pl.ACTIONS), "outputs": ["logits", "throttle", "steer"],
            "steer_scale": 0.7}
    assert md.contract_diff(base, {**base, **pli.meta()}) == ["uses_lidar", "lidar_layout"]
    # An old sidecar reads as no lidar, and a stray layout on one means nothing.
    assert md.contract_of(base)["uses_lidar"] is False
    assert md.contract_diff(base, {**base, "lidar_layout": pli.LAYOUT}) == []


# --- training -------------------------------------------------------------------------

def test_the_sensor_dropout_only_blanks_ticks_with_a_scan():
    rng = np.random.default_rng(0)
    present = np.arange(10000) % 2 == 0
    blank = pt.lidar_blanks(present, 0.3, rng)
    assert not blank[~present].any()
    assert blank[present].mean() == pytest.approx(0.3, abs=0.02)
    assert not pt.lidar_blanks(present, 0.0, rng).any()


def test_the_eval_feeds_a_lidar_model_its_scan_or_none():
    arrays = {"lidar_bins": np.stack([bins_with(b0=700), bins_with()]),
              "lidar_present": np.array([True, False])}
    np.testing.assert_array_equal(pe.lidar_input(arrays, 0), pli.encode(bins_with(b0=700)))
    np.testing.assert_array_equal(pe.lidar_input(arrays, 1), pli.missing())
    assert not pe.lidar_input({}, 0).any()   # a dataset from before the lidar
    meta = pt.lidar_meta(True, 0.3)
    assert pl.check_lidar(meta, ["x", "prev", "lidar"]) and meta["lidar_dropout"] == 0.3
    assert pt.lidar_meta(False) == {}
