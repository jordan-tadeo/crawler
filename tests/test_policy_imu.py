"""The policy's IMU input: one builder, the recording and the car agreeing.

The features are made twice -- by the dataset tool off recorded /footprint
rows, and by PolicyDriver off the live camera view -- and the only thing
keeping those two the same is that both go through policy_imu. So the test
that matters most here is the parity one: the same samples through the
recording side and the runtime side come out as the same numbers. Around it,
the cases a recording really has: no sample yet, a stalled stream, a session
from before the up vector existed, and a model that predates the IMU input
entirely, which has to load and run exactly as it did.
"""

import json
import math
import os
import pathlib
import sys

import depthai as dai
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tools"))

import footprint as fp  # noqa: E402
import oakd  # noqa: E402
import policy as pl  # noqa: E402
import policy_adapters as pa  # noqa: E402
import policy_dataset as pd  # noqa: E402
import policy_eval as pe  # noqa: E402
import policy_imu as pimu  # noqa: E402
import telemetry as tm  # noqa: E402

from conftest import FakeCamera, FakeController  # noqa: E402
from test_policy_tools import footprint_session, row  # noqa: E402

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
SMOKE = FIXTURES / "policy-smoke.onnx"    # from before the IMU input
IMU_MODEL = FIXTURES / "policy-imu.onnx"  # untrained, profile + prev + mode + imu

F = {name: i for i, name in enumerate(pimu.FEATURES)}
T0 = 100.0
LEVEL = (1.0, 0.0, 0.0)  # up on this car's mounting, x vertical


def sample(accel=(0.1, 0.2, 0.3), gyro=(0.4, 0.05, -0.1), up=None):
    out = {"accel": accel, "gyro": gyro}
    if up is not None:
        out["up"] = up
    return out


def push_all(rows, tilt=True, dt=0.1):
    feats = pimu.ImuFeatures(tilt=tilt)
    return [feats.push(r, T0 + k * dt) for k, r in enumerate(rows)]


# --- the builder ---------------------------------------------------------------

def test_no_sample_yet_is_all_zeros_and_says_so():
    """imu_age_ms -1 and zeros is how a row with no IMU reads. Every feature
    is 0, valid included, so the model can tell it from a parked car."""
    row_ = pimu.row_fields(None, 0.0, 1.0)
    assert row_["imu_age_ms"] == -1.0
    out = push_all([row_])[0]
    assert out.shape == (pimu.IMU_SIZE,) and out.dtype == np.float32
    assert not out.any()


def test_a_parked_car_is_valid_and_near_zero_not_the_same_as_no_sample():
    out = push_all([pimu.row_fields(sample((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), 1.0, 1.0)])[0]
    assert out[F["valid"]] == 1.0
    assert not np.delete(out, F["valid"]).any()


def test_a_stalled_stream_is_not_a_reading():
    """The view keeps handing back the last sample's time; past MAX_AGE_MS
    the numbers are history, not the car."""
    fresh = pimu.row_fields(sample(), 1.0, 1.0 + pimu.MAX_AGE_MS / 1000.0 - 0.01)
    stale = pimu.row_fields(sample(), 1.0, 1.0 + pimu.MAX_AGE_MS / 1000.0 + 0.01)
    assert push_all([fresh])[0][F["valid"]] == 1.0
    assert not push_all([stale])[0].any()


def test_values_are_scaled_and_clipped():
    out = push_all([pimu.row_fields(sample((100.0, -5.0, 0.0), (0.0, 1.0, -50.0)), 1.0, 1.0)])[0]
    assert out[F["accel_x"]] == pimu.CLIP and out[F["accel_y"]] == pytest.approx(-0.5)
    assert out[F["gyro_y"]] == pytest.approx(0.5) and out[F["gyro_z"]] == -pimu.CLIP
    assert out[F["accel_peak"]] == pimu.CLIP


def test_the_peak_holds_a_knock_for_the_window_and_a_gap_starts_it_over():
    knock = pimu.row_fields(sample((6.0, 8.0, 0.0), (0.0, 0.0, 0.0)), 0.0, 0.0)
    calm = pimu.row_fields(sample((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), 0.0, 0.0)
    peaks = [f[F["accel_peak"]] for f in push_all([knock] + [calm] * 6)]
    assert peaks[:pimu.WINDOW] == [pytest.approx(10.0 / pimu.ACCEL_SCALE)] * pimu.WINDOW
    assert peaks[pimu.WINDOW:] == [0.0, 0.0]

    feats = pimu.ImuFeatures()
    feats.push(knock, T0)
    after_gap = feats.push(calm, T0 + pimu.GAP_S + 0.05)
    assert after_gap[F["accel_peak"]] == 0.0


def test_an_old_session_with_no_up_vector_reads_no_tilt_and_the_rest_as_usual():
    old = row(1, imu_age_ms=3.0, imu_accel_x=1.0, imu_accel_y=0.0, imu_accel_z=0.0,
              imu_gyro_x=0.2, imu_gyro_y=0.0, imu_gyro_z=0.0)
    assert not any(k in old for k in pimu.ROW_UP)
    out = push_all([old])[0]
    assert out[F["valid"]] == 1.0 and out[F["accel_x"]] == pytest.approx(0.1)
    assert out[F["tilt_valid"]] == 0.0 and not out[F["up_x"]:F["up_z"] + 1].any()


def test_a_model_trained_without_tilt_never_sees_it_even_when_the_car_has_it():
    rows = [pimu.row_fields(sample(up=(0.97, 0.0, 0.243)), 0.0, 0.0)]
    with_tilt = push_all(rows, tilt=True)[0]
    without = push_all(rows, tilt=False)[0]
    assert with_tilt[F["tilt_valid"]] == 1.0
    assert with_tilt[F["up_x"]] == pytest.approx(0.97, abs=1e-3)
    assert without[F["tilt_valid"]] == 0.0 and not without[F["up_x"]:].any()
    np.testing.assert_array_equal(without, pimu.without_tilt(with_tilt))


def test_up_off_the_rotation_vector():
    """Android's convention, which the BNO08x follows: the quaternion takes
    device axes to a world whose z is up. Identity is device z up; a quarter
    turn about y puts world up along device -x."""
    assert pimu.up_from_quaternion(0, 0, 0, 1) == pytest.approx((0.0, 0.0, 1.0))
    s = math.sqrt(0.5)
    assert pimu.up_from_quaternion(0, s, 0, s) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-9)
    assert pimu.up_from_quaternion(s, 0, 0, s) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert pimu.up_from_quaternion(0, 0, 0, 0) is None        # not reported
    assert pimu.up_from_quaternion(float("nan"), 0, 0, 1) is None


def test_the_mirror_flips_yaw_roll_and_everything_lateral():
    f = np.arange(1, pimu.IMU_SIZE + 1, dtype=np.float32)
    m = pimu.mirror(f)
    flipped = {n for n in pimu.FEATURES if m[F[n]] == -f[F[n]]}
    assert flipped == {"gyro_x", "gyro_z", "accel_y", "up_y"}
    np.testing.assert_array_equal(pimu.mirror(m), f)


# --- what the car records -----------------------------------------------------

def _packet(accel=(0.1, 0.2, 0.3), gyro=(0.4, 0.5, 0.6), quat=None):
    p = dai.IMUPacket()
    p.acceleroMeter.x, p.acceleroMeter.y, p.acceleroMeter.z = accel
    p.gyroscope.x, p.gyroscope.y, p.gyroscope.z = gyro
    if quat is not None:
        rv = p.rotationVector
        rv.i, rv.j, rv.k, rv.real = quat
    return p


def test_the_pipeline_adds_orientation_and_keeps_the_old_two_sensors():
    imu = [n for n in oakd.build_pipeline().getAllNodes() if n.getName() == "IMU"]
    assert len(imu) == 1
    sensors = {s.sensorId: s.reportRate for s in imu[0].properties.imuSensors}
    assert sensors == {dai.IMUSensor.LINEAR_ACCELERATION: 100,
                       dai.IMUSensor.GYROSCOPE_RAW: 100,
                       dai.IMUSensor.GAME_ROTATION_VECTOR: 100}


def test_a_reading_keeps_accel_and_gyro_and_adds_up_only_when_reported():
    data = dai.IMUData()
    data.packets = [_packet()]
    s = oakd.imu_reading(data)
    assert s["accel"] == pytest.approx((0.1, 0.2, 0.3))
    assert s["gyro"] == pytest.approx((0.4, 0.5, 0.6))
    assert "up" not in s

    data.packets = [_packet(), _packet((9.0, 9.0, 9.0), quat=(0, 0, 0, 1))]
    s = oakd.imu_reading(data)
    assert s["accel"] == pytest.approx((0.1, 0.2, 0.3))   # still the first packet's
    assert s["up"] == pytest.approx((0.0, 0.0, 1.0))
    assert oakd.imu_reading(None) is None


def test_rows_and_imu_messages_carry_up_and_zeros_without_it():
    with_up = fp.imu_fields(sample(up=LEVEL), 1.0, 1.01)
    assert (with_up["imu_up_x"], with_up["imu_up_y"], with_up["imu_up_z"]) == LEVEL
    without = fp.imu_fields(sample(), 1.0, 1.01)
    assert (without["imu_up_x"], without["imu_up_y"], without["imu_up_z"]) == (0.0, 0.0, 0.0)
    # The old columns mean what they meant.
    assert without["imu_accel_y"] == 0.2 and without["imu_gyro_x"] == 0.4
    assert without["imu_age_ms"] == pytest.approx(10.0)
    assert set(fp.imu_fields(None, 0, 0)) == set(pimu.ROW_FIELDS)

    msg = tm.imu_sample(sample(up=LEVEL))
    assert (msg["up_x"], msg["up_y"], msg["up_z"]) == LEVEL
    assert tm.imu_sample(sample())["up_x"] == 0.0
    for key in ("imu_up_x", "imu_up_y", "imu_up_z"):
        assert key in tm.JSON_SCHEMAS["footprint"]["schema"]["properties"]
    for key in ("up_x", "up_y", "up_z"):
        assert key in tm.JSON_SCHEMAS["imu"]["schema"]["properties"]


# --- training and the car agree -------------------------------------------------

class ImuStub:
    """A model with the IMU input that records what it is fed."""

    contract = pl.CONTRACT_DEPTH_SMALL
    representation = pl.PROFILE
    has_mode = False

    def __init__(self, uses_prev=False, tilt=True):
        self.uses_prev = uses_prev
        self.uses_imu = True
        self.imu_tilt = tilt
        self.fed = []
        self.prevs = []

    def run(self, x, prev, mode=None, imu=None):
        self.fed.append(None if imu is None else imu.copy())
        self.prevs.append(prev.copy())
        return {"logits": np.array([[5.0, 0.0, 0.0]], np.float32),
                "throttle": np.array([[0.5]], np.float32),
                "steer": np.array([[0.25]], np.float32)}


def _samples():
    """Ten ticks of a drive: moving, a knock, a dropped sample, a slope."""
    out = []
    for k in range(10):
        accel = (0.3, 0.1 * k, -0.2) if k != 3 else (7.0, -4.0, 1.0)
        out.append(None if k == 6 else
                   sample(accel, (0.5, 0.02 * k, -0.05), up=(0.94, 0.0, 0.342) if k >= 7 else None))
    return out


def test_the_recording_and_the_car_build_the_same_features():
    """The same IMU samples, once through what record writes and the dataset
    tool reads, once through PolicyDriver live: the same numbers, tick by
    tick. A drift here is a model running on input it was never shown."""
    samples = _samples()
    times = [T0 + 0.1 * k for k in range(len(samples))]

    # Record: the tap holds the newest sample and its time, the row goes out
    # as JSON, the adapter reads it back and the dataset builds the window.
    held, held_at, rows = None, 0.0, []
    for k, (s, now) in enumerate(zip(samples, times)):
        if s is not None:
            held, held_at = s, now
        r = row(k, throttle=0.3, **fp.imu_fields(held, held_at, now))
        rows.append((now - T0, json.loads(json.dumps(r))))
    ticks, _, _ = pd.align(footprint_session(rows), rules=None)
    recorded = np.stack([t.imu for t in ticks])

    # The car: the same samples off the driver's camera view at its ticks.
    model = ImuStub()
    cam = FakeCamera(depth=np.full((400, 640), 3000, np.uint16))
    d = pl.PolicyDriver(FakeController(), cam, model)
    for s, now in zip(samples, times):
        cam.imu = s
        d.tick(now)
    live = np.concatenate(model.fed)

    assert len(ticks) == len(samples) == len(live)
    np.testing.assert_allclose(live, recorded, rtol=0, atol=1e-6)
    # And the cases are really in there.
    assert recorded[3, F["accel_peak"]] > recorded[2, F["accel_peak"]]
    assert recorded[7:, F["tilt_valid"]].all() and not recorded[:7, F["tilt_valid"]].any()


def test_the_dataset_says_whether_its_sessions_had_tilt(tmp_path, monkeypatch):
    def session(up):
        extra = dict(imu_age_ms=2.0, imu_accel_x=0.1, imu_accel_y=0.0, imu_accel_z=0.0,
                     imu_gyro_x=0.3, imu_gyro_y=0.0, imu_gyro_z=0.0)
        if up:
            extra.update(imu_up_x=1.0, imu_up_y=0.0, imu_up_z=0.0)
        rows = [(k * 0.1, row(k, throttle=0.4, forward_limit=pa.RUNTIME_CAPS[0],
                              reverse_limit=pa.RUNTIME_CAPS[1], **extra)) for k in range(20)]
        return footprint_session(rows)._replace(name="s")

    for up in (False, True):
        monkeypatch.setattr(pd, "read_session", lambda path, adapter=None: (session(up), False))
        out = tmp_path / f"d{up}.npz"
        assert pd.main([str(out), str(tmp_path / "s.mcap"), "--no-idle-rules"]) == 0
        a = pe.load(str(out))
        assert a["imu"].shape == (20, pimu.IMU_SIZE)
        assert a["meta"]["imu_features"] == list(pimu.FEATURES)
        assert a["meta"]["imu_tilt"] is up
        assert a["imu"][:, F["valid"]].all()
        assert bool(a["imu"][:, F["tilt_valid"]].any()) is up


def test_eval_feeds_a_model_without_tilt_no_tilt():
    model = ImuStub(tilt=False)
    feats = pimu.ImuFeatures().push(pimu.row_fields(sample(up=LEVEL), 0.0, 0.0), 0.0)
    data = {"frames": np.full((1, pl.INPUT_HEIGHT, pl.INPUT_WIDTH), 1500, np.uint16),
            "frame": np.zeros(1, np.int64), "prev": np.zeros((1, pl.PREV_SIZE), np.float32),
            "mode_in": np.zeros(1, np.int8), "imu": feats[None]}
    pe._run(model, data, 0)
    assert model.fed[0][0, F["tilt_valid"]] == 0.0
    assert model.fed[0][0, F["valid"]] == 1.0


def test_prev_and_imu_go_in_together():
    model = ImuStub(uses_prev=True)
    cam = FakeCamera(depth=np.full((400, 640), 3000, np.uint16), imu=sample())
    d = pl.PolicyDriver(FakeController(), cam, model)
    d.tick(T0)
    d.tick(T0 + 0.1)
    assert model.fed[1][0, F["valid"]] == 1.0
    np.testing.assert_allclose(model.prevs[1], pl.encode_prev(pl.FORWARD, 0.5, 0.25))


# --- models, old and new ----------------------------------------------------------

def test_a_model_from_before_the_imu_input_loads_and_runs_without_one():
    model = pl.PolicyModel(str(SMOKE))
    assert (model.uses_imu, model.imu_tilt) == (False, False)
    assert pl.IMU_INPUT not in {i.name for i in model.session.get_inputs()}
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=np.full((400, 640), 3000, np.uint16),
                                                     imu=sample()), model)
    d.tick(T0)
    assert d.last_decision.reason.startswith("model")
    assert d.last_imu is None


def test_an_imu_model_loads_and_is_fed_its_features():
    model = pl.PolicyModel(str(IMU_MODEL))
    assert model.uses_imu and model.uses_prev and model.has_mode and not model.imu_tilt
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=np.full((400, 640), 3000, np.uint16),
                                                     imu=sample(up=LEVEL)), model)
    d.tick(T0)
    assert d.last_decision.reason.startswith("model")
    assert d.last_imu.shape == (1, pimu.IMU_SIZE)
    assert d.last_imu[0, F["valid"]] == 1.0 and d.last_imu[0, F["tilt_valid"]] == 0.0


def _copy(tmp_path, onnx, meta):
    (tmp_path / "m.onnx").write_bytes(onnx.read_bytes())
    (tmp_path / "m.json").write_text(json.dumps(meta))
    return str(tmp_path / "m.onnx")


def test_a_sidecar_and_graph_that_disagree_about_the_imu_are_refused(tmp_path):
    imu_meta = json.loads(IMU_MODEL.with_suffix(".json").read_text())
    smoke_meta = json.loads(SMOKE.with_suffix(".json").read_text())

    no_claim = {k: v for k, v in imu_meta.items() if not k.startswith("imu") and k != "uses_imu"}
    with pytest.raises(ValueError, match="does not declare"):
        pl.PolicyModel(_copy(tmp_path, IMU_MODEL, no_claim))
    with pytest.raises(ValueError, match="no 'imu' input"):
        pl.PolicyModel(_copy(tmp_path, SMOKE, {**smoke_meta, "uses_imu": True,
                                               "imu_features": list(pimu.FEATURES)}))
    with pytest.raises(ValueError, match="imu_features"):
        pl.PolicyModel(_copy(tmp_path, IMU_MODEL, {**imu_meta,
                                                   "imu_features": list(pimu.FEATURES)[::-1]}))
    with pytest.raises(ValueError, match="imu_window"):
        pl.PolicyModel(_copy(tmp_path, IMU_MODEL, {**imu_meta, "imu_window": 7}))
