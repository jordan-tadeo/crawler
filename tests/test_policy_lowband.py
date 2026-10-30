"""The lower band: the bottom third of the depth frame, as a second profile.

column_profile.py draws it; a model whose sidecar says lowband gets it as
three more rows of x under the footprint band's. These hold that it reads
the rows it says and no others, that the flat-floor layout reads a flat
floor as nothing, that the car and the dataset draw the same numbers off
the same tick, that the mirror is a column flip of both bands, and that a
model from before it is fed exactly what it was.
"""

import json
import math
import pathlib

import numpy as np
import pytest

from crawler.autopilot import column_profile as cp
from crawler.autopilot import driver as dv
from crawler.autopilot import policy as pl
from crawler.autopilot import policy_imu as pimu
import policy_eval as pe
import policy_scorecard as sc
import retrain as rt

from conftest import FakeCamera, FakeController

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
LOWBAND_MODEL = FIXTURES / "policy-lowband.onnx"
H, W = pl.INPUT_HEIGHT, pl.INPUT_WIDTH
T0 = 100.0


def _loop(small, top, bottom, columns=cp.PROFILE_COLUMNS):
    band = small[top:bottom].astype(np.float32)
    edges = np.linspace(0, small.shape[1], columns + 1).round().astype(int)
    out = np.empty((3, columns), np.float32)
    for i in range(columns):
        cell = band[:, edges[i]:edges[i + 1]].ravel()
        valid = cell[cell > 0]
        if valid.size:
            out[0, i] = min(1.0, valid.min() / cp.PROFILE_SCALE_MM)
            out[1, i] = min(1.0, float(np.median(valid)) / cp.PROFILE_SCALE_MM)
        else:
            out[0, i] = out[1, i] = 1.0
        out[2, i] = valid.size / max(1, cell.size)
    return out


def floor_frame(pitch=0.0, h=H, w=W):
    """The depth a flat floor reads, every row below the horizon, 0 above."""
    z = cp.floor_depth_mm((h, w), pitch)
    col = np.where(np.isfinite(z) & (z < 60000), np.round(z), 0)
    return np.repeat(col[:, None], w, axis=1).astype(np.uint16)


# --- what it reads ---------------------------------------------------------

def test_the_band_is_the_bottom_third_of_the_frame():
    assert cp.lowband_rows((50, 80)) == (33, 50)
    assert cp.lowband_rows((400, 640)) == (267, 400)


def test_the_plain_band_is_the_per_column_loop_over_the_bottom_third():
    rng = np.random.default_rng(11)
    top, bottom = cp.lowband_rows((H, W))
    for k in range(30):
        small = rng.integers(0, 5000, (H, W)).astype(np.uint16)
        small[rng.random(small.shape) < rng.random()] = 0
        if k % 4 == 0:
            small[:, 50:60] = 0
        np.testing.assert_allclose(cp.lowband_profile(small, cp.LOWBAND_PLAIN),
                                   _loop(small, top, bottom), atol=1e-6)


def test_the_plain_band_reads_its_own_rows_and_not_the_footprints():
    small = np.full((H, W), 2400, np.uint16)
    top, _ = cp.lowband_rows(small.shape)
    small[:top] = 300
    band = cp.lowband_profile(small, cp.LOWBAND_PLAIN)
    assert np.allclose(band[0], 0.8) and np.allclose(band[1], 0.8) and np.allclose(band[2], 1.0)
    # And the footprint band does not move when only the lower rows do.
    other = small.copy()
    other[top + 5:] = 700
    f_top, f_bottom = cp.profile_rows(small.shape)
    assert f_bottom <= top + 5
    np.testing.assert_array_equal(cp.column_profile(other), cp.column_profile(small))
    assert not np.allclose(cp.lowband_profile(other, cp.LOWBAND_PLAIN), band)


@pytest.mark.parametrize("layout", cp.LOWBAND_LAYOUTS)
def test_a_blind_column_reads_nothing_there(layout):
    small = floor_frame()
    small[:, :5] = 0
    band = cp.lowband_profile(small, layout)
    blank = cp.lowband_blank(layout)
    assert band[:, 0].tolist() == blank[:, 0].tolist()
    assert band[2, -1] == 1.0


def test_a_flat_floor_reads_as_no_deviation_at_its_own_pitch():
    for pitch in (0.0, math.radians(4), math.radians(-3)):
        band = cp.lowband_profile(floor_frame(pitch), cp.LOWBAND_FLOOR, pitch)
        assert np.abs(band[:2]).max() < 0.01, pitch
        assert np.allclose(band[2], 1.0)


def test_the_floor_layout_needs_the_pitch_to_call_a_tilted_floor_flat():
    tilted = floor_frame(math.radians(5))
    level = cp.lowband_profile(tilted, cp.LOWBAND_FLOOR, 0.0)
    assert np.abs(level[1]).min() > 0.1


def test_a_rock_on_the_floor_reads_nearer_and_a_drop_reads_further():
    small = floor_frame()
    top, _ = cp.lowband_rows(small.shape)
    small[top:top + 6, 0:10] = 250        # something standing up, left
    small[top:, 70:80] = np.minimum(small[top:, 70:80].astype(int) * 2, 60000).astype(np.uint16)
    band = cp.lowband_profile(small, cp.LOWBAND_FLOOR)
    assert band[0, 0] < -0.5 and band[1, 0] < 0
    assert band[1, -1] > 0.5
    assert abs(band[1, 16]) < 0.01


def test_a_row_above_the_horizon_reads_nearer_than_the_floor_not_blind():
    small = np.full((H, W), 2000, np.uint16)
    pitch = -math.radians(30)            # nose up: every lower row sees the sky
    band = cp.lowband_profile(small, cp.LOWBAND_FLOOR, pitch)
    assert np.allclose(band[2], 1.0)
    assert np.allclose(band[0], -1.0)


def test_the_pitch_is_nose_down_positive_and_level_with_no_reading():
    assert cp.floor_pitch(-math.sin(math.radians(5))) == pytest.approx(math.radians(5))
    assert cp.floor_pitch(0.2, valid=False) == 0.0
    assert cp.floor_pitch(float("nan")) == 0.0
    assert cp.floor_pitch(None) == 0.0
    assert pl.lowband_pitch(cp.LOWBAND_FLOOR, None) == 0.0
    assert pl.lowband_pitch(cp.LOWBAND_PLAIN, (-1.0, 0.0, -0.1)) == 0.0
    assert pl.lowband_pitch(cp.LOWBAND_FLOOR, (-1.0, 0.0, -0.1)) > 0


@pytest.mark.parametrize("layout", cp.LOWBAND_LAYOUTS)
def test_the_mirror_image_is_both_bands_with_the_columns_reversed(layout):
    rng = np.random.default_rng(5)
    small = floor_frame()
    small[rng.random(small.shape) < 0.3] = 0
    small[30:, 3:9] = 300
    x = pl.encode_input(pl.PROFILE, small, layout, 0.07)
    m = pl.encode_input(pl.PROFILE, small[:, ::-1].copy(), layout, 0.07)
    # 80 pixels into 32 columns is not symmetric to the pixel, so the
    # check is on a frame whose columns are whole 5-pixel blocks: the trainer
    # and the scorecard both mirror x by its last axis.
    blocks = np.repeat(small[:, ::5], 5, axis=1)
    xb = pl.encode_input(pl.PROFILE, blocks, layout, 0.07)
    mb = pl.encode_input(pl.PROFILE, blocks[:, ::-1].copy(), layout, 0.07)
    assert x.shape == m.shape == (1, 6, cp.PROFILE_COLUMNS)
    np.testing.assert_allclose(mb, xb[..., ::-1], atol=1e-6)
    # And left is left: the thing at the frame's left is in the first columns.
    near = xb[0, 3]
    assert near[2:4].max() < near[8:-8].min()


# --- the input, old and new ----------------------------------------------

def test_a_model_without_a_lower_band_is_fed_the_footprint_band_alone():
    small = floor_frame()
    x = pl.encode_input(pl.PROFILE, small)
    assert x.shape == (1, cp.PROFILE_FEATURES, cp.PROFILE_COLUMNS)
    np.testing.assert_array_equal(x[0], cp.column_profile(small))


def test_with_one_it_is_six_rows_footprint_first():
    small = floor_frame()
    x = pl.encode_input(pl.PROFILE, small, cp.LOWBAND_PLAIN)
    np.testing.assert_array_equal(x[0, :3], cp.column_profile(small))
    np.testing.assert_array_equal(x[0, 3:], cp.lowband_profile(small, cp.LOWBAND_PLAIN))
    with pytest.raises(ValueError):
        pl.encode_input(pl.IMAGE, small, cp.LOWBAND_PLAIN)


def test_the_lowband_model_loads_and_the_band_changes_its_answer():
    model = pl.PolicyModel(str(LOWBAND_MODEL))
    assert model.lowband == cp.LOWBAND_PLAIN
    small = floor_frame()
    prev = np.zeros((1, pl.PREV_SIZE), np.float32)
    mode = pl.encode_mode(False)
    x = pl.encode_input(pl.PROFILE, small, model.lowband)
    blank = x.copy()
    blank[0, 3:] = cp.lowband_blank(model.lowband)
    a, b = model.run(x, prev, mode), model.run(blank, prev, mode)
    assert not np.allclose(a["logits"], b["logits"])


@pytest.mark.parametrize("name", ["policy-smoke", "policy-imu", "policy-per-gear",
                                  "policy-map", "policy-goal", "policy-chunk5"])
def test_every_model_from_before_loads_with_no_lower_band(name):
    model = pl.PolicyModel(str(FIXTURES / f"{name}.onnx"))
    assert model.lowband is None


def test_a_lowband_sidecar_this_car_cannot_draw_is_refused(tmp_path):
    meta = json.loads((FIXTURES / "policy-lowband.json").read_text())
    without = {k: v for k, v in meta.items() if k != "lowband"}
    for bad in (dict(meta, lowband="lower_half"), dict(meta, lowband_camera_height_mm=120.0),
                dict(meta, lowband_fy_at_400=400.0), dict(meta, lowband_fraction=0.5),
                without):
        (tmp_path / "m.onnx").write_bytes(LOWBAND_MODEL.read_bytes())
        (tmp_path / "m.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            pl.PolicyModel(str(tmp_path / "m.onnx"))
    # And an old graph with a sidecar that claims a lower band.
    old = json.loads((FIXTURES / "policy-imu.json").read_text())
    (tmp_path / "o.onnx").write_bytes((FIXTURES / "policy-imu.onnx").read_bytes())
    (tmp_path / "o.json").write_text(json.dumps(dict(old, **pl.lowband_meta(cp.LOWBAND_PLAIN))))
    with pytest.raises(ValueError):
        pl.PolicyModel(str(tmp_path / "o.onnx"))


# --- the car and the dataset draw the same thing ----------------------------

class Recorder:
    contract = pl.CONTRACT_DEPTH_SMALL
    representation = pl.PROFILE
    uses_prev = False

    def __init__(self, lowband=None):
        self.lowband = lowband
        self.fed = []

    def run(self, x, prev):
        self.fed.append(x.copy())
        return {"logits": np.array([[5.0, 0.0, 0.0]], np.float32),
                "throttle": np.array([[0.5]], np.float32),
                "steer": np.array([[0.0]], np.float32)}


def raw_floor(pitch):
    """A raw 400x640 frame of a flat floor (and far wall above it)."""
    z = cp.floor_depth_mm((400, 640), pitch)
    col = np.where(np.isfinite(z) & (z < 3000), np.round(z), 3000).astype(np.uint16)
    return np.repeat(col[:, None], 640, axis=1)


TIP = math.radians(4)
IMU = {"accel": (0.0, 0.0, 9.8), "gyro": (0.0, 0.0, 0.0),
       "up": (-math.cos(TIP), 0.0, -math.sin(TIP))}


@pytest.mark.parametrize("layout", [None, *cp.LOWBAND_LAYOUTS])
def test_the_car_and_the_dataset_draw_the_same_x_off_the_same_tick(layout):
    raw = raw_floor(TIP)
    model = Recorder(layout)
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=raw, imu=IMU), model)
    d.tick(T0)
    live = model.fed[-1]
    # The dataset's side: the frame through the recording's downsample, the
    # tick's imu array off the row through ImuFeatures with the tilt on.
    row = pimu.row_fields(IMU, T0, T0)
    data = {"frames": pl.depth_small_function()(raw)[None], "frame": np.array([0]),
            "imu": pimu.ImuFeatures(tilt=True).push(dict(row), T0)[None]}
    offline = pe.x_input(model, data, 0)
    np.testing.assert_array_equal(live, offline)
    assert live.shape[1] == (3 if layout is None else 6)
    if layout == cp.LOWBAND_FLOOR:
        assert d.last_lowband_pitch == pytest.approx(TIP, abs=1e-6)
        # At its pitch, the floor in front of it is a flat floor.
        assert np.abs(live[0, 3:5]).max() < 0.05


def test_a_model_without_a_lower_band_never_reads_the_pitch(monkeypatch):
    called = []
    monkeypatch.setattr(pl, "lowband_pitch", lambda *a: called.append(a) or 0.0)
    for layout in (None, cp.LOWBAND_PLAIN):
        d = pl.PolicyDriver(FakeController(), FakeCamera(depth=raw_floor(0.0), imu=IMU),
                            Recorder(layout))
        d.tick(T0)
    assert called == []


def test_the_up_of_a_row_is_what_imu_features_keeps():
    for sample in (IMU, {"accel": (0, 0, 9.8), "gyro": (0, 0, 0)},
                   {"accel": (0, 0, 9.8), "gyro": (0, 0, 0), "up": (0.0, 0.0, 0.5)}):
        row = pimu.row_fields(sample, T0, T0)
        f = pimu.ImuFeatures(tilt=True).push(dict(row), T0)
        up = pimu.up_of(row)
        assert (up is not None) == bool(f[12])
        if up is not None:
            np.testing.assert_allclose(f[9:12], up, atol=1e-6)
    stale = pimu.row_fields(IMU, T0 - 1.0, T0)
    assert pimu.up_of(stale) is None


# --- the trainer's cache and the scorecard ---------------------------------

def test_the_x_cache_is_named_for_the_band_it_was_drawn_with():
    assert rt.input_key(pl.PROFILE) == "profile"
    keys = {rt.input_key(pl.PROFILE, b) for b in cp.LOWBAND_LAYOUTS}
    assert len(keys) == 2 and "profile" not in keys
    for b in (None, *cp.LOWBAND_LAYOUTS):
        assert rt.parse_input_key(rt.input_key(pl.PROFILE, b)) == (pl.PROFILE, b)


def test_the_scorecard_blanks_only_the_lower_band():
    model = pl.PolicyModel(str(LOWBAND_MODEL))
    data = {"frames": floor_frame()[None], "frame": np.array([0]),
            "mode_in": np.array([0])}
    x, *_ = sc.inputs(model, data, 0)
    xb, *_ = sc.inputs(model, data, 0, blank_lowband=True)
    np.testing.assert_array_equal(x[0, :3], xb[0, :3])
    np.testing.assert_array_equal(xb[0, 3:], cp.lowband_blank(model.lowband))
    assert not np.array_equal(x[0, 3:], xb[0, 3:])
    assert "lowband_use" in sc.KEYS
