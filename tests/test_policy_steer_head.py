"""The steering head: a regression, or a direction class and a magnitude.

A single tanh steer averages a hard left and a hard right into straight,
which is how a model ends up backing out of a 3-point turn with the wheels
straight. The direction head answers which way first and how far second.
Both are read by one function, policy.decode_steer, on the car and in the
eval, and every model from before the direction head has to load and drive
exactly as it did. No torch here: the direction model is an untrained
fixture run through onnxruntime, as the car runs it.
"""

import json
import math
import pathlib

import numpy as np
import pytest

from crawler.autopilot import footprint as fp
from crawler.autopilot import policy as pl
from crawler.telemetry import telemetry as tm
import models as md
import policy_train as pt

from conftest import FakeCamera, FakeController

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
SMOKE = FIXTURES / "policy-smoke.onnx"           # a regression head, from before steer_head
STEER_DIR = FIXTURES / "policy-steer-dir.onnx"   # untrained, profile + mode, direction head
T0 = 100.0
L, S, R = (pl.STEER_DIRS.index(d) for d in (pl.LEFT, pl.STRAIGHT, pl.RIGHT))


def outputs(steer=None, dir_logits=None, mag=None, logits=(5.0, 0.0, 0.0), throttle=1.0):
    out = {"logits": np.array([logits], np.float32),
           "throttle": np.array([[throttle]], np.float32)}
    if steer is not None:
        out["steer"] = np.array([[steer]], np.float32)
    if dir_logits is not None:
        out["steer_dir"] = np.array([dir_logits], np.float32)
    if mag is not None:
        out["steer_mag"] = np.array([[mag]], np.float32)
    return out


def clear():
    return np.full((400, 640), 3000, dtype=np.uint16)


# --- decode, both heads ----------------------------------------------------------

def test_a_regression_head_decodes_as_its_one_steer():
    steer, p, why = pl.decode_steer(outputs(steer=-0.4))
    assert steer == pytest.approx(-0.4) and p is None and why == ""
    assert pl.read_output(outputs(steer=-0.4))[0][2] == pytest.approx(-0.4)
    assert pl.steer_dir_p(outputs(steer=-0.4)) == pl.NO_STEER_DIR_P


@pytest.mark.parametrize("dir_logits,want", [((3.0, 0.0, 0.0), -0.8),
                                             ((0.0, 3.0, 0.0), 0.0),
                                             ((0.0, 0.0, 3.0), 0.8)],
                         ids=["left", "straight", "right"])
def test_a_direction_head_is_its_sides_sign_times_the_magnitude(dir_logits, want):
    """Straight is 0 whatever the magnitude says: the magnitude is only
    ever trained on turns."""
    steer, p, why = pl.decode_steer(outputs(dir_logits=dir_logits, mag=0.8))
    assert why == "" and steer == pytest.approx(want)
    assert len(p) == 3 and sum(p) == pytest.approx(1.0)
    assert int(np.argmax(p)) == int(np.argmax(dir_logits))
    decided, _ = pl.read_output(outputs(dir_logits=dir_logits, mag=0.8))
    assert decided[2] == pytest.approx(want)


def test_two_hard_turns_the_regression_would_average_stay_a_hard_turn():
    """The point of the head: 51% left at full lock is full left, not 0.02."""
    steer, p, _ = pl.decode_steer(outputs(dir_logits=(0.04, -5.0, 0.0), mag=1.0))
    assert steer == pytest.approx(-1.0)
    assert p[L] > p[R] > p[S]


BAD = [
    ("nan class", dict(dir_logits=(math.nan, 0.0, 0.0), mag=0.5), "not finite"),
    ("inf magnitude", dict(dir_logits=(1.0, 0.0, 0.0), mag=math.inf), "not finite"),
    ("magnitude over 1", dict(dir_logits=(1.0, 0.0, 0.0), mag=1.2), "steer out of range"),
    ("negative magnitude", dict(dir_logits=(1.0, 0.0, 0.0), mag=-0.1), "steer out of range"),
    ("no magnitude", dict(dir_logits=(1.0, 0.0, 0.0)), "missing"),
    ("no class", dict(mag=0.5), "missing"),
    ("two classes", dict(dir_logits=(1.0, 0.0), mag=0.5), "shape"),
    ("both heads", dict(steer=0.1, dir_logits=(1.0, 0.0, 0.0), mag=0.5), "missing"),
    ("no steering at all", dict(), "missing"),
]


@pytest.mark.parametrize("name,kwargs,why", BAD, ids=[b[0] for b in BAD])
def test_a_broken_direction_head_is_refused_and_drives_neutral(name, kwargs, why):
    out = outputs(**kwargs)
    assert pl.decode_steer(out)[0] is None
    decided, got = pl.read_output(out)
    assert decided is None and got == why

    class Stub:
        contract, representation, uses_prev = pl.CONTRACT_LEGACY_JET, pl.PROFILE, False

        def run(self, x, prev):
            return out

    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()), Stub())
    d.tick(T0)
    assert d.controller.throttles[-1] == 0 and d.controller.steerings[-1] == (0, 0)
    assert d.policy_fields()["policy_steer_dir_p"] == list(pl.NO_STEER_DIR_P)


# --- the flip --------------------------------------------------------------------

def test_the_flip_swaps_left_and_right_and_leaves_straight():
    steer = np.array([-0.9, -0.3, -0.05, 0.0, 0.05, 0.3, 0.9], np.float32)
    cls = pt.steer_classes(steer)
    assert cls.tolist() == [L, L, S, S, S, R, R]
    flipped, fcls = pt.flip_steer_labels(steer, cls)
    np.testing.assert_array_equal(flipped, -steer)
    assert fcls.tolist() == [R, R, S, S, S, L, L]
    # The flipped classes are the classes of the flipped steer, band and all.
    assert fcls.tolist() == pt.steer_classes(flipped).tolist()


def test_the_straight_band_edge_is_straight():
    band = pt.STRAIGHT_BAND
    assert pt.steer_classes([-band, band]).tolist() == [S, S]
    assert pt.steer_classes([-band - 1e-3, band + 1e-3]).tolist() == [L, R]


# --- the sidecar -----------------------------------------------------------------

def test_the_sidecar_round_trips_and_names_the_head():
    for head in pl.STEER_HEADS:
        meta = json.loads(json.dumps(pt.steer_meta(head)))
        assert meta["steer_head"] == head and meta["outputs"] == pt.output_names(head)
        assert pl.check_steer_head(meta, meta["outputs"]) == head
    direction = pt.steer_meta(pl.STEER_DIRECTION)
    assert direction["steer_dirs"] == list(pl.STEER_DIRS)
    assert direction["steer_straight_band"] == pt.STRAIGHT_BAND
    assert pl.STEER_DIR_OUTPUT in direction["outputs"] and "steer" not in direction["outputs"]
    assert pt.OUTPUTS == pt.output_names(pl.STEER_REGRESSION)


def test_a_sidecar_and_graph_that_disagree_about_the_head_are_refused():
    reg, dirn = pt.output_names(pl.STEER_REGRESSION), pt.output_names(pl.STEER_DIRECTION)
    # Before steer_head: a regression, and only a graph with `steer` is one.
    assert pl.check_steer_head({}, reg) == pl.STEER_REGRESSION
    for meta, outs in (({}, dirn),
                       (pt.steer_meta(pl.STEER_DIRECTION), reg),
                       (pt.steer_meta(pl.STEER_REGRESSION), dirn),
                       ({"steer_head": "bins"}, reg),
                       ({**pt.steer_meta(pl.STEER_DIRECTION),
                         "steer_dirs": ["right", "straight", "left"]}, dirn)):
        with pytest.raises(ValueError, match="steer_"):
            pl.check_steer_head(meta, outs)


def test_the_steering_head_is_a_contract_key_and_changing_it_is_a_major():
    old = {"input_contract": "depth_small", "representation": "profile",
           "actions": list(pl.ACTIONS), "outputs": pt.OUTPUTS, "steer_scale": 0.7,
           "steer_modes": list(pl.STEER_MODES)}
    reg = md.empty_registry()
    reg["releases"].append({"name": "kblock", "version": "1.2",
                            "contract": md.contract_of(old)})
    # A sidecar from before the key is a regression, and not a change.
    assert md.contract_of(old)["steer_head"] == pl.STEER_REGRESSION
    assert md.next_version(reg, "kblock", md.contract_of(
        {**old, **pt.steer_meta(pl.STEER_REGRESSION)}))[0] == "1.3"
    v, why = md.next_version(reg, "kblock", md.contract_of(
        {**old, **pt.steer_meta(pl.STEER_DIRECTION)}))
    assert v == "2.0" and "steer_head" in why


# --- the real graphs, through onnxruntime ------------------------------------------

def test_an_old_model_loads_as_a_regression_and_drives_as_it_did():
    meta = json.loads(SMOKE.with_suffix(".json").read_text())
    assert "steer_head" not in meta
    model = pl.PolicyModel(str(SMOKE))
    assert model.steer_head == pl.STEER_REGRESSION
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()), model)
    d.tick(T0)
    assert d.last_decision.reason.startswith("model")
    # What the wheels got is the graph's own `steer`, untouched by the new decode.
    out = model.run(pl.encode_input(model.representation, d.downsample(clear())),
                    np.zeros((1, pl.PREV_SIZE), np.float32))
    steer = float(np.asarray(out["steer"]).ravel()[0])
    assert d.controller.steerings[-1][0] == pytest.approx(steer * pl.STEER_SCALE, abs=1e-6)
    assert d.policy_fields()["policy_steer_dir_p"] == list(pl.NO_STEER_DIR_P)


def test_a_direction_model_loads_drives_and_logs_its_probabilities():
    model = pl.PolicyModel(str(STEER_DIR))
    assert model.steer_head == pl.STEER_DIRECTION
    assert {pl.STEER_DIR_OUTPUT, pl.STEER_MAG_OUTPUT} <= set(model.output_names)
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()), model)
    # Already reversing, so a reverse answer is not a fresh one the commit
    # would hold at its own steer: the wheels get the decoded steer.
    d._prev = (pl.REVERSE, 1.0, 0.0)
    d.tick(T0)
    assert d.last_decision.reason == "model"
    out = model.run(pl.encode_input(model.representation, d.downsample(clear())),
                    np.zeros((1, pl.PREV_SIZE), np.float32), pl.encode_mode(False))
    steer, p, _ = pl.decode_steer(out)
    # Off the raw outputs, not through the decode under test.
    k = int(np.argmax(np.asarray(out["steer_dir"]).ravel()))
    want = (0.0 if k == S else (-1.0 if k == L else 1.0)) * float(np.asarray(out["steer_mag"]).ravel()[0])
    assert d.controller.steerings[-1][0] == pytest.approx(want * pl.STEER_SCALE, abs=1e-6)
    fields = d.policy_fields()
    assert fields["policy_steer_dir_p"] == pytest.approx(list(p), abs=1e-6)
    assert sum(fields["policy_steer_dir_p"]) == pytest.approx(1.0, abs=1e-5)


def test_a_direction_sidecar_on_a_regression_graph_will_not_load(tmp_path):
    meta = {**json.loads(SMOKE.with_suffix(".json").read_text()),
            **pt.steer_meta(pl.STEER_DIRECTION)}
    (tmp_path / "m.json").write_text(json.dumps(meta))
    (tmp_path / "m.onnx").write_bytes(SMOKE.read_bytes())
    with pytest.raises(ValueError, match="steer_head"):
        pl.PolicyModel(str(tmp_path / "m.onnx"))


# --- the logged field ----------------------------------------------------------------

def test_the_steering_probabilities_are_a_footprint_field_in_the_schema():
    props = tm.JSON_SCHEMAS["footprint"]["schema"]["properties"]
    assert props["policy_steer_dir_p"]["type"] == "array"
    assert props["policy_steer_dir_p"]["minItems"] == props["policy_steer_dir_p"]["maxItems"] == 3
    assert fp.AUTOPILOT_FIELDS["policy_steer_dir_p"] == [0.0, 0.0, 0.0]
    assert fp.autopilot_fields(None)["policy_steer_dir_p"] == [0.0, 0.0, 0.0]


def test_a_row_carries_the_probabilities_three_wide_whatever_the_driver_hands_it():
    class Driver:
        def __init__(self, p):
            self.p = p

        def policy_fields(self):
            return {"autopilot": pl.AUTOPILOT_POLICY, "policy_steer_dir_p": self.p,
                    "policy_infer_ms": 2.0}

    assert fp.autopilot_fields(Driver((0.7, 0.2, 0.1)))["policy_steer_dir_p"] == \
        pytest.approx([0.7, 0.2, 0.1])
    for bad in ((0.5, 0.5), (math.nan, 0.5, 0.5)):
        row = fp.autopilot_fields(Driver(bad))
        assert len(row["policy_steer_dir_p"]) == 3 and -1.0 in row["policy_steer_dir_p"]
        # The rest of the row is still the driver's.
        assert row["policy_infer_ms"] == 2.0 and row["autopilot"] == pl.AUTOPILOT_POLICY
