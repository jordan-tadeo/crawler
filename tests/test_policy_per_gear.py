"""The per-gear steering head: one regression for forward, one for reverse.

I swap lock on 95% of my reversals, so a single steer had to flip its sign on
exactly the tick the gear flipped. The per-gear model has a head for each
gear, each trained only on its own gear's ticks (a stop on the gear that
comes next), and the car steers with the head of the gear it actually
commands, after every rule that changes the gear. These hold the decode,
the chunk blend and the driver to that, every model from before it to
driving exactly as it did, and the trainer's gear labels. No torch here:
the per-gear model is an untrained fixture run through onnxruntime.
"""

import json
import math
import pathlib

import numpy as np
import pytest

from crawler.autopilot import driver as dv
from crawler.autopilot import policy as pl
import models as md
import policy_train as pt

from conftest import FakeCamera, FakeController

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
PER_GEAR = FIXTURES / "policy-per-gear.onnx"   # untrained, profile + mode, chunk 5
CHUNK = FIXTURES / "policy-chunk5.onnx"        # one regression steer, chunk 5
SMOKE = FIXTURES / "policy-smoke.onnx"
T0 = 100.0
FWD = (5.0, 0.0, 0.0)
REV = (0.0, 5.0, 0.0)
STOP_THEN_REV = (0.0, 1.0, 5.0)
STOP_THEN_FWD = (1.0, 0.0, 5.0)


def geared(logits=FWD, fwd=0.3, rev=-0.6, throttle=0.5):
    return {"logits": np.array([logits], np.float32),
            "throttle": np.array([[throttle]], np.float32),
            "steer_forward": np.array([np.atleast_1d(fwd)], np.float32),
            "steer_reverse": np.array([np.atleast_1d(rev)], np.float32)}


def clear(mm=3000):
    return np.full((400, 640), mm, dtype=np.uint16)


# --- the decode -------------------------------------------------------------------

@pytest.mark.parametrize("logits,want", [(FWD, 0.3), (REV, -0.6),
                                         (STOP_THEN_REV, -0.6), (STOP_THEN_FWD, 0.3)],
                         ids=["forward", "reverse", "stop before a reverse",
                              "stop before a forward"])
def test_the_steer_is_the_head_of_the_gear_and_a_stop_takes_the_next_best(logits, want):
    out = geared(logits)
    assert pl.steer_head_of(out) == pl.STEER_PER_GEAR
    steer, p, why = pl.decode_steer(out)
    assert why == "" and p is None and steer == pytest.approx(want)
    decided, _ = pl.read_output(out)
    assert decided[0] == pl.ACTIONS[int(np.argmax(logits))]
    assert decided[2] == pytest.approx(want)


def test_the_decode_takes_the_action_it_is_told_over_the_logits():
    out = geared(FWD)
    assert pl.decode_steer(out, pl.REVERSE)[0] == pytest.approx(-0.6)
    assert pl.decode_steer(out, pl.FORWARD)[0] == pytest.approx(0.3)
    assert pl.gear_steer(out, pl.REVERSE) == pytest.approx(-0.6)
    # Every other head has its one steer, whatever the gear.
    assert pl.gear_steer({"logits": out["logits"], "steer": np.array([[0.1]])},
                         pl.REVERSE) is None


BAD = [
    ("nan in the head not used", dict(rev=math.nan), "not finite"),
    ("inf forward", dict(fwd=math.inf), "not finite"),
    ("over full lock", dict(rev=1.2), "steer out of range"),
    ("two wide", dict(fwd=(0.1, 0.2)), "shape"),
]


@pytest.mark.parametrize("name,kw,why", BAD, ids=[b[0] for b in BAD])
def test_a_broken_per_gear_answer_is_refused_whichever_head_is_broken(name, kw, why):
    out = geared(FWD, **kw)
    assert pl.decode_steer(out)[0] is None
    assert pl.read_output(out) == (None, why)


def test_one_head_alone_or_a_per_gear_head_beside_a_regression_is_missing():
    one = geared(FWD)
    one.pop("steer_reverse")
    assert pl.decode_steer(one) == (None, None, "missing")
    both = {**geared(FWD), "steer": np.array([[0.1]], np.float32)}
    assert pl.steer_head_of(both) is None and pl.read_output(both) == (None, "missing")


# --- the chunk blend ------------------------------------------------------------------

def test_each_gears_steer_is_blended_on_its_own_offset_and_the_first_is_offset_0():
    k, m = 3, -0.5
    ens = pl.ChunkEnsembler(k, m)
    a = geared(FWD * k, fwd=(0.1, 0.2, 0.3), rev=(-0.1, -0.2, -0.3))
    b = geared(FWD * k, fwd=(0.5, 0.6, 0.7), rev=(-0.5, -0.6, -0.7))
    first = pl.first_of_chunk(a)
    assert first["steer_forward"].tolist() == pytest.approx([0.1])
    assert first["steer_reverse"].tolist() == pytest.approx([-0.1])
    ens.apply(a, T0)
    out = ens.apply(b, T0 + 0.1)
    w0, w1 = 1.0, math.exp(m)
    # b is the newest (age 0, its offset 0); a is age 1, its offset 1.
    assert out["steer_forward"][0] == pytest.approx((w0 * 0.5 + w1 * 0.2) / (w0 + w1))
    assert out["steer_reverse"][0] == pytest.approx((w0 * -0.5 + w1 * -0.2) / (w0 + w1))
    assert pl.read_output(out)[0][2] == pytest.approx(out["steer_forward"][0])


def test_a_nan_on_a_later_offset_of_either_head_is_never_kept():
    ens = pl.ChunkEnsembler(3, 0.0)
    bad = geared(FWD * 3, fwd=(0.1, 0.2, 0.3), rev=(-0.1, math.nan, -0.3))
    out = ens.apply(bad, T0)
    assert ens.depth == 0
    assert pl.read_output(out) == (None, "not finite")


# --- the driver -----------------------------------------------------------------------

class Stub:
    """A per-gear model that says what it is told."""

    contract = pl.CONTRACT_DEPTH_SMALL
    representation = pl.PROFILE
    has_mode = False
    uses_prev = False
    uses_imu = False
    uses_lidar = False
    uses_map = False

    def __init__(self, logits=FWD, fwd=0.3, rev=-0.6, throttle=0.8):
        self.say(logits, fwd, rev)
        self.throttle = throttle

    def say(self, logits, fwd=None, rev=None):
        self.logits = logits
        if fwd is not None:
            self.fwd = fwd
        if rev is not None:
            self.rev = rev

    def run(self, x, prev, mode=None, imu=None, lidar=None, map=None):
        return geared(self.logits, self.fwd, self.rev, self.throttle)


class MapStub(Stub):
    uses_lidar = True
    uses_map = True


def map_driver(model, rear_mm=None):
    from test_policy_map import FakeSlam, reader
    return pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()), model,
                           lidar=reader(rear_mm), slam=FakeSlam())


def test_a_map_model_steers_each_gear_with_its_own_head():
    model = MapStub(FWD)
    d = map_driver(model)
    d.tick(T0)
    assert d.controller.throttles[-1] > 0
    assert d.controller.steerings[-1][0] == pytest.approx(0.3 * pl.STEER_SCALE)
    model.say(REV)
    d.tick(T0 + 0.1)
    assert d.controller.throttles[-1] < 0
    assert d.controller.steerings[-1][0] == pytest.approx(-0.6 * pl.STEER_SCALE)
    # A stop between the two turns the wheels for the gear that comes next.
    model.say(STOP_THEN_FWD)
    d.tick(T0 + 0.2)
    assert d.controller.throttles[-1] == 0
    assert d.controller.steerings[-1][0] == pytest.approx(0.3 * pl.STEER_SCALE)


def test_a_reverse_the_rear_stop_turns_forward_steers_with_the_forward_head():
    near = int(dv.REAR_SELF_MM + dv.REAR_STOP_CLEAR_MM - 40)
    d = map_driver(MapStub((1.0, 5.0, 0.0)), rear_mm=near)
    d.tick(T0)
    dec = d.last_decision
    assert dec.action == pl.FORWARD and dec.reason == "model rear masked"
    assert d.controller.throttles[-1] > 0
    assert d.controller.steerings[-1][0] == pytest.approx(0.3 * pl.STEER_SCALE)


def test_the_hand_rules_refractory_forward_steers_with_the_forward_head():
    model = Stub(REV)
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()), model)
    assert d.hand_rules
    d.tick(T0)
    assert d.reverse_commit
    t = T0 + 0.1
    while d.reverse_commit:
        model.say(FWD)
        d.tick(t)
        t += 0.1
    # Straight back into the reverse it just came out of: the refractory
    # sends it forward, and forward is the forward head's.
    model.say(REV)
    d.tick(t)
    assert d.reverse_refractory
    assert d.controller.throttles[-1] > 0
    assert d.controller.steerings[-1][0] == pytest.approx(0.3 * pl.STEER_SCALE)


def test_the_fixture_loads_and_drives_on_the_head_of_the_gear_it_commands():
    model = pl.PolicyModel(str(PER_GEAR))
    assert model.steer_head == pl.STEER_PER_GEAR and model.chunk == 5
    assert set(pl.GEAR_STEER_OUTPUTS) <= set(model.output_names)
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()), model)
    d._prev = (pl.FORWARD, 1.0, 0.0)
    d.tick(T0)
    assert d.last_decision.reason == "model"
    out = model.run(pl.encode_input(model.representation, d.downsample(clear())),
                    np.zeros((1, pl.PREV_SIZE), np.float32), pl.encode_mode(False))
    action = d.last_decision.action
    assert action in (pl.FORWARD, pl.REVERSE)
    # Off the raw graph, not through the decode under test: offset 0 of the
    # head named for the gear.
    name = "steer_forward" if action == pl.FORWARD else "steer_reverse"
    other = "steer_reverse" if action == pl.FORWARD else "steer_forward"
    want = float(np.asarray(out[name]).ravel()[0])
    assert abs(want - float(np.asarray(out[other]).ravel()[0])) > 1e-3
    assert d.controller.steerings[-1][0] == pytest.approx(want * pl.STEER_SCALE, abs=1e-6)


def test_old_single_steer_models_still_load_and_drive_on_their_one_steer():
    for path in (SMOKE, CHUNK):
        model = pl.PolicyModel(str(path))
        assert model.steer_head == pl.STEER_REGRESSION
        d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()), model)
        d._prev = (pl.FORWARD, 1.0, 0.0)
        d.tick(T0)
        assert d.last_decision.reason.startswith("model"), path.name
        args = [pl.encode_input(model.representation, d.downsample(clear())),
                np.zeros((1, pl.PREV_SIZE), np.float32)]
        if model.has_mode:
            args.append(pl.encode_mode(False))
        out = model.run(*args)
        if d.last_decision.action == pl.FORWARD:
            steer = float(np.asarray(out["steer"]).ravel()[0])
            assert d.controller.steerings[-1][0] == pytest.approx(
                steer * pl.STEER_SCALE, abs=1e-6), path.name


# --- the sidecar ----------------------------------------------------------------------

def test_a_per_gear_sidecar_needs_both_heads_in_the_graph(tmp_path):
    meta = pt.steer_meta(pl.STEER_PER_GEAR)
    assert meta["outputs"] == pt.output_names(pl.STEER_PER_GEAR)
    assert pl.check_steer_head(meta, meta["outputs"]) == pl.STEER_PER_GEAR
    with pytest.raises(ValueError, match="steer_"):
        pl.check_steer_head(meta, [n for n in meta["outputs"] if n != "steer_reverse"])
    with pytest.raises(ValueError, match="steer_head"):
        pl.check_steer_head(meta, pt.output_names(pl.STEER_REGRESSION))
    # The per-gear sidecar on a one-steer graph will not load.
    side = {**json.loads(CHUNK.with_suffix(".json").read_text()), **meta}
    (tmp_path / "m.json").write_text(json.dumps(side))
    (tmp_path / "m.onnx").write_bytes(CHUNK.read_bytes())
    with pytest.raises(ValueError):
        pl.PolicyModel(str(tmp_path / "m.onnx"))


def test_a_chunk_sidecar_wider_than_the_gear_heads_is_refused(tmp_path):
    side = json.loads(PER_GEAR.with_suffix(".json").read_text())
    side["chunk"] = 3
    (tmp_path / "m.json").write_text(json.dumps(side))
    (tmp_path / "m.onnx").write_bytes(PER_GEAR.read_bytes())
    with pytest.raises(ValueError):
        pl.PolicyModel(str(tmp_path / "m.onnx"))


def test_the_per_gear_head_is_a_major_contract_change():
    old = {"input_contract": "depth_small", "representation": "profile",
           "actions": list(pl.ACTIONS), "outputs": pt.OUTPUTS, "steer_scale": 0.7,
           "steer_modes": list(pl.STEER_MODES)}
    reg = md.empty_registry()
    reg["releases"].append({"name": "kblock", "version": "1.2",
                            "contract": md.contract_of(old)})
    v, why = md.next_version(reg, "kblock", md.contract_of(
        {**old, **pt.steer_meta(pl.STEER_PER_GEAR)}))
    assert v == "2.0" and "steer_head" in why


# --- the trainer's gear labels --------------------------------------------------------

F, R, S = (pl.ACTIONS.index(a) for a in (pl.FORWARD, pl.REVERSE, pl.STOP))


def gear_data(actions, t_s, sessions=None):
    return {"action": np.array(actions), "t_ns": (np.array(t_s) * 1e9).astype(np.int64),
            "session": np.array(sessions if sessions is not None else [0] * len(actions))}


def test_a_moving_tick_trains_its_own_gears_head():
    g = pt.steer_gears(gear_data([F, R, F, R], [0.0, 0.1, 0.2, 0.3]))
    assert g.tolist() == [pt.GEAR_FORWARD, pt.GEAR_REVERSE, pt.GEAR_FORWARD, pt.GEAR_REVERSE]


def test_a_stop_trains_the_head_of_the_gear_that_comes_next():
    # Forward, a stop, then a reverse 1.5 s on: the stop is where I put on
    # the reverse lock, so it trains the reverse head.
    g = pt.steer_gears(gear_data([F, S, S, R], [0.0, 0.1, 1.5, 1.6]))
    assert g.tolist()[1:3] == [pt.GEAR_REVERSE, pt.GEAR_REVERSE]


def test_a_stop_with_nothing_soon_after_takes_the_gear_before_or_forward():
    g = pt.steer_gears(gear_data([R, S, F], [0.0, 1.0, 9.0]))
    assert g[1] == pt.GEAR_REVERSE, "nothing next within the window: the gear before"
    g = pt.steer_gears(gear_data([R, S, F], [0.0, 5.0, 9.0]))
    assert g[1] == pt.GEAR_FORWARD, "nothing either side within the window: forward"
    # The window is the one the sidecar records.
    g = pt.steer_gears(gear_data([F, S, R], [0.0, 1.0, 3.5]), stop_gear_s=3.0)
    assert g[1] == pt.GEAR_REVERSE


def test_the_gear_labels_never_reach_across_a_session_or_out_of_time_order():
    # Rows out of order and two sessions: session 1's stop is followed in
    # time by its own forward, never by session 0's reverse.
    g = pt.steer_gears(gear_data([S, R, F, S], [0.5, 0.6, 0.9, 0.4], [1, 0, 1, 0]))
    assert g[0] == pt.GEAR_FORWARD
    assert g[3] == pt.GEAR_REVERSE
    # Session 0 ends on a stop after a reverse, and session 1 starts on a
    # stop long before its forward: neither takes the other's gear.
    g = pt.steer_gears(gear_data([R, S, F], [0.0, 0.1, 0.15], [0, 0, 1]))
    assert g[1] == pt.GEAR_REVERSE, "session 0's own reverse, not session 1's forward"
    g = pt.steer_gears(gear_data([R, S, F], [0.0, 0.0, 5.0], [0, 1, 1]))
    assert g[1] == pt.GEAR_FORWARD, "nothing of its own within the window"
