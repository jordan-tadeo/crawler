"""Action chunks: a model that answers for the next K decisions at once, and
the car blending its last K answers (policy.ChunkEnsembler).

These hold the blend to ACT's temporal ensemble with the trainer's weights
(exp(m * age), age 0 the newest answer), K = 1 to exactly the one-step model
it always was, the chain to starting over wherever the answers stop being
one drive's run, and the sidecar to refusing a chunk the graph does not have.
"""

import json
import math
import pathlib

import numpy as np
import pytest

from crawler.autopilot import policy as pl
import policy_train as pt

from conftest import FakeCamera, FakeController

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CHUNK_MODEL = FIXTURES / "policy-chunk5.onnx"   # untrained, profile + mode, chunk 5
SMOKE_MODEL = FIXTURES / "policy-smoke.onnx"
T0 = 100.0
FWD = (5.0, 0.0, 0.0)
REV = (0.0, 5.0, 0.0)


def chunked(gears, steers, throttle=0.5):
    """Raw chunk outputs: one gear logit triple and one steer per offset."""
    return {"logits": np.array([np.concatenate(gears)], np.float32),
            "throttle": np.array([[throttle]], np.float32),
            "steer": np.array([steers], np.float32)}


def same(gear, steer, k=5):
    return chunked([gear] * k, [steer] * k)


def clear(mm=3000):
    return np.full((400, 640), mm, dtype=np.uint16)


class ChunkStub:
    """A chunk model that says whatever it is told, on every offset."""

    contract = pl.CONTRACT_DEPTH_SMALL
    representation = pl.PROFILE
    uses_prev = False
    has_mode = False
    chunk = 5
    ensemble_m = 0.0
    ensemble_link_s = pl.CHUNK_LINK_S

    def __init__(self, gear=FWD, steer=0.2):
        self.gear, self.steer = gear, steer

    def run(self, x, prev, mode=None, **extra):
        return same(self.gear, self.steer, self.chunk)


def blend_of(outputs):
    decided, why = pl.read_output(outputs)
    assert decided is not None, why
    return decided


# --- K = 1 is the one-step model ----------------------------------------------------

def test_k1_passes_every_answer_through_untouched():
    e = pl.ChunkEnsembler(1)
    for out in (same(FWD, 0.3, 1), same((np.nan, 0.0, 0.0), 0.0, 1), {"logits": None}):
        assert e.apply(out, T0) is out
    assert pl.check_chunk({}, {"logits": [1, 3], "steer": [1, 1]}) == (1, 0.0, pl.CHUNK_LINK_S)
    assert pt.chunk_meta(1, -0.5) == {}, "a one-step sidecar is the sidecar from before"


def test_a_model_from_before_the_chunk_heads_drives_through_a_k1_ensembler():
    model = pl.PolicyModel(str(SMOKE_MODEL))
    assert (model.chunk, model.ensemble_m) == (1, 0.0)
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()), model)
    assert d.ensembler.k == 1
    d.tick(T0)
    assert d.ensembler.depth == 0


def test_a_chunk_that_agrees_with_itself_decides_as_one_step_would():
    """Every offset the same answer: the blend is that answer, gear, steer and
    the confidence the speed scale reads."""
    one = same((2.0, 0.5, -1.0), -0.4, 1)
    e = pl.ChunkEnsembler(5, -0.5)
    for k in range(7):
        out = e.apply(same((2.0, 0.5, -1.0), -0.4), T0 + 0.1 * k)
        assert blend_of(out)[0] == blend_of(one)[0]
        assert blend_of(out)[2] == pytest.approx(blend_of(one)[2], abs=1e-6)
        assert pl.committed_confidence(out, pl.FORWARD) == pytest.approx(
            pl.committed_confidence(one, pl.FORWARD), abs=1e-6)
    assert e.depth == 5


def test_read_output_takes_a_raw_chunk_as_its_first_offset():
    out = chunked([FWD, REV, REV, REV, REV], [0.1, 0.9, 0.9, 0.9, 0.9])
    action, _, steer, _ = blend_of(out)
    assert action == pl.FORWARD and steer == pytest.approx(0.1)
    assert pl.read_output(chunked([FWD] * 5, [0.1] * 4))[0] is None, "steer not K wide"


# --- the blend ----------------------------------------------------------------------

@pytest.mark.parametrize("m,ticks", [(0.0, 3), (-0.5, 2), (-3.0, 1)])
def test_a_step_from_forward_to_reverse_takes_the_ticks_its_weights_say(m, ticks):
    """Five sure forwards, then the model says reverse on every offset. With m
    = 0 the blend is a plain mean of the five, so reverse wins on the third
    new answer; m = -0.5 weighs the newest 1 against 0.61, 0.37, 0.22, 0.14,
    so on the second; m = -3 all but drops the old, so on the first."""
    e = pl.ChunkEnsembler(5, m)
    t = T0
    for _ in range(5):
        assert blend_of(e.apply(same(FWD, 0.0), t))[0] == pl.FORWARD
        t += 0.1
    took = None
    for k in range(1, 6):
        if blend_of(e.apply(same(REV, 1.0), t))[0] == pl.REVERSE:
            took = k
            break
        t += 0.1
    assert took == ticks


def test_the_steer_is_the_weighted_mean_of_each_answers_own_offset():
    """Offset `age` of each answer, not offset 0 of all of them."""
    e = pl.ChunkEnsembler(3, -0.5)
    e.apply(chunked([FWD] * 3, [0.0, 0.3, 0.6]), T0)
    e.apply(chunked([FWD] * 3, [0.1, 0.4, 0.7]), T0 + 0.1)
    out = e.apply(chunked([FWD] * 3, [0.2, 0.5, 0.8]), T0 + 0.2)
    w = [math.exp(-0.5 * a) for a in range(3)]
    want = (w[0] * 0.2 + w[1] * 0.4 + w[2] * 0.6) / sum(w)
    assert blend_of(out)[2] == pytest.approx(want, abs=1e-6)


# --- where the chain starts over ----------------------------------------------------

def test_a_gap_over_the_link_starts_the_chain_over():
    e = pl.ChunkEnsembler(5)
    for k in range(4):
        e.apply(same(FWD, 0.0), T0 + 0.1 * k)
    assert e.depth == 4
    e.apply(same(FWD, 0.0), T0 + 0.3 + pl.CHUNK_LINK_S + 0.01)
    assert e.depth == 1
    e.apply(same(FWD, 0.0), T0 + 0.3 + pl.CHUNK_LINK_S + 0.01 + pl.CHUNK_LINK_S - 0.01)
    assert e.depth == 2, "just under the link is one chain"


def test_a_mode_change_starts_the_chain_over():
    e = pl.ChunkEnsembler(5)
    e.apply(same(FWD, 0.0), T0, False)
    e.apply(same(FWD, 0.0), T0 + 0.1, False)
    e.apply(same(FWD, 0.0), T0 + 0.2, True)
    assert e.depth == 1


def test_an_answer_that_is_not_finite_is_never_kept_and_goes_on_to_be_refused():
    e = pl.ChunkEnsembler(5)
    e.apply(same(FWD, 0.0), T0)
    e.apply(same(FWD, 0.0), T0 + 0.1)
    bad = chunked([FWD, (np.nan, 0.0, 0.0), FWD, FWD, FWD], [0.0] * 5)
    out = e.apply(bad, T0 + 0.2)
    assert e.depth == 0
    assert pl.read_output(out) == (None, "not finite")


def test_my_hand_on_the_pad_starts_the_chain_over():
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()), ChunkStub())
    for k in range(6):
        d.tick(T0 + 0.1 * k)
    assert d.ensembler.depth == 5
    d.human_steering = True
    d.tick(T0 + 0.6)
    assert d.ensembler.depth == 1, "only the answer made under my hand"
    d.human_steering = False
    d.tick(T0 + 0.7)
    assert d.ensembler.depth == 1, "and none of that once I let go"
    d.tick(T0 + 0.8)
    assert d.ensembler.depth == 2
    d.abort()
    assert d.ensembler.depth == 0


def test_map_lost_starts_the_chain_over():
    pytest.importorskip("scipy")
    from test_policy_map import FakeSlam, reader

    class MapChunk(ChunkStub):
        uses_lidar = True
        uses_map = True

    slam = FakeSlam()
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()), MapChunk(),
                        lidar=reader(), slam=slam)
    assert not d.hand_rules, "a map model's chunk drives with the hand rules off"
    t = T0
    for _ in range(5):
        slam.at(t)
        d.tick(t)
        t += 0.1
    assert d.ensembler.depth == 5 and d.controller.throttles[-1] > 0
    while d.last_decision.reason != "map lost":
        slam.at(T0, present=False)
        d.tick(t)
        t += 0.1
        assert t < T0 + 5.0
    assert d.ensembler.depth == 0 and d.controller.throttles[-1] == 0


# --- the sidecar --------------------------------------------------------------------

def test_the_chunk_fixture_loads_and_its_blend_drives():
    model = pl.PolicyModel(str(CHUNK_MODEL))
    assert (model.chunk, model.ensemble_m, model.ensemble_link_s) == (5, -0.5, pl.CHUNK_LINK_S)
    out = model.run(pl.encode_input(model.representation, np.full((50, 80), 1500, np.uint16)),
                    np.zeros((1, pl.PREV_SIZE), np.float32), pl.encode_mode(False))
    assert out["logits"].shape == (1, 15) and out["steer"].shape == (1, 5)
    assert pl.read_output(out)[0] is not None
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()), model)
    for k in range(3):
        d.tick(T0 + 0.1 * k)
    assert d.ensembler.k == 5 and d.ensembler.depth == 3
    assert d.last_decision.reason != "exception"


@pytest.mark.parametrize("edit", [{"chunk": 3}, {"chunk": None}, {"ensemble_m": None},
                                  {"ensemble_m": float("nan")}, {"chunk": 5.0}])
def test_a_sidecar_that_disagrees_with_the_chunk_graph_is_refused(tmp_path, edit):
    meta = json.loads(CHUNK_MODEL.with_suffix(".json").read_text())
    for k, v in edit.items():
        if v is None:
            meta.pop(k)
        else:
            meta[k] = v
    (tmp_path / "m.onnx").write_bytes(CHUNK_MODEL.read_bytes())
    (tmp_path / "m.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        pl.PolicyModel(str(tmp_path / "m.onnx"))


def test_a_one_step_graph_with_a_chunk_sidecar_is_refused(tmp_path):
    meta = json.loads(SMOKE_MODEL.with_suffix(".json").read_text())
    meta.update(pt.chunk_meta(5, 0.0))
    (tmp_path / "m.onnx").write_bytes(SMOKE_MODEL.read_bytes())
    (tmp_path / "m.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        pl.PolicyModel(str(tmp_path / "m.onnx"))
    with pytest.raises(ValueError):
        pl.check_chunk({"chunk": 5, "ensemble_m": 0.0}, {}, pl.STEER_DIRECTION)


# --- the trainer's labels -----------------------------------------------------------

def test_chunk_labels_chain_through_short_gaps_and_stop_at_a_long_one_or_a_session():
    """Offset i is the i-th next decision tick: a non-decision tick is
    skipped, a gap over the link and a new session end the chain."""
    t = np.array([0.0, 0.1, 0.15, 0.2, 0.3, 0.7, 0.8, 0.0, 0.1]) * 1e9
    data = {"t_ns": t.astype(np.int64),
            "session": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1]),
            "decision": np.array([1, 1, 0, 1, 1, 1, 1, 1, 1], bool)}
    fut = pt.chunk_index(data, 3)
    assert fut[0].tolist() == [0, 1, 3]
    assert fut[1].tolist() == [1, 3, 4]
    assert fut[2].tolist() == [-1, -1, -1], "not a decision tick"
    assert fut[3].tolist() == [3, 4, -1], "0.3 -> 0.7 is past the link"
    assert fut[5].tolist() == [5, 6, -1], "the session ends"
    assert fut[7].tolist() == [7, 8, -1]
    assert pt.chunk_meta(5, -0.5) == {"chunk": 5, "ensemble_m": -0.5,
                                      "ensemble_link_s": pl.CHUNK_LINK_S}
