"""Branched heads by goal bearing (policy_train --goal-branches).

The goal weighs four heads inside the graph (policy_train.goal_gate): one with
no goal, one each for a goal to the left, ahead and to the right. These hold
the gate to summing to one, to mirroring left onto right, and to never moving
on a step as the bearing sweeps round (an edge, or straight behind), and the
exported graph to the gate: the fixture is untrained but for its steer
biases, planted at +2 on the left head, -2 on the right and 0 on the others,
so its steer is exactly tanh(2 (left - right)) of the gate. No torch here.
"""

import json
import math
import pathlib

import numpy as np
import pytest

from crawler.autopilot import policy as pl
from crawler.autopilot import policy_goal as pg
from crawler.mapping import egomap as em
import models as md
import policy_train as pt

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
BRANCHED = FIXTURES / "policy-goal-branches.onnx"
NONE, LEFT, AHEAD, RIGHT = (pt.GOAL_BRANCHES.index(n) for n in ("none", "left", "ahead", "right"))


def at(deg, dist_m=2.0):
    b = np.radians(np.atleast_1d(deg))
    return pg.encode_many(dist_m * np.cos(b), dist_m * np.sin(b), np.ones(len(b), bool))


def sweep(step=0.25):
    return np.arange(-180.0, 180.0 + step / 2, step)


# --- the gate ----------------------------------------------------------------------

def test_no_goal_is_all_the_no_goal_head_and_a_goal_none_of_it():
    w = pt.goal_gate(np.concatenate([pg.missing(), at([0, 90, -90, 180])]))
    np.testing.assert_allclose(w[0], [1, 0, 0, 0])
    assert np.all(w[1:, NONE] == 0)
    assert np.all(np.isfinite(w))


@pytest.mark.parametrize("deg,branch", [(0, AHEAD), (10, AHEAD), (-10, AHEAD), (35, LEFT),
                                        (90, LEFT), (150, LEFT), (-35, RIGHT), (-90, RIGHT),
                                        (-150, RIGHT)])
def test_a_bearing_well_inside_a_bin_is_mostly_that_bins_head(deg, branch):
    w = pt.goal_gate(at(deg))[0]
    assert w[branch] > 0.85 and w.argmax() == branch


def test_every_row_sums_to_one_and_no_weight_is_negative():
    w = pt.goal_gate(at(sweep()))
    np.testing.assert_allclose(w.sum(1), 1.0, atol=1e-6)
    assert w.min() >= -1e-6


def test_the_mirror_swaps_left_and_right_exactly():
    g = at(sweep())
    w, wm = pt.goal_gate(g), pt.goal_gate(pg.mirror(g))
    np.testing.assert_allclose(wm[:, [NONE, LEFT, AHEAD, RIGHT]], w[:, [NONE, RIGHT, AHEAD, LEFT]],
                               atol=1e-6)


def test_the_weights_never_step_as_the_bearing_sweeps_round_through_behind():
    w = pt.goal_gate(at(sweep(0.25)))
    # A quarter of a degree moves no weight more than ~1.3% (the logistic's
    # steepest is 0.25 / soft per radian, the behind ramp 0.5 / behind).
    worst = max(0.25 / math.radians(pt.GOAL_BRANCH_SOFT_DEG),
                0.5 / math.radians(pt.GOAL_BRANCH_BEHIND_DEG)) * math.radians(0.25)
    assert np.abs(np.diff(w, axis=0)).max() <= worst * 1.05
    # ...and round the back: -180 and +180 are one bearing.
    np.testing.assert_allclose(w[0], w[-1], atol=1e-6)
    np.testing.assert_allclose(w[0, [LEFT, RIGHT]], [0.5, 0.5], atol=1e-6)


# --- the exported graph --------------------------------------------------------------

def _inputs():
    x = np.zeros((1, pt.profile_rows_of("lower_third"), 32), np.float32)
    prev = np.zeros((1, pl.PREV_SIZE), np.float32)
    m = np.zeros((1, em.CHANNELS, em.SIZE, em.SIZE), np.float32)
    m[0, em.PRESENT] = 1.0
    return x, prev, m


def test_the_branched_model_loads_with_the_inputs_and_outputs_of_a_one_head_goal_model():
    model = pl.PolicyModel(str(BRANCHED))
    assert model.uses_goal and model.uses_map and model.uses_lidar and model.uses_imu
    meta = json.loads(BRANCHED.with_suffix(".json").read_text())
    assert meta["outputs"] == list(pt.output_names(pl.STEER_PER_GEAR))
    assert meta["goal_branches"]["names"] == list(pt.GOAL_BRANCHES)
    # The branches are a record, not a contract: the car feeds it as any goal model.
    assert "goal_branches" not in md.contract_of(meta)


def test_the_graphs_steer_is_the_gate_of_the_goal_it_is_fed():
    model = pl.PolicyModel(str(BRANCHED))
    x, prev, m = _inputs()
    degs = sweep(5.0)
    got = np.array([model.run(x, prev, map=m, goal=at(d))["steer_forward"][0, 0] for d in degs])
    w = pt.goal_gate(at(degs))
    np.testing.assert_allclose(got, np.tanh(2.0 * (w[:, LEFT] - w[:, RIGHT])), atol=1e-5)
    # Left is a goal with sin > 0, and it steers to the left head's side.
    assert got[degs == 90][0] > 0.9 and got[degs == -90][0] < -0.9


def test_no_goal_through_the_graph_is_finite_and_the_no_goal_head():
    model = pl.PolicyModel(str(BRANCHED))
    x, prev, m = _inputs()
    out = model.run(x, prev, map=m, goal=pg.missing())
    assert all(np.all(np.isfinite(v)) for v in out.values())
    assert out["steer_forward"][0, 0] == pytest.approx(0.0, abs=1e-6)
    off = model.run(x, prev, map=m)
    np.testing.assert_array_equal(off["logits"], out["logits"])


# --- the trainer's flag ----------------------------------------------------------------

def test_the_flag_needs_the_goal_and_a_regression_steer_head():
    assert pt.main(["data.npz", "out.onnx", "--goal-branches", "3"]) == 2


def test_no_branches_leaves_the_goal_sidecar_as_it_was():
    assert "goal_branches" not in pt.goal_meta(True)
    assert pt.goal_meta(True) == pt.goal_meta(True, branches=0)
    assert pt.goal_meta(False, branches=3) == {}
