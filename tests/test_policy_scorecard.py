"""The scorecard: drives no model trains on, and the grader that reads them.

tools/scorecard.txt names the sessions; policy_train.py keeps every model
off them, so tools/policy_scorecard.py can put any two models side by side
on ticks neither has seen. These hold the list, the holdout, the grader's
mirror test and reversal finder, and that a steering I make standing still
stays a label with its own value.
"""

import re

import numpy as np
import pytest

from crawler.autopilot import policy as pl
import policy_dataset as pd
import policy_eval as pe
import policy_scorecard as sc
import policy_train as pt

from test_policy_tools import R, footprint_session, record

F, V, S = (pl.ACTIONS.index(a) for a in (pl.FORWARD, pl.REVERSE, pl.STOP))


def test_the_committed_scorecard_is_names_only_and_never_shrinks_below_four():
    names = pt.scorecard_sessions()
    assert len(names) >= 4 and len(set(names)) == len(names)
    assert all(re.fullmatch(r"crawler-\d{8}-\d{6}Z-p\d+-\d{3}", n) for n in names), names


def test_the_list_reads_past_comments_and_blank_lines(tmp_path):
    p = tmp_path / "card.txt"
    p.write_text("# a comment\n\na-1  # why\n  b-2\n#c-3\n")
    assert pt.scorecard_sessions(str(p)) == ["a-1", "b-2"]
    assert pt.scorecard_sessions(str(tmp_path / "none.txt")) == []


def test_the_holdout_marks_every_tick_of_a_scorecard_session_and_no_other():
    data = {"meta": {"sessions": ["a", "b", "c"]}, "session": np.array([0, 1, 2, 1, 0])}
    assert pt.scorecard_mask(data, ["b", "not here"]).tolist() == [False, True, False, True, False]
    assert not pt.scorecard_mask(data, []).any()


def test_the_grader_takes_the_scorecards_decision_ticks_in_time_order():
    data = {"meta": {"sessions": ["a", "b"]}, "session": np.array([1, 0, 1, 1]),
            "decision": np.array([True, True, False, True]),
            "t_ns": np.array([30, 10, 20, 5])}
    assert sc.scorecard_ticks(data, ["b"]).tolist() == [3, 0]


# --- the grader on a drive of its own --------------------------------------------

def drive():
    """Forward on the right lock, a short stop, reverse on the left lock,
    forward again on the right: two swaps of lock across a reversal."""
    spec = [(2, 0.3, 0.5, 0.1), (0.2, 0.0, 0.5, 0.0), (2, -0.6, -0.5, 0.3),
            (0.2, 0.0, -0.5, 0.0), (2, 0.3, 0.5, 0.1)]
    session = footprint_session(record(spec))
    ticks, _, _ = pd.align(session, rules=R)
    a = pd.to_arrays([(session, ticks)])
    a["meta"] = {"sessions": ["s"], "marks": {}}
    return a


class Stub:
    contract = pl.CONTRACT_DEPTH_SMALL
    representation = pl.PROFILE
    has_mode = uses_prev = uses_imu = uses_lidar = uses_map = uses_goal = False
    chunk, ensemble_m, ensemble_link_s = 1, 0.0, pl.CHUNK_LINK_S

    def __init__(self, steer_of):
        self.steer_of = steer_of

    def run(self, x, prev, mode=None, **_):
        return {"logits": np.array([[5.0, 0.0, 0.0]], np.float32),
                "throttle": np.array([[0.5]], np.float32),
                "steer": np.array([[self.steer_of(x)]], np.float32)}


def test_the_reversals_are_my_gear_swaps_through_a_short_stop():
    a = drive()
    ticks = np.flatnonzero(a["decision"])
    ok = np.ones(len(ticks), bool)
    found = sc.reversals(a, ticks, ok)
    assert len(found) == 2
    for before, after in found:
        assert len(before) == len(after) == sc.AFTER
        assert len(set(a["action"][ticks[before]])) == 1
        assert a["action"][ticks[before][0]] != a["action"][ticks[after][0]]


def test_a_model_that_ignores_the_mirror_fails_it_and_one_that_follows_it_passes():
    a = drive()
    ticks = np.flatnonzero(a["decision"])
    # The left-right asymmetry of the frame, so the mirror image is its negative.
    lean = lambda x: float(np.clip(x[..., :16].sum() - x[..., 16:].sum(), -1, 1))  # noqa: E731
    frames = a["frames"].copy()
    frames[:, :, :20] = 800   # nearer on the left
    a["frames"] = frames
    follows = sc.score(Stub(lean), a, ticks)
    ignores = sc.score(Stub(lambda x: 0.5), a, ticks)
    assert follows["mirror_fail"] == 0.0
    assert ignores["mirror_fail"] == 1.0
    # Steering one way all drive is on my side of half the lock swaps' ticks.
    assert ignores["lock"] == pytest.approx(0.5)
    assert np.isnan(ignores.get("map_use", np.nan)), "no map, no map_use"


# --- steering standing still -------------------------------------------------------

def test_steering_standing_still_is_a_stop_decision_labelled_with_its_own_lock():
    """A stop is where I put the lock on for the next gear. Nothing between
    the recording and the loss may centre that steering or drop it: it is
    the label the per-gear stop and every steer head train on."""
    spec = [(2, 0.3, 0.0, 0.1), (1.0, 0.0, 0.6, 0.0), (2, -0.6, 0.6, 0.3)]
    session = footprint_session(record(spec))
    ticks, _, _ = pd.align(session, rules=R)
    a = pd.to_arrays([(session, ticks)])
    stops = (a["action"] == S) & a["decision"]
    assert stops.sum() >= 8
    assert a["steer_label"][stops].all()
    assert a["steer"][stops] == pytest.approx(np.full(stops.sum(), 0.6 / pl.STEER_SCALE))
    # And they are training ticks: inside the trainer's mask and its steer labels.
    a["meta"] = {"sessions": ["s"], "marks": {}}
    steer_ok, throttle_ok = pe.label_masks(a, "pad")
    mask = pt.training_mask(a, np.zeros(len(a["t_ns"]), bool), steer_ok | throttle_ok)
    assert mask[stops].all() and steer_ok[stops].all()
