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
from crawler.autopilot import policy_lidar as pli
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


# --- the car's own drives: open floor, the rejected drives, the ensemble ------------

A, T = pd.SOURCES.index(pd.SOURCE_AUTO), pd.SOURCES.index(pd.SOURCE_TAKEOVER)


def scans(n, front_mm, side_mm=300):
    """n scans with every front bin (within OPEN_FRONT_DEG) at front_mm and
    the rest at side_mm: a wall close by on the sides is still open floor."""
    ang = np.arange(pli.BINS) * pli.BIN_DEG
    front = (ang <= sc.OPEN_FRONT_DEG) | (ang >= 360 - sc.OPEN_FRONT_DEG)
    b = np.where(front, front_mm, side_mm).astype(np.uint16)
    return np.tile(b, (n, 1))


def test_open_floor_is_nothing_near_in_front_whatever_is_beside():
    n = 4
    d = {"lidar_bins": scans(n, 1500), "lidar_present": np.array([True, True, True, False])}
    d["lidar_bins"][1, 0] = 900        # one bin dead ahead too near
    d["lidar_bins"][2, 0] = 0          # a bin with no return is not near
    assert sc.open_front(d, np.arange(n)).tolist() == [True, False, True, False]
    assert not sc.open_front({}, np.arange(n)).any()


def car_drive():
    """drive()'s ticks as the car's: auto, forward, on open floor, but for a
    takeover at the end, which makes the 3 s before it not accepted."""
    a = drive()
    n = len(a["t_ns"])
    a["meta"]["sources"] = list(pd.SOURCES)
    a["source"] = np.full(n, A, np.int8)
    a["source"][-3:] = T
    a["throttle_label"] = a["source"] == T
    a["steer_label"] = a["source"] == T
    a["action"] = np.full(n, F, np.int64)
    a["lidar_bins"] = scans(n, 2000)
    a["lidar_present"] = np.ones(n, bool)
    return a


def test_the_accepted_ticks_are_auto_forward_open_and_not_before_a_takeover():
    a = car_drive()
    ticks = np.arange(len(a["t_ns"]))
    accepted, took = sc.open_sets(a, ticks)
    t = a["t_ns"] / 1e9
    first_take = t[a["source"] == T].min()
    assert accepted[t < first_take - sc.ACCEPT_S - 0.05].all()
    assert not accepted[(t >= first_take - sc.ACCEPT_S) & (a["source"] == A)].any()
    assert took.tolist() == (a["source"] == T).tolist()
    a["lidar_bins"][:] = 500
    accepted, took = sc.open_sets(a, ticks)
    assert not accepted.any() and not took.any()


class Gear(Stub):
    """A model that always answers this gear."""

    def __init__(self, gear, chunk=1):
        super().__init__(lambda x: 0.0)
        self.gear = gear
        self.chunk = chunk

    def run(self, x, prev, mode=None, **_):
        lg = np.zeros((1, 3 * self.chunk), np.float32)
        lg[0, self.gear::3] = 5.0
        return {"logits": lg, "throttle": np.array([[0.5]], np.float32),
                "steer": np.zeros((1, self.chunk), np.float32)}


def test_open_fwd_rev_is_the_share_of_accepted_open_ticks_it_reverses_on():
    a = car_drive()
    ticks = np.arange(len(a["t_ns"]))
    s = sc.open_scores(Gear(V), a, ticks)
    assert s["open_fwd_rev"] == 1.0 and s["open_fwd_stop"] == 0.0
    assert s["L_rev"] == 1.0 and s["L_ticks"] == 3
    s = sc.open_scores(Gear(F), a, ticks)
    assert s["open_fwd_rev"] == 0.0 and s["L_stop"] == 0.0
    assert s["open_fwd_ticks"] > 0
    d = sc.drive_scores(Gear(S), a, ticks)
    assert d["drive_open_stop"] == 1.0 and d["drive_open_ticks"] == len(ticks)


def test_the_ensemble_averages_the_gear_probabilities_per_offset():
    class Two(Gear):
        def run(self, x, prev, mode=None, **_):
            out = super().run(x, prev, mode)
            out["logits"] = np.log(np.array([[0.6, 0.3, 0.1, 0.2, 0.7, 0.1]], np.float32))
            return out

    one, two = Gear(F, chunk=2), Two(F, chunk=2)
    one.meta = two.meta = {"train_sessions": []}
    e = sc.Ensemble([one, two])
    lg = e.run(None, None)["logits"].reshape(2, 3)
    p1 = np.exp(np.array([[5, 0, 0], [5, 0, 0]], float))
    p1 /= p1.sum(1, keepdims=True)
    want = (p1 + np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])) / 2
    assert np.exp(lg) == pytest.approx(want, abs=1e-5)
    assert e.chunk == 2 and e.sha256 is None
    with pytest.raises(ValueError, match="differ"):
        sc.Ensemble([Gear(F, chunk=1), Gear(F, chunk=2)])


def test_a_model_spec_with_a_plus_is_an_ensemble_and_named_so():
    assert sc.model_label("/m/a.onnx") == "a"
    assert sc.model_label("/m/a.onnx+/m/b.onnx") == "ensemble(a+b)"


def test_the_hindsight_goal_never_takes_the_fed_one(monkeypatch):
    seen = {}

    def grab(data, rows, shown, rng, **kw):
        seen.update(data)
        return np.zeros((len(rows), 4), np.float32)
    monkeypatch.setattr(pt, "goal_inputs", grab)
    d = {"goal_poses": np.zeros(2), "map_on": np.ones(2, bool), "fed_ok": np.ones(2, bool),
         "fed_ahead": np.ones(2), "fed_left": np.ones(2)}
    sc.hind_goals(d, np.arange(2))
    assert "goal_poses" in seen and not any(k.startswith("fed_") for k in seen)
    assert "fed_ok" in d


def test_the_rejected_drives_list_is_names_only():
    names = pt.rejected_sessions()
    assert names and len(set(names)) == len(names)
    assert all(re.fullmatch(r"crawler-\d{8}-\d{6}Z-p\d+-\d{3}", n) for n in names), names
    assert not set(names) & set(pt.scorecard_sessions())
