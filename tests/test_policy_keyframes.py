"""Keyframes (--keyframe-weight): the decisions a copycat of my last few
actions gets wrong count more. policy_keyframes on arrays; the copycat is a
small torch MLP, trained for real on a toy drive."""

import numpy as np
import pytest

import policy_keyframes as pk
import policy_train as pt

FWD, REV, STOP = 0, 1, 2


def drive(sessions, n=200, dt=0.1, change_every=40, gap_at=None):
    """`sessions` sessions of n decisions each, dt apart, forward and
    straight but for a turn to full right lock every change_every ticks
    (held 10 ticks) and a stop at each turn's end. A 2 s hole before tick
    gap_at of each session, if given."""
    t, session, action, steer = [], [], [], []
    for s in range(sessions):
        ts = np.arange(n) * dt
        if gap_at is not None:
            ts[gap_at:] += 2.0
        a = np.full(n, FWD)
        st = np.zeros(n, np.float32)
        for start in range(change_every // 2, n, change_every):
            st[start:start + 10] = 1.0
            a[start + 10:start + 13] = STOP
        t.append(ts + s * 1000.0)
        session.append(np.full(n, s))
        action.append(a)
        steer.append(st)
    total = sessions * n
    return {"t_ns": (np.concatenate(t) * 1e9).astype(np.int64),
            "session": np.concatenate(session).astype(np.int16),
            "decision": np.ones(total, bool),
            "action": np.concatenate(action).astype(np.int8),
            "steer": np.concatenate(steer),
            "mark": np.zeros(total, np.int8),
            "source": np.zeros(total, np.int8),
            "meta": {"sessions": [str(i) for i in range(sessions)], "sources": ["record"]}}


def test_the_history_is_the_decisions_before_in_the_same_session():
    d = drive(2, n=20)
    past = pk.history_index(d, k=3)
    assert list(past[5]) == [4, 3, 2]
    assert list(past[1]) == [0, -1, -1]
    assert (past[0] == -1).all()
    # Session 1 starts over: nothing of session 0 is its history.
    assert (past[20] == -1).all() and list(past[22]) == [21, 20, -1]


def test_a_hole_in_the_ticks_starts_the_history_over():
    d = drive(1, n=20, gap_at=10)
    past = pk.history_index(d, k=2)
    assert (past[10] == -1).all() and list(past[11]) == [10, -1]
    assert list(past[9]) == [8, 7]


def test_a_tick_that_is_no_decision_is_skipped_and_has_no_history():
    d = drive(1, n=10, dt=0.05)
    d["decision"][1::2] = False
    past = pk.history_index(d, k=2)
    assert list(past[4]) == [2, 0]
    assert (past[3] == -1).all()


def test_the_copycat_input_is_each_past_decision_one_hot_with_its_steer():
    d = drive(1, n=30)
    past = pk.history_index(d, k=2)
    x = pk.copycat_inputs(d, np.array([0, 21]), past)
    assert x.shape == (2, 2 * pk.STEP_FEATURES)
    assert (x[0] == 0).all()
    # Tick 21's past: tick 20 (the turn's first, full lock) and 19 (straight).
    assert list(x[1]) == [1, 0, 0, 1, 1, 1, 0, 0, 0, 1]


def test_the_sessions_are_dealt_into_folds_and_never_more_folds_than_sessions():
    fold_of, n = pk.session_folds(np.arange(12), n_folds=5)
    assert n == 5 and sorted(set(fold_of.values())) == list(range(5))
    assert len(fold_of) == 12
    assert pk.session_folds(np.array([3, 3, 7]), n_folds=5)[1] == 2


def test_the_moments_i_changed_my_mind_are_the_keyframes():
    d = drive(6)
    every = np.ones(len(d["t_ns"]), bool)
    ape, parts = pk.action_prediction_error(d, every, every, every, k=4, epochs=40)
    w, keyframe = pk.keyframe_weights(ape, every, 5.0, share=0.05)
    # A turn starting cannot be told from the straight before it, nor its
    # end (with the stop) from four ticks of a ten-tick turn: keyframes.
    # Going again after three ticks of stop the four past ticks do show,
    # so that is no keyframe, however much the action changed.
    turn_on = np.flatnonzero(np.diff(d["steer"]) > 0) + 1
    turn_off = np.flatnonzero(np.diff(d["steer"]) < 0) + 1
    go = np.flatnonzero((d["action"][:-1] == STOP) & (d["action"][1:] == FWD)) + 1
    assert keyframe[turn_on].all() and keyframe[turn_off].all()
    assert not keyframe[go].any()
    assert keyframe.sum() <= 0.06 * len(keyframe)
    assert set(np.unique(w)) == {1.0, 5.0} and (w[keyframe] == 5.0).all()
    assert parts["steer"][turn_on].mean() > 10 * parts["steer"][~keyframe].mean()


def test_the_apes_two_parts_are_each_scaled_to_their_mean():
    d = drive(4)
    every = np.ones(len(d["t_ns"]), bool)
    ape, parts = pk.action_prediction_error(d, every, every, every, k=3, epochs=5)
    want = parts["gear"] / parts["gear"].mean() + parts["steer"] / parts["steer"].mean()
    assert np.allclose(ape, want)


def test_a_part_with_no_label_adds_nothing_and_a_tick_outside_the_fit_is_zero():
    d = drive(4)
    n = len(d["t_ns"])
    fit = np.ones(n, bool)
    fit[:10] = False
    no_steer = np.zeros(n, bool)
    ape, parts = pk.action_prediction_error(d, fit, no_steer, fit, k=3, epochs=5)
    assert (parts["steer"] == 0).all() and (ape[:10] == 0).all()


def test_the_top_share_by_ape_is_upweighted_and_a_zero_ape_never():
    ape = np.r_[np.arange(1, 11, dtype=float), np.zeros(10)]
    fit = np.r_[np.ones(10, bool), np.zeros(10, bool)]
    w, keyframe = pk.keyframe_weights(ape, fit, 5.0, share=0.2)
    assert list(np.flatnonzero(keyframe)) == [8, 9]
    assert w[8] == w[9] == 5.0 and (np.delete(w, [8, 9]) == 1.0).all()
    w0, k0 = pk.keyframe_weights(np.zeros(10), np.ones(10, bool), 5.0)
    assert not k0.any() and (w0 == 1.0).all()


def test_their_ten_percent_at_five_is_a_third_of_the_weight():
    keyframe = np.r_[np.ones(10, bool), np.zeros(90, bool)]
    w = np.where(keyframe, 5.0, 1.0)
    assert pk.weight_share(np.ones(100), w, keyframe) == pytest.approx(50 / 140)
    # The other weights multiply in: a keyframe that is also a takeover at 3.
    base = np.ones(100)
    base[0] = 3.0
    assert pk.weight_share(base, w, keyframe) == pytest.approx(60 / 150)


def test_the_keyframe_weight_multiplies_every_other_tick_weight():
    d = drive(1, n=10)
    idx = np.arange(10)
    plain = pt.tick_weights(d, idx, 2.0, 3.0)
    d["keyframe_w"] = np.where(idx == 4, 5.0, 1.0).astype(np.float32)
    d["accepted_w"] = np.where(idx >= 4, 0.3, 1.0).astype(np.float32)
    w = pt.tick_weights(d, idx, 2.0, 3.0)
    assert w[4] == pytest.approx(plain[4] * 5.0 * 0.3)
    assert w[5] == pytest.approx(plain[5] * 0.3) and w[0] == plain[0]


def test_no_keyframe_flag_leaves_the_weights_as_they_always_were():
    d = drive(1, n=10)
    idx = np.arange(10)
    before = pt.tick_weights(d, idx, 2.0, 3.0)
    assert before.dtype == np.float32 and (before == 1.0).all()
    assert "keyframe_w" not in d


def test_the_gear_changes_among_the_keyframes_are_counted():
    d = drive(1, n=40)
    past = pk.history_index(d, 1)
    keyframe = np.zeros(40, bool)
    keyframe[[20, 30, 31]] = True   # 20 a steer change, 30 into stop, 31 stop held
    assert pk.gear_changes(d, keyframe, past) == 1


def test_steer_only_keyframes_rank_by_the_steer_and_weight_the_steer_loss_alone():
    d = drive(1, n=40)
    n = len(d["t_ns"])
    every = np.ones(n, bool)
    # The gear's surprise at the stops (30-32), the steer's at the turn (20).
    parts = {"gear": np.where(np.isin(np.arange(n), [30, 31, 32]), 9.0, 0.1),
             "steer": np.where(np.arange(n) == 20, 9.0, 0.1)}
    ape = parts["gear"] + parts["steer"]
    kf_w, keyframe, steer_w = pt.place_keyframes(ape, parts, every, 5.0, 0.025,
                                                 steer_only=True)
    assert list(np.flatnonzero(keyframe)) == [20] and steer_w is kf_w
    idx = np.arange(n)
    # The gear's weights stay as they were: nothing goes on data["keyframe_w"].
    base = pt.tick_weights(d, idx, 2.0, 3.0)
    assert "keyframe_w" not in d and (base == 1.0).all()
    w = pt.steer_weights(base, steer_w, idx)
    assert w.dtype == np.float32 and w[20] == 5.0 and (np.delete(w, 20) == 1.0).all()
    # A chunk's (rows, offsets) take each offset at its own tick's weight.
    fut0 = np.stack([idx, np.minimum(idx + 1, n - 1)], axis=1)
    wc = pt.steer_weights(np.ones(fut0.shape, np.float32), steer_w, fut0)
    assert wc[20, 0] == wc[19, 1] == 5.0 and wc[20, 1] == wc[19, 0] == 1.0


def test_keyframes_on_every_head_rank_by_the_whole_ape_and_leave_the_steer_as_is():
    n = 40
    parts = {"gear": np.where(np.arange(n) == 31, 9.0, 0.1),
             "steer": np.where(np.arange(n) == 20, 4.0, 0.1)}
    ape = parts["gear"] + parts["steer"]
    kf_w, keyframe, steer_w = pt.place_keyframes(ape, parts, np.ones(n, bool), 5.0, 0.025)
    assert list(np.flatnonzero(keyframe)) == [31] and steer_w is None
    base = np.ones(n, np.float32)
    assert pt.steer_weights(base, None, np.arange(n)) is base
