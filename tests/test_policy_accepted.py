"""The car's own accepted driving as weak labels, and the softer class weights.

An auto tick the model drove, with no takeover in the 3 s after it or the
2 s before it and nothing missing in between, is a decision I let stand
(policy_train.accepted_auto); --accepted-weight makes it a label at that
weight. --class-weight-power takes the gear's class weights to a power.
Arrays only, never torch.
"""

import numpy as np
import pytest

import policy_dataset as pd
import policy_train as pt

AUTO, TAKE = pd.SOURCES.index(pd.SOURCE_AUTO), pd.SOURCES.index(pd.SOURCE_TAKEOVER)


def drive(sources, dt=0.1, session=None, policy=True, gap_at=None):
    """One tick each dt s with these sources, a hole of 2 s before tick
    gap_at if given."""
    n = len(sources)
    t = np.arange(n) * dt
    if gap_at is not None:
        t[gap_at:] += 2.0
    return {"t_ns": (t * 1e9).astype(np.int64),
            "session": np.zeros(n, np.int64) if session is None else np.asarray(session),
            "source": np.asarray(sources, np.int8),
            "decision": np.ones(n, bool),
            "policy_drove": np.full(n, policy),
            "action": np.zeros(n, np.int64),
            "steer_label": np.asarray(sources) == TAKE,
            "throttle_label": np.asarray(sources) == TAKE,
            "meta": {"sessions": ["a", "b"], "sources": list(pd.SOURCES)}}


def test_a_takeover_marks_the_three_seconds_before_it_and_the_two_after():
    d = drive([AUTO] * 50 + [TAKE] + [AUTO] * 49)
    near = pt.takeover_near(d, 3.0, 2.0)
    t = d["t_ns"] / 1e9
    assert near[(t >= 2.0) & (t < 5.0)].all()
    assert not near[t < 1.95].any()
    assert near[(t > 5.0) & (t <= 7.0)].all()
    assert not near[t > 7.05].any()


def test_a_takeover_in_another_session_marks_nothing():
    d = drive([AUTO] * 10 + [TAKE] * 10, session=[0] * 10 + [1] * 10)
    assert not pt.takeover_near(d, 3.0, 2.0)[:10].any()


def test_the_end_of_a_session_or_a_hole_in_it_is_not_letting_it_stand():
    d = drive([AUTO] * 100, gap_at=60)
    near = pt.takeover_near(d, 3.0, 0.0, hole_s=1.0)
    t = d["t_ns"] / 1e9
    # The 3 s before the hole (it starts after tick 59, at 5.9 s) and before the end.
    assert near[(t > 2.9) & (t <= 5.9)].all() and not near[t <= 2.85].any()
    assert near[t > t[-1] - 3.0].all()
    assert not near[(t >= 7.9) & (t < t[-1] - 3.05)].any()


def test_only_the_models_own_auto_decisions_are_accepted():
    d = drive([AUTO] * 100 + [TAKE] + [AUTO] * 100)
    d["decision"][5] = False
    d["policy_drove"][6] = False
    acc = pt.accepted_auto(d)
    assert not acc[[5, 6]].any() and not acc[100]
    assert acc[10] and acc[150]
    assert not acc[75] and not acc[105]   # just before and after my hand


def test_accepted_ticks_are_labels_only_when_asked():
    d = drive([AUTO] * 100 + [TAKE] * 5)
    steer, throttle = pt.label_masks(d, "pad")
    assert steer.sum() == throttle.sum() == 5
    d["accepted"] = pt.accepted_auto(d)
    steer, throttle = pt.label_masks(d, "pad")
    assert (steer == (d["accepted"] | (d["source"] == TAKE))).all()
    assert (throttle == steer).all() and d["accepted"].sum() > 0


def test_an_accepted_tick_weighs_its_weight_and_nothing_else_moves():
    d = drive([AUTO] * 100 + [TAKE] * 5)
    idx = np.arange(len(d["t_ns"]))
    before = pt.tick_weights(d, idx, 2.0, 3.0)
    acc = pt.accepted_auto(d)
    d["accepted_w"] = np.where(acc, 0.3, 1.0).astype(np.float32)
    w = pt.tick_weights(d, idx, 2.0, 3.0)
    assert w[acc] == pytest.approx(0.3 * before[acc])
    assert (w[~acc] == before[~acc]).all()


def test_half_the_log_weight_is_the_square_root_and_a_missing_class_stays_zero():
    w = np.array([1.0, 1.85, 3.29, 0.0])
    assert pt.soften_weights(w, 1.0) == pytest.approx(w)
    assert pt.soften_weights(w, 0.5) == pytest.approx([1.0, 1.85 ** 0.5, 3.29 ** 0.5, 0.0])
    assert pt.soften_weights(w, 0.0) == pytest.approx([1.0, 1.0, 1.0, 0.0])


def test_the_flags_default_to_the_recipe_as_it_was():
    ap_defaults = {}
    import argparse
    real = argparse.ArgumentParser.parse_args

    def grab(self, argv=None, namespace=None):
        ns = real(self, argv, namespace)
        ap_defaults.update(vars(ns))
        raise SystemExit(0)
    argparse.ArgumentParser.parse_args = grab
    try:
        with pytest.raises(SystemExit):
            pt.main(["data.npz", "out.onnx"])
    finally:
        argparse.ArgumentParser.parse_args = real
    assert ap_defaults["accepted_weight"] == 0.0
    assert ap_defaults["class_weight_power"] == 1.0


def test_the_rejected_drives_are_held_out_beside_the_scorecard():
    rejected = pt.rejected_sessions()
    assert "crawler-20261030-205939Z-p1276-001" in rejected
    held = pt.held_out_sessions()
    assert held[:len(pt.scorecard_sessions())] == pt.scorecard_sessions()
    assert set(rejected) <= set(held) and len(set(held)) == len(held)


# --- the model's own call (tools/auto_overrides.py, --accepted-own) ---------

import auto_overrides as ao  # noqa: E402


def row(reason, **kw):
    return {"autopilot": "policy", "mode": "auto", "throttle_by": "driver",
            "policy_reason": reason, **kw}


def test_only_the_models_own_gear_is_its_own_call():
    assert ao.is_own(row("model")) == (True, True)
    assert ao.is_own(row("model creep")) == (True, True)
    for rule in ("goal reached", "model floor", "rear stop", "model rear masked", "tilt veto",
                 "model tilt masked", "model commit", "standoff", "stuck"):
        assert ao.is_own(row(rule)) == (True, False), rule
    assert ao.is_own(row("model", reverse_refractory=True)) == (True, False)


def test_a_row_the_policy_did_not_drive_is_not_counted_and_an_old_row_is_own():
    assert ao.is_own(row("model", throttle_by="pad")) == (False, False)
    assert ao.is_own(row("model", autopilot="rules")) == (False, False)
    assert ao.is_own(row("model", mode="manual")) == (False, False)
    old = row("model")
    del old["policy_reason"]
    assert ao.is_own(old) == (True, True)


def test_a_tick_pairs_with_the_newest_row_not_too_old():
    rows = np.array([1.0, 1.1, 1.2, 2.0])
    j = ao.pair(rows, np.array([0.9, 1.15, 1.2, 1.5, 2.05]))
    assert j.tolist() == [-1, 1, 2, -1, 3]


def test_an_accepted_tick_the_cars_rules_made_is_dropped(tmp_path):
    d = drive([AUTO] * 100 + [TAKE] * 5)
    t = d["t_ns"] / 1e9
    own = np.ones(len(t), bool)
    own[10:20] = False
    np.savez(tmp_path / "a.npz", log_t=t, own=own, reason=np.array(["x"] * len(t)),
             layout=np.array(ao.LAYOUT))
    got, found = pt.load_accepted_own(d, str(tmp_path))
    assert found == 1
    assert not got[10:20].any() and got[:10].all() and got[20:].all()


def test_a_session_with_no_file_is_all_own(tmp_path):
    d = drive([AUTO] * 20)
    got, found = pt.load_accepted_own(d, str(tmp_path))
    assert found == 0 and got.all()


# --- Sirius's shares (--intervention-share) and set gear weights -------------

def test_the_intervention_share_is_what_it_was_asked_and_record_keeps_its_own():
    # 60 record, 20 interventions, 120 accepted.
    src = [pd.SOURCES.index(pd.SOURCE_RECORD)] * 60 + [TAKE] * 20 + [AUTO] * 120
    d = drive(src)
    d["accepted"] = np.array([False] * 80 + [True] * 120)
    mask = np.ones(len(src), bool)
    tw, aw = pt.intervention_share_weights(d, mask, 0.5)
    total = 60 + 20 * tw + 120 * aw
    assert 20 * tw / total == pytest.approx(0.5)
    assert 60 / total == pytest.approx(60 / 200)
    assert 120 * aw / total == pytest.approx(1 - 0.5 - 0.3)


def test_no_room_for_the_accepted_ticks_is_an_error():
    src = [pd.SOURCES.index(pd.SOURCE_RECORD)] * 80 + [TAKE] * 10 + [AUTO] * 10
    d = drive(src)
    d["accepted"] = np.array([False] * 90 + [True] * 10)
    with pytest.raises(ValueError):
        pt.intervention_share_weights(d, np.ones(len(src), bool), 0.5)


def test_the_new_flags_default_to_the_recipe_as_it_was():
    ap_defaults = {}
    import argparse
    real = argparse.ArgumentParser.parse_args

    def grab(self, argv=None, namespace=None):
        ns = real(self, argv, namespace)
        ap_defaults.update(vars(ns))
        raise SystemExit(0)
    argparse.ArgumentParser.parse_args = grab
    try:
        with pytest.raises(SystemExit):
            pt.main(["data.npz", "out.onnx"])
    finally:
        argparse.ArgumentParser.parse_args = real
    assert ap_defaults["gear_weights"] is None and ap_defaults["accepted_own"] is None
    assert ap_defaults["intervention_share"] is None
    assert ap_defaults["accept_after_s"] == pt.ACCEPT_AFTER_S


def test_a_shorter_window_before_a_takeover_lets_more_stand():
    d = drive([AUTO] * 100 + [TAKE] + [AUTO] * 100)
    t = d["t_ns"] / 1e9
    three, two = pt.accepted_auto(d, after_s=3.0), pt.accepted_auto(d, after_s=2.0)
    band = (t > 7.05) & (t < 7.95)
    assert two[band].all() and not three[band].any()


def test_the_gear_weights_are_square_rooted_inverse_frequency_unless_set():
    counts = np.array([900.0, 100.0, 0.0])
    w = pt.gear_class_weights(counts)
    assert w == pytest.approx([1.0, 3.0, 0.0])
    assert pt.gear_class_weights(counts, 0.5) == pytest.approx([1.0, 3.0 ** 0.5, 0.0])
    assert pt.gear_class_weights(counts, 0.5, [1.0, 1.36, 3.26]) == pytest.approx([1.0, 1.36, 3.26])
