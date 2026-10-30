"""Training against a tip: what a tip is in the data, and what the trainer does with it.

A tip is a tick whose up vector is TIP_DEG or more off the session's own
upright (the median up of the session: a car that drives all session on a
10 degree lean is not on its side). Its run-up is the lead_s before the
first tipped tick of each crossing. The trainer drops every label on both
(the pad ticks of a car lying on its side, and my hand in the second before
it went over, are not driving to copy) and puts the tip penalty on the
run-up's moving decisions, whoever drove them: the gear the car took there,
and the throttle, are made less likely. That is a training objective, the
model still decides; nothing here runs on the car.
"""

import numpy as np
import pytest

from crawler.autopilot import policy as pl
from crawler.autopilot import policy_imu as pimu
import policy_scorecard as sc
import policy_train as pt

F, V, S = (pl.ACTIONS.index(a) for a in (pl.FORWARD, pl.REVERSE, pl.STOP))
UP = pimu.FEATURES.index("up_x")
VALID = pimu.FEATURES.index("tilt_valid")


def rolled(deg, base=(-1.0, 0.0, 0.0)):
    """An up vector rolled deg about the car's forward axis (z) off base."""
    r = np.radians(deg)
    bx, by, bz = base
    return (bx * np.cos(r) - by * np.sin(r), bx * np.sin(r) + by * np.cos(r), bz)


def data_of(sessions, hz=10):
    """{session id: [tilt deg per tick]} -> the arrays tip_marks reads."""
    imu, sess, t = [], [], []
    for sid, tilts in sessions.items():
        for k, deg in enumerate(tilts):
            row = np.zeros(pimu.IMU_SIZE, np.float32)
            if deg is not None:
                row[UP:UP + 3] = rolled(deg)
                row[VALID] = 1.0
            imu.append(row)
            sess.append(sid)
            t.append(int(k * 1e9 / hz))
    n = len(t)
    return {"imu": np.array(imu), "session": np.array(sess), "t_ns": np.array(t, np.int64),
            "decision": np.ones(n, bool), "action": np.full(n, F, np.int64),
            "meta": {"sessions": [str(s) for s in sessions]}}


def test_a_tip_is_the_ticks_past_the_angle_and_its_run_up_the_second_before():
    tilts = [0] * 20 + [10, 20, 30, 40, 50, 70, 90, 90] + [0] * 10
    tipped, runup = pt.tip_marks(data_of({0: tilts}), tip_deg=45, lead_s=1.0)
    assert np.flatnonzero(tipped).tolist() == [24, 25, 26, 27]
    # 1.0 s at 10 Hz before tick 24: ticks 14..23, never the tipped ones.
    assert np.flatnonzero(runup).tolist() == list(range(14, 24))


def test_the_angle_is_off_the_sessions_own_upright_not_the_world():
    # A whole session parked on a 40 degree lean, then 15 more: not a tip.
    lean = [40] * 30 + [55] * 5
    tipped, runup = pt.tip_marks(data_of({0: lean}), tip_deg=45)
    assert not tipped.any() and not runup.any()
    # The same 15 degrees on a session that is level is not one either; 50 is.
    tipped, _ = pt.tip_marks(data_of({0: [0] * 30 + [50] * 5}), tip_deg=45)
    assert tipped.sum() == 5


def test_no_reading_is_never_a_tip_and_a_run_up_never_reaches_into_another_session():
    # Session 1 starts its clock at 0 too, so its run-up's times are session 0's.
    d = data_of({0: [0] * 30, 1: [None] * 3 + [0] * 5 + [80] * 4})
    tipped, runup = pt.tip_marks(d, tip_deg=45, lead_s=1.0)
    s = d["session"]
    assert not tipped[s == 0].any() and not runup[s == 0].any()
    assert np.flatnonzero(tipped).tolist() == [38, 39, 40, 41]
    # The ticks with no reading are in the second before it all the same.
    assert np.flatnonzero(runup).tolist() == list(range(30, 38))


def test_every_crossing_gets_its_own_run_up():
    tilts = [0] * 15 + [80] * 3 + [0] * 15 + [80] * 3
    tipped, runup = pt.tip_marks(data_of({0: tilts}), tip_deg=45, lead_s=0.5)
    assert np.flatnonzero(runup).tolist() == [10, 11, 12, 13, 14, 28, 29, 30, 31, 32]


def test_the_tip_ticks_and_their_run_up_are_never_a_label():
    tipped = np.array([0, 0, 1, 1, 0, 0], bool)
    runup = np.array([1, 1, 0, 0, 0, 0], bool)
    labelled = np.array([1, 1, 1, 1, 1, 0], bool)
    assert pt.tip_labelled(labelled, tipped, runup).tolist() == [0, 0, 0, 0, 1, 0]


def test_the_penalty_goes_on_the_run_ups_moving_decisions_outside_val_and_bad_marks():
    d = data_of({0: [0] * 20 + [80] * 3})
    tipped, runup = pt.tip_marks(d, tip_deg=45, lead_s=1.0)
    d["action"][12] = S                       # a stop in the run-up is what we want
    d["decision"][13] = False                 # a recover's machine, not a decision
    in_val = np.zeros(len(d["t_ns"]), bool)
    in_val[14] = True
    d["mark"] = np.zeros(len(d["t_ns"]), np.int8)
    d["mark"][15] = pt.pe.MARK_BAD
    rows = pt.tip_rows(d, in_val, runup)
    assert np.flatnonzero(rows).tolist() == [10, 11, 16, 17, 18, 19]
    assert not (rows & tipped).any()


def test_the_penalty_is_the_unlikelihood_of_the_gear_taken_plus_the_throttle():
    p = np.array([0.9, 0.5, 0.0])
    thr = np.array([0.4, 0.0, 0.0])
    got = pt.tip_penalty(p, thr, np)
    want = -np.log(1 - p + pt.TIP_EPS) + thr
    assert got == pytest.approx(want)
    # Surer of the gear it tipped on, or more throttle, always costs more.
    assert got[0] > got[1] > got[2]
    assert pt.tip_penalty(np.array([0.5]), np.array([0.6]), np)[0] > got[1]


def test_the_tip_arguments_are_off_by_default_and_written_to_the_sidecar():
    assert pt.tip_meta(0.0, 45.0, 1.0) == {"tip": None}
    m = pt.tip_meta(3.0, 45.0, 1.0)["tip"]
    assert m == {"weight": 3.0, "deg": 45.0, "lead_s": 1.0, "hygiene": True}
    assert pt.tip_meta(0.0, 45.0, 1.0, hygiene=True)["tip"]["weight"] == 0.0


# --- the grader on a tip ---------------------------------------------------------------

class Gear:
    """A stub model that answers the gear gear_of(i) for the i-th call."""
    contract = pl.CONTRACT_DEPTH_SMALL
    representation = pl.PROFILE
    has_mode = uses_prev = uses_imu = uses_lidar = uses_map = uses_goal = False
    chunk, ensemble_m, ensemble_link_s = 1, 0.0, pl.CHUNK_LINK_S

    def __init__(self, gear_of, throttle=0.4):
        self.gear_of, self.throttle, self.calls = gear_of, throttle, 0

    def run(self, x, prev, mode=None, **_):
        g = self.gear_of(self.calls)
        self.calls += 1
        logits = np.zeros((1, 3), np.float32)
        logits[0, g] = 5.0
        return {"logits": logits, "throttle": np.array([[self.throttle]], np.float32),
                "steer": np.array([[0.0]], np.float32)}


def tip_drive():
    d = data_of({0: [0] * 30 + [80] * 3})
    n = len(d["t_ns"])
    d["frames"] = np.full((n, 12, 32), 2000, np.uint16)
    d["frame"] = np.arange(n)
    d["mode_in"] = np.zeros(n, np.int64)
    d["frame_pitch"] = np.zeros(n, np.float32)
    return d


def test_the_tip_grade_is_the_share_of_the_run_up_still_driving_and_the_throttle_on_it(monkeypatch):
    d = tip_drive()
    monkeypatch.setattr(sc.pe, "x_input", lambda model, data, i: np.zeros((1, 1, 32), np.float32))
    # Drives to tick 24, stops from 25: half of the run-up (20..29) is still moving.
    s = sc.tip_scores(Gear(lambda i: F if i < 25 else S), d, lead_s=1.0)
    assert s["tip_ticks"] == 10 and s["tip_crossings"] == 1
    assert s["tip_go"] == pytest.approx(0.5)
    assert s["tip_thr"] == pytest.approx(0.2)
    # Away from the tip it drove every tick it was asked: calm_go keeps a
    # model that only ever stops from looking safe.
    assert s["calm_go"] == pytest.approx(1.0)
    never = sc.tip_scores(Gear(lambda i: S), d, lead_s=1.0)
    assert never["tip_go"] == 0.0 and never["calm_go"] == 0.0
