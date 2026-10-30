"""The tilt veto: on a lean against a wall, the direction the car was going
is taken off the table and the model picks from what is left.

What it must never do, each held here: steer, block both directions, latch
past its trigger, stop auto, or let the blocked direction's throttle out by
any path. pigpio holds the last pulse forever, so every cut is checked as a
0 on the wheels, never a skipped write. Every loop is a bounded range().
"""

import math

import numpy as np
import pytest

import crawler.__main__ as mn
from crawler.autopilot import driver as dv
from crawler.autopilot import footprint as fp
from crawler.autopilot import policy as pl
from crawler.hardware import lidar as ld
from crawler.telemetry import telemetry as tm

from conftest import FakeCamera, FakeController
from test_policy_map import FakeSlam, MapStub, clear

T0 = 100.0
FORWARD = (5.0, 0.0, 0.0)
REVERSE = (0.0, 5.0, 0.0)
# The model's answer is forward; the rest is a reverse it would take over a stop.
FWD_THEN_REV = (5.0, 2.0, 0.0)
# Forward, and a stop it would take over a reverse.
FWD_THEN_STOP = (5.0, 0.0, 2.0)
REV_THEN_FWD = (2.0, 5.0, 0.0)


def rolled(deg):
    """A unit up vector rolled `deg` to the left, device axes (level -x)."""
    r = math.radians(deg)
    return (-math.cos(r), math.sin(r), 0.0)


def imu(deg):
    return {"accel": (0.0, 0.0, 9.8), "gyro": (0.3, 0.0, 0.0), "up": rolled(deg)}


def side_bins(gap_mm=None, side=1):
    """Open all round, and one return `gap_mm` outside the wheels dead
    beside the car (bearing +-90), or none."""
    bins = np.full(ld.BINS, 2500, np.uint16)
    # Nothing beside the body at 2500 mm: the beam at 90 lands 2364 past the wheels.
    if gap_mm is not None:
        bins[int(round((90.0 * side) / ld.BIN_DEG)) % ld.BINS] = int(round(ld.HALF_WIDTH_MM + gap_mm))
    return bins


class Reader:
    """A real LidarReader on a pinned clock whose one scan can be swapped."""

    def __init__(self, bins):
        self.r = ld.LidarReader(clock=lambda: T0)
        self.set(bins)

    def set(self, bins):
        self.r._scan = ld.Scan(t=T0, angles_deg=np.zeros(0),
                               distances_mm=np.zeros(0, np.uint16),
                               intensities=np.zeros(0, np.uint8), speed_dps=3570.0,
                               packets=40, bins=bins)


def map_driver(model, deg=0.0, gap=None):
    reader = Reader(side_bins(gap))
    cam = FakeCamera(depth=clear(), imu=imu(deg))
    d = pl.PolicyDriver(FakeController(), cam, model, lidar=reader.r, slam=FakeSlam())
    return d, cam, reader


def tick(d, k):
    d.slam.at(T0 + k * 0.1)
    d.tick(T0 + k * 0.1)


# --- the pieces ----------------------------------------------------------------

def test_tilt_is_the_angle_off_level_and_junk_is_no_tilt():
    assert dv.tilt_deg((-1.0, 0.0, 0.0)) == pytest.approx(0.0)
    assert dv.tilt_deg(rolled(30.0)) == pytest.approx(30.0)
    pitched = (-math.cos(math.radians(25)), 0.0, math.sin(math.radians(25)))
    assert dv.tilt_deg(pitched) == pytest.approx(25.0)
    for bad in (None, (0.0, 0.0, 0.0), (math.nan, 0.0, 0.0), "abc", (1.0,)):
        assert math.isnan(dv.tilt_deg(bad))


def test_the_side_gap_is_beside_the_body_only():
    assert dv.side_gap_mm(side_bins(30)) == pytest.approx(30.0, abs=1.0)
    assert dv.side_gap_mm(side_bins(12, side=-1)) == pytest.approx(12.0, abs=1.0)
    assert dv.side_gap_mm(side_bins(None)) > 2000, "only the far room beside it"
    # Something 20 mm off the front bumper, dead ahead, is not beside it.
    ahead = np.zeros(ld.BINS, np.uint16)
    ahead[0] = int(ld.FRONT_BUMPER_X_MM + 20)
    assert dv.side_gap_mm(ahead) is None
    assert dv.side_gap_mm(None) is None and dv.side_gap_mm("junk") is None


def veto():
    lines = []
    return dv.TiltVeto(say=lines.append), lines


def test_it_engages_only_on_a_lean_with_something_at_a_side():
    v, _ = veto()
    v.note_throttle(0.3, T0)
    assert v.update(T0, rolled(25), side_bins(None)) == 0, "a lean in the open"
    assert v.update(T0 + 0.1, rolled(19), side_bins(10)) == 0, "a wall, but under 20"
    assert v.update(T0 + 0.2, rolled(21), side_bins(10)) == 1, "both: forward blocked"
    assert v.blocks(0.3) and not v.blocks(-0.6) and not v.blocks(0.0)


def test_a_gap_seen_inside_the_window_counts_and_one_older_does_not():
    v, _ = veto()
    v.note_throttle(-0.6, T0)
    v.update(T0, rolled(5), side_bins(20))
    # Away from the wall now, and leaning: the wall was 1.9 s ago.
    assert v.update(T0 + 1.9, rolled(22), side_bins(None)) == -1, "reverse blocked"
    v, _ = veto()
    v.note_throttle(-0.6, T0 + 2.0)
    v.update(T0, rolled(5), side_bins(20))
    assert v.update(T0 + 2.2, rolled(22), side_bins(None)) == 0


def test_it_blocks_the_direction_it_was_going_and_only_that_one():
    for sign in (1, -1):
        v, _ = veto()
        v.note_throttle(0.4 * sign, T0)
        v.note_throttle(0.0, T0 + 0.1)   # a 0 is not a direction
        assert v.update(T0 + 0.2, rolled(25), side_bins(5)) == sign
        # And the open direction going out does not move it, nor block it too.
        for k in range(10):
            v.note_throttle(-0.6 * sign, T0 + 0.3 + k * 0.1)
            assert v.update(T0 + 0.3 + k * 0.1, rolled(25), side_bins(5)) == sign
            assert v.blocks(0.4 * sign) and not v.blocks(-0.6 * sign)


def test_nothing_moving_is_nothing_blocked():
    v, lines = veto()
    v.note_throttle(0.4, T0)
    assert v.update(T0 + 2.5, rolled(25), side_bins(5)) == 0
    assert v.engaged and "nothing moving" in lines[-1]


def test_it_never_latches_and_the_release_is_15_degrees():
    v, lines = veto()
    v.note_throttle(0.4, T0)
    assert v.update(T0, rolled(21), side_bins(5)) == 1
    assert v.update(T0 + 0.1, rolled(16), side_bins(5)) == 1, "hysteresis: still over 15"
    assert v.update(T0 + 0.2, rolled(14.9), side_bins(5)) == 0, "released under 15"
    assert not v.engaged and "released" in lines[-1]
    # It does not come back at 16 on the way up again: 20 to engage.
    assert v.update(T0 + 0.3, rolled(18), side_bins(5)) == 0
    # The gap window running out lets go as well, and so does no tilt.
    v.note_throttle(0.4, T0 + 0.35)
    assert v.update(T0 + 0.4, rolled(25), side_bins(5)) == 1
    assert v.update(T0 + 2.3, rolled(25), side_bins(None)) == 1, "the wall 1.9 s ago"
    assert v.update(T0 + 2.5, rolled(25), side_bins(None)) == 0, "2.1 s ago"
    v.note_throttle(0.4, T0 + 2.55)
    assert v.update(T0 + 2.6, rolled(25), side_bins(5)) == 1
    assert v.update(T0 + 2.7, None, side_bins(5)) == 0, "no tilt reading"


def test_the_journal_says_engage_and_release_and_does_not_spam():
    v, lines = veto()
    t = T0
    for k in range(6):
        v.note_throttle(0.4, t)
        v.update(t, rolled(25), side_bins(8))
        v.update(t + 0.1, rolled(10), side_bins(8))
        t += 0.5
    said = [x for x in lines if x.startswith("Tilt veto - ")]
    assert len(said) == 1, lines
    assert "forward blocked" in said[0] and "tilt 25.0 deg" in said[0] and "side gap 8 mm" in said[0]
    assert sum("released" in x for x in lines) == 1, "only a release whose engage was said"
    v.note_throttle(0.4, T0 + 5.1)
    v.update(T0 + 5.1, rolled(25), side_bins(8))
    assert "5 more since the last line" in lines[-1]


def test_off_is_off():
    v, _ = veto()
    v.enabled = False
    v.note_throttle(0.4, T0)
    assert v.update(T0, rolled(30), side_bins(1)) == 0 and not v.engaged


# --- in the policy driver ---------------------------------------------------------

def lean_into_a_wall(d, cam, reader, model, logits, n=10, deg=25.0, gap=10.0):
    """Driving forward level and in the open, then a lean with a wall at the side."""
    model.say(FORWARD)
    for k in range(3):
        tick(d, k)
    assert d.controller.throttles[-1] > 0
    cam.imu = imu(deg)
    reader.set(side_bins(gap))
    model.say(logits)
    for k in range(3, 3 + n):
        tick(d, k)


def test_a_map_models_blocked_forward_becomes_its_own_next_best_reverse():
    model = MapStub(FORWARD, throttle=0.8, steer=0.5)
    d, cam, reader = map_driver(model)
    lean_into_a_wall(d, cam, reader, model, FWD_THEN_REV)
    assert all(t == dv.ESCAPE_SPEED for t in d.controller.throttles[-10:]), \
        "the open direction, at the reverse every autonomous reverse runs at"
    dec = d.last_decision
    assert dec.action == pl.REVERSE and dec.tilt_masked and dec.model_action == pl.FORWARD
    assert dec.reason == "model tilt masked"
    f = d.policy_fields()
    assert f["tilt_veto"] == "forward" and f["policy_tilt_masked"] is True
    assert f["tilt_veto_deg"] == pytest.approx(25.0, abs=0.1) and f["tilt_veto_gap_mm"] == pytest.approx(10.0, abs=1.0)
    # It levels out: forward is the model's again on the very next tick.
    cam.imu = imu(10.0)
    model.say(FORWARD)
    tick(d, 20)
    assert d.controller.throttles[-1] > 0 and d.policy_fields()["tilt_veto"] == ""


def test_a_map_model_that_would_rather_stop_stops_with_its_steering_out():
    model = MapStub(FORWARD, throttle=0.8, steer=0.5)
    d, cam, reader = map_driver(model)
    lean_into_a_wall(d, cam, reader, model, FWD_THEN_STOP)
    assert all(t == 0 for t in d.controller.throttles[-10:]), "a 0 on the wheels"
    assert d.last_decision.reason == pl.TILT_REASON and d.last_decision.action == pl.STOP
    assert d.controller.steerings[-1] != (0, 0), "the steering is the model's, not centred"
    # The moment the model would reverse, it does: nothing else is held.
    model.say((5.0, 3.0, 2.0))
    tick(d, 30)
    assert d.controller.throttles[-1] == dv.ESCAPE_SPEED


def test_the_open_direction_is_fully_the_models():
    """Reverse blocked, a model asking for forward: forward exactly as unvetoed."""
    model = MapStub(REVERSE, throttle=1.0, steer=-0.4)
    d, cam, reader = map_driver(model)
    for k in range(3):
        tick(d, k)
    assert d.controller.throttles[-1] < 0
    cam.imu = imu(24.0)
    reader.set(side_bins(15))
    model.say(FORWARD, throttle=0.6)
    tick(d, 3)
    assert d.policy_fields()["tilt_veto"] == "reverse"
    ref, _, _ = map_driver(MapStub(FORWARD, throttle=0.6, steer=-0.4))
    tick(ref, 0)
    assert ref.controller.throttles[-1] > 0
    assert d.controller.throttles[-1] == pytest.approx(ref.controller.throttles[-1])
    assert not d.last_decision.tilt_masked


@pytest.mark.parametrize("seed", range(3))
def test_the_blocked_directions_throttle_never_goes_out(seed):
    """Whatever the model says, tick after tick: while a direction is
    blocked no throttle of its sign is commanded, and the other sign is."""
    rng = np.random.default_rng(seed)
    model = MapStub(FORWARD, throttle=0.8, steer=0.2)
    d, cam, reader = map_driver(model)
    lean_into_a_wall(d, cam, reader, model, FORWARD, n=1)
    blocked = d.tilt_veto.blocked
    assert blocked == 1
    other = 0
    for k in range(4, 80):
        model.say(tuple(rng.normal(0, 3, 3)), throttle=float(rng.uniform(0.1, 1.0)))
        tick(d, k)
        thr = d.controller.throttles[-1]
        assert d.tilt_veto.blocked == blocked, "one direction, the same one, until release"
        assert thr * blocked <= 0, f"tick {k}: {thr} went out the blocked way"
        other += thr * blocked < 0
    assert other > 10, "and the other direction does go out"


def test_my_trigger_is_never_vetoed_and_sets_the_direction():
    model = MapStub(FORWARD, throttle=0.8)
    d, cam, reader = map_driver(model)
    tick(d, 0)
    # I am reversing along the wall, then it leans; the model would go forward.
    d.human_throttle, d.pad_throttle = True, -0.5
    tick(d, 1)
    cam.imu = imu(25.0)
    reader.set(side_bins(5))
    model.say(REVERSE)
    for k in range(2, 4):
        tick(d, k)
        assert not d.last_decision.tilt_masked, "the throttle is mine"
        assert d.last_decision.reason != pl.TILT_REASON
    assert d.tilt_veto.blocked == -1, "the way my trigger was going"
    model.say(FORWARD)
    # I let go: reverse is blocked, the model's forward is its own.
    d.human_throttle, d.pad_throttle = False, None
    tick(d, 4)
    assert d.controller.throttles[-1] > 0 and not d.last_decision.tilt_masked
    model.say(REV_THEN_FWD)
    tick(d, 5)
    assert d.controller.throttles[-1] > 0 and d.last_decision.tilt_masked


def test_it_never_parks_auto_and_a_broken_reading_lets_go():
    model = MapStub(FORWARD, throttle=0.8)
    d, cam, reader = map_driver(model)
    lean_into_a_wall(d, cam, reader, model, FWD_THEN_REV, n=3)
    assert d.tilt_veto.blocked == 1
    cam.imu = {"accel": (0, 0, 9.8), "gyro": (0, 0, 0), "up": ("x", None, 1)}
    model.say(FORWARD)
    tick(d, 10)
    assert d.tilt_veto.blocked == 0 and d.controller.throttles[-1] > 0

    def boom(*a, **k):
        raise RuntimeError("bad scan")
    d2, cam2, reader2 = map_driver(MapStub(FORWARD))
    d2.tilt_veto.update = boom
    tick(d2, 0)     # does not raise: the tick drives on, unvetoed
    assert d2.controller.throttles[-1] > 0


def test_the_hand_rules_commit_ends_when_reverse_is_blocked():
    """An old model's reverse commit holds reverse 0.7 s whatever it says;
    a blocked reverse ends it on the tick, and the forward is its own."""
    from test_rear_stop import StubModel
    model = StubModel(REVERSE, throttle=1.0)
    cam = FakeCamera(depth=clear(), imu=imu(0.0))
    reader = Reader(side_bins(None))
    d = pl.PolicyDriver(FakeController(), cam, model, lidar=reader.r)
    assert d.hand_rules
    d.tick(T0)
    assert d.reverse_commit and d.controller.throttles[-1] < 0
    cam.imu = imu(25.0)
    reader.set(side_bins(5))
    model.say(FORWARD)
    d.tick(T0 + 0.1)
    assert not d.reverse_commit, "the commit ended"
    assert d.controller.throttles[-1] > 0 and d.policy_fields()["tilt_veto"] == "reverse"


def test_the_hand_rules_go_the_other_way_never_stop():
    from test_rear_stop import StubModel
    model = StubModel(FORWARD, throttle=1.0)
    cam = FakeCamera(depth=clear(), imu=imu(0.0))
    reader = Reader(side_bins(None))
    d = pl.PolicyDriver(FakeController(), cam, model, lidar=reader.r)
    d.tick(T0)
    cam.imu = imu(25.0)
    reader.set(side_bins(5))
    d.tick(T0 + 0.1)
    assert d.controller.throttles[-1] == dv.ESCAPE_SPEED and d.last_decision.tilt_masked


# --- the row and the switch ---------------------------------------------------------

def test_the_row_carries_the_veto():
    props = tm.JSON_SCHEMAS["footprint"]["schema"]["properties"]
    for key, kind in (("tilt_veto", "string"), ("tilt_veto_deg", "number"),
                      ("tilt_veto_gap_mm", "number"), ("policy_tilt_masked", "boolean")):
        assert props[key]["type"] == kind
        assert key in fp.AUTOPILOT_FIELDS
    d, _, _ = map_driver(MapStub(FORWARD))
    tick(d, 0)
    row = fp.autopilot_fields(d)
    assert row["tilt_veto"] == "" and row["policy_tilt_masked"] is False


def test_on_by_default_and_a_flag_turns_it_off():
    assert mn.parse_args([]).tilt_veto is True
    assert mn.parse_args(["--no-tilt-veto"]).tilt_veto is False
    assert mn.AutoPilot(None, None).tilt_veto is True
    assert mn.AutoPilot(None, None, tilt_veto=False).tilt_veto is False
    assert dv.TiltVeto().enabled is True


def hand_rules_driver(logits):
    from test_rear_stop import StubModel
    model = StubModel(logits, throttle=1.0)
    cam = FakeCamera(depth=clear(), imu=imu(0.0))
    reader = Reader(side_bins(None))
    return pl.PolicyDriver(FakeController(), cam, model, lidar=reader.r), cam, reader


def test_a_back_out_into_a_blocked_reverse_ends_at_0():
    """The hand rules' STUCK back-out is a scripted reverse: a lean that
    blocks it ends it on the tick, at 0, and nothing puts it back."""
    d, cam, reader = hand_rules_driver(FORWARD)
    d.tick(T0)
    d.stuck = lambda imu, now: True
    d.tick(T0 + 0.1)
    d.stuck = lambda imu, now: False
    d.tick(T0 + 0.2)
    assert d._phase == dv._PHASE_REVERSE and d.controller.throttles[-1] < 0
    cam.imu = imu(25.0)
    reader.set(side_bins(5))
    d.tick(T0 + 0.3)
    assert d.tilt_veto.blocked == -1
    assert d._phase != dv._PHASE_REVERSE and d._backout_exit == dv.EXIT_TILT
    for k in range(3, 30):
        assert d.controller.throttles[-1] >= 0, k
        d.tick(T0 + (k + 1) * 0.1)


def test_a_stuck_with_reverse_blocked_does_not_back_out():
    d, cam, reader = hand_rules_driver(FORWARD)
    d.tilt_veto.note_throttle(-0.6, T0)
    cam.imu = imu(25.0)
    reader.set(side_bins(5))
    d.stuck = lambda imu, now: True
    d.tick(T0 + 0.1)
    assert d.tilt_veto.blocked == -1
    assert d._phase is None and d.controller.throttles[-1] == 0


def test_the_last_word_holds_whatever_path_brings_the_blocked_direction(monkeypatch):
    """If anything upstream ever hands the blocked direction back, the net
    before the write still puts a 0 on the wheels."""
    monkeypatch.setattr(pl, "tilt_open_action",
                        lambda outputs, blocked, hand_rules=False: blocked)
    model = MapStub(FORWARD, throttle=0.8, steer=0.5)
    d, cam, reader = map_driver(model)
    lean_into_a_wall(d, cam, reader, model, FWD_THEN_REV, n=5)
    assert all(t == 0 for t in d.controller.throttles[-5:])
    assert d.last_decision.reason == pl.TILT_REASON
