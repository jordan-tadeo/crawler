"""The stuck check. This is the bug that shipped, so this is the file that matters.

What the journal showed: "Switched to auto" and "STUCK - backing up to
recover." stamped on the same second, every time, and then STUCK again at :47,
:52, :57, :02, :07 -- a flat five-second metronome on a car that was driving
along perfectly happily.

Three separate faults produced that, and there is a test for each:

  1. is_moving() returned False for a missing IMU sample and the caller read
     False as "not moving". An absent packet and a car wedged in a doorway
     gave the same command. motion_evidence() has a third answer now and
     UNKNOWN must never walk the stuck counter.

  2. The check did not care whether the car had been asked to move. A motion
     check on a car nobody is commanding measures nothing.

  3. last_recovery_time was stamped on every false positive, so the cooldown
     re-armed on a metronome rather than on a stick actually confirmed.

Everything here is arithmetic on numbers and numpy arrays. No OAK-D, no pigpio.
"""

import pathlib
import time

import numpy as np
import pytest

from crawler.autopilot import driver as dv


# --- fixtures ---------------------------------------------------------------

def clear_depth(mm=3000):
    """A depth frame with nothing in front of the car.

    Uniform and far, so obstacle_in_front is False and every pixel is valid --
    the centre patch is fully populated, which matters because that function
    fails closed on a sparse patch.
    """
    return np.full((400, 640), mm, dtype=np.uint16)


def blocked_depth(mm=200):
    return np.full((400, 640), mm, dtype=np.uint16)


def imu(gyro, accel=(0.0, 0.0, 0.0)):
    return {"accel": accel, "gyro": gyro}


STILL_IMU = imu(gyro=(0.002, 0.002, 0.002))          # inside the measured parked floor
MOVING_IMU = imu(gyro=(0.5, 0.0, 0.0))               # well over GYRO_MOVING_MIN


# The `d` fixture -- a Driver with a recording controller and an empty camera
# under it -- comes from tests/conftest.py.


# --- motion_evidence --------------------------------------------------------

def test_a_missing_imu_sample_is_unknown_not_still(d):
    """Fault 1, at the source. This returning STILL is what made a dropped
    packet indistinguishable from a buried wheel."""
    assert d.motion_evidence(None) == dv.UNKNOWN


@pytest.mark.parametrize("malformed", [{}, {"accel": (0, 0, 0)}, {"gyro": (0, 0, 0)}, 7, "nope"])
def test_a_malformed_imu_sample_is_unknown(d, malformed):
    assert d.motion_evidence(malformed) == dv.UNKNOWN


def test_a_shaking_chassis_reads_as_moving(d):
    assert d.motion_evidence(MOVING_IMU) == dv.MOVING


def test_the_measured_parked_floor_reads_as_still(d):
    """Parked, measured on the car: gyro magnitude max 0.0069 rad/s. That has
    to come out STILL or the stuck check can never fire at all."""
    assert d.motion_evidence(imu(gyro=(0.004, 0.004, 0.004))) == dv.STILL


def test_the_gyro_threshold_clears_the_measured_parked_maximum(d):
    """GYRO_MOVING_MIN sits between the parked max and the rolling p1. If
    someone drops it to the floor itself, a parked car starts reading as
    moving."""
    assert dv.GYRO_MOVING_MIN > 0.0069


def test_the_slowest_measured_creep_reads_as_moving(d):
    """The rolling end, measured off the 2026-10-27 run: the p1 of the
    five-sample gyro mean while armed at forward_speed 0.2 was 0.0146 rad/s,
    with the accel at its own p1 of 0.106 m/s^2. The old 0.035 threshold read
    that as STILL -- it read a quarter of all honest creeping as STILL -- and
    that is where four STUCK reverses on a moving car came from."""
    creeping = imu(gyro=(0.0146, 0.0, 0.0), accel=(0.106, 0.0, 0.0))
    for _ in range(d.imu_window_size):
        evidence = d.motion_evidence(creeping)
    assert evidence == dv.MOVING


def test_a_creeping_car_is_never_called_stuck(d):
    """The same thing through stuck(), over the length it takes to fire: the
    slowest two seconds before any of the run's four STUCKs never had a window
    under 0.0142 rad/s, with the accel term underneath stuck_threshold the
    whole way. Armed, cooled down, thirty ticks of it -- three times the run of
    STILL a stick needs -- and it must not fire once."""
    creeping = imu(gyro=(0.0142, 0.0, 0.0), accel=(0.16, 0.0, 0.0))
    d._throttle_since = 0.0
    for k in range(dv.STUCK_CONFIRM_SAMPLES * 3):
        assert d.stuck(creeping, 100.0 + 0.1 * k) is False, f"fired on tick {k}"


def test_the_accelerometer_is_a_second_opinion_not_the_primary_test(d):
    """LINEAR_ACCELERATION is gravity compensated, so it reads the parked floor
    at a steady creep too. A hard knock still counts as motion."""
    assert d.motion_evidence(imu(gyro=(0.0, 0.0, 0.0), accel=(5.0, 0.0, 0.0))) == dv.MOVING


def test_an_unknown_sample_does_not_pollute_the_smoothing_window(d):
    """A rejected sample must not land in the history, or the next real
    reading is a blend with a packet that never existed."""
    d.motion_evidence(MOVING_IMU)
    before = list(d.gyro_history)
    d.motion_evidence(None)
    assert list(d.gyro_history) == before


# --- stuck: fault 1, UNKNOWN must never count -------------------------------

def test_unknown_readings_can_never_confirm_a_stick(d):
    """Fault 1, at the consequence, and the single most important test here.

    Armed, cooldown long expired, and nothing but missing packets. If UNKNOWN
    is allowed to walk the counter this fires on the tenth tick.
    """
    d._throttle_since = 0.0
    for _ in range(STICK := 50):
        assert d.stuck(None, 100.0) is False
    assert d._still_samples == 0


def test_an_unknown_in_the_middle_holds_the_count_rather_than_adding_to_it(d):
    d._throttle_since = 0.0
    for _ in range(5):
        d.stuck(STILL_IMU, 100.0)
    assert d._still_samples == 5

    for _ in range(20):
        d.stuck(None, 100.0)
    assert d._still_samples == 5


# --- stuck: fault 2, it must be commanded to move ---------------------------

def test_a_car_nobody_asked_to_move_is_never_stuck(d):
    """Fault 2. _throttle_since is None whenever the driver is not asking the
    car to go anywhere, and a motion check then measures nothing."""
    assert d._throttle_since is None
    for _ in range(50):
        assert d.stuck(STILL_IMU, 100.0) is False


def test_the_check_does_not_arm_until_throttle_has_been_up_long_enough(d):
    """A car just handed the wheels has not been given a chance to move. The
    old code declared it stuck on the very first tick of auto mode."""
    d._throttle_since = 100.0
    for _ in range(50):
        assert d.stuck(STILL_IMU, 100.0 + dv.STUCK_ARM_SECONDS - 0.01) is False


def test_the_counter_is_held_down_while_the_check_is_unarmed(d):
    """Unarmed must not quietly bank samples that fire the instant it arms."""
    d._throttle_since = 100.0
    for _ in range(50):
        d.stuck(STILL_IMU, 100.1)
    assert d._still_samples == 0


# --- stuck: the confirmed case ----------------------------------------------

def test_a_confirmed_stick_takes_a_full_run_of_still_samples(d):
    d._throttle_since = 0.0
    for i in range(dv.STUCK_CONFIRM_SAMPLES - 1):
        assert d.stuck(STILL_IMU, 100.0) is False, f"fired early on sample {i + 1}"
    assert d.stuck(STILL_IMU, 100.0) is True


def test_one_moving_reading_resets_the_whole_run(d):
    """Evidence has to be consecutive. A car that moves at all is not stuck."""
    d._throttle_since = 0.0
    for _ in range(dv.STUCK_CONFIRM_SAMPLES - 1):
        d.stuck(STILL_IMU, 100.0)
    assert d.stuck(MOVING_IMU, 100.0) is False
    assert d._still_samples == 0
    assert d.stuck(STILL_IMU, 100.0) is False


def test_a_confirmed_stick_clears_its_own_evidence(d):
    """Otherwise the tick after a recover starts one sample from firing again."""
    d._throttle_since = 0.0
    for _ in range(dv.STUCK_CONFIRM_SAMPLES):
        d.stuck(STILL_IMU, 100.0)
    assert d._still_samples == 0


def test_the_cooldown_blocks_a_second_recover(d):
    d._throttle_since = 0.0
    d.last_recovery_time = 99.0
    for _ in range(dv.STUCK_CONFIRM_SAMPLES * 3):
        assert d.stuck(STILL_IMU, 100.0) is False


def test_confirming_a_stick_costs_a_second_of_throttle_and_a_second_of_stillness(d):
    """STUCK_CONFIRM_SAMPLES is a count of ticks and not a wall-clock deadline
    on purpose: the loop does not always make its rate, and a count can only
    make confirming a stick take longer in real time, never fire early."""
    assert dv.STUCK_CONFIRM_SAMPLES == 10          # a second at the 10 Hz tick
    assert dv.STUCK_ARM_SECONDS == 1.0


# --- fault 3: last_recovery_time is only stamped on a real stick ------------

def test_driving_along_happily_never_stamps_the_recovery_clock(d):
    """Fault 3, through tick(), which is where the stamping lives.

    Clear depth, no IMU packets at all -- the exact condition that used to
    produce the five-second metronome. The clock must still read zero after
    a long run of it.
    """
    d.camera.depth = clear_depth()
    d.camera.imu = None

    now = 100.0
    for _ in range(200):
        d.tick(now)
        now += 0.1

    assert d.last_recovery_time == 0.0
    assert d.controller.throttles[-1] == d.forward_speed


def test_a_moving_car_never_stamps_the_recovery_clock(d):
    d.camera.depth = clear_depth()
    d.camera.imu = MOVING_IMU

    now = 100.0
    for _ in range(200):
        d.tick(now)
        now += 0.1

    assert d.last_recovery_time == 0.0


def test_a_genuinely_stuck_car_does_stamp_it_and_recovers(d):
    """The other half: having gated it three ways, it still has to fire."""
    d.camera.depth = clear_depth()
    d.camera.imu = STILL_IMU

    now = 100.0
    for _ in range(60):
        d.tick(now)
        now += 0.1

    assert d.last_recovery_time > 0.0
    assert any(t < 0 for t in d.controller.throttles)


# --- throttle bookkeeping ---------------------------------------------------

def test_forward_throttle_stamps_when_it_was_first_commanded(d):
    d.camera.depth = clear_depth()
    d.camera.imu = None

    d.tick(100.0)
    assert d._throttle_since == 100.0

    d.tick(100.1)
    assert d._throttle_since == 100.0, "an unbroken run must keep its start time"


def test_losing_depth_stops_the_car_and_forgets_the_evidence(d):
    """None is a real return: no packet waiting, or the X_LINK error path.
    pigpio holds the last pulse width forever, so doing nothing means
    continuing to drive."""
    d.camera.depth = clear_depth()
    d.camera.imu = STILL_IMU
    d.tick(100.0)
    assert d._throttle_since is not None

    d.camera.depth = None
    d.tick(100.1)

    assert d.controller.throttles[-1] == 0
    assert d._throttle_since is None
    assert d._still_samples == 0
    assert d.gyro_history == []


# --- obstacle_in_front fails closed ----------------------------------------

# Truthiness and not `is True` on the next two: the normal path ends in
# `np.mean(...) > 0.5`, which hands back a numpy bool_ rather than Python's
# True, while the fail-closed path returns a literal. Both are correct for an
# `if`, so the tests ask the question an `if` asks.

def test_a_wall_in_front_is_an_obstacle(d):
    assert d.obstacle_in_front(blocked_depth())


def test_clear_space_is_not_an_obstacle(d):
    assert not d.obstacle_in_front(clear_depth())


def test_an_unmeasurable_centre_patch_fails_closed(d):
    """Dropping the zero pixels and then taking a mean used to hand np.mean()
    an empty array, which is nan, and nan > 0.5 is False -- so a patch with
    nothing measurable in it read as clear and the car drove into it."""
    assert d.obstacle_in_front(np.zeros((400, 640), dtype=np.uint16)) is True


def test_a_mostly_unmeasurable_centre_patch_fails_closed(d):
    """Under a tenth valid is not enough to call it clear."""
    depth = np.zeros((400, 640), dtype=np.uint16)
    depth[200, 320] = 3000
    assert d.obstacle_in_front(depth) is True


def test_obstacle_detection_decides_before_it_commands(d):
    """The old order commanded forward speed and only then looked, so every
    tick facing a wall lurched into it for a tick first."""
    d.camera.depth = blocked_depth()
    d.camera.imu = None

    d.tick(100.0)

    assert d.forward_speed not in d.controller.throttles
    assert d.controller.throttles[-1] < 0


# --- the recover phase machine ---------------------------------------------

def test_a_recover_never_blocks_and_gives_the_wheels_back(d):
    """The old recover() awaited 2.5 seconds with the throttle pinned in
    reverse, and for those 2.5 seconds nothing polled the stop signal."""
    d.camera.depth = blocked_depth()
    d.begin_recover(100.0)
    assert d.recovering(100.0) is True
    assert d.recovering(100.0 + dv.REVERSE_SECONDS) is True          # into settle
    assert d._backout_exit == dv.EXIT_CAP
    assert d.recovering(100.0 + dv.REVERSE_SECONDS + dv.SETTLE_SECONDS) is False
    assert d._phase is None


def test_the_back_out_side_is_my_rule_not_a_coin(d):
    """It used to be random.choice of four angles. Now it is the side I
    reverse on myself: opposite the last forward turn, left with none, and
    the same answer every time it is asked the same question."""
    sides = set()
    for _ in range(50):
        d.abort()
        d.begin_recover(100.0)
        sides.add(d._turn_dir)
    assert sides == {dv.REVERSE_DEFAULT_SIDE * d.max_turn}

    d.abort()
    d._note_turn(0.6, 99.5)                 # turned right half a second ago
    d.begin_recover(100.0)
    assert d._turn_dir == pytest.approx(-d.max_turn)

    d.abort()
    d._note_turn(-0.6, 99.5)                # turned left
    d.begin_recover(100.0)
    assert d._turn_dir == pytest.approx(d.max_turn)

    d.abort()
    d._note_turn(-0.6, 100.0 - dv.REVERSE_TURN_WINDOW_S - 0.1)   # too long ago
    d.begin_recover(100.0)
    assert d._turn_dir == pytest.approx(dv.REVERSE_DEFAULT_SIDE * d.max_turn)


def test_the_back_out_side_comes_off_the_rules_own_last_turn(d):
    """Through tick(): an evade to the right, then a wall, reverses left."""
    d.camera.imu = MOVING_IMU
    d.camera.depth = probe_scene(EVADE_LEFT)
    d.tick(100.0)
    assert d._evade_dir == 1.0
    d.camera.depth = blocked_depth()
    d.tick(100.1)
    assert d._phase == dv._PHASE_REVERSE
    assert d._turn_dir == pytest.approx(-d.max_turn)
    assert d._backout_reason == dv.BACKOUT_OBSTACLE


def test_reverse_side_ignores_a_nan(d):
    assert dv.reverse_side(float("nan"), 0.1) == dv.REVERSE_DEFAULT_SIDE
    assert dv.reverse_side(None, 0.0, float("nan")) == dv.REVERSE_DEFAULT_SIDE
    assert dv.reverse_side(0.5, float("nan")) == dv.REVERSE_DEFAULT_SIDE


def test_abort_drops_a_recover_mid_reverse_and_stops(d):
    d.begin_recover(100.0)
    d.abort()
    assert d.controller.throttles[-1] == 0
    assert d.recovering(100.1) is False


def test_abort_is_safe_to_call_twice(d):
    d.begin_recover(100.0)
    d.abort()
    d.abort()
    assert d._phase is None


def test_a_recover_hands_back_leaning_toward_the_side_that_opened(d):
    """Not hard-centred: the car reversed on one side, so its nose swung to
    the other, and the wheels come back leaning that way, counter-phase, at
    HANDBACK_STEER of max_turn -- never the reverse's own lock."""
    d.camera.depth = blocked_depth()
    d.begin_recover(100.0)
    side = d._backout_side
    d.recovering(100.0 + dv.REVERSE_SECONDS)
    d.recovering(100.0 + dv.REVERSE_SECONDS + dv.SETTLE_SECONDS)
    lean = -side * dv.HANDBACK_STEER * d.max_turn
    assert d.controller.steerings[-1] == pytest.approx(dv.steering_pair(lean))
    assert abs(lean) < d.max_turn


def test_the_hand_back_lean_decays_to_nothing(d):
    """Through tick() on open floor after a back-out: the first forward leans
    the full HANDBACK_STEER, it shrinks every decision, and HANDBACK_S later
    the cruise is straight again."""
    d.camera.imu = MOVING_IMU
    d.camera.depth = blocked_depth()
    d.begin_recover(100.0)
    d.camera.depth = clear_depth()
    now = recover_to_handback(d, 100.0, d.tick)
    fronts = []
    for _ in range(int(dv.HANDBACK_S * 10) + 3):
        fronts.append(d.controller.steerings[-1][0])
        now = round(now + 0.1, 3)
        d.tick(now)
    side = -d._backout_side
    assert fronts[0] == pytest.approx(side * dv.HANDBACK_STEER * d.max_turn)
    assert all(abs(b) <= abs(a) + 1e-12 for a, b in zip(fronts, fronts[1:]))
    assert fronts[-1] == 0.0


# --- the back-out: slow, watched, escalating ---------------------------------

def test_a_back_out_never_commands_full_reverse(d):
    """The whole complaint. Every reverse it puts on the wheels is a fraction
    of ESCAPE_SPEED, which sits well clear of -1.0, and it eases in rather
    than stepping straight to it."""
    d.camera.depth = blocked_depth()
    d.camera.imu = MOVING_IMU
    d.begin_recover(100.0)
    for i in range(1, 40):
        d.recovering(100.0 + i * 0.1)
    reverses = [t for t in d.controller.throttles if t < 0]
    assert reverses
    assert min(reverses) == pytest.approx(dv.ESCAPE_SPEED)
    assert dv.ESCAPE_SPEED > -0.9
    assert reverses[0] > dv.ESCAPE_SPEED, "it stepped straight to the escape speed"
    ramp = reverses[:dv.ESCAPE_RAMP_DECISIONS]
    assert ramp == sorted(ramp, reverse=True)


def test_the_escape_speed_still_moves_the_car(d):
    """Slow, not stalled: past the reverse floor my drives bracket, and never
    under the policy's own reverse floor."""
    from crawler.autopilot import policy as pl
    assert abs(dv.ESCAPE_SPEED) > dv.REVERSE_MOVING_FLOOR
    assert abs(dv.ESCAPE_SPEED) >= pl.MOVING_FLOOR_REVERSE


def test_an_esc_buzz_is_not_a_reverse_that_is_moving(d):
    """Held at a reverse too slow to move it, the chassis buzzes over
    GYRO_MOVING_MIN; the bumper judges against its own line and still fires.
    Real reversing, well over that line, never does."""
    buzz = imu(gyro=(0.03, 0.0, 0.0))
    assert d.motion_evidence(buzz) == dv.MOVING
    d.camera.depth = blocked_depth()
    d.begin_recover(100.0)
    assert reverse_for(d, ticks=30, imu_sample=buzz) is not None
    assert d._backout_exit == dv.EXIT_BUMPER
    d.abort()
    d.begin_recover(200.0)
    reversing = imu(gyro=(0.33, 0.0, 0.0))
    assert reverse_for(d, ticks=19, imu_sample=reversing, start=200.0) is None


def open_after(d, blocked_ticks, imu_sample=MOVING_IMU, reason=dv.BACKOUT_OBSTACLE):
    """A back-out that sees a wall for `blocked_ticks` decisions and then
    open floor. Returns the decision the reverse ended on, or None."""
    d.camera.imu = imu_sample
    d.camera.depth = blocked_depth()
    d.begin_recover(100.0, reason)
    for i in range(1, 40):
        if i > blocked_ticks:
            d.camera.depth = clear_depth()
        d.recovering(round(100.0 + i * 0.1, 3))
        if d._phase != dv._PHASE_REVERSE:
            return i
    return None


def test_a_back_out_ends_when_the_way_forward_opens(d):
    """Not a timer: open for BACKOUT_OPEN_DECISIONS decisions in a row and it
    stops, well before the cap, with throttle 0 on that decision."""
    ended = open_after(d, blocked_ticks=6)
    assert ended == 6 + dv.BACKOUT_OPEN_DECISIONS
    assert ended * 0.1 < dv.REVERSE_SECONDS
    assert d._backout_exit == dv.EXIT_OPENED
    assert d.controller.throttles[-1] == 0
    assert d._phase == dv._PHASE_SETTLE


def test_one_open_frame_is_not_an_exit(d):
    """Open, wall, open: the count is consecutive, so a flicker of an opening
    starts again from nothing."""
    d.camera.imu = MOVING_IMU
    d.begin_recover(100.0)
    frames = [blocked_depth()] * 6 + [clear_depth()] * (dv.BACKOUT_OPEN_DECISIONS - 1) \
        + [blocked_depth()] + [clear_depth()] * (dv.BACKOUT_OPEN_DECISIONS - 1)
    for i, frame in enumerate(frames, start=1):
        d.camera.depth = frame
        d.recovering(round(100.0 + i * 0.1, 3))
        assert d._phase == dv._PHASE_REVERSE, i


def test_an_open_front_still_reverses_for_the_minimum(d):
    """Open from the first frame: it still backs out BACKOUT_MIN_S, so it
    does not twitch back and straight in again."""
    ended = open_after(d, blocked_ticks=0)
    assert d._backout_exit == dv.EXIT_OPENED
    assert ended * 0.1 >= dv.BACKOUT_MIN_S - 1e-9
    assert ended * 0.1 < dv.BACKOUT_MIN_S + 0.1 + 1e-9


def test_a_stick_reverses_longer_on_an_open_front(d):
    """STUCK is the one the camera did not see, so an open fan says nothing
    about whether the car is free: it gets BACKOUT_STUCK_MIN_S."""
    ended = open_after(d, blocked_ticks=0, reason=dv.BACKOUT_STUCK)
    assert ended * 0.1 >= dv.BACKOUT_STUCK_MIN_S - 1e-9
    assert dv.BACKOUT_STUCK_MIN_S > dv.BACKOUT_MIN_S


def test_a_back_out_that_never_opens_stops_at_the_cap(d):
    ended = open_after(d, blocked_ticks=100)
    assert ended == pytest.approx(dv.REVERSE_SECONDS * 10)
    assert d._backout_exit == dv.EXIT_CAP
    assert d.controller.throttles[-1] == 0


def test_the_rear_bumper_still_cuts_a_back_out(d):
    ended = open_after(d, blocked_ticks=100, imu_sample=STILL_IMU)
    assert d._backout_exit == dv.EXIT_BUMPER
    assert ended == d.imu_window_size + dv.REVERSE_STALL_SAMPLES - 1
    assert d.controller.throttles[-1] == 0


def test_a_lost_frame_mid_back_out_is_neutral(d):
    """Stale camera is neutral, as everywhere: a car that cannot see does not
    keep backing up on the strength of the last picture."""
    d.camera.imu = MOVING_IMU
    d.camera.depth = blocked_depth()
    d.begin_recover(100.0)
    d.recovering(100.1)
    d.camera.depth = None
    d.recovering(100.2)
    assert d._backout_exit == dv.EXIT_NO_FRAME
    assert d.controller.throttles[-1] == 0
    assert d._phase == dv._PHASE_SETTLE
    d.recovering(100.3)
    assert d.controller.throttles[-1] == 0


def test_abort_drops_a_back_out_inside_one_tick(d):
    """START, a pad dropout and SIGTERM all land on abort(): one call, throttle
    0, and the next decision commands no reverse at all."""
    d.camera.imu = MOVING_IMU
    d.camera.depth = blocked_depth()
    d.begin_recover(100.0)
    d.recovering(100.1)
    assert d.controller.throttles[-1] < 0
    d.abort()
    assert d.controller.throttles[-1] == 0
    assert d._phase is None
    assert d._backout_exit == dv.EXIT_ABORT
    n = len(d.controller.throttles)
    assert d.recovering(100.2) is False
    assert len(d.controller.throttles) == n


MAX_BACKOUT_TICKS = round(
    (dv.ESCALATED_REVERSE_SECONDS + dv.SETTLE_SECONDS) / 0.1) + 5


def recover_to_handback(d, now, step=None):
    """Tick a back-out at 10 Hz until it hands the wheels back. Bounded: the
    longest one is the escalated cap plus the settle, so a back-out that never
    ends fails here instead of hanging the suite."""
    step = step or d.recovering
    for _ in range(MAX_BACKOUT_TICKS):
        if d._phase is None:
            return now
        now = round(now + 0.1, 3)
        step(now)
    raise AssertionError("the back-out never handed the wheels back")


def run_back_out(d, start, blocked=True):
    """One whole back-out from `start` against a wall that never opens, to
    the hand-back. Returns the time the wheels came back."""
    d.camera.imu = MOVING_IMU
    d.camera.depth = blocked_depth() if blocked else clear_depth()
    d.begin_recover(start)
    return recover_to_handback(d, start)


def test_a_back_out_at_the_same_spot_escalates(d):
    """Back again soon after the last one: the next reverses longer, and the
    one after that takes the other side, a three-point turn instead of the
    same move forever."""
    back = run_back_out(d, 100.0)
    first = (d._backout_side, d._backout_reverse_s, d._backout_level)
    assert first[2] == 0

    back = run_back_out(d, back + 1.0)
    second = (d._backout_side, d._backout_reverse_s, d._backout_level)
    assert second[2] == 1
    assert second[1] > first[1]
    assert second[1] == pytest.approx(dv.ESCALATED_REVERSE_SECONDS)

    back = run_back_out(d, back + 1.0)
    third = (d._backout_side, d._backout_reverse_s, d._backout_level)
    assert third[2] == 2
    assert third[0] == -second[0]

    run_back_out(d, back + 1.0)
    assert d._backout_level == 3
    assert d._backout_side == -third[0]


def test_a_back_out_somewhere_else_starts_from_scratch(d):
    back = run_back_out(d, 100.0)
    run_back_out(d, back + 1.0)
    assert d._backout_level == 1
    back = run_back_out(d, 200.0)
    assert d._backout_level == 0
    assert d._backout_reverse_s == pytest.approx(dv.REVERSE_SECONDS)


def test_abort_forgets_the_spot(d):
    back = run_back_out(d, 100.0)
    d.abort()
    run_back_out(d, back + 1.0)
    assert d._backout_level == 0


def test_backout_opened_reuses_the_rules_notion_of_clear():
    """A frame the rules would reverse on is never open; open floor is; and
    something past the budget but inside BACKOUT_OPEN_MM is not open straight
    ahead, because that is a car one tick from reversing again."""
    assert dv.backout_opened(dv.free_by_column(blocked_depth()), 1.0) == ""
    assert dv.backout_opened(dv.free_by_column(clear_depth()), 1.0) == "straight"
    middle = clear_depth(int((dv.CLEARANCE_MM + dv.BACKOUT_OPEN_MM) / 2))
    assert dv.backout_opened(dv.free_by_column(middle), 1.0) == ""
    blind = np.zeros((400, 640), dtype=np.uint16)
    assert dv.backout_opened(dv.free_by_column(blind), 1.0) == ""


def test_backout_opened_accepts_the_side_it_is_about_to_turn_to():
    """Near side still inside BACKOUT_OPEN_MM but past the budget, the side
    the hand-back leans to open: open that way, and only that way."""
    scan = dv.free_by_column(clear_depth())
    d_mm = np.full(dv.PROBE_COLUMNS, 899.0)
    d_mm[:3] = 450.0                       # the left edge, past 400, short of 600
    scan = scan._replace(distance_mm=d_mm)
    assert dv.backout_opened(scan, 1.0) == "right"
    assert dv.backout_opened(scan, -1.0) == ""


def test_backout_fields_say_what_the_back_out_is_doing(d):
    d.camera.imu = MOVING_IMU
    d.camera.depth = blocked_depth()
    assert d.backout_fields(100.0)["backout_phase"] == ""
    d.begin_recover(100.0, dv.BACKOUT_STANDOFF)
    d.recovering(100.1)
    f = d.backout_fields(100.1)
    assert f["backout_phase"] == dv._PHASE_REVERSE
    assert f["backout_reason"] == dv.BACKOUT_STANDOFF
    assert f["backout_elapsed_s"] == pytest.approx(0.1)
    assert f["backout_side"] == dv.REVERSE_DEFAULT_SIDE
    assert f["backout_speed"] < 0 and f["backout_exit"] == ""
    now = recover_to_handback(d, 100.1)
    f = d.backout_fields(now)
    assert f["backout_phase"] == dv.HANDBACK
    assert f["backout_exit"] == dv.EXIT_CAP
    assert f["backout_reverse_s"] == pytest.approx(dv.REVERSE_SECONDS)
    assert f["backout_speed"] == 0.0


# --- the gyro as a rear bumper ---------------------------------------------
#
# Nothing on this car points backwards, so a reverse is blind. The gyro is
# the contact sensor: commanded in reverse at reverse_speed and not shaking means the
# wheels are against something, and the remaining seconds of the reverse are
# seconds of pushing.
#
# All of it runs off the FakeCamera's single imu slot, which hands back the
# same reading every call, so "reversing into a wall" is just loading STILL
# and stepping the phase machine at 10 Hz.

def reverse_for(d, ticks, imu_sample, start=100.0, step=0.1):
    """Step recovering() at the 10 Hz tick with one IMU reading loaded.

    Returns the tick number the reverse phase ended on, or None if it was
    still reversing when the ticks ran out.
    """
    d.camera.imu = imu_sample
    if d.camera.depth is None:
        # Something in front that never moves, so only the bumper or the cap
        # can end the reverse.
        d.camera.depth = blocked_depth()
    for i in range(1, ticks + 1):
        d.recovering(start + i * step)
        if d._phase != dv._PHASE_REVERSE:
            return i
    return None


def test_a_blind_reverse_into_a_wall_ends_early(d):
    """The whole point. Backed into something, so stop backing into it."""
    d.begin_recover(100.0)
    ended = reverse_for(d, ticks=20, imu_sample=STILL_IMU)
    assert ended is not None, "it reversed into a wall for the full two seconds"
    assert ended * 0.1 < dv.REVERSE_SECONDS
    assert d._phase == dv._PHASE_SETTLE


def test_the_bumper_waits_for_the_smoothing_window_to_fill(d):
    """begin_recover() empties the window, so the first ticks of a reverse are
    a mean over one or two readings taken while the ESC was still spinning up
    -- which reads STILL on a car that is about to move off fine. This is also
    the half second that stops it firing before the car has gone anywhere,
    which is why there is no separate arm delay."""
    d.begin_recover(100.0)
    ended = reverse_for(d, ticks=20, imu_sample=STILL_IMU)
    assert ended >= d.imu_window_size, f"fired on tick {ended}"
    assert ended == d.imu_window_size + dv.REVERSE_STALL_SAMPLES - 1


def test_a_reverse_that_is_actually_moving_runs_its_full_duration(d):
    """The car is backing up happily. Do not cut it short."""
    d.begin_recover(100.0)
    assert reverse_for(d, ticks=19, imu_sample=MOVING_IMU) is None
    assert d._reverse_still_samples == 0


def test_unknown_imu_samples_can_never_end_a_reverse(d):
    """The same rule as the forward detector, in the one place guessing is
    worst: no evidence is not evidence of no motion, and this one is acting on
    a direction it cannot see."""
    d.begin_recover(100.0)
    assert reverse_for(d, ticks=19, imu_sample=None) is None
    assert d._reverse_still_samples == 0


def test_an_unknown_in_the_middle_of_a_reverse_holds_the_count(d):
    """Held where it is, not added to and not reset -- the forward detector's
    rule exactly. Every tick here stays inside the fixed reverse window, so
    the only thing that could end the phase is the bumper."""
    d.camera.depth = blocked_depth()
    d.begin_recover(100.0)
    one_short = d.imu_window_size + dv.REVERSE_STALL_SAMPLES - 2
    assert (one_short + 10) * 0.1 < dv.REVERSE_SECONDS, "the window has to fit"

    d.camera.imu = STILL_IMU
    for i in range(1, one_short + 1):
        d.recovering(100.0 + i * 0.1)
    assert d._reverse_still_samples == dv.REVERSE_STALL_SAMPLES - 1

    d.camera.imu = None
    for i in range(one_short + 1, one_short + 11):
        d.recovering(100.0 + i * 0.1)
    assert d._reverse_still_samples == dv.REVERSE_STALL_SAMPLES - 1
    assert d._phase == dv._PHASE_REVERSE


def test_one_moving_reading_resets_the_reverse_count(d):
    """Evidence has to be consecutive, same as the forward detector. A car
    that moves at all has not hit anything."""
    d.camera.depth = blocked_depth()
    d.begin_recover(100.0)
    for i in range(1, d.imu_window_size + dv.REVERSE_STALL_SAMPLES - 1):
        d.camera.imu = STILL_IMU
        d.recovering(100.0 + i * 0.1)
    assert d._reverse_still_samples == dv.REVERSE_STALL_SAMPLES - 1

    d.camera.imu = MOVING_IMU
    d.recovering(100.9)
    assert d._reverse_still_samples == 0
    assert d._phase == dv._PHASE_REVERSE


def test_the_bumper_is_deliberately_quicker_than_the_forward_detector(d):
    """Reversing into a wall wants a fast answer and the two failures are not
    symmetric: a false positive here ends a recover early, a false positive
    forward starts a reverse nobody asked for."""
    assert dv.REVERSE_STALL_SAMPLES < dv.STUCK_CONFIRM_SAMPLES
    total = (d.imu_window_size + dv.REVERSE_STALL_SAMPLES - 1) * 0.1
    assert total < dv.REVERSE_SECONDS / 2.0


def test_the_bumper_commands_neutral_when_it_fires(d):
    """pigpio latches the last pulse width forever and there is no hardware
    failsafe, so any path that ends the reverse and does not say so is a
    command to keep reversing."""
    d.begin_recover(100.0)
    assert d.controller.throttles[-1] < 0
    reverse_for(d, ticks=20, imu_sample=STILL_IMU)
    assert d.controller.throttles[-1] == 0


def test_the_bumper_cannot_shorten_the_settle(d):
    """Ending the reverse early must not hand the wheels back early. Settle is
    there so the car has stopped before the forward decision is taken on it."""
    d.begin_recover(100.0)
    ended = reverse_for(d, ticks=20, imu_sample=STILL_IMU)
    handed_back_at = 100.0 + ended * 0.1 + dv.SETTLE_SECONDS
    assert d.recovering(handed_back_at - 0.01) is True
    assert d.recovering(handed_back_at) is False


# --- the two detectors must not corrupt each other -------------------------

def test_the_bumper_never_touches_the_forward_stuck_evidence(d):
    """A new counter, not _still_samples, and _throttle_since is gated on
    FORWARD throttle and stays that way. Sharing either would let one phase
    walk the other's evidence up."""
    d.begin_recover(100.0)
    reverse_for(d, ticks=20, imu_sample=STILL_IMU)
    assert d._reverse_still_samples >= dv.REVERSE_STALL_SAMPLES
    assert d._still_samples == 0
    assert d._throttle_since is None


def test_a_recover_leaves_no_imu_evidence_behind_it(d):
    """The windows ARE sampled during a reverse now, so they hold real
    readings at hand-back -- and every one of them was taken going the other
    way. The first forward reading afterwards must not be a mean with them."""
    d.begin_recover(100.0)
    ended = reverse_for(d, ticks=20, imu_sample=STILL_IMU)
    assert d.gyro_history, "the bumper has to have been sampling"

    d.recovering(100.0 + ended * 0.1 + dv.SETTLE_SECONDS)
    assert d._phase is None
    assert d.gyro_history == []
    assert d.imu_history == []
    assert d._reverse_still_samples == 0
    assert d._still_samples == 0
    assert d._throttle_since is None


def test_a_bumped_reverse_does_not_leave_the_car_one_tick_from_stuck(d):
    """The whole thing through tick(), and the failure mode worth pinning.

    A car wedged against a wall reverses, the bumper cuts the reverse because
    it is wedged at the back too, and the wheels come back. Every IMU sample
    taken during that reverse said STILL. If any of them reaches the forward
    detector, the car declares a stick before it has been given its full
    second of commanded throttle and its full second of stillness -- and the
    shipped bug this whole file is about was exactly a stick confirmed on
    evidence it had not earned.

    So the test is not "it does not fire", it is "it takes just as long to
    fire as it would from cold".
    """
    # A second of ordinary forward driving first, so the forward detector has
    # real state to be corrupted: throttle commanded, smoothing window full of
    # STILL readings, and not yet armed.
    d.camera.depth = clear_depth()
    d.camera.imu = STILL_IMU
    now = 100.0
    for _ in range(10):
        d.tick(now)
        now += 0.1
    assert d._throttle_since is not None
    assert d.gyro_history
    assert d.last_recovery_time == 0.0

    # Then a wall. Long enough for a couple of bumper-cut recovers: the bumper
    # fires 0.8 s in and settle is another 0.5 s.
    d.camera.depth = blocked_depth()
    for _ in range(30):
        d.tick(now)
        now += 0.1
    assert any(t < 0 for t in d.controller.throttles)
    assert d.last_recovery_time == 0.0, "a wall is not a stick"

    # Wall gone, car still not moving. A stick is a legitimate verdict from
    # here -- it just has to cost full price again, timed from the first
    # forward command after the wheels came back.
    d.camera.depth = clear_depth()
    first_forward = None
    for _ in range(100):
        d.tick(now)
        if first_forward is None and d.controller.throttles[-1] == d.forward_speed:
            first_forward = now
        now += 0.1
        if d.last_recovery_time:
            break
    assert d.last_recovery_time, "never confirmed a stick at all"

    took = d.last_recovery_time - first_forward
    full_price = dv.STUCK_ARM_SECONDS + (dv.STUCK_CONFIRM_SAMPLES - 1) * 0.1
    assert took >= full_price - 1e-9, (
        f"confirmed a stick {took:.2f} s after the wheels came back, against "
        f"{full_price:.2f} s from cold -- evidence survived the recover")


# --- the footprint check ----------------------------------------------------
#
# The bug this half of the file is about: obstacle_in_front() sampled a 50x50
# patch at the dead centre of a 640x400 frame. 2,500 of 256,000 pixels, 0.98%
# of the frame, a 45 mm square window at the 400 mm trigger distance -- on a
# car 267 mm wide. A wall or a doorway corner that catches a wheel was never
# in it. Worse vertically: the lowest row it read was 25 px below the optical
# centre, which with the lens 165 mm up is 142 mm of air at 400 mm, so it
# looked straight over every rock on the floor.
#
# All of this is arithmetic on numpy arrays. No OAK-D, no pigpio, no car.

def ground_depth(invalid=0.0, wall_mm=2500, seed=5):
    """A believable clear scene: floor sloping away, a far wall above the horizon.

    Level ground at distance Z appears fy * CAMERA_HEIGHT_MM / Z px below the
    optical centre, so this inverts that per row. It matters that the clear
    case is this and not a uniform slab, because the floor is what guarantees
    every probe far evidence, and that guarantee is the whole reason
    fail-closed does not become routine once the band reaches down.

    `invalid` drops that fraction of pixels to 0, the way an unmeasurable
    pixel comes off the device.
    """
    rows = np.arange(400) - 200
    floor = np.where(rows > 0,
                     dv.FY_AT_400 * dv.CAMERA_HEIGHT_MM / np.maximum(rows, 1),
                     1e9)
    frame = np.broadcast_to(np.minimum(floor, wall_mm).astype(np.uint16)[:, None],
                            (400, 640)).copy()
    if invalid:
        rng = np.random.default_rng(seed)
        frame[rng.random(frame.shape) < invalid] = 0
    return frame


def row_for(height_mm, range_mm=dv.CLEARANCE_MM):
    """Image row where something `height_mm` off the ground sits at `range_mm`."""
    return 200 + int(round(dv.FY_AT_400 * (dv.CAMERA_HEIGHT_MM - height_mm) / range_mm))


def marks(scan):
    """The scan as one readable string: N near, ? no-return, . clear."""
    return "".join("N" if r == dv.NEAR else "?" if r == dv.NO_RETURN else "."
                   for r in scan.reason)


def sampled_cells(geo, probe):
    """The pixels one probe actually reads, after decimation.

    Needed because a test that scatters pixels at random mostly misses the
    stride grid and then proves nothing: five random pixels inside the band
    land zero sampled cells. Anything testing a count has to put the pixels
    where the probe will look.
    """
    return [(r, c)
            for r in range(geo.row_top, geo.row_bottom, dv.ROW_STRIDE)
            for c in range(geo.col_edges[probe], geo.col_edges[probe + 1], dv.COL_STRIDE)]


def probe_k(geo, probe):
    """How many samples inside the budget it takes to block one probe."""
    n = len(sampled_cells(geo, probe))
    return max(dv.NEAR_SAMPLES_FLOOR, int(round(dv.NEAR_SAMPLE_FRACTION * n)))


def speckled_depth(near_samples, probe=4):
    """Clear ground with `near_samples` isolated near pixels in one probe.

    Isolated on purpose: each one is a single pixel on the stride grid with
    untouched neighbours, which is what sensor speckle looks like rather than
    an object.
    """
    frame = ground_depth()
    geo = dv.footprint_geometry(frame.shape)
    for r, c in sampled_cells(geo, probe)[::3][:near_samples]:
        frame[r, c] = 120
    return frame



# The stereo pair's minimum range, which is the floor every near reading sits
# above: fx * baseline / disparity_levels = 441.25 * 75 / 190 = 174.2 mm.
# 190 levels because oakd.py runs extended disparity; on subpixel it was 95
# and this read 348.4. Inside the floor nothing comes back at all, so an
# obstacle that gets there flips to NO_RETURN instead of reading nearer and
# nearer -- but at 174 mm that is now well below REVERSE_FLOOR_MM, so in
# practice the car reverses long before a real obstacle arrives down here.
MIN_Z_MM = dv.FX_AT_640 * 75.0 / 190.0


def sampled_columns(geo):
    """Every column the probes actually read, after COL_STRIDE."""
    cols = []
    for i in range(dv.PROBE_COLUMNS):
        cols.extend(range(geo.col_edges[i], geo.col_edges[i + 1], dv.COL_STRIDE))
    return np.array(cols)


def wall(d_mm=390.0, beta_deg=30.0, invalid=0.0):
    """A flat vertical wall at beta degrees to the image plane, over level floor.

    beta = 0 is head-on. Positive beta recedes to the right, so the wall is
    nearest on the left and the car should steer right away from it; negative
    is the mirror of that.

    The plane is z = d + x tan beta, and a pixel at horizontal tangent u sees
    it at z = d / (1 - u tan beta) -- one divide per column, no ray casting.
    It is then scaled so the nearest column the probes actually sample reads
    exactly d_mm, because that is the number the car decides on: anchoring on
    the near edge is what keeps the whole wall measurable as beta steepens,
    instead of sliding the near end into the blind zone and turning every
    oblique scene into a NO_RETURN.

    Anything closer than MIN_Z_MM is written as 0, which is how an
    unmeasurable pixel arrives off the device.
    """
    frame = ground_depth(invalid=invalid)
    h, w = frame.shape
    geo = dv.footprint_geometry(frame.shape)
    u = (np.arange(w) - w / 2.0) / geo.fx
    t = np.tan(np.radians(beta_deg))
    denom = 1.0 - u * t
    behind = denom <= 1e-6
    z = np.where(behind, np.inf, 1.0 / np.where(behind, 1.0, denom))
    z = z * (d_mm / z[sampled_columns(geo)].min())

    out = np.minimum(frame.astype(np.float64), z[None, :])
    out[out < MIN_Z_MM] = 0.0
    return np.minimum(out, 65535).astype(np.uint16)


def probe_scene(depths):
    """A frame where probe i reads exactly depths[i] mm.

    Not a photograph of anything. It writes the scan's own axis directly, so
    the arithmetic in cruise_bias() and read_the_gradient() can be checked
    against exact numbers rather than approximate ones, and a mirrored scene
    is exactly mirrored.

    That last part is the reason this exists alongside wall(). The probe fan
    is not symmetric about the frame centre to the pixel -- col_edges run
    151..489 on a 640 wide frame, so the outermost sampled columns are 169 px
    left of centre and 167 px right of it -- which means a flipped IMAGE is
    not a flipped SCAN and nothing built out of pixels can state an exact
    negation.
    """
    frame = ground_depth()
    geo = dv.footprint_geometry(frame.shape)
    for i, mm in enumerate(depths):
        frame[geo.row_top:geo.row_bottom,
              geo.col_edges[i]:geo.col_edges[i + 1]] = mm
    return frame


def blind_probe(frame, probe):
    """Zero one probe's columns outright, the way a dead column arrives."""
    frame = frame.copy()
    geo = dv.footprint_geometry(frame.shape)
    frame[:, geo.col_edges[probe]:geo.col_edges[probe + 1]] = 0
    return frame


EVADE_LEFT = [390.0] + [899.0] * 8      # blocked on the left edge, steer right
EVADE_RIGHT = [899.0] * 8 + [390.0]     # blocked on the right edge, steer left


# --- the geometry, against the two numbers measured off the car -------------

def test_the_probes_span_the_measured_track_width():
    """Track width outside-to-outside is 266.7 mm, measured on the car, so the
    probes have to reach 133.35 mm each side plus the clearance margin. The old
    patch reached 22.7 mm, which is 5.9x too narrow."""
    geo = dv.footprint_geometry((400, 640))
    half_px = (geo.col_edges[-1] - geo.col_edges[0]) / 2.0
    half_mm = dv.CLEARANCE_MM * half_px / geo.fx
    assert half_mm == pytest.approx(dv.HALF_TRACK_MM + dv.FOOTPRINT_MARGIN_MM, abs=1.0)
    assert half_px / 25.0 > 5.0, "no wider than the old patch is no fix"


def test_the_probes_tile_the_footprint_with_no_gaps():
    """Contiguous, not sampled. A gap between probes is somewhere a wheel can
    hide, and the probe width has to stay under a tyre face."""
    geo = dv.footprint_geometry((400, 640))
    edges = geo.col_edges
    assert len(edges) == dv.PROBE_COLUMNS + 1
    widths = [edges[i + 1] - edges[i] for i in range(dv.PROBE_COLUMNS)]
    assert all(w > 0 for w in widths)
    assert sum(widths) == edges[-1] - edges[0]
    widest_mm = dv.CLEARANCE_MM * max(widths) / geo.fx
    assert widest_mm < 50.0, "a probe wider than a tyre blurs which wheel is blocked"


def test_the_probes_reach_down_toward_the_near_ground():
    """The part his 165.1 mm lens-height measurement unblocked. The old band
    was a 3.24 degree stripe around the optical axis and saw nothing below
    142 mm above the ground at the trigger range."""
    geo = dv.footprint_geometry((400, 640))
    lowest = geo.row_bottom - 1
    assert lowest > 225, "no lower than the old patch is no fix"
    above_ground = dv.CAMERA_HEIGHT_MM - dv.CLEARANCE_MM * (lowest - 200) / geo.fy
    assert above_ground < 100.0


def test_the_bottom_probe_row_cannot_read_level_ground_as_an_obstacle():
    """The reason the band stops where it does. A probe takes the near
    distance in its column, so a row low enough to see flat floor inside the
    budget reads the floor as a wall and the car reverses forever."""
    geo = dv.footprint_geometry((400, 640))
    floor_mm = geo.fy * dv.CAMERA_HEIGHT_MM / (geo.row_bottom - 1 - 200)
    assert floor_mm == pytest.approx(dv.FLOOR_CLEAR_MM, rel=0.02)
    assert floor_mm > 2.0 * dv.CLEARANCE_MM, "needs room for the car to pitch"


def test_the_geometry_scales_with_the_frame_it_is_given():
    """The intrinsics are quoted at 640x400 and the field of view does not
    change with resolution, so an 800P frame needs no edit here."""
    small = dv.footprint_geometry((400, 640))
    big = dv.footprint_geometry((800, 1280))
    assert big.fx == pytest.approx(2 * small.fx)
    assert (big.col_edges[-1] - big.col_edges[0]) == pytest.approx(
        2 * (small.col_edges[-1] - small.col_edges[0]), abs=2)


# --- the four scenes that matter -------------------------------------------

def test_an_all_far_frame_leaves_every_probe_clear():
    scan = dv.free_by_column(np.full((400, 640), 3000, dtype=np.uint16))
    assert marks(scan) == "." * dv.PROBE_COLUMNS
    assert not scan.blocked.any()


def test_a_wall_blocks_every_probe_for_being_near():
    scan = dv.free_by_column(np.full((400, 640), 200, dtype=np.uint16))
    assert marks(scan) == "N" * dv.PROBE_COLUMNS
    assert np.all(scan.distance_mm == 200)


def test_a_corner_blocks_only_the_probes_it_covers():
    """The case the old check could not see at all: something across one side
    of the car. Left half of the frame is a wall, right half is open floor."""
    frame = ground_depth(invalid=0.2)
    frame[:, :320] = 250
    scan = dv.free_by_column(frame)
    assert set(marks(scan)) == {"N", "."}
    assert marks(scan).startswith("N"), "the blocked side has to be the left one"
    assert marks(scan).endswith("."), "the open side has to stay open"


def test_an_all_zeros_frame_blocks_every_probe_for_no_return():
    """A stereo pair that returns nothing is too close, too dark or too
    featureless, and all three are reasons to stop."""
    scan = dv.free_by_column(np.zeros((400, 640), dtype=np.uint16))
    assert marks(scan) == "?" * dv.PROBE_COLUMNS
    assert np.all(np.isinf(scan.distance_mm))


def test_a_single_near_column_blocks_only_itself():
    frame = ground_depth(invalid=0.25)
    geo = dv.footprint_geometry(frame.shape)
    frame[geo.row_top:geo.row_bottom, geo.col_edges[-2]:geo.col_edges[-1]] = 300
    scan = dv.free_by_column(frame)
    assert marks(scan) == "." * (dv.PROBE_COLUMNS - 1) + "N"
    assert scan.distance_mm[-1] == 300


# --- fail-closed inflation, which is the risk this change carries ----------

def test_a_clear_floor_is_not_blocked_even_with_most_of_it_unmeasurable():
    """THE test for the risk. Under 10% valid was rare across 2,500 centre
    pixels; spread over a band that includes frame edges and far field it
    could have become routine, and the car would reverse on a metronome --
    which would read as a regression in the stuck detector that just shipped.

    The rule asks for FAR evidence per probe, not validity, so a scene that is
    plainly far survives heavy dropout."""
    for dropout in (0.0, 0.3, 0.5, 0.7, 0.85):
        scan = dv.free_by_column(ground_depth(invalid=dropout))
        assert not scan.blocked.any(), f"blocked at {dropout:.0%} invalid: {marks(scan)}"


def test_a_probe_that_is_mostly_far_and_partly_invalid_is_not_blocked():
    """Invalidity only means "obstacle" inside the band where a reading would
    have meant "too close". Half of this frame is missing and the rest is at
    3 m, which is an answer, not an absence of one."""
    frame = np.full((400, 640), 3000, dtype=np.uint16)
    frame[np.random.default_rng(3).random(frame.shape) < 0.5] = 0
    scan = dv.free_by_column(frame)
    assert marks(scan) == "." * dv.PROBE_COLUMNS
    assert np.all(scan.valid_fraction < 0.6)


def test_a_probe_with_almost_nothing_valid_is_blocked_even_though_it_reads_far():
    """The other side of the same rule. At 95% dropout a probe cannot produce
    MIN_VALID_FRACTION of its samples as valid-and-far, so it has not earned
    the right to call itself clear."""
    frame = np.full((400, 640), 3000, dtype=np.uint16)
    frame[np.random.default_rng(4).random(frame.shape) < 0.95] = 0
    scan = dv.free_by_column(frame)
    assert marks(scan) == "?" * dv.PROBE_COLUMNS


def test_a_dead_probe_blocks_its_own_column_and_nothing_else():
    """The validity rule is per probe, never over the whole band. One dead
    column in an otherwise clean frame leaves the band 89% valid, so a
    whole-region test would wave it through."""
    frame = ground_depth()
    geo = dv.footprint_geometry(frame.shape)
    frame[:, geo.col_edges[3]:geo.col_edges[4]] = 0
    scan = dv.free_by_column(frame)
    assert marks(scan) == "...?....."
    assert scan.valid_fraction.mean() > 0.8, "the band as a whole is healthy"


def test_a_speckle_cannot_claim_a_probe():
    """A plain minimum would hand the whole column to one bad pixel. The
    reported distance is the k-th smallest, k being the same count the near
    test wants, so a scatter below that count is ignored.

    k is 13 samples out of 650 for a full probe, which is about 100 raw
    pixels, and the spatial and speckle filters in oakd.py are already
    supposed to have cleared anything that small."""
    geo = dv.footprint_geometry((400, 640))
    k = probe_k(geo, 4)
    assert k > 1, "a k of 1 is a plain minimum and there is nothing to test"

    scan = dv.free_by_column(speckled_depth(k - 1))
    assert not scan.blocked.any(), marks(scan)
    assert scan.distance_mm[4] > dv.CLEARANCE_MM

    # And the other side of the line, so this is pinning a threshold and not
    # just asserting that nothing ever blocks.
    assert dv.free_by_column(speckled_depth(k)).reason[4] == dv.NEAR


def test_the_reported_distance_and_the_near_verdict_are_the_same_statement():
    """distance_mm < CLEARANCE_MM has to mean exactly "blocked for being
    near", or the number in the telemetry and the decision on the car can
    drift apart. The two speckle frames are the interesting ones: they sit
    either side of the count, which is where a second statistic would show
    up."""
    corner = ground_depth()
    corner[:, :320] = 250
    k = probe_k(dv.footprint_geometry((400, 640)), 4)
    frames = [ground_depth(), ground_depth(invalid=0.5),
              np.full((400, 640), 200, dtype=np.uint16),
              np.zeros((400, 640), dtype=np.uint16),
              corner,
              speckled_depth(k - 1), speckled_depth(k),
              speckled_depth(max(1, k // 3))]
    for frame in frames:
        scan = dv.free_by_column(frame)
        for i, reason in enumerate(scan.reason):
            assert (scan.distance_mm[i] < dv.CLEARANCE_MM) == (reason == dv.NEAR)


# --- the two scenes the old check got wrong --------------------------------

def test_the_old_centre_patch_would_have_missed_a_rock_in_front_of_a_wheel():
    """The bug, stated as a scene. A 100 mm rock at 380 mm, directly in front
    of the left wheel. It is 20 mm inside the clearance budget and it will
    stop the car dead.

    The old 50x50 centre patch is reproduced inline, and it reads the scene as
    clear twice over: the rock is outside its 45 mm wide window, and it is 70
    rows below the lowest row that window reached."""
    frame = ground_depth(invalid=0.2)
    geo = dv.footprint_geometry(frame.shape)
    frame[row_for(100):, geo.col_edges[0]:geo.col_edges[2]] = 380

    old_patch = frame[200 - 25:200 + 25, 320 - 25:320 + 25]
    valid = old_patch[old_patch > 0]
    assert valid.size >= old_patch.size * 0.1, "the old patch had plenty to look at"
    assert np.mean(valid < dv.CLEARANCE_MM) <= 0.5, "and it called this clear"

    scan = dv.free_by_column(frame)
    assert marks(scan) == "NN......."
    assert scan.distance_mm[0] == 380


def test_an_overhang_the_car_drives_under_does_not_block():
    """The reason the band has a top at all. Nothing above the lens axis can
    be ground, so going higher costs nothing in false floor hits -- but a
    doorway lintel the car passes under must not stop it."""
    frame = ground_depth(invalid=0.2)
    lintel = row_for(280)
    assert lintel < dv.footprint_geometry(frame.shape).row_top
    frame[:lintel, :] = 400
    scan = dv.free_by_column(frame)
    assert marks(scan) == "." * dv.PROBE_COLUMNS


# --- what obstacle_in_front hands back and leaves behind -------------------

def test_a_near_obstacle_and_a_blind_frame_report_different_reasons(d):
    """Two completely different behaviours used to wear one function name, and
    the journal could not tell them apart. On this stereo pair most real
    triggers should be the near kind: extended disparity puts minimum range at
    174 mm against a 400 mm budget, so there are 226 mm of measurable approach
    before the blind zone, and a no-return now means dark or featureless
    rather than merely close."""
    assert d.obstacle_in_front(np.full((400, 640), 200, dtype=np.uint16)) is True
    assert d.last_block_reason == dv.NEAR

    assert d.obstacle_in_front(np.zeros((400, 640), dtype=np.uint16)) is True
    assert d.last_block_reason == dv.NO_RETURN

    assert d.obstacle_in_front(ground_depth()) is False
    assert d.last_block_reason == dv.CLEAR


def test_both_reasons_are_reported_when_both_fired(d):
    frame = ground_depth()
    geo = dv.footprint_geometry(frame.shape)
    frame[:, geo.col_edges[0]:geo.col_edges[1]] = 250    # near
    frame[:, geo.col_edges[-2]:geo.col_edges[-1]] = 0    # blind
    assert d.obstacle_in_front(frame) is True
    assert d.last_block_reason == dv.NEAR + "+" + dv.NO_RETURN


def test_the_scan_is_left_on_the_driver_for_telemetry_to_publish(d):
    """One read of the frame, several consumers. Nothing in the driver decides
    on these -- they are there so whoever is publishing does not have to run
    the check a second time."""
    assert d.last_scan is None
    d.obstacle_in_front(ground_depth())
    assert d.last_scan is not None
    assert d.last_scan.distance_mm.shape == (dv.PROBE_COLUMNS,)
    assert d.last_scan.valid_fraction.shape == (dv.PROBE_COLUMNS,)
    assert len(d.last_scan.reason) == dv.PROBE_COLUMNS
    assert d.last_scan_ms > 0.0


def test_a_frame_too_small_for_the_band_still_decides_and_fails_closed():
    """Degenerate shapes must not raise on the one thread that commands the
    ESC. Nothing on the car produces these; the guards are there because an
    exception out of tick() costs auto mode."""
    assert dv.free_by_column(np.zeros((32, 32), dtype=np.uint16)).blocked.all()
    assert not dv.free_by_column(np.full((32, 32), 3000, dtype=np.uint16)).blocked.any()
    # Narrower than the probe count cannot be laid out at all, and that too
    # has to come back as a stop rather than an exception.
    assert dv.free_by_column(np.full((4, 4), 3000, dtype=np.uint16)).blocked.any()


# --- four-wheel steering ----------------------------------------------------
#
# The car has two steering servos and ordinary driving only ever used one.
# tick() commanded set_steering(bias, 0), so the rear was hard-centred on
# every forward tick while begin_recover() -- the one path that did use it --
# commanded the full counter-phase pair. Tight indoor corners were the thing
# that could not be taken. steering_pair() is now the single definition of
# how this car steers and both auto call sites go through it.


def test_counter_phase_puts_the_rear_opposite_the_front():
    """Opposite, not the same. In-phase would crab the body sideways without
    rotating it, which is not what a tight corner needs."""
    front, rear = dv.steering_pair(0.4)
    assert front == 0.4
    assert rear == -0.4


def test_the_ratio_is_full_counter_phase():
    """1.0 is the tightest radius the linkage gives and it is what
    begin_recover() has always commanded. Anything less is a looser turn than
    the car can do."""
    assert dv.REAR_STEER_RATIO == 1.0


def test_centre_stays_centred_on_both_axles():
    assert dv.steering_pair(0) == (0, 0)
    assert dv.steering_pair(0.0)[1] == 0.0


def test_forward_driving_steers_the_rear_too(d):
    """The ask, at the call site that was wrong. A scene asking for a turn
    used to command a rear of exactly 0."""
    d.camera.depth = probe_scene(EVADE_LEFT)
    d.camera.imu = MOVING_IMU

    d.tick(100.0)

    front, rear = d.controller.steerings[-1]
    assert front != 0, "the scene has to be lopsided enough to ask for a turn"
    assert rear == -front


def test_forward_driving_goes_through_the_one_steering_definition(d, monkeypatch):
    """Not a second copy of the arithmetic. Turn the one constant and the
    forward path has to follow it."""
    monkeypatch.setattr(dv, "REAR_STEER_RATIO", 0.5)
    d.camera.depth = probe_scene(EVADE_LEFT)
    d.camera.imu = MOVING_IMU

    d.tick(100.0)

    front, rear = d.controller.steerings[-1]
    assert rear == pytest.approx(-0.5 * front)


def test_a_recover_goes_through_the_one_steering_definition(d, monkeypatch):
    """begin_recover() predates the helper and its literal (turn_dir,
    -turn_dir) happened to agree with it at ratio 1.0. It has to be reading
    the ratio, or there are two definitions again and only one of them moves
    when I turn the knob."""
    monkeypatch.setattr(dv, "REAR_STEER_RATIO", 0.5)
    d.begin_recover(100.0)

    front, rear = d.controller.steerings[-1]
    assert front == d._turn_dir
    assert rear == pytest.approx(-0.5 * d._turn_dir)


@pytest.mark.parametrize("ratio", [1.0, 1.5, 4.0, 100.0])
@pytest.mark.parametrize("front", [-1.0, -0.6, -0.25, 0.0, 0.25, 0.6, 1.0])
def test_the_rear_can_never_exceed_the_front(monkeypatch, ratio, front):
    """max_turn = 0.6 is my limit on how hard a servo may be held against an
    object, and it bounds the front. Capping the rear at the front's own
    magnitude means the rear is under whatever bound the front is under, so a
    ratio raised past 1.0 can never put the rear servo somewhere the front is
    not allowed to go."""
    monkeypatch.setattr(dv, "REAR_STEER_RATIO", ratio)
    out_front, rear = dv.steering_pair(front)
    assert out_front == front, "the front is clamped, never scaled down"
    assert abs(rear) <= abs(front) + 1e-12


@pytest.mark.parametrize("ratio", [1.0, 1.5, 100.0])
def test_a_recover_never_commands_past_max_turn_on_either_axle(d, monkeypatch, ratio):
    """The recover path picks its magnitudes off max_turn, so the bound can be
    checked end to end here. The turn direction is a random pick of four, so
    this asks for all of them. The forward path is bounded too now -- see
    further down -- but it was not when this test was written."""
    monkeypatch.setattr(dv, "REAR_STEER_RATIO", ratio)
    for _ in range(50):
        d.begin_recover(100.0)
        front, rear = d.controller.steerings[-1]
        assert abs(front) <= d.max_turn + 1e-12
        assert abs(rear) <= d.max_turn + 1e-12


def test_the_rear_keeps_the_front_sign_flipped_both_ways():
    assert dv.steering_pair(-0.3)[1] == pytest.approx(0.3)
    assert dv.steering_pair(0.3)[1] == pytest.approx(-0.3)


def test_a_nan_angle_is_left_for_the_one_place_that_writes_hardware():
    """steering_pair does not guard it and must not pretend to. _usable() in
    VehicleController is the single gate and it commands neutral on both
    channels, which is the behaviour a nan already had before the rear was
    wired."""
    front, rear = dv.steering_pair(float("nan"))
    assert np.isnan(front) and np.isnan(rear)


# --- max_turn bounds every steering command --------------------------------
#
# self.max_turn = 0.6 carries the comment "limit to avoid servo damage when
# steering against an object", and for a long time that sentence was not true
# of the path it describes: the constant was referenced in exactly one place,
# begin_recover(), and tick() fed the old get_steering_bias() straight into
# steering_pair(). That function saturated at exactly 1.0000 on a frame with
# one side at 1500 mm and the other at 4000, so the car commanded full lock on
# the front while it leaned against the wall, and the four-wheel change put a
# second servo under the same load.
#
# Every steering decision is now a FRACTION of max_turn -- cruise, evade and
# recover alike -- so the bound is one multiplication at one call site and
# there is nothing left for a decision to saturate past. The number 0.6 has
# not moved; what changed is that it governs.


def test_an_evade_asks_for_the_full_fraction(d):
    """The premise, asserted rather than assumed. An evade is the one
    decision that asks for everything it is allowed, so if this ever stops
    coming out at exactly 1.0 the clamp below is being tested against
    nothing. This is the test that used to pin get_steering_bias() saturating
    at 1.0; the saturation is deliberate and bounded now instead of
    accidental and unbounded."""
    verdict = dv.read_the_gradient(dv.free_by_column(probe_scene(EVADE_LEFT)))
    assert verdict.action == dv.ACT_EVADE
    assert verdict.steer == 1.0
    assert dv.read_the_gradient(
        dv.free_by_column(probe_scene(EVADE_RIGHT))).steer == -1.0


@pytest.mark.parametrize("depths,expected_front",
                         [(EVADE_LEFT, 0.6), (EVADE_RIGHT, -0.6)])
def test_the_forward_path_never_commands_past_max_turn(d, depths, expected_front):
    """The fix. Asked for everything both ways, and the sign has to survive
    the bound -- a limit that forgot which way to turn would steer into the
    thing it was avoiding."""
    d.camera.depth = probe_scene(depths)
    d.camera.imu = MOVING_IMU

    d.tick(100.0)

    front, rear = d.controller.steerings[-1]
    assert front == pytest.approx(expected_front)
    assert rear == pytest.approx(-expected_front)
    assert abs(front) <= d.max_turn + 1e-12
    assert abs(rear) <= d.max_turn + 1e-12


def test_a_cruise_correction_is_passed_through_as_its_own_fraction(d):
    """A bound, not a gain, and the fraction is not quantised on the way
    through. Cruise asks for a fraction well inside the limit and the command
    has to be exactly that fraction of it."""
    depths = [880.0] * 5 + [500.0] * 4          # room on the left
    scan = dv.free_by_column(probe_scene(depths))
    bias = dv.cruise_bias(scan)
    assert -1.0 < bias < 0.0, "the scene has to ask for a left correction"
    assert abs(bias) < 1.0

    d.camera.depth = probe_scene(depths)
    d.camera.imu = MOVING_IMU
    d.tick(100.0)

    front, rear = d.controller.steerings[-1]
    assert front == pytest.approx(bias * d.max_turn)
    assert rear == pytest.approx(-bias * d.max_turn)


def test_the_bound_is_the_constant_and_not_a_literal(d):
    """max_turn is the knob. Turning it has to move the forward path, or 0.6
    has been copied into tick() and there are two limits again."""
    d.max_turn = 0.25
    d.camera.depth = probe_scene(EVADE_LEFT)
    d.camera.imu = MOVING_IMU

    d.tick(100.0)

    front, rear = d.controller.steerings[-1]
    assert front == pytest.approx(0.25)
    assert rear == pytest.approx(-0.25)


def test_the_clamp_does_not_turn_a_nan_into_a_hard_turn(d, monkeypatch):
    """The trap worth a test of its own. min(0.6, nan) is 0.6, so a plain
    max/min clamp converts an unmeasurable steer into a commanded full
    deflection -- the exact opposite of a safety limit, and the same shape as
    the bug that once made a nan throttle read as full forward. A nan has to
    reach set_steering(), which is the one place that writes hardware and the
    one place that rejects it into neutral on both channels."""
    monkeypatch.setattr(dv, "cruise_bias", lambda scan: float("nan"))
    d.camera.depth = ground_depth()
    d.camera.imu = MOVING_IMU

    d.tick(100.0)

    front, rear = d.controller.steerings[-1]
    assert np.isnan(front), "a clamp must not invent an angle out of a nan"
    assert np.isnan(rear)


def test_max_turn_bounds_every_auto_steering_command(d):
    """All three paths, end to end, which is the claim the comment on the
    constant now makes: cruise and evade through tick(), recovery through
    begin_recover()'s max_turn-derived magnitudes."""
    d.camera.imu = MOVING_IMU
    for depths in (EVADE_LEFT, EVADE_RIGHT, [880.0] * 5 + [500.0] * 4,
                   [500.0] * 4 + [880.0] * 5):
        d.camera.depth = probe_scene(depths)
        d._clear_evade()
        d.tick(100.0)

    for _ in range(50):
        d.begin_recover(100.0)

    assert d.controller.steerings
    for front, rear in d.controller.steerings:
        assert abs(front) <= d.max_turn + 1e-12
        assert abs(rear) <= d.max_turn + 1e-12


# --- cruise steering, which replaced get_steering_bias ---------------------
#
# The old function took two 100x80 patches at the quarter columns, averaged
# every valid pixel in each and divided the difference by a 1200 mm literal.
# Three things were wrong with it: it read a band the footprint check now
# covers properly, it saturated at exactly 1.0 with a wall down one side, and
# the normaliser made its answer depend on how far away the scene was rather
# than on its shape.
#
# cruise_bias() reads the scan that already ran, is scale free by
# construction, and is deliberately silent on anything past FLOOR_CLEAR_MM.


def test_an_even_scene_steers_straight(d):
    """Level floor reads 899 on every probe, so this has to be exactly zero
    and the command exactly zero -- not a small number that still rounds up
    to a servo degree."""
    assert dv.cruise_bias(dv.free_by_column(ground_depth())) == 0.0

    d.camera.depth = ground_depth()
    d.camera.imu = MOVING_IMU
    d.tick(100.0)

    front, rear = d.controller.steerings[-1]
    assert front == 0.0
    assert rear == 0.0


def test_cruise_turns_toward_the_open_side_and_mirrors_exactly(d):
    """Positive is nose right, so a scene nearer on the right has to come out
    negative, and the mirror of it has to come out as the exact negation --
    not approximately, or the two sides of the car are not the same car.

    (Was test_steering_bias_turns_toward_the_open_side.)
    """
    nearer_right = dv.cruise_bias(
        dv.free_by_column(probe_scene([880.0] * 5 + [500.0] * 4)))
    nearer_left = dv.cruise_bias(
        dv.free_by_column(probe_scene([500.0] * 4 + [880.0] * 5)))

    assert nearer_right < 0.0
    assert nearer_left == -nearer_right


def test_an_evade_turns_toward_the_open_side_and_mirrors_exactly():
    """The same claim on the other decision. A wall nearer on the right is
    steered away from to the left, and the mirrored wall gives the exact
    negation because the magnitude is a constant and not a measurement."""
    right = dv.read_the_gradient(dv.free_by_column(wall(390.0, -60.0)))
    left = dv.read_the_gradient(dv.free_by_column(wall(390.0, 60.0)))

    assert right.action == dv.ACT_EVADE and right.steer < 0.0
    assert left.action == dv.ACT_EVADE
    assert left.steer == -right.steer


def test_cruise_is_silent_on_anything_past_the_floor_clip(d):
    """Documented behaviour, pinned so nobody later reads it as dead code and
    "fixes" it. Every probe terminates on the floor inside FLOOR_CLEAR_MM, so
    a scene whose walls are all further away than that reads as open floor and
    gets no correction at all. A wall at 1500 mm down one side used to command
    full lock through the old normaliser."""
    far_wall = np.full((400, 640), 4000, dtype=np.uint16)
    far_wall[:, :320] = 1500
    assert dv.cruise_bias(dv.free_by_column(far_wall)) == 0.0

    # Bring the same asymmetry inside the clip and it does ask for a turn.
    assert dv.cruise_bias(
        dv.free_by_column(probe_scene([500.0] * 4 + [880.0] * 5))) > 0.0


def test_a_correction_smaller_than_the_deadband_is_not_commanded():
    """The deadband, at the number. A 4% difference in room is servo noise and
    has to come out as an exact zero rather than a twitch."""
    inside = probe_scene([880.0] * 5 + [880.0 * (1 - 2 * 0.04)] * 4)
    outside = probe_scene([880.0] * 5 + [880.0 * (1 - 2 * 0.07)] * 4)

    raw_in = dv.free_by_column(inside)
    left = float(np.minimum(raw_in.distance_mm, dv.FLOOR_CLEAR_MM)[:4].mean())
    right = float(np.minimum(raw_in.distance_mm, dv.FLOOR_CLEAR_MM)[5:].mean())
    assert abs((right - left) / (right + left)) < dv.CRUISE_DEADBAND

    assert dv.cruise_bias(raw_in) == 0.0
    assert dv.cruise_bias(dv.free_by_column(outside)) != 0.0


def test_cruise_is_a_correction_and_not_a_turn():
    """CRUISE_TURN_FRACTION, at the number. The hardest cruise command the
    899 clip can produce has to stay well inside the evade magnitude, or
    leaning away from a wall and steering around an obstacle stop being
    distinguishable on the servos."""
    hardest = dv.cruise_bias(
        dv.free_by_column(probe_scene([899.0] * 5 + [400.0] * 4)))
    ceiling = ((dv.FLOOR_CLEAR_MM - 1 - dv.CLEARANCE_MM)
               / (dv.FLOOR_CLEAR_MM - 1 + dv.CLEARANCE_MM)
               * dv.CRUISE_TURN_FRACTION)
    assert abs(hardest) == pytest.approx(abs(ceiling), abs=1e-3)
    assert abs(hardest) < 0.2


def test_probe_four_is_left_out_of_both_sides():
    """It straddles the centreline and contributes to both sides equally, so
    including it could only dilute the difference. Moving it has to not move
    the answer."""
    depths = [880.0] * 5 + [500.0] * 4
    with_centre = dv.cruise_bias(dv.free_by_column(probe_scene(depths)))
    depths[4] = 400.0
    assert dv.cruise_bias(dv.free_by_column(probe_scene(depths))) == with_centre


def test_the_bias_is_scale_free(d):
    """The actual fix for the 1200 mm normaliser, and the reason it is not
    simply a smaller normaliser. The same wall angle at two different standoff
    distances has to ask for the same correction, because what is being
    measured is the angle."""
    close = dv.cruise_bias(dv.free_by_column(probe_scene([600.0] * 5 + [450.0] * 4)))
    far = dv.cruise_bias(dv.free_by_column(probe_scene([800.0] * 5 + [600.0] * 4)))
    assert close == pytest.approx(far, abs=1e-9)


# --- the gradient rules, R1 to R6 ------------------------------------------
#
# Six gates in order, every one of them a reason to reverse, and the evade is
# only what falls out of the bottom. One test per gate, so the narrowing of
# begin_recover()'s domain can be audited one condition at a time.


def test_an_all_zeros_frame_reverses_rather_than_dividing(d):
    """INVERTED on purpose, and the inversion is the point.

    This was test_no_valid_depth_either_side_steers_straight_rather_than_
    dividing: get_steering_bias() returned a neutral 0 when neither patch had
    a valid pixel, tick() clamped that to 0 and drove on at forward_speed. A
    frame with nothing measurable in it is a frame the stereo pair cannot see
    -- too close, too dark or too featureless -- and driving straight ahead
    was the fail-open bug. R1 vetoes the whole feature on a single NO_RETURN
    probe, so the test that pinned the bug is now the test that pins the fix.
    """
    blind = np.zeros((400, 640), dtype=np.uint16)
    verdict = dv.read_the_gradient(dv.free_by_column(blind))
    assert verdict.action == dv.ACT_REVERSE
    assert verdict.steer == 0.0
    assert verdict.reason == "no_return"

    d.camera.depth = blind
    d.camera.imu = MOVING_IMU
    d.tick(100.0)
    assert d.controller.throttles[-1] < 0


@pytest.mark.parametrize("probe", range(dv.PROBE_COLUMNS))
def test_a_no_return_probe_can_never_be_the_direction_turned_toward(probe):
    """R1, at every position in the fan, and the hardest constraint in the
    whole feature.

    The base scene is a clean evade: one blocked probe on the left edge with
    open floor beside it, which steers right. Put one unmeasurable column
    anywhere in that fan -- including the one being steered toward -- and the
    answer has to become a reverse. An unmeasurable column is usually
    something too close to measure, and steering at it is the worst available
    move.
    """
    frame = blind_probe(probe_scene(EVADE_LEFT), probe)
    verdict = dv.read_the_gradient(dv.free_by_column(frame))
    assert verdict.action == dv.ACT_REVERSE
    assert verdict.steer == 0.0


def test_an_obstacle_closing_in_stops_being_an_evade():
    """The other half of R1 and R2 together, and the reason no separate
    progress monitor is needed. A wall being grazed that keeps closing runs
    out of room: first it drops under REVERSE_FLOOR_MM, and then it stops
    being measurable at all and the R1 veto takes it. Either way the episode
    ends on the scan itself rather than on a timer."""
    for d_mm in (390.0, 370.0):
        assert dv.read_the_gradient(
            dv.free_by_column(wall(d_mm, 60.0))).action == dv.ACT_EVADE

    closed = dv.read_the_gradient(dv.free_by_column(wall(MIN_Z_MM - 5.0, 60.0)))
    assert closed.action == dv.ACT_REVERSE
    assert closed.steer == 0.0
    assert closed.reason.split()[0] in ("floor", "no_return"), closed.reason

    # And a column that has gone entirely unmeasurable -- which is what an
    # obstacle filling the blind zone looks like -- is the R1 veto itself.
    gone = blind_probe(probe_scene(EVADE_LEFT), 0)
    assert dv.read_the_gradient(dv.free_by_column(gone)).reason == "no_return"


def test_something_as_close_as_the_sensor_can_see_reverses():
    """R2, at the number. Inside REVERSE_FLOOR_MM there is no room left to
    rotate along a surface and the next few millimetres are the blind zone.
    195 and 205 mm bracket the shipped 200, written as literals so that moving
    the constant does not quietly move the scene with it.

    This used to bracket 360, and 355 used to reverse here. It now steers:
    one edge probe at 355 is well outside the full-lock sweep, and the floor
    is only the backstop under R7. Just above the floor still reverses, on the
    sweep rather than the floor."""
    verdict = dv.read_the_gradient(
        dv.free_by_column(probe_scene([195.0] + [899.0] * 8)))
    assert verdict.action == dv.ACT_REVERSE
    assert verdict.reason.startswith("floor")
    assert verdict.steer == 0.0

    just_over = dv.read_the_gradient(
        dv.free_by_column(probe_scene([205.0] + [899.0] * 8)))
    assert just_over.action == dv.ACT_REVERSE
    assert just_over.reason == "tight 205"

    assert dv.read_the_gradient(
        dv.free_by_column(probe_scene([355.0] + [899.0] * 8))).action == dv.ACT_EVADE


def test_the_reverse_floor_sits_between_the_blind_zone_and_the_trigger():
    """Where R2's number has to live. Below the stereo minimum it could never
    fire -- nothing is measurable down there -- and at or above the trigger it
    would swallow the whole feature."""
    assert MIN_Z_MM < dv.REVERSE_FLOOR_MM < dv.CLEARANCE_MM


def test_the_fixtures_depth_floor_matches_the_stereo_config_on_the_car():
    """MIN_Z_MM above is the one number in this file that describes hardware
    rather than arithmetic, and nothing else here can catch it going stale.

    Every near-range scene is clipped at it, so if oakd.py went back to
    subpixel the whole gradient suite would quietly keep modelling a camera
    with a 174 mm floor while the car had a 348 mm one -- and every test would
    still pass, because the scenes it clips are all well above both numbers.
    So assert the coupling directly: 190 disparity levels is extended
    disparity with subpixel off, and that is a claim about oakd.py.

    Read as text rather than imported, because importing oakd constructs a
    dai.Device and there is no camera on the laptop.
    """
    source = (pathlib.Path(__file__).resolve().parent.parent / "crawler" / "hardware" / "oakd.py").read_text()
    assert "stereo.setExtendedDisparity(True)" in source
    assert "stereo.setSubpixel(False)" in source
    assert MIN_Z_MM == pytest.approx(dv.FX_AT_640 * 75.0 / 190.0)


def test_two_separate_obstacles_are_not_a_surface():
    """R3. A blocked probe at each end is two things, and there is no single
    direction that goes around both."""
    verdict = dv.read_the_gradient(
        dv.free_by_column(probe_scene([390.0] + [899.0] * 7 + [390.0])))
    assert verdict.action == dv.ACT_REVERSE
    assert verdict.steer == 0.0
    assert verdict.reason == "split"


@pytest.mark.parametrize("depths, steer", [
    ([899.0] * 7 + [390.0] + [899.0], -1.0),          # .......N.  left, 7 open
    ([899.0] * 6 + [390.0] + [899.0] * 2, -1.0),      # ......N..  left, 6 open
    ([899.0] * 2 + [390.0] + [899.0] * 6, 1.0),       # ..N......  right, 6 open
    ([899.0] * 3 + [390.0] * 2 + [899.0] * 4, 1.0),   # ...NN....  right, 4 vs 3
])
def test_a_block_in_the_middle_is_dodged_toward_the_open_side(depths, steer):
    """Middle blocks used to reverse, all of them, as "middle". The run had
    one at about 380 mm, [...N.....], that reversed off something small the
    car had a whole fan of room to go round. It now turns toward the side
    with more open probes, and the reason says it was a dodge rather than a
    graze so the journal keeps the two apart."""
    verdict = dv.read_the_gradient(dv.free_by_column(probe_scene(depths)))
    assert verdict.action == dv.ACT_EVADE
    assert verdict.steer == steer
    assert verdict.reason.startswith("dodge")


def test_a_centred_block_breaks_its_tie_on_room_and_an_exact_tie_reverses():
    """A single probe on the centreline leaves four open probes either side,
    so the count cannot pick. The measured room can; and when that is exactly
    equal too there is no honest side, so it reverses as "middle" rather than
    picking one by coin."""
    roomier_left = [899.0] * 4 + [390.0] + [700.0] + [899.0] * 3
    verdict = dv.read_the_gradient(dv.free_by_column(probe_scene(roomier_left)))
    assert verdict.action == dv.ACT_EVADE
    assert verdict.steer == -1.0

    roomier_right = [899.0] * 3 + [700.0] + [390.0] + [899.0] * 4
    assert dv.read_the_gradient(
        dv.free_by_column(probe_scene(roomier_right))).steer == 1.0

    even = [899.0] * 4 + [390.0] + [899.0] * 4
    verdict = dv.read_the_gradient(dv.free_by_column(probe_scene(even)))
    assert verdict.action == dv.ACT_REVERSE
    assert verdict.steer == 0.0
    assert verdict.reason == "middle"


@pytest.mark.parametrize("depths, tight, clears", [
    # pattern            reverses at     steers at   (sweep cut, R = 590)
    ("N........",        250.0,          265.0),     # ~257
    ("NN.......",        275.0,          295.0),     # ~284
    ("...N.....",        340.0,          362.0),     # ~351
])
def test_the_turn_sweep_decides_how_close_is_too_close(depths, tight, clears):
    """R7. The single 360 mm floor is gone; how close a block can be and still
    be turned away from now depends on where it is in the fan, because the
    car's swinging front corner reaches an edge probe last and a centred one
    first. Literals either side of each cut, so moving TURN_RADIUS_MM or
    CAMERA_AHEAD_MM fails here rather than quietly moving the scenes."""
    def at(mm):
        return probe_scene([mm if c == "N" else 899.0 for c in depths])

    close = dv.read_the_gradient(dv.free_by_column(at(tight)))
    assert close.action == dv.ACT_REVERSE
    assert close.steer == 0.0
    assert close.reason == f"tight {tight:.0f}"

    assert dv.read_the_gradient(dv.free_by_column(at(clears))).action == dv.ACT_EVADE


def test_the_sweep_is_asked_of_every_probe_not_just_the_blocked_one():
    """A clear probe can still be in the way of the corner. An edge block at
    300 mm clears on its own. Put the far probe -- the one the car is turning
    toward, whose edge the swinging corner reaches first -- at 450 mm, past
    the budget and so CLEAR, and the swing no longer fits: on that side the
    cut is about 520 mm."""
    alone = [300.0] + [899.0] * 8
    assert dv.turn_sweep_clears(dv.free_by_column(probe_scene(alone)), 1.0)

    crowded = [300.0] + [899.0] * 7 + [450.0]
    scan = dv.free_by_column(probe_scene(crowded))
    assert scan.reason[8] == dv.CLEAR
    assert not dv.turn_sweep_clears(scan, 1.0)
    verdict = dv.read_the_gradient(scan)
    assert verdict.action == dv.ACT_REVERSE
    assert verdict.reason == "tight 300"


def test_the_two_walls_marked_on_the_floor_land_on_the_side_he_marked():
    """The /marks probe distances off the 2026-10-27 run, VERIFIED, both of
    them oblique walls across the whole fan nearest at one edge. The first
    reversed and was marked bad; the second reversed and was marked good. The
    other two bad marks from that run, BAD3 and BAD5, are not here because
    they still reverse: on the estimated radius their corners would clip."""
    bad = [373.0, 390.0, 419.0, 455.0, 465.0, 514.0, 538.0, 617.0, 636.0]
    verdict = dv.read_the_gradient(dv.free_by_column(probe_scene(bad)))
    assert verdict.action == dv.ACT_EVADE
    assert verdict.steer == 1.0

    good = [497.0, 481.0, 466.0, 446.0, 432.0, 420.0, 409.0, 395.0, 380.0]
    verdict = dv.read_the_gradient(dv.free_by_column(probe_scene(good)))
    assert verdict.action == dv.ACT_REVERSE
    assert verdict.reason.startswith("shallow")


def test_a_doorway_straight_ahead_reverses_and_does_not_steer(d):
    """The scene R4 exists for, asserted on the steer as well as the action.
    A steer of anything but exactly zero on a reverse would be a servo angle
    held against whatever is in front of the car for the length of a
    recover."""
    verdict = dv.read_the_gradient(
        dv.free_by_column(probe_scene([899.0] * 3 + [390.0] * 3 + [899.0] * 3)))
    assert verdict.action == dv.ACT_REVERSE
    assert verdict.steer == 0.0


def test_a_block_spanning_the_whole_fan_reverses():
    """R4's other half. Touching both edges is a wall, and the open side the
    ratio would need does not exist."""
    verdict = dv.read_the_gradient(dv.free_by_column(probe_scene([390.0] * 9)))
    assert verdict.action == dv.ACT_REVERSE
    assert verdict.reason == "spans"


def test_too_much_of_the_car_in_it_reverses():
    """R5, at the number. The probe boundaries put one blocked edge probe
    14.5 mm inside the tyre line, two at 48.2 mm -- about one tyre face -- and
    three at 82 mm, which is most of the wheel."""
    for width in (1, 2):
        depths = [390.0] * width + [899.0] * (9 - width)
        assert dv.read_the_gradient(
            dv.free_by_column(probe_scene(depths))).action == dv.ACT_EVADE

    for width in (3, 4):
        depths = [390.0] * width + [899.0] * (9 - width)
        verdict = dv.read_the_gradient(dv.free_by_column(probe_scene(depths)))
        assert verdict.action == dv.ACT_REVERSE
        assert verdict.reason == f"wide {width}"
        assert verdict.steer == 0.0


def test_a_surface_too_steep_to_rotate_along_reverses():
    """R6, at the number. Without the ratio gate the car would steer away from
    a wall it is driving straight into, because one edge probe lighting up
    first is exactly what a slightly-off-square approach looks like.

    The open side was 700 here, a ratio of 1.79, when the threshold was 2.0.
    That now steers -- it is steeper than the wall marked bad in the run -- so
    the scene is 580, a ratio of 1.49, just under 1.5."""
    shallow = dv.read_the_gradient(
        dv.free_by_column(probe_scene([390.0] + [580.0] * 8)))
    assert shallow.action == dv.ACT_REVERSE
    assert shallow.reason.startswith("shallow")
    assert shallow.steer == 0.0

    assert dv.read_the_gradient(
        dv.free_by_column(probe_scene([390.0] + [899.0] * 8))).action == dv.ACT_EVADE


def test_the_open_side_cannot_be_inflated_past_the_floor_clip():
    """R6's ceiling. Every probe terminates on the floor inside
    FLOOR_CLEAR_MM, so a column that happened to see past it -- a dropout
    straight down the band, say -- must not be allowed to manufacture a ratio
    the geometry cannot support."""
    depths = [390.0] + [899.0] * 7 + [40000.0]
    ratio_source = dv.free_by_column(probe_scene(depths))
    assert ratio_source.distance_mm[8] > dv.FLOOR_CLEAR_MM
    verdict = dv.read_the_gradient(ratio_source)
    assert verdict.action == dv.ACT_EVADE
    assert float(verdict.reason.split()[1]) == pytest.approx(
        dv.FLOOR_CLEAR_MM / 390.0, abs=0.01)


def test_the_ratio_threshold_is_reachable_against_its_own_ceiling():
    """OPEN_RATIO_MIN has to leave headroom under the ceiling the 899 clip
    imposes, or the feature is unreachable in principle. The ceiling is the
    clip over the closest reading that is still allowed to steer."""
    ceiling = dv.FLOOR_CLEAR_MM / dv.REVERSE_FLOOR_MM
    assert dv.OPEN_RATIO_MIN < ceiling
    assert dv.OPEN_RATIO_MIN <= 2.2, "past 2.2 there is no headroom left"


def test_clear_always_means_a_finite_distance():
    """The invariant that makes the old fail-open bug unreachable.

    get_steering_bias() divided by a count it had not checked and handed back
    a neutral zero when the count was nothing. Every consumer of this scan --
    cruise_bias(), R6's open side -- reads distance_mm on probes marked CLEAR,
    so CLEAR has to imply a real number and not an infinity.

    It holds because the far-evidence rule is stricter than the near-sample
    rule: a CLEAR probe needs MIN_VALID_FRACTION of its samples valid AND
    beyond the budget, which on any probe with at least
    NEAR_SAMPLES_FLOOR / MIN_VALID_FRACTION samples in it is more than the k
    the distance statistic needs.
    """
    assert dv.MIN_VALID_FRACTION >= dv.NEAR_SAMPLE_FRACTION

    frames = [ground_depth(), ground_depth(invalid=0.3), ground_depth(invalid=0.5),
              ground_depth(invalid=0.7), ground_depth(invalid=0.85),
              clear_depth(), blocked_depth(), np.zeros((400, 640), np.uint16),
              probe_scene(EVADE_LEFT), probe_scene(EVADE_RIGHT),
              probe_scene([390.0] * 9), speckled_depth(3), speckled_depth(40)]
    frames += [wall(390.0, beta) for beta in range(-80, 81, 10)]
    frames += [blind_probe(probe_scene(EVADE_LEFT), i)
               for i in range(dv.PROBE_COLUMNS)]

    for frame in frames:
        scan = dv.free_by_column(frame)
        for i in range(dv.PROBE_COLUMNS):
            if scan.reason[i] == dv.CLEAR:
                assert np.isfinite(scan.distance_mm[i]), (
                    f"clear probe {i} with no distance: {marks(scan)}")


@pytest.mark.parametrize("dropout", [0.0, 0.3, 0.5, 0.7, 0.85])
def test_open_floor_drives_straight_on_however_patchy_it_is(dropout):
    """The risk this change carries, on the steering as well as the stop. A
    scene that is plainly open floor has to come out as forward with an exact
    zero -- if heavy dropout can shift the per-probe distances enough to ask
    for a turn, the car wanders down an empty room."""
    verdict = dv.read_the_gradient(dv.free_by_column(ground_depth(invalid=dropout)))
    assert verdict.action == dv.ACT_FORWARD
    assert verdict.steer == 0.0


# --- the envelope: which wall angles are steered around --------------------


@pytest.mark.parametrize("beta", [-70, -60, -50, -40, 40, 50, 60, 70])
def test_a_shallow_enough_wall_is_steered_away_from(beta):
    """Measured on the planar fixture at the trigger range, near edge at
    390 mm:

        beta   20     30     40     50     60     70
        ratio  1.28   1.47   1.76   2.26   2.31   2.31

    The threshold sits between 30 and 40 degrees, which is where
    OPEN_RATIO_MIN = 1.5 puts it: the fan's outer columns are at tan +/-0.383,
    so a plane reads (1 + 0.383 tan beta) / (1 - 0.383 tan beta) across it and
    1.5 is beta = 28 degrees on the formula, a hair over 30 on the fixture.
    It was 2.0 and 41 degrees, with 40 on the reverse side, until the run's
    marks put it lower. The table flattens at 2.31 because the open side is
    the floor at 899 against a near edge at 390.
    """
    verdict = dv.read_the_gradient(dv.free_by_column(wall(390.0, beta)))
    assert verdict.action == dv.ACT_EVADE
    assert verdict.steer == (1.0 if beta > 0 else -1.0), "steer away, not into"


@pytest.mark.parametrize("beta", [-30, -20, 20, 30])
def test_a_wall_being_driven_into_is_backed_away_from(beta):
    """The other side of the same threshold, and the one that matters for
    safety: a 30 degree approach still reverses. Steering along a wall the car
    is driving into is how it ends up leaning on a servo. 40 degrees used to
    be on this side; it steers now, see the table above."""
    verdict = dv.read_the_gradient(dv.free_by_column(wall(390.0, beta)))
    assert verdict.action == dv.ACT_REVERSE
    assert verdict.steer == 0.0


@pytest.mark.parametrize("beta", [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
def test_a_head_on_wall_always_reverses(beta):
    """The scene the whole feature must never get wrong. Square on to a wall,
    plus or minus five degrees of approach, is a reverse every time -- and the
    reasons vary (spans, wide, shallow) which is fine, the action does not."""
    verdict = dv.read_the_gradient(dv.free_by_column(wall(390.0, beta)))
    assert verdict.action == dv.ACT_REVERSE
    assert verdict.steer == 0.0


@pytest.mark.parametrize("d_mm", [365.0, 380.0, 395.0])
@pytest.mark.parametrize("beta", [-60, -50, -40, 40, 50, 60])
def test_the_envelope_holds_across_the_trigger_band(d_mm, beta):
    """The whole measurable band between REVERSE_FLOOR_MM and the trigger, so
    the verdict is not an accident of one standoff distance."""
    verdict = dv.read_the_gradient(dv.free_by_column(wall(d_mm, beta)))
    assert verdict.action == dv.ACT_EVADE
    assert verdict.steer == (1.0 if beta > 0 else -1.0)


def test_the_measured_journal_patterns_arm_about_two_in_five():
    """The sanity check on the real car's own record: 88 obstacle events read
    out of the journal's bracketed probe strings, classified by shape. 47 of
    them are the evade shape. If a change to the rules moved this a long way
    in either direction, it changed what the car does on the floor rather than
    what it does in a test.

    37 until middle blocks could be dodged. The ten that moved, at 390 mm, are
    exactly the off-centre middle shapes: .......N. (5), ......N.. (2),
    ...NN.... (1), .N....... (1) and ..N...... (1). The two ....N.... stay
    reverses, an exact tie with nothing to choose a side by."""
    counts = {
        "........N": 15, "NNNNNNNNN": 13, "N........": 12, "....NNNNN": 7,
        "NN.......": 7, ".......N.": 5, "NNNNNN...": 3, "NNN......": 3,
        "......NNN": 3, ".......NN": 3, "....N....": 2, "......N..": 2,
        "NN.NNNNNN": 1, "..NNNNNN.": 1, "...NNNNNN": 1, "NNNNN....": 1,
        "NNN....NN": 1, "..NNNNN..": 1, "NNNN.....": 1, "....NNNN.": 1,
        ".....NNNN": 1, ".....NNN.": 1, "...NN....": 1, ".N.......": 1,
        "..N......": 1,
    }
    armed = 0
    for pattern, seen in counts.items():
        frame = probe_scene([390.0 if c == "N" else 899.0 for c in pattern])
        scan = dv.free_by_column(frame)
        assert marks(scan) == pattern, "the fixture has to reproduce the shape"
        if dv.read_the_gradient(scan).action == dv.ACT_EVADE:
            armed += seen
    assert armed == 47, f"{armed} of {sum(counts.values())} events"


# --- the commitment latch --------------------------------------------------
#
# The latch lives on the Driver. read_the_gradient() is pure and sees none of
# it, so what gets commanded is a function of one scan plus one latched
# direction and nothing else.


def test_an_evade_holds_its_side_through_clear_frames_and_then_lets_go(d):
    """EVADE_SECONDS. The obstacle rotates out of the fan about one tick into
    the turn, so a car that un-steers the moment the fan reads clear has not
    gone anywhere. It has to keep the turn in and then hand it back."""
    d.camera.imu = MOVING_IMU
    d.camera.depth = probe_scene(EVADE_LEFT)
    d.tick(100.0)
    assert d.controller.steerings[-1][0] == pytest.approx(d.max_turn)

    d.camera.depth = ground_depth()
    for k in range(1, 5):                     # 100.1 .. 100.4
        d.tick(100.0 + 0.1 * k)
        assert d.controller.steerings[-1][0] == pytest.approx(d.max_turn), (
            f"let go after {0.1 * k:.1f} s, inside EVADE_SECONDS")

    d.tick(100.0 + dv.EVADE_SECONDS)
    assert d.controller.steerings[-1][0] == 0.0
    assert d._evade_dir == 0.0


def test_an_evade_that_never_ends_reverses_instead(d):
    """EVADE_LIMIT_SECONDS. A surface that is still blocked after thirty
    decisions of steering the same way is a wall being followed into a
    corner, and none of R1, R2, R5 or R6 will have fired. This is the
    backstop."""
    assert dv.EVADE_LIMIT_SECONDS <= 10.0, (
        "a backstop longer than ten seconds is not a backstop")

    d.camera.imu = MOVING_IMU
    d.camera.depth = probe_scene(EVADE_LEFT)

    for k in range(int(dv.EVADE_LIMIT_SECONDS / 0.1)):
        d.tick(100.0 + 0.1 * k)
        assert d.controller.throttles[-1] == d.forward_speed, (
            f"gave up after {0.1 * k:.1f} s, inside EVADE_LIMIT_SECONDS")

    d.tick(100.0 + dv.EVADE_LIMIT_SECONDS)
    assert d.controller.throttles[-1] < 0
    assert d._evade_dir == 0.0


def test_an_evade_asking_for_the_other_side_becomes_a_reverse(d):
    """The anti-dither mechanism, at the source. A sign reversal mid-episode
    is never obeyed: it is a scene the car cannot read, not a turn to make,
    and a reverse is both the safe answer and an assertable one. A low-pass
    filter would only have damped it, and would have added lag to the floor
    check as well."""
    d.camera.imu = MOVING_IMU
    d.camera.depth = probe_scene(EVADE_LEFT)
    d.tick(100.0)
    assert d.controller.throttles[-1] == d.forward_speed

    d.camera.depth = probe_scene(EVADE_RIGHT)
    d.tick(100.1)
    assert d.controller.throttles[-1] < 0
    assert d._evade_dir == 0.0


def test_a_reverse_verdict_drops_the_latch_at_the_decision_too(d):
    """Belt and braces with begin_recover()'s own clear, and asked of _commit()
    directly because that is where it has to live. _commit() is the only thing
    in this file with a memory in it, so it drops the latch itself rather than
    trusting every caller to recover afterwards."""
    d._evade_dir = 1.0
    d._evade_until = 1e9

    out = d._commit(dv.SteerVerdict(dv.ACT_REVERSE, 0.0, "spans"), 100.0)

    assert out.action == dv.ACT_REVERSE
    assert d._evade_dir == 0.0


def test_the_decision_is_left_on_the_driver_beside_the_scan(d):
    """Same read-only bookkeeping as last_scan, for the same reason: whoever
    publishes telemetry should not have to re-run the decision to report it."""
    assert d.last_verdict is None
    d.camera.imu = MOVING_IMU
    d.camera.depth = probe_scene(EVADE_LEFT)

    d.tick(100.0)

    assert d.last_verdict.action == dv.ACT_EVADE
    assert d.last_verdict.steer == 1.0
    assert d.last_verdict.reason.startswith("graze")


def test_the_probe_string_tells_the_two_block_reasons_apart():
    """N and ? are two different events -- something measurably inside the
    budget, and a stereo pair that cannot see at all -- and the journal line is
    where I read which one it was. One character wrong and a week of logs says
    the wrong thing; the 88 events behind these constants were read out of
    exactly this string."""
    near_left_blind_right = blind_probe(probe_scene(EVADE_LEFT), 8)
    assert dv.probe_marks(dv.free_by_column(near_left_blind_right)) == "N.......?"
    assert dv.probe_marks(dv.free_by_column(ground_depth())) == "." * 9


def test_a_reverse_beats_the_commitment_every_time(d):
    """Safety beats commitment. Mid-evade, a scene that reverses reverses --
    the latch does not get to hold the turn in for another half second."""
    d.camera.imu = MOVING_IMU
    d.camera.depth = probe_scene(EVADE_LEFT)
    d.tick(100.0)

    d.camera.depth = probe_scene([390.0] * 9)
    d.tick(100.1)
    assert d.controller.throttles[-1] < 0
    assert d._evade_dir == 0.0


def test_a_recover_clears_the_latch(d):
    """Whatever the car was steering around is behind the decision once a
    recover has taken the wheels, and a latch that survived it would steer the
    old way on the first tick back."""
    d.camera.imu = MOVING_IMU
    d.camera.depth = probe_scene(EVADE_LEFT)
    d.tick(100.0)
    assert d._evade_dir == 1.0

    d.begin_recover(200.0)
    assert d._evade_dir == 0.0
    assert d._evade_until == 0.0

    d.camera.depth = ground_depth()
    d.tick(203.0)        # ends the reverse phase
    d.tick(204.0)        # settle is up, wheels come back, decides afresh
    # Afresh means the hand-back's lean and nothing of the evade's full lock.
    assert d._evade_dir == 0.0
    lean = d._handback_side * dv.HANDBACK_STEER * d.max_turn
    assert d.controller.steerings[-1] == pytest.approx(dv.steering_pair(lean))
    assert abs(lean) < d.max_turn


def test_abort_clears_the_latch(d):
    """Same claim for the other way out. abort() is what a START press, a
    controller dropout and a systemctl stop all land on."""
    d.camera.imu = MOVING_IMU
    d.camera.depth = probe_scene(EVADE_LEFT)
    d.tick(100.0)
    assert d._evade_dir == 1.0

    d.abort()
    assert d._evade_dir == 0.0


def test_the_servos_never_see_a_sign_flip_on_consecutive_decisions(d):
    """Anti-dither on the hardware record rather than on the decision.

    The scene alternates left-blocked and right-blocked every tick, which is
    the worst input this feature can be handed -- a scene balanced on
    OPEN_RATIO_MIN at a 10 Hz decision rate. What must never appear is two
    consecutive decisions commanding opposite front angles, which on four
    wheels is both steering servos slamming across at 10 Hz.
    """
    d.camera.imu = MOVING_IMU
    commanded = {}
    for k in range(100):
        d.camera.depth = probe_scene(EVADE_LEFT if k % 2 == 0 else EVADE_RIGHT)
        before = len(d.controller.steerings)
        d.tick(100.0 + 0.1 * k)
        if (len(d.controller.steerings) > before
                and d.controller.throttles[-1] == d.forward_speed):
            commanded[k] = d.controller.steerings[-1][0]

    assert commanded, "the alternation has to command forward steering at all"
    for k, front in commanded.items():
        previous = commanded.get(k - 1)
        if previous:
            assert front * previous > 0, (
                f"tick {k - 1} commanded {previous:+.2f} and tick {k} "
                f"commanded {front:+.2f}")


# --- what every tick does, whatever it decides -----------------------------


def tick_scenarios():
    """Every shape of decision tick() can take, by the frame that causes it."""
    return {
        "open floor": ground_depth(),
        "patchy floor": ground_depth(invalid=0.7),
        "cruise correction": probe_scene([880.0] * 5 + [500.0] * 4),
        "evade left": probe_scene(EVADE_LEFT),
        "evade right": probe_scene(EVADE_RIGHT),
        "grazeable wall": wall(390.0, 60.0),
        "head-on wall": wall(390.0, 0.0),
        "doorway": probe_scene([899.0] * 3 + [390.0] * 3 + [899.0] * 3),
        "split": probe_scene([390.0] + [899.0] * 7 + [390.0]),
        "blind frame": np.zeros((400, 640), dtype=np.uint16),
        "no depth at all": None,
    }


@pytest.mark.parametrize("name", list(tick_scenarios()))
def test_every_tick_commands_something(d, name):
    """pigpio latches the last servo pulse width forever and there is no
    hardware failsafe on this car, so a tick that decides and commands
    nothing is a tick that keeps doing whatever it was doing. Exactly one
    throttle command per tick, and never a tick that writes neither the ESC
    nor the servos."""
    d.camera.depth = tick_scenarios()[name]
    d.camera.imu = MOVING_IMU

    d.tick(100.0)

    assert len(d.controller.throttles) == 1, d.controller.throttles
    assert d.controller.throttles or d.controller.steerings


@pytest.mark.parametrize("name", list(tick_scenarios()))
def test_a_tick_decides_before_it_commands(d, name):
    """The old order commanded forward speed and only then looked for an
    obstacle, so every tick facing a wall lurched into it for a tick first.
    Whatever the scene, the first throttle of the tick is the only one."""
    d.camera.depth = tick_scenarios()[name]
    d.camera.imu = MOVING_IMU

    d.tick(100.0)

    assert len(set(d.controller.throttles)) == 1


def test_the_decision_prints_on_transitions_and_not_on_every_tick(d, capsys):
    """The decision rate is 10 Hz and journald rate-limits, so a line a tick
    costs me the lines I actually want. The bracketed probe string is what
    makes the journal readable after the fact -- the 88 events behind
    MAX_BLOCKED_PROBES and OPEN_RATIO_MIN were read straight out of it."""
    d.camera.imu = MOVING_IMU
    d.camera.depth = probe_scene(EVADE_LEFT)
    for k in range(10):
        d.tick(100.0 + 0.1 * k)

    printed = [line for line in capsys.readouterr().out.splitlines()
               if "[" in line]
    assert len(printed) == 1, printed
    assert "[N........]" in printed[0]
    assert "steering right" in printed[0]


def test_the_scan_and_the_verdict_fit_inside_the_tick_budget():
    """The control loop targets 100 Hz, measures 75, and has a 190 ms stall in
    its history, so the budget for this is sub-millisecond. Timed as the tick
    pays for it: one scan plus the verdict read off it. The verdict measures a
    few microseconds against a scan of about 70 us on a laptop, so the bar
    here is loose on purpose -- it catches a tenfold regression and never a
    slow machine."""
    frame = ground_depth(invalid=0.25)
    dv.read_the_gradient(dv.free_by_column(frame))
    times = []
    for _ in range(200):
        started = time.perf_counter()
        dv.read_the_gradient(dv.free_by_column(frame))
        times.append((time.perf_counter() - started) * 1000.0)
    assert float(np.median(times)) < 2.0, f"median {np.median(times):.3f} ms"


# --- the forward floor: takeover's and the policy's veto ---------------------
#
# Fake scan sequences, ten a second like the driver's tick. "N<mm>" is a
# probe NEAR at that distance, "?" NO_RETURN, "." clear.

TICK = 0.1


def fake_scan(*probes):
    """Nine probes, each "N<mm>", "?" or "." -- the rest of the fan clear."""
    probes = list(probes) + ["."] * (dv.PROBE_COLUMNS - len(probes))
    dist, reason = [], []
    for p in probes:
        if p.startswith("N"):
            dist.append(float(p[1:]))
            reason.append(dv.NEAR)
        else:
            dist.append(np.inf)
            reason.append(dv.NO_RETURN if p == "?" else dv.CLEAR)
    blocked = np.array([r != dv.CLEAR for r in reason])
    return dv.FootprintScan(np.array(dist), np.zeros(len(dist)), blocked,
                            tuple(reason), None)


def centre(p):
    """One reading on probe 4, dead ahead, and open floor either side."""
    return fake_scan(".", ".", ".", ".", p)


def replay(scans, veto=None):
    """Each scan a tick apart into one veto; what it said after each."""
    veto = veto or dv.ForwardFloorVeto()
    said = []
    for k, scan in enumerate(scans):
        veto.observe(scan, k * TICK)
        said.append(veto.vetoes())
    return said, veto


def test_the_forward_floor_is_built_on_forward_speed():
    """REVERSE_FLOOR_MM plus the travel at the cap while a picture ages its
    worst: 200 + 1050 x 0.4 x 0.388 = 363 mm today, and under the clearance
    budget, past which nothing can read NEAR to trip it."""
    assert dv.FORWARD_FLOOR_MM == pytest.approx(
        dv.REVERSE_FLOOR_MM
        + dv.MM_S_PER_THROTTLE * dv.FORWARD_SPEED * dv.WORST_PICTURE_AGE_S)
    assert dv.FORWARD_FLOOR_MM == pytest.approx(362.96, abs=0.01)
    assert dv.REVERSE_FLOOR_MM < dv.FORWARD_FLOOR_MM < dv.CLEARANCE_MM
    assert dv.WORST_PICTURE_AGE_S == pytest.approx(
        dv.FRAME_AGE_S + dv.FLOOR_SCAN_MAX_AGE)


def test_the_forward_floor_moves_with_the_speed(monkeypatch):
    """Read off FORWARD_SPEED when asked, never a literal: a slower cap is a
    nearer floor, and a NEAR 300 that holds the car at 0.4 does not at 0.2."""
    at_04 = dv.forward_floor_mm()
    assert replay([centre("N300")])[0] == [True]
    monkeypatch.setattr(dv, "FORWARD_SPEED", 0.2)
    at_02 = dv.forward_floor_mm()
    assert at_02 == pytest.approx(200.0 + 210.0 * dv.WORST_PICTURE_AGE_S)
    assert at_02 < 300.0 < at_04
    assert replay([centre("N300")])[0] == [False]
    assert dv.forward_floor_mm(0.0) == dv.REVERSE_FLOOR_MM
    assert dv.ForwardFloorVeto().vetoes()  # no scan yet is held


def test_an_approach_stays_held_through_the_blind_tick():
    said, veto = replay([centre("N400"), centre("N300"), centre("N190"),
                         centre("?"), centre("?")])
    assert said == [False, True, True, True, True]
    assert veto.latch.latched[4]


def test_a_blind_probe_out_of_a_clear_or_far_scene_holds_nothing():
    """The dark or featureless surface: out of open floor, out of a reading
    well outside the floor on the tick before, and with no history at all."""
    assert replay([centre("."), centre("."), centre("?"), centre("?")])[0] == [
        False, False, False, False]
    assert replay([centre("N390"), centre("?")])[0] == [False, False]
    assert replay([centre("?")])[0] == [False]
    # Something near on the far side of the fan says nothing about probe 4.
    assert replay([fake_scan("N250"), centre("?")])[0] == [True, False]


def test_a_neighbour_going_blind_counts_as_the_approach():
    """An angled approach walks the near edge across the fan."""
    said, veto = replay([fake_scan(".", ".", ".", "N250", "."), centre("?")])
    assert said == [True, True]
    assert veto.latch.latched[4]


def test_an_old_reading_counts_once_the_car_could_have_closed_it():
    """NEAR 390 then blind: not an approach on the next tick (42 mm of travel
    at 0.4 cannot cover 190), but it is once the car could have."""
    said, _ = replay([centre("N390")] + [centre("?") for _ in range(6)])
    # 200 + 420 x age passes 390 at age 0.45 s.
    assert said == [False, False, False, False, False, True, True]


def test_the_latch_lets_go_on_a_valid_reading_past_the_floor():
    # 330 is measured but still under the floor, so the latch holds through
    # it; open floor lets go, and forgets the approach, so a blind scan right
    # after is a dark patch.
    said, veto = replay([centre("N300"), centre("?"), centre("N330"),
                         centre("?"), centre("."), centre("?")])
    assert said == [True, True, True, True, False, False]
    said, veto = replay([centre("N300"), centre("?"), centre("N370"),
                         centre("N370")])
    assert said == [True, True, False, False]
    assert not any(veto.latch.latched)


def test_the_latch_lets_go_after_backing_off_and_forgets_the_approach():
    """FLOOR_RELEASE_REVERSE_MM of ESTIMATED reverse, and a reverse short of
    REVERSE_CREDIT_MIN counts for nothing however long it runs."""
    _, veto = replay([centre("N190"), centre("?")])
    assert veto.vetoes()
    # -0.30: the band that did not move the car, short of the credit line.
    veto.note_throttle(-0.30, 1.0)
    veto.note_throttle(0.0, 6.0)
    assert veto.vetoes() and veto.latch.reversed_mm == 0.0
    # At the cap, 164 mm/s: 1.22 s is the release.
    assert dv.REVERSE_SPEED <= dv.REVERSE_CREDIT_MIN
    need = dv.FLOOR_RELEASE_REVERSE_MM / dv.REVERSE_CREDIT_MM_S
    assert need == pytest.approx(1.22, abs=0.005)
    veto.note_throttle(dv.REVERSE_SPEED, 10.0)
    veto.note_throttle(dv.REVERSE_SPEED, 10.0 + need - 0.02)
    assert veto.vetoes()
    veto.note_throttle(0.0, 10.0 + need + 0.01)
    assert not veto.vetoes()
    assert veto.latch == dv.EMPTY_FLOOR_LATCH
    # Still blind where it stopped: a dark surface now, not a near one.
    veto.observe(centre("?"), 11.0)
    assert not veto.vetoes()


def test_a_slow_back_out_earns_credit_in_proportion():
    """The back-out reverses at ESCAPE_SPEED, past the moving floor: it earns
    credit, at its magnitude's share of REVERSE_CREDIT_MM_S, never the
    full-speed rate."""
    assert dv.ESCAPE_SPEED <= dv.REVERSE_CREDIT_MIN
    one_s = -dv.floor_travel_mm(dv.ESCAPE_SPEED, 1.0)
    assert one_s == pytest.approx(dv.REVERSE_CREDIT_MM_S * -dv.ESCAPE_SPEED)
    assert 0.0 < one_s < -dv.floor_travel_mm(dv.REVERSE_SPEED, 1.0)
    assert dv.floor_travel_mm(-(dv.REVERSE_MOVING_FLOOR - 0.05), 1.0) == 0.0
    _, veto = replay([centre("N190"), centre("?")])
    need = dv.FLOOR_RELEASE_REVERSE_MM / one_s
    veto.note_throttle(dv.ESCAPE_SPEED, 10.0)
    veto.note_throttle(dv.ESCAPE_SPEED, 10.0 + need - 0.05)
    assert veto.vetoes()
    veto.note_throttle(0.0, 10.0 + need + 0.01)
    assert not veto.vetoes()


def test_driving_forward_while_latched_takes_back_the_reverse_credit():
    _, veto = replay([centre("N190"), centre("?")])
    veto.note_throttle(dv.REVERSE_SPEED, 1.0)
    veto.note_throttle(dv.FORWARD_SPEED, 2.0)
    veto.note_throttle(0.0, 2.2)
    assert veto.vetoes()
    assert veto.latch.reversed_mm == pytest.approx(
        1.0 * 164.0 - 0.2 * 420.0, abs=0.5)


def test_the_same_scan_shown_twice_counts_once():
    scan = centre("N190")
    veto = dv.ForwardFloorVeto()
    for k in range(5):
        veto.observe(scan, k * 0.01)
    assert len(veto.latch.history) == 1


def test_the_floor_veto_is_pure():
    empty = dv.EMPTY_FLOOR_LATCH
    a = dv.advance_floor_latch(empty, centre("N190"), 0.0)
    b = dv.advance_floor_latch(a, centre("?"), 0.1)
    assert empty == dv.EMPTY_FLOOR_LATCH and not any(a.latched)
    assert b.latched[4]
    assert dv.advance_floor_latch(a, centre("?"), 0.1) == b
    assert dv.forward_floor_vetoes(b, centre("?"))
    assert not dv.forward_floor_vetoes(empty, centre("?"))
