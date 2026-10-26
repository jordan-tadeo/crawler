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

import numpy as np
import pytest

import driver as dv


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
    """GYRO_MOVING_MIN sits at 5x the parked max. If someone drops it to the
    floor itself, a parked car starts reading as moving."""
    assert dv.GYRO_MOVING_MIN > 0.0069


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
    assert any(t == d.reverse_speed for t in d.controller.throttles)


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
    assert d.controller.throttles[-1] == d.reverse_speed


# --- the recover phase machine ---------------------------------------------

def test_a_recover_never_blocks_and_gives_the_wheels_back(d):
    """The old recover() awaited 2.5 seconds with the throttle pinned in
    reverse, and for those 2.5 seconds nothing polled the stop signal."""
    d.begin_recover(100.0)
    assert d.recovering(100.0) is True
    assert d.recovering(100.0 + dv.REVERSE_SECONDS) is True          # into settle
    assert d.recovering(100.0 + dv.REVERSE_SECONDS + dv.SETTLE_SECONDS) is False
    assert d._phase is None


def test_a_recover_can_turn_either_way(d):
    """It used to choose between max_turn/4 and max_turn, both positive, so
    the car always pivoted the same way and a corner that beat it beat it
    forever."""
    seen = set()
    for _ in range(200):
        d.begin_recover(100.0)
        seen.add(d._turn_dir > 0)
    assert seen == {True, False}


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


# --- steering bias ----------------------------------------------------------

def test_an_even_scene_steers_straight(d):
    assert d.get_steering_bias(clear_depth()) == 0


def test_steering_bias_turns_toward_the_open_side(d):
    """Positive is steer right, so more room on the right has to come out
    positive."""
    depth = np.full((400, 640), 500, dtype=np.uint16)
    depth[:, 320:] = 4000
    assert d.get_steering_bias(depth) > 0


def test_steering_bias_stays_in_the_unit_range(d):
    depth = np.full((400, 640), 1, dtype=np.uint16)
    depth[:, 320:] = 65000
    assert -1.0 <= d.get_steering_bias(depth) <= 1.0


def test_no_valid_depth_either_side_steers_straight_rather_than_dividing(d):
    assert d.get_steering_bias(np.zeros((400, 640), dtype=np.uint16)) == 0
