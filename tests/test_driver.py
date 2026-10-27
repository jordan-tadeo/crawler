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

import time

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


def test_a_recover_hands_back_centred_steering(d):
    """It used to hand back set_steering(-self._turn_dir, 0), which is the
    opposite front lock and a centred rear -- not centre, and not the
    counter-phase pair begin_recover() set. Nothing is moving by then, so
    there was nothing to counter-steer with it."""
    d.begin_recover(100.0)
    d.recovering(100.0 + dv.REVERSE_SECONDS)
    d.recovering(100.0 + dv.REVERSE_SECONDS + dv.SETTLE_SECONDS)
    assert d.controller.steerings[-1] == (0, 0)


# --- the gyro as a rear bumper ---------------------------------------------
#
# Nothing on this car points backwards, so a reverse is blind. The gyro is
# the contact sensor: commanded in reverse at -0.6 and not shaking means the
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
    assert d.controller.throttles[-1] == d.reverse_speed
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
    assert any(t == d.reverse_speed for t in d.controller.throttles)
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
    triggers are probably the no-return kind: subpixel with extended
    disparity off puts minimum range at 348 mm against a 400 mm budget, so
    there are 52 mm of graded warning before the blind zone."""
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


def test_the_check_fits_inside_the_tick_budget():
    """The control loop targets 100 Hz, measures 75, and has a 190 ms stall in
    its history, so the budget for this is sub-millisecond. Measured p50 is
    about 0.07 ms on a laptop; the bar here is loose on purpose so it catches
    a tenfold regression and never a slow machine."""
    frame = ground_depth(invalid=0.25)
    dv.free_by_column(frame)
    times = []
    for _ in range(200):
        started = time.perf_counter()
        dv.free_by_column(frame)
        times.append((time.perf_counter() - started) * 1000.0)
    assert float(np.median(times)) < 2.0, f"median {np.median(times):.3f} ms"


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
    """The ask, at the call site that was wrong. A clear scene with the room
    all on one side used to command a rear of exactly 0."""
    depth = clear_depth()
    depth[:, 320:] = 4000
    d.camera.depth = depth
    d.camera.imu = MOVING_IMU

    d.tick(100.0)

    front, rear = d.controller.steerings[-1]
    assert front != 0, "the scene has to be lopsided enough to ask for a turn"
    assert rear == -front


def test_forward_driving_goes_through_the_one_steering_definition(d, monkeypatch):
    """Not a second copy of the arithmetic. Turn the one constant and the
    forward path has to follow it."""
    monkeypatch.setattr(dv, "REAR_STEER_RATIO", 0.5)
    depth = clear_depth()
    depth[:, 320:] = 4000
    d.camera.depth = depth
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
    """The recover path is the one place the front is bounded by max_turn, so
    it is the one place the bound can be checked end to end. The turn
    direction is a random pick of four, so this asks for all of them."""
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
