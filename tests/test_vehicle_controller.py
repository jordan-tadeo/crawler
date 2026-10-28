"""The pulse-width and servo-angle arithmetic, and the nan guard over it.

Why this file exists: until the hardware imports moved inside __init__, none
of this arithmetic could run anywhere but on the car, and the one time it went
wrong it put 1787 us on an ESC with nobody touching the pad. The nan guard
underneath it was added after working out that min()/max() do not clamp a nan
at all -- so a nan throttle came out of the clamp as a clean 1.0 and went to
the ESC as 2000 us, full forward.

Both of those are endpoint arithmetic on numbers. Nothing here needs a daemon.
"""

import math

import pytest

from crawler.hardware import vehicle_controller as vc


# --- throttle endpoints -----------------------------------------------------

def test_neutral_throttle_is_the_calibrated_neutral(vecon):
    assert vecon.set_throttle(0) == vc.ESC_NEUTRAL_PW == 1575


def test_full_forward_is_the_calibrated_maximum(vecon):
    assert vecon.set_throttle(1.0) == vc.ESC_FULL_FORWARD_PW == 2000


def test_full_reverse_is_the_calibrated_minimum(vecon):
    assert vecon.set_throttle(-1.0) == vc.ESC_FULL_REVERSE_PW == 1000


def test_half_forward_lands_where_it_did_on_the_car(vecon):
    """1787 us, the number off the app.log from the launch.

    int(1575 + 0.5 * 425) truncates to 1787, and 1787 is exactly what the ESC
    was being fed when the car took off on its own. If this number ever moves,
    the ESC calibration moved with it and that is a bench job, not a desk one.
    """
    assert vecon.set_throttle(0.5) == 1787


def test_throttle_is_monotonic_across_its_range(vecon):
    pulses = [vecon.set_throttle(v / 10) for v in range(-10, 11)]
    assert pulses == sorted(pulses)
    assert pulses[0] == vc.ESC_FULL_REVERSE_PW
    assert pulses[-1] == vc.ESC_FULL_FORWARD_PW


@pytest.mark.parametrize("asked", [2.0, 97.0, 1.0001])
def test_above_range_clamps_to_full_forward(vecon, asked):
    assert vecon.set_throttle(asked) == vc.ESC_FULL_FORWARD_PW


@pytest.mark.parametrize("asked", [-2.0, -97.0, -1.0001])
def test_below_range_clamps_to_full_reverse(vecon, asked):
    assert vecon.set_throttle(asked) == vc.ESC_FULL_REVERSE_PW


@pytest.mark.parametrize("asked", [-1.0, -0.6, -0.2, 0.0, 0.2, 0.6, 1.0])
def test_a_finite_throttle_is_recorded_exactly_as_asked(vecon, asked):
    """The guard must not round, re-scale or otherwise touch a good number."""
    vecon.set_throttle(asked)
    assert vecon.last_throttle == asked


# --- the nan and inf guard --------------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_throttle_parks_the_esc_at_neutral(vecon, bad):
    """This is the whole reason _usable exists.

    Without the isnan/isinf check, max(-1.0, min(1.0, nan)) returns 1.0 --
    every comparison against a nan is False, so min() never swaps -- and the
    ESC gets 2000 us. Full forward on a bad reading.
    """
    assert vecon.set_throttle(bad) == vc.ESC_NEUTRAL_PW
    assert vecon.pi.pulses[vc.ESC_GPIO_PIN] == vc.ESC_NEUTRAL_PW


def test_the_bare_clamp_really_would_have_floored_it(vecon):
    """Pins down the mechanism the guard is defending against.

    Not a test of my code -- a test of the claim in _usable's docstring, so
    that if some future Python makes min() nan-aware the guard's reasoning
    gets re-read rather than trusted.
    """
    nan = float("nan")
    assert max(-1.0, min(1.0, nan)) == 1.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_throttle_is_recorded_as_the_neutral_it_commanded(vecon, bad):
    """last_throttle feeds telemetry and the CSV. A nan there reads as a fault."""
    vecon.set_throttle(bad)
    assert vecon.last_throttle == 0
    assert vecon.last_esc_pulse == vc.ESC_NEUTRAL_PW
    assert not math.isnan(vecon.last_throttle)


def test_a_rejected_throttle_is_logged_once_not_once_per_tick(vecon):
    """The latch. This sits in a 100 Hz loop and the logger flushes a file."""
    for _ in range(50):
        vecon.set_throttle(float("nan"))
    assert len(vecon.logger.warnings()) == 1


def test_the_latch_clears_so_a_second_episode_is_a_second_line(vecon):
    vecon.set_throttle(float("nan"))
    vecon.set_throttle(0.4)
    vecon.set_throttle(float("nan"))
    assert len(vecon.logger.warnings()) == 2


def test_a_plain_zero_is_neutral_without_a_complaint(vecon):
    """0 is falsy, so _usable returns False for it -- but it means neutral,
    not a bad reading, and it must not log."""
    assert vecon.set_throttle(0) == vc.ESC_NEUTRAL_PW
    assert vecon.logger.warnings() == []


# --- steering ---------------------------------------------------------------

def test_neutral_steering_centres_both_servos(vecon):
    assert vecon.set_steering(0, 0) == (90, 90)


def test_steering_endpoints_reach_both_stops(vecon):
    """The servo direction is the inverse of the ask, which is why +1 is 0."""
    assert vecon.set_steering(1.0, 1.0) == (0, 0)
    assert vecon.set_steering(-1.0, -1.0) == (180, 180)


def test_steering_clamps_outside_the_range(vecon):
    assert vecon.set_steering(5.0, -5.0) == (0, 180)


def test_steering_angles_stay_inside_the_servo_actuation_range(vecon):
    for v in range(-20, 21):
        front, rear = vecon.set_steering(v / 10, -v / 10)
        assert 0 <= front <= 180
        assert 0 <= rear <= 180


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_steering_centres_that_servo(vecon, bad):
    """Same hole as the throttle, and it reached 180 degrees -- full lock."""
    assert vecon.set_steering(bad, bad) == (90, 90)


def test_a_bad_front_does_not_cost_a_good_rear(vecon):
    front, rear = vecon.set_steering(float("nan"), -1.0)
    assert front == 90
    assert rear == 180


def test_steering_records_what_was_asked_not_the_inverted_value(vecon):
    """A telemetry channel showing the servo-side sign reads as a steering bug."""
    vecon.set_steering(0.5, -0.25)
    assert vecon.last_steering_front == 0.5
    assert vecon.last_steering_rear == -0.25


def test_a_rejected_steering_ask_is_recorded_as_the_neutral_it_commanded(vecon):
    vecon.set_steering(float("nan"), float("nan"))
    assert vecon.last_steering_front == 0.0
    assert vecon.last_steering_rear == 0.0


def test_one_bad_steering_ask_logs_one_line_per_channel(vecon):
    """_usable is called twice per channel per call, once for the clamp and
    once for the record. It must not log twice for one bad value."""
    vecon.set_steering(float("nan"), float("nan"))
    assert len(vecon.logger.warnings()) == 2


# --- return_neutral and close -----------------------------------------------

def test_return_neutral_disarms_the_esc_and_centres_the_steering(vecon):
    vecon.set_throttle(1.0)
    vecon.set_steering(1.0, -1.0)
    vecon.return_neutral()
    assert vecon.pi.pulses[vc.ESC_GPIO_PIN] == vc.ESC_NEUTRAL_PW
    assert vecon.front_steering_servo.angle == 90
    assert vecon.rear_steering_servo.angle == 90
    assert vecon.last_throttle == 0.0
    assert vecon.last_esc_pulse == vc.ESC_NEUTRAL_PW


def test_return_neutral_does_not_touch_the_gimbal_channels(vecon):
    """Pan and tilt are not bolted to the car. return_neutral runs on every
    mode change and every controller dropout, and two I2C writes to absent
    hardware in that path is two more chances to raise."""
    before = len(vecon.pan_servo.writes), len(vecon.tilt_servo.writes)
    vecon.return_neutral()
    assert (len(vecon.pan_servo.writes), len(vecon.tilt_servo.writes)) == before


def test_close_stops_the_esc_pulse_entirely(vecon):
    vecon.set_throttle(0.8)
    vecon.close()
    assert vecon.pi.pulses[vc.ESC_GPIO_PIN] == 0
    # Neutral first, then the pulse off, and in that order.
    assert vecon.pi.writes[-2:] == [vc.ESC_NEUTRAL_PW, 0]


def test_close_still_stops_the_esc_when_a_servo_raises(vecon):
    """The ESC goes first and on its own, because one I2C error in the steering
    used to throw straight past the one step that actually stops the car."""

    class Raising:
        writes = []

        def __setattr__(self, name, value):
            raise OSError("no I2C here")

    vecon.front_steering_servo = Raising()
    vecon.close()
    assert vecon.pi.pulses[vc.ESC_GPIO_PIN] == 0


# --- the speed caps through the real mapping --------------------------------

def test_reverse_cap_is_full_reverse_through_the_real_mapping(vecon):
    """Reverse is the end of the range: 1000 us, full reverse.

    Equal pulse offset with forward was measured too slow on the floor, so
    reverse is not derived from forward any more. Through set_throttle's own
    arithmetic, not a copy of it, and inside [-1, 0) so it is still a reverse
    and the clamp never moves it.
    """
    from crawler.autopilot import driver as dv

    assert -1.0 <= dv.REVERSE_SPEED < 0
    assert vecon.set_throttle(dv.REVERSE_SPEED) == vc.ESC_FULL_REVERSE_PW == 1000
    assert vecon.set_throttle(dv.FORWARD_SPEED) > vc.ESC_NEUTRAL_PW
    d = dv.Driver(None, None)
    assert (d.forward_speed, d.reverse_speed) == (dv.FORWARD_SPEED, dv.REVERSE_SPEED)
