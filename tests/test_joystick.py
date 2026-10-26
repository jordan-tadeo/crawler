"""The pad map and the trigger scaling.

Two separate bugs live behind this file.

The map: XBOX_AXES used to say LT 5 and RT 4 with the right stick at 2 and 3.
Axis 2 is actually the left trigger, so a trigger resting at -1.0 was being
handed to set_steering as the rear angle and held that servo at full lock the
whole time the car sat still. The map in the module is measured off the pad
under xpadneo; these tests hold its shape so a rearrangement has to be
deliberate.

The scaling: trigger_pull was (value + 1) / 2, which turns a reading of 0.0 --
what an axis reads before anything has reported on it -- into half throttle.
That is the 1787 us launch.

Nothing in here constructs a Joystick. That needs pygame to find a pad. The
two functions under test are module-level and pure.
"""

import pytest

import Joystick as js


# --- the axis and button maps -----------------------------------------------

def test_the_pad_reports_seven_axes_and_every_index_is_in_range():
    """Measured: 7 axes, 0 through 6. Axis 6 is the pad's profile switch and
    is deliberately unread, so every name here has to land in 0..5."""
    assert all(0 <= i <= 6 for i in js.XBOX_AXES.values())


def test_no_two_axis_names_share_an_index():
    """The old map had the right stick sitting on top of the triggers. Two
    names on one index is that bug's signature."""
    indices = list(js.XBOX_AXES.values())
    assert len(indices) == len(set(indices))


def test_the_triggers_are_where_the_pad_actually_puts_them():
    """LT 2 and RT 5, with the right stick between them at 3 and 4. This is
    the plain xpad layout and it is what the pad was measured reporting.
    Getting this pair wrong is what put a trigger on the rear steering servo."""
    assert js.XBOX_AXES["LT"] == 2
    assert js.XBOX_AXES["RT"] == 5
    assert js.XBOX_AXES["RIGHT_X"] == 3
    assert js.XBOX_AXES["RIGHT_Y"] == 4


def test_the_left_stick_is_where_steering_expects_it():
    assert js.XBOX_AXES["LEFT_X"] == 0
    assert js.XBOX_AXES["LEFT_Y"] == 1


def test_no_two_button_names_share_an_index():
    indices = list(js.XBOX_BUTTONS.values())
    assert len(indices) == len(set(indices))


def test_every_button_index_is_in_range_for_an_eleven_button_pad():
    """Measured: 11 buttons, 0 through 10."""
    assert all(0 <= i <= 10 for i in js.XBOX_BUTTONS.values())
    assert len(js.XBOX_BUTTONS) == 11


def test_the_guide_button_is_named_so_it_is_not_rediscovered_as_a_stick_click():
    """xpadneo puts the guide at 8, which pushes the two stick clicks to 9 and
    10. An unnamed 8 is how that gets mistaken for LS."""
    assert js.XBOX_BUTTONS["GUIDE"] == 8
    assert js.XBOX_BUTTONS["LS"] == 9
    assert js.XBOX_BUTTONS["RS"] == 10


def test_the_mode_toggle_button_main_asks_for_still_exists():
    """main.py names "START" for the mode toggle. If that key ever leaves this
    map, the mode button stops working with no import error to say so."""
    assert js.XBOX_BUTTONS["START"] == 7
    assert js.XBOX_BUTTONS["BACK"] == 6


# --- plausible_rest ---------------------------------------------------------

def test_a_trigger_resting_at_its_candidate_calibrates_to_it():
    assert js.plausible_rest(-1.0) == -1.0
    assert js.plausible_rest(0.0) == 0.0


def test_a_rest_just_above_its_candidate_is_kept_as_it_reads():
    """-0.97 calibrated to -1.0 would leave a 1.5% pull standing at rest."""
    assert js.plausible_rest(-0.97) == -0.97


def test_a_rest_below_its_candidate_is_pulled_back_up_to_it():
    """A rest lower than the real one is the direction that creeps."""
    assert js.plausible_rest(-1.05) == -1.0
    assert js.plausible_rest(-0.06) == 0.0


@pytest.mark.parametrize("between", [-0.5, -0.4, -0.75, 0.5, 0.9])
def test_a_rest_between_the_two_candidates_is_not_believed(between):
    """Either the trigger was held while I sampled or the axis is not what I
    think it is. Both mean fall back to safe."""
    assert js.plausible_rest(between) == js.SAFE_TRIGGER_REST


@pytest.mark.parametrize("junk", [None, "abc", float("nan"), object()])
def test_junk_calibrates_to_the_safe_rest(junk):
    assert js.plausible_rest(junk) == js.SAFE_TRIGGER_REST


def test_the_safe_rest_is_the_one_that_commands_nothing_at_zero():
    """A rest of 0.0 means only the positive half of the travel commands
    anything. That is what stopped the car launching on its own, so it is
    never relaxed."""
    assert js.SAFE_TRIGGER_REST == 0.0
    assert js.trigger_pull(0.0, js.SAFE_TRIGGER_REST) == 0.0


# --- trigger_pull -----------------------------------------------------------

def test_a_trigger_at_rest_commands_nothing():
    assert js.trigger_pull(-1.0, rest=-1.0) == 0.0
    assert js.trigger_pull(0.0, rest=0.0) == 0.0


def test_a_fully_pulled_trigger_commands_everything():
    assert js.trigger_pull(1.0, rest=-1.0) == 1.0
    assert js.trigger_pull(1.0, rest=0.0) == 1.0


def test_a_calibrated_trigger_gets_its_whole_travel_back():
    """Rest at -1.0 is (value + 1) / 2 again, which is the honest conversion
    for a trigger that really does rest at -1.0."""
    assert js.trigger_pull(0.0, rest=-1.0) == 0.5
    assert js.trigger_pull(-0.5, rest=-1.0) == 0.25


def test_an_uncalibrated_zero_reading_is_not_half_throttle():
    """The launch, in one line. (0.0 + 1) / 2 is 0.5, and 0.5 of the throttle
    range is 1787 us. 0.0 is exactly what an axis reads before anything has
    reported on it."""
    assert js.trigger_pull(0.0) == 0.0
    assert js.trigger_pull(0.0, rest=js.SAFE_TRIGGER_REST) == 0.0


@pytest.mark.parametrize("rest", [-1.0, -0.97, 0.0])
def test_a_pull_below_rest_never_goes_negative(rest):
    """A negative pull handed to set_throttle is reverse nobody asked for."""
    assert js.trigger_pull(-1.0, rest=rest) >= 0.0
    assert js.trigger_pull(rest - 0.5, rest=rest) == 0.0


@pytest.mark.parametrize("rest", [-1.0, 0.0])
def test_a_pull_is_monotonic_and_stays_in_the_unit_range(rest):
    pulls = [js.trigger_pull(v / 20, rest=rest) for v in range(-20, 21)]
    assert pulls == sorted(pulls)
    assert all(0.0 <= p <= 1.0 for p in pulls)


@pytest.mark.parametrize("junk", [None, "abc", float("nan")])
def test_junk_pulls_command_nothing(junk):
    assert js.trigger_pull(junk) == 0.0
    assert js.trigger_pull(0.5, rest=junk) == 0.0


def test_a_rest_at_or_above_full_pull_commands_nothing_instead_of_dividing():
    """span = 1 - rest, and a rest of 1.0 makes that zero. Neutral beats a
    ZeroDivisionError on the thread that commands the ESC."""
    assert js.trigger_pull(1.0, rest=1.0) == 0.0
    assert js.trigger_pull(1.0, rest=2.0) == 0.0


def test_the_deadzone_is_a_fraction_of_the_pull_not_a_raw_axis_reading():
    assert 0.0 < js.TRIGGER_DEADZONE < 1.0
