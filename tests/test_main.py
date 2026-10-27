"""The steering command path in manual mode.

Two things live in here.

The mapping. The left stick used to drive the front axle and the right stick
the rear, which is a split nobody asked for every time they turn a corner --
the car has two steering servos and a corner wants both of them, counter-phase.
The left stick now drives both, and it does it by calling
driver.steering_pair(), the same function every auto steering decision goes
through. That shared call is the point and there is a test below whose whole
job is to fail if somebody copies the arithmetic into main.py instead: turn
REAR_STEER_RATIO and manual has to follow it. Nobody has yet put the wheels off
the ground and confirmed that (f, -f) is counter-phase rather than a crab on
this linkage, so the one-definition property is what makes that answer a
one-line fix instead of two.

L3 splits them back, as a toggle. The edge detection is ButtonEdge, shared
with START's mode switch, so a hold cannot retrigger either of them and
neither can raise out of a stale pad handle.

The safety shape these tests are holding: pigpio latches the last servo pulse
width and emits it forever. Every path has to command, and a read that raises
must not be able to leave the car driving or leave the two axles under
different mappings. So the toggle is polled outside drive_manual's try/except
-- inside it, a thumb on a stale handle would be swallowed into "dropped a
tick" and read as a silent neutral -- and ButtonEdge swallows the pad fault
itself, before anything has been commanded.

Nothing in here constructs a Joystick, a VehicleController or a Telemetry.
"""

import threading

import pytest

import driver as dv
import main as mn


def limited(value):
    """A stick reading as read_pad_intent sees it.

    STEERING_LIMIT_PERC is 70 and every axis read goes through it, so the
    mapping tests below would otherwise be full of 0.28s. The limit itself has
    its own test, which pins the literal.
    """
    return value * mn.STEERING_LIMIT_PERC / 100


class FakeLogger:
    def __init__(self):
        self.lines = []

    def log(self, level, subject, message):
        self.lines.append((level, subject, message))

    def csv(self, *args, **kwargs):
        pass

    def of(self, level):
        return [line for line in self.lines if line[0] == level]


class FakeHandle:
    """The pygame handle, as far as rumble() is concerned.

    Here only so the no-rumble rule has something to assert against.
    """

    def __init__(self):
        self.calls = []

    def rumble(self, low, high, duration):
        self.calls.append((low, high, duration))
        return True


class FakePad:
    """A controller. Sticks and buttons are whatever I last set them to.

    `script` is one frame of button names per loop tick, advanced by step().
    The control loop calls pygame.event.pump() exactly once a tick, so the
    integration test hangs the advance off that.
    """

    def __init__(self, raises=False, script=None):
        self.axes = {"LEFT_X": 0.0, "RIGHT_X": 0.0}
        self.throttle = 0.0
        self.down = {}
        self.raises = raises
        self.joystick = FakeHandle()
        self.script = list(script or [])
        self.frames = 0

    # ---- what main.py reads ---------------------------------------------
    def read_throttle(self):
        if self.raises:
            raise RuntimeError("stale handle")
        return self.throttle

    def get_axis(self, name, limit_perc=100):
        if self.raises:
            raise RuntimeError("stale handle")
        return self.axes[name] * (limit_perc / 100)

    def get_button(self, name):
        if self.raises:
            raise RuntimeError("stale handle")
        return self.down.get(name, False)

    def is_connected(self):
        return True

    def update_connection_status(self):
        pass

    def wait_for_connection(self, stop_event):
        pass

    # ---- test helpers ----------------------------------------------------
    def press(self, *names):
        self.down = {name: True for name in names}

    def release(self):
        self.down = {}

    def step(self):
        self.frames += 1
        if self.script:
            self.press(*self.script.pop(0))


class FakeVecon:
    def __init__(self):
        self.steerings = []
        self.throttles = []
        self.neutrals = 0
        self.closed = 0
        self.last_throttle = 0.0
        self.last_steering_front = 0.0
        self.last_steering_rear = 0.0
        self.last_esc_pulse = 1575

    def set_throttle(self, value):
        self.throttles.append(value)
        self.last_throttle = value

    def set_steering(self, front, rear):
        self.steerings.append((front, rear))
        self.last_steering_front = front
        self.last_steering_rear = rear

    def return_neutral(self):
        self.neutrals += 1
        self.set_steering(0.0, 0.0)
        self.set_throttle(0.0)

    def get_state(self):
        return {"throttle": 1575, "front_s": 90, "rear_s": 90}

    def close(self):
        self.closed += 1


class FakeAutopilot:
    unavailable = False
    driver = None

    def __init__(self):
        self.steps = 0
        self.resumes = 0
        self.pauses = 0

    def resume(self):
        self.resumes += 1
        return True

    def pause(self):
        self.pauses += 1

    def step(self, now):
        self.steps += 1
        return True

    def stop(self):
        pass

    def camera_state(self):
        return "off"

    def camera_error(self):
        return ""


class FakeMarkPad:
    def poll(self, *args, **kwargs):
        pass

    def counts(self):
        return {"marks_good": 0, "marks_bad": 0}

    def stop(self):
        pass


class FakeTele:
    """Says yes only to the channels a test asks about."""

    def __init__(self, channels=("vehicle",)):
        self.channels = set(channels)
        self.offers = []

    def wants(self, key, now):
        return key in self.channels

    def offer(self, key, payload, now):
        self.offers.append((key, dict(payload)))

    def recording_status(self):
        return {}

    def of(self, key):
        return [payload for name, payload in self.offers if name == key]


class TickLimit(threading.Event):
    """A stop_event that stops the control loop after a fixed number of passes."""

    def __init__(self, ticks):
        super().__init__()
        self.ticks = ticks

    def is_set(self):
        if self.ticks <= 0:
            return True
        self.ticks -= 1
        return False


@pytest.fixture
def pad():
    return FakePad()


@pytest.fixture
def log():
    return FakeLogger()


# --- the default mapping ----------------------------------------------------


def test_the_left_stick_drives_both_axles_by_default(pad, log):
    """The ask. Both axles off one stick, counter-phase, and the right stick
    is not read at all -- it is loaded here with a value that would be
    unmistakable if it leaked onto the rear."""
    pad.axes["LEFT_X"] = 0.4
    pad.axes["RIGHT_X"] = 0.9
    vecon = FakeVecon()

    commanded = mn.drive_manual(pad, vecon, log)

    assert vecon.steerings[-1] == pytest.approx((limited(0.4), -limited(0.4)))
    assert commanded[1:] == pytest.approx((limited(0.4), -limited(0.4)))


def test_unified_steering_is_the_one_definition_the_driver_uses(pad, log,
                                                                monkeypatch):
    """Not a second copy of the counter-phase arithmetic. REAR_STEER_RATIO is
    the single knob, and it is the one place a sign flips if the rear linkage
    turns out to be mirrored -- so my hand on the stick has to move when it
    does."""
    monkeypatch.setattr(dv, "REAR_STEER_RATIO", 0.5)
    pad.axes["LEFT_X"] = 0.4
    vecon = FakeVecon()

    mn.drive_manual(pad, vecon, log)

    front, rear = vecon.steerings[-1]
    assert front == pytest.approx(limited(0.4))
    assert rear == pytest.approx(-0.5 * limited(0.4))


def test_centre_stays_centred_on_both_axles(pad, log):
    vecon = FakeVecon()
    mn.drive_manual(pad, vecon, log)
    assert vecon.steerings[-1] == (0.0, 0.0)


def test_the_stick_limit_still_applies_to_the_unified_pair(pad, log):
    """STEERING_LIMIT_PERC is 70 and it is applied to the stick, once, before
    the pair is built -- so it bounds both axles rather than only the front."""
    pad.axes["LEFT_X"] = 1.0
    vecon = FakeVecon()

    mn.drive_manual(pad, vecon, log)

    front, rear = vecon.steerings[-1]
    assert front == pytest.approx(0.7)
    assert rear == pytest.approx(-0.7)


# --- L3 and the split -------------------------------------------------------


def test_split_gives_each_stick_its_own_axle(pad, log):
    """The current behaviour, kept as the escape hatch. No ratio, no pairing
    -- a crab walk and an odd pose are the whole point of it and a ratio is
    exactly what would take them away."""
    pad.axes["LEFT_X"] = 0.4
    pad.axes["RIGHT_X"] = 0.9
    vecon = FakeVecon()

    mn.drive_manual(pad, vecon, log, split=True)

    assert vecon.steerings[-1] == pytest.approx((limited(0.4), limited(0.9)))


def test_split_does_not_go_through_steering_pair(pad, log, monkeypatch):
    """The mirror of the unified test. Turning the ratio must NOT move split,
    or the pose it exists to make is not available."""
    monkeypatch.setattr(dv, "REAR_STEER_RATIO", 0.5)
    pad.axes["LEFT_X"] = 0.4
    pad.axes["RIGHT_X"] = 0.9
    vecon = FakeVecon()

    mn.drive_manual(pad, vecon, log, split=True)

    assert vecon.steerings[-1] == pytest.approx((limited(0.4), limited(0.9)))


def test_the_starting_state_is_unified(log):
    steer = mn.SteerMode(log)
    assert steer.split is False
    assert steer.name == mn.UNIFIED


def test_l3_flips_into_split_and_back(pad, log):
    steer = mn.SteerMode(log)

    pad.press(mn.STEER_SPLIT_BUTTON)
    assert steer.poll(pad) is True
    assert steer.split is True
    assert steer.name == mn.SPLIT

    pad.release()
    steer.poll(pad)
    pad.press(mn.STEER_SPLIT_BUTTON)
    assert steer.poll(pad) is True
    assert steer.split is False


def test_l3_is_a_toggle_and_not_a_hold(pad, log):
    """Holding the stick clicked while I steer must not chatter the mapping
    back and forth under my thumb. One press, one flip."""
    steer = mn.SteerMode(log)
    pad.press(mn.STEER_SPLIT_BUTTON)

    flips = sum(bool(steer.poll(pad)) for _ in range(40))

    assert flips == 1
    assert steer.split is True


def test_releasing_l3_does_not_flip_it_back(pad, log):
    steer = mn.SteerMode(log)
    pad.press(mn.STEER_SPLIT_BUTTON)
    steer.poll(pad)
    pad.release()

    assert steer.poll(pad) is False
    assert steer.split is True


def test_the_toggle_does_not_rumble(pad, log):
    """A mark buzzes once for a good call and twice for a bad one. A third
    meaning on the same motor would muddy a signal I have only just started
    trusting, so the journal line and the /vehicle field are the receipt."""
    steer = mn.SteerMode(log)
    for _ in range(4):
        pad.press(mn.STEER_SPLIT_BUTTON)
        steer.poll(pad)
        pad.release()
        steer.poll(pad)

    assert pad.joystick.calls == []


def test_the_journal_says_which_way_the_toggle_went(pad, log):
    steer = mn.SteerMode(log)

    pad.press(mn.STEER_SPLIT_BUTTON)
    steer.poll(pad)
    pad.release()
    steer.poll(pad)
    pad.press(mn.STEER_SPLIT_BUTTON)
    steer.poll(pad)

    messages = [line[2] for line in log.of("info")]
    assert len(messages) == 2
    assert mn.SPLIT in messages[0]
    assert mn.UNIFIED in messages[1]


def test_l3_is_a_button_nothing_else_reads(log):
    """LS was measured at 9 and left unread. If the mode toggle or a mark ever
    lands on the same index, one press would do two things."""
    import Joystick as js

    assert mn.STEER_SPLIT_BUTTON == "LS"
    assert mn.STEER_SPLIT_BUTTON != mn.MODE_TOGGLE_BUTTON
    assert (js.XBOX_BUTTONS[mn.STEER_SPLIT_BUTTON]
            != js.XBOX_BUTTONS[mn.MODE_TOGGLE_BUTTON])


# --- a pad fault cannot strand anything -------------------------------------


def test_a_pad_fault_in_manual_still_commands_neutral(log):
    """The guarantee that was already here and has to stay: pigpio holds the
    last pulse width forever, so a dropped tick has to command, not decline."""
    pad = FakePad(raises=True)
    vecon = FakeVecon()

    assert mn.drive_manual(pad, vecon, log, split=False) == (0.0, 0.0, 0.0)
    assert vecon.throttles[-1] == 0
    assert log.of("warning")


def test_a_pad_fault_cannot_raise_out_of_the_toggle_read(log):
    """The toggle is a button read and a button read off a stale handle
    raises. It has to be swallowed here, where nothing has been commanded
    yet, rather than inside drive_manual's try where it would be filed as a
    dropped tick and read as a silent neutral."""
    pad = FakePad(raises=True)
    steer = mn.SteerMode(log)

    assert steer.poll(pad) is False
    assert steer.split is False


def test_a_pad_fault_does_not_lose_a_mapping_i_already_chose(log):
    """Split through a dropout and back. Losing it would silently re-pair the
    axles under a hand that is mid-pose."""
    pad = FakePad()
    steer = mn.SteerMode(log)
    pad.press(mn.STEER_SPLIT_BUTTON)
    steer.poll(pad)

    pad.raises = True
    for _ in range(5):
        assert steer.poll(pad) is False
    pad.raises = False

    assert steer.split is True


def test_a_pad_that_comes_back_still_down_is_an_edge_again(log):
    """A re-paired handle reports whatever the button is doing now and carries
    no history with it, so the latch has to be dropped on a fault. This only
    bites once the latch has something in it: click L3, lose the pad, get it
    back with my thumb still down, and a latch kept across the dropout swallows
    that press as a continuing hold."""
    pad = FakePad()
    steer = mn.SteerMode(log)

    pad.press(mn.STEER_SPLIT_BUTTON)
    assert steer.poll(pad) is True, "down and seen -- the latch now says down"

    pad.raises = True
    assert steer.poll(pad) is False
    pad.raises = False

    assert steer.poll(pad) is True
    assert steer.split is False, "two presses, two flips"


def test_the_mode_switch_shares_the_same_edge_detection(log):
    """One definition of a press edge, not two. ModeSwitch gets the same
    hold-does-not-retrigger and fault-cannot-raise behaviour from the same
    class."""
    pad = FakePad()
    modes = mn.ModeSwitch(mn.MANUAL, log)

    pad.press(mn.MODE_TOGGLE_BUTTON)
    assert sum(bool(modes.poll(pad)) for _ in range(20)) == 1
    assert modes.mode == mn.AUTO

    pad.raises = True
    assert modes.poll(pad) is False
    assert modes.mode == mn.AUTO


# --- wired into the loop, and published -------------------------------------


def run_loop(pad, log, script, mode=mn.MANUAL, monkeypatch=None, tele=None):
    """Drive control_loop for len(script) passes with a scripted pad.

    pygame.event.pump() is called exactly once a tick at the top of the loop,
    so the pad's script advance hangs off it.
    """
    import pygame

    pad.script = list(script)
    monkeypatch.setattr(pygame.event, "pump", pad.step)
    monkeypatch.setattr(mn, "LOOP_INTERVAL", 0)

    vecon = FakeVecon()
    autopilot = FakeAutopilot()
    modes = mn.ModeSwitch(mode, log)
    steer = mn.SteerMode(log)
    tele = tele or FakeTele()

    mn.control_loop(pad, vecon, autopilot, modes, log, TickLimit(len(script)),
                    tele, FakeMarkPad(), steer)
    return vecon, modes, steer, tele


def test_the_loop_polls_the_toggle_every_tick(pad, log, monkeypatch):
    """Wired in, not just written. The mapping has to change the commands the
    loop actually sends."""
    pad.axes["LEFT_X"] = 0.4
    pad.axes["RIGHT_X"] = 0.9

    vecon, _, steer, _ = run_loop(
        pad, log, [(), (mn.STEER_SPLIT_BUTTON,), ()], monkeypatch=monkeypatch)

    assert steer.split is True
    # First tick unified, last tick split, and the right stick only ever
    # reaches the rear in the second of those.
    assert vecon.steerings[0] == pytest.approx((limited(0.4), -limited(0.4)))
    assert vecon.steerings[-1] == pytest.approx((limited(0.4), limited(0.9)))


def test_the_loop_publishes_which_mapping_is_in_force(pad, log, monkeypatch):
    """Foxglove has to be able to say which one it is, because it changes what
    the two steering numbers next to it mean."""
    _, _, _, tele = run_loop(
        pad, log, [(), (mn.STEER_SPLIT_BUTTON,), ()], monkeypatch=monkeypatch)

    published = [payload["steering_mode"] for payload in tele.of("vehicle")]
    assert published[0] == mn.UNIFIED
    assert published[-1] == mn.SPLIT


def test_the_published_field_is_in_the_vehicle_schema():
    """A field Studio has no schema for is a field Studio will not plot."""
    import telemetry as tm

    properties = tm.JSON_SCHEMAS["vehicle"]["schema"]["properties"]
    assert properties["steering_mode"] == {"type": "string"}


def test_the_mapping_survives_a_lap_in_auto(pad, log, monkeypatch):
    """It is not a property of the drive mode. Going to auto and back must
    leave the sticks mapped the way I left them."""
    pad.axes["LEFT_X"] = 0.4
    pad.axes["RIGHT_X"] = 0.9

    # split, release, START to auto, release, START back to manual, release.
    script = [(mn.STEER_SPLIT_BUTTON,), (), (mn.MODE_TOGGLE_BUTTON,), (),
              (mn.MODE_TOGGLE_BUTTON,), ()]
    vecon, modes, steer, _ = run_loop(pad, log, script, monkeypatch=monkeypatch)

    assert modes.mode == mn.MANUAL
    assert steer.split is True
    assert vecon.steerings[-1] == pytest.approx((limited(0.4), limited(0.9)))


def test_a_flip_does_not_neutral_the_wheels(pad, log, monkeypatch):
    """Unlike a mode switch, nothing is changing hands -- the next tick is
    10 ms away and commands the new mapping off the sticks where they are
    actually sitting. A neutral in between is a jolt for no reason, and the
    throttle is not touched at all."""
    pad.axes["LEFT_X"] = 0.4
    pad.throttle = 0.3

    vecon, _, _, _ = run_loop(pad, log, [(), (mn.STEER_SPLIT_BUTTON,)],
                              monkeypatch=monkeypatch)

    # One neutral, and it is the one control_loop's finally does on the way
    # out through vecon.close(). Nothing commanded a neutral on the flip.
    assert vecon.neutrals == 0
    assert all(throttle == pytest.approx(0.3) for throttle in vecon.throttles)
