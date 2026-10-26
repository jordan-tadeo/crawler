import pygame

# xpadneo reports 11 buttons, which is the count I measured, and it puts the
# Xbox guide button in among them at 8 -- so the two stick clicks sit at 9 and
# 10, not 8 and 9 as this used to say. Nothing reads LS or RS today, so that
# was a wrong number waiting rather than a bug, but GUIDE belongs in here by
# name: an unnamed 8 is how it gets rediscovered as a stick click.
#
# The count and the guide at 8 are measured. The identity of A/B/X/Y/LB/RB/
# BACK/START below is xpadneo's order, which the count agrees with; I have not
# pressed each one and watched its index move. BACK 6 and START 7 are what
# they already were, so main.py's MODE_TOGGLE_BUTTON = "START" is untouched
# and still the Menu button to the right of the guide.
XBOX_BUTTONS = {
    "A": 0,
    "B": 1,
    "X": 2,
    "Y": 3,
    "LB": 4,
    "RB": 5,
    "BACK": 6,
    "START": 7,
    "GUIDE": 8,
    "LS": 9,
    "RS": 10
}

# Measured on the car under xpadneo, not copied off a diagram. The pad comes
# up to SDL as "Xbox 360 Controller", guid 050003f05e0400008e02000030110000,
# with 7 axes, 11 buttons and 1 hat, and at rest it reads:
#
#   axis 0  +0.000   left stick X
#   axis 1  +0.000   left stick Y
#   axis 2  -1.000   LEFT trigger
#   axis 3  +0.000   right stick X
#   axis 4  +0.000   right stick Y
#   axis 5  -1.000   RIGHT trigger
#   axis 6  -1.000   ABS_PROFILE, the pad's own profile switch. Unread.
#
# Triggers at 2 and 5 with the right stick between them, which is the plain
# xpad layout. This map used to say LT 5 and RT 4 with the right stick at 2
# and 3, and everything that was wrong with driving came out of that. Axis 2
# is the left trigger, and it was being handed to set_steering as the rear
# steering angle -- so a trigger resting at -1.0 held the rear servo at full
# lock the whole time the car sat still, and pulling LT swung it to the other
# stop instead of touching throttle. The code's "RT" landed on axis 4, a
# thumbstick, and that is the only reason throttle stayed at neutral:
# trigger_pull clamps a rest of 0.0, exactly as it was built to. Running the
# right stick down its own travel would have commanded throttle.
XBOX_AXES = {
    "LEFT_X": 0,
    "LEFT_Y": 1,
    "LT": 2,
    "RIGHT_X": 3,
    "RIGHT_Y": 4,
    "RT": 5
}

# How far a trigger has to come off its rest position before it counts as
# pressed at all. A fraction of the pull now, not a raw axis reading.
TRIGGER_DEADZONE = 0.05

# The only two readings this pad produces with nothing touched: an axis
# bound as a trigger sits at -1.0, and a thumbstick sits at 0.0.
TRIGGER_REST_CANDIDATES = (-1.0, 0.0)

# How close a calibration sample has to land to one of those before I
# believe it. Anything in between means the trigger was held while I
# sampled, or that axis is not what I think it is.
TRIGGER_REST_TOLERANCE = 0.1

# What I assume before I have calibrated, and whenever the sample made no
# sense. Rest at 0.0 means only the positive half of the travel commands
# anything, which is what stopped the car launching on its own, so this is
# the one fallback and it is never relaxed.
SAFE_TRIGGER_REST = 0.0


def plausible_rest(sample) -> float:
    """A rest reading I am willing to calibrate to, or the safe default.

    A sample that lands above its candidate I keep as it reads, so a trigger
    sitting at -0.97 calibrates to -0.97 and commands exactly nothing while
    it sits there -- calling it -1.0 would leave a 1.5% pull standing at
    rest. A sample that lands below its candidate I take as the candidate
    instead, because a rest lower than the real one is the one direction
    that creeps: a stick drifting to -0.06 would otherwise command 6%
    throttle sitting dead centre, which is worse than the plain clamp was.
    So rest is never below the candidate and never in between two of them.
    """
    try:
        s = float(sample)
    except (TypeError, ValueError):
        return SAFE_TRIGGER_REST
    if s != s:  # NaN
        return SAFE_TRIGGER_REST
    for candidate in TRIGGER_REST_CANDIDATES:
        if abs(s - candidate) <= TRIGGER_REST_TOLERANCE:
            return max(s, candidate)
    return SAFE_TRIGGER_REST


def trigger_pull(value, rest: float = SAFE_TRIGGER_REST) -> float:
    """A trigger axis reading turned into 0.0 at rest, 1.0 at full pull.

    (value - rest) / (1 - rest), clamped to [0, 1].

    This used to be (value + 1) / 2, which is the honest conversion for an
    Xbox trigger that reports -1.0 at rest and +1.0 pulled -- and which turns
    a reading of 0.0 into HALF THROTTLE. 0.0 is exactly what an axis reads
    when nothing has reported on it yet, and it is what a resting thumbstick
    reads when the same pad binds through xpad over USB instead of Bluetooth
    and the axis order is not the one in XBOX_AXES above. The first tick then
    commanded 0.5 and the car launched at 1787 us with nobody touching it.

    After that it was a bare clamp. Safe under either binding, but it eats
    the bottom half of the pull: the trigger does nothing until it is halfway
    in, then goes 0 to full over the back half. So I measure rest at the
    moment I pick the pad up instead of assuming it -- see
    Joystick._calibrate_triggers -- and scale from there. A rest of -1.0 is
    (value + 1) / 2 again and I get the whole travel back. A rest of 0.0,
    which is what I fall back to whenever I have no sample I trust, is the
    bare clamp again, so a resting stick on a wrongly mapped axis still
    commands nothing. Nothing in between ever gets calibrated to.
    """
    try:
        v = float(value)
        r = float(rest)
    except (TypeError, ValueError):
        return 0.0
    if v != v or r != r:  # NaN in either one
        return 0.0
    span = 1.0 - r
    if span <= 0.0:
        # Nonsense rest. Neutral beats a divide.
        return 0.0
    return max(0.0, min(1.0, (v - r) / span))


class Joystick:
    def __init__(self, disabled: bool = False, stop_event=None):
        if not pygame.joystick.get_init():
            pygame.joystick.init()

        self.connected = False
        self.joystick = None
        # Safe until a sample proves otherwise, every time I take a handle.
        self._invalidate_calibration()

        if disabled:
            self.connected = True
            return

        if pygame.joystick.get_count() == 0:
            # Can come back not connected, if the stop signal arrived first.
            self.wait_for_connection(stop_event)
            return

        try:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.connected = True
        except pygame.error as e:
            raise RuntimeError(f"Failed to initialize joystick: {e}")

        self._calibrate_triggers()

    def _invalidate_calibration(self):
        """Forget what rest read. Uncalibrated has to mean safe, not unclamped."""
        self._trigger_rest = {"RT": SAFE_TRIGGER_REST, "LT": SAFE_TRIGGER_REST}

    def _calibrate_triggers(self):
        """Learn what the two trigger axes read with nobody touching them.

        Only called the moment I take a handle, which is the one moment I
        know the pad is not being driven: the service boots and sits in
        wait_for_connection until a pad shows up, and a re-pair comes back
        through the same door. The axis order can change between hid-generic
        over Bluetooth and xpad over USB across that re-pair, which is the
        whole reason the half throttle bug existed, so the old numbers go in
        the bin every time rather than carrying over.

        Never raises, never waits. One sample, and if it is not a sample I
        believe I keep the safe rest and take it again on the next handle.
        """
        self._invalidate_calibration()
        if self.joystick is None:
            return

        try:
            pygame.event.pump()
            sampled = {
                "RT": plausible_rest(self.get_axis("RT")),
                "LT": plausible_rest(self.get_axis("LT")),
            }
        except Exception as e:
            # A handle that went stale between the device count and the read.
            # The control loop runs at 100 Hz and does not want to hear it.
            print(f"\nTrigger calibration failed, holding the safe rest: {e}")
            return

        self._trigger_rest = sampled
        # Off the end of the "Waiting..." carriage return, and the only way
        # to tell from the service log which axis map I ended up on.
        print("\nTrigger rest: RT={:.3f} LT={:.3f}".format(sampled["RT"], sampled["LT"]))

    def read_throttle(self) -> float:
        """One throttle value in [-1, 1]. Never None, never a surprise."""
        try:
            rest = self._trigger_rest
            rt = trigger_pull(self.get_axis("RT"), rest.get("RT", SAFE_TRIGGER_REST))
            lt = trigger_pull(self.get_axis("LT"), rest.get("LT", SAFE_TRIGGER_REST))
        except Exception:
            # A stale handle after a re-pair raises in here. Neutral.
            return 0.0

        if rt >= TRIGGER_DEADZONE:
            return rt
        if lt >= TRIGGER_DEADZONE:
            return -lt
        # Neither trigger pressed. This used to fall off the end of the
        # function and hand back None.
        return 0.0

    def wait_for_connection(self, stop_event=None, poll_interval: float = 1.0):
        """Wait for a controller. Returns True once one is attached.

        Interruptible, which it was not: the old loop sat on
        pygame.time.delay(1000) with no way out, so a systemctl stop blocked
        here until TimeoutStopSec expired and systemd followed up with
        SIGKILL -- which skips vecon.close() and leaves the ESC holding
        whatever pulse width it had.
        """
        while not self.connected:
            if stop_event is not None and stop_event.is_set():
                return False

            pygame.event.pump()
            print("No joystick connected. Waiting...", end="\r", flush=True)

            if pygame.joystick.get_count() > 0:
                try:
                    self.joystick = pygame.joystick.Joystick(0)
                    self.joystick.init()
                    self.connected = True
                    # New handle, possibly a new axis map. Measure rest again.
                    self._calibrate_triggers()
                    break
                except pygame.error as e:
                    raise RuntimeError(f"Failed to initialize joystick: {e}")

            if stop_event is not None:
                stop_event.wait(poll_interval)
            else:
                pygame.time.delay(int(poll_interval * 1000))

        return self.connected

    def is_connected(self):
        return self.connected

    def get_axis(self, axis_name: str, limit_perc: float = 100):
        if axis_name in XBOX_AXES:
            return self.joystick.get_axis(XBOX_AXES[axis_name]) * (limit_perc / 100)
        else:
            raise ValueError(f"Invalid axis name: {axis_name}")

    def get_button(self, button_name: str):
        if button_name in XBOX_BUTTONS:
            return self.joystick.get_button(XBOX_BUTTONS[button_name])
        else:
            raise ValueError(f"Invalid button name: {button_name}")

    def update_connection_status(self):
        """Is the handle I am holding still good?

        The device count on its own is not enough. A pad that drops and
        re-pairs can come back at a different SDL index while the count never
        reaches zero, and the handle I kept then returns stale axis values or
        raises. Clearing self.connected here sends the control loop back
        through wait_for_connection(), which re-attaches at index 0 and
        measures the trigger rest again. The calibration dies with the
        handle: whatever rest read on the old binding is not evidence about
        the next one, and an uncalibrated trigger is the clamped, safe one.
        """
        if pygame.joystick.get_count() == 0 or self.joystick is None:
            self.connected = False
            self._invalidate_calibration()
            return

        try:
            self.joystick.get_axis(0)
        except Exception:
            self.connected = False
            self._invalidate_calibration()
            return

        self.connected = True
