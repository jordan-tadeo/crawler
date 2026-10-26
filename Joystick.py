import pygame

XBOX_BUTTONS = {
    "A": 0,
    "B": 1,
    "X": 2,
    "Y": 3,
    "LB": 4,
    "RB": 5,
    "BACK": 6,
    "START": 7,
    "LS": 8,
    "RS": 9
}

XBOX_AXES = {
    "LEFT_X": 0,
    "LEFT_Y": 1,
    "LT": 5,
    "RIGHT_X": 2,
    "RIGHT_Y": 3,
    "RT": 4
}

# How far a trigger has to come off its rest position before it counts as
# pressed at all.
TRIGGER_DEADZONE = 0.05


def trigger_pull(value) -> float:
    """A trigger axis reading turned into 0.0 at rest, 1.0 at full pull.

    This used to be (value + 1) / 2, which is the honest conversion for an
    Xbox trigger that reports -1.0 at rest and +1.0 pulled -- and which turns
    a reading of 0.0 into HALF THROTTLE. 0.0 is exactly what an axis reads
    when nothing has reported on it yet, and it is what a resting thumbstick
    reads when the same pad binds through xpad over USB instead of Bluetooth
    and the axis order is not the one in XBOX_AXES above. The first tick then
    commanded 0.5 and the car launched at 1787 us with nobody touching it.

    So only the positive half of the travel commands anything: zero is
    neutral, and so is everything below it. That costs me the bottom half of
    the trigger pull as a dead band and I am happy to pay it, because the
    failure I am designing against is the car driving off on its own.
    """
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


class Joystick:
    def __init__(self, disabled: bool = False, stop_event=None):
        if not pygame.joystick.get_init():
            pygame.joystick.init()

        self.connected = False
        self.joystick = None

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

    def read_throttle(self) -> float:
        """One throttle value in [-1, 1]. Never None, never a surprise."""
        try:
            rt = trigger_pull(self.get_axis("RT"))
            lt = trigger_pull(self.get_axis("LT"))
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
        through wait_for_connection(), which re-attaches at index 0.
        """
        if pygame.joystick.get_count() == 0 or self.joystick is None:
            self.connected = False
            return

        try:
            self.joystick.get_axis(0)
        except Exception:
            self.connected = False
            return

        self.connected = True
