"""Stand-ins for the four pieces of hardware, so the logic can be tested off the car.

None of these pretend to be good emulators. They record what was commanded and
hand back what was last written, which is all any of these tests ask of them.

The reason this works at all is that VehicleController imports pigpio, board,
busio and the two adafruit modules inside __init__ rather than at module scope,
so the file is importable on a laptop. Constructing one for real still needs a
pigpio daemon and an I2C bus, so these tests build the object with __new__ and
fill in the attributes __init__ would have set. That is deliberate: the thing
worth testing in there is the pulse-width and servo-angle arithmetic, and that
arithmetic had never run anywhere but on the car.
"""

import pytest

import VehicleController as vc


class FakeLogger:
    """Records log lines instead of writing a file.

    VehicleController._usable logs a rejected command through this, and it
    latches so a stream of bad values says so once. The tests check that latch,
    so the lines have to be counted.
    """

    def __init__(self):
        self.lines = []

    def log(self, level, subject, message):
        self.lines.append((level, subject, message))

    def warnings(self):
        return [line for line in self.lines if line[0] == "warning"]


class FakePi:
    """The pigpio handle. Latches a pulse width the way the real one does."""

    def __init__(self):
        self.connected = True
        self.pulses = {}
        self.writes = []

    def set_mode(self, pin, mode):
        pass

    def set_servo_pulsewidth(self, pin, pulse):
        self.pulses[pin] = pulse
        self.writes.append(pulse)

    def get_servo_pulsewidth(self, pin):
        return self.pulses.get(pin, 0)

    def stop(self):
        pass


class FakeServo:
    """An adafruit_motor servo. Remembers every angle written to it."""

    def __init__(self):
        self.angle = None
        self.writes = []

    def __setattr__(self, name, value):
        if name == "angle" and "writes" in self.__dict__:
            self.__dict__["writes"].append(value)
        self.__dict__[name] = value


class FakeReleasable:
    """The PCA9685 and the I2C bus, as far as close() is concerned.

    Only deinit() matters, and it has to exist rather than be None: close()
    builds its (name, release) tuple before entering the try, so the attribute
    lookups happen outside the per-step error handling.
    """

    def __init__(self):
        self.deinit_calls = 0

    def deinit(self):
        self.deinit_calls += 1


@pytest.fixture
def vecon():
    """A VehicleController with fake hardware under it.

    Built with __new__ and then handed the attributes __init__ sets, ending
    with the real return_neutral() -- the same thing __init__ does last -- so
    the last_* bookkeeping starts out the way it does on the car.
    """
    v = vc.VehicleController.__new__(vc.VehicleController)
    v.logger = FakeLogger()
    v.pi = FakePi()
    v.i2c = FakeReleasable()
    v.pca = FakeReleasable()
    v.pan_servo = FakeServo()
    v.tilt_servo = FakeServo()
    v.front_steering_servo = FakeServo()
    v.rear_steering_servo = FakeServo()
    v._rejected = {}
    v.return_neutral()
    return v


class FakeController:
    """Stands in for VehicleController where the driver is the thing under test."""

    def __init__(self):
        self.throttles = []
        self.steerings = []

    def set_throttle(self, value):
        self.throttles.append(value)

    def set_steering(self, front, rear):
        self.steerings.append((front, rear))


class FakeCamera:
    """Stands in for a camera view. Hands back whatever it was loaded with."""

    def __init__(self, depth=None, imu=None):
        self.depth = depth
        self.imu = imu

    def get_depth_frame(self):
        return self.depth

    def get_imu_sample(self):
        return self.imu


@pytest.fixture
def d():
    """A Driver with a recording controller and an empty camera under it.

    Named short because the driver tests say it a lot. Load the camera by
    assigning d.camera.depth and d.camera.imu.
    """
    import driver as dv

    return dv.Driver(FakeController(), FakeCamera())
