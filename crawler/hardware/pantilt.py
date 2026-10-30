"""The hood camera's pan/tilt mount and its picture.

What is on the car (2026-10-29, late)
-------------------------------------
The pan/tilt mount broke in a crash and is gone. The camera now is a fixed
UVC "Global Shutter Camera", USB 32e4:0234 (an AR0234 sensor, 1920x1200
global shutter, MJPG up to 90 fps), on the windshield and looking about 45
degrees up, by id
/dev/v4l/by-id/usb-Global_Shutter_Camera_Global_Shutter_Camera_01.00.00-video-index0.
deploy/pantilt.json describes it: mount.kind "none" (NullMount, no servo
writes), 1280x720 at 15 fps, pitch_deg 45.

Its field of view, fitted on a recording rather than read off a datasheet
(the lens is not named anywhere on the device): 21 pairs of a person's box
against the same person in the roof lidar's 0.5 degree scan, the truck
standing still, 2026-10-30 03:33Z. The ray through each box edge and centre
(track.ray(), pitch 45) was fitted against the lidar's bearing to the
cluster seen from the camera's place. The fit is flat from 100 to 115
degrees across (4.0-4.2 degrees rms) and clearly worse at the old guess of
90 (5.1) and below; 105 is the middle of the flat. vfov follows from square
pixels at 1280x720 (tan(vfov/2) = tan(hfov/2) * 720/1280): 72.5. Both are
ESTIMATES good to about 10 degrees: a person's box is wider than the slice
the lidar cuts through them, boxes near the frame's edge are clipped, and
the lens is wide enough to bend straight lines, which a pinhole does not
model. The camera was unplugged when this was done, so nothing was read
off v4l2. To do better, with the camera back: stand one person still at
2 m, 0 / +-20 / +-40 degrees by the lidar, and grab one frame at each;
the u of the person's centre against the lidar bearing is the curve.

x_m, y_m, z_m say where the lens is from the lidar's centre: x 0.15 forward
(on the windshield, between the lidar and the OAK-D's lens at 0.235), z
0.22 above the floor. Both are ESTIMATES, not taped.

The servo mount and the camera before (deploy/pantilt-servo.json)
------------------------------------------------------------------
The camera was a Suyin "HD Camera", USB 1e45:8022, UVC, on the Pi's USB-2
hub at 480 Mbit/s. It shows up as /dev/video0 and by id as
/dev/v4l/by-id/usb-Suyin_HD_Camera_200910120001-video-index0, which is the
name used here because video numbers move. MJPG at 320x240 up to 1280x800,
30 fps at every size.

It reports pan_absolute (-57600..57600) and tilt_absolute (-43200..43200)
controls in arc-seconds, steps of 3600, one degree. Those are not what moves
it: they come with zoom_absolute 0..3, which makes them a digital crop of the
sensor, and at zoom 0 there is nothing to crop. The mount that turns it is
two hobby servos on the PCA9685 at 0x40 on /dev/i2c-1 -- the same chip the
steering servos are on. So the config says "servo" and UvcMount is here for
a camera that really does move on its own controls.

The shared chip
---------------
VehicleController owns the PCA9685: it resets it, sets MODE1 and the 50 Hz
prescale, and writes the two steering channels. Nothing in here does any of
that. A mount writes the four LEDn_ON/OFF registers of its own two channels
and nothing else, at whatever frequency the chip is already running, and it
refuses outright to be pointed at a steering channel. In the service it
writes through VehicleController's own PCA9685 object, so it takes the same
bus lock as the steering writes and the two never interleave on the wire. A
write is about half a millisecond at 100 kHz, which is the longest a
steering write can wait behind it. tools/pantilt_cal.py, a separate process,
writes the registers raw (RawPca9685) for the same reason: constructing an
adafruit PCA9685 resets the chip, and that is not something a second process
gets to do under a running car.

Angles
------
Pan is degrees counter-clockwise seen from above, left positive, the same way
round as /goal/status's bearing_deg. Tilt is degrees up. 0, 0 is the centre
pulse of each servo, and mount.yaw_deg and mount.pitch_deg say where that
points on the truck. The pulse for an angle is centre_us + angle / deg_per_us,
negated with invert; min_us and max_us are the soft limits, and the angle
range follows from them. Every number comes from the config, which
tools/pantilt_cal.py fills in.
"""

import json
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict, is_dataclass

# The channels VehicleController steers on, by value so this file never
# imports the ESC's module. tests/test_pantilt.py checks they match.
STEERING_CHANNELS = frozenset((14, 15))
# Where the hood camera's servos are plugged in, found by sweeping each
# channel with the main battery in (it feeds the servo rail).
PAN_CHANNEL = 10
TILT_CHANNEL = 11

DEFAULT_CONFIG = os.path.join("deploy", "pantilt.json")

PCA_ADDRESS = 0x40
PCA_OSC_HZ = 25_000_000
REG_MODE1 = 0x00
REG_LED0_ON_L = 0x06
REG_PRESCALE = 0xFE
MODE1_AI = 0x20

# No servo pulse outside this, whatever a config says. Hobby servos take
# about 500-2500 us; past it a cheap one drives into its own end stop.
PULSE_FLOOR_US = 500
PULSE_CEIL_US = 2500


@dataclass
class AxisConfig:
    channel: int = PAN_CHANNEL
    min_us: float = 1100.0
    max_us: float = 1900.0
    centre_us: float = 1500.0
    invert: bool = False
    # A 180-degree hobby servo over 500-2500 us is 0.09 degrees a
    # microsecond. Measure it with tools/pantilt_cal.py and put it here.
    deg_per_us: float = 0.09

    def angle_limits(self):
        """(lowest, highest) angle the soft limits allow, in degrees."""
        a = self.angle_of(self.min_us)
        b = self.angle_of(self.max_us)
        return min(a, b), max(a, b)

    def angle_of(self, us):
        sign = -1.0 if self.invert else 1.0
        return sign * (us - self.centre_us) * self.deg_per_us

    def pulse_of(self, angle_deg):
        """The pulse for an angle, clamped to the soft limits."""
        sign = -1.0 if self.invert else 1.0
        us = self.centre_us + sign * angle_deg / self.deg_per_us
        return min(max(us, self.min_us), self.max_us)


@dataclass
class CameraConfig:
    device: str = "/dev/v4l/by-id/usb-Suyin_HD_Camera_200910120001-video-index0"
    width: int = 320
    height: int = 240
    fps: int = 15
    fourcc: str = "MJPG"
    # Unmeasured: a typical webcam. The tracker turns pixels into degrees
    # with these, so a wrong one only makes each correction a little long or
    # short, which the next detection fixes.
    hfov_deg: float = 60.0
    vfov_deg: float = 46.0


@dataclass
class MountConfig:
    kind: str = "servo"            # servo | uvc | none
    yaw_deg: float = 0.0           # where pan 0 points, left of straight ahead
    pitch_deg: float = 0.0         # where tilt 0 points, above level
    x_m: float = 0.0               # the lens from the lidar origin, forward
    y_m: float = 0.0               # left
    z_m: float = 0.22              # and above the floor
    pan: AxisConfig = field(
        default_factory=lambda: AxisConfig(channel=PAN_CHANNEL))
    tilt: AxisConfig = field(
        default_factory=lambda: AxisConfig(channel=TILT_CHANNEL))
    # For kind "uvc": the v4l2 controls and how many control units a degree
    # is. UVC's pan/tilt_absolute are arc-seconds, 3600 to the degree.
    pan_control: str = "pan_absolute"
    tilt_control: str = "tilt_absolute"
    units_per_deg: float = 3600.0


@dataclass
class DetectorConfig:
    model: str = os.path.join("models", "person.onnx")
    conf: float = 0.4
    iou: float = 0.45
    # Detections a second when the Pi has room, and the floor it backs off
    # to when it does not.
    base_hz: float = 3.0
    min_hz: float = 0.5
    # The most of one core the detector may average: the interval is never
    # shorter than the last inference time divided by this.
    max_duty: float = 0.5
    # Below this fraction of the Pi's CPU idle, halve the rate; above
    # recover_idle, creep back up.
    low_idle: float = 0.15
    recover_idle: float = 0.35
    worker: str = "process"        # process | thread
    nice: int = 10
    threads: int = 1


@dataclass
class TrackConfig:
    # Fraction of the measured angle error taken per detection.
    gain: float = 0.7
    deadband_deg: float = 1.5
    # Fastest the mount is slewed, degrees a second.
    rate_deg_s: float = 120.0
    step_hz: float = 20.0
    # No person for this long and the mount goes home or scans.
    lost_s: float = 3.0
    on_lost: str = "centre"        # centre | scan | hold
    scan_deg_s: float = 25.0


@dataclass
class PanTiltConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    mount: MountConfig = field(default_factory=MountConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    track: TrackConfig = field(default_factory=TrackConfig)

    def to_dict(self):
        return asdict(self)


class ConfigError(ValueError):
    pass


def _fill(cls, data, where):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected an object")
    known = {f for f in cls.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    defaults = cls()
    kwargs = {}
    for name, value in data.items():
        default = getattr(defaults, name)
        if is_dataclass(default):
            kwargs[name] = _fill(type(default), value, f"{where}.{name}")
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"{where}.{name}: expected true or false")
            kwargs[name] = value
        elif isinstance(default, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{where}.{name}: expected a number")
            kwargs[name] = type(default)(value) if isinstance(default, float) \
                else value
        else:
            kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(data) -> PanTiltConfig:
    cfg = _fill(PanTiltConfig, data or {}, "pantilt")
    validate(cfg)
    return cfg


def load_config(path=None) -> PanTiltConfig:
    path = path or DEFAULT_CONFIG
    with open(path) as f:
        return config_from_dict(json.load(f))


def save_config(cfg: PanTiltConfig, path):
    validate(cfg)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def validate(cfg: PanTiltConfig):
    """Raise ConfigError on anything that could move the wrong servo or
    drive one into its stop."""
    m = cfg.mount
    if m.kind not in ("servo", "uvc", "none"):
        raise ConfigError(f"mount.kind must be servo, uvc or none, not {m.kind!r}")
    if m.kind == "servo":
        if m.pan.channel == m.tilt.channel:
            raise ConfigError("pan and tilt are on the same channel")
        for name, axis in (("pan", m.pan), ("tilt", m.tilt)):
            check_channel(axis.channel)
            for key in ("min_us", "max_us", "centre_us"):
                v = getattr(axis, key)
                if not (isinstance(v, (int, float)) and math.isfinite(v)
                        and PULSE_FLOOR_US <= v <= PULSE_CEIL_US):
                    raise ConfigError(f"mount.{name}.{key} {v!r} is outside "
                                      f"{PULSE_FLOOR_US}-{PULSE_CEIL_US} us")
            if not axis.min_us <= axis.centre_us <= axis.max_us:
                raise ConfigError(f"mount.{name}: centre_us is not between "
                                  "min_us and max_us")
            if not (math.isfinite(axis.deg_per_us) and axis.deg_per_us > 0):
                raise ConfigError(f"mount.{name}.deg_per_us must be positive")
    c = cfg.camera
    for key in ("hfov_deg", "vfov_deg"):
        v = getattr(c, key)
        if not 5.0 <= v <= 170.0:
            raise ConfigError(f"camera.{key} {v!r} is not a field of view")
    d = cfg.detector
    if not 0 < d.min_hz <= d.base_hz:
        raise ConfigError("detector: need 0 < min_hz <= base_hz")
    if not 0 < d.max_duty <= 1:
        raise ConfigError("detector.max_duty must be in (0, 1]")
    if d.worker not in ("process", "thread"):
        raise ConfigError("detector.worker must be process or thread")
    t = cfg.track
    if t.on_lost not in ("centre", "scan", "hold"):
        raise ConfigError("track.on_lost must be centre, scan or hold")
    if not (t.rate_deg_s > 0 and t.step_hz > 0 and 0 < t.gain <= 1):
        raise ConfigError("track: rate_deg_s and step_hz positive, gain in (0, 1]")


def check_channel(channel):
    if not isinstance(channel, int) or not 0 <= channel <= 15:
        raise ConfigError(f"PCA9685 channel {channel!r} is not 0-15")
    if channel in STEERING_CHANNELS:
        raise ConfigError(f"channel {channel} is steering; a mount never "
                          "writes it")


# ---- the servo pulse writers ----------------------------------------------

def off_count(us, freq_hz):
    """The 12-bit LEDn_OFF count for a pulse of `us` at `freq_hz`."""
    return int(round(us * freq_hz * 4096 / 1e6))


class AdafruitChannels:
    """Pulses through VehicleController's own adafruit PCA9685 object.

    `channels[n].duty_cycle = x` writes LEDn_ON/OFF of channel n and nothing
    else, under the bus lock the steering writes take. The frequency is read
    once, never set.
    """

    def __init__(self, pca):
        self.pca = pca
        self.freq_hz = float(pca.frequency)

    def write_us(self, channel, us):
        check_channel(channel)
        count = off_count(us, self.freq_hz)
        # adafruit's duty_cycle is 16-bit and keeps the top 12 bits.
        self.pca.channels[channel].duty_cycle = min(count << 4, 0xFFFE)

    def release(self, channel):
        check_channel(channel)
        self.pca.channels[channel].duty_cycle = 0


class RawPca9685:
    """LEDn registers only, written straight to the chip. For a process that
    is not the service (tools/pantilt_cal.py): it never writes MODE1 or
    PRESCALE, never resets the chip, and reads the frequency the service set.

    `i2c` is a busio.I2C (or anything with try_lock, unlock, writeto and
    writeto_then_readfrom).
    """

    def __init__(self, i2c, address=PCA_ADDRESS):
        self.i2c = i2c
        self.address = address
        mode1 = self._read(REG_MODE1)
        if not mode1 & MODE1_AI:
            # Without auto-increment a 4-byte write lands in one register.
            raise RuntimeError(f"PCA9685 MODE1 is {mode1:#04x}, auto-increment "
                               "off: is the service up? Not writing.")
        prescale = self._read(REG_PRESCALE)
        self.freq_hz = PCA_OSC_HZ / 4096.0 / (prescale + 1)

    def _locked(self, fn):
        while not self.i2c.try_lock():
            time.sleep(0)
        try:
            return fn()
        finally:
            self.i2c.unlock()

    def _read(self, reg, n=1):
        buf = bytearray(n)
        self._locked(lambda: self.i2c.writeto_then_readfrom(
            self.address, bytes([reg]), buf))
        return buf[0] if n == 1 else bytes(buf)

    def _write_led(self, channel, on, off):
        check_channel(channel)
        reg = REG_LED0_ON_L + 4 * channel
        data = bytes([reg, on & 0xFF, on >> 8, off & 0xFF, off >> 8])
        self._locked(lambda: self.i2c.writeto(self.address, data))

    def read_us(self, channel):
        """The pulse a channel is putting out now, or None if it is off."""
        on_l, on_h, off_l, off_h = self._read(REG_LED0_ON_L + 4 * channel, 4)
        if off_h & 0x10:
            return None
        on = on_l | (on_h & 0x0F) << 8
        off = off_l | (off_h & 0x0F) << 8
        return (off - on) * 1e6 / (self.freq_hz * 4096)

    def write_us(self, channel, us):
        self._write_led(channel, 0, off_count(us, self.freq_hz))

    def release(self, channel):
        # Full-off is bit 4 of LEDn_OFF_H.
        self._write_led(channel, 0, 0x1000)


# ---- mounts -----------------------------------------------------------------

class ServoMount:
    """Two hobby servos. set() takes angles, clamps them to the soft limits and
    writes a channel only when its pulse has moved by a count or more."""

    kind = "servo"

    def __init__(self, cfg: MountConfig, writer):
        check_channel(cfg.pan.channel)
        check_channel(cfg.tilt.channel)
        self.cfg = cfg
        self.writer = writer
        self.pan_deg = 0.0
        self.tilt_deg = 0.0
        self._last = {}
        self.writes = 0

    def limits(self):
        return self.cfg.pan.angle_limits(), self.cfg.tilt.angle_limits()

    def _axis(self, axis: AxisConfig, angle):
        if not math.isfinite(angle):
            angle = 0.0
        lo, hi = axis.angle_limits()
        angle = min(max(angle, lo), hi)
        us = axis.pulse_of(angle)
        count = off_count(us, getattr(self.writer, "freq_hz", 50.0))
        if self._last.get(axis.channel) != count:
            self.writer.write_us(axis.channel, us)
            self._last[axis.channel] = count
            self.writes += 1
        return angle

    def set(self, pan_deg, tilt_deg):
        self.pan_deg = self._axis(self.cfg.pan, pan_deg)
        self.tilt_deg = self._axis(self.cfg.tilt, tilt_deg)
        return self.pan_deg, self.tilt_deg

    def centre(self):
        return self.set(0.0, 0.0)

    def close(self):
        try:
            self.centre()
        except Exception:
            pass


def _v4l2_set(device, controls, run=subprocess.run):
    arg = ",".join(f"{k}={int(v)}" for k, v in controls.items())
    run(["v4l2-ctl", "-d", device, "-c", arg], check=True,
        capture_output=True, timeout=2.0)


class UvcMount:
    """A camera that pans and tilts on its own UVC controls."""

    kind = "uvc"

    def __init__(self, cfg: MountConfig, device, limits_deg=((-16, 16), (-12, 12)),
                 run=subprocess.run):
        self.cfg = cfg
        self.device = device
        self._limits = limits_deg
        self._run = run
        self.pan_deg = 0.0
        self.tilt_deg = 0.0
        self._last = None
        self.writes = 0

    def limits(self):
        return self._limits

    def set(self, pan_deg, tilt_deg):
        (plo, phi), (tlo, thi) = self._limits
        pan = min(max(pan_deg if math.isfinite(pan_deg) else 0.0, plo), phi)
        tilt = min(max(tilt_deg if math.isfinite(tilt_deg) else 0.0, tlo), thi)
        u = self.cfg.units_per_deg
        # UVC pan_absolute is positive to the right; ours is left.
        want = {self.cfg.pan_control: round(-pan) * u,
                self.cfg.tilt_control: round(tilt) * u}
        if want != self._last:
            _v4l2_set(self.device, want, self._run)
            self._last = want
            self.writes += 1
        self.pan_deg, self.tilt_deg = pan, tilt
        return pan, tilt

    def centre(self):
        return self.set(0.0, 0.0)

    def close(self):
        try:
            self.centre()
        except Exception:
            pass


class NullMount:
    """Nothing moves: detection and bearings only, at pan 0, tilt 0."""

    kind = "none"

    def __init__(self):
        self.pan_deg = 0.0
        self.tilt_deg = 0.0
        self.writes = 0

    def limits(self):
        return (0.0, 0.0), (0.0, 0.0)

    def set(self, pan_deg, tilt_deg):
        return 0.0, 0.0

    def centre(self):
        return 0.0, 0.0

    def close(self):
        pass


def build_mount(cfg: PanTiltConfig, pca=None, run=subprocess.run):
    """The mount the config asks for. A servo mount needs the service's PCA9685
    object (VehicleController.pca)."""
    kind = cfg.mount.kind
    if kind == "servo":
        if pca is None:
            raise RuntimeError("a servo mount needs the PCA9685")
        return ServoMount(cfg.mount, AdafruitChannels(pca))
    if kind == "uvc":
        return UvcMount(cfg.mount, cfg.camera.device, run=run)
    return NullMount()


# ---- the picture ------------------------------------------------------------

class FrameGrabber:
    """The newest camera frame, read on a thread of its own.

    VideoCapture.read() waits for the next frame, so it gets its own thread
    and everybody else takes the newest frame without waiting. A camera that
    will not open or stops delivering is opened again every few seconds; until
    then latest() is None.
    """

    REOPEN_S = 3.0
    STALL_S = 3.0

    def __init__(self, cfg: CameraConfig, open_capture=None, logger=None):
        self.cfg = cfg
        self._open_capture = open_capture or self._cv2_capture
        self.logger = logger
        self._latest = None
        self._stop = threading.Event()
        self._thread = None
        self.frames = 0
        self.error = ""
        self.up = False

    def _cv2_capture(self):
        import cv2
        cap = cv2.VideoCapture(self.cfg.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {self.cfg.device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self):
        self._thread = threading.Thread(target=self._run, name="pantilt-cam",
                                        daemon=True)
        self._thread.start()

    def latest(self):
        """(frame, monotonic time it was read) or None. Never waits."""
        return self._latest

    def _log(self, level, msg):
        if self.logger is not None:
            try:
                self.logger.log(level, "PanTilt", msg)
                return
            except Exception:
                pass
        print(f"[pantilt] {msg}")

    def _run(self):
        logged = False
        while not self._stop.is_set():
            try:
                cap = self._open_capture()
            except Exception as e:
                self.error = str(e)
                if not logged:
                    self._log("warning", f"Camera would not open: {e}; trying "
                              f"again every {self.REOPEN_S:g} s")
                    logged = True
                self._stop.wait(self.REOPEN_S)
                continue
            self.up = True
            self.error = ""
            if logged:
                self._log("info", "Camera open")
            logged = False
            last = time.monotonic()
            try:
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    now = time.monotonic()
                    if ok and frame is not None:
                        self._latest = (frame, now)
                        self.frames += 1
                        last = now
                    elif now - last > self.STALL_S:
                        self.error = "no frames"
                        self._log("warning", f"Camera gave no frame for "
                                  f"{self.STALL_S:g} s, opening it again")
                        break
                    else:
                        self._stop.wait(0.01)
            finally:
                self.up = False
                try:
                    cap.release()
                except Exception:
                    pass
            self._stop.wait(self.REOPEN_S)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
