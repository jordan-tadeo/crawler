"""/footprint: what the scan saw on each decision, and what the car was told.

What this is for
----------------
Training data. The next steering policy is a small CNN over a downsampled
depth image, and the person whose steering and throttle it should learn is
me. So on every decision in record mode -- hold Y in manual -- the car writes
three things at one log time: the 80x50 depth picture the policy will see
(policy_input.depth_small, on /policy/depth_small), the command my hands put
on the wheels, and the IMU. In auto it writes the same three for the
autopilot's own decisions, off the frame the driver scanned. The nine-probe
footprint scan and the gradient rules' verdict ride along in the same row:
they are what the hand-written driver would have done, which is the baseline
a learned one has to beat. The schema is in TRAINING_DATA.md.

Nothing in here is a second copy of the rules. The scan is
driver.free_by_column() and the verdict is driver.read_the_gradient(), called,
so whatever the rules become, the rows say what the rules said.

Nothing in here can drive the car
---------------------------------
pigpio latches the last pulse width forever and there is no hardware failsafe.
A recording-side computation that could reach set_steering() or set_throttle()
in record mode would be the autopilot driving under my hands without either of
us knowing which one had the wheels. So this file is never handed the vehicle
controller at all. The command it writes is a tuple of three floats the
control loop read off the controller after commanding it, and the driver it
reads in auto is read and never called -- last_scan and last_verdict are
attributes. A test holds both properties.

The rate
--------
10 Hz, which is AUTO_TICK_INTERVAL: the autopilot's decision rate, not the
camera's 16 fps and not the control loop's 100 Hz. A row is a decision, and
a policy trained at one rate and run at another is learning the wrong thing.

The cost
--------
In record the scan runs here, on the control loop, at 10 Hz -- the same scan,
on the same thread, at the same rate auto has always run it. The JSON is built
here and encoded on the publisher thread like every other channel. In auto the
scan is the driver's and this adds a read_the_gradient() call, which is a few
dozen comparisons. In both, the policy picture is shrunk here, on the control
loop, once a tick: 2.8 ms p50 and 5.0 ms p99 on the Pi 4, measured. It is
not handed to another thread to shrink because the car will shrink it on this
thread at inference, and the recording should cost what driving will. The
PNG encode is on the publisher thread.
"""

import math
import time

import driver as dv
import policy_input

# The decision rate. main.AUTO_TICK_INTERVAL is the same number; it is not
# imported from there because main imports this file.
INTERVAL = 0.1

# Every Nth record row also carries the raw depth frame it was computed from,
# lossless, on /camera/depth_raw at the row's exact log time. 5 is 2 Hz:
# enough to re-run a changed scan over a drive and check it against the rows,
# without the frames being most of the card. See the README for the measured
# size.
RAW_DEPTH_EVERY = 5

RECORD = "record"
AUTO = "auto"

IMU_FIELDS = ("imu_accel_x", "imu_accel_y", "imu_accel_z",
              "imu_gyro_x", "imu_gyro_y", "imu_gyro_z")


def _num(value, fallback=-1.0) -> float:
    try:
        value = float(value)
    except Exception:
        return fallback
    return value if math.isfinite(value) else fallback


def _sweep_clears(scan, side) -> bool:
    """driver.turn_sweep_clears(), and False if it cannot say."""
    try:
        return bool(dv.turn_sweep_clears(scan, side))
    except Exception:
        return False


def feature_row(scan, verdict, command, mode, source, shape, scan_ms=0.0,
                committed=None, pad_throttle=None, limits=None,
                steering_mode=""):
    """One /footprint message. Pure: values in, a dict out."""
    throttle, front, rear = command
    forward, reverse = limits if limits is not None else (-1.0, -1.0)
    return {
        "mode": mode,
        "source": source,
        "frame_width": int(shape[1]) if shape is not None else 0,
        "frame_height": int(shape[0]) if shape is not None else 0,
        "probe_distance_mm": [_num(d) for d in scan.distance_mm],
        "probe_valid_fraction": [_num(v, 0.0) for v in scan.valid_fraction],
        "probe_blocked": [bool(b) for b in scan.blocked],
        "probe_reason": [str(r) for r in scan.reason],
        "probe_marks": dv.probe_marks(scan),
        # Could a full-lock turn that way begin now without touching anything
        # the fan sees? The driver's own sweep test, asked both ways: side -1
        # is left and +1 right, as read_the_gradient passes it.
        "sweep_clears_left": _sweep_clears(scan, -1.0),
        "sweep_clears_right": _sweep_clears(scan, 1.0),
        "scan_ms": _num(scan_ms, 0.0),
        "verdict_action": str(verdict.action),
        "verdict_steer": _num(verdict.steer, 0.0),
        "verdict_reason": str(verdict.reason),
        "committed_action": str(committed.action) if committed else "",
        "committed_steer": _num(committed.steer, 0.0) if committed else 0.0,
        "committed_reason": str(committed.reason) if committed else "",
        "throttle": _num(throttle, 0.0),
        "steering_front": _num(front, 0.0),
        "steering_rear": _num(rear, 0.0),
        "steering_mode": str(steering_mode),
        "pad_throttle": _num(pad_throttle, 0.0),
        "forward_limit": _num(forward),
        "reverse_limit": _num(reverse),
        "depth_raw": False,
        "depth_small": False,
    }


def imu_fields(sample, sampled_at, now):
    """The IMU part of a row: a camera.get_imu_sample() dict, flattened.

    Zeros and an age of -1 when there has been no sample, so every row has
    the same columns whether or not the IMU has spoken yet.
    """
    out = dict.fromkeys(IMU_FIELDS, 0.0)
    out["imu_age_ms"] = -1.0
    if not sample:
        return out
    try:
        accel = [float(v) for v in sample["accel"]]
        gyro = [float(v) for v in sample["gyro"]]
    except Exception:
        return out
    for key, value in zip(IMU_FIELDS, accel + gyro):
        out[key] = _num(value, 0.0)
    out["imu_age_ms"] = _num((now - sampled_at) * 1000.0, -1.0)
    return out


class FootprintTap:
    """The recording side of a decision. Reads; never commands.

    `view` is a camera.StreamView of its own ("footprint"), so taking a frame
    here costs the driver and the telemetry nothing. `tele` is the Telemetry.
    Neither of them can move a wheel, and there is no third thing in here.
    """

    def __init__(self, view=None, tele=None, interval=INTERVAL,
                 raw_every=RAW_DEPTH_EVERY, auto_view=None):
        self.view = view
        self.tele = tele
        # The driver's own camera view, read for its last_depth: the frame
        # the driver scanned on the tick that made its last_scan. Read only;
        # a take here would steal the driver's next frame.
        self.auto_view = auto_view
        self.interval = interval
        self.raw_every = raw_every

        self.seq = 0
        self.rows = 0
        self._next_at = 0.0
        self._record_rows = 0
        self._last_driver_scan = None
        self._failed = False
        self._imu = None
        self._imu_at = 0.0

    def after_drive(self, now, mode, command, driver=None, pad_throttle=None,
                    limits=None, steering_mode=""):
        """Called by the control loop once a pass, after the drive call.

        `command` is (throttle, front, rear) as the controller holds it after
        this pass's command. Returns the row offered, or None. Never raises:
        a recording is never worth a tick, and this runs after the wheels have
        already been told what to do.
        """
        try:
            self._poll_imu(now)
            if mode == AUTO:
                return self._auto(now, command, driver, steering_mode)
            if mode == RECORD:
                return self._record(now, command, pad_throttle, limits,
                                    steering_mode)
            return None
        except Exception as e:
            if not self._failed:
                self._failed = True
                print(f"[footprint] dropped a row and stopped complaining: {e}")
            return None

    def _poll_imu(self, now):
        """Keep the newest IMU sample. One attribute read a pass."""
        if self.view is None:
            return
        get = getattr(self.view, "get_imu_sample", None)
        sample = get() if get is not None else None
        if sample is not None:
            self._imu, self._imu_at = sample, now

    # ---- auto: the driver's own decision ---------------------------------

    def _auto(self, now, command, driver, steering_mode):
        """A row on every pass where the driver made a new scan.

        The driver's tick is already gated at 10 Hz, so this is too. A recover
        makes no scan and so makes no row: those 2.5 seconds are a fixed
        manoeuvre, not a decision.
        """
        if driver is None or self.tele is None:
            return None
        scan = getattr(driver, "last_scan", None)
        if scan is None or scan is self._last_driver_scan:
            return None
        self._last_driver_scan = scan
        if not self.tele.wants("footprint", now):
            return None

        depth = getattr(self.auto_view, "last_depth", None)
        row = feature_row(
            scan, dv.read_the_gradient(scan), command, AUTO, "driver",
            depth.shape if depth is not None else None,
            getattr(driver, "last_scan_ms", 0.0),
            committed=getattr(driver, "last_verdict", None),
            steering_mode=steering_mode)
        return self._offer(row, now, depth)

    # ---- record: our own scan, my hands' command -------------------------

    def _record(self, now, command, pad_throttle, limits, steering_mode):
        if self.view is None or self.tele is None:
            return None
        if now < self._next_at:
            return None
        depth = self.view.get_depth_frame()
        if depth is None:
            # Camera still coming up, or no frame since the last row. The
            # gate does not advance, so the next pass with a frame is a row.
            return None
        self._next_at = now + self.interval

        if not self.tele.wants("footprint", now):
            return None

        started = time.perf_counter()
        scan = dv.free_by_column(depth)
        scan_ms = (time.perf_counter() - started) * 1000.0
        verdict = dv.read_the_gradient(scan)

        row = feature_row(scan, verdict, command, RECORD, "tap", depth.shape,
                          scan_ms, pad_throttle=pad_throttle, limits=limits,
                          steering_mode=steering_mode)

        # The frame first, at the same `now`, so its log time is the row's.
        if (self._record_rows % self.raw_every == 0
                and self.tele.wants("depth_raw", now)):
            self.tele.offer("depth_raw", depth, now)
            row["depth_raw"] = True
        self._record_rows += 1
        return self._offer(row, now, depth)

    def _offer(self, row, now, depth=None):
        """The row, its IMU snapshot and its policy picture, all at `now`.

        The picture goes first and carries frame_id "footprint/<seq>", so a
        reader can pair it to its row by log time or by seq, and the two can
        never disagree.
        """
        self.seq += 1
        row["seq"] = self.seq
        row.update(imu_fields(self._imu, self._imu_at, now))
        if depth is not None and self.tele.wants("depth_small", now):
            try:
                small = policy_input.depth_small(depth)
            except ValueError as e:
                # A frame the policy could not take. The row still goes,
                # saying it has no picture.
                small = None
                if not self._failed:
                    self._failed = True
                    print(f"[footprint] no policy picture: {e}")
            if small is not None:
                self.tele.offer("depth_small",
                                (small, f"footprint/{self.seq}"), now)
                row["depth_small"] = True
        self.tele.offer("footprint", row, now)
        self.rows += 1
        return row
