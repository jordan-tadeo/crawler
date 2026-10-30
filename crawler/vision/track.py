"""The arithmetic of keeping a person in the middle of the hood camera.

Nothing in here touches hardware or a clock; every function takes the time
it needs. runner.py feeds it.

The one idea worth stating: a detection is turned into an ABSOLUTE target
angle using the pan and tilt the mount had when that frame was taken, not
the ones it has by the time the detector answers. The detector is a few
hundred milliseconds behind on the Pi, and a relative correction applied to
a mount that has already moved double-counts the move and oscillates. An
absolute target does not care how late it arrives.
"""

import math
from dataclasses import dataclass


@dataclass
class Detection:
    """One person: a confidence and a box in fractions of the frame
    (0, 0 top left, 1, 1 bottom right)."""
    score: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self):
        return 0.5 * (self.x0 + self.x1)

    @property
    def cy(self):
        return 0.5 * (self.y0 + self.y1)

    @property
    def area(self):
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def as_list(self):
        return [round(self.x0, 4), round(self.y0, 4), round(self.x1, 4),
                round(self.y1, 4), round(self.score, 3)]


def pixel_angles(u, v, hfov_deg, vfov_deg):
    """(left, up) degrees off the optical axis of a point at fractions u, v
    of the frame, through a pinhole: u = 0.5 is straight ahead, u = 0 is half
    the field of view to the left."""
    tx = math.tan(math.radians(hfov_deg) / 2.0)
    ty = math.tan(math.radians(vfov_deg) / 2.0)
    left = -math.degrees(math.atan((2.0 * u - 1.0) * tx))
    up = -math.degrees(math.atan((2.0 * v - 1.0) * ty))
    return left, up


def pick(detections, prefer=None, gate=None):
    """The person to follow, or None.

    With nobody followed yet (prefer None) the biggest box, which is the
    nearest person more often than not, by confidence on a tie. Once one is
    followed, prefer is (cx, cy) of the last box, and the box nearest it
    within `gate` (a fraction of the frame) wins, so a second person walking
    past does not steal the camera. Nobody within the gate: the biggest.
    """
    if not detections:
        return None
    if prefer is not None and gate is not None:
        px, py = prefer
        near = [(math.hypot(d.cx - px, d.cy - py), d) for d in detections]
        near = [nd for nd in near if nd[0] <= gate]
        if near:
            return min(near, key=lambda nd: nd[0])[1]
    return max(detections, key=lambda d: (d.area, d.score))


def bearing(mount_yaw_deg, pan_deg, left_deg):
    """Degrees left of the truck's straight ahead, wrapped to (-180, 180]."""
    b = (mount_yaw_deg + pan_deg + left_deg + 180.0) % 360.0 - 180.0
    return 180.0 if b == -180.0 else b


def step_toward(current, target, max_step, deadband=0.0):
    """current moved toward target by at most max_step; unmoved inside the
    deadband."""
    err = target - current
    if abs(err) <= deadband:
        return current
    return current + max(-max_step, min(max_step, err))


class Tracker:
    """Where the mount should point, updated by detections and stepped by time.

    on_detections() is called once per detector answer with the angles the
    mount had when that frame was taken; step() is called at the mount's
    step rate and returns the angles to command.

    States: "idle" before anybody is seen, "tracking" with a person in the
    last lost_s seconds, then "lost" (whatever on_lost says: back to centre,
    a scan, or hold where it is).
    """

    def __init__(self, track_cfg, camera_cfg, limits, mount_yaw_deg=0.0,
                 mount_pitch_deg=0.0):
        self.t = track_cfg
        self.cam = camera_cfg
        (self.pan_lo, self.pan_hi), (self.tilt_lo, self.tilt_hi) = limits
        self.mount_yaw = mount_yaw_deg
        self.mount_pitch = mount_pitch_deg
        self.state = "idle"
        self.target_pan = 0.0
        self.target_tilt = 0.0
        self.pan = 0.0
        self.tilt = 0.0
        self.last_seen = None
        self.last_box = None
        self.person = None
        self.bearing_deg = None
        self.elevation_deg = None
        self._scan_dir = 1.0
        self.acquired = 0

    def _clamp(self, pan, tilt):
        return (min(max(pan, self.pan_lo), self.pan_hi),
                min(max(tilt, self.tilt_lo), self.tilt_hi))

    def on_detections(self, detections, pan_at_frame, tilt_at_frame, now):
        """Take one detector answer. Returns the Detection followed, or None."""
        gate = 0.25 if self.state == "tracking" else None
        prefer = self.last_box if self.state == "tracking" else None
        d = pick(detections, prefer, gate)
        if d is None:
            self.person = None
            return None
        left, up = pixel_angles(d.cx, d.cy, self.cam.hfov_deg,
                                self.cam.vfov_deg)
        # gain under 1 under-corrects on purpose: the next detection finishes
        # the move, and a field of view that is a little off never overshoots.
        if abs(left) > self.t.deadband_deg:
            self.target_pan = pan_at_frame + self.t.gain * left
        if abs(up) > self.t.deadband_deg:
            self.target_tilt = tilt_at_frame + self.t.gain * up
        self.target_pan, self.target_tilt = self._clamp(self.target_pan,
                                                        self.target_tilt)
        if self.state != "tracking":
            self.acquired += 1
        self.state = "tracking"
        self.last_seen = now
        self.last_box = (d.cx, d.cy)
        self.person = d
        self.bearing_deg = bearing(self.mount_yaw, pan_at_frame, left)
        self.elevation_deg = self.mount_pitch + tilt_at_frame + up
        return d

    def step(self, now, dt):
        """The angles to command now, dt seconds after the last step."""
        if self.state == "tracking" and self.last_seen is not None \
                and now - self.last_seen > self.t.lost_s:
            self.state = "lost"
            self.person = None
            self.last_box = None
            self.bearing_deg = None
            self.elevation_deg = None
        if self.state == "lost":
            if self.t.on_lost == "centre":
                self.target_pan, self.target_tilt = 0.0, 0.0
            elif self.t.on_lost == "scan":
                self.target_tilt = 0.0
                nxt = self.pan + self._scan_dir * self.t.scan_deg_s * dt
                if nxt >= self.pan_hi or nxt <= self.pan_lo:
                    self._scan_dir = -self._scan_dir
                self.target_pan = nxt
            # hold: leave the targets where they were
        self.target_pan, self.target_tilt = self._clamp(self.target_pan,
                                                        self.target_tilt)
        max_step = self.t.rate_deg_s * max(0.0, dt)
        self.pan = step_toward(self.pan, self.target_pan, max_step)
        self.tilt = step_toward(self.tilt, self.target_tilt, max_step)
        return self.pan, self.tilt


class RateGovernor:
    """How long to wait between detections.

    Never more often than base_hz. Never so often that the detector averages
    more than max_duty of a core: the interval is at least the last inference
    time over max_duty. And when the Pi as a whole is short of idle CPU --
    under low_idle -- the interval doubles, down to min_hz, and only creeps
    back once there is recover_idle to spare. The SLAM worker drops scans
    when it falls behind, so the detector is the one that gives way.
    """

    def __init__(self, det_cfg):
        self.c = det_cfg
        self.base = 1.0 / det_cfg.base_hz
        self.slowest = 1.0 / det_cfg.min_hz
        self.backoff = 1.0
        self.interval = self.base

    def update(self, infer_s=None, idle_frac=None):
        if idle_frac is not None:
            if idle_frac < self.c.low_idle:
                self.backoff = min(self.backoff * 2.0, self.slowest / self.base)
            elif idle_frac > self.c.recover_idle:
                self.backoff = max(1.0, self.backoff * 0.8)
        interval = self.base * self.backoff
        if infer_s is not None and infer_s > 0:
            interval = max(interval, infer_s / self.c.max_duty)
        self.interval = min(max(interval, self.base), self.slowest)
        return self.interval


class CpuIdle:
    """The Pi's idle fraction across all cores since the last call, off
    /proc/stat. None where there is no /proc/stat or on the first call."""

    def __init__(self, path="/proc/stat"):
        self.path = path
        self._last = None

    def sample(self):
        try:
            with open(self.path) as f:
                fields = f.readline().split()
        except OSError:
            return None
        if not fields or fields[0] != "cpu":
            return None
        vals = [int(x) for x in fields[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals[:8])
        last, self._last = self._last, (idle, total)
        if last is None or total <= last[1]:
            return None
        return (idle - last[0]) / float(total - last[1])
