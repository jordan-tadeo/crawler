"""Find the reverse throttle whose ground speed matches forward 0.4.

On the car, with the service stopped (this opens the ESC, the pad and the
OAK-D itself, and the service owns all three while it runs):

    sudo systemctl stop crawler
    cd /home/jt/Documents/py/crawler
    ./.venv/bin/python deploy/speed_cal.py
    sudo systemctl start crawler

Anywhere, with no hardware at all -- a simulated car with an ESC deadband
model, a simulated owner who presses RB when asked, synthetic depth:

    ./.venv/bin/python deploy/speed_cal.py --dry-run

What it does. The car faces a flat box or a wall about 1.6 m away. It drives
legs, one per RB press, and measures each one off the depth camera: forward
0.4 toward the box, reverse at -0.3, -0.4 ... -1.0 away from it. Each leg
skips its first 0.4 s (the ESC spinning up) and fits a line through centre
distance against time over the rest, so a leg is a speed in mm/s with a fit
residual and a frame count. Then both axles go to full lock and it steps the
throttle up until the car actually moves, both ways. Out the other end comes
the reverse throttle that matches forward 0.4, the lowest throttle that moves
the car at full lock, and a REVERSE_SPEED to put in driver.py. Everything goes
to log/speed_cal-<utc>.json as well, raw samples included.

The car only moves while RB is held, and every leg needs its own fresh press:
let go between legs, press and hold when the pad buzzes. A press held over
from the last leg does not start the next one, so a pad that drops out with
RB down can drive at most the one leg it was already in, and every leg has a
hard end of its own (see the bounds below). B stops the whole run.

pigpio latches the last pulse width forever and there is no failsafe under
it, so stopping here always means commanding neutral, never just returning:
every leg ends through _halt(), every exit path goes through the finally in
main(), and SIGINT, SIGTERM and SIGHUP (the ssh session dropping) all command
neutral from inside the handler before anything else runs. The one thing that
can still leave a pulse latched is SIGKILL.
"""

import argparse
import datetime as _dt
import json
import math
import os
import signal
import sys
import time
from typing import NamedTuple, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import driver as dv  # noqa: E402
import main as car_main  # noqa: E402  (light: pygame, pigpio and depthai are all deferred)

# ---------------------------------------------------------------------------
# The protocol's numbers
# ---------------------------------------------------------------------------

# The forward speed being matched. A literal and not dv.FORWARD_SPEED on
# purpose: this run answers "what matches 0.4", and if FORWARD_SPEED moves the
# answer to that question does not.
FORWARD_LEVEL = 0.4
REVERSE_LEVELS = (-0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0)
LOCK_REVERSE_LEVELS = REVERSE_LEVELS
LOCK_FORWARD_LEVELS = (0.2, 0.25, 0.3, 0.35, 0.4)

# Full lock exactly as manual sends it at full stick: stick_steer(1.0) is
# STEERING_LIMIT_PERC of travel, and steering_pair puts the rear counter-phase.
LOCK_STEERING = dv.steering_pair(car_main.stick_steer(1.0))
CENTRE = (0.0, 0.0)

LOOP_S = 0.01               # the leg loop, same 100 Hz main runs at
SETTLE_S = 0.4              # dropped from the start of every fit
MIN_FIT_FRAMES = 5
MIN_FIT_SPAN_S = 0.25

# Hard bounds. These end a leg early whatever the leg was doing.
FORWARD_MIN_MM = 450.0      # never forward with the box closer than this, or unmeasured
REVERSE_MAX_S = 1.2         # never reverse longer than this in one leg
REVERSE_MAX_MM = 2200.0     # never reverse past this, or with the range lost
CAMERA_STALL_S = 0.3        # no new depth frame for this long: neutral
IMU_STALL_S = 0.5           # lock legs only, they decide on the gyro
CAMERA_DEAD_S = 10.0        # nothing at all from the camera while waiting: exit

# Where the straight legs start and stop. Forward runs from far down to
# FORWARD_STOP_MM and is the measurement; reverse then runs from near back
# out. A forward leg needs room for a fit, so it is skipped when the car is
# already close, and a reverse leg needs room behind it, so the car is
# brought in first when it is far.
FORWARD_STOP_MM = 700.0
FORWARD_MAX_S = 4.0
FORWARD_MIN_START_MM = 1100.0
REVERSE_START_MAX_MM = 1000.0
MIN_FORWARD_FITS = 3

# The lock test.
LOCK_STEP_S = 0.8
STEER_SETTLE_S = 0.4        # servos reach lock before the throttle comes up
LOCK_REST_S = 1.5           # gyro sampled at rest, at lock, to set the threshold
LOCK_WINDOW_S = 0.25        # yaw is the mean over this trailing window
LOCK_DETECT_DELAY_S = 0.15
LOCK_GYRO_FLOOR = 0.02      # rad/s; the rest measurement can only raise it
LOCK_GYRO_REST_FACTOR = 3.0
LOCK_CONFIRM_S = 0.1        # yaw over threshold this long, unbroken, before it counts
# Or the box moved this much. A backstop and not the main test: with extended
# disparity and no subpixel, depth comes in whole disparity levels, and one
# level is Z^2 / (fx * baseline) = 44 mm at 1.2 m, so anything under about two
# levels is one level flickering.
LOCK_DEPTH_MM = 100.0
LOCK_MARGIN = 0.05          # added to the lock threshold for the recommendation

REST_S = 0.8                # neutral before the next prompt, so the car has stopped
MAX_ATTEMPTS = 3            # an aborted leg is offered again this many times

# The centre distance. NOT free_by_column's centre probes, and this is the
# one place I departed from the plan. Those probes are the obstacle check's
# band, and its bottom rows are cut to land on the floor inside 900 mm --
# "the band's bottom rows always terminate on the ground", in driver.py's own
# words -- and a probe's distance is its 2nd-percentile depth. So with the box
# past about 900 mm the centre probes read the floor in front of the car, not
# the box, and a 1.6 m approach would fit at nearly zero mm/s. Instead: the
# median of a small patch just above the optical centre, which can only ever
# see the box, because a level camera 165 mm up never sees the floor above its
# own horizon. At 640 wide it is 48 px across and 10 px tall, which on the box
# is 240 mm wide and 50 mm tall at 2.2 m. The probes still guard the forward
# bound, where reading the floor at 900 is harmless.
ROI_HALF_WIDTH_AT_640 = 24
ROI_TOP_AT_400 = 12         # rows above centre
ROI_BOTTOM_AT_400 = 2
ROI_MIN_VALID = 0.5
CENTRE_PROBES = (3, 4, 5)

FORWARD, REVERSE = "forward", "reverse"
ABORTS = ("released", "pad", "camera stall", "imu stall")


class Stop(Exception):
    """The owner asked for the run to end: B, Ctrl-C, SIGTERM, SIGHUP."""


def say(*parts):
    """Print, and never raise. After a SIGHUP the terminal is gone."""
    try:
        print(*parts, flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pure pieces: distance, fit, match, recommendation
# ---------------------------------------------------------------------------

class Range(NamedTuple):
    centre_mm: Optional[float]  # the ROI median, or None when not measurable
    probe_min_mm: float         # nearest of all nine probes (inf when none)
    centre_probe_mm: float      # nearest of the centre three (inf when blind)


def roi_bounds(shape):
    h, w = shape[0], shape[1]
    cy, cx = h // 2, w // 2
    half = max(1, int(round(ROI_HALF_WIDTH_AT_640 * w / 640.0)))
    top = max(0, cy - int(round(ROI_TOP_AT_400 * h / 400.0)))
    bottom = max(top + 1, cy - int(round(ROI_BOTTOM_AT_400 * h / 400.0)))
    return top, bottom, max(0, cx - half), min(w, cx + half)


def measure_range(depth) -> Range:
    top, bottom, left, right = roi_bounds(depth.shape)
    patch = np.asarray(depth[top:bottom, left:right]).astype(np.float64).ravel()
    valid = patch[patch > 0]
    centre = None
    if patch.size and valid.size >= ROI_MIN_VALID * patch.size:
        centre = float(np.median(valid))
    scan = dv.free_by_column(depth)
    dist = np.asarray(scan.distance_mm, dtype=np.float64)
    return Range(centre, float(np.min(dist)),
                 float(np.min(dist[list(CENTRE_PROBES)])))


class Fit(NamedTuple):
    speed_mm_s: Optional[float]  # positive = moving the way the leg commanded
    residual_mm: Optional[float]
    frames: int


def fit_speed(samples, start_t, direction, settle_s=SETTLE_S) -> Fit:
    """A line through (time, centre distance) after the settle.

    Forward closes on the box, so its speed is the negated slope; reverse
    opens the distance, so it is the slope. Signed that way, a leg that went
    the wrong way comes out negative instead of looking fine.
    """
    pts = [(t, d) for t, d in samples
           if t >= start_t + settle_s and d is not None and math.isfinite(d)]
    n = len(pts)
    if n < MIN_FIT_FRAMES or pts[-1][0] - pts[0][0] < MIN_FIT_SPAN_S:
        return Fit(None, None, n)
    t = np.array([p[0] for p in pts])
    d = np.array([p[1] for p in pts])
    slope, intercept = np.polyfit(t - t[0], d, 1)
    resid = d - (slope * (t - t[0]) + intercept)
    speed = -slope if direction == FORWARD else slope
    return Fit(float(speed), float(np.sqrt(np.mean(resid ** 2))), n)


class Match(NamedTuple):
    throttle: Optional[float]   # negative, or None
    note: str


def match_reverse(points, target_mm_s) -> Match:
    """The reverse throttle whose speed is target, by linear interpolation.

    points is (throttle, speed) for the reverse legs that fit. Walked in
    order of |throttle| and the first crossing wins, so one noisy leg past
    the crossing cannot move it.
    """
    pts = sorted(((abs(x), s) for x, s in points if s is not None),
                 key=lambda p: p[0])
    if target_mm_s is None or not pts:
        return Match(None, "nothing to match: no forward speed or no reverse legs")
    if pts[0][1] >= target_mm_s:
        return Match(-pts[0][0], f"already at or over forward speed at the lowest "
                     f"level tested, {-pts[0][0]:.2f}; the match is at or below it")
    for (a, sa), (b, sb) in zip(pts, pts[1:]):
        if sa < target_mm_s <= sb:
            x = a + (target_mm_s - sa) * (b - a) / (sb - sa)
            return Match(-x, f"between {-a:.2f} ({sa:.0f} mm/s) and "
                         f"{-b:.2f} ({sb:.0f} mm/s)")
    return Match(None, f"reverse never reached {target_mm_s:.0f} mm/s; fastest was "
                 f"{max(s for _, s in pts):.0f} mm/s")


class Recommendation(NamedTuple):
    reverse_speed: Optional[float]
    pulse_us: Optional[int]
    conflict: bool
    note: str


def esc_pulse(throttle) -> int:
    """The pulse VehicleController.set_throttle sends for this value."""
    import VehicleController as vc
    if throttle > 0:
        return int(vc.ESC_NEUTRAL_PW + throttle * (vc.ESC_FULL_FORWARD_PW - vc.ESC_NEUTRAL_PW))
    if throttle < 0:
        return int(vc.ESC_NEUTRAL_PW - abs(throttle) * (vc.ESC_NEUTRAL_PW - vc.ESC_FULL_REVERSE_PW))
    return vc.ESC_NEUTRAL_PW


def _up(magnitude):
    """Round a magnitude up to the next 0.01, ignoring float dust."""
    return min(1.0, math.ceil(round(magnitude * 100.0, 6)) / 100.0)


def recommend(matched, lock_threshold, lock_tested=True, margin=LOCK_MARGIN) -> Recommendation:
    """max(|matched|, |lock threshold| + margin), as a negative throttle.

    lock_threshold is the lowest reverse level that moved at full lock, or
    None when none did. lock_tested False means the lock test never ran, which
    is not the same thing as "never moved" and is reported differently.
    """
    lock_need = None if lock_threshold is None else abs(lock_threshold) + margin
    if lock_tested and lock_threshold is None:
        mag = 1.0
        note = ("reverse never moved at full lock, even at -1.0: nothing "
                "here both matches forward 0.4 and gets out of a corner at lock")
        if matched is not None:
            note += f"; the speed match alone is {matched:.3f}"
        return Recommendation(-mag, esc_pulse(-mag), True, note)
    if matched is None and lock_need is None:
        return Recommendation(None, None, False, "no speed match and no lock test: no recommendation")
    if matched is None:
        mag = _up(lock_need)
        return Recommendation(-mag, esc_pulse(-mag), False,
                              "no speed match; this is the lock threshold plus margin only")
    if lock_need is None:
        mag = _up(abs(matched))
        return Recommendation(-mag, esc_pulse(-mag), False,
                              "lock test not run; this is the speed match only")
    if lock_need > abs(matched):
        mag = _up(lock_need)
        return Recommendation(-mag, esc_pulse(-mag), True,
                              f"CONFLICT: the speed match {matched:.3f} stalls at full lock "
                              f"(lock moves at {lock_threshold:.2f}); taking the lock "
                              f"threshold plus {margin:.2f}, so reverse is faster than "
                              f"forward 0.4")
    mag = _up(abs(matched))
    return Recommendation(-mag, esc_pulse(-mag), False,
                          f"the speed match; it clears the lock threshold "
                          f"{lock_threshold:.2f} plus {margin:.2f}")


# ---------------------------------------------------------------------------
# The calibrator
# ---------------------------------------------------------------------------

class PadState(NamedTuple):
    ok: bool
    rb: bool
    abort: bool
    reason: str = ""


class LegSpec(NamedTuple):
    name: str
    direction: str
    throttle: float
    max_s: float
    steering: tuple = CENTRE
    stop_at_mm: Optional[float] = None
    lock: bool = False


class Calibrator:
    """Runs legs against a vehicle, a pad and a camera, all injected.

    vehicle: throttle(v), steer(front, rear), close()
    pad:     poll() -> PadState, prompt()
    camera:  poll(now) -> (depth [(t_capture, t_rx, frame)], gyro [(t, gx)]), close()
    clock:   now(), sleep(dt)
    """

    def __init__(self, vehicle, pad, camera, clock, carry=None):
        self.vehicle = vehicle
        # Called with a distance in mm whenever a prompt asks the owner to
        # carry the car somewhere. None on the car, where he does it by hand;
        # the simulation moves its car.
        self.carry = carry
        self.pad = pad
        self.camera = camera
        self.clock = clock
        self.range = Range(None, math.inf, math.inf)
        self.last_frame_rx = None
        self.last_imu_rx = None
        self.started = clock.now()
        self.steering = None
        self.legs = []          # every LegResult dict, aborted ones too
        self.gyro_bias = 0.0
        self.gyro_threshold = None
        self.gyro_rest = None
        self._buf = None        # (depth samples, gyro samples) while a leg runs
        self._yaw_over_since = None

    # -- hardware wrappers ---------------------------------------------------

    def _halt(self, centre=False):
        """Neutral now. Every leg ends through here."""
        try:
            self.vehicle.throttle(0.0)
        finally:
            if centre:
                self._steer(CENTRE)

    def _steer(self, pair):
        self.vehicle.steer(*pair)
        self.steering = tuple(pair)

    def _read(self, now):
        depth, gyro = self.camera.poll(now)
        for t_cap, t_rx, frame in depth:
            self.range = measure_range(frame)
            self.last_frame_rx = t_rx
            if self._buf is not None:
                self._buf[0].append((t_cap, self.range.centre_mm))
        for t, gx in gyro:
            self.last_imu_rx = t
            if self._buf is not None:
                self._buf[1].append((t, gx))

    def _camera_fresh(self, now):
        return self.last_frame_rx is not None and now - self.last_frame_rx <= CAMERA_STALL_S

    def distance(self):
        return self.range.centre_mm

    # -- waiting for the owner -----------------------------------------------

    def wait_for_press(self, prompt, rest_s=REST_S, carry_to=None):
        """Block until a FRESH RB press with the pad and camera both healthy.

        Neutral the whole time. A press only counts after RB has been seen
        up, so holding RB from the last leg never starts the next one.
        """
        self._halt()
        start = self.clock.now()
        say(prompt)
        try:
            self.pad.prompt()
        except Exception:
            pass
        if carry_to is not None and self.carry is not None:
            self.carry(carry_to)
        seen_up = False
        last_note = None
        while True:
            now = self.clock.now()
            pad = self.pad.poll()
            if pad.abort:
                raise Stop("B pressed")
            self._read(now)
            note = None
            if self.last_frame_rx is None and now - start > CAMERA_DEAD_S:
                raise RuntimeError("no depth frame at all in "
                                   f"{CAMERA_DEAD_S:.0f} s; the camera is not running")
            if (self.last_frame_rx is not None
                    and now - self.last_frame_rx > CAMERA_DEAD_S):
                raise RuntimeError(f"the camera has sent nothing for {CAMERA_DEAD_S:.0f} s")
            if not pad.ok:
                seen_up = False
                note = f"  (pad not usable: {pad.reason})"
            elif not pad.rb:
                seen_up = True
            elif seen_up and now - start >= rest_s:
                if self._camera_fresh(now):
                    return
                note = "  (waiting for the camera)"
            if note and note != last_note:
                say(note)
            last_note = note
            self.clock.sleep(LOOP_S)

    # -- one leg ---------------------------------------------------------------

    def _bound(self, spec, now, t0, pad, depth_buf, gyro_buf):
        """Why this leg must stop right now, or None. Checked every loop."""
        if not pad.ok:
            return "pad", pad.reason
        if not pad.rb:
            return "released", "RB let go"
        if not self._camera_fresh(now):
            return "camera stall", "no depth frame for over %.1f s" % CAMERA_STALL_S
        if spec.lock and (self.last_imu_rx is None or now - self.last_imu_rx > IMU_STALL_S):
            return "imu stall", "no gyro for over %.1f s" % IMU_STALL_S
        d = self.range.centre_mm
        if spec.direction == FORWARD:
            near = min(self.range.probe_min_mm, self.range.centre_probe_mm)
            if d is None or d < FORWARD_MIN_MM or near < FORWARD_MIN_MM:
                return "near", f"centre {d} mm, probes {near:.0f} mm, under {FORWARD_MIN_MM:.0f}"
            if spec.stop_at_mm is not None and d <= spec.stop_at_mm:
                return "target", f"reached {d:.0f} mm"
        else:
            if d is None:
                return "range lost", "no centre distance while reversing"
            if d >= REVERSE_MAX_MM:
                return "far", f"{d:.0f} mm, at the {REVERSE_MAX_MM:.0f} mm limit"
        if t0 is not None:
            # The reverse cap is the constant, not whatever the leg asked for,
            # so no spec can buy a longer blind reverse.
            limit = spec.max_s if spec.direction == FORWARD else min(spec.max_s, REVERSE_MAX_S)
            if now - t0 >= limit:
                return "time", f"{limit:.1f} s"
            if spec.lock and self._moved(t0, now, depth_buf, gyro_buf):
                return "moved", "the car moved at lock"
        return None

    def _yaw(self, t0, now, gyro_buf):
        """Mean bias-corrected gyro_x over the trailing window, 0 until it is full.

        A part-filled window is a mean over a handful of samples, and that is
        noise big enough to read as motion, so nothing is said until the
        window has been filling for LOCK_WINDOW_S after the detect delay.
        """
        start = t0 + LOCK_DETECT_DELAY_S
        if now - start < LOCK_WINDOW_S:
            return 0.0
        window = [gx - self.gyro_bias for t, gx in gyro_buf if t >= now - LOCK_WINDOW_S]
        if len(window) < 3:
            return 0.0
        return float(np.mean(window))

    def _depth_change(self, t0, depth_buf):
        pts = [d for t, d in depth_buf if t >= t0 and d is not None]
        if len(pts) < 2:
            return 0.0
        return float(pts[-1] - pts[0])

    def _moved(self, t0, now, depth_buf, gyro_buf):
        """Yaw over threshold for LOCK_CONFIRM_S unbroken, or the box moved.

        Unbroken because a stalled ESC still buzzes the chassis, which the
        rest measurement cannot see (it is taken at neutral), and one window
        that happens to average high is not a car turning. A car that is
        really turning stays over it.
        """
        threshold = self.gyro_threshold if self.gyro_threshold is not None else LOCK_GYRO_FLOOR
        if abs(self._yaw(t0, now, gyro_buf)) > threshold:
            if self._yaw_over_since is None:
                self._yaw_over_since = now
        else:
            self._yaw_over_since = None
        turning = (self._yaw_over_since is not None
                   and now - self._yaw_over_since >= LOCK_CONFIRM_S)
        return turning or abs(self._depth_change(t0, depth_buf)) > LOCK_DEPTH_MM

    def run_leg(self, spec: LegSpec) -> dict:
        """Drive one leg. Returns its record. Always leaves the car at neutral."""
        depth_buf, gyro_buf = [], []
        self._buf = (depth_buf, gyro_buf)
        self._yaw_over_since = None
        t0 = None
        start_mm = self.range.centre_mm
        ended, why = None, ""
        peak_yaw = 0.0
        try:
            if self.steering != tuple(spec.steering):
                self._steer(spec.steering)
                settle_until = self.clock.now() + STEER_SETTLE_S
            else:
                settle_until = self.clock.now()
            while True:
                now = self.clock.now()
                pad = self.pad.poll()
                if pad.abort:
                    raise Stop("B pressed")
                self._read(now)
                if spec.lock and t0 is not None:
                    peak_yaw = max(peak_yaw, abs(self._yaw(t0, now, gyro_buf)))
                stop = self._bound(spec, now, t0, pad, depth_buf, gyro_buf)
                if stop:
                    ended, why = stop
                    break
                if now >= settle_until:
                    if t0 is None:
                        t0 = now
                        start_mm = self.range.centre_mm
                    self.vehicle.throttle(spec.throttle)
                self.clock.sleep(LOOP_S)
        finally:
            self._buf = None
            self._halt(centre=(ended in ABORTS) or ended is None)
        end_now = self.clock.now()
        if ended in ABORTS:
            outcome = "aborted"
        elif t0 is None:
            outcome = "refused"
        else:
            outcome = "done"
        fit = fit_speed(depth_buf, t0, spec.direction) if t0 is not None else Fit(None, None, 0)
        rec = {
            "name": spec.name,
            "direction": spec.direction,
            "throttle": spec.throttle,
            "pulse_us": esc_pulse(spec.throttle),
            "steering": list(spec.steering),
            "lock": spec.lock,
            "outcome": outcome,
            "ended": ended,
            "why": why,
            "at_s": round((t0 if t0 is not None else end_now) - self.started, 3),
            "drive_s": round(end_now - t0, 3) if t0 is not None else 0.0,
            "start_mm": start_mm,
            "end_mm": self.range.centre_mm,
            "speed_mm_s": fit.speed_mm_s,
            "residual_mm": fit.residual_mm,
            "fit_frames": fit.frames,
            "frames": len(depth_buf),
            "valid_frames": sum(1 for _, d in depth_buf if d is not None),
            "samples": [[round(t - (t0 or 0.0), 4), d] for t, d in depth_buf],
        }
        if spec.lock and t0 is not None:
            rec["moved"] = ended == "moved"
            rec["peak_yaw_rad_s"] = peak_yaw
            rec["depth_change_mm"] = self._depth_change(t0, depth_buf)
        self.legs.append(rec)
        say("  " + describe(rec))
        return rec

    def leg_with_retries(self, spec, prompt):
        rec = None
        for attempt in range(MAX_ATTEMPTS):
            extra = "" if attempt == 0 else f"  (try {attempt + 1} of {MAX_ATTEMPTS})"
            self.wait_for_press(prompt + extra)
            rec = self.run_leg(spec)
            if rec["outcome"] != "aborted":
                return rec
            say(f"  stopped: {rec['why']}. Neutral. Same leg again on the next press.")
        return rec

    # -- the protocol ----------------------------------------------------------

    def forward_leg(self):
        return LegSpec(f"forward {FORWARD_LEVEL:+.2f}", FORWARD, FORWARD_LEVEL,
                       FORWARD_MAX_S, CENTRE, FORWARD_STOP_MM)

    def forward_fits(self):
        return [r for r in self.legs if r["direction"] == FORWARD and not r["lock"]
                and r["throttle"] == FORWARD_LEVEL and r["speed_mm_s"] is not None]

    def need_distance(self, prompt, carry_to=None):
        """A press, then another, until the camera has the box in its centre."""
        self.wait_for_press(prompt, rest_s=0.0, carry_to=carry_to)
        while self.distance() is None:
            self.wait_for_press("I cannot see the box in the middle of the picture. "
                                "Aim the car straight at it and press RB.", rest_s=0.0,
                                carry_to=carry_to)

    def sweep(self):
        say("\n== Straight sweep ==")
        self.need_distance("Press RB when the car is set up (it will not move yet).",
                           carry_to=1600.0)
        for level in REVERSE_LEVELS:
            d = self.distance()
            if d is not None and d > REVERSE_START_MAX_MM:
                self.leg_with_retries(self.forward_leg(),
                                      f"Next: forward {FORWARD_LEVEL} toward the box. "
                                      f"Press and hold RB.")
            d = self.distance()
            if d is None or d > REVERSE_START_MAX_MM:
                self.need_distance("Too far out to back up. Carry the car to about "
                                   "0.8 m from the box, straight, then press RB.",
                                   carry_to=800.0)
            spec = LegSpec(f"reverse {level:+.2f}", REVERSE, level, REVERSE_MAX_S)
            self.leg_with_retries(spec, f"Next: reverse {level}. Press and hold RB.")
        # One forward leg last, so the forward speed is bracketed in time
        # around the reverse legs and battery drift shows up as a spread.
        # Then more, carried, until there are MIN_FORWARD_FITS of them.
        for attempt in range(MIN_FORWARD_FITS + 1):
            if attempt > 0 and len(self.forward_fits()) >= MIN_FORWARD_FITS:
                break
            d = self.distance()
            if d is None or d < FORWARD_MIN_START_MM:
                self.need_distance("Forward again, for battery drift. Carry the car "
                                   "to about 1.6 m from the box, straight, then press RB.",
                                   carry_to=1600.0)
            self.leg_with_retries(self.forward_leg(),
                                  f"Next: forward {FORWARD_LEVEL}. Press and hold RB.")

    def calibrate_gyro(self):
        """Gyro at rest, wheels at lock: the bias and the moving threshold."""
        self._steer(LOCK_STEERING)
        t_end = self.clock.now() + STEER_SETTLE_S + LOCK_REST_S
        t_from = self.clock.now() + STEER_SETTLE_S
        samples = []
        self._buf = ([], samples)
        try:
            while self.clock.now() < t_end:
                pad = self.pad.poll()
                if pad.abort:
                    raise Stop("B pressed")
                self._read(self.clock.now())
                self.clock.sleep(LOOP_S)
        finally:
            self._buf = None
        rest = [(t, gx) for t, gx in samples if t >= t_from]
        if len(rest) < 20:
            raise RuntimeError(f"only {len(rest)} gyro samples in {LOCK_REST_S} s at rest")
        g = np.array([gx for _, gx in rest])
        t = np.array([t for t, _ in rest])
        self.gyro_bias = float(np.mean(g))
        worst = 0.0
        for i in range(len(t)):
            w = g[(t > t[i] - LOCK_WINDOW_S) & (t <= t[i])] - self.gyro_bias
            if len(w) >= 3 and t[i] - t[0] >= LOCK_WINDOW_S:
                worst = max(worst, abs(float(np.mean(w))))
        self.gyro_threshold = max(LOCK_GYRO_FLOOR, LOCK_GYRO_REST_FACTOR * worst)
        self.gyro_rest = {"bias_rad_s": self.gyro_bias, "worst_window_rad_s": worst,
                          "std_rad_s": float(np.std(g)), "samples": len(g),
                          "threshold_rad_s": self.gyro_threshold}
        say(f"  gyro at rest: bias {self.gyro_bias:+.4f}, worst {LOCK_WINDOW_S}s window "
            f"{worst:.4f}, threshold {self.gyro_threshold:.4f} rad/s")

    def lock_steps(self, direction, levels):
        for level in levels:
            spec = LegSpec(f"lock {direction} {level:+.2f}", direction, level,
                           LOCK_STEP_S, LOCK_STEERING, None, lock=True)
            rec = self.leg_with_retries(spec, f"Next: {direction} {level} at full lock. "
                                              f"Press and hold RB.")
            if rec.get("moved"):
                return level
        return None

    def lock_test(self):
        say("\n== Lock test ==")
        self.need_distance("Set the car about 1.2 m from the box, straight at it. "
                           "The wheels go to full lock and it measures the gyro at "
                           "rest for 2 s, so leave it alone. Press RB.", carry_to=1200.0)
        self.calibrate_gyro()
        rev = self.lock_steps(REVERSE, LOCK_REVERSE_LEVELS)
        self.need_distance("Reverse lock done. Aim the car back at the box, about "
                           "1.2 m, then press RB.", carry_to=1200.0)
        self.calibrate_gyro()
        fwd = self.lock_steps(FORWARD, LOCK_FORWARD_LEVELS)
        self._steer(CENTRE)
        return rev, fwd


def describe(rec):
    speed = ("%5.0f mm/s" % rec["speed_mm_s"]) if rec["speed_mm_s"] is not None else "   no fit "
    resid = ("±%.1f mm" % rec["residual_mm"]) if rec["residual_mm"] is not None else ""
    extra = ""
    if rec.get("lock"):
        extra = " moved" if rec.get("moved") else " no move"
        extra += " (yaw %.3f, depth %+.0f mm)" % (rec.get("peak_yaw_rad_s", 0.0),
                                                  rec.get("depth_change_mm", 0.0))
    return (f"{rec['name']:<20} {speed} {resid:<10} {rec['fit_frames']:>3}/{rec['frames']:<3} "
            f"frames  {rec['outcome']}: {rec['ended']}{extra}")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def summarize(cal, lock_rev=None, lock_fwd=None, lock_tested=False, complete=False):
    fwd = cal.forward_fits()
    speeds = [r["speed_mm_s"] for r in fwd]
    fwd_mean = float(np.mean(speeds)) if speeds else None
    fwd_spread = (float(max(speeds) - min(speeds)) if len(speeds) > 1 else None)
    fwd_std = float(np.std(speeds)) if len(speeds) > 1 else None
    reverse = [(r["throttle"], r["speed_mm_s"]) for r in cal.legs
               if r["direction"] == REVERSE and not r["lock"] and r["outcome"] == "done"]
    match = match_reverse(reverse, fwd_mean)
    rec = recommend(match.throttle, lock_rev, lock_tested=lock_tested)
    return {
        "complete": complete,
        "forward_level": FORWARD_LEVEL,
        "forward_speeds_mm_s": [[r["at_s"], r["speed_mm_s"]] for r in fwd],
        "forward_mean_mm_s": fwd_mean,
        "forward_spread_mm_s": fwd_spread,
        "forward_std_mm_s": fwd_std,
        "reverse_table": [[x, s] for x, s in reverse],
        "matched_reverse": match.throttle,
        "match_note": match.note,
        "lock_tested": lock_tested,
        "lock_reverse_threshold": lock_rev,
        "lock_forward_threshold": lock_fwd,
        "gyro_rest": cal.gyro_rest,
        "recommended_reverse_speed": rec.reverse_speed,
        "recommended_pulse_us": rec.pulse_us,
        "conflict": rec.conflict,
        "recommendation_note": rec.note,
        "legs": cal.legs,
    }


def report(summary):
    lines = ["", "== Results ==", "throttle   pulse   mm/s     fit      frames  ended"]
    for r in summary["legs"]:
        if r["lock"] or r["outcome"] == "aborted":
            continue
        s = "%6.0f" % r["speed_mm_s"] if r["speed_mm_s"] is not None else "    --"
        f = "±%.1f" % r["residual_mm"] if r["residual_mm"] is not None else "--"
        lines.append(f"{r['throttle']:+6.2f}   {r['pulse_us']:5d}  {s}   {f:<8} "
                     f"{r['fit_frames']:>3}/{r['frames']:<3}  {r['ended']}")
    m = summary["forward_mean_mm_s"]
    if m is not None:
        spread = summary["forward_spread_mm_s"]
        lines.append(f"forward {FORWARD_LEVEL}: {m:.0f} mm/s mean over "
                     f"{len(summary['forward_speeds_mm_s'])} legs"
                     + (f", spread {spread:.0f} mm/s (first to last: "
                        + " -> ".join("%.0f" % s for _, s in summary["forward_speeds_mm_s"])
                        + ")" if spread is not None else ""))
    else:
        lines.append(f"forward {FORWARD_LEVEL}: no leg fit")
    mt = summary["matched_reverse"]
    lines.append("matched reverse: " + (f"{mt:.3f}" if mt is not None else "none")
                 + f"  ({summary['match_note']})")
    if summary["lock_tested"]:
        lr, lf = summary["lock_reverse_threshold"], summary["lock_forward_threshold"]
        lines.append("moves at full lock: reverse "
                     + (f"{lr:+.2f}" if lr is not None else "NEVER (to -1.0)")
                     + ", forward " + (f"{lf:+.2f}" if lf is not None else "NEVER (to 0.4)"))
        if lf is None:
            lines.append("forward 0.4 itself stalls at full lock")
    rs = summary["recommended_reverse_speed"]
    lines.append("REVERSE_SPEED = " + (f"{rs:.2f}  ({summary['recommended_pulse_us']} us)"
                                       if rs is not None else "none")
                 + f"  -- {summary['recommendation_note']}")
    if not summary["complete"]:
        lines.append("(run did not finish; this is what it had)")
    return "\n".join(lines)


def write_json(summary, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(log_dir, f"speed_cal-{stamp}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=1, default=float)
    return path


# ---------------------------------------------------------------------------
# The real car
# ---------------------------------------------------------------------------

class RealClock:
    def now(self):
        return time.monotonic()

    def sleep(self, dt):
        time.sleep(dt)


class CarVehicle:
    """VehicleController, with the tool's own finite check on top of its own."""

    def __init__(self):
        import Logger as lg
        import VehicleController as vc
        self.vc = vc.VehicleController(logger=lg.Logger())
        self.vc.return_neutral()

    def throttle(self, value):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            value = 0.0
        self.vc.set_throttle(value)

    def steer(self, front, rear):
        self.vc.set_steering(front, rear)

    def close(self):
        self.vc.close()


class CarPad:
    """The Joystick, read once a loop: RB, B, and whether to trust it at all."""

    def __init__(self, stop_event):
        import pygame
        import Joystick as js
        self.pygame = pygame
        self.js = js.Joystick(stop_event=stop_event)

    def poll(self) -> PadState:
        try:
            self.pygame.event.pump()
            self.js.update_connection_status()
            if not self.js.heard():
                # Take the handle again if it is back, so a re-pair does not
                # mean restarting the run. Only when a device is there, so
                # wait_for_connection returns at once instead of waiting.
                if self.pygame.joystick.get_count() > 0:
                    self.js.connected = False
                    self.js.wait_for_connection(poll_interval=0.0)
                return PadState(False, False, False, "pad lost")
            for axis in ("LEFT_X", "LEFT_Y", "LT", "RT"):
                if not math.isfinite(float(self.js.get_axis(axis))):
                    return PadState(False, False, False, f"{axis} is not a number")
            rb = bool(self.js.get_button("RB"))
            b = bool(self.js.get_button("B"))
            return PadState(True, rb, b)
        except Exception as e:
            return PadState(False, False, False, f"pad read failed: {e}")

    def prompt(self):
        handle = getattr(self.js, "joystick", None)
        if handle is not None:
            try:
                handle.rumble(0.6, 0.6, 160)
            except Exception:
                pass


class CarCamera:
    """The OAK-D, every queue drained every loop.

    OakD opens all its queues shallow and non-blocking, which is what keeps
    XLink from jamming at ~43 frames, and this drains every one of them with
    tryGetAll on top of that, so nothing builds up either. Depth is timed by
    the packet's own capture timestamp mapped onto time.monotonic, not by when
    it reached me, so the fit is on when the picture was taken.
    """

    def __init__(self):
        import depthai as dai
        from oakd import OakD
        self.dai = dai
        self.oak = OakD(rectified=False)

    def _host_time(self, ts, now):
        try:
            return now - (self.dai.Clock.now() - ts).total_seconds()
        except Exception:
            return now

    def poll(self, now):
        depth, gyro = [], []
        oak = self.oak
        for packet in oak.depth_queue.tryGetAll() or []:
            try:
                frame = packet.getFrame()
            except RuntimeError:
                continue
            depth.append((self._host_time(packet.getTimestamp(), now), now, frame))
        if oak.disparity_queue is not None:
            oak.disparity_queue.tryGetAll()
        for data in oak.imu_queue.tryGetAll() or []:
            for p in data.packets:
                gyro.append((now, float(p.gyroscope.x)))
        return depth, gyro

    def close(self):
        self.oak.close()


# ---------------------------------------------------------------------------
# The simulated car, for --dry-run and the tests
# ---------------------------------------------------------------------------

class EscModel(NamedTuple):
    """Steady ground speed against throttle: dead below a threshold, linear past it.

    Defaults are a guess shaped by what is measured: forward 0.2 lands at
    187 mm/s (164-217 on the floor), reverse at -0.2957 barely moves, reverse
    -0.6 is faster than forward 0.2, and the reverse threshold is the higher.
    Lock adds drag, which shows up as a higher threshold.
    """

    fwd_dead: float = 0.05
    fwd_gain: float = 1250.0     # mm/s per unit past the threshold
    rev_dead: float = 0.25
    rev_gain: float = 1100.0
    lock_fwd_dead: float = 0.15
    lock_rev_dead: float = 0.45
    tau_s: float = 0.15

    def speed(self, throttle, lock=False):
        """Signed steady speed, positive toward the box."""
        if throttle > 0:
            dead = self.lock_fwd_dead if lock else self.fwd_dead
            return max(0.0, throttle - dead) * self.fwd_gain
        if throttle < 0:
            dead = self.lock_rev_dead if lock else self.rev_dead
            return -max(0.0, -throttle - dead) * self.rev_gain
        return 0.0


class SimClock:
    def __init__(self, world=None):
        self.t = 0.0
        self.world = world

    def now(self):
        return self.t

    def sleep(self, dt):
        if self.world is not None:
            self.world.advance(self.t, dt)
        self.t += dt


class SimWorld:
    """One car, one box, a floor. Steps in the clock's sleep."""

    TURN_RADIUS_MM = 590.0
    FPS = 16.0
    LATENCY_S = 0.088
    IMU_HZ = 100.0
    SHAPE = (200, 320)

    # The pair's baseline, for quantising depth to whole disparity levels the
    # way extended disparity without subpixel does on the car.
    BASELINE_MM = 75.0

    def __init__(self, start_mm=1600.0, esc=EscModel(), seed=0):
        self.esc = esc
        self.gyro_noise = 0.003     # rad/s per sample at rest
        self.gyro_buzz = 0.01       # extra per sample with throttle up
        self.quantize = True
        self.rng = np.random.default_rng(seed)
        self.d = float(start_mm)
        self.v = 0.0
        self.heading = 0.0
        self.yaw = 0.0
        self.throttle = 0.0
        self.front = 0.0
        self.pending = []           # (t_capture, t_arrival, frame) not yet delivered
        self.gyro = []              # (t, gx) not yet delivered
        self.next_frame = 0.0
        self.next_imu = 0.0
        self.camera_stalled_from = None
        self.imu_stalled_from = None
        self.distance_offset = 0.0  # tests: pretend the box is somewhere else
        h, w = self.SHAPE
        fy = dv.FY_AT_400 * h / 400.0
        rows = np.arange(h) - h // 2
        with np.errstate(divide="ignore"):
            self.floor = np.where(rows > 0, dv.CAMERA_HEIGHT_MM * fy / np.maximum(rows, 1), np.inf)

    def frame(self, d):
        h, w = self.SHAPE
        d = d + self.rng.normal(0.0, 3.0)
        if self.quantize and d > 0:
            # 640-wide intrinsics, which is what the car's frames are.
            fb = dv.FX_AT_640 * self.BASELINE_MM
            d = fb / max(1.0, round(fb / d))
        col = np.minimum(self.floor, d)
        col = np.clip(col, 0, 65000)
        return np.repeat(col.astype(np.uint16)[:, None], w, axis=1)

    def advance(self, t, dt):
        lock = abs(self.front) >= 0.5
        target = self.esc.speed(self.throttle, lock)
        self.v += (target - self.v) * min(1.0, dt / self.esc.tau_s)
        sign = 1.0 if self.front >= 0 else -1.0
        self.yaw = (self.v / self.TURN_RADIUS_MM) * sign if lock else 0.0
        self.heading += self.yaw * dt
        self.d -= self.v * math.cos(self.heading) * dt
        t1 = t + dt
        while self.next_frame <= t1:
            tc = self.next_frame
            if self.camera_stalled_from is None or tc < self.camera_stalled_from:
                self.pending.append((tc, tc + self.LATENCY_S,
                                     self.frame(self.d + self.distance_offset)))
            self.next_frame += 1.0 / self.FPS
        while self.next_imu <= t1:
            ti = self.next_imu
            if self.imu_stalled_from is None or ti < self.imu_stalled_from:
                buzz = self.gyro_buzz if self.throttle != 0 else 0.0
                gx = self.yaw + 0.002 + self.rng.normal(0.0, self.gyro_noise + buzz)
                self.gyro.append((ti, float(gx)))
            self.next_imu += 1.0 / self.IMU_HZ


class SimVehicle:
    def __init__(self, world, clock):
        self.world = world
        self.clock = clock
        self.log = []           # (t, "throttle"/"steer", value)
        self.closed = False
        self.raise_on_throttle_at = None

    def throttle(self, value):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            value = 0.0
        if (self.raise_on_throttle_at is not None and value != 0.0
                and self.clock.now() >= self.raise_on_throttle_at):
            raise RuntimeError("simulated ESC fault")
        self.world.throttle = max(-1.0, min(1.0, float(value)))
        self.log.append((self.clock.now(), "throttle", self.world.throttle))

    def steer(self, front, rear):
        self.world.front = front if math.isfinite(front) else 0.0
        self.log.append((self.clock.now(), "steer", (front, rear)))

    def close(self):
        self.world.throttle = 0.0
        self.world.front = 0.0
        self.closed = True
        self.log.append((self.clock.now(), "close", None))


class SimCamera:
    def __init__(self, world):
        self.world = world
        self.closed = False
        self.raise_at = None

    def poll(self, now):
        if self.raise_at is not None and now >= self.raise_at:
            raise RuntimeError("simulated camera fault")
        w = self.world
        depth = [p for p in w.pending if p[1] <= now]
        w.pending = [p for p in w.pending if p[1] > now]
        gyro, w.gyro = w.gyro, []
        return depth, gyro

    def close(self):
        self.closed = True


class SimPad:
    """A patient owner. Lets go on every prompt and presses again 0.3 s later.

    faults: list of (t_from, t_to, kind) in sim time, kind one of "release",
    "nan", "lost", "abort". hooks: list of (t, fn) called once when the clock
    passes t, for things like sending a signal mid-leg.
    """

    def __init__(self, clock, press_delay=0.3):
        self.clock = clock
        self.press_delay = press_delay
        self.rb = False
        self.press_at = None
        self.faults = []
        self.hooks = []
        self.prompts = 0

    def prompt(self):
        self.prompts += 1
        self.rb = False
        self.press_at = self.clock.now() + self.press_delay

    def hold(self):
        self.rb = True
        self.press_at = None

    def poll(self) -> PadState:
        now = self.clock.now()
        for hook in list(self.hooks):
            if now >= hook[0]:
                self.hooks.remove(hook)
                hook[1]()
        if self.press_at is not None and now >= self.press_at:
            self.rb, self.press_at = True, None
        for t_from, t_to, kind in self.faults:
            if t_from <= now < t_to:
                if kind == "abort":
                    return PadState(True, self.rb, True)
                if kind == "release":
                    return PadState(True, False, False)
                if kind == "nan":
                    return PadState(False, False, False, "LEFT_X is not a number")
                if kind == "lost":
                    return PadState(False, False, False, "pad lost")
        return PadState(True, self.rb, False)


class SimHardware(NamedTuple):
    clock: SimClock
    world: SimWorld
    vehicle: SimVehicle
    pad: SimPad
    camera: SimCamera

    def carry(self, mm):
        """The owner picking the car up and setting it down, straight."""
        self.world.d = float(mm)
        self.world.v = 0.0
        self.world.heading = 0.0


def build_sim(start_mm=1600.0, esc=EscModel(), seed=0) -> SimHardware:  # noqa: B008
    world = SimWorld(start_mm, esc, seed)
    clock = SimClock(world)
    return SimHardware(clock, world, SimVehicle(world, clock), SimPad(clock),
                       SimCamera(world))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SETUP = """\
Setup: the car on the floor it drives on, facing a flat box or a wall about
1.6 m away (the box at least 30 cm wide and 30 cm tall), straight on, with
1.5 m clear behind it. Each leg drives only while RB is held: let go between
legs, press and hold when the pad buzzes, let go any time to stop. B ends the
run. Ctrl-C ends it too.
"""


def emergency_stop(vehicle):
    """Neutral and centre, each on its own so one failure cannot skip the other."""
    if vehicle is None:
        return
    try:
        vehicle.throttle(0.0)
    except Exception as e:
        say(f"[speed_cal] could not neutral the ESC: {e}")
    try:
        vehicle.steer(0.0, 0.0)
    except Exception as e:
        say(f"[speed_cal] could not centre the steering: {e}")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry-run", action="store_true",
                   help="simulated car, pad and camera; touches no hardware")
    p.add_argument("--no-lock", action="store_true", help="skip the lock test")
    p.add_argument("--log-dir", default=os.path.join(ROOT, "log"))
    p.add_argument("--seed", type=int, default=0, help="dry run only")
    return p.parse_args(argv)


def main(argv=None, hardware=None):
    """Run the whole protocol. hardware is a SimHardware for tests; None builds one."""
    args = parse_args(argv)
    held = {"vehicle": None, "quiet": False}

    def on_signal(signum, _frame):
        # Neutral first, from inside the handler, whatever the main thread
        # was in the middle of. Then unwind -- except once cleanup has
        # started, where a second Ctrl-C must not cut the close short.
        emergency_stop(held["vehicle"])
        if not held["quiet"]:
            raise Stop(signal.Signals(signum).name)

    old = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        old[sig] = signal.signal(sig, on_signal)

    vehicle = pad = camera = cal = None
    lock_rev = lock_fwd = None
    lock_tested = complete = False
    exit_code = 1
    try:
        if hardware is None and args.dry_run:
            hardware = build_sim(seed=args.seed)
        if hardware is not None:
            clock, vehicle, pad, camera = (hardware.clock, hardware.vehicle,
                                           hardware.pad, hardware.camera)
            held["vehicle"] = vehicle
        else:
            import threading
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            import pygame
            pygame.init()
            clock = RealClock()
            vehicle = CarVehicle()          # first, so the ESC is at neutral
            held["vehicle"] = vehicle
            say("ESC at neutral. Waiting for the pad...")
            pad = CarPad(threading.Event())
            say("Pad up. Starting the OAK-D (a few seconds)...")
            camera = CarCamera()
        carry = getattr(hardware, "carry", None) if hardware is not None else None
        cal = Calibrator(vehicle, pad, camera, clock, carry=carry)
        say(SETUP)
        cal.sweep()
        if not args.no_lock:
            lock_tested = True
            lock_rev, lock_fwd = cal.lock_test()
        complete = True
        exit_code = 0
    except Stop as e:
        emergency_stop(vehicle)
        say(f"\n[speed_cal] stopped: {e}")
        exit_code = 130
    except BaseException as e:
        emergency_stop(vehicle)
        say(f"\n[speed_cal] failed: {e!r}")
        if not isinstance(e, Exception):
            exit_code = 130
    finally:
        held["quiet"] = True
        emergency_stop(vehicle)
        for thing in (camera, vehicle):
            if thing is not None:
                try:
                    thing.close()
                except Exception as e:
                    say(f"[speed_cal] close failed: {e}")
        for sig, handler in old.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass
    if cal is not None and cal.legs:
        summary = summarize(cal, lock_rev, lock_fwd, lock_tested, complete)
        say(report(summary))
        try:
            say("wrote " + write_json(summary, args.log_dir))
        except Exception as e:
            say(f"[speed_cal] could not write the JSON: {e}")
        main.last_summary = summary
    return exit_code


main.last_summary = None


if __name__ == "__main__":
    sys.exit(main())
