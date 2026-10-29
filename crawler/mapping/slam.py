"""Live 2D lidar SLAM, one scan at a time: track against the map, then write.

Slam.step() is the whole thing, and it never looks at anything later than
the scan it is given. The order on every revolution:

  1. Predict. Constant velocity from the last two fits, with the heading
     taken from the IMU (gyro . up, integrated over the time between scans)
     when there is one.
  2. De-skew. The truck turns and moves during the 100 ms the sensor takes
     to go round, so each point is moved to where it would have been at the
     scan's end: IMU yaw rate over the revolution, last fitted velocity.
  3. Match. Point-to-line ICP against the map's occupied hit means within
     LOCAL_R_M, with the IMU heading as a soft prior. A fit under
     RESEED_BELOW (70%) inliers is tried again from the last pose at heading
     offsets out to +-35 degrees and from the prediction moved 25 cm each way,
     and the best one kept.
     A fit whose heading is more than BAND_YAW_DEG off the IMU's, or whose
     position is more than BAND_M off the prediction, is not believed: the
     prediction is kept and the scan counts as a failed fit. Along a
     direction the scene does not pin down (a corridor's length) the fit
     keeps the prediction too (matcher.remap).
  4. Write, only if all of these hold:
       the fit is good (WRITE_MIN_INLIERS, WRITE_MAX_RMS_M) and the scene
       pins the pose down (WRITE_MIN_CONSTRAINT);
       the IMU says the truck is near level, pitch under
       WRITE_MAX_PITCH_DEG and roll under WRITE_MAX_ROLL_DEG, because a
       tilted scan plane sees the floor and the tops of things;
       nothing is lost or rolled over.
     Every usable return is written, not only the ones on a line (those are
     what is matched), so legs and clutter are in the map; only the line
     points move the hit means the matcher matches against. Points the tilted
     plane might be putting on the ground are left out even then
     (scan.ground_mask).

The states (Result.state):

  tracking   matched, and written if the gates above allowed
  tilted     past the pitch or roll gate: matched, not written
  rolled     past ROLL_DEG, upside down or on its side or in a hand: the pose
             is frozen and nothing is matched or written
  lost       MAX_FAILS fits in a row failed, or it just came back from
             rolled. Nothing is written; every scan tries RECOVER_SEEDS
             headings round the whole circle from the frozen pose until one
             fits well (RECOVER_MIN_INLIERS), and then it is tracking again.

With no IMU at all (the camera shut) it still runs: no heading prior, the
de-skew rate from the last fits, and the level gate open, because there is
nothing to say the truck is not level. Result.tilt is NaN then.

The map is crawler.mapping.grid.Grid, reachable as Slam.grid; the pose is
Slam.pose. That and Result are the interface a planner builds on: see
crawler/mapping/__init__.py.

The seam for loop closure, later: every written scan goes through
_write(pose, points) with the pose it was written at, so a pose graph can
log (pose, points) there and re-render the grid from corrected poses
without anything else changing.
"""

import math
import time
from typing import NamedTuple, Optional

import numpy as np
from scipy.spatial import cKDTree

from crawler.mapping import grid as gr
from crawler.mapping import matcher as mt
from crawler.mapping import scan as sc
from crawler.mapping.imu import ImuTrack

# Map points within this square of the truck are matched against.
LOCAL_R_M = 9.0

# The heading prior: 1.5 degrees a scan, widened with the time since the
# last one.
SIGMA_YAW_DEG = 1.5

# A fit is a fit.
GOOD_INLIERS = 0.4
# Below this the fit is not trusted for the pose either: the prediction is
# used instead.
TRUST_INLIERS = 0.25
RESEED_DEG = (0, 10, -10, 20, -20, 35, -35)
RESEED_SHIFT_M = ((0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25))
# A fit under this many inliers is tried again from the seeds above, even if
# it is good enough to keep. 0.4 alone let a 125 deg/s spin in 035323 (530 s)
# settle half a wall off, at 0.52, and the whole of the rest of the drive
# carried the 27 cm.
RESEED_BELOW = 0.7

# The write gates. Pitch and roll separately (imu.ImuTrack.pitch_roll):
# pitch tips the plane into the floor ahead and behind, roll into it to the
# sides, and the truck sits rolled a few degrees on its springs far more
# often than it sits pitched.
WRITE_MAX_PITCH_DEG = 3.0
WRITE_MAX_ROLL_DEG = 5.5
WRITE_MIN_INLIERS = 0.6
WRITE_MAX_RMS_M = 0.02
# matcher.Fit.constraint below this and the scene does not say where along
# it the truck is: matched (on the prediction there), not written.
WRITE_MIN_CONSTRAINT = 0.02

# The band a fit must land in round the prediction to be believed. The
# heading's against the IMU integral, when there is an IMU; the position's
# widens with the time since the last scan, at 1 m/s (the truck's top speed
# indoors is under that).
BAND_YAW_DEG = 5.0
BAND_M = 0.3
BAND_M_PER_S = 1.0
# The band only holds a fit to a prediction made from a good fit the scan
# before. After a refused or failed fit, or a gap, the pose it would hold to
# is the thing in doubt: in 035203 a bad fit across a 1 s gap put the truck
# 46 cm off, and a band applied after it refused every fit that would have
# brought it back, until it was lost (610 scans).
BAND_MAX_GAP_S = 0.25

# The matcher's tree is rebuilt when the truck has moved this far since the
# last build, or the map has had this many writes: every scan used to be a
# rebuild.
TREE_MOVE_M = 0.2
TREE_WRITES = 5

# Past this the truck is on its side, upside down, or being carried.
ROLL_DEG = 60.0
# This many failed fits in a row and the pose is not believed any more.
MAX_FAILS = 10
# Lost: headings tried per scan, and how good a fit must be to end it.
RECOVER_STEP_DEG = 10
RECOVER_SEEDS = 8
RECOVER_MIN_INLIERS = 0.75

# The revolution period when there is nothing better to go on: 0.1009 s
# measured over the four drives.
PERIOD_S = 0.101

TRACKING = "tracking"
TILTED = "tilted"
ROLLED = "rolled"
LOST = "lost"
NO_POINTS = "no_points"
STATES = (TRACKING, TILTED, ROLLED, LOST, NO_POINTS)


class Result(NamedTuple):
    """One scan's outcome. Everything a planner needs to trust it or not."""
    t: float                  # the scan's end time, monotonic seconds
    x: float                  # map metres
    y: float
    yaw: float                # radians CCW from the boot heading
    state: str                # STATES
    matched: bool             # the fit is good (GOOD_INLIERS, converged)
    written: bool             # this scan went into the map
    why: str                  # why not written, "" if it was
    inliers: float            # 0..1
    rms_m: float              # point-to-line residual, NaN if not matched
    constraint: float         # 0 (corridor, open field) .. 0.5 (a room)
    returns: int              # usable ranges in the scan
    points: int               # of those, points on a line (matched)
    ground_dropped: int       # points kept out of the write as maybe-ground
    tilt_deg: float           # NaN with no IMU
    iters: int
    ms: float                 # time step() took
    pitch_deg: float = float("nan")   # |pitch|, |roll|, NaN with no IMU
    roll_deg: float = float("nan")
    held: int = 0             # directions the fit kept at the prediction
    banded: bool = False      # the fit landed outside the band and was refused


class Slam:
    """The map, the pose, and the step that moves them on."""

    def __init__(self, imu: Optional[ImuTrack] = None, grid=None,
                 l_max=gr.L_MAX, write_max_pitch=WRITE_MAX_PITCH_DEG,
                 write_max_roll=WRITE_MAX_ROLL_DEG, ground_guard=True,
                 band=True, clock=time.perf_counter):
        self.imu = imu if imu is not None else ImuTrack()
        self.grid = grid if grid is not None else gr.Grid(l_max=l_max)
        self.write_max_pitch = write_max_pitch
        self.write_max_roll = write_max_roll
        self.ground_guard = ground_guard
        self.band = band
        self.banded = 0
        self._clock = clock
        self.pose = np.zeros(3)
        self.vel = np.zeros(3)      # map frame dx, dy, dyaw per second
        self.bias = 0.0             # gyro yaw-rate bias, rad/s
        self.last_t = None
        self.started = False
        self.fails = 0
        self.lost = False
        self._seed = 0
        self._tree = None
        self._tgt = None
        self._tree_at = None
        self._tree_version = -1
        self._tree_writes = 0
        self._tree_origin = None
        self._att = (float("nan"), float("nan"))
        self._banded = False
        self.scans = 0
        self.writes = 0
        self.retries = 0
        self.recoveries = 0
        self.last_points = None

    # ---- the step -------------------------------------------------------

    def step(self, t, ranges, period=None) -> Result:
        t0 = self._clock()
        self.scans += 1
        self.last_points = None
        imu = self.imu
        last_t = self.last_t
        if period is None:
            period = (PERIOD_S if last_t is None
                      else float(np.clip(t - last_t, 0.09, 0.115)))
        dt = PERIOD_S if last_t is None else t - last_t
        returns = sc.valid_returns(ranges)
        tilt = imu.tilt(t)
        self._att = imu.pitch_roll(t)
        self._banded = False

        if tilt > ROLL_DEG:
            # Frozen. Whatever the scan sees now is the floor, the ceiling or
            # a hand; none of it is a wall to match.
            self.lost = True
            self.vel[:] = 0.0
            self.last_t = t
            return self._result(t, ROLLED, False, False, "rolled", None,
                                returns, 0, 0, tilt, t0)

        p, nrm, idx = sc.scan_points(ranges)
        if len(p) < sc.MIN_POINTS:
            if self.last_t is None:
                self.last_t = t
            return self._result(t, NO_POINTS, False, False, "few points",
                                None, returns, len(p), 0, tilt, t0)
        allp, alli = sc.all_points(ranges)
        on_line = np.isin(alli, idx)

        pose = self.pose
        vcar = mt.rot(-pose[2]) @ self.vel[:2]
        imu_rate = (imu.rate_between(last_t, t, self.bias)
                    if last_t is not None else None)
        om = imu.rate_between(t - period, t, self.bias)
        if om is None:
            om = self.vel[2]
        p, _ = sc.deskew(p, idx, period, om, vcar)
        nrm, _ = sc.deskew(nrm, idx, period, om, np.zeros(2))
        allp, _ = sc.deskew(allp, alli, period, om, vcar)
        # The de-skewed car-frame points, for anyone who wants this scan
        # drawn at the fitted pose (the replay's grading does).
        self.last_points = p

        pred = pose + self.vel * min(dt, 0.5)
        if imu_rate is not None:
            pred[2] = pose[2] + imu_rate * dt

        fit = None
        if not self.started:
            newpose, good = pose.copy(), True
            self.started = True
        elif self._tree is None:
            newpose, good = pred, False
        elif self.lost:
            fit = self._recover(p, nrm, pred)
            good = fit is not None
            newpose = np.array(fit.pose) if good else pose.copy()
            if good:
                self.lost = False
                self.fails = 0
                self.recoveries += 1
                self.vel[:] = 0.0
        else:
            fit = self._match(p, nrm, pred, imu_rate is not None, dt)
            good = fit.converged and fit.inliers >= GOOD_INLIERS
            newpose = np.array(fit.pose)
            if not good and fit.inliers < TRUST_INLIERS:
                newpose = pred
            elif (self.band and self.fails == 0 and dt <= BAND_MAX_GAP_S
                  and self._out_of_band(newpose, pred, dt,
                                        imu_rate is not None)):
                # A fit this far from where the truck can have got to is the
                # scan lining up with the wrong wall: in 035323 at 89 s, the
                # truck going over, one did. Keep the prediction.
                good = False
                newpose = pred
                self._banded = True
                self.banded += 1

        if self._tree is not None and not good and not self.lost:
            self.fails += 1
            self.vel *= 0.5
            if self.fails >= MAX_FAILS:
                self.lost = True
                self.vel[:] = 0.0
                newpose = pose.copy()
        elif good:
            self.fails = 0

        if last_t is not None and good and fit is not None:
            d = max(dt, 1e-3)
            v_new = (newpose - pose) / d
            v_new[2] = ((newpose[2] - pose[2] + np.pi) % (2 * np.pi)
                        - np.pi) / d
            self.vel = 0.5 * self.vel + 0.5 * v_new if dt < 0.3 else v_new * 0
            if imu_rate is not None and dt < 0.2:
                # The gyro's slow bias, from how far it disagrees with the
                # fits: an EMA slow enough that one bad fit is noise.
                self.bias += 0.002 * (imu_rate - v_new[2])
        self.pose = newpose

        pitch, roll = self._att
        state = LOST if self.lost else (
            TILTED if (pitch > self.write_max_pitch
                       or roll > self.write_max_roll) else TRACKING)
        why = self._why_not(fit, good, state)
        dropped = 0
        written = False
        if not why:
            keep = np.ones(len(allp), bool)
            if self.ground_guard and imu.up is not None:
                keep = ~sc.ground_mask(allp, imu.up)
                dropped = int((~keep).sum())
            self._write(self.pose, allp[keep], on_line[keep])
            written = True
        self._refresh_target()
        self.last_t = t
        return self._result(t, state, good, written, why, fit, returns,
                            len(p), dropped, tilt, t0)

    # ---- the parts -------------------------------------------------------

    def _match(self, p, nrm, pred, have_prior, dt):
        sig = math.radians(SIGMA_YAW_DEG) * max(dt / 0.1, 1.0)
        yp = pred[2] if have_prior else None
        fit = mt.icp(p, nrm, self._tree, self._tgt, pred, yp, sig)
        if fit.converged and fit.inliers >= RESEED_BELOW:
            return fit
        self.retries += 1
        best = fit
        for seed in self._seeds(pred):
            c = mt.icp(p, nrm, self._tree, self._tgt, seed, iters=25)
            if c.converged and c.inliers > best.inliers + 0.05:
                best = c
            if best.converged and best.inliers >= RESEED_BELOW:
                break
        return best

    def _seeds(self, pred):
        """Where to start again from when a fit is poor: the last pose at
        heading offsets, then the prediction moved a quarter metre each way."""
        for dyaw in RESEED_DEG:
            seed = self.pose.copy()
            seed[2] += math.radians(dyaw)
            yield seed
        for dx, dy in RESEED_SHIFT_M:
            seed = pred.copy()
            seed[0] += dx
            seed[1] += dy
            yield seed

    def _recover(self, p, nrm, pred):
        """A few headings round the circle per scan, from the frozen pose."""
        n = 360 // RECOVER_STEP_DEG
        offsets = [0]
        for k in range(1, n // 2 + 1):
            offsets += [k * RECOVER_STEP_DEG, -k * RECOVER_STEP_DEG]
        offsets = offsets[:n]
        best = None
        for _ in range(RECOVER_SEEDS):
            dyaw = offsets[self._seed % len(offsets)]
            self._seed += 1
            seed = self.pose.copy()
            seed[2] = pred[2] + math.radians(dyaw)
            c = mt.icp(p, nrm, self._tree, self._tgt, seed, iters=25)
            if c.converged and (best is None or c.inliers > best.inliers):
                best = c
        if best is not None and best.inliers >= RECOVER_MIN_INLIERS:
            self._seed = 0
            return best
        return None

    def _why_not(self, fit, good, state):
        if state == LOST:
            return "lost"
        if state == TILTED:
            return "tilted"
        if fit is None:
            # The first scan, or a map too small to match: written on the
            # prediction, which is how the map gets started.
            return "" if (good or self._tree is None) else "no fit"
        if not good:
            return "no fit"
        if fit.inliers < WRITE_MIN_INLIERS:
            return "few inliers"
        if not fit.rms <= WRITE_MAX_RMS_M:
            return "high residual"
        if fit.constraint < WRITE_MIN_CONSTRAINT:
            return "degenerate"
        return ""

    def _out_of_band(self, pose, pred, dt, have_imu):
        """True if the fit is further from the prediction than the truck can
        have got: BAND_YAW_DEG off the IMU heading, or BAND_M (plus
        BAND_M_PER_S a second since the last scan past the first tenth)."""
        if have_imu:
            dyaw = (pose[2] - pred[2] + math.pi) % (2 * math.pi) - math.pi
            if abs(dyaw) > math.radians(BAND_YAW_DEG):
                return True
        lim = BAND_M + BAND_M_PER_S * max(dt - PERIOD_S, 0.0)
        return math.hypot(pose[0] - pred[0], pose[1] - pred[1]) > lim

    def _write(self, pose, points, on_line=None):
        """The one place a scan enters the map (the loop-closure seam).
        `on_line` marks the points that move the hit means (grid.integrate)."""
        g = self.grid
        g.scroll_to(pose[0], pose[1])
        wp = points @ mt.rot(pose[2]).T + pose[:2]
        g.integrate(pose[0], pose[1], wp[:, 0], wp[:, 1], mean=on_line)
        self.writes += 1

    def _refresh_target(self):
        """Rebuild the matcher's tree once the truck has moved TREE_MOVE_M
        from where it was built or the map has had TREE_WRITES writes since,
        or the map was moved under it (a grow)."""
        g = self.grid
        x, y = self.pose[0], self.pose[1]
        if (self._tree is not None and self._tree_at is not None
                and self._tree_origin == (g.ox, g.oy)
                and self.writes - self._tree_writes < TREE_WRITES
                and math.hypot(x - self._tree_at[0],
                               y - self._tree_at[1]) < TREE_MOVE_M):
            return
        tgt = g.local_points(x, y, LOCAL_R_M)
        self._tgt = tgt
        self._tree = cKDTree(tgt) if len(tgt) >= 50 else None
        self._tree_at = (x, y)
        self._tree_version = g.version
        self._tree_writes = self.writes
        self._tree_origin = (g.ox, g.oy)

    def _result(self, t, state, matched, written, why, fit, returns, points,
                dropped, tilt, t0):
        x, y, yaw = (float(v) for v in self.pose)
        return Result(
            t=float(t), x=x, y=y, yaw=yaw, state=state, matched=bool(matched),
            written=bool(written), why=why,
            inliers=float(fit.inliers) if fit is not None else (
                1.0 if written else 0.0),
            rms_m=float(fit.rms) if fit is not None else float("nan"),
            constraint=float(fit.constraint) if fit is not None else 0.0,
            returns=int(returns), points=int(points),
            ground_dropped=int(dropped), tilt_deg=float(tilt),
            iters=int(fit.iters) if fit is not None else 0,
            ms=(self._clock() - t0) * 1000.0,
            pitch_deg=float(self._att[0]), roll_deg=float(self._att[1]),
            held=int(getattr(fit, "held", 0)) if fit is not None else 0,
            banded=bool(self._banded))
