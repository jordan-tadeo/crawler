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
     and the best one kept. The retries stop once the step has taken
     RETRY_STEP_S, and the next poor scan carries on from the seed after the
     last one tried, so a slow Pi still gets through all of them over a few
     revolutions instead of falling behind the lidar.
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
  lost       MAX_FAILS fits in a row failed, or it fitted too poorly to
             write (under WRITE_MIN_INLIERS) for POOR_LOST_S of level
             driving, or it just came back from rolled. The pose is in doubt; the map is not. Nothing is
             written, and every scan relocalizes against the map
             (_relocalize): a correlative search (reloc.py) over x, y and
             heading within reloc.SEARCH_R_M of the frozen pose, the
             headings outwards from the heading the gyro has carried it to
             since (projected on up, so through a rollover too), then ICP on
             the best few. A fit over RECOVER_MIN_INLIERS is picked if no
             different pose fits nearly as well (a room turned round can),
             or if the gyro's heading settles which (reloc.Search.pick), and
             believed once ICP from it fits as well on the next scan too;
             then it is tracking again. Measured on the synthetic room: set
             down in place after 4 s upside down, tracking on the 2nd scan;
             1.5 m off and turned 120 degrees, on the 14th; 1.2-1.4 cm off.

Never losing the map. A tilt, a knock or a rollover only ever makes the pose
uncertain: the map stays as it was. Only lost for RESEED_LOST_S (counted from
when it was last rolled, so the time in a hand does not count) is a fresh map
started from the scan in hand (Result.reseed "lost"), and even then the old
one is kept on the shelf: the truck is relocalized against it, round where
it is reckoned to be on it, and when that is found (twice running, the same
way) the fresh map is merged into the old one and the old one is the map
again (Result.reseed "merged"). That search is done a piece at a time,
SHELF_SHARE_S of it a scan (Slam._shelf_slice), never whole in one step:
the fresh map is the one being driven on meanwhile.

Where the truck is on the shelf. From the reseed that shelved it, the shelf
reckons the truck on itself every scan (Shelf.here, Slam._reckon_shelf): the
heading by the gyro, the position by the fresh map's own step turned onto
the shelf by the gyro's heading, or by the commanded speed when nothing
fits. None of that goes through the fresh map's frame, so a fresh map that
spins on its own, starts again young or relocalizes onto its own twin does
not move it. The search is drawn round it, as wide as its doubt, over only
the headings within SHELF_DRIFT_DEG (and more with time) of its heading, and
a fit outside that is never merged however well it fits: the 180 degree
twin of a room fits it as well as the truth does. Before this the search
was drawn where the truck had frozen, every heading, and 174641
(2026-10-29) merged the twin. Long lost, the position's doubt is many metres
and the heading alone cannot tell a wrong place: a merge is only made if
the fresh map's walls land on the shelf's walls, not its open floor
(SHELF_AGREE).

How a fresh map sits on the one before it. A fresh map goes on from the
pose the truck was last believed at, which after a loss is where it froze,
not where it is, so a point on the old map is not the same point on the new
one. Every reseed says how far off (Result.link): the old map's frame onto
the new one, from where the truck is reckoned to have got since its last
good fit the gyro agrees with. A fit that is good by its inliers but has
turned the truck where the gyro did not (DR_AGREE_DEG) is the scan lining up
with the wrong walls on the way to getting lost: in replays of 052302 with
the scans blinded at 520 s, good fits turned it 10 to 20 degrees the gyro
never saw, the reckoning restarted from them, and the shelf held the merge
to a heading 85 degrees off. The reckoning carries on through such a fit,
unless DR_TRUST_RUN good fits in a row each agree with the gyro scan to
scan (DR_STEP_DEG), which is the map tracking again from a heading the gyro
has drifted off. The heading off the gyro (bias taken off, projected on up, so a
rollover too); the position off the commanded speed step() is handed (the
throttle through nav/truck.ground_speed) along that heading; and an error
estimate for both, LINK_ERR_* below. A merge's link is the fit that merged
(old = T . fresh) with no error. A goal clicked on the map rides these
(autopilot/goal.py ClickGoal); the shelf starts its own reckoning from the
shelving link and goes on by itself.

A map is young until it has had SEED_WRITES writes. A young map that a scan
does not fit is started again from that scan (Result.reseed "young"), and
nothing is shelved: one or two scans are no map to hold the truck to. On
10-29 the truck was carried to the start after boot, the map was its first
scan from where it had stood, and nothing after fitted it: lost from one
second in, for the whole drive.

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
from crawler.mapping import reloc as rl
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
# The retries stop once the step is this long on Slam's clock, after one
# seed at least, and the next poor scan goes on round the seeds from the one
# after the last it tried. All eleven every scan was about 400 ms on the Pi
# (p1201, 2026-10-29, a seed's ICP 30-35 ms there): the truck was being
# carried, three scans in four were dropped behind the retries, and the pose
# was left behind on a spot that fitted half. On the Mac all eleven take
# under 40 ms, so a replay there tries them all as before.
RETRY_STEP_S = 0.06

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
# Nor is it after this many seconds of level scans whose fits are good
# (GOOD_INLIERS) but too poor to write (WRITE_MIN_INLIERS), with no fit good
# enough to write between them; tilted scans pause the count, so a climb is
# never lost to it. Such a fit is the pose on the wrong spot of the map,
# and lost is what searches the map for the right one: p1201 (2026-10-29)
# fitted 0.45 for the last five minutes of the recording, wrote nothing, and
# was never lost; 201721 fitted 0.4-0.6 for 25 s before it failed outright,
# and the search found it at 0.94 eight seconds later. Over the replays of
# five 10-29 drives the longest run of them anywhere else was 3.2 s, the
# next 2.7.
POOR_LOST_S = 3.0
# Lost: how good a relocalized fit must be to end it.
RECOVER_MIN_INLIERS = 0.75
# A confirmed fit is one ICP from the candidate on the next scan agrees with
# to within these.
CONFIRM_M = 0.25
CONFIRM_DEG = 8.0
# Lost this long, in scan time, without a relocalization, and a fresh map is
# started from the scan in hand, the old one shelved. The search covers the
# whole circle every 3 scans, so this is the window searched 25 times over.
RESEED_LOST_S = 8.0
# The search for the truck on a shelved map is done a piece at a time: a
# heading scored (reloc.Search, about 5 ms on the Pi) or an ICP from a
# candidate (30-40 ms). It may spend SHELF_SHARE_S a scan on average, and
# stops once the step that runs it is SHELF_STEP_S long (Slam's clock, from
# the step's start), after one piece at least. It used to be the whole of a
# reloc.Search.step and its four ICPs every fifth scan, 370 ms on the Pi
# (p1432, 2026-10-29), and the crop went stale behind each one
# (egomap.STALE_S): the fresh map tracked every scan, the policy never had
# it the MAP_BACK_S it waits for, and the car drove 16 minutes on its
# fallback. A tracking step is about 30 ms there and a scan comes every 100.
SHELF_SHARE_S = 0.015
SHELF_STEP_S = 0.04
# The search is drawn round the truck again once it is this far from where
# it was drawn.
SHELF_MOVE_M = 1.0
# Under this many writes since the map started a scan that does not fit
# starts it again instead of failing.
SEED_WRITES = 5

# How sure a link (Result.link) is. The dead reckoning's position is off by
# LINK_ERR_FRAC of the distance it counted (wheels spin on rock, and the
# speed curve is flat floor), plus LINK_ERR_M; with no speed handed in it
# cannot count any, and it owns up to LINK_UNKNOWN_M_S of movement it did
# not see. The gyro's heading drifts LINK_ERR_DEG_S a second on top of
# LINK_ERR_DEG. Rolled over or picked up, the truck may be anywhere near:
# LINK_ROLLED_M once for each time.
LINK_ERR_FRAC = 0.3
LINK_ERR_M = 0.1
LINK_UNKNOWN_M_S = 0.3
LINK_ERR_DEG = 1.0
LINK_ERR_DEG_S = 0.1
LINK_ROLLED_M = 1.0
# A good fit restarts the reckoning only if its heading is within
# DR_AGREE_DEG (plus the gyro's LINK_ERR_DEG_S a second) of the reckoned one,
# or DR_TRUST_RUN good fits in a row have each turned within DR_STEP_DEG of
# the gyro. Scan to scan, fit and gyro part by 1.1 degrees at p99, 3.2 at
# p99.99 and 5.0 at most over 71k tracking scan pairs of the 10-29 replays.
DR_AGREE_DEG = 5.0
DR_STEP_DEG = 3.0
DR_TRUST_RUN = 10

# A shelved map keeps reckoning where the truck is on it (Shelf.here, see
# "Where the truck is on the shelf"), and a merge is held to that. Its
# heading may be off the reckoned one by the shelving link's err_yaw plus
# SHELF_DRIFT_DEG, and SHELF_DRIFT_DEG_S more for every second since the map
# was shelved: how far SLAM's heading parts from the bias-corrected gyro's
# while it tracks, over windows on one map in replays of the 10-29 drives
# (gaps in the gyro left out), p99 6 degrees over 10 s, 11 over 60 s, 12.5
# over 120 s, 17-18 over 180-300 s, the worst 20 over 300 s bar one drive
# whose SLAM spun on its own. A 180 degree twin, the merge 174641 took at
# 250 s, is far outside it. The search only tries headings within it, so a
# circle is a few headings, not ninety. A gap in the gyro since the map was
# shelved and the heading is anyone's: the whole circle, as before.
SHELF_DRIFT_DEG = 12.0
SHELF_DRIFT_DEG_S = 0.03
# The position's doubt: the shelving link's err_m, plus SHELF_TRACK_FRAC of
# every metre the fresh map tracked (its steps are good; what turns them
# onto the shelf is the gyro's heading, off by up to the tolerance above),
# plus what the link would own up to for the metres counted off the
# commanded speed or not counted at all (LINK_*). The search is drawn round
# Shelf.here as wide as that, never under reloc.SEARCH_R_M, never over
# SHELF_MAX_R_M.
SHELF_TRACK_FRAC = 0.15
SHELF_MAX_R_M = 6.0
# A merge the walls do not bear out is not made: of the fresh map's walls
# that land, through the merge, where the shelf knows something, at least
# SHELF_AGREE must land within a cell of one of its walls rather than on its
# open floor (agreement()). Under SHELF_AGREE_MIN of them is no evidence
# either way. In replays of the 10-29 drives the right merges agreed 0.73 to
# 1.00 (68 with a loss injected, 2 of the losses that happened); the wrong
# ones 0.38 (191314, the right heading 6 m off where the truck was reckoned
# after 200 s lost), 0.42 (the 174641 twin) and 0.60 (an 85 degree one).
SHELF_AGREE = 0.65
SHELF_AGREE_MIN = 50

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
    reseed: str = ""          # "young" / "lost": this scan started a fresh map
    #                           "merged": the old one is the map again
    # With a reseed: how the map before this scan sits on the one after it,
    # new = T . old, as (x, y, yaw, err_m, err_yaw, px, py): the transform,
    # how far off its position may be (metres) and its heading (radians),
    # and where on the old map the truck was reckoned to be (the heading's
    # error swings a point round there). () without one.
    link: tuple = ()
    # A "lost" reseed that left the shelf as it was: what it had there was
    # the bigger map, and the one this scan left is gone.
    kept_shelf: bool = False


class Shelf:
    """A map a lost reseed put aside, and where the search for the truck on
    it has got to (Slam._shelf_slice)."""

    def __init__(self, grid, writes):
        self.grid = grid
        self.writes = writes        # its writes when it was put aside
        self.search = None          # reloc.Search round the truck
        self.tree = None            # the old map's hit means near it
        self.tgt = None
        self.batch = []             # (score, T) of the headings scored
        self.headings = 0           # headings in the batch
        self.todo = []              # T to ICP from, best first
        self.circle = False         # a circle is done: pick once todo is
        self.pending = None         # T the last circle picked
        self.credit = 0.0           # seconds the search may still spend
        # Where the truck is reckoned to be on it (Slam._reckon_shelf), the
        # doubt in that (metres, radians), and the scan time it was shelved.
        # here is None when nothing reckoned it (it was put there by hand);
        # err_yaw is pi once the gyro has had a gap.
        self.here = None
        self.err_m = 0.0
        self.err_yaw = math.pi
        self.since = 0.0
        self.rolled = False         # the last reckoned step was rolled


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
        # Scan time the pose stopped being believed (or the truck was last
        # rolled); None while it is.
        self._lost_since = None
        # Writes since the map was last started, for SEED_WRITES.
        self._map_writes = 0
        self.reseeds = 0
        self.merges = 0
        # Shelf merges found twice that the walls did not bear out.
        self.shelf_refused = 0
        self._reseed = ""
        # The relocalization in progress while lost (reloc.Search), and a fit
        # waiting to be confirmed on the next scan.
        self._search = None
        self._pending = None
        # While lost: the heading the gyro has carried the truck to since the
        # pose stopped being believed (map radians), or None once a gap in it
        # means it is not known. It orders the search and settles a fit with
        # a twin (reloc.Search.pick).
        self._lost_yaw = None
        # A map put aside by a reseed and the search for the truck on it.
        self._shelf: Optional[Shelf] = None
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
        # The seed the next poor fit's retries start from (_match), and the
        # seconds of level poor fits in a row (_poor_too_long).
        self._seed_next = 0
        self._poor_s = 0.0
        self.recoveries = 0
        self.last_points = None
        # Dead reckoning since the last good fit, on the current map: the
        # pose it was at, moved on by the gyro and the commanded speed while
        # nothing fits (see "How a fresh map sits"). And what it has counted
        # since: metres, seconds, metres it could not count, rollovers.
        self._dr = np.zeros(3)
        self._dr_m = 0.0
        self._dr_s = 0.0
        self._dr_unknown_m = 0.0
        self._dr_rolled = 0
        self._dr_blind = False
        # Good fits in a row that each turned as the gyro did.
        self._dr_run = 0
        self._link = ()
        self._shelved = False

    # ---- the step -------------------------------------------------------

    def step(self, t, ranges, period=None, speed=None) -> Result:
        """One revolution. `speed`, the commanded ground speed (m/s, signed)
        as of this scan, or None: only ever used to reckon where the truck
        has got to while nothing fits, for Result.link."""
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
        self._reseed = ""
        self._link = ()
        self._shelved = False

        if tilt > ROLL_DEG:
            # Frozen. Whatever the scan sees now is the floor, the ceiling or
            # a hand; none of it is a wall to match. The map is kept. The
            # gyro is projected on up (imu.py), so it still says how far the
            # truck has turned about the vertical while it is over.
            if not self.lost:
                self._lost_yaw = float(self.pose[2])
            rolled_rate = (imu.rate_between(last_t, t, self.bias)
                           if last_t is not None else None)
            self._carry_yaw(rolled_rate,
                            t - last_t if last_t is not None else 0.0)
            if last_t is not None:
                self._reckon(rolled_rate, None, t - last_t, rolled=True)
                self._reckon_shelf(rolled_rate, None, t - last_t, rolled=True)
            self.lost = True
            self._lost_since = t
            self._search = None
            self._pending = None
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
        recovered = False
        poor_out = False
        if not self.started:
            newpose, good = pose.copy(), True
            self.started = True
        elif self._tree is None:
            newpose, good = pred, False
        elif self.lost:
            self._carry_yaw(imu_rate, dt)
            fit, good = self._relocalize(p, nrm, pred, self._lost_yaw)
            newpose = np.array(fit.pose) if good else pose.copy()
            if good:
                self.lost = False
                self._lost_since = None
                self.fails = 0
                self.recoveries += 1
                self.vel[:] = 0.0
                recovered = True
            elif (self._lost_since is not None
                  and t - self._lost_since >= RESEED_LOST_S - 1e-9):
                self._reseed = "lost"
        else:
            fit = self._match(p, nrm, pred, imu_rate is not None, dt, t0)
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
            if not good and self._map_writes < SEED_WRITES:
                self._reseed = "young"
            elif good and self._poor_too_long(fit, dt):
                good = False
                poor_out = True

        trusted = (good and not self._reseed
                   and (recovered or last_t is None
                        or self._agrees(newpose, pose, imu_rate, dt)))
        if not good:
            self._dr_run = 0
        if last_t is not None and not trusted:
            self._reckon(imu_rate, speed, dt)
        if self._reseed:
            # A fresh map from this scan, at the pose the truck is believed
            # to be at, heading off the IMU: written below as the first scan
            # of a map is. A lost one's old map is shelved, never dropped.
            self._shelved = self._start_over(keep=self._reseed == "lost")
            newpose, good, fit = pred.copy(), True, None
            self._link = self._link_from_reckoning(newpose)
            if self._shelved:
                self._shelf_from_reckoning(t)
        if self._tree is not None and not good and not self.lost:
            self.fails += 1
            self.vel *= 0.5
            if self.fails >= MAX_FAILS or poor_out:
                self.lost = True
                self._lost_since = t
                self._poor_s = 0.0
                # The failing scans went on the IMU's heading (pred), so it
                # is still the gyro's from here.
                self._lost_yaw = (float(newpose[2]) if imu_rate is not None
                                  else None)
                self.vel[:] = 0.0
                newpose = pose.copy()
        elif good:
            self.fails = 0

        # Not off a recovery: the jump from the frozen pose to the found one
        # is no speed, and taken as one it carried the next scans' prediction
        # off at metres a second (and the gyro's bias with it).
        if last_t is not None and good and fit is not None and not recovered:
            d = max(dt, 1e-3)
            v_new = (newpose - pose) / d
            v_new[2] = ((newpose[2] - pose[2] + np.pi) % (2 * np.pi)
                        - np.pi) / d
            self.vel = 0.5 * self.vel + 0.5 * v_new if dt < 0.3 else v_new * 0
            if imu_rate is not None and dt < 0.2:
                # The gyro's slow bias, from how far it disagrees with the
                # fits: an EMA slow enough that one bad fit is noise.
                self.bias += 0.002 * (imu_rate - v_new[2])
        if (self._shelf is not None and last_t is not None
                and not self._shelved):
            moved = (newpose[:2] - pose[:2]
                     if trusted and fit is not None and not recovered
                     else None)
            self._reckon_shelf(imu_rate, speed, dt, moved=moved,
                               fresh_yaw=float(newpose[2]))
        self.pose = newpose
        if trusted or (good and self._reseed):
            self._dr_reset(newpose)

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
        if (self._shelf is not None and good and not self._reseed
                and state in (TRACKING, TILTED)
                and self._shelf_slice(p, nrm, t0, t)):
            self._reseed = "merged"
            self._dr_reset(self.pose)
        self._refresh_target()
        self.last_t = t
        return self._result(t, state, good, written, why, fit, returns,
                            len(p), dropped, tilt, t0)

    # ---- the parts -------------------------------------------------------

    def _match(self, p, nrm, pred, have_prior, dt, t0=None):
        """ICP from the prediction, and from the seeds if that fits under
        RESEED_BELOW: as many as fit in the step's RETRY_STEP_S (begun at
        t0, Slam's clock), one at least, going on round them from where the
        last poor scan stopped."""
        sig = math.radians(SIGMA_YAW_DEG) * max(dt / 0.1, 1.0)
        yp = pred[2] if have_prior else None
        fit = mt.icp(p, nrm, self._tree, self._tgt, pred, yp, sig)
        if fit.converged and fit.inliers >= RESEED_BELOW:
            self._seed_next = 0
            return fit
        self.retries += 1
        best = fit
        seeds = list(self._seeds(pred))
        start = self._seed_next % len(seeds)
        for k in range(len(seeds)):
            if (k and t0 is not None
                    and self._clock() - t0 >= RETRY_STEP_S):
                break
            seed = seeds[(start + k) % len(seeds)]
            self._seed_next = (start + k + 1) % len(seeds)
            c = mt.icp(p, nrm, self._tree, self._tgt, seed, iters=25)
            if c.converged and c.inliers > best.inliers + 0.05:
                best = c
            if best.converged and best.inliers >= RESEED_BELOW:
                self._seed_next = 0
                break
        return best

    def _poor_too_long(self, fit, dt):
        """Whether a good fit ends POOR_LOST_S of level scans fitted too
        poorly to write, with none good enough to write between them."""
        if fit.inliers >= WRITE_MIN_INLIERS:
            self._poor_s = 0.0
            return False
        pitch, roll = self._att
        if not (pitch > self.write_max_pitch or roll > self.write_max_roll):
            self._poor_s += min(max(dt, 0.0), BAND_MAX_GAP_S)
        return self._poor_s >= POOR_LOST_S - 1e-9

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

    def _start_over(self, keep=False):
        """A fresh map, from nothing: no tree, nothing lost or failing. With
        `keep` the old one goes on the shelf (the bigger of it and whatever
        is there already), else it is cleared in place. The pose is kept, so
        the map frame goes on where it was. True if the map it left went on
        the shelf."""
        shelved = False
        if keep:
            if self._shelf is None or self._map_writes > self._shelf.writes:
                self._shelf = Shelf(self.grid, self._map_writes)
                shelved = True
            g = self.grid
            self.grid = gr.Grid(size=g.size, res=g.res, l_max=g.l_max,
                                x0=float(self.pose[0]), y0=float(self.pose[1]),
                                max_size=g.max_size)
        else:
            self.grid.clear()
        self._tree = None
        self._tgt = None
        self._tree_at = None
        self.lost = False
        self._lost_since = None
        self._search = None
        self._pending = None
        self.fails = 0
        self._poor_s = 0.0
        self.vel[:] = 0.0
        self._map_writes = 0
        self.reseeds += 1
        return shelved

    # ---- dead reckoning, for Result.link ---------------------------------

    def _dr_reset(self, pose):
        self._dr = np.array(pose, float)
        self._dr_m = self._dr_s = self._dr_unknown_m = 0.0
        self._dr_rolled = 0
        self._dr_blind = False

    def _agrees(self, newpose, pose, rate, dt):
        """Whether a good fit to newpose (from pose, dt ago) is one the
        reckoning starts again from: its heading within DR_AGREE_DEG of the
        reckoned one, or the last DR_TRUST_RUN good fits each turned as the
        gyro did. With no gyro to say, any good fit is."""
        if rate is None or not math.isfinite(rate) or self._dr_blind:
            return True
        step = abs(_wrap(float(newpose[2]) - float(pose[2]) - rate * dt))
        self._dr_run = (self._dr_run + 1
                        if step <= math.radians(DR_STEP_DEG) else 0)
        off = abs(_wrap(float(newpose[2]) - (self._dr[2] + rate * dt)))
        tol = math.radians(DR_AGREE_DEG + LINK_ERR_DEG_S * (self._dr_s + dt))
        return off <= tol or self._dr_run >= DR_TRUST_RUN

    def _reckon(self, rate, speed, dt, rolled=False):
        """Move the dead-reckoned pose on over dt: the heading by the gyro's
        rate (kept with none), the position by `speed` along it. Rolled, it
        is not driving; it may be being carried (LINK_ROLLED_M)."""
        if not dt > 0.0:
            return
        if rate is not None and math.isfinite(rate):
            self._dr[2] = _wrap(self._dr[2] + rate * dt)
        else:
            # A gap in the gyro: how far it turned meanwhile is anyone's.
            self._dr_blind = True
        self._dr_s += dt
        if rolled:
            self._dr_rolled = 1
            return
        if speed is None or not math.isfinite(speed) or dt > 1.0:
            # A second's gap is not one speed held across it.
            self._dr_unknown_m += LINK_UNKNOWN_M_S * dt
            return
        step = float(speed) * dt
        self._dr[0] += step * math.cos(self._dr[2])
        self._dr[1] += step * math.sin(self._dr[2])
        self._dr_m += abs(step)

    def _link_from_reckoning(self, new_pose):
        """Result.link for a reseed: the old map onto the new one, off where
        the truck is reckoned to be on the old map and where the new one
        puts it (new_pose = T . reckoned)."""
        T = _between(new_pose, self._dr)
        err_m = (LINK_ERR_M + LINK_ERR_FRAC * self._dr_m + self._dr_unknown_m
                 + LINK_ROLLED_M * self._dr_rolled)
        err_yaw = (math.pi if self._dr_blind else
                   math.radians(LINK_ERR_DEG + LINK_ERR_DEG_S * self._dr_s))
        return (T[0], T[1], T[2], float(err_m), float(err_yaw),
                float(self._dr[0]), float(self._dr[1]))

    # ---- where the truck is on the shelf ---------------------------------

    def _shelf_from_reckoning(self, t):
        """A map just shelved: the truck is where the reckoning has it on
        that map (the link's own pivot and heading), as sure as the link."""
        sh = self._shelf
        _, _, _, err_m, err_yaw = self._link[:5]
        sh.here = np.array(self._dr, float)
        sh.err_m, sh.err_yaw, sh.since = float(err_m), float(err_yaw), t
        sh.search = None

    def _reckon_shelf(self, rate, speed, dt, moved=None, fresh_yaw=0.0,
                      rolled=False):
        """Move Shelf.here on over dt. The heading by the gyro, as the
        reckoning for a link is; the position by the fresh map's own step
        when it fitted (`moved`, its x, y change, turned onto the shelf by
        how far the gyro's heading there is from the fresh map's), else by
        `speed` along the heading. Neither depends on the fresh map's frame,
        so a fresh map that is started again, or relocalizes onto a twin of
        itself, does not move it."""
        sh = self._shelf
        if sh is None or sh.here is None or not dt > 0.0:
            return
        h = sh.here
        if rate is not None and math.isfinite(rate):
            h[2] = _wrap(h[2] + rate * dt)
        else:
            sh.err_yaw = math.pi
        if rolled:
            sh.err_m += LINK_ROLLED_M if not sh.rolled else 0.0
            sh.rolled = True
            return
        sh.rolled = False
        if moved is not None:
            d = mt.rot(h[2] - fresh_yaw) @ np.asarray(moved, float)
            h[0] += d[0]
            h[1] += d[1]
            sh.err_m += SHELF_TRACK_FRAC * math.hypot(d[0], d[1])
        elif speed is None or not math.isfinite(speed) or dt > 1.0:
            sh.err_m += LINK_UNKNOWN_M_S * dt
        else:
            step = float(speed) * dt
            h[0] += step * math.cos(h[2])
            h[1] += step * math.sin(h[2])
            sh.err_m += LINK_ERR_FRAC * abs(step)

    def _on_shelf(self, sh):
        """The truck on the shelved map: where it is reckoned to be, or with
        nothing reckoned the pose as it is (the fresh map went on from the
        old frame)."""
        if sh.here is None:
            return np.array(self.pose, float)
        return np.array(sh.here, float)

    def _shelf_tol(self, sh, t):
        """How far a merge's heading may be off the reckoned one, radians
        (pi: anything goes)."""
        if sh.here is None:
            return math.pi
        drift = SHELF_DRIFT_DEG + SHELF_DRIFT_DEG_S * max(t - sh.since, 0.0)
        return min(math.pi, sh.err_yaw + math.radians(drift))

    def _shelf_r(self, sh):
        """The search's radius round Shelf.here: its position doubt."""
        if sh.here is None:
            return rl.SEARCH_R_M
        return min(max(rl.SEARCH_R_M, sh.err_m), SHELF_MAX_R_M)

    def _carry_yaw(self, rate, dt):
        """Turn the lost heading on by the gyro's rate over dt; a scan with
        no rate (a gap) leaves it unknown for the rest of the episode."""
        if self._lost_yaw is None:
            return
        if rate is None:
            self._lost_yaw = None
        else:
            self._lost_yaw = _wrap(self._lost_yaw + rate * dt)

    def _candidates(self, search, p, nrm, tree, tgt, yaw_now, prior=None):
        """ICP on the search's best candidates this scan, each kept on the
        search. (the best Fit this scan or None, the pose the search picks
        if this scan finished a circle, else None)."""
        best = None
        for _, cand in search.step(p, yaw_now):
            c = mt.icp(p, nrm, tree, tgt, cand, iters=25)
            if c.converged:
                search.add(c.inliers, c.pose)
                if best is None or c.inliers > best.inliers:
                    best = c
        if search.swept:
            picked = search.pick(RECOVER_MIN_INLIERS, prior=prior)
        elif prior is not None:
            picked = search.pick(RECOVER_MIN_INLIERS, prior=prior, early=True)
        else:
            picked = None
        return best, picked

    def _relocalize(self, p, nrm, pred, prior=None):
        """One scan's worth of looking for the truck on the map it lost its
        place on, `prior` the gyro's heading (or None). (best fit or None,
        whether it ends lost). A pick (reloc.Search.pick) ends it only if
        ICP from it on the next scan lands within CONFIRM_M / CONFIRM_DEG and
        fits as well."""
        if self._tree is None:
            return None, False
        best = None
        if self._pending is not None:
            c = mt.icp(p, nrm, self._tree, self._tgt, self._pending, iters=25)
            pend, self._pending = self._pending, None
            if (c.converged and c.inliers >= RECOVER_MIN_INLIERS
                    and _close(c.pose, pend)):
                self._search = None
                return c, True
            best = c if c.converged else None
        if self._search is None:
            self._search = rl.Search(self.grid, self.pose[0], self.pose[1],
                                     self.pose[2])
        c, picked = self._candidates(self._search, p, nrm, self._tree,
                                     self._tgt,
                                     pred[2] if prior is None else prior,
                                     prior)
        if c is not None and (best is None or c.inliers > best.inliers):
            best = c
        if picked is not None:
            self._pending = np.array(picked)
        # The best of a failed search still goes out, so /slam/status says
        # how near it came.
        return best, False

    def _shelf_slice(self, p, nrm, t0, t):
        """On a fresh map with one on the shelf: the next pieces of the
        search for the truck on the shelved one (_shelf_piece). Each step
        adds SHELF_SHARE_S to what the search may spend, never banking more
        than that, and every piece costs what it took on Slam's clock; the
        pieces run while there is some left, and stop once the step (begun
        at t0) is SHELF_STEP_S long, after one at least. True if the
        shelved map was found and is the map again."""
        sh = self._shelf
        sh.credit = min(sh.credit + SHELF_SHARE_S, SHELF_SHARE_S)
        first = True
        while sh.credit > 0 and (first or self._clock() - t0 < SHELF_STEP_S):
            first = False
            t1 = self._clock()
            done = self._shelf_piece(sh, p, nrm, t)
            sh.credit -= self._clock() - t1
            if done is not None:
                return done
        return False

    def _shelf_piece(self, sh, p, nrm, t):
        """One piece of the shelf's search, the first of these that is due:

          draw the search (reloc.Search) and the old map's tree round where
          the truck is reckoned to be on it (_on_shelf), only the headings
          within _shelf_tol of the reckoned one and as wide as _shelf_r, the
          first time and again once it is SHELF_MOVE_M off;
          ICP from the next candidate of the last reloc.YAWS_PER_SCAN
          headings scored, the best reloc.TOP_K of them;
          once a circle's candidates are all fitted, the pick (as a lost
          relocalization picks) and ICP from it (_shelf_pick);
          else score the next heading.

        A candidate is kept as the old-from-new transform (_between), the
        one thing about it that does not change while the truck drives on,
        and put back on the truck's pose of the scan that fits it. None to
        go on, False to stop for this scan, True if it merged."""
        x, y, yaw = (float(v) for v in self._on_shelf(sh))
        tol = self._shelf_tol(sh, t)
        span = tol if tol < math.pi else None
        if sh.search is None or math.hypot(sh.search.field.cx - x,
                                           sh.search.field.cy - y) > SHELF_MOVE_M:
            r = self._shelf_r(sh)
            sh.search = rl.Search(sh.grid, x, y, yaw, r_m=r, span=span)
            tgt = sh.grid.local_points(x, y, LOCAL_R_M + r + SHELF_MOVE_M)
            sh.tgt = tgt
            sh.tree = cKDTree(tgt) if len(tgt) >= 50 else None
            sh.batch, sh.headings, sh.todo, sh.circle = [], 0, [], False
            return None
        if sh.tree is None or sh.search.field.points == 0:
            return False
        if sh.todo:
            c = mt.icp(p, nrm, sh.tree, sh.tgt,
                       _apply(sh.todo.pop(0), self.pose), iters=25)
            if c.converged:
                sh.search.add(c.inliers, _between(c.pose, self.pose))
            return None
        if sh.circle:
            sh.circle = False
            return True if self._shelf_pick(sh, p, nrm, t) else None
        sh.search.span = span
        cands = sh.search.step(p, yaw, per_scan=1, top_k=rl.PER_YAW)
        if not cands and not sh.search.swept:
            return False
        sh.batch += [(score, _between(cand, self.pose))
                     for score, cand in cands]
        sh.headings += 1
        if sh.headings >= rl.YAWS_PER_SCAN or sh.search.swept:
            sh.batch.sort(key=lambda b: -b[0])
            sh.todo = [T for _, T in sh.batch[:rl.TOP_K]]
            sh.batch, sh.headings = [], 0
            sh.circle = sh.search.swept
        return None

    def _shelf_pick(self, sh, p, nrm, t):
        """A circle of the shelf's search done: its pick, ICP from it on
        this scan, and the merge if it is the transform the last circle
        picked too. Only a fit within _shelf_tol of the reckoned heading is
        an answer at all, before the pick and after the ICP: a twin or a
        wrong room is never merged however well it fits. True if it
        merged."""
        search = sh.search
        search.fits = [(i, _apply(T, self.pose)) for i, T in search.fits]
        here = self._on_shelf(sh)
        tol = self._shelf_tol(sh, t)
        gated = tol < math.pi
        if gated:
            search.fits = [f for f in search.fits
                           if abs(_wrap(f[1][2] - here[2])) <= tol]
        picked = search.pick(RECOVER_MIN_INLIERS,
                             prior=float(here[2]) if gated else None)
        if picked is None:
            sh.pending = None
            return False
        c = mt.icp(p, nrm, sh.tree, sh.tgt, picked, iters=25)
        if not (c.converged and c.inliers >= RECOVER_MIN_INLIERS
                and (not gated or abs(_wrap(c.pose[2] - here[2])) <= tol)):
            sh.pending = None
            return False
        T = _between(c.pose, self.pose)
        if sh.pending is None or not _close(T, sh.pending):
            sh.pending = T
            return False
        share, counted = agreement(sh.grid, self.grid, T)
        if counted >= SHELF_AGREE_MIN and share < SHELF_AGREE:
            # Found twice, and the fresh map's walls stand on the shelf's
            # open floor: the wrong place, however well the scan fits it.
            sh.pending = None
            self.shelf_refused += 1
            return False
        # Found twice the same way: the fresh map into the old, and the old
        # one is the map.
        merge(sh.grid, self.grid, T)
        self.grid = sh.grid
        self.pose = np.array(c.pose, float)
        self._shelf = None
        self._map_writes = sh.writes + self._map_writes
        self._tree = None
        self._tree_at = None
        self.vel[:] = 0.0
        self.merges += 1
        self._link = (float(T[0]), float(T[1]), float(T[2]), 0.0, 0.0,
                      float(self.pose[0]), float(self.pose[1]))
        return True

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
        self._map_writes += 1

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
            banded=bool(self._banded), reseed=self._reseed,
            link=self._link if self._reseed else (),
            kept_shelf=self._reseed == "lost" and not self._shelved)


# ---- poses and maps ---------------------------------------------------------

def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _close(a, b, m=CONFIRM_M, deg=CONFIRM_DEG):
    """Two (x, y, yaw) within m metres and deg degrees of each other."""
    return (math.hypot(a[0] - b[0], a[1] - b[1]) <= m
            and abs(_wrap(a[2] - b[2])) <= math.radians(deg))


def _apply(T, pose):
    """T . pose: a pose in the new map's frame in the old one's."""
    R = mt.rot(T[2])
    xy = R @ np.asarray(pose[:2], float) + (T[0], T[1])
    return np.array([xy[0], xy[1], _wrap(pose[2] + T[2])])


def _between(old_pose, new_pose):
    """T with old_pose = T . new_pose: the old map's frame from the new's,
    as (x, y, yaw)."""
    dyaw = _wrap(old_pose[2] - new_pose[2])
    R = mt.rot(dyaw)
    t = np.asarray(old_pose[:2], float) - R @ np.asarray(new_pose[:2], float)
    return (float(t[0]), float(t[1]), dyaw)


def agreement(old, new, T):
    """How far `new`'s walls bear out the transform T (old = T . new): of
    its wall cells that land on a cell `old` knows (or next to a wall of
    it), the share that land within a cell of an old wall rather than on old
    open floor, and how many that is. (share, count); (1.0, 0) with none."""
    ii, jj = np.nonzero(new.lo > gr.L_OCC)
    if not len(ii):
        return 1.0, 0
    wx = (ii + new.ox + 0.5) * new.res
    wy = (jj + new.oy + 0.5) * new.res
    q = np.stack([wx, wy], 1) @ mt.rot(T[2]).T + (T[0], T[1])
    hx = np.floor(q[:, 0] / old.res).astype(np.int64) - old.ox
    hy = np.floor(q[:, 1] / old.res).astype(np.int64) - old.oy
    n = old.size
    ok = (hx >= 1) & (hx < n - 1) & (hy >= 1) & (hy < n - 1)
    hx, hy = hx[ok], hy[ok]
    near = np.zeros(len(hx), bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            near |= old.lo[hx + dx, hy + dy] > gr.L_OCC
    agree = int(near.sum())
    clash = int(((~near) & (old.lo[hx, hy] < 0)).sum())
    counted = agree + clash
    return (agree / counted if counted else 1.0), counted


def merge(old, new, T):
    """Every cell `new` has seen, into `old` through T (old = T . new): the
    log-odds added and clamped, and a hit mean carried over where `old` has
    none. Cells that land outside `old`'s window are dropped (it is 100 m a
    side; the fresh map is a few minutes of driving)."""
    known = new.lo != 0
    ii, jj = np.nonzero(known)
    if not len(ii):
        old.version += 1
        return
    lo = new.lo[ii, jj].astype(np.int16)
    has = new.cnt[ii, jj] > 0
    fx = np.where(has, (new.fx[ii, jj] + 0.5) / 256.0, 0.5)
    fy = np.where(has, (new.fy[ii, jj] + 0.5) / 256.0, 0.5)
    wx = (ii + new.ox + fx) * new.res
    wy = (jj + new.oy + fy) * new.res
    R = mt.rot(T[2])
    q = np.stack([wx, wy], 1) @ R.T + (T[0], T[1])
    gx = np.floor(q[:, 0] / old.res).astype(np.int64)
    gy = np.floor(q[:, 1] / old.res).astype(np.int64)
    hx, hy = gx - old.ox, gy - old.oy
    n = old.size
    ok = (hx >= 0) & (hx < n) & (hy >= 0) & (hy < n)
    hx, hy, lo, has, q, gx, gy = hx[ok], hy[ok], lo[ok], has[ok], q[ok], gx[ok], gy[ok]
    cur = old.lo[hx, hy].astype(np.int16)
    old.lo[hx, hy] = np.clip(cur + lo, gr.L_MIN, old.l_max).astype(np.int8)
    m = has & (old.cnt[hx, hy] == 0)
    old.fx[hx[m], hy[m]] = np.clip((q[m, 0] / old.res - gx[m]) * 256.0, 0, 255).astype(np.uint8)
    old.fy[hx[m], hy[m]] = np.clip((q[m, 1] / old.res - gy[m]) * 256.0, 0, 255).astype(np.uint8)
    old.cnt[hx[m], hy[m]] = 1
    old.version += 1
