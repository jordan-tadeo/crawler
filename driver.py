"""Depth-driven roomba mode: crawl forward, back out of whatever it hits.

Everything here is synchronous and non-blocking. The control loop owns the
ESC and the servos on one thread and calls tick() about ten times a second,
so nothing in this file is allowed to sleep: the old recover() awaited 2.5
seconds with the throttle pinned in reverse, and for those 2.5 seconds
nothing was polling the controller, the stop signal or the camera. SIGTERM
arrived and the car kept reversing.

recover() is a phase machine instead. begin_recover() commands the first
phase and returns, recovering() advances it on the next tick, and abort()
drops it on the floor and zeroes the throttle. A START press, a controller
dropout and a systemctl stop are all honoured inside one 10 ms tick, even
halfway through a reverse.
"""

import random
import time
from typing import NamedTuple

import numpy as np

# Recover phases, in order.
_PHASE_REVERSE = "reverse"
_PHASE_SETTLE = "settle"

# How long each phase holds. These two came out of the original recover()'s
# asyncio.sleep() calls, so it is the same feel, just not blocking.
REVERSE_SECONDS = 2.0
SETTLE_SECONDS = 0.5

# What the IMU is allowed to say about motion. Three answers, not two: the old
# is_moving() had only "moving" and "not moving" and it filed a missing sample
# under the second one, so a tick with no IMU packet in it was indistinguishable
# from a car with its wheels buried.
MOVING = "moving"
STILL = "still"
UNKNOWN = "unknown"

# The parked floor, measured on the car on 2026-10-26: powered, ESC armed at
# neutral 1575, nobody touching it. Three recordings -- 95 s off the /imu
# telemetry channel (1673 samples at 17.6 Hz), and 60 s taken straight off the
# camera view the driver reads, both at the driver's own 10 Hz tick (587
# samples) and at the device rate (5256 samples at 87.6 Hz). All three agree:
#
#   linear accel magnitude   p50 0.017   p95 0.033   p99 0.039   max 0.058 m/s^2
#   gyro magnitude           p50 0.0026  p95 0.0045  p99 0.0055  max 0.0069 rad/s
#
# The gyro floor is the tighter of the two by about a factor of eight, and the
# part that actually matters: LINEAR_ACCELERATION is gravity-compensated, so it
# reads that same floor whether the car is parked or rolling at a steady creep.
# It cannot tell those two apart at all. The gyro can -- a crawler rolling over
# real ground shakes, and the shake is the witness.
#
# Both channels come off the BNO086 as fixed point, which is why a component
# sometimes reads an exact 0.0 while the other two show noise -- nothing is
# being dropped, it is the zero bucket. Every accel component in 1673 samples
# is an exact multiple of 1/256 = 0.00390625 m/s^2 and every gyro component an
# exact multiple of 0.00106526 rad/s. Parked, the whole accel floor fits inside
# 11 of its steps and the whole gyro floor inside 6 of its own.
#
# GYRO_MOVING_MIN sits at five times the measured parked maximum and about eight
# times p99, which is 33 gyro steps against a parked floor of 6 -- clear of the
# quantisation either way. The parked end of that range is measured. The rolling
# end is not -- see the note in motion_evidence().
GYRO_MOVING_MIN = 0.035  # rad/s

# How much evidence before the car is allowed to call itself stuck. At the 10 Hz
# tick this is a full second of consecutive confident no-motion readings, every
# one of them taken while forward throttle was already up.
#
# A count of ticks on purpose, not a wall-clock duration. The control loop does
# not always make its 100 Hz -- I have watched it sit at 75 Hz with a 190 ms
# stall in it -- and a count can only ever make confirming a stick take longer
# in real time, never make it fire early. A deadline would do the opposite.
STUCK_CONFIRM_SAMPLES = 10

# And how long forward throttle must have been commanded before the stuck check
# arms at all. A car that has just been handed the wheels has not been given a
# chance to move yet. The old code declared it stuck on the very first tick of
# auto mode -- last_recovery_time started at 0.0, so the cooldown was already
# expired -- which is exactly what the journal shows: "Switched to auto" and
# "STUCK - backing up to recover." stamped on the same second, every time.
STUCK_ARM_SECONDS = 1.0


# ---------------------------------------------------------------------------
# The footprint the obstacle check has to cover
# ---------------------------------------------------------------------------
#
# What the old obstacle_in_front() actually looked at: a 50x50 patch at the
# dead centre of a 640x400 depth frame. 2,500 pixels out of 256,000, which is
# 0.98% of the frame, and in the world it is a 45 mm x 45 mm window at the
# 400 mm trigger distance. The car is 267 mm wide. A wall or a doorway corner
# that catches a wheel was never in that patch at all, and the one thing the
# car does for a living is drive into walls.
#
# The vertical half of it is worse and less obvious. The lowest row that patch
# read is row 225, 25 px below the optical centre, which is 3.24 degrees down.
# At 400 mm that is 22.7 mm below the lens axis, and with the lens 165 mm off
# the ground it is 142 mm of air. Anything shorter than that -- which is to
# say every rock, kerb lip and chair leg base on the floor -- passed
# underneath the check entirely. It was looking over the obstacle.
#
# get_steering_bias() did not cover for it either. Its two patches sit at 14.0
# to 25.5 degrees off centre and the centre patch covers 3.24 degrees, so the
# band from 3.24 to 14.0 degrees was read by nothing in this file -- and the
# wheel line is at 18.4 degrees, inside the steering patches but only ever as
# one term of a 1200 mm average, never as a stop.
#
# So the check is now N column probes tiling the width the car actually
# occupies, each one reaching from a little above the lens down toward the
# near ground, and each one failing closed on its own.
#
# --- the two measurements everything below is derived from ------------------
#
# Both taken off the car on 2026-10-26, so these are the authored numbers and
# the rest of this block is arithmetic on them:
#
#   track width, outside of tyre to outside of tyre   10.5 in = 266.7 mm
#   lens height above ground, suspension loaded        6.5 in = 165.1 mm
#
HALF_TRACK_MM = 133.35      # half of 266.7
CAMERA_HEIGHT_MM = 165.1

# How much wider than the tyres to treat as the car. The wheels sit at
# +/-133.35 mm in the camera's own frame and stay there whatever the chassis is
# doing, so this is not a yaw allowance -- yaw does not move the footprint in
# the image at all. It is clearance: the car sweeps outside its static width
# while it is steering, and I do not know the camera is exactly on the
# centreline. 20 mm is about three quarters of an inch outside each tyre.
# Widening this only ever costs me stopping for gaps I would have squeezed
# through, which is the cheap direction.
FOOTPRINT_MARGIN_MM = 20.0

# How close something has to be before it counts. Same 400 mm the old default
# had and the same meaning, promoted to a name because three things derive
# from it now. Worth knowing where it sits: setSubpixel(True) with extended
# disparity off gives this stereo pair a minimum range of
# fx * baseline / disparity_levels = 441.25 * 75 / 95 = 348 mm, so there are
# 52 mm of graded warning between first sight and the blind zone. Most real
# triggers will therefore be the no-return kind rather than a measured
# distance, which is exactly why the two are reported apart.
CLEARANCE_MM = 400.0

# Mono camera intrinsics at THE_400_P, and the loudest assumption in this
# file.
#
# Not read off the device. readCalibration() is an XLink round trip that can
# block and can raise, and the only place a Driver gets built is
# AutoPilot.resume() on the control-loop thread, which is documented to do
# neither -- so a calibration read would put a device call on the one thread
# that must never wait. These are the published OV9282 numbers instead: 72
# degree HFOV at 1280x800 native, halved to 640x400. fx = 640/tan(35.93 deg)
# /2 = 441.6, and 441.25 back-solves to a 71.90 degree HFOV and a 48.77
# degree VFOV, both of which match the datasheet. Square pixels, so fy = fx.
# They are scaled to whatever frame actually arrives, so a resolution change
# needs no edit here.
#
# The complication, and it is unresolved: oakd.py calls
# stereo.setDepthAlign(dai.CameraBoardSocket.RGB) while there is no
# ColorCamera node in the pipeline. The EEPROM carries RGB calibration whether
# or not that node exists, so the align can take effect, and if it does the
# depth frame is in the IMX378's frame rather than the mono pair's -- 69
# degree HFOV, and a 4:3 sensor squeezed into a 1.6 output. Worst case that is
# fx = 466 (5.5% high) and fy = 388 (12% low). The margins below were picked
# to survive it: at fy = 388 the bottom probe row still reads flat floor at
# 792 mm against a 400 mm budget. Do not change the stereo config to find out
# -- that is its own experiment. The cheap check is to park a measured 400 mm
# off a flat wall and see whether the probe span lands on 267 mm of it.
FX_AT_640 = 441.25
FY_AT_400 = 441.25

# Where the probe band stops, top and bottom. The two edges are parameterised
# differently on purpose, because different things bound them.
#
# The bottom is bounded by the floor. A probe takes the near distance in its
# column, so a row low enough to see flat ground inside the clearance budget
# reads the ground as an obstacle and the car reverses forever. The lowest row
# allowed is therefore the one where level ground reads FLOOR_CLEAR_MM:
#
#   level ground at distance Z appears fy * CAMERA_HEIGHT_MM / Z px below the
#   optical centre, so 900 mm puts it 80.9 px down, row 281 of 400.
#
# 900 mm against a 400 mm budget leaves 12.0 degrees of nose-down pitch before
# the floor itself trips a probe. A crawler pitches, so that margin is the
# whole reason this is not set lower -- 600 mm would buy sight of shorter
# rocks and leave only 7.0 degrees. The cost is stated rather than hidden: at
# 400 mm the lowest row sees 91.7 mm above the ground, so an obstacle shorter
# than that is still invisible. That is a little over tyre radius, which is
# roughly where the car ought to be climbing rather than stopping, but it is
# a judgement by feel and the number to move if he wants it more or less
# twitchy over small stuff.
#
# The top is bounded by the vehicle, not the floor. Nothing above the lens
# axis can ever be ground, so going higher is free as far as false positives
# go -- but an overhang the car drives under must not stop it, and at 400 mm
# the top of the frame is 346 mm off the ground, well over the roofline. So
# the band stops ENVELOPE_TOP_MM above the lens axis at the budget range,
# which with the floor edge makes the band cover 92 mm to 225 mm above the
# ground at 400 mm. A wall fills all of it; a doorway lintel misses all of it.
FLOOR_CLEAR_MM = 900.0
ENVELOPE_TOP_MM = 60.0

# How many probes across the footprint, and they tile it rather than sampling
# it -- contiguous, no gaps, so there is nothing for a wheel to hide between.
#
# The span is 2 * fx * (133.35 + 20) / 400 = 338 px at the budget range, so
# nine probes are 37.6 px each, which is 34 mm of real width out there. A tyre
# face is about 50 mm (estimated -- he measured the track, not the tyre), so
# anything wide enough to catch a wheel lights at least one whole probe and
# usually two. Fewer and the probes get wider than a tyre, which costs nothing
# in detection -- the statistic is a near distance, not an average -- but
# blurs which side of the car is blocked, and that is the array the steering
# decision and the telemetry both want. More and each probe holds fewer valid
# pixels, which makes the per-probe validity test noisier, and noisy validity
# is how this change turns into a car that reverses constantly.
PROBE_COLUMNS = 9

# Decimation. "Is anything within 40 cm" is not a high-frequency signal and
# the control loop does not have the microseconds to spare: it targets 100 Hz,
# measures 75, and has a 190 ms stall in its history. Stride 3 in both axes
# turns 147 x 338 pixels into about 637 samples per probe, which is still far
# more than any of the tests below need to be confident.
ROW_STRIDE = 3
COL_STRIDE = 3

# His 0.1, and the single most important line in this block.
#
# The old fail-closed rule was "under a tenth of the patch came back valid, so
# call it an obstacle", applied to one 2,500 pixel patch where under 10%
# validity is rare. Spread naively over the whole footprint -- frame edges,
# dark corners, sky, far field -- low validity stops being rare and the car
# reverses on a metronome again, which would read as a regression in the stuck
# detector that just shipped. Three things stop that:
#
#   the rule is per probe, never over the whole band, so one dead probe blocks
#   its own column and nothing else;
#
#   it is applied to FAR evidence, not to validity. A probe is only blocked
#   for no-return if it cannot produce MIN_VALID_FRACTION of its samples as
#   valid-and-beyond-the-budget. A probe that is mostly far and partly
#   unmeasurable has its answer already and is left alone. Absence of a
#   reading only means "obstacle" where a reading would have meant "too
#   close";
#
#   and the band's bottom rows always terminate on the ground inside 899 mm,
#   so in ordinary driving every probe has abundant far evidence and
#   no-return is structurally rare rather than routine.
MIN_VALID_FRACTION = 0.1

# How much of a probe has to be inside the budget before it counts, as a
# fraction of that probe's samples. Speckle rejection: a plain minimum would
# hand the whole column over to one bad pixel.
#
# 2% of 637 samples is 12, which is about 100 raw pixels. Sized off the
# smallest thing that can appear in the band at all: a 100 mm obstacle at
# 400 mm has its top 71.8 px below centre against a band bottom of 80.9, so
# it fills 9 px of rows across a whole probe -- 39 samples at stride 3, three
# times what it needs. Raise it if he sees spurious "near" reverses; the
# failure it protects against (missing a real wall) is worse than the failure
# it causes (a short reverse for nothing), so it is set sensitive.
NEAR_SAMPLE_FRACTION = 0.02
NEAR_SAMPLES_FLOOR = 4

# depthai reports an unmeasurable pixel as 0 and everything else as real
# millimetres. The sentinel is what invalid pixels are rewritten to so they
# sort to the far end of a partition instead of needing a mask: it is uint16
# max, so a genuine 65.5 m reading would be mistaken for invalid, which this
# stereo pair cannot produce.
FAR_SENTINEL = 65535

# Why a probe is blocked. Two completely different behaviours used to wear one
# function name and the journal could not tell them apart.
NEAR = "near"           # something is measurably inside the budget
NO_RETURN = "no_return" # nothing measurable, which on a stereo pair means too
                        # close, too dark or too featureless -- all reasons to
                        # stop
CLEAR = ""


class FootprintGeometry(NamedTuple):
    """Where the probes land on one particular frame shape, in pixels."""

    row_top: int
    row_bottom: int
    col_edges: tuple          # PROBE_COLUMNS + 1 column boundaries
    fx: float
    fy: float


class FootprintScan(NamedTuple):
    """One pass of the footprint check. Several consumers, one read of the frame.

    distance_mm     the near distance in each probe, or inf where the probe had
                    nothing to say. Robust: it is the NEAR_SAMPLE_FRACTION-th
                    smallest valid depth, not the minimum, so one speckle
                    cannot claim a column.
    valid_fraction  how much of each probe came back measurable at all. Pure
                    diagnostics -- nothing decides on it.
    blocked         whether each probe is inside its budget.
    reason          NEAR, NO_RETURN or CLEAR per probe.
    geometry        the pixel layout the numbers came from.
    """

    distance_mm: np.ndarray
    valid_fraction: np.ndarray
    blocked: np.ndarray
    reason: tuple
    geometry: FootprintGeometry


def footprint_geometry(shape, threshold_mm=CLEARANCE_MM) -> FootprintGeometry:
    """Where the probes go on a frame of this shape.

    Everything scales off the frame that actually arrived, so an 800P depth
    frame needs no edit: the intrinsics above are quoted at 640x400 and the
    field of view does not change with resolution.
    """
    h, w = shape[0], shape[1]
    fx = FX_AT_640 * w / 640.0
    fy = FY_AT_400 * h / 400.0

    # The footprint subtends atan(half_width / range), and a pinhole puts that
    # at fx * half_width / range pixels off the optical centre. Sized at the
    # budget range because that is the range at which the footprint is the
    # question -- further out the probes see a wider swath than the car, which
    # only ever reads as far and is therefore harmless.
    half_px = fx * (HALF_TRACK_MM + FOOTPRINT_MARGIN_MM) / threshold_mm
    left = max(0, int(round(w / 2.0 - half_px)))
    right = min(w, int(round(w / 2.0 + half_px)))
    # A span too narrow to cut into PROBE_COLUMNS distinct probes would leave
    # zero-width probes, and a probe with no pixels in it has no evidence and
    # so fails closed -- a stopped car for a reason that is arithmetic rather
    # than an obstacle. Widen to the frame first. Nothing off the device gets
    # here; a frame narrower than the probe count still fails closed.
    if right - left < PROBE_COLUMNS:
        left, right = 0, w

    row_top = max(0, int(round(h / 2.0 - fy * ENVELOPE_TOP_MM / threshold_mm)))
    row_bottom = min(h, int(round(h / 2.0 + fy * CAMERA_HEIGHT_MM / FLOOR_CLEAR_MM)) + 1)
    if row_bottom <= row_top:
        row_top, row_bottom = 0, h

    edges = np.linspace(left, right, PROBE_COLUMNS + 1).round().astype(int)
    return FootprintGeometry(row_top, row_bottom, tuple(int(e) for e in edges),
                             fx, fy)


def free_by_column(depth, threshold_mm=CLEARANCE_MM) -> FootprintScan:
    """Scan the footprint. Pure: an ndarray in, numbers out, no state touched.

    Returns the per-probe distances the plan asked for plus the two things a
    distance alone cannot say -- whether the probe is blocked, and which of the
    two reasons fired. One pass of the frame feeds the stop decision, the
    steering and the telemetry.

    Pure on purpose. This is the whole reason the check is testable on a
    laptop with no car attached: a wall, a corner, a blind frame and a single
    near column are all just arrays.
    """
    geo = footprint_geometry(depth.shape, threshold_mm)
    rows = slice(geo.row_top, geo.row_bottom, ROW_STRIDE)

    distance = np.full(PROBE_COLUMNS, np.inf)
    valid_fraction = np.zeros(PROBE_COLUMNS)
    blocked = np.zeros(PROBE_COLUMNS, dtype=bool)
    reason = []

    for i in range(PROBE_COLUMNS):
        probe = depth[rows, geo.col_edges[i]:geo.col_edges[i + 1]:COL_STRIDE]
        n = probe.size
        if n == 0:
            # A probe with no pixels in it has no evidence of anything, which
            # fails closed like any other.
            blocked[i] = True
            reason.append(NO_RETURN)
            continue

        # Invalid pixels rewritten to the sentinel rather than masked out, so
        # the partition below needs no per-probe boolean indexing and the
        # counts come off one array. Widened to int32 first: the sentinel does
        # not fit every dtype a depth frame could arrive in, and numpy 2 raises
        # on that rather than wrapping.
        filled = probe.astype(np.int32).ravel()
        filled[filled <= 0] = FAR_SENTINEL
        n_invalid = int(np.count_nonzero(filled == FAR_SENTINEL))
        n_valid = n - n_invalid
        n_near = int(np.count_nonzero(filled < threshold_mm))
        n_far = n_valid - n_near
        valid_fraction[i] = n_valid / n

        # The k-th smallest, not the smallest. k is the same count the near
        # test uses, which makes "distance_mm[i] < threshold_mm" and "at least
        # k samples are inside the budget" the identical statement -- the
        # number reported and the decision taken cannot drift apart.
        k = max(NEAR_SAMPLES_FLOOR, int(round(NEAR_SAMPLE_FRACTION * n)))
        if n >= k:
            kth = float(np.partition(filled, k - 1)[k - 1])
            if kth < FAR_SENTINEL:
                distance[i] = kth

        if distance[i] < threshold_mm:
            # Positive evidence first. A measured obstacle is an obstacle
            # whatever the rest of the probe looks like.
            blocked[i] = True
            reason.append(NEAR)
        elif n_far < MIN_VALID_FRACTION * n:
            # Not enough far evidence to claim the column is clear. This is
            # his 0.1, per probe, and asked about FAR samples rather than
            # valid ones -- see MIN_VALID_FRACTION.
            blocked[i] = True
            reason.append(NO_RETURN)
        else:
            reason.append(CLEAR)

    return FootprintScan(distance, valid_fraction, blocked, tuple(reason), geo)


class Driver:
    def __init__(self, controller, oakd):
        self.controller = controller
        self.camera = oakd

        self.last_recovery_time = 0.0
        self.recovery_cooldown = 5.0  # seconds before the stuck check re-arms
        self.imu_history = []
        self.gyro_history = []
        self.imu_window_size = 5  # window for smoothing both magnitudes

        # Stuck detection state. _throttle_since is when forward throttle was
        # first commanded in the current unbroken run of commanding it, and
        # None whenever the driver is not asking the car to go anywhere.
        self._throttle_since = None
        self._still_samples = 0

        # Tunable parameters. I set all four of these by feel on the bench and
        # they are not to be nudged from a desk.
        self.forward_speed = 0.2    # very slow forward motion
        self.reverse_speed = -0.6   # very slow backup
        self.max_turn = 0.6         # limit to avoid servo damage when steering against an object
        # Still my bench number, and still the same quantity: the smoothed
        # linear accel magnitude in m/s^2. What changed is that it is no longer
        # the only test. It is one of two OR'd terms now and the gyro is the one
        # carrying the weight. Measured parked, the accel magnitude tops out at
        # 0.058, so 0.2 never fired on a parked car -- the threshold was never
        # the problem and I have not touched it.
        self.stuck_threshold = 0.2

        # Recover phase machine. _phase is None whenever recover does not own
        # the wheels, which is the normal case.
        self._phase = None
        self._phase_until = 0.0
        self._turn_dir = 0.0

        # The last footprint scan, and what it cost. Read-only bookkeeping for
        # whoever is publishing telemetry -- set as a side effect of the check
        # that was going to run anyway, so it costs one assignment and never a
        # second read of the frame. Nothing in here decides on them.
        #
        # last_block_reason is NEAR, NO_RETURN, both joined by a "+", or the
        # empty string when the footprint is clear.
        self.last_scan = None
        self.last_scan_ms = 0.0
        self.last_block_reason = CLEAR

    def motion_evidence(self, imu) -> str:
        """What the IMU can say about motion: MOVING, STILL or UNKNOWN.

        UNKNOWN is the whole point of this function. The old is_moving()
        returned False for a missing sample, the caller read that as "not
        moving", and "not moving" went straight to "stuck" -- so an absent IMU
        packet and a car wedged in a doorway produced the same command.

        The signal is the gyro first and the accelerometer second. The camera's
        IMU node is enabled for LINEAR_ACCELERATION, which is gravity
        compensated: near zero at rest AND near zero at a constant velocity.
        forward_speed = 0.2 is a creep that reaches steady state in well under
        a second, so on the accel magnitude alone a car rolling along perfectly
        happily reads the same as a car that is not moving at all. Measured,
        that floor is p50 0.017 / max 0.058 m/s^2, nowhere near
        stuck_threshold, which is why the accel term is now a second opinion
        for a knock or a wheel breaking loose rather than the primary test.

        The gyro survives constant velocity. It reads the chassis shaking, and
        a crawler on real ground always shakes while it rolls.

        What I could not measure from a desk: how much gyro a creep at 0.2
        actually produces. GYRO_MOVING_MIN is anchored off the parked end of
        the range only, at 5x the measured parked maximum, which errs toward
        calling the car moving -- that is, toward never recovering for no
        reason, which is the failure being fixed. The number still wants a pass
        on the bench with the wheels up before I treat it as a feel number.
        """
        if imu is None:
            return UNKNOWN

        try:
            accel = imu["accel"]
            gyro = imu["gyro"]
        except (KeyError, TypeError):
            return UNKNOWN

        self.imu_history.append(float(np.linalg.norm(accel)))
        if len(self.imu_history) > self.imu_window_size:
            self.imu_history.pop(0)

        self.gyro_history.append(float(np.linalg.norm(gyro)))
        if len(self.gyro_history) > self.imu_window_size:
            self.gyro_history.pop(0)

        if (np.mean(self.gyro_history) > GYRO_MOVING_MIN
                or np.mean(self.imu_history) > self.stuck_threshold):
            return MOVING

        return STILL

    def _clear_motion_evidence(self):
        """Forget everything the IMU said. Called whenever the car stops.

        A recover reverses for two seconds and the smoothing windows are not
        sampled while recover owns the wheels, so without this the first
        readings after handing them back are a blend of before and after.
        """
        self.imu_history.clear()
        self.gyro_history.clear()
        self._still_samples = 0
        self._throttle_since = None

    def stuck(self, imu, now) -> bool:
        """True only on confirmed evidence that a car being told to go is not.

        Three gates and all three have to be open:

          forward throttle has been commanded for at least STUCK_ARM_SECONDS,
          because a motion check on a car nobody asked to move measures
          nothing;

          STUCK_CONFIRM_SAMPLES samples in a row have come back STILL, with an
          UNKNOWN holding the count where it is rather than adding to it, so a
          dropped packet can never contribute to the verdict;

          and the cooldown has expired.

        last_recovery_time is stamped by the caller and only when this returns
        True, which is the other half of the old bug: it used to be stamped on
        every false positive, so the car recovered on a flat 5 s metronome --
        the journal shows STUCK at :47, :52, :57, :02, :07. The clock can only
        be restamped by a stick the car actually confirmed now, and confirming
        one costs a second of commanded throttle plus a second of unbroken
        stillness.
        """
        evidence = self.motion_evidence(imu)

        armed = (self._throttle_since is not None
                 and now - self._throttle_since >= STUCK_ARM_SECONDS)

        if evidence == MOVING or not armed:
            self._still_samples = 0
            return False

        if evidence == STILL:
            self._still_samples += 1
        # UNKNOWN falls through on purpose: no evidence is not evidence of no
        # motion, and it must not be able to walk the counter up on its own.

        if self._still_samples < STUCK_CONFIRM_SAMPLES:
            return False

        if now - self.last_recovery_time <= self.recovery_cooldown:
            return False

        self._still_samples = 0
        return True

    def get_steering_bias(self, depth, region_width=100, region_height=80, threshold_mm=1200):
        """
        Compare average depth in left vs right side.
        Return a steering value from -1 (steer left) to 1 (steer right).
        """
        h, w = depth.shape
        top = h // 2 - region_height // 2
        bottom = h // 2 + region_height // 2

        # Regions left and right of center
        left_region = depth[top:bottom, w//4 - region_width//2:w//4 + region_width//2]
        right_region = depth[top:bottom, 3*w//4 - region_width//2:3*w//4 + region_width//2]

        left_valid = left_region[left_region > 0]
        right_valid = right_region[right_region > 0]

        # Avoid divide-by-zero
        if len(left_valid) == 0 or len(right_valid) == 0:
            return 0  # neutral

        left_avg = np.mean(left_valid)
        right_avg = np.mean(right_valid)

        # Normalize difference to [-1, 1]
        delta = right_avg - left_avg
        max_range = threshold_mm
        steering = np.clip(delta / max_range, -1.0, 1.0)
        return steering

    def obstacle_in_front(self, depth_frame, threshold_mm=CLEARANCE_MM):
        """Is any part of the footprint inside its clearance budget?

        Fails closed, same as it always did, and for the same reason: on a
        stereo pair a no-return usually means too close, too dark or too
        featureless, and all three are reasons to stop. Dropping the zero
        pixels and then taking a mean used to hand np.mean() an empty array,
        which is nan, and nan > 0.5 is False -- a patch with nothing
        measurable in it read as "clear" and the car drove straight at it.

        What changed is what gets asked. It used to be one 50x50 patch at the
        dead centre, 0.98% of the frame and 45 mm square out in the world on a
        267 mm wide car. It is now PROBE_COLUMNS probes tiling the real
        footprint, each reaching from a little above the lens down toward the
        near ground, each with its own budget and its own fail-closed test.
        See the block above the class for the derivation.

        The scan is kept on the Driver rather than printed or published from
        here, so telemetry can report the column distances, the validity and
        the reason without a second read of the frame.
        """
        started = time.perf_counter()
        scan = free_by_column(depth_frame, threshold_mm)
        self.last_scan = scan
        self.last_scan_ms = (time.perf_counter() - started) * 1000.0

        near = bool(np.any(scan.blocked & np.array(
            [r == NEAR for r in scan.reason])))
        no_return = bool(np.any(scan.blocked & np.array(
            [r == NO_RETURN for r in scan.reason])))

        # Both reasons reported when both fired. They are two different
        # behaviours and a log that calls them the same thing is a log that
        # cannot tell me which one I am looking at.
        if near and no_return:
            self.last_block_reason = NEAR + "+" + NO_RETURN
        elif near:
            self.last_block_reason = NEAR
        elif no_return:
            self.last_block_reason = NO_RETURN
        else:
            self.last_block_reason = CLEAR

        return bool(scan.blocked.any())

    def begin_recover(self, now):
        """Start a backup-and-turn. Commands the first phase and returns."""
        print("[Driver] reversing and turning")

        # Pick a random direction, and a random amount of it. The old version
        # chose between max_turn/4 and max_turn -- two magnitudes, both
        # positive -- so the car always pivoted the same way and a corner that
        # beat it beat it forever. Both signs now.
        self._turn_dir = random.choice([
            -self.max_turn,
            -self.max_turn / 4,
            self.max_turn / 4,
            self.max_turn,
        ])
        self.controller.set_steering(self._turn_dir, -self._turn_dir)
        self.controller.set_throttle(self.reverse_speed)

        self._phase = _PHASE_REVERSE
        self._phase_until = now + REVERSE_SECONDS
        self._clear_motion_evidence()

    def recovering(self, now):
        """Advance the recover phases. True while recover owns the wheels."""
        if self._phase is None:
            return False

        if now < self._phase_until:
            return True

        if self._phase == _PHASE_REVERSE:
            self.controller.set_throttle(0)
            self._phase = _PHASE_SETTLE
            self._phase_until = now + SETTLE_SECONDS
            return True

        # Settle is up: unwind the steering and hand the wheels back.
        self.controller.set_steering(-self._turn_dir, 0)
        self._phase = None
        self._phase_until = 0.0
        return False

    def abort(self):
        """Give up on a recover in progress and stop. Safe to call twice."""
        self._phase = None
        self._phase_until = 0.0
        self.controller.set_throttle(0)
        self._clear_motion_evidence()

    def tick(self, now=None):
        """One pass of autonomy. Never blocks, never sleeps."""
        if now is None:
            now = time.monotonic()

        if self.recovering(now):
            return

        depth = self.camera.get_depth_frame()
        if depth is None:
            # None is a real return value: no packet waiting, or the X_LINK
            # error path. Without depth there is nothing to decide on, and the
            # only safe command is to stop -- pigpio holds the last pulse
            # width forever, so "do nothing" means "keep driving".
            self.controller.set_throttle(0)
            # Throttle is down, so the stuck check has nothing to measure and
            # whatever it had measured is about to be stale.
            self._clear_motion_evidence()
            return

        imu = self.camera.get_imu_sample()

        # Decide first, command second. The old order commanded forward speed
        # and only then checked for an obstacle, so every tick facing a wall
        # lurched into it for a tick before reversing.
        if self.obstacle_in_front(depth):
            # Which probes and why. "near" and "no_return" are two different
            # events -- a measured obstacle and a stereo pair that cannot see
            # -- and the old single line could not tell me which I was reading.
            where = "".join(
                "N" if r == NEAR else "?" if r == NO_RETURN else "."
                for r in self.last_scan.reason)
            print(f"Obstacle detected ({self.last_block_reason}) "
                  f"[{where}] - backing up.")
            self.begin_recover(now)
            return

        if self.stuck(imu, now):
            self.last_recovery_time = now
            print("STUCK - backing up to recover.")
            self.begin_recover(now)
            return

        self.controller.set_steering(self.get_steering_bias(depth), 0)
        self.controller.set_throttle(self.forward_speed)
        if self._throttle_since is None:
            self._throttle_since = now
