"""Depth-driven roomba mode: crawl forward, steer around an edge, back out of
the rest.

One decision per tick, taken before anything is commanded, off one read of
the depth frame. read_the_gradient() turns a footprint scan into one of three
answers -- drive on, steer around it, or back out -- and it is pure, so the
whole decision is testable on a laptop with no car attached. Everything with
a memory in it is on the Driver: the recover phase machine, the stuck
evidence, and one latched evade direction.

The steering half of that is deliberately narrow. Every probe inside the
400 mm budget used to mean a reverse; now a small block -- off one edge or
in the middle of the fan -- steers away and keeps driving when the car's own
full-lock sweep says there is room, and the seven gates that define that are
in the block above the class. Everything else reverses.

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

The reverse phase reads the IMU as it goes. Nothing on this car points
backwards, so backing up is blind -- but the gyro is a contact sensor: a car
commanded in reverse that has stopped shaking has hit something, and the
reverse can end there instead of pushing for the rest of its two seconds.
"""

import math
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

# How this car steers, in one place.
#
# It is four-wheel steered -- FRONT_STEERING_CHANNEL 15 and
# REAR_STEERING_CHANNEL 14, both of them real servos -- and until now only the
# recover path used the second one. Ordinary forward driving commanded
# set_steering(bias, 0) and hard-centred the rear on every tick, so the car
# steered one way when it was driving and another way when it was backing out
# of a corner. Indoors that showed up as corners it could not take: a tight
# hallway turn needs a radius the front axle alone does not have.
#
# Counter-phase, which is what the negative sign is. The rear wheels turn
# opposite the front ones, both axles contribute yaw, and the turning radius
# comes down. In-phase (both axles the same way) is a crab -- it slides the
# body sideways without rotating it -- which is a different and occasionally
# useful thing, but nothing on this car senses a reason to pick it, so there
# is no mode here to pick.
#
# 1.0 is full counter-phase, the tightest radius this linkage can give, and
# it is the ratio begin_recover() has always commanded -- it sets
# (turn_dir, -turn_dir). So this is not a new behaviour so much as the one
# behaviour, now used everywhere a steering angle gets chosen instead of only
# in recovery. Tightest is the ask; the feel is mine to judge on the floor and
# this is the one number to turn.
REAR_STEER_RATIO = 1.0


def steering_pair(front):
    """The (front, rear) pair to command for one chosen front angle.

    Every steering decision in auto goes through here, so "how this car
    steers" is defined once rather than re-decided at each call site. The
    hand-back at the end of a recover does not: that commands a literal
    (0, 0), because centre is centre and routing a zero through a ratio is
    just a way to put a nonzero number on a servo by accident.

    The rear magnitude can never exceed the front's. That is a clamp, not a
    scale -- the front is handed back exactly as it came in. max_turn = 0.6 is
    my limit on holding a servo at lock against an obstacle, and it is the
    front that is bounded by it; capping the rear at the front means whatever
    bound the front is under, the rear is under it too, and a ratio raised
    above 1.0 can lean on the front but can never push the rear past it.

    No numpy, no allocation beyond the tuple, five float operations. Measured
    on the car's own interpreter: 1.43 us a call against 0.22 us for the
    (front, 0) literal it replaced, so 1.2 us a tick. The footprint scan in
    the same tick measures 1.88-2.34 ms, which is more than a thousand times
    that, and the tick itself has about 13 ms. It is not a cost, but it is
    measured rather than assumed, because the last estimate in this loop came
    in 5-9x light.

    A nan or an infinity in gets the same treatment it would have got
    straight: it comes out the other side and set_steering() rejects it and
    commands neutral on both channels. That check lives at the one place that
    writes the hardware and it is not duplicated here.
    """
    rear = -REAR_STEER_RATIO * front
    limit = abs(front)
    return front, max(-limit, min(limit, rear))

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
# The rolling end, measured on 2026-10-27 off the /imu recording of a real auto
# run: forward_speed 0.2 on a hard floor, armed (forward throttle for at least
# STUCK_ARM_SECONDS), 772 windows of the same five-sample mean the driver takes:
#
#   gyro mean while rolling    p1 0.0146   p5 0.0173   p50 0.0356 rad/s
#   accel mean while rolling   p1 0.106    p5 0.125    p50 0.188  m/s^2
#
# This was 0.035, five times the parked maximum, set before anyone had seen the
# rolling number -- and the rolling median turned out to be 0.0356. Half of
# every second of honest creeping read STILL on the gyro, the accel term at
# stuck_threshold 0.2 sits on its own rolling median too, so 27% of armed
# windows read STILL on both and ten of them in a row was a matter of time.
# Four STUCK reverses in that run, three of them marked BAD, and in all four
# the car was moving: the depth frames were changing at the same rate as a
# known straight approach (frame-to-frame median 112-165 mm/s against 0 parked
# and 166 approaching a wall), and the two seconds before each one never had a
# window under 0.0142 rad/s.
#
# 0.010 is the geometric middle of the parked maximum (0.0069) and the rolling
# p1 (0.0146), about nine gyro steps against the parked floor of six. Replayed
# over that run nothing armed reads STILL at all. What has never been recorded
# is a genuinely stalled car -- wheels against something, ESC pushing at 0.2 --
# so whether a stall reads under 0.010 or buzzes above it is NOT measured.
# That is a test to do with a hand on the car.
#
# Every number above was measured at forward_speed 0.2. Forward is 0.4 now
# (FORWARD_SPEED), twice the throttle and four times the pulse offset off
# neutral, so these 0.2-era numbers say very little about it and 0.010 has
# not been checked against it. A faster car should shake the chassis more
# while rolling, which moves the rolling p1 up and away from this threshold
# rather than toward it, but that is a guess -- and a stall at 0.4 pushes a
# lot harder and may buzz above it. UNVERIFIED at 0.4.
GYRO_MOVING_MIN = 0.010  # rad/s

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

# How many consecutive still samples end a reverse early. The gyro as a rear
# bumper: there is no camera pointed backwards, so reversing is blind, but
# there is a contact sensor on the car already. If I have commanded reverse
# and the chassis is not shaking, I have backed into something.
#
# Deliberately much shorter than STUCK_CONFIRM_SAMPLES = 10. Reversing into a
# wall wants a fast answer -- every tick spent pushing against it is a tick
# not spent getting out -- and the two failures are not symmetric: a false
# positive here costs me a recover that ended early, while a false positive on
# the forward detector costs me a reverse I did not ask for.
#
# Four is not the whole delay and that is the point. A sample only counts once
# the smoothing window is full, and begin_recover() empties it, so the first
# five ticks of a reverse are spent filling it -- half a second, during which
# the ESC has spun up and the car is either moving or it is not. Four samples
# after that puts the verdict 0.9 s into a 2.0 s reverse: 0.5 s of smoothing
# plus 0.4 s of confirmation, eight distinct IMU readings all agreeing. Going
# to five would buy 10% more confidence for half of the time saved.
#
# A count of ticks and not a deadline, for the same reason STUCK_CONFIRM_SAMPLES
# is: the loop does not always make its rate, and a count can only ever take
# longer in real time, never fire early.
REVERSE_STALL_SAMPLES = 4


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
# from it now. Worth knowing where it sits: oakd.py runs extended disparity
# with subpixel off, which gives this stereo pair 190 disparity levels and a
# minimum range of
# fx * baseline / disparity_levels = 441.25 * 75 / 190 = 174 mm, so the whole
# band from 400 mm down to 174 is measurable -- 226 mm of graded warning
# before anything falls into the blind zone.
#
# That is the opposite of how this read when the pair was on subpixel and 95
# levels. 348 mm of minimum range left about 52 mm of measurable approach, so
# most triggers were expected to be the no-return kind: something arriving
# already too close to measure. With 226 mm of room the usual trigger should
# now be a measured NEAR distance, and a NO_RETURN means what it says --
# dark, featureless or outside the overlap -- rather than just "near". The two
# are still reported apart because they still mean different things; it is the
# expected mix that has moved.
#
# Both those figures are the arithmetic, and the arithmetic is a little
# pessimistic at the near end: measured on the car the old config returned
# depth down to 319 mm rather than stopping dead at 348, and the new one down
# to 165 rather than 174, because the post-processing filters interpolate past
# the edge of the search range. The shape of the argument survives it -- the
# measurable approach went from about 80 mm to about 235 mm -- but do not read
# 174 as a cliff.
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
# There was one complication hanging over these two numbers, and it is closed.
# oakd.py calls stereo.setDepthAlign(dai.CameraBoardSocket.RGB) while there is
# no ColorCamera node in the pipeline. The EEPROM carries RGB calibration
# whether or not that node exists, so the align does take effect, and the fear
# was that depth therefore lands in the IMX378's frame rather than the mono
# pair's -- 69 degree HFOV, and a 4:3 sensor squeezed into a 1.6 output. Worst
# case that is fx = 466 (5.5% high) and fy = 388 (12% low), which would make
# the two numbers below wrong by a few percent and every millimetre in this
# file suspect with them.
#
# They are not wrong. Measured on the car against a tape measure, flat box
# target, 300 frames per config at a healthy 16.08 fps with every queue
# drained:
#
#   target at 203.2 mm (8 in)     depth read 207.0 mm
#   target at 1041.4 mm (41 in)   depth read 1048.0 mm   align on, as deployed
#                                 depth read 1051.0 mm   align off
#
# Those two distances are 5x apart, which is what makes offset and scale
# separable. Fit them jointly for a constant offset plus a scale factor and
# they come out at +3.1 mm of offset and +0.33% of scale. The offset is the
# optical centre sitting a few millimetres behind the housing face I held the
# tape against, so it is me measuring from the wrong datum rather than an
# error in the camera at all, and a third of a percent of scale is inside the
# sensor's own spec and far below anything the thresholds in this file can
# resolve.
#
# So 441.25 is correct, the align does not meaningfully warp depth, and the
# few-percent error feared above does not exist. Do not spend the afternoon
# re-deriving it, and do not treat the millimetre figures anywhere in this
# file as suspect on account of it. An earlier bench run had the align
# shifting a centre-ROI median by 19.1 mm, 3.7% of it; that did not reproduce
# against the tape and reads as an artifact of the one scene it was measured
# on. What did reproduce is the valid-pixel density the align buys, which is
# the actual reason it stays, and that is written up next to the call in
# oakd.py.
#
# The probe margins below were originally sized to survive fy = 388. They keep
# that slack, because slack there only ever costs me gaps I would have
# squeezed through -- not because the error is still on the table.
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


def probe_marks(scan) -> str:
    """One scan as a readable string: N near, ? no-return, . clear.

    The format the journal has carried since the footprint check
    shipped, defined once now that two places want to print it.
    """
    return "".join("N" if r == NEAR else "?" if r == NO_RETURN else "."
                   for r in scan.reason)


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


# ---------------------------------------------------------------------------
# Reading the gradient: the one near obstacle that gets steered around
# ---------------------------------------------------------------------------
#
# Until now every probe inside the clearance budget meant the same thing:
# stop, reverse for two seconds, turn a random way, try again. That is the
# right answer to a wall and the wrong answer to the thing this car actually
# meets indoors, which is a door jamb or a skirting board catching one edge of
# the fan while the other eight probes read open floor. 88 obstacle events
# read out of a week of the journal, and 37 of them are a single edge probe or
# a pair of them -- [........N] 15 times, [N........] 12, [NN.......] 7,
# [.......NN] 3.
#
# So a small block now steers instead of reversing: entirely measured,
# contiguous, at most MAX_BLOCKED_PROBES wide, not spanning the fan, with
# OPEN_RATIO_MIN times as much room on the side being turned toward, nothing
# closer than REVERSE_FLOOR_MM, and every measured point in the fan outside the
# circle the car sweeps at full lock. Everything else reverses.
#
# That used to be one edge only. A block that touched neither edge -- the
# journal's "middle" -- reversed on the argument that steering away from it
# meant steering into whatever was beside it. The run on 2026-10-27 says that
# was the wrong call for this car in this house: 20 reversals in six minutes,
# and the shapes it backed away from were [...N.....], [.N.......] and
# [...NN....], single chair-leg sized things with eight probes of open floor
# around them. The question was never "is there a gap to thread", it is "is
# there room to turn away", and the probes beside the block answer that
# directly: they are measured, and they are clear past the budget. So a middle
# block now turns toward the side with more open probes, and the sweep check
# below decides whether there is room to.
#
# Still a narrowing of begin_recover()'s domain and not a behaviour alongside
# it. Seven gates in order, every one of them a reason to reverse, and the
# evade is only what falls out of the bottom -- so the safety argument reads
# one condition at a time, which is why the rules are numbered in the code as
# well as here. Wide (3+), split and spanning blocks reverse exactly as before.

# The hard floor: inside this nothing steers, whatever the pattern.
#
# 360 until 2026-10-27, which was a deliberately conservative stand-in for a
# turn radius nobody had measured. Two things moved. The camera now measures
# cleanly down to about 165 mm (VERIFIED against a tape at 203 mm, read 207),
# so a reading at 200-360 is a real distance and not the edge of the blind
# zone. And the question of how close is too close to turn away is now asked
# per scene, against the car's own sweep -- see TURN_RADIUS_MM -- rather than
# answered once with a number that had to cover the worst pattern. With that
# check in place a single floor only has to cover what the sweep model cannot:
# readings so close that the frame age and the servo slew are most of the
# distance. 200 mm is 35 mm above the measured near edge of the sensor and four
# times the TURN_LAG_MM allowance, and nothing below it steers.
#
# Of the run's eight "floor" reversals, the ones at 160, 226 and 247 mm still
# reverse (the first on this floor, the other two on the sweep check). This
# number is not the knob for those; TURN_RADIUS_MM is.
#
# At forward 0.4 the margin is thinner, and I have NOT changed the number.
# Ground speed at 0.4 is ESTIMATED at 330-420 mm/s, twice what 0.2 measured.
# The band this floor can see -- from 200 down to the sensor's ~165 mm near
# edge -- is 35 mm, which was about 1.7 ticks of travel at 0.2 and is about
# one tick (33-42 mm) at 0.4, so a forward decision on one tick can carry an
# obstacle straight past it. A picture is 88 ms old when it is decided on, and
# takeover's forward trigger accepts one up to FLOOR_SCAN_MAX_AGE 0.3 s old:
# 0.19-0.39 s between the picture and the next chance to stop, 60-80 mm at the
# common end and 130-165 mm at the worst, so something read at 201 mm can be
# inside 165 before anything looks again. The rules autopilot is still safe
# there because R1 reverses on any NO_RETURN, and too close reads NO_RETURN.
# Takeover's forward trigger and the policy runtime used to ask this floor
# about NEAR probes only, so inside 165 mm they stopped vetoing. They ask
# ForwardFloorVeto now, which holds them further out (FORWARD_FLOOR_MM) and
# keeps holding when a probe goes blind on the way in. This number is still
# the rules autopilot's R2, and the base the forward floor is built on.
REVERSE_FLOOR_MM = 200.0

# The full-lock turn, as one circle. ESTIMATED, off the 2026-10-27 recording,
# and the single number most worth measuring with a tape: drive a full-lock
# circle at forward_speed and halve its diameter.
#
# How it was estimated. Yaw is the IMU's gyro_x on this mounting (VERIFIED: it
# goes positive on a right steer under throttle, and it is the axis that
# carries 0.66-0.74 rad/s in a hand-driven right turn). During the two
# graze-steers in the run, max_turn counter-phase at forward_speed 0.2, it read
# 0.39 rad/s (20:10:44) and 0.32 rad/s (20:12:47) in the steady part of the
# turn. Ground speed at forward_speed 0.2 is about 210 mm/s, off the closing
# rate of three straight wall approaches in the depth stream (206, 210 and
# 217 mm/s on the centre probes). 210 / 0.355 = 590 mm. The spread on that is
# wide -- the turn may be slower than the straight line, and the two turns
# disagree by 20% -- so read it as 450-750 mm. Counter-phase rear steer is
# roughly half the radius front-only steering would give, so this is the
# radius of a car that would turn in about 1.2 m on its front axle alone.
#
# All of that was at forward_speed 0.2. Forward is 0.4 now; a turning circle
# is geometry and should not move much with speed, so this number stays, but
# the faster the car the more a slipping tyre widens it, and at twice the
# throttle that is no longer a small effect. The tape settles it, at 0.4.
TURN_RADIUS_MM = 590.0

# How far the camera sits ahead of the point the car pivots about, which on a
# symmetric counter-phase car is midway between the axles. ESTIMATED: nobody
# has measured the wheelbase. 160 mm is half a 1/10 crawler's ~320 mm
# wheelbase with the camera at the front axle, and if it is mounted ahead of
# the axle the real number is larger, which only ever makes the sweep check
# more conservative than it needs to be. Another number a tape settles.
CAMERA_AHEAD_MM = 160.0

# How much closer an obstacle gets before the turn has actually started.
# ESTIMATED: the frame behind a decision is 88 ms old (measured, see
# EVADE_SECONDS) and the servos take about 0.15 s to reach lock (yaw rises
# from nothing to the turn rate in two or three 20 Hz samples in the run),
# so about 0.24 s at 210 mm/s, which is 50 mm.
#
# 210 mm/s was ground speed at forward_speed 0.2 in an auto run (and the
# first record session closed on a wall at only 164 mm/s at the same cap).
# Ground speed at 0.4 is NOT measured, and at twice the throttle the 0.2-era
# speed is no basis for guessing it. The allowance is time times speed, so it
# grows with the car: at an ESTIMATED 330-420 mm/s (2x the 164-210 measured
# at 0.2) the same 0.24 s is 80-100 mm, so 50 is an UNDERestimate at 0.4 and
# the sweep check is less conservative than it reads. I have left it at 50
# until the speed at 0.4 is measured off a recording rather than guessed;
# 50 is also the number REVERSE_FLOOR_MM's "four times the allowance" was
# written against, and at 0.4 that ratio is about two.
TURN_LAG_MM = 50.0

# How much of the car is allowed to be in the thing being steered around.
# The probe boundaries land at these lateral offsets at the 400 mm budget,
# in mm: -153.2 -118.8 -85.2 -50.8 -17.2 +17.2 +50.8 +85.2 +118.8 +153.2.
# The outer face of a tyre is at +/-133.35, so one blocked edge probe reaches
# at most 14.5 mm inside the tyre line, two reach 48.2 mm -- about one tyre
# face -- and three reach 82 mm, which is most of the wheel. Two is the last
# width where steering away is still a graze rather than a hope.
#
# The same limit holds for a block in the middle of the fan, where two probes
# are about 68 mm of real width at the budget range -- a chair leg, a bin, a
# shoe. Three in the middle is a third of the fan and is the start of something
# the car is driving into rather than past, and it reverses as "wide".
MAX_BLOCKED_PROBES = 2

# How much more room the open side needs before the surface counts as shallow
# enough to rotate along rather than back away from.
#
# For a flat surface the ratio is a statement about its angle and nothing
# else: the fan's outer columns sit at tan = +/-0.383, so a plane at beta to
# the image plane reads (1 + 0.383 tan beta) / (1 - 0.383 tan beta) across it.
# On the planar test fixture at a 390 mm near edge, 20 degrees reads 1.28, 30
# reads 1.47 and 40 reads 1.76.
#
# 2.0 until 2026-10-27, which is 41 degrees and was chosen without a turn
# radius. Two marks from the run bracket the real answer, and both are oblique
# walls across the whole fan, nearest at one edge (probe distances off the
# /marks snapshot, VERIFIED):
#
#   BAD   [373 390 419 455 465 514 538 617 636]  ratio 1.71, about 38 degrees
#   GOOD  [497 481 466 446 432 420 409 395 380]  ratio 1.31, about 21 degrees
#
# I wanted the first one steered along and the second one backed out of. 1.5
# sits between them, and it is also about where the geometry puts it: a wall
# can be turned away from when its distance from the full-lock turn centre is
# more than the car's swept corner radius, and with TURN_RADIUS_MM at 590 and a
# centre-probe distance of 430-470 mm that cut lands at 26-31 degrees, a ratio
# of 1.45-1.55. ESTIMATED, because the radius is. The sweep check below asks
# the same question directly of every probe; this is the angle limit that holds
# whatever the radius turns out to be.
#
# One of the run's reversals, [........N] at 1.49, sits right on it and still
# reverses. Whether that one should have steered is a feel call.
#
# The ceiling no longer binds. distance_mm on an open probe is about 899 (its
# lower rows end on the floor), so the ratio tops out at 899 over the nearest
# reading still allowed to steer, which is 4.5 with the floor at 200 instead of
# 2.5 with it at 360.
OPEN_RATIO_MIN = 1.5

# Cruise steering: how much of max_turn a gradient with no obstacle in it is
# allowed to ask for, and how much asymmetry it takes before it asks at all.
#
# Cruise is a correction and not a turn -- the turns are evades and recovers
# -- so it gets half the bench limit. With the 899 clip the raw bias tops out
# at (899 - 400) / (899 + 400) = 0.384, which puts the hardest cruise command
# at 0.384 * 0.5 * 0.6 = 0.115, or 10 servo degrees.
#
# The deadband is on the raw bias, so 0.05 is a 5% left/right difference in
# room and commands 0.05 * 0.5 * 0.6 = 0.015, which is 1.3 servo degrees. That
# is the floor below which a correction is servo noise rather than a steer.
CRUISE_TURN_FRACTION = 0.5
CRUISE_DEADBAND = 0.05

# How long an evade is committed to once it starts, and the longest one
# allowed. The autopilot decides at 10 Hz -- AUTO_TICK_INTERVAL in crawler/__main__.py,
# not the camera's 16 fps and not the control loop's 100 Hz -- so 0.5 s is
# five decisions.
#
# 30 fps is what that used to say, and the camera has never delivered it. 30
# is the mono sensors' default shutter rate; what reaches this file is the
# stereo node's output rate, which was 12.25 fps measured and is now 16.0 --
# oakd.CAPTURE_FPS, which is a cap set deliberately under what the node can
# finish so that frames stop queueing inside the device. Either way the
# decision rate is the slower of the two and this constant is counted in
# decisions, so nothing below moved. What did change is how old the frame
# behind a decision is: 642 ms, now 88 ms, which is most of one decision
# interval returned.
#
# EVADE_SECONDS is there to hold the turn after the obstacle has left the fan.
# The fan is 338 px wide at the budget range and the thing being steered
# around is at its edge, so the moment the nose starts rotating the obstacle
# stops being measured, and a car that un-steers then has not gone anywhere.
# Five decisions is coarse; it is not lower for that reason.
#
# EVADE_LIMIT_SECONDS is the backstop for a surface that never ends: 30
# decisions of steering the same way and still being blocked is a wall being
# followed into a corner, and it reverses. ESTIMATED -- it wants a ground
# speed measurement off the car to pin properly.
EVADE_SECONDS = 0.5
EVADE_LIMIT_SECONDS = 3.0

# What tick() does about one scan.
ACT_FORWARD = "forward"
ACT_EVADE = "evade"
ACT_REVERSE = "reverse"


class SteerVerdict(NamedTuple):
    """One decision about one scan, taken before anything is commanded.

    action  ACT_FORWARD, ACT_EVADE or ACT_REVERSE.
    steer   the front angle as a FRACTION of max_turn, in [-1, 1], positive
            noses the car right. A fraction and not an angle on purpose: the
            one bench constant that bounds every auto steering command is
            then applied at the one place that commands it, and nothing
            decided in here can get around it.
    reason  a short phrase for the journal, and the thing _announce() keys
            its transitions off.
    """

    action: str
    steer: float
    reason: str


def cruise_bias(scan) -> float:
    """Which way to lean with nothing inside the clearance budget.

    The old get_steering_bias() took two 100x80 patches at the quarter
    columns, averaged every valid pixel in each, and divided the difference by
    a 1200 mm literal. Three things were wrong with it: it read a band this
    file now covers properly, it saturated at exactly 1.0 against a wall down
    one side, and the normaliser made the answer depend on how far away the
    scene was rather than on its shape.

    This is scale free by construction. For a plane the standoff distance is a
    common factor of both sides and cancels, so what comes out is the wall's
    ANGLE and not its distance -- which is the actual fix for the 1200 mm
    normaliser, rather than a smaller number in the same place.

    It reads the footprint scan that the obstacle check already ran, so it
    costs no second look at the frame, and it inherits the scan's fail-closed
    validity rules for free.

    Probe 4 is left out. It straddles the centreline, contributes to both
    sides equally and can only dilute the difference.

    Deliberately silent until something is inside FLOOR_CLEAR_MM. Every probe
    terminates on the floor inside 899 mm, so open floor reads 899 across the
    fan and the bias is exactly zero -- this says nothing at all about an
    empty room, which is correct and is not a bug to be fixed later. What it
    does is lean away from a wall that is already in sight but not yet inside
    the trigger, and that mild pre-turn raises the angle the wall is read at,
    which is what moves a scene from the reverse band into the evade band.
    """
    room = np.minimum(scan.distance_mm, FLOOR_CLEAR_MM)
    left = float(room[:4].mean())
    right = float(room[5:].mean())
    total = left + right
    if not total > 0.0:
        # Unreachable from tick(): cruise only runs on a scan with nothing
        # blocked, and a clear probe has a finite distance of at least the
        # trigger range. It is here so the function is total, because the
        # caller is the one thread that commands the ESC.
        return 0.0
    bias = (right - left) / total
    return 0.0 if abs(bias) < CRUISE_DEADBAND else bias * CRUISE_TURN_FRACTION


def _turn_side_mm(scan, i, side, distance):
    """Where probe i's edge nearest the turn sits, in mm, at `distance`.

    Measured along the direction being turned toward: positive is on the turn
    side of the centreline, negative is on the side being turned away from.
    The probe boundaries are rays from the lens, so the lateral offset of a
    boundary scales with the distance it is read at, and the edge nearest the
    turn is the worst case for anything inside that probe -- it is the point
    the car's swinging corner gets to first.
    """
    geo = scan.geometry
    edges = geo.col_edges
    centre = (edges[0] + edges[-1]) / 2.0
    edge = edges[i + 1] if side > 0 else edges[i]
    return side * (edge - centre) / geo.fx * distance


def turn_sweep_clears(scan, side) -> bool:
    """Can a full-lock turn toward `side` begin now without touching anything
    the fan can see? Pure, closed form, nine multiply-adds.

    A counter-phase car at constant lock rotates about one fixed point: a
    distance TURN_RADIUS_MM to the turn side of its pivot, which is
    CAMERA_AHEAD_MM behind the lens. Every part of the car stays inside the
    circle its outer front corner traces about that point, and the corner gets
    to anything ahead of it before the rest of the body does. So a point the
    camera measured is safe to turn away from exactly when it lies outside that
    circle, once TURN_LAG_MM of approach is taken off for the frame age and the
    servo slew:

        (z + a)^2 + (R - p)^2  >  a^2 + (R + H)^2

    with z the distance less the lag, p the point's lateral offset toward the
    turn, a = CAMERA_AHEAD_MM, R = TURN_RADIUS_MM and H the half width
    including FOOTPRINT_MARGIN_MM. Every probe is asked, blocked or clear, at
    its turn-side edge -- so a wall that recedes into the clear probes is
    tested along its whole visible length, not just at its nearest point.

    Conservative, deliberately: it assumes the turn is held for as long as it
    takes, where the real evade straightens out once the fan is clear. At the
    estimated 590 mm radius this puts the closest a thing can be and still be
    turned away from at about 257 mm for one edge probe, 284 for two, 316 for
    a block two probes in, 351 for three in and 390 for one on the centreline
    -- which leaves a centred block 10 mm of the trigger band, so in practice
    it reverses. All of those move with TURN_RADIUS_MM, which is the point.
    """
    a = CAMERA_AHEAD_MM
    r = TURN_RADIUS_MM
    swept = a * a + (r + HALF_TRACK_MM + FOOTPRINT_MARGIN_MM) ** 2
    for i in range(PROBE_COLUMNS):
        distance = float(scan.distance_mm[i])
        if not np.isfinite(distance):
            # Only reachable on a NO_RETURN probe, which R1 has already
            # reversed on. A point with no distance cannot be shown clear.
            return False
        z = distance - TURN_LAG_MM
        if z <= 0.0:
            return False
        p = _turn_side_mm(scan, i, side, distance)
        if (z + a) ** 2 + (r - p) ** 2 <= swept:
            return False
    return True


def _roomier_side(scan, blocked) -> float:
    """Which way to turn away from a block that touches neither edge.

    More open probes first: they are measured, they are past the budget, and
    they are where the car is going to be pointed. A tie -- only a single probe
    dead on the centreline can make one -- goes to whichever side has more
    measured room, each side's probes capped at FLOOR_CLEAR_MM so a column that
    saw past the floor cannot win it. An exact tie on that too is 0.0, which
    the caller reverses on rather than picking a side by coin.
    """
    left_open = blocked[0]
    right_open = PROBE_COLUMNS - 1 - blocked[-1]
    if left_open != right_open:
        return 1.0 if right_open > left_open else -1.0
    room = np.minimum(scan.distance_mm, FLOOR_CLEAR_MM)
    left = float(room[:blocked[0]].mean())
    right = float(room[blocked[-1] + 1:].mean())
    if right == left:
        return 0.0
    return 1.0 if right > left else -1.0


def read_the_gradient(scan) -> SteerVerdict:
    """What to do about one footprint scan. Pure: a scan in, a verdict out.

    Seven gates, in order, and every one of them reverses. The evade is what
    comes out of the bottom when all seven are satisfied, so this reads as the
    old "anything inside 400 mm reverses" with its domain narrowed rather than
    as a second behaviour competing with it.

    R1 is first and has to stay first. A NO_RETURN probe is a column the
    stereo pair cannot measure -- too close, too dark or too featureless --
    and such a column can never be a direction to turn toward, can never
    contribute to the open side of the ratio and can never be counted in the
    width. One anywhere in the fan vetoes the whole feature.

    R2 is the hard floor and R7 is the per-scene one. An obstacle being steered
    around that keeps closing fails R7 first, while it is still perfectly
    measurable, and the next scan reverses; R2 is what is left for readings
    so close that the lag is most of the distance.
    """
    blocked = [i for i in range(PROBE_COLUMNS) if scan.blocked[i]]

    # R1. Anything unmeasurable anywhere vetoes the whole feature.
    if any(r == NO_RETURN for r in scan.reason):
        return SteerVerdict(ACT_REVERSE, 0.0, "no_return")

    if not blocked:
        return SteerVerdict(ACT_FORWARD, cruise_bias(scan), "clear")

    near = min(float(scan.distance_mm[i]) for i in blocked)

    # R2. Too close to steer, whatever the shape.
    if near < REVERSE_FLOOR_MM:
        return SteerVerdict(ACT_REVERSE, 0.0, f"floor {near:.0f}")

    # R3. Two separate obstacles is not a surface.
    if blocked != list(range(blocked[0], blocked[-1] + 1)):
        return SteerVerdict(ACT_REVERSE, 0.0, "split")

    # R4. Touching both edges is a wall across the whole car, and there is no
    # open side to turn toward.
    at_left = blocked[0] == 0
    at_right = blocked[-1] == PROBE_COLUMNS - 1
    if at_left and at_right:
        return SteerVerdict(ACT_REVERSE, 0.0, "spans")

    # R5. How much of the car may be in it.
    if len(blocked) > MAX_BLOCKED_PROBES:
        return SteerVerdict(ACT_REVERSE, 0.0, f"wide {len(blocked)}")

    # Which way. Away from an edge it touches; otherwise toward the roomier
    # side, and a block with no roomier side reverses.
    if at_left:
        side = 1.0
    elif at_right:
        side = -1.0
    else:
        side = _roomier_side(scan, blocked)
        if not side:
            return SteerVerdict(ACT_REVERSE, 0.0, "middle")
    turn_side = (range(blocked[-1] + 1, PROBE_COLUMNS) if side > 0
                 else range(0, blocked[0]))

    # R6. Shallow enough to rotate along? Asked of the side being turned
    # toward, which for an edge block is every open probe and for a middle
    # block is the half it is heading into. Capped at the same FLOOR_CLEAR_MM
    # the probes themselves terminate at, so the ratio cannot be inflated by a
    # column that happened to see past the floor, and an infinity cannot reach
    # the division. Every probe in turn_side is CLEAR -- R1 has excluded
    # NO_RETURN and the block is contiguous -- so the max has something in it.
    open_mm = min(FLOOR_CLEAR_MM,
                  max(float(scan.distance_mm[i]) for i in turn_side))
    ratio = open_mm / near
    if ratio < OPEN_RATIO_MIN:
        return SteerVerdict(ACT_REVERSE, 0.0, f"shallow {ratio:.2f}")

    # R7. Room to actually turn away, off the car's own sweep.
    if not turn_sweep_clears(scan, side):
        return SteerVerdict(ACT_REVERSE, 0.0, f"tight {near:.0f}")

    # At the full fraction. Nothing in between: the magnitude is not a
    # function of the ratio, so it cannot dither, and the one number that
    # bounds it is max_turn at the call site. "graze" off an edge, "dodge"
    # around something in the middle, so the journal says which.
    kind = "graze" if (at_left or at_right) else "dodge"
    return SteerVerdict(ACT_EVADE, side, f"{kind} {ratio:.2f}")


# The speed caps. Auto drives at exactly these, record scales full RT and
# full LT to them (main.auto_speed_limits), takeover's trigger is scaled to
# them, and the policy runtime multiplies its magnitude by them.
#
# Forward was 0.2, 1660 us against neutral 1575. The first record session
# (2026-10-27 22:18) says that sits on the motor's start threshold: RT was
# at full for 638 of 645 forward ticks, the one clean straight approach
# closed at 164 mm/s, and two other straight full-RT stretches closed at 4
# and 19 mm/s -- a car with my whole finger on the trigger, barely moving.
# 0.4 is 1745 us, +170 us off neutral: twice the throttle and twice the
# pulse offset of what I drove that session.
FORWARD_SPEED = 0.4

# Reverse was -0.6, 1230 us, -345 us off neutral. Then it was derived to
# sit the same pulse offset off neutral as forward: -0.2957 at forward 0.4,
# 1405 us, -170 us. On the floor that was too slow. Reverse barely moved,
# and with the wheels turned it did not move at all. This ESC needs a much
# bigger pulse to start in reverse than in forward, so an equal offset is
# not an equal speed here.
#
# -1.0 is my call: the end of the range, 1000 us, -575 us off neutral, full
# reverse. set_throttle clamps to [-1, 1], so there is nothing past it.
REVERSE_SPEED = -1.0


# ---------------------------------------------------------------------------
# The forward floor: the veto on a forward the rules autopilot did not choose
# ---------------------------------------------------------------------------
#
# Two things drive this car forward in auto without going through
# read_the_gradient(): my trigger under takeover (main.AutoPilot) and the
# policy runtime (policy.PolicyDriver). Both used to ask one scan whether a
# NEAR probe was under REVERSE_FLOOR_MM, and at forward 0.4 that had two holes.
# Inside about 165 mm the stereo pair measures nothing, so a probe read
# NO_RETURN and the veto let go exactly when the car was closest. And the
# 200-165 band is about one tick of travel at 0.4, against a picture that can
# be 0.39 s old by the time anything looks again. ForwardFloorVeto closes
# both: the floor is pushed out by the travel during the worst-case picture
# age, and a probe that goes blind on the way in stays a veto. The rules
# autopilot never asks it; its R1 and R2 are what they were.

# Ground speed per unit of forward throttle. ESTIMATED, a fit through zero
# off the only two measurements there are, both at forward_speed 0.2: 210
# mm/s off three straight wall approaches in an auto run (VERIFIED, see
# TURN_LAG_MM) and 164 mm/s off the one clean approach in the first record
# session. The fast one, because a car that is faster than assumed is the one
# that overruns the floor. At 0.4 that is 420 mm/s, the top of the 330-420
# guessed for it. Nothing has measured 0.4. A motor with a start threshold --
# and 0.2 sat on this one's -- rises faster than a line through zero above
# it, so 420 may be low; drive a recorded straight approach at 0.4 and
# replace this. Forward only: reverse is not this line (see REVERSE_SPEED).
MM_S_PER_THROTTLE = 210.0 / 0.2

# How old a picture is when the next chance to stop comes round, at worst.
# The frame behind a decision is 88 ms old (measured, see EVADE_SECONDS), and
# takeover lets my trigger through on a scan up to FLOOR_SCAN_MAX_AGE old.
# The policy decides on a fresh frame every tick and so sees at most 0.19 s
# of this; one number covers both callers, and it is the takeover's.
FRAME_AGE_S = 0.088

# How old the last scan may be and still clear my forward throttle past the
# floor veto. Three decision ticks. A recover makes no scans, so for most of
# one my forward trigger is held at 0 -- the camera has not been asked, and
# no picture is not a clear picture. Reverse is never held. (crawler/__main__.py reads it
# from here: the floor is built on it.)
FLOOR_SCAN_MAX_AGE = 0.3
WORST_PICTURE_AGE_S = FRAME_AGE_S + FLOOR_SCAN_MAX_AGE

# How far back a blind probe looks for the approach that would explain it.
# One second is ten scans and 420 mm of travel at 0.4, which is the whole
# NEAR band (under CLEARANCE_MM) with room to spare: past a second, nothing
# the probe measured says anything about where it is now.
FLOOR_LATCH_SECONDS = 1.0

# A latch also lets go once the car has backed off this far. A latched probe
# was at worst touching something; backed off REVERSE_FLOOR_MM, which is 35 mm
# past the ~165 mm near edge of the sensor, the thing is inside the band the
# camera measures, so a probe still reading NO_RETURN there is a dark or
# featureless surface and not a near one. The usual release is sooner and
# needs no estimate -- the probe reads a distance again as soon as the car
# backs out of the blind zone. This one is for the probe that stays blind.
FLOOR_RELEASE_REVERSE_MM = REVERSE_FLOOR_MM

# How a reverse is counted toward that, and it is deliberately crude, because
# reverse ground speed has never been measured at any pulse. What is known:
# at -0.296 (170 us) reverse barely moved, and with the wheels turned not at
# all, so this ESC's reverse starts much later than its forward and no
# forward fit says anything about it. So a reverse counts only at or past
# REVERSE_CREDIT_MIN, -0.5 (287 us, 1.7 times the pulse that barely moved),
# and then only at REVERSE_CREDIT_MM_S, 164 mm/s, the slowest clean ground
# speed this car has been measured at in either direction. At the -1.0 cap
# that is ESTIMATED 1.22 s of backing up, about twelve ticks -- inside the
# recover's own reverse. Both numbers err toward holding the latch longer.
REVERSE_CREDIT_MIN = -0.5
REVERSE_CREDIT_MM_S = 164.0


def ground_speed_mm_s(throttle) -> float:
    """ESTIMATED forward ground speed for a forward throttle, mm/s. Anything
    that is not a finite forward is 0: reverse is floor_travel_mm's."""
    try:
        t = float(throttle)
    except Exception:
        return 0.0
    if not math.isfinite(t) or t <= 0.0:
        return 0.0
    return MM_S_PER_THROTTLE * min(t, 1.0)


def forward_floor_mm(forward_speed=None) -> float:
    """Where a forward the rules did not choose is held, for this cap.

    REVERSE_FLOOR_MM plus how far the car goes at `forward_speed` while the
    picture ages WORST_PICTURE_AGE_S, so something read just outside it is
    still outside REVERSE_FLOOR_MM when the veto next looks. None is the
    module's FORWARD_SPEED, read when called.
    """
    if forward_speed is None:
        forward_speed = FORWARD_SPEED
    speed = ground_speed_mm_s(max(0.0, float(forward_speed)))
    return REVERSE_FLOOR_MM + speed * WORST_PICTURE_AGE_S


# 200 + 420 x 0.388 = 363 mm at forward 0.4. Under CLEARANCE_MM, as it has to
# be: a NEAR reading only exists inside the clearance budget, so a floor past
# 400 would be a floor nothing can trip.
FORWARD_FLOOR_MM = forward_floor_mm(FORWARD_SPEED)


class FloorLatch(NamedTuple):
    """What the forward veto remembers. Immutable; the functions below make
    the next one.

    history      ((time, near_mm per probe), ...) for the last
                 FLOOR_LATCH_SECONDS of scans, inf where a probe was not NEAR
    latched      per probe: blind, and blind because it got too close
    reversed_mm  ESTIMATED backing-off since a probe latched
    """

    history: tuple
    latched: tuple
    reversed_mm: float


EMPTY_FLOOR_LATCH = FloorLatch((), (), 0.0)


def _blind_from_approach(history, i, now, speed) -> bool:
    """Could probe i, or a probe beside it, have closed into the blind zone?

    A NEAR reading at d mm, `age` seconds before this scan, explains a probe
    going blind now if the car could have covered d - REVERSE_FLOOR_MM since:
    d under REVERSE_FLOOR_MM + speed x age. Never less than the forward floor
    itself, so a probe that was holding the veto keeps holding it when it goes
    blind. The neighbours count because an approach at an angle walks the
    near edge across the fan.
    """
    for t, near in history:
        reach = REVERSE_FLOOR_MM + speed * max(WORST_PICTURE_AGE_S, now - t)
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(near) and near[j] < reach:
                return True
    return False


def advance_floor_latch(latch, scan, now, forward_speed=None) -> FloorLatch:
    """The latch after one more scan. Pure: a latch and a scan in, a latch out.

    Per probe:
      NO_RETURN out of an approach (see _blind_from_approach)   latches
      NO_RETURN while already latched                            stays
      NO_RETURN out of a clear or far scene                      never latches
      NEAR under the forward floor                               stays as it was
      a measured distance at or past the forward floor           lets go, and
                                                                 forgets what
                                                                 it read on
                                                                 the way in
    The last line is the "reads a valid distance above the floor again". A
    NO_RETURN out of nothing is the dark or featureless surface the old veto
    ignored on purpose, and this still ignores it.
    """
    if scan is None:
        return latch
    if forward_speed is None:
        forward_speed = FORWARD_SPEED
    floor = forward_floor_mm(forward_speed)
    speed = ground_speed_mm_s(max(0.0, float(forward_speed)))
    n = len(scan.reason)
    history = tuple(h for h in latch.history
                    if 0.0 <= now - h[0] <= FLOOR_LATCH_SECONDS)
    was = latch.latched if len(latch.latched) == n else (False,) * n

    latched = []
    for i in range(n):
        reason = scan.reason[i]
        if reason == NO_RETURN:
            hold = was[i] or _blind_from_approach(history, i, now, speed)
        elif reason == NEAR and float(scan.distance_mm[i]) < floor:
            hold = was[i]
        else:
            hold = False
        latched.append(bool(hold))

    # A probe measured at or past the floor has said where it is now, and
    # that supersedes anything it read on the way in: its old NEAR readings
    # are forgotten, so a dark patch right after it is not an approach.
    measured_far = [scan.reason[i] != NO_RETURN
                    and float(scan.distance_mm[i]) >= floor for i in range(n)]
    if any(measured_far):
        history = tuple(
            (t, tuple(math.inf if i < n and measured_far[i] else d
                      for i, d in enumerate(near)))
            for t, near in history)

    near = tuple(float(scan.distance_mm[i]) if scan.reason[i] == NEAR
                 else math.inf for i in range(n))
    return FloorLatch(history + ((float(now), near),), tuple(latched),
                      latch.reversed_mm if any(latched) else 0.0)


def floor_latch_moved(latch, travelled_mm) -> FloorLatch:
    """The latch after the car moved `travelled_mm`, negative backwards. Pure.

    Backed off FLOOR_RELEASE_REVERSE_MM in all, it lets go of everything and
    forgets the approach too, or the next blind scan would latch again off
    the same readings.
    """
    if not any(latch.latched):
        return latch._replace(reversed_mm=0.0)
    back = max(0.0, latch.reversed_mm - float(travelled_mm))
    if back >= FLOOR_RELEASE_REVERSE_MM:
        return EMPTY_FLOOR_LATCH
    return latch._replace(reversed_mm=back)


def floor_travel_mm(throttle, seconds) -> float:
    """ESTIMATED travel at `throttle` for `seconds`, as the latch counts it.

    Backwards at REVERSE_CREDIT_MM_S, and only at or past REVERSE_CREDIT_MIN;
    forwards on the fast fit. Both choices keep a latch on longer, never
    shorter.
    """
    if seconds <= 0.0:
        return 0.0
    try:
        t = float(throttle)
    except Exception:
        return 0.0
    if math.isfinite(t) and t <= REVERSE_CREDIT_MIN:
        return -REVERSE_CREDIT_MM_S * seconds
    return ground_speed_mm_s(t) * seconds


def forward_floor_vetoes(latch, scan, forward_speed=None) -> bool:
    """Hold a forward? Pure. Anything NEAR under the forward floor, or any
    probe latched blind. A missing scan is a veto: no picture is not a clear
    picture. Reverse is never asked about -- the camera cannot see behind.

    R1's NO_RETURN veto and the sweep check are the autopilot's reasons not
    to TURN, and neither my hand nor a model is the autopilot, so a blind
    probe on its own vetoes nothing here.
    """
    if scan is None:
        return True
    floor = forward_floor_mm(forward_speed)
    for i in range(len(scan.reason)):
        if scan.reason[i] == NEAR and float(scan.distance_mm[i]) < floor:
            return True
    return any(latch.latched)


class ForwardFloorVeto:
    """The forward floor with its memory. One per caller, owned by it.

    The caller shows it every new scan (observe) and every throttle that went
    out (note_throttle), and asks vetoes() before a forward. All the deciding
    is in the pure functions above; this only holds the latch between scans.
    The same scan shown twice counts once, so a caller asking at 100 Hz off a
    10 Hz scan cannot age its own history.

    `forward_speed` is the cap the forward is scaled to; None reads the
    module's FORWARD_SPEED when asked.
    """

    def __init__(self, forward_speed=None):
        self.forward_speed = forward_speed
        self.reset()

    def reset(self):
        """Nothing seen and the wheels at neutral."""
        self.latch = EMPTY_FLOOR_LATCH
        self.scan = None
        self._throttle = 0.0
        self._throttle_at = None

    def observe(self, scan, now):
        if scan is None or scan is self.scan:
            return
        self.scan = scan
        self.latch = advance_floor_latch(self.latch, scan, now,
                                         self.forward_speed)

    def note_throttle(self, throttle, now):
        """`throttle` is on the wheels from `now`. The one before it is
        credited for the time it was on them -- pigpio holds a pulse until
        it is told otherwise, so that is how long it drove."""
        if self._throttle_at is not None:
            self.latch = floor_latch_moved(
                self.latch, floor_travel_mm(self._throttle,
                                            now - self._throttle_at))
        try:
            t = float(throttle)
        except Exception:
            t = 0.0
        self._throttle = t if math.isfinite(t) else 0.0
        self._throttle_at = now

    def vetoes(self) -> bool:
        return forward_floor_vetoes(self.latch, self.scan, self.forward_speed)


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

        # Rear bumper state, and it is its own counter on purpose. The forward
        # detector and the reverse bumper ask the same question of the same
        # sensor in two completely different situations -- one while commanding
        # forward throttle for a second, one while commanding reverse for two
        # -- and sharing a count would let either one walk the other's
        # evidence up. _throttle_since in particular is gated on FORWARD
        # throttle and nothing in here writes it.
        self._reverse_still_samples = 0

        # Tunable parameters. The two speeds are the module's FORWARD_SPEED
        # and REVERSE_SPEED -- see there for why they are what they are. The
        # other two I set by feel on the bench and they are not to be nudged
        # from a desk.
        self.forward_speed = FORWARD_SPEED
        self.reverse_speed = REVERSE_SPEED
        # Limit to avoid servo damage when steering against an object -- and
        # it now bounds every steering angle auto commands, which is what that
        # sentence always claimed and did not do. It used to be read only by
        # begin_recover(); tick() drove the forward path straight off the old
        # get_steering_bias(), which saturated at 1.0 with a wall down one
        # side. Every steering decision is a fraction of this now -- cruise,
        # evade and recover alike -- so the bound is one multiplication at one
        # call site. Same 0.6 as it has always been, still a bench number,
        # still not to be nudged from a desk.
        self.max_turn = 0.6
        # Still my bench number, and still the same quantity: the smoothed
        # linear accel magnitude in m/s^2. What changed is that it is no longer
        # the only test. It is one of two OR'd terms now and the gyro is the one
        # carrying the weight. Measured parked, the accel magnitude tops out at
        # 0.058, so 0.2 never fired on a parked car -- the threshold was never
        # the problem and I have not touched it. This 0.2 is m/s^2 and has
        # nothing to do with forward_speed's old 0.2. The rolling accel it
        # was weighed against (p50 0.188) was measured with forward at 0.2;
        # at 0.4 the rolling accel is not measured, and the number stays.
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
        # The decision taken off that scan, same deal: read-only bookkeeping,
        # one assignment, nothing in here decides on it. Publishing it on
        # /marks is a one-line change in marks.py and is not made here.
        self.last_verdict = None

        # The evade latch. _evade_dir is the side the current episode is
        # committed to, 0.0 whenever there is no episode, and it is the only
        # thing carried from one decision to the next -- read_the_gradient()
        # is pure and sees none of this. _evade_since dates the episode for
        # EVADE_LIMIT_SECONDS and _evade_until is how long a clear frame
        # keeps steering the way it was.
        self._evade_dir = 0.0
        self._evade_since = 0.0
        self._evade_until = 0.0

        # What _announce() last printed, so it prints on transitions rather
        # than ten times a second.
        self._last_announced = None

        # True while my hand has the throttle in auto (main.Takeover). Set by
        # the control loop before each tick and read in one place: it disarms
        # the STUCK detector. stuck() asks whether a car the driver is telling
        # to go is not going, and while I hold the triggers the driver is not
        # telling it anything -- I might be holding it still on purpose, and
        # half a second of stop is exactly what letting go of a trigger
        # commands before the throttle goes back. A STUCK then would start a
        # recover and swing the steering, which may still be the driver's, at
        # a random lock under a car I am driving. So under my throttle the arm
        # is dropped on every tick, and after I let go it has to be earned
        # again: STUCK_ARM_SECONDS of the driver's own forward command, then
        # STUCK_CONFIRM_SAMPLES of stillness.
        self.human_throttle = False

    def motion_evidence(self, imu) -> str:
        """What the IMU can say about motion: MOVING, STILL or UNKNOWN.

        UNKNOWN is the whole point of this function. The old is_moving()
        returned False for a missing sample, the caller read that as "not
        moving", and "not moving" went straight to "stuck" -- so an absent IMU
        packet and a car wedged in a doorway produced the same command.

        The signal is the gyro first and the accelerometer second. The camera's
        IMU node is enabled for LINEAR_ACCELERATION, which is gravity
        compensated: near zero at rest AND near zero at a constant velocity.
        forward_speed is a creep (0.2 when this was written, 0.4 now) that
        reaches steady state in well under a second, so on the accel
        magnitude alone a car rolling along perfectly happily reads the same
        as a car that is not moving at all. Measured, that floor is p50
        0.017 / max 0.058 m/s^2, nowhere near stuck_threshold, which is why
        the accel term is now a second opinion for a knock or a wheel
        breaking loose rather than the primary test.

        The gyro survives constant velocity. It reads the chassis shaking, and
        a crawler on real ground always shakes while it rolls.

        What I could not measure from a desk: how much gyro a creep at 0.2
        actually produces. That one is measured now -- see GYRO_MOVING_MIN, and
        the old 0.035 was its median. At 0.4 it is not measured yet. What
        follows was written when it was anchored off the parked end of the
        range only, which erred toward
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

        A recover reverses for two seconds and the smoothing windows ARE
        sampled while recover owns the wheels now -- the rear bumper needs
        them -- so this runs at both ends of a recover rather than just the
        start. Without it the first forward readings after handing the wheels
        back would be a mean over samples taken while the car was going the
        other way.

        Both counters, because both measure consecutive evidence and neither
        has any business carrying a count across a stop.
        """
        self.imu_history.clear()
        self.gyro_history.clear()
        self._still_samples = 0
        self._reverse_still_samples = 0
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

    def scan_footprint(self, depth_frame, threshold_mm=CLEARANCE_MM) -> FootprintScan:
        """Scan the footprint once, time it, and leave the result behind.

        One read of the frame per tick, and this is it. tick() calls this and
        then decides off the scan it is handed; obstacle_in_front() is the
        same scan asked a single boolean question. Nothing in auto looks at a
        frame twice.

        The timing and the three read-only attributes telemetry publishes --
        last_scan, last_scan_ms and last_block_reason -- are set here, which
        is where they were always set, so /marks keeps reporting the same
        fields off the same assignments.

        last_block_reason is NEAR, NO_RETURN, both joined by a "+", or the
        empty string when the footprint is clear. Both are reported when both
        fired: they are two different behaviours and a log that calls them the
        same thing is a log that cannot tell me which one I am looking at.
        """
        started = time.perf_counter()
        scan = free_by_column(depth_frame, threshold_mm)
        self.last_scan = scan
        self.last_scan_ms = (time.perf_counter() - started) * 1000.0

        near = bool(np.any(scan.blocked & np.array(
            [r == NEAR for r in scan.reason])))
        no_return = bool(np.any(scan.blocked & np.array(
            [r == NO_RETURN for r in scan.reason])))

        if near and no_return:
            self.last_block_reason = NEAR + "+" + NO_RETURN
        elif near:
            self.last_block_reason = NEAR
        elif no_return:
            self.last_block_reason = NO_RETURN
        else:
            self.last_block_reason = CLEAR

        return scan

    def obstacle_in_front(self, depth_frame, threshold_mm=CLEARANCE_MM):
        """Is any part of the footprint inside its clearance budget?

        One question asked of scan_footprint(), kept because it is the honest
        name for that question even though tick() now wants the whole scan
        rather than the boolean.

        Fails closed, same as it always did, and for the same reason: on a
        stereo pair a no-return usually means too close, too dark or too
        featureless, and all three are reasons to stop. Dropping the zero
        pixels and then taking a mean used to hand np.mean() an empty array,
        which is nan, and nan > 0.5 is False -- a patch with nothing
        measurable in it read as "clear" and the car drove straight at it.

        What changed before this was what gets asked. It used to be one 50x50
        patch at the dead centre, 0.98% of the frame and 45 mm square out in
        the world on a 267 mm wide car. It is now PROBE_COLUMNS probes tiling
        the real footprint, each reaching from a little above the lens down
        toward the near ground, each with its own budget and its own
        fail-closed test. See the block above the class for the derivation.
        """
        return bool(self.scan_footprint(depth_frame, threshold_mm).blocked.any())

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
        self.controller.set_steering(*steering_pair(self._turn_dir))
        self.controller.set_throttle(self.reverse_speed)

        self._phase = _PHASE_REVERSE
        self._phase_until = now + REVERSE_SECONDS
        self._clear_motion_evidence()
        # Whatever the car was steering around, it is behind the decision now.
        self._clear_evade()

    def _rear_bumper_hit(self) -> bool:
        """Has the car reversed into something?

        There is no camera pointed backwards, so a reverse is blind -- but
        there is a contact sensor bolted to the car already and it is the
        gyro. The car has been commanded in reverse at reverse_speed and the
        chassis is
        not shaking, so either the wheels are against something or it is up on
        a ledge. Either way the reverse has stopped being useful and the two
        remaining seconds of it are two seconds of pushing.

        Its own counter, never the forward detector's. Same sensor, same
        notion of motion -- I do not want two definitions of "is it moving" --
        but the counts are separate and both are cleared at every stop, so
        neither phase can walk the other's evidence up.

        A sample only counts once the smoothing window is full.
        begin_recover() empties it, so the first ticks of a reverse are a mean
        over one or two readings taken while the ESC was still spinning up,
        and those read STILL on a car that is about to move off perfectly
        well. Waiting for the five samples is also the half second that stops
        this firing before the car has had a chance to go anywhere, which is
        why there is no separate arm delay.

        UNKNOWN holds the count where it is and never adds to it, exactly as
        in the forward detector: a dropped packet is not evidence of no
        motion, and a blind reverse is the worst place to start guessing.
        """
        evidence = self.motion_evidence(self.camera.get_imu_sample())

        if len(self.gyro_history) < self.imu_window_size:
            return False

        if evidence == MOVING:
            self._reverse_still_samples = 0
            return False

        if evidence == STILL:
            self._reverse_still_samples += 1

        return self._reverse_still_samples >= REVERSE_STALL_SAMPLES

    def recovering(self, now):
        """Advance the recover phases. True while recover owns the wheels.

        This reads the IMU during the reverse phase, which it did not use to.
        tick() returns the moment this says True, so for the whole 2.5 seconds
        of REVERSE plus SETTLE nothing in this file looked at depth OR at
        motion -- and reverse is the one direction with nothing pointed down
        it. The reverse phase is where the only sensor that can see behind the
        car is worth reading.
        """
        if self._phase is None:
            return False

        if self._phase == _PHASE_REVERSE:
            # Asked every tick, and before the deadline, because it is
            # accumulating evidence rather than answering a question. Short
            # circuiting it on the last tick of a reverse would be harmless;
            # skipping it on any other tick would not be.
            hit = self._rear_bumper_hit()
            if not hit and now < self._phase_until:
                return True
            if hit:
                print("Rear bumper - something behind me, cutting the reverse.")
            self.controller.set_throttle(0)
            self._phase = _PHASE_SETTLE
            self._phase_until = now + SETTLE_SECONDS
            return True

        if now < self._phase_until:
            return True

        # Settle is up: centre the steering and hand the wheels back.
        #
        # This used to command set_steering(-self._turn_dir, 0), which is the
        # opposite front lock and a centred rear -- not centre, and not the
        # counter-phase pair begin_recover() set either. Nothing is moving by
        # here (the throttle went to zero a settle ago) so there was nothing
        # to counter-steer, and it held a servo at full lock against nothing
        # for up to a tick. Centre is what "unwind the steering" meant.
        self.controller.set_steering(0, 0)
        self._phase = None
        self._phase_until = 0.0
        # The bumper has been sampling the IMU for the length of the reverse,
        # so unlike before there is real evidence in the windows now, and all
        # of it was taken going the other way.
        self._clear_motion_evidence()
        return False

    def abort(self):
        """Give up on a recover in progress and stop. Safe to call twice."""
        self._phase = None
        self._phase_until = 0.0
        self.controller.set_throttle(0)
        self._clear_motion_evidence()
        self._clear_evade()

    def _commit(self, verdict, now):
        """Hold an evade to the side it started on, and time-limit it.

        The latch lives on the Driver and not in read_the_gradient(), so the
        decision function stays pure and this stays the only place with a
        memory in it.

        Three jobs, in the order they are allowed to win:

          a reverse clears the latch and is obeyed immediately -- safety beats
          commitment, always;

          an evade that asks for the opposite side of an evade already running
          is never obeyed. It becomes a reverse. Dither is the failure this
          feature could plausibly have -- the decision rate is 10 Hz and a
          scene balanced on OPEN_RATIO_MIN could flip the verdict tick to tick
          -- and a sign flip at 10 Hz on two steering servos is the one
          outcome worth making unreachable rather than merely damped. Turning
          it into a reverse is both safe and assertable on the recorded
          commands, which is what a low-pass filter would not be;

          and a clear frame inside EVADE_SECONDS of the last evade keeps
          steering the way it was. Without that the car un-steers the instant
          the obstacle rotates out of the fan, which is about one tick in, and
          it has not gone anywhere yet.

        No separate progress monitor. R1, R2, R5 and R6 re-run on every
        decision and already are one: the obstacle closing, widening, going
        unmeasurable or flattening out all end the episode by themselves.
        EVADE_LIMIT_SECONDS is only the backstop for the case where none of
        them ever fire.
        """
        if verdict.action == ACT_REVERSE:
            self._evade_dir = 0.0
            return verdict

        if verdict.action == ACT_EVADE:
            side = 1.0 if verdict.steer > 0 else -1.0
            if self._evade_dir and side != self._evade_dir:
                self._evade_dir = 0.0
                return SteerVerdict(ACT_REVERSE, 0.0, "evade flipped")
            if not self._evade_dir:
                self._evade_since = now
            elif now - self._evade_since >= EVADE_LIMIT_SECONDS:
                self._evade_dir = 0.0
                return SteerVerdict(ACT_REVERSE, 0.0, "evade timeout")
            self._evade_dir = side
            self._evade_until = now + EVADE_SECONDS
            return verdict

        if self._evade_dir and now < self._evade_until:
            return SteerVerdict(ACT_EVADE, self._evade_dir, "holding out")

        self._evade_dir = 0.0
        return verdict

    def _clear_evade(self):
        """Forget an evade episode. Called wherever recover takes the wheels."""
        self._evade_dir = 0.0
        self._evade_since = 0.0
        self._evade_until = 0.0
        self._last_announced = None

    def _announce(self, verdict, scan):
        """Print the decision, on transitions only.

        Keyed on (action, reason), so a car driving down a corridor prints one
        line when it starts leaning away from the wall and one when it stops,
        rather than ten a second. The decision rate is 10 Hz and journald
        rate-limits, so a per-tick line would cost me the lines I actually
        want.

        The bracketed probe string stays exactly as it was. It is what makes
        the journal readable after the fact and I have been mining it -- the
        88 obstacle events behind MAX_BLOCKED_PROBES and OPEN_RATIO_MIN were
        read straight out of these lines.
        """
        key = (verdict.action, verdict.reason)
        if key == self._last_announced:
            return
        self._last_announced = key

        where = probe_marks(scan)
        if verdict.action == ACT_REVERSE:
            print(f"Obstacle detected ({self.last_block_reason}) "
                  f"[{where}] {verdict.reason} - backing up.")
        elif verdict.action == ACT_EVADE:
            side = "right" if verdict.steer > 0 else "left"
            print(f"Obstacle detected ({self.last_block_reason}) "
                  f"[{where}] {verdict.reason} - steering {side}.")
        elif where.strip("."):
            # Cruising, but with something in sight. Worth one line; an empty
            # fan is not.
            print(f"Clear [{where}] - driving on.")

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
        #
        # One scan, three consumers: the verdict, the journal line and the
        # telemetry attributes. read_the_gradient() is pure and _commit() is
        # the only thing here with a memory, so what gets commanded below is
        # a function of this scan plus one latched direction and nothing else.
        scan = self.scan_footprint(depth)
        verdict = self._commit(read_the_gradient(scan), now)
        self.last_verdict = verdict
        self._announce(verdict, scan)

        if verdict.action == ACT_REVERSE:
            self.begin_recover(now)
            return

        if self.human_throttle:
            self._throttle_since = None
            self._still_samples = 0
        elif self.stuck(imu, now):
            self.last_recovery_time = now
            print("STUCK - backing up to recover.")
            self.begin_recover(now)
            return

        # max_turn, applied where it was always meant to be. Its comment says
        # "limit to avoid servo damage when steering against an object" and
        # for a long time that was a lie: the constant was referenced only by
        # begin_recover(), so the forward path -- the one path that actually
        # drives at a wall -- was unbounded, and the old get_steering_bias()
        # saturated at exactly 1.0000 with a wall down one side.
        #
        # The verdict hands over a FRACTION of max_turn rather than an angle,
        # so the multiplication here is the whole of the bound and there is
        # nothing for a decision to saturate past. The clip is belt and
        # braces on a fraction that is already constructed inside [-1, 1].
        #
        # np.clip and not max(-1, min(1, v)), which is not a style choice.
        # min(1.0, nan) is 1.0, so the plain clamp turns a nan into a
        # commanded hard turn -- and a nan is meant to reach set_steering()
        # and be rejected into neutral on both channels there. np.clip
        # propagates it.
        front = float(np.clip(verdict.steer, -1.0, 1.0) * self.max_turn)
        self.controller.set_steering(*steering_pair(front))
        self.controller.set_throttle(self.forward_speed)
        if self._throttle_since is None:
            self._throttle_since = now
