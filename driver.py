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
# GYRO_MOVING_MIN sits at five times the measured parked maximum and about eight
# times p99. The parked end of that range is measured. The rolling end is not --
# see the note in motion_evidence().
GYRO_MOVING_MIN = 0.035  # rad/s

# How much evidence before the car is allowed to call itself stuck. At the 10 Hz
# tick this is a full second of consecutive confident no-motion readings, every
# one of them taken while forward throttle was already up.
STUCK_CONFIRM_SAMPLES = 10

# And how long forward throttle must have been commanded before the stuck check
# arms at all. A car that has just been handed the wheels has not been given a
# chance to move yet. The old code declared it stuck on the very first tick of
# auto mode -- last_recovery_time started at 0.0, so the cooldown was already
# expired -- which is exactly what the journal shows: "Switched to auto" and
# "STUCK - backing up to recover." stamped on the same second, every time.
STUCK_ARM_SECONDS = 1.0


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

    def obstacle_in_front(self, depth_frame, threshold_mm=400, region_size=(50, 50)):
        """Check if there's an obstacle in the center of the depth frame.

        Fails closed. Dropping the zero pixels and then taking a mean used to
        hand np.mean() an empty array, which is nan, and nan > 0.5 is False --
        so a centre patch with nothing measurable in it read as "clear" and
        the car drove straight at it. On a stereo pair a no-return usually
        means too close, too dark or too featureless, every one of which is a
        reason to stop. So if under a tenth of the patch came back valid,
        call it an obstacle.
        """
        h, w = depth_frame.shape
        dh, dw = region_size
        center = depth_frame[h//2 - dh//2:h//2 + dh//2, w//2 - dw//2:w//2 + dw//2]
        total = center.size
        center = center[center > 0]  # ignore zero-depth pixels

        if total == 0 or center.size < total * 0.1:
            return True

        return np.mean(center < threshold_mm) > 0.5

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
            print("Obstacle detected - backing up.")
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
