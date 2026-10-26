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


class Driver:
    def __init__(self, controller, oakd):
        self.controller = controller
        self.camera = oakd

        self.last_recovery_time = 0.0
        self.recovery_cooldown = 5.0  # seconds before the stuck check re-arms
        self.imu_history = []
        self.imu_window_size = 5  # window for smoothing the accel magnitude

        # Tunable parameters. I set all four of these by feel on the bench and
        # they are not to be nudged from a desk.
        self.forward_speed = 0.2    # very slow forward motion
        self.reverse_speed = -0.6   # very slow backup
        self.max_turn = 0.6         # limit to avoid servo damage when steering against an object
        self.stuck_threshold = 0.2  # or whatever threshold makes sense for imu

        # Recover phase machine. _phase is None whenever recover does not own
        # the wheels, which is the normal case.
        self._phase = None
        self._phase_until = 0.0
        self._turn_dir = 0.0

    def is_moving(self, imu) -> bool:
        """True if the car looks like it is actually moving.

        Careful: the IMU node is enabled for LINEAR_ACCELERATION, and linear
        acceleration is near zero at a constant velocity. A smooth roll across
        a flat floor can therefore read as "stuck" and trigger a recover for
        no reason. stuck_threshold = 0.2 is my bench number on the accel
        magnitude, so it mostly rides the noise floor -- but whether this
        wants a gyro term or a wheel-speed source instead needs a run on the
        bench, not a guess. Leaving the threshold alone until then.
        """
        if imu is None:
            return False

        accel = imu["accel"]
        magnitude = np.linalg.norm(accel)

        self.imu_history.append(magnitude)
        if len(self.imu_history) > self.imu_window_size:
            self.imu_history.pop(0)

        avg_motion = np.mean(self.imu_history)

        return avg_motion > self.stuck_threshold

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
            return

        imu = self.camera.get_imu_sample()

        # Decide first, command second. The old order commanded forward speed
        # and only then checked for an obstacle, so every tick facing a wall
        # lurched into it for a tick before reversing.
        if self.obstacle_in_front(depth):
            print("Obstacle detected - backing up.")
            self.begin_recover(now)
            return

        if not self.is_moving(imu) and now - self.last_recovery_time > self.recovery_cooldown:
            self.last_recovery_time = now
            print("STUCK - backing up to recover.")
            self.begin_recover(now)
            return

        self.controller.set_steering(self.get_steering_bias(depth), 0)
        self.controller.set_throttle(self.forward_speed)
