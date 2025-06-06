import asyncio
import numpy as np
from oakd import OakD
from VehicleController import VehicleController
import random

class Driver:
    def __init__(self, controller: VehicleController, oakd: OakD):
        self.controller = controller
        self.camera = oakd
        self.turning = False

        # Tunable parameters
        self.forward_speed = 0.2    # very slow forward motion
        self.reverse_speed = -0.2   # very slow backup
        self.turn_speed = 0.3       # light steering during recovery

    def get_steering_bias(self, depth, region_width=40, region_height=30, threshold_mm=1000):
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


    def obstacle_in_front(self, depth_frame, threshold_mm=500, region_size=(20, 80)):
        """Check if there's an obstacle in the center of the depth frame."""
        h, w = depth_frame.shape
        dh, dw = region_size
        center = depth_frame[h//2 - dh//2:h//2 + dh//2, w//2 - dw//2:w//2 + dw//2]
        center = center[center > 0]  # ignore zero-depth pixels
        return np.any(center < threshold_mm)

    async def recover(self):
        """Backup and turn to avoid obstacle."""
        print("[Driver] Obstacle detected — reversing and turning")

        self.controller.set_throttle(self.reverse_speed)
        await asyncio.sleep(0.8)

        self.controller.set_throttle(0)

        turn_dir = random.choice([-self.turn_speed, self.turn_speed])
        self.controller.set_steering(turn_dir, turn_dir)
        await asyncio.sleep(0.6)

        self.controller.set_steering(0, 0)
        self.controller.set_throttle(0)
        await asyncio.sleep(0.2)

    async def tick(self):
        """Main loop logic: move or recover based on depth."""
        depth = self.camera.get_depth_frame()

        if self.obstacle_in_front(depth):
            await self.recover()
        else:
            steering = self.get_steering_bias(depth)
            print(f"[Driver] Path clear — steering bias: {steering:.2f}")
            self.controller.set_steering(steering, steering)
            self.controller.set_throttle(self.forward_speed)

