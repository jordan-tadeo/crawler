import asyncio
import numpy as np
from oakd import OakD
from VehicleController import VehicleController
import random

class Driver:
    def __init__(self, controller: VehicleController):
        self.controller = controller
        self.camera = OakD()
        self.turning = False

    def obstacle_in_front(self, depth_frame, threshold_mm=500, region_size=(20, 80)):
        """Check if there's an obstacle in the center of the depth frame."""
        h, w = depth_frame.shape
        dh, dw = region_size
        center = depth_frame[h//2 - dh//2:h//2 + dh//2, w//2 - dw//2:w//2 + dw//2]
        center = center[center > 0]  # filter out invalid values
        return np.any(center < threshold_mm)

    async def recover(self):
        """Backup and turn in place to recover from obstacle."""
        print("[Driver] Obstacle detected — backing up and turning")
        self.controller.set_throttle(-0.5)
        await asyncio.sleep(0.7)

        self.controller.set_throttle(0)
        turn_dir = random.choice([-0.6, 0.6])  # left or right
        self.controller.set_steering(turn_dir, turn_dir)
        await asyncio.sleep(0.5)

        self.controller.set_steering(0, 0)
        self.controller.set_throttle(0)
        await asyncio.sleep(0.1)

    async def tick(self):
        """Called repeatedly in a loop to make drive decisions."""
        depth = self.camera.get_depth_frame()

        if self.obstacle_in_front(depth):
            await self.recover()
        else:
            print("[Driver] Path is clear — moving forward")
            self.controller.set_steering(0, 0)
            self.controller.set_throttle(0.55)
