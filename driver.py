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
            print("[Driver] Path is clear — creeping forward")
            self.controller.set_steering(0, 0)
            self.controller.set_throttle(self.forward_speed)
