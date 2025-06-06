import asyncio
import numpy as np
from oakd import OakD
from VehicleController import VehicleController

class Driver:
    def __init__(self, controller):
        self.controller = controller
        self.camera = OakD()

    def obstacle_in_front(self, depth_frame, threshold_mm=400):
        h, w = depth_frame.shape
        center = depth_frame[h//2 - 10:h//2 + 10, w//2 - 40:w//2 + 40]
        return np.any(center < threshold_mm)

    async def tick(self):
        depth = self.camera.get_depth_frame()
        if self.obstacle_in_front(depth):
            self.controller.set_throttle(0)
            self.controller.set_throttle(-0.5)
            await asyncio.sleep(1.0)
            self.controller.set_throttle(0)
        else:
            self.controller.set_throttle(0.6)
