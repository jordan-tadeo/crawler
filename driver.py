import numpy as np
from oakd import OakD
from VehicleController import VehicleController

class Driver:
    async def tick(self):
        depth_frame = self.camera.get_depth_frame()
        if self.obstacle_in_front(depth_frame):
            self.controller.stop()
            self.controller.set_throttle(-0.5)
            await asyncio.sleep(1.0)
            self.controller.set_throttle(0)
        else:
            self.controller.move_forward()

