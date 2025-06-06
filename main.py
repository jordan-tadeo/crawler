
from PyQt5.QtWidgets import QApplication
import sys
import asyncio
import threading
from VehicleController import VehicleController
from Joystick import Joystick
from Logger import Logger
import USBCamera as uc
import Dashboard as db
import pygame
import PersonFollower as pf
from driver import Driver
from oakd import OakD

# === Main Control Loop ===
async def control_loop(vc: VehicleController, driver: Driver):

        try:
            while True:
                
                # Do something
                await driver.tick()
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[Shutdown] Stopping ESC and Servos...")
        finally:
            vc.close()

if __name__ == "__main__":
    log = Logger()

    vc = VehicleController(logger=log)

    oakd = OakD()

    app = QApplication(sys.argv)
    dashboard = db.Dashboard(None, oakd)
    dashboard.show()  

    driver = Driver(vc)

    # Run the asyncio control loop in a separate thread
    loop = asyncio.get_event_loop()
    asyncio_thread = threading.Thread(target=loop.run_until_complete, args=(control_loop(vc, driver),))
    asyncio_thread.start()

    sys.exit(app.exec_())
