
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
import os

# os.environ['DEPTHAI_LEVEL'] = 'debug'

# === Main Control Loop ===
async def control_loop(vc: VehicleController, driver: Driver):

        try:
            t = 0
            while True:
                
                # Do something
                # await driver.tick()
                print(f"{t = }")
                vc.set_throttle(t)
                await asyncio.sleep(0.1)
                t -= 0.1
                if t > -1:
                     t = 0
        except KeyboardInterrupt:
            print("\n[Shutdown] Stopping ESC and Servos...")
        finally:
            print("[Shutdown] Stopping vehicle safely.")
            vc.close()

if __name__ == "__main__":
    log = Logger()

    vc = VehicleController(logger=log)

    oakd = OakD()

    app = QApplication(sys.argv)
    dashboard = db.Dashboard(None, oakd)
    dashboard.show()  

    driver = Driver(vc, oakd)

    # Run the asyncio control loop in a separate thread
    try:
        loop = asyncio.get_event_loop()
        asyncio_thread = threading.Thread(target=loop.run_until_complete, args=(control_loop(vc, driver),))
        asyncio_thread.start()

        sys.exit(app.exec_())
    except:
            print("\n[Shutdown] Stopping ESC and Servos...")
    finally:
        print("[Shutdown] Stopping vehicle safely.")
        vc.close()
