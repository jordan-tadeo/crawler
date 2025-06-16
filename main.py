
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

ESC_NEUTRAL_PW = 1325
ESC_FULL_FORWARD_PW = 1865
ESC_FULL_REVERSE_PW = 1100

# === Main Control Loop ===
async def control_loop(vc: VehicleController, driver: Driver):

        try:
            while True:
                # Do something
                # await driver.tick()
                await vc.throttle_sweep(1100, 1901, 100, 2)
                await asyncio.sleep(0.1)
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
