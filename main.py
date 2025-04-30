from PyQt5.QtWidgets import QApplication
import sys
import asyncio
import threading
import VehicleController as vc
import Joystick as js
import Logger as lg
import USBCamera as uc
import Dashboard as db
import pygame
import PersonFollower as pf

# === Main Control Loop ===
async def control_loop(joystick: js.Joystick, vecon: vc.VehicleController, person_follower: pf.PersonFollower):
        await asyncio.sleep(1)  # Allow time for joystick to initialize

        vehicle_state = None
        # frame_counter = 0  # Counter to throttle console output
        try:
            while True:
                joystick.update_connection_status()
                if not joystick.is_connected():
                    joystick.wait_for_connection()

                pygame.event.pump()
                throttle = joystick.read_throttle()
                steering_front = joystick.get_axis("LEFT_X", limit_perc=70)
                steering_rear = joystick.get_axis("RIGHT_X", limit_perc=70)

                vecon.set_throttle(throttle)
                vecon.set_steering(steering_front, steering_rear)

                new_vehicle_state = vecon.get_state()

                if new_vehicle_state != vehicle_state:
                    log.log("info", "Vehicle State Change", f"New state: {new_vehicle_state}")
                    log.csv("log/vehicle_state_log.csv", "Vehicle State Change", new_vehicle_state)

                vehicle_state = new_vehicle_state

                # # Throttle console output to every 10th frame
                # if frame_counter % 10 == 0:
                #     print(f"Vehicle State: {vehicle_state}", end="\r", flush=True)

                # frame_counter += 1
                person_follower.process_and_adjust()

                await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            print("\n[Shutdown] Stopping ESC and Servos...")
        finally:
            vecon.close()

if __name__ == "__main__":
    pygame.init()
    log = lg.Logger()

    joystick = js.Joystick()
    vecon = vc.VehicleController(logger=log)
    usb_cam = uc.USBCamera(camera_index=0, fps=30)
    person_follower = pf.PersonFollower(vecon, usb_cam)

    person_follower.start()  # Start YOLO processing thread

    app = QApplication(sys.argv)
    dashboard = db.Dashboard(person_follower)
    dashboard.show()    

    # Run the asyncio control loop in a separate thread
    loop = asyncio.get_event_loop()
    asyncio_thread = threading.Thread(target=loop.run_until_complete, args=(control_loop(joystick, vecon, person_follower),))
    asyncio_thread.start()

    try:
        sys.exit(app.exec_())
    finally:
        person_follower.stop()  # Stop YOLO processing thread
