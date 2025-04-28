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

# === Main Control Loop ===
async def control_loop():
    pygame.init()

    joystick = js.Joystick()
    vecon = vc.VehicleController()

    log = lg.Logger()

    await asyncio.sleep(1)  # Allow time for joystick to initialize

    vehicle_state = vecon.get_state()
    print(f"Initial State: {vehicle_state}")
    try:
        while True:
            joystick.update_connection_status()
            if not joystick.is_connected():
                joystick.wait_for_connection()
            print(f"Vehicle State: {vehicle_state}", end="\r", flush=True)

            pygame.event.pump()
            throttle = joystick.read_throttle()
            steering_front = joystick.get_axis("LEFT_X")
            steering_rear = joystick.get_axis("RIGHT_X")

            if throttle and steering_front and steering_rear:
                print(f"Throttle: {throttle:.2f}, Front Steering: {steering_front:.2f}, Rear Steering: {steering_rear:.2f}", end="\r", flush=True)

            vecon.set_throttle(throttle)
            vecon.set_steering(steering_front, steering_rear)  

            new_vehicle_state = vecon.get_state()    

            if new_vehicle_state != vehicle_state:
                log.log("info", "Vehicle State Change", f"New state: {new_vehicle_state}")
                log.csv("log/vehicle_state_log.csv", "Vehicle State Change", new_vehicle_state)

            vehicle_state = new_vehicle_state

            await asyncio.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping ESC and Servos...")
    finally:
        vecon.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = db.Dashboard()

    # Run the asyncio control loop in a separate thread
    loop = asyncio.get_event_loop()
    asyncio_thread = threading.Thread(target=loop.run_until_complete, args=(control_loop(),))
    asyncio_thread.start()

    sys.exit(app.exec_())
