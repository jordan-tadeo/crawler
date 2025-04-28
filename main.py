import time
import pygame
import VehicleController as vc
import Joystick as js


# === Main Control Loop ===
def control_loop():
    joystick = js.Joystick()
    vecon = vc.VehicleController()

    pygame.init()
    time.sleep(1)  # Allow time for joystick to initialize
    try:
        while True:
            joystick.update_connection_status()
            if not joystick.is_connected():
                joystick.wait_for_connection()

            pygame.event.pump()
            throttle = joystick.read_throttle()
            steering_front = joystick.get_axis("LEFT_X")
            steering_rear = joystick.get_axis("RIGHT_X")

            if throttle and steering_front and steering_rear:
                print(f"Throttle: {throttle:.2f}, Front Steering: {steering_front:.2f}, Rear Steering: {steering_rear:.2f}", end="\r", flush=True)
            
            vecon.set_throttle(throttle)
            vecon.set_steering(steering_front, steering_rear)      

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping ESC and Servos...")
    finally:
        vecon.close()

if __name__ == "__main__":
    control_loop()
