import time
import sys
import pygame
import pigpio
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# === GPIO PWM for ESC using pigpio ===
ESC_GPIO_PIN = 18
ESC_NEUTRAL_PW = 1575
ESC_FULL_FORWARD_PW = 2000
ESC_FULL_REVERSE_PW = 1000

# === PCA9685 PWM Channels ===
PAN_CHANNEL = 14
TILT_CHANNEL = 15
SERVO_FREQ = 50

# === Setup pigpio for ESC control ===
pi = pigpio.pi()
if not pi.connected:
    sys.exit("pigpio daemon not running")
pi.set_mode(ESC_GPIO_PIN, pigpio.OUTPUT)

# === Setup I2C and PCA9685 ===
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = SERVO_FREQ
pan_servo = servo.Servo(pca.channels[PAN_CHANNEL], actuation_range=180)
tilt_servo = servo.Servo(pca.channels[TILT_CHANNEL], actuation_range=180)

# === Initialize Pygame and Controller ===
def init_controller():
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("🎮 Controller connected. RT = throttle, LT = steering, Right Stick = pan/tilt")
    return joystick

# === ESC Control ===
def set_esc_throttle(value):
    pulse = int(ESC_NEUTRAL_PW + value * (ESC_FULL_FORWARD_PW - ESC_NEUTRAL_PW))
    pi.set_servo_pulsewidth(ESC_GPIO_PIN, pulse)
    return pulse

# === Servo Control: Pan & Tilt ===
def set_pan_tilt(x_val, y_val):
    # Clamp joystick range and convert to servo angles
    x_val = max(-1.0, min(1.0, x_val))
    y_val = max(-1.0, min(1.0, y_val))
    pan_angle = int((x_val + 1.0) * 90)     # -1 to 1 → 0 to 180
    tilt_angle = int((1.0 - y_val) * 90)    # -1 to 1 → 180 to 0 (invert)
    pan_servo.angle = pan_angle
    tilt_servo.angle = tilt_angle
    return pan_angle, tilt_angle

# === Main Control Loop ===
def control_loop():
    joystick = init_controller()
    last_snapshot = None

    try:
        while True:
            pygame.event.pump()

            # Throttle (RT Trigger, axis 4): -1 to 1 → 0 to 1
            throttle_raw = joystick.get_axis(4)
            throttle_norm = max(0.0, (throttle_raw + 1) / 2)
            throttle_pulse = set_esc_throttle(throttle_norm)

            # Pan/Tilt (Right stick X/Y → axes 3/4)
            pan_raw = joystick.get_axis(3)
            tilt_raw = joystick.get_axis(0)
            pan_angle, tilt_angle = set_pan_tilt(pan_raw, tilt_raw)

            if joystick.get_button(0):  # A Button
                last_snapshot = (throttle_pulse, pan_angle, tilt_angle)

            status = (
                f"Throttle: {throttle_pulse}µs | Pan: {pan_angle}° | Tilt: {tilt_angle}°"
            )
            if last_snapshot:
                status += f" | 🔸 Snapshot (A): {last_snapshot}"
            print(f"\r{status.ljust(100)}", end="", flush=True)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping ESC and Servos...")
    finally:
        pi.set_servo_pulsewidth(ESC_GPIO_PIN, ESC_NEUTRAL_PW)
        time.sleep(1)
        pi.set_servo_pulsewidth(ESC_GPIO_PIN, 0)
        pi.stop()
        pca.deinit()

if __name__ == "__main__":
    control_loop()
