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
PAN_CHANNEL = 13
TILT_CHANNEL = 14
SERVO_CHANNEL = 15 
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
steering_servo = servo.Servo(pca.channels[SERVO_CHANNEL], actuation_range=180)

# === Initialize Pygame and Controller ===
def init_controller():
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("\U0001F3AE Controller connected. RT = forward, LT = reverse, Left Stick X = steering, Right Stick = pan/tilt")
    return joystick

# === ESC Control ===
def set_esc_throttle(forward_value, reverse_value):
    if forward_value > 0.05:
        pulse = int(ESC_NEUTRAL_PW + forward_value * (ESC_FULL_FORWARD_PW - ESC_NEUTRAL_PW))
    elif reverse_value > 0.05:
        pulse = int(ESC_NEUTRAL_PW - reverse_value * (ESC_NEUTRAL_PW - ESC_FULL_REVERSE_PW))
    else:
        pulse = ESC_NEUTRAL_PW
    pi.set_servo_pulsewidth(ESC_GPIO_PIN, pulse)
    return pulse

# === Servo Control: Steering ===
def set_steering(x_val):
    x_val = max(-1.0, min(1.0, -x_val))
    steering_angle = int((x_val + 1.0) * 90)
    steering_servo.angle = steering_angle
    return steering_angle

# === Servo Control: Pan & Tilt ===
def set_pan_tilt(x_val, y_val):
    x_val = max(-1.0, min(1.0, x_val))
    y_val = max(-1.0, min(1.0, y_val))
    pan_angle = int((x_val + 1.0) * 90)
    tilt_angle = int((1.0 - y_val) * 90)
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

            # Throttle Forward (RT Trigger, axis 4): -1 to 1 => 0 to 1
            forward_raw = joystick.get_axis(4)
            forward_norm = max(0.0, (forward_raw + 1) / 2)

            # Throttle Reverse (LT Trigger, axis 5): -1 to 1 => 0 to 1
            reverse_raw = joystick.get_axis(5)
            reverse_norm = max(0.0, (reverse_raw + 1) / 2)

            throttle_pulse = set_esc_throttle(forward_norm, reverse_norm)

            # Steering (Left stick X axis -> axis 1)
            steering_raw = joystick.get_axis(0)
            steering_angle = set_steering(steering_raw)

            # # Pan/Tilt (Right stick X/Y -> axes 3/0)
            # pan_raw = joystick.get_axis(3)
            # tilt_raw = joystick.get_axis(0)
            # pan_angle, tilt_angle = set_pan_tilt(pan_raw, tilt_raw)

            # if joystick.get_button(0):  # A Button
            #     last_snapshot = (throttle_pulse, steering_angle, pan_angle, tilt_angle)

            status = (
                f"Throttle: {throttle_pulse}µs | Steering: {steering_angle}°"# | Pan: {pan_angle}° | Tilt: {tilt_angle}°"
            )
            if last_snapshot:
                status += f" | 🔸 Snapshot (A): {last_snapshot}"
            print(f"\r{status.ljust(120)}", end="", flush=True)

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
