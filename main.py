import time
import sys
import pygame
import pigpio
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# === GPIO PWM for ESC using pigpio ===
ESC_GPIO_PIN = 18  # GPIO pin for ESC signal
ESC_NEUTRAL_PW = 1500  # microseconds
ESC_FULL_FORWARD_PW = 2000
ESC_FULL_REVERSE_PW = 1000

# === PCA9685 PWM Channels ===
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
steering_servo = servo.Servo(pca.channels[SERVO_CHANNEL], actuation_range=180)

# === Initialize Pygame and Controller ===
def init_controller():
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("🎮 Controller connected. Use RT for throttle, LT for steering.")
    return joystick

# === ESC Control ===
def set_esc_throttle(value):
    pulse = int(ESC_NEUTRAL_PW + value * (ESC_FULL_FORWARD_PW - ESC_FULL_REVERSE_PW))
    if not hasattr(set_esc_throttle, "last_pulse") or abs(pulse - set_esc_throttle.last_pulse) >= 5:
        pi.set_servo_pulsewidth(ESC_GPIO_PIN, pulse)
        set_esc_throttle.last_pulse = pulse
    return pulse

# === Servo Control ===
def set_servo_position(value):
    value = max(-1.0, min(1.0, value))
    angle = int((value + 1.0) * 90)  # map -1.0 to 1.0 → 0 to 180
    steering_servo.angle = angle
    return angle

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

            # Steering (Left stick vertical, axis 1): -1 to 1
            steering_raw = joystick.get_axis(1)
            steering_angle = set_servo_position(steering_raw)

            if joystick.get_button(0):  # A Button
                last_snapshot = (throttle_pulse, steering_angle)

            # Display live status
            status = f"Throttle: {throttle_pulse}µs | Steering Angle: {steering_angle}°"
            if last_snapshot:
                status += f" | 🔸 Snapshot (A): {last_snapshot}"
            print(f"\r{status.ljust(80)}", end="", flush=True)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping ESC and Servo...")
    finally:
        pi.set_servo_pulsewidth(ESC_GPIO_PIN, ESC_NEUTRAL_PW)
        time.sleep(1)
        pi.set_servo_pulsewidth(ESC_GPIO_PIN, 0)  # stop signal
        pi.stop()
        pca.deinit()

if __name__ == "__main__":
    control_loop()