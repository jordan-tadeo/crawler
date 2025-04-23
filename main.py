import time
import sys
import pygame
import RPi.GPIO as GPIO
import board
import busio
from adafruit_pca9685 import PCA9685

# === PCA9685 PWM Channels ===
ESC_CHANNEL = 3
SERVO_CHANNEL = 15
SERVO_FREQ = 50
SERVO_MIN = 205
SERVO_MAX = 410
SERVO_NEUTRAL = 307

# === ESC Pulse Range (PCA-style, 0–4095) ===
ESC_MIN_PULSE = 205
ESC_MAX_PULSE = 410
ESC_NEUTRAL = 307

# === Setup I2C and PCA9685 ===
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = SERVO_FREQ
pca.channels[ESC_CHANNEL].duty_cycle = ESC_NEUTRAL
time.sleep(1)

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
    value = max(0.0, min(1.0, value))
    pulse = int(ESC_NEUTRAL + (value * (ESC_MAX_PULSE - ESC_MIN_PULSE) / 2))
    pulse = max(ESC_MIN_PULSE, min(ESC_MAX_PULSE, pulse))
    pca.channels[ESC_CHANNEL].duty_cycle = pulse
    return pulse

# === Servo Control ===
def set_servo_position(value):
    value = max(-1.0, min(1.0, value))
    pulse = int(SERVO_NEUTRAL + value * ((SERVO_MAX - SERVO_MIN) / 2))
    pulse = max(SERVO_MIN, min(SERVO_MAX, pulse))
    pca.channels[SERVO_CHANNEL].duty_cycle = pulse
    return pulse

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
            steering_pulse = set_servo_position(steering_raw)

            if joystick.get_button(0):  # A Button
                last_snapshot = (throttle_pulse, steering_pulse)

            # Display live status
            status = f"Throttle: {throttle_pulse} | Steering Pulse: {steering_pulse}"
            if last_snapshot:
                status += f" | 🔸 Snapshot (A): {last_snapshot}"
            print(f"\r{status.ljust(80)}", end="", flush=True)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping ESC and Servo...")
    finally:
        pca.channels[ESC_CHANNEL].duty_cycle = ESC_NEUTRAL
        pca.deinit()

if __name__ == "__main__":
    control_loop()
