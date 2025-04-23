import time
import sys
import pygame
import RPi.GPIO as GPIO
import board
import busio
from adafruit_pca9685 import PCA9685

# === GPIO PWM for ESC ===
ESC_GPIO_PIN = 18  # GPIO pin for ESC signal
ESC_PWM_FREQ = 50  # 50Hz standard for ESCs
ESC_NEUTRAL_DUTY = 7.5
ESC_FULL_FORWARD = 10.0
ESC_FULL_REVERSE = 5.0

# === PCA9685 PWM for Servo ===
SERVO_CHANNEL = 15
SERVO_FREQ = 50
SERVO_NEUTRAL = 307  # 1.5ms pulse width

# === ESC Pulse Range (PCA-style, 0–4095) ===
ESC_MIN_PULSE = 205
ESC_MAX_PULSE = 410

# === Setup GPIO for ESC control ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(ESC_GPIO_PIN, GPIO.OUT)
esc_pwm = GPIO.PWM(ESC_GPIO_PIN, ESC_PWM_FREQ)
esc_pwm.start(ESC_NEUTRAL_DUTY)

# === Setup I2C and PCA9685 for Servo ===
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = SERVO_FREQ

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
    # Convert normalized value (0.0 to 1.0) to PWM duty %
    duty = ESC_FULL_REVERSE + (value * (ESC_FULL_FORWARD - ESC_FULL_REVERSE))
    esc_pwm.ChangeDutyCycle(duty)
    return duty

# === Servo Control ===
def set_servo_position(value):
    # Convert normalized value (-1.0 to 1.0) to pulse width
    pulse = int(SERVO_NEUTRAL + (value * 100))  # ±100 around neutral
    pca.channels[SERVO_CHANNEL].duty_cycle = pulse
    return pulse

# === Main Control Loop ===
def control_loop():
    joystick = init_controller()
    last_snapshot = None

    try:
        while True:
            pygame.event.pump()

            # Throttle (RT Trigger, axis 5): -1 to 1 → 0 to 1
            throttle_raw = joystick.get_axis(5)
            throttle_norm = max(0.0, (throttle_raw + 1) / 2)
            throttle_duty = set_esc_throttle(throttle_norm)

            # Steering (LT Trigger or axis 2): -1 to 1
            steering_raw = joystick.get_axis(2)
            servo_pulse = set_servo_position(steering_raw)

            if joystick.get_button(0):  # A Button
                last_snapshot = (throttle_duty, servo_pulse)

            # Display live status
            status = f"Throttle: {throttle_duty:.2f}% | Steering Pulse: {servo_pulse}"
            if last_snapshot:
                status += f" | 🔸 Snapshot (A): {last_snapshot}"
            print(f"\r{status.ljust(80)}", end="", flush=True)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping ESC and Servo...")
    finally:
        esc_pwm.ChangeDutyCycle(ESC_NEUTRAL_DUTY)
        esc_pwm.stop()
        GPIO.cleanup()
        pca.deinit()

if __name__ == "__main__":
    control_loop()
