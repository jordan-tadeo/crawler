import pygame
import RPi.GPIO as GPIO
import time

# === Constants ===
SERVO_PIN = 18      # GPIO pin connected to ESC/servo signal wire
PWM_FREQ = 50       # 50Hz is standard for servos and ESCs

# === GPIO Setup ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, PWM_FREQ)
pwm.start(0)  # Neutral position

# === Controller Setup ===
pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()

print("🎮 Controller connected. Use RT trigger to control throttle.")

def sweep_duty_cycle():
    for duty in range(0, 101, 5):
        pwm.ChangeDutyCycle(duty)
        print(f"\rPWM: {duty:.2f}", end="", flush=True)
        time.sleep(1)

try:
    while True:
        pygame.event.pump()

        # # Read RT trigger (usually axis 5 on Xbox controllers)
        # rt_value = joystick.get_axis(5)  # Range: -1.0 to 1.0
        # throttle = (rt_value + 1) / 2    # Normalize to 0.0 to 1.0
        # duty = (throttle * 100)    # Map to PWM: 7.5 (neutral) → 10 (full)

        # pwm.ChangeDutyCycle(duty)

        # # ✅ Clean one-line console output
        # print(f"\rRT: {throttle:.2f} → PWM: {duty:.2f}", end="", flush=True)
        # time.sleep(0.05)

        sweep_duty_cycle()

except KeyboardInterrupt:
    print("\nStopping... Resetting servo.")
finally:
    pwm.ChangeDutyCycle(7.5)  # Back to neutral
    pwm.stop()
    GPIO.cleanup()
