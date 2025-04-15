import pygame
import RPi.GPIO as GPIO
import time
import sys

# === Constants ===
SERVO_PIN = 18      # GPIO pin connected to ESC/servo signal wire
PWM_FREQ = 50       # 50Hz is standard for servos and ESCs

# === GPIO Setup ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, PWM_FREQ)
pwm.start(7)  # Neutral position

# === Controller Setup ===
pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()

print("🎮 Controller connected. Use RT trigger to control throttle.")
print("")  # For second line display

last_snapshot = None  # NEW

try:
    while True:
        pygame.event.pump()

        # Read RT trigger (usually axis 5 on Xbox controllers)
        rt_value = joystick.get_axis(2)  # Range: -1.0 to 1.0
        throttle = (rt_value + 1) / 2    # Normalize to 0.0 to 1.0
        duty = 4 + (throttle * 5)    # Scale to 0–10% PWM

        pwm.ChangeDutyCycle(duty)

        # Detect A button press (button 0)
        if joystick.get_button(0):
            last_snapshot = duty  # NEW

        # Print live line + frozen snapshot line
        print(f"\rRT: {throttle:.2f} → PWM: {duty:.2f}", end="", flush=True)  # overwrite
        if last_snapshot is not None:
            sys.stdout.write(f"🔸 Snapshot (A): PWM was {last_snapshot:.2f}     \r")
        else:
            sys.stdout.write("                                         \r")  # Clear line

        sys.stdout.flush()
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping... Resetting servo.")
finally:
    pwm.ChangeDutyCycle(7.5)  # Back to neutral
    pwm.stop()
    GPIO.cleanup()
