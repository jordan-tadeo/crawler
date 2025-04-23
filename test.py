import RPi.GPIO as GPIO
import time

ESC_PIN = 18  # GPIO18 = pin 12 (supports hardware PWM)
PWM_FREQ = 50  # 50Hz for ESC input

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(ESC_PIN, GPIO.OUT)

pwm = GPIO.PWM(ESC_PIN, PWM_FREQ)
pwm.start(0)  # Start with 0% duty

def set_throttle(pulse_ms):
    # Convert ms pulse width to 0–100% duty cycle at 50Hz
    duty = (pulse_ms / 20.0) * 100
    print(f"Setting throttle: {pulse_ms}ms -> {duty:.2f}% duty")
    pwm.ChangeDutyCycle(duty)

try:
    print("Arming ESC with neutral signal...")
    set_throttle(1.5)  # neutral
    time.sleep(5)

    print("Sending throttle...")
    set_throttle(1.7)  # slight forward
    time.sleep(3)

    print("Back to neutral")
    set_throttle(1.5)
    time.sleep(2)

    print("Full stop")
    set_throttle(1.0)
    time.sleep(2)

finally:
    pwm.stop()
    GPIO.cleanup()
