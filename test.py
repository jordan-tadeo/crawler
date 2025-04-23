import RPi.GPIO as GPIO
import time

ESC_PIN = 18
PWM_FREQ = 50
GPIO.setmode(GPIO.BCM)
GPIO.setup(ESC_PIN, GPIO.OUT)

pwm = GPIO.PWM(ESC_PIN, PWM_FREQ)
pwm.start(7.5)  # Neutral

try:
    print("Neutral (7.5%)")
    time.sleep(3)

    print("Full throttle (10.0%)")
    pwm.ChangeDutyCycle(10.0)
    time.sleep(3)

    print("Neutral (7.5%)")
    pwm.ChangeDutyCycle(7.5)
    time.sleep(3)

    print("Full reverse (5.0%)")
    pwm.ChangeDutyCycle(5.0)
    time.sleep(3)

    print("Neutral again")
    pwm.ChangeDutyCycle(7.5)
    time.sleep(3)

finally:
    pwm.stop()
    GPIO.cleanup()
