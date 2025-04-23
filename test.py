import RPi.GPIO as GPIO
import time

ESC_PIN = 18       # GPIO18, physical pin 12
PWM_FREQ = 50      # 50Hz is standard for ESCs

GPIO.setmode(GPIO.BCM)
GPIO.setup(ESC_PIN, GPIO.OUT)

pwm = GPIO.PWM(ESC_PIN, PWM_FREQ)
pwm.start(7.5)  # 1.5ms pulse = neutral (1.5 / 20 * 100)

try:
    print("Neutral signal (7.5%)... give ESC time to arm")
    time.sleep(5)

    print("Throttle up (8.0%)")
    pwm.ChangeDutyCycle(8.0)
    time.sleep(3)

    print("Throttle up more (8.5%)")
    pwm.ChangeDutyCycle(8.5)
    time.sleep(3)

    print("Back to neutral (7.5%)")
    pwm.ChangeDutyCycle(7.5)
    time.sleep(2)

    print("Full stop / brake (5.0%)")
    pwm.ChangeDutyCycle(5.0)
    time.sleep(3)

    print("Neutral again")
    pwm.ChangeDutyCycle(7.5)
    time.sleep(2)

except KeyboardInterrupt:
    print("Interrupted, stopping...")

finally:
    pwm.stop()
    GPIO.cleanup()
    print("Cleaned up GPIO")
