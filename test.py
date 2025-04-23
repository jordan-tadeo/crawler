import RPi.GPIO as GPIO
import time

ESC_PIN = 18
PWM_FREQ = 50

GPIO.setmode(GPIO.BCM)
GPIO.setup(ESC_PIN, GPIO.OUT)

pwm = GPIO.PWM(ESC_PIN, PWM_FREQ)
pwm.start(7.5)  # Neutral: 1.5ms = (1.5 / 20) * 100 = 7.5%

try:
    print("Holding neutral signal at 7.5% duty cycle...")
    while True:
        time.sleep(1)  # Just keep the script alive
except KeyboardInterrupt:
    print("Stopping...")
finally:
    pwm.stop()
    GPIO.cleanup()
