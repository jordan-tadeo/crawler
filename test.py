import time
import board
import busio
from adafruit_pca9685 import PCA9685

i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

channel = 15

# Full throttle (2.0ms = ~410)
print("Step 1: Power OFF the ESC now if it's on.")
print("Step 2: Starting calibration.")
print("Step 3: Plug in ESC power NOW.")
pca.channels[channel].duty_cycle = 410
print("Sending FULL THROTTLE (410)")
time.sleep(10)

# Full brake (1.0ms = ~205)
pca.channels[channel].duty_cycle = 205
print("Sending FULL BRAKE (205)")
time.sleep(5)

# Neutral (1.5ms = ~307)
pca.channels[channel].duty_cycle = 307
print("Sending NEUTRAL (307)")
time.sleep(5)

print("Calibration complete. ESC should be armed.")
