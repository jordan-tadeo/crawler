import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import time

i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

pan_servo = servo.Servo(pca.channels[15], actuation_range=180)
tilt_servo = servo.Servo(pca.channels[14], actuation_range=180)

# Send both to center
print("Centering servos to 90°...")
pan_servo.angle = 90
tilt_servo.angle = 90

time.sleep(5)
pca.deinit()
print("Done.")
