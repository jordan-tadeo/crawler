# Hand-run bench scratchpad, not a test: importing this file commands a servo.
# It also drives pca.channels[0], which is none of the four channels this
# vehicle actually uses (12 pan, 13 tilt, 14 rear steering, 15 front
# steering), so it proves nothing about the car.
from adafruit_pca9685 import PCA9685
import board
import busio
from adafruit_motor import servo

i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

test_servo = servo.Servo(pca.channels[0])
test_servo.angle = 90  # Move to neutral position
print("Servo moved to 90 degrees")