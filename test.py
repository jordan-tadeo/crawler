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