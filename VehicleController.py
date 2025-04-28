import pigpio
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from typing import Tuple

# Constants for control of the Quicrun 880 ESC
ESC_GPIO_PIN = 18
ESC_NEUTRAL_PW = 1575
ESC_FULL_FORWARD_PW = 2000
ESC_FULL_REVERSE_PW = 1000

# Constants for PCA9685 PWM Channels (Servos)
PAN_CHANNEL = 12
TILT_CHANNEL = 13
FRONT_STEERING_CHANNEL = 15
REAR_STEERING_CHANNEL = 14
SERVO_FREQ = 50

class VehicleController:
    ''' 
    This class defines an interface for controlling 
    the ESC and servos of an RC vehicle. The "set_" methods
    expect values between -1 and 1, where 0 is neutral.
    '''

    def __init__(self):
        # Setup pigpio for ESC control
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon not running")
        self.pi.set_mode(ESC_GPIO_PIN, pigpio.OUTPUT)

        # Setup I2C and PCA9685
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c)
        self.pca.frequency = SERVO_FREQ

        # Initialize servos
        self.pan_servo = servo.Servo(self.pca.channels[PAN_CHANNEL], actuation_range=180)
        self.tilt_servo = servo.Servo(self.pca.channels[TILT_CHANNEL], actuation_range=180)
        self.front_steering_servo = servo.Servo(self.pca.channels[FRONT_STEERING_CHANNEL], actuation_range=180)
        self.rear_steering_servo = servo.Servo(self.pca.channels[REAR_STEERING_CHANNEL], actuation_range=180)
    
    def set_throttle(self, value) -> int:
        ''' 
        Set the throttle for the ESC.
        '''

        # Restrict value to the range [-1, 1]
        value = max(-1.0, min(1.0, value))

        if value > 0:
            pulse = int(ESC_NEUTRAL_PW + value * (ESC_FULL_FORWARD_PW - ESC_NEUTRAL_PW))
        elif value < 0:
            pulse = int(ESC_NEUTRAL_PW - abs(value) * (ESC_NEUTRAL_PW - ESC_FULL_REVERSE_PW))
        else:
            pulse = ESC_NEUTRAL_PW
        self.pi.set_servo_pulsewidth(ESC_GPIO_PIN, pulse)
        return pulse
    
    def set_steering(self, front, rear) -> Tuple[int, int]:
        ''' 
        Set the steering angle for the servos.
        '''

        # Restrict values to the range [-1, 1]
        front = max(-1.0, min(1.0, front))
        rear = max(-1.0, min(1.0, rear))
        
        # Map the range [-1, 1] to [0, 180] for servo angle
        front_angle = int((front + 1.0) * 90)
        rear_angle = int((rear + 1.0) * 90)
        
        self.front_steering_servo.angle = front_angle
        self.rear_steering_servo.angle = rear_angle
        
        return front_angle, rear_angle
    
    def set_pan_tilt(self, pan, tilt) -> Tuple[int, int]:
        ''' 
        Set the pan and tilt angles for the servos.
        '''

        # Restrict values to the range [-1, 1]
        pan = max(-1.0, min(1.0, pan))
        tilt = max(-1.0, min(1.0, tilt))
        
        # Map the range [-1, 1] to [0, 180] for servo angle
        pan_angle = int((pan + 1.0) * 90)
        tilt_angle = int((1.0 - tilt) * 90)
        
        self.pan_servo.angle = pan_angle
        self.tilt_servo.angle = tilt_angle
        
        return pan_angle, tilt_angle
    
    def return_neutral(self) -> None:
        ''' 
        Set the ESC and servos to neutral position.
        '''

        self.pi.set_servo_pulsewidth(ESC_GPIO_PIN, ESC_NEUTRAL_PW)
        self.pan_servo.angle = 90
        self.tilt_servo.angle = 90
        self.front_steering_servo.angle = 90
        self.rear_steering_servo.angle = 90
    
    def close(self) -> None:
        ''' 
        Set neutral position and cleanup resources.
        '''
    
        self.return_neutral()
        self.pi.set_servo_pulsewidth(ESC_GPIO_PIN, 0)
        self.pi.stop()
        self.pca.deinit()
        self.i2c.deinit()

    