from typing import Tuple
import Logger

# pigpio, board, busio, adafruit_pca9685 and adafruit_motor are imported
# inside __init__, not here. None of them exist off the Pi, and at module
# scope they made this file impossible to even import on my laptop -- which
# meant the pulse-width arithmetic in set_throttle and the angle maths in the
# steering setters had never run anywhere but on the car. Constructing a
# VehicleController still needs every one of them, so nothing changes on the
# Pi; it is only reading the file that got cheaper.

# Constants for control of the Quicrun 880 ESC.
#
# These are my bench calibration and nothing in this repo gets to change
# them. There is a second set of candidates sitting on the experimental
# branch -- neutral 1350 (and 1325 in its main.py), full forward 1850, full
# reverse 1100, plus an ESC_MIN_FORWARD_PW of 1500 -- which is why I pulled
# only oakd.py and driver.py off that branch instead of merging it: a merge
# would have taken 1350 with no conflict and parked the car 225 us into
# reverse on every "neutral", including the shutdown path. Which set is
# right is a question for the ESC with a wheel off the ground, so until I
# have armed it and watched it, 1575 stands.
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

    def __init__(self, logger: Logger.Logger) -> None:
        import pigpio
        import board
        import busio
        from adafruit_pca9685 import PCA9685
        from adafruit_motor import servo

        self.logger = logger
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

        # Disarm before anything else gets a chance to command. pigpio latches
        # a pulse width and keeps emitting it, so if the last process was
        # SIGKILLed mid-throttle the ESC is still being fed that value right
        # now -- and it will be until something overwrites it. Constructing
        # this object is the first moment I can.
        self.return_neutral()

    def get_state(self) -> dict:
        ''' 
        Get the current state of the ESC and servos.
        Returns a tuple of (throttle, front_steering, rear_steering, pan, tilt).
        '''

        try:
            throttle = self.pi.get_servo_pulsewidth(ESC_GPIO_PIN)
        except Exception as e:
            self.logger.log("error", "Error reading state", f"Error reading throttle: {e}")
            throttle = None

        try:
            front_steering = self.front_steering_servo.angle
        except Exception as e:
            self.logger.log("error", "Error reading state", f"Error reading front steering: {e}")
            front_steering = None

        try:
            rear_steering = self.rear_steering_servo.angle
        except Exception as e:
            self.logger.log("error", "Error reading state", f"Error reading rear steering: {e}")
            rear_steering = None

        try:
            pan = self.pan_servo.angle
        except Exception as e:
            self.logger.log("error", "Error reading state", f"Error reading pan angle: {e}")
            pan = None

        try:
            tilt = self.tilt_servo.angle
        except Exception as e:
            self.logger.log("error", "Error reading state", f"Error reading tilt angle: {e}")
            tilt = None

        return {"throttle": throttle,
                "front_s": front_steering,
                "rear_s": rear_steering,
                "pan": pan,
                "tilt": tilt}
    
    def set_throttle(self, value) -> int:
        ''' 
        Set the throttle for the ESC.
        '''

        # Restrict value to the range [-1, 1]
        value = max(-1.0, min(1.0, value)) if value else 0

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
        front = max(-1.0, min(1.0, -front)) if front else 0
        rear = max(-1.0, min(1.0, -rear)) if rear else 0
        
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
        Stop the ESC, then centre the servos, then let the hardware go.

        The ESC goes first and on its own. return_neutral() touches the I2C
        bus for four servos before it is done, and one I2C error in there used
        to throw straight past set_servo_pulsewidth(pin, 0) -- so the one step
        that actually stops the car got skipped because a servo I do not even
        have attached did not answer. Every step is on its own now and a
        failure in one cannot cost me the next.
        '''

        try:
            self.pi.set_servo_pulsewidth(ESC_GPIO_PIN, ESC_NEUTRAL_PW)
        except Exception as e:
            print(f"[Shutdown] could not neutral the ESC: {e}")

        try:
            self.pi.set_servo_pulsewidth(ESC_GPIO_PIN, 0)
        except Exception as e:
            print(f"[Shutdown] could not stop the ESC pulse: {e}")

        for name, s in (("front steering", self.front_steering_servo),
                        ("rear steering", self.rear_steering_servo),
                        ("pan", self.pan_servo),
                        ("tilt", self.tilt_servo)):
            try:
                s.angle = 90
            except Exception as e:
                print(f"[Shutdown] could not centre {name}: {e}")

        for name, release in (("pigpio", self.pi.stop),
                              ("pca9685", self.pca.deinit),
                              ("i2c", self.i2c.deinit)):
            try:
                release()
            except Exception as e:
                print(f"[Shutdown] could not release {name}: {e}")

