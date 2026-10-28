from typing import Tuple
import math
from crawler.telemetry import logger as Logger

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

        # Initialize servos. Building the pan and tilt objects costs no bus
        # traffic -- a Servo only writes when something sets .angle -- and
        # set_pan_tilt still needs them, so they stay as the seam for putting
        # the gimbal back on. Nothing on the live path touches them now.
        self.pan_servo = servo.Servo(self.pca.channels[PAN_CHANNEL], actuation_range=180)
        self.tilt_servo = servo.Servo(self.pca.channels[TILT_CHANNEL], actuation_range=180)
        self.front_steering_servo = servo.Servo(self.pca.channels[FRONT_STEERING_CHANNEL], actuation_range=180)
        self.rear_steering_servo = servo.Servo(self.pca.channels[REAR_STEERING_CHANNEL], actuation_range=180)

        # The last thing commanded, recorded as a side effect of commanding it.
        #
        # Read-only bookkeeping. Every one of these is a number the setters
        # below have already worked out and are about to write anyway, so
        # keeping it costs nothing -- and it means telemetry can report what
        # was asked for without asking the hardware what it got. Asking is the
        # expensive direction: get_state() is a pigpio round trip and two I2C
        # reads, which is exactly why the control loop only polls it once a
        # second.
        #
        # Nothing reads these to make a driving decision and nothing writes
        # hardware off the back of them. There is still exactly one writer to
        # the ESC and the servos, and it is still this class.
        self.last_throttle = 0.0
        self.last_esc_pulse = ESC_NEUTRAL_PW
        self.last_steering_front = 0.0
        self.last_steering_rear = 0.0
        self.last_front_angle = 90
        self.last_rear_angle = 90

        # One latch per channel, so a stream of bad commands says so once
        # instead of once per tick. See _usable().
        self._rejected = {}

        # Disarm before anything else gets a chance to command. pigpio latches
        # a pulse width and keeps emitting it, so if the last process was
        # SIGKILLed mid-throttle the ESC is still being fed that value right
        # now -- and it will be until something overwrites it. Constructing
        # this object is the first moment I can.
        self.return_neutral()

    def get_state(self) -> dict:
        '''
        Get the current state of the ESC and the steering.
        Returns a dict of throttle, front_s and rear_s.

        Pan and tilt are not in here. The gimbal is off the vehicle, so
        reading its two angles was two I2C round trips a second spent asking
        hardware that is not bolted on. Put them back here if it goes back on.
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

        return {"throttle": throttle,
                "front_s": front_steering,
                "rear_s": rear_steering}
    
    def _usable(self, value, channel) -> bool:
        '''
        Can this number be clamped into a command, or does the channel have to
        go neutral?

        False for anything falsy -- 0, 0.0, False, None -- which is what the
        `if value else 0` these guards replaced already did, and which is
        correct: all of those mean neutral.

        False for a nan or an infinity, which is the whole reason this exists.
        min() and max() do not clamp a nan. Every comparison against a nan is
        False, so `min(1.0, nan)` never swaps and hands back 1.0, and
        `max(-1.0, 1.0)` agrees -- a nan throttle came out of that clamp as a
        clean 1.0 and went to the ESC as 2000 us, full forward. The steering
        did the same thing and reached 180 degrees, full lock. An infinity got
        through the same hole on the throttle. Nothing upstream sends me a nan
        today; the joystick and the autonomous path both filter first. But
        1787 us once left here as arithmetic I had not run anywhere but on the
        car, and this is the same shape of mistake with a worse number at the
        end of it, so it is checked here, at the one place that writes the
        hardware, rather than trusted to every caller.

        Logged on the way down, because silently driving at neutral when I am
        asking for throttle is its own confusing afternoon. Logged once per
        run of bad values, not once per value: this sits in a 100 Hz loop and
        self.logger writes and flushes a file on the calling thread, so a
        sustained nan would mean hundreds of synchronous writes a second on
        the thread that commands the ESC. The latch clears as soon as the
        channel reports a number again, so a second episode is a second line.
        '''

        if not value:
            return False

        if math.isnan(value) or math.isinf(value):
            if not self._rejected.get(channel):
                self.logger.log("warning", "Rejected command",
                                f"{channel} was {value}, which is not a finite "
                                f"number -- commanding neutral instead")
            self._rejected[channel] = True
            return False

        self._rejected[channel] = False
        return True

    def set_throttle(self, value) -> int:
        '''
        Set the throttle for the ESC.
        '''

        # Restrict value to the range [-1, 1]
        value = max(-1.0, min(1.0, value)) if self._usable(value, "throttle") else 0

        if value > 0:
            pulse = int(ESC_NEUTRAL_PW + value * (ESC_FULL_FORWARD_PW - ESC_NEUTRAL_PW))
        elif value < 0:
            pulse = int(ESC_NEUTRAL_PW - abs(value) * (ESC_NEUTRAL_PW - ESC_FULL_REVERSE_PW))
        else:
            pulse = ESC_NEUTRAL_PW
        self.pi.set_servo_pulsewidth(ESC_GPIO_PIN, pulse)
        # After the write, not before. If pigpio raises, nothing was commanded
        # and the record should not claim otherwise.
        self.last_throttle = value
        self.last_esc_pulse = pulse
        return pulse
    
    def set_steering(self, front, rear) -> Tuple[int, int]:
        ''' 
        Set the steering angle for the servos.
        '''

        # What was asked for, before the sign flip below turns it into a servo
        # direction. This is the number the caller handed over, and it is the
        # one worth reporting: a telemetry channel that showed the inverted
        # value would read as a steering bug every time I looked at it.
        asked_front, asked_rear = front, rear

        # Asked once, read twice: once for the clamp and once for the record
        # below. Calling _usable again down there would log the same rejection
        # a second time.
        front_ok = self._usable(front, "front steering")
        rear_ok = self._usable(rear, "rear steering")

        # Restrict values to the range [-1, 1]
        front = max(-1.0, min(1.0, -front)) if front_ok else 0
        rear = max(-1.0, min(1.0, -rear)) if rear_ok else 0

        # Map the range [-1, 1] to [0, 180] for servo angle
        front_angle = int((front + 1.0) * 90)
        rear_angle = int((rear + 1.0) * 90)

        self.front_steering_servo.angle = front_angle
        self.rear_steering_servo.angle = rear_angle

        # A rejected ask is recorded as the neutral that was actually
        # commanded. Reporting the nan back out would put it on the telemetry
        # channels and into the CSV, where it reads as a steering fault
        # instead of a rejected input.
        self.last_steering_front = asked_front if front_ok else 0.0
        self.last_steering_rear = asked_rear if rear_ok else 0.0
        self.last_front_angle = front_angle
        self.last_rear_angle = rear_angle

        return front_angle, rear_angle
    
    def set_pan_tilt(self, pan, tilt) -> Tuple[int, int]:
        '''
        Set the pan and tilt angles for the servos.

        Nothing calls this. The pan/tilt gimbal is not on the vehicle any
        more, and its only callers were in PersonFollower, which is retired.
        I am keeping this method and PAN_CHANNEL/TILT_CHANNEL on disk on
        purpose: if the gimbal goes back on, this is the seam -- wire a caller
        to it, and put pan and tilt back into get_state and return_neutral.
        Until then the live path does not touch channels 12 and 13 at all.
        '''

        # Restrict values to the range [-1, 1]. Same nan hole as the throttle
        # and the steering -- this one is unreachable twice over, since nothing
        # calls it and the gimbal is off the car, but it is the third clamp of
        # that shape in this file and leaving one unguarded is how it comes
        # back when the gimbal does.
        pan = max(-1.0, min(1.0, pan)) if self._usable(pan, "pan") else 0
        tilt = max(-1.0, min(1.0, tilt)) if self._usable(tilt, "tilt") else 0
        
        # Map the range [-1, 1] to [0, 180] for servo angle
        pan_angle = int((pan + 1.0) * 90)
        tilt_angle = int((1.0 - tilt) * 90)
        
        self.pan_servo.angle = pan_angle
        self.tilt_servo.angle = tilt_angle
        
        return pan_angle, tilt_angle
    
    def return_neutral(self) -> None:
        '''
        Set the ESC and the steering to neutral position.

        This runs on every mode change, on every controller dropout and once
        at the end of __init__, so it is the hottest path in the class. Pan
        and tilt used to be centred in here too: two I2C writes to a gimbal
        that is not attached, every single time, each one able to raise.
        '''

        self.pi.set_servo_pulsewidth(ESC_GPIO_PIN, ESC_NEUTRAL_PW)
        self.front_steering_servo.angle = 90
        self.rear_steering_servo.angle = 90

        # This path writes the hardware directly instead of going through the
        # two setters, so it has to keep the record straight itself. It is set
        # last for the same reason the setters set it last: if one of the three
        # writes above raised, the car is not at neutral and this must not say
        # it is.
        self.last_throttle = 0.0
        self.last_esc_pulse = ESC_NEUTRAL_PW
        self.last_steering_front = 0.0
        self.last_steering_rear = 0.0
        self.last_front_angle = 90
        self.last_rear_angle = 90
    
    def close(self) -> None:
        '''
        Stop the ESC, then centre the servos, then let the hardware go.

        The ESC goes first and on its own. return_neutral() touches the I2C
        bus for the steering before it is done, and one I2C error in there
        used to throw straight past set_servo_pulsewidth(pin, 0) -- so the one
        step that actually stops the car got skipped because a servo I do not
        even have attached did not answer. Every step is on its own now and a
        failure in one cannot cost me the next. The two servos I do not have
        attached are no longer centred in here either.
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
                        ("rear steering", self.rear_steering_servo)):
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

