import pygame

XBOX_BUTTONS = {
    "A": 0,
    "B": 1,
    "X": 2,
    "Y": 3,
    "LB": 4,
    "RB": 5,
    "BACK": 6,
    "START": 7,
    "LS": 8,
    "RS": 9
}

XBOX_AXES = {
    "LEFT_X": 0,
    "LEFT_Y": 1,
    "LT": 5,
    "RIGHT_X": 2,
    "RIGHT_Y": 3,
    "RT": 4
}

class Joystick:
    def __init__(self):
        if not pygame.joystick.get_init():
            pygame.joystick.init()

        self.type = type
        self.connected = False

        if pygame.joystick.get_count() == 0:
            self.wait_for_connection()

        try:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.connected = True
        except pygame.error as e:
            raise RuntimeError(f"Failed to initialize joystick: {e}")
    
    def read_throttle(self) -> float:
        deadzone = 0.01

        # translate RT and LT to one throttle value between -1 and 1
        rt = self.get_axis("RT")
        lt = self.get_axis("LT")

        if rt + 1 > deadzone:
            # return rt mapped to [0, 1]
            rt = (rt + 1) / 2
            return rt
        elif lt + 1 > deadzone:
            # return lt mapped to [-1, 0]
            lt = (lt + 1) / 2
            return -lt


    def wait_for_connection(self):
        while not self.connected:
            pygame.event.pump()
            print("No joystick connected. Waiting...", end="\r", flush=True)
            if pygame.joystick.get_count() > 0:
                try:
                    self.joystick = pygame.joystick.Joystick(0)
                    self.joystick.init()
                    self.connected = True
                except pygame.error as e:
                    raise RuntimeError(f"Failed to initialize joystick: {e}")
            pygame.time.delay(1000)

    def is_connected(self):
        return self.connected
    
    def get_axis(self, axis_name: str):
        if axis_name in XBOX_AXES:
            return self.joystick.get_axis(XBOX_AXES[axis_name])
        else:
            raise ValueError(f"Invalid axis name: {axis_name}")
    
    def get_button(self, button_name: str):
        if button_name in XBOX_BUTTONS:
            return self.joystick.get_button(XBOX_BUTTONS[button_name])
        else:
            raise ValueError(f"Invalid button name: {button_name}")
    
    def update_connection_status(self):
        # Check if the joystick is still connected
        self.connected = pygame.joystick.get_count() > 0