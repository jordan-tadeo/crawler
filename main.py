import pygame

pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()

while True:
    pygame.event.pump()
    left_trigger = joystick.get_axis(2)  # Axis number may vary
    right_trigger = joystick.get_axis(5)
    a_button = joystick.get_button(0)
    print(f"LT: {left_trigger:.2f}  RT: {right_trigger:.2f}  A: {a_button}")
