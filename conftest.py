"""Put the checkout on sys.path so the tests can import the car's modules.

This project is a flat pile of modules driven by `python main.py` -- main.py
does `import Joystick as js` and `import VehicleController as vc`, and the
systemd unit runs it with WorkingDirectory set to the checkout, so those names
resolve off the current directory. There is no package and `tool.uv.package` is
false on purpose.

pytest prepends the directory holding this conftest.py to sys.path, which is
exactly the checkout root, so `import driver` works from inside tests/ with no
src layout, no installed package and no PYTHONPATH to remember.

Nothing in here imports anything off the car. pigpio, depthai, PyQt5 and a
plugged-in pad are all absent from the laptop, which is where these run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
