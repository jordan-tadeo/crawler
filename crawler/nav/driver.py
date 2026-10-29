"""NavDriver: the autopilot chassis, with the navigator choosing the command.

It is driver.Driver underneath, the way policy.PolicyDriver is, so every
safety layer the rules and the learned policy sit on is under this too, in
the same order and off the same code:

  the back-out    Driver.recovering() runs first, every decision; while it
                  runs the navigator is not asked anything
  STUCK           Driver.stuck() off the IMU, on a forward this driver
                  commanded: the back-out
  the floor       driver.ForwardFloorVeto off the camera's footprint scan:
                  a forward inside FORWARD_FLOOR_MM is held at 0 (the 200 mm
                  floor pushed out by the picture's age); held STANDOFF_S
                  in a row, the back-out
  the rear stop   driver.rear_blocked() off the roof lidar: a reverse with
                  something inside REAR_STOP_CLEAR_MM of the bumper is
                  neutral, never just not commanded
  reverse speed   every reverse goes out through auto_reverse_throttle(),
                  so at ESCAPE_SPEED

The navigator's own "no progress" asks for the same back-out, and when a
back-out ends the navigator is told (Navigator.recovered) and plans again
from wherever it left the truck.

What it writes is a request into the controller it is handed -- on the car,
main.DriverCommand, which AutoPilot.step() mixes with my hand and writes at
most once a pass. Nothing here touches pigpio.

The map comes from `map_source`, anything with view() -> MapView or None
(the live SLAM, later), and optionally local() -> LocalMap. No map source is
no NavDriver: the constructor refuses, so --nav without a map is auto
unavailable, never a truck driving blind.
"""

import math
import time

import numpy as np

from crawler.autopilot import driver as dv
from crawler.nav import navigator as nv
from crawler.nav import truck as tk

# The least forward throttle a moving command goes out at. MEASURED: under
# about 0.15 the truck does not move at all and 0.18 moves half the time;
# 0.25 is 0.22 m/s and always moves.
MIN_FORWARD = 0.25
STANDOFF_S = 1.0


def request_for(out, forward_speed, max_turn=tk.MAX_TURN):
    """(throttle, front, rear) for a navigator Output. Pure.

    Forward is `forward_speed` scaled by the output's speed, never under
    MIN_FORWARD; reverse is always the driver's ESCAPE_SPEED. Steering is the
    measured curve's inverse (truck.front_for_curvature), counter-phase
    through driver.steering_pair, so the rear can never exceed the front.
    A stop keeps the wheels where the curvature says, at 0 throttle."""
    front = tk.front_for_curvature(out.curvature, max_turn)
    if out.direction > 0:
        throttle = max(MIN_FORWARD, float(forward_speed) * float(out.speed))
        throttle = min(throttle, float(forward_speed))
    elif out.direction < 0:
        throttle = dv.auto_reverse_throttle(-1.0)
    else:
        throttle = 0.0
    f, r = dv.steering_pair(front)
    return float(throttle), float(f), float(r)


class NavDriver(dv.Driver):
    """driver.Driver with the navigator deciding. See the module docstring."""

    def __init__(self, controller, oakd, map_source, navigator=None,
                 lidar=None):
        if map_source is None:
            raise ValueError("nav needs a map source, and there is none")
        super().__init__(controller, oakd, lidar=lidar)
        self.map_source = map_source
        self.nav = navigator if navigator is not None else nv.Navigator()
        self.floor_veto = dv.ForwardFloorVeto(self.forward_speed)
        self.last_output = None
        self.last_request = (0.0, 0.0, 0.0)
        self.floor_held = False
        self._standoff_since = None
        self._was_recovering = False
        self.human_steering = False
        self.pad_split = None

    def _neutral(self):
        try:
            self.controller.set_throttle(0)
        except Exception:
            pass
        try:
            self.controller.set_steering(0, 0)
        except Exception:
            pass
        self._clear_motion_evidence()

    def tick(self, now=None):
        if now is None:
            now = time.monotonic()
        try:
            self._tick(now)
        except Exception:
            self._neutral()
            raise

    def _tick(self, now):
        self.rear_stop = False
        if self.recovering(now):
            self.floor_veto.observe(self.last_scan, now)
            self.floor_veto.note_throttle(self._backout_throttle, now)
            self._was_recovering = True
            return
        if self._was_recovering:
            self._was_recovering = False
            self.nav.recovered(now)
        self.read_rear()

        depth = self.camera.get_depth_frame()
        if depth is None or getattr(depth, "ndim", 0) != 2:
            self._neutral()
            return
        imu = self.camera.get_imu_sample()
        scan = self.scan_footprint(depth)
        self.floor_veto.forward_speed = self.forward_speed
        self.floor_veto.observe(scan, now)

        if self.human_throttle:
            self._throttle_since = None
            self._still_samples = 0
        elif self.stuck(imu, now):
            self.last_recovery_time = now
            print("STUCK - backing up to recover.")
            self.begin_recover(now, dv.BACKOUT_STUCK)
            return

        try:
            view = self.map_source.view()
        except Exception:
            view = None
        local = getattr(self.map_source, "local", None)
        local = local() if callable(local) else None
        out = self.nav.tick(now, view, local)
        self.last_output = out

        if out.recover:
            self.last_recovery_time = now
            print("No progress - backing up to recover.")
            self.begin_recover(now, dv.BACKOUT_STUCK)
            return

        throttle, front, rear = request_for(out, self.forward_speed, self.max_turn)
        self.floor_held = False
        if throttle > 0.0 and self.floor_veto.vetoes():
            throttle = 0.0
            self.floor_held = True
            if not self.human_throttle:
                if self._standoff_since is None:
                    self._standoff_since = now
                if now - self._standoff_since >= STANDOFF_S - 1e-3:
                    self._standoff_since = None
                    print("Standoff with the floor - backing up to recover.")
                    self.begin_recover(now, dv.BACKOUT_STANDOFF)
                    return
        else:
            self._standoff_since = None

        if throttle < 0.0 and not self.human_throttle and dv.rear_blocked(self.rear_clearance):
            print(f"Rear stop - {self.rear_clearance:.0f} mm behind me, "
                  "cutting the reverse.")
            self.rear_stop = True
            self._neutral()
            self.floor_veto.note_throttle(0.0, now)
            return

        self.controller.set_steering(front, rear)
        self.controller.set_throttle(throttle)
        self.floor_veto.note_throttle(throttle, now)
        self.last_request = (throttle, front, rear)
        if throttle > 0.0:
            if self._throttle_since is None:
                self._throttle_since = now
            self._note_turn(front, now)
        else:
            self._clear_motion_evidence()
