"""NavDriver: the autopilot chassis, with the navigator choosing the command.

It is driver.Driver underneath, the way policy.PolicyDriver is, so the
safety layer the rules and the learned policy sit on is under this too, off
the same code:

  STUCK           Driver.stuck() off the IMU, on a forward this driver
                  commanded: neutral, and a fresh plan with the spot marked
                  solid in front of the bumper (Navigator.blocked_ahead)
  the floor       driver.ForwardFloorVeto off the camera's footprint scan:
                  a forward inside FORWARD_FLOOR_MM is held at 0 (the 200 mm
                  floor pushed out by the picture's age); held STANDOFF_S
                  in a row, the spot is marked solid and the path planned
                  round it
  the rear stop   driver.rear_blocked() off the roof lidar: a reverse with
                  something inside REAR_STOP_CLEAR_MM of the bumper is
                  neutral, never just not commanded
  reverse speed   every reverse goes out through auto_reverse_throttle(),
                  so at ESCAPE_SPEED
  no frame        neutral

No back-out. The way out of a spot is a path on the map, planned forward
and reverse (a three-point turn is a plan), never the hand-written reverse
at lock: the navigator's "no progress" plans again rather than asking for
one (navigator.Params.backout). One that is running anyway -- the policy's,
handed over mid-way by pointnav -- still runs to its end first.

My hand. With either group on the pad the navigator is not followed: its
path is dropped (Navigator.hand), its own groups are neutral, and the tick
after I let go plans from wherever I left the truck. A held goal stays held.

What it writes is a request into the controller it is handed -- on the car,
main.DriverCommand, which AutoPilot.step() mixes with my hand and writes at
most once a pass. Nothing here touches pigpio.

The map comes from `map_source`, anything with view() -> MapView or None
(on the car nav/live.SlamMap, the live SLAM), and optionally local() ->
LocalMap. Goals come from `goals`, anything with take() -> click.ClickGoal,
or from set_goal() (pointnav.PointNavDriver hands them over that way). No map
source is no NavDriver: the constructor refuses, so --nav without a map is
auto unavailable, never a truck driving blind.
"""

import math
import time

from crawler.autopilot import driver as dv
from crawler.nav import navigator as nv
from crawler.nav import truck as tk

# The least forward throttle a moving command goes out at. MEASURED: under
# about 0.15 the truck does not move at all and 0.18 moves half the time;
# 0.25 is 0.22 m/s and always moves.
MIN_FORWARD = 0.25
STANDOFF_S = 1.0
# Where the floor's hold is marked, ahead of the pose: the forward floor's
# own distance off the lens (truck.FLOOR_REACH_M). It was that distance past
# the bumper, 28 mm further out than the veto reaches.
FLOOR_AHEAD_M = tk.FLOOR_REACH_M


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
                 lidar=None, goals=None):
        if map_source is None:
            raise ValueError("nav needs a map source, and there is none")
        super().__init__(controller, oakd, lidar=lidar)
        self.map_source = map_source
        # Anything with take() -> click.ClickGoal or None: on the car,
        # click.ClickGoals, the Foxglove listener. None, no clicks.
        self.goals = goals
        self.goals_taken = 0
        # The car plans on its own thread (navigator.ThreadPlanner), so a
        # search never holds up a decision.
        self.nav = (navigator if navigator is not None
                    else nv.Navigator(planner=nv.ThreadPlanner()))
        self.floor_veto = dv.ForwardFloorVeto(self.forward_speed)
        self.last_output = None
        self.last_request = (0.0, 0.0, 0.0)
        self.last_pose = None
        self.floor_held = False
        self._standoff_since = None
        self._was_recovering = False
        self.human_steering = False
        self.pad_split = None

    def set_goal(self, x, y, yaw=None):
        self.nav.set_goal(x, y, yaw)
        self.goals_taken += 1

    def _take_goal(self, view):
        """The newest Foxglove click, if one came, as the navigator's goal.
        A click in base_link needs a pose to place it by; with none it is
        dropped, said, and the truck keeps doing what it was."""
        if self.goals is None:
            return
        try:
            goal = self.goals.take()
        except Exception:
            goal = None
        if goal is None:
            return
        from crawler.nav import click
        placed = click.to_map(goal, None if view is None else view.pose)
        if placed is None:
            print(f"[nav] goal in frame {goal.frame_id!r} dropped: "
                  "no pose to place it by")
            return
        x, y, yaw = placed
        self.set_goal(x, y, yaw)
        print(f"[nav] goal ({x:.2f}, {y:.2f}) in the map")

    def abort(self):
        """Driver's abort, and the navigator forgets its goal."""
        super().abort()
        self.nav.stop()
        self._standoff_since = None
        self.last_request = (0.0, 0.0, 0.0)

    def close(self):
        """Stop the planner's thread. Never raises."""
        try:
            self.nav.planner.close()
        except Exception:
            pass

    def _neutral(self):
        try:
            self.controller.set_throttle(0)
        except Exception:
            pass
        try:
            self.controller.set_steering(0, 0)
        except Exception:
            pass
        self.last_request = (0.0, 0.0, 0.0)
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

        try:
            view = self.map_source.view()
        except Exception:
            view = None
        if view is not None:
            self.last_pose = view.pose
        self._take_goal(view)

        if self.human_throttle or self.human_steering:
            # My hand: the navigator waits, and plans afresh when I let go.
            self._throttle_since = None
            self._still_samples = 0
            self._standoff_since = None
            self.nav.hand(now)
            self.last_output = nv.Output(0, 0.0, 0.0, False, self.nav.status,
                                         self.nav.reason)
            self._neutral()
            self.floor_veto.note_throttle(0.0, now)
            return
        if self.stuck(imu, now):
            self.last_recovery_time = now
            print("STUCK - neutral, and a fresh plan round it.")
            if self.last_request[0] > 0.0 and self.last_pose is not None:
                self.nav.blocked_ahead(now, self.last_pose, tk.FRONT_M)
            self.nav.recovered(now)
            self._neutral()
            self.floor_veto.note_throttle(0.0, now)
            return

        local = getattr(self.map_source, "local", None)
        local = local() if callable(local) else None
        out = self.nav.tick(now, view, local)
        self.last_output = out

        if out.recover:
            # Only with navigator.Params(backout=True); off by default.
            self.last_recovery_time = now
            print("No progress - backing up to recover.")
            self.begin_recover(now, dv.BACKOUT_STUCK)
            return
        if out.direction == 0 and (self.nav.holding or out.status in (
                nv.ST_IDLE, nv.ST_NO_MAP)):
            # Holding at a goal, no goal, or no pose to trust: neutral, the
            # steering centred too, for as long as it lasts.
            self._standoff_since = None
            self._neutral()
            self.floor_veto.note_throttle(0.0, now)
            return

        throttle, front, rear = request_for(out, self.forward_speed, self.max_turn)
        self.floor_held = False
        if throttle > 0.0 and self.floor_veto.vetoes():
            throttle = 0.0
            self.floor_held = True
            if self._standoff_since is None:
                self._standoff_since = now
            if now - self._standoff_since >= STANDOFF_S - 1e-3:
                self._standoff_since = None
                print("Standoff with the floor - planning round it.")
                if self.last_pose is not None:
                    self.nav.blocked_ahead(now, self.last_pose,
                                           FLOOR_AHEAD_M)
        else:
            self._standoff_since = None

        if throttle < 0.0 and dv.rear_blocked(self.rear_clearance):
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
