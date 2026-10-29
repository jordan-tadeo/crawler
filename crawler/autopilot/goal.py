"""Whether the car has got where it was going. The one place auto asks.

A learned policy's "stop" is only committed when this says the goal is
reached; until then PolicyDriver drives on with the better of forward and
reverse (policy.py, "The goal"). A plain Goal is never set, so it is always
False and the model never gets to give up in place.

The seam: AutoPilot owns one Goal (pilot.goal) and hands it to the
PolicyDriver it builds, which calls reached() once every decision. Whatever
decides the car has arrived calls set_reached(True) on that same object, or
replaces it with anything else that has a reached() method. ClickGoal is one.

This only ever stops the car. Every safety stop in the driver stops whatever
this says.

ClickGoal: a point clicked on the SLAM map
------------------------------------------
With --goal, Foxglove's Publish pose tool on the 3D panel sends a goal
(crawler/nav/click.py receives it). ClickGoal keeps the newest one, in the
map frame and on the map frame it was clicked on (egomap.Crop.frame), from
the moment it arrives in any mode until a new one replaces it: a click
before auto, or during it, is the goal the model drives to whenever auto
drives. A goal model (policy_goal.py) is fed it every decision as
encode(crop); nothing else in auto reads where it is. The model does all
of the driving.

Arriving is the one rule, by the owner's ruling a safety exception like the
rear stop: once the truck's map pose is within ARRIVE_M of the goal,
reached() is True and the driver holds neutral ("goal reached"), and it
stays True until a new goal is clicked that is more than ARRIVE_M from it.
It never steers. A goal on a map frame the truck is not on now (a reseed
since) is kept, never placed, and never arrives; the model drives without
it until the map it is on comes back (a merge), as it does with the map
lost.
"""

import math

from crawler.autopilot import policy_goal as pg


class Goal:
    """A goal flag, False until something sets it."""

    def __init__(self, reached=False):
        self._reached = bool(reached)

    def reached(self):
        return self._reached

    def set_reached(self, reached):
        self._reached = bool(reached)

    def encode(self, crop):
        """The goal input against this crop: none."""
        return pg.missing()

    def observe(self, crop):
        pass


class ClickGoal(Goal):
    """The newest clicked goal, where it is from the truck, and whether the
    truck has arrived. Control loop only: the Foxglove listener's thread
    only ever touches click.ClickGoals, which poll() drains."""

    def __init__(self, arrive_m=pg.ARRIVE_M, say=None):
        super().__init__()
        self.arrive_m = float(arrive_m)
        self._say = say
        self.goal = None            # (x, y, frame) on the map, or None
        self.yaw = None             # the click's heading, kept for the status only
        self.clicks = 0
        self.refused = 0
        self.last = None            # (distance m, bearing rad) off the newest crop
        self.on_map = False         # the newest crop is present and on the goal's frame
        self.frame_now = None

    def _log(self, level, text):
        print(text)
        if self._say is not None:
            try:
                self._say(level, "Goal", text)
            except Exception:
                pass

    # ---- in ------------------------------------------------------------------

    def set_goal(self, x, y, frame, yaw=None):
        """A new goal on the map. True if it is one; a click within arrive_m
        of the goal the truck has arrived at is the same goal, and the hold
        stays (False)."""
        x, y = float(x), float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            self.refused += 1
            return False
        self.clicks += 1
        if (self._reached and self.goal is not None and int(frame) == self.goal[2]
                and math.hypot(x - self.goal[0], y - self.goal[1]) <= self.arrive_m):
            return False
        self.goal = (x, y, int(frame))
        self.yaw = yaw
        self.last = None
        if self._reached:
            self._log("info", "New goal - the arrival hold is off, auto drives again")
        self._reached = False
        self._log("info", f"Goal set at ({x:.2f}, {y:.2f}) m on map {int(frame)}")
        return True

    def poll(self, clicks, crop):
        """Take the newest click off a click.ClickGoals (or None) and place
        it on the map off this crop's pose, then observe(crop)."""
        click = clicks.take() if clicks is not None else None
        if click is not None:
            self.take_click(click, crop)
        self.observe(crop)

    def take_click(self, click, crop):
        from crawler.nav import click as ck
        pose = None if crop is None else (crop.x, crop.y, crop.yaw)
        if crop is None:
            self.refused += 1
            self._log("warning", "A goal was clicked with no SLAM map yet; not taken")
            return False
        placed = ck.to_map(click, pose)
        if placed is None:
            self.refused += 1
            self._log("warning", f"A goal in frame {click.frame_id!r} cannot be put on "
                      "the map; set the 3D panel's display frame to map")
            return False
        return self.set_goal(placed[0], placed[1], int(getattr(crop, "frame", 0)), placed[2])

    # ---- each tick -------------------------------------------------------------

    def observe(self, crop):
        """Where the goal is from this crop's pose, and the arrival latch."""
        self.frame_now = None if crop is None else int(getattr(crop, "frame", 0))
        if self.goal is None or crop is None or not getattr(crop, "present", False):
            self.on_map = False
            return
        db = pg.distance_bearing(crop, self.goal)
        self.on_map = db is not None
        if db is None:
            self.last = None
            return
        self.last = db
        if not self._reached and db[0] <= self.arrive_m:
            self._reached = True
            self._log("info", f"Goal reached ({db[0]:.2f} m off) - holding neutral "
                      "until a new goal is clicked")

    def encode(self, crop):
        """The model's goal input this decision (policy_goal.encode)."""
        return pg.encode(crop, self.goal)

    def status(self):
        """/goal/status."""
        d, b = self.last if self.last is not None else (-1.0, 0.0)
        x, y, frame = self.goal if self.goal is not None else (0.0, 0.0, -1)
        return {"goal_set": self.goal is not None, "x": float(x), "y": float(y),
                "frame": int(frame),
                "frame_now": -1 if self.frame_now is None else int(self.frame_now),
                "on_map": bool(self.on_map),
                "distance_m": float(d) if self.goal is not None else -1.0,
                "bearing_deg": float(math.degrees(b)) if self.goal is not None else 0.0,
                "arrived": bool(self._reached), "arrive_m": self.arrive_m,
                "clicks": int(self.clicks), "refused": int(self.refused)}
