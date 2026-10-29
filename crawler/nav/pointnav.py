"""Point navigation: click a spot in Foxglove, the car drives there and holds.

Who drives. Two drivers write the one held command (main.DriverCommand),
never both in the same decision:

  free   the auto driver the car would have without --nav: the learned
         policy with --policy (its fallback and all), else the rules Driver.
  nav    nav.driver.NavDriver: the navigator on the live SLAM map.

From START until the first goal, free drives and nothing about it changes:
it is ticked exactly as it is without --nav. A goal hands the wheels to nav
from that decision (free is aborted, the way leaving auto aborts it), and nav
keeps them until auto is left: it drives to the goal, and once there -- or at
the reachable point nearest a goal it cannot get to, or having given up --
holds neutral for as long as it takes for another goal. A goal within
NEW_GOAL_M of one it has arrived at is the same goal and changes nothing; any
other goes. Leaving auto forgets the goal, so START again is free driving
again.

A click only counts while auto runs. One made in manual is still waiting in
the listener when START is pressed, and is thrown away on that first
decision (and said): a spot clicked minutes ago is not where I want the car
to go the moment I hand it the wheels.

The safety layer is the same on both sides (NavDriver's docstring has nav's),
and my hand on the pad overrides either the way it always has
(AutoPilot.step). Under nav, the navigator waits while my hand is on and
plans from wherever I left the truck once I let go; a held goal stays held.

What it says: a journal line (stdout) on every goal received, on arriving,
on stopping at the closest reachable point, on giving up, and on pausing for
a pose it cannot trust and resuming; and nav_status() / nav_path() for Foxglove's
/nav/status and /nav/path (telemetry/nav_view.py).

Everything else AutoPilot, the footprint tap and the mark pad read off a
driver (last_scan, _phase, backout_fields, forward_speed ...) is the active
driver's, through __getattr__. A /footprint row nav drove says autopilot
"nav" (policy_fields), so nothing reads it as the policy's own driving.
"""

import math

from crawler.nav import click
from crawler.nav import navigator as nv

NEW_GOAL_M = nv.ARRIVE_M

# /nav/status's state, off the navigator's own status.
IDLE = "idle"
PLANNING = "planning"
DRIVING = "driving"
ARRIVED = "arrived"
CLOSEST = "closest"
PAUSED = "paused"
FAILED = "failed"
STATES = (IDLE, PLANNING, DRIVING, ARRIVED, CLOSEST, PAUSED, FAILED)

_STATE_OF = {
    nv.ST_IDLE: IDLE, nv.ST_EXPLORED: IDLE,
    nv.ST_PLANNING: PLANNING,
    nv.ST_DRIVING: DRIVING, nv.ST_CUSP: DRIVING, nv.ST_WAIT: DRIVING,
    nv.ST_RECOVER: DRIVING,
    nv.ST_ARRIVED: ARRIVED,
    nv.ST_CLOSEST: CLOSEST,
    nv.ST_NO_MAP: PAUSED, nv.ST_HAND: PAUSED,
    nv.ST_FAILED: FAILED,
}

# At most this many points of the path go to Foxglove.
PATH_POINTS = 200

_OWN = frozenset(("free", "nav", "goals", "active", "goals_received",
                  "goals_dropped", "_engaged", "_said"))


class PointNavDriver:
    """See the module docstring. `free` and `nav` are built on the same
    command and camera view; `goals` is click.ClickGoals (anything with
    take()), or None."""

    def __init__(self, free, nav, goals=None):
        self.free = free
        self.nav = nav
        self.goals = goals
        self.active = free
        self.goals_received = 0
        self.goals_dropped = 0
        self._engaged = False
        self._said = None

    # ---- what AutoPilot sets before every tick: both drivers see it -------

    @property
    def human_throttle(self):
        return self.active.human_throttle

    @human_throttle.setter
    def human_throttle(self, v):
        self.free.human_throttle = v
        self.nav.human_throttle = v

    @property
    def human_steering(self):
        return getattr(self.active, "human_steering", False)

    @human_steering.setter
    def human_steering(self, v):
        self.free.human_steering = v
        self.nav.human_steering = v

    @property
    def pad_split(self):
        return getattr(self.active, "pad_split", None)

    @pad_split.setter
    def pad_split(self, v):
        self.free.pad_split = v
        self.nav.pad_split = v

    def __getattr__(self, name):
        # Only for what this object does not have itself.
        if name in _OWN or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.__dict__["active"], name)

    @property
    def policy_fields(self):
        """The free driver's own, while it drives; under nav a row that says
        so. None for the rules driver before any goal, as without --nav."""
        if self.active is self.nav:
            return self._nav_fields
        return getattr(self.free, "policy_fields", None)

    def _nav_fields(self):
        n = self.nav.nav
        return {"autopilot": "nav", "policy_reason": f"nav {n.status} {n.reason}"}

    @property
    def driving(self):
        """Who has the wheels: "nav", or the free driver's kind."""
        if self.active is self.nav:
            return "nav"
        return "policy" if getattr(self.free, "policy_fields", None) else "rules"

    # ---- the decision -------------------------------------------------------

    def tick(self, now=None):
        if not self._engaged:
            self._engaged = True
            self._drain()
        self._take_goal(now)
        if self.active is self.nav:
            self.nav.tick(now)
            self._say()
        else:
            self.free.tick(now)

    def abort(self):
        """Leaving auto: both drivers stop, the goal is forgotten, and the
        next engage starts on the free driver."""
        try:
            self.free.abort()
        finally:
            self.nav.abort()
            self.active = self.free
            self._engaged = False
            self._said = None

    def close(self):
        for d in (self.nav, self.free):
            close = getattr(d, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def _drain(self):
        if self.goals is None:
            return
        try:
            stale = self.goals.take()
        except Exception:
            stale = None
        if stale is not None:
            self.goals_dropped += 1
            print(f"[nav] goal ({stale.x:.2f}, {stale.y:.2f}) was clicked before auto "
                  "started, ignored: click again")

    def _take_goal(self, now):
        if self.goals is None:
            return
        try:
            goal = self.goals.take()
        except Exception:
            goal = None
        if goal is None:
            return
        try:
            view = self.nav.map_source.view()
        except Exception:
            view = None
        pose = None if view is None else view.pose
        placed = click.to_map(goal, pose)
        if placed is None:
            self.goals_dropped += 1
            print(f"[nav] goal in frame {goal.frame_id!r} dropped: "
                  "no pose to place it by")
            return
        x, y, yaw = placed
        n = self.nav.nav
        if (self.active is self.nav and n.status == nv.ST_ARRIVED and n.holding
                and n.goal is not None
                and math.hypot(x - n.goal[0], y - n.goal[1]) < NEW_GOAL_M):
            print(f"[nav] goal ({x:.2f}, {y:.2f}) is the one I am at, still holding")
            return
        self.goals_received += 1
        away = ("" if pose is None else
                f", {math.hypot(x - pose[0], y - pose[1]):.1f} m away")
        print(f"[nav] goal {self.goals_received} received: ({x:.2f}, {y:.2f}) "
              f"in the map{away}")
        if self.active is not self.nav:
            self.free.abort()
            self.active = self.nav
            print("[nav] the navigator has the wheels until auto is left")
        self.nav.set_goal(x, y, yaw)
        self._said = None

    def _say(self):
        """One journal line per change of state that matters."""
        n = self.nav.nav
        state = _STATE_OF.get(n.status, n.status)
        if state == self._said:
            return
        before, self._said = self._said, state
        g = n.goal
        pose = self.nav.last_pose
        off = ("" if g is None or pose is None else
               f", {math.hypot(g[0] - pose[0], g[1] - pose[1]):.2f} m from the goal")
        if state == ARRIVED:
            print(f"[nav] arrived at ({g[0]:.2f}, {g[1]:.2f}){off}; holding")
        elif state == CLOSEST:
            a = n.aim or (pose[:2] if pose is not None else (math.nan, math.nan))
            print(f"[nav] goal ({g[0]:.2f}, {g[1]:.2f}) cannot be reached; holding at "
                  f"the closest reachable point ({a[0]:.2f}, {a[1]:.2f}){off}")
        elif state == FAILED:
            print(f"[nav] gave up on ({g[0]:.2f}, {g[1]:.2f}): {n.reason}{off}; holding")
        elif state == PAUSED:
            print(f"[nav] paused: {self._reason()}")
        elif before == PAUSED:
            print(f"[nav] resumed, {state}")

    def _reason(self):
        n = self.nav.nav
        if n.status == nv.ST_NO_MAP:
            why = getattr(self.nav.map_source, "why", "") or "no map"
            return f"slam {why}"
        return n.reason

    # ---- what Foxglove is shown ------------------------------------------------

    def nav_status(self):
        """/nav/status: a flat dict of plain numbers, strings and bools."""
        n = self.nav.nav
        on = self.active is self.nav
        state = _STATE_OF.get(n.status, n.status) if on else IDLE
        g = n.goal if on else None
        a = n.aim if on else None
        pose = self.nav.last_pose
        dist = (-1.0 if g is None or pose is None
                else math.hypot(g[0] - pose[0], g[1] - pose[1]))
        path_m = -1.0
        if on and n.path is not None:
            try:
                path_m = float(n.follower.remaining_m)
            except Exception:
                path_m = -1.0
        return {
            "state": state,
            "reason": self._reason() if on else "",
            "status": n.status if on else "",
            "driving": self.driving,
            "has_goal": g is not None,
            "goal_x": float(g[0]) if g is not None else 0.0,
            "goal_y": float(g[1]) if g is not None else 0.0,
            "has_aim": a is not None,
            "aim_x": float(a[0]) if a is not None else 0.0,
            "aim_y": float(a[1]) if a is not None else 0.0,
            "closest": bool(on and n.closest),
            "distance_m": float(dist),
            "path_m": path_m,
            "local_map": bool(on and n.local),
            "plans": int(n.plans),
            "plan_ms": float(n.last_plan_ms),
            "goals": int(self.goals_received),
            "dropped": int(self.goals_dropped),
        }

    def nav_path(self):
        """/nav/path: {"path": [(x, y), ...] still to drive, "goal": (x, y)
        or None, "aim": the point the path ends at when it is not the goal}."""
        n = self.nav.nav
        if self.active is not self.nav:
            return {"path": [], "goal": None, "aim": None}
        pts = []
        p = n.path
        if p is not None:
            try:
                i = int(getattr(n.follower, "index", 0))
                xy = p.poses[i:, :2]
                step = max(1, int(math.ceil(len(xy) / PATH_POINTS)))
                pts = [(float(x), float(y)) for x, y in xy[::step]]
                if len(xy) and (len(xy) - 1) % step:
                    pts.append((float(xy[-1, 0]), float(xy[-1, 1])))
            except Exception:
                pts = []
        g = None if n.goal is None else (float(n.goal[0]), float(n.goal[1]))
        a = None
        if n.closest and n.aim is not None:
            a = (float(n.aim[0]), float(n.aim[1]))
        return {"path": pts, "goal": g, "aim": a}
