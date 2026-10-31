"""Explore: the truck drives itself round the house, to the edge of what the
SLAM map has seen and on.

The model drives. This only picks the point the goal input points at, the
same way follow's search does once the person is gone (autopilot/search.py),
with no person to go along after: a frontier first (a free cell next to
unknown, the edge of the map, reachable, nearest with a bonus for a long
edge), then with none left the farthest free reachable cell not tried yet
(wander). The path planner (pathgoal.PathPlanner) routes to it like any
goal. Nothing here steers or touches the throttle; with nothing to pick it
is the arrival hold, neutral, until the map gives it somewhere.

It lives in the patrol (patrol.Patrol, use "explore") so the map page and
/patrol/cmd start and stop it, and so it and the patrol are never both on:

  {"cmd": "explore"}               on if off, off if on
  {"cmd": "explore", "on": true}   on (stops the patrol, keeps its list)
  {"cmd": "explore", "on": false}  off, holding neutral

Start (the patrol), pause, a single goal, edit or trace stop it; a loop drawn
on the map page meanwhile is kept, paused, for a Start.
Like the patrol, it is only armed by the command: the truck drives it in
auto (START on the pad), and a pad loss parks auto as ever.

The target's clock (arrival, no progress, timeout: search.py's numbers)
runs only on auto's decisions; a pass that is not driving only picks a
target when there is none, so the map page shows where it will go before
auto.
"""

from crawler.autopilot import search as sr


class Explore:
    """Search's frontier and wander targets, with no person hint."""

    def __init__(self, map_view=None, say=None):
        self.search = sr.Search(map_view=map_view, say=say, topic="Explore",
                                noun="Explore", doing="Exploring")
        self.on = False
        self.started = 0            # times turned on since boot

    def start(self, crop=None):
        self.search.reset()
        self.search.start(crop, None)
        self.on = True
        self.started += 1

    def stop(self):
        self.search.reset()
        self.on = False

    @property
    def target(self):
        """(x, y, frame) of the current target, or None."""
        return self.search.target if self.on else None

    @property
    def kind(self):
        """frontier / wander / "" with no target."""
        return self.search.kind if self.target is not None else ""

    def step(self, crop, clocked=True):
        if self.on:
            self.search.step(crop, clocked=clocked)

    def follow(self, L):
        self.search.follow(L)

    def since_best(self):
        """The current target's seconds without getting PROGRESS_M closer,
        -1 with none."""
        return float(self.search._since_best) if self.target is not None else -1.0

    def counts(self):
        s = self.search
        return {"targets": int(s.targets), "arrived": int(s.arrived),
                "skipped": int(s.skipped), "frontiers": int(s.frontiers)}
