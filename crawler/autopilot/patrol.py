"""Waypoints on the SLAM map, driven through in order: once, in laps, or
back and forth.

The goal model drives. All this does is say which waypoint is the goal right
now: the model is fed it through the goal input it already has
(policy_goal.encode: present, distance over FAR_M, the bearing's sin and
cos), exactly as it is fed a single clicked goal. No path is planned or
followed and nothing here steers; by the owner's ruling only the model ever
owns the wheels. The only things it adds to the wheels are holds, the same
neutral the arrival hold is (reached()): at the last waypoint of a "once",
while paused, and through a dwell.

Using it, with --goal. Foxglove's Publish pose tool on the map panel sends
/move_base_simple/goal; the Publish panel buttons in the drive layout send
{"cmd": ...} on /patrol/cmd (nav/click.py). The commands (COMMANDS):

  edit    patrol editing: from now on every pose click is appended as the
          next waypoint. The patrol is paused until start.
  goal    back to a single goal: a click is the goal, as before --goal had
          patrols. The list is kept.
  start   drive the list (resumes where it was paused; from the first after
          a finished "once"). Refused with no waypoints.
  pause   hold neutral where it is, the list and the place in it kept.
  undo    drop the last waypoint.
  clear   drop them all, and pause.
  mode    {"cmd": "mode", "mode": "once" | "loop" | "patrol"}; without a
          mode, the next one round.
  trace   on, then off: while on, the truck's own map pose is appended every
          TRACE_M it moves. A path drawn by driving it, as Foxglove has no
          tool to draw a polyline on a non-ROS connection.

Arriving. A waypoint is passed once the truck is within PASS_M of it, and the
next one is the goal from that decision on; or, with the path planner, once
it has gone by it round the corner (pathgoal.gone_by, told through
passed()). The planner is told the waypoints after the current one
(ahead) and plans through them, so the point the model is fed bends round
to the next before this one is reached. The last of a "once" wants
ARRIVE_M, and then the arrival hold holds the car there, as for a single
goal. dwell_s, 0 unless set, holds neutral that long at each waypoint first.
"loop" goes from the last back to the first, a lap each time round;
"patrol" turns round at either end, a lap each time it is back at the first.
One waypoint in a loop or a patrol is a "once": there is nowhere to go on to.

A waypoint it cannot get to. While auto drives towards a waypoint on the map
(a decision with the waypoint placed, observe(); not manual, not a lost map),
its clock runs: no PROGRESS_M closer for NO_PROGRESS_S, or TIMEOUT_S in all,
and it is skipped with a journal line. Every waypoint skipped in a row, a
lap with no arrival at all, pauses the patrol.

Across a reseed. Every waypoint is carried as a single clicked goal is
(goal.ClickGoal: its per-frame places, the links followed once each, the snap
back on a merge), each in a ClickGoal of its own whose journal lines are
gathered into one line a link.

Across a restart. With a store, the list is written on every change (on a
thread, never in the control loop), with the walls of the map it is on. The
SLAM map is saved too (mapping/mapfile.py) and comes back as map 0 in the
frame it was saved in, so the old list is on it again; without that file
(the first boot, --no-slam-map-file, a file that would not load) a restart
is a new map 0 with its origin wherever the truck boots. Either way nothing
is restored until the first map comes up and its walls are checked against
the saved ones (MATCH_SHARE of them within one cell); a match restores it
paused, anything else clears it with a journal line, as does a list that
was on a map other than 0 (a fresh map after a reseed comes back after a
restart only as map 0, if it was the one saved, so its old number means
nothing).
"""

import json
import math
import os
import threading
import time

import numpy as np

from crawler.autopilot import goal as gl
from crawler.autopilot import policy_goal as pg

MODES = ("once", "loop", "patrol")
COMMANDS = ("edit", "goal", "start", "pause", "undo", "clear", "mode", "trace")
# Within this of a waypoint it is passed and the next is the goal. Wider
# than ARRIVE_M: a waypoint on the way is somewhere to go past, not to stop
# on, and a model asked to hit 0.25 m on the way round circles for it.
PASS_M = 0.5
DWELL_S = 0.0
# A waypoint's clock: skipped after this long without getting PROGRESS_M
# closer, or TIMEOUT_S driving at it in all.
NO_PROGRESS_S = 20.0
PROGRESS_M = 0.3
TIMEOUT_S = 120.0
# The most of a decision's gap the clock counts: auto off between two
# decisions is not time driven.
CLOCK_STEP_MAX_S = 0.5
TRACE_M = 1.0
MAX_WAYPOINTS = 100
# The waypoints after the current one the path planner is told of
# (pathgoal.MAX_AHEAD plans through no more).
AHEAD_MAX = 3
# The restart check: the boot map's walls against the saved ones.
MATCH_SHARE = 0.6
MIN_WALLS = 50
WALLS_KEPT = 3000
WALLS_EVERY_S = 30.0
# Present crops on map 0 with too few walls to check before giving up.
RESTORE_WAIT_S = 30.0


class _Waypoint(gl.ClickGoal):
    """One waypoint: a ClickGoal for its carry along the links, with its
    journal lines kept for the patrol to sum up instead of said."""

    def __init__(self):
        super().__init__()
        self.said = []

    def _log(self, level, text):
        self.said.append(text)


class _One:
    def __init__(self, click):
        self.click = click

    def take(self):
        click, self.click = self.click, None
        return click


class Store:
    """The list on disk, written off the control thread. background=False
    writes in the caller, for tests."""

    def __init__(self, path, background=True):
        self.path = str(path)
        self.background = bool(background)
        self._lock = threading.Lock()
        self._pending = None
        self._wake = threading.Event()
        self._thread = None
        self.errors = 0

    def load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def save(self, data):
        if not self.background:
            self._write(data)
            return
        with self._lock:
            self._pending = data
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="patrol-store",
                                                daemon=True)
                self._thread.start()
        self._wake.set()

    def _loop(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                data, self._pending = self._pending, None
            if data is not None:
                self._write(data)

    def _write(self, data):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        except Exception:
            self.errors += 1


def _occupied_points(view):
    """(N, 2) map metres of a runner.MapView's occupied pixel centres."""
    from crawler.mapping import grid as gr
    if view is None:
        return np.zeros((0, 2))
    i, j = np.nonzero(np.asarray(view.codes) == gr.OCCUPIED)
    return np.column_stack([view.x0 + (i + 0.5) * view.res,
                            view.y0 + (j + 0.5) * view.res])


def walls_match(saved, view):
    """The share of the view's walls within one pixel of a saved wall."""
    now = _occupied_points(view)
    saved = np.asarray(saved, float).reshape(-1, 2)
    if len(now) == 0 or len(saved) == 0:
        return 0.0
    res = float(view.res)
    have = set()
    for i, j in np.floor(saved / res).astype(np.int64):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                have.add((int(i) + di, int(j) + dj))
    cells = np.floor(now / res).astype(np.int64)
    return float(np.mean([(int(i), int(j)) in have for i, j in cells]))


class Patrol(gl.Goal):
    """The single clicked goal, or the patrol. Control loop only."""

    def __init__(self, arrive_m=pg.ARRIVE_M, pass_m=PASS_M, dwell_s=DWELL_S,
                 say=None, store=None, map_view=None, clock=time.monotonic):
        super().__init__()
        self.arrive_m, self.pass_m, self.dwell_s = float(arrive_m), float(pass_m), float(dwell_s)
        self._say = say
        self.single = gl.ClickGoal(arrive_m=arrive_m, say=say)
        self.store = store
        self.map_view = map_view
        self.clock = clock
        self.use = "goal"
        self.mode = "loop"
        self.running = False
        self.finished = False
        self.tracing = False
        self.points = []
        self.index = 0
        self.direction = 1
        self.laps = 0
        self.skipped = 0
        self.passed_on_path = 0       # passed by the path planner, not PASS_M
        self.refused = 0
        self.restored = False
        self._in_row = 0              # skips since the last arrival
        self._seq = 0                 # the newest egomap.Link taken
        self.frame_now = None
        self.present = False
        self.last = None              # (distance, bearing) to the current one
        self.pose = None              # the newest present crop's (x, y, yaw)
        self._dwell_until = None
        self._wp_time = 0.0
        self._best = math.inf
        self._since_best = 0.0
        self._last_obs = None
        self._walls = None            # map 0's walls, for the restart check
        self._walls_at = None
        self._pending = None          # a saved list waiting for the map
        self._pending_wait = 0.0
        if store is not None:
            data = store.load()
            if data and data.get("points"):
                self._pending = data

    def _log(self, level, text):
        print(text)
        if self._say is not None:
            try:
                self._say(level, "Patrol", text)
            except Exception:
                pass

    # ---- the goal the model is fed ---------------------------------------------------

    @property
    def goal(self):
        """(x, y, frame) the model drives to now, or None."""
        if self.use == "goal":
            return self.single.goal
        if not self.running or self.holding():
            return None
        place = self.current_place()
        return None if place is None else (place[0], place[1], self.frame_now)

    @property
    def ahead(self):
        """[(x, y, frame)]: the waypoints after the current one, in the
        order they will be driven, up to AHEAD_MAX, so the path planner
        (pathgoal.py) can carry the point it feeds on round the corner
        before this one is passed. Nothing at the last stop of a "once",
        with a dwell (every waypoint is a stop), or with no goal now."""
        if self.use != "patrol" or self.goal is None or self.dwell_s > 0.0:
            return []
        out = []
        i, direction = self.index, self.direction
        n = len(self.points)
        for _ in range(AHEAD_MAX):
            if n == 1 or (self.mode == "once" and i >= n - 1):
                break
            i, direction = self._next(i, direction)
            place = self.points[i].places.get(self.frame_now)
            if place is None:
                break
            out.append((place[0], place[1], self.frame_now))
        return out

    def passed(self, goal, crop=None):
        """The path planner says the truck has gone by `goal` (cut the
        corner wide of PASS_M). Passed as on PASS_M if it is the current
        waypoint and not a stop; True if it was."""
        if self.use != "patrol" or self.holding() or self.dwell_s > 0.0 or self._last_stop():
            return False
        g = self.goal
        if g is None or goal is None or int(goal[2]) != int(g[2]) or math.hypot(
                float(goal[0]) - g[0], float(goal[1]) - g[1]) > 0.05:
            return False
        self._in_row = 0
        self.passed_on_path += 1
        self._advance()
        if crop is not None and getattr(crop, "present", False):
            self._measure(crop)
        return True

    def holding(self):
        return (self._dwell_until is not None) or self.finished or not self.running

    def reached(self):
        if self.use == "goal":
            return self.single.reached()
        return self.holding()

    def encode(self, crop):
        if self.use == "goal":
            return self.single.encode(crop)
        g = self.goal
        return pg.missing() if g is None else pg.encode(crop, g)

    # ---- in ----------------------------------------------------------------------

    def poll(self, clicks, crop):
        """Every control pass, any mode: the clicks and commands since the
        last pass, in order, then where everything is off this crop."""
        if clicks is None:
            events = []
        elif hasattr(clicks, "take_events"):
            events = clicks.take_events()
        else:
            click = clicks.take()
            events = [] if click is None else [("goal", click)]
        links = getattr(crop, "links", ()) if crop is not None else ()
        # Links up to this crop are from before anything in these events.
        self._follow(links)
        for kind, what in events:
            if kind == "cmd":
                self.command(what, crop)
            elif self.use == "goal":
                self.single.poll(_One(what), crop)
            else:
                self._take_click(what, crop)
        self._place(crop)

    def observe(self, crop):
        """A decision in auto (PolicyDriver._tick): _place, and the
        current waypoint's clock."""
        self._follow(getattr(crop, "links", ()) if crop is not None else ())
        self._place(crop)
        now = float(self.clock())
        dt = 0.0 if self._last_obs is None else min(max(now - self._last_obs, 0.0),
                                                   CLOCK_STEP_MAX_S)
        self._last_obs = now
        if (self.use != "patrol" or self.holding() or self.last is None
                or not self.present):
            return
        self._wp_time += dt
        d = self.last[0]
        if d < self._best - PROGRESS_M:
            self._best, self._since_best = d, 0.0
        else:
            self._since_best += dt
        why = None
        if self._since_best >= NO_PROGRESS_S - 1e-6:
            why = f"no {PROGRESS_M:.1f} m closer in {NO_PROGRESS_S:.0f} s"
        elif self._wp_time >= TIMEOUT_S - 1e-6:
            why = f"not there in {TIMEOUT_S:.0f} s"
        if why is not None:
            self.skipped += 1
            self._in_row += 1
            self._log("warning", f"Waypoint {self.index + 1} of {len(self.points)} "
                      f"skipped: {why}, {d:.2f} m off")
            if self._in_row >= len(self.points):
                self.running = False
                self._log("warning", "Every waypoint skipped in a row - the patrol "
                          "is paused; holding neutral")
                self._save()
                return
            self._advance()
            self._measure(crop)

    def command(self, msg, crop=None):
        """One {"cmd": ...}. True if it did something, False if it was
        refused (said why), None if it is not a command at all."""
        name = msg.get("cmd") if isinstance(msg, dict) else None
        if name not in COMMANDS:
            self._log("warning", f"{name!r} is not a patrol command; one of "
                      f"{', '.join(COMMANDS)}")
            return None
        if name == "edit":
            if self.use != "patrol":
                self.use, self.running = "patrol", False
            self._log("info", f"Patrol editing: each pose click is waypoint "
                      f"{len(self.points) + 1} on")
        elif name == "goal":
            self.use, self.running, self.tracing = "goal", False, False
            self._log("info", "Single goal: a pose click is the goal")
        elif name == "start":
            if not self.points:
                self._log("warning", "Start refused: no waypoints")
                return False
            self.use = "patrol"
            if self.finished or not (0 <= self.index < len(self.points)):
                self.index, self.direction, self.laps, self.finished = 0, 1, 0, False
            self.running, self._in_row = True, 0
            self._reset_clock()
            self._log("info", f"Patrol {self.mode} started at waypoint {self.index + 1} "
                      f"of {len(self.points)}")
        elif name == "pause":
            self.running = False
            self._dwell_until = None
            self._log("info", "Patrol paused - holding neutral")
        elif name == "undo":
            if not self.points:
                return False
            self.points.pop()
            if self.index >= len(self.points):
                self.index = max(len(self.points) - 1, 0)
                self._reset_clock()
            if not self.points:
                self.running = False
            self._log("info", f"Waypoint {len(self.points) + 1} undone, "
                      f"{len(self.points)} left")
        elif name == "clear":
            self.points, self.running, self.tracing, self.finished = [], False, False, False
            self.index, self.direction = 0, 1
            self._log("info", "Patrol cleared")
        elif name == "mode":
            mode = msg.get("mode")
            if mode is None:
                mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
            if mode not in MODES:
                self._log("warning", f"Patrol has no mode {mode!r}; one of {', '.join(MODES)}")
                return False
            self.mode = mode
            self.finished = False
            self._log("info", f"Patrol mode {mode}")
        elif name == "trace":
            self.tracing = not self.tracing
            if self.tracing:
                self.use, self.running = "patrol", False
            self._log("info", "Tracing: a waypoint every "
                      f"{TRACE_M:.1f} m driven" if self.tracing else
                      f"Tracing off, {len(self.points)} waypoints")
        self._save()
        return True

    def _take_click(self, click, crop):
        from crawler.nav import click as ck
        if crop is None:
            self.refused += 1
            self._log("warning", "A waypoint was clicked with no SLAM map yet; not taken")
            return False
        placed = ck.to_map(click, (crop.x, crop.y, crop.yaw))
        if placed is None:
            self.refused += 1
            self._log("warning", f"A waypoint in frame {click.frame_id!r} cannot be put "
                      "on the map; set the 3D panel's display frame to map")
            return False
        return self.append(placed[0], placed[1], int(getattr(crop, "frame", 0)))

    def append(self, x, y, frame):
        x, y = float(x), float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            self.refused += 1
            return False
        if len(self.points) >= MAX_WAYPOINTS:
            self.refused += 1
            self._log("warning", f"Waypoint refused: {MAX_WAYPOINTS} is the most")
            return False
        if self._pending is not None:
            self._drop_pending("a new waypoint was set before the map was checked")
        w = _Waypoint()
        w._seq = self._seq
        w.set_goal(x, y, frame)
        w.said.clear()
        self.points.append(w)
        self._save()
        return True

    # ---- each pass -------------------------------------------------------------------

    def _follow(self, links):
        """Carry the single goal and every waypoint along the links none
        has taken yet, one journal line a link."""
        if self.use != "goal":
            self.single.follow(links)
        for L in links:
            if L.seq <= self._seq:
                continue
            self._seq = L.seq
            moved = snapped = 0
            offs, errs = [], []
            for w in self.points:
                had = len(w.transfer_errors)
                w.follow([L])
                if len(w.transfer_errors) > had:
                    snapped += 1
                    offs.append(w.transfer_errors[-1])
                elif w.said:
                    moved += 1
                    errs.append(w.places[L.to][2])
                w.said.clear()
            if snapped:
                self._log("info", f"Back on map {L.to}: {snapped} waypoints exactly "
                          f"where they were set again; the carried ones were "
                          f"{min(offs):.2f}-{max(offs):.2f} m off")
            if moved and L.why == "lost":
                self._log("info", f"Lost the map the patrol was on; carried {moved} "
                          f"waypoints onto map {L.to}, up to {max(errs):.2f} m off")
            elif moved and L.why == "merged":
                self._log("info", f"Map {L.frm} merged into map {L.to}; {moved} "
                          "waypoints moved with it")
            elif moved:
                self._log("info", f"Map {L.frm} restarted as map {L.to}; carried "
                          f"{moved} waypoints, up to {max(errs):.2f} m off")
            if (snapped or moved) and self.store is not None:
                self._save()

    def current_place(self):
        """(x, y, approx_m) of the current waypoint on the map the truck is
        on, or None."""
        if not self.points or self.frame_now is None:
            return None
        return self.points[min(self.index, len(self.points) - 1)].places.get(self.frame_now)

    def waypoints(self):
        """[(x, y, frame)] on the map the truck is on now (each one's place
        there; one with none there is on the frame it was set on)."""
        out = []
        for w in self.points:
            place = w.places.get(self.frame_now) if self.frame_now is not None else None
            out.append((place[0], place[1], self.frame_now) if place is not None
                       else w.goal)
        return out

    def _place(self, crop):
        self.frame_now = None if crop is None else int(getattr(crop, "frame", 0))
        self.present = crop is not None and bool(getattr(crop, "present", False))
        if self.present:
            self.pose = (float(crop.x), float(crop.y), float(crop.yaw))
        if self.use == "goal":
            self.single.observe(crop)
        if self._pending is not None:
            self._check_pending(crop)
        if not self.present:
            self.last = None
            return
        self._keep_walls()
        if self.tracing:
            self._trace(crop)
        db = self._measure(crop)
        if self.use != "patrol" or not self.running or self.finished:
            return
        if self._dwell_until is not None:
            if float(self.clock()) >= self._dwell_until:
                self._dwell_until = None
                self._advance()
                self._measure(crop)
            return
        if db is None:
            return
        radius = self.arrive_m if self._last_stop() else self.pass_m
        if db[0] <= radius:
            self._in_row = 0
            if self._last_stop():
                self._advance()
            elif self.dwell_s > 0.0:
                self._dwell_until = float(self.clock()) + self.dwell_s
            else:
                self._advance()
                self._measure(crop)

    def _measure(self, crop):
        """self.last: (distance, bearing) from this crop to the current
        waypoint, or None."""
        place = self.current_place()
        self.last = (None if place is None else
                     pg.distance_bearing(crop, (place[0], place[1], self.frame_now)))
        return self.last

    def _last_stop(self):
        """Whether the current waypoint is the last stop: the end of a
        "once", or the only waypoint there is."""
        n = len(self.points)
        return n == 1 or (self.mode == "once" and self.index >= n - 1)

    def _advance(self):
        n = len(self.points)
        if n == 0:
            return
        if self._last_stop():
            self.finished = True
            self.laps += 1
            self._log("info", f"Patrol done: waypoint {self.index + 1} of {n} reached - "
                      "holding neutral until start")
            self._save()
            return
        i, self.direction = self._next(self.index, self.direction)
        if i == 0 and self.mode == "loop":
            self.laps += 1
            self._log("info", f"Lap {self.laps} of {n} waypoints done, "
                      f"{self.skipped} skipped so far")
        elif i == 0 and self.mode == "patrol":
            self.laps += 1
            self._log("info", f"Lap {self.laps} there and back done, "
                      f"{self.skipped} skipped so far")
        self.index = i
        self._reset_clock()

    def _next(self, i, direction):
        """(index, direction) after waypoint i, not at a last stop."""
        n = len(self.points)
        if self.mode == "loop":
            return (i + 1) % n, direction
        if self.mode == "patrol":
            if not 0 <= i + direction < n:
                direction = -direction
            return i + direction, direction
        return i + 1, direction

    def _reset_clock(self):
        self._wp_time, self._best, self._since_best = 0.0, math.inf, 0.0
        self._dwell_until = None

    def _trace(self, crop):
        if not self.points:
            self.append(crop.x, crop.y, self.frame_now)
            return
        last = self.points[-1].places.get(self.frame_now)
        if last is None or math.hypot(crop.x - last[0], crop.y - last[1]) >= TRACE_M:
            self.append(crop.x, crop.y, self.frame_now)

    # ---- across a restart ------------------------------------------------------------

    def _keep_walls(self):
        """Map 0's walls, every WALLS_EVERY_S while there is a list to save."""
        if (self.store is None or self.map_view is None or self.frame_now != 0
                or not self.points):
            return
        now = float(self.clock())
        if self._walls_at is not None and now - self._walls_at < WALLS_EVERY_S:
            return
        self._walls_at = now
        try:
            pts = _occupied_points(self.map_view())
        except Exception:
            return
        if len(pts) >= MIN_WALLS:
            step = max(1, int(math.ceil(len(pts) / WALLS_KEPT)))
            self._walls = np.round(pts[::step], 3).tolist()
            self._save()

    def _saved(self):
        # Their places on map 0 if they all have one; else on the map the
        # newest was set on, which a restart never has.
        frame = 0
        if not all(w.places.get(0) is not None for w in self.points):
            frame = self.points[-1].goal[2]
        places = [w.places.get(frame) for w in self.points]
        pts = [[float(p[0]), float(p[1])] for p in places if p is not None]
        return {"version": 1, "frame": int(frame), "mode": self.mode, "points": pts,
                "walls": self._walls or []}

    def _save(self):
        if self.store is None or self._pending is not None:
            return
        self.store.save(self._saved())

    def _drop_pending(self, why):
        n = len(self._pending.get("points") or [])
        self._pending = None
        self._log("warning", f"The saved patrol ({n} waypoints) is cleared: {why}")
        self._save()

    def _check_pending(self, crop):
        data = self._pending
        if crop is None or not getattr(crop, "present", False):
            return
        if int(data.get("frame", -1)) != 0:
            self._drop_pending(f"it was on map {data.get('frame')}, a fresh map that "
                               "is not there after a restart")
            return
        if self.frame_now != 0:
            self._drop_pending("the map reseeded before its walls could be checked")
            return
        try:
            view = self.map_view() if self.map_view is not None else None
        except Exception:
            view = None
        if len(_occupied_points(view)) < MIN_WALLS:
            now = float(self.clock())
            if self._pending_wait == 0.0:
                self._pending_wait = now
            elif now - self._pending_wait >= RESTORE_WAIT_S:
                self._drop_pending("the map never had walls enough to check it against")
            return
        share = walls_match(data.get("walls") or [], view)
        if share < MATCH_SHARE:
            self._drop_pending(f"this map is not the one it was set on ({share:.0%} of "
                               "the walls agree)")
            return
        self._pending = None
        mode = data.get("mode")
        self.mode = mode if mode in MODES else self.mode
        for x, y in data["points"]:
            self.append(x, y, 0)
        self._walls = data.get("walls") or None
        self.use, self.running, self.restored = "patrol", False, True
        self._save()
        self._log("info", f"Patrol restored: {len(self.points)} waypoints on the same "
                  f"map ({share:.0%} of the walls agree), {self.mode}, paused until start")

    # ---- out -------------------------------------------------------------------------

    def state(self):
        if self.use == "goal":
            return "goal"
        if self.tracing:
            return "tracing"
        if not self.points:
            return "empty"
        if self.finished:
            return "finished"
        if not self.running:
            return "paused"
        if self._dwell_until is not None:
            return "dwell"
        return "driving"

    def approx_m(self):
        place = self.current_place()
        return float(place[2]) if place is not None else 0.0

    def status(self):
        """/goal/status: the goal the model is fed, in goal.ClickGoal's schema."""
        if self.use == "goal":
            return self.single.status()
        place = self.current_place()
        d, b = self.last if self.last is not None else (-1.0, 0.0)
        have = place is not None
        x, y = (place[0], place[1]) if have else (0.0, 0.0)
        e = self.approx_m()
        return {"goal_set": bool(self.points), "x": float(x), "y": float(y),
                "frame": int(self.frame_now) if have else -1,
                "frame_now": -1 if self.frame_now is None else int(self.frame_now),
                "on_map": bool(have and self.present),
                "distance_m": float(d) if have else -1.0,
                "bearing_deg": float(math.degrees(b)) if have else 0.0,
                "arrived": bool(self.reached()), "arrive_m": self.arrive_m,
                "clicks": int(self.single.clicks + len(self.points)),
                "refused": int(self.single.refused + self.refused),
                "approx": e > 0.0, "approx_m": e}

    def patrol_status(self):
        """/patrol/status."""
        tx, ty, tyaw = self.pose if self.pose is not None else (0.0, 0.0, 0.0)
        tyaw = math.degrees(tyaw)
        place = self.current_place()
        have = place is not None
        d = self.last[0] if (self.last is not None and have) else -1.0
        x, y = (place[0], place[1]) if have else (0.0, 0.0)
        e = self.approx_m()
        return {"use": self.use, "state": self.state(), "mode": self.mode,
                "running": bool(self.running), "tracing": bool(self.tracing),
                "index": int(self.index), "number": int(self.index + 1) if self.points else 0,
                "count": len(self.points), "laps": int(self.laps),
                "skipped": int(self.skipped),
                "passed_on_path": int(self.passed_on_path), "x": float(x), "y": float(y),
                "frame_now": -1 if self.frame_now is None else int(self.frame_now),
                "on_map": bool(have and self.present), "distance_m": float(d),
                "approx": e > 0.0, "approx_m": e, "restored": bool(self.restored),
                "pending": self._pending is not None,
                # The truck's map pose the waypoint is measured from, on the
                # same map: a recording pairs the two with no other topic.
                "truck_x": float(tx), "truck_y": float(ty), "truck_yaw_deg": float(tyaw)}

    def marker(self):
        """/patrol/marker's payload: the waypoints on the map the truck is on
        now, in order, which is the current one and how far off it may be.
        {} with none there, which clears every marker."""
        if not self.points or self.frame_now is None:
            return {"points": []}
        places = [w.places.get(self.frame_now) for w in self.points]
        if any(p is None for p in places):
            return {"points": []}
        return {"points": [(p[0], p[1]) for p in places],
                "current": min(self.index, len(places) - 1)
                if self.use == "patrol" else None,
                "closed": self.mode == "loop",
                "approx_m": float(places[min(self.index, len(places) - 1)][2])}
