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

A near miss. A waypoint the truck has come within near_pass_m of (1.0 m
unless set) is passed too, by="near", once it can no longer be got to by
driving on: it is abeam or behind (NEAR_ABEAM_DEG or more off the nose), the
distance has risen NEAR_RISE_M past the closest it came, or it lies so far
inside a forward turning circle (TURN_RADIUS_M) that no forward arc comes
within PASS_M of it; and only once the truck has moved NEAR_MOVE_M since the
waypoint became the goal. On the 2026-10-30 patrols 56 of 58 drives away from a
waypoint began that way: within 0.5-1.3 m, never inside PASS_M, the
waypoint stayed the goal beside or behind the truck, and the model drove on
away from it for a median 26 s before it came back. Only the goal changes;
the model still drives. Never at a last stop or with a dwell (both are
places to stop on), and never for a waypoint the truck was not near: this is
not the stuck skip. near_pass_m 0 is the old rule, PASS_M only. Set with
--patrol-near-pass-m, or {"cmd": "near", "m": 1.0}, which is kept in the
settings file across a restart and wins over the flag, as the skip is.

A waypoint it cannot get to. While auto drives towards a waypoint on the map
(a decision with the waypoint placed, observe(); not manual, not a lost map),
its clock runs: no PROGRESS_M closer for NO_PROGRESS_S, or TIMEOUT_S in all,
and it is stuck. With the skip on it is skipped with a journal line, and
every waypoint skipped in a row, a lap with no arrival at all, pauses the
patrol. The skip is off unless asked for (--patrol-skip-stuck, or the
toggle on the map page, {"cmd": "skip", "on": true}, kept in a settings
file across a restart): the owner found a timed skip threw a lap off more
than it helped. Off, the waypoint stays the goal until it is passed, the
stall gets one journal line, and /patrol/status still says how long it has
gone without getting closer (stuck_s).

Across a reseed. Every waypoint is carried as a single clicked goal is
(goal.ClickGoal: its per-frame places, the links followed once each, the snap
back on a merge), each in a ClickGoal of its own whose journal lines are
gathered into one line a link.

Across a restart. With a store, the list is written on every change (on a
thread, never in the control loop) as a track (autopilot/tracks.py): the
points and a sample of the walls round them, in one frame. A restart is a
fresh map with its origin wherever the truck boots, so the saved track is
"not placed yet" until mapping/trackfit.py has fitted its walls onto the
live map (turned and moved, never scaled), on a thread, tried again every
FIT_EVERY_S while the map grows. Only a confident fit places it, and then
the points are put through the same fit and the patrol is paused until
start. Until then nothing is lost: the file is not written over, and the
owner can place it by hand (the "place" command: the map page's nudge).
The walls are only ever used to fit; the map is not restored.

Named tracks. "save" keeps the list under a name (tracks.Library), "load"
brings one back as a track to place, "forget" deletes one. Nothing
automatic deletes or writes over a named track; an unplaced or unsaved
track that is about to be replaced is kept in the library first under a
name of its own (AUTOSAVE).

Explore (autopilot/explore.py): {"cmd": "explore"} (the map page's Explore
button) makes the goal the search's frontier, then wander, target instead of
a waypoint, use "explore"; the list is kept, paused. It and the patrol are
never both on: explore on pauses the patrol, and start, pause, a single goal,
edit or trace turn explore off, holding neutral. /patrol/status, /goal/status
and the /footprint row say use "explore", state exploring or holding.

Tags (telemetry/tags.py): {"cmd": "tags", "clean": true} (the map page's
Clean button) labels the recordings from then on. The patrol only passes
it on to the SessionTags it was given; without one it is refused. Nothing about driving changes with a tag.

Editing from the map page (telemetry/map.html), on the map the truck is on:
"stroke" appends a run of points drawn with a finger as one step that
"unstroke" takes back, and makes the patrol a loop; "move" and "delete"
take a waypoint's index.
"""

import collections
import datetime
import json
import math
import os
import threading
import time

import numpy as np

from crawler.autopilot import goal as gl
from crawler.autopilot import policy_goal as pg
from crawler.autopilot import tracks as tk

MODES = ("once", "loop", "patrol")
# The drive layout's buttons send these; the map page (telemetry/map.html)
# sends them and MAP_COMMANDS, which take points, indices or names.
BUTTON_COMMANDS = ("edit", "goal", "start", "pause", "undo", "clear", "mode", "trace")
MAP_COMMANDS = ("stroke", "unstroke", "move", "delete", "save", "load", "forget", "place",
                "skip", "near", "explore", "tags")
COMMANDS = BUTTON_COMMANDS + MAP_COMMANDS
# Within this of a waypoint it is passed and the next is the goal. Wider
# than ARRIVE_M: a waypoint on the way is somewhere to go past, not to stop
# on, and a model asked to hit 0.25 m on the way round circles for it.
PASS_M = 0.5
# A near miss (the docstring's): within NEAR_PASS_M and then abeam, going
# away, or deep inside the turning circle, and it is passed as well. Off the
# 2026-10-30 patrols replayed (/patrol/status + /slam/status): 1.0 m takes
# 39 of the 58 drives away off the evening's patrols and 58 of 120 off the
# day's before; 1.3 m would take 56 and 91, but passes further from where
# the owner put the point. NEAR_PASS_MAX_M is the most it can be set to.
NEAR_PASS_M = 1.0
NEAR_PASS_MAX_M = 2.0
# Abeam: 80 degrees is past where the goal model's steer for the goal falls
# off (policy_goal.STAGE_DEG's note: +0.04 at 75-90 on the left, wrong-way
# past 90), so a waypoint this far round is one it will not turn in to.
NEAR_ABEAM_DEG = 80.0
# Going away: this much further than the closest it came. Well above the
# SLAM pose's jitter; on the replay 0.15 m passed 11 and 13 more waypoints
# (the two days) that the truck then got inside PASS_M of within 3 s, and
# took off no more drives away.
NEAR_RISE_M = 0.25
# The tightest circle the truck drives forward (nav/truck.py
# DRIVEN_TURN_RADIUS_M, kept equal by a test; not imported, as truck.py
# brings the driver in). A waypoint closer than TURN_RADIUS_M - PASS_M to
# either circle's centre cannot be got within PASS_M of on any forward arc.
TURN_RADIUS_M = 0.70
# A near pass only once the truck has moved this far from where it was as
# the waypoint became the goal, so the waypoints of a tight bend drawn at
# the map page's 1 m spacing are never passed one after another where the
# truck stands. It cost the replay no drive away taken off.
NEAR_MOVE_M = 0.25
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
# The walls kept with the track, for fitting it on the next boot's map.
MIN_WALLS = 50
WALLS_KEPT = 3000
WALLS_EVERY_S = 30.0
# A saved track's fit is tried at most this often, and again sooner than
# FIT_STALE_S only once the live map has FIT_GROWTH times the walls it had.
FIT_EVERY_S = 10.0
FIT_STALE_S = 60.0
FIT_GROWTH = 1.15
# The name a track about to be replaced is kept under, with the time.
AUTOSAVE = "kept {:%m-%d %H:%M:%S}"
# Journal lines kept for the map page.
SAID_KEPT = 6
# /patrol/event messages waiting for the control loop to send (pop_event);
# it sends one a pass, so this only fills with nothing taking them.
EVENTS_KEPT = 32
# The /footprint columns the patrol fills (row_fields), as they read with
# no patrol. tools/lap_report.py joins each decision to its lap on these.
ROW_FIELDS = {
    "patrol_use": "",       # goal / patrol / explore / follow, "" with none
    "patrol_state": "",     # state(): driving, paused, finished... (exploring, holding)
    "patrol_track": "",     # track_id()
    "patrol_index": -1,     # the current waypoint, from 0
    "patrol_count": 0,
    "patrol_lap": -1,       # laps done
    "patrol_stuck_s": -1.0,  # the current waypoint's (explore: target's) seconds with no
                             # PROGRESS_M closer
}
# A command that carries an "id" (the map page's) is answered in
# page_state()'s acks, [id, taken, why], the newest ACKS_KEPT of them, so the
# page can tell a stroke the car took from one still on its way. why is ""
# when taken, else the warning the refusal said (refusal_why).
ACKS_KEPT = 64


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


def _thin(pts, n=WALLS_KEPT):
    pts = np.asarray(pts, float).reshape(-1, 2)
    if len(pts) > n:
        pts = pts[np.linspace(0, len(pts) - 1, n).astype(np.int64)]
    return pts


def merged_walls(view, old):
    """The view's walls, and those of `old` (same frame) that land where the
    view has seen nothing: a room driven last time and not this time is
    kept for the next fit. Thinned to WALLS_KEPT."""
    now = _occupied_points(view)
    old = np.asarray(old if old is not None else [], float).reshape(-1, 2)
    if len(old):
        from crawler.mapping import grid as gr
        codes = np.asarray(view.codes)
        i = np.floor((old[:, 0] - view.x0) / view.res).astype(np.int64)
        j = np.floor((old[:, 1] - view.y0) / view.res).astype(np.int64)
        inside = (i >= 0) & (i < codes.shape[0]) & (j >= 0) & (j < codes.shape[1])
        seen = np.zeros(len(old), bool)
        seen[inside] = codes[i[inside], j[inside]] != gr.UNKNOWN
        now = np.concatenate([now, old[~seen]])
    return np.round(_thin(now), 3).tolist()


def _xy_list(v):
    """[[x, y], ...] of finite numbers off a command's list, or []."""
    out = []
    if not isinstance(v, (list, tuple)):
        return out
    for p in v:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError, KeyError):
            return []
        if not (math.isfinite(x) and math.isfinite(y)):
            return []
        out.append((x, y))
    return out


def _near_m(v):
    """A near pass radius, metres, off a setting or a command: a number from
    0 to NEAR_PASS_MAX_M (bools are not), else None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) and 0.0 <= v <= NEAR_PASS_MAX_M else None


def _near_said(m):
    if m <= 0.0:
        return f"Near passes are off: a waypoint is passed within {PASS_M:.1f} m only"
    return f"Near passes are on: a waypoint come within {m:.2f} m of is passed once it is behind"


def near_miss(d, bearing, closest, near_m, pass_m=PASS_M, abeam_deg=NEAR_ABEAM_DEG,
              rise_m=NEAR_RISE_M, turn_radius_m=TURN_RADIUS_M):
    """Why a waypoint `d` metres off at `bearing` (radians CCW from the
    nose), `closest` the nearest the truck has been to it (d included),
    counts as passed though never within pass_m: "abeam", "away" or
    "circle"; None if it does not (or near_m is 0, or it was never within
    near_m)."""
    if near_m <= 0.0 or closest > near_m:
        return None
    if abs(math.degrees(bearing)) >= abeam_deg:
        return "abeam"
    if d >= closest + rise_m:
        return "away"
    ahead, side = d * math.cos(bearing), abs(d * math.sin(bearing))
    if d <= near_m and math.hypot(ahead, side - turn_radius_m) < turn_radius_m - pass_m:
        return "circle"
    return None


class Patrol(gl.Goal):
    """The single clicked goal, or the patrol. Control loop only."""

    def __init__(self, arrive_m=pg.ARRIVE_M, pass_m=PASS_M, dwell_s=DWELL_S,
                 say=None, store=None, map_view=None, clock=time.monotonic,
                 library=None, fitter=None, skip_stuck=False, settings=None,
                 near_pass_m=NEAR_PASS_M, tags=None):
        super().__init__()
        self.arrive_m, self.pass_m, self.dwell_s = float(arrive_m), float(pass_m), float(dwell_s)
        self._say = say
        self.single = gl.ClickGoal(arrive_m=arrive_m, say=say)
        self.store = store
        self.map_view = map_view
        self.clock = clock
        self.use = "goal"           # goal / patrol / explore
        # Explore (autopilot/explore.py): on while use is "explore", its
        # target the goal; the waypoint list is kept, paused, meanwhile.
        # (Imported here: search.py takes the patrol's clock numbers.)
        from crawler.autopilot import explore as ex
        self.explore = ex.Explore(map_view=map_view, say=self._explore_say)
        self._explore_last = None     # (distance, bearing) to its target
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
        self.passed_near = 0          # passed as a near miss (near_pass_m)
        self._closest = math.inf      # the closest the current one has been
        self._closest_at = None       # (frame, x, y) it was measured to
        self._near_from = None        # (frame, x, y): the truck as it became the goal
        self._near_moved = 0.0        # the furthest the truck has been from there
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
        self._walls = None            # the walls round the list, for a later fit
        self._walls_frame = None      # the map frame they are on
        self._walls_at = None
        self.name = None              # the track's name, if it was saved or loaded as one
        self.dirty = False            # changed since it was saved under its name
        self._strokes = []            # points appended per step, for unstroke
        self.library = library        # tracks.Library, or None
        self.fitter = fitter          # trackfit.Fitter, or None: fitted in line
        self._pending = None          # a track (tracks.py) not placed on this map yet
        self.fit_state = {"state": "none", "why": "", "tries": 0, "share": 0.0,
                          "agree": 0, "guess": None}
        self._fit_at = None
        self._fit_occ = 0
        self._fit_why = None
        self.said = collections.deque(maxlen=SAID_KEPT)
        self._said_n = 0              # lines said ever, for refusal_why
        # The track id (track_id()): the base it was saved or started as,
        # and the edits to the list since, so an edited track's laps are
        # never counted with the one it was before.
        self._base = None
        self._edits = 0
        # Each pass, skip, start and pause, for /patrol/event (pop_event).
        self._events = collections.deque(maxlen=EVENTS_KEPT)
        self._event_seq = 0
        self._ran_track = None        # the track id the last start drove
        self.acks = collections.deque(maxlen=ACKS_KEPT)
        # Whether a stuck waypoint is skipped: the settings file's, else
        # the default given (--patrol-skip-stuck).
        # The same for the near pass's radius (--patrol-near-pass-m).
        self.settings = settings      # a Store of {"skip_stuck": bool, "near_pass_m": m}, or None
        saved = settings.load() if settings is not None else None
        self._settings = dict(saved) if isinstance(saved, dict) else {}
        kept = self._settings.get("skip_stuck")
        self.skip_stuck = kept if isinstance(kept, bool) else bool(skip_stuck)
        self._log("info", f"Stuck waypoints are {'skipped' if self.skip_stuck else 'kept'}"
                  f" (skip {'on' if self.skip_stuck else 'off'}, "
                  f"{'as last set' if isinstance(kept, bool) else 'the default'})")
        near = _near_m(self._settings.get("near_pass_m"))
        given = _near_m(near_pass_m)
        if given is None:
            self._log("warning", f"Near pass radius {near_pass_m!r} is not from 0 to "
                      f"{NEAR_PASS_MAX_M:.1f} m; {NEAR_PASS_M:.1f} m instead")
            given = NEAR_PASS_M
        self.near_pass_m = near if near is not None else given
        self._log("info", f"{_near_said(self.near_pass_m)} "
                  f"({'as last set' if near is not None else 'the default'})")
        # The recordings' tags (telemetry/tags.SessionTags), or None. Its
        # lines go where explore's do: the journal and the map page's toasts.
        self.tags = tags
        if tags is not None:
            tags.say = self._explore_say
        if store is not None:
            data = tk.migrate(store.load())
            if data and data.get("points"):
                self._set_pending(data)

    def _keep_setting(self, key, value):
        """One setting into the settings file, the others in it kept."""
        self._settings[key] = value
        if self.settings is not None:
            self.settings.save(dict(self._settings))

    def _log(self, level, text):
        print(text)
        self.said.append((level, text))
        self._said_n += 1
        if self._say is not None:
            try:
                self._say(level, "Patrol", text)
            except Exception:
                pass

    def _explore_say(self, level, topic, text):
        """Explore's journal lines (it prints them itself): the journal under
        its own topic, and the map page's toasts as the patrol's are."""
        self.said.append((level, text))
        self._said_n += 1
        if self._say is not None:
            try:
                self._say(level, topic, text)
            except Exception:
                pass

    # ---- the goal the model is fed ---------------------------------------------------

    @property
    def goal(self):
        """(x, y, frame) the model drives to now, or None."""
        if self.use == "goal":
            return self.single.goal
        if self.use == "explore":
            return self.explore.target
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
        self._advance("path")
        if crop is not None and getattr(crop, "present", False):
            self._measure(crop)
        return True

    def holding(self):
        return (self._dwell_until is not None) or self.finished or not self.running

    def reached(self):
        if self.use == "goal":
            return self.single.reached()
        if self.use == "explore":
            # Nothing picked yet (no map, nowhere left): the arrival hold.
            return self.explore.target is None
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
                said = self._said_n
                took = self.command(what, crop)
                cid = what.get("id") if isinstance(what, dict) else None
                if isinstance(cid, str) and 0 < len(cid) <= 64:
                    why = "" if took is True else self.refusal_why(what, said)
                    self.acks.append([cid, took is True, why])
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
        if self.use == "explore":
            # Its target's clock, on the crop's time, only here: in auto.
            self.explore.step(crop)
            self._measure_explore(crop)
        now = float(self.clock())
        dt = 0.0 if self._last_obs is None else min(max(now - self._last_obs, 0.0),
                                                   CLOCK_STEP_MAX_S)
        self._last_obs = now
        if (self.use != "patrol" or self.holding() or self.last is None
                or not self.present):
            return
        was_wp, was_best = self._wp_time, self._since_best
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
        if why is not None and not self.skip_stuck:
            # Kept as the goal: said once as it goes stuck, not skipped.
            if (was_best < NO_PROGRESS_S - 1e-6 <= self._since_best
                    or was_wp < TIMEOUT_S - 1e-6 <= self._wp_time):
                self._log("warning", f"Waypoint {self.index + 1} of {len(self.points)} "
                          f"stuck: {why}, {d:.2f} m off; kept (skip is off)")
            return
        if why is not None:
            self.skipped += 1
            self._in_row += 1
            self._log("warning", f"Waypoint {self.index + 1} of {len(self.points)} "
                      f"skipped: {why}, {d:.2f} m off")
            if self._in_row >= len(self.points):
                self.running = False
                self._event("skip", self.index, "stuck")
                self._event("pause", self.index, "all skipped")
                self._log("warning", "Every waypoint skipped in a row - the patrol "
                          "is paused; holding neutral")
                self._save()
                return
            self._advance("stuck")
            self._measure(crop)

    def refusal_why(self, msg, said_before):
        """Why a command was refused, for its ack: the newest warning it
        said (said_before is _said_n before it ran), else its name. The page
        shows this word for word, so a refusal never reads as just "refused"
        or as some older, unrelated warning."""
        k = min(self._said_n - said_before, len(self.said))
        for level, text in reversed(list(self.said)[len(self.said) - k:]):
            if level == "warning":
                return text
        name = msg.get("cmd") if isinstance(msg, dict) else None
        return f"{str(name).capitalize()} refused"

    def command(self, msg, crop=None):
        """One {"cmd": ...}. True if it did something, False if it was
        refused (said why), None if it is not a command at all."""
        was = self.use
        took = self._command(msg, crop)
        if was == "explore" and self.use != "explore":
            # Start, pause, a single goal or an edit: explore and the patrol
            # are never both on.
            self._stop_explore()
        return took

    def _stop_explore(self):
        self.explore.stop()
        self._explore_last = None
        self._log("info", "Explore off" + (" - the patrol has the goal" if self.running
                                            else " - holding neutral"))

    def _explore_command(self, msg, crop):
        on = msg.get("on")
        if on is None:
            on = self.use != "explore"
        if not isinstance(on, bool):
            self._log("warning", f"Explore refused: on is {on!r}, not true or false")
            return False
        if not on:
            if self.use == "explore":
                # command() says it and stops it, as for any way off.
                self.use, self.running = "patrol", False
            return True
        if self.use == "explore":
            return True
        if self.running:
            self._event("pause", self.index, "explore")
        self.use, self.running, self.tracing = "explore", False, False
        self._dwell_until = None
        self.explore.start(crop)
        self._log("info", "Explore on: the edge of the map is the goal, then the far "
                  "corners; START on the pad drives it")
        return True

    def _command(self, msg, crop=None):
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
            restart = self.finished or not (0 <= self.index < len(self.points))
            if restart:
                self.index, self.direction, self.laps, self.finished = 0, 1, 0, False
            # Fresh: from the first, or on a list it has not driven as it is
            # now; a resume carries on the waypoint it was paused on.
            fresh = restart or self._ran_track != self.track_id()
            self._ran_track = self.track_id()
            was_running = self.running
            self.running, self._in_row = True, 0
            self._reset_clock()
            if not was_running:
                self._event("start", self.index, "fresh" if fresh else "resume")
            self._log("info", f"Patrol {self.mode} started at waypoint {self.index + 1} "
                      f"of {len(self.points)}")
        elif name == "explore":
            return self._explore_command(msg, crop)
        elif name == "pause":
            if self.use == "explore":
                self.use = "patrol"
            if self.running:
                self._event("pause", self.index, "pause")
            self.running = False
            self._dwell_until = None
            self._log("info", "Patrol paused - holding neutral")
        elif name == "undo":
            if not self.points:
                self._log("warning", "Nothing to undo: no waypoints")
                return False
            self._remove(len(self.points) - 1)
            self._log("info", f"Waypoint {len(self.points) + 1} undone, "
                      f"{len(self.points)} left")
        elif name == "clear":
            if self._pending is not None:
                self._keep_pending("the track was cleared")
            elif self.points and (self.dirty or self.name is None):
                self._keep_current("the track was cleared")
            self.points, self.running, self.tracing, self.finished = [], False, False, False
            self.index, self.direction = 0, 1
            self._strokes, self.name, self.dirty = [], None, False
            self._walls, self._walls_frame = None, None
            self._base, self._edits = None, 0
            self._log("info", "Patrol cleared")
        elif name in ("stroke", "unstroke", "move", "delete"):
            if self._edit(name, msg, crop) is False:
                return False
        elif name in ("save", "load", "forget"):
            if self._library_command(name, msg) is False:
                return False
        elif name == "place":
            if self._place_command(msg) is False:
                return False
        elif name == "skip":
            on = msg.get("on")
            if on is None:
                on = not self.skip_stuck
            if not isinstance(on, bool):
                self._log("warning", f"Skip refused: on is {on!r}, not true or false")
                return False
            self.skip_stuck = on
            self._keep_setting("skip_stuck", on)
            self._log("info", "Stuck waypoints are skipped from now on" if on else
                      "Stuck waypoints are kept from now on: no timed skip")
            return True
        elif name == "tags":
            if self.tags is None:
                self._log("warning", "Tags refused: this car keeps no tags")
                return False
            return self.tags.command(msg)
        elif name == "near":
            m = _near_m(msg.get("m"))
            if m is None:
                self._log("warning", f"Near refused: m is {msg.get('m')!r}, not a distance "
                          f"from 0 to {NEAR_PASS_MAX_M:.1f} m")
                return False
            self.near_pass_m = m
            self._keep_setting("near_pass_m", m)
            self._log("info", f"{_near_said(m)}, from now on")
            return True
        elif name == "mode":
            mode = msg.get("mode")
            if mode is None:
                mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
            if mode not in MODES:
                self._log("warning", f"Patrol has no mode {mode!r}; one of {', '.join(MODES)}")
                return False
            if mode != self.mode and self.points:
                self._changed()
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

    def append(self, x, y, frame, stroke=False):
        """One waypoint on the end. stroke=True: part of the newest stroke
        (unstroke takes it back with the rest), else a step of its own."""
        x, y = float(x), float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            self.refused += 1
            return False
        if len(self.points) >= MAX_WAYPOINTS:
            self.refused += 1
            self._log("warning", f"Waypoint refused: {MAX_WAYPOINTS} is the most")
            return False
        if self._pending is not None:
            self._keep_pending("a new waypoint was set before it was placed")
        self.points.append(self._waypoint(x, y, frame))
        self._changed()
        if stroke and self._strokes:
            self._strokes[-1] += 1
        else:
            self._strokes.append(1)
        self.dirty = True
        self._save()
        return True

    def _waypoint(self, x, y, frame):
        w = _Waypoint()
        w._seq = self._seq
        w.set_goal(float(x), float(y), int(frame))
        w.said.clear()
        return w

    def _remove(self, i):
        """Take waypoint i out, the place in the list kept on the same one."""
        self.points.pop(i)
        k, at = 0, 0
        while k < len(self._strokes):
            if i < at + self._strokes[k]:
                self._strokes[k] -= 1
                if self._strokes[k] == 0:
                    self._strokes.pop(k)
                break
            at += self._strokes[k]
            k += 1
        if i < self.index:
            self.index -= 1
        elif i == self.index:
            self._reset_clock()
        if self.index >= len(self.points):
            self.index = max(len(self.points) - 1, 0)
            self._reset_clock()
        if not self.points:
            self.running = False
            self._strokes = []
        self.dirty = True
        self._changed()

    # ---- editing from the map page ---------------------------------------------------

    def _edit(self, name, msg, crop):
        """stroke, unstroke, move, delete. False if refused."""
        if name == "unstroke":
            if not self._strokes:
                self._log("warning", "Nothing to undo: every stroke is already "
                          "taken back")
                return False
            n = self._strokes[-1]
            for _ in range(n):
                self._remove(len(self.points) - 1)
            self._log("info", f"Stroke of {n} waypoints undone, {len(self.points)} left")
            return True
        frame = self.frame_now
        if frame is None and crop is not None:
            frame = int(getattr(crop, "frame", 0))
        if frame is None:
            self._log("warning", f"{name.capitalize()} refused: no SLAM map yet")
            return False
        if name == "stroke":
            pts = _xy_list(msg.get("points"))
            if not pts:
                self._log("warning", "Stroke refused: it had no points the car could read")
                return False
            if self._pending is not None:
                self._keep_pending("a new track was drawn before it was placed")
            had = len(self.points)
            self._strokes.append(0)
            for x, y in pts:
                if not self.append(x, y, frame, stroke=True):
                    break
            if self._strokes and self._strokes[-1] == 0:
                self._strokes.pop()
            if len(self.points) == had:
                return False
            self.mode, self.finished = "loop", False
            if self.use not in ("patrol", "explore"):
                # Drawn while exploring: kept, paused, for a Start.
                self.use, self.running = "patrol", False
            self._log("info", f"Drew waypoints {had + 1}-{len(self.points)}; the loop "
                      f"closes back to 1")
            return True
        i = msg.get("i")
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(self.points):
            self._log("warning", f"{name.capitalize()} refused: no waypoint {i!r}")
            return False
        if name == "delete":
            self._remove(i)
            self._log("info", f"Waypoint {i + 1} deleted, {len(self.points)} left")
            return True
        xy = _xy_list([[msg.get("x"), msg.get("y")]])
        if not xy:
            return False
        self.points[i] = self._waypoint(xy[0][0], xy[0][1], frame)
        if i == self.index:
            self._reset_clock()
        self.dirty = True
        self._changed()
        self._log("info", f"Waypoint {i + 1} moved to ({xy[0][0]:.2f}, {xy[0][1]:.2f})")
        return True

    # ---- named tracks ----------------------------------------------------------------

    def track(self):
        """The track as it would be saved now (tracks.py), or None."""
        if self._pending is not None:
            return dict(self._pending)
        if not self.points:
            return None
        return dict(self._saved(), name=self.name)

    def _later(self, fn, *args):
        """Library writes, off the control loop."""
        if getattr(self.library, "synchronous", False):
            fn(*args)
            return
        threading.Thread(target=fn, args=args, name="tracks", daemon=True).start()

    def _keep(self, data, why):
        if self.library is None or not data or not data.get("points"):
            return
        name = tk.clean_name(f"{data.get('name') or 'track'} "
                             f"{AUTOSAVE.format(datetime.datetime.now())}")
        self._later(self.library.save, name, data)
        self._log("info", f"Kept the track as '{name}' in the saved tracks: {why}")

    def _keep_pending(self, why):
        data, self._pending = self._pending, None
        self._fit_reset()
        self._keep(data, why)

    def _keep_current(self, why):
        self._keep(self.track(), why)

    def _library_command(self, name, msg):
        if self.library is None:
            self._log("warning", "No saved tracks on this run")
            return False
        track_name = tk.clean_name(msg.get("name"))
        if track_name is None:
            self._log("warning", f"{name.capitalize()} refused: no name")
            return False
        if name == "save":
            data = self.track()
            if data is None:
                self._log("warning", "Save refused: no waypoints")
                return False
            if self.library.has(track_name) and not msg.get("overwrite"):
                self._log("warning", f"Save refused: '{track_name}' is saved already")
                return False
            # Saved under a name, it is that track from here on: a base of
            # its own, no edits, written into the file it is saved as.
            base = tk.base_of(dict(data, name=track_name))
            data.update(base=base, edits=0)
            self._later(self.library.save, track_name, data)
            if self._pending is not None:
                self._pending.update(name=track_name, base=base, edits=0)
            self.name, self.dirty = track_name, False
            self._base, self._edits = base, 0
            self._log("info", f"Saved '{track_name}': {len(data['points'])} waypoints")
            return True
        if name == "forget":
            if not self.library.has(track_name):
                self._log("warning", f"No saved track '{track_name}'")
                return False
            self._later(self.library.delete, track_name)
            self._log("info", f"Deleted the saved track '{track_name}'")
            return True
        data = self.library.load(track_name)
        if data is None or not data.get("points"):
            self._log("warning", f"No saved track '{track_name}'")
            return False
        if self._pending is not None and self._pending.get("name") != track_name:
            self._keep_pending("another track was loaded")
        elif self.points and (self.dirty or self.name is None):
            self._keep_current("another track was loaded")
        self.points, self.running, self.tracing, self.finished = [], False, False, False
        self.index, self.direction, self._strokes = 0, 1, []
        self._walls, self._walls_frame = None, None
        self._set_pending(data)
        if self.use != "explore":
            self.use = "patrol"
        self._log("info", f"Loaded '{track_name}': {len(data['points'])} waypoints, "
                  "placing it on this map")
        return True

    # ---- placing a saved track -------------------------------------------------------

    def _set_pending(self, data):
        self._pending = data
        self.name = data.get("name")
        self._base = data.get("base") or tk.base_of(data)
        self._edits = int(data.get("edits") or 0)
        self.mode = data["mode"] if data.get("mode") in MODES else self.mode
        self.dirty = False
        self._fit_reset()
        if not data.get("walls"):
            self.fit_state["state"] = "no walls"

    def _fit_reset(self):
        self.fit_state = {"state": "none", "why": "", "tries": 0, "share": 0.0,
                          "agree": 0, "guess": None}
        self._fit_at, self._fit_occ, self._fit_why = None, 0, None

    def _place_command(self, msg):
        """{"cmd": "place", "yaw": rad, "x": m, "y": m}: the unplaced track
        put on this map through that transform (its own frame -> this map);
        with none unplaced, the list on this map moved by it (a nudge)."""
        try:
            fit = (float(msg.get("yaw", 0.0)), float(msg.get("x", 0.0)),
                   float(msg.get("y", 0.0)))
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(v) for v in fit) or self.frame_now is None:
            self._log("warning", "Place refused: no SLAM map yet" if self.frame_now is None
                      else "Place refused: not a transform")
            return False
        if self._pending is not None:
            self._place_pending(fit, "placed by hand")
            return True
        places = [w.places.get(self.frame_now) for w in self.points]
        if not places or any(p is None for p in places):
            self._log("warning", "Nudge refused: no waypoints on this map")
            return False
        from crawler.mapping import trackfit as tf
        moved = tf.transform(fit, [p[:2] for p in places])
        self.points = [self._waypoint(x, y, self.frame_now) for x, y in moved]
        if self._walls and self._walls_frame == self.frame_now:
            self._walls = np.round(tf.transform(fit, self._walls), 3).tolist()
        self._reset_clock()
        self.dirty = True
        self._log("info", f"Track nudged {math.degrees(fit[0]):+.0f} deg, "
                  f"({fit[1]:+.2f}, {fit[2]:+.2f}) m")
        return True

    def _place_pending(self, fit, how):
        from crawler.mapping import trackfit as tf
        data, self._pending = self._pending, None
        frame = self.frame_now
        pts = tf.transform(fit, data["points"])
        self.points = [self._waypoint(x, y, frame) for x, y in pts[:MAX_WAYPOINTS]]
        self._strokes = [len(self.points)] if self.points else []
        self.index, self.direction, self.finished, self.laps = 0, 1, False, 0
        walls = data.get("walls") or []
        self._walls = np.round(tf.transform(fit, walls), 3).tolist() if walls else None
        self._walls_frame = frame if walls else None
        self._walls_at = None
        self.name, self.dirty = data.get("name"), False
        if self.use != "explore":
            # A track placed while exploring waits paused; explore goes on.
            self.use = "patrol"
        self.running, self.restored = False, True
        self.fit_state = dict(self.fit_state, state="placed", why=how)
        self._save()
        name = f"'{self.name}'" if self.name else "The saved track"
        self._log("info", f"{name} placed on map {frame} ({how}): {len(self.points)} "
                  f"waypoints, turned {math.degrees(fit[0]):.0f} deg and moved "
                  f"({fit[1]:.2f}, {fit[2]:.2f}) m, {self.mode}, paused until start")

    def _fit_pending(self, crop):
        """Try the unplaced track's walls on the live map: a result in, and
        another try out when it is due."""
        if self._pending is None or self.frame_now is None or not self.present:
            return
        fitter = self.fitter
        if fitter is not None:
            done = fitter.result()
            if done is not None:
                self._fit_result(*done)
                if self._pending is None:
                    return
        walls = self._pending.get("walls") or []
        if len(walls) < MIN_WALLS or self.map_view is None:
            return
        if fitter is not None and fitter.busy:
            return
        now = float(self.clock())
        if self._fit_at is not None and now - self._fit_at < FIT_EVERY_S:
            return
        try:
            view = self.map_view()
        except Exception:
            view = None
        if view is None:
            return
        occ = int(np.count_nonzero(np.asarray(view.codes) == 2))
        if occ < MIN_WALLS:
            return
        if (self._fit_at is not None and occ < self._fit_occ * FIT_GROWTH
                and now - self._fit_at < FIT_STALE_S):
            return
        self._fit_at, self._fit_occ = now, occ
        self.fit_state["state"] = "searching" if self.fit_state["state"] == "none" \
            else self.fit_state["state"]
        tag = (self.frame_now, self._seq, id(self._pending))
        if fitter is None:
            from crawler.mapping import trackfit as tf
            self._fit_result(tag, tf.fit(walls, view))
        else:
            fitter.submit(tag, walls, view)

    def _fit_result(self, tag, fit):
        if self._pending is None or tag != (self.frame_now, self._seq, id(self._pending)):
            self._fit_at = None          # the map moved on under it; try again
            return
        st = self.fit_state
        st["tries"] += 1
        if fit is None:
            st.update(state="unsure", why="nothing on the map to fit to yet")
            return
        st.update(share=fit.share, agree=fit.agree, guess=[fit.yaw, fit.x, fit.y])
        if fit.ok:
            self._place_pending((fit.yaw, fit.x, fit.y),
                                f"fitted, {fit.share:.0%} of {fit.agree + fit.disagree} "
                                "seen walls agree")
            return
        st.update(state="unsure", why=fit.why)
        kind = fit.why.split(" ")[0]
        if kind != self._fit_why:
            self._fit_why = kind
            name = f"'{self._pending.get('name')}'" if self._pending.get("name") else \
                "The saved track"
            self._log("info", f"{name} is not placed yet: {fit.why}; trying again as "
                      "the map grows (or place it by hand on the map page)")

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
            self.explore.follow(L)
            self._carry_walls(L)
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
        if self.use == "explore":
            self.explore.step(crop, clocked=False)
            self._measure_explore(crop)
        if self._pending is not None:
            self._fit_pending(crop)
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
                self._advance("dwell")
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
            return
        if self._last_stop() or self.dwell_s > 0.0:
            return
        why = self._near(db)
        if why is not None:
            self._in_row = 0
            self.passed_near += 1
            self._log("info", f"Waypoint {self.index + 1} of {len(self.points)} passed near "
                      f"({why}): {self._closest:.2f} m at the closest, now {db[0]:.2f} m "
                      f"at {math.degrees(db[1]):+.0f} deg")
            self._advance("near")
            self._measure(crop)

    def _near(self, db):
        """near_miss() for the current waypoint off this (distance,
        bearing), with the closest it has come. The closest starts again
        whenever the waypoint's place moves (a carry along a link, a move),
        so a near pass is never off where the point used to be."""
        place = self.current_place()
        at = (self.frame_now, place[0], place[1])
        if (self._closest_at is None or at[0] != self._closest_at[0]
                or math.hypot(at[1] - self._closest_at[1], at[2] - self._closest_at[2]) > 0.05):
            self._closest, self._closest_at = math.inf, at
        self._closest = min(self._closest, db[0])
        x, y = self.pose[0], self.pose[1]
        if self._near_from is None or self._near_from[0] != self.frame_now:
            self._near_from, self._near_moved = (self.frame_now, x, y), 0.0
        self._near_moved = max(self._near_moved, math.hypot(x - self._near_from[1],
                                                            y - self._near_from[2]))
        if self._near_moved < NEAR_MOVE_M:
            return None
        return near_miss(db[0], db[1], self._closest, self.near_pass_m, self.pass_m)

    def _measure_explore(self, crop):
        g = self.explore.target
        self._explore_last = (pg.distance_bearing(crop, g)
                              if (g is not None and self.present
                                  and int(g[2]) == self.frame_now) else None)

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

    def _advance(self, by="radius"):
        """On from the current waypoint: passed within its radius ("radius"),
        gone by on the path ("path"), missed near and behind ("near"), after
        its dwell ("dwell"), or skipped ("stuck")."""
        n = len(self.points)
        if n == 0:
            return
        was = self.index
        kind = "skip" if by == "stuck" else "pass"
        if self._last_stop():
            self.finished = True
            self.laps += 1
            self._event("finish" if kind == "pass" else kind, was, by)
            self._log("info", f"Patrol done: waypoint {self.index + 1} of {n} reached - "
                      "holding neutral until start")
            self._save()
            return
        i, self.direction = self._next(self.index, self.direction)
        # The event before the lap count and the index move on, so it says
        # the waypoint it was about and the clock it had; next and lap after.
        event = self._event(kind, was, by)
        if i == 0 and self.mode == "loop":
            self.laps += 1
            self._log("info", f"Lap {self.laps} of {n} waypoints done, "
                      f"{self.skipped} skipped so far")
        elif i == 0 and self.mode == "patrol":
            self.laps += 1
            self._log("info", f"Lap {self.laps} there and back done, "
                      f"{self.skipped} skipped so far")
        self.index = i
        event.update(next=int(i), lap=int(self.laps))
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
        self._closest, self._closest_at = math.inf, None
        self._near_from, self._near_moved = None, 0.0

    def _trace(self, crop):
        if not self.points:
            self.append(crop.x, crop.y, self.frame_now)
            return
        last = self.points[-1].places.get(self.frame_now)
        if last is None or math.hypot(crop.x - last[0], crop.y - last[1]) >= TRACE_M:
            self.append(crop.x, crop.y, self.frame_now)

    # ---- across a restart ------------------------------------------------------------

    def _keep_walls(self):
        """The walls round the list, every WALLS_EVERY_S while there is one
        on this map: the live map's, and the kept ones it has not seen."""
        if (self.store is None or self.map_view is None or self.frame_now is None
                or not self.points or self._pending is not None):
            return
        if any(w.places.get(self.frame_now) is None for w in self.points):
            return
        now = float(self.clock())
        if self._walls_at is not None and now - self._walls_at < WALLS_EVERY_S:
            return
        self._walls_at = now
        try:
            view = self.map_view()
        except Exception:
            return
        if view is None or len(_occupied_points(view)) < MIN_WALLS:
            return
        old = self._walls if self._walls_frame == self.frame_now else None
        self._walls = merged_walls(view, old)
        self._walls_frame = self.frame_now
        self._save()

    def _carry_walls(self, L):
        """The kept walls along a link, as the waypoints go."""
        if self._walls and self._walls_frame == L.frm:
            from crawler.mapping import trackfit as tf
            self._walls = np.round(tf.transform((L.yaw, L.x, L.y), self._walls), 3).tolist()
            self._walls_frame = L.to

    def _saved(self):
        """The track as written: the points on the frame the walls are on
        if they all have a place there, else on the one the truck is on (or
        the newest was set on), without walls when those are elsewhere."""
        frame = self._walls_frame
        if frame is None or any(w.places.get(frame) is None for w in self.points):
            frame = self.frame_now
            if frame is None or any(w.places.get(frame) is None for w in self.points):
                frame = self.points[-1].goal[2] if self.points else 0
        places = [w.places.get(frame) for w in self.points]
        pts = [[float(p[0]), float(p[1])] for p in places if p is not None]
        walls = self._walls if (self._walls and self._walls_frame == frame) else []
        return {"version": tk.VERSION, "name": self.name, "mode": self.mode,
                "points": pts, "walls": walls, "saved_at": time.time(),
                "base": self._base, "edits": int(self._edits)}

    def _save(self):
        """The working track to the store. Never while a saved track is
        waiting to be placed: that one stays on disk as it is."""
        if self.store is None or self._pending is not None:
            return
        self.store.save(self._saved())

    # ---- the lap report's ids and events ---------------------------------------------

    def _changed(self):
        """The waypoint list (or its order) changed: one more edit on the
        track id. A list started from nothing gets a base of its own."""
        if self._base is None:
            self._base = "n" + os.urandom(3).hex()
        self._edits += 1

    def track_id(self):
        """The track the laps are on: "<name or unsaved>@<base>", with
        "+<edits>" once the list has changed since that base. The base of a
        saved track is the hash of its file (tracks.base_of), so the same
        track loaded on another boot has the same id; "" with no list."""
        if not self.points and self._pending is None:
            return ""
        out = f"{self.name or 'unsaved'}@{self._base or 'new'}"
        return out + (f"+{self._edits}" if self._edits else "")

    def _event(self, kind, index, by=""):
        """One /patrol/event, queued for the control loop: kind pass, skip,
        finish (the last stop of a once), start or pause; index the
        waypoint it is about; wp_s and stuck_s that waypoint's clocks;
        mono_s the patrol's clock at the decision it happened on. Returned,
        so _advance can fill in next and lap once they have moved on."""
        self._event_seq += 1
        tx, ty, _ = self.pose if self.pose is not None else (0.0, 0.0, 0.0)
        event = {"seq": self._event_seq, "kind": kind, "by": by, "index": int(index),
                 "next": int(self.index), "count": len(self.points), "lap": int(self.laps),
                 "mode": self.mode, "track": self.track_id(),
                 "wp_s": round(float(self._wp_time), 2),
                 "stuck_s": round(float(self._since_best), 2),
                 "mono_s": float(self.clock()),
                 "frame": -1 if self.frame_now is None else int(self.frame_now),
                 "truck_x": float(tx), "truck_y": float(ty)}
        self._events.append(event)
        return event

    def pop_event(self):
        """The oldest /patrol/event not sent yet, or None."""
        return self._events.popleft() if self._events else None

    def row_fields(self):
        """ROW_FIELDS for this decision's /footprint row."""
        stuck = self.explore.since_best() if self.use == "explore" else self._since_best
        return {"patrol_use": self.use, "patrol_state": self.state(),
                "patrol_track": self.track_id(), "patrol_index": int(self.index),
                "patrol_count": len(self.points), "patrol_lap": int(self.laps),
                "patrol_stuck_s": round(float(stuck), 2)}

    # ---- out -------------------------------------------------------------------------

    def state(self):
        if self.use == "goal":
            return "goal"
        if self.use == "explore":
            return "exploring" if self.explore.target is not None else "holding"
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

    def _explore_place(self):
        """(x, y, 0.0) of explore's target on the map the truck is on, or
        None: what current_place() is for a waypoint."""
        g = self.explore.target
        if g is None or self.frame_now is None or int(g[2]) != self.frame_now:
            return None
        return (g[0], g[1], 0.0)

    def approx_m(self):
        if self.use == "explore":
            return 0.0
        place = self.current_place()
        return float(place[2]) if place is not None else 0.0

    def status(self):
        """/goal/status: the goal the model is fed, in goal.ClickGoal's schema."""
        if self.use == "goal":
            return self.single.status()
        if self.use == "explore":
            place = self._explore_place()
            last = self._explore_last
        else:
            place, last = self.current_place(), self.last
        d, b = last if last is not None else (-1.0, 0.0)
        have = place is not None
        x, y = (place[0], place[1]) if have else (0.0, 0.0)
        e = self.approx_m()
        goal_set = have if self.use == "explore" else bool(self.points)
        return {"goal_set": bool(goal_set), "x": float(x), "y": float(y),
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
        exploring = self.use == "explore"
        # Exploring: the place and distance are its target's.
        place = self._explore_place() if exploring else self.current_place()
        last = self._explore_last if exploring else self.last
        have = place is not None
        d = last[0] if (last is not None and have) else -1.0
        x, y = (place[0], place[1]) if have else (0.0, 0.0)
        e = self.approx_m()
        return {"use": self.use, "state": self.state(), "mode": self.mode,
                "running": bool(self.running), "tracing": bool(self.tracing),
                "index": int(self.index), "number": int(self.index + 1) if self.points else 0,
                "count": len(self.points), "laps": int(self.laps),
                "skipped": int(self.skipped),
                # The current waypoint's time without getting PROGRESS_M
                # closer, and whether that skips it at NO_PROGRESS_S.
                "stuck_s": round(float(self.explore.since_best() if exploring
                                       else self._since_best), 2),
                "skip_stuck": bool(self.skip_stuck),
                "passed_on_path": int(self.passed_on_path),
                # Those passed as a near miss, and the radius (0: off).
                "passed_near": int(self.passed_near), "near_pass_m": float(self.near_pass_m),
                "x": float(x), "y": float(y),
                "frame_now": -1 if self.frame_now is None else int(self.frame_now),
                "on_map": bool(have and self.present), "distance_m": float(d),
                "approx": e > 0.0, "approx_m": e, "restored": bool(self.restored),
                "pending": self._pending is not None,
                # The truck's map pose the waypoint is measured from, on the
                # same map: a recording pairs the two with no other topic.
                "truck_x": float(tx), "truck_y": float(ty), "truck_yaw_deg": float(tyaw),
                "track": self.track_id()}

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

    def page_state(self):
        """The map page's view of the patrol (telemetry/buttons.py), built in
        the control loop when the page asks: the waypoints on the map the
        truck is on (None for one with no place there), and a track not
        placed yet in its own frame, with the best fit so far as a guess."""
        frame = self.frame_now
        pts = []
        for w in self.points:
            p = w.places.get(frame) if frame is not None else None
            pts.append(None if p is None else [round(p[0], 3), round(p[1], 3)])
        pending = None
        if self._pending is not None:
            pending = {"name": self._pending.get("name"), "mode": self._pending.get("mode"),
                       "points": self._pending.get("points") or [],
                       "walls": self._pending.get("walls") or [],
                       "id": id(self._pending)}
        fit = dict(self.fit_state)
        fit["busy"] = bool(getattr(self.fitter, "busy", False))
        place = self._explore_place()
        explore = dict(self.explore.counts(), on=self.use == "explore",
                       target=None if place is None else [round(place[0], 3),
                                                           round(place[1], 3)],
                       kind=self.explore.kind,
                       distance_m=round(float(self._explore_last[0]), 2)
                       if (self._explore_last is not None and place is not None) else -1.0)
        return {"frame": frame, "present": bool(self.present),
                "pose": list(self.pose) if (self.pose is not None and self.present) else None,
                "use": self.use, "state": self.state(), "mode": self.mode,
                "running": bool(self.running), "tracing": bool(self.tracing),
                "index": int(self.index), "points": pts, "laps": int(self.laps),
                "skipped": int(self.skipped), "name": self.name, "dirty": bool(self.dirty),
                "stuck_s": round(float(self._since_best), 1),
                "skip_stuck": bool(self.skip_stuck), "near_pass_m": float(self.near_pass_m),
                "strokes": len(self._strokes), "pending": pending, "fit": fit,
                "explore": explore,
                "tags": self.tags.state() if self.tags is not None else None,
                "said": [list(x) for x in self.said],
                "acks": [list(a) for a in self.acks]}
