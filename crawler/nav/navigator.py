"""The goal handling: where to go, the path there, and when to give up.

Two ways to have a goal:

  goto   a spot, clicked in Foxglove (click.py) or handed over by anything
         else. Arrives within ARRIVE_M and stops, and holds there. A spot it
         cannot stand on or get to (inside a wall, in unknown, shut off) is
         the reachable point nearest it instead, by path cost among the
         nearest few, AIM_CLEAR_M clear of anything (_aim); held there as
         "closest". While driving to one the closest point is looked for
         again every CLOSEST_REFRESH_S, and on getting there once more, as
         the map has grown: a better one by CLOSEST_GAIN_M goes on.
  rove   no spot: the best frontier (frontier.py), then the next, for as long
         as auto runs. When the map has no reachable frontier left it patrols
         the place it has been least.

Every decision (tick), in order:

  1. Which map. The global one while the SLAM is sure of itself (confidence
     at or over CONFIDENCE_MIN); below that, the truck-centred LocalMap,
     which only holds the last few seconds of scans and so cannot have been
     smeared by a pose that has drifted. A goal past the local window is
     steered for: the reachable point in the window nearest it. Nothing else
     changes between the two, so indoors and out are one mode. No view at
     all (the SLAM lost, or not started) is a stop with the path dropped:
     a path from before cannot be trusted against a pose after, so the
     goal is planned afresh once there is a view again.
  2. The cost map, rebuilt when the map changed (its version) or the truck
     has moved off the window it was built for.
  3. Arrived? Then stop.
  4. A path, re-planned when there is none, when the next PATH_CHECK_M of it
     now runs through something (a door shut, a person, a wall the map had
     not seen yet), or when the truck is CROSS_TRACK_REPLAN_M off it.
     Planning is rate-limited to once every REPLAN_MIN_S, and it never runs
     inside tick(): see "Planning off the decision" below.
  5. Give up. A goal with no plan MAX_PLAN_FAILS times running is given up
     (rove blacklists the frontier for BLACKLIST_S and takes the next one);
     between tries it waits WAIT_S, which is what a person standing in a
     doorway needs. A goal the truck makes no progress toward for
     PROGRESS_S while driving is a wait first (WAIT_S stopped, then a fresh
     plan), and after that a fresh plan again from where it stands, and
     after MAX_RECOVERS of those with no progress between them it is given
     up. There is no back-out: the map is the way out, and what the
     forward floor or a STUCK stopped the truck on goes into the cost map
     (blocked_ahead) for the next plan to go round. Params.backout puts the
     old reverse at lock back, for the simulator to compare against.
  6. Follow: the follower's command for this pose. A cusp, where the path
     changes direction, is CUSP_STOP_DECISIONS of standing still first.
  7. Look ahead: the command is run forward FORECAST_S at the speed the
     truck is doing (FORECAST_MIN_M at least), on the curvature it will
     actually drive, with the footprint. If that touches anything (closer
     than FORECAST_CLEAR_M, or closer than the truck already is), it stops
     and plans again instead.

Planning off the decision. A plan is up to a second or two of search on
the laptop and several times that on the Pi, and tick() runs inside the
autopilot's 10 Hz decision; pigpio holds the last pulse it was given, so a
decision that blocks is a truck driving on an old command. So tick() hands
the search (frontier choice, goal snapping and the plan) to a planner
(ThreadPlanner on the car: one worker thread, one job at a time, a
PLAN_BUDGET_S deadline) and goes on deciding. Until a path is in hand the
output is a stop. A plan that comes back is checked against the cost map as
it is now before it is used, and one asked for before a new goal, a stop or
a back-out is thrown away.

The plan radius follows the ground. The planner's curvature is the
measured lock's (planner.PLAN_CURVATURE) times the follower's learned gain
when that is under 1 (to GAIN_PLAN_MIN): an axle that only turns 60% of
what it is told is planned for as one, instead of being handed turns it
cannot make.

Rove. A refresh every ROVE_REPLAN_S looks at the frontiers again and only
changes target for one that scores SWITCH_BONUS_M better than the current
(so it does not flip between two), or when the current one is no longer a
frontier at all.

It never touches a servo. tick() returns an Output, and the driver decides
what reaches the wheels (driver.py): the forward floor, the rear stop and
STUCK are underneath every one of these.
"""

import math
import threading
import time
from typing import NamedTuple, Optional

import numpy as np

from crawler.nav import costmap as cmod
from crawler.nav import follower as fl
from crawler.nav import frontier as fr
from crawler.nav import planner as pl
from crawler.nav import truck as tk

IDLE = "idle"
GOTO = "goto"
ROVE = "rove"

# What status says. Driving is the only one that moves the truck.
ST_IDLE = "idle"
ST_DRIVING = "driving"
ST_PLANNING = "planning"
ST_ARRIVED = "arrived"
ST_FAILED = "gave up"
ST_EXPLORED = "explored"
ST_NO_MAP = "no map"
ST_RECOVER = "recover"
ST_CUSP = "cusp"
ST_WAIT = "wait"
# Point navigation's two more: holding at the reachable point nearest a goal
# that cannot be reached itself, and my hand on the pad (driver.NavDriver).
ST_CLOSEST = "closest"
ST_HAND = "takeover"
# What a stopped goto holds at until the next goal: it never moves on its own
# from any of these.
HELD = (ST_ARRIVED, ST_CLOSEST, ST_FAILED)

CONFIDENCE_MIN = 0.35
ARRIVE_M = 0.25
PATH_CHECK_M = 2.5
CROSS_TRACK_REPLAN_M = 0.45
REPLAN_MIN_S = 0.5
ROVE_REPLAN_S = 6.0
MAX_PLAN_FAILS = 3
PROGRESS_S = 6.0
PROGRESS_M = 0.15
MAX_RECOVERS = 3
CUSP_STOP_DECISIONS = 3
COSTMAP_MAX_AGE_S = 1.0
WAIT_S = 2.0
BLACKLIST_S = 60.0
SWITCH_BONUS_M = 1.0
# A goal the truck cannot stand on or get to is driven to the reachable point
# nearest it (_aim). While it drives there that point is looked for again
# every CLOSEST_REFRESH_S, because the map grows as it goes, and a new one is
# only taken if it is CLOSEST_SWITCH_M nearer the goal. At the point, one more
# search: a point CLOSEST_GAIN_M nearer the goal than the truck already is is
# driven to; anything less and it holds there, "closest".
CLOSEST_REFRESH_S = 2.0
CLOSEST_SWITCH_M = 0.1
CLOSEST_GAIN_M = 0.3
# How clear of anything a goto's point has to be (_aim). The forward floor
# holds a forward FORWARD_FLOOR_MM (363 mm) off the LENS, which is LENS_M
# (235 mm) ahead of the pose, so head on to a wall the pose stops about
# 0.60 m off it (truck.FLOOR_REACH_M), and a point nearer the wall than
# ARRIVE_M inside that could never be arrived at that way; 7 cm more for the
# brake and the pose's noise. 0.42 m. It was 0.45, which took the 363 as off
# the bumper (FRONT_M, 28 mm further on).
AIM_CLEAR_M = tk.FLOOR_REACH_M - ARRIVE_M + 0.07
# A forward the camera's floor held off for driver.STANDOFF_S is something
# the lidar map does not have (a shoe under the lidar's plane): a disc of
# BLOCK_R_M is put there in the cost map for BLOCK_S and the path planned
# round it. Never a back-out (Params.backout).
BLOCK_R_M = 0.12
BLOCK_S = 20.0
# The wall-clock budget for one search on the car's planning thread. The
# truck stands still while it has no path, so a long budget costs time and
# nothing else; the hard searches (a three-point turn out of a hallway) are
# about 1 s on the laptop, so several on the Pi.
PLAN_BUDGET_S = 4.0
# A search that ran out of budget is tried again this much greedier
# (planner H_WEIGHT times this), which finds a longer path much sooner.
GREEDY_FACTOR = 2.0
# The least learned gain the plan radius follows (follower.GainEstimate): at
# 0.5 a plan is twice as wide as on a good floor, and no wider.
GAIN_PLAN_MIN = 0.5
FORECAST_S = 1.0
FORECAST_MIN_M = 0.25
FORECAST_CLEAR_M = 0.02
# The speed estimate off successive poses (when the caller has none).
SPEED_ALPHA = 0.5
# Patrol, once rove has run out of frontiers: a point this far off by road,
# in the least-visited VISIT_CELL_M square.
PATROL_MIN_M = 2.0
PATROL_MAX_M = 8.0
VISIT_CELL_M = 0.5


class Params(NamedTuple):
    margin: float = cmod.MARGIN_M
    planner: pl.Params = pl.Params()
    adaptive_gain: bool = True
    lookahead_min: float = fl.LOOKAHEAD_MIN_M
    lookahead_s: float = fl.LOOKAHEAD_S
    confidence_min: float = CONFIDENCE_MIN
    arrive: float = ARRIVE_M
    progress_s: float = PROGRESS_S
    max_recovers: int = MAX_RECOVERS
    path_check: float = PATH_CHECK_M
    cross_track_replan: float = CROSS_TRACK_REPLAN_M
    plan_budget_s: Optional[float] = PLAN_BUDGET_S
    forecast_s: float = FORECAST_S
    delay_s: float = fl.DELAY_S
    # No progress asks for the driver's back-out (Output.recover) when True.
    # False, the default: it plans again from where it is instead, because
    # the escape is a path on the map and never a hand-written manoeuvre.
    backout: bool = False


class Output(NamedTuple):
    direction: int          # +1, -1, 0
    speed: float            # 0..1 of the cap that way
    curvature: float        # 1/m, left positive
    recover: bool = False   # run the driver's back-out now
    status: str = ST_IDLE
    reason: str = ""
    local: bool = False     # deciding on the LocalMap
    target: Optional[tuple] = None

    @property
    def moving(self):
        return self.direction != 0 and not self.recover


STOP = Output(0, 0.0, 0.0)


class PlanResult(NamedTuple):
    """What one search came back with. `target` is what the path is to
    (None: nothing to go to); `fail` is why there is no path, or "".
    For a goto, `aim` is the point the path ends at and `closest` whether
    that is the reachable point nearest a goal the truck cannot get to;
    `start` is the pose the search was from."""
    why: str
    target: Optional[tuple]
    path: Optional[pl.Path]
    fail: str = ""
    note: str = ""
    explored: bool = False
    aim: Optional[tuple] = None
    closest: bool = False
    start: Optional[tuple] = None


class InlinePlanner:
    """Runs a planning job on the caller's thread the moment it is
    submitted, and hands the answer back `delay_scale` times the time it
    took later, on the caller's clock. 0 (the default) is at once; the
    simulator sets it to how much slower the Pi is, so a simulated truck
    waits for its plans the way the real one will."""

    def __init__(self, delay_scale=0.0):
        self.delay_scale = float(delay_scale)
        self._pending = None

    @property
    def busy(self):
        return self._pending is not None

    def submit(self, now, job):
        t0 = time.perf_counter()
        try:
            res = job()
        except Exception as e:          # a bug in a search is a failed plan
            res = PlanResult("error", None, None, fail=f"planner error {e!r}")
        ms = (time.perf_counter() - t0) * 1000.0
        self._pending = (now + ms / 1000.0 * self.delay_scale, res, ms)

    def poll(self, now):
        """(PlanResult, ms) once it is ready, else None."""
        if self._pending is None or now < self._pending[0] - 1e-9:
            return None
        _, res, ms = self._pending
        self._pending = None
        return res, ms

    def close(self):
        self._pending = None


class ThreadPlanner:
    """One worker thread, one job at a time: the car's planner. submit()
    never blocks; poll() hands back a finished job's answer once."""

    def __init__(self):
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._job = None
        self._result = None
        self._closed = False
        self._thread = None

    @property
    def busy(self):
        with self._lock:
            return self._job is not None or self._result is not None

    def submit(self, now, job):
        with self._lock:
            if self._job is not None or self._result is not None or self._closed:
                return
            self._job = job
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="nav-planner",
                                                daemon=True)
                self._thread.start()
        self._wake.set()

    def _run(self):
        while not self._closed:
            self._wake.wait(1.0)
            self._wake.clear()
            with self._lock:
                job = self._job
            if job is None:
                continue
            t0 = time.perf_counter()
            try:
                res = job()
            except Exception as e:
                res = PlanResult("error", None, None, fail=f"planner error {e!r}")
            ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self._result = (res, ms)
                self._job = None

    def poll(self, now):
        with self._lock:
            r, self._result = self._result, None
        return r

    def close(self):
        self._closed = True
        self._wake.set()


class Navigator:
    """One goal at a time. Not thread-safe: tick it from the control loop and
    hand it goals there too (click.ClickGoals.take() is the thread-safe end).
    """

    def __init__(self, params=Params(), seed=0, planner=None):
        self.p = params
        self.planner = planner if planner is not None else InlinePlanner()
        self.mode = IDLE
        self.status = ST_IDLE
        self.goal = None            # (x, y, yaw or None) in the map frame
        self.target = None          # what the path is to: the goal, a frontier
        self.path = None
        self.follower = fl.Follower(adaptive=params.adaptive_gain,
                                    lookahead_min=params.lookahead_min,
                                    lookahead_s=params.lookahead_s,
                                    delay_s=params.delay_s)
        self.cm = None
        self._cm_key = None
        self._cm_at = -math.inf
        self._cm_pose = None
        self.blacklist = []         # (x, y, when)
        self.plan_fails = 0
        self._gen = 0               # bumped by anything that makes a plan stale
        self._job_gen = None
        self._wait_until = -math.inf
        self._waited = False
        self._speed = 0.0
        self._last_pose = None
        self.forecast_stops = 0
        self._greedy = False
        self.recovers = 0
        self._planned_at = -math.inf
        self._progress_at = None
        self._progress_best = math.inf
        self._cusp_wait = 0
        self._last_dir = 0
        self._visits = {}
        self._rng = np.random.default_rng(seed)
        self.plans = 0
        self.last_plan_ms = 0.0
        self.local = False
        self.events = []            # (time, what) for the simulator's log
        self.aim = None             # the point a goto's path ends at
        self.closest = False        # the aim is the nearest to a goal it cannot reach
        self._confirm = False       # at a closest aim: one more search first
        self.blocks = []            # (x, y, until): the floor's holds, see BLOCK_S
        self.reason = ""            # the last Output's reason

    # ---- the goal -------------------------------------------------------

    def set_goal(self, x, y, yaw=None):
        self.mode = GOTO
        self.goal = (float(x), float(y), None if yaw is None else float(yaw))
        self._new_target(self.goal)
        self.status = ST_PLANNING

    @property
    def holding(self):
        """A goto that has stopped for good: arrived, at the closest point,
        or given up. It stays stopped until set_goal() or stop()."""
        return self.mode == IDLE and self.status in HELD

    def hand(self, now):
        """My hand is on the pad. Whatever path there was is not where the
        truck will be when I let go, so it is dropped, and the progress clock
        stops; the next tick without the hand plans from where I left it. A
        held goto stays held."""
        if self.mode == IDLE:
            return
        self._drop_path()
        self._progress_at = None
        self._wait_until = -math.inf
        self._cusp_wait = 0
        self._last_dir = 0
        self.status = ST_HAND
        self.reason = "takeover"

    def blocked_ahead(self, now, pose, ahead_m):
        """The floor has held a forward: something `ahead_m` ahead of the
        pose that the map may not have. Planned round for BLOCK_S."""
        x, y, yaw = (float(v) for v in pose[:3])
        d = float(ahead_m) + BLOCK_R_M
        self.blocks = [b for b in self.blocks if b[2] > now]
        self.blocks.append((x + d * math.cos(yaw), y + d * math.sin(yaw),
                            now + BLOCK_S))
        self.events.append((now, "floor block"))
        self._cm_key = None             # the cost map is rebuilt with it
        self._drop_path()
        self._planned_at = -math.inf

    def _drop_path(self):
        """No path, and any search in flight thrown away when it lands."""
        self._gen += 1
        self.path = None
        self.follower.set_path(None)
        self._confirm = False

    def rove(self):
        self.mode = ROVE
        self.goal = None
        self.blacklist = []
        self._new_target(None)
        self.status = ST_PLANNING

    def stop(self):
        self.mode = IDLE
        self.goal = None
        self._new_target(None)
        self.status = ST_IDLE

    def _new_target(self, target):
        self._gen += 1
        self._waited = False
        self._wait_until = -math.inf
        self.target = target
        self.path = None
        self.follower.set_path(None)
        self.plan_fails = 0
        self.recovers = 0
        self._progress_at = None
        self._progress_best = math.inf
        self._cusp_wait = 0
        self.aim = None
        self.closest = False
        self._confirm = False

    def recovered(self, now):
        """The driver's back-out has handed the wheels back. Plan afresh
        from wherever it left the truck."""
        self._gen += 1
        self.path = None
        self.follower.set_path(None)
        self._planned_at = -math.inf
        self._progress_at = now
        self._last_dir = 0

    # ---- the decision ---------------------------------------------------

    def tick(self, now, view, local_map=None, speed_m_s=None):
        """One decision. `speed_m_s` is the truck's ground speed if the
        caller knows it; None estimates it off the poses."""
        if self.mode == IDLE:
            if self.status not in HELD:
                self.status = ST_IDLE
            return self._stop("holding" if self.status in HELD else "no goal")
        pose = None
        use = view
        self.local = False
        if view is not None and view.confidence < self.p.confidence_min:
            use = None
            pose = view.pose
        if use is None and local_map is not None:
            pose = view.pose if view is not None else pose
            if pose is not None:
                use = local_map.view(pose, 0.0 if view is None else view.confidence)
                self.local = True
        if use is None:
            # No pose to trust: stop, and the path goes with it. It was
            # planned from a pose that is now in doubt, and the truck is
            # never driven on it again; once there is a pose, a fresh plan.
            self.status = ST_NO_MAP
            self._drop_path()
            self._progress_at = None
            self._last_pose = None
            return self._stop("no map")
        pose = use.pose
        self._visit(pose)
        self._estimate_speed(now, pose)
        if speed_m_s is None:
            speed_m_s = self._speed

        if self.mode == GOTO and math.hypot(self.goal[0] - pose[0],
                                            self.goal[1] - pose[1]) <= self.p.arrive:
            self.status = ST_ARRIVED
            self.events.append((now, "arrived"))
            self.mode = IDLE
            self._drop_path()
            return self._stop("arrived")
        if (self.mode == GOTO and self.closest and self.aim is not None
                and not self._confirm
                and math.hypot(self.aim[0] - pose[0],
                               self.aim[1] - pose[1]) <= self.p.arrive):
            # At the closest point. The map has grown on the way; one more
            # search says whether there is anywhere nearer now.
            self._drop_path()
            self._confirm = True
            self._planned_at = -math.inf

        self._costmap(now, use)
        cm = self.cm

        done = self.planner.poll(now)
        if done is not None:
            out = self._apply(now, cm, *done)
            if out is not None:
                return out
            if self.holding:
                return self._stop(self.reason)

        if now < self._wait_until:
            self.status = ST_WAIT
            return self._stop("wait")

        if self._cusp_wait > 0:
            self._cusp_wait -= 1
            self.status = ST_CUSP
            return self._stop("cusp")

        need = self._needs_plan(now, cm, pose)
        if need == "blocked":
            self.events.append((now, "path blocked"))
            self.path = None
            self.follower.set_path(None)
        if need and not self.planner.busy and now - self._planned_at >= REPLAN_MIN_S:
            self._submit(now, cm, pose, need)
            done = self.planner.poll(now)       # an inline planner is done already
            if done is not None:
                out = self._apply(now, cm, *done)
                if out is not None:
                    return out
                if self.holding:
                    return self._stop(self.reason)
        if self.path is None:
            if self.status not in (ST_EXPLORED,):
                self.status = ST_PLANNING
            return self._stop("no path")

        # Progress, on the path's remaining length.
        remaining = self.follower.remaining_m
        if self._progress_at is None:
            self._progress_at, self._progress_best = now, remaining
        elif remaining < self._progress_best - PROGRESS_M:
            self._progress_best = remaining
            self._progress_at = now
            self.recovers = 0
            self._waited = False
        elif now - self._progress_at >= self.p.progress_s:
            return self._stuck(now)

        cmd = self.follower.step(pose, speed_m_s)
        if cmd.done:
            self.path = None
            self.follower.set_path(None)
            if self.mode == ROVE:
                self._new_target(None)
                self.status = ST_PLANNING
                return self._stop("frontier reached")
            # goto: the path ends inside goal_tol but not inside arrive; plan
            # the last bit again. At a closest point that search is the
            # confirming one.
            self._planned_at = -math.inf
            if self.closest and self.aim is not None and math.hypot(
                    self.aim[0] - pose[0], self.aim[1] - pose[1]) <= 2 * self.p.arrive:
                # The end of a path to the closest point, near it: as far as
                # the follower goes (a head-on stop can end it short).
                self._confirm = True
            return self._stop("path end")
        if cmd.cusp:
            self._cusp_wait = CUSP_STOP_DECISIONS - 1
            self.status = ST_CUSP
            return self._stop("cusp")
        if cmd.cross_track > self.p.cross_track_replan:
            self.path = None
            return self._stop("off the path")
        if self._last_dir and cmd.direction != self._last_dir:
            # A change of direction the path did not mark: stand first.
            self._last_dir = 0
            return self._stop("direction change")
        if self._forecast_hits(cm, pose, cmd, speed_m_s):
            self.forecast_stops += 1
            self.events.append((now, "forecast hit"))
            self.path = None
            self.follower.set_path(None)
            return self._stop("command would hit")
        self._last_dir = cmd.direction
        self.status = ST_DRIVING
        self.reason = "follow"
        return Output(cmd.direction, cmd.speed, cmd.curvature, False,
                      ST_DRIVING, "follow", self.local, self.target)

    # ---- pieces ---------------------------------------------------------

    def _stop(self, reason):
        self._last_dir = 0
        self.reason = reason
        return Output(0, 0.0, 0.0, False, self.status, reason, self.local,
                      self.target)

    def _stuck(self, now):
        self.path = None
        self.follower.set_path(None)
        self._gen += 1
        if not self._waited:
            # A wait first: whatever is in the way may move.
            self._waited = True
            self.events.append((now, "no progress: wait"))
            self._wait_until = now + WAIT_S
            self._progress_at = now + WAIT_S
            self._planned_at = -math.inf
            self.status = ST_WAIT
            return self._stop("no progress: wait")
        self.recovers += 1
        self.events.append((now, "no progress"))
        self._progress_at = now
        if self.recovers > self.p.max_recovers:
            return self._give_up(now, "no progress")
        if not self.p.backout:
            # A fresh plan from where it is, greedier, which finds a
            # different way sooner: the map is the way out.
            self._planned_at = -math.inf
            self.status = ST_PLANNING
            return self._stop("no progress: plan again")
        self.status = ST_RECOVER
        self.reason = "no progress"
        return Output(0, 0.0, 0.0, True, ST_RECOVER, "no progress",
                      self.local, self.target)

    def _give_up(self, now, why):
        self.events.append((now, f"gave up: {why}"))
        self._drop_path()
        if self.mode == ROVE:
            if self.target is not None:
                self.blacklist.append((self.target[0], self.target[1], now))
            self._new_target(None)
            self.status = ST_PLANNING
            return self._stop(f"frontier dropped: {why}")
        self.mode = IDLE
        self.status = ST_FAILED
        return self._stop(f"gave up: {why}")

    def _costmap(self, now, view):
        self.blocks = [b for b in self.blocks if b[2] > now]
        key = (view.version, view.local, len(self.blocks))
        moved = (self._cm_pose is None
                 or math.hypot(view.pose[0] - self._cm_pose[0],
                               view.pose[1] - self._cm_pose[1]) > 1.0)
        if (self.cm is not None and key == self._cm_key and not moved
                and now - self._cm_at < COSTMAP_MAX_AGE_S):
            return
        crop = None
        if not view.local:
            pts = [self.target[:2]] if self.target is not None else []
            if self.mode == GOTO and self.goal is not None:
                pts.append(self.goal[:2])
            crop = _union(cmod.window(view, pts), _known_box(view))
        if self.blocks:
            view = view._replace(layers=tuple(view.layers) + (
                _discs(view, [b[:2] for b in self.blocks], BLOCK_R_M),))
        self.cm = cmod.build(view, margin=self.p.margin, crop=crop)
        self._cm_key, self._cm_at, self._cm_pose = key, now, view.pose

    def _needs_plan(self, now, cm, pose):
        if self._confirm and self.path is None:
            return "confirm"
        if self.target is None or self.path is None:
            return "new"
        clear = pl.path_clearance(cm, self.path, self.follower.index,
                                  self.p.path_check)
        if clear < cm.margin * 0.5:
            return "blocked"
        if self.mode == ROVE and now - self._planned_at >= ROVE_REPLAN_S:
            return "refresh"
        if (self.mode == GOTO and self.closest
                and now - self._planned_at >= CLOSEST_REFRESH_S):
            return "refresh"
        return ""

    def _submit(self, now, cm, pose, why):
        """Hand a search to the planner. Everything it needs is copied in:
        the job never reads the navigator while tick() runs."""
        self._planned_at = now
        self._job_gen = self._gen
        mode, target, goal = self.mode, self.target, self.goal
        blacklist = [(x, y) for x, y, t in self.blacklist if now - t < BLACKLIST_S]
        last_dir = self._last_dir
        visits = dict(self._visits)
        budget = self.p.plan_budget_s
        local = self.local
        params = self.p.planner
        # A confirm asks whether anywhere is nearer than where the truck is
        # now, which is the closest point it was driving to, give or take
        # the arrival tolerance and the pose's noise.
        here = self.aim if why == "confirm" else None
        keep = self.aim if self.closest else None
        if self.p.adaptive_gain:
            # A surface that gives back less of the steering gets a wider plan:
            # the plan radius is the measured one over the learned gain.
            g = min(max(self.follower.gain.gain, GAIN_PLAN_MIN), 1.0)
            params = params._replace(curvature=params.curvature * g)
        if self._greedy:
            params = params._replace(h_weight=params.h_weight * GREEDY_FACTOR)

        def job():
            deadline = None if budget is None else time.perf_counter() + budget
            return self._search(cm, pose, why, mode, target, goal, blacklist,
                                last_dir, visits, deadline, local, params, here,
                                keep)
        self.planner.submit(now, job)

    def _search(self, cm, pose, why, mode, target, goal, blacklist, last_dir,
                visits, deadline, local, params, here=None, keep=None):
        """The planning job: a target (rove), the goal snapped (goto), the
        plan. Reads only its arguments and the navigator's constants."""
        note, explored = "", False
        if mode == ROVE:
            found = fr.find(cm, pose, blacklist)
            current = None
            if target is not None:
                current = next((f for f in found if math.hypot(
                    f.x - target[0], f.y - target[1]) < fr.BLACKLIST_M), None)
            if target is None or (why == "refresh" and (
                    current is None or (found and found[0] is not current
                                        and current.score - found[0].score > SWITCH_BONUS_M))):
                if target is not None and current is None and not found:
                    pass        # a patrol point, and still nothing new: keep it
                elif found:
                    f = found[0]
                    target = (f.x, f.y)
                    note = f"frontier {f.size_m:.1f} m at {f.distance_m:.1f}"
                else:
                    explored = True
                    target = self._patrol_point(cm, pose, visits)
            if target is None:
                return PlanResult(why, None, None, fail="nothing to explore",
                                  explored=True)
            aim = target
        closest = False
        start = tuple(float(v) for v in pose[:3])
        if mode != ROVE:
            found = self._aim(cm, pose, goal, local, keep)
            if found is None:
                return PlanResult(why, target, None, fail="goal blocked",
                                  start=start)
            aim, closest = found
            now_off = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
            if here is not None:
                now_off = min(now_off, math.hypot(goal[0] - here[0],
                                                  goal[1] - here[1]))
            if closest and why == "confirm" and (
                    now_off - math.hypot(goal[0] - aim[0], goal[1] - aim[1])
                    < CLOSEST_GAIN_M):
                # Nowhere reachable is worth the drive: this is the closest.
                return PlanResult(why, target, None, aim=aim, closest=True,
                                  start=start)
        path = pl.plan(cm, pose, aim[:2], params=params,
                       start_dir=last_dir, deadline=deadline)
        if path is None:
            over = deadline is not None and time.perf_counter() > deadline
            return PlanResult(why, target, None,
                              fail="over budget" if over else "no path",
                              note=note, explored=explored,
                              aim=aim if mode != ROVE else None,
                              closest=closest, start=start)
        return PlanResult(why, target, path, note=note, explored=explored,
                          aim=aim if mode != ROVE else None, closest=closest,
                          start=start)

    def _apply(self, now, cm, res, ms):
        """A search has come back: use it, unless it is stale."""
        self.last_plan_ms = ms
        self.plans += 1
        if self._job_gen != self._gen:
            return None
        if self.mode == ROVE:
            if res.target is not None and res.target != self.target:
                if self.target is not None:
                    self.events.append((now, "frontier switch"))
                self._new_target(res.target)
                self._job_gen = self._gen
            if res.note:
                self.events.append((now, res.note))
            if res.explored:
                self.status = ST_EXPLORED
        if self.mode == GOTO and res.why == "confirm":
            self._confirm = False
            if res.path is None and (res.closest or res.fail):
                # At the closest point, and nowhere nearer to go to (or no
                # way to get there from here): hold.
                return self._hold_closest(now, res)
        if (self.mode == GOTO and res.why == "refresh" and self.path is not None
                and res.closest and self.aim is not None
                and math.hypot(self.goal[0] - res.aim[0], self.goal[1] - res.aim[1])
                > math.hypot(self.goal[0] - self.aim[0], self.goal[1] - self.aim[1])
                - CLOSEST_SWITCH_M):
            return None         # no nearer than the point it is going to
        if res.fail:
            if res.fail == "nothing to explore":
                self.path = None
                self.status = ST_EXPLORED
                return self._stop(res.fail)
            self._greedy = res.fail == "over budget"
            return self._fail_plan(now, res.fail)
        self._greedy = False
        path = res.path
        if pl.path_clearance(cm, path, 0, self.p.path_check) < cm.margin * 0.5:
            # The map has moved on since the search started.
            self._planned_at = -math.inf
            return None
        self.plan_fails = 0
        self.path = path
        self.follower.set_path(path)
        if self.mode == GOTO:
            if res.closest and not self.closest:
                self.events.append((now, "goal unreachable: closest "
                                         f"({res.aim[0]:.2f}, {res.aim[1]:.2f})"))
            self.aim, self.closest = res.aim, bool(res.closest)
        self.events.append((now, f"plan {res.why} {path.length:.1f} m "
                                 f"{path.reverses} rev {ms:.0f} ms"))
        return None

    def _hold_closest(self, now, res):
        self.mode = IDLE
        self.status = ST_CLOSEST
        # The point it drove to and is at, not the search's: that is no
        # nearer, and can be as near the other side of the wall.
        self.aim = self.aim if self.aim is not None else res.aim
        self.events.append((now, "closest"))
        self._drop_path()
        return self._stop("closest reachable")

    def _fail_plan(self, now, why):
        self.plan_fails += 1
        self.path = None
        self.follower.set_path(None)
        if self.plan_fails >= MAX_PLAN_FAILS:
            return self._give_up(now, why)
        self._wait_until = now + WAIT_S
        self.status = ST_PLANNING
        return self._stop(why)

    def _estimate_speed(self, now, pose):
        last = self._last_pose
        self._last_pose = (now, pose[0], pose[1])
        if last is None or now <= last[0]:
            return
        v = math.hypot(pose[0] - last[1], pose[1] - last[2]) / (now - last[0])
        if v > 3.0:             # a pose jump (a SLAM correction), not speed
            return
        self._speed += SPEED_ALPHA * (v - self._speed)

    def _forecast_hits(self, cm, pose, cmd, speed):
        """Does the command, held FORECAST_S, run the footprint into
        anything? On the curvature it will drive (the learned gain)."""
        if cmd.direction == 0:
            return False
        gain = self.follower.gain.gain if self.p.adaptive_gain else 1.0
        k = cmd.curvature * gain
        dist = max(FORECAST_MIN_M, float(speed) * self.p.forecast_s)
        n = max(2, int(math.ceil(dist / 0.05)))
        s = cmd.direction * dist * np.arange(1, n + 1) / n
        x, y, th = (float(v) for v in pose[:3])
        ax, ay, ath = pl._arc(x, y, th, np.full(n, k), s)
        ahead = float(np.min(cm.footprint_clearance(ax, ay, ath)))
        here = float(cm.footprint_clearance(x, y, th))
        return ahead < min(FORECAST_CLEAR_M, here - 0.01)

    def _aim(self, cm, pose, goal, local=False, keep=None):
        """(point, closest) for a goto: the goal itself when the truck can
        stand on it and get to it from here; otherwise the reachable point
        nearest the goal (closest True), and of those within a cell of the
        nearest, the one the shortest drive away. None when nothing at all
        is reachable.

        Reachable is over known floor (planner.grid_distances, known_only):
        a goal inside a wall, in the unknown or in a room with no way into it
        is aimed at the nearest known floor the truck can drive to. On the
        local map, which is only the last few seconds of scans, unknown
        counts as passable, as it always has there.

        A goal the footprint fits on is the goal however near a wall. A
        point instead of the goal is one AIM_CLEAR_M clear of anything where
        there is one (the footprint's clearance where there is not): facing
        a wall the forward floor stops the truck with its pose about 0.60 m
        off it, and a point nearer the wall than that could not be arrived
        at head on. `keep`, the point a closest goto is
        already going to, stays the point while it can still be stood on and
        reached and is within CLOSEST_SWITCH_M of the nearest."""
        gx, gy = float(goal[0]), float(goal[1])
        need = cm.inscribed + 0.02
        clear = max(need, AIM_CLEAR_M)
        reach = pl.grid_distances(cm, [pose[:2]], known_only=not local,
                                  clearance=need)
        # The coarse distances onto the cost map's own cells.
        k = max(1, int(round(reach.res / cm.res)))
        nx, ny = cm.shape
        cost = np.repeat(np.repeat(reach.dist, k, 0), k, 1)[:nx, :ny]
        stand = np.isfinite(cost) & (cm.dist >= need)
        if not local:
            stand &= cm.known
        if not stand.any():
            return None
        ix, iy = (int(v) for v in cm.world_to_cell(gx, gy))
        if cm.inside(ix, iy) and stand[ix, iy]:
            # A goal it can stand on and get to is the goal, however near a
            # wall: ARRIVE_M round it is what it has to get inside.
            return (gx, gy), False
        ok = stand & (cm.dist >= clear)
        if not ok.any():
            ok = stand
        ii, jj = np.nonzero(ok)
        wx = cm.origin[0] + (ii + 0.5) * cm.res
        wy = cm.origin[1] + (jj + 0.5) * cm.res
        d = np.hypot(wx - gx, wy - gy)
        if keep is not None:
            # The point it is already going to, while it can still stand
            # there and get there (not held to AIM_CLEAR_M again, which a
            # cell of new scan can tip): a fresh plan from a pose further on
            # is no reason to swap it for one as near the other side of a
            # wall.
            kx, ky = (int(v) for v in cm.world_to_cell(keep[0], keep[1]))
            if (cm.inside(kx, ky) and stand[kx, ky]
                    and math.hypot(keep[0] - gx, keep[1] - gy)
                    <= float(d.min()) + CLOSEST_SWITCH_M):
                return (float(keep[0]), float(keep[1])), True
        # Nearest the goal; between points as near as each other to half a
        # cell (either side of a pillar), the one that is the shorter drive.
        near = d <= float(d.min()) + 0.5 * cm.res
        m = int(np.argmin(np.where(near, cost[ii, jj], np.inf)))
        return (float(wx[m]), float(wy[m])), True

    def _visit(self, pose):
        k = (int(math.floor(pose[0] / VISIT_CELL_M)),
             int(math.floor(pose[1] / VISIT_CELL_M)))
        self._visits[k] = self._visits.get(k, 0) + 1

    def _patrol_point(self, cm, pose, visits):
        reach = pl.grid_distances(cm, [pose[:2]], known_only=True)
        d = reach.dist
        ok = np.isfinite(d) & (d >= PATROL_MIN_M) & (d <= PATROL_MAX_M)
        if not ok.any():
            return None
        ii, jj = np.nonzero(ok)
        wx = reach.origin[0] + (ii + 0.5) * reach.res
        wy = reach.origin[1] + (jj + 0.5) * reach.res
        visits = np.array([visits.get((int(math.floor(x / VISIT_CELL_M)),
                                       int(math.floor(y / VISIT_CELL_M))), 0)
                           for x, y in zip(wx, wy)])
        low = np.nonzero(visits == visits.min())[0]
        k = int(self._rng.choice(low))
        return float(wx[k]), float(wy[k])


def _discs(view, points, r_m):
    """bool [view.shape]: discs of r_m round map points, for MapView.layers."""
    out = np.zeros(view.shape, bool)
    n = int(math.ceil(r_m / view.res))
    for x, y in points:
        ix, iy = (int(v) for v in view.world_to_cell(x, y))
        i0, i1 = max(ix - n, 0), min(ix + n + 1, out.shape[0])
        j0, j1 = max(iy - n, 0), min(iy + n + 1, out.shape[1])
        if i1 <= i0 or j1 <= j0:
            continue
        gi, gj = np.mgrid[i0:i1, j0:j1]
        out[i0:i1, j0:j1] |= (gi - ix) ** 2 + (gj - iy) ** 2 <= n * n
    return out


def _known_box(view, pad_m=1.0):
    k = view.known
    rows = np.flatnonzero(k.any(1))
    if not rows.size:
        return None
    cols = np.flatnonzero(k.any(0))
    p = int(math.ceil(pad_m / view.res))
    nx, ny = k.shape
    return (max(int(rows[0]) - p, 0), min(int(rows[-1]) + 1 + p, nx),
            max(int(cols[0]) - p, 0), min(int(cols[-1]) + 1 + p, ny))


def _union(a, b):
    if b is None:
        return a
    return (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))
