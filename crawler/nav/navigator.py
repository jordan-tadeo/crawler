"""The goal handling: where to go, the path there, and when to give up.

Two ways to have a goal:

  goto   a spot, clicked in Foxglove (click.py) or handed over by anything
         else. Arrives within ARRIVE_M and stops.
  rove   no spot: the best frontier (frontier.py), then the next, for as long
         as auto runs. When the map has no reachable frontier left it patrols
         the place it has been least.

Every decision (tick), in order:

  1. Which map. The global one while the SLAM is sure of itself (confidence
     at or over CONFIDENCE_MIN); below that, the truck-centred LocalMap,
     which only holds the last few seconds of scans and so cannot have been
     smeared by a pose that has drifted. A goal past the local window is
     steered for: the reachable point in the window nearest it. Nothing else
     changes between the two, so indoors and out are one mode.
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
     plan), and after that asks for the back-out (`recover`), which is the
     driver's own -- the same one STUCK runs -- and after MAX_RECOVERS of
     those with no progress between them it is given up.
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
GOAL_SNAP_M = 0.6
COSTMAP_MAX_AGE_S = 1.0
WAIT_S = 2.0
BLACKLIST_S = 60.0
SWITCH_BONUS_M = 1.0
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
    (None: nothing to go to); `fail` is why there is no path, or ""."""
    why: str
    target: Optional[tuple]
    path: Optional[pl.Path]
    fail: str = ""
    note: str = ""
    explored: bool = False


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

    # ---- the goal -------------------------------------------------------

    def set_goal(self, x, y, yaw=None):
        self.mode = GOTO
        self.goal = (float(x), float(y), None if yaw is None else float(yaw))
        self._new_target(self.goal)
        self.status = ST_PLANNING

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
            self.status = ST_IDLE
            return STOP
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
            self.status = ST_NO_MAP
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
            return self._stop("arrived")

        self._costmap(now, use)
        cm = self.cm

        done = self.planner.poll(now)
        if done is not None:
            out = self._apply(now, cm, *done)
            if out is not None:
                return out

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
            # the last bit again.
            self._planned_at = -math.inf
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
        return Output(cmd.direction, cmd.speed, cmd.curvature, False,
                      ST_DRIVING, "follow", self.local, self.target)

    # ---- pieces ---------------------------------------------------------

    def _stop(self, reason):
        self._last_dir = 0
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
        self.status = ST_RECOVER
        return Output(0, 0.0, 0.0, True, ST_RECOVER, "no progress",
                      self.local, self.target)

    def _give_up(self, now, why):
        self.events.append((now, f"gave up: {why}"))
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
        key = (view.version, view.local)
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
        self.cm = cmod.build(view, margin=self.p.margin, crop=crop)
        self._cm_key, self._cm_at, self._cm_pose = key, now, view.pose

    def _needs_plan(self, now, cm, pose):
        if self.target is None or self.path is None:
            return "new"
        clear = pl.path_clearance(cm, self.path, self.follower.index,
                                  self.p.path_check)
        if clear < cm.margin * 0.5:
            return "blocked"
        if self.mode == ROVE and now - self._planned_at >= ROVE_REPLAN_S:
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
                                last_dir, visits, deadline, local, params)
        self.planner.submit(now, job)

    def _search(self, cm, pose, why, mode, target, goal, blacklist, last_dir,
                visits, deadline, local, params):
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
        else:
            aim = self._reachable_goal(cm, pose, goal, local)
            if aim is None:
                return PlanResult(why, target, None, fail="goal blocked")
        path = pl.plan(cm, pose, aim[:2], params=params,
                       start_dir=last_dir, deadline=deadline)
        if path is None:
            over = deadline is not None and time.perf_counter() > deadline
            return PlanResult(why, target, None,
                              fail="over budget" if over else "no path",
                              note=note, explored=explored)
        return PlanResult(why, target, path, note=note, explored=explored)

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
        self.events.append((now, f"plan {res.why} {path.length:.1f} m "
                                 f"{path.reverses} rev {ms:.0f} ms"))
        return None

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

    def _reachable_goal(self, cm, pose, goal, local=False):
        """The goal, or the nearest point to it the truck can stand on. On
        the local map, a goal past the window is the reachable point in it
        nearest the goal."""
        gx, gy = goal[0], goal[1]
        ix, iy = cm.world_to_cell(gx, gy)
        inside = bool(cm.inside(ix, iy))
        need = cm.inscribed + 0.02
        if inside and cm.dist[ix, iy] >= need:
            return (gx, gy)
        if inside:
            r = int(math.ceil(GOAL_SNAP_M / cm.res))
            x0, x1 = max(ix - r, 0), min(ix + r + 1, cm.shape[0])
            y0, y1 = max(iy - r, 0), min(iy + r + 1, cm.shape[1])
            win = cm.dist[x0:x1, y0:y1] >= need
            ii, jj = np.nonzero(win)
            if len(ii):
                d2 = (ii + x0 - ix) ** 2 + (jj + y0 - iy) ** 2
                k = int(np.argmin(d2))
                if d2[k] <= r * r:
                    return tuple(float(v) for v in cm.cell_to_world(ii[k] + x0, jj[k] + y0))
            if not local:
                return None
        # Outside the window (or blocked inside a local one): head for it.
        reach = pl.grid_distances(cm, [pose[:2]], known_only=False)
        ok = np.isfinite(reach.dist)
        if not ok.any():
            return None
        ii, jj = np.nonzero(ok)
        wx = reach.origin[0] + (ii + 0.5) * reach.res
        wy = reach.origin[1] + (jj + 0.5) * reach.res
        k = int(np.argmin(np.hypot(wx - gx, wy - gy)))
        return float(wx[k]), float(wy[k])

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
