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
     Planning is rate-limited to once every REPLAN_MIN_S.
  5. Give up. A goal with no plan MAX_PLAN_FAILS times running is given up
     (rove blacklists the frontier and takes the next one). A goal the truck
     makes no progress toward for PROGRESS_S while driving asks for the
     back-out (`recover`), which is the driver's own -- the same one STUCK
     runs -- and after MAX_RECOVERS of those with no progress between them
     it is given up.
  6. Follow: the follower's command for this pose. A cusp, where the path
     changes direction, is CUSP_STOP_DECISIONS of standing still first.

It never touches a servo. tick() returns an Output, and the driver decides
what reaches the wheels (driver.py): the forward floor, the rear stop and
STUCK are underneath every one of these.
"""

import math
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


class Navigator:
    """One goal at a time. Not thread-safe: tick it from the control loop and
    hand it goals there too (click.ClickGoals.take() is the thread-safe end).
    """

    def __init__(self, params=Params(), seed=0):
        self.p = params
        self.mode = IDLE
        self.status = ST_IDLE
        self.goal = None            # (x, y, yaw or None) in the map frame
        self.target = None          # what the path is to: the goal, a frontier
        self.path = None
        self.follower = fl.Follower(adaptive=params.adaptive_gain,
                                    lookahead_min=params.lookahead_min,
                                    lookahead_s=params.lookahead_s)
        self.cm = None
        self._cm_key = None
        self._cm_at = -math.inf
        self._cm_pose = None
        self.blacklist = []
        self.plan_fails = 0
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
        self.path = None
        self.follower.set_path(None)
        self._planned_at = -math.inf
        self._progress_at = now
        self._last_dir = 0

    # ---- the decision ---------------------------------------------------

    def tick(self, now, view, local_map=None, speed_m_s=0.0):
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

        if self.mode == GOTO and math.hypot(self.goal[0] - pose[0],
                                            self.goal[1] - pose[1]) <= self.p.arrive:
            self.status = ST_ARRIVED
            self.events.append((now, "arrived"))
            self.mode = IDLE
            return self._stop("arrived")

        self._costmap(now, use)
        cm = self.cm

        if self._cusp_wait > 0:
            self._cusp_wait -= 1
            self.status = ST_CUSP
            return self._stop("cusp")

        need = self._needs_plan(now, cm, pose)
        if need and now - self._planned_at >= REPLAN_MIN_S:
            out = self._replan(now, cm, pose, need)
            if out is not None:
                return out
        if self.path is None:
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
        self.recovers += 1
        self.events.append((now, "no progress"))
        self.path = None
        self.follower.set_path(None)
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
                self.blacklist.append(self.target[:2])
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

    def _replan(self, now, cm, pose, why):
        self._planned_at = now
        if self.mode == ROVE and (self.target is None or why == "refresh"):
            if why == "refresh" and self.target is not None:
                # Is the frontier still a frontier? The map moved on.
                if not self._still_frontier(cm, self.target):
                    self._new_target(None)
            if self.target is None:
                target = self._pick_frontier(cm, pose)
                if target is None:
                    return self._stop("nothing to explore")
                self.target = target
        target = self.target
        if self.mode == GOTO:
            target = self._reachable_goal(cm, pose)
            if target is None:
                return self._fail_plan(now, "goal blocked")
        t0 = time.perf_counter()
        path = pl.plan(cm, pose, target[:2], params=self.p.planner,
                       start_dir=self._last_dir)
        self.last_plan_ms = (time.perf_counter() - t0) * 1000.0
        self.plans += 1
        if path is None:
            return self._fail_plan(now, "no path")
        self.plan_fails = 0
        self.path = path
        self.follower.set_path(path)
        self.events.append((now, f"plan {why} {path.length:.1f} m "
                                 f"{path.reverses} rev {self.last_plan_ms:.0f} ms"))
        return None

    def _fail_plan(self, now, why):
        self.plan_fails += 1
        self.path = None
        self.follower.set_path(None)
        if self.plan_fails >= MAX_PLAN_FAILS:
            return self._give_up(now, why)
        self.status = ST_PLANNING
        return self._stop(why)

    def _reachable_goal(self, cm, pose):
        """The goal, or the nearest point to it the truck can stand on. On
        the local map, a goal past the window is the reachable point in it
        nearest the goal."""
        gx, gy = self.goal[0], self.goal[1]
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
            if not self.local:
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

    def _pick_frontier(self, cm, pose):
        found = fr.find(cm, pose, self.blacklist)
        if found:
            f = found[0]
            self.events.append((0.0, f"frontier {f.size_m:.1f} m at {f.distance_m:.1f}"))
            self.status = ST_PLANNING
            return (f.x, f.y)
        self.status = ST_EXPLORED
        return self._patrol_point(cm, pose)

    def _still_frontier(self, cm, target):
        mask = fr.frontier_mask(cm)
        ix, iy = cm.world_to_cell(target[0], target[1])
        r = int(math.ceil(0.3 / cm.res))
        x0, x1 = max(ix - r, 0), min(ix + r + 1, cm.shape[0])
        y0, y1 = max(iy - r, 0), min(iy + r + 1, cm.shape[1])
        return bool(mask[x0:x1, y0:y1].any())

    def _visit(self, pose):
        k = (int(math.floor(pose[0] / VISIT_CELL_M)),
             int(math.floor(pose[1] / VISIT_CELL_M)))
        self._visits[k] = self._visits.get(k, 0) + 1

    def _patrol_point(self, cm, pose):
        reach = pl.grid_distances(cm, [pose[:2]], known_only=True)
        d = reach.dist
        ok = np.isfinite(d) & (d >= PATROL_MIN_M) & (d <= PATROL_MAX_M)
        if not ok.any():
            return None
        ii, jj = np.nonzero(ok)
        wx = reach.origin[0] + (ii + 0.5) * reach.res
        wy = reach.origin[1] + (jj + 0.5) * reach.res
        visits = np.array([self._visits.get((int(math.floor(x / VISIT_CELL_M)),
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
