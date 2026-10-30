"""Where on the way to its goal the goal model is pointed: a point along a
path round the walls, in place of the goal itself.

The model drives. This only chooses the point its goal input
(policy_goal.encode: present, distance over FAR_M, the bearing's sin and
cos) is the encoding of. Fed the goal itself, the model heads down the
straight-line bearing, into whatever is between; fed a point LOOKAHEAD_M
along a path that goes round, it heads round. The encoding is the one a
click is fed through, so every goal model drives on it unchanged. Nothing
here steers or touches the throttle, and the arrival hold and every skip
in a patrol still read the real goal, never the point fed.

The goal source
---------------
GoalSource is the seam: anything with `goal`, the (x, y, frame) on the map
the model should get to or None, and `encode(crop)`, its own goal input,
used whenever no path is (policy_goal.encode of `goal`, for every source so
far). The single click (goal.ClickGoal), the patrol's current waypoint
(patrol.Patrol) and a person to follow are all one: PathPlanner reads
`goal` off the one AutoPilot holds, whatever it is, and never anything else.

The plan
--------
On the SLAM runner's /map picture (runner.MapView, 10 cm pixels, made twice
a second anyway, so planning costs no map of its own), cut to the box round
the truck and the goal padded by PAD_M, unknown past the edge of what has
been seen:

  inflated  nav.costmap.build: a cell is out of bounds within the truck's
            half-width plus WIDE_MARGIN_M of an occupied one (0.266 m), and
            costs more the closer it is, out to SOFT_M, paid at SOFT_WEIGHT,
            so a path runs down the middle of a gap and goes wide round a
            corner instead of shaving it (the truck rubbed walls and caught
            a wheel on corners on the old 0.196 m and 0.6 m band). With no
            way through at that, the plan is made again at the truck's own
            fit, NARROW_MARGIN_M (0.196 m), soft band and all, so a tight
            doorway still routes: on the 10 cm map an opening of 0.55 m
            and up is planned wide, down to about 0.35-0.4 m narrow.
            /path/status says which (clear_m).
  unknown   costed, never banned (costmap.UNKNOWN_COST), so a goal past the edge of the map gets a path
            into the unknown towards it, and the truck maps it on the way.
  search    Dijkstra out from the goal over the 8-connected free cells
            (nav.planner.grid_distances, scipy's), then down the distances
            from the truck. The goal and the truck are each snapped to the
            nearest cell the search can stand in, within SNAP_M.

On a thread, never in the control loop: every PLAN_EVERY_S at most (2 Hz),
when the map, the truck (REPLAN_MOVE_M) or the goal has changed since the
last plan; a new goal is planned at once.

The point fed
-------------
Each decision, off the newest plan and the crop the model is fed:

  the path point nearest the truck, then the farthest point at most
  LOOKAHEAD_M on along the path that the truck can see in a straight line
  (no wall within the plan's own clearance of the line, wide or narrow).
  LOOKAHEAD_M 1.5 m is inside
  what the goal head was trained on: hindsight goals 3 to 15 s ahead, on
  the ticks the truck was moving in, are 0.9 m at the 25th percentile and
  1.9 m at the median (maps-pose, all drives to 2026-10-29).

and the goal itself, exact, when the goal is within LOOKAHEAD_M in a clear
line, or that far along the path is the goal. With no plan to use, the
goal itself too, and /path/status says why: no map yet, no way through,
the goal or the truck boxed in, a plan gone STALE_S old, the truck more
than OFF_PATH_M off it, or a goal on another map frame.
"""

import math
import threading
import time
from typing import NamedTuple, Optional, Protocol

import numpy as np

from crawler.autopilot import policy_goal as pg
from crawler.nav import costmap as cmap

LOOKAHEAD_M = 1.5
PLAN_EVERY_S = 0.5
REPLAN_MOVE_M = 0.2
PAD_M = 3.0
SNAP_M = 0.6
# Off the walls. The plan is first looked for with the truck's middle kept
# WIDE_MARGIN_M past its half-width off every wall (hard), and paying more
# the nearer than SOFT_M it goes (the soft band, costmap.build's, at
# SOFT_WEIGHT), so it runs down the middle of a gap and swings wide of a
# corner; with nothing through that way, again at costmap.MARGIN_M (the
# narrow, the truck's own fit), so a tight doorway still routes.
WIDE_MARGIN_M = 0.13
SOFT_M = 1.0
SOFT_WEIGHT = 4.0
NARROW_MARGIN_M = cmap.MARGIN_M
# The line to the point keeps as clear of walls as the path itself must:
# the truck drives the chord, not the path. The plan's own clearance (its
# cost map's inscribed), this when there is none to hand.
LOS_CLEAR_M = cmap.INSCRIBED_M
LOS_STEP_M = 0.05
# The first of the line this close to the truck is where it already is.
LOS_SKIP_M = 0.2
STALE_S = 3.0
OFF_PATH_M = 1.0
# A goal moved more than this is a new goal: planned at once.
GOAL_MOVED_M = 0.05
# Points on /path: every Nth of the 10 cm path, the ends always.
MARKER_EVERY = 2
MAX_PATH_CELLS = 4000

# The map's codes (mapping/grid.py), here so this does not import the mapper.
UNKNOWN, FREE, OCCUPIED = 0, 1, 2


class GoalSource(Protocol):
    """What PathPlanner reads off the goal the autopilot holds."""

    @property
    def goal(self) -> Optional[tuple]:
        """(x, y, frame) on the map, or None for no goal."""

    def encode(self, crop) -> np.ndarray:
        """Its own goal input against this crop (policy_goal.encode)."""


class Plan(NamedTuple):
    goal: tuple             # (x, y, frame) planned to
    start: tuple            # (x, y) planned from
    made: float             # clock seconds
    map_t: float            # the MapView's own time
    points: Optional[np.ndarray]   # (n, 2) map metres, truck to goal; None with no path
    cum: Optional[np.ndarray]      # (n,) metres along it
    cm: object              # the nav.costmap.CostMap it was found on
    why: str                # "" with a path, else why there is none
    ms: float               # wall time to make it
    cpu_ms: float           # this thread's CPU time to make it


def _window(view, start, goal, pad=PAD_M):
    """The view's codes cut to the box round start and goal, padded, as a
    nav.mapview.MapView: unknown outside what the view holds."""
    from crawler.nav import mapview as mv
    res = float(view.res)
    x0, y0 = float(view.x0), float(view.y0)
    lo_x = min(start[0], goal[0]) - pad
    lo_y = min(start[1], goal[1]) - pad
    hi_x = max(start[0], goal[0]) + pad
    hi_y = max(start[1], goal[1]) + pad
    # On the view's own lattice.
    i0 = int(math.floor((lo_x - x0) / res))
    j0 = int(math.floor((lo_y - y0) / res))
    i1 = int(math.ceil((hi_x - x0) / res))
    j1 = int(math.ceil((hi_y - y0) / res))
    codes = np.asarray(view.codes)
    nx, ny = codes.shape
    out = np.full((i1 - i0, j1 - j0), UNKNOWN, np.uint8)
    a0, a1 = max(i0, 0), min(i1, nx)
    b0, b1 = max(j0, 0), min(j1, ny)
    if a1 > a0 and b1 > b0:
        out[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = codes[a0:a1, b0:b1]
    return mv.MapView(out == OCCUPIED, out != UNKNOWN, res,
                      (x0 + i0 * res, y0 + j0 * res), (start[0], start[1], 0.0))


def find_path(view, start, goal, cost_weight=SOFT_WEIGHT, snap_m=SNAP_M,
              margins=(WIDE_MARGIN_M, NARROW_MARGIN_M), soft_m=SOFT_M):
    """(points (n, 2) or None, why, CostMap) from map point `start` to map
    point `goal` on a runner.MapView. The points run from start itself to
    goal itself through the centres of the cells between. Tried at each of
    `margins` in turn (the hard clearance off the walls past the truck's
    half-width), the first with a path kept; the CostMap's `inscribed` says
    which. With none, the last one's why."""
    win = _window(view, start, goal)
    out = None
    for margin in margins:
        out = _find_at(win, start, goal, margin, soft_m, cost_weight, snap_m)
        if out[0] is not None:
            return out
    return out


def _find_at(win, start, goal, margin, soft_m, cost_weight, snap_m):
    from crawler.nav import planner as pl
    cm = cmap.build(win, margin=margin, inflation=soft_m)
    g = pl.grid_distances(cm, [goal[:2]], coarsen=1, cost_weight=cost_weight,
                          snap_m=snap_m)
    if not np.isfinite(g.dist).any():
        return None, "goal boxed in", cm
    s = pl._snap(g.passable, g, start[0], start[1], snap_m)
    if s is None:
        return None, "truck boxed in", cm
    d = g.dist
    if not np.isfinite(d[s]):
        return None, "no way through", cm
    nx, ny = d.shape
    i, j = s
    cells = [(i, j)]
    steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
             (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)))
    while d[i, j] > 0.0 and len(cells) < MAX_PATH_CELLS:
        best, nxt = d[i, j], None
        for di, dj, length in steps:
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and d[a, b] < best:
                best, nxt = d[a, b], (a, b)
        if nxt is None:
            break
        i, j = nxt
        cells.append(nxt)
    if d[i, j] > 0.0:
        return None, "no way through", cm
    c = np.asarray(cells, float)
    pts = np.column_stack([g.origin[0] + (c[:, 0] + 0.5) * g.res,
                           g.origin[1] + (c[:, 1] + 0.5) * g.res])
    pts = np.vstack([[start[0], start[1]], pts, [goal[0], goal[1]]])
    return pts, "", cm


def line_clear(cm, a, b, clear_m=None, step=LOS_STEP_M, skip_m=LOS_SKIP_M):
    """Whether the straight line from map point a to b keeps clear_m from
    every wall, bar its first skip_m. clear_m None: the cost map's own
    inscribed, the clearance its plan was found at."""
    if clear_m is None:
        clear_m = float(getattr(cm, "inscribed", LOS_CLEAR_M))
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length <= skip_m:
        return True
    s = np.arange(skip_m, length + 1e-9, step)
    s = np.append(s, length)
    t = s / length
    x = a[0] + t * (b[0] - a[0])
    y = a[1] + t * (b[1] - a[1])
    return bool((cm.dist_at(x, y) >= clear_m).all())


def subgoal(plan, pose, lookahead_m=LOOKAHEAD_M):
    """(x, y, kind) of the point to feed off `plan` from truck map point
    `pose`, kind "subgoal" or "goal" (the goal itself), or (None, None, why)."""
    pts, cum = plan.points, plan.cum
    p = np.asarray(pose[:2], float)
    d = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
    k0 = int(np.argmin(d))
    if d[k0] > OFF_PATH_M:
        return None, None, "off the path"
    reach = cum[k0] + lookahead_m
    last = int(np.searchsorted(cum, reach, side="right")) - 1
    last = max(min(last, len(pts) - 1), k0)
    for k in range(last, k0, -1):
        if line_clear(plan.cm, p, pts[k]):
            if k == len(pts) - 1:
                return float(pts[k][0]), float(pts[k][1]), "goal"
            return float(pts[k][0]), float(pts[k][1]), "subgoal"
    k = min(k0 + 1, len(pts) - 1)
    return float(pts[k][0]), float(pts[k][1]), ("goal" if k == len(pts) - 1 else "subgoal")


class _Request(NamedTuple):
    goal: tuple             # (x, y, frame)
    pose: tuple             # (x, y, yaw) on that frame
    frame_since: float      # the first crop time on this frame
    seq: int                # bumped when the goal moves


class PathPlanner:
    """Plans to the goal the autopilot holds and picks the point the goal
    model is fed. update() and encode() are the control loop's and only
    ever take a lock and do arithmetic; the plan is made on a thread of its
    own (background=False makes it in the caller when due, for tests)."""

    def __init__(self, slam, lookahead_m=LOOKAHEAD_M, every_s=PLAN_EVERY_S,
                 background=True, clock=time.monotonic, say=None):
        self.slam = slam
        self.lookahead_m = float(lookahead_m)
        self.every_s = float(every_s)
        self.background = bool(background)
        self.clock = clock
        self._say = say
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._req = None
        self._plan = None
        self._frame = None
        self._frame_since = 0.0
        self._seq = 0
        self._last_goal = None
        self._done = None          # (seq, pose, map_t, made) of the last plan made
        # The last decision's choice, for /path/status and /path.
        self.fed = "none"          # "subgoal" | "goal" | "none"
        self.why = "no goal"
        self.aim = None            # (x, y) fed, map frame
        self.plans = 0
        self.failed = 0
        self.ms_total = 0.0
        self.cpu_total = 0.0
        self.ms_max = 0.0
        self._said = ""

    # ---- the control loop ------------------------------------------------------

    def update(self, crop, source):
        """Every control pass: what the plan is wanted for now. Cheap."""
        goal = getattr(source, "goal", None) if source is not None else None
        if crop is None or not getattr(crop, "present", False) or goal is None:
            with self._lock:
                self._req = None
            return
        frame = int(getattr(crop, "frame", 0))
        if frame != self._frame:
            self._frame, self._frame_since = frame, float(crop.t)
        g = (float(goal[0]), float(goal[1]), int(goal[2]))
        if (self._last_goal is None or g[2] != self._last_goal[2]
                or math.hypot(g[0] - self._last_goal[0], g[1] - self._last_goal[1])
                > GOAL_MOVED_M):
            self._seq += 1
            self._last_goal = g
            new = True
        else:
            new = False
        with self._lock:
            self._req = _Request(g, (float(crop.x), float(crop.y), float(crop.yaw)),
                                 self._frame_since, self._seq)
        if self.background:
            self._ensure_thread()
            if new:
                self._wake.set()
        else:
            self._step()

    def encode(self, crop, source):
        """The goal input this decision: the encoding of the point along the
        path, or the source's own (the goal itself) when there is none."""
        self.update(crop, source)
        goal = getattr(source, "goal", None) if source is not None else None
        if goal is None or crop is None or not getattr(crop, "present", False):
            self._choose("none", "no goal", None)
            return source.encode(crop) if source is not None else pg.missing()
        frame = int(getattr(crop, "frame", 0))
        if int(goal[2]) != frame:
            self._choose("none", "goal on another map", None)
            return source.encode(crop)
        pose = (float(crop.x), float(crop.y))
        # In a straight clear line and close: the goal itself, no plan needed.
        with self._lock:
            plan = self._plan
        if plan is not None and plan.goal[2] != frame:
            plan = None
        near = math.hypot(goal[0] - pose[0], goal[1] - pose[1]) <= self.lookahead_m
        if near and plan is not None and plan.cm is not None and line_clear(plan.cm, pose, goal):
            self._choose("goal", "in sight", (goal[0], goal[1]))
            return source.encode(crop)
        why = self._usable(plan, goal)
        if why:
            self._choose("goal", why, (goal[0], goal[1]))
            return source.encode(crop)
        x, y, kind = subgoal(plan, pose, self.lookahead_m)
        if x is None:
            self._choose("goal", kind, (goal[0], goal[1]))
            return source.encode(crop)
        if kind == "goal":
            self._choose("goal", "along the path", (goal[0], goal[1]))
            return source.encode(crop)
        self._choose("subgoal", "", (x, y))
        return pg.encode(crop, (x, y, frame))

    def _usable(self, plan, goal):
        if plan is None:
            return "no plan yet"
        if math.hypot(plan.goal[0] - goal[0], plan.goal[1] - goal[1]) > GOAL_MOVED_M:
            return "planning to the new goal"
        if plan.points is None:
            return plan.why
        if float(self.clock()) - plan.made > STALE_S:
            return "plan stale"
        return ""

    def _choose(self, fed, why, aim):
        self.fed, self.why, self.aim = fed, why, aim
        said = f"{fed}:{why}"
        if fed == "goal" and why not in ("in sight", "along the path") and said != self._said:
            self._log("info", f"No path to feed: {why}; the model is fed the goal itself")
        elif fed == "subgoal" and self._said and not self._said.startswith("subgoal"):
            self._log("info", "Feeding the model a point along the path")
        self._said = said

    def _log(self, level, text):
        print(text)
        if self._say is not None:
            try:
                self._say(level, "Path", text)
            except Exception:
                pass

    # ---- the planning thread -----------------------------------------------------

    def _ensure_thread(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="path-planner",
                                            daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()

    def _loop(self):
        while not self._stop.is_set():
            self._wake.wait(timeout=self.every_s / 2.0)
            self._wake.clear()
            try:
                self._step()
            except Exception as e:
                self.failed += 1
                print(f"[path] {e!r}")

    def _due(self, req, view):
        done = self._done
        if done is None or done[0] != req.seq:
            return True
        _, pose, map_t, made = done
        if float(self.clock()) - made < self.every_s - 1e-6:
            return False
        moved = math.hypot(req.pose[0] - pose[0], req.pose[1] - pose[1]) > REPLAN_MOVE_M
        return moved or float(view.t) != map_t or float(self.clock()) - made > STALE_S / 2

    def _step(self):
        """One look: plan if the goal is new, or the map or the truck have
        moved and the last plan is PLAN_EVERY_S old."""
        with self._lock:
            req = self._req
        if req is None:
            return
        try:
            view = self.slam.map() if self.slam is not None else None
        except Exception:
            view = None
        now = float(self.clock())
        if view is None or float(view.t) < req.frame_since:
            why = "no map yet" if view is None else "no map on this frame yet"
            with self._lock:
                plan = self._plan
            if plan is None or plan.goal != req.goal or plan.why != why:
                # Said, not counted as a plan: the first map is planned on at once.
                with self._lock:
                    self._plan = Plan(req.goal, req.pose[:2], now, -1.0, None, None,
                                      None, why, 0.0, 0.0)
                self._done = None
            return
        if not self._due(req, view):
            return
        t0, c0 = time.perf_counter(), time.thread_time()
        pts, why, cm = find_path(view, req.pose[:2], req.goal[:2])
        ms = (time.perf_counter() - t0) * 1000.0
        cpu = (time.thread_time() - c0) * 1000.0
        cum = None
        if pts is not None:
            cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        self._set(Plan(req.goal, req.pose[:2], now, float(view.t), pts, cum, cm, why,
                       ms, cpu), req, float(view.t))

    def _set(self, plan, req, map_t):
        self.plans += 1
        if plan.points is None:
            self.failed += 1
        self.ms_total += plan.ms
        self.cpu_total += plan.cpu_ms
        self.ms_max = max(self.ms_max, plan.ms)
        self._done = (req.seq, req.pose, map_t, plan.made)
        with self._lock:
            self._plan = plan

    # ---- out ----------------------------------------------------------------------

    def status(self):
        """/path/status."""
        with self._lock:
            plan = self._plan
        have = plan is not None and plan.points is not None
        aim = self.aim if self.aim is not None else (0.0, 0.0)
        n = max(self.plans, 1)
        return {"fed": self.fed, "why": self.why,
                "aim_x": float(aim[0]), "aim_y": float(aim[1]),
                "path": bool(have),
                "plan_why": "" if plan is None else plan.why,
                "length_m": float(plan.cum[-1]) if have else -1.0,
                "age_s": float(self.clock()) - plan.made if plan is not None else -1.0,
                "plan_ms": float(plan.ms) if plan is not None else 0.0,
                "plan_cpu_ms": float(plan.cpu_ms) if plan is not None else 0.0,
                "plan_ms_mean": self.ms_total / n, "plan_ms_max": float(self.ms_max),
                "plan_cpu_ms_mean": self.cpu_total / n,
                "plans": int(self.plans), "failed": int(self.failed),
                "lookahead_m": self.lookahead_m,
                "clear_m": float(plan.cm.inscribed) if have else -1.0}

    def marker(self):
        """/path's payload: the path from the truck on, and the point fed.
        {} clears both."""
        with self._lock:
            plan = self._plan
        out = {}
        if plan is not None and plan.points is not None and plan.goal[2] == self._frame:
            pts = plan.points
            keep = list(range(0, len(pts), MARKER_EVERY))
            if keep[-1] != len(pts) - 1:
                keep.append(len(pts) - 1)
            out["path"] = [(float(pts[k][0]), float(pts[k][1])) for k in keep]
        if self.aim is not None and self.fed != "none":
            out["aim"] = (float(self.aim[0]), float(self.aim[1]))
            out["fed"] = self.fed
        return out
