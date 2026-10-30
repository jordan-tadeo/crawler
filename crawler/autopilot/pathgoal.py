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

Two more, both optional: `ahead`, the goals that come after `goal` in
order (the patrol's next waypoints; nothing for a click, a person or a
search target, which have no next one known), and `passed(goal, crop)`,
told when the plan says the truck has gone by `goal` (True if the source
moved on). A source without them is planned to as before.

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

Through the waypoints after it. The plan runs on from the goal through
`ahead`, a leg at a time (the same search, the leg's own window), until
it is LOOKAHEAD_M past the goal or MAX_AHEAD goals on, so the point fed
carries on round a corner made of waypoints before the truck gets to the
first one, instead of sitting on the waypoint until it is passed and then
swinging up to 180 degrees to the next. A leg with no way through ends the
plan there. A plan made through the next goal is still used once the
source has moved on to it, from its next leg, until the new one is in.

Passed. The source passes a goal as it always has (the patrol's PASS_M),
and the plan passes it too: once the truck is within PASS_NEAR_M of it and
nearer the path after it than the path up to it by PASS_SIDE_M, the truck
has cut the corner and gone by, and source.passed() is told. A goal the path
comes back from the way it went (the end of a there-and-back, a dead end)
is never passed this way: both are the same distance off.

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

The nearest point is looked for only on the leg to the goal the source
is on (and BACK_M before it), so a path that doubles back is not jumped
along. The goal itself, exact, when there is nothing after it and the goal
is within LOOKAHEAD_M in a clear line, or that far along the path is the
goal. With no plan to use, the
goal itself too, and /path/status says why: no map yet, no way through,
the goal or the truck boxed in, a plan gone STALE_S old, the truck more
than OFF_PATH_M off it, or a goal on another map frame.

The lookahead is the one the model's path goals were drawn at (the car
sets it off the sidecar, policy_goal.lookahead_of), LOOKAHEAD_M without.

Staged
------
Whatever is picked, its bearing goes in staged (policy_goal.stage): one
further round than STAGE_DEG from the nose is fed as STAGE_DEG on its side,
as the goal models steer the wrong way for a goal well round to the left
(policy_goal's measures), so a hairpin is fed as a hard turn until the point
comes inside it. One BEHIND_DEG or more round goes in as it is: the goal is
behind, and turning round for it, backing up included, is the model's. And
the bearing fed turns on the map no faster than AIM_RATE_DPS from the last
decision's, so a replan that moves the point round a corner (+15 to +120 in
one tick at the 2026-10-30 hairpin) comes in over a second, not at once;
the truck's own turning moves it freely. Coming round from a goal behind it
turns through the staged band at that rate. One passed as it is goes on
passing to BEHIND_HYST_DEG inside BEHIND_DEG, so a bearing hovering on 150
is not fed 60 and 150 by turns. The rate limit is for a point moving round
a corner, not for a new direction: a bearing wanted more than AIM_SNAP_DEG
round from the last one fed (a plan flipped from behind to ahead), and the
first one for a new goal (the patrol moving on to its next waypoint), are
fed at once. Held at the rate, each of those was fed pointing the old way
for up to a second and a half, sometimes round the far side of the truck
from the path. Only the bearing changes:
present and distance are the point's own, and aim (/path/status, so
tools/fed_goals.py) is moved to the bearing fed, the point the model got.
"""

import math
import threading
import time
from typing import NamedTuple, Optional, Protocol

import numpy as np

from crawler.autopilot import policy_goal as pg
from crawler.nav import costmap as cmap

LOOKAHEAD_M = pg.PATH_M
# Staged and rate-limited (the docstring's "Staged"): policy_goal.stage's
# band, and the most the bearing fed may turn on the map in a second. The
# point picked turned at 127 deg/s or less on 95% of the 2026-10-30 p1261
# decisions (replayed); the other 5% are the jumps, up to 179 in a tick.
STAGE_DEG = pg.STAGE_DEG
BEHIND_DEG = pg.BEHIND_DEG
AIM_RATE_DPS = 120.0
# Decisions further apart than this are not rate-limited: the first one
# after is fed where it points.
AIM_GAP_S = 0.5
# The bearing wanted further than this round from the last one fed is a
# new direction, not a replan's step round a corner: it is fed at once, as
# it is after a new goal (the patrol moving on). Held at the rate, a plan
# that flipped from behind to dead ahead was fed pointing behind for over
# a second (replayed 2026-10-30 patrols). Coming round from a goal behind
# to one ahead on its own side is less than this and still turns at the rate.
AIM_SNAP_DEG = 150.0
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
# Through the goals after this one: no more than MAX_AHEAD of them, and
# no more once the plan is lookahead_m past the goal.
MAX_AHEAD = 3
# Passed on the path: within PASS_NEAR_M of the goal and nearer the path
# after it than the path up to it by PASS_SIDE_M (more than a 10 cm cell).
PASS_NEAR_M = 1.0
PASS_SIDE_M = 0.15
# On a leg after the first, the nearest point is looked for from this far
# back along the path before the goal it starts at, so the switch to it is
# made from where the truck is, not jumped ahead.
BACK_M = 0.6
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

    # Optional:
    #   ahead -> [(x, y, frame)], the goals after `goal`, in order
    #   passed(goal, crop) -> bool, the plan says the truck has gone by `goal`


def _ahead(source, frame):
    """The source's goals after its goal on `frame`, up to MAX_AHEAD, as
    (x, y, frame); () for a source with none."""
    try:
        got = getattr(source, "ahead", None)
    except Exception:
        return ()
    out = []
    for a in got or ():
        if a is None or int(a[2]) != frame or len(out) >= MAX_AHEAD:
            break
        out.append((float(a[0]), float(a[1]), int(a[2])))
    return tuple(out)


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _same(a, b):
    return (a[2] == b[2] and math.hypot(a[0] - b[0], a[1] - b[1]) <= GOAL_MOVED_M)


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
    route: tuple = ()       # (goal, then the goals after it planned through)
    vias: tuple = ()        # each route goal's index in points


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


def _leg(plan, j):
    """(lo, hi) indices into plan.points of the leg to route goal j: from
    the truck for the first, from BACK_M before goal j - 1 after that."""
    pts, cum = plan.points, plan.cum
    vias = plan.vias or (len(pts) - 1,)
    hi = int(vias[j])
    if j == 0:
        return 0, hi
    lo = int(np.searchsorted(cum, cum[int(vias[j - 1])] - BACK_M, side="left"))
    return min(lo, hi), hi


def subgoal(plan, pose, lookahead_m=LOOKAHEAD_M, j=0):
    """(x, y, kind) of the point to feed off `plan` from truck map point
    `pose`, on the way to route goal j (0, the plan's goal), kind "subgoal"
    or "goal" (the goal itself, with nothing after it), or
    (None, None, why)."""
    pts, cum = plan.points, plan.cum
    p = np.asarray(pose[:2], float)
    lo, hi = _leg(plan, j)
    d = np.hypot(pts[lo:hi + 1, 0] - p[0], pts[lo:hi + 1, 1] - p[1])
    k0 = lo + int(np.argmin(d))
    if d[k0 - lo] > OFF_PATH_M:
        return None, None, "off the path"
    end = len(pts) - 1
    last_goal = len(plan.route or (None,)) - 1 == j

    def kind(k):
        return "goal" if (k == end and last_goal) else "subgoal"

    reach = cum[k0] + lookahead_m
    last = int(np.searchsorted(cum, reach, side="right")) - 1
    last = max(min(last, end), k0)
    # Past the goal, a point no nearer the truck than the goal (or half the
    # lookahead, if that is less): a path that comes back the way it went
    # is not fed as a point on top of the truck.
    floor = min(math.hypot(pts[hi][0] - p[0], pts[hi][1] - p[1]), 0.5 * lookahead_m)
    for k in range(last, k0, -1):
        if k > hi and math.hypot(pts[k][0] - p[0], pts[k][1] - p[1]) < floor:
            continue
        if line_clear(plan.cm, p, pts[k]):
            return float(pts[k][0]), float(pts[k][1]), kind(k)
    k = min(k0 + 1, end)
    return float(pts[k][0]), float(pts[k][1]), kind(k)


def gone_by(plan, pose, j=0, near_m=PASS_NEAR_M, side_m=PASS_SIDE_M):
    """Whether the truck at map point `pose` has gone by route goal j on
    `plan`: within near_m of it, and nearer the path after it (up to the
    next goal) than the path up to it by side_m. False for the last goal
    on the plan, and for a path that comes back from it the way it went."""
    route, vias = plan.route or (), plan.vias or ()
    if plan.points is None or j >= len(route) - 1:
        return False
    p = np.asarray(pose[:2], float)
    g = route[j]
    if math.hypot(g[0] - p[0], g[1] - p[1]) > near_m:
        return False
    pts = plan.points
    lo, hi = _leg(plan, j)
    before = np.hypot(pts[lo:hi + 1, 0] - p[0], pts[lo:hi + 1, 1] - p[1]).min()
    nxt = int(vias[j + 1])
    if nxt <= hi:
        return False
    after = np.hypot(pts[hi + 1:nxt + 1, 0] - p[0], pts[hi + 1:nxt + 1, 1] - p[1]).min()
    return bool(after < before - side_m)


def _cum(pts):
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])


def plan_route(view, start, route, lookahead_m=LOOKAHEAD_M, **kw):
    """(points, cum, CostMap, why, route, vias): a path from map point
    `start` through each (x, y, frame) of `route` in turn, a find_path leg
    each, stopped once it runs lookahead_m past route[0] or at the first
    leg with no way through after the first. route and vias are the goals
    it went through and each one's index in points. The first leg's
    CostMap (its clearance is what /path/status says, and the line of sight
    is checked on it). points None with no first leg, why its reason."""
    pts, why, cm = find_path(view, start, route[0][:2], **kw)
    if pts is None:
        return None, None, cm, why, (route[0],), ()
    legs, vias = [pts], [len(pts) - 1]
    done = [route[0]]
    past = 0.0
    for a, b in zip(route, route[1:]):
        if past >= lookahead_m:
            break
        more, _, _ = find_path(view, a[:2], b[:2], **kw)
        if more is None:
            break
        more = more[1:]
        legs.append(more)
        vias.append(vias[-1] + len(more))
        done.append(b)
        past += float(_cum(np.vstack([legs[-2][-1:], more]))[-1])
    pts = np.vstack(legs)
    return pts, _cum(pts), cm, "", tuple(done), tuple(vias)


class _Request(NamedTuple):
    goal: tuple             # (x, y, frame)
    pose: tuple             # (x, y, yaw) on that frame
    frame_since: float      # the first crop time on this frame
    seq: int                # bumped when the goal moves
    route: tuple = ()       # goal, then the source's goals after it


class PathPlanner:
    """Plans to the goal the autopilot holds and picks the point the goal
    model is fed. update() and encode() are the control loop's and only
    ever take a lock and do arithmetic; the plan is made on a thread of its
    own (background=False makes it in the caller when due, for tests)."""

    def __init__(self, slam, lookahead_m=LOOKAHEAD_M, every_s=PLAN_EVERY_S,
                 background=True, clock=time.monotonic, say=None,
                 stage_deg=STAGE_DEG, behind_deg=BEHIND_DEG, aim_rate_dps=AIM_RATE_DPS):
        self.slam = slam
        self.lookahead_m = float(lookahead_m)
        # None turns staging, or the rate limit, off.
        self.stage_deg = stage_deg
        self.behind_deg = float(behind_deg)
        self.aim_rate_dps = aim_rate_dps
        self._last_fed = None      # (map bearing fed, crop t, frame) of the last decision
        self._walking = False      # walking round from a goal behind
        self._passing = False      # the last bearing picked was passed as it is, behind
        self._shaped_goal = None   # the source's goal the last bearing fed was shaped for
        self.bearing_deg = 0.0     # the bearing fed this decision, truck frame
        self.raw_bearing_deg = 0.0  # the one picked, before staging and the rate limit
        self.staged = 0            # decisions fed other than the bearing picked
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
        self._last_route = ()
        self._done = None          # (seq, pose, map_t, made) of the last plan made
        # The last decision's choice, for /path/status and /path.
        self.fed = "none"          # "subgoal" | "goal" | "none"
        self.why = "no goal"
        self.aim = None            # (x, y) fed, map frame
        self.plans = 0
        self.failed = 0
        self.passes = 0            # goals passed on the path (gone_by)
        self._at = (None, 0)       # (plan, route goal) the last decision was on
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
        route = (g,) + _ahead(source, g[2])
        if (self._last_goal is None or len(route) != len(self._last_route)
                or not all(_same(a, b) for a, b in zip(route, self._last_route))):
            self._seq += 1
            self._last_goal = g
            self._last_route = route
            new = True
        else:
            new = False
        with self._lock:
            self._req = _Request(g, (float(crop.x), float(crop.y), float(crop.yaw)),
                                 self._frame_since, self._seq, route)
        if self.background:
            self._ensure_thread()
            if new:
                self._wake.set()
        else:
            self._step()

    def encode(self, crop, source):
        """The goal input this decision: the encoding of the point along the
        path, or the source's own (the goal itself) when there is none,
        staged and rate-limited (_shape)."""
        g = self._pick(crop, source)
        goal = getattr(source, "goal", None) if source is not None else None
        return self._shape(crop, g, goal)

    def _shape(self, crop, g, goal=None):
        """The goal input g as it is fed: its bearing staged
        (policy_goal.stage, passing on from the last decision's), then
        turned on the map no faster than aim_rate_dps from the last
        decision's. Coming round from a goal behind it turns through the
        staged band at that rate, and is staged again once inside it. A
        bearing wanted more than AIM_SNAP_DEG round from the last one fed,
        or the first for a new `goal` (the source's, the patrol's waypoint),
        is fed at once. Present, distance and the encoding are g's own; aim
        moves to the bearing fed at its own distance, so /path/status (and
        tools/fed_goals.py off it) has the point the model got."""
        g = np.asarray(g, np.float32)
        if (g.shape != (1, pg.SIZE) or not g[0, pg.PRESENT] > 0 or crop is None
                or not getattr(crop, "present", False)):
            self._last_fed, self._walking, self._passing = None, False, False
            self._shaped_goal = None
            self.bearing_deg = self.raw_bearing_deg = 0.0
            return g
        goal = None if goal is None else (float(goal[0]), float(goal[1]), int(goal[2]))
        new_goal = (goal is not None and self._shaped_goal is not None
                    and not _same(goal, self._shaped_goal))
        self._shaped_goal = goal
        raw = math.atan2(float(g[0, pg.SIN]), float(g[0, pg.COS]))
        want = float(pg.stage(raw, self.stage_deg, self.behind_deg, self._passing))
        self._passing = (self.stage_deg is not None
                         and bool(pg.passes(raw, self.behind_deg, self._passing)))
        yaw, t, frame = float(crop.yaw), float(crop.t), int(getattr(crop, "frame", 0))
        heading = yaw + want
        last = self._last_fed
        if (self.aim_rate_dps is not None and last is not None and last[2] == frame
                and 0.0 <= t - last[1] <= AIM_GAP_S and not new_goal):
            turn = _wrap(heading - last[0])
            if abs(turn) <= math.radians(AIM_SNAP_DEG):
                step = math.radians(float(self.aim_rate_dps)) * (t - last[1])
                heading = last[0] + max(-step, min(step, turn))
        fed = _wrap(heading - yaw)
        if self.stage_deg is not None:
            lim = math.radians(float(self.stage_deg))
            self._walking = abs(want) > lim or (self._walking and abs(fed) > lim)
            if not self._walking:
                fed = max(-lim, min(lim, fed))
        self._last_fed = (yaw + fed, t, frame)
        self.raw_bearing_deg = math.degrees(raw)
        self.bearing_deg = math.degrees(fed)
        if abs(_wrap(fed - raw)) < 1e-6:
            return g
        self.staged += 1
        if self.aim is not None:
            d = math.hypot(self.aim[0] - float(crop.x), self.aim[1] - float(crop.y))
            self.aim = (float(crop.x) + d * math.cos(yaw + fed),
                        float(crop.y) + d * math.sin(yaw + fed))
        out = g.copy()
        out[0, pg.SIN] = math.sin(fed)
        out[0, pg.COS] = math.cos(fed)
        return out

    def _pick(self, crop, source):
        """The goal input picked off the plan, before _shape."""
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
        with self._lock:
            plan = self._plan
        if plan is not None and plan.goal[2] != frame:
            plan = None
        j = self._on(plan, goal)
        # Gone by the goal on the path: the source moves on, and the same
        # plan is used from its next leg until the new one is in.
        if (j is not None and not self._usable(plan, goal)
                and callable(getattr(source, "passed", None)) and gone_by(plan, pose, j)):
            if source.passed(goal, crop):
                self.passes += 1
                goal = getattr(source, "goal", None)
                if goal is None:
                    self._choose("none", "no goal", None)
                    return source.encode(crop)
                j = self._on(plan, goal)
        # In a straight clear line and close, with nothing after it: the goal
        # itself, no plan needed. With goals after it, the path on round.
        alone = not _ahead(source, frame)
        near = math.hypot(goal[0] - pose[0], goal[1] - pose[1]) <= self.lookahead_m
        if (alone and near and plan is not None and plan.cm is not None
                and line_clear(plan.cm, pose, goal)):
            self._choose("goal", "in sight", (goal[0], goal[1]))
            return source.encode(crop)
        why = self._usable(plan, goal)
        if why:
            self._choose("goal", why, (goal[0], goal[1]))
            return source.encode(crop)
        self._at = (plan, j)
        x, y, kind = subgoal(plan, pose, self.lookahead_m, j)
        if x is None:
            self._choose("goal", kind, (goal[0], goal[1]))
            return source.encode(crop)
        if kind == "goal":
            self._choose("goal", "along the path", (goal[0], goal[1]))
            return source.encode(crop)
        self._choose("subgoal", "", (x, y))
        return pg.encode(crop, (x, y, frame))

    def _on(self, plan, goal):
        """Which of plan's route goals `goal` is (0 its own), or None. On
        from the one the last decision was on with the same plan, as a loop
        of two waypoints plans through the same goal twice."""
        if plan is None or goal is None:
            return None
        g = (float(goal[0]), float(goal[1]), int(goal[2]))
        since = self._at[1] if self._at[0] is plan else 0
        route = plan.route or (plan.goal,)
        for j in range(since, len(route)):
            if _same(route[j], g):
                return j
        return None

    def _usable(self, plan, goal):
        if plan is None:
            return "no plan yet"
        if self._on(plan, goal) is None:
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
                                      None, why, 0.0, 0.0, (req.goal,), ())
                self._done = None
            return
        if not self._due(req, view):
            return
        t0, c0 = time.perf_counter(), time.thread_time()
        pts, cum, cm, why, route, vias = plan_route(view, req.pose[:2],
                                                    req.route or (req.goal,),
                                                    self.lookahead_m)
        ms = (time.perf_counter() - t0) * 1000.0
        cpu = (time.thread_time() - c0) * 1000.0
        self._set(Plan(req.goal, req.pose[:2], now, float(view.t), pts, cum, cm, why,
                       ms, cpu, route, vias), req, float(view.t))

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
                "clear_m": float(plan.cm.inscribed) if have else -1.0,
                "route_n": len(plan.route) if have else 0,
                "passes": int(self.passes),
                "bearing_deg": float(self.bearing_deg),
                "raw_bearing_deg": float(self.raw_bearing_deg),
                "staged": int(self.staged)}

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
