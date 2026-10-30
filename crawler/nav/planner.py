"""Paths the truck can drive: hybrid A* over arcs, forward and reverse.

A plain grid A* hands back a path with corners in it, and this truck cannot
turn on the spot: its tightest circle is 0.75 m on the weaker (right) side
at full lock (truck.TURN_RADIUS_M, MEASURED), and with a lazy axle on loose
ground it is wider than that. So
the search moves the way the truck moves. Every expansion drives PRIM_STEP_M
along an arc, at one of a handful of curvatures, forward or in reverse, and
keeps the continuous pose it ends at. The closed set is binned (XY_BIN_M,
YAW_BINS) so the search ends, but the poses on the path are the real ones.

What makes it a three-point turn when it needs one: reverse arcs are in the
move set, they cost REVERSE_FACTOR times a forward metre (the truck backs
up at a quarter of its forward speed, and blind but for the lidar), and a
change of direction costs SWITCH_PENALTY_M. So it reverses when the forward
way round is longer than those costs, and not otherwise.

The plan curvature is a fraction of the truck's measured lock
(PLAN_CURVATURE of truck.MAX_CURVATURE, the weaker side): 0.88, so a radius
of 0.85 m. The follower keeps the other 12% in hand to correct with, and the
learned gain (follower.GainEstimate) covers a surface that gives back less;
a path planned at the best lock is one the truck cannot follow on a bad day,
and one planned much wider than the truck can turn sends every turnaround
into a hunt for room it does not need.

Two heuristics, the larger taken. A 2D search on the same cost map,
coarsened (H_COARSEN), run from the goal outwards (grid_distances): it knows
the walls, which a straight-line distance does not, so the search does not
spend itself on the wrong side of one. And the obstacle-free distance for a
truck that cannot turn tighter than the plan radius (turn_distance): it
knows a goal beside or behind the truck is a turn away, which the 2D one
does not, and that is where a plain hybrid A* floods. Weighted (H_WEIGHT),
which trades a slightly longer path for far fewer expansions.

The analytic shot. Every SHOT_EVERY expansions within SHOT_M of the goal,
the search tries the obstacle-free answer straight off: one arc at the plan
radius, then straight to the goal, forward or in reverse, either way round
(_shot). If that is clear by the margin all the way, it is the rest of the
path, and the search ends there instead of creeping up on the goal a
PRIM_STEP_M at a time.

The floor veto. The camera holds any forward with something in a box
FORWARD_FLOOR_MM off the lens (truck.PROBE_POINTS), so a path that
noses up to a wall before turning away from it is a path the car stops on
and then backs out of. With Params.floor_probe (off by default), forward
moves keep that box PROBE_CLEAR_M clear, except within PROBE_FREE_M of the
goal.

Bounded twice: MAX_EXPANSIONS outright, and a wall-clock deadline the caller
can pass (the navigator's planning thread does), checked every 64.
"""

import heapq
import math
import time
from typing import NamedTuple, Optional

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from crawler.nav import truck as tk

PRIM_STEP_M = 0.2
CURVATURE_STEPS = (-1.0, -0.5, 0.0, 0.5, 1.0)
PLAN_CURVATURE = 0.88         # of truck.MAX_CURVATURE: R 0.85 m
REVERSE_FACTOR = 3.0
SWITCH_PENALTY_M = 1.0
STEER_CHANGE_M = 0.05         # per unit of curvature fraction changed
COST_WEIGHT = 2.0
GOAL_TOL_M = 0.2
GOAL_YAW_TOL = math.radians(30.0)
# 15 cm and 10 degrees. Halving either about doubles the search in a tight
# spot (a three-point turn in a hallway: 4600 expansions at 10 cm / 5
# degrees, 1960 at these) for paths that come out within a few percent of
# the same length.
XY_BIN_M = 0.15
YAW_BINS = 36
MAX_EXPANSIONS = 12000
H_WEIGHT = 1.4
H_COARSEN = 2
SAMPLE_M = 0.05               # spacing of the poses handed to the follower
SHOT_EVERY = 8
SHOT_M = 5.0
# Forward moves keep the floor-veto box (truck.PROBE_POINTS) this clear of
# anything, except within PROBE_FREE_M of the goal, where arriving facing a
# wall is the point. 1 cm: the box only has to be off solid cells (their
# distance is 0, the next cell's is one cell).
#
# Off by default (Params.floor_probe). In the house sim it cost about 10
# points of arrivals at both 5 cm and 1 cm, almost all "no path" in places
# the truck does fit, while the floor holds it saves are about one back-out
# an episode. Worth another look on the car, where a floor hold may cost
# more than the sim's stand-in does.
PROBE_CLEAR_M = 0.01
PROBE_FREE_M = 0.6
# A shot is only worth trying where the straight line to the goal is clear
# of the lethal band: one cheap look along it first.


class Params(NamedTuple):
    step: float = PRIM_STEP_M
    curvature: float = PLAN_CURVATURE
    reverse: bool = True
    reverse_factor: float = REVERSE_FACTOR
    switch_penalty: float = SWITCH_PENALTY_M
    steer_change: float = STEER_CHANGE_M
    cost_weight: float = COST_WEIGHT
    goal_tol: float = GOAL_TOL_M
    yaw_tol: float = GOAL_YAW_TOL
    xy_bin: float = XY_BIN_M
    yaw_bins: int = YAW_BINS
    max_expansions: int = MAX_EXPANSIONS
    h_weight: float = H_WEIGHT
    shot: bool = True
    floor_probe: bool = False   # keep the floor-veto box off walls (off: see PROBE_CLEAR_M)


class Path(NamedTuple):
    """poses (n, 3) x, y, yaw every SAMPLE_M; dirs (n,) +1 forward, -1
    reverse, the direction the truck is moving INTO each pose (dirs[0] is
    the first segment's); curv (n,) the planned curvature into each pose."""
    poses: np.ndarray
    dirs: np.ndarray
    curv: np.ndarray
    length: float
    cost: float
    expansions: int
    reverses: int

    def segments(self):
        """[(start, end)] index ranges of constant direction, end exclusive.
        Each starts at the cusp the previous one ended on."""
        d = self.dirs
        cuts = [0] + [i for i in range(1, len(d)) if d[i] != d[i - 1]] + [len(d)]
        out = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            out.append((max(a - 1, 0), b))
        return out


class Grid2D(NamedTuple):
    """Distances on the coarse grid from a set of sources, cost-weighted
    metres; inf where unreachable."""
    dist: np.ndarray
    res: float
    origin: tuple
    passable: np.ndarray

    def at(self, x, y):
        ix = np.floor((np.asarray(x) - self.origin[0]) / self.res).astype(np.int64)
        iy = np.floor((np.asarray(y) - self.origin[1]) / self.res).astype(np.int64)
        nx, ny = self.dist.shape
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        out = np.full(np.shape(ix), np.inf)
        out[ok] = self.dist[ix[ok], iy[ok]]
        return out

    def cell(self, x, y):
        return (int(math.floor((x - self.origin[0]) / self.res)),
                int(math.floor((y - self.origin[1]) / self.res)))


def _coarse(cm, k):
    """The cost map pooled k x k: passable where any fine cell is (the 2D
    search is a heuristic and a reachability test, so it is generous; the
    arc search is the one that is strict), cost the block's mean."""
    nx, ny = cm.dist.shape
    px, py = (-nx) % k, (-ny) % k
    dist = np.pad(cm.dist, ((0, px), (0, py)), constant_values=0.0)
    cost = np.pad(cm.cost, ((0, px), (0, py)), constant_values=1.0)
    known = np.pad(cm.known, ((0, px), (0, py)), constant_values=False)
    sx, sy = dist.shape[0] // k, dist.shape[1] // k
    dist = dist.reshape(sx, k, sy, k).max(axis=(1, 3))
    cost = cost.reshape(sx, k, sy, k).mean(axis=(1, 3))
    known = known.reshape(sx, k, sy, k).any(axis=(1, 3))
    return dist, cost, known


_OFFSETS = ((1, 0, 1.0), (0, 1, 1.0), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)))


def grid_distances(cm, sources, coarsen=H_COARSEN, cost_weight=COST_WEIGHT,
                   known_only=False, snap_m=0.6, clearance=None):
    """Cost-weighted distances over the coarse passable grid from the cells
    holding `sources` (map points), each snapped to the nearest passable
    cell within snap_m. `known_only` keeps the search off unknown cells, so
    a frontier is reached through mapped floor. Returns a Grid2D."""
    k = max(1, int(coarsen))
    dist, cost, known = _coarse(cm, k)
    res = cm.res * k
    need = cm.inscribed if clearance is None else clearance
    passable = dist >= need
    if known_only:
        passable &= known
    nx, ny = passable.shape
    idx = -np.ones((nx, ny), np.int64)
    ii, jj = np.nonzero(passable)
    idx[ii, jj] = np.arange(len(ii))
    grid = Grid2D(np.full((nx, ny), np.inf), res, cm.origin, passable)
    if not len(ii):
        return grid
    rows, cols, vals = [], [], []
    for dx, dy, length in _OFFSETS:
        # Cell a and its neighbour b at (+dx, +dy), as two aligned views.
        a_sl = (slice(max(0, -dx), nx - max(0, dx)), slice(max(0, -dy), ny - max(0, dy)))
        b_sl = (slice(max(0, dx), nx - max(0, -dx)), slice(max(0, dy), ny - max(0, -dy)))
        a = passable[a_sl]
        b = passable[b_sl]
        both = a & b
        ia = idx[a_sl][both]
        ib = idx[b_sl][both]
        ca = cost[a_sl][both]
        cb = cost[b_sl][both]
        w = length * res * (1.0 + cost_weight * 0.5 * (ca + cb))
        rows.append(ia)
        cols.append(ib)
        vals.append(w)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    n = len(ii)
    graph = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    src = []
    for (x, y) in sources:
        c = _snap(passable, grid, x, y, snap_m)
        if c is not None:
            src.append(idx[c])
    if not src:
        return grid
    d = dijkstra(graph, directed=False, indices=sorted(set(src)), min_only=True)
    out = np.full((nx, ny), np.inf)
    out[ii, jj] = d
    return grid._replace(dist=out)


def _snap(passable, grid, x, y, snap_m):
    """The passable coarse cell nearest (x, y), within snap_m, or None."""
    nx, ny = passable.shape
    cx, cy = grid.cell(x, y)
    r = int(math.ceil(snap_m / grid.res))
    x0, x1 = max(cx - r, 0), min(cx + r + 1, nx)
    y0, y1 = max(cy - r, 0), min(cy + r + 1, ny)
    if x0 >= x1 or y0 >= y1:
        return None
    win = passable[x0:x1, y0:y1]
    ii, jj = np.nonzero(win)
    if not len(ii):
        return None
    d2 = (ii + x0 - cx) ** 2 + (jj + y0 - cy) ** 2
    k = int(np.argmin(d2))
    if d2[k] > r * r:
        return None
    return int(ii[k] + x0), int(jj[k] + y0)


def heuristic_field(cm, goal_xy, params=Params()):
    """The goal's 2D distance field, with the holes round walls filled from
    their neighbours so a node squeezed past the inflated band (the start,
    after a back-out) still gets a finite estimate."""
    g = grid_distances(cm, [goal_xy], cost_weight=params.cost_weight,
                       clearance=tk.HALF_WIDTH_M)
    d = g.dist
    if np.isfinite(d).any():
        big = np.where(np.isfinite(d), d, 1e9)
        filled = ndimage.minimum_filter(big, size=9) + 4 * g.res
        d = np.where(np.isfinite(d), d, np.where(filled < 1e8, filled, np.inf))
    return g._replace(dist=d)


def turn_distance(x, y, th, gx, gy, radius, reverse_factor=REVERSE_FACTOR,
                  reverse=True):
    """The shortest obstacle-free drive from poses (x, y, th) to the point
    (gx, gy), any heading at the end, for a truck that turns no tighter than
    `radius`: one arc then straight, forward or (at reverse_factor the cost)
    in reverse, either way round. Vectorised over poses.

    This is the turn-then-straight family of Reeds-Shepp paths, which is the
    whole of the answer for a goal outside both turning circles; a goal
    inside them is reached by a path with a cusp in it, which this family
    does not have, and gets the straight-line distance times the reverse
    factor instead (backing up to it is always one way there)."""
    x, y, th = np.broadcast_arrays(*(np.asarray(v, float) for v in (x, y, th)))
    dx, dy = gx - x, gy - y
    c, s = np.cos(th), np.sin(th)
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    R = float(radius)
    best = np.hypot(lx, ly) * reverse_factor
    # All four (direction, side) at once: the point mirrored into the
    # forward-left case for each.
    ds = np.array((1, 1, -1, -1) if reverse else (1, 1), float)
    sides = np.array((1, -1, 1, -1) if reverse else (1, -1), float)
    w = np.where(ds > 0, 1.0, reverse_factor)
    shape = (len(ds),) + (1,) * lx.ndim
    length = _turn_then_straight(ds.reshape(shape) * lx[None],
                                 sides.reshape(shape) * ly[None], R)[0]
    return np.minimum(best, (w.reshape(shape) * length).min(0))


def _turn_then_straight(px, py, R):
    """(length, arc angle, straight) of the left-turn-then-straight path from
    the origin heading +x to the point (px, py). inf where the point is
    inside the left circle."""
    cx, cy = px, py - R                       # the point from the circle's centre
    L = np.hypot(cx, cy)
    inside = L < R - 1e-9
    Ls = np.where(inside, R, L)
    straight = np.sqrt(np.maximum(Ls * Ls - R * R, 0.0))
    phi = np.arctan2(cy, cx)
    # The truck starts at angle -pi/2 round the centre and goes anticlockwise
    # to the tangent point whose tangent runs through the point.
    t = phi - np.arccos(np.clip(R / Ls, -1.0, 1.0))
    arc = np.mod(t + math.pi / 2, 2 * math.pi)
    # A point dead ahead is an arc of 0, which rounding can make a lap.
    arc = np.where(arc > 2 * math.pi - 1e-6, 0.0, arc)
    length = np.where(inside, np.inf, R * arc + straight)
    return length, arc, straight


def _line_clear(cm, x, y, gx, gy, step=0.1):
    n = max(2, int(math.ceil(math.hypot(gx - x, gy - y) / step)) + 1)
    t = np.linspace(0.0, 1.0, n)
    return bool((cm.dist_at(x + t * (gx - x), y + t * (gy - y)) >= cm.inscribed).all())


def _shot(cm, x, y, th, gx, gy, kmax, p, need):
    """The cheapest clear turn-then-straight path from (x, y, th) to the
    goal: (cost, poses (n, 3), dirs, curv) or None. Clear means every pose
    on it has footprint clearance `need`."""
    R = 1.0 / kmax
    c, s = math.cos(th), math.sin(th)
    lx = c * (gx - x) + s * (gy - y)
    ly = -s * (gx - x) + c * (gy - y)
    options = []
    for d in ((1, -1) if p.reverse else (1,)):
        for side in (1, -1):
            length, arc, straight = _turn_then_straight(np.float64(d * lx),
                                                        np.float64(side * ly), R)
            if not math.isfinite(float(length)):
                continue
            w = 1.0 if d > 0 else p.reverse_factor
            options.append((w * float(length), d, side, float(arc), float(straight)))
    options.sort()
    for _, d, side, arc, straight in options:
        k = side * kmax
        n_arc = max(1, int(math.ceil(R * arc / SAMPLE_M)))
        n_st = max(1, int(math.ceil(straight / SAMPLE_M)))
        s_arc = d * R * arc * np.arange(1, n_arc + 1) / n_arc
        ax, ay, ath = _arc(x, y, th, np.full(n_arc, k), s_arc)
        s_st = d * straight * np.arange(1, n_st + 1) / n_st
        bx, by, bth = _arc(float(ax[-1]), float(ay[-1]), float(ath[-1]),
                           np.zeros(n_st), s_st)
        px = np.concatenate([ax, bx])
        py = np.concatenate([ay, by])
        pth = np.concatenate([ath, bth])
        # Every fourth pose first: most shots that fail, fail cheaply.
        if not (cm.footprint_clearance(px[::4], py[::4], pth[::4]) >= need).all():
            continue
        if not (cm.footprint_clearance(px, py, pth) >= need).all():
            continue
        if d > 0 and p.floor_probe:
            far = np.hypot(px - gx, py - gy) > PROBE_FREE_M
            if far.any() and (cm.probe_clearance(px[far][::2], py[far][::2],
                                                 pth[far][::2]) < PROBE_CLEAR_M).any():
                continue
        cost = cm.cost_at(px, py)
        w = 1.0 if d > 0 else p.reverse_factor
        total = (R * arc + straight) * w + p.cost_weight * float(cost.mean()) * (
            R * arc + straight) + p.steer_change * 1.0
        poses = np.stack([px, py, pth], 1)
        dirs = np.full(len(px), d, int)
        curv = np.concatenate([np.full(n_arc, k), np.zeros(n_st)])
        return total, poses, dirs, curv
    return None


def _arc(x, y, th, k, s):
    """Poses s metres (signed) along curvature k from (x, y, th). Broadcasts."""
    th1 = th + k * s
    small = np.abs(k) < 1e-6
    ks = np.where(small, 1.0, k)
    ax = np.where(small, x + s * np.cos(th), x + (np.sin(th1) - np.sin(th)) / ks)
    ay = np.where(small, y + s * np.sin(th), y - (np.cos(th1) - np.cos(th)) / ks)
    return ax, ay, th1


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def plan(cm, start, goal_xy, goal_yaw=None, params=Params(), h=None,
         start_dir=0, deadline=None):
    """A Path from `start` (x, y, yaw) to within goal_tol of goal_xy, or
    None. `h` is a heuristic_field for this goal if the caller has one.
    `start_dir` is the way the truck is already moving (+1, -1 or 0); a
    first move the other way pays the switch penalty. `deadline` is a
    time.perf_counter() past which it gives up (None)."""
    p = params
    if h is None:
        h = heuristic_field(cm, goal_xy, p)
    kmax = tk.MAX_CURVATURE * p.curvature
    prims = []
    for d in ((1, -1) if p.reverse else (1,)):
        for f in CURVATURE_STEPS:
            prims.append((d, f, f * kmax))
    pd = np.array([q[0] for q in prims], float)
    pf = np.array([q[1] for q in prims], float)
    pk = np.array([q[2] for q in prims], float)
    fr = np.array([1 / 3, 2 / 3, 1.0])
    S = (pd * p.step)[:, None] * fr[None, :]          # (P, 3) signed arc lengths
    K = np.repeat(pk[:, None], 3, 1)

    sx, sy, sth = (float(v) for v in start[:3])
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    # A start already inside the margin -- squeezed by noise, or left there
    # by a back-out -- may move as long as it does not get any closer.
    start_clear = float(cm.footprint_clearance(sx, sy, sth))
    need = min(cm.margin, start_clear - 0.01)
    relax_r = 0.5

    h0 = float(h.at(sx, sy))
    if not math.isfinite(h0):
        h0 = math.hypot(gx - sx, gy - sy) * 3.0
    if goal_yaw is None:
        h0 = max(h0, float(turn_distance(sx, sy, sth, gx, gy, 1.0 / kmax,
                                         p.reverse_factor, p.reverse)))

    R = 1.0 / kmax

    def hval(xs, ys, ths):
        v = h.at(xs, ys)
        bad = ~np.isfinite(v)
        if bad.any():
            v = np.where(bad, np.hypot(gx - xs, gy - ys) * 3.0, v)
        if goal_yaw is None:
            v = np.maximum(v, turn_distance(xs, ys, ths, gx, gy, R,
                                            p.reverse_factor, p.reverse))
        return v

    def key(x, y, th):
        return (int(math.floor(x / p.xy_bin)), int(math.floor(y / p.xy_bin)),
                int(math.floor(_wrap(th) / (2 * math.pi) * p.yaw_bins)) % p.yaw_bins)

    # node: (x, y, th, dir, curv_frac, g, parent, prim)
    nodes = [(sx, sy, sth, int(start_dir), 0.0, 0.0, -1, -1)]
    best = {key(sx, sy, sth): 0.0}
    heap = [(p.h_weight * h0, 0)]
    expansions = 0
    goal_node = None
    tail = None
    while heap and expansions < p.max_expansions:
        if deadline is not None and expansions % 64 == 0 and time.perf_counter() > deadline:
            return None
        f, ni = heapq.heappop(heap)
        x, y, th, dprev, fprev, g, _, _ = nodes[ni]
        if g > best.get(key(x, y, th), math.inf) + 1e-9:
            continue
        if math.hypot(x - gx, y - gy) <= p.goal_tol and (
                goal_yaw is None or abs(_wrap(th - goal_yaw)) <= p.yaw_tol):
            goal_node = ni
            break
        if (p.shot and goal_yaw is None and expansions % SHOT_EVERY == 0
                and math.hypot(x - gx, y - gy) <= SHOT_M
                and _line_clear(cm, x, y, gx, gy)):
            near_start = math.hypot(x - sx, y - sy) < relax_r
            shot = _shot(cm, x, y, th, gx, gy, kmax, p,
                         need if near_start else cm.margin)
            if shot is not None:
                goal_node = ni
                tail = shot
                break
        expansions += 1
        ax, ay, ath = _arc(x, y, th, K, S)               # (P, 3)
        clear = cm.footprint_clearance(ax, ay, ath)       # (P, 3)
        near_start = math.hypot(x - sx, y - sy) < relax_r
        ok = (clear >= (need if near_start else cm.margin)).all(1)
        # A forward move ending with the floor-veto box on something is one
        # the car will stop; except on the last stretch to the goal.
        fwd = ok & (pd > 0)
        if p.floor_probe and fwd.any():
            ex_, ey_ = ax[fwd, -1], ay[fwd, -1]
            far = np.hypot(ex_ - gx, ey_ - gy) > PROBE_FREE_M
            if far.any():
                probe = cm.probe_clearance(ex_, ey_, ath[fwd, -1])
                bad = far & (probe < PROBE_CLEAR_M)
                idx = np.nonzero(fwd)[0]
                ok[idx[bad]] = False
        if not ok.any():
            continue
        cost = cm.cost_at(ax, ay).mean(1)
        ex, ey, eth = ax[:, -1], ay[:, -1], ath[:, -1]
        step_g = p.step * (1.0 + p.cost_weight * cost)
        step_g = np.where(pd < 0, step_g * p.reverse_factor, step_g)
        switch = (pd != dprev) & (dprev != 0)
        step_g = step_g + np.where(switch, p.switch_penalty, 0.0)
        step_g = step_g + p.steer_change * np.abs(pf - fprev)
        hs = hval(ex, ey, eth)
        for j in np.nonzero(ok)[0]:
            g2 = g + float(step_g[j])
            kk = key(ex[j], ey[j], eth[j])
            if g2 >= best.get(kk, math.inf) - 1e-9:
                continue
            best[kk] = g2
            nodes.append((float(ex[j]), float(ey[j]), float(eth[j]), int(pd[j]),
                          float(pf[j]), g2, ni, int(j)))
            heapq.heappush(heap, (g2 + p.h_weight * float(hs[j]), len(nodes) - 1))
    if goal_node is None:
        return None
    return _reconstruct(nodes, goal_node, prims, kmax, p, expansions, tail)


def _reconstruct(nodes, gi, prims, kmax, p, expansions, tail=None):
    chain = []
    i = gi
    while i >= 0:
        chain.append(nodes[i])
        i = nodes[i][6]
    chain.reverse()
    x0, y0, th0 = chain[0][0], chain[0][1], chain[0][2]
    poses = [(x0, y0, th0)]
    dirs = []
    curv = []
    length = 0.0
    n_sub = max(1, int(round(p.step / SAMPLE_M)))
    for node in chain[1:]:
        prim = prims[node[7]]
        d, _, k = prim
        px, py, pth = poses[-1]
        for t in range(1, n_sub + 1):
            s = d * p.step * t / n_sub
            ax, ay, ath = _arc(px, py, pth, np.array(k), np.array(s))
            poses.append((float(ax), float(ay), float(ath)))
            dirs.append(d)
            curv.append(k)
        # Snap to the node's own pose, so rounding never accumulates.
        poses[-1] = (node[0], node[1], node[2])
        length += p.step
    cost = float(nodes[gi][5])
    if tail is not None:
        t_cost, t_poses, t_dirs, t_curv = tail
        d_last = dirs[-1] if dirs else int(chain[-1][3])
        if d_last and int(t_dirs[0]) != d_last:
            t_cost += p.switch_penalty
        poses.extend(tuple(float(v) for v in q) for q in t_poses)
        dirs.extend(int(v) for v in t_dirs)
        curv.extend(float(v) for v in t_curv)
        length += float(np.sum(np.hypot(np.diff(t_poses[:, 0], prepend=poses[-len(t_poses) - 1][0]),
                                        np.diff(t_poses[:, 1], prepend=poses[-len(t_poses) - 1][1]))))
        cost += t_cost
    if not dirs:
        dirs, curv = [1], [0.0]
        poses.append(poses[0])
    dirs = [dirs[0]] + dirs
    curv = [curv[0]] + curv
    dirs_a = np.array(dirs, int)
    reverses = int(np.count_nonzero(np.diff(dirs_a) != 0))
    return Path(np.array(poses), dirs_a, np.array(curv), length,
                cost, expansions, reverses)


def path_clearance(cm, path, start_index=0, ahead_m=None):
    """The least footprint clearance along a path from start_index, over at
    most ahead_m of it. Positive is clear by that much past nothing; the
    caller compares it with the margin."""
    poses = path.poses[start_index:]
    if ahead_m is not None:
        n = int(math.ceil(ahead_m / SAMPLE_M)) + 1
        poses = poses[:n]
    if not len(poses):
        return math.inf
    return float(np.min(cm.footprint_clearance(poses[:, 0], poses[:, 1], poses[:, 2])))
