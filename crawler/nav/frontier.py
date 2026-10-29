"""Frontiers: the edge of what the map knows, for rove.

A frontier cell is a known-free cell with an unknown cell beside it. Driving
to one shows the lidar what is past it, which grows the map, which moves the
frontier: that loop is the whole of exploration. Frontier-based exploration
(Yamauchi, 1997) is the standard way to do it and it needs nothing a 2D
occupancy grid does not have, indoors or out.

Which one to go to. The cells are grouped into connected frontiers, and the
small ones dropped (MIN_CELLS: a gap a hand's width across between two chair
legs is not a way anywhere). Each is scored by how far the truck has to drive
to reach it -- through mapped floor only, off grid_distances, so the distance
is a real one and an unreachable frontier never scores -- less a credit for
its size, so a wide opening a little further on beats a crack next to the
truck. Frontiers that failed before (no plan, or the truck got stuck going
there) are on a blacklist and skipped.

The point to drive to starts as the frontier cell the truck can reach most
cheaply, which is on the near side of the opening, where the truck sees
through it -- and is then moved (snap_clear) to the reachable cell within
SNAP_M of it with the most room round it, up to CLEAR_CAP_M. A frontier
cell is often hard against a wall (the lidar sees past a door frame's edge
first), and a goal there is a goal the truck can only reach by shaving the
frame.
"""

import math
from typing import NamedTuple

import numpy as np
from scipy import ndimage

from crawler.nav import planner as pl

MIN_CELLS = 8                 # 0.4 m of frontier at 5 cm
SIZE_CREDIT = 1.0             # metres of driving one metre of frontier is worth
BLACKLIST_M = 0.6
# A frontier closer than this to the truck is one it is already looking
# through; going there would not grow the map.
MIN_DIST_M = 0.4
SNAP_M = 0.5
CLEAR_CAP_M = 0.5
SNAP_COST = 0.3               # metres of clearance one metre of moving it costs


class Frontier(NamedTuple):
    x: float            # the point to drive to, map metres
    y: float
    size_m: float       # the frontier's length, roughly
    distance_m: float   # cost-weighted driving distance from the truck
    score: float
    cells: int


def frontier_mask(cm):
    """bool on the cost map's grid: known-free cells next to unknown ones."""
    free = cm.known & ~cm.solid
    unknown = ~cm.known
    near_unknown = ndimage.binary_dilation(
        unknown, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
    mask = free & near_unknown
    # The window's own edge is not a frontier: past it is just not loaded.
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    return mask


def find(cm, pose, blacklist=(), min_cells=MIN_CELLS, size_credit=SIZE_CREDIT,
         reach=None):
    """Every reachable frontier, best first. `reach` is a grid_distances off
    the truck if the caller already has one."""
    mask = frontier_mask(cm)
    if not mask.any():
        return []
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if reach is None:
        reach = pl.grid_distances(cm, [pose[:2]], known_only=True)
    out = []
    ii, jj = np.nonzero(labels)
    lab = labels[ii, jj]
    wx, wy = cm.cell_to_world(ii, jj)
    dist = reach.at(wx, wy)
    order = np.argsort(lab, kind="stable")
    lab, wx, wy, dist = lab[order], wx[order], wy[order], dist[order]
    bounds = np.searchsorted(lab, np.arange(1, n + 2))
    for k in range(n):
        a, b = bounds[k], bounds[k + 1]
        cells = b - a
        if cells < min_cells:
            continue
        d = dist[a:b]
        finite = np.isfinite(d)
        if not finite.any():
            continue
        far = np.hypot(wx[a:b] - pose[0], wy[a:b] - pose[1]) >= MIN_DIST_M
        pick = finite & far
        if not pick.any():
            continue
        dd = np.where(pick, d, np.inf)
        m = int(np.argmin(dd))
        x, y = snap_clear(cm, reach, float(wx[a + m]), float(wy[a + m]))
        if any(math.hypot(x - bx, y - by) < BLACKLIST_M for bx, by in blacklist):
            continue
        size = cells * cm.res
        score = float(dd[m]) - size_credit * min(size, 3.0)
        out.append(Frontier(x, y, size, float(dd[m]), score, int(cells)))
    out.sort(key=lambda f: f.score)
    return out


def snap_clear(cm, reach, x, y, snap_m=SNAP_M, cap=CLEAR_CAP_M, cost=SNAP_COST):
    """(x, y) moved to the cell within snap_m with the most clearance (to
    cap), less `cost` a metre moved, among those the truck can stand on
    and reach. The point itself if none is better."""
    ix, iy = cm.world_to_cell(x, y)
    r = int(math.ceil(snap_m / cm.res))
    nx, ny = cm.shape
    x0, x1 = max(int(ix) - r, 0), min(int(ix) + r + 1, nx)
    y0, y1 = max(int(iy) - r, 0), min(int(iy) + r + 1, ny)
    if x0 >= x1 or y0 >= y1:
        return x, y
    ii, jj = np.mgrid[x0:x1, y0:y1]
    ii, jj = ii.ravel(), jj.ravel()
    off = np.hypot(ii - ix, jj - iy) * cm.res
    clear = cm.dist[ii, jj]
    wx, wy = cm.cell_to_world(ii, jj)
    ok = ((off <= snap_m) & cm.known[ii, jj] & (clear >= cm.inscribed)
          & np.isfinite(reach.at(wx, wy)))
    if not ok.any():
        return x, y
    score = np.minimum(clear, cap) - cost * off
    score = np.where(ok, score, -np.inf)
    k = int(np.argmax(score))
    here = min(float(cm.dist[ix, iy]), cap) if cm.inside(ix, iy) else -np.inf
    if score[k] <= here:
        return x, y
    return float(wx[k]), float(wy[k])
