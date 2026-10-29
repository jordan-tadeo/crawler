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

The point to drive to is the frontier cell the truck can reach most cheaply,
which is on the near side of the opening, where the truck sees through it.
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
        x, y = float(wx[a + m]), float(wy[a + m])
        if any(math.hypot(x - bx, y - by) < BLACKLIST_M for bx, by in blacklist):
            continue
        size = cells * cm.res
        score = float(dd[m]) - size_credit * min(size, 3.0)
        out.append(Frontier(x, y, size, float(dd[m]), score, int(cells)))
    out.sort(key=lambda f: f.score)
    return out
