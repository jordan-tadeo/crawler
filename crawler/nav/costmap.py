"""Obstacles, grown by the truck, with a cost that falls away beyond them.

Built off one MapView, on the view's own grid (cropped to a window round
the truck and the goal, see build()):

  dist     metres from each cell to the nearest solid one, 0 inside it. The
           footprint check reads this: the truck's three circles
           (truck.CIRCLE_X_M) are clear when the distance at each centre is
           past CIRCLE_R_M plus the margin.
  lethal   cells closer than INSCRIBED_M, the half-width plus the margin. A
           cell here is one the middle of the truck can never be in without
           a wheel in the wall, whatever its heading: the "obstacles inflated
           by the half-width plus a margin" grid. The 2D search runs on its
           complement.
  cost     0..1, 1 at the lethal edge falling off exponentially to 0 at
           INFLATION_M. Paths pay for it, so they keep to the middle of a
           doorway instead of shaving the frame.
  unknown  cells never observed. Not lethal -- a path across the unmapped
           side of the room is the only way to a goal out there, and
           outdoors most of the world is unknown -- but they cost
           UNKNOWN_COST.

The margin is the one number that trades doorways against scrapes. The
footprint check needs CIRCLE_R_M + margin (0.161 + 0.06 m) at each circle
centre, so the truck fits an opening of 0.44 m, and the middle of the truck
is lethal within 0.196 m of anything. House doorways are 0.7 m and up.
"""

import math
from typing import NamedTuple

import numpy as np
from scipy import ndimage

from crawler.nav import truck as tk

MARGIN_M = 0.06
INSCRIBED_M = tk.HALF_WIDTH_M + MARGIN_M
INFLATION_M = 0.6
UNKNOWN_COST = 0.3

# How far round the truck and the goal a cost map reaches, past the box that
# holds them both. 3 m is room for a path to go the long way round a sofa.
WINDOW_PAD_M = 3.0


class CostMap(NamedTuple):
    dist: np.ndarray       # float32 metres to the nearest solid cell
    lethal: np.ndarray     # bool, dist < inscribed
    cost: np.ndarray       # float32 0..1 (lethal cells 1)
    unknown: np.ndarray    # bool
    solid: np.ndarray      # bool, the occupied cells themselves
    known: np.ndarray      # bool
    res: float
    origin: tuple
    margin: float
    inscribed: float
    version: object = None

    @property
    def shape(self):
        return self.dist.shape

    def world_to_cell(self, x, y):
        ix = np.floor((np.asarray(x) - self.origin[0]) / self.res).astype(np.int64)
        iy = np.floor((np.asarray(y) - self.origin[1]) / self.res).astype(np.int64)
        return ix, iy

    def cell_to_world(self, ix, iy):
        return (self.origin[0] + (np.asarray(ix) + 0.5) * self.res,
                self.origin[1] + (np.asarray(iy) + 0.5) * self.res)

    def inside(self, ix, iy):
        nx, ny = self.dist.shape
        return (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    def dist_at(self, x, y):
        """Distance to the nearest solid cell at map points. Outside the
        window is 0: the planner never leaves the map it was given."""
        ix, iy = self.world_to_cell(x, y)
        ok = self.inside(ix, iy)
        out = np.zeros(np.shape(ix), np.float32)
        out[ok] = self.dist[ix[ok], iy[ok]]
        return out

    def cost_at(self, x, y):
        ix, iy = self.world_to_cell(x, y)
        ok = self.inside(ix, iy)
        out = np.ones(np.shape(ix), np.float32)
        out[ok] = self.cost[ix[ok], iy[ok]]
        return out

    def footprint_clearance(self, x, y, yaw):
        """The truck's least clearance at poses: the smallest of
        dist - CIRCLE_R_M over the three circles. Negative is touching.
        Vectorised over poses."""
        x, y, yaw = (np.asarray(v, float) for v in (x, y, yaw))
        c, s = np.cos(yaw), np.sin(yaw)
        worst = None
        for cx in tk.CIRCLE_X_M:
            d = self.dist_at(x + c * cx, y + s * cx) - tk.CIRCLE_R_M
            worst = d if worst is None else np.minimum(worst, d)
        return worst

    def footprint_free(self, x, y, yaw, margin=None):
        m = self.margin if margin is None else margin
        return self.footprint_clearance(x, y, yaw) >= m


def window(view, points=(), pad=WINDOW_PAD_M):
    """(i0, i1, j0, j1) of the view's grid to build a cost map over: the
    known cells near the truck plus the box round the truck and any `points`
    (the goal), padded, clipped to the grid."""
    nx, ny = view.occupied.shape
    xs = [view.pose[0]] + [p[0] for p in points]
    ys = [view.pose[1]] + [p[1] for p in points]
    ix, iy = view.world_to_cell(np.array(xs), np.array(ys))
    p = int(math.ceil(pad / view.res))
    i0, i1 = int(ix.min()) - p, int(ix.max()) + p + 1
    j0, j1 = int(iy.min()) - p, int(iy.max()) + p + 1
    return max(i0, 0), min(i1, nx), max(j0, 0), min(j1, ny)


def build(view, margin=MARGIN_M, inflation=INFLATION_M,
          unknown_cost=UNKNOWN_COST, crop=None):
    """A CostMap off a MapView. `crop` is (i0, i1, j0, j1) or None for all
    of it; the crop's origin is carried, so world coordinates stay right."""
    solid = view.solid()
    known = view.known
    origin = view.origin
    if crop is not None:
        i0, i1, j0, j1 = crop
        solid = solid[i0:i1, j0:j1]
        known = known[i0:i1, j0:j1]
        origin = (origin[0] + i0 * view.res, origin[1] + j0 * view.res)
    res = float(view.res)
    if solid.size == 0:
        solid = np.zeros((1, 1), bool)
        known = np.zeros((1, 1), bool)
    if solid.any():
        dist = ndimage.distance_transform_edt(~solid).astype(np.float32) * res
    else:
        dist = np.full(solid.shape, 1e3, np.float32)
    inscribed = tk.HALF_WIDTH_M + margin
    lethal = dist < inscribed
    # 1 at the lethal edge, e-folding over a third of the band, 0 at its end.
    band = max(inflation - inscribed, 1e-3)
    u = np.clip((dist - inscribed) / band, 0.0, 1.0)
    cost = (np.exp(-3.0 * u) - math.exp(-3.0)) / (1.0 - math.exp(-3.0))
    cost = cost.astype(np.float32)
    unknown = ~known
    cost = np.where(unknown, np.maximum(cost, unknown_cost), cost)
    cost[lethal] = 1.0
    return CostMap(dist, lethal, cost.astype(np.float32), unknown, solid,
                   known, res, origin, float(margin), float(inscribed),
                   view.version)
