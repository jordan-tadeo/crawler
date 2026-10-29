"""What the navigator is handed each decision: a map, a pose, and how much to
trust them.

MapView is the whole interface to the SLAM. It is deliberately small, so the
live SLAM (crawler/mapping, Slam.grid and Slam.pose) and the simulator can
both fill it:

  occupied    bool [nx, ny], cells to keep the truck out of
  known       bool [nx, ny], cells that have been observed at all
  res         metres a cell
  origin      (x, y) map metres of cell (0, 0)'s outer corner
  pose        (x, y, yaw) of the lidar, map frame
  confidence  0..1, how far the pose and map are to be trusted
  version     anything that changes when the map does, so a cost map is
              not rebuilt for a map that did not move
  scan        the newest scan's returns in the map frame, (n, 2), or None:
              what LocalMap is built from

Indexing is [ix, iy], x first, as crawler/mapping/grid.py stores it.

The extra layers seam. `layers` is a tuple of more bool [nx, ny] arrays on
the same grid, OR'd into occupied before anything else sees it. A camera
depth layer -- the low things the roof lidar's plane passes over, a kerb, a
rock, a shoe -- goes in there later and nothing downstream changes.

LocalMap is the fallback when the global confidence is low. It is a window
round the truck built from only the last few seconds of scans, so a pose
that has drifted has not had time to smear it. It is never wrong by more
than a few seconds of odometry, and it knows nothing beyond the lidar's
range, which is all obstacle avoidance needs.
"""

import math
from collections import deque
from typing import NamedTuple, Optional, Tuple

import numpy as np


class MapView(NamedTuple):
    occupied: np.ndarray
    known: np.ndarray
    res: float
    origin: Tuple[float, float]
    pose: Tuple[float, float, float]
    confidence: float = 1.0
    version: object = None
    scan: Optional[np.ndarray] = None
    layers: tuple = ()
    local: bool = False

    @property
    def shape(self):
        return self.occupied.shape

    def solid(self):
        """occupied with every extra layer OR'd in."""
        occ = self.occupied
        for layer in self.layers:
            if layer is not None and layer.shape == occ.shape:
                occ = occ | layer
        return occ

    def world_to_cell(self, x, y):
        ix = np.floor((np.asarray(x) - self.origin[0]) / self.res).astype(np.int64)
        iy = np.floor((np.asarray(y) - self.origin[1]) / self.res).astype(np.int64)
        return ix, iy

    def cell_to_world(self, ix, iy):
        return (self.origin[0] + (np.asarray(ix) + 0.5) * self.res,
                self.origin[1] + (np.asarray(iy) + 0.5) * self.res)


# What live SLAM states mean for trust. crawler/mapping/slam.py's Result.state
# and its inlier fraction; anything not tracking is not trusted at all.
SLAM_TRACKING = "tracking"


def slam_confidence(result) -> float:
    """0..1 off a live SLAM Result (crawler/mapping/slam.Result).

    Tracking scales with the fit's inlier fraction, so a corridor match that
    barely holds reads lower than a room. Tilted still tracks but writes
    nothing, so it is half. Anything else -- lost, rolled, no points -- is 0.
    Anything without those fields is 0, never a guess.
    """
    try:
        state = str(result.state)
        inliers = float(result.inliers)
    except Exception:
        return 0.0
    if not math.isfinite(inliers):
        return 0.0
    if state == SLAM_TRACKING and bool(getattr(result, "matched", True)):
        return float(np.clip(inliers, 0.0, 1.0))
    if state == "tilted":
        return 0.5 * float(np.clip(inliers, 0.0, 1.0))
    return 0.0


def from_grid(grid, pose, confidence=1.0, scan=None, layers=()):
    """A MapView off anything shaped like crawler/mapping/grid.Grid.

    It asks the grid for occupied() and known() and reads res and origin, so
    the cells and the frame are the grid's own. version is the grid's, so a
    cost map knows when to rebuild."""
    return MapView(
        occupied=np.asarray(grid.occupied(), bool),
        known=np.asarray(grid.known(), bool),
        res=float(grid.res),
        origin=tuple(float(v) for v in grid.origin),
        pose=tuple(float(v) for v in pose[:3]),
        confidence=float(confidence),
        version=getattr(grid, "version", None),
        scan=scan,
        layers=tuple(layers))


# The local window: 8 m square round the truck, 5 cm cells, the last 3 s of
# scans. 4 m each way is past the furthest a decision ever looks ahead, and
# 3 s at 0.42 m/s is 1.3 m of travel: short enough for odometry drift to be
# a centimetre or two, long enough to remember a wall the truck just turned
# its lidar plane away from.
LOCAL_HALF_M = 4.0
LOCAL_RES = 0.05
LOCAL_MEMORY_S = 3.0


class LocalMap:
    """A truck-centred occupancy window off the last LOCAL_MEMORY_S of scans.

    add(t, points) takes a scan already in the map frame (the pose it was
    taken at applied). view(pose) rasterises what is remembered round that
    pose: occupied where a return landed, known along every ray from the pose
    the scan was taken at. Bounded: the deque holds at most `max_scans`.
    """

    def __init__(self, half=LOCAL_HALF_M, res=LOCAL_RES,
                 memory_s=LOCAL_MEMORY_S, max_scans=60):
        self.half = float(half)
        self.res = float(res)
        self.memory_s = float(memory_s)
        self._scans = deque(maxlen=int(max_scans))
        self.version = 0

    def add(self, t, origin_xy, points):
        pts = np.asarray(points, float).reshape(-1, 2)
        pts = pts[np.all(np.isfinite(pts), 1)]
        self._scans.append((float(t), (float(origin_xy[0]), float(origin_xy[1])), pts))
        while self._scans and t - self._scans[0][0] > self.memory_s:
            self._scans.popleft()
        self.version += 1

    def clear(self):
        self._scans.clear()
        self.version += 1

    def view(self, pose, confidence=0.0, layers=()):
        n = int(round(2 * self.half / self.res))
        ox = math.floor((pose[0] - self.half) / self.res) * self.res
        oy = math.floor((pose[1] - self.half) / self.res) * self.res
        occ = np.zeros((n, n), bool)
        known = np.zeros((n, n), bool)
        for _, (sx, sy), pts in self._scans:
            if not len(pts):
                continue
            # Free along each ray, sampled at half a cell; then the hits.
            d = pts - (sx, sy)
            r = np.hypot(d[:, 0], d[:, 1])
            m = int(np.ceil(r.max() / (self.res * 0.5)))
            if m > 0:
                steps = np.linspace(0.0, 1.0, m, endpoint=False)
                fx = sx + d[:, 0:1] * steps
                fy = sy + d[:, 1:2] * steps
                ix = np.floor((fx - ox) / self.res).astype(np.int64).ravel()
                iy = np.floor((fy - oy) / self.res).astype(np.int64).ravel()
                ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
                known[ix[ok], iy[ok]] = True
            hx = np.floor((pts[:, 0] - ox) / self.res).astype(np.int64)
            hy = np.floor((pts[:, 1] - oy) / self.res).astype(np.int64)
            ok = (hx >= 0) & (hx < n) & (hy >= 0) & (hy < n)
            occ[hx[ok], hy[ok]] = True
            known[hx[ok], hy[ok]] = True
        return MapView(occ, known, self.res, (ox, oy),
                       tuple(float(v) for v in pose[:3]), float(confidence),
                       ("local", self.version, ox, oy), None, tuple(layers),
                       True)
