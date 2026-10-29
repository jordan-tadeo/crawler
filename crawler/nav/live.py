"""The live SLAM as the navigator's map: SlamMap, the map_source NavDriver
reads with --slam --nav.

view() is a MapView off the runner's newest pose and its full-resolution
nav_map() (runner.NAV_HALF_M round the truck, the grid's own 5 cm cells),
or None. None is the navigator standing still ("no map"), and it is what
view() gives for every pose it would not drive on, by the same rule the map
policy uses ("map lost", autopilot/policy.py):

  no Result yet, or one that is not tracking and matched (egomap.trusted)
  a Result older than POSE_STALE_S: the mapper is behind or gone
  a worker that is not alive
  a confidence (runner.confidence) under TRUST_MIN
  no nav map, or one older than MAP_STALE_S

The map policy waits MAP_LOST_S before it goes neutral; this stops on the
first decision. A navigator only ever moves on a path it planned on this
map, so a pose it does not trust is a path it cannot follow.

Between TRUST_MIN and the navigator's own CONFIDENCE_MIN the navigator
drives on local(): the LocalMap of the last few seconds of scans, placed by
poses that were still tracking. The newest lidar scan goes into it on every
view() that is trusted, and into MapView.scan.

The low-obstacle layer (--nav-low-obstacles, off by default) is the OAK-D's
LowView off --voxel-map, on the same 5 cm lattice. It is advisory: an
obstacle cell is OR'd in as solid, through MapView.layers, and nothing it
says ever takes anything out. Its "floor" never clears a lidar cell, and an
unseen cell is just unseen. A LowView older than LOW_STALE_S is left out.
Its phantom rate is not settled, which is why it is off.
"""

import math
import time

import numpy as np

from crawler.nav import mapview as mv

# Under this the pose is not driven on at all. runner.confidence() is the
# inlier fraction scaled down where the scene does not pin the pose; a
# room is about 0.9, a long corridor goes under 0.3.
TRUST_MIN = 0.15
# A Result older than this (monotonic seconds against its scan time) is the
# mapper behind or gone: egomap.STALE_S, the map policy's number.
POSE_STALE_S = 0.3
# The nav map comes twice a second; three missed is a stale one.
MAP_STALE_S = 2.0
# The LowView comes once a second.
LOW_STALE_S = 3.0

# grid.UNKNOWN / FREE / OCCUPIED and voxel.LOW_OBSTACLE, here so the
# navigator does not import the mapper (tests check they agree).
UNKNOWN, FREE, OCCUPIED = 0, 1, 2
LOW_OBSTACLE = 2


def low_layer(low, shape, x0, y0, res):
    """bool [shape] on the nav map's grid: the LowView's obstacle cells,
    where the two overlap. Both are on the same map-frame lattice, so it is
    an integer offset; a LowView at another resolution is no layer."""
    out = np.zeros(shape, bool)
    if low is None or abs(float(low.res) - float(res)) > 1e-6:
        return out
    di = int(round((float(low.x0) - x0) / res))
    dj = int(round((float(low.y0) - y0) / res))
    obs = np.asarray(low.codes) == LOW_OBSTACLE
    nx, ny = shape
    a0, a1 = max(di, 0), min(di + obs.shape[0], nx)
    b0, b1 = max(dj, 0), min(dj + obs.shape[1], ny)
    if a1 <= a0 or b1 <= b0:
        return out
    out[a0:a1, b0:b1] = obs[a0 - di:a1 - di, b0 - dj:b1 - dj]
    return out


class SlamMap:
    """map_source for NavDriver off a mapping.runner.SlamRunner.

    `lidar` is the LidarReader (its latest() scan feeds MapView.scan and the
    LocalMap), `voxels` the VoxelMapper (its low()), read only with
    `low_obstacles`. Every read is the runners' own non-blocking getters,
    and view() never raises: anything wrong is None.
    """

    def __init__(self, slam, lidar=None, voxels=None, low_obstacles=False,
                 clock=time.monotonic, trust_min=TRUST_MIN):
        self.slam = slam
        self.lidar = lidar
        self.voxels = voxels
        self.low_obstacles = bool(low_obstacles)
        self.trust_min = float(trust_min)
        self._clock = clock
        self.local_map = mv.LocalMap()
        self._scan_t = None
        self.why = "no result"      # why the last view() was None, "" if it was not

    def view(self):
        try:
            return self._view()
        except Exception as e:
            self.why = f"error {e!r}"
            return None

    def local(self):
        return self.local_map

    def _refuse(self, why):
        self.why = why
        return None

    def _view(self):
        now = self._clock()
        slam = self.slam
        try:
            alive = bool(slam.stats().get("alive", False))
        except Exception:
            alive = False
        if not alive:
            return self._refuse("mapper not running")
        res = slam.latest()
        if res is None:
            return self._refuse("no result")
        if not (res.state == "tracking" and bool(res.matched)):
            return self._refuse(f"not tracking ({res.state})")
        if now - float(res.t) > POSE_STALE_S:
            return self._refuse("pose stale")
        from crawler.mapping import runner
        conf = float(runner.confidence(res))
        if not math.isfinite(conf) or conf < self.trust_min:
            return self._refuse(f"confidence {conf:.2f}")
        m = slam.nav_map()
        if m is None:
            return self._refuse("no map")
        if now - float(m.t) > MAP_STALE_S:
            return self._refuse("map stale")
        pose = (float(res.x), float(res.y), float(res.yaw))
        codes = np.asarray(m.codes)
        scan = self._scan(now, pose)
        layers = ()
        if self.low_obstacles and self.voxels is not None:
            low = self.voxels.low()
            if low is not None and now - float(low.t) <= LOW_STALE_S:
                layers = (low_layer(low, codes.shape, float(m.x0), float(m.y0),
                                    float(m.res)),)
        self.why = ""
        return mv.MapView(codes == OCCUPIED, codes != UNKNOWN, float(m.res),
                          (float(m.x0), float(m.y0)), pose, conf,
                          ("slam", int(m.version), float(m.t)), scan, layers)

    def _scan(self, now, pose):
        """The newest lidar scan in the map frame, (n, 2), placed by `pose`,
        and into the LocalMap once. None with no fresh scan."""
        if self.lidar is None:
            return None
        scan = self.lidar.latest()
        if scan is None:
            return None
        from crawler.mapping import runner, scan as sc
        pts, _ = sc.all_points(runner.ranges_of(scan, self.slam.mount))
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        world = np.stack([x + c * pts[:, 0] - s * pts[:, 1],
                          y + s * pts[:, 0] + c * pts[:, 1]], 1)
        if self._scan_t != scan.t:
            self._scan_t = scan.t
            self.local_map.add(now, (x, y), world)
        return world
