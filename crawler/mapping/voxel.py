"""The OAK-D's depth as a 3D voxel map, and the low obstacles the lidar cannot see.

The roof lidar sweeps one plane at cab height. Everything shorter than that --
a shoe, a rock, a cat toy, the bottom rail of a chair -- passes underneath
it. The OAK-D on the front axle sees it. This turns each depth frame into
two maps in the SLAM "map" frame, placed by the lidar SLAM pose:

  3D    int8 log-odds voxels, 5 x 5 cm in plan and 2.5 cm tall, hits from
        every depth pixel inside MAX_RANGE_M and free space ray-cast along a
        decimated set of rays. Heights are above the floor plane fitted to
        that frame, so the floor is z = 0 wherever the truck is.
  low   the 2.5D low-obstacle layer, 5 cm cells: a pixel standing LOW_MIN_M
        to LOW_MAX_M above its LOCAL ground (not above gravity's plane) and
        within LOW_RANGE_M raises its cell by one, a floor pixel lowers it.
        A cell is a low obstacle at LOW_CELL_SCORE and up: two frames that
        saw it more than saw floor there.

Off unless the service runs with --voxel-map (which needs --slam). Mapping
only: nothing here commands the truck, and nothing here runs on the control
thread. The camera's reader thread hands each depth frame to on_depth(), which
keeps the newest one in a one-deep slot and returns. A feeder thread places
it (the SLAM pose and the IMU at its capture time) and hands it to a worker
that owns the maps: a process of its own by default, niced, at most
MAP_MAX_HZ frames a second. On the Pi, beside a 100 Hz loop, either a thread
or a process added ~10 ms to the loop's p99 (cores and memory are shared);
the thread also stalled the loop up to 56 ms through the GIL, the process
34 ms at worst. A frame that arrives while the worker is busy replaces the
one waiting, and the replaced one is counted in stats()["dropped"]: it
drops frames rather than fall behind.

Why each step is there, and the measurements behind it, are next to it. The
offline proof against tonight's drives is the commit message's.
"""

import math
import queue
import threading
import time
from collections import deque
from typing import NamedTuple, Optional

import numpy as np

from crawler.autopilot import driver as dv

# ---- the camera ---------------------------------------------------------------

# Depth comes in at 640 x 400, aligned to RGB, fx = fy = 441.25 there (tape
# verified, driver.py). The centre is the frame's own.
SRC_W, SRC_H = 640, 400
FX = dv.FX_AT_640
CX, CY = (SRC_W - 1) / 2.0, (SRC_H - 1) / 2.0
CAMERA_HEIGHT_M = dv.CAMERA_HEIGHT_MM / 1000.0

# Where the lens sits in base_link (the lidar's centre), and its yaw. Fitted
# offline by placing camera points 6-28 cm above the floor (walls, cabinet
# faces) on the lidar's own points through the SLAM poses, 278 frames over two
# drives, grid search then refine: x 0.22 m, yaw -0.25 deg. 0.22 is plausible
# against the tape (front bumper 0.27 m ahead of the lidar, the camera a few
# cm behind it). The offset sideways was not separable from zero.
CAMERA_X_M = 0.22
CAMERA_Y_M = 0.0
CAMERA_YAW = math.radians(-0.25)

# The IMU's axes to the camera's (x right, y down, z forward). IMU x is
# vertical, y lateral, z along the car (policy_imu.py); against floor planes
# fitted to 382 frames the signs are cam x = -imu y, cam y = imu x,
# cam z = imu z, plus a 1.45 degree mount offset folded in here. With it the
# IMU's up and the fitted floor normal agree to 0.99 deg median, 2.3 p90.
IMU_TO_CAMERA = np.array([[0.0244, -0.9997, 0.0],
                          [0.9997, 0.0244, -0.0067],
                          [0.0067, 0.0002, 1.0]])
# With no IMU at all: the median floor normal over 1137 raw frames.
MOUNT_UP = np.array([0.00654, -0.99993, 0.00947])
MOUNT_UP = MOUNT_UP / np.linalg.norm(MOUNT_UP)

# ---- the picture the mapper reads ----------------------------------------------

# Its own downsample, not policy_input.depth_small(). That one is the NEAREST
# depth of an 8 x 8 block, which is right for not hitting things and wrong
# for mapping: on the floor the nearest pixel of a block is its bottom row,
# and one speckle of near depth wins the whole block. This is the MEDIAN of
# the valid pixels of a BLOCK x BLOCK block, 160 x 100.
BLOCK = 4
MIN_VALID = 8               # of 16; fewer and the block says nothing

# Flying pixels and edge skirts. The OAK-D's spatial hole filling paints
# a skirt of in-between depths behind every leg, which lands exactly in the
# low band. In inverse depth a plane is linear across the image whatever its
# angle, so a pixel whose inverse depth is more than EDGE_REL off the mean of
# its two neighbours (across, or down) is not on a surface: dropped.
EDGE_REL = 0.06

# The left edge of the frame bends the floor up. Offline, on hand-checked
# empty floor, the phantom low pixels left after everything else sat in the
# first 17 of 160 columns (90th percentile), standing 5-13 cm off a floor
# that was flat at full resolution everywhere else. Those columns are not
# mapped. LEFT_MARGIN columns of the downsampled frame, 4 raw pixels each.
LEFT_MARGIN = 20

RANGE_MIN_M = 0.17          # extended disparity's measured floor
MAX_RANGE_M = 3.0           # hits and free space, the 3D map

# ---- the floor ---------------------------------------------------------------

# RANSAC over points near the IMU's floor, seeded by it, then least squares
# on the inliers. The BNO086 alone is 1.5 deg static and worse moving, which
# at 2 m is 5 cm of invented height: more than LOW_MIN_M. So the IMU is the
# seed and the fallback, never the answer. Offline, over 1137 raw frames, the
# fitted plane put the lens 0.168 m up (tape: 0.1651) with 4 mm RMS on its
# inliers.
FIT_GATE_M = 0.06
FIT_INLIER_M = 0.012
FIT_MAX_TILT_DEG = 12.0
FIT_NEAR_M = 2.2
FIT_ITERS = 48
FIT_MIN_POINTS = 30

# ---- classes -----------------------------------------------------------------

UNKNOWN, GROUND, LOW, HIGH, DROP = 0, 1, 2, 3, 4
LOW_MIN_M = 0.03
LOW_MAX_M = 0.25            # under the lidar's plane, ESTIMATED ~0.3 m
DROP_MIN_M = 0.06
# Ground keeps going up a slope this steep; a steeper rise is an obstacle.
MAX_SLOPE_DEG = 30.0
# The ground a column is measured against only moves on once the floor has
# come this far: compared row to row, a vertical face rises a centimetre at a
# time and would climb into the ground with it.
GROUND_STEP_M = 0.05

# ---- the maps ----------------------------------------------------------------

RES_M = 0.05
RES_Z_M = 0.025
Z0_M = -0.30
NZ = 40                     # -0.30 .. +0.70 m
SIZE = 400                  # cells a side: 20 m, recentred as the truck moves
RECENTRE_MARGIN_M = 4.0

# Log-odds in int8, the lidar grid's steps (0.05). Occupied needs two hits:
# one hit is +17 and occupied is above 25.
L_HIT, L_FREE, L_MIN, L_MAX, L_OCC = 17, -8, -40, 50, 25
FREE_SHORT_M = 0.075        # stop clearing this far short of the hit
FREE_STEP_M = 0.05
FREE_RAY_STRIDE = 2         # every 2nd ray across and down
FREE_FRAME_EVERY = 2        # and every 2nd frame

LOW_RANGE_M = 1.5
LOW_CELL_SCORE = 2
LOW_SCORE_MIN, LOW_SCORE_MAX = -3, 5

# ---- the pose ----------------------------------------------------------------

MIN_CONFIDENCE = 0.5
# A device timestamp further than this from the host's receipt is not one I
# believe; the frame is then placed at receipt less the measured 88 ms age.
CAPTURE_SANITY_S = 1.0
FRAME_AGE_S = 0.088
IMU_KEEP = 200
IMU_MAX_GAP_S = 0.1

# ---- publishing --------------------------------------------------------------

PUBLISH_EVERY_S = 1.0
# The worker owes a "done" for every frame it is sent. Past this it is taken
# as lost and the next frame goes anyway.
IN_FLIGHT_GIVE_UP_S = 2.0
# At most this many frames a second are mapped; the rest are dropped. The
# camera gives ~6. On the Pi each mapped frame is ~60 ms of a core alone and
# more under load, and the control loop shares those cores and their memory.
MAP_MAX_HZ = 3.0
# The worker process runs at this niceness, so the control loop wins a core.
WORKER_NICE = 10
POINTS_MAX = 40000
LOW_VIEW_HALF_M = 10.0


# =============================================================================
# one frame
# =============================================================================

def downsample(frame, block=BLOCK, min_valid=MIN_VALID):
    """(400, 640) uint16 mm -> (100, 160) float32 metres, NaN where too few
    pixels were valid. Median of the valid pixels in each block."""
    frame = np.asarray(frame)
    h, w = frame.shape[0] // block, frame.shape[1] // block
    b = frame[:h * block, :w * block].reshape(h, block, w, block)
    b = b.transpose(0, 2, 1, 3).reshape(h, w, block * block)
    s = np.sort(b, axis=2)                       # zeros (invalid) first
    nvalid = np.count_nonzero(s, axis=2)
    first = block * block - nvalid
    mid = np.minimum(first + (nvalid - 1) // 2, block * block - 1)
    med = np.take_along_axis(s, mid[..., None], axis=2)[..., 0]
    out = med.astype(np.float32) * 1e-3
    out[nvalid < min_valid] = np.nan
    return out


def drop_edges(z, rel=EDGE_REL):
    """NaN out pixels that are not on a surface: see EDGE_REL."""
    inv = 1.0 / z
    bad = np.zeros(z.shape, bool)
    with np.errstate(invalid="ignore"):
        mid = 0.5 * (inv[:, :-2] + inv[:, 2:])
        bad[:, 1:-1] |= np.abs(inv[:, 1:-1] - mid) > rel * inv[:, 1:-1]
        mid = 0.5 * (inv[:-2] + inv[2:])
        bad[1:-1] |= np.abs(inv[1:-1] - mid) > rel * inv[1:-1]
    out = z.copy()
    out[bad] = np.nan
    return out


_RAYS = {}


def rays(shape):
    """(ux, uy) per pixel of a downsampled frame: x/z and y/z at the block's
    centre in the 640 x 400 frame."""
    if shape not in _RAYS:
        h, w = shape
        sy, sx = SRC_H / h, SRC_W / w
        u = (np.arange(w) * sx + (sx - 1) / 2.0 - CX) / FX
        v = (np.arange(h) * sy + (sy - 1) / 2.0 - CY) / FX
        ux, uy = np.meshgrid(u, v)
        _RAYS[shape] = (ux.astype(np.float32), uy.astype(np.float32))
    return _RAYS[shape]


def backproject(z):
    """(H, W) metres -> (H, W, 3) camera-frame points, NaN outside range."""
    z = np.where((z >= RANGE_MIN_M) & (z <= MAX_RANGE_M + 1.0), z, np.nan)
    ux, uy = rays(z.shape)
    return np.stack([ux * z, uy * z, z], -1)


def camera_up(imu_up):
    """The IMU's up vector (its own axes) -> camera frame, unit."""
    c = IMU_TO_CAMERA @ np.asarray(imu_up, np.float64)
    n = np.linalg.norm(c)
    return c / n if n > 0 else MOUNT_UP


class Floor(NamedTuple):
    up: np.ndarray          # unit, camera frame, away from the floor
    height: float           # lens above the floor, m
    inliers: int
    fitted: bool            # False: the seed, as is


def fit_floor(P, seed_up, rng, seed_h=CAMERA_HEIGHT_M):
    """RANSAC + least squares floor plane over finite points P (N, 3)."""
    n0 = np.asarray(seed_up, np.float64)
    h = P @ n0 + seed_h
    cand = P[(np.abs(h) < FIT_GATE_M) & (P[:, 2] < FIT_NEAR_M)]
    if len(cand) < FIT_MIN_POINTS:
        return Floor(n0, seed_h, 0, False)
    idx = rng.integers(0, len(cand), size=(FIT_ITERS, 3))
    a, b, c = cand[idx[:, 0]], cand[idx[:, 1]], cand[idx[:, 2]]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(nn > 0, nn, 1.0)
    d = -np.sum(n * a, axis=1)
    flip = n @ n0 < 0
    n[flip] *= -1
    d[flip] *= -1
    ok = (n @ n0 > math.cos(math.radians(FIT_MAX_TILT_DEG))) & (d > 0.08) & (d < 0.30)
    if not ok.any():
        return Floor(n0, seed_h, 0, False)
    n, d = n[ok], d[ok]
    score = (np.abs(cand @ n.T + d) < FIT_INLIER_M).sum(0)
    k = int(np.argmax(score))
    nb, db = n[k], float(d[k])
    for _ in range(2):
        inl = cand[np.abs(cand @ nb + db) < FIT_INLIER_M]
        if len(inl) < 10:
            break
        centre = inl.mean(0)
        _, _, vt = np.linalg.svd(inl - centre, full_matrices=False)
        nb = vt[2] if vt[2] @ n0 > 0 else -vt[2]
        db = float(-nb @ centre)
    if math.acos(min(1.0, float(nb @ n0))) > math.radians(FIT_MAX_TILT_DEG) or not 0.08 < db < 0.30:
        return Floor(n0, seed_h, 0, False)
    inliers = int((np.abs(cand @ nb + db) < FIT_INLIER_M).sum())
    return Floor(nb, db, inliers, True)


def ground_axes(up):
    """Forward and left along the floor, for a camera-frame up vector."""
    f = np.array([0.0, 0.0, 1.0]) - up[2] * up
    f = f / np.linalg.norm(f)
    return f, np.cross(up, f)


def classify(P, floor):
    """Per pixel: class, height above LOCAL ground, height above the plane,
    and distance along the floor.

    Column by column from the bottom of the frame up, which is near to far
    along the floor, following the ground: a pixel is ground while it rises
    over the last ground pixel in its column by less than the slope allows
    (MAX_SLOPE_DEG, and never less than LOW_MIN_M). One that rises more is an
    obstacle, its height taken from that last ground pixel, so a ramp or a
    trail that bends away from the fitted plane stays ground and a shoe on it
    still stands up. That last ground pixel only moves on once the floor has
    come GROUND_STEP_M further, so a face, which has no run, cannot climb in.
    """
    up = floor.up
    h_abs = P @ up + floor.height
    f, l = ground_axes(up)
    r = np.hypot(P @ f, P @ l)
    H, W = h_abs.shape
    tan_s = math.tan(math.radians(MAX_SLOPE_DEG))
    cls = np.full((H, W), UNKNOWN, np.uint8)
    h_loc = np.full((H, W), np.nan, np.float32)
    rg = np.zeros(W)
    hg = np.zeros(W)
    with np.errstate(invalid="ignore"):
        for row in range(H - 1, -1, -1):
            hr, rr = h_abs[row], r[row]
            fin = np.isfinite(hr)
            dr = np.maximum(rr - rg, 0.0)
            dh = hr - hg
            tol = np.maximum(LOW_MIN_M, dr * tan_s)
            dtol = np.maximum(DROP_MIN_M, dr * tan_s)
            ground = fin & (dh < tol) & (dh > -dtol)
            drop = fin & (dh <= -dtol)
            obst = fin & ~ground & ~drop
            c = cls[row]
            c[ground] = GROUND
            c[drop] = DROP
            c[obst & (dh <= LOW_MAX_M)] = LOW
            c[obst & (dh > LOW_MAX_M)] = HIGH
            h_loc[row] = np.where(ground, 0.0, np.where(fin, dh, np.nan))
            step = ground & (dr >= GROUND_STEP_M)
            rg = np.where(step, rr, rg)
            hg = np.where(step, hr, hg)
    return cls, h_loc, h_abs, r


class FrameResult(NamedTuple):
    floor: Floor
    cls: np.ndarray         # (H, W) uint8
    X: np.ndarray           # base_link metres, forward
    Y: np.ndarray           # left
    Z: np.ndarray           # above the fitted floor
    r: np.ndarray           # along the floor from the lens


def process_frame(frame_mm, imu_up=None, rng=None):
    """One raw depth frame -> classes and base_link points. Pure."""
    rng = np.random.default_rng(0) if rng is None else rng
    z = drop_edges(downsample(frame_mm))
    z[:, :LEFT_MARGIN] = np.nan
    P = backproject(z)
    seed = camera_up(imu_up) if imu_up is not None else MOUNT_UP
    flat = P.reshape(-1, 3)
    floor = fit_floor(flat[np.isfinite(flat[:, 2])], seed, rng)
    cls, _h_loc, h_abs, r = classify(P, floor)
    f, l = ground_axes(floor.up)
    a, b = P @ f, P @ l
    c, s = math.cos(CAMERA_YAW), math.sin(CAMERA_YAW)
    X = CAMERA_X_M + c * a - s * b
    Y = CAMERA_Y_M + s * a + c * b
    return FrameResult(floor, cls, X, Y, h_abs, r)


# =============================================================================
# the maps
# =============================================================================

class VoxelMaps:
    """The 3D log-odds and the 2.5D low layer, one window, world-indexed.

    A cell is identified by floor(x / RES_M) in the map frame, the same as
    the lidar grid, so a cell here and a cell there are the same 5 cm of
    floor. The window is SIZE cells a side and recentres on the truck when it
    gets within RECENTRE_MARGIN_M of an edge; what falls off is forgotten.
    Nothing fades otherwise: a cell changes only when it is observed.
    """

    def __init__(self, x=0.0, y=0.0, size=SIZE):
        self.size = size
        self.lo = np.zeros((size, size, NZ), np.int8)
        self.low = np.zeros((size, size), np.int8)
        self.ox = int(math.floor(x / RES_M)) - size // 2
        self.oy = int(math.floor(y / RES_M)) - size // 2
        self.version = 0
        self.recentres = 0

    def cells(self, x, y):
        i = np.floor(np.asarray(x) / RES_M).astype(np.int64) - self.ox
        j = np.floor(np.asarray(y) / RES_M).astype(np.int64) - self.oy
        return i, j

    def follow(self, x, y):
        """Recentre on (x, y) if it is near an edge."""
        m = int(RECENTRE_MARGIN_M / RES_M)
        i, j = self.cells(x, y)
        if m <= i < self.size - m and m <= j < self.size - m:
            return False
        nox = int(math.floor(x / RES_M)) - self.size // 2
        noy = int(math.floor(y / RES_M)) - self.size // 2
        for name in ("lo", "low"):
            old = getattr(self, name)
            new = np.zeros_like(old)
            dx, dy = nox - self.ox, noy - self.oy
            si = slice(max(0, dx), min(self.size, self.size + dx))
            sj = slice(max(0, dy), min(self.size, self.size + dy))
            di = slice(max(0, -dx), max(0, -dx) + (si.stop - si.start))
            dj = slice(max(0, -dy), max(0, -dy) + (sj.stop - sj.start))
            if si.stop > si.start and sj.stop > sj.start:
                new[di, dj] = old[si, sj]
            setattr(self, name, new)
        self.ox, self.oy = nox, noy
        self.recentres += 1
        return True

    def _flat3(self, x, y, z):
        i, j = self.cells(x, y)
        k = np.floor((np.asarray(z) - Z0_M) / RES_Z_M).astype(np.int64)
        ok = ((i >= 0) & (i < self.size) & (j >= 0) & (j < self.size)
              & (k >= 0) & (k < NZ))
        return ((i * self.size + j) * NZ + k)[ok]

    def integrate_3d(self, origin, pts, free_from=None):
        """Hits at pts (N, 3) map metres; free space along the rays from
        origin to free_from (M, 3), if given. One update per voxel."""
        flat = self.lo.reshape(-1)
        hits = np.unique(self._flat3(pts[:, 0], pts[:, 1], pts[:, 2]))
        if free_from is not None and len(free_from):
            d = free_from - origin
            r = np.linalg.norm(d, axis=1)
            u = d / r[:, None]
            n = np.floor((np.minimum(r, MAX_RANGE_M) - FREE_SHORT_M)
                         / FREE_STEP_M).astype(np.int64)
            m = int(n.max(initial=0))
            if m > 0:
                steps = np.arange(1, m + 1)
                keep = steps[None, :] <= n[:, None]
                q = origin + (u[:, None, :] * (steps * FREE_STEP_M)[None, :, None])[keep]
                free = np.unique(self._flat3(q[:, 0], q[:, 1], q[:, 2]))
                free = free[~np.isin(free, hits, assume_unique=True)]
                v = flat[free].astype(np.int16) + L_FREE
                flat[free] = np.maximum(v, L_MIN).astype(np.int8)
        v = flat[hits].astype(np.int16) + L_HIT
        flat[hits] = np.minimum(v, L_MAX).astype(np.int8)
        self.version += 1
        return len(hits)

    def integrate_low(self, x_low, y_low, x_floor, y_floor):
        """One frame's low-obstacle and floor pixels, map metres. A cell goes
        up one if anything low landed in it this frame, else down one if
        floor did."""
        n = self.size

        def flat(x, y):
            i, j = self.cells(x, y)
            ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
            return np.unique((i * n + j)[ok])

        up = flat(x_low, y_low)
        down = flat(x_floor, y_floor)
        down = down[~np.isin(down, up, assume_unique=True)]
        low = self.low.reshape(-1)
        low[up] = np.minimum(low[up].astype(np.int16) + 1, LOW_SCORE_MAX)
        low[down] = np.maximum(low[down].astype(np.int16) - 1, LOW_SCORE_MIN)
        self.version += 1

    # ---- reading ---------------------------------------------------------------

    def occupied(self):
        return self.lo > L_OCC

    def low_cells(self):
        return self.low >= LOW_CELL_SCORE

    def occupied_points(self, limit=POINTS_MAX):
        """(N, 3) float32 map-frame centres of the occupied voxels."""
        idx = np.flatnonzero(self.lo.reshape(-1) > L_OCC)
        if len(idx) > limit:
            idx = idx[:: int(math.ceil(len(idx) / limit))]
        k = idx % NZ
        ij = idx // NZ
        i, j = ij // self.size, ij % self.size
        return np.stack([(i + self.ox + 0.5) * RES_M,
                         (j + self.oy + 0.5) * RES_M,
                         Z0_M + (k + 0.5) * RES_Z_M], 1).astype(np.float32)


class LowView(NamedTuple):
    """The low-obstacle layer round the truck, cheap to send anywhere."""
    t: float
    x0: float               # map metres of cell (0, 0)'s outer corner
    y0: float
    res: float
    codes: np.ndarray       # uint8 [nx, ny]: LOW_UNSEEN / LOW_FLOOR / LOW_OBSTACLE


LOW_UNSEEN, LOW_FLOOR, LOW_OBSTACLE = 0, 1, 2


def low_view(maps, x, y, t, half_m=LOW_VIEW_HALF_M):
    """A LowView of maps.low, half_m either side of (x, y)."""
    i, j = maps.cells(x, y)
    h = int(half_m / RES_M)
    i0, i1 = max(0, int(i) - h), min(maps.size, int(i) + h)
    j0, j1 = max(0, int(j) - h), min(maps.size, int(j) + h)
    if i1 <= i0 or j1 <= j0:
        return None
    s = maps.low[i0:i1, j0:j1]
    codes = np.full(s.shape, LOW_UNSEEN, np.uint8)
    codes[s < 0] = LOW_FLOOR
    codes[s >= LOW_CELL_SCORE] = LOW_OBSTACLE
    return LowView(t, (i0 + maps.ox) * RES_M, (j0 + maps.oy) * RES_M, RES_M, codes)


def to_map(X, Y, pose):
    """base_link (X, Y) -> map frame, by a pose (x, y, yaw)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * X - s * Y, y + s * X + c * Y


def integrate(maps, res, pose, free=True):
    """One FrameResult into the maps at a pose (x, y, yaw). Returns hits."""
    fin = np.isfinite(res.Z)
    X, Y, Z = res.X[fin], res.Y[fin], res.Z[fin]
    r = res.r[fin]
    near = r <= MAX_RANGE_M
    wx, wy = to_map(X[near], Y[near], pose)
    pts = np.stack([wx, wy, Z[near]], 1)
    origin = None
    rays_to = None
    if free:
        ox, oy = to_map(CAMERA_X_M, CAMERA_Y_M, pose)
        origin = np.array([ox, oy, res.floor.height])
        sub = np.zeros(res.Z.shape, bool)
        sub[::FREE_RAY_STRIDE, ::FREE_RAY_STRIDE] = True
        sel = (sub & np.isfinite(res.Z) & (res.r <= MAX_RANGE_M))
        rx, ry = to_map(res.X[sel], res.Y[sel], pose)
        rays_to = np.stack([rx, ry, res.Z[sel]], 1)
    maps.follow(pose[0], pose[1])
    hits = maps.integrate_3d(origin if origin is not None else np.zeros(3), pts, rays_to)
    # The low layer: inside LOW_RANGE_M only. DROP is never written: a
    # reflective floor reads as a hole, and only the lidar could say otherwise.
    cls = res.cls
    lowm = (cls == LOW) & (res.r <= LOW_RANGE_M)
    flo = (cls == GROUND) & (res.r <= LOW_RANGE_M)
    lx, ly = to_map(res.X[lowm], res.Y[lowm], pose)
    fx, fy = to_map(res.X[flo], res.Y[flo], pose)
    maps.integrate_low(lx, ly, fx, fy)
    return hits


# =============================================================================
# the live half
# =============================================================================

MODE_THREAD = "thread"
MODE_PROCESS = "process"
MODES = (MODE_THREAD, MODE_PROCESS)
OUTBOX = 8


class _Maps:
    """The worker's half: the maps, and what to send back for each frame."""

    def __init__(self, publish_every_s=PUBLISH_EVERY_S):
        self.maps = None
        self.rng = np.random.default_rng(0)
        self.mapped = 0
        self.publish_every_s = publish_every_s
        self.next_publish = -math.inf

    def frame(self, t, frame, pose, up):
        """One placed frame -> the messages for the main process."""
        t0 = time.perf_counter()
        res = process_frame(frame, up, self.rng)
        if self.maps is None:
            self.maps = VoxelMaps(pose[0], pose[1])
        free = (self.mapped % FREE_FRAME_EVERY) == 0
        integrate(self.maps, res, pose, free=free)
        self.mapped += 1
        out = [("done", (time.perf_counter() - t0) * 1e3, bool(res.floor.fitted))]
        if t >= self.next_publish:
            self.next_publish = t + self.publish_every_s
            out.append(("pub", self.maps.occupied_points(),
                        low_view(self.maps, pose[0], pose[1], t)))
        return out


def _work(inbox, outbox, publish_every_s, nice=0):
    """The worker, thread or process: placed frames in, results out. None in
    the inbox ends it. `nice` only for a process: a thread's would be the
    whole service's."""
    if nice:
        try:
            import os
            os.nice(nice)
        except Exception:
            pass
    m = _Maps(publish_every_s)
    while True:
        item = inbox.get()
        if item is None:
            return
        try:
            out = m.frame(*item)
        except Exception as e:      # a bad frame costs that frame
            out = [("error", repr(e))]
        for msg in out:
            try:
                # A "done" must arrive or the feeder waits IN_FLIGHT_GIVE_UP_S;
                # a "pub" can be lost.
                outbox.put(msg, timeout=1.0 if msg[0] != "pub" else 0.0)
            except Exception:
                pass


class VoxelMapper:
    """Depth frames in, the two maps out, on a worker of its own.

    on_depth() and on_imu() are called from the camera's reader thread and
    return at once. Nothing here commands the truck, and every public method
    returns at once and never raises.
    """

    def __init__(self, slam, tele=None, logger=None, clock=time.monotonic,
                 mode=MODE_PROCESS):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self.slam = slam
        self.tele = tele
        self.logger = logger
        self.mode = mode
        self._clock = clock
        self._slot = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._feeder = None
        self._pump = None
        self._worker = None
        self._inbox = None
        self._outbox = None
        self._in_flight_at = None
        self._sent_at = None
        self._imu = deque(maxlen=IMU_KEEP)
        self._said_error = False
        # counts since boot
        self.frames = 0          # handed to on_depth
        self.dropped = 0         # replaced in the slot before the worker took them
        self.sent = 0            # placed and handed to the worker
        self.mapped = 0          # of those, in the maps
        self.skipped_pose = 0    # no pose worth placing them by
        self.errors = 0
        self.unfitted = 0        # the floor fell back to the IMU's
        self.lost = 0            # sent, and no word back in IN_FLIGHT_GIVE_UP_S
        self.last_ms = 0.0
        self.ms = deque(maxlen=200)

    # ---- lifecycle -------------------------------------------------------------

    def start(self):
        if self._worker is not None:
            return
        self._stop.clear()
        args = None
        if self.mode == MODE_PROCESS:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            self._inbox = ctx.Queue(maxsize=1)
            self._outbox = ctx.Queue(maxsize=OUTBOX)
            args = (self._inbox, self._outbox, PUBLISH_EVERY_S, WORKER_NICE)
            self._worker = ctx.Process(target=_work, args=args, name="voxel",
                                       daemon=True)
        else:
            self._inbox = queue.Queue(maxsize=1)
            self._outbox = queue.Queue(maxsize=OUTBOX)
            args = (self._inbox, self._outbox, PUBLISH_EVERY_S)
            self._worker = threading.Thread(target=_work, args=args,
                                            name="voxel", daemon=True)
        self._worker.start()
        self._feeder = threading.Thread(target=self._feed_loop,
                                        name="voxel-feed", daemon=True)
        self._feeder.start()
        self._pump = threading.Thread(target=self._pump_loop,
                                      name="voxel-pump", daemon=True)
        self._pump.start()

    def close(self):
        """Stop the worker, the feeder and the pump. Never raises."""
        self._stop.set()
        self._wake.set()
        worker, self._worker = self._worker, None
        if worker is None:
            return
        try:
            while True:
                self._inbox.get_nowait()
        except Exception:
            pass
        try:
            self._inbox.put_nowait(None)
        except Exception:
            pass
        try:
            worker.join(timeout=1.0)
            if self.mode == MODE_PROCESS and worker.is_alive():
                worker.terminate()
                worker.join(timeout=1.0)
        except Exception:
            pass
        for t in (self._feeder, self._pump):
            if t is not None:
                t.join(timeout=1.0)
        self._feeder = self._pump = None
        if self.mode == MODE_PROCESS:
            for q in (self._inbox, self._outbox):
                try:
                    q.cancel_join_thread()
                    q.close()
                except Exception:
                    pass

    # ---- the producers ---------------------------------------------------------

    def on_depth(self, now, frame, capture_t=None):
        """One raw depth frame from the reader thread. Never blocks."""
        if frame is None:
            return
        t = capture_t
        if t is None or abs(t - now) > CAPTURE_SANITY_S:
            t = now - FRAME_AGE_S
        with self._lock:
            if self._slot is not None:
                self.dropped += 1
            self._slot = (t, frame)
            self.frames += 1
        self._wake.set()

    def on_imu(self, now, sample):
        """One IMU sample from the reader thread: its up vector, if it has one."""
        try:
            up = sample.get("up") if isinstance(sample, dict) else None
        except Exception:
            up = None
        if up is not None:
            self._imu.append((now, tuple(up)))

    # ---- the main process's half -----------------------------------------------

    def imu_up_at(self, t):
        """The IMU up vector nearest t, or None if none is within IMU_MAX_GAP_S."""
        best = None
        for ts, up in list(self._imu):
            d = abs(ts - t)
            if best is None or d < best[0]:
                best = (d, up)
        if best is None or best[0] > IMU_MAX_GAP_S:
            return None
        return best[1]

    def busy(self):
        """True while the worker has a frame and has not answered for it."""
        at = self._in_flight_at
        if at is None:
            return False
        if self._clock() - at > IN_FLIGHT_GIVE_UP_S:
            self._in_flight_at = None
            self.lost += 1
            return False
        return True

    def too_soon(self):
        """True while the last frame sent is under 1 / MAP_MAX_HZ old."""
        at = self._sent_at
        return at is not None and self._clock() - at < 1.0 / MAP_MAX_HZ

    def step(self, t, frame):
        """Place one frame and hand it to the worker. True if it was sent."""
        pose = self.slam.pose_at(t) if self.slam is not None else None
        if (pose is None or pose.confidence < MIN_CONFIDENCE
                or pose.state != "tracking"):
            self.skipped_pose += 1
            return False
        if self._inbox is None:
            return False
        item = (t, frame, (pose.x, pose.y, pose.yaw), self.imu_up_at(t))
        self._in_flight_at = self._clock()
        try:
            self._inbox.put_nowait(item)
        except Exception:
            self._in_flight_at = None
            self.dropped += 1
            return False
        self.sent += 1
        self._sent_at = self._clock()
        return True

    def _feed_loop(self):
        while not self._stop.is_set():
            self._wake.wait(0.5)
            self._wake.clear()
            if self._stop.is_set() or self.busy():
                continue
            if self.too_soon():
                self._stop.wait(0.02)
                self._wake.set()
                continue
            with self._lock:
                item, self._slot = self._slot, None
            if item is None:
                continue
            try:
                self.step(*item)
            except Exception:
                self.errors += 1

    def _pump_loop(self):
        while not self._stop.is_set():
            try:
                msg = self._outbox.get(timeout=0.2)
            except queue.Empty:
                continue
            except Exception:
                return
            try:
                self._take(msg)
            except Exception:
                pass

    def _take(self, msg):
        kind = msg[0]
        if kind == "done":
            self.mapped += 1
            self.last_ms = msg[1]
            self.ms.append(msg[1])
            if not msg[2]:
                self.unfitted += 1
            self._in_flight_at = None
            self._wake.set()
        elif kind == "error":
            self.errors += 1
            self._in_flight_at = None
            self._wake.set()
            if not self._said_error and self.logger is not None:
                self._said_error = True
                try:
                    self.logger.log("warning", "VOXEL", f"A frame failed: {msg[1]}")
                except Exception:
                    pass
        elif kind == "pub":
            tele = self.tele
            if tele is None:
                return
            now = self._clock()
            if tele.wants("voxel_points", now):
                tele.offer("voxel_points", msg[1], now)
            if msg[2] is not None and tele.wants("voxel_low", now):
                tele.offer("voxel_low", msg[2], now)

    # ---- the readers ---------------------------------------------------------

    def stats(self) -> dict:
        ms = np.array(self.ms) if self.ms else np.zeros(1)
        alive = self._worker is not None and self._worker.is_alive()
        return {"mode": self.mode, "frames": self.frames, "dropped": self.dropped,
                "sent": self.sent, "mapped": self.mapped,
                "skipped_pose": self.skipped_pose, "unfitted": self.unfitted,
                "errors": self.errors, "lost": self.lost,
                "ms_p50": float(np.median(ms)), "ms_p95": float(np.percentile(ms, 95)),
                "alive": bool(alive)}
