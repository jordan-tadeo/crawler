"""The policy's view of the map: an egocentric crop of the SLAM grid.

One builder, used by the dataset and the car, as policy_lidar.py is for the
scan. The SLAM worker calls EgoMap.after_step() right after every Slam.step(),
live (runner.py) and over a recording (tools/map_crops.py), so the crop a
model trains on and the crop it drives on come out of the same function off
the same grid.

The crop
--------
SIZE x SIZE pixels of RES metres, truck-centred and heading-up: row r is
(r + 0.5 - SIZE / 2) * RES metres ahead of the truck (row 0 is the furthest
behind, row SIZE - 1 the furthest ahead), column c is the same distance to
its left. 64 at 10 cm is 6.4 m square, 3.2 m each way. A pixel covers
(RES / grid.RES)^2 grid cells and is occupied if any of them is, free if
any is and none is occupied, unknown otherwise: a 5 cm wall is never lost
to the downsample.

  codes   uint8 (SIZE, SIZE): grid.UNKNOWN / FREE / OCCUPIED
  trail   uint8 (SIZE, SIZE): where the truck has been in the last TRAIL_S,
          255 now fading to 1 at TRAIL_S, 0 never (or longer ago)
  present whether the pose this crop is drawn at is one to trust: matched,
          tracking or tilted (see trusted). A crop that is not present goes
          in as no map.

The input
---------
(1, CHANNELS, SIZE, SIZE) float32: occupied, free, unknown, trail / 255,
and a plane of present. No map -- no --slam, a crop older than STALE_S, a
pose that is not trusted -- is every value 0, present included.

Mirroring: column c and column SIZE - 1 - c swap; the trainer's flip
augmentation flips the last axis, and nothing else moves.

numpy only.
"""

from typing import NamedTuple

import numpy as np

from crawler.mapping import grid as gr

SIZE = 64
RES = 0.10
TRAIL_S = 60.0
# The newest crop is used while it is this fresh: three scans.
STALE_S = 0.3
CHANNELS = 5
LAYOUT = f"ego{SIZE}_{int(round(RES * 100))}cm_trail{int(TRAIL_S)}"
PRESENT = 4


class Crop(NamedTuple):
    t: float              # the scan time it is as of (Result.t)
    present: bool
    codes: np.ndarray     # uint8 (SIZE, SIZE)
    trail: np.ndarray     # uint8 (SIZE, SIZE)


# The states a matched pose is good in. Tilted is the map's write gate, not
# the pose's: the scan still matched, only it is not written, so the crop is
# the map as last written at a pose that keeps updating. Leaving tilted out
# parked the car on "map lost" whenever the chassis sat rolled past the
# 5.5 degree gate after a knock: 9 of the 11 stops on 2026-10-29 17:46Z,
# rolled 6 to 8 degrees, with 659 of 668 tilted scans matched (median
# inliers 0.89).
TRUSTED_STATES = ("tracking", "tilted")


def trusted(res) -> bool:
    """A Result whose pose a crop may be drawn at: matched, and tracking or
    tilted. Lost, rolled and no_points never are."""
    return res is not None and res.state in TRUSTED_STATES and bool(res.matched)


class EgoMap:
    """Keeps the trail and draws crops. One per Slam, on the Slam's thread."""

    def __init__(self, size=SIZE, res=RES, trail_s=TRAIL_S):
        self.size, self.res, self.trail_s = int(size), float(res), float(trail_s)
        sub = max(int(round(self.res / gr.RES)), 1)
        self.sub = sub
        k = (np.arange(self.size * sub) + 0.5) / sub - self.size / 2.0
        u = k * self.res
        U, V = np.meshgrid(u, u, indexing="ij")
        self._u = U.ravel()
        self._v = V.ravel()
        self._tx, self._ty, self._tt = [], [], []

    def after_step(self, grid, res) -> Crop:
        """The Result the step just made, and the grid it wrote: the crop."""
        if res.state not in ("rolled", "no_points"):
            self._tx.append(res.x)
            self._ty.append(res.y)
            self._tt.append(res.t)
            # forget what is past the window, in one cut now and then
            if len(self._tt) > 64 and res.t - self._tt[0] > self.trail_s + 5.0:
                k = int(np.searchsorted(np.asarray(self._tt), res.t - self.trail_s))
                del self._tx[:k], self._ty[:k], self._tt[:k]
        codes, trail = self.render(grid, res.x, res.y, res.yaw, res.t)
        return Crop(float(res.t), trusted(res), codes, trail)

    def render(self, grid, x, y, yaw, t):
        S, sub = self.size, self.sub
        c, s = np.cos(yaw), np.sin(yaw)
        wx = x + c * self._u - s * self._v
        wy = y + s * self._u + c * self._v
        n = grid.size
        ix = np.floor(wx / grid.res).astype(np.int64) - grid.ox
        iy = np.floor(wy / grid.res).astype(np.int64) - grid.oy
        ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        flat = np.where(ok, ix * n + iy, 0)
        lo = np.where(ok, grid.lo.reshape(-1)[flat], 0)
        cn = np.where(ok, grid.cnt.reshape(-1)[flat], 0)
        occ = ((lo > gr.L_OCC) & (cn > 0)).reshape(S, sub, S, sub).any((1, 3))
        free = (lo < 0).reshape(S, sub, S, sub).any((1, 3))
        codes = np.full((S, S), gr.UNKNOWN, np.uint8)
        codes[free] = gr.FREE
        codes[occ] = gr.OCCUPIED
        trail = np.zeros((S, S), np.uint8)
        if self._tt:
            tt = t - np.asarray(self._tt)
            m = (tt < self.trail_s) & (tt >= -1e-6)
            if m.any():
                dx = np.asarray(self._tx)[m] - x
                dy = np.asarray(self._ty)[m] - y
                u = c * dx + s * dy
                v = -s * dx + c * dy
                r = np.floor(u / self.res + S / 2).astype(np.int64)
                q = np.floor(v / self.res + S / 2).astype(np.int64)
                k = (r >= 0) & (r < S) & (q >= 0) & (q < S)
                val = np.clip(255.0 * (1.0 - tt[m][k] / self.trail_s), 1, 255).astype(np.uint8)
                np.maximum.at(trail, (r[k], q[k]), val)
        return codes, trail


def encode_many(codes, trail, present):
    """(N, S, S) codes and trail, (N,) present -> (N, CHANNELS, S, S) float32.
    A row that is not present is all zeros."""
    codes = np.asarray(codes)
    n = len(codes)
    out = np.zeros((n, CHANNELS) + codes.shape[1:], np.float32)
    p = np.asarray(present, dtype=bool)
    if p.any():
        c = codes[p]
        out[p, 0] = c == gr.OCCUPIED
        out[p, 1] = c == gr.FREE
        out[p, 2] = c == gr.UNKNOWN
        out[p, 3] = np.asarray(trail)[p] / 255.0
        out[p, PRESENT] = 1.0
    return out


def missing(size=SIZE):
    return np.zeros((1, CHANNELS, size, size), np.float32)


def encode(crop, now=None, stale_s=STALE_S):
    """The car's input off the runner's newest Crop (or None): no map unless
    it is present and fresh. `now` on the Result.t clock."""
    if crop is None or not crop.present:
        return missing()
    if now is not None and not (0.0 <= now - crop.t <= stale_s):
        return missing()
    return encode_many(crop.codes[None], crop.trail[None], [True])


def pair_ticks(log_t, tick_t, stale_s=STALE_S):
    """A recording's crops (tools/map_crops.py, one per scan, by the scan's
    log time) against dataset tick times: the index of the newest crop at or
    before each tick no more than stale_s older, else -1. encode()'s rule
    for the runner's newest crop, over a file."""
    log_t = np.asarray(log_t, float)
    tick_t = np.asarray(tick_t, float)
    j = np.searchsorted(log_t, tick_t, side="right") - 1
    ok = j >= 0
    ok[ok] &= (tick_t[ok] - log_t[j[ok]]) <= stale_s
    return np.where(ok, j, -1)


def mirror(m):
    """The flip augmentation's mirror of an encoded map: columns swap."""
    return m[..., ::-1]


def meta():
    return {"map_layout": LAYOUT, "map_size": SIZE, "map_res_m": RES,
            "map_trail_s": TRAIL_S, "map_stale_s": STALE_S}
