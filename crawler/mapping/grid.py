"""The occupancy grid the truck builds as it drives: log-odds, 5 cm cells.

One square of SIZE x SIZE cells, 51.2 m on a side at the default, centred on
wherever the truck was at boot. It never forgets on a clock: a cell is only
ever changed by an observation. A return lands in a cell and makes it more
occupied; a ray that passes through a cell on its way to a return makes it
more free. The log-odds are clamped at both ends, so a wall that was looked
at for ten minutes is no harder to clear than one looked at for ten seconds:
see L_MAX for the number and why.

Storage, 5 bytes a cell (5.2 MB at 1024 x 1024):

  lo     int8, log-odds in steps of LO_STEP (0.05). 0 is unknown.
  fx,fy  uint8, the mean position of the returns that landed in the cell,
         as a fraction of the cell (0.2 mm steps). The scan matcher matches
         against these, not the cell centres: a 5 cm cell centre is up to
         3.5 cm off the wall, a hit mean is a few millimetres off it.
  cnt    uint8, how many returns the mean is over, capped at MEAN_CAP so it
         keeps following a wall that settles. 0 means no mean yet.

Coordinates: the map frame is metres, x and y, the truck's pose at boot is
the origin facing +x. A cell is identified by its world index floor(x / RES);
the array holds the window of world cells starting at (ox, oy). Integer
origins, so scrolling never accumulates a rounding error.

Scrolling: when the truck gets within SCROLL_MARGIN_M of an edge the window
shifts to put it back in the middle, and what falls off the far side is gone.
That is what lets it drive a trail for kilometres in fixed memory, and the
cost is that a place left more than about 25 m behind is forgotten. Loop
closure would want those kept; it is not in this round.

Pure numpy, no scipy, so nothing here needs anything the car does not have.
"""

import numpy as np

RES = 0.05
SIZE = 1024

# Log-odds in int8 steps. 0.85 and -0.4 a return are the prototype's, which
# was measured on the four drives: a cell needs one return to go from unknown
# to occupied and a few clear passes to come back.
LO_STEP = 0.05
L_HIT = 17          # +0.85
L_FREE = -8         # -0.40
L_MIN = -40         # -2.0: three returns turn a long-free cell occupied
L_OCC = 10          # +0.5: above this a cell is a wall to match against
# The occupied clamp. A wall observed for any length of time sits at L_MAX,
# so the number of clear passes it takes to flip it is (L_MAX - L_OCC) / 8,
# whatever its history: 8 at 70 (3.5), 3 at 30 (1.5). A cell is crossed at
# most once a scan, so that is 0.8 s against 0.3 s of looking through where
# a thing was, at 10 Hz.
#
# Measured, tools/slam_replay.py --clamp 1.0 / 1.5 / 2.0 / 3.5 on the four
# drives. The higher clamp held walls better on long revisits (300 s or more
# apart): 035323 median 0.94 cm and max 2.6 at 3.5 against 1.09 and 3.7 at
# 1.5; 045557 1.04 and 3.3 against 1.42 and 4.4. Every other metric moved by
# noise. A 30 cm box drawn into the 045557 scans for its first 60 s and then
# taken away was gone from the map 51.2 s after at 1.0, 51.8 s at 3.5: how
# long a removed thing lingers is how long until the truck looks through
# where it was, not the clamp. So 3.5.
L_MAX = 70          # +3.5

# A mean over at most this many returns, then an exponential average with
# that weight: a wall a few millimetres off its first sighting moves.
MEAN_CAP = 10

# Returns further than this are matched against but not written. The
# sensor's noise and the angular gap between points both grow with range,
# and at 8 m the points are 10 cm apart.
MAP_MAX_R = 8.0

# The truck is kept at least this far from the edge of the window.
SCROLL_MARGIN_M = 12.0


class Grid:
    """The map. Every method is a few numpy calls; nothing allocates SIZE^2
    per scan except scroll(), which happens once every ~14 m of travel at
    most."""

    def __init__(self, size=SIZE, res=RES, l_max=L_MAX, x0=0.0, y0=0.0):
        self.size = int(size)
        self.res = float(res)
        self.l_max = int(l_max)
        n = self.size
        self.lo = np.zeros((n, n), np.int8)
        self.fx = np.zeros((n, n), np.uint8)
        self.fy = np.zeros((n, n), np.uint8)
        self.cnt = np.zeros((n, n), np.uint8)
        self._mark = np.zeros(n * n, bool)
        self.ox = int(np.floor(x0 / self.res)) - n // 2
        self.oy = int(np.floor(y0 / self.res)) - n // 2
        self.scrolls = 0
        # Bumped on every write, so a reader can tell a grid that changed.
        self.version = 0

    # ---- frames --------------------------------------------------------

    @property
    def origin(self):
        """The map-frame metres of cell (0, 0)'s outer corner."""
        return self.ox * self.res, self.oy * self.res

    def world_to_cell(self, x, y):
        """Map metres -> array indices (may be outside the window)."""
        ix = np.floor(np.asarray(x) / self.res).astype(np.int64) - self.ox
        iy = np.floor(np.asarray(y) / self.res).astype(np.int64) - self.oy
        return ix, iy

    def cell_to_world(self, ix, iy):
        """Array indices -> map metres of the cell centres."""
        return ((np.asarray(ix) + self.ox + 0.5) * self.res,
                (np.asarray(iy) + self.oy + 0.5) * self.res)

    def nbytes(self):
        return sum(a.nbytes for a in (self.lo, self.fx, self.fy, self.cnt))

    # ---- reading -------------------------------------------------------

    def occupied(self):
        """bool (SIZE, SIZE): the cells a matcher or a planner treats as solid."""
        return (self.lo > L_OCC) & (self.cnt > 0)

    def known(self):
        return self.lo != 0

    def log_odds(self):
        """float32 (SIZE, SIZE) log-odds, a copy."""
        return self.lo.astype(np.float32) * LO_STEP

    def local_points(self, x, y, radius):
        """The occupied cells' hit means within a square of `radius` around
        (x, y), map metres, float64 (n, 2). What the scan matcher matches."""
        n, res = self.size, self.res
        ix, iy = self.world_to_cell(x, y)
        w = int(radius / res)
        a0, a1 = max(int(ix) - w, 0), min(int(ix) + w, n)
        b0, b1 = max(int(iy) - w, 0), min(int(iy) + w, n)
        if a0 >= a1 or b0 >= b1:
            return np.zeros((0, 2))
        occ = ((self.lo[a0:a1, b0:b1] > L_OCC)
               & (self.cnt[a0:a1, b0:b1] > 0))
        ii, jj = np.nonzero(occ)
        fx = self.fx[a0:a1, b0:b1][ii, jj]
        fy = self.fy[a0:a1, b0:b1][ii, jj]
        px = (ii + a0 + self.ox + (fx + 0.5) / 256.0) * res
        py = (jj + b0 + self.oy + (fy + 0.5) / 256.0) * res
        return np.stack([px, py], 1)

    def bounds(self):
        """(i0, i1, j0, j1) of the cells ever observed, or None."""
        k = self.lo != 0
        rows = np.flatnonzero(k.any(1))
        if not rows.size:
            return None
        cols = np.flatnonzero(k.any(0))
        return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1

    # ---- writing -------------------------------------------------------

    def scroll_to(self, x, y, margin=SCROLL_MARGIN_M):
        """Shift the window if (x, y) is within `margin` of its edge. True if
        it moved."""
        n = self.size
        ix, iy = self.world_to_cell(x, y)
        m = int(margin / self.res)
        if m <= ix < n - m and m <= iy < n - m:
            return False
        sx = int(ix) - n // 2
        sy = int(iy) - n // 2
        for name in ("lo", "fx", "fy", "cnt"):
            a = getattr(self, name)
            b = np.zeros_like(a)
            src_x = slice(max(sx, 0), min(n + sx, n))
            dst_x = slice(max(-sx, 0), min(n - sx, n))
            src_y = slice(max(sy, 0), min(n + sy, n))
            dst_y = slice(max(-sy, 0), min(n - sy, n))
            b[dst_x, dst_y] = a[src_x, src_y]
            setattr(self, name, b)
        self.ox += sx
        self.oy += sy
        self.scrolls += 1
        self.version += 1
        return True

    def integrate(self, sx, sy, wx, wy):
        """One scan's returns (wx, wy), map metres, seen from (sx, sy).

        Each cell is changed at most once a scan: a cell a dozen rays cross
        is one free observation, not twelve, and a cell holding a return is
        never also cleared by a ray that grazes it on the way to the next
        point along the wall. The second is what keeps a wall seen end-on
        from eroding itself.
        """
        n, res = self.size, self.res
        wx = np.asarray(wx, np.float64)
        wy = np.asarray(wy, np.float64)
        r = np.hypot(wx - sx, wy - sy)
        keep = r < MAP_MAX_R
        wx, wy, r = wx[keep], wy[keep], r[keep]
        if not len(r):
            return
        gx = np.floor(wx / res).astype(np.int64)
        gy = np.floor(wy / res).astype(np.int64)
        hx, hy = gx - self.ox, gy - self.oy
        inside = (hx >= 0) & (hx < n) & (hy >= 0) & (hy < n)
        hi = (hx * n + hy)[inside]
        mark = self._mark
        mark[hi] = True

        lo = self.lo.reshape(-1)
        cnt = self.cnt.reshape(-1)
        # Free cells: one sample every cell length from the sensor to two
        # cells short of the return.
        nstep = np.maximum((r / res).astype(np.int64) - 2, 0)
        tot = int(nstep.sum())
        if tot:
            ray = np.repeat(np.arange(len(r)), nstep)
            k = np.arange(tot) - np.repeat(np.cumsum(nstep) - nstep, nstep)
            f = (k + 0.5) * res / r[ray]
            fx = np.floor((sx + f * (wx[ray] - sx)) / res).astype(np.int64) - self.ox
            fy = np.floor((sy + f * (wy[ray] - sy)) / res).astype(np.int64) - self.oy
            ok = (fx >= 0) & (fx < n) & (fy >= 0) & (fy < n)
            fi = (fx * n + fy)[ok]
            fi = fi[~mark[fi]]
            # Repeated indices read the same old value and write the same
            # new one, so a cell crossed by many rays is cleared once.
            old = lo[fi].astype(np.int16)
            new = np.maximum(old + L_FREE, L_MIN)
            lo[fi] = new.astype(np.int8)
            # A cell that stops being a wall starts its mean again the next
            # time something lands in it: what is there now is not what was.
            cnt[fi[new <= L_OCC]] = 0
        mark[hi] = False

        if not len(hi):
            self.version += 1
            return
        # Returns: one log-odds step per cell, and the hit mean.
        px = (wx - gx * res)[inside] / res     # 0..1 within the cell
        py = (wy - gy * res)[inside] / res
        u, inv, c = np.unique(hi, return_inverse=True, return_counts=True)
        mx = np.bincount(inv, px) / c
        my = np.bincount(inv, py) / c
        old = lo[u].astype(np.int16)
        lo[u] = np.minimum(old + L_HIT, self.l_max).astype(np.int8)
        c0 = cnt[u].astype(np.float64)
        nn = np.minimum(c0 + 1, MEAN_CAP)
        fxf = self.fx.reshape(-1)
        fyf = self.fy.reshape(-1)
        ox_ = (fxf[u] + 0.5) / 256.0
        oy_ = (fyf[u] + 0.5) / 256.0
        fresh = c0 == 0
        ex = np.where(fresh, mx, ox_ + (mx - ox_) / nn)
        ey = np.where(fresh, my, oy_ + (my - oy_) / nn)
        fxf[u] = np.clip(ex * 256.0, 0, 255).astype(np.uint8)
        fyf[u] = np.clip(ey * 256.0, 0, 255).astype(np.uint8)
        cnt[u] = nn.astype(np.uint8)
        self.version += 1

    # ---- the picture ---------------------------------------------------

    def codes(self, step=1, bounds=None):
        """The map as uint8 codes, UNKNOWN / FREE / OCCUPIED, over `bounds`
        (default: everything observed), `step` cells to a pixel.

        A pixel is occupied if any cell in it is, free if any is and none is
        occupied. Returns (codes[nx, ny], x0, y0) with (x0, y0) the map
        metres of pixel (0, 0)'s outer corner, or None for an empty map.
        """
        b = self.bounds() if bounds is None else bounds
        if b is None:
            return None
        i0, i1, j0, j1 = b
        step = max(int(step), 1)
        i1 = i0 + -(-(i1 - i0) // step) * step
        j1 = j0 + -(-(j1 - j0) // step) * step
        lo = np.zeros((i1 - i0, j1 - j0), np.int8)
        cn = np.zeros_like(lo, dtype=np.uint8)
        si = slice(i0, min(i1, self.size))
        sj = slice(j0, min(j1, self.size))
        lo[:si.stop - i0, :sj.stop - j0] = self.lo[si, sj]
        cn[:si.stop - i0, :sj.stop - j0] = self.cnt[si, sj]
        occ = (lo > L_OCC) & (cn > 0)
        free = lo < 0
        nx, ny = (i1 - i0) // step, (j1 - j0) // step
        occ = occ.reshape(nx, step, ny, step).any((1, 3))
        free = free.reshape(nx, step, ny, step).any((1, 3))
        out = np.full((nx, ny), UNKNOWN, np.uint8)
        out[free] = FREE
        out[occ] = OCCUPIED
        return out, (i0 + self.ox) * self.res, (j0 + self.oy) * self.res


UNKNOWN = 0
FREE = 1
OCCUPIED = 2
