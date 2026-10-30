"""Finding the truck again on a map it has lost its place on.

A lost Slam (slam.py) knows the map and roughly where it was: the pose it
froze at. Picked up, knocked over or set down somewhere else, the truck can be
metres off that and facing anywhere, which is out of reach of ICP from the
frozen pose (it converges from a few tens of centimetres and degrees). So the
search is in two steps, a correlative one over the whole window and ICP on the
best few:

  1. The map's hit means within SEARCH_R_M + FIELD_PAD_M of the centre are
     drawn into a COARSE_M raster once per centre, and each cell gets a score
     off its distance to the nearest of them, exp(-d^2 / 2 SIGMA_M^2). A
     candidate pose's score is the mean over a fixed subsample of the scan's
     points (SCORE_POINTS) at that pose: lookups only, no tree.
  2. Candidates are every COARSE_M of x and y within SEARCH_R_M of the centre
     at every YAW_STEP_DEG of heading; YAWS_PER_SCAN headings a scan, in
     order outwards from the heading the IMU says, so a truck set down facing
     the way it was is found on the first scan and the whole circle is
     covered every few. The best TOP_K are handed back for ICP.

Nothing here decides anything: slam.py runs ICP on the candidates, and only
once the whole circle has been searched does it pick: the best fit, if it is
good enough and no different pose (DISTINCT_M / DISTINCT_DEG away) fits
within AMBIGUOUS of it, and then it wants ICP from it to fit on the next scan
too. A room that looks the same turned round (a bare rectangle does) is no
answer at all: the synthetic room's 180 degree twin fitted at 0.79 and was
taken, before this, ahead of the true pose two scans of headings later.
"""

import math

import numpy as np

# How far from the centre the truck may be, metres.
SEARCH_R_M = 3.0
# The scan's points reach this far past the search window.
FIELD_PAD_M = 8.0
# The raster's cell and the candidate step, metres.
COARSE_M = 0.2
# The score's fall-off from a wall: half a step.
SIGMA_M = 0.12
YAW_STEP_DEG = 4.0
YAWS_PER_SCAN = 30
SCORE_POINTS = 48
TOP_K = 4
# Two fits are different answers if they are this far apart.
DISTINCT_M = 0.5
DISTINCT_DEG = 20.0
# The best fit must beat every different answer by this much inlier fraction.
AMBIGUOUS = 0.08
# A twin is settled by the gyro: the one answer within this of the heading
# the gyro carried the truck to while it was lost.
PRIOR_DEG = 30.0
# Candidates kept from one heading, so one good heading cannot fill TOP_K with
# its neighbours a step apart.
PER_YAW = 1


class Field:
    """The score raster round one centre of one grid."""

    def __init__(self, grid, cx, cy, r_m=SEARCH_R_M, pad_m=FIELD_PAD_M,
                 res=COARSE_M, sigma=SIGMA_M):
        from scipy.ndimage import distance_transform_edt
        self.cx, self.cy, self.r_m, self.res = float(cx), float(cy), float(r_m), float(res)
        half = r_m + pad_m
        self.x0, self.y0 = cx - half, cy - half
        n = int(math.ceil(2 * half / res)) + 1
        self.n = n
        pts = grid.local_points(cx, cy, half)
        self.points = len(pts)
        occ = np.zeros((n, n), bool)
        if len(pts):
            i = np.floor((pts[:, 0] - self.x0) / res).astype(np.int64)
            j = np.floor((pts[:, 1] - self.y0) / res).astype(np.int64)
            ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
            occ[i[ok], j[ok]] = True
        if occ.any():
            d = distance_transform_edt(~occ) * res
            self.score = np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)
        else:
            self.score = np.zeros((n, n), np.float32)
        # Candidate offsets, a disc of SEARCH_R_M at the raster's step.
        k = int(r_m / res)
        u = np.arange(-k, k + 1) * res
        X, Y = np.meshgrid(u, u, indexing="ij")
        disc = X ** 2 + Y ** 2 <= r_m ** 2 + 1e-9
        self.offsets = np.stack([X[disc], Y[disc]], 1)
        # The offsets are whole cells of the raster, so a point's cell at
        # every offset is its cell at none plus the offset's cells: one
        # flat index a point and one an offset, added. The raster is padded
        # with 2k + 1 cells of zeros all round and a point's cell clipped into
        # the padding, so a point off the raster scores 0 at every offset
        # without a test per pair. 2.5 ms a scan (30 headings) on the Mac,
        # was 12.
        ki = np.rint(X[disc] / res).astype(np.int64)
        kj = np.rint(Y[disc] / res).astype(np.int64)
        self._pad = pad = 2 * k + 1
        self._w = w = n + 2 * pad
        flat = np.zeros((w, w), np.float32)
        flat[pad:pad + n, pad:pad + n] = self.score
        self._flat = flat.ravel()
        self._koff = (ki * w + kj).astype(np.int32)
        self._lo, self._hi = k, n - 1 + 2 * pad - k     # a cell's clip

    def scores(self, pts, yaw):
        """Score of every candidate offset at heading `yaw` (map radians)
        for car-frame points `pts` (m, 2): (len(offsets),) float32."""
        c, s = math.cos(yaw), math.sin(yaw)
        q = pts @ np.array([[c, s], [-s, c]])          # rotated, (m, 2)
        i = np.floor((q[:, 0] + (self.cx - self.x0)) / self.res)
        j = np.floor((q[:, 1] + (self.cy - self.y0)) / self.res)
        i = np.clip(i + self._pad, self._lo, self._hi).astype(np.int32)
        j = np.clip(j + self._pad, self._lo, self._hi).astype(np.int32)
        idx = self._koff[:, None] + (i * self._w + j)[None, :]
        return self._flat[idx].mean(1, dtype=np.float32)


def _angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _same(a, b, distinct_m, distinct_deg):
    return (math.hypot(a[0] - b[0], a[1] - b[1]) <= distinct_m
            and abs(_angle(a[2] - b[2])) <= math.radians(distinct_deg))


def subsample(pts, n=SCORE_POINTS):
    """n points spread evenly through the scan (they are in bearing order)."""
    if len(pts) <= n:
        return pts
    return pts[np.linspace(0, len(pts) - 1, n).astype(np.int64)]


def yaw_order(yaw0, step_deg=YAW_STEP_DEG, span=None):
    """Every heading a step apart round the circle, outwards from yaw0; with
    `span` (radians) only those within it of yaw0, and one step past it, so
    a span between two steps is still covered."""
    n = int(round(360.0 / step_deg))
    out = [0]
    for k in range(1, n // 2 + 1):
        out += [k, -k]
    out = out[:n]
    if span is not None and span < math.pi:
        last = int(math.ceil(math.degrees(span) / step_deg))
        out = [k for k in out if abs(k) <= last]
    return [yaw0 + math.radians(step_deg) * k for k in out]


class Search:
    """One lost episode's search: a Field round the centre and the headings
    still to try, YAWS_PER_SCAN of them each step(), starting over once the
    circle is done."""

    def __init__(self, grid, cx, cy, yaw0, r_m=SEARCH_R_M, span=None):
        self.field = Field(grid, cx, cy, r_m)
        self.yaw0 = float(yaw0)
        # Only headings within this of the one handed to step() (radians),
        # or None for the whole circle: a circle is then that arc.
        self.span = span
        self._next = 0
        self.sweeps = 0
        self.swept = False      # the last step() finished a circle
        self.fits = []          # (inliers, pose) slam.py found this circle

    def add(self, inliers, pose):
        self.fits.append((float(inliers), tuple(float(v) for v in pose)))

    def pick(self, min_inliers, prior=None, distinct_m=DISTINCT_M,
             distinct_deg=DISTINCT_DEG, margin=AMBIGUOUS, prior_deg=PRIOR_DEG,
             early=False):
        """After a circle: the pose to believe, or None. The best fit, if it
        is over min_inliers and no different answer comes within `margin` of
        it. If some do, `prior` (the gyro's heading, or None) settles it when
        exactly one of those answers is within prior_deg of it; otherwise
        None. Starts the next circle's list either way.

        `early`, before the circle is done: only an answer within prior_deg
        of the prior, and only if the headings searched so far hold no other.
        Every position has been tried at every heading searched, so a twin
        near the gyro's heading would be among them, and one far from it
        loses to the gyro anyway. The list is kept for the rest of the
        circle unless it picks. This is what finds a truck set down facing
        the way the gyro says on its first scan."""
        fits = sorted(self.fits, key=lambda f: -f[0])
        if not early:
            self.fits = []
        if not fits:
            return None
        best_i, best = fits[0]
        if best_i < min_inliers:
            return None
        # One pose per answer, the best fit of it, for every answer that
        # comes within margin of the best.
        answers = []
        for i, pose in fits:
            if i < best_i - margin:
                break
            if not any(_same(pose, a, distinct_m, distinct_deg)
                       for a in answers):
                answers.append(pose)
        if prior is None:
            return best if len(answers) == 1 and not early else None
        near = [a for a in answers
                if abs(_angle(a[2] - prior)) <= math.radians(prior_deg)]
        if early:
            if len(answers) == 1 and len(near) == 1:
                self.fits = []
                return near[0]
            return None
        if len(answers) == 1:
            return best
        return near[0] if len(near) == 1 else None

    def step(self, pts, yaw_now=None, per_scan=YAWS_PER_SCAN, top_k=TOP_K):
        """The best top_k (score, (x, y, yaw)) for this scan's car-frame
        points over the next per_scan headings, best first. The headings run
        outwards from yaw_now (the IMU's) when it is given, else from the
        heading the search started at."""
        order = yaw_order(self.yaw0 if yaw_now is None else yaw_now,
                          span=self.span)
        sub = subsample(pts)
        if not len(sub) or self.field.points == 0:
            return []
        cands = []
        for _ in range(min(per_scan, len(order))):
            yaw = order[self._next % len(order)]
            self._next += 1
            sc = self.field.scores(sub, yaw)
            best = np.argsort(sc)[-PER_YAW:]
            for b in best:
                ox, oy = self.field.offsets[b]
                cands.append((float(sc[b]), (self.field.cx + float(ox),
                                             self.field.cy + float(oy), yaw)))
        self.swept = self._next >= len(order)
        if self.swept:
            self._next = 0
            self.sweeps += 1
        cands.sort(key=lambda c: -c[0])
        return cands[:top_k]
