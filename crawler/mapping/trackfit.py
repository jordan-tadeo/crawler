"""Putting a saved track on this boot's map: the saved walls fitted to the
live SLAM map, turned and moved, never scaled.

Every boot starts a fresh map at wherever the truck is (mapping/mapfile.py
says why the old map is not loaded), so a track saved on another boot is in
a frame of its own. It is saved with a sample of the walls of the map it
was on (autopilot/patrol.py), and those walls are the only thing used to
find it again: they are never loaded as the map or drawn into it.

The lidar is metric, so the fit is rigid: a heading and an x, y shift.

  1. Search. The live map is made into a score field at COARSE_M: near a
     wall is worth up to 1 (exp(-d^2 / 2 SIGMA_M^2), d the distance to the
     nearest occupied cell), a free cell clear of every wall costs
     FREE_COST, unknown is 0. For each heading every YAW_STEP_DEG round the
     whole circle the saved walls are rasterised turned that way and
     correlated with the field by FFT, which scores every shift at once.
     The best PER_YAW shifts of each heading are kept.
  2. Refine. The TOP_K best are each taken to a fit by ICP against the
     live map's occupied cells (point to point, closed form, trimmed).
  3. Judge. Every saved wall that lands on a part of the live map that has
     been seen is either within NEAR_M of a live wall (it agrees) or on
     free floor (it disagrees); unknown says nothing. A fit is accepted
     only if it has MIN_AGREE agreeing walls and MIN_AGREE_SHARE of the
     saved ones, if AGREE_SHARE of the seen ones agree, and if every
     different answer (DISTINCT_M or DISTINCT_DEG away) scores no more than
     MARGIN of it. A room that looks the same turned round has two answers
     and is not accepted until the map has seen something that tells them
     apart.

Retried by the patrol as the map grows; nothing here keeps state. About a
second on the Pi 4 for a whole house (measured, see the commit), so it runs
on a thread of its own, between the passes of the control loop.
"""

import math
import os
import sys
import threading
import time
from typing import NamedTuple

import numpy as np

COARSE_M = 0.2
SIGMA_M = 0.15
# A free cell further than this from any wall costs FREE_COST in the search.
FREE_CLEAR_M = 0.3
FREE_COST = 1.0
YAW_STEP_DEG = 2.0
PER_YAW = 2
TOP_K = 8
ICP_ITERS = 12
ICP_TRIM_M = (0.5, 0.3, 0.2)
NEAR_M = 0.15
# Two fits are different answers this far apart.
DISTINCT_M = 0.5
DISTINCT_DEG = 10.0
# What an accepted fit needs.
MIN_AGREE = 150
MIN_AGREE_SHARE = 0.15
AGREE_SHARE = 0.85
MARGIN = 0.6
# The most saved walls used; more are thinned evenly.
MAX_WALLS = 3000


class Fit(NamedTuple):
    """A rigid fit, saved frame -> live frame: p_live = R(yaw) p + (x, y)."""
    yaw: float            # radians
    x: float
    y: float
    agree: int            # saved walls within NEAR_M of a live wall
    disagree: int         # saved walls on live free floor
    share: float          # agree / (agree + disagree)
    score: float          # agree - disagree
    second: float         # the best different answer's score (0 if none)
    ok: bool
    why: str              # why it was not accepted, "" if it was
    walls: int = 0        # saved walls used


def transform(fit, pts):
    """(N, 2) points through a fit (or any (yaw, x, y))."""
    yaw, x, y = float(fit[0]), float(fit[1]), float(fit[2])
    pts = np.asarray(pts, float).reshape(-1, 2)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.column_stack([c * pts[:, 0] - s * pts[:, 1] + x,
                            s * pts[:, 0] + c * pts[:, 1] + y])


def compose(a, b):
    """The transform `a` after `b`: transform(compose(a, b), p) ==
    transform(a, transform(b, p))."""
    ya, xa, y_a = float(a[0]), float(a[1]), float(a[2])
    yb, xb, y_b = float(b[0]), float(b[1]), float(b[2])
    c, s = math.cos(ya), math.sin(ya)
    return (_wrap(ya + yb), c * xb - s * y_b + xa, s * xb + c * y_b + y_a)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def occupied_points(view):
    from crawler.mapping import grid as gr
    i, j = np.nonzero(np.asarray(view.codes) == gr.OCCUPIED)
    return np.column_stack([view.x0 + (i + 0.5) * view.res,
                            view.y0 + (j + 0.5) * view.res])


def _coarse(codes, res, step):
    """codes [nx, ny] -> the same at `step` cells a pixel, the highest code
    in each (occupied over free over unknown)."""
    if step <= 1:
        return codes
    nx, ny = codes.shape
    px, py = -(-nx // step) * step, -(-ny // step) * step
    pad = np.zeros((px, py), codes.dtype)
    pad[:nx, :ny] = codes
    return pad.reshape(px // step, step, py // step, step).max((1, 3))


def _field(codes):
    """The search's score field over coarse codes."""
    from scipy.ndimage import distance_transform_edt
    from crawler.mapping import grid as gr
    occ = codes == gr.OCCUPIED
    d = distance_transform_edt(~occ) * COARSE_M
    w = np.exp(-0.5 * (d / SIGMA_M) ** 2)
    w[(codes == gr.FREE) & (d > FREE_CLEAR_M)] = -FREE_COST
    return w.astype(np.float32)


def thin(walls, n=MAX_WALLS):
    walls = np.asarray(walls, float).reshape(-1, 2)
    walls = walls[np.isfinite(walls).all(1)]
    if len(walls) > n:
        walls = walls[np.linspace(0, len(walls) - 1, n).astype(np.int64)]
    return walls


def search(walls, view, yaw_step_deg=YAW_STEP_DEG, pause=None):
    """[(score, yaw, x, y)] the best few coarse fits, best first.
    `pause`, if given, is called between headings (a thread yields)."""
    from scipy import fft as sfft
    walls = thin(walls)
    step = max(1, int(round(COARSE_M / float(view.res))))
    codes = _coarse(np.asarray(view.codes), view.res, step)
    res = float(view.res) * step
    w = _field(codes)
    lx, ly = w.shape
    # Every turned raster fits in a square as wide as the walls' spread.
    centre = walls.mean(0)
    rel = walls - centre
    reach = float(np.max(np.hypot(rel[:, 0], rel[:, 1]))) if len(rel) else 0.0
    rn = int(math.ceil(2 * reach / res)) + 2
    sx, sy = sfft.next_fast_len(lx + rn), sfft.next_fast_len(ly + rn)
    fw = sfft.rfft2(w, s=(sx, sy))
    out = []
    for yaw in np.radians(np.arange(0.0, 360.0, yaw_step_deg)):
        c, s = math.cos(yaw), math.sin(yaw)
        p = np.column_stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]])
        a = np.floor((p + reach) / res).astype(np.int64) + 1
        r = np.zeros((rn, rn), np.float32)
        np.add.at(r, (np.clip(a[:, 0], 0, rn - 1), np.clip(a[:, 1], 0, rn - 1)), 1.0)
        corr = sfft.irfft2(fw * np.conj(sfft.rfft2(r, s=(sx, sy))), s=(sx, sy))
        for _ in range(PER_YAW):
            k = int(np.argmax(corr))
            u, v = divmod(k, sy)
            best = float(corr[u, v])
            if best <= 0:
                break
            # Raster cell a (centre (a - 0.5) res - reach, turned walls
            # round their centre) lands on live coarse cell a + u (centre
            # x0 + (a + u + 0.5) res); negative shifts are wrapped.
            uu = u - sx if u > sx - rn else u
            vv = v - sy if v > sy - rn else v
            tx = view.x0 + (uu + 1) * res + reach
            ty = view.y0 + (vv + 1) * res + reach
            out.append((best, float(yaw), float(tx - (c * centre[0] - s * centre[1])),
                        float(ty - (s * centre[0] + c * centre[1]))))
            corr[max(0, u - 3):u + 4, max(0, v - 3):v + 4] = -np.inf
        if pause is not None:
            pause()
    out.sort(key=lambda t: -t[0])
    picked = []
    for cand in out:
        if all(_distinct(cand[1:], q[1:]) for q in picked):
            picked.append(cand)
        if len(picked) >= TOP_K:
            break
    return picked


def _distinct(a, b):
    dyaw = abs(_wrap(a[0] - b[0]))
    return dyaw > math.radians(DISTINCT_DEG) or math.hypot(a[1] - b[1], a[2] - b[2]) > DISTINCT_M


def icp(walls, tree, start, iters=ICP_ITERS):
    """(yaw, x, y) from `start`, point to point against a cKDTree of the
    live walls, trimmed at ICP_TRIM_M (tightening)."""
    yaw, x, y = start
    pts = thin(walls)
    for k in range(iters):
        trim = ICP_TRIM_M[min(k, len(ICP_TRIM_M) - 1)]
        p = transform((yaw, x, y), pts)
        d, idx = tree.query(p, distance_upper_bound=trim)
        ok = np.isfinite(d)
        if ok.sum() < 10:
            break
        a, b = p[ok], tree.data[idx[ok]]
        ca, cb = a.mean(0), b.mean(0)
        h = (a - ca).T @ (b - cb)
        dyaw = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
        c, s = math.cos(dyaw), math.sin(dyaw)
        t = cb - np.array([c * ca[0] - s * ca[1], s * ca[0] + c * ca[1]])
        yaw, x, y = compose((dyaw, t[0], t[1]), (yaw, x, y))
        if abs(dyaw) < 1e-4 and math.hypot(*t) < 1e-3:
            break
    return yaw, x, y


def judge(walls, view, tree, pose):
    """(agree, disagree) of the saved walls at `pose` on the live map."""
    from crawler.mapping import grid as gr
    p = transform(pose, thin(walls))
    d, _ = tree.query(p, distance_upper_bound=NEAR_M + 1e-6)
    near = np.isfinite(d)
    codes = np.asarray(view.codes)
    i = np.floor((p[:, 0] - view.x0) / view.res).astype(np.int64)
    j = np.floor((p[:, 1] - view.y0) / view.res).astype(np.int64)
    inside = (i >= 0) & (i < codes.shape[0]) & (j >= 0) & (j < codes.shape[1])
    free = np.zeros(len(p), bool)
    free[inside] = codes[i[inside], j[inside]] == gr.FREE
    return int(near.sum()), int((free & ~near).sum())


def fit(walls, view, pause=None):
    """The best Fit of the saved walls on this MapView, accepted or not, or
    None with nothing to fit."""
    from scipy.spatial import cKDTree
    walls = thin(walls)
    if view is None or len(walls) < 10:
        return None
    live = occupied_points(view)
    if len(live) < 10:
        return None
    tree = cKDTree(live)
    fits = []
    for _, yaw, x, y in search(walls, view, pause=pause):
        pose = icp(walls, tree, (yaw, x, y))
        agree, disagree = judge(walls, view, tree, pose)
        fits.append((agree - disagree, pose, agree, disagree))
        if pause is not None:
            pause()
    if not fits:
        return None
    fits.sort(key=lambda f: -f[0])
    score, pose, agree, disagree = fits[0]
    second = max([f[0] for f in fits[1:] if _distinct(f[1], pose)] + [0.0])
    share = agree / max(agree + disagree, 1)
    why = ""
    if agree < MIN_AGREE or agree < MIN_AGREE_SHARE * len(walls):
        why = f"only {agree} of {len(walls)} saved walls seen to agree yet"
    elif share < AGREE_SHARE:
        why = f"{share:.0%} of the seen walls agree"
    elif second > MARGIN * score:
        why = f"two places fit as well ({second:.0f} against {score:.0f})"
    return Fit(yaw=float(pose[0]), x=float(pose[1]), y=float(pose[2]),
               agree=int(agree), disagree=int(disagree), share=float(share),
               score=float(score), second=float(second), ok=not why, why=why,
               walls=len(walls))


class Fitter:
    """fit() on a thread of its own, niced on Linux, one job at a time:
    submit() hands it walls and a MapView and returns at once, result()
    hands back (tag, Fit or None) once it is done. Between headings it
    sleeps PAUSE_S so the control loop's thread gets the interpreter.
    background=False runs the job inside submit(), for tests."""

    PAUSE_S = 0.002
    NICE = 10

    def __init__(self, background=True):
        self.background = bool(background)
        self._lock = threading.Lock()
        self._job = None
        self._done = None
        self._wake = threading.Event()
        self._thread = None
        self.busy = False
        self.runs = 0
        self.errors = 0
        self.last_s = 0.0

    def submit(self, tag, walls, view):
        """False if a job is still running."""
        with self._lock:
            if self.busy:
                return False
            self.busy = True
            self._job = (tag, walls, view)
        if not self.background:
            self._run()
            return True
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="trackfit", daemon=True)
            self._thread.start()
        self._wake.set()
        return True

    def result(self):
        with self._lock:
            done, self._done = self._done, None
        return done

    def _loop(self):
        if sys.platform.startswith("linux"):
            try:
                os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), self.NICE)
            except Exception:
                pass
        while True:
            self._wake.wait()
            self._wake.clear()
            self._run()

    def _run(self):
        with self._lock:
            job, self._job = self._job, None
        if job is None:
            return
        tag, walls, view = job
        t0 = time.perf_counter()
        pause = (lambda: time.sleep(self.PAUSE_S)) if self.background else None
        try:
            out = fit(walls, view, pause=pause)
        except Exception:
            self.errors += 1
            out = None
        with self._lock:
            self.last_s = time.perf_counter() - t0
            self.runs += 1
            self._done = (tag, out)
            self.busy = False
