"""Looking for a person nobody can see: where on the SLAM map the goal goes
next, while following is on and the person is gone.

The model drives. This only picks the next point the goal input points at,
and the path planner (pathgoal.PathPlanner) routes to it like any goal.
Nothing here steers or touches the throttle; it never turns the truck on the
spot, and it has no turn of its own to make. Order of targets:

  along     where the person was last seen, carried on EXTEND_M further
            down the bearing they were seen on from the truck, stopped short
            of the first wall on the way (CLEAR_M off it). The way they
            went, which is the likeliest place to see them again.
  frontier  a free cell next to unknown on the map (the edge of what has
            been mapped): the clusters of them within RADIUS_M, at least
            MIN_FRONTIER_CELLS long, reachable from the truck over free
            cells CLEAR_M off every wall, not round a speck of unknown
            under MIN_UNKNOWN_CELLS left inside the map, nearest first with a bonus for a
            long edge (score = distance - INFO_WEIGHT * edge length).
  wander    with no frontier left (a room mapped wall to wall): the free,
            clear, reachable cell within RADIUS_M farthest from the truck
            and from every target tried so far.

A target is done on arrival (ARRIVE_M), or skipped with no PROGRESS_M
closer in NO_PROGRESS_S, or after TARGET_MAX_S in all; those are the patrol's
waypoint numbers (patrol.py), generous on purpose, as the model does not
always go where its goal input says. A target done or skipped is not picked
again (TRIED_M round it) until there is nothing else.

Clock: the crop's time (crop.t), the same clock the sightings are on.
"""

import math

import numpy as np

from crawler.autopilot import patrol as pt
from crawler.nav import costmap as cmap

EXTEND_M = 2.0
RADIUS_M = 8.0
# Off every wall, for a target and the cells reaching it: the truck's lethal
# band (costmap.INSCRIBED_M) plus 0.15 m.
CLEAR_M = cmap.INSCRIBED_M + 0.15
MIN_FRONTIER_CELLS = 6
# Unknown in a patch smaller than this (cells, 4-connected) is a speck left
# inside what is mapped, not the edge of it: 0.25 m2 at 10 cm.
MIN_UNKNOWN_CELLS = 25
INFO_WEIGHT = 0.5
# A target nearer the truck than this is where it already is.
MIN_TARGET_M = 1.0
ARRIVE_M = 0.5
PROGRESS_M = pt.PROGRESS_M
NO_PROGRESS_S = pt.NO_PROGRESS_S
TARGET_MAX_S = 90.0
TRIED_M = 1.0
TRIED_KEPT = 20
# Nothing to pick: look again this often.
RETRY_S = 1.0
CLOCK_STEP_MAX_S = 0.5
# Along the bearing: the person's own cell and this round it can be mapped
# as a wall (they were standing in it).
PERSON_R_M = 0.5
RAY_STEP_M = 0.1

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


def _window(view, cx, cy, radius):
    """(codes, x0, y0, res) of the view cut to the box radius round (cx, cy),
    or None."""
    res = float(view.res)
    codes = np.asarray(view.codes)
    nx, ny = codes.shape
    i0 = max(int(math.floor((cx - radius - view.x0) / res)), 0)
    j0 = max(int(math.floor((cy - radius - view.y0) / res)), 0)
    i1 = min(int(math.ceil((cx + radius - view.x0) / res)), nx)
    j1 = min(int(math.ceil((cy + radius - view.y0) / res)), ny)
    if i1 <= i0 or j1 <= j0:
        return None
    return (codes[i0:i1, j0:j1], float(view.x0) + i0 * res,
            float(view.y0) + j0 * res, res)


def _clearance(codes, res):
    """Metres from each cell to the nearest occupied one (inf with none)."""
    from scipy import ndimage
    occ = codes == OCCUPIED
    if not occ.any():
        return np.full(codes.shape, np.inf)
    return ndimage.distance_transform_edt(~occ) * res


def _reachable(clear_ok, x0, y0, res, px, py):
    """The cells of clear_ok connected to the one nearest (px, py), or None
    when the truck is not on one (it is then not used as a filter)."""
    from scipy import ndimage
    lab, n = ndimage.label(clear_ok, structure=np.ones((3, 3), bool))
    if n == 0:
        return None
    i, j = int((px - x0) / res), int((py - y0) / res)
    r = int(math.ceil(0.6 / res))
    nx, ny = clear_ok.shape
    best = None
    for a in range(max(i - r, 0), min(i + r + 1, nx)):
        for b in range(max(j - r, 0), min(j + r + 1, ny)):
            if lab[a, b]:
                d = (a - i) ** 2 + (b - j) ** 2
                if best is None or d < best[0]:
                    best = (d, lab[a, b])
    return None if best is None else lab == best[1]


def frontiers(view, pose, radius=RADIUS_M, clear_m=CLEAR_M,
              min_cells=MIN_FRONTIER_CELLS):
    """[(x, y, edge_m, distance_m)] of the frontier clusters round the map
    point pose, each the member cell nearest the cluster's middle."""
    from scipy import ndimage
    w = _window(view, pose[0], pose[1], radius)
    if w is None:
        return []
    codes, x0, y0, res = w
    clear = _clearance(codes, res)
    ok = (codes == FREE) & (clear >= clear_m)
    unknown = codes == UNKNOWN
    ulab, un = ndimage.label(unknown)
    if un:
        sizes = np.bincount(ulab.ravel())
        unknown = unknown & (sizes[ulab] >= MIN_UNKNOWN_CELLS)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    edge = ok & ndimage.binary_dilation(unknown, structure=cross)
    reach = _reachable(ok, x0, y0, res, pose[0], pose[1])
    if reach is not None:
        edge &= reach
    lab, n = ndimage.label(edge, structure=np.ones((3, 3), bool))
    out = []
    for k in range(1, n + 1):
        ii, jj = np.nonzero(lab == k)
        if len(ii) < min_cells:
            continue
        mi, mj = ii.mean(), jj.mean()
        m = int(np.argmin((ii - mi) ** 2 + (jj - mj) ** 2))
        x = x0 + (ii[m] + 0.5) * res
        y = y0 + (jj[m] + 0.5) * res
        d = math.hypot(x - pose[0], y - pose[1])
        if d <= radius:
            out.append((float(x), float(y), float(len(ii) * res), float(d)))
    return out


def along(view, sight, person, extend_m=EXTEND_M, clear_m=CLEAR_M):
    """(x, y): the person's point carried extend_m on down the bearing from
    sight (the truck's map point when they were seen), short of any wall."""
    dx, dy = person[0] - sight[0], person[1] - sight[1]
    r = math.hypot(dx, dy)
    if r < 1e-6:
        return None
    ux, uy = dx / r, dy / r
    far = (person[0] + extend_m * ux, person[1] + extend_m * uy)
    if view is None:
        return far
    w = _window(view, person[0], person[1], extend_m + 1.0)
    if w is None:
        return far
    codes, x0, y0, res = w
    clear = _clearance(codes, res)
    nx, ny = codes.shape
    best = (person[0], person[1])
    for s in np.arange(0.0, extend_m + 1e-9, RAY_STEP_M):
        x, y = person[0] + s * ux, person[1] + s * uy
        i, j = int((x - x0) / res), int((y - y0) / res)
        if not (0 <= i < nx and 0 <= j < ny):
            best = (x, y)
            continue
        if s > PERSON_R_M and codes[i, j] == OCCUPIED:
            break
        if clear[i, j] >= clear_m or s <= PERSON_R_M:
            best = (x, y)
    return (float(best[0]), float(best[1]))


def wander(view, pose, avoid, radius=RADIUS_M, clear_m=CLEAR_M):
    """(x, y): the free, clear, reachable cell within radius farthest from
    pose and every point in avoid, or None."""
    w = _window(view, pose[0], pose[1], radius)
    if w is None:
        return None
    codes, x0, y0, res = w
    clear = _clearance(codes, res)
    ok = (codes == FREE) & (clear >= clear_m)
    reach = _reachable(ok, x0, y0, res, pose[0], pose[1])
    if reach is not None:
        ok &= reach
    ii, jj = np.nonzero(ok)
    if not len(ii):
        return None
    x = x0 + (ii + 0.5) * res
    y = y0 + (jj + 0.5) * res
    d_truck = np.hypot(x - pose[0], y - pose[1])
    keep = (d_truck >= MIN_TARGET_M + 0.5) & (d_truck <= radius)
    if not keep.any():
        return None
    x, y = x[keep], y[keep]
    near = np.hypot(x - pose[0], y - pose[1])
    for ax, ay in avoid:
        near = np.minimum(near, np.hypot(x - ax, y - ay))
    k = int(np.argmax(near))
    return float(x[k]), float(y[k])


class Search:
    """The search's current target, rotated on arrival or no progress.

    map_view  t -> runner.MapView or None (mapping runner's map), optional
    """

    def __init__(self, map_view=None, say=None):
        self.map_view = map_view
        self._say = say
        self.target = None          # (x, y, frame) or None
        self.kind = ""              # along / frontier / wander / ""
        self.targets = 0            # picked since the search began
        self.arrived = 0
        self.skipped = 0
        self.frontiers = 0          # clusters at the last pick
        self.started_t = None
        self._hint = None           # ((sx, sy), (px, py), frame) for along
        self._tried = []            # [(x, y)] on frame self._tried_frame
        self._tried_frame = None
        self._frame_since = None
        self._frame = None
        self._t = None
        self._best = math.inf
        self._since_best = 0.0
        self._age = 0.0
        self._retry_t = None
        self._idle_said = False

    def _log(self, text):
        print(text)
        if self._say is not None:
            try:
                self._say("info", "Follow", text)
            except Exception:
                pass

    def reset(self):
        self.target, self.kind = None, ""
        self.started_t, self._hint = None, None
        self._tried, self._retry_t, self._idle_said = [], None, False
        self.targets = self.arrived = self.skipped = 0

    @property
    def active(self):
        return self.started_t is not None

    def start(self, crop, hint=None):
        """Begin a search off this crop. hint: ((sx, sy), (px, py), frame),
        the truck's map point when the person was last seen and theirs."""
        self.reset()
        self.started_t = float(getattr(crop, "t", 0.0)) if crop is not None else 0.0
        self._hint = hint

    def follow(self, L):
        """One egomap.Link: the target and the tried points carried along."""
        c, s = math.cos(L.yaw), math.sin(L.yaw)

        def tf(x, y):
            return c * x - s * y + L.x, s * x + c * y + L.y
        if self.target is not None and self.target[2] == L.frm:
            self.target = (*tf(self.target[0], self.target[1]), L.to)
        if self._tried_frame == L.frm:
            self._tried = [tf(x, y) for x, y in self._tried]
            self._tried_frame = L.to
        if self._hint is not None and self._hint[2] == L.frm:
            (sx, sy), (px, py), _ = self._hint
            self._hint = (tf(sx, sy), tf(px, py), L.to)

    def _view(self, crop):
        if self.map_view is None:
            return None
        try:
            view = self.map_view()
        except Exception:
            return None
        if view is None or (self._frame_since is not None
                            and float(view.t) < self._frame_since):
            return None
        return view

    def step(self, crop):
        """One pass: the target's clock, and a new target when it is done.
        A crop without a map place does nothing."""
        if not self.active or crop is None or not getattr(crop, "present", False):
            return
        t = float(crop.t)
        frame = int(getattr(crop, "frame", 0))
        if frame != self._frame:
            # The first frame seen: any map; a new frame: one made since.
            self._frame_since = -math.inf if self._frame is None else t
            self._frame = frame
        dt = 0.0 if self._t is None else min(max(t - self._t, 0.0), CLOCK_STEP_MAX_S)
        self._t = t
        if self.target is not None and self.target[2] != frame:
            self.target = None
        if self.target is not None:
            d = math.hypot(self.target[0] - crop.x, self.target[1] - crop.y)
            self._age += dt
            if d < self._best - PROGRESS_M:
                self._best, self._since_best = d, 0.0
            else:
                self._since_best += dt
            why = None
            if d <= ARRIVE_M:
                self.arrived += 1
                why = "there"
            elif self._since_best >= NO_PROGRESS_S - 1e-6:
                self.skipped += 1
                why = f"skipped, no {PROGRESS_M:.1f} m closer in {NO_PROGRESS_S:.0f} s"
            elif self._age >= TARGET_MAX_S - 1e-6:
                self.skipped += 1
                why = f"skipped, not there in {TARGET_MAX_S:.0f} s"
            if why is None:
                return
            self._log(f"Search target ({self.kind}) {why}, {d:.1f} m off")
            self._remember(self.target)
            self.target = None
        if self._retry_t is not None and t - self._retry_t < RETRY_S:
            return
        self._pick(crop, frame, t)

    def _remember(self, target):
        if self._tried_frame != target[2]:
            self._tried, self._tried_frame = [], target[2]
        self._tried.append((target[0], target[1]))
        self._tried = self._tried[-TRIED_KEPT:]

    def _tried_near(self, x, y, frame):
        if self._tried_frame != frame:
            return False
        return any(math.hypot(x - a, y - b) < TRIED_M for a, b in self._tried)

    def _set(self, x, y, frame, kind, crop):
        self.target, self.kind = (float(x), float(y), frame), kind
        self.targets += 1
        self._best = math.hypot(x - crop.x, y - crop.y)
        self._since_best, self._age = 0.0, 0.0
        self._retry_t, self._idle_said = None, False
        self._log(f"Searching: heading for a {kind} point {self._best:.1f} m off")

    def _pick(self, crop, frame, t):
        view = self._view(crop)
        pose = (float(crop.x), float(crop.y))
        hint, self._hint = self._hint, None
        if hint is not None and hint[2] == frame:
            p = along(view, hint[0], hint[1])
            if (p is not None and math.hypot(p[0] - pose[0], p[1] - pose[1]) >= MIN_TARGET_M
                    and not self._tried_near(p[0], p[1], frame)):
                self._set(p[0], p[1], frame, "along", crop)
                return
        if view is not None:
            fs = frontiers(view, pose)
            self.frontiers = len(fs)
            fs = [f for f in fs if f[3] >= MIN_TARGET_M and not self._tried_near(f[0], f[1], frame)]
            if fs:
                x, y, _, _ = min(fs, key=lambda f: f[3] - INFO_WEIGHT * f[2])
                self._set(x, y, frame, "frontier", crop)
                return
            avoid = list(self._tried) if self._tried_frame == frame else []
            p = wander(view, pose, avoid)
            if p is not None:
                self._set(p[0], p[1], frame, "wander", crop)
                return
        self._retry_t, self.kind = t, ""
        if not self._idle_said:
            self._idle_said = True
            self._log("Searching: nowhere to go on the map yet - holding neutral")
