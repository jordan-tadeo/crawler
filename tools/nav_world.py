"""The 2D world the navigation is tested in: a floor plan, the truck, its lidar.

Laptop only. tools/nav_sim.py drives it; the tests build synthetic rooms with
it. Nothing here runs on the car.

The world is an occupancy grid at 2.5 cm, from one of two places:

  load_map()    a real SLAM map of the house, read from wherever it is on
                disk at run time and thresholded into solid and free. Those
                maps never go in the repo: this is a public one and they are
                the inside of a house.
  room()        synthetic rooms, for the tests.

The truck (Truck) is kinematic: a pose, a speed, and a curvature off the
steering command, with the response randomised the way the real one varies.
The model is the one measured off four drives in the house (crawler/nav/
truck.py has the curve itself), and each episode draws its own:

  curvature   k(u_f) x a speed factor clamp(1.24 - 0.55|v|, 0.85, 1.10),
              x1.1 in reverse, x exp(eta), eta an Ornstein-Uhlenbeck process
              (sigma 0.18 forward, 0.25 reverse, about 1 s correlation); plus
              an additive wander of 0.10 /m on the same clock; plus, now and
              then (1% a second of turning), half a second where the wheels
              scrub and the turn is 0-30% of what it should be
  lag         u_f is the command delayed 0.10 s then lagged 0.12 s
  speed       the measured throttle map, delayed 0.09 s and lagged 0.11 s,
              x 0.82 at full lock, with a slow 6% wobble; the ESC's drag brake
              stops it in about 0.3 s
  axle        a constant gain on the curve, 1 by default; the tests run it at
              0.6 (an axle that turns 60% of what it is told)

The response levels nav_sim reports by (Response.level()):

  ideal       no noise, gain 1, the lag only
  measured    the recipe above
  harsh       sigma doubled, gain 0.75, lag 1.5x, the scrub three times as
              often: a bad surface
  axle60      measured, with the curve's gain at 0.6

The lidar (Lidar) ray-casts the grid from the truck's pose: 360 beams to
8 m, a centimetre of range noise, 5% of beams dropped and the odd sector
gone. The map the navigator gets (SimMap) is built from those scans by the
same log-odds rule as the live grid (crawler/mapping/grid.py, same
constants), at the pose the navigator believes, which is the true pose
plus a slowly wandering error of a few centimetres: what a SLAM that is
tracking gives. It is not the live SLAM itself.
"""

import json
import math
from typing import NamedTuple

import numpy as np
from scipy import ndimage

from crawler.nav import mapview as mv
from crawler.nav import truck as tk

WORLD_RES = 0.025

# The live grid's log-odds (crawler/mapping/grid.py): int8 steps of 0.05.
L_HIT = 17
L_FREE = -8
L_MIN = -40
L_OCC = 10
L_MAX = 30
MAP_RES = 0.05
MAP_MAX_R = 8.0


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------

class World(NamedTuple):
    solid: np.ndarray       # bool [nx, ny] at res
    res: float
    edt: np.ndarray         # metres to the nearest solid cell
    name: str = ""

    @property
    def size_m(self):
        return self.solid.shape[0] * self.res, self.solid.shape[1] * self.res

    def cell(self, x, y):
        return (np.floor(np.asarray(x) / self.res).astype(np.int64),
                np.floor(np.asarray(y) / self.res).astype(np.int64))

    def solid_at(self, x, y):
        ix, iy = self.cell(x, y)
        nx, ny = self.solid.shape
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        out = np.ones(np.shape(ix), bool)
        out[ok] = self.solid[ix[ok], iy[ok]]
        return out

    def clearance_at(self, x, y):
        ix, iy = self.cell(x, y)
        nx, ny = self.solid.shape
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        out = np.zeros(np.shape(ix))
        out[ok] = self.edt[ix[ok], iy[ok]]
        return out


def make_world(solid, res=WORLD_RES, name=""):
    solid = np.asarray(solid, bool).copy()
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    edt = ndimage.distance_transform_edt(~solid) * res
    return World(solid, float(res), edt, name)


def room(kind="box", size=(6.0, 5.0), res=WORLD_RES):
    """Synthetic rooms for the tests. Walls are 5 cm.

      box       an empty rectangle
      wall      a box with a wall across the middle and a 0.9 m gap at one end
      pillar    a box with a 0.6 m square pillar in the middle
      corridor  a 1.0 m corridor with a right-angle bend
    """
    nx, ny = int(round(size[0] / res)), int(round(size[1] / res))
    s = np.zeros((nx, ny), bool)
    w = max(1, int(round(0.05 / res)))
    s[:w, :] = s[-w:, :] = s[:, :w] = s[:, -w:] = True

    def box(x0, x1, y0, y1):
        s[int(x0 / res):int(math.ceil(x1 / res)), int(y0 / res):int(math.ceil(y1 / res))] = True

    if kind == "wall":
        mx = size[0] / 2
        box(mx - 0.025, mx + 0.025, 0.0, size[1] - 0.9)
    elif kind == "pillar":
        box(size[0] / 2 - 0.3, size[0] / 2 + 0.3, size[1] / 2 - 0.3, size[1] / 2 + 0.3)
    elif kind == "corridor":
        s[:] = True
        box_free = np.zeros_like(s)
        a = slice(int(0.5 / res), int((size[0] - 0.5) / res))
        box_free[a, int(0.5 / res):int(1.5 / res)] = True
        bx = slice(int((size[0] - 1.5) / res), int((size[0] - 0.5) / res))
        box_free[bx, int(0.5 / res):int((size[1] - 0.5) / res)] = True
        s[box_free] = False
    elif kind != "box":
        raise ValueError(kind)
    return make_world(s, res, kind)


# How the house maps are read. BreezySLAM's byte map: 127 is unknown, 0
# solid, 255 free, with a blur either side of a wall. Anything short of
# clearly free is solid to the truck, so the unexplored side of a doorway is
# a wall, and smudges of furniture stay furniture.
FREE_MIN = 170


def load_map(path, free_min=FREE_MIN, pad_m=0.5):
    """A World off a SLAM result .npz (map, map_px, map_m), cropped to the
    free floor plus pad_m. The axes are the array's: x along axis 0."""
    d = np.load(path)
    m = d["map"]
    res = float(d["map_m"]) / float(d["map_px"])
    free = m >= free_min
    # Only the biggest connected piece of floor is the house.
    lab, n = ndimage.label(free)
    if n > 1:
        sizes = ndimage.sum(free, lab, range(1, n + 1))
        free = lab == (1 + int(np.argmax(sizes)))
    # Specks of solid inside the floor smaller than a 5 cm square are noise.
    free = ndimage.binary_closing(free, iterations=1) | free
    rows = np.flatnonzero(free.any(1))
    cols = np.flatnonzero(free.any(0))
    p = int(pad_m / res)
    r0, r1 = max(rows[0] - p, 0), min(rows[-1] + p + 1, m.shape[0])
    c0, c1 = max(cols[0] - p, 0), min(cols[-1] + p + 1, m.shape[1])
    solid = ~free[r0:r1, c0:c1]
    if abs(res - WORLD_RES) > 1e-6:
        k = WORLD_RES / res
        solid = ndimage.zoom(solid.astype(np.uint8), 1 / k, order=0).astype(bool)
        res = WORLD_RES
    return make_world(solid, res, str(path))


def sample_poses(world, rng, n, clearance=0.35, mask=None):
    """n random poses on free floor at least `clearance` from anything."""
    ok = world.edt >= clearance
    if mask is not None:
        ok &= mask
    ii, jj = np.nonzero(ok)
    if not len(ii):
        return []
    k = rng.integers(0, len(ii), n)
    x = (ii[k] + rng.random(n)) * world.res
    y = (jj[k] + rng.random(n)) * world.res
    yaw = rng.uniform(-math.pi, math.pi, n)
    return list(zip(x, y, yaw))


def reachable_mask(world, x, y, clearance):
    """Cells a disc of `clearance` can reach from (x, y)."""
    ok = world.edt >= clearance
    lab, _ = ndimage.label(ok)
    ix, iy = world.cell(x, y)
    if not ok[ix, iy]:
        return np.zeros_like(ok)
    return lab == lab[ix, iy]


# ---------------------------------------------------------------------------
# The truck's response
# ---------------------------------------------------------------------------

class Response(NamedTuple):
    gain: float = 1.0           # on the whole curve (an axle at 60% is 0.6)
    sigma_fwd: float = 0.18
    sigma_rev: float = 0.25
    corr_s: float = 1.0
    wander: float = 0.10        # additive curvature noise, 1/m
    scrub_rate: float = 0.01    # per second of steady turning
    dead_s: float = 0.10
    tau_s: float = 0.12
    speed_dead_s: float = 0.09
    speed_tau_s: float = 0.11
    speed_wobble: float = 0.06
    speed_effect: bool = True
    name: str = "measured"

    @staticmethod
    def level(name, rng=None):
        if name == "ideal":
            return Response(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.10, 0.12, 0.09,
                            0.11, 0.0, False, "ideal")
        if name == "measured":
            return Response()
        if name == "harsh":
            return Response(0.75, 0.36, 0.5, 1.0, 0.2, 0.03, 0.15, 0.18,
                            0.12, 0.15, 0.10, True, "harsh")
        if name == "axle60":
            return Response(gain=0.6, name="axle60")
        raise ValueError(name)


def load_response(path):
    """The measured lag and noise off a steering response.json, where it has
    them; anything missing keeps the default (which is the same model)."""
    try:
        with open(path) as f:
            j = json.load(f)
        lag = j["lag"]
        eff = j["effectiveness_ratio"]["sample_as"]
        r = Response(dead_s=float(lag["Td_s"]["value"]),
                     tau_s=float(lag["tau_s"]["value"]),
                     corr_s=float(eff["eta_correlation_time_s"]["value"]),
                     wander=float(j["straight_line_noise"]["kappa_std_per_m"]))
        return r
    except Exception:
        return Response()


class Truck:
    """The truck in the world. step(dt, throttle, front) moves it, unless
    the move would put the footprint into something, in which case it stays
    put, stops, and that is a collision."""

    def __init__(self, world, pose, response=Response(), rng=None, dt=0.02):
        self.world = world
        self.x, self.y, self.yaw = (float(v) for v in pose)
        self.v = 0.0
        self.r = response
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.dt = dt
        nd = max(1, int(round(response.dead_s / dt)))
        self._steer_q = [0.0] * nd
        ns = max(1, int(round(response.speed_dead_s / dt)))
        self._thr_q = [0.0] * ns
        self.u_f = 0.0
        self.eta = 0.0
        self.wander = 0.0
        self.wobble = 0.0
        self.scrub_left = 0.0
        self.scrub_ratio = 1.0
        self.in_contact = False
        self.collisions = 0
        self.contact_s = 0.0
        self.min_clearance = math.inf
        self.travel = 0.0
        self.yaw_rate = 0.0

    @property
    def pose(self):
        return (self.x, self.y, self.yaw)

    def _ou(self, value, sigma):
        if sigma <= 0:
            return 0.0
        a = math.exp(-self.dt / self.r.corr_s)
        return a * value + sigma * math.sqrt(1 - a * a) * self.rng.standard_normal()

    def clearance(self):
        pts = tk.footprint_points(self.x, self.y, self.yaw, self.world.res)
        return float(self.world.clearance_at(pts[:, 0], pts[:, 1]).min())

    def step(self, throttle, front):
        r, dt = self.r, self.dt
        # Steering: dead time, then a first-order lag.
        self._steer_q.append(float(front))
        u = self._steer_q.pop(0)
        self.u_f += (u - self.u_f) * (1 - math.exp(-dt / r.tau_s))
        # Speed: dead time, lag, the brake.
        self._thr_q.append(float(throttle))
        t = self._thr_q.pop(0)
        target = tk.ground_speed(t)
        if target != 0.0:
            target *= 1.0 - 0.18 * min(abs(self.u_f) / tk.MAX_TURN, 1.0)
            self.wobble = self._ou(self.wobble, r.speed_wobble)
            target *= 1.0 + self.wobble
        tau = r.speed_tau_s if target != 0.0 else 0.13
        self.v += (target - self.v) * (1 - math.exp(-dt / tau))
        if abs(self.v) < 1e-3 and target == 0.0:
            self.v = 0.0

        rev = self.v < 0
        self.eta = self._ou(self.eta, r.sigma_rev if rev else r.sigma_fwd)
        self.wander = self._ou(self.wander, r.wander)
        k = tk.curvature_for_front(self.u_f) * r.gain
        if r.speed_effect:
            k *= min(max(1.24 - 0.55 * abs(self.v), 0.85), 1.10)
            if rev:
                k *= 1.10
        k *= math.exp(self.eta)
        if self.scrub_left > 0:
            self.scrub_left -= dt
            k *= self.scrub_ratio
        elif abs(self.u_f) > 0.3 and self.rng.random() < r.scrub_rate * dt:
            self.scrub_left = 0.5
            self.scrub_ratio = self.rng.uniform(-0.2, 0.3)
        k += self.wander

        ds = self.v * dt
        nyaw = self.yaw + k * ds
        mid = self.yaw + 0.5 * k * ds
        nx = self.x + ds * math.cos(mid)
        ny = self.y + ds * math.sin(mid)
        self.yaw_rate = k * self.v
        if ds != 0.0:
            pts = tk.footprint_points(nx, ny, nyaw, self.world.res)
            if self.world.solid_at(pts[:, 0], pts[:, 1]).any():
                if not self.in_contact:
                    self.collisions += 1
                self.in_contact = True
                self.contact_s += dt
                self.v = 0.0
                self.yaw_rate = 0.0
                return
        self.in_contact = False
        self.x, self.y, self.yaw = nx, ny, (nyaw + math.pi) % (2 * math.pi) - math.pi
        self.travel += abs(ds)


# ---------------------------------------------------------------------------
# The lidar
# ---------------------------------------------------------------------------

class Lidar:
    """360 beams from the truck's pose, ranges in metres, NaN for nothing."""

    def __init__(self, world, rng, beams=360, max_r=MAP_MAX_R, noise=0.01,
                 dropout=0.05, sector_dropout=0.02):
        self.world = world
        self.rng = rng
        self.beams = beams
        self.max_r = max_r
        self.noise = noise
        self.dropout = dropout
        self.sector_dropout = sector_dropout
        self.angles = np.arange(beams) * (2 * math.pi / beams)
        step = world.res * 0.5
        self.steps = np.arange(step, max_r, step)

    def scan(self, pose):
        x, y, yaw = pose
        a = self.angles + yaw
        ca, sa = np.cos(a), np.sin(a)
        px = x + ca[:, None] * self.steps[None, :]
        py = y + sa[:, None] * self.steps[None, :]
        hit = self.world.solid_at(px, py)
        first = np.argmax(hit, axis=1)
        any_hit = hit[np.arange(self.beams), first]
        r = np.where(any_hit, self.steps[first], np.nan)
        r = r + self.rng.normal(0, self.noise, self.beams) * np.isfinite(r)
        drop = self.rng.random(self.beams) < self.dropout
        if self.rng.random() < self.sector_dropout:
            c = self.rng.integers(0, self.beams)
            w = self.rng.integers(5, 30)
            drop[(np.arange(self.beams) - c) % self.beams < w] = True
        r[drop] = np.nan
        return r

    def rear_clearance(self, ranges):
        """mm past the rear bumper to the nearest return in the rear stop's
        sector (driver.REAR_STOP_SECTOR, 180 +- 20 degrees), or None."""
        deg = np.degrees(self.angles)
        sector = np.abs(((deg - 180.0) + 180.0) % 360.0 - 180.0) <= 20.0
        r = ranges[sector]
        r = r[np.isfinite(r) & (r * 1000.0 >= abs(tk.REAR_M) * 1000.0)]
        if not len(r):
            return None
        return float(r.min() * 1000.0 - abs(tk.REAR_M) * 1000.0)


# ---------------------------------------------------------------------------
# The map the navigator sees
# ---------------------------------------------------------------------------

class SimMap:
    """A log-odds grid built from the simulated scans, the live grid's rule:
    a return makes its cell more occupied, a ray makes the cells it passes
    more free, each cell changed once a scan, a hit never cleared in the
    scan that hit it."""

    def __init__(self, size_m, res=MAP_RES, margin=1.0):
        self.res = res
        self.origin = (-margin, -margin)
        n = (int(math.ceil((size_m[0] + 2 * margin) / res)),
             int(math.ceil((size_m[1] + 2 * margin) / res)))
        self.lo = np.zeros(n, np.int16)
        self.version = 0

    def _cells(self, x, y):
        return (np.floor((x - self.origin[0]) / self.res).astype(np.int64),
                np.floor((y - self.origin[1]) / self.res).astype(np.int64))

    def integrate(self, pose, ranges, angles):
        x, y, yaw = pose
        ok = np.isfinite(ranges) & (ranges < MAP_MAX_R)
        r = ranges[ok]
        a = angles[ok] + yaw
        if not len(r):
            return np.zeros((0, 2))
        hx = x + r * np.cos(a)
        hy = y + r * np.sin(a)
        nx, ny = self.lo.shape
        n = int(math.ceil(r.max() / (self.res * 0.5)))
        f = np.linspace(0, 1, n, endpoint=False)
        fx = x + (hx - x)[:, None] * f[None, :]
        fy = y + (hy - y)[:, None] * f[None, :]
        # Only the part of each ray short of its return.
        keep = (f[None, :] * r[:, None]) < (r[:, None] - self.res)
        cx, cy = self._cells(fx[keep], fy[keep])
        ix, iy = self._cells(hx, hy)
        inb = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
        free = np.unique(cx[inb] * ny + cy[inb])
        hb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        hits = np.unique(ix[hb] * ny + iy[hb])
        free = np.setdiff1d(free, hits, assume_unique=True)
        flat = self.lo.reshape(-1)
        flat[free] = np.maximum(flat[free] + L_FREE, L_MIN)
        flat[hits] = np.minimum(flat[hits] + L_HIT, L_MAX)
        self.version += 1
        return np.stack([hx, hy], 1)

    def fill_from(self, world):
        """Start already mapped: the world, as the grid would hold it after a
        good look round. Each 5 cm cell is solid if any 2.5 cm cell in it is."""
        k = int(round(self.res / world.res))
        nx, ny = self.lo.shape
        ox, oy = int(round(-self.origin[0] / self.res)), int(round(-self.origin[1] / self.res))
        s = world.solid
        px, py = (-s.shape[0]) % k, (-s.shape[1]) % k
        s = np.pad(s, ((0, px), (0, py)), constant_values=True)
        s = s.reshape(s.shape[0] // k, k, s.shape[1] // k, k).any(axis=(1, 3))
        w, h = min(s.shape[0], nx - ox), min(s.shape[1], ny - oy)
        block = np.where(s[:w, :h], L_MAX, L_MIN).astype(np.int16)
        # Solid deep inside walls was never seen: leave it unknown.
        inner = ndimage.binary_erosion(s[:w, :h], iterations=2)
        block[inner] = 0
        self.lo[ox:ox + w, oy:oy + h] = block
        self.version += 1

    def view(self, pose, confidence=1.0, scan=None):
        return mv.MapView(self.lo > L_OCC, self.lo != 0, self.res, self.origin,
                          tuple(pose), confidence, self.version, scan)

    def known_free(self):
        return (self.lo < 0)


class PoseError:
    """The difference between where the truck is and where the navigator
    thinks it is: a slow wander, a couple of centimetres and half a degree,
    plus a centimetre of jitter. What a SLAM that is tracking gives."""

    def __init__(self, rng, xy_sigma=0.02, yaw_sigma=math.radians(0.5),
                 jitter=0.008, corr_s=3.0, dt=0.1):
        self.rng = rng
        self.a = math.exp(-dt / corr_s)
        self.b = math.sqrt(1 - self.a * self.a)
        self.xy_sigma, self.yaw_sigma, self.jitter = xy_sigma, yaw_sigma, jitter
        self.e = np.zeros(3)

    def apply(self, pose):
        s = np.array([self.xy_sigma, self.xy_sigma, self.yaw_sigma])
        self.e = self.a * self.e + self.b * s * self.rng.standard_normal(3)
        j = np.array([self.jitter, self.jitter, 0.0]) * self.rng.standard_normal(3)
        p = np.asarray(pose) + self.e + j
        return (float(p[0]), float(p[1]), float(p[2]))
