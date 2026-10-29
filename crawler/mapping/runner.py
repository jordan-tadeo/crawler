"""SLAM beside the control loop, never in its way.

The lidar thread hands every scan to on_scan() and the camera's reader thread
hands every IMU sample to on_imu(). Both return in microseconds: on_imu
appends to a deque, on_scan puts the scan's 720 ranges and the IMU samples
since the last scan into an inbox INBOX deep with put_nowait. A worker takes
them out and runs Slam.step(). When the worker is behind and the inbox is
full the scan is dropped and counted (stats()["dropped"]), and its IMU
samples go out with the next one, so the heading integral has no hole in it.

The worker is a thread or a process (MODE_THREAD, MODE_PROCESS). Both run
the same _work() loop over the same two queues; only the queue and the
worker type differ. A process has its own interpreter and its own GIL, so
nothing it does can stretch a control-loop tick; a thread shares the GIL,
but every step is mostly numpy and scipy, which let go of it. Which one is
the default is a measurement, see tools/slam_bench.py --jitter.

What comes back (the outbox) is read by a pump thread here, which keeps the
newest Result, the newest MapView and the path, and offers the Foxglove
messages: /slam/status every scan, /tf (map -> base_link) every scan, /map
and /slam/path at their INTERVALS. The pump never blocks anything either:
telemetry's offer() is drop-oldest.

Nothing here commands the truck. latest(), pose() and map() are the reads a
planner will make; see crawler/mapping/__init__.py.
"""

import math
import queue
import threading
import time
from collections import deque
from typing import NamedTuple, Optional

import numpy as np

MODE_THREAD = "thread"
MODE_PROCESS = "process"
MODES = (MODE_THREAD, MODE_PROCESS)

# Two scans in hand at most. A third waiting would be 200 ms old by the
# time it was matched, and a map 200 ms behind is no use to anybody.
INBOX = 2
# Results and map views the pump has not read yet. It reads them in
# microseconds; this only matters if it dies.
OUTBOX = 64
# IMU samples held between scans: 88 a second live, so two seconds' worth.
IMU_KEEP = 200

# The /map picture: once a second, 10 cm pixels (two cells), at most
# MAP_VIEW_HALF_M either side of the truck. 400 x 400 RGBA is 640 KB before
# zstd, which takes the unknown and free runs down to almost nothing.
MAP_EVERY_S = 1.0
MAP_STEP = 2
MAP_VIEW_HALF_M = 20.0

# The navigator's map (nav_map(), only with nav_half_m set, i.e. --nav): the
# grid's own 5 cm cells, NAV_HALF_M either side of the truck, twice a
# second. 320 x 320 bytes is 100 KB a message. The /map picture's 10 cm
# pixels mark a whole pixel solid for one cell in it, which grows every wall
# by up to a cell and narrows a doorway by two.
NAV_EVERY_S = 0.5
NAV_HALF_M = 8.0

# The path: a point every PATH_STEP_M moved, PATH_MAX of them, halved when
# full so the oldest part of a long drive gets coarser instead of vanishing.
# A Point3 is three doubles on the wire, and the live view is not compressed:
# at 5 cm and 4000 points the path alone was 19 KiB/s over a 672 s drive.
PATH_STEP_M = 0.10
PATH_MAX = 2000

# The lidar's own /tf (base_link -> lidar only) stands back while a SLAM
# result has come in within this long, and takes over again if they stop.
OWN_TF_S = 1.0


# Poses kept for pose_at(): ten seconds of scans.
POSE_KEEP = 100
# How far past the newest scan pose_at() still answers with it: a depth
# frame arrives before the scan that covers its moment has been matched.
POSE_MAX_AHEAD_S = 0.25
# constraint (matcher.Fit.constraint, 0..0.5) at and above which a scene
# pins the pose in both directions; below it the confidence is scaled down.
# On the four indoor drives the median tracking scan was 0.37-0.42 and the
# 5th percentile 0.11-0.24, so indoors it is almost always 1.
CONSTRAINT_FULL = 0.15


class PoseStamp(NamedTuple):
    """map -> truck (base_link) at one moment, and how far to trust it.

    t is time.monotonic() seconds, the same clock the camera hub stamps its
    frames with: the scan's de-jittered end time. x, y metres and yaw
    radians CCW, in the map frame (the truck's boot pose is the origin,
    facing +x). confidence 0..1, see runner.confidence(); 0 means do not
    place anything by this pose. state is slam.Result.state."""
    t: float
    x: float
    y: float
    yaw: float
    confidence: float
    state: str


class MapView(NamedTuple):
    """A cropped, downsampled picture of the map, cheap to send anywhere."""
    t: float              # the scan time it is as of
    version: int          # Grid.version then
    x0: float             # map metres of pixel (0, 0)'s outer corner
    y0: float
    res: float            # metres per pixel
    codes: np.ndarray     # uint8 [nx, ny], grid.UNKNOWN / FREE / OCCUPIED


def ranges_of(scan, mount):
    """A lidar.Scan as the 720 lidar-frame ranges Slam.step() takes: the
    same slots /lidar/scan records, so a replay sees what the live run saw."""
    from crawler.telemetry import lidar_view
    return lidar_view.scan_slots(scan.angles_deg, scan.distances_mm,
                                 mount).astype(np.float32)


def imu_row(t, sample):
    """A camera get_imu_sample() dict as (t, gx, gy, gz, ux, uy, uz), the up
    vector NaN when the sample has none. None if it has no gyro."""
    try:
        gx, gy, gz = (float(v) for v in sample["gyro"])
    except Exception:
        return None
    up = sample.get("up") if hasattr(sample, "get") else None
    try:
        ux, uy, uz = (float(v) for v in up)
    except (TypeError, ValueError):
        ux = uy = uz = float("nan")
    return (float(t), gx, gy, gz, ux, uy, uz)


def map_view(slam, t, step=MAP_STEP, half_m=MAP_VIEW_HALF_M):
    """The MapView of slam.grid round slam.pose, or None for an empty map."""
    g = slam.grid
    b = g.bounds()
    if b is None:
        return None
    x, y = float(slam.pose[0]), float(slam.pose[1])
    ci, cj = g.world_to_cell(x, y)
    h = int(half_m / g.res)
    i0, i1 = max(b[0], int(ci) - h), min(b[1], int(ci) + h)
    j0, j1 = max(b[2], int(cj) - h), min(b[3], int(cj) + h)
    if i1 <= i0 or j1 <= j0:
        return None
    out = g.codes(step, (i0, i1, j0, j1))
    if out is None:
        return None
    codes, x0, y0 = out
    return MapView(t=float(t), version=int(g.version), x0=float(x0),
                   y0=float(y0), res=g.res * step, codes=codes)


def _work(inbox, outbox, l_max, map_every_s, nav_half_m=None,
          nav_every_s=NAV_EVERY_S):
    """The worker, thread or process: scans in, Results and MapViews out.
    None in the inbox ends it. With nav_half_m, a full-resolution MapView
    for the navigator too, every nav_every_s."""
    from crawler.mapping.clock import SensorClock
    from crawler.mapping.egomap import EgoMap
    from crawler.mapping.imu import ImuTrack
    from crawler.mapping.slam import Slam

    imu = ImuTrack()
    kw = {} if l_max is None else {"l_max": int(l_max)}
    slam = Slam(imu=imu, **kw)
    # The policy's crop of the map, drawn here after every step, on the
    # grid the step just wrote: the same EgoMap tools/map_crops.py runs
    # over a recording, so training and the car see the same picture.
    ego = EgoMap()
    clock = SensorClock()
    next_map = 0.0
    next_nav = 0.0
    lost_out = 0
    grid_seen = (0, 0)
    while True:
        item = inbox.get()
        if item is None:
            return
        host_t, stamp_ms, ranges, rows = item
        for r in rows:
            up = r[4:7]
            imu.add(r[0], r[1:4], None if math.isnan(up[0]) else up)
        t = clock.fit(host_t, stamp_ms)
        try:
            res = slam.step(t, ranges)
        except Exception as e:      # a bad scan costs that scan
            try:
                outbox.put_nowait(("error", repr(e)))
            except queue.Full:
                pass
            continue
        try:
            outbox.put_nowait(("r", res, lost_out))
        except queue.Full:
            lost_out += 1
        g = slam.grid
        if (g.grew, g.dropped) != grid_seen:
            grid_seen = (g.grew, g.dropped)
            # Waits up to a second for room: this happens a handful of
            # times a drive and a dropped scroll must not go unsaid.
            try:
                outbox.put(("grid", g.grew, g.dropped, g.size), timeout=1.0)
            except queue.Full:
                pass
        try:
            outbox.put_nowait(("crop", ego.after_step(slam.grid, res)))
        except queue.Full:
            pass
        except Exception as e:      # a crop that fails costs that crop
            try:
                outbox.put_nowait(("error", repr(e)))
            except queue.Full:
                pass
        if host_t >= next_map:
            next_map = host_t + map_every_s
            view = map_view(slam, t)
            if view is not None:
                try:
                    outbox.put_nowait(("map", view))
                except queue.Full:
                    pass
        if nav_half_m is not None and host_t >= next_nav:
            next_nav = host_t + nav_every_s
            view = map_view(slam, t, step=1, half_m=nav_half_m)
            if view is not None:
                try:
                    outbox.put_nowait(("nav", view))
                except queue.Full:
                    pass


class SlamRunner:
    """Scans and IMU in, the newest pose and map out, on a worker of its own.

    start() returns at once and close() never raises. Every public method
    is safe to call from any thread and none of them waits on the worker.
    """

    def __init__(self, mode=MODE_THREAD, tele=None, logger=None, mount=None,
                 l_max=None, inbox=INBOX, map_every_s=MAP_EVERY_S,
                 clock=time.monotonic, nav_half_m=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        if mount is None:
            from crawler.hardware import lidar as ld
            mount = ld.MOUNT
        self.mode = mode
        self.tele = tele
        self.logger = logger
        self.mount = mount
        self.l_max = l_max
        self.inbox_size = inbox
        self.map_every_s = map_every_s
        # --nav's full-resolution map, NAV_HALF_M round the truck; None, the
        # default, never draws it.
        self.nav_half_m = nav_half_m
        self._clock = clock

        self._imu = deque(maxlen=IMU_KEEP)
        self._imu_lock = threading.Lock()
        self._inbox = None
        self._outbox = None
        self._worker = None
        self._pump = None
        self._stop = threading.Event()

        self.scans = 0          # handed to on_scan
        self.dropped = 0        # of those, not matched: the worker was behind
        self.results = 0
        self.errors = 0
        self.imu_samples = 0
        self.grid_grew = 0
        self.grid_dropped = 0
        self._latest = None
        self._latest_at = None
        self._map = None
        self._nav_map = None
        self._crop = None
        self._path = []
        self._path_version = 0
        self._poses = deque(maxlen=POSE_KEEP)
        self._poses_lock = threading.Lock()
        self._said_error = False

    # ---- lifecycle ---------------------------------------------------------

    def start(self):
        if self._worker is not None:
            return
        self._stop.clear()
        args = None
        if self.mode == MODE_PROCESS:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            self._inbox = ctx.Queue(maxsize=self.inbox_size)
            self._outbox = ctx.Queue(maxsize=OUTBOX)
            args = (self._inbox, self._outbox, self.l_max, self.map_every_s,
                    self.nav_half_m)
            self._worker = ctx.Process(target=_work, args=args, name="slam",
                                       daemon=True)
        else:
            self._inbox = queue.Queue(maxsize=self.inbox_size)
            self._outbox = queue.Queue(maxsize=OUTBOX)
            args = (self._inbox, self._outbox, self.l_max, self.map_every_s,
                    self.nav_half_m)
            self._worker = threading.Thread(target=_work, args=args,
                                            name="slam", daemon=True)
        self._worker.start()
        self._pump = threading.Thread(target=self._pump_loop,
                                      name="slam-pump", daemon=True)
        self._pump.start()

    def close(self):
        """Stop the worker and the pump. Never raises, waits a second at most."""
        self._stop.set()
        worker, self._worker = self._worker, None
        if worker is None:
            return
        try:
            # Make room for the None: whatever is waiting is thrown away.
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
        pump, self._pump = self._pump, None
        if pump is not None:
            pump.join(timeout=1.0)
        if self.mode == MODE_PROCESS:
            for q in (self._inbox, self._outbox):
                try:
                    q.cancel_join_thread()
                    q.close()
                except Exception:
                    pass

    # ---- the producers' half -------------------------------------------------

    def on_imu(self, t, sample):
        """One IMU sample, from the camera's reader thread."""
        row = imu_row(t, sample)
        if row is None:
            return
        with self._imu_lock:
            self._imu.append(row)
            self.imu_samples += 1

    def on_scan(self, scan):
        """One revolution, from the lidar's thread. Never blocks."""
        if self._worker is None:
            return
        self.scans += 1
        try:
            ranges = ranges_of(scan, self.mount)
        except Exception:
            self.dropped += 1
            return
        with self._imu_lock:
            rows = list(self._imu)
            self._imu.clear()
        try:
            self._inbox.put_nowait(
                (float(scan.t), getattr(scan, "stamp_ms", None), ranges, rows))
        except queue.Full:
            self.dropped += 1
            # The samples are not the scan's to lose: they go with the next.
            with self._imu_lock:
                self._imu.extendleft(reversed(rows))
        except Exception:
            self.dropped += 1

    # ---- the readers' half ---------------------------------------------------

    def pose_stamped(self):
        """The newest PoseStamp, or None before the first scan."""
        with self._poses_lock:
            return self._poses[-1] if self._poses else None

    def pose_at(self, t, max_ahead=POSE_MAX_AHEAD_S):
        """The PoseStamp at monotonic time `t`, interpolated between the two
        scans either side of it, for placing something seen at `t` (a depth
        frame) in the map. Past the newest scan by up to `max_ahead` it is
        the newest, as is; further ahead, or older than the last POSE_KEEP
        scans, or across a jump in state, None. The confidence is the lower
        of the two scans'."""
        with self._poses_lock:
            poses = list(self._poses)
        return interpolate(poses, t, max_ahead)

    def latest(self):
        """The newest slam.Result, or None before the first."""
        return self._latest

    def pose(self):
        """(x, y, yaw) in the map frame from the newest Result, or None."""
        r = self._latest
        return None if r is None else (r.x, r.y, r.yaw)

    def crop(self):
        """The newest egomap.Crop, drawn after the newest step, or None."""
        return self._crop

    def map(self) -> Optional[MapView]:
        """The newest MapView, at most MAP_EVERY_S old, or None."""
        return self._map

    def nav_map(self) -> Optional[MapView]:
        """The newest full-resolution MapView round the truck, or None:
        always None unless built with nav_half_m."""
        return self._nav_map

    def path(self):
        """The (x, y) points driven so far, a copy."""
        return list(self._path)

    def owns_tf(self, now=None) -> bool:
        """True while results are coming in: the lidar leaves /tf to me."""
        at = self._latest_at
        if at is None:
            return False
        now = self._clock() if now is None else now
        return now - at <= OWN_TF_S

    def stats(self) -> dict:
        w = self._worker
        return {"mode": self.mode, "scans": self.scans,
                "dropped": self.dropped, "results": self.results,
                "errors": self.errors, "imu_samples": self.imu_samples,
                "grid_grew": self.grid_grew,
                "grid_dropped": self.grid_dropped,
                "alive": bool(w is not None and w.is_alive())}

    # ---- the pump --------------------------------------------------------------

    def _pump_loop(self):
        while not self._stop.is_set():
            try:
                item = self._outbox.get(timeout=0.2)
            except queue.Empty:
                continue
            except Exception:
                return
            try:
                self._take(item)
            except Exception:
                pass

    def _take(self, item):
        kind = item[0]
        now = self._clock()
        if kind == "r":
            res = item[1]
            self._latest = res
            self._latest_at = now
            with self._poses_lock:
                self._poses.append(pose_stamp(res))
            self.results += 1
            self._extend_path(res)
            self._offer_result(res, now)
        elif kind == "crop":
            self._crop = item[1]
        elif kind == "map":
            self._map = item[1]
            tele = self.tele
            if tele is not None and tele.wants("map", now):
                tele.offer("map", item[1], now)
        elif kind == "nav":
            self._nav_map = item[1]
        elif kind == "grid":
            _, grew, dropped, size = item
            was = self.grid_dropped
            self.grid_grew, self.grid_dropped = grew, dropped
            self._say("error" if dropped > was else "info",
                      f"The map is {size * 0.05:.0f} m a side and lost cells "
                      f"to a scroll ({dropped} so far)" if dropped > was else
                      f"The map grew to {size * 0.05:.0f} m a side")
        elif kind == "error":
            self.errors += 1
            if not self._said_error and self.logger is not None:
                self._said_error = True
                try:
                    self.logger.log("warning", "SLAM",
                                    f"A scan failed to match: {item[1]}")
                except Exception:
                    pass

    def _say(self, level, message):
        if self.logger is None:
            return
        try:
            self.logger.log(level, "SLAM", message)
        except Exception:
            pass

    def _extend_path(self, res):
        if res.state in ("rolled", "no_points"):
            return
        p = self._path
        if p and math.hypot(res.x - p[-1][0], res.y - p[-1][1]) < PATH_STEP_M:
            return
        p.append((res.x, res.y))
        if len(p) > PATH_MAX:
            self._path = p[::2]
        self._path_version += 1

    def _offer_result(self, res, now):
        tele = self.tele
        if tele is None:
            return
        try:
            if tele.wants("slam_status", now):
                tele.offer("slam_status", status_row(res, self.stats()), now)
            if tele.wants("tf", now):
                tele.offer("tf", (self.mount, (res.x, res.y, res.yaw), res.t),
                           now)
            if tele.wants("slam_path", now):
                tele.offer("slam_path", self.path(), now)
        except Exception:
            pass


def confidence(res) -> float:
    """0..1, how far to trust a Result's pose: 0 unless it matched and is
    tracking or tilted (tilted is matched but not written, so half). Then
    the fit's inlier fraction, scaled down where the scene does not pin the
    pose in every direction (constraint under CONSTRAINT_FULL: a corridor,
    one wall, an open space)."""
    if not res.matched or res.state not in ("tracking", "tilted"):
        return 0.0
    c = max(0.0, min(1.0, float(res.inliers)))
    c *= max(0.0, min(1.0, float(res.constraint) / CONSTRAINT_FULL))
    if res.state == "tilted":
        c *= 0.5
    return round(c, 3)


def pose_stamp(res) -> "PoseStamp":
    return PoseStamp(t=res.t, x=res.x, y=res.y, yaw=res.yaw,
                     confidence=confidence(res), state=res.state)


def interpolate(poses, t, max_ahead=POSE_MAX_AHEAD_S):
    """pose_at() over a time-ordered list of PoseStamps."""
    if not poses:
        return None
    last = poses[-1]
    if t >= last.t:
        return last if t - last.t <= max_ahead else None
    if t < poses[0].t:
        return None
    lo, hi = 0, len(poses) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if poses[mid].t <= t:
            lo = mid
        else:
            hi = mid
    a, b = poses[lo], poses[hi]
    if a.state != b.state or b.t - a.t > 0.5:
        return None
    f = (t - a.t) / (b.t - a.t) if b.t > a.t else 0.0
    dyaw = (b.yaw - a.yaw + math.pi) % (2 * math.pi) - math.pi
    return PoseStamp(t=float(t), x=a.x + f * (b.x - a.x),
                     y=a.y + f * (b.y - a.y), yaw=a.yaw + f * dyaw,
                     confidence=min(a.confidence, b.confidence),
                     state=a.state)


def status_row(res, stats) -> dict:
    """A Result and the runner's counts as a /slam/status row."""
    def num(v, nd=4):
        v = float(v)
        return round(v, nd) if math.isfinite(v) else -1.0
    return {
        "state": res.state,
        "x": num(res.x), "y": num(res.y),
        "yaw_deg": num(math.degrees(res.yaw), 2),
        "matched": bool(res.matched), "written": bool(res.written),
        "why": res.why,
        "inliers": num(res.inliers, 3),
        "rms_mm": num(res.rms_m * 1000.0, 2),
        "constraint": num(res.constraint, 4),
        "returns": int(res.returns), "points": int(res.points),
        "ground_dropped": int(res.ground_dropped),
        "tilt_deg": num(res.tilt_deg, 2),
        "pitch_deg": num(res.pitch_deg, 2), "roll_deg": num(res.roll_deg, 2),
        "held": int(res.held), "banded": bool(res.banded),
        "iters": int(res.iters), "ms": num(res.ms, 2),
        "scans": int(stats.get("scans", 0)),
        "dropped": int(stats.get("dropped", 0)),
        "errors": int(stats.get("errors", 0)),
    }
