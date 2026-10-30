"""The SLAM map on disk: a record of the run, and where the next boot
starts only if asked.

Nothing kept the map before this: it lived in the mapping worker's memory
and nowhere else. 10-29 at 21:39 the truck went over, was carried 3.3 m and
turned round by hand, lost its place on a 20 minute map and shelved it; the
Pi came back up at 21:40, and the map was gone with the process, though it
had been sitting on the shelf, whole, until then. The patrol's waypoints
came back (autopilot/patrol.py Store) and had no map to be on.

What is kept is one map, the bigger of the current one and the one on the
shelf (by writes: slam.Slam.snapshot), with where the truck is reckoned to
be on it. After a loss the shelf holds the map worth having and the
current one is a few seconds old, so that is the one saved. The runner
takes a snapshot every SAVE_EVERY_S while the map or the pose has moved on,
and once more on a clean stop, and hands it to a Saver, which writes it on
a thread of its own: the snapshot is only the observed cells copied out
(Grid.bounds), a few ms; the compressing and writing are on the thread.

Nothing loads it at boot unless asked (--slam-map-restore): every boot
starts a fresh map. On 10-30 the restored map came back at every restart:
the truck booted elsewhere, and the map showed rooms it had not driven
that run, and at 11:33 a heading 24 degrees off was written into it, and
that came back at 11:35 and 11:47 too. So the
file is a record of the last run, kept for looking at: at boot the runner
moves it to slam_map.prev.npz (set_aside()) before the first save of the
new run goes over it, and a restart leaves the run before it to look at.

Asked to (restore=True on the runner), the runner loads it (load()), and
the Slam starts lost on it (Slam.restore): the scan in hand is searched for
round where the truck was (reloc.py), and if it is not found inside
slam.RESEED_LOST_S the restored map goes on the shelf as any lost map does
and is searched for from the fresh map, merged back when it is found.

Never in the way of SLAM starting. A file that is missing, too big, of
another version, cut short or not what it says it is, is ignored with a
reason (load() returns (None, why)) and the map starts from nothing as it
always did. Written to a temporary file in the same directory, flushed to
the card and renamed over the old one, so a power cut leaves the old file
or the new, never half of one.

The format is a numpy .npz (np.load with allow_pickle off, so loading runs
no code from the file): the four cell arrays of the observed window and a
"meta" entry, UTF-8 JSON, with FORMAT and VERSION in it. The default path
is log/slam_map.npz, which is gitignored with the rest of log/: a map is a
floor plan of the house and never goes in the repo.
"""

import io
import json
import math
import os
import threading
import time
from typing import NamedTuple, Optional

import numpy as np

from crawler.mapping import grid as gr

FORMAT = "crawler-slam-map"
VERSION = 1
# Bigger than this is not a map this code wrote: 4 bytes a cell over the
# whole of the largest window (grid.MAX_SIZE square) is 64 MiB before it is
# compressed, and a house compresses ten to one.
MAX_BYTES = 80 * 1024 * 1024
# How often the runner saves while the map or the pose has changed. A save
# of the 10-29 house map (135 x 302 cells observed, 9453 writes) is 33 KB
# and 3 ms of compressing on the Mac.
SAVE_EVERY_S = 10.0
# The pose has moved on for a save if it has gone this far.
MOVED_M = 0.25
MOVED_DEG = 10.0
_ARRAYS = ("lo", "fx", "fy", "cnt")
_DTYPES = {"lo": np.int8, "fx": np.uint8, "fy": np.uint8, "cnt": np.uint8}


class Snapshot(NamedTuple):
    """One map as it was, cut down to the cells it has observed."""
    size: int             # the grid's window, cells a side
    max_size: int
    res: float            # metres a cell
    l_max: int
    ox: int               # the window's origin, world cells
    oy: int
    i0: int               # where the arrays sit in the window
    j0: int
    lo: np.ndarray
    fx: np.ndarray
    fy: np.ndarray
    cnt: np.ndarray
    pose: tuple           # (x, y, yaw) where the truck is reckoned to be on it
    writes: int           # scans written into it
    saved_at: float = 0.0     # time.time() when it was taken

    def grid(self):
        """A Grid with these cells where they were, everything else
        unknown."""
        g = gr.Grid(size=self.size, res=self.res, l_max=self.l_max,
                    max_size=self.max_size)
        g.ox, g.oy = int(self.ox), int(self.oy)
        n, m = self.lo.shape
        for name in _ARRAYS:
            getattr(g, name)[self.i0:self.i0 + n, self.j0:self.j0 + m] = \
                getattr(self, name)
        g.version = 1
        return g


def snapshot_of(grid, pose, writes, saved_at=None):
    """A Snapshot of `grid` (copied, so the grid can go on changing), or None
    if it has observed nothing."""
    b = grid.bounds()
    if b is None:
        return None
    i0, i1, j0, j1 = b
    cut = {name: getattr(grid, name)[i0:i1, j0:j1].copy() for name in _ARRAYS}
    return Snapshot(size=int(grid.size), max_size=int(grid.max_size),
                    res=float(grid.res), l_max=int(grid.l_max),
                    ox=int(grid.ox), oy=int(grid.oy), i0=int(i0), j0=int(j0),
                    pose=tuple(float(v) for v in pose), writes=int(writes),
                    saved_at=float(time.time() if saved_at is None
                                   else saved_at), **cut)


def encode(snap):
    """The file's bytes for a Snapshot."""
    meta = {"format": FORMAT, "version": VERSION, "size": snap.size,
            "max_size": snap.max_size, "res": snap.res, "l_max": snap.l_max,
            "ox": snap.ox, "oy": snap.oy, "i0": snap.i0, "j0": snap.j0,
            "pose": list(snap.pose), "writes": snap.writes,
            "saved_at": snap.saved_at}
    buf = io.BytesIO()
    np.savez_compressed(
        buf, meta=np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8),
        **{name: getattr(snap, name) for name in _ARRAYS})
    return buf.getvalue()


def save(path, snap, max_bytes=MAX_BYTES):
    """Write a Snapshot to `path`: a temporary file beside it, flushed to
    disk, renamed over it. Raises on any failure, and leaves whatever was
    at `path` as it was."""
    data = encode(snap)
    if len(data) > max_bytes:
        raise ValueError(f"the map is {len(data)} bytes, over {max_bytes}")
    path = os.fspath(path)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:
        # The rename itself, to the card: without it a power cut can come
        # back to the directory as it was before.
        fd = os.open(folder, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass
    return len(data)


def load(path, max_bytes=MAX_BYTES):
    """(Snapshot, "") from `path`, or (None, why not). Never raises."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return None, "no map file"
    if size > max_bytes:
        return None, f"the file is {size} bytes, over {max_bytes}"
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            if not isinstance(meta, dict) or meta.get("format") != FORMAT:
                return None, "not a map file"
            if meta.get("version") != VERSION:
                return None, f"version {meta.get('version')!r}, not {VERSION}"
            arrays = {name: z[name] for name in _ARRAYS}
        return _checked(meta, arrays), ""
    except Exception as e:
        return None, f"unreadable: {type(e).__name__}: {e}"


def previous_path(path):
    """Where the last run's map is put aside at boot: slam_map.npz ->
    slam_map.prev.npz."""
    root, ext = os.path.splitext(os.fspath(path))
    return f"{root}.prev{ext or '.npz'}"


def set_aside(path):
    """Move the last run's map file to previous_path() before this run's
    first save goes over it. The path it went to, or None if there was no
    file. Never raises."""
    if not os.path.exists(path):
        return None
    prev = previous_path(path)
    try:
        os.replace(os.fspath(path), prev)
    except OSError:
        return None
    return prev


def _checked(meta, arrays):
    """The Snapshot `meta` and `arrays` describe, or ValueError if any of it
    does not hold together."""
    size, max_size = int(meta["size"]), int(meta["max_size"])
    res, l_max = float(meta["res"]), int(meta["l_max"])
    i0, j0 = int(meta["i0"]), int(meta["j0"])
    pose = tuple(float(v) for v in meta["pose"])
    writes = int(meta["writes"])
    if not 0 < size <= max_size <= gr.MAX_SIZE:
        raise ValueError(f"a {size} cell window, at most {max_size}")
    if not (math.isfinite(res) and res > 0):
        raise ValueError(f"{res} m cells")
    if not 0 < l_max <= 127:
        raise ValueError(f"a clamp of {l_max}")
    if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
        raise ValueError(f"the pose {pose}")
    shape = arrays["lo"].shape
    if len(shape) != 2:
        raise ValueError(f"cells of shape {shape}")
    for name in _ARRAYS:
        a = arrays[name]
        if a.shape != shape or a.dtype != _DTYPES[name]:
            raise ValueError(f"{name} is {a.dtype} {a.shape}")
    if not (0 <= i0 and 0 <= j0 and i0 + shape[0] <= size
            and j0 + shape[1] <= size):
        raise ValueError("the cells are outside the window")
    return Snapshot(size=size, max_size=max_size, res=res, l_max=l_max,
                    ox=int(meta["ox"]), oy=int(meta["oy"]), i0=i0, j0=j0,
                    pose=pose, writes=writes,
                    saved_at=float(meta.get("saved_at", 0.0)), **arrays)


def moved(a, b, m=MOVED_M, deg=MOVED_DEG):
    """Whether pose b has gone further than m metres or deg degrees from a."""
    dyaw = (b[2] - a[2] + math.pi) % (2 * math.pi) - math.pi
    return (math.hypot(b[0] - a[0], b[1] - a[1]) > m
            or abs(dyaw) > math.radians(deg))


class Saver:
    """Writes the newest Snapshot handed to offer() on a thread of its own:
    one waiting at most, a newer one replacing it. flush() waits for the
    one in hand. Failures are counted (errors, last_error), never raised."""

    def __init__(self, path):
        self.path = os.fspath(path)
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._pending: Optional[Snapshot] = None
        self._busy = False
        self._thread = None
        self.saves = 0
        self.errors = 0
        self.last_error = ""
        self.last_bytes = 0
        self.last_ms = 0.0

    def offer(self, snap):
        with self._lock:
            self._pending = snap
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop,
                                                name="slam-map-save",
                                                daemon=True)
                self._thread.start()
            self._idle.notify_all()

    def flush(self, timeout=2.0):
        """Wait until nothing is waiting or being written, up to `timeout`
        seconds. True if it all got written (or failed) in time."""
        end = time.monotonic() + timeout
        with self._lock:
            while self._pending is not None or self._busy:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._idle.wait(left)
        return True

    def _loop(self):
        while True:
            with self._lock:
                while self._pending is None:
                    self._idle.wait()
                snap, self._pending = self._pending, None
                self._busy = True
            try:
                t0 = time.perf_counter()
                self.last_bytes = save(self.path, snap)
                self.last_ms = (time.perf_counter() - t0) * 1000.0
                self.saves += 1
            except Exception as e:
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {e}"
            with self._lock:
                self._busy = False
                self._idle.notify_all()
