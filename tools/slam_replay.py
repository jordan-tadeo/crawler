"""Run the car's own SLAM over a recording, scan by scan, and grade it.

    uv run --with mcap==1.6.0 python tools/slam_replay.py RUN.mcap --out DIR

The same crawler.mapping.slam.Slam the car runs with --slam, fed the same
inputs in the same order: every IMU sample up to a scan's log time, then
the scan. Nothing from later in the file reaches a step, so what it does here
is what it would have done live.

The IMU is /imu/batch when the file has it: every sample the car's SLAM was
handed, about 88 a second, each at its own time. A file from before that
channel has only /imu, gated to about 16 a second, and a replay of one is
coarser than the live run was -- by enough to change a verdict on anything
that compares the scan heading to the gyro.

The scan's time is the best one the file has: the lidar's own packet stamp
through clock.SensorClock when /lidar carries it (recordings from this
change on), otherwise the host log time through clock.HostDejitter, or raw
with --raw-time.

Writes into DIR (never into the repo: these are maps of a house):

  <run>.png          the final map, free white, walls black, unknown grey,
                     the path over it in colour by time
  <run>.csv          t, x, y, yaw, state, written, inliers, rms, constraint
  <run>.json         the metrics below
  <run>.npz          with --export: the inputs, for tools/slam_bench.py

The metrics, the same definitions the prototype was graded on:

  fail_frac          scans with no good fit (or too few points)
  jump_frac          scans that moved over 15 cm in under 0.2 s
  rms_mm_med         point-to-line residual of the good fits
  yaw_rms10_med_deg  per 10 s window, RMS of (scan heading change - IMU
                     heading change), with one gyro bias per drive fitted
                     offline
  wall_mm_med        how thick the accumulated walls are: the spread across
                     the best local line of the 20 nearest points
  revisit_*          drift on coming back: a scan at least 30 s after an
                     earlier one within 1 m is aligned to that one's
                     neighbours; the correction is the drift. ge300 = 300 s
                     or more apart.
  ms_med / ms_p95    the step's own time, this machine

Needs scipy (pyproject.toml) and, to read an MCAP, the `mcap` package.
"""

import argparse
import json
import os
import struct
import sys
import time
import zlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.mapping import clock as ck          # noqa: E402
from crawler.mapping import grid as gr           # noqa: E402
from crawler.mapping import matcher as mt        # noqa: E402
from crawler.mapping import scan as sc           # noqa: E402
from crawler.mapping.imu import ImuTrack         # noqa: E402
from crawler.mapping.slam import Slam            # noqa: E402

SCAN_TOPIC = "/lidar/scan"
LIDAR_TOPIC = "/lidar"
IMU_TOPIC = "/imu"
IMU_BATCH_TOPIC = "/imu/batch"


# ---- reading -----------------------------------------------------------------

def _varint(buf, i):
    shift = value = 0
    while True:
        b = buf[i]
        i += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, i
        shift += 7


def laser_ranges(buf):
    """A foxglove.LaserScan's ranges (field 6, packed doubles), no protobuf."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        num, wire = key >> 3, key & 7
        if wire == 0:
            _, i = _varint(buf, i)
        elif wire == 1:
            i += 8
        elif wire == 2:
            size, i = _varint(buf, i)
            if num == 6:
                return np.frombuffer(bytes(buf[i:i + size]), "<f8")
            i += size
        elif wire == 5:
            i += 4
        else:
            break
    return None


def load_mcap(path):
    """The inputs, in log-time order: scans and IMU rows."""
    from mcap.records import Channel, Message
    from mcap.stream_reader import StreamReader
    channels = {}
    scans, lidar, imu, batch = [], [], [], []
    with open(path, "rb") as fh:
        try:
            for rec in StreamReader(fh).records:
                if isinstance(rec, Channel):
                    channels[rec.id] = rec.topic
                    continue
                if not isinstance(rec, Message):
                    continue
                topic = channels.get(rec.channel_id)
                t = rec.log_time * 1e-9
                if topic == SCAN_TOPIC:
                    r = laser_ranges(rec.data)
                    if r is not None and len(r) == sc.SLOTS:
                        scans.append((t, r.astype(np.float32)))
                elif topic == LIDAR_TOPIC:
                    d = json.loads(rec.data)
                    lidar.append((t, d.get("stamp_ms"), d.get("rpm")))
                elif topic == IMU_TOPIC:
                    d = json.loads(rec.data)
                    imu.append((t, d["gyro_x"], d["gyro_y"], d["gyro_z"],
                                d.get("up_x", 0.0), d.get("up_y", 0.0),
                                d.get("up_z", 0.0)))
                elif topic == IMU_BATCH_TOPIC:
                    batch.extend(imu_batch_rows(json.loads(rec.data)))
        except Exception as e:  # a file cut off by the battery coming out
            print(f"[replay] stopped reading at: {e}", file=sys.stderr)
    scans.sort(key=lambda s: s[0])
    t = np.array([s[0] for s in scans])
    ranges = np.array([s[1] for s in scans], np.float32).reshape(-1, sc.SLOTS)
    # /lidar and /lidar/scan of one revolution share a log time.
    stamp = np.full(len(t), -1, np.int64)
    rpm = np.zeros(len(t))
    if lidar:
        lt = {round(a, 6): (b, c) for a, b, c in lidar}
        for k, tk in enumerate(t):
            b, c = lt.get(round(tk, 6), (None, None))
            if b is not None:
                stamp[k] = int(b)
            if c:
                rpm[k] = float(c)
    imu = imu_rows(imu, batch)
    return dict(t=t, ranges=ranges, stamp=stamp, rpm=rpm, imu=imu)


def imu_batch_rows(d):
    """An /imu/batch message as the same (t, gx, gy, gz, ux, uy, uz) rows an
    /imu message makes, one per sample, each at its own time."""
    cols = [d["t"], d["gyro_x"], d["gyro_y"], d["gyro_z"],
            d.get("up_x"), d.get("up_y"), d.get("up_z")]
    n = len(cols[0])
    cols = [c if c is not None and len(c) == n else [0.0] * n for c in cols]
    return list(zip(*cols))


def imu_rows(gated, batch):
    """The IMU a replay is fed, as a time-ordered (n, 7) array: every sample
    off /imu/batch when the file has that channel, the gated /imu otherwise.
    Never the two mixed: /imu is a subset of the same samples."""
    rows = batch if batch else gated
    return np.array(sorted(rows), float).reshape(-1, 7)


def load(path):
    if path.endswith(".npz"):
        z = np.load(path)
        n = len(z["t"])
        return dict(t=z["t"], ranges=z["ranges"],
                    stamp=z["stamp"] if "stamp" in z and z["stamp"].dtype.kind == "i"
                    else np.full(n, -1, np.int64),
                    rpm=z["rpm"] if "rpm" in z else np.zeros(n),
                    imu=z["imu"].reshape(-1, 7))
    return load_mcap(path)


# ---- running -----------------------------------------------------------------

def scan_times(d, raw=False):
    """Each scan's end time: sensor stamp, else host de-jitter, else raw."""
    t = d["t"]
    if raw:
        return t.copy(), "raw"
    if (d["stamp"] >= 0).mean() > 0.9:
        c = ck.SensorClock()
        return np.array([c.fit(a, b if b >= 0 else None)
                         for a, b in zip(t, d["stamp"])]), "sensor"
    h = ck.HostDejitter()
    return np.array([h.fit(a) for a in t]), "host"


def run(d, raw_time=False, use_imu=True, keep_points=True, **slam_kw):
    """Slam over the whole file. Returns the Slam, per-scan arrays, points."""
    imu = ImuTrack()
    slam = Slam(imu=imu, **slam_kw)
    ts, how = scan_times(d, raw_time)
    rows = d["imu"] if use_imu else np.zeros((0, 7))
    k = 0
    out = []
    pts = []
    for i in range(len(ts)):
        log_t = d["t"][i]
        while k < len(rows) and rows[k, 0] <= log_t:
            r = rows[k]
            up = r[4:7]
            imu.add(r[0], r[1:4], up if np.linalg.norm(up) > 0.5 else None)
            k += 1
        res = slam.step(float(ts[i]), d["ranges"][i])
        out.append(res)
        if keep_points:
            p = slam.last_points
            if res.state in ("tracking", "tilted") and res.matched and p is not None:
                wp = p @ mt.rot(res.yaw).T + (res.x, res.y)
                pts.append(wp.astype(np.float32))
            else:
                pts.append(None)
    return slam, out, pts, how


# ---- grading (the prototype's definitions) ----------------------------------

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def imu_yaw(rows):
    """Offline cumulative heading off /imu rows, as the prototype graded."""
    t = rows[:, 0]
    up = rows[:, 4:7]
    n = np.linalg.norm(up, axis=1)
    up = np.where(n[:, None] > 0.5, up / np.maximum(n, 1e-9)[:, None],
                  np.array([-1.0, 0, 0]))
    rate = (rows[:, 1:4] * up).sum(1)
    cum = np.concatenate([[0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
    return t, cum


def yaw_windows(t, yl, yi, imu_t, win=10.0):
    out = []
    t0 = max(t[0], imu_t[0])
    gaps = imu_t[1:][np.diff(imu_t) > 0.3]
    while t0 + win <= min(t[-1], imu_t[-1]):
        m = (t >= t0) & (t < t0 + win) & np.isfinite(yl)
        if m.sum() > 50 and not ((gaps > t0) & (gaps < t0 + win + 0.3)).any():
            dd = (yl[m] - yl[m][0]) - (yi[m] - yi[m][0])
            out.append(np.sqrt(np.mean(dd ** 2)))
        t0 += win
    return np.degrees(np.array(out))


def sharpness(pts, rng):
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    sel = rng.choice(len(pts), min(20000, len(pts)), replace=False)
    dd, j = tree.query(pts[sel], k=20, distance_upper_bound=0.10, workers=1)
    full = np.isfinite(dd).all(1)
    nb = pts[j[full]]
    c = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / 20
    ev = np.linalg.eigvalsh(cov)
    return float(np.median(np.sqrt(np.clip(ev[:, 0], 0, None))) * 1000)


def revisits(t, P, pts_world, ok):
    from scipy.spatial import cKDTree
    idx = np.nonzero(ok)[0]
    if not len(idx):
        return []
    sub = idx[np.minimum(np.searchsorted(t[idx], np.arange(t[idx[0]], t[idx[-1]], 1.0)),
                         len(idx) - 1)]
    sub = np.unique(sub)
    res, used = [], set()
    tr = cKDTree(P[sub, :2])
    for ia in sub:
        for b in tr.query_ball_point(P[ia, :2], 1.0):
            ib = sub[b]
            if t[ib] - t[ia] < 30 or pts_world[ib] is None:
                continue
            key = (int(t[ia] // 20), int(t[ib] // 20))
            if key in used:
                continue
            used.add(key)
            near = [pts_world[k] for k in range(max(ia - 3, 0), min(ia + 4, len(t)))
                    if pts_world[k] is not None]
            if not near:
                continue
            tgt = np.concatenate(near).astype(np.float64)
            if len(tgt) < 50:
                continue
            src_w = pts_world[ib].astype(np.float64)
            src = (src_w - P[ib, :2]) @ mt.rot(-P[ib, 2]).T
            tt = cKDTree(tgt)
            _, jn = tt.query(tgt, k=6, workers=1)
            c = tgt[jn] - tgt[jn].mean(1, keepdims=True)
            _, v = np.linalg.eigh(np.einsum("nki,nkj->nij", c, c))
            tn = v[:, :, 0]
            pose = P[ib].copy()
            for md in [0.5] * 4 + [0.25] * 4 + [0.1] * 8:
                q = src @ mt.rot(pose[2]).T + pose[:2]
                dd, j = tt.query(q, distance_upper_bound=md, workers=1)
                m = np.isfinite(dd)
                if m.sum() < 50:
                    break
                nw = tn[j[m]]
                e = ((q[m] - tgt[j[m]]) * nw).sum(1)
                sp = src[m]
                th = pose[2]
                dq = np.stack([-np.sin(th) * sp[:, 0] - np.cos(th) * sp[:, 1],
                               np.cos(th) * sp[:, 0] - np.sin(th) * sp[:, 1]], 1)
                J = np.stack([nw[:, 0], nw[:, 1], (dq * nw).sum(1)], 1)
                wt = 1 / (1 + (e / 0.03) ** 2)
                pose += -np.linalg.solve((J * wt[:, None]).T @ J + 1e-9 * np.eye(3),
                                         (J * wt[:, None]).T @ e)
            q = src @ mt.rot(pose[2]).T + pose[:2]
            dd, j = tt.query(q, distance_upper_bound=0.08, workers=1)
            frac = np.isfinite(dd).mean()
            if frac < 0.6:
                continue
            res.append(dict(ti=float(t[ia] - t[0]), tj=float(t[ib] - t[0]),
                            gap=float(t[ib] - t[ia]),
                            dxy=float(np.hypot(*(pose[:2] - P[ib, :2]))),
                            dyaw=float(np.degrees(abs(wrap(pose[2] - P[ib, 2]))))))
    return res


def grade(d, out, pts):
    t = np.array([r.t for r in out])
    P = np.array([[r.x, r.y, r.yaw] for r in out])
    ok = np.array([r.matched for r in out])
    st = np.array([r.state for r in out])
    ran = st != "rolled"
    step = np.r_[0, np.hypot(*np.diff(P[:, :2], axis=0).T)]
    dt = np.r_[0.1, np.diff(t)]
    jump = (step > 0.15) & (dt < 0.2)
    rms = np.array([r.rms_m for r in out])
    ms = np.array([r.ms for r in out])
    m = dict(scans=int(len(t)),
             fail_frac=float((~ok[ran][1:]).mean()) if ran.sum() > 1 else 0.0,
             jump_frac=float(jump[ran].mean()),
             rolled_scans=int((~ran).sum()),
             lost_scans=int((st == "lost").sum()),
             written_frac=float(np.mean([r.written for r in out])),
             ground_dropped_frac=float(sum(r.ground_dropped for r in out)
                                       / max(sum(r.points for r in out if r.written), 1)),
             inliers_med=float(np.median([r.inliers for r in out if r.matched])),
             rms_mm_med=float(np.nanmedian(rms[ok]) * 1000),
             rms_mm_p95=float(np.nanpercentile(rms[ok], 95) * 1000),
             constraint_med=float(np.median([r.constraint for r in out if r.matched])),
             constraint_p5=float(np.percentile([r.constraint for r in out if r.matched], 5)),
             ms_med=float(np.median(ms)), ms_p95=float(np.percentile(ms, 95)),
             ms_max=float(ms.max()),
             path_m=float(np.nansum(step[ok])))
    if len(d["imu"]) > 10:
        it, cum = imu_yaw(d["imu"])
        yi = np.interp(t, it, cum)
        yl = np.unwrap(P[:, 2])
        sel = np.isfinite(yl) & (t >= it[0]) & (t <= it[-1])
        b = np.polyfit(t[sel] - t[0], (yi - yl)[sel], 1)[0]
        w = yaw_windows(t, yl, yi - b * (t - t[0]), it)
        if len(w):
            m["yaw_rms10_med_deg"] = float(np.median(w))
            m["yaw_rms10_p95_deg"] = float(np.percentile(w, 95))
            m["yaw_rms10_max_deg"] = float(w.max())
    have = [p for p in pts[::2] if p is not None]
    if have:
        m["wall_mm_med"] = sharpness(np.concatenate(have).astype(np.float64),
                                     np.random.default_rng(0))
    rv = revisits(t, P, pts, ok)
    for name, s in (("revisit_lt300", [x for x in rv if x["gap"] < 300]),
                    ("revisit_ge300", [x for x in rv if x["gap"] >= 300])):
        m[name + "_n"] = len(s)
        if s:
            m[name + "_dxy_med_m"] = float(np.median([x["dxy"] for x in s]))
            m[name + "_dxy_p90_m"] = float(np.percentile([x["dxy"] for x in s], 90))
            m[name + "_dxy_max_m"] = float(np.max([x["dxy"] for x in s]))
            m[name + "_dyaw_med_deg"] = float(np.median([x["dyaw"] for x in s]))
    m["revisits"] = rv
    return m


# ---- pictures, no image library ----------------------------------------------

def write_png(path, rgb):
    """An (h, w, 3) uint8 array as a PNG, with zlib and nothing else."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind, data):
        c = struct.pack(">I", len(data)) + kind + data
        return c + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def map_picture(grid, out):
    got = grid.codes(1)
    if got is None:
        return np.zeros((8, 8, 3), np.uint8)
    codes, x0, y0 = got
    img = np.full(codes.shape + (3,), 150, np.uint8)
    img[codes == gr.FREE] = 255
    img[codes == gr.OCCUPIED] = 0
    n = len(out)
    for k, r in enumerate(out):
        i = int((r.x - x0) / grid.res)
        j = int((r.y - y0) / grid.res)
        if 0 <= i < img.shape[0] and 0 <= j < img.shape[1]:
            f = k / max(n - 1, 1)
            img[i, j] = (int(255 * f), 60, int(255 * (1 - f)))
    # Rows are x, columns y: turn so +x is right and +y is up.
    return np.ascontiguousarray(np.transpose(img, (1, 0, 2))[::-1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", help=".mcap recordings (or --export .npz)")
    ap.add_argument("--out", required=True, help="output directory (not the repo)")
    ap.add_argument("--clamp", type=float, default=None,
                    help="occupied log-odds clamp, in log-odds (default grid.L_MAX)")
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--no-ground-guard", action="store_true")
    ap.add_argument("--pitch", type=float, default=None,
                    help="pitch write gate, degrees (default slam.WRITE_MAX_PITCH_DEG)")
    ap.add_argument("--roll", type=float, default=None,
                    help="roll write gate, degrees (default slam.WRITE_MAX_ROLL_DEG)")
    ap.add_argument("--no-band", action="store_true",
                    help="believe every fit, however far from the prediction")
    ap.add_argument("--raw-time", action="store_true",
                    help="use the host log time as the scan time")
    ap.add_argument("--export", action="store_true",
                    help="also write the inputs as <run>.npz for slam_bench.py")
    ap.add_argument("--tag", default="", help="suffix for the output names")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    kw = {}
    if a.clamp is not None:
        kw["l_max"] = int(round(a.clamp / gr.LO_STEP))
    if a.pitch is not None:
        kw["write_max_pitch"] = a.pitch
    if a.roll is not None:
        kw["write_max_roll"] = a.roll
    if a.no_band:
        kw["band"] = False
    if a.no_ground_guard:
        kw["ground_guard"] = False
    for path in a.inputs:
        name = os.path.basename(path).rsplit(".", 1)[0] + a.tag
        d = load(path)
        if a.export:
            np.savez_compressed(os.path.join(a.out, name + ".npz"), **d)
        t0 = time.perf_counter()
        slam, out, pts, how = run(d, raw_time=a.raw_time,
                                  use_imu=not a.no_imu, **kw)
        wall = time.perf_counter() - t0
        m = grade(d, out, pts)
        m.update(run=name, time_source=how, wall_s=round(wall, 1),
                 grid_MB=slam.grid.nbytes() / 1e6, retries=slam.retries,
                 recoveries=slam.recoveries, scrolls=slam.grid.scrolls,
                 grew=slam.grid.grew, dropped_scrolls=slam.grid.dropped,
                 banded=slam.banded, reseeds=slam.reseeds)
        with open(os.path.join(a.out, name + ".json"), "w") as f:
            json.dump(m, f, indent=1)
        with open(os.path.join(a.out, name + ".csv"), "w") as f:
            f.write("t,x,y,yaw,state,written,inliers,rms_m,constraint,tilt_deg,pitch_deg,roll_deg,held,banded,why\n")
            for r in out:
                f.write(f"{r.t:.4f},{r.x:.4f},{r.y:.4f},{r.yaw:.5f},{r.state},"
                        f"{int(r.written)},{r.inliers:.3f},{r.rms_m:.4f},"
                        f"{r.constraint:.3f},{r.tilt_deg:.2f},{r.pitch_deg:.2f},"
                        f"{r.roll_deg:.2f},{r.held},{int(r.banded)},{r.why}\n")
        write_png(os.path.join(a.out, name + ".png"), map_picture(slam.grid, out))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in m.items() if k != "revisits"}), flush=True)


if __name__ == "__main__":
    main()
