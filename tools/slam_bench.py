"""Time the car's SLAM on whatever machine this runs on. Drives nothing.

Imports crawler.mapping (numpy, scipy) and, for --bandwidth, foxglove-sdk.
Never imports pigpio or anything that touches the ESC or the servo, so it
is safe on the Pi with the service stopped or running.

The input is a recording's scans and IMU rows, exported once on the laptop:

    uv run --with mcap==1.6.0 python tools/slam_replay.py RUN.mcap \\
        --out /tmp/slam --export          # writes /tmp/slam/RUN.npz
    scp /tmp/slam/RUN.npz pi:/tmp/

or an .mcap straight off the car (needs the mcap package).

Three measurements:

  (default)    Slam.step() per scan, back to back on one core: mean, p50,
               p99, max ms. The number that says whether a 10 Hz lidar fits.

  --jitter     A stand-in for the control loop, a 100 Hz thread doing a
               little pure-Python work each tick, run three times: with no
               mapper, with the mapper on a thread, and on a process
               (runner.SlamRunner, fed the scans in real time at their
               recorded spacing). Reports the loop's period p50 / p99 / max
               and how many scans the runner dropped. --seconds caps each.

  --bandwidth  The --slam Foxglove topics (/map, /tf with map, /slam/path,
               /slam/status) for the whole drive, through the SDK's own MCAP
               writer, at the telemetry INTERVALS. Bytes per second each.

    python tools/slam_bench.py /tmp/RUN.npz
    python tools/slam_bench.py /tmp/RUN.npz --jitter --seconds 60
    python tools/slam_bench.py /tmp/RUN.npz --bandwidth --out /tmp/bw.mcap
"""

import argparse
import json
import os
import sys
import threading
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.mapping import clock as ck        # noqa: E402
from crawler.mapping.imu import ImuTrack       # noqa: E402
from crawler.mapping.slam import Slam          # noqa: E402


def load(path):
    if path.endswith(".npz"):
        z = np.load(path)
        n = len(z["t"])
        stamp = (z["stamp"] if "stamp" in z and z["stamp"].dtype.kind == "i"
                 else np.full(n, -1, np.int64))
        return dict(t=z["t"], ranges=z["ranges"], stamp=stamp,
                    imu=z["imu"].reshape(-1, 7))
    import slam_replay
    return slam_replay.load(path)


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def summary(ms):
    ms = np.asarray(ms, float)
    return {"n": int(len(ms)), "mean": round(float(ms.mean()), 2),
            "p50": round(pct(ms, 50), 2), "p99": round(pct(ms, 99), 2),
            "max": round(float(ms.max()), 2)}


# ---- per scan ------------------------------------------------------------------

def per_scan(d, limit=None):
    imu = ImuTrack()
    slam = Slam(imu=imu)
    if (d["stamp"] >= 0).mean() > 0.9:
        c = ck.SensorClock()
        ts = [c.fit(a, b if b >= 0 else None) for a, b in zip(d["t"], d["stamp"])]
    else:
        h = ck.HostDejitter()
        ts = [h.fit(a) for a in d["t"]]
    rows = d["imu"]
    k = 0
    ms = []
    n = len(ts) if limit is None else min(limit, len(ts))
    for i in range(n):
        while k < len(rows) and rows[k, 0] <= d["t"][i]:
            r = rows[k]
            up = r[4:7]
            imu.add(r[0], r[1:4], up if np.linalg.norm(up) > 0.5 else None)
            k += 1
        t0 = time.perf_counter()
        slam.step(float(ts[i]), d["ranges"][i])
        ms.append((time.perf_counter() - t0) * 1000.0)
    return summary(ms)


# ---- jitter --------------------------------------------------------------------

class FakeScan:
    """What runner.on_scan reads off a lidar.Scan, from recorded ranges."""

    def __init__(self, t, ranges, stamp):
        self.t = t
        self.stamp_ms = None if stamp < 0 else int(stamp)
        self.ranges = ranges


def loop_100hz(stop, periods):
    """The control loop's shape: sleep to the next 10 ms tick, do ~0.3 ms of
    Python. Records each period, ms."""
    nxt = time.perf_counter()
    last = nxt
    while not stop.is_set():
        nxt += 0.01
        dt = nxt - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        now = time.perf_counter()
        periods.append((now - last) * 1000.0)
        last = now
        s = 0
        for j in range(1500):
            s += j * j


def jitter(d, mode, seconds):
    from crawler.mapping import runner as rn
    # The recorded ranges go in as they are: skip the scan -> slots step.
    real_ranges_of = rn.ranges_of
    rn.ranges_of = lambda scan, mount: scan.ranges
    r = None
    try:
        if mode is not None:
            r = rn.SlamRunner(mode=mode)
            r.start()
            if mode == rn.MODE_PROCESS:
                time.sleep(3.0)     # the spawn's imports, not the mapping
        stop = threading.Event()
        periods = []
        th = threading.Thread(target=loop_100hz, args=(stop, periods))
        th.start()
        t_rec = d["t"]
        rows = d["imu"]
        k = 0
        start = time.monotonic()
        base = t_rec[0]
        i = 0
        while i < len(t_rec):
            rel = t_rec[i] - base
            if rel > seconds:
                break
            wait = start + rel - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            if r is not None:
                host = start + rel
                while k < len(rows) and rows[k, 0] <= t_rec[i]:
                    q = rows[k]
                    sample = {"gyro": q[1:4]}
                    if np.linalg.norm(q[4:7]) > 0.5:
                        sample["up"] = q[4:7]
                    r.on_imu(start + (q[0] - base), sample)
                    k += 1
                r.on_scan(FakeScan(host, d["ranges"][i], d["stamp"][i]))
            i += 1
        time.sleep(0.5)
        stop.set()
        th.join()
        p = np.asarray(periods[10:])
        out = {"worker": mode or "none", "ticks": int(len(p)),
               "period_p50_ms": round(pct(p, 50), 2),
               "period_p99_ms": round(pct(p, 99), 2),
               "period_max_ms": round(float(p.max()), 2),
               "ticks_over_12ms": int((p > 12.0).sum())}
        if r is not None:
            st = r.stats()
            res = r.latest()
            out.update(scans=st["scans"], dropped=st["dropped"],
                       results=st["results"],
                       last_ms=round(res.ms, 2) if res else None)
        return out
    finally:
        if r is not None:
            r.close()
        rn.ranges_of = real_ranges_of


# ---- bandwidth -----------------------------------------------------------------

def bandwidth(d, out_path):
    import foxglove
    from foxglove import Channel, Schema
    from foxglove.channels import (FrameTransformsChannel, GridChannel,
                                   SceneUpdateChannel)
    from crawler.hardware import lidar as ld
    from crawler.mapping import runner as rn
    from crawler.telemetry import map_view as mv
    from crawler.telemetry import telemetry as tm

    spec = tm.JSON_SCHEMAS["slam_status"]
    writer = foxglove.open_mcap(out_path, allow_overwrite=True)
    chans = {
        "map": GridChannel(topic="/map"),
        "tf": FrameTransformsChannel(topic="/tf"),
        "slam_path": SceneUpdateChannel(topic="/slam/path"),
        "slam_status": Channel(topic="/slam/status", message_encoding="json",
                               schema=Schema(name=spec["name"],
                                             encoding="jsonschema",
                                             data=json.dumps(spec["schema"]).encode())),
    }
    sizes = {k: 0 for k in chans}
    counts = {k: 0 for k in chans}
    imu = ImuTrack()
    slam = Slam(imu=imu)
    h = ck.HostDejitter()
    rows = d["imu"]
    k = 0
    path = []
    next_at = {k: 0.0 for k in chans}
    for i in range(len(d["t"])):
        tl = float(d["t"][i])
        while k < len(rows) and rows[k, 0] <= tl:
            q = rows[k]
            up = q[4:7]
            imu.add(q[0], q[1:4], up if np.linalg.norm(up) > 0.5 else None)
            k += 1
        res = slam.step(h.fit(tl), d["ranges"][i])
        ns = int(tl * 1e9)
        if not path or np.hypot(res.x - path[-1][0], res.y - path[-1][1]) >= rn.PATH_STEP_M:
            path.append((res.x, res.y))
            if len(path) > rn.PATH_MAX:
                path = path[::2]
        msgs = {
            "tf": lambda: mv.frame_transforms(ns, ld.MOUNT, (res.x, res.y, res.yaw), ns),
            "slam_status": lambda: rn.status_row(res, {}),
            "slam_path": lambda: mv.path_scene(path, ns),
            "map": lambda: (lambda v: None if v is None else mv.grid_message(v, ns))(
                rn.map_view(slam, tl)),
        }
        for key, make in msgs.items():
            if tl < next_at[key]:
                continue
            next_at[key] = tl + tm.INTERVALS[key]
            m = make()
            if m is None:
                continue
            if key == "slam_status":
                b = json.dumps(m).encode()
                chans[key].log(m, log_time=ns)
            else:
                b = m.encode()
                chans[key].log(m, log_time=ns)
            sizes[key] += len(b)
            counts[key] += 1
    writer.close()
    dur = float(d["t"][-1] - d["t"][0])
    return {"seconds": round(dur, 1),
            "raw_KiB_per_s": {k: round(sizes[k] / 1024 / dur, 2) for k in chans},
            "messages": counts,
            "mcap_bytes": os.path.getsize(out_path),
            "mcap_KiB_per_s": round(os.path.getsize(out_path) / 1024 / dur, 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", help="an --export .npz (or an .mcap)")
    ap.add_argument("--limit", type=int, default=None, help="first N scans")
    ap.add_argument("--jitter", action="store_true")
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="--jitter: seconds of drive per run")
    ap.add_argument("--bandwidth", action="store_true")
    ap.add_argument("--out", default="/tmp/slam_bench.mcap",
                    help="--bandwidth: where the MCAP goes (not the repo)")
    a = ap.parse_args(argv)
    d = load(a.input)
    if a.jitter:
        for mode in (None, "thread", "process"):
            print(json.dumps(jitter(d, mode, a.seconds)), flush=True)
    elif a.bandwidth:
        print(json.dumps(bandwidth(d, a.out)), flush=True)
    else:
        print(json.dumps(dict(input=os.path.basename(a.input),
                              **per_scan(d, a.limit))), flush=True)


if __name__ == "__main__":
    main()
