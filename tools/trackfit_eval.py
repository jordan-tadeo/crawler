#!/usr/bin/env python3
"""How well a saved track is fitted onto the next boot's map, off real drives.

    train/bin/python tools/trackfit_eval.py extract    # needs the `mcap` packages
    .venv/bin/python tools/trackfit_eval.py evaluate

A case is a track the owner traced on one run (the /patrol/marker points
and the newest /map on the frame they were on: what autopilot/patrol.py
saves) and the /map pictures of the run after it, which booted on a fresh
map. extract writes one small npz a case under <data>/trackfit/ (never in
git: a map is a floor plan of the house). evaluate runs
mapping/trackfit.fit on the next run's map every EVERY_S as it grew and
says, for each:

  accepted   whether the fit was confident enough to place the track
  err        how far each waypoint landed from where the reference fit
             puts it, mean and max: the reference is the fit on the
             newest map of the same frame, all the walls that run saw
  mirror     the same walls on the map flipped left to right, which no
             turn and shift can match: any accept there is a false accept

tests/test_trackfit.py reads the same files and holds the numbers.
"""

import argparse
import bisect
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# (name, the run the track was traced on, the frame it was saved on, the
# time from the start of that run it was last saved before, or None for
# the end; the next run, mcap name stems)
CASES = [
    ("A-0912", "crawler-20261030-1[56]*-p1290-00?", 0, 2003.0, "crawler-20261030-161219Z-p1462-001"),
    ("B-1136", "crawler-20261030-18*-p1261-00?", 0, 2028.0, "crawler-20261030-183518Z-p1374-001"),
    ("C-1407", "crawler-20261030-205939Z-p1276-001", 0, None, "crawler-20261030-21[01]*-p2607-00?"),
    ("D-1439", "crawler-20261030-21[01]*-p2607-00?", 0, None, "crawler-20261030-213718Z-p1200-001"),
]
EVERY_S = 20.0
# No map within this of a frame change is used: /map and /patrol/status
# are separate topics, and either can be a beat behind the other.
GUARD_S = 2.0


def data_dir():
    import data_paths as dp
    return os.path.join(dp.data_root(), "trackfit")


def _read(paths):
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    maps, status, markers = [], [], []
    for path in paths:
        with open(path, "rb") as f:
            r = make_reader(f)
            df, decs = DecoderFactory(), {}
            for schema, ch, msg in r.iter_messages(
                    topics=["/map", "/patrol/status", "/patrol/marker"], log_time_order=True):
                t = msg.log_time / 1e9
                if ch.topic == "/patrol/status":
                    status.append((t, json.loads(msg.data)["frame_now"]))
                    continue
                if schema.id not in decs:
                    decs[schema.id] = df.decoder_for("protobuf", schema)
                dec = decs[schema.id](msg.data)
                if ch.topic == "/map":
                    nx = dec.column_count
                    rgba = np.frombuffer(dec.data, np.uint8).reshape(-1, nx, 4)
                    codes = np.zeros(rgba.shape[:2], np.uint8)
                    codes[rgba[..., 3] == 200] = 1      # map_view.COLOURS
                    codes[rgba[..., 3] == 255] = 2
                    maps.append((t, dec.pose.position.x, dec.pose.position.y,
                                 dec.cell_size.x, codes.T.copy()))
                else:
                    pts = {}
                    for e in dec.entities:
                        if e.id.startswith("patrol_wp_"):
                            p = e.cylinders[0].pose.position
                            pts[int(e.id.rsplit("_", 1)[1])] = (p.x, p.y)
                    markers.append((t, [pts[k] for k in sorted(pts)]))
    return maps, status, markers


class _Frames:
    def __init__(self, status):
        self.t = [s[0] for s in status]
        self.f = [s[1] for s in status]

    def at(self, t):
        return self.f[max(bisect.bisect_right(self.t, t) - 1, 0)]

    def settled(self, t):
        f = self.at(t)
        return f if self.at(t - GUARD_S) == f and self.at(t + GUARD_S) == f else None


def extract(mcap_dir, out_dir):
    import glob
    os.makedirs(out_dir, exist_ok=True)
    for name, src, frame, cut, tgt in CASES:
        maps, status, markers = _read(sorted(glob.glob(os.path.join(mcap_dir, src + ".mcap"))))
        if not maps:
            print(f"{name}: no recording for {src}, skipped")
            continue
        fr = _Frames(status)
        t0 = maps[0][0]
        before = math.inf if cut is None else t0 + cut
        pts = next(p for t, p in reversed(markers) if t < before and p and fr.at(t) == frame)
        m = next(m for m in reversed(maps) if m[0] < before and fr.settled(m[0]) == frame)
        from crawler.mapping import trackfit as tf
        from crawler.mapping import runner as rn
        walls = tf.occupied_points(rn.MapView(0, 0, m[1], m[2], m[3], m[4]))
        maps, status, _ = _read(sorted(glob.glob(os.path.join(mcap_dir, tgt + ".mcap"))))
        fr = _Frames(status)
        t0, last, keep = maps[0][0], -math.inf, []
        for k, m in enumerate(maps):
            f = fr.settled(m[0])
            if f is None:
                continue
            final = all(fr.settled(n[0]) != f for n in maps[k + 1:])
            if m[0] - last >= EVERY_S or final:
                last = m[0] if m[0] - last >= EVERY_S else last
                keep.append((m[0] - t0, f, final) + m[1:])
        np.savez_compressed(
            os.path.join(out_dir, name + ".npz"), walls=walls, points=np.array(pts),
            t=np.array([k[0] for k in keep]), frame=np.array([k[1] for k in keep]),
            final=np.array([k[2] for k in keep]),
            meta=np.array([k[3:6] for k in keep]),
            codes=np.array([k[6] for k in keep], dtype=object), allow_pickle=True)
        print(f"{name}: {len(pts)} waypoints, {len(walls)} walls, {len(keep)} maps")


def load(path):
    """(walls, points, [(t, frame, final, MapView)]) of one case file."""
    from crawler.mapping import runner as rn
    d = np.load(path, allow_pickle=True)
    views = [(float(t), int(f), bool(fin),
              rn.MapView(float(t), k, float(m[0]), float(m[1]), float(m[2]), np.asarray(c)))
             for k, (t, f, fin, m, c) in enumerate(zip(d["t"], d["frame"], d["final"],
                                                       d["meta"], d["codes"]))]
    return d["walls"], d["points"], views


def mirrored(view):
    codes = np.asarray(view.codes)[:, ::-1].copy()
    return view._replace(codes=codes, y0=-(view.y0 + view.res * codes.shape[1]))


def evaluate(path, every=1):
    """One case: rows (t, frame, accepted, err mean, err max, mirror
    accepted), on every `every`th map."""
    from crawler.mapping import trackfit as tf
    walls, pts, views = load(path)
    refs = {f: tf.fit(walls, v) for _, f, final, v in views if final}
    rows = []
    for t, f, _, v in views[::every]:
        fit, ref = tf.fit(walls, v), refs.get(f)
        if fit is None or ref is None:
            continue
        err = np.hypot(*(tf.transform(fit, pts) - tf.transform(ref, pts)).T)
        m = tf.fit(walls, mirrored(v))
        rows.append((t, f, fit.ok, float(err.mean()), float(err.max()),
                     bool(m is not None and m.ok), fit))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("extract", "evaluate"))
    ap.add_argument("--mcap", help="the recordings (default <data>/mcap)")
    args = ap.parse_args(argv)
    out = data_dir()
    if args.what == "extract":
        import data_paths as dp
        extract(args.mcap or os.path.join(dp.data_root(), "mcap"), out)
        return 0
    for fn in sorted(os.listdir(out)):
        rows = evaluate(os.path.join(out, fn))
        ok = [r for r in rows if r[2]]
        first = next((r[0] for r in ok), None)
        print(f"{fn[:-4]}: {len(ok)} of {len(rows)} accepted, first at "
              f"{'never' if first is None else f'{first:.0f} s'}; "
              + (f"err mean {np.mean([r[3] for r in ok]):.3f} m, worst max "
                 f"{max(r[4] for r in ok):.3f} m; " if ok else "")
              + f"wrong accepts (max > 0.5 m) {sum(r[4] > 0.5 for r in ok)}; "
              f"mirror accepts {sum(r[5] for r in rows)} of {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
