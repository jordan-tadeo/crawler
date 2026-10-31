"""When a patrol had missed its waypoint and was driving away from it, off a drive's own recording.

    python tools/drive_aways.py OUT_DIR RUN.mcap [RUN.mcap ...]

On the 2026-10-30 patrols the truck came within 0.5-1.3 m of a waypoint,
never inside PASS_M, and drove on away from it with the waypoint still the
goal (crawler/autopilot/patrol.py's "A near miss"). I took over and backed it
round on open floor to get it pointed at the waypoint again. Those reverses
are labels like any other, but they are a rescue of the patrol's mistake,
not what the truck should do on open floor: 154 of the 203 open-floor
reverse labels in the 10-31 drives are inside one of these windows, and the
other 49 were not on a patrol at all. policy_train.py --rescue-reverse-weight
weighs them down.

A window, per waypoint: the run of /patrol/status rows with that waypoint
the goal while a patrol drives on the map (one index, track and frame, no
gap over SEGMENT_GAP_S), from the first row where either

  near miss   the closest it has come is within NEAR_M but not PASS_M
  drive away  it is RISE_M further than the closest it has come (the
              wrong-way survey's rise)

to that run's last row plus TAIL_S. So a waypoint the truck gets near opens
one, and a plain pass (inside PASS_M, the next waypoint the goal) shuts it
a second or two later; one it misses stays open until the near pass, a skip
or my rescue gets it past. Only my reverses on open floor in there are
weighed down, so a window on a good approach costs nothing: I do not back
up on open floor a metre short of a waypoint the truck is about to pass.

Writes OUT_DIR/<run>.npz:

  t0, t1   each window's start and end, log time in seconds
  layout   LAYOUT

policy_train.py --drive-aways OUT_DIR reads them; a session with no file has
no window. Needs the `mcap` package.
"""

import argparse
import json
import os
import sys

import numpy as np

LAYOUT = "away1"
# patrol.PASS_M (not imported: patrol.py brings the car's runtime in).
PASS_M = 0.5
# The near-miss band the patrol docstring measured: 56 of 58 drives away
# began within 1.3 m of the waypoint.
NEAR_M = 1.3
# The wrong-way survey's drive away: 0.5 m further than the closest it came.
RISE_M = 0.5
# One waypoint's run is split where /patrol/status (2 Hz) goes quiet this long.
SEGMENT_GAP_S = 3.0
# A window runs on this long after its run's last row.
TAIL_S = 0.5
TOPIC = "/patrol/status"


def status_arrays(rows):
    """/patrol/status payloads, each with its log time as "_t" -> dict of
    arrays: t, driving, distance, index, track, frame."""
    t = np.array([float(r["_t"]) for r in rows], float)
    driving = np.array([r.get("state") == "driving" and bool(r.get("running"))
                        and (r.get("use") or "patrol") == "patrol" and bool(r.get("on_map"))
                        and (r.get("distance_m") if r.get("distance_m") is not None else -1) >= 0
                        for r in rows], bool)
    dist = np.array([-1.0 if r.get("distance_m") is None else float(r["distance_m"])
                     for r in rows], float)
    index = np.array([-1 if r.get("index") is None else int(r["index"]) for r in rows], int)
    track = np.array([str(r.get("track")) for r in rows], dtype=str)
    frame = np.array([-1 if r.get("frame_now") is None else int(r["frame_now"]) for r in rows], int)
    o = np.argsort(t, kind="stable")
    return {"t": t[o], "driving": driving[o], "distance": dist[o], "index": index[o],
            "track": track[o], "frame": frame[o]}


def segments(s, gap_s=SEGMENT_GAP_S):
    """[(i, j)] row ranges, inclusive: one waypoint the goal while driving."""
    t, drv = s["t"], s["driving"]
    out, i, n = [], 0, len(t)
    while i < n:
        if not drv[i]:
            i += 1
            continue
        j = i
        while (j + 1 < n and drv[j + 1] and s["index"][j + 1] == s["index"][i]
               and s["track"][j + 1] == s["track"][i] and s["frame"][j + 1] == s["frame"][i]
               and t[j + 1] - t[j] < gap_s):
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def windows(s, near_m=NEAR_M, rise_m=RISE_M, pass_m=PASS_M, tail_s=TAIL_S):
    """(t0, t1) arrays: the docstring's windows over status_arrays' s."""
    t0, t1 = [], []
    for i, j in segments(s):
        d = s["distance"][i:j + 1]
        closest = np.minimum.accumulate(d)
        hit = ((closest <= near_m) & (closest > pass_m)) | (d - closest >= rise_m)
        if hit.any():
            t0.append(s["t"][i + int(np.argmax(hit))])
            t1.append(s["t"][j] + tail_s)
    return np.asarray(t0, float), np.asarray(t1, float)


def inside(t0, t1, tick_t):
    """Per tick time, whether it falls in any window [t0, t1]."""
    tick_t = np.asarray(tick_t, float)
    out = np.zeros(len(tick_t), bool)
    for a, b in zip(np.asarray(t0, float), np.asarray(t1, float)):
        out |= (tick_t >= a) & (tick_t <= b)
    return out


def read(path):
    """One MCAP -> its /patrol/status payloads, each with "_t"."""
    from mcap.reader import make_reader
    rows = []
    with open(path, "rb") as f:
        try:
            for _s, _ch, m in make_reader(f).iter_messages(topics=[TOPIC]):
                try:
                    d = json.loads(m.data)
                except Exception:
                    continue
                d["_t"] = m.log_time / 1e9
                rows.append(d)
        except Exception:
            pass  # a file cut off mid-chunk: what was read stands
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for path in args.runs:
        name = os.path.basename(path).rsplit(".", 1)[0]
        rows = read(path)
        if not rows:
            print(f"{name}: no patrol")
            continue
        t0, t1 = windows(status_arrays(rows))
        np.savez_compressed(os.path.join(args.out, name + ".npz"), t0=t0, t1=t1,
                            layout=np.array(LAYOUT))
        print(f"{name}: {len(t0)} windows, {float((t1 - t0).sum()):.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
