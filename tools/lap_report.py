"""Lap and segment times off patrol drives, per model and per track.

    ~/crawler/train/bin/python tools/lap_report.py ~/crawler/data/mcap/crawler-20261030-*.mcap
    ... --json laps.json      # everything, laps and segments too
    ... --segments            # a line per segment under each lap

The question is which model gets round a track fastest with the least
tipping, so the numbers are what a lap cost: time, skips, takeovers, tips.
Nothing here feeds the car; the planner still owns the goal.

What a segment and a lap are
----------------------------

A segment is the drive from one waypoint to the next: it opens on the event
that made waypoint j the goal (waypoint i passed or skipped) and closes on
the event that passes or skips j. One opened by a start rather than a pass
is an approach (from wherever the car stood): kept, but never "chained",
and not timed against the others. A lap is the segments driven while the
patrol's lap count read L, one chain on one track: full when it runs from
the event that made it L to the one that made it L+1 with every segment
chained.

Clean: a full lap (or a chained segment) with no takeover, no skip, no tip
and no row on the map-free fallback.

Where the times come from
-------------------------

A drive recorded since /patrol/event: the event at the decision it happened
(10 Hz or better), and every /footprint row says the track, the waypoint and
the lap (patrol_track, patrol_index, patrol_lap). The track id
(patrol.Patrol.track_id) changes with every edit, so an edited track's laps
are never counted with the one it was.

An older one: the events are read off /patrol/status, 2 Hz, so every pass is
placed half way between two status messages (to +-0.25 s), a message that
moved two waypoints on gives two at the same time, and the track is only
known to change when the waypoint count does (an edit that kept the count is
missed, and the lap counter ran on across edits). It says so.

The per-row numbers are the /footprint auto rows (10 Hz) with the patrol
driving, each counted for the gap to the next (at most ROW_GAP_S):
  time_s       drive_s + manual_s: the patrol driving, auto or not
  drive_s      in auto
  manual_s     in manual or record (/vehicle), the patrol still driving
  takeover_s   auto rows with my hand on either group, plus manual_s;
               takeovers counts each start of one
  reverse_s, zero_s   the wheels' throttle below 0, at 0
  rear_stops   each rear stop (the lidar ended or refused a reverse)
  tilts        each start of a tilt veto block
  tips         each start of a tip: policy_train.tip_marks, 45 deg off the
               run's own median up (the training's tip mask)
  max_tilt_deg the most off that up
  fallback     share of the auto rows the map-free fallback drove
  map_lost_s   rows a map model had no trusted map for
  stuck_s      time past STUCK_FROM_S without getting PROGRESS_M closer
  path_m       /slam/status track length, steps over STEP_MAX_M dropped
  leg_m        straight line from the waypoint before to this one
  s_per_m      time_s / leg_m: a segment's time against its own length

The summary, per model and track: the median clean lap time and the
median clean segment s_per_m, and skips, takeovers and tips per lap
(counted per lap-equivalent: all segments driven over segments a lap, so a
part lap counts its part), each with a 95% bootstrap interval resampling
laps (BOOT draws, seed BOOT_SEED). A segment's model is the one that drove
most of its non-fallback rows; a lap's, most of its segments' rows.

Needs the `mcap` package (the training venv has it).
"""

import argparse
import collections
import json
import math
import os
import re
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from crawler.autopilot import policy_imu as pimu  # noqa: E402
import policy_train as ptr  # noqa: E402

TOPICS = ("/footprint", "/patrol/status", "/patrol/event", "/slam/status", "/vehicle")
ROW_GAP_S = 0.2           # a row counts for the gap to the next, no more
VEHICLE_GAP_S = 0.1       # /vehicle is 20 Hz
STATUS_GAP_S = 0.5        # /patrol/status is 2 Hz
STEP_MAX_M = 0.5          # a /slam/status step longer is a jump, not driving
STUCK_FROM_S = 5.0        # stuck: this long without PROGRESS_M closer
PROGRESS_M = 0.3          # patrol.PROGRESS_M
LEG_MIN_M = 0.3           # a shorter leg's s_per_m is noise
BOOT = 2000
BOOT_SEED = 0
COARSE = ("timing off /patrol/status at 2 Hz: each pass to +-0.25 s, and an edit "
          "that kept the waypoint count is not seen")
_FILE = re.compile(r"-p(\d+)-(\d+)\.mcap$")


# ---- reading -------------------------------------------------------------------------

def _row(d):
    """The /footprint fields this reads, so a run's rows stay small."""
    keep = ("mode", "policy_model", "policy_fallback", "policy_map_present", "steer_by",
            "throttle_by", "throttle", "rear_stop", "tilt_veto", "patrol_use",
            "patrol_state", "patrol_track", "patrol_index", "patrol_lap",
            "patrol_stuck_s") + pimu.ROW_FIELDS
    return {k: d[k] for k in keep if k in d}


def read_mcap(path):
    """[(topic, t_s, dict)] off one file, in log time order. A file cut
    off mid-chunk gives what was read before the cut."""
    from mcap.reader import make_reader
    out = []
    with open(path, "rb") as f:
        try:
            for _s, ch, m in make_reader(f).iter_messages(topics=list(TOPICS),
                                                           log_time_order=True):
                try:
                    d = json.loads(m.data)
                except Exception:
                    continue
                if ch.topic == "/footprint":
                    if d.get("mode") != "auto":
                        continue
                    d = _row(d)
                elif ch.topic == "/vehicle":
                    d = {"mode": d.get("mode")}
                elif ch.topic == "/slam/status":
                    d = {k: d.get(k) for k in ("state", "x", "y", "reseed")}
                out.append((ch.topic, m.log_time / 1e9, d))
        except Exception:
            pass
    return out


def runs_of(paths):
    """The files grouped into runs, one a process: in name (time) order,
    a new run at each -001 or a new process id. [(name, [paths])]."""
    files = sorted((p for p in paths if _FILE.search(os.path.basename(p))),
                   key=os.path.basename)
    runs = []
    for p in files:
        pid, seq = _FILE.search(os.path.basename(p)).groups()
        if not runs or runs[-1][0] != pid or int(seq) == 1:
            runs.append((pid, [p]))
        else:
            runs[-1][1].append(p)
    return [(f"p{pid} {os.path.basename(ps[0])[8:24]}", ps) for pid, ps in runs]


# ---- events --------------------------------------------------------------------------

def events_from_status(status, run):
    """Events off /patrol/status alone (before /patrol/event), and a track
    id for each status message. status [(t, dict)] -> (events, ids)."""
    events, ids = [], []
    prev, track, k, shape = None, None, 0, None
    for t, s in status:
        if s.get("use") != "patrol":
            ids.append(track)
            prev = None
            continue
        n = int(s.get("count") or 0)
        if (n, s.get("mode")) != shape:
            k, shape = k + 1, (n, s.get("mode"))
            track = f"{run}#{k}:{s.get('mode')}{n}"
        ids.append(track)
        driving = s.get("state") == "driving"
        if prev is not None:
            ps = prev[1]
            mid = 0.5 * (prev[0] + t)
            was_driving = ps.get("state") == "driving"
            same = track == prev[2]
            if driving and not was_driving:
                fresh = (not same or int(s.get("laps") or 0) < int(ps.get("laps") or 0)
                         or int(s.get("index") or 0) != int(ps.get("index") or 0))
                events.append(dict(t=mid, kind="start", by="fresh" if fresh else "resume",
                                   index=int(s.get("index") or 0), track=track))
            # Only a message that was driving can have passed anything: an
            # index that moved while paused or finished is a reset.
            if same and n > 0 and was_driving:
                i0, i1 = int(ps.get("index") or 0), int(s.get("index") or 0)
                steps = _steps(i0, i1, n, s.get("mode"))
                skips = max(int(s.get("skipped") or 0) - int(ps.get("skipped") or 0), 0)
                lap0, lap1 = int(ps.get("laps") or 0), int(s.get("laps") or 0)
                for j, (a, b) in enumerate(steps):
                    last = j == len(steps) - 1
                    lap = lap1 if last else lap0 + sum(1 for _, bb in steps[:j + 1] if bb == 0)
                    events.append(dict(t=mid, kind="skip" if j < skips else "pass",
                                       by="status", index=a, next=b, lap=lap, track=track,
                                       count=n, mode=s.get("mode")))
            if was_driving and not driving and s.get("state") == "finished":
                events.append(dict(t=mid, kind="finish", by="status",
                                   index=int(s.get("index") or 0), track=track))
        prev = (t, s, track)
    return events, ids


def _steps(i0, i1, n, mode):
    """[(from, to)] the patrol went through between two status messages."""
    if i0 == i1:
        return []
    if mode == "loop":
        k = (i1 - i0) % n
        return [((i0 + j) % n, (i0 + j + 1) % n) for j in range(k)]
    d = 1 if i1 > i0 else -1
    return [(i, i + d) for i in range(i0, i1, d)]


# ---- one run -------------------------------------------------------------------------

def _rising(flags):
    flags = np.asarray(flags, bool)
    if not len(flags):
        return np.zeros(0, bool)
    return flags & ~np.concatenate([[False], flags[:-1]])


def _gaps(t, cap):
    t = np.asarray(t, float)
    if not len(t):
        return t
    return np.minimum(np.diff(np.append(t, t[-1] + cap)), cap)


def _before(times, t):
    """Index of the newest of sorted `times` at or before t, -1 with none."""
    return int(np.searchsorted(times, t, side="right")) - 1


def _tilt(rows, sid):
    """(tipped onsets, degrees off the run's median up) per row, by
    policy_train.tip_marks over the up vector as the dataset builds it."""
    n = len(rows)
    imu = np.zeros((n, pimu.IMU_SIZE), np.float32)
    c = pimu.FEATURES.index("up_x")
    v = pimu.FEATURES.index("tilt_valid")
    for i, r in enumerate(rows):
        up = pimu.up_of(r)
        if up is not None:
            imu[i, c:c + 3] = up
            imu[i, v] = 1.0
    t_ns = np.array([int(r["_t"] * 1e9) for r in rows], np.int64)
    tipped, _ = ptr.tip_marks({"imu": imu, "session": np.full(n, sid), "t_ns": t_ns})
    deg = np.full(n, np.nan)
    ok = imu[:, v] > 0.5
    if ok.any():
        ref = np.median(imu[ok, c:c + 3].astype(float), axis=0)
        ref /= max(np.linalg.norm(ref), 1e-9)
        deg[ok] = np.degrees(np.arccos(np.clip(imu[ok, c:c + 3] @ ref, -1.0, 1.0)))
    return _rising(tipped), deg


def analyse_run(name, msgs, sid=0):
    """One run's messages [(topic, t, dict)] -> {name, coarse, segments,
    laps, warnings}."""
    by = collections.defaultdict(list)
    for topic, t, d in sorted(msgs, key=lambda m: m[1]):
        by[topic].append((t, d))
    status = by["/patrol/status"]
    warnings = []
    exact = bool(by["/patrol/event"])
    if exact:
        events = []
        for t, e in by["/patrol/event"]:
            events.append(dict(e, t=t))
        seqs = [e.get("seq") for _, e in by["/patrol/event"]]
        lost = sum(max(b - a - 1, 0) for a, b in zip(seqs, seqs[1:])
                   if isinstance(a, int) and isinstance(b, int))
        if lost:
            warnings.append(f"{name}: {lost} /patrol/event messages missing (seq gaps)")
        ids = [s.get("track") or None for _, s in status]
    else:
        events, ids = events_from_status(status, name)
        if events:
            warnings.append(f"{name}: {COARSE}")
    if not any(e["kind"] in ("pass", "skip", "finish") for e in events):
        return {"name": name, "coarse": not exact, "segments": [], "laps": [],
                "warnings": []}

    st_t = np.array([t for t, _ in status], float)
    rows = []
    for t, d in by["/footprint"]:
        d = dict(d, _t=t)
        if "patrol_state" not in d:
            j = _before(st_t, t)
            s = status[j][1] if j >= 0 else {}
            d["patrol_use"], d["patrol_state"] = s.get("use", ""), s.get("state", "")
            d["patrol_track"] = (ids[j] if j >= 0 else None) or ""
        rows.append(d)
    row_t = np.array([r["_t"] for r in rows], float)
    gap = _gaps(row_t, ROW_GAP_S)
    tip_on, tilt_deg = _tilt(rows, sid) if rows else (np.zeros(0, bool), np.zeros(0))
    driving = np.array([r.get("patrol_use") == "patrol" and r.get("patrol_state") == "driving"
                        for r in rows], bool)
    hand = np.array([r.get("steer_by") == "pad" or r.get("throttle_by") == "pad"
                     for r in rows], bool)
    fallback = np.array([bool(r.get("policy_fallback")) for r in rows], bool)
    map_models = {r.get("policy_model") for r in rows if r.get("policy_map_present")}
    map_lost = np.array([bool(r.get("policy_fallback")) or (
        r.get("policy_model") in map_models and not r.get("policy_map_present"))
        for r in rows], bool)
    thr = np.array([float(r.get("throttle") or 0.0) for r in rows], float)
    rear_on = _rising([bool(r.get("rear_stop")) for r in rows])
    tilt_on = _rising([bool(r.get("tilt_veto")) for r in rows])
    hand_on = _rising(hand)
    stuck_row = np.array([float(r.get("patrol_stuck_s", -1.0)) for r in rows], float)
    has_row_stuck = bool(rows) and "patrol_stuck_s" in rows[0]

    veh = by["/vehicle"]
    veh_t = np.array([t for t, _ in veh], float)
    veh_gap = _gaps(veh_t, VEHICLE_GAP_S)
    veh_manual = np.array([d.get("mode") not in ("auto", None) for _, d in veh], bool)
    veh_on = _rising(veh_manual)
    st_driving = np.array([s.get("use") == "patrol" and s.get("state") == "driving"
                           for _, s in status], bool)

    slam = [(t, d) for t, d in by["/slam/status"] if d.get("state") == "tracking"]
    sl_t = np.array([t for t, _ in slam], float)
    steps = np.zeros(len(slam))
    for i in range(1, len(slam)):
        a, b = slam[i - 1][1], slam[i][1]
        try:
            s = math.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"]))
        except (TypeError, ValueError, KeyError):
            continue
        if s <= STEP_MAX_M and not b.get("reseed"):
            steps[i] = s

    # Where each waypoint is, per track: the status's current one.
    places = collections.defaultdict(dict)
    for (t, s), tid in zip(status, ids):
        if tid and s.get("use") == "patrol" and s.get("on_map"):
            places[tid][int(s.get("index") or 0)] = (float(s.get("x") or 0.0),
                                                    float(s.get("y") or 0.0))

    def window(t0, t1, track, target):
        i0, i1 = np.searchsorted(row_t, t0), np.searchsorted(row_t, t1)
        sel = np.zeros(len(rows), bool)
        sel[i0:i1] = driving[i0:i1]
        if exact:
            sel[i0:i1] &= np.array([r.get("patrol_track") == track
                                    and r.get("patrol_index") == target
                                    for r in rows[i0:i1]], bool)
        return sel

    def seg_stats(t0, t1, track, target):
        sel = window(t0, t1, track, target)
        g = gap[sel]
        out = {"drive_s": float(g.sum())}
        v0, v1 = np.searchsorted(veh_t, t0), np.searchsorted(veh_t, t1)
        vm = veh_manual[v0:v1].copy()
        for k in range(len(vm)):
            j = _before(st_t, veh_t[v0 + k])
            vm[k] &= j >= 0 and st_driving[j]
        out["manual_s"] = float(veh_gap[v0:v1][vm].sum())
        out["time_s"] = out["drive_s"] + out["manual_s"]
        out["takeover_s"] = float(g[hand[sel]].sum()) + out["manual_s"]
        out["takeovers"] = int(hand_on[sel].sum() + (veh_on[v0:v1] & vm).sum())
        out["reverse_s"] = float(g[thr[sel] < 0].sum())
        out["zero_s"] = float(g[thr[sel] == 0].sum())
        out["rear_stops"] = int(rear_on[sel].sum())
        out["tilts"] = int(tilt_on[sel].sum())
        out["tips"] = int(tip_on[sel].sum())
        td = tilt_deg[sel]
        out["max_tilt_deg"] = float(np.nanmax(td)) if np.isfinite(td).any() else None
        n = int(sel.sum())
        out["rows"] = n
        out["fallback_rows"] = int(fallback[sel].sum())
        out["fallback"] = out["fallback_rows"] / n if n else 0.0
        out["map_lost_s"] = float(g[map_lost[sel]].sum())
        if has_row_stuck:
            out["stuck_s"] = float(g[stuck_row[sel] >= STUCK_FROM_S].sum())
        else:
            out["stuck_s"] = _stuck_from_status(status, st_t, t0, t1)
        # The steps after t0 up to and with the one that arrives at t1.
        s0, s1 = np.searchsorted(sl_t, t0), np.searchsorted(sl_t, t1, side="right")
        out["path_m"] = float(steps[s0 + 1:s1].sum()) if s1 > s0 else 0.0
        names = collections.Counter(r.get("policy_model") or "" for r, on, fb in
                                    zip(rows, sel, fallback) if on and not fb)
        allnames = collections.Counter(r.get("policy_model") or "" for r, on in
                                       zip(rows, sel) if on)
        out["model_rows"] = dict(names)
        out["models"] = sorted(m for m in allnames if m)
        out["model"] = names.most_common(1)[0][0] if names else (
            allnames.most_common(1)[0][0] if allnames else "")
        return out

    segments = []
    open_ = None
    chain = 0
    for e in sorted(events, key=lambda e: e["t"]):
        kind = e["kind"]
        track = e.get("track") or ""
        if kind == "start":
            keep = (e.get("by") == "resume" and open_ is not None and open_["track"] == track
                    and open_["target"] == e["index"])
            if not keep:
                chain += 1
                open_ = {"t": e["t"], "track": track, "target": int(e["index"]),
                         "from": None, "lap": int(e.get("lap", 0) or 0), "chain": chain,
                         "lap_start": False}
            continue
        if kind == "pause":
            continue
        if kind not in ("pass", "skip", "finish"):
            continue
        if open_ is not None and open_["track"] == track and open_["target"] == e["index"]:
            seg = {"run": name, "track": track, "from": open_["from"], "to": int(e["index"]),
                   "t0": open_["t"], "t1": e["t"], "end": kind, "by": e.get("by", ""),
                   "lap": open_["lap"], "chain": open_["chain"],
                   "chained": open_["from"] is not None, "lap_start": open_["lap_start"],
                   "lap_end": int(e.get("lap", 0) or 0) != open_["lap"],
                   "count": int(e.get("count") or 0), "mode": e.get("mode") or "",
                   "wp_s": e.get("wp_s")}
            seg.update(seg_stats(seg["t0"], seg["t1"], track, seg["to"]))
            a = places[track].get(seg["from"]) if seg["from"] is not None else None
            b = places[track].get(seg["to"])
            seg["leg_m"] = math.hypot(b[0] - a[0], b[1] - a[1]) if a and b else None
            seg["s_per_m"] = (seg["time_s"] / seg["leg_m"]
                              if seg["leg_m"] and seg["leg_m"] >= LEG_MIN_M
                              and seg["rows"] else None)
            seg["skip"] = kind == "skip"
            seg["clean"] = bool(seg["chained"] and _clean(seg))
            segments.append(seg)
        else:
            chain += 1
        if kind == "finish":
            open_ = None
            continue
        nxt = e.get("next")
        if nxt is None:
            open_ = None
            continue
        lap = int(e.get("lap", 0) or 0)
        if open_ is None or open_["track"] != track:
            chain += 1
        # The pass that made the first waypoint the goal again starts a lap
        # (patrol._advance counts it there, in a loop and there and back).
        open_ = {"t": e["t"], "track": track, "target": int(nxt), "from": int(e["index"]),
                 "lap": lap, "chain": chain,
                 "lap_start": int(nxt) == 0 and e.get("mode") in ("loop", "patrol")}
    return {"name": name, "coarse": not exact, "segments": segments,
            "laps": laps_of(segments), "warnings": warnings}


def _clean(s):
    """Driven (some auto rows), and nothing a fast time could hide."""
    return (s["rows"] > 0 and s["takeover_s"] == 0 and s["takeovers"] == 0 and not s["skip"]
            and s["tips"] == 0 and s["fallback_rows"] == 0)


def _stuck_from_status(status, st_t, t0, t1):
    """stuck_s off /patrol/status's distance (2 Hz), by the patrol's rule."""
    i0, i1 = np.searchsorted(st_t, t0), np.searchsorted(st_t, t1)
    best, since, out, last = math.inf, 0.0, 0.0, None
    for t, s in status[i0:i1]:
        dt = 0.0 if last is None else min(t - last, STATUS_GAP_S)
        last = t
        if s.get("state") != "driving" or not s.get("on_map"):
            continue
        d = float(s.get("distance_m", -1.0))
        if d < 0:
            continue
        if d < best - PROGRESS_M:
            best, since = d, 0.0
        else:
            since += dt
            if since >= STUCK_FROM_S:
                out += dt
    return out


SUMS = ("time_s", "drive_s", "manual_s", "takeover_s", "takeovers", "reverse_s", "zero_s",
        "rear_stops", "tilts", "tips", "map_lost_s", "stuck_s", "path_m", "rows",
        "fallback_rows")


def per_lap(mode, n):
    """Segments in a lap of n waypoints."""
    return 2 * (n - 1) if mode == "patrol" else n


def laps_of(segments):
    """The segments grouped into laps: one chain, one track, one lap count."""
    groups = collections.OrderedDict()
    for s in segments:
        groups.setdefault((s["run"], s["track"], s["chain"], s["lap"]), []).append(s)
    laps = []
    for (run, track, chain, lap), ss in groups.items():
        n, mode = ss[0]["count"], ss[0]["mode"]
        out = {"run": run, "track": track, "lap": lap, "chain": chain, "segments": len(ss),
               "count": n, "mode": mode, "t0": ss[0]["t0"], "t1": ss[-1]["t1"]}
        for k in SUMS:
            out[k] = sum(s[k] for s in ss)
        out["skips"] = sum(s["skip"] for s in ss)
        out["fallback"] = out["fallback_rows"] / out["rows"] if out["rows"] else 0.0
        tilts = [s["max_tilt_deg"] for s in ss if s["max_tilt_deg"] is not None]
        out["max_tilt_deg"] = max(tilts) if tilts else None
        rows = collections.Counter()
        for s in ss:
            rows.update(s["model_rows"])
        out["model"] = rows.most_common(1)[0][0] if rows else ss[0]["model"]
        out["models"] = sorted({m for s in ss for m in s["models"]})
        out["full"] = bool(ss[0]["lap_start"] and ss[-1]["lap_end"]
                           and all(s["chained"] for s in ss)
                           and len(ss) == per_lap(mode, n))
        out["clean"] = bool(out["full"] and all(_clean(s) for s in ss))
        out["lap_equiv"] = len(ss) / max(per_lap(mode, n), 1)
        laps.append(out)
    return laps


# ---- the summary ---------------------------------------------------------------------

def _boot(groups, stat, rng, draws=BOOT):
    """(estimate, lo, hi): stat over the groups pooled, and its 95%
    percentile interval resampling whole groups."""
    groups = [g for g in groups if len(g)]
    if not groups:
        return None, None, None
    est = stat([x for g in groups for x in g])
    if est is None:
        return None, None, None
    vals = []
    for _ in range(draws):
        pick = rng.integers(0, len(groups), len(groups))
        v = stat([x for i in pick for x in groups[i]])
        if v is not None:
            vals.append(v)
    if not vals:
        return est, None, None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return est, float(lo), float(hi)


def _median(xs):
    return float(statistics.median(xs)) if xs else None


def _rate(key):
    def stat(items):
        lapeq = sum(e for e, _ in items)
        return sum(v for _, v in items) / lapeq if lapeq > 0 else None
    return stat


def summarise(runs, draws=BOOT, seed=BOOT_SEED):
    """[{model, track, ...}] per model and track."""
    rng = np.random.default_rng(seed)
    laps = [lp for r in runs for lp in r["laps"]]
    segs = [s for r in runs for s in r["segments"]]
    keyed = collections.defaultdict(list)
    for lp in laps:
        keyed[(lp["model"], _track_key(lp["track"]))].append(lp)
    out = []
    for (model, track), lps in sorted(keyed.items()):
        ids = {(lp["run"], lp["track"], lp["chain"], lp["lap"]) for lp in lps}
        mine = [s for s in segs if (s["run"], s["track"], s["chain"], s["lap"]) in ids]
        by_lap = collections.defaultdict(list)
        for s in mine:
            by_lap[(s["run"], s["track"], s["chain"], s["lap"])].append(s)
        clean_laps = [[lp["time_s"]] for lp in lps if lp["clean"]]
        spm = [[s["s_per_m"] for s in ss if s["clean"] and s["s_per_m"] is not None]
               for ss in by_lap.values()]
        lap_items = {k: [] for k in ("skips", "takeovers", "tips", "tilts", "time_s")}
        for lp in lps:
            for k in lap_items:
                lap_items[k].append([(lp["lap_equiv"], lp[k])])
        row = {"model": model, "track": track, "laps": len(lps),
               "full_laps": sum(lp["full"] for lp in lps),
               "clean_laps": len(clean_laps),
               "lap_equiv": round(sum(lp["lap_equiv"] for lp in lps), 2),
               "segments": len(mine), "clean_segments": sum(s["clean"] for s in mine),
               "runs": sorted({lp["run"] for lp in lps})}
        row["clean_lap_s"] = _boot(clean_laps, _median, rng, draws)
        row["clean_s_per_m"] = _boot(spm, _median, rng, draws)
        for k in ("skips", "takeovers", "tips", "tilts"):
            row[f"{k}_per_lap"] = _boot(lap_items[k], _rate(k), rng, draws)
        row["time_per_lap_s"] = _boot(lap_items["time_s"], _rate("time_s"), rng, draws)
        row["fallback"] = (sum(lp["fallback_rows"] for lp in lps)
                           / max(sum(lp["rows"] for lp in lps), 1))
        out.append(row)
    return [r for r in out if r["model"]]


def by_model(runs, draws=BOOT, seed=BOOT_SEED):
    """[{model, ...}] across every track, on the segments a model drove
    itself (its own, not its lap's): the median clean segment s_per_m, and
    skips, takeovers, tips and tilt blocks per 100 m of legs driven (the
    straight line waypoint to waypoint, so a model that wanders is not
    credited the metres), each with a 95% bootstrap over laps."""
    rng = np.random.default_rng(seed)
    keyed = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in runs:
        for s in r["segments"]:
            if s["model"] and s["leg_m"]:
                keyed[s["model"]][(s["run"], s["track"], s["chain"], s["lap"])].append(s)
    out = []
    for model, laps in sorted(keyed.items()):
        groups = list(laps.values())
        segs = [s for g in groups for s in g]
        row = {"model": model, "segments": len(segs),
               "clean_segments": sum(s["clean"] for s in segs),
               "leg_m": round(sum(s["leg_m"] for s in segs), 1),
               "time_s": round(sum(s["time_s"] for s in segs), 1),
               "tracks": sorted({s["track"] for s in segs})}
        row["clean_s_per_m"] = _boot(
            [[s["s_per_m"] for s in g if s["clean"] and s["s_per_m"] is not None]
             for g in groups], _median, rng, draws)
        row["s_per_m_all"] = (row["time_s"] / row["leg_m"]) if row["leg_m"] else None
        for k, field in (("skips", "skip"), ("takeovers", "takeovers"), ("tips", "tips"),
                         ("tilts", "tilts")):
            items = [[(s["leg_m"] / 100.0, float(s[field])) for s in g] for g in groups]
            row[f"{k}_per_100m"] = _boot(items, _rate(k), rng, draws)
        out.append(row)
    return out


def _track_key(track):
    """The summary's track: the id itself (a saved track is the same id on
    every boot)."""
    return track


# ---- out -----------------------------------------------------------------------------

def _f(v, fmt="{:.1f}"):
    return "-" if v is None else fmt.format(v)


def _ci(t, fmt="{:.1f}"):
    est, lo, hi = t
    if est is None:
        return "-"
    if lo is None:
        return fmt.format(est)
    return f"{fmt.format(est)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def _short(model):
    return model.replace("kblock candidate ", "") or "?"


LAP_HEAD = (f"{'lap':>4} {'segs':>5} {'full':>4} {'clean':>5} {'time_s':>7} {'path_m':>6} "
            f"{'skip':>4} {'stuck':>5} {'take_s':>6} {'take%':>5} {'rev_s':>5} {'zero_s':>6} "
            f"{'rstop':>5} {'tilt':>4} {'tip':>3} {'maxdeg':>6} {'fb%':>4} {'mlost':>5}  model")


def _lap_line(lp):
    t = lp["time_s"]
    return (f"{lp['lap']:>4} {lp['segments']:>3}/{per_lap(lp['mode'], lp['count']):<2}"
            f"{'y' if lp['full'] else '-':>3} {'y' if lp['clean'] else '-':>5} "
            f"{t:>7.1f} {lp['path_m']:>6.1f} {lp['skips']:>4} {lp['stuck_s']:>5.0f} "
            f"{lp['takeover_s']:>6.1f} {100 * lp['takeover_s'] / t if t else 0:>5.0f} "
            f"{lp['reverse_s']:>5.1f} {lp['zero_s']:>6.1f} {lp['rear_stops']:>5} "
            f"{lp['tilts']:>4} {lp['tips']:>3} {_f(lp['max_tilt_deg'], '{:.0f}'):>6} "
            f"{100 * lp['fallback']:>4.0f} {lp['map_lost_s']:>5.1f}  "
            f"{', '.join(_short(m) for m in lp['models'])}")


def _seg_line(s):
    return (f"       {_f(s['from'], '{}'):>3}->{s['to']:<3} {s['end']:<6} "
            f"{s['time_s']:>6.1f} s {_f(s['leg_m'], '{:.1f}'):>5} m leg "
            f"{_f(s['s_per_m'], '{:.2f}'):>5} s/m  take {s['takeover_s']:.1f} s  "
            f"tips {s['tips']}  {'clean' if s['clean'] else ''}")


def text(runs, summary, segments=False, models=None):
    out = []
    for r in runs:
        if not r["laps"]:
            continue
        out.append(f"== {r['name']}{'  (coarse: 2 Hz status)' if r['coarse'] else ''}")
        for track in dict.fromkeys(lp["track"] for lp in r["laps"]):
            lps = [lp for lp in r["laps"] if lp["track"] == track]
            out.append(f"  track {track} ({lps[0]['mode']}, {lps[0]['count']} waypoints)")
            out.append("  " + LAP_HEAD)
            for lp in lps:
                out.append("  " + _lap_line(lp))
                if segments:
                    for s in r["segments"]:
                        if (s["track"], s["chain"], s["lap"]) == (track, lp["chain"], lp["lap"]):
                            out.append(_seg_line(s))
    out.append("")
    out.append("== per model and track (95% bootstrap over laps)")
    for row in summary:
        out.append(f"  {_short(row['model'])}  on  {row['track']}")
        out.append(f"    laps {row['laps']} ({row['lap_equiv']} lap-equivalents), full "
                   f"{row['full_laps']}, clean {row['clean_laps']}; segments "
                   f"{row['segments']}, clean {row['clean_segments']}; fallback "
                   f"{100 * row['fallback']:.1f}% of rows")
        out.append(f"    clean lap s {_ci(row['clean_lap_s'])}   clean segment s/m "
                   f"{_ci(row['clean_s_per_m'], '{:.2f}')}   time per lap s "
                   f"{_ci(row['time_per_lap_s'])}")
        out.append(f"    per lap: skips {_ci(row['skips_per_lap'], '{:.2f}')}"
                   f"   takeovers {_ci(row['takeovers_per_lap'], '{:.2f}')}"
                   f"   tips {_ci(row['tips_per_lap'], '{:.2f}')}"
                   f"   tilt blocks {_ci(row['tilts_per_lap'], '{:.2f}')}")
    if models:
        out.append("")
        out.append("== per model, every track (per 100 m of legs; 95% bootstrap over laps)")
        out.append(f"  {'model':<20} {'segs':>5} {'clean':>5} {'leg_m':>6} "
                   f"{'s/m all':>7}  {'clean s/m':<20} {'skips':<20} {'takeovers':<20} "
                   f"{'tips':<20}")
        for row in models:
            out.append(f"  {_short(row['model']):<20} {row['segments']:>5} "
                       f"{row['clean_segments']:>5} {row['leg_m']:>6.0f} "
                       f"{_f(row['s_per_m_all'], '{:.2f}'):>7}  "
                       f"{_ci(row['clean_s_per_m'], '{:.2f}'):<20} "
                       f"{_ci(row['skips_per_100m'], '{:.2f}'):<20} "
                       f"{_ci(row['takeovers_per_100m'], '{:.2f}'):<20} "
                       f"{_ci(row['tips_per_100m'], '{:.2f}'):<20}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mcap", nargs="+")
    ap.add_argument("--json", help="write runs, laps, segments and the summary here")
    ap.add_argument("--segments", action="store_true", help="a line per segment")
    ap.add_argument("--boot", type=int, default=BOOT)
    args = ap.parse_args(argv)
    runs, warnings = [], []
    for sid, (name, paths) in enumerate(runs_of(args.mcap)):
        msgs = [m for p in paths for m in read_mcap(p)]
        r = analyse_run(name, msgs, sid)
        r["files"] = [os.path.basename(p) for p in paths]
        warnings += r["warnings"]
        runs.append(r)
    summary = summarise(runs, draws=args.boot)
    models = by_model(runs, draws=args.boot)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(text(runs, summary, args.segments, models))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"runs": runs, "summary": summary, "by_model": models,
                       "warnings": warnings}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
