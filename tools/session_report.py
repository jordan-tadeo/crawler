"""Say whether a drive is any good as training data, one file at a time.

    uv run --with mcap python tools/session_report.py ~/crawler-data/mcap/crawler-record-*.mcap
    uv run --with mcap --with numpy==1.26.4 --with opencv-python-headless \
        python tools/session_report.py run.mcap      # depth off /camera/depth too

The question a session has to answer is not "did it record", which
deploy/mcap_check.py already answers, but "is there anything in here a policy
can learn from". A drive that is all open floor at half throttle teaches
nothing about the one situation that matters, the wall in front of the car, so
what this counts is the time spent close to things and the reversals that
happened there.

Read-only. It never repairs a file and it never writes anything; a run with no
index is read in streaming order exactly like mcap_check does.

Where the numbers come from
---------------------------

  Ticks are /vehicle, 20 Hz, in every mode: the commanded throttle and
  steering as they reached the wheels, and which mode the car was in. Mode
  time is the sum of the gaps between ticks, with any gap longer than
  TICK_GAP_CAP_S counted as a hole rather than as driving.

  Nearest obstacle is /footprint when the file has it (record and auto on the
  manual-recording build): the smallest of the nine probe distances, which is
  the same footprint band the driver blocks on. Without it, /camera/depth is
  read back -- but that topic is a JET-colourmapped JPEG clipped to 300..3000
  mm, so the read-back is an estimate (JPEG noise is a few colourmap steps,
  roughly +-30 mm) and it CANNOT see under 300 mm: a pixel at the bottom of the
  colourmap is either nearer than 300 or no reading at all, and the picture
  cannot say which, so those pixels are treated as no reading. Close under
  300 mm is therefore only ever counted from /footprint.

  Everything the new record-mode topics are named is in the adapter right
  below, so a renamed field is a one-line fix here and nowhere else.
"""

import argparse
import json
import os
import statistics
import sys
from typing import NamedTuple

# ---------------------------------------------------------------------------
# The adapter. Topic and field names for the record-mode topics, which are
# still moving on the manual-recording branch. TRAINING_DATA.md there is the
# contract; if it changes, this is the only block that should have to.
# ---------------------------------------------------------------------------

FOOTPRINT_TOPIC = "/footprint"
DEPTH_SMALL_TOPIC = "/policy/depth_small"


def footprint_row(row):
    """One /footprint JSON row as the handful of things this report reads.

    Returns {mode, throttle, steering_front, steering_rear, steering_mode,
    nearest_mm, seq}. nearest_mm is None when no probe had a reading: a probe
    distance of -1 means infinite in driver.py, and that is "nothing near",
    not "touching".
    """
    probes = row.get("probe_distance_mm") or []
    real = [float(p) for p in probes if _is_num(p) and float(p) >= 0]
    return {
        "mode": str(row.get("mode", "")),
        "throttle": _f(row.get("throttle")),
        "steering_front": _f(row.get("steering_front")),
        "steering_rear": _f(row.get("steering_rear")),
        "steering_mode": str(row.get("steering_mode", "")),
        "nearest_mm": min(real) if real else None,
        "seq": int(_f(row.get("seq"), -1)),
    }


# ---------------------------------------------------------------------------
# Thresholds. Every one of these is printed with the report it shaped.
# ---------------------------------------------------------------------------

VEHICLE_TOPIC = "/vehicle"
CONTROLLER_TOPIC = "/controller"
DIAGNOSTICS_TOPIC = "/diagnostics"
DEPTH_TOPIC = "/camera/depth"
MARKS_TOPIC = "/marks"

# |throttle| under this is stopped. A trigger at rest reads exactly 0 off the
# pad, so this only has to absorb a resting finger.
STOP_BAND = 0.05

# Two reverse runs closer together than this are one reversal: a feathered
# trigger that dips through the stop band for a tick is not two decisions.
MERGE_GAP_S = 0.3

# A reversal shorter than this is a twitch, not a reversal.
MIN_REVERSE_S = 0.15

# Near an obstacle: the nearest depth anywhere in the second before the
# reversal started (the approach that caused it) up to a moment after.
NEAR_MM = 500.0
VERY_NEAR_MM = 300.0
NEAR_LOOKBACK_S = 1.0
NEAR_LEAD_S = 0.2

# A tick whose newest depth reading is older than this has stale depth.
DEPTH_STALE_S = 0.5

# A hole in the depth stream this long, while the camera says it is ready, is
# a dropout. The JPEG topic is 8 Hz and /footprint 10 Hz, so 1 s is eight
# missed frames.
DEPTH_DROPOUT_S = 1.0

# Any gap between /vehicle ticks longer than this is not counted as time in a
# mode. 20 Hz is 0.05 s; 0.5 s is ten missed ticks.
TICK_GAP_CAP_S = 0.5

# A hole in /vehicle this long is the pad gone: the control loop parks.
PAD_HOLE_S = 2.0

# Modes the camera is open in, so the only modes depth can be asked about.
DEPTH_MODES = ("auto", "record")

# The verdict.
MIN_DEPTH_MINUTES = 1.0       # less depth-mode time than this is too short
GOOD_CLOSE_FRACTION = 0.15    # of depth-mode time spent under NEAR_MM
GOOD_NEAR_REVERSALS = 3       # reversals that started near something
STRAIGHT_STEER = 0.1          # |steering_front| under this is straight
TOO_STRAIGHT_FRACTION = 0.8   # of moving ticks

THROTTLE_EDGES = [-1.0, -0.6, -0.3, -STOP_BAND, STOP_BAND, 0.3, 0.6, 1.0]
# The pad's LEFT_X times 70%, so the servos see -0.7..0.7.
STEER_EDGES = [-0.7, -0.5, -0.3, -STRAIGHT_STEER, STRAIGHT_STEER, 0.3, 0.5, 0.7]

# Where the footprint band sits on a depth frame, as fractions of it, for the
# /camera/depth fallback: driver.footprint_geometry() on 640x400 is rows
# 134..282 and columns 151..489 in nine probes. Fractions so the 320x200 JPEG
# lands on the same patch. Copied rather than imported, because this runs on
# the laptop under `uv run --with mcap` and driver.py is moving on another
# branch.
BAND_ROWS = (134 / 400.0, 282 / 400.0)
BAND_COLS = (151 / 640.0, 489 / 640.0)
BAND_PROBES = 9
# The k-th smallest per probe, as driver.py does, so a speckle cannot claim it.
BAND_NEAR_FRACTION = 0.02
# The colourmap window telemetry.py renders depth across.
JET_NEAR_MM = 300.0
JET_FAR_MM = 3000.0
# Colourmap steps at the bottom that mean "<=300 or no reading" once JPEG has
# smeared them, and at the top that mean ">=3000".
JET_FLOOR_STEPS = 4
JET_CEIL_STEPS = 4


def _is_num(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _f(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# The pure half: ticks in, numbers out. Everything the tests drive.
# ---------------------------------------------------------------------------

class Tick(NamedTuple):
    t: float               # seconds into the file
    mode: str
    throttle: float
    steering_front: float
    steering_rear: float
    steering_mode: str


class Event(NamedTuple):
    start: float
    end: float
    mean_throttle: float
    peak_throttle: float   # the most negative
    mode: str = ""         # the mode the reversal started in

    @property
    def duration(self):
        return self.end - self.start


def tick_durations(ticks, cap=TICK_GAP_CAP_S):
    """How long each tick stood for: the gap to the next one, capped.

    The last tick gets the median gap, so a two-tick file is not zero long.
    """
    if not ticks:
        return []
    gaps = [b.t - a.t for a, b in zip(ticks, ticks[1:])]
    typical = statistics.median(gaps) if gaps else 0.05
    out = [g if 0 <= g <= cap else 0.0 for g in gaps]
    out.append(min(typical, cap))
    return out


def mode_split(ticks):
    """{mode: seconds}."""
    split = {}
    for tick, dt in zip(ticks, tick_durations(ticks)):
        split[tick.mode] = split.get(tick.mode, 0.0) + dt
    return split


def throttle_counts(ticks, band=STOP_BAND):
    """(forward, reverse, stop) tick counts."""
    fwd = sum(1 for t in ticks if t.throttle > band)
    rev = sum(1 for t in ticks if t.throttle < -band)
    return fwd, rev, len(ticks) - fwd - rev


def histogram(values, edges):
    """Counts per [edges[i], edges[i+1]) bin, the last bin closed. Values
    outside the edges land in the end bins, so nothing is silently dropped."""
    counts = [0] * (len(edges) - 1)
    for v in values:
        i = 0
        while i < len(counts) - 1 and v >= edges[i + 1]:
            i += 1
        counts[i] += 1
    return counts


def reverse_events(ticks, band=STOP_BAND, merge_gap=MERGE_GAP_S,
                   min_duration=MIN_REVERSE_S):
    """Every reversal: a run of ticks under -band, with short dips merged.

    A run ends at the first tick that is not reversing; its end time is that
    tick's time (or the last tick's, if the file ends mid-reverse). Two runs
    whose gap is under merge_gap become one. A run shorter than min_duration
    after merging is dropped.
    """
    runs = []  # [start, end, [throttles]]
    current = None
    for i, tick in enumerate(ticks):
        if tick.throttle < -band:
            if current is None:
                current = [tick.t, tick.t, [], tick.mode]
            current[2].append(tick.throttle)
            nxt = ticks[i + 1].t if i + 1 < len(ticks) else tick.t
            current[1] = nxt
        elif current is not None:
            runs.append(current)
            current = None
    if current is not None:
        runs.append(current)

    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] < merge_gap:
            merged[-1][1] = run[1]
            merged[-1][2].extend(run[2])
        else:
            merged.append([run[0], run[1], list(run[2]), run[3]])

    return [Event(s, e, sum(th) / len(th), min(th), mode)
            for s, e, th, mode in merged if e - s >= min_duration]


class DepthSeries:
    """(t, nearest_mm or None) samples, oldest first, with time lookups."""

    def __init__(self, samples):
        self.samples = sorted(samples, key=lambda s: s[0])
        self.times = [s[0] for s in self.samples]

    def __bool__(self):
        return bool(self.samples)

    def latest_at(self, t):
        """The newest sample at or before t, or None."""
        import bisect
        i = bisect.bisect_right(self.times, t) - 1
        return self.samples[i] if i >= 0 else None

    def min_between(self, start, end):
        """Nearest reading in [start, end], or None if none had one."""
        import bisect
        lo = bisect.bisect_left(self.times, start)
        hi = bisect.bisect_right(self.times, end)
        vals = [mm for _, mm in self.samples[lo:hi] if mm is not None]
        return min(vals) if vals else None


def near_reversals(events, depth, threshold=NEAR_MM,
                   lookback=NEAR_LOOKBACK_S, lead=NEAR_LEAD_S):
    """How many reversals had something under threshold around their start,
    and how many could be asked at all (had any depth reading in the window)."""
    near = asked = 0
    for ev in events:
        mm = depth.min_between(ev.start - lookback, ev.start + lead)
        if mm is None:
            continue
        asked += 1
        if mm < threshold:
            near += 1
    return near, asked


def depth_tick_stats(ticks, depth, thresholds=(NEAR_MM, VERY_NEAR_MM),
                     stale=DEPTH_STALE_S, modes=DEPTH_MODES):
    """Over the ticks in a camera mode: time under each threshold, time with
    fresh depth, and how many ticks had stale or no depth.

    Returns {"depth_s", "fresh_s", "stale_ticks", "close_s": {mm: s}}.
    """
    out = {"depth_s": 0.0, "fresh_s": 0.0, "stale_ticks": 0,
           "close_s": {mm: 0.0 for mm in thresholds}}
    for tick, dt in zip(ticks, tick_durations(ticks)):
        if tick.mode not in modes:
            continue
        out["depth_s"] += dt
        sample = depth.latest_at(tick.t) if depth else None
        if sample is None or tick.t - sample[0] > stale:
            out["stale_ticks"] += 1
            continue
        out["fresh_s"] += dt
        mm = sample[1]
        if mm is None:
            continue
        for limit in thresholds:
            if mm < limit:
                out["close_s"][limit] += dt
    return out


def disconnects(samples):
    """[(t, connected)] -> (count of connected->disconnected drops, seconds
    spent disconnected). A file that starts disconnected counts that stretch
    as time but not as a drop."""
    drops, down, since = 0, 0.0, None
    prev = None
    for t, connected in samples:
        if prev is not None and prev and not connected:
            drops += 1
        if not connected and since is None:
            since = t
        if connected and since is not None:
            down += t - since
            since = None
        prev = connected
    if since is not None and samples:
        down += samples[-1][0] - since
    return drops, down


def dropouts(depth_times, ready_spans, gap=DEPTH_DROPOUT_S):
    """Holes in the depth stream longer than gap that fall inside a span the
    camera reported ready. ready_spans: [(start, end)]. Returns [(start, s)]."""
    holes = []
    times = sorted(depth_times)
    for start, end in ready_spans:
        inside = [t for t in times if start <= t <= end]
        edges = [start] + inside + [end]
        for a, b in zip(edges, edges[1:]):
            if b - a > gap:
                holes.append((a, b - a))
    return holes


def tick_holes(tick_times, span_end, gap=PAD_HOLE_S):
    """Stretches with no /vehicle tick longer than gap, as [(start, s)],
    including one running off the end of the file.

    This is how a lost pad shows up in a recording. crawler/__main__.py parks the loop
    while the pad is gone, so /vehicle and /controller just stop; the one
    /diagnostics row meant to say "disconnected" is behind the 4 Hz rate gate
    and, in every file checked, never made it in.
    """
    holes = []
    times = sorted(tick_times)
    for a, b in zip(times, times[1:]):
        if b - a > gap:
            holes.append((a, b - a))
    if times and span_end - times[-1] > gap:
        holes.append((times[-1], span_end - times[-1]))
    return holes


def spans(samples, predicate):
    """[(t, value)] -> [(start, end)] stretches where predicate(value) held."""
    out, start = [], None
    for t, value in samples:
        if predicate(value):
            if start is None:
                start = t
        elif start is not None:
            out.append((start, t))
            start = None
    if start is not None and samples:
        out.append((start, samples[-1][0]))
    return out


def verdict(stats):
    """One line, from the dict summarise() builds."""
    depth_min = stats["depth_s"] / 60.0
    if stats["depth_s"] <= 0:
        return ("no auto or record time: manual writes no training rows, so "
                "this is driving habits only")
    if not stats["has_depth"]:
        return ("no depth in this file at all: nothing a policy can train on")
    if depth_min < MIN_DEPTH_MINUTES:
        return (f"too short: {stats['depth_s']:.0f} s in auto/record, want at "
                f"least {MIN_DEPTH_MINUTES:.0f} min")
    record_s = stats["mode_s"].get("record", 0.0)
    who = "" if record_s >= stats["depth_s"] / 2 else \
        " (mostly auto, so these are the autopilot's recoveries, not mine)"
    close = stats["close_s"].get(NEAR_MM, 0.0) / stats["depth_s"]
    near = stats["near_reversals"]
    if close >= GOOD_CLOSE_FRACTION and near >= GOOD_NEAR_REVERSALS:
        line = "good recovery coverage"
    elif close >= GOOD_CLOSE_FRACTION:
        line = ("plenty of close passes but few reversals near anything: "
                "back out of more of them")
    else:
        line = "mostly open-floor driving, record more close passes"
    if stats["moving_ticks"] and stats["straight_fraction"] >= TOO_STRAIGHT_FRACTION:
        line += (f"; and {stats['straight_fraction']:.0%} of moving ticks "
                 f"were straight, steer more")
    return line + who


def verdict_rules():
    return (f"verdict rules: good = >={GOOD_CLOSE_FRACTION:.0%} of camera time "
            f"under {NEAR_MM:.0f} mm AND >={GOOD_NEAR_REVERSALS} reversals "
            f"starting near something; under {MIN_DEPTH_MINUTES:.0f} min of "
            f"camera time is too short; >={TOO_STRAIGHT_FRACTION:.0%} of moving "
            f"ticks with |steer|<{STRAIGHT_STEER} is too straight")


def summarise(ticks, depth, controller, ready_spans, depth_times, marks,
              footprint_seq_gaps=0, has_depth=False):
    """Everything the report prints, as one dict. Pure."""
    durations = tick_durations(ticks)
    events = reverse_events(ticks)
    near, asked = near_reversals(events, depth) if depth else (0, 0)
    dstats = depth_tick_stats(ticks, depth)
    moving = [t for t in ticks if abs(t.throttle) > STOP_BAND]
    straight = sum(1 for t in moving if abs(t.steering_front) < STRAIGHT_STEER)
    split_ticks = sum(1 for t in ticks if t.steering_mode == "split")
    split_s = sum(dt for t, dt in zip(ticks, durations)
                  if t.steering_mode == "split")
    # Split shows as rear != -front even if the mode string is ever missing.
    rear_independent = sum(
        1 for t in ticks
        if abs(t.steering_rear + t.steering_front) > 0.02)
    drops, down = disconnects(controller)
    return {
        "mode_s": mode_split(ticks),
        "driven_s": sum(durations),
        "throttle_counts": throttle_counts(ticks),
        "throttle_hist": histogram([t.throttle for t in ticks], THROTTLE_EDGES),
        "steer_hist": histogram([t.steering_front for t in moving], STEER_EDGES),
        "moving_ticks": len(moving),
        "straight_fraction": straight / len(moving) if moving else 0.0,
        "split_ticks": split_ticks,
        "split_s": split_s,
        "rear_independent_ticks": rear_independent,
        "events": events,
        "near_reversals": near,
        "near_asked": asked,
        "depth_s": dstats["depth_s"],
        "fresh_s": dstats["fresh_s"],
        "stale_ticks": dstats["stale_ticks"],
        "close_s": dstats["close_s"],
        "pad_drops": drops,
        "pad_down_s": down,
        "dropouts": dropouts(depth_times, ready_spans),
        "footprint_seq_gaps": footprint_seq_gaps,
        "marks": marks,
        "has_depth": has_depth,
    }


def manual_reversing(events, manual_s):
    """The habit numbers for hand driving: (per minute, median duration,
    median mean-throttle, median peak-throttle). None where there are none."""
    if manual_s <= 0:
        return None
    if not events:
        return (0.0, None, None, None)
    return (len(events) / (manual_s / 60.0),
            statistics.median(e.duration for e in events),
            statistics.median(e.mean_throttle for e in events),
            statistics.median(e.peak_throttle for e in events))


# ---------------------------------------------------------------------------
# The reading half: an MCAP in, the inputs above out.
# ---------------------------------------------------------------------------

def _proto_fields(buf):
    """Top-level length-delimited fields of a protobuf message, {num: bytes}.

    Just enough protobuf to get the image bytes out of a
    foxglove.CompressedImage (data is field 2) without a protobuf dependency.
    """
    fields, i, n = {}, 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        num, wire = key >> 3, key & 7
        if wire == 0:
            _, i = _varint(buf, i)
        elif wire == 1:
            i += 8
        elif wire == 2:
            size, i = _varint(buf, i)
            fields[num] = buf[i:i + size]
            i += size
        elif wire == 5:
            i += 4
        else:
            break
    return fields


def _varint(buf, i):
    shift = value = 0
    while True:
        byte = buf[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, i
        shift += 7


class JetReader:
    """Nearest depth in the footprint band, read back off a JET JPEG.

    numpy and cv2 are imported here and only here: without them the report
    still runs and just says it could not read /camera/depth.
    """

    def __init__(self):
        import cv2
        import numpy as np
        self.cv2, self.np = cv2, np
        lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1),
                                cv2.COLORMAP_JET).reshape(256, 3).astype(np.int32)
        # A 64^3 cube from quantised BGR to the nearest colourmap step, so
        # each pixel is one table lookup instead of 256 distances. 64 and not
        # 32: the bottom of JET moves 4 levels of blue per step, so a coarser
        # cube can only tell every other step apart.
        q = (np.arange(64) * 4 + 2).astype(np.int32)
        b, g, r = np.meshgrid(q, q, q, indexing="ij")
        cube = np.stack([b, g, r], axis=-1).reshape(-1, 3)
        out = np.empty(len(cube), np.uint8)
        for i in range(0, len(cube), 16384):
            chunk = cube[i:i + 16384, None, :]
            out[i:i + 16384] = ((chunk - lut[None, :, :]) ** 2).sum(-1).argmin(1)
        self.cube = out.reshape(64, 64, 64)
        self.shift = 2

    def nearest(self, jpeg):
        cv2, np = self.cv2, self.np
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        band = img[int(h * BAND_ROWS[0]):int(h * BAND_ROWS[1]),
                   int(w * BAND_COLS[0]):int(w * BAND_COLS[1])] >> self.shift
        steps = self.cube[band[..., 0], band[..., 1], band[..., 2]].astype(np.int32)
        cols = np.array_split(steps, BAND_PROBES, axis=1)
        best = None
        for probe in cols:
            flat = probe.ravel()
            valid = flat[flat >= JET_FLOOR_STEPS]
            if valid.size == 0:
                continue
            k = max(1, int(round(BAND_NEAR_FRACTION * flat.size)))
            if valid.size < k:
                continue
            kth = int(np.partition(valid, k - 1)[k - 1])
            if kth >= 255 - JET_CEIL_STEPS:
                continue
            mm = JET_NEAR_MM + kth * (JET_FAR_MM - JET_NEAR_MM) / 255.0
            best = mm if best is None else min(best, mm)
        return best


def read_session(path, want_jpeg_depth=True):
    """One pass over a file. Returns the raw material summarise() wants."""
    from mcap.records import Channel, Message, Schema
    from mcap.stream_reader import StreamReader

    channels, schemas = {}, {}
    vehicle, controller, diag, marks = [], [], [], []
    footprint, jpeg_depth = [], []
    pad_diag = []
    seqs = []
    t_first = t_last = None
    truncated = False
    jet = None
    jet_error = ""
    have_footprint = False

    def topic_of(message):
        ch = channels.get(message.channel_id)
        return ch.topic if ch else ""

    with open(path, "rb") as handle:
        try:
            for record in StreamReader(handle).records:
                if isinstance(record, Schema):
                    schemas[record.id] = record
                    continue
                if isinstance(record, Channel):
                    channels[record.id] = record
                    if record.topic == FOOTPRINT_TOPIC:
                        have_footprint = True
                    continue
                if not isinstance(record, Message):
                    continue
                t = record.log_time / 1e9
                t_first = t if t_first is None else min(t_first, t)
                t_last = t if t_last is None else max(t_last, t)
                topic = topic_of(record)
                if topic == DEPTH_TOPIC:
                    if not want_jpeg_depth or have_footprint:
                        continue
                    if jet is None and not jet_error:
                        try:
                            jet = JetReader()
                        except Exception as e:  # no numpy / cv2
                            jet_error = str(e)
                    if jet is None:
                        jpeg_depth.append((t, None))
                        continue
                    data = _proto_fields(record.data).get(2, b"")
                    jpeg_depth.append((t, jet.nearest(data)))
                    continue
                if topic not in (VEHICLE_TOPIC, CONTROLLER_TOPIC,
                                 DIAGNOSTICS_TOPIC, MARKS_TOPIC,
                                 FOOTPRINT_TOPIC):
                    continue
                try:
                    row = json.loads(record.data.decode("utf-8"))
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                if topic == VEHICLE_TOPIC:
                    vehicle.append((t, row))
                elif topic == CONTROLLER_TOPIC:
                    controller.append((t, bool(row.get("connected", False))))
                elif topic == DIAGNOSTICS_TOPIC:
                    diag.append((t, row))
                    if "controller_connected" in row:
                        pad_diag.append(
                            (t, bool(row.get("controller_connected"))))
                elif topic == MARKS_TOPIC:
                    marks.append((t, row))
                else:
                    fp = footprint_row(row)
                    footprint.append((t, fp["nearest_mm"]))
                    seqs.append(fp["seq"])
        except Exception:
            truncated = True

    t0 = t_first or 0.0
    vehicle.sort(key=lambda p: p[0])
    ticks = [Tick(t - t0, str(r.get("mode", "?")), _f(r.get("throttle")),
                  _f(r.get("steering_front")), _f(r.get("steering_rear")),
                  str(r.get("steering_mode", "")))
             for t, r in vehicle]
    source = "footprint" if footprint else ("camera/depth JPEG"
                                            if jpeg_depth else "")
    samples = footprint if footprint else jpeg_depth
    depth = DepthSeries([(t - t0, mm) for t, mm in samples])
    diag.sort(key=lambda p: p[0])
    ready = spans([(t - t0, r.get("camera_state")) for t, r in diag],
                  lambda s: s == "ready")
    seq_sorted = sorted(s for s in seqs if s >= 0)
    gaps = sum(max(0, b - a - 1) for a, b in zip(seq_sorted, seq_sorted[1:]))
    return {
        "path": path,
        "size": os.path.getsize(path),
        "span_s": (t_last - t_first) if t_first is not None else 0.0,
        "truncated": truncated,
        "ticks": ticks,
        "depth": depth,
        "depth_source": source,
        "jet_error": jet_error,
        # /diagnostics first: when the pad goes, the control loop parks and
        # /controller simply stops, so the only message that ever says
        # "disconnected" is the one diagnostics row it sends on the way out.
        "controller": [(t - t0, c) for t, c in sorted(pad_diag or controller)],
        "ready_spans": ready,
        "depth_times": [t for t, _ in depth.samples],
        "marks": [(t - t0, r) for t, r in sorted(marks, key=lambda p: p[0])],
        "seq_gaps": gaps,
    }


# ---------------------------------------------------------------------------
# Printing.
# ---------------------------------------------------------------------------

def _bars(counts, edges, width=30):
    total = sum(counts) or 1
    peak = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        label = f"{edges[i]:+.2f}..{edges[i + 1]:+.2f}"
        bar = "#" * int(round(width * c / peak))
        lines.append(f"      {label:>13}  {c:>6}  {c / total:>4.0%}  {bar}")
    return lines


def _mmss(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def render(session, stats):
    lines = []
    name = os.path.basename(session["path"])
    mb = session["size"] / 1e6
    span = session["span_s"]
    lines.append(f"{name}")
    note = " (no index / truncated tail; read in streaming order)" \
        if session["truncated"] else ""
    lines.append(f"  duration   {_mmss(span)} ({span:.0f} s), {mb:.1f} MB, "
                 f"{mb / (span / 60.0) if span > 0 else 0:.1f} MB/min{note}")
    if not session["ticks"]:
        lines.append("  no /vehicle ticks: nothing driven in this file")
        return lines

    split = " / ".join(f"{m} {_mmss(s)}" for m, s in
                       sorted(stats["mode_s"].items(), key=lambda kv: -kv[1]))
    lines.append(f"  modes      {split}")

    fwd, rev, stop = stats["throttle_counts"]
    n = fwd + rev + stop
    lines.append(f"  throttle   {n} ticks: forward {fwd} ({fwd / n:.0%}), "
                 f"reverse {rev} ({rev / n:.0%}), stopped {stop} "
                 f"({stop / n:.0%})  [stopped = |t|<{STOP_BAND}]")
    lines += _bars(stats["throttle_hist"], THROTTLE_EDGES)
    lines.append(f"  steering   over {stats['moving_ticks']} moving ticks, "
                 f"{stats['straight_fraction']:.0%} straight "
                 f"(|front|<{STRAIGHT_STEER})")
    lines += _bars(stats["steer_hist"], STEER_EDGES)
    if stats["split_ticks"]:
        lines.append(f"             split steering used: {_mmss(stats['split_s'])} "
                     f"({stats['split_ticks']} ticks)")
    else:
        lines.append("             split steering never used")

    events = stats["events"]
    minutes = stats["driven_s"] / 60.0
    lines.append(f"  reversals  {len(events)} "
                 f"({len(events) / minutes if minutes else 0:.1f}/min)"
                 + (f", median {statistics.median(e.duration for e in events):.1f} s, "
                    f"median throttle {statistics.median(e.mean_throttle for e in events):+.2f} "
                    f"(peak {statistics.median(e.peak_throttle for e in events):+.2f})"
                    if events else ""))
    by_mode = {}
    for ev in events:
        by_mode[ev.mode] = by_mode.get(ev.mode, 0) + 1
    if len(by_mode) > 1 or (by_mode and "manual" not in by_mode):
        lines.append("             by mode: " + ", ".join(
            f"{m} {n}" for m, n in sorted(by_mode.items())))
    if session["depth"]:
        lines.append(f"             {stats['near_reversals']} of "
                     f"{stats['near_asked']} with depth started near something "
                     f"(<{NEAR_MM:.0f} mm within -{NEAR_LOOKBACK_S:.1f}/"
                     f"+{NEAR_LEAD_S:.1f} s of the start)")
    else:
        lines.append("             near an obstacle: unknown, no depth in "
                     "this file")
    lines.append(f"             [one reversal = throttle <-{STOP_BAND} for "
                 f">={MIN_REVERSE_S} s, dips under {MERGE_GAP_S} s merged]")

    if stats["depth_s"] > 0:
        src = session["depth_source"] or "none"
        lines.append(f"  depth      from {src}; {_mmss(stats['depth_s'])} in "
                     f"{'/'.join(DEPTH_MODES)}, {_mmss(stats['fresh_s'])} with "
                     f"depth under {DEPTH_STALE_S} s old")
        if session["jet_error"]:
            lines.append(f"             could not read /camera/depth: "
                         f"{session['jet_error']} (needs numpy + "
                         f"opencv-python-headless)")
        for limit, secs in stats["close_s"].items():
            frac = secs / stats["depth_s"] if stats["depth_s"] else 0.0
            if limit < JET_NEAR_MM + 1 and session["depth_source"] != "footprint":
                lines.append(f"             under {limit:.0f} mm: not measurable "
                             f"from the JPEG (it clips at {JET_NEAR_MM:.0f})")
                continue
            lines.append(f"             under {limit:.0f} mm: {_mmss(secs)} "
                         f"({frac:.0%})")
        if session["depth_source"] != "footprint" and session["depth"]:
            lines.append("             [JPEG read-back: ESTIMATED, about "
                         "+-30 mm, and blind under 300 mm]")

    gaps = []
    holes = tick_holes([t.t for t in session["ticks"]], session["span_s"])
    if holes:
        gaps.append(f"pad gone {len(holes)}x by the ticks, "
                    f"{sum(s for _, s in holes):.0f} s with no /vehicle "
                    f"(the loop parks only when the pad drops; first at "
                    f"t+{holes[0][0]:.0f}s)")
    if stats["pad_drops"] or stats["pad_down_s"] > 0:
        gaps.append(f"pad dropped {stats['pad_drops']}x by /diagnostics, "
                    f"{stats['pad_down_s']:.1f} s disconnected")
    if stats["dropouts"]:
        worst = max(s for _, s in stats["dropouts"])
        gaps.append(f"{len(stats['dropouts'])} depth dropouts "
                    f">{DEPTH_DROPOUT_S:.0f} s (worst {worst:.1f} s)")
    if stats["stale_ticks"]:
        gaps.append(f"{stats['stale_ticks']} ticks with stale or no depth "
                    f"(>{DEPTH_STALE_S} s)")
    if stats["footprint_seq_gaps"]:
        gaps.append(f"{stats['footprint_seq_gaps']} /footprint rows missing "
                    f"by seq")
    lines.append("  gaps       " + ("; ".join(gaps) if gaps else "none"))

    marks = stats["marks"]
    if marks:
        good = sum(1 for _, r in marks if r.get("kind") == "good")
        lines.append(f"  marks      {len(marks)} ({good} good, "
                     f"{len(marks) - good} bad)")
        for t, r in marks:
            lines.append(f"             {str(r.get('kind', '?')).upper():<4} "
                         f"t+{t:.1f}s  {r.get('utc', '?')}  "
                         f"{r.get('mode', '?')}")
    else:
        lines.append("  marks      none")

    lines.append(f"  verdict    {verdict(stats)}")
    return lines


def collect(argv):
    """Files on the command line, a directory counting as its recordings."""
    targets = []
    for arg in argv:
        if os.path.isdir(arg):
            targets += [os.path.join(arg, n) for n in sorted(os.listdir(arg))
                        if n.startswith("crawler-") and n.endswith(".mcap")]
        else:
            targets.append(arg)
    return targets


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Is this drive any use as training data?")
    parser.add_argument("targets", nargs="+",
                        help="mcap files, or directories of them")
    parser.add_argument("--no-jpeg-depth", action="store_true",
                        help="skip reading /camera/depth back (fast; only "
                             "/footprint gives depth then)")
    parser.add_argument("--min-seconds", type=float, default=5.0,
                        help="skip files with fewer seconds of ticks than "
                             "this (default 5)")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    targets = collect(args.targets)
    if not targets:
        print("No crawler-*.mcap files to report on.")
        return 0
    manual_events, manual_s, manual_moving_s = [], 0.0, 0.0
    for path in targets:
        try:
            session = read_session(path, not args.no_jpeg_depth)
        except Exception as e:
            print(f"{os.path.basename(path)}\n  could not read it: {e}\n")
            continue
        stats = summarise(session["ticks"], session["depth"],
                          session["controller"], session["ready_spans"],
                          session["depth_times"], session["marks"],
                          session["seq_gaps"], bool(session["depth"]))
        if stats["driven_s"] < args.min_seconds:
            print(f"{os.path.basename(path)}  {session['size'] / 1e6:.1f} MB "
                  f"over {session['span_s']:.0f} s, {stats['driven_s']:.0f} s "
                  f"of /vehicle ticks: skipped"
                  + (" (camera kept recording with the pad gone)"
                     if session["span_s"] > args.min_seconds else "") + "\n")
            continue
        print("\n".join(render(session, stats)))
        print()
        manual_s += stats["mode_s"].get("manual", 0.0)
        manual_moving_s += sum(
            dt for t, dt in zip(session["ticks"],
                                tick_durations(session["ticks"]))
            if t.mode == "manual" and abs(t.throttle) > STOP_BAND)
        manual_events += [e for e in stats["events"] if e.mode == "manual"]
    print(verdict_rules())
    if manual_s > 0:
        habit = manual_reversing(manual_events, manual_s)
        moving_rate = (len(manual_events) / (manual_moving_s / 60.0)
                       if manual_moving_s > 0 else 0.0)
        line = (f"hand driving, all files: {len(manual_events)} reversals in "
                f"{_mmss(manual_s)} of manual, {habit[0]:.1f}/min "
                f"({moving_rate:.1f} per minute with the throttle open, "
                f"{_mmss(manual_moving_s)} of that)")
        if manual_events:
            line += (f", median {habit[1]:.1f} s, median throttle "
                     f"{habit[2]:+.2f} (peak {habit[3]:+.2f})")
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
