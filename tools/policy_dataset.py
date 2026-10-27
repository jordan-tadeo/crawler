"""MCAP recordings -> one .npz training set for the driving policy.

    python tools/policy_dataset.py OUT.npz [REC.mcap ...]

With no files it reads every ~/crawler-data/mcap/crawler-record-*.mcap, where
tools/pull_mcaps.py leaves the car's recordings.

Each file is one session, and a session is the unit the train/val split is
made in (policy_train.py --val), never a random tick: ticks a tenth of a
second apart are nearly the same picture, and a random split would grade the
model on frames it has effectively seen.

What one tick is
----------------
A 10 Hz decision, the rate the autopilot decides at and the rate a model will
be run at. In a legacy recording there is no tick id: /vehicle is sampled
down to 10 Hz and each tick takes the newest /camera/depth frame at or
before it, which is the frame a decision at that instant could have used.
The alignment error is that frame's age. In a record-mode recording the row
and the frame share a seq, and the error is their log-time difference.

What gets dropped, and what is only left out of training
--------------------------------------------------------
Dropped, and counted in the report: a mode that makes no ticks, no frame at
all yet, a frame older than --max-age-ms, the pad disconnected, a command
that is not finite, and split steering (the model has no rear head until
there is data that needs one -- the report says how much there was).

Kept, with a mask: whether each group -- steering, and direction plus
magnitude -- came off the pad. policy_adapters.control_sources() decides it
from steer_by / throttle_by; the trainer drops a "driver" group from that
head's loss unless it is told to clone the autopilot.

Kept but not trained on: the ticks inside a recover. A reverse is a two-second
phase machine, and only the tick that started it was a decision; the rest
are the machine. They stay in the file so the eval can say what the model
would have done at a /marks window that landed in one.

Dropped as not driving (pad-driven modes only, record and manual; IdleRules, every number a
CLI option): the car parked or in my hands is not a decision, and a model
that learns from it learns to sit still.

  idle lead / idle trail  everything before the first and after the last
                          throttle in the session: setting up, putting down.
  exit reach              EXIT_REACH_S before each end of a record run (the
                          Y tap, or START): that is my thumb leaving the
                          stick for the button.
  parked                  no input -- throttle inside THROTTLE_DEADBAND and
                          front steering inside the stick's rest band -- for
                          longer than PARKED_S. The span goes, all but
                          PARKED_EDGE_S at each end: the stop before a
                          reverse is a decision, and so is the start after.
                          A stop shorter than PARKED_S is kept whole.
  handled                 no input, at least HANDLED_SETTLE_S after the last
                          input, and the IMU moving like nothing that
                          sitting still does: the car being picked up or
                          carried. Spread HANDLED_MARGIN_S either side, over
                          no-input ticks only. A tick with any pad input is
                          never dropped by this, however violent the IMU:
                          that is rough ground, and rough ground is wanted.

Flagged by a /marks press: on a tick the pad was driving (RECORD, or the
pad-owned group of a takeover), a mark is a verdict on the label. Every such
tick inside a BAD mark's window gets mark = MARK_BAD, and the trainer and the
val scores leave it out; inside a GOOD window it gets MARK_GOOD, and the
trainer weights it up (policy_train.py --good-weight). BAD wins where the two
overlap. The window is the mark row's own window_start_s..window_end_s, put
on the log clock through the row's mono_s, never a hardcoded 4 s / 1 s. On a
tick the autopilot was driving the mark judges the autopilot, not a label, so
it stays a flag for policy_eval's marks section and nothing else.

Needs `mcap` and `mcap-protobuf-support`, which are tools for the laptop and
not dependencies of the car. The alignment below needs neither and is what
the tests exercise.
"""

import argparse
import bisect
import glob
import json
import math
import os
import sys
from collections import Counter
from typing import NamedTuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import driver as dv  # noqa: E402
import policy as pl  # noqa: E402
import policy_adapters as pa  # noqa: E402

TICK_S = 0.1
# Slack on the 10 Hz sampling of a 20 Hz channel, so jitter in /vehicle does
# not turn every other tick into a skipped one.
TICK_SLACK_S = 0.02
# The legacy depth channel arrives every 130-190 ms (measured), so a frame up
# to one frame interval plus a tick old is normal there; record mode writes
# the frame on the row's own log time and anything past a few ms is wrong.
MAX_AGE_MS = {"legacy": 250.0, "footprint": 20.0}
# Ticks further apart than this are not consecutive, and get no prev action.
PREV_GAP_S = 0.25
# How long after the last reverse tick a stop is the recover's settle phase
# rather than a decision: SETTLE_SECONDS plus a tick.
SETTLE_S = dv.SETTLE_SECONDS + TICK_S

# /marks verdicts on a pad-driven tick, as stored in the .npz "mark" array.
MARK_NONE, MARK_GOOD, MARK_BAD = 0, 1, -1

DROP_REASONS = ("mode", "no frame", "stale", "pad", "bad command", "split",
                "idle lead", "idle trail", "exit reach", "handled", "parked")
IDLE_REASONS = DROP_REASONS[6:]


class IdleRules(NamedTuple):
    """The not-driving rules' numbers. Measured where there was data to measure.

    The IMU was measured on 2026-10-27 over the eleven rolling recordings
    (OAK-D LINEAR_ACCELERATION + GYROSCOPE_RAW at /imu's 20 Hz, against
    /vehicle's throttle):

      state                             |accel| p99 / max   |gyro| p99 / max
      auto forward at 0.2 (12523)          2.21 / 15.3        0.36 / 1.91
      auto reverse at -0.6 (4266)          6.78 / 22.8        1.32 / 7.52
      at rest 0.5-1.0 s, manual (385)      1.35 / 3.33        0.15 / 0.23
      at rest > 10 s, manual (84614)       0.12 / 11.2        0.009 / 2.63

    Driving at the record caps shakes the car as hard as handling it does
    (reverse at -0.6 is the most violent thing in the files), which is why
    the rule only ever looks at ticks with no input. Half a second after
    the sticks are let go the coast is over -- nothing past it in 385
    manual samples reached 0.23 rad/s or 3.4 m/s^2 -- so the thresholds sit
    above that and above the forward-at-caps p99.9 (0.79 rad/s, 4.97 m/s^2)
    with room: 1.0 rad/s and 5.0 m/s^2. The long-rest tail past them is
    the handling these exist to catch. No recording has a pickup labelled
    as one, so how many real ones this misses is NOT measured.
    """

    steer_rest: float = 0.12 * 0.7   # the stick's 0.12 deadzone, through the 70% steering scale
    parked_s: float = 1.5
    parked_edge_s: float = 0.5
    exit_reach_s: float = 1.0
    handled_settle_s: float = 0.5
    handled_gyro: float = 1.0        # rad/s
    handled_accel: float = 5.0       # m/s^2
    handled_margin_s: float = 1.0
    run_gap_s: float = 0.5           # record rows further apart than this are two runs
    modes: tuple = pa.PAD_MODES      # which ticks the rules judge: the pad-driven ones


class Tick(NamedTuple):
    t_ns: int
    frame: int
    age_ms: float
    action: str
    magnitude: float
    steer: float
    rule_action: str
    rule_steer: float
    decision: bool
    mode: str
    steer_label: bool     # steering came off the pad (steer_by == "pad")
    throttle_label: bool  # direction and magnitude came off the pad
    throttle: float = 0.0  # the commanded values, for the idle rules
    front: float = 0.0
    accel: float = float("nan")
    gyro: float = float("nan")
    policy_drove: bool = False  # auto under --policy: a "driver" group is the model's


def _is_split(cmd):
    """L3 split steering: the rear is its own stick, which no head learns yet."""
    return cmd.steering_mode == pa.SPLIT


def _rear_off_pair(cmd):
    """A unified command whose rear is not steering_pair(front)'s.

    That is every recording from before the rear servo was driven (rear held
    at 0). The front is still a label, so these are counted, not dropped.
    """
    _, rear = dv.steering_pair(cmd.front)
    return not _is_split(cmd) and abs(cmd.rear - rear) > 0.02


def _connected_at(connected, t_ns):
    """The pad's state at t, from the newest /controller sample before it.

    No sample at all reads as connected: a legacy file with the channel
    missing should not lose every tick to a field it never had.
    """
    if not connected:
        return True
    i = bisect.bisect_right([c[0] for c in connected], t_ns) - 1
    return connected[i][1] if i >= 0 else connected[0][1]


def sample_ticks(commands, tick_s=TICK_S):
    """A 20 Hz command stream down to the commands at a 10 Hz decision rate."""
    out, last = [], None
    for cmd in commands:
        if last is None or (cmd.t_ns - last) / 1e9 >= tick_s - TICK_SLACK_S:
            out.append(cmd)
            last = cmd.t_ns
    return out


def _no_input(t, rules):
    return abs(t.throttle) <= pa.THROTTLE_DEADBAND and abs(t.front) <= rules.steer_rest


def _runs(times_ns, gap_s):
    """Index ranges [(i, j)] of consecutive times no more than gap_s apart."""
    runs, start = [], 0
    for i in range(1, len(times_ns) + 1):
        if i == len(times_ns) or (times_ns[i] - times_ns[i - 1]) / 1e9 > gap_s:
            if len(times_ns):
                runs.append((start, i))
            start = i
    return runs


def idle_reasons(ticks, record_times_ns, rules=IdleRules()):
    """Per tick an IDLE_REASONS entry, or "" to keep it.

    `ticks` are the session's aligned ticks in time order; only those in
    rules.modes are judged. `record_times_ns` is every row of those modes in
    the session, aligned or not, so a run's end is where the car stopped
    writing rows, not where the last usable frame happened to be.
    """
    out = [""] * len(ticks)
    idx = [i for i, t in enumerate(ticks) if t.mode in rules.modes]
    if not idx:
        return out
    tk = [ticks[i] for i in idx]
    t_s = np.array([t.t_ns for t in tk], dtype=np.int64) / 1e9
    reason = [""] * len(tk)

    def mark(k, why):
        if not reason[k]:
            reason[k] = why

    # Lead and trail: before the first and after the last throttle.
    active = [k for k, t in enumerate(tk) if abs(t.throttle) > pa.THROTTLE_DEADBAND]
    if not active:
        reason = ["idle lead"] * len(tk)
    else:
        for k in range(active[0]):
            mark(k, "idle lead")
        for k in range(active[-1] + 1, len(tk)):
            mark(k, "idle trail")

    # The reach for the button that ended each run of rows.
    rec = sorted(record_times_ns)
    for _, j in _runs(rec, rules.run_gap_s):
        end = rec[j - 1] / 1e9
        for k in np.flatnonzero((t_s > end - rules.exit_reach_s) & (t_s <= end)):
            mark(int(k), "exit reach")

    quiet = [_no_input(t, rules) for t in tk]

    # Handled: no input, settled, and the IMU says otherwise.
    last_input = None
    flagged = []
    for k, t in enumerate(tk):
        if not quiet[k]:
            last_input = t_s[k]
            continue
        settled = last_input is None or t_s[k] - last_input >= rules.handled_settle_s
        if settled and (t.gyro > rules.handled_gyro or t.accel > rules.handled_accel):
            flagged.append(k)
    for k in flagged:
        # Spread over the no-input ticks around it, never across an input.
        for step in (-1, 1):
            n = k
            while 0 <= n < len(tk) and quiet[n] and abs(t_s[n] - t_s[k]) <= rules.handled_margin_s:
                mark(n, "handled")
                n += step

    # Parked: a no-input span longer than parked_s, less its two edges. A
    # span is unbroken no-input ticks, no two further apart than PREV_GAP_S.
    spans, cur = [], []
    for k in range(len(tk)):
        if quiet[k] and cur and t_s[k] - t_s[cur[-1]] <= PREV_GAP_S and k == cur[-1] + 1:
            cur.append(k)
            continue
        if cur:
            spans.append(cur)
        cur = [k] if quiet[k] else []
    if cur:
        spans.append(cur)
    for span in spans:
        first, last = t_s[span[0]], t_s[span[-1]]
        if last - first + TICK_S <= rules.parked_s:
            continue
        for k in span:
            if t_s[k] - first >= rules.parked_edge_s and last - t_s[k] >= rules.parked_edge_s:
                mark(k, "parked")

    for k, i in enumerate(idx):
        out[i] = reason[k]
    return out


def align(session, max_age_ms=None, rules=IdleRules(), timeline=None):
    """A Session -> (ticks, drop counts, alignment errors in ms of kept ticks).

    rules None turns the not-driving rules off. timeline, if a list, gets
    (t_ns, reason) for every tick that made it to those rules, "" for kept.
    """
    if max_age_ms is None:
        max_age_ms = MAX_AGE_MS[session.adapter]
    drops = Counter()
    errors = []
    ticks = []
    frame_times = [f.t_ns for f in session.frames]
    seq_index = {f.seq: i for i, f in enumerate(session.frames)} if session.by_seq else {}

    commands = session.commands if session.by_seq else sample_ticks(session.commands)
    last_reverse = None
    prev_was_reverse = False
    prev_t = None
    for cmd in commands:
        if cmd.mode not in session.label_modes:
            drops["mode"] += 1
            prev_was_reverse = False
            continue
        if session.by_seq:
            fi = seq_index.get(cmd.seq)
            if fi is None:
                drops["no frame"] += 1
                continue
            age = abs(cmd.t_ns - frame_times[fi]) / 1e6
        else:
            fi = bisect.bisect_right(frame_times, cmd.t_ns) - 1
            if fi < 0:
                drops["no frame"] += 1
                continue
            age = (cmd.t_ns - frame_times[fi]) / 1e6
        if age > max_age_ms:
            drops["stale"] += 1
            continue
        if not _connected_at(session.connected, cmd.t_ns):
            drops["pad"] += 1
            continue
        if not all(math.isfinite(v) for v in (cmd.throttle, cmd.front, cmd.rear)):
            drops["bad command"] += 1
            continue
        if _is_split(cmd):
            drops["split"] += 1
            continue

        action = pa.action_of(cmd.throttle)
        consecutive = prev_t is not None and (cmd.t_ns - prev_t) / 1e9 <= PREV_GAP_S
        if action == pl.REVERSE:
            decision = not (consecutive and prev_was_reverse)
            last_reverse = cmd.t_ns
        elif (action == pl.STOP and last_reverse is not None
              and (cmd.t_ns - last_reverse) / 1e9 <= SETTLE_S):
            decision = False
        else:
            decision = True
        prev_was_reverse = action == pl.REVERSE
        prev_t = cmd.t_ns

        frame = session.frames[fi]
        rule_action = cmd.rule_action or frame.rule_action
        rule_steer = cmd.rule_steer if math.isfinite(cmd.rule_steer) else frame.rule_steer
        ticks.append(Tick(
            cmd.t_ns, fi, age, action,
            pa.magnitude_of(cmd.throttle, cmd.forward_cap, cmd.reverse_cap),
            float(np.clip(cmd.front / pa.STEER_SCALE, -1.0, 1.0)),
            rule_action, float(rule_steer), decision, cmd.mode,
            cmd.steer_by == pa.BY_PAD, cmd.throttle_by == pa.BY_PAD,
            float(cmd.throttle), float(cmd.front), float(cmd.accel), float(cmd.gyro),
            cmd.autopilot == pa.AUTOPILOT_POLICY))
        errors.append(age)

    if rules is not None:
        record_times = [c.t_ns for c in commands if c.mode in rules.modes]
        why = idle_reasons(ticks, record_times, rules)
        if timeline is not None:
            timeline.extend((t.t_ns, w) for t, w in zip(ticks, why) if t.mode in rules.modes)
        kept = [(t, e) for t, e, w in zip(ticks, errors, why) if not w]
        for w in why:
            if w:
                drops[w] += 1
        ticks = [t for t, _ in kept]
        errors = [e for _, e in kept]
    return ticks, drops, errors


TIMELINE_CHARS = {"": "#", "idle lead": "i", "idle trail": "i", "exit reach": "x",
                  "handled": "h", "parked": "p"}


def format_timeline(timeline, width=60):
    """One character a second: # kept, p parked, h handled, x exit reach,
    i idle, . no tick. The most common verdict in each second wins."""
    if not timeline:
        return []
    t0 = min(t for t, _ in timeline)
    seconds = {}
    for t, w in timeline:
        seconds.setdefault(int((t - t0) // 1_000_000_000), Counter())[w] += 1
    n = max(seconds) + 1
    chars = "".join(TIMELINE_CHARS[seconds[s].most_common(1)[0][0]] if s in seconds else "."
                    for s in range(n))
    return [f"    {i:5d}s {chars[i:i + width]}" for i in range(0, n, width)]


def mark_window_ns(t_ns, mark):
    """(lo, hi) log-time ns of one mark row's window, or None if it has none.

    The row's window is on the car's monotonic clock and its log_time is on
    the epoch clock; mono_s is the press on the first, t_ns on the second.
    """
    try:
        mono = float(mark["mono_s"])
        lo = float(mark["window_start_s"]) - mono
        hi = float(mark["window_end_s"]) - mono
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (lo, hi)) or hi < lo:
        return None
    return t_ns + int(lo * 1e9), t_ns + int(hi * 1e9)


def mark_flags(ticks, marks):
    """Per tick MARK_NONE / MARK_GOOD / MARK_BAD, off [(t_ns, mark row)]."""
    flags = np.full(len(ticks), MARK_NONE, dtype=np.int8)
    if not len(ticks):
        return flags
    times = np.array([t.t_ns for t in ticks], dtype=np.int64)
    pad = np.array([t.steer_label or t.throttle_label for t in ticks], dtype=bool)
    for kind, flag in (("good", MARK_GOOD), ("bad", MARK_BAD)):  # bad last: it wins
        for t_ns, mark in marks:
            if str(mark.get("kind", "")).lower() != kind:
                continue
            window = mark_window_ns(int(t_ns), mark)
            if window is None:
                continue
            inside = (times >= window[0]) & (times <= window[1]) & pad
            flags[inside] = flag
    return flags


def prev_inputs(ticks):
    """(N, PREV_SIZE): each tick's previous tick as the model would see it."""
    out = np.zeros((len(ticks), pl.PREV_SIZE), dtype=np.float32)
    for i in range(1, len(ticks)):
        a, b = ticks[i - 1], ticks[i]
        if (b.t_ns - a.t_ns) / 1e9 <= PREV_GAP_S:
            out[i] = pl.encode_prev(a.action, a.magnitude, a.steer)[0]
        else:
            out[i] = pl.encode_prev(pl.STOP, 0.0, 0.0)[0]
    if len(ticks):
        out[0] = pl.encode_prev(pl.STOP, 0.0, 0.0)[0]
    return out


def to_arrays(sessions_ticks):
    """[(Session, ticks)] -> the dict of arrays the .npz holds."""
    # Only the frames some tick points at. A legacy file is mostly manual
    # driving with the camera up, and those frames have no label.
    frames, rows = [], []
    for si, (session, ticks) in enumerate(sessions_ticks):
        index = {}
        prev = prev_inputs(ticks)
        flags = mark_flags(ticks, session.marks)
        for t, p, f in zip(ticks, prev, flags):
            if t.frame not in index:
                index[t.frame] = len(frames)
                frames.append(session.frames[t.frame].small)
            rows.append((si, index[t.frame], t, p, f))
    n = len(rows)
    a = {
        "frames": (np.stack(frames) if frames
                   else np.zeros((0, pl.INPUT_HEIGHT, pl.INPUT_WIDTH), np.uint16)),
        "session": np.array([r[0] for r in rows], dtype=np.int16),
        "frame": np.array([r[1] for r in rows], dtype=np.int64),
        "t_ns": np.array([r[2].t_ns for r in rows], dtype=np.int64),
        "age_ms": np.array([r[2].age_ms for r in rows], dtype=np.float32),
        "action": np.array([pl.ACTIONS.index(r[2].action) for r in rows], dtype=np.int8),
        "magnitude": np.array([r[2].magnitude for r in rows], dtype=np.float32),
        "steer": np.array([r[2].steer for r in rows], dtype=np.float32),
        "rule_action": np.array([pl.ACTIONS.index(r[2].rule_action)
                                 if r[2].rule_action in pl.ACTIONS else -1
                                 for r in rows], dtype=np.int8),
        "rule_steer": np.array([r[2].rule_steer for r in rows], dtype=np.float32),
        "decision": np.array([r[2].decision for r in rows], dtype=bool),
        "steer_label": np.array([r[2].steer_label for r in rows], dtype=bool),
        "throttle_label": np.array([r[2].throttle_label for r in rows], dtype=bool),
        "policy_drove": np.array([r[2].policy_drove for r in rows], dtype=bool),
        "mark": np.array([r[4] for r in rows], dtype=np.int8),
        "prev": (np.stack([r[3] for r in rows]) if n
                 else np.zeros((0, pl.PREV_SIZE), np.float32)),
    }
    return a


# ---- reading MCAPs (laptop only) ------------------------------------------

def read_session(path, adapter_name=None):
    """One MCAP file -> a Session. Survives a file cut off mid-chunk.

    The stream reader and not the indexed one: a recording the car was
    killed in the middle of has no summary and ends in a half-written chunk,
    and the indexed reader gives up on the whole file. This reads up to the
    cut and stops.
    """
    from mcap.records import Channel, Message, Schema
    from mcap.stream_reader import StreamReader
    from mcap_protobuf.decoder import DecoderFactory

    factory = DecoderFactory()
    schemas, channels, decoders = {}, {}, {}
    name = os.path.splitext(os.path.basename(path))[0]
    adapter = None
    pending = []
    truncated = False

    def decode(channel, data):
        if channel.message_encoding == "json":
            return json.loads(data)
        dec = decoders.get(channel.id)
        if dec is None:
            dec = factory.decoder_for(channel.message_encoding,
                                      schemas.get(channel.schema_id))
            decoders[channel.id] = dec
        msg = dec(data)
        return {"data": msg.data, "format": getattr(msg, "format", ""),
                "frame_id": getattr(msg, "frame_id", ""),
                "width": getattr(msg, "width", 0), "height": getattr(msg, "height", 0),
                "encoding": getattr(msg, "encoding", "")}

    with open(path, "rb") as f:
        try:
            for rec in StreamReader(f, skip_magic=False, validate_crcs=False).records:
                if isinstance(rec, Schema):
                    schemas[rec.id] = rec
                elif isinstance(rec, Channel):
                    channels[rec.id] = rec
                elif isinstance(rec, Message):
                    pending.append(rec) if adapter is None else None
                    if adapter is None:
                        topics = {c.topic for c in channels.values()}
                        cls = (pa.ADAPTERS[adapter_name] if adapter_name
                               else pa.pick_adapter(topics))
                        # Wait for enough channels to know the recording's kind,
                        # but not forever: one is never announced in some files.
                        if cls is None and len(pending) < 2000:
                            continue
                        if cls is None:
                            cls = pa.LegacyAdapter
                        adapter = cls(name)
                        for m in pending:
                            ch = channels[m.channel_id]
                            if ch.topic in adapter.topics:
                                adapter.feed(ch.topic, m.log_time, decode(ch, m.data))
                        pending = []
                        continue
                    ch = channels[rec.channel_id]
                    if ch.topic in adapter.topics:
                        adapter.feed(ch.topic, rec.log_time, decode(ch, rec.data))
        except Exception as e:
            truncated = f"{type(e).__name__}"
    if adapter is None:
        adapter = (pa.ADAPTERS[adapter_name] if adapter_name else pa.LegacyAdapter)(name)
        for m in pending:
            ch = channels[m.channel_id]
            if ch.topic in adapter.topics:
                adapter.feed(ch.topic, m.log_time, decode(ch, m.data))
    return adapter.finish(), truncated


def _pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


# Where the recordings are pulled to on the laptop, and the files a bare
# `policy_dataset.py OUT.npz` reads.
DEFAULT_MCAPS = "~/crawler-data/mcap/crawler-record-*.mcap"


def default_mcaps(pattern=DEFAULT_MCAPS):
    """Every file the default glob matches, oldest name first."""
    return sorted(glob.glob(os.path.expanduser(pattern)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("mcaps", nargs="*",
                    help=f"recordings (default: {DEFAULT_MCAPS})")
    ap.add_argument("--adapter", choices=sorted(pa.ADAPTERS), default=None,
                    help="force one; default picks off the topics in each file")
    ap.add_argument("--modes", default=None,
                    help="comma list of recorded modes to label from "
                         "(default: auto for legacy files, record for new ones)")
    ap.add_argument("--max-age-ms", type=float, default=None)
    d = IdleRules()
    idle = ap.add_argument_group("not-driving rules (IdleRules; record and manual ticks only)")
    idle.add_argument("--no-idle-rules", action="store_true", help="keep every tick")
    idle.add_argument("--steer-rest", type=float, default=d.steer_rest,
                      help=f"commanded |front| that counts as no steering input ({d.steer_rest:.3f})")
    idle.add_argument("--parked-s", type=float, default=d.parked_s,
                      help=f"a no-input span longer than this is parked ({d.parked_s})")
    idle.add_argument("--parked-edge-s", type=float, default=d.parked_edge_s,
                      help=f"kept at each end of a parked span ({d.parked_edge_s})")
    idle.add_argument("--exit-reach-s", type=float, default=d.exit_reach_s,
                      help=f"dropped before each end of a record run ({d.exit_reach_s})")
    idle.add_argument("--handled-settle-s", type=float, default=d.handled_settle_s,
                      help=f"no input this long before the IMU is judged ({d.handled_settle_s})")
    idle.add_argument("--handled-gyro", type=float, default=d.handled_gyro,
                      help=f"|gyro| rad/s over this with no input is handling ({d.handled_gyro})")
    idle.add_argument("--handled-accel", type=float, default=d.handled_accel,
                      help=f"|linear accel| m/s^2 over this with no input is handling ({d.handled_accel})")
    idle.add_argument("--handled-margin-s", type=float, default=d.handled_margin_s,
                      help=f"spread either side of a handled tick, over no-input ticks ({d.handled_margin_s})")
    args = ap.parse_args(argv)
    if not args.mcaps:
        args.mcaps = default_mcaps()
        if not args.mcaps:
            ap.error(f"no recordings given and none match {DEFAULT_MCAPS}")
    rules = None if args.no_idle_rules else IdleRules(
        args.steer_rest, args.parked_s, args.parked_edge_s, args.exit_reach_s,
        args.handled_settle_s, args.handled_gyro, args.handled_accel, args.handled_margin_s)

    kept, contracts, report = [], set(), []
    all_errors = []
    for path in args.mcaps:
        session, truncated = read_session(path, args.adapter)
        if args.modes:
            session = session._replace(label_modes=tuple(args.modes.split(",")))
        timeline = []
        ticks, drops, errors = align(session, args.max_age_ms, rules, timeline)
        splits = sum(1 for c in session.commands if _is_split(c))
        front_only = sum(1 for c in session.commands if _rear_off_pair(c))
        line = (f"{session.name}: {session.adapter}, {len(session.frames)} frames, "
                f"{len(session.commands)} commands -> {len(ticks)} ticks "
                f"({sum(t.decision for t in ticks)} decisions)"
                f"{' [cut off: ' + truncated + ']' if truncated else ''}\n"
                f"    dropped {dict(drops)}; split-steering commands in file: {splits}, "
                f"rear not steering_pair(front): {front_only}\n"
                f"    pad-labelled ticks: steering {sum(t.steer_label for t in ticks)}, "
                f"throttle {sum(t.throttle_label for t in ticks)}")
        flags = mark_flags(ticks, session.marks)
        kinds = Counter(str(m.get("kind", "?")).lower() for _, m in session.marks)
        line += (f"\n    marks: {kinds.get('good', 0)} good, {kinds.get('bad', 0)} bad; "
                 f"pad ticks dropped by BAD {int((flags == MARK_BAD).sum())}, "
                 f"upweighted by GOOD {int((flags == MARK_GOOD).sum())}")
        if timeline:
            judged = len(timeline)
            line += (f"\n    not driving: {judged} {'/'.join(rules.modes)} ticks judged, "
                     f"{sum(1 for _, w in timeline if not w)} kept, "
                     + ", ".join(f"{r} {drops.get(r, 0)}" for r in IDLE_REASONS)
                     + "\n    timeline (# kept  p parked  h handled  x exit reach  i idle  . none):\n"
                     + "\n".join(format_timeline(timeline)))
        if errors:
            line += (f"\n    alignment ms p50 {_pct(errors, 50):.0f} p90 {_pct(errors, 90):.0f} "
                     f"p99 {_pct(errors, 99):.0f} max {max(errors):.0f}")
        print(line)
        report.append(line)
        if ticks:
            kept.append((session, ticks))
            contracts.add(session.contract)
            all_errors.extend(errors)

    if not kept:
        print("no ticks in any file; nothing written")
        return 1
    if len(contracts) > 1:
        print(f"mixed input contracts {sorted(contracts)}; build them separately")
        return 1

    arrays = to_arrays(kept)
    hist_edges = [0, 25, 50, 100, 150, 200, 250, 1e9]
    hist = np.histogram(all_errors, bins=hist_edges)[0].tolist()
    meta = {
        "input_contract": contracts.pop(),
        "actions": list(pl.ACTIONS),
        "sessions": [s.name for s, _ in kept],
        "steer_scale": pa.STEER_SCALE,
        "marks": {s.name: [m for m in s.marks] for s, _ in kept},
        "alignment_ms_histogram": {"edges": hist_edges[:-1], "counts": hist},
        "report": report,
    }
    np.savez_compressed(args.out, meta=json.dumps(meta), **arrays)
    counts = Counter(pl.ACTIONS[a] for a in arrays["action"][arrays["decision"]])
    print(f"wrote {args.out}: {len(arrays['t_ns'])} ticks, "
          f"{int(arrays['decision'].sum())} decisions {dict(counts)}, "
          f"{len(arrays['frames'])} frames")
    print(f"alignment ms, all kept ticks: p50 {_pct(all_errors, 50):.0f} "
          f"p90 {_pct(all_errors, 90):.0f} p99 {_pct(all_errors, 99):.0f}; "
          f"histogram {dict(zip(['<25', '25-50', '50-100', '100-150', '150-200', '200-250', '>250'], hist))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
