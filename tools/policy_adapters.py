"""Recorded messages -> sessions the dataset tool can align. Field names live HERE.

A recording is a pile of timestamped messages on named topics, and two kinds
of recording exist:

  LegacyAdapter     what every run so far has written: /camera/depth as a
                    colourmapped 320x200 JPEG at 6-8 Hz, and the commanded
                    throttle and steering on /vehicle at 20 Hz. No tick id,
                    so a frame and a command are matched by timestamp.
  FootprintAdapter  what record mode writes (docs/TRAINING_DATA.md):
                    /policy/depth_small, policy_input.depth_small() of the
                    frame as a 16-bit PNG, and /footprint carrying the
                    command, the probes and the gradient verdict -- one of
                    each per 10 Hz decision, on one log time and one seq.

Every topic and field name either one reads is in the two blocks of
constants below and nowhere else. When the record-mode schema changes, those
blocks are the edit.

Pure: decoded payloads in, a Session out. Nothing here opens an MCAP -- that
is policy_dataset.py, so these can be tested with dicts.
"""

import math
from typing import NamedTuple

import numpy as np

from crawler.autopilot import driver as dv
from crawler.autopilot import policy as pl
from crawler.autopilot import policy_imu as pimu

# ---- legacy recordings -----------------------------------------------------

LEGACY_TOPICS = {
    "depth": "/camera/depth",          # foxglove.CompressedImage, JET JPEG
    "vehicle": "/vehicle",             # JSON, 20 Hz
    "controller": "/controller",       # JSON, 20 Hz
    "marks": "/marks",                 # JSON, one per press
}
LEGACY_VEHICLE = {
    "mode": "mode",
    "throttle": "throttle",
    "front": "steering_front",
    "rear": "steering_rear",
    "steering_mode": "steering_mode",  # absent before L3 split existed
}
LEGACY_CONNECTED = "connected"         # on /controller

# ---- record-mode recordings: docs/TRAINING_DATA.md is the contract ------------

FOOTPRINT_TOPICS = {
    "depth": "/policy/depth_small",    # foxglove.CompressedImage, 16-bit PNG
    "row": "/footprint",               # JSON, crawler.Footprint
    "controller": "/controller",
    "marks": "/marks",
}
FOOTPRINT_ROW = {
    "seq": "seq",
    "mode": "mode",
    "throttle": "throttle",
    "front": "steering_front",
    "rear": "steering_rear",
    "steering_mode": "steering_mode",
    "rule_action": "verdict_action",
    "rule_steer": "verdict_steer",
    # The speed caps, so a magnitude is measured against the caps that were
    # actually in force rather than assumed. Record rows always carry them;
    # auto rows carry auto's caps on every row now, and -1 only in older
    # recordings, on rows no trigger drove.
    "forward_cap": "forward_limit",
    "reverse_cap": "reverse_limit",
    # TAKEOVER: who had each group of controls on this tick, "driver" (the
    # autopilot) or "pad" (a hand on it). Only "pad" is a label.
    "steer_by": "steer_by",
    "throttle_by": "throttle_by",
    # "driver" is auto's own scan; "tap" is the footprint tap's, which is
    # every record row and every auto row while a recover owns the wheels.
    "source": "source",
    # Who the auto driver was: "rules" or "policy" (python -m crawler --policy); ""
    # in record and in files from before the field existed. A "driver"
    # group under "policy" is the model's own output, never a label.
    "autopilot": "autopilot",
    # The newest OAK-D sample at the tick, for policy_dataset's "handled"
    # rule. LINEAR_ACCELERATION: gravity is already taken out, so a parked
    # car reads ~0, not 1 g. imu_age_ms -1 (and zeros) means no sample yet.
    "accel": ("imu_accel_x", "imu_accel_y", "imu_accel_z"),
    "gyro": ("imu_gyro_x", "imu_gyro_y", "imu_gyro_z"),
    "imu_age": "imu_age_ms",
    # Every IMU column, for the policy's IMU input. Named in policy_imu
    # because the car builds the same row live; the up vector is absent from
    # recordings before GAME_ROTATION_VECTOR and reads as no tilt.
    "imu_row": pimu.ROW_FIELDS,
}
# The picture's frame_id is "footprint/<seq>", the row's tick id. Both also
# carry the same log_time; the seq is the join, the log time the check.
DEPTH_SMALL_FRAME_ID_PREFIX = "footprint/"
DEPTH_SMALL_FORMAT = "png"

# Which recorded modes become ticks at all, per adapter. Whether a tick's
# steering or throttle is a LABEL is a separate question, answered per group
# by control_sources(): only a "pad" group is one, and the trainer masks a
# "driver" group out of that head's loss unless it is told to clone the
# autopilot. Legacy keeps auto only (manual in those files is mostly a
# parked car, and every legacy auto tick is driver/driver -- a smoke test).
# Record-mode files keep record and auto, so a TAKEOVER hand in auto counts.
LEGACY_LABEL_MODES = ("auto",)
FOOTPRINT_LABEL_MODES = ("record", "auto")

# Steering is learned in manual's units: a fraction of full lock, 0.7 of
# servo travel (main.STEERING_LIMIT_PERC), which is what the pad commands
# and so what a label is. policy.py multiplies by the same constant.
STEER_SCALE = pl.STEER_SCALE

# read_the_gradient()'s steer is a fraction of the autopilot's max_turn, a
# smaller lock, so a verdict is rescaled into STEER_SCALE's units before it
# is ever compared with a label or a model.
RULE_MAX_TURN = 0.6


AUTOPILOT_POLICY = pl.AUTOPILOT_POLICY


def rule_steer_units(fraction):
    """A verdict's steer (fraction of max_turn) in STEER_SCALE units."""
    return float(fraction) * RULE_MAX_TURN / STEER_SCALE

# The unit of a throttle label: a FRACTION OF THE CAP IN FORCE ON THAT ROW,
# never an absolute throttle. Record mode scales the triggers to
# Driver.forward_speed and Driver.reverse_speed, so throttle / cap is exactly
# how far I pulled the trigger, and policy.py turns a magnitude back into a
# throttle by multiplying it by the live caps. One unit on both sides, and a
# model asked for "full" can never command more than today's cap whatever cap
# it was trained under. An absolute label would do the opposite: a reverse
# learned at -0.6 would ask for about twice today's reverse cap and lean on
# a clamp to stay legal.
#
# What the unit does not fix is that a pull is not the same decision at a
# different cap -- at forward 0.2 I held RT flat because the car barely
# moved. So a session recorded at other caps is left out of a dataset by
# default (policy_dataset.py --include-other-caps), and session_caps() is
# how it is told.
#
# The caps every recording before the move to 0.4 was made at: forward 0.2,
# reverse -0.6. A legacy file carries no caps and neither does a /footprint
# row from before forward_limit existed; both are from then, so these are
# what they are measured against.
LEGACY_FORWARD_CAP = 0.2
LEGACY_REVERSE_CAP = -0.6

# What auto, record and the policy runtime drive at now. Read off driver.py,
# never copied.
RUNTIME_CAPS = (float(dv.FORWARD_SPEED), float(dv.REVERSE_SPEED))

# Two caps closer than this are the same cap. footprint.py writes the float
# as it was, so anything bigger is a real difference.
CAP_TOLERANCE = 1e-4

# Throttle to action, and it must not eat a creep. Joystick.read_throttle()
# returns exactly 0.0 inside TRIGGER_DEADZONE = 0.05 and the trigger value
# itself above it, so the smallest throttle record mode can command is 0.05
# times the smallest cap: 0.05 x 0.2 = 0.010 at the old forward cap, 0.015 at
# today's reverse. Half the smallest of those, across old and new caps:
# anything the triggers can produce is motion, and only an untouched pad is
# a stop. 0.005, as it always was.
THROTTLE_DEADBAND = 0.5 * 0.05 * min(LEGACY_FORWARD_CAP, -LEGACY_REVERSE_CAP,
                                     RUNTIME_CAPS[0], -RUNTIME_CAPS[1])

SPLIT = "split"

# Who commanded a group of controls on one tick.
BY_PAD = "pad"
BY_DRIVER = "driver"
# The modes in which a hand is on everything. A row from one of these with no
# steer_by / throttle_by is pad/pad; from any other mode it is driver/driver.
PAD_MODES = ("record", "manual")


def control_sources(mode, steer_by=None, throttle_by=None):
    """(steer_by, throttle_by) for one row, with the defaults filled in.

    A field that is present wins. A field that is missing is "pad" in
    record (and manual), and "driver" everywhere else -- an auto row from
    before TAKEOVER existed is the autopilot's own output on both groups,
    which is a label only for cloning the autopilot.
    """
    default = BY_PAD if mode in PAD_MODES else BY_DRIVER
    steer = steer_by if steer_by in (BY_PAD, BY_DRIVER) else default
    throttle = throttle_by if throttle_by in (BY_PAD, BY_DRIVER) else default
    return steer, throttle


class Command(NamedTuple):
    t_ns: int
    mode: str
    throttle: float
    front: float
    rear: float
    steering_mode: str
    rule_action: str      # "" when the adapter cannot say
    rule_steer: float     # nan when the adapter cannot say
    seq: int              # -1 when there is none
    forward_cap: float = LEGACY_FORWARD_CAP
    reverse_cap: float = LEGACY_REVERSE_CAP
    steer_by: str = BY_DRIVER
    throttle_by: str = BY_DRIVER
    accel: float = float("nan")   # |linear acceleration|, m/s^2; nan with no sample
    gyro: float = float("nan")    # |gyro|, rad/s; nan with no sample
    source: str = ""              # "driver" / "tap" on a /footprint row
    autopilot: str = ""           # "rules" / "policy" in auto, "" otherwise
    imu: tuple = ()               # FOOTPRINT_ROW["imu_row"] values, nan where absent


def imu_magnitudes(payload):
    """(|accel|, |gyro|) off a /footprint row, nan for both with no sample."""
    r = FOOTPRINT_ROW
    if not _f(payload.get(r["imu_age"]), -1.0) >= 0:
        return float("nan"), float("nan")
    out = []
    for key in ("accel", "gyro"):
        parts = [_f(payload.get(k)) for k in r[key]]
        out.append(math.sqrt(sum(v * v for v in parts)))  # nan if any part is
    return tuple(out)


class Frame(NamedTuple):
    t_ns: int
    small: np.ndarray     # (INPUT_HEIGHT, INPUT_WIDTH) uint16 mm, 0 invalid
    rule_action: str
    rule_steer: float
    seq: int


class Session(NamedTuple):
    name: str
    adapter: str
    contract: str
    commands: list
    frames: list
    connected: list       # [(t_ns, bool)]
    marks: list           # [(t_ns, dict)]
    label_modes: tuple
    by_seq: bool          # frames and commands share a seq


def _f(value, fallback=float("nan")):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if math.isfinite(value) else fallback


def action_of(throttle):
    if throttle > THROTTLE_DEADBAND:
        return pl.FORWARD
    if throttle < -THROTTLE_DEADBAND:
        return pl.REVERSE
    return pl.STOP


def magnitude_of(throttle, forward_cap=LEGACY_FORWARD_CAP,
                 reverse_cap=LEGACY_REVERSE_CAP,
                 fallback=(LEGACY_FORWARD_CAP, LEGACY_REVERSE_CAP)):
    """A signed throttle as a fraction of its direction's cap, in [0, 1].

    A forward cap on the wrong side of zero is the row's "no limits"
    placeholder (footprint.py writes -1 for both on an auto row no trigger
    drove), and the PAIR falls back to `fallback`: the caps the rest of the
    session recorded, when the caller knows them, and the legacy caps when it
    does not. The pair, not each cap on its own: -1 is on the right side of
    zero for a reverse cap, and read as one it turned a placeholder row's
    reverse into its absolute throttle rather than a fraction of any cap.
    """
    if not _valid_caps(forward_cap, reverse_cap):
        forward_cap, reverse_cap = fallback
    action = action_of(throttle)
    if action == pl.FORWARD:
        return float(min(1.0, throttle / forward_cap))
    if action == pl.REVERSE:
        return float(min(1.0, throttle / reverse_cap))
    return 0.0


def _valid_caps(forward_cap, reverse_cap):
    return (math.isfinite(forward_cap) and math.isfinite(reverse_cap)
            and forward_cap > 0 and reverse_cap < 0)


def session_caps(session):
    """Every (forward, reverse) cap pair a session's ticks recorded, sorted.

    Only commands in the session's label modes count, and only real caps --
    the -1 placeholder on an auto row is "nobody said", not a cap. A legacy
    session is its commands' default, the legacy caps. Empty when nothing in
    the file says: a record-mode file of nothing but untouched auto.
    """
    found = []
    for cmd in session.commands:
        if cmd.mode not in session.label_modes:
            continue
        if not _valid_caps(cmd.forward_cap, cmd.reverse_cap):
            continue
        pair = (float(cmd.forward_cap), float(cmd.reverse_cap))
        if not any(caps_match(pair, seen) for seen in found):
            found.append(pair)
    return sorted(found)


def caps_match(a, b=RUNTIME_CAPS, tol=CAP_TOLERANCE):
    """Are two (forward, reverse) cap pairs the same caps?"""
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def fallback_caps(session):
    """What a placeholder row in `session` is measured against.

    The session's own caps when it recorded exactly one pair, which is every
    session there can be today (the caps are constants for a process). Else
    the legacy caps, which is what an auto-only file from before this change
    was driven at -- and NOT necessarily what one from after was. That case
    is the one session_caps() returns empty for, and policy_dataset says so.
    """
    caps = session_caps(session)
    if len(caps) == 1:
        return caps[0]
    return (LEGACY_FORWARD_CAP, LEGACY_REVERSE_CAP)


def rule_action_of(verdict_action):
    """A SteerVerdict action in the policy's three-way vocabulary."""
    if verdict_action in (dv.ACT_FORWARD, dv.ACT_EVADE):
        return pl.FORWARD
    if verdict_action == dv.ACT_REVERSE:
        return pl.REVERSE
    return ""


# ---- JET JPEG back to millimetres -----------------------------------------

_JET_LUT = None
_JET_BITS = 6
# A decoded pixel further than this (8-bit RGB, euclidean) from every colour
# on the JET curve is a JPEG ringing artefact at an edge, and is invalid.
JET_MAX_DISTANCE = 48.0


def _jet_lut():
    """A (2**bits)^3 table from quantised BGR to (level, distance)."""
    global _JET_LUT
    if _JET_LUT is None:
        import cv2

        jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None],
                                cv2.COLORMAP_JET)[:, 0, :].astype(np.float32)
        n = 1 << _JET_BITS
        step = 256 // n
        axis = np.arange(n, dtype=np.float32) * step + step / 2.0
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
        level = np.empty(len(grid), dtype=np.uint8)
        dist = np.empty(len(grid), dtype=np.float32)
        for start in range(0, len(grid), 8192):
            chunk = grid[start:start + 8192]
            d = ((chunk[:, None, :] - jet[None, :, :]) ** 2).sum(-1)
            level[start:start + 8192] = d.argmin(1)
            dist[start:start + 8192] = np.sqrt(d.min(1))
        _JET_LUT = (level.reshape(n, n, n), dist.reshape(n, n, n))
    return _JET_LUT


def jet_jpeg_to_mm(data):
    """A /camera/depth JPEG back to a (200, 320) uint16 mm frame, 0 invalid."""
    import cv2

    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    level, dist = _jet_lut()
    q = (bgr >> (8 - _JET_BITS)).astype(np.intp)
    lv = level[q[..., 0], q[..., 1], q[..., 2]]
    far = dist[q[..., 0], q[..., 1], q[..., 2]] > JET_MAX_DISTANCE
    mm = pl.legacy_levels_to_mm(lv)
    mm[far] = 0
    return mm


def gradient_rule(mm):
    """read_the_gradient() on one frame: (action, steer fraction)."""
    verdict = dv.read_the_gradient(dv.free_by_column(mm))
    return rule_action_of(verdict.action), rule_steer_units(verdict.steer)


# ---- adapters --------------------------------------------------------------

class LegacyAdapter:
    name = "legacy"
    contract = pl.CONTRACT_LEGACY_JET
    topics = tuple(LEGACY_TOPICS.values())

    def __init__(self, session_name, label_modes=LEGACY_LABEL_MODES):
        self.session_name = session_name
        self.label_modes = tuple(label_modes)
        self.commands, self.frames, self.connected, self.marks = [], [], [], []

    def feed(self, topic, t_ns, payload):
        """One decoded message. Images are dicts with 'data' (the JPEG bytes)."""
        if topic == LEGACY_TOPICS["vehicle"]:
            v = LEGACY_VEHICLE
            mode = str(payload.get(v["mode"], ""))
            steer_by, throttle_by = control_sources(mode)
            self.commands.append(Command(
                int(t_ns), mode,
                _f(payload.get(v["throttle"])), _f(payload.get(v["front"])),
                _f(payload.get(v["rear"])),
                str(payload.get(v["steering_mode"]) or ""), "", float("nan"), -1,
                steer_by=steer_by, throttle_by=throttle_by))
        elif topic == LEGACY_TOPICS["depth"]:
            self.feed_mm(t_ns, jet_jpeg_to_mm(payload["data"]))
        elif topic == LEGACY_TOPICS["controller"]:
            self.connected.append((int(t_ns), bool(payload.get(LEGACY_CONNECTED, False))))
        elif topic == LEGACY_TOPICS["marks"]:
            self.marks.append((int(t_ns), dict(payload)))

    def feed_mm(self, t_ns, mm):
        """A decoded legacy frame, already in mm at LEGACY_SHAPE."""
        if mm is None:
            return
        action, steer = gradient_rule(mm)
        self.frames.append(Frame(int(t_ns), pl.legacy_small_from_mm(mm),
                                 action, steer, -1))

    def finish(self):
        return Session(self.session_name, self.name, self.contract,
                       sorted(self.commands), sorted(self.frames, key=lambda f: f.t_ns),
                       sorted(self.connected), sorted(self.marks, key=lambda m: m[0]),
                       self.label_modes, False)


def decode_depth_small(payload):
    """A /policy/depth_small CompressedImage -> ((50, 80) uint16, seq), or None.

    payload is the decoded message as a dict: data, format and frame_id.
    """
    import cv2

    if str(payload.get("format", "")) != DEPTH_SMALL_FORMAT:
        return None
    img = cv2.imdecode(np.frombuffer(bytes(payload.get("data", b"")), np.uint8),
                       cv2.IMREAD_UNCHANGED)
    if img is None or img.dtype != np.uint16 or img.shape != (pl.INPUT_HEIGHT, pl.INPUT_WIDTH):
        return None
    frame_id = str(payload.get("frame_id", ""))
    seq = -1
    if frame_id.startswith(DEPTH_SMALL_FRAME_ID_PREFIX):
        try:
            seq = int(frame_id[len(DEPTH_SMALL_FRAME_ID_PREFIX):])
        except ValueError:
            seq = -1
    return img, seq


class FootprintAdapter:
    name = "footprint"
    contract = pl.CONTRACT_DEPTH_SMALL
    topics = tuple(FOOTPRINT_TOPICS.values())

    def __init__(self, session_name, label_modes=FOOTPRINT_LABEL_MODES):
        self.session_name = session_name
        self.label_modes = tuple(label_modes)
        self.commands, self.frames, self.connected, self.marks = [], [], [], []
        self.bad_frames = 0

    def feed(self, topic, t_ns, payload):
        if topic == FOOTPRINT_TOPICS["row"]:
            r = FOOTPRINT_ROW
            self.commands.append(Command(
                int(t_ns), str(payload.get(r["mode"], "")),
                _f(payload.get(r["throttle"])), _f(payload.get(r["front"])),
                _f(payload.get(r["rear"])),
                str(payload.get(r["steering_mode"]) or ""),
                rule_action_of(payload.get(r["rule_action"])),
                rule_steer_units(_f(payload.get(r["rule_steer"]))),
                int(payload.get(r["seq"], -1)),
                _f(payload.get(r["forward_cap"]), LEGACY_FORWARD_CAP),
                _f(payload.get(r["reverse_cap"]), LEGACY_REVERSE_CAP),
                *control_sources(str(payload.get(r["mode"], "")),
                                 payload.get(r["steer_by"]),
                                 payload.get(r["throttle_by"])),
                *imu_magnitudes(payload),
                str(payload.get(r["source"]) or ""),
                str(payload.get(r["autopilot"]) or ""),
                tuple(_f(payload.get(k)) for k in r["imu_row"])))
        elif topic == FOOTPRINT_TOPICS["depth"]:
            decoded = decode_depth_small(payload)
            if decoded is None:
                self.bad_frames += 1
                return
            small, seq = decoded
            self.frames.append(Frame(int(t_ns), small, "", float("nan"), seq))
        elif topic == FOOTPRINT_TOPICS["controller"]:
            self.connected.append((int(t_ns), bool(payload.get(LEGACY_CONNECTED, False))))
        elif topic == FOOTPRINT_TOPICS["marks"]:
            self.marks.append((int(t_ns), dict(payload)))

    def finish(self):
        by_seq = bool(self.frames) and all(f.seq >= 0 for f in self.frames)
        return Session(self.session_name, self.name, self.contract,
                       sorted(self.commands), sorted(self.frames, key=lambda f: f.t_ns),
                       sorted(self.connected), sorted(self.marks, key=lambda m: m[0]),
                       self.label_modes, by_seq)


ADAPTERS = {LegacyAdapter.name: LegacyAdapter, FootprintAdapter.name: FootprintAdapter}


def pick_adapter(topics, settled=True):
    """The adapter a recording's topic list calls for, or None.

    settled False means more channels may still be announced. A legacy match
    waits then: record mode writes /camera/depth and /vehicle too, and when
    those are announced before /policy/depth_small the whole recording would
    be read as legacy and every record tick dropped by mode.
    """
    topics = set(topics)
    if {FOOTPRINT_TOPICS["depth"], FOOTPRINT_TOPICS["row"]} <= topics:
        return FootprintAdapter
    if settled and {LEGACY_TOPICS["depth"], LEGACY_TOPICS["vehicle"]} <= topics:
        return LegacyAdapter
    return None
