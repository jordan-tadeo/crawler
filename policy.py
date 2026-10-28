"""A learned driving decision, behind the same safety the autopilot has.

What this is
------------
A small model, trained off the car by behaviour cloning, that looks at one
downsampled depth frame and says three things: a direction (forward,
reverse or stop), how hard as a fraction of that direction's speed cap, and
a front steering fraction -- plus a rear one, if a model is ever trained on
split steering -- and, if it has the head, which steering mode it wants. The frame reaches it either whole (IMAGE, a small 2-D CNN)
or as the column profile off the footprint's rows (PROFILE, column_profile.py),
and the sidecar says which. PolicyDriver runs it where Driver.tick() runs read_the_gradient(),
and it is a Driver: the recover phase machine, the stuck detector, the rear
bumper and the footprint scan are all inherited, not copied. The only thing
the model replaces is the verdict.

It is opt-in. main.py builds one only when --policy names a model file, and
with no flag the car runs exactly the autopilot it runs today.

What the model can never do
---------------------------
pigpio latches the last pulse width forever and there is no hardware
failsafe, so every way out of a tick below commands something, and every
failure commands neutral. In the order they are checked:

  a STUCK recover owns the wheels   the model is not consulted at all
  no new depth frame (a stale one
  is not handed out twice), or a
  frame that is not a 2-D array     neutral (throttle 0, steering 0,0)
  the whole fan unmeasurable        neutral -- a blind camera is not a
                                    picture to decide on
  the stuck detector fires          the ordinary recover; disarmed while
                                    my hand has the throttle, as Driver's is
  model output not finite, the
  wrong shape, throttle past [0, 1]
  or steering past +-1              neutral
  a forward with anything measured
  inside FORWARD_FLOOR_MM, or a
  probe gone blind on the way in    throttle 0, the model's steering still
                                    goes out (driver.ForwardFloorVeto, the
                                    same veto the takeover holds my
                                    trigger with, its own instance). A
                                    model reverse is never vetoed: the
                                    camera cannot see behind, so the floor
                                    has nothing to say about it
  anything raises                   neutral, then the exception goes on up
                                    to AutoPilot.step(), which parks auto

It is handed main.DriverCommand as its controller, exactly as Driver is, so
it never writes the hardware: AutoPilot.step() mixes in whatever groups my
hand has taken and writes once a pass. Takeover works the same on top of it.

The throttle is a magnitude, never a signed number. Forward and reverse are
both right answers in the same scene often enough that one regressed signed
throttle averages them into a stop, so the direction is a classification and
the magnitude a sigmoid in [0, 1] that only ever multiplies a bench cap --
forward_speed for forward, reverse_speed for reverse. That is the label's
unit too: policy_adapters measures each recorded throttle against the caps
in force on its own row, so "how hard I pulled" is what is learned and what
is replayed, at whatever caps the car drives at now. Every continuous output
is checked with isfinite() and against its range before it is scaled, and
never clamped into range first: min(1.0, nan) is 1.0, and 1.0 times
forward_speed is the most throttle this file can command, but a broken model
gets neutral, not that.

A reverse is the model's, tick by tick: reverse_speed times its magnitude,
on its steering, decided again at the next 10 Hz tick like every other
answer. The canned 2 s recover is the thing being replaced, so it is kept
for STUCK only.

Steering is in manual's units, a fraction of STEER_SCALE (main's
STEERING_LIMIT_PERC, 70%), not of the autopilot's max_turn 0.6: the labels
are what my stick put on the servos, and full lock is 0.7. The trainer
writes the scale into the sidecar and PolicyModel refuses one that differs.

Steering mode
-------------
L3 flips main.SteerMode between unified and split. I have only ever driven
split with the left stick, so in my data split is two-wheel steering (rear
held straight) and unified is four-wheel (rear counter-phase through
driver.steering_pair()). A model with the mode head (the sidecar lists
steer_modes) says which of the two it wants each tick, given the mode it is
in now -- a third input, `mode`, a one-hot over STEER_MODES -- and the rear
follows the rule manual's does: steering_pair(front) in unified, 0 in split.
A split with my right stick on the rear is not modelled; the dataset drops
those rows.

A toggle is sticky and a head's answer is not, so the head does not switch
the car. ModeLatch does: the other mode has to be above MODE_THRESHOLD for
MODE_DWELL_S, and the last switch has to be MODE_MIN_INTERVAL_S ago. The
sidecar carries the three numbers; these are the defaults. Who else sets it:

  auto engages               the latch starts at the pad's SteerMode
  my hand has the steering   my SteerMode and my sticks drive, as always;
                             the latch follows my mode the whole time, so
                             when I hand the steering back it is in my mode,
                             and its interval starts again
  L3 while the policy drives my press wins: the latch takes the pad's new
                             mode and its interval starts again, so the model
                             cannot undo the press for MODE_MIN_INTERVAL_S

A model with no mode head runs fixed unified, which is what every model
before the head did, whatever the pad's mode.

The input contract
------------------
A model is a .onnx file with a .json sidecar beside it. The sidecar says which
downsample produced the frames it was trained on, and the runtime refuses a
model whose contract it cannot reproduce. Two exist:

  depth_small   the recording's own downsample, policy_input.depth_small(),
                imported by name (DEPTH_SMALL_MODULE below). Never
                reimplemented here: a model trained on one function and run
                on a copy of it is a model running on input it never saw.
  legacy_jet    the old recordings, which only kept /camera/depth as a
                colourmapped 320x200 JPEG. legacy_small_from_raw() puts a raw
                frame through the same halve, clip and quantise, then the
                same depth_small(), so a model
                trained on those files sees the same picture live. It cannot
                reproduce the JPEG noise; that is the smoke-test model's
                problem, not the next one's.

The IMU
-------
A model whose sidecar says uses_imu has a fourth input, `imu`: the
policy_imu features of the tick (gyro, linear accel, their half-second
peaks, a validity flag, and the up vector when its data had one). The
dataset tool makes them by pushing each /footprint row through
policy_imu.ImuFeatures; PolicyDriver makes them by building that same row
live (policy_imu.row_fields, the function footprint.py writes the row with)
and pushing it through its own ImuFeatures. check_imu() refuses a sidecar
whose feature list is not policy_imu.FEATURES. A model with no uses_imu --
every model before this, the deployed one included -- has no `imu` input
and is never fed one.

numpy only at import. onnxruntime is imported when a model is loaded, so this
file imports on a laptop and in every test without it.
"""

import hashlib
import importlib
import json
import math
import os
import time
from typing import NamedTuple

import numpy as np

import column_profile as cp
import driver as dv
import policy_imu as pimu

# ---- the frame the model sees ----------------------------------------------

INPUT_HEIGHT = 50
INPUT_WIDTH = 80

# Depth is scaled by this and clipped to 1. Past 3 m this car does not care,
# which is the same window the telemetry colourmap has always used.
DEPTH_SCALE_MM = 3000.0

# Where the recording's downsample lives. The ONE place the name is written:
# the dataset tool and the runtime both resolve it through
# depth_small_function(), so if the module is renamed this is the edit.
DEPTH_SMALL_MODULE = "policy_input"
DEPTH_SMALL_FUNCTION = "depth_small"

CONTRACT_DEPTH_SMALL = "depth_small"
CONTRACT_LEGACY_JET = "legacy_jet"
CONTRACTS = (CONTRACT_DEPTH_SMALL, CONTRACT_LEGACY_JET)

# The legacy pipeline, as telemetry.py rendered /camera/depth: nearest-neighbour
# down to 320x200, clip to 300..3000 mm, 256 levels, JET, JPEG at quality 60.
LEGACY_NEAR_MM = 300.0
LEGACY_FAR_MM = 3000.0
LEGACY_LEVELS = 255
LEGACY_SHAPE = (200, 320)
# The colourmap clipped every invalid pixel to the near end, so a 0 and a
# reading at 300 mm come out the same dark blue. Levels at or below this are
# called invalid, which is what most of them are on a stereo pair, and the
# margin above zero absorbs the JPEG's few levels of noise.
LEGACY_INVALID_LEVEL = 2

# ---- the decision the model makes -----------------------------------------

FORWARD = "forward"
REVERSE = "reverse"
STOP = "stop"
ACTIONS = (FORWARD, REVERSE, STOP)

# Two ways to show the model the frame. Both come off the same downsample, so
# the recording format does not choose between them; the sidecar does.
IMAGE = "image"        # the whole (2, H, W): scaled depth and valid mask
PROFILE = "profile"    # column_profile(): (3, 32) off the footprint's rows
REPRESENTATIONS = (IMAGE, PROFILE)

# The steering unit, train and run: a model's steer of 1.0 is a front servo
# command of this. main.STEERING_LIMIT_PERC / 100, written here as a number
# because main imports pygame and this must not; a test holds them equal.
STEER_SCALE = 0.7

# /footprint's autopilot field: who was driving the "driver" groups.
AUTOPILOT_RULES = "rules"
AUTOPILOT_POLICY = "policy"

# Previous-action input: a one-hot over ACTIONS, the throttle magnitude and
# the steering fraction.
PREV_SIZE = len(ACTIONS) + 2

# Steering modes, main.UNIFIED / main.SPLIT, in the order the mode head and
# the mode input use. unified is four-wheel (rear = steering_pair(front)),
# split is two-wheel (rear 0) -- the only split my data has.
UNIFIED = "unified"
SPLIT = "split"
STEER_MODES = (UNIFIED, SPLIT)
MODE_SIZE = len(STEER_MODES)
# The graph's names for them. Two names because ONNX will not let an input
# and an output share one.
MODE_INPUT = "mode"
MODE_OUTPUT = "mode_logits"

# The IMU input, for a model whose sidecar says uses_imu: (1, pimu.IMU_SIZE),
# built by policy_imu.ImuFeatures off the same columns /footprint records.
IMU_INPUT = "imu"

# The mode latch. 0.7 for 0.5 s: six ticks in a row at 10 Hz of a head that
# clearly wants the other mode, not one tick of a coin flip. 2 s between
# switches, so it cannot flap even if the head does: my own toggles in the
# data are 1.2 s apart at the closest, and a policy that toggles faster than
# I do is not copying me.
MODE_THRESHOLD = 0.7
MODE_DWELL_S = 0.5
MODE_MIN_INTERVAL_S = 2.0

# How far past its range a continuous output may be before it is called
# broken. The heads are a tanh and a sigmoid, so anything outside this is a
# model that is not the one that was trained, and that is a reason to stop,
# not to clamp.
OUTPUT_TOLERANCE = 1e-4


def legacy_levels_to_mm(levels):
    """Colourmap levels (0..255) back to millimetres, invalid as 0."""
    levels = np.asarray(levels)
    mm = LEGACY_NEAR_MM + levels.astype(np.float32) * (
        (LEGACY_FAR_MM - LEGACY_NEAR_MM) / LEGACY_LEVELS)
    return np.where(levels <= LEGACY_INVALID_LEVEL, 0, np.round(mm)).astype(np.uint16)


def legacy_mm_from_raw(raw):
    """A raw depth frame through the legacy render, minus the colour and JPEG.

    Downscale to LEGACY_SHAPE nearest-neighbour exactly as telemetry did
    (cv2.resize INTER_NEAREST at an exact 2x is every second pixel), clip,
    quantise to the same 256 levels and back.
    """
    raw = np.asarray(raw)
    sy = max(1, raw.shape[0] // LEGACY_SHAPE[0])
    sx = max(1, raw.shape[1] // LEGACY_SHAPE[1])
    small = raw[::sy, ::sx][:LEGACY_SHAPE[0], :LEGACY_SHAPE[1]].astype(np.float32)
    clipped = np.clip(small, LEGACY_NEAR_MM, LEGACY_FAR_MM)
    levels = ((clipped - LEGACY_NEAR_MM)
              * (LEGACY_LEVELS / (LEGACY_FAR_MM - LEGACY_NEAR_MM))).astype(np.uint8)
    return legacy_levels_to_mm(levels)


def legacy_small_from_mm(mm_legacy):
    """A LEGACY_SHAPE millimetre frame down to the model's input size.

    Through the recording's own depth_small(), at 4x4 blocks instead of 8x8
    because the legacy frame is already half size -- so even the smoke-test
    model sees the same nearest-valid rule the real data is made with.
    """
    return depth_small_function()(np.ascontiguousarray(mm_legacy, dtype=np.uint16))


def legacy_small_from_raw(raw):
    return legacy_small_from_mm(legacy_mm_from_raw(raw))


def depth_small_function():
    """The recording's own downsample, imported. ImportError if it is absent."""
    module = importlib.import_module(DEPTH_SMALL_MODULE)
    return getattr(module, DEPTH_SMALL_FUNCTION)


def downsample_for(contract):
    """The raw-frame-to-input function a model of this contract was trained on."""
    if contract == CONTRACT_LEGACY_JET:
        return legacy_small_from_raw
    if contract == CONTRACT_DEPTH_SMALL:
        return depth_small_function()
    raise ValueError(f"unknown input contract {contract!r}")


def encode_depth(small_mm):
    """(H, W) uint16 mm -> (1, 2, H, W) float32: scaled depth, valid mask."""
    small = np.asarray(small_mm)
    valid = small > 0
    depth = np.where(valid, np.minimum(small, DEPTH_SCALE_MM) / DEPTH_SCALE_MM, 0.0)
    return np.stack([depth, valid]).astype(np.float32)[None]


def encode_input(representation, small_mm):
    """The model's first input for one downsampled frame."""
    if representation == IMAGE:
        return encode_depth(small_mm)
    if representation == PROFILE:
        return cp.column_profile(small_mm)[None]
    raise ValueError(f"unknown representation {representation!r}")


def encode_mode(split):
    """The mode the car is in as the model's third input, (1, MODE_SIZE)."""
    out = np.zeros((1, MODE_SIZE), dtype=np.float32)
    out[0, STEER_MODES.index(SPLIT if split else UNIFIED)] = 1.0
    return out


def encode_prev(action, magnitude, steer):
    """The previous decision as the model's second input, (1, PREV_SIZE)."""
    out = np.zeros((1, PREV_SIZE), dtype=np.float32)
    if action in ACTIONS:
        out[0, ACTIONS.index(action)] = 1.0
    for i, value in ((-2, magnitude), (-1, steer)):
        value = float(value)
        out[0, i] = value if math.isfinite(value) else 0.0
    return out


def encode_imu(features):
    """One decision's policy_imu features as the model's imu input."""
    out = np.asarray(features, dtype=np.float32).reshape(1, pimu.IMU_SIZE)
    return np.where(np.isfinite(out), out, 0.0).astype(np.float32)


def check_imu(meta, inputs, where="sidecar"):
    """The IMU half of the contract -> (uses_imu, tilt). ValueError when the
    sidecar and the graph disagree, or the features are not this car's.

    A sidecar with no uses_imu (every model before the IMU input) needs a
    graph with no IMU_INPUT, and gets (False, False). One with uses_imu true
    needs the input, imu_features equal to policy_imu.FEATURES in order, and
    says in imu_tilt whether its training data had orientation.
    """
    inputs = set(inputs)
    if not meta.get("uses_imu", False):
        if IMU_INPUT in inputs:
            raise ValueError(f"{where}: the graph has an {IMU_INPUT!r} input the "
                             f"sidecar does not declare (no uses_imu)")
        return False, False
    if IMU_INPUT not in inputs:
        raise ValueError(f"{where}: uses_imu declared but the graph has no "
                         f"{IMU_INPUT!r} input")
    names = meta.get("imu_features")
    if not isinstance(names, (list, tuple)) or tuple(names) != pimu.FEATURES:
        raise ValueError(f"{where}: imu_features {names} != {list(pimu.FEATURES)}")
    window = meta.get("imu_window", pimu.WINDOW)
    if window != pimu.WINDOW:
        raise ValueError(f"{where}: imu_window {window!r} != {pimu.WINDOW}")
    return True, bool(meta.get("imu_tilt", False))


def read_mode(outputs):
    """The mode head -> (p_split, ""), (None, "") with no head, or None + why.

    Checked like the other heads: two finite logits, or the tick is neutral.
    """
    if MODE_OUTPUT not in outputs:
        return None, ""
    try:
        logits = np.asarray(outputs[MODE_OUTPUT], dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None, "mode missing"
    if logits.shape != (MODE_SIZE,):
        return None, "mode shape"
    if not all(math.isfinite(v) for v in logits):
        return None, "mode not finite"
    e = np.exp(logits - logits.max())
    return float(e[STEER_MODES.index(SPLIT)] / e.sum()), ""


class ModeLatch:
    """The steering mode the policy drives with. Switches slowly, on purpose.

    update() takes the head's p(split) each tick and switches only when the
    other mode has been above `threshold` for `dwell_s` and the last switch
    was `min_interval_s` ago. set() is the pad -- auto engaging, my hand on
    the steering, an L3 press: it takes the mode and restarts the interval.
    """

    def __init__(self, threshold=MODE_THRESHOLD, dwell_s=MODE_DWELL_S,
                 min_interval_s=MODE_MIN_INTERVAL_S):
        self.threshold = float(threshold)
        self.dwell_s = float(dwell_s)
        self.min_interval_s = float(min_interval_s)
        self.split = False
        self._since = None      # the last switch or set()
        self._wanted = None     # when the other mode first went over threshold

    def set(self, split, now):
        self.split = bool(split)
        self._since = now
        self._wanted = None

    def update(self, p_split, now):
        """One tick of the head. True when this tick switched."""
        p_other = (1.0 - p_split) if self.split else p_split
        if not p_other > self.threshold:
            self._wanted = None
            return False
        if self._wanted is None:
            self._wanted = now
        # 1 ms of slack: ticks are 0.1 s apart in float seconds, and five of
        # them must make 0.5, not 0.49999.
        if (now - self._wanted >= self.dwell_s - 1e-3
                and (self._since is None
                     or now - self._since >= self.min_interval_s - 1e-3)):
            self.split = not self.split
            self._since = now
            self._wanted = None
            return True
        return False


def check_sidecar(meta, inputs, outputs, where="sidecar"):
    """The mode head's half of the contract. The latch's numbers, or None for
    a model with no head (fixed unified). ValueError when the sidecar and the
    graph disagree.

    A sidecar that lists steer_modes needs a graph with a MODE_INPUT input
    and a MODE_OUTPUT output, the modes in STEER_MODES' order, and latch numbers in range. One
    that does not list them needs a graph with neither: a graph wanting a
    mode input the runtime would not know to feed is not one this car runs.
    """
    inputs, outputs = set(inputs), set(outputs)
    modes = meta.get("steer_modes")
    if modes is None:
        if MODE_INPUT in inputs or MODE_OUTPUT in outputs:
            raise ValueError(f"{where}: the graph has a mode head the sidecar "
                             f"does not declare (no steer_modes)")
        return None
    if not isinstance(modes, (list, tuple)) or tuple(modes) != STEER_MODES:
        raise ValueError(f"{where}: steer_modes {modes} != {list(STEER_MODES)}")
    if MODE_INPUT not in inputs or MODE_OUTPUT not in outputs:
        raise ValueError(f"{where}: steer_modes declared but the graph has no "
                         f"{MODE_INPUT!r} input and {MODE_OUTPUT!r} output")
    switch = meta.get("mode_switch")
    try:
        out = {k: float(switch[k]) for k in ("threshold", "dwell_s", "min_interval_s")}
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{where}: mode_switch {switch!r} needs threshold, "
                         f"dwell_s and min_interval_s as numbers") from None
    if not (0.5 <= out["threshold"] < 1.0 and 0.0 <= out["dwell_s"] <= 10.0
            and 0.0 <= out["min_interval_s"] <= 60.0):
        raise ValueError(f"{where}: mode_switch {switch!r} out of range")
    return out


class Decision(NamedTuple):
    """What the policy decided on one tick, and why. Read-only bookkeeping.

    magnitude is the fraction of the speed cap for `action` (forward_speed
    or reverse_speed); steer and rear are fractions of STEER_SCALE, rear None
    when the model has no rear head and steering_pair() derives it.
    throttle and front are what went into the command, after the floor veto.
    """

    action: str
    magnitude: float
    steer: float
    rear: object
    reason: str
    throttle: float = 0.0
    front: float = 0.0


def _in(value, low, high):
    return low - OUTPUT_TOLERANCE <= value <= high + OUTPUT_TOLERANCE


def read_output(outputs):
    """Model outputs -> (action, magnitude, steer, rear), or None + why.

    `outputs` maps logits, throttle, steer and optionally rear to arrays.
    isfinite() before anything else touches the numbers. Nothing here
    clamps: a value past its range from a sigmoid or tanh head is a broken
    model, and the answer to a broken model is neutral.
    """
    try:
        logits = np.asarray(outputs["logits"], dtype=np.float64).ravel()
        mag = np.asarray(outputs["throttle"], dtype=np.float64).ravel()
        steer = np.asarray(outputs["steer"], dtype=np.float64).ravel()
        rear = outputs.get("rear")
        rear = None if rear is None else np.asarray(rear, dtype=np.float64).ravel()
    except (KeyError, TypeError, ValueError):
        return None, "missing"
    if (logits.shape != (len(ACTIONS),) or mag.shape != (1,) or steer.shape != (1,)
            or (rear is not None and rear.shape != (1,))):
        return None, "shape"
    values = [*logits, mag[0], steer[0]] + ([] if rear is None else [rear[0]])
    if not all(math.isfinite(v) for v in values):
        return None, "not finite"
    if not _in(mag[0], 0.0, 1.0):
        return None, "throttle out of range"
    if not _in(steer[0], -1.0, 1.0) or (rear is not None and not _in(rear[0], -1.0, 1.0)):
        return None, "steer out of range"
    # Inside the tolerance, now known finite, so a clip cannot invent a value.
    return (ACTIONS[int(np.argmax(logits))], float(np.clip(mag[0], 0.0, 1.0)),
            float(np.clip(steer[0], -1.0, 1.0)),
            None if rear is None else float(np.clip(rear[0], -1.0, 1.0))), ""


class PolicyModel:
    """An exported policy: the ONNX session plus the sidecar it came with."""

    def __init__(self, path, threads=1):
        meta_path = os.path.splitext(path)[0] + ".json"
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.contract = self.meta.get("input_contract")
        if self.contract not in CONTRACTS:
            raise ValueError(f"{meta_path}: unknown input contract {self.contract!r}")
        self.representation = self.meta.get("representation")
        if self.representation not in REPRESENTATIONS:
            raise ValueError(f"{meta_path}: unknown representation {self.representation!r}")
        if tuple(self.meta.get("actions", ())) != ACTIONS:
            raise ValueError(f"{meta_path}: actions {self.meta.get('actions')} != {ACTIONS}")
        self.uses_prev = bool(self.meta.get("uses_prev_action", False))
        scale = self.meta.get("steer_scale")
        if not isinstance(scale, (int, float)) or abs(float(scale) - STEER_SCALE) > 1e-6:
            raise ValueError(f"{meta_path}: trained at steer_scale {scale!r}, "
                             f"this car steers at {STEER_SCALE}")
        with open(path, "rb") as f:
            # Which model drove, in every /footprint row it wrote.
            self.sha256 = hashlib.sha256(f.read()).hexdigest()

        import onnxruntime as ort

        options = ort.SessionOptions()
        # One thread by default. The control loop shares these four cores and
        # both models are measured on the car at one thread -- see the README.
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, options,
                                            providers=["CPUExecutionProvider"])
        self.output_names = [o.name for o in self.session.get_outputs()]
        # None: no mode head, fixed unified. Else the latch's numbers.
        input_names = [i.name for i in self.session.get_inputs()]
        self.mode_switch = check_sidecar(
            self.meta, input_names, self.output_names, meta_path)
        self.has_mode = self.mode_switch is not None
        # (False, False) for every model before the IMU input: it is never
        # fed one, and runs exactly as it did.
        self.uses_imu, self.imu_tilt = check_imu(self.meta, input_names, meta_path)
        self.path = path

    def run(self, x, prev, mode=None, imu=None):
        feed = {"x": x, "prev": prev}
        if self.has_mode:
            feed[MODE_INPUT] = mode if mode is not None else encode_mode(False)
        if self.uses_imu:
            feed[IMU_INPUT] = (imu if imu is not None
                               else np.zeros((1, pimu.IMU_SIZE), dtype=np.float32))
        values = self.session.run(self.output_names, feed)
        return dict(zip(self.output_names, values))


class PolicyDriver(dv.Driver):
    """Driver, with the verdict coming from a model instead of the gradient.

    `model` is anything with .contract, .representation, .uses_prev and
    .run(x, prev) -> {name: array} -- a PolicyModel on the car, a stub in the
    tests. One with .has_mode true is run as .run(x, prev, mode), and its
    .mode_switch dict sets the latch. One with .uses_imu true also gets
    imu=, its policy_imu features, built by this driver's own ImuFeatures
    off the row footprint.py would have written for the tick.

    AutoPilot.step() sets two things before each tick, as it sets
    human_throttle: pad_split, the pad's SteerMode (None reads as unified),
    and human_steering, whether my hand has the steering group.
    """

    def __init__(self, controller, oakd, model, downsample=None):
        super().__init__(controller, oakd)
        self.model = model
        self.downsample = downsample or downsample_for(model.contract)
        self._prev = (STOP, 0.0, 0.0)
        # The forward floor and its memory of the last second of scans. This
        # driver's own: it is shown every scan this driver makes and every
        # throttle it commands.
        self.floor_veto = dv.ForwardFloorVeto(self.forward_speed)
        self._now = None  # the tick in progress, for the neutral path
        self.has_mode = bool(getattr(model, "has_mode", False))
        switch = getattr(model, "mode_switch", None) or {}
        self.mode = ModeLatch(switch.get("threshold", MODE_THRESHOLD),
                              switch.get("dwell_s", MODE_DWELL_S),
                              switch.get("min_interval_s", MODE_MIN_INTERVAL_S))
        self.pad_split = None
        self.human_steering = False
        self._pad_seen = None       # the pad's mode last tick; None until engaged
        self._had_steering = False
        # Read-only bookkeeping, like last_scan and last_verdict.
        self.last_decision = None
        self.last_infer_ms = 0.0
        self.last_split_p = -1.0    # the head's p(split); -1 with no head
        # The IMU input. The newest sample and when it was taken, held as
        # footprint.FootprintTap holds them, so the row this driver builds
        # its features off is the row a recording of this tick would carry.
        self.uses_imu = bool(getattr(model, "uses_imu", False))
        self.imu_features = pimu.ImuFeatures(tilt=bool(getattr(model, "imu_tilt", False)))
        self._imu_sample = None
        self._imu_at = 0.0
        self.last_imu = None        # the features fed on the last model run

    def abort(self):
        """Driver's abort, and the mode latch forgets the pad: the next tick
        is an engage, and starts at whatever SteerMode is then."""
        super().abort()
        self._pad_seen = None
        self.imu_features.reset()

    def _sync_mode(self, now):
        """The pad's say in the policy's mode, before the model has its own.

        Engaging, an L3 press and my hand on the steering all set the latch
        to the pad's mode; handing the steering back sets it once more, which
        restarts its interval.
        """
        pad = bool(self.pad_split)
        if not self.has_mode:
            self.mode.split = False
        elif (self._pad_seen is None or pad != self._pad_seen
                or self.human_steering or self._had_steering):
            self.mode.set(pad, now)
        self._pad_seen = pad
        self._had_steering = bool(self.human_steering)

    def tick(self, now=None):
        """One pass. Anything that raises commands neutral first."""
        if now is None:
            now = time.monotonic()
        try:
            self._sync_mode(now)
            self._tick(now)
        except Exception:
            self._neutral("exception")
            raise

    def _neutral(self, reason):
        """Throttle to zero, steering to centre, evidence forgotten.

        Each write on its own try, throttle first: a steering servo that
        raises must not stop the throttle reaching zero, and this is called
        from the exception path, where raising again would lose the original.
        """
        try:
            self.controller.set_throttle(0)
        except Exception:
            pass
        try:
            self.controller.set_steering(0, 0)
        except Exception:
            pass
        self._commanded(0.0, self._now)
        self._clear_motion_evidence()
        self._prev = (STOP, 0.0, 0.0)
        self._note(Decision(STOP, 0.0, 0.0, None, reason))

    def _note(self, decision):
        """Leave the decision behind for telemetry, as the autopilot does.

        last_verdict is what /footprint's committed_* fields and /marks read
        off an auto driver, so it is filled in the autopilot's vocabulary:
        forward and reverse as theirs, stop as itself, the reason prefixed
        so a row says the model made it.
        """
        self.last_decision = decision
        action = {FORWARD: dv.ACT_FORWARD, REVERSE: dv.ACT_REVERSE}.get(decision.action, STOP)
        self.last_verdict = dv.SteerVerdict(action, decision.steer,
                                            f"policy {decision.reason}")

    def policy_fields(self):
        """What /footprint records about the policy's own command this tick.

        throttle and front are what the model put into the command, after
        the floor veto; the row's throttle / steering_* are what reached the
        hardware once my hand's groups were mixed in, which is not the same
        thing during a takeover.
        """
        d = self.last_decision
        return {
            "autopilot": AUTOPILOT_POLICY,
            "policy_model": str(getattr(self.model, "sha256", "")),
            "policy_action": d.action if d is not None else "",
            "policy_throttle": float(d.throttle) if d is not None else 0.0,
            "policy_steering_front": float(d.front) if d is not None else 0.0,
            "policy_reason": d.reason if d is not None else "",
            "policy_infer_ms": float(self.last_infer_ms),
            # The mode the policy drove with, and the head's probability of
            # that mode on its last answer: low while it wants to switch. -1
            # with no head, which always drives unified.
            "policy_steer_mode": SPLIT if self.mode.split else UNIFIED,
            "policy_steer_mode_p": (-1.0 if self.last_split_p < 0
                                    else self.last_split_p if self.mode.split
                                    else 1.0 - self.last_split_p),
        }

    def _recover(self, now, reason):
        """The canned backup-and-turn. STUCK only: every other reverse is
        the model's own, tick by tick."""
        self._note(Decision(REVERSE, 1.0, 0.0, None, reason, self.reverse_speed, 0.0))
        self._prev = (REVERSE, 1.0, 0.0)
        self.begin_recover(now)
        self._commanded(self.reverse_speed, now)

    def _commanded(self, throttle, now):
        """Tell the floor veto what this driver just put on the throttle.
        Never raises: it runs on the neutral path too."""
        try:
            if now is not None:
                self.floor_veto.note_throttle(throttle, now)
        except Exception:
            pass

    def _tick(self, now):
        self._now = now
        if self.recovering(now):
            # The recover's reverse, then its settle at 0, as the veto's
            # record of what the wheels were told.
            self._commanded(self.reverse_speed
                            if self._phase == dv._PHASE_REVERSE else 0.0, now)
            return

        depth = self.camera.get_depth_frame()
        if depth is None or getattr(depth, "ndim", 0) != 2:
            self._neutral("no frame")
            return

        imu = self.camera.get_imu_sample()
        if imu is not None:
            self._imu_sample, self._imu_at = imu, now
        # Every tick with a frame pushes, blind and stuck ones too, so the
        # window is the last half second of the car and not of the model.
        imu_x = self.imu_features.push(
            pimu.row_fields(self._imu_sample, self._imu_at, now), now)
        scan = self.scan_footprint(depth)
        # Every scan, blind ones too: a blind scan is exactly the one the
        # latch is for. The cap is read each tick, as the throttle is.
        self.floor_veto.forward_speed = self.forward_speed
        self.floor_veto.observe(scan, now)

        if all(r == dv.NO_RETURN for r in scan.reason):
            self._neutral("blind")
            return

        # Driver's own rule: under my throttle the driver is not telling the
        # car to go, so the stuck detector has nothing to judge.
        if self.human_throttle:
            self._throttle_since = None
            self._still_samples = 0
        elif self.stuck(imu, now):
            self.last_recovery_time = now
            print("STUCK - backing up to recover.")
            self._recover(now, "stuck")
            return

        x = encode_input(self.model.representation, self.downsample(depth))
        prev = (encode_prev(*self._prev) if self.model.uses_prev
                else np.zeros((1, PREV_SIZE), dtype=np.float32))
        mode = encode_mode(self.mode.split) if self.has_mode else None
        started = time.perf_counter()
        if self.uses_imu:
            self.last_imu = encode_imu(imu_x)
            outputs = self.model.run(x, prev, mode, imu=self.last_imu)
        elif self.has_mode:
            outputs = self.model.run(x, prev, mode)
        else:
            outputs = self.model.run(x, prev)
        self.last_infer_ms = (time.perf_counter() - started) * 1000.0

        decided, why = read_output(outputs)
        if decided is None:
            self._neutral(f"model {why}")
            return
        action, magnitude, steer, rear = decided
        if self.has_mode:
            p_split, why = read_mode(outputs)
            if p_split is None:
                self._neutral(f"model {why or 'mode missing'}")
                return
            self.last_split_p = p_split
            # My hand on the steering means my mode: _sync_mode holds the
            # latch to it, and the head does not get to move it.
            if not self.human_steering:
                self.mode.update(p_split, now)

        reason = "model"
        if action == FORWARD and self.floor_veto.vetoes():
            # Blocks the forward, not the decision: the steering still goes
            # out, and next tick the model can choose to reverse.
            throttle, reason = 0.0, "model floor"
        elif action == FORWARD:
            throttle = self.forward_speed * magnitude
        elif action == REVERSE:
            throttle = self.reverse_speed * magnitude
        else:
            throttle = 0.0

        # magnitude, steer and rear are already known finite and in range,
        # so none of these can exceed a cap or STEER_SCALE.
        front = steer * STEER_SCALE
        if rear is not None:
            pair = (front, rear * STEER_SCALE)
        elif self.mode.split:
            # Split as I drive it: the left stick on the front, the rear
            # straight. A literal 0, never through steering_pair.
            pair = (front, 0.0)
        else:
            pair = dv.steering_pair(front)
        # Steering first, as the autopilot orders them; main.DriverCommand
        # holds both and AutoPilot.step() writes them once.
        self.controller.set_steering(*pair)
        self.controller.set_throttle(throttle)
        self._commanded(throttle, now)
        if throttle > 0.0:
            if self._throttle_since is None:
                self._throttle_since = now
        else:
            # Only a forward the driver itself commanded arms STUCK.
            self._clear_motion_evidence()
        self._prev = (action, magnitude, steer)
        self._note(Decision(action, magnitude, steer, rear, reason, throttle, front))
