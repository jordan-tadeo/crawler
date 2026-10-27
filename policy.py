"""A learned driving decision, behind the same safety the autopilot has.

What this is
------------
A small model, trained off the car by behaviour cloning, that looks at one
downsampled depth frame and says three things: a direction (forward,
reverse or stop), how hard as a fraction of that direction's speed cap, and
a front steering fraction -- plus a rear one, if a model is ever trained on
split steering. The frame reaches it either whole (IMAGE, a small 2-D CNN)
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

  recover owns the wheels           the model is not consulted at all
  no new depth frame, or a frame
  that is not a 2-D array           neutral (throttle 0, steering 0,0)
  the whole fan unmeasurable        neutral -- a blind camera is not a
                                    picture to decide on
  any probe nearer than
  REVERSE_FLOOR_MM                  the ordinary recover, before the model
                                    runs; the model cannot see this frame
  the stuck detector fires          the ordinary recover, same
  model output not finite, the
  wrong shape, throttle past [0, 1]
  or steering past +-1              neutral
  anything raises                   neutral, then the exception goes on up
                                    to AutoPilot.step(), which parks auto

The throttle is a magnitude, never a signed number. Forward and reverse are
both right answers in the same scene often enough that one regressed signed
throttle averages them into a stop, so the direction is a classification and
the magnitude a sigmoid in [0, 1] that only ever multiplies a bench cap --
forward_speed for forward, reverse_speed for reverse. Every continuous output
is checked with isfinite() and against its range before it is scaled, and
never clamped into range first: min(1.0, nan) is 1.0, and 1.0 times
forward_speed is the most throttle this file can command, but a broken model
gets neutral, not that.

Reverse means a recover, at the model's speed and steered the way it asked
instead of a random way. That keeps the blind direction inside the phase machine that already
has a rear bumper and a fixed length, rather than letting a model hold a
reverse open tick by tick with nothing pointed behind the car.

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

numpy only at import. onnxruntime is imported when a model is loaded, so this
file imports on a laptop and in every test without it.
"""

import importlib
import json
import math
import os
import time
from typing import NamedTuple

import numpy as np

import column_profile as cp
import driver as dv

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

# Previous-action input: a one-hot over ACTIONS, the throttle magnitude and
# the steering fraction.
PREV_SIZE = len(ACTIONS) + 2

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


def encode_prev(action, magnitude, steer):
    """The previous decision as the model's second input, (1, PREV_SIZE)."""
    out = np.zeros((1, PREV_SIZE), dtype=np.float32)
    if action in ACTIONS:
        out[0, ACTIONS.index(action)] = 1.0
    for i, value in ((-2, magnitude), (-1, steer)):
        value = float(value)
        out[0, i] = value if math.isfinite(value) else 0.0
    return out


class Decision(NamedTuple):
    """What the policy decided on one tick, and why. Read-only bookkeeping.

    magnitude is the fraction of the speed cap for `action` (forward_speed
    or reverse_speed); steer and rear are fractions of max_turn, rear None
    when the model has no rear head and steering_pair() derives it.
    """

    action: str
    magnitude: float
    steer: float
    rear: object
    reason: str


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

        import onnxruntime as ort

        options = ort.SessionOptions()
        # One thread by default. The control loop shares these four cores and
        # both models are measured on the car at one thread -- see the README.
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, options,
                                            providers=["CPUExecutionProvider"])
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.path = path

    def run(self, x, prev):
        values = self.session.run(self.output_names, {"x": x, "prev": prev})
        return dict(zip(self.output_names, values))


class PolicyDriver(dv.Driver):
    """Driver, with the verdict coming from a model instead of the gradient.

    `model` is anything with .contract, .representation, .uses_prev and
    .run(x, prev) -> {name: array} -- a PolicyModel on the car, a stub in the
    tests.
    """

    def __init__(self, controller, oakd, model, downsample=None):
        super().__init__(controller, oakd)
        self.model = model
        self.downsample = downsample or downsample_for(model.contract)
        self._prev = (STOP, 0.0, 0.0)
        # Read-only bookkeeping, like last_scan and last_verdict.
        self.last_decision = None
        self.last_infer_ms = 0.0

    def tick(self, now=None):
        """One pass. Anything that raises commands neutral first."""
        if now is None:
            now = time.monotonic()
        try:
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

    def _recover(self, now, reason, magnitude=1.0, steer=None):
        """Back out through the ordinary recover phase machine.

        steer None is the random turn a veto has always got. A model reverse
        picks the turn and the speed; magnitude is already checked to be in
        [0, 1], so the speed can never exceed reverse_speed.
        """
        self._note(Decision(REVERSE, magnitude, 0.0 if steer is None else steer,
                            None, reason))
        self._prev = (REVERSE, magnitude, 0.0 if steer is None else steer)
        self.begin_recover(
            now,
            turn=None if steer is None else steer * self.max_turn,
            speed=self.reverse_speed * magnitude)

    def _tick(self, now):
        if self.recovering(now):
            return

        depth = self.camera.get_depth_frame()
        if depth is None or getattr(depth, "ndim", 0) != 2:
            self._neutral("no frame")
            return

        imu = self.camera.get_imu_sample()
        scan = self.scan_footprint(depth)

        if all(r == dv.NO_RETURN for r in scan.reason):
            self._neutral("blind")
            return

        measured = scan.distance_mm[np.isfinite(scan.distance_mm)]
        if measured.size and float(measured.min()) < dv.REVERSE_FLOOR_MM:
            self._recover(now, f"floor {float(measured.min()):.0f}")
            return

        if self.stuck(imu, now):
            self.last_recovery_time = now
            print("STUCK - backing up to recover.")
            self._recover(now, "stuck")
            return

        x = encode_input(self.model.representation, self.downsample(depth))
        prev = (encode_prev(*self._prev) if self.model.uses_prev
                else np.zeros((1, PREV_SIZE), dtype=np.float32))
        started = time.perf_counter()
        outputs = self.model.run(x, prev)
        self.last_infer_ms = (time.perf_counter() - started) * 1000.0

        decided, why = read_output(outputs)
        if decided is None:
            self._neutral(f"model {why}")
            return
        action, magnitude, steer, rear = decided

        if action == REVERSE:
            self._recover(now, "model", magnitude, steer)
            return

        # A stop takes the throttle off before it touches the steering; a
        # forward steers first, as the autopilot does, so the car never rolls
        # on last tick's angle.
        if action == STOP:
            self.controller.set_throttle(0)
            self._clear_motion_evidence()
        front = steer * self.max_turn
        if rear is None:
            self.controller.set_steering(*dv.steering_pair(front))
        else:
            self.controller.set_steering(front, rear * self.max_turn)
        if action == FORWARD:
            self.controller.set_throttle(self.forward_speed * magnitude)
            if self._throttle_since is None:
                self._throttle_since = now
        self._prev = (action, magnitude, steer)
        self._note(Decision(action, magnitude, steer, rear, "model"))
