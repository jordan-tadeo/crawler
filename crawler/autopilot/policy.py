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

It is opt-in. crawler/__main__.py builds one only when --policy names a model file, and
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
  the floor holds a model forward
  for STANDOFF_S in a row           the STUCK recover, reason "standoff";
                                    the model has the car back after it
  anything raises                   neutral, then the exception goes on up
                                    to AutoPilot.step(), which parks auto
  the goal is reached               neutral, whatever the model said

The reverse commit
------------------
A reverse the model starts after not reversing is held for REVERSE_COMMIT_S
at REVERSE_STEER, on the side reverse_commit_front() picks off my record
data, whatever the model says in between; then the model decides again.
The model is still run every tick of it and every row above still stops it
at once: no frame, blind, a bad output, STUCK, the goal. So does my hand on
either group, and the recover's rear bumper -- though at 10 Hz the bumper
needs 0.9 s of reverse to speak (driver.REVERSE_STALL_SAMPLES), so at
REVERSE_COMMIT_S 0.7 it cannot, and only would if I raise it. The
confidence scale still slows its throttle. The steering mode is the
latch's, as always: in unified the rear goes counter-phase through
steering_pair(), which tightens the arc and swings the nose the same way,
faster; in split the rear stays straight. /footprint says reverse_commit
and standoff_ticks.

The goal
--------
The model's own "stop" is never committed on its own say. goal.Goal is the
one source of whether the car has arrived, asked every decision; while it
says no (always, for now) a stop answer is driven as the better of forward
and reverse by the model's own logits, at the magnitude and steering the
model gave -- the magnitude head is only ever trained on moving ticks, so
it is a moving answer even on a stop. Only the model's choice is masked:
every row above that commands neutral, the recover, and the floor's hold on
a forward are all where they were, and a masked forward goes through the
floor like any other. When the goal says yes, the tick is neutral with
reason "goal reached". /footprint keeps the model's raw choice in
policy_action and says goal_reached and policy_stop_masked beside it.

Confidence
----------
The less sure the model is, the slower it drives. Confidence is the softmax
probability of the action committed (after the goal mask), smoothed over
CONF_EMA_DECISIONS decisions, ramped from SPEED_FLOOR at CONF_FLOOR to 1.0
at CONF_FULL, and moved at most MAX_SCALE_STEP a decision. The scale only
multiplies a forward or reverse throttle's magnitude, after the caps, and
never takes it under MOVING_FLOOR_FORWARD / MOVING_FLOOR_REVERSE, so the
car slows rather than stalls. Every stop in the table above returns before
it, so they are all immediate and unscaled, and a neutral is never slewed.
The EMA starts over whenever auto is paused. /footprint carries
policy_confidence, policy_confidence_smoothed and policy_speed_scale.

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
answer -- except the first REVERSE_COMMIT_S of a fresh one (below). The
canned 2 s recover is kept for STUCK and for a standoff with the floor.

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
whose feature list is not policy_imu.FEATURES, or NO_YAW_FEATURES when it
says imu_yaw false: a model trained without the yaw rate (policy_imu,
"Without yaw"), whose ImuFeatures is built yaw=False. A model with no uses_imu --
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

from crawler.autopilot import column_profile as cp
from crawler.autopilot import driver as dv
from crawler.autopilot import goal as gl
from crawler.autopilot import policy_imu as pimu

# ---- the frame the model sees ----------------------------------------------

INPUT_HEIGHT = 50
INPUT_WIDTH = 80

# Depth is scaled by this and clipped to 1. Past 3 m this car does not care,
# which is the same window the telemetry colourmap has always used.
DEPTH_SCALE_MM = 3000.0

# Where the recording's downsample lives. The ONE place the name is written:
# the dataset tool and the runtime both resolve it through
# depth_small_function(), so if the module is renamed this is the edit.
DEPTH_SMALL_MODULE = "crawler.autopilot.policy_input"
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

# The steering head, which the sidecar names as steer_head. A model with no
# steer_head is a REGRESSION: one tanh output, `steer`. That head averages
# when I could have gone either way: backing out of a 3-point turn I go hard
# left or hard right, the flip augmentation makes those scenes exactly
# bimodal, and the L1 answer between full left and full right is straight.
# DIRECTION is a class over STEER_DIRS (`steer_dir`, three logits) and how
# far (`steer_mag`, a sigmoid, the fraction of full lock), and the steer is
# the class's side times the magnitude, 0 for straight. decode_steer() is
# the one place either is read, on the car and in the eval.
STEER_REGRESSION = "regression"
STEER_DIRECTION = "direction"
STEER_HEADS = (STEER_REGRESSION, STEER_DIRECTION)
STEER_OUTPUT = "steer"
STEER_DIR_OUTPUT = "steer_dir"
STEER_MAG_OUTPUT = "steer_mag"
# Negative steering is left, as driver.turn_sweep_clears(scan, -1) is.
LEFT = "left"
STRAIGHT = "straight"
RIGHT = "right"
STEER_DIRS = (LEFT, STRAIGHT, RIGHT)
STEER_SIDE = (-1.0, 0.0, 1.0)
# What /footprint's policy_steer_dir_p says for a regression head.
NO_STEER_DIR_P = (-1.0, -1.0, -1.0)

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

# ---- how sure it is: the throttle slows as the model's confidence drops ------
#
# Confidence is the softmax probability, over the three action logits, of the
# action actually committed -- after the goal mask, so a masked stop driven as
# a reverse is as sure as the model was of reverse. It is smoothed, ramped to
# a speed scale and slew-limited, and the scale multiplies the throttle's
# magnitude after the caps. Every stop below is untouched by it. I tune these
# by feel.
CONF_FULL = 0.9          # smoothed confidence at or above this drives at full speed
CONF_FLOOR = 0.5         # at or below this the scale bottoms out at SPEED_FLOOR
SPEED_FLOOR = 0.5        # the slowest the scale goes, as a fraction of the throttle
MAX_SCALE_STEP = 0.1     # the most the scale moves in one decision, either way
CONF_EMA_DECISIONS = 5   # EMA length in decisions; 5 at 10 Hz is about 0.5 s

# The least throttle that moves the car, by direction, as magnitudes. The
# scale never takes a throttle under these; a throttle the model already put
# under them is left where it was. Not calibrated: deploy/speed_cal.py
# measures the real ones at full lock and has never been run into this file.
# Forward 0.2 is the start threshold the first record session found (driver.py,
# above FORWARD_SPEED). Reverse 0.6 is the old REVERSE_SPEED, which drove
# every backup-and-turn recover with the wheels turned; -0.2957 barely moved
# straight and did not move at lock.
MOVING_FLOOR_FORWARD = 0.2
MOVING_FLOOR_REVERSE = 0.6

# ---- the standoff: the model pushing forward into the floor ----------------
#
# The floor holds a forward at 0 and the model keeps saying forward, so
# neither gives way and the car sits there until I take it. The first drive
# on base17 had four of these, 4.5 to 9.8 s each. A floor-held forward
# commands no throttle, so STUCK never arms on one. I tune this by feel.
STANDOFF_S = 1.0         # a floor hold this long runs the STUCK recover

# ---- the reverse commit: a fresh model reverse is held, at lock -------------
#
# Tick by tick, the model reverses almost straight (median |front| 0.10) for
# about 3 ticks, so it rocks back and forth on one line. I reverse at lock
# (median 0.70) for 7 ticks. So a reverse the model starts after not
# reversing is held for REVERSE_COMMIT_S at REVERSE_STEER, then it decides
# again. Every stop still ends it at once. I tune these by feel.
REVERSE_COMMIT_S = 0.7   # how long a fresh reverse is held; 7 ticks, my median run
REVERSE_STEER = 0.7      # the commit's front servo command, my median reversing; full lock
REVERSE_TURN_MIN = 0.15  # a front past this is a turn, not straight (servo units)
REVERSE_TURN_WINDOW_S = 1.5  # a turning forward this recent sets the commit's side
REVERSE_DEFAULT_SIDE = -1.0  # with no turn to go on: left, as I did 15 of 19 times


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
    out = np.asarray(features, dtype=np.float32).reshape(1, -1)
    return np.where(np.isfinite(out), out, 0.0).astype(np.float32)


def check_imu(meta, inputs, where="sidecar"):
    """The IMU half of the contract -> (uses_imu, tilt, yaw). ValueError when the
    sidecar and the graph disagree, or the features are not this car's.

    A sidecar with no uses_imu (every model before the IMU input) needs a
    graph with no IMU_INPUT, and gets (False, False, False). One with uses_imu
    true needs the input, imu_features equal to policy_imu.features(yaw) in
    order -- yaw is imu_yaw, true when absent (every model before it) --
    and says in imu_tilt whether its training data had orientation.
    """
    inputs = set(inputs)
    if not meta.get("uses_imu", False):
        if IMU_INPUT in inputs:
            raise ValueError(f"{where}: the graph has an {IMU_INPUT!r} input the "
                             f"sidecar does not declare (no uses_imu)")
        return False, False, False
    if IMU_INPUT not in inputs:
        raise ValueError(f"{where}: uses_imu declared but the graph has no "
                         f"{IMU_INPUT!r} input")
    yaw = meta.get("imu_yaw", True)
    if not isinstance(yaw, bool):
        raise ValueError(f"{where}: imu_yaw {yaw!r} is not true or false")
    want = pimu.features(yaw)
    names = meta.get("imu_features")
    if not isinstance(names, (list, tuple)) or tuple(names) != want:
        raise ValueError(f"{where}: imu_features {names} != {list(want)} "
                         f"(imu_yaw {str(yaw).lower()})")
    window = meta.get("imu_window", pimu.WINDOW)
    if window != pimu.WINDOW:
        raise ValueError(f"{where}: imu_window {window!r} != {pimu.WINDOW}")
    return True, bool(meta.get("imu_tilt", False)), yaw


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


def committed_confidence(outputs, action):
    """The softmax probability of `action` over the model's action logits.

    Only called on outputs read_output() has already checked, so the logits
    are three finite numbers.
    """
    logits = np.asarray(outputs["logits"], dtype=np.float64).ravel()
    e = np.exp(logits - logits.max())
    return float(e[ACTIONS.index(action)] / e.sum())


def speed_scale_for(confidence):
    """The linear ramp: CONF_FLOOR -> SPEED_FLOOR, CONF_FULL -> 1.0, clamped."""
    t = (confidence - CONF_FLOOR) / (CONF_FULL - CONF_FLOOR)
    t = min(1.0, max(0.0, t))
    return SPEED_FLOOR + t * (1.0 - SPEED_FLOOR)


def slowed_throttle(throttle, scale, floor):
    """A signed throttle with its magnitude scaled, never under `floor`.

    A throttle already under the floor is left as it is: the scale only
    slows a throttle down to the floor, it never speeds one up. 0 stays 0.
    """
    mag = abs(float(throttle))
    slowed = max(mag * scale, min(mag, floor))
    return math.copysign(slowed, throttle) if mag > 0.0 else 0.0


def reverse_commit_front(turn_front, turn_age_s, model_front):
    """The front servo command a reverse commit holds. Pure.

    My record reverses say one thing clearly: I reverse to the opposite side
    of the forward turn I was just in -- 83 of 85 times when I had turned
    within 1.5 s -- which keeps the car yawing the same way, a K-turn. Which
    side had more room said nothing: steering toward the tighter side fits 36
    of 69, a coin. So:

      a forward turn (|front| > REVERSE_TURN_MIN) inside REVERSE_TURN_WINDOW_S
                         the other side
      else the model's own reverse steer, if it is past REVERSE_TURN_MIN
                         its side
      else               REVERSE_DEFAULT_SIDE

    turn_front is None when there has been no turn; turn_age_s is how long
    ago it was. Positive is right, as everywhere.
    """
    if (turn_front is not None and turn_age_s <= REVERSE_TURN_WINDOW_S
            and abs(turn_front) > REVERSE_TURN_MIN):
        side = -math.copysign(1.0, turn_front)
    elif abs(model_front) > REVERSE_TURN_MIN:
        side = math.copysign(1.0, model_front)
    else:
        side = REVERSE_DEFAULT_SIDE
    return side * min(REVERSE_STEER, STEER_SCALE)


class ConfidenceScale:
    """The smoothed confidence and the speed scale it sets. One per driver.

    update() takes one committed decision's raw confidence. The first after a
    reset() seeds the EMA and sets the scale straight off it -- the car is
    starting from a neutral, so there is nothing to be jerky from -- and every
    one after moves the scale by at most MAX_SCALE_STEP.
    """

    def __init__(self):
        self.alpha = 2.0 / (CONF_EMA_DECISIONS + 1)
        self.reset()

    def reset(self):
        self.raw = -1.0         # -1 on a tick with no committed model action
        self.smoothed = None
        self.scale = 1.0

    def update(self, confidence):
        self.raw = float(confidence)
        if self.smoothed is None:
            self.smoothed = self.raw
            self.scale = speed_scale_for(self.smoothed)
            return self.scale
        self.smoothed += self.alpha * (self.raw - self.smoothed)
        target = speed_scale_for(self.smoothed)
        step = min(MAX_SCALE_STEP, max(-MAX_SCALE_STEP, target - self.scale))
        self.scale += step
        return self.scale


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


def check_steer_head(meta, outputs, where="sidecar"):
    """The steering head's half of the contract -> STEER_REGRESSION or
    STEER_DIRECTION. A sidecar with no steer_head is every model from
    before the direction head, a regression. ValueError when the sidecar
    names a head the runtime does not know, or one the graph does not have.
    """
    head = meta.get("steer_head", STEER_REGRESSION)
    if head not in STEER_HEADS:
        raise ValueError(f"{where}: unknown steer_head {head!r}")
    if steer_head_of({name: None for name in outputs}) != head:
        raise ValueError(f"{where}: steer_head {head!r} but the graph's outputs are "
                         f"{sorted(outputs)}")
    if head == STEER_DIRECTION and tuple(meta.get("steer_dirs", ())) != STEER_DIRS:
        raise ValueError(f"{where}: steer_dirs {meta.get('steer_dirs')} != {list(STEER_DIRS)}")
    return head


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
    # The model's own answer, when it is not `action`: a stop that was
    # masked, or any answer on a tick the goal was reached. "" otherwise.
    model_action: str = ""
    stop_masked: bool = False


def _in(value, low, high):
    return low - OUTPUT_TOLERANCE <= value <= high + OUTPUT_TOLERANCE


def steer_head_of(outputs):
    """Which steering head these outputs come from, by their names. None
    when it is neither or both: a graph that answers both ways is not one
    anything here knows how to read."""
    has_dir = STEER_DIR_OUTPUT in outputs or STEER_MAG_OUTPUT in outputs
    has_reg = STEER_OUTPUT in outputs
    if has_dir == has_reg:
        return None
    return STEER_DIRECTION if has_dir else STEER_REGRESSION


def decode_steer(outputs):
    """Either steering head -> (steer, p_dir, ""), or (None, None, why).

    steer is a fraction of full lock in [-1, 1]. p_dir is the direction
    head's softmax over STEER_DIRS, None for a regression head. The one
    decode for both: the car's read_output and the eval go through here.
    Checked as read_output checks the rest, finite and in range first, and
    never clamped into range.
    """
    head = steer_head_of(outputs)
    if head is None:
        return None, None, "missing"
    try:
        if head == STEER_REGRESSION:
            steer = np.asarray(outputs[STEER_OUTPUT], dtype=np.float64).ravel()
        else:
            dir_logits = np.asarray(outputs[STEER_DIR_OUTPUT], dtype=np.float64).ravel()
            mag = np.asarray(outputs[STEER_MAG_OUTPUT], dtype=np.float64).ravel()
    except (KeyError, TypeError, ValueError):
        return None, None, "missing"
    if head == STEER_REGRESSION:
        if steer.shape != (1,):
            return None, None, "shape"
        if not math.isfinite(steer[0]):
            return None, None, "not finite"
        if not _in(steer[0], -1.0, 1.0):
            return None, None, "steer out of range"
        return float(np.clip(steer[0], -1.0, 1.0)), None, ""
    if dir_logits.shape != (len(STEER_DIRS),) or mag.shape != (1,):
        return None, None, "shape"
    if not all(math.isfinite(v) for v in (*dir_logits, mag[0])):
        return None, None, "not finite"
    if not _in(mag[0], 0.0, 1.0):
        return None, None, "steer out of range"
    e = np.exp(dir_logits - dir_logits.max())
    p = e / e.sum()
    side = STEER_SIDE[int(np.argmax(dir_logits))]
    return side * float(np.clip(mag[0], 0.0, 1.0)), tuple(float(v) for v in p), ""


def steer_dir_p(outputs):
    """The direction head's [left, straight, right] probabilities, or
    NO_STEER_DIR_P for a regression head or an answer decode_steer refuses."""
    _, p, _ = decode_steer(outputs)
    return NO_STEER_DIR_P if p is None else p


def read_output(outputs):
    """Model outputs -> (action, magnitude, steer, rear), or None + why.

    `outputs` maps logits, throttle, either steering head (decode_steer) and
    optionally rear to arrays. isfinite() before anything else touches the
    numbers. Nothing here clamps: a value past its range from a sigmoid or
    tanh head is a broken model, and the answer to a broken model is neutral.
    """
    try:
        logits = np.asarray(outputs["logits"], dtype=np.float64).ravel()
        mag = np.asarray(outputs["throttle"], dtype=np.float64).ravel()
        rear = outputs.get("rear")
        rear = None if rear is None else np.asarray(rear, dtype=np.float64).ravel()
    except (KeyError, TypeError, ValueError):
        return None, "missing"
    if (logits.shape != (len(ACTIONS),) or mag.shape != (1,)
            or (rear is not None and rear.shape != (1,))):
        return None, "shape"
    values = [*logits, mag[0]] + ([] if rear is None else [rear[0]])
    if not all(math.isfinite(v) for v in values):
        return None, "not finite"
    steer, _, why = decode_steer(outputs)
    if steer is None:
        return None, why
    if not _in(mag[0], 0.0, 1.0):
        return None, "throttle out of range"
    if rear is not None and not _in(rear[0], -1.0, 1.0):
        return None, "steer out of range"
    # Inside the tolerance, now known finite, so a clip cannot invent a value.
    return (ACTIONS[int(np.argmax(logits))], float(np.clip(mag[0], 0.0, 1.0)), steer,
            None if rear is None else float(np.clip(rear[0], -1.0, 1.0))), ""


def moving_action(outputs):
    """The better of forward and reverse by the model's action logits.

    Only called on outputs read_output() has already checked, so the logits
    are three finite numbers. A tie is forward, as argmax over all three is.
    """
    logits = np.asarray(outputs["logits"], dtype=np.float64).ravel()
    f, r = logits[ACTIONS.index(FORWARD)], logits[ACTIONS.index(REVERSE)]
    return FORWARD if f >= r else REVERSE


def model_label(meta, path):
    """The name a model drives under: "kblock 1.1" off a registered sidecar.

    tools/models.py writes model_name and model_version into the sidecar of
    every copy it registers ("1.2" for a release, "candidate <tag>" for a
    candidate, so "kblock candidate <tag>"). A sidecar from before has neither,
    and is named by its file instead -- the real file, past the
    models/policy.onnx symlink the service is pointed at, because "policy"
    says nothing about which model drove.
    """
    name = meta.get("model_name")
    version = meta.get("model_version")
    if isinstance(name, str) and name and version not in (None, ""):
        return f"{name} {version}"
    return os.path.splitext(os.path.basename(os.path.realpath(path)))[0]


class PolicyModel:
    """An exported policy: the ONNX session plus the sidecar it came with.

    Keys in the sidecar this does not read are ignored, so a registered copy's
    model_name / model_version and the trainer's provenance fields load the
    same as before.
    """

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
        # And by name, which is what a person reads off a row.
        self.label = model_label(self.meta, path)

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
        self.steer_head = check_steer_head(self.meta, self.output_names, meta_path)
        # (False, False, False) for every model before the IMU input: it is
        # never fed one, and runs exactly as it did.
        self.uses_imu, self.imu_tilt, self.imu_yaw = check_imu(
            self.meta, input_names, meta_path)
        self.path = path

    def run(self, x, prev, mode=None, imu=None):
        feed = {"x": x, "prev": prev}
        if self.has_mode:
            feed[MODE_INPUT] = mode if mode is not None else encode_mode(False)
        if self.uses_imu:
            feed[IMU_INPUT] = (imu if imu is not None
                               else np.zeros((1, pimu.size(self.imu_yaw)), dtype=np.float32))
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

    def __init__(self, controller, oakd, model, downsample=None, goal=None):
        super().__init__(controller, oakd)
        self.model = model
        # Whether the car has arrived: the only thing that lets a model stop
        # be committed (goal.py). Asked every tick.
        self.goal = goal if goal is not None else gl.Goal()
        self.goal_reached = False
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
        self.last_steer_dir_p = NO_STEER_DIR_P  # [left, straight, right]
        # The IMU input. The newest sample and when it was taken, held as
        # footprint.FootprintTap holds them, so the row this driver builds
        # its features off is the row a recording of this tick would carry.
        self.uses_imu = bool(getattr(model, "uses_imu", False))
        # A model with no imu_yaw is a full-IMU one (and one with no IMU
        # input is never fed these at all).
        self.imu_features = pimu.ImuFeatures(tilt=bool(getattr(model, "imu_tilt", False)),
                                             yaw=bool(getattr(model, "imu_yaw", True)))
        self._imu_sample = None
        self._imu_at = 0.0
        self.last_imu = None        # the features fed on the last model run
        # How sure the model is of what it commits, and the speed scale off
        # it. Reset with the latch, whenever auto stops.
        self.confidence = ConfidenceScale()
        # The standoff: when the floor first held the current run of model
        # forwards, and how many decisions it has held. None / 0 otherwise.
        self._standoff_since = None
        self.standoff_ticks = 0
        # The reverse commit: when it started (None while there is none), and
        # the magnitude and front it holds.
        self._commit_since = None
        self._commit_magnitude = 0.0
        self._commit_front = 0.0
        # The last forward this driver drove with a turn in it, and when: the
        # commit's side comes off it.
        self._turn_front = None
        self._turn_at = None
        # What the IMU said this tick, for the commit's rear bumper.
        self._evidence = dv.UNKNOWN

    def abort(self):
        """Driver's abort, and the mode latch forgets the pad: the next tick
        is an engage, and starts at whatever SteerMode is then."""
        super().abort()
        self._pad_seen = None
        self.imu_features.reset()
        self.confidence.reset()
        self._end_commit()
        self._reset_standoff()
        self._turn_front = self._turn_at = None

    @property
    def reverse_commit(self):
        """True while a reverse commit holds the wheels."""
        return self._commit_since is not None

    def _end_commit(self):
        self._commit_since = None

    def _reset_standoff(self):
        self._standoff_since = None
        self.standoff_ticks = 0

    def motion_evidence(self, imu):
        """Driver's, remembered for the tick, so the commit's rear bumper
        counts the sample stuck() already took instead of taking another."""
        self._evidence = super().motion_evidence(imu)
        return self._evidence

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
        self._end_commit()
        self._reset_standoff()
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
            # Which model drove, by name ("kblock 1.1"), and exactly, by hash.
            "policy_model": str(getattr(self.model, "label", "")
                                or getattr(self.model, "sha256", "")),
            "policy_model_sha256": str(getattr(self.model, "sha256", "")),
            # The model's own choice; committed_action is what was done.
            "policy_action": (d.model_action or d.action) if d is not None else "",
            "policy_throttle": float(d.throttle) if d is not None else 0.0,
            "policy_steering_front": float(d.front) if d is not None else 0.0,
            "policy_reason": d.reason if d is not None else "",
            "goal_reached": bool(self.goal_reached),
            "policy_stop_masked": bool(d.stop_masked) if d is not None else False,
            # The committed action's softmax probability (-1 on a tick the
            # model committed nothing), its EMA, and the scale on the throttle.
            "policy_confidence": float(self.confidence.raw),
            "policy_confidence_smoothed": float(self.confidence.smoothed
                                                if self.confidence.smoothed is not None
                                                else -1.0),
            "policy_speed_scale": float(self.confidence.scale),
            "policy_infer_ms": float(self.last_infer_ms),
            # The mode the policy drove with, and the head's probability of
            # that mode on its last answer: low while it wants to switch. -1
            # with no head, which always drives unified.
            "policy_steer_mode": SPLIT if self.mode.split else UNIFIED,
            "policy_steer_mode_p": (-1.0 if self.last_split_p < 0
                                    else self.last_split_p if self.mode.split
                                    else 1.0 - self.last_split_p),
            # Whether a reverse commit held this tick, and how many decisions
            # in a row the floor has held a model forward (0 when it is not).
            "reverse_commit": bool(self.reverse_commit),
            "standoff_ticks": int(self.standoff_ticks),
            # The direction head's [left, straight, right] on its last
            # answer; all -1 for a regression head.
            "policy_steer_dir_p": [float(v) for v in self.last_steer_dir_p],
        }

    def _recover(self, now, reason):
        """The canned backup-and-turn, for STUCK and for a standoff with the
        floor. Every other reverse is the model's own."""
        self._end_commit()
        self._standoff_since = None   # standoff_ticks stays on this row
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
        # Only a tick that commits a model action says how sure it was.
        self.confidence.raw = -1.0
        # First, so every row says it, whichever path the tick takes. A goal
        # that raises is the exception path: neutral, and auto parks.
        was = self.goal_reached
        self.goal_reached = bool(self.goal.reached())
        if self.goal_reached != was:
            print("Goal reached - stopping." if self.goal_reached
                  else "Goal no longer reached - driving on.")
        if self.recovering(now):
            # The recover's reverse, then its settle at 0, as the veto's
            # record of what the wheels were told.
            self._commanded(self.reverse_speed
                            if self._phase == dv._PHASE_REVERSE else 0.0, now)
            self._reset_standoff()
            return
        # My hand on either group ends a commit: the pad wins, and when I let
        # go the model decides from scratch. My throttle also ends a
        # standoff, for the reason it disarms STUCK below.
        if self.human_throttle or self.human_steering:
            self._end_commit()
        if self.human_throttle:
            self._reset_standoff()

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
        self._evidence = dv.UNKNOWN
        if self.human_throttle:
            self._throttle_since = None
            self._still_samples = 0
        elif self.stuck(imu, now):
            self.last_recovery_time = now
            print("STUCK - backing up to recover.")
            self._recover(now, "stuck")
            return
        elif self.reverse_commit and self._rear_bumper_counts(self._evidence):
            # The recover's rear bumper, on the commit's reverse: the gyro
            # says the car stopped moving while told to back up, so it is
            # against something. Off the sample stuck() just took.
            print("Rear bumper - something behind me, cutting the reverse.")
            self._neutral("rear bumper")
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
        self.last_steer_dir_p = steer_dir_p(outputs)
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

        if self.goal_reached:
            self._neutral("goal reached")
            self._note(self.last_decision._replace(model_action=action))
            return
        model_action = action
        stop_masked = action == STOP
        if stop_masked:
            # Not arrived, so not a stop. Only the direction changes; the
            # floor below still holds a forward, as it holds any other.
            action = moving_action(outputs)

        # The reverse commit. One running past REVERSE_COMMIT_S ends, and the
        # model's answer this tick stands. One still running holds reverse,
        # whatever the model said. A reverse after anything but a reverse
        # starts one. 1 ms of slack, as the mode latch has, so seven ticks
        # 0.1 s apart make 0.7.
        if (self.reverse_commit
                and now - self._commit_since >= REVERSE_COMMIT_S - 1e-3):
            self._end_commit()
        starting = False
        if self.reverse_commit:
            action, magnitude = REVERSE, self._commit_magnitude
            steer, rear = self._commit_front / STEER_SCALE, None
        elif (action == REVERSE and self._prev[0] != REVERSE
                and not (self.human_throttle or self.human_steering)):
            starting = True
            self._commit_since = now
            self._commit_magnitude = magnitude
            age = (now - self._turn_at) if self._turn_at is not None else math.inf
            self._commit_front = reverse_commit_front(
                self._turn_front, age, steer * STEER_SCALE)
            steer, rear = self._commit_front / STEER_SCALE, None

        # Every neutral has returned by here, so this is a committed model
        # action. The floor's hold below is a 0 no scale touches, so the
        # scale only ever multiplies a moving throttle.
        scale = self.confidence.update(committed_confidence(outputs, action))
        reason = "model commit" if self.reverse_commit else "model"
        if action == FORWARD and self.floor_veto.vetoes():
            # Blocks the forward, not the decision: the steering still goes
            # out, and next tick the model can choose to reverse.
            throttle, reason = 0.0, "model floor"
            # Under my trigger the floor is holding me, not the model, so
            # it is no standoff (the count was reset above).
            if not self.human_throttle:
                if self._standoff_since is None:
                    self._standoff_since = now
                self.standoff_ticks += 1
                if now - self._standoff_since >= STANDOFF_S - 1e-3:
                    # Neither side is giving way. Back out the way STUCK
                    # does, then the model has the car again.
                    self.last_recovery_time = now
                    print("Standoff with the floor - backing up to recover.")
                    self._recover(now, "standoff")
                    return
        elif action == FORWARD:
            throttle = slowed_throttle(self.forward_speed * magnitude, scale,
                                       MOVING_FLOOR_FORWARD)
        elif action == REVERSE:
            throttle = slowed_throttle(self.reverse_speed * magnitude, scale,
                                       MOVING_FLOOR_REVERSE)
        else:
            throttle = 0.0
        if reason != "model floor":
            self._reset_standoff()

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
            if (abs(front) > REVERSE_TURN_MIN
                    and not (self.human_throttle or self.human_steering)):
                self._turn_front, self._turn_at = front, now
        elif self.reverse_commit and not starting:
            # The commit's reverse is sampled for its rear bumper, as the
            # recover's is. Nothing arms STUCK: _throttle_since stays None.
            pass
        else:
            # Only a forward the driver itself commanded arms STUCK. A
            # commit starts with the windows empty, as begin_recover() does.
            self._clear_motion_evidence()
        self._prev = (action, magnitude, steer)
        self._note(Decision(action, magnitude, steer, rear, reason, throttle, front,
                            model_action if (stop_masked or self.reverse_commit) else "",
                            stop_masked))
