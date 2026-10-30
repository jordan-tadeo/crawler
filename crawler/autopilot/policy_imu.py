"""The IMU the policy feels. One builder, used by the dataset and the car.

A learned driver that only sees depth cannot tell "I just hit something"
from "something is close", and cannot tell a slope from a floor. The OAK-D's
IMU can. This file is the one place a stream of IMU readings becomes the
policy's IMU input, for the same reason policy_input.py is the one place a
depth frame becomes its picture: the dataset tool builds the training
features by pushing a session's /footprint rows through ImuFeatures, and
PolicyDriver builds the live ones by pushing the row it would have written
through the very same object. Nothing else turns IMU numbers into features.

The row
-------
row_fields() is the IMU half of a /footprint row (footprint.py calls it):

  imu_accel_x/y/z   linear acceleration, gravity removed, m/s^2
  imu_gyro_x/y/z    raw gyro, rad/s
  imu_up_x/y/z      the unit vector pointing away from the ground, in the
                    IMU's own axes, off GAME_ROTATION_VECTOR (oakd.py). All
                    three 0 when there is no orientation: an OAK-D running
                    an older pipeline, and every recording made before it.
  imu_age_ms        age of the sample at the tick; -1 and all zeros when
                    there has been none

The IMU's axes on this car, measured off the three 0.4/-1.0 recordings
(4565 /footprint rows): x is vertical (gyro_x against front steering while
moving forward, r = 0.73), y is lateral (accel_y against steering, r =
-0.25, the centripetal term), z is along the car (accel_z against the change
in throttle, r = 0.33). MIRROR below is built off that.

The features
------------
FEATURES, in order, one float32 each:

  gyro_x/y/z      the newest sample over GYRO_SCALE
  accel_x/y/z     over ACCEL_SCALE
  accel_peak      the largest |accel| of the last WINDOW decisions, over
                  ACCEL_SCALE: a knock lasts a few tens of milliseconds and
                  one sample a decision catches it on one tick at most, so
                  the peak keeps it in view for half a second
  gyro_peak       the same for |gyro|, over GYRO_SCALE
  valid           1 when the newest sample is a real one no older than
                  MAX_AGE_MS. 0 means every feature above is 0 and means
                  nothing: no IMU yet, or a stalled stream
  up_x/y/z        imu_up, as it is (it is already a unit vector)
  tilt_valid      1 when up is a real reading

Every scaled value is clipped to +-CLIP. An invalid sample does not join the
window and its tick is all zeros, peaks too, so the peaks only ever say what
a real reading said, and only next to one.

A model trained without tilt (every recording up to now) gets up and
tilt_valid zeroed always -- ImuFeatures(tilt=False) -- even on a car that
now records orientation, because inputs it never saw a non-zero value on are
not inputs it knows what to do with. The sidecar says which.

Rows further apart than GAP_S restart the window: the car was not deciding
in between (a recover, a manual stretch, a dropped row), and a peak from
before the gap is not "the last half second".

Without yaw
-----------
gyro_x is yaw on this car, and yaw is what steering does: a model fed it
learns to read back the turn it is being graded on (held out, the steering
gain of a full-IMU model was gone with the gyro zeroed), and that is no help
on the car, where the turn has not happened yet. ImuFeatures(yaw=False)
builds NO_YAW_FEATURES instead: FEATURES without gyro_x, and with gyro_peak
replaced by pitch_roll_peak, the same peak over |(gyro_y, gyro_z)|, because
a peak over all three would carry the yaw back in. What is left is what the
camera cannot see -- a knock, a wheel held by a chair leg, a bump, a tilt.
The sidecar says which list a model takes (imu_features, and imu_yaw false),
and features(yaw) / mirror_of(yaw) / DATASET_KEY[yaw] are the one place each
list's names, mirror and dataset array live. A sidecar without imu_yaw is a
full-IMU model, every one from before this. A car on code from before this
refuses a no-yaw model rather than feed it yaw: its check wants imu_features
to be FEATURES, and the input is a column narrower.

numpy only, no depthai: the car imports this from footprint.py and policy.py,
and the laptop from the tools.
"""

import math

import numpy as np

ROW_ACCEL = ("imu_accel_x", "imu_accel_y", "imu_accel_z")
ROW_GYRO = ("imu_gyro_x", "imu_gyro_y", "imu_gyro_z")
ROW_UP = ("imu_up_x", "imu_up_y", "imu_up_z")
ROW_AGE = "imu_age_ms"
ROW_FIELDS = ROW_ACCEL + ROW_GYRO + ROW_UP + (ROW_AGE,)

FEATURES = ("gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z",
            "accel_peak", "gyro_peak", "valid", "up_x", "up_y", "up_z",
            "tilt_valid")
IMU_SIZE = len(FEATURES)

# The no-yaw list (see "Without yaw"). The up vector and tilt_valid stay the
# last four, as in FEATURES, so without_tilt() is the same for both.
NO_YAW_FEATURES = ("gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z",
                   "accel_peak", "pitch_roll_peak", "valid", "up_x", "up_y",
                   "up_z", "tilt_valid")
NO_YAW_SIZE = len(NO_YAW_FEATURES)
# Which FEATURES column each NO_YAW_FEATURES one is built in (the peak is
# built in gyro_peak's place, over pitch and roll only).
_NO_YAW_COLUMNS = [FEATURES.index("gyro_peak" if n == "pitch_roll_peak" else n)
                   for n in NO_YAW_FEATURES]
# The dataset array each list is stored in.
DATASET_KEY = {True: "imu", False: "imu_no_yaw"}

# The scales, off the same three recordings at 0.4/-1.0 (/footprint rows,
# VERIFIED): |gyro| p99 1.63 rad/s moving forward, 1.46 reversing, 4.2 with
# the car stopped (being handled); |accel| p99 9.4 forward, 11.0 reversing,
# 15.1 stopped. So driving fits inside +-1 and handling inside the clip.
GYRO_SCALE = 2.0     # rad/s
ACCEL_SCALE = 10.0   # m/s^2
CLIP = 3.0

# Half a second of decisions at the 10 Hz tick. A count, as driver.py's IMU
# windows are, so a slow loop can only make it longer in real time.
WINDOW = 5
GAP_S = 0.25
# A sample this old at the tick is a stalled stream, not a reading. The hub
# delivers at ~88 Hz and a recording's imu_age_ms is 0 at p50, 13 ms at p99.
MAX_AGE_MS = 250.0

# Left-right mirror (policy_train's flip augmentation): yaw (gyro_x) and roll
# (gyro_z, about the forward axis) change sign, and so does everything
# lateral (accel_y, up_y). Pitch, vertical and forward do not.
MIRROR = np.array([-1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1], dtype=np.float32)
# Without yaw: the same signs, gyro_x's left out. |(pitch, roll)| has no side.
MIRROR_NO_YAW = MIRROR[_NO_YAW_COLUMNS]


def features(yaw=True):
    """The feature names a model with (yaw=True) or without yaw takes."""
    return FEATURES if yaw else NO_YAW_FEATURES


def size(yaw=True):
    return len(features(yaw))


def mirror_of(yaw=True):
    """The left-right mirror's signs for that list."""
    return MIRROR if yaw else MIRROR_NO_YAW


def _num(value, fallback):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if math.isfinite(value) else fallback


def up_from_quaternion(i, j, k, real):
    """A rotation vector quaternion -> the up unit vector in the device's axes.

    The BNO08x rotation vectors follow Android's: the quaternion turns device
    coordinates into a world frame whose z is up, so up in device coordinates
    is the third row of that rotation matrix. None for the all-zero
    quaternion depthai reports when the sensor has not spoken, or anything
    not finite.
    """
    try:
        x, y, z, w = (float(v) for v in (i, j, k, real))
    except (TypeError, ValueError):
        return None
    n = x * x + y * y + z * z + w * w
    if not math.isfinite(n) or n < 0.5:
        return None
    s = 2.0 / n
    up = (s * (x * z - w * y), s * (y * z + w * x), 1.0 - s * (x * x + y * y))
    norm = math.sqrt(sum(v * v for v in up))
    if not math.isfinite(norm) or norm < 1e-6:
        return None
    return tuple(v / norm for v in up)


def row_fields(sample, sampled_at, now):
    """The IMU part of a /footprint row: a get_imu_sample() dict, flattened.

    Zeros and an age of -1 when there has been no sample, so every row has
    the same columns whether or not the IMU has spoken yet. "up" is optional
    in the sample; without it the three up fields are 0.
    """
    out = dict.fromkeys(ROW_FIELDS, 0.0)
    out[ROW_AGE] = -1.0
    if not sample:
        return out
    try:
        accel = [float(v) for v in sample["accel"]]
        gyro = [float(v) for v in sample["gyro"]]
    except Exception:
        return out
    if len(accel) != 3 or len(gyro) != 3:
        return out
    for key, value in zip(ROW_ACCEL + ROW_GYRO, accel + gyro):
        out[key] = _num(value, 0.0)
    up = sample.get("up") if hasattr(sample, "get") else None
    try:
        up = [float(v) for v in up] if up is not None else None
    except (TypeError, ValueError):
        up = None
    if up is not None and len(up) == 3 and all(math.isfinite(v) for v in up):
        for key, value in zip(ROW_UP, up):
            out[key] = value
    out[ROW_AGE] = _num((now - sampled_at) * 1000.0, -1.0)
    return out


def _clip(value):
    return max(-CLIP, min(CLIP, value))


def up_of(row):
    """A /footprint row's up vector, unit length, or None: the up and
    tilt_valid columns ImuFeatures puts in, by the same rule (a fresh,
    finite sample, and an up within 10% of unit length). Stateless, so the
    lower band's pitch (column_profile.floor_pitch) can be read off a row on
    the car exactly as the dataset reads it off its imu array."""
    get = row.get if hasattr(row, "get") else (lambda k, d=None: d)
    age = _num(get(ROW_AGE), -1.0)
    readings = [_num(get(k), float("nan")) for k in ROW_ACCEL + ROW_GYRO]
    if not (0.0 <= age <= MAX_AGE_MS and all(math.isfinite(v) for v in readings)):
        return None
    up = [_num(get(k), 0.0) for k in ROW_UP]
    norm = math.sqrt(sum(v * v for v in up))
    if not 0.9 <= norm <= 1.1:
        return None
    return tuple(v / norm for v in up)


class ImuFeatures:
    """IMU rows in, one decision at a time; the policy's IMU input out.

    push(row, t_s) takes a mapping with ROW_FIELDS (missing keys read as no
    reading, which is how a recording from before a field existed reads)
    and the decision's time in seconds, and returns (size(yaw),) float32:
    FEATURES, or NO_YAW_FEATURES with yaw=False.
    """

    def __init__(self, tilt=True, yaw=True, window=WINDOW, gap_s=GAP_S):
        self.tilt = bool(tilt)
        self.yaw = bool(yaw)
        self.window = int(window)
        self.gap_s = float(gap_s)
        self.reset()

    def reset(self):
        self._accel = []
        self._gyro = []
        self._last_t = None

    def push(self, row, t_s):
        t_s = float(t_s)
        if self._last_t is not None and not 0.0 <= t_s - self._last_t <= self.gap_s:
            self.reset()
        self._last_t = t_s
        out = np.zeros(IMU_SIZE, dtype=np.float32)
        get = row.get if hasattr(row, "get") else (lambda k, d=None: d)

        age = _num(get(ROW_AGE), -1.0)
        accel = [_num(get(k), float("nan")) for k in ROW_ACCEL]
        gyro = [_num(get(k), float("nan")) for k in ROW_GYRO]
        valid = (0.0 <= age <= MAX_AGE_MS
                 and all(math.isfinite(v) for v in accel + gyro))
        if valid:
            self._accel.append(math.sqrt(sum(v * v for v in accel)))
            # Without yaw the peak never sees gyro_x.
            self._gyro.append(math.sqrt(sum(v * v for v in (gyro if self.yaw
                                                            else gyro[1:]))))
            del self._accel[:-self.window], self._gyro[:-self.window]
            out[0:3] = [_clip(v / GYRO_SCALE) for v in gyro]
            out[3:6] = [_clip(v / ACCEL_SCALE) for v in accel]
            out[6] = _clip(max(self._accel) / ACCEL_SCALE)
            out[7] = _clip(max(self._gyro) / GYRO_SCALE)
            out[8] = 1.0

        up = up_of(row) if valid else None
        if up is not None:
            out[9:12] = up
            out[12] = 1.0
        if not self.tilt:
            out = without_tilt(out)
        return out if self.yaw else out[_NO_YAW_COLUMNS]


# The up vector and tilt_valid: the last four columns of either list.
_TILT = slice(FEATURES.index("up_x") - IMU_SIZE, None)
assert NO_YAW_FEATURES[_TILT] == FEATURES[_TILT]


def without_tilt(features):
    """Features, (..., IMU_SIZE) or (..., NO_YAW_SIZE), with up and
    tilt_valid zeroed: what a model trained without orientation is fed.
    ImuFeatures(tilt=False) is this; the trainer applies it to a dataset
    built with tilt."""
    out = np.array(features, dtype=np.float32, copy=True)
    out[..., _TILT] = 0.0
    return out


def mirror(feats, yaw=True):
    """Features of the left-right mirror image of the same moment."""
    return np.asarray(feats, dtype=np.float32) * mirror_of(yaw)
