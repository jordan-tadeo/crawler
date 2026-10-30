"""One revolution of the roof lidar as the scan matcher wants it.

The input is the 720-slot range array /lidar/scan carries
(telemetry/lidar_view.scan_slots): slot i is bearing i * 0.5 degrees
counter-clockwise from the arrow on the lid, in metres, NaN for nothing. The
same array the live reader makes and every recording holds, so the replay and
the car run identical input through identical code.

Frames: the car's, x forward, y left, counter-clockwise bearings, origin at
the lidar (lidar.py). The arrow points at car bearing -90 (lidar.MOUNT).

Timing inside a revolution. The sensor sweeps clockwise from above, raw angle
phi 0 -> 360, and the revolution closes on the wrap. Slot bearing b is raw
phi = 360 - b, so slot b was measured b / 360 of a revolution before the
scan's end time. That is all de-skew needs.
"""

import numpy as np

from crawler.hardware import lidar as ld
from crawler.telemetry import lidar_view as lv

SLOTS = lv.SCAN_SLOTS
SLOT_B = np.radians(np.arange(SLOTS) * lv.SCAN_STEP_DEG)
CAR_B = SLOT_B + np.radians(ld.MOUNT.yaw_deg)
FRAC_BEFORE_END = (np.arange(SLOTS) * lv.SCAN_STEP_DEG) / 360.0

# The truck's own body, as a box round the lidar in the car frame, metres:
# the measured half width over the wheels (lidar.HALF_WIDTH_MM, 135.89),
# the measured rear bumper (-247.65) and the taped front one (+263),
# each with BODY_MARGIN_M on it. A return inside it is the truck or
# something touching it, and is neither matched nor written. A circle of
# 0.32 m used to stand in for this, and it was blind to anything within
# 18 cm of the wheels abeam. On 045557, 0.28% of returns fall in the box.
BODY_MARGIN_M = 0.04
BODY_FRONT_M = ld.FRONT_BUMPER_X_MM / 1000.0 + BODY_MARGIN_M
BODY_REAR_M = ld.REAR_BUMPER_X_MM / 1000.0 - BODY_MARGIN_M
BODY_HALF_W_M = float(ld.HALF_WIDTH_MM or 150.0) / 1000.0 + BODY_MARGIN_M
# Nearer than this is inside the sensor's own blind spot whatever the box.
MIN_R = 0.05
# Further than this the points are too sparse and noisy to match a line
# through.
MAX_R = 10.0
_COS_B = np.cos(CAR_B)
_SIN_B = np.sin(CAR_B)

# A point gets a normal only if its four angular neighbours are close by and
# the five lie on a line within about 1.2 cm. Corners, edges and clutter get
# none and are not matched.
LINE_FIT_M = 0.012

# A scan with fewer usable points than this is not matched at all.
MIN_POINTS = 40


def valid_mask(ranges):
    """bool (720,): a usable return, in range and outside the body box."""
    r = np.asarray(ranges, np.float64)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(r) & (r > MIN_R) & (r < MAX_R)
        x = r * _COS_B
        y = r * _SIN_B
        body = ((x > BODY_REAR_M) & (x < BODY_FRONT_M)
                & (np.abs(y) < BODY_HALF_W_M))
    return ok & ~body


def all_points(ranges):
    """Every usable return as car-frame points (n, 2) and its slot idx: what
    is written to the map, line or no line, so legs and clutter are there."""
    idx = np.nonzero(valid_mask(ranges))[0]
    r = np.asarray(ranges, np.float64)[idx]
    return np.stack([r * _COS_B[idx], r * _SIN_B[idx]], 1), idx


def scan_points(ranges):
    """720 ranges -> car-frame points (n, 2), unit normals (n, 2), slot idx.

    Only points with a good local line, so every point has a normal: what is
    matched.
    """
    ranges = np.asarray(ranges, np.float64)
    idx = np.nonzero(valid_mask(ranges))[0]
    r = ranges[idx]
    b = CAR_B[idx]
    p = np.stack([r * np.cos(b), r * np.sin(b)], 1)
    if len(p) < 20:
        return p[:0], p[:0], idx[:0]
    nb = [np.roll(p, k, 0) for k in (-2, -1, 1, 2)]
    d = np.stack([np.linalg.norm(q - p, axis=1) for q in nb], 1)
    good = (d < np.maximum(0.06, 0.03 * r)[:, None]
            * np.array([2, 1, 1, 2])).all(1)
    pts = np.stack([p] + nb, 1)
    c = pts - pts.mean(1, keepdims=True)
    sxx = (c[..., 0] ** 2).sum(1)
    syy = (c[..., 1] ** 2).sum(1)
    sxy = (c[..., 0] * c[..., 1]).sum(1)
    ang = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    nrm = np.stack([-np.sin(ang), np.cos(ang)], 1)
    lam_min = 0.5 * (sxx + syy) - np.sqrt(0.25 * (sxx - syy) ** 2 + sxy ** 2)
    good &= lam_min / 5 < LINE_FIT_M ** 2
    return p[good], nrm[good], idx[good]


def valid_returns(ranges):
    """How many slots hold a usable return at all, before the line test."""
    return int(np.count_nonzero(valid_mask(ranges)))


def deskew(p, idx, period, omega, v):
    """Move each point into the car frame at the scan's END.

    `omega` is the yaw rate (rad/s, CCW +) and `v` the car-frame velocity
    (m/s, (2,)) over the revolution, both taken constant across it. A point
    measured tau seconds before the end is where it was then, seen from a
    car that has since turned omega * tau and moved v * tau; this undoes
    that. Returns the moved points and each one's rotation angle.
    """
    tau = -FRAC_BEFORE_END[idx] * period
    th = omega * tau
    c, s = np.cos(th), np.sin(th)
    x = c * p[:, 0] - s * p[:, 1] + v[0] * tau
    y = s * p[:, 0] + c * p[:, 1] + v[1] * tau
    return np.stack([x, y], 1), th


# ---- the ground --------------------------------------------------------------

# The scan plane's height above the ground under the truck, metres. The
# lidar is on the cab roof; 0.30 is an ESTIMATE, not measured.
LIDAR_HEIGHT_M = 0.30

# A point is "maybe ground" once the tilted scan plane, followed out along
# its bearing, has dropped this much of LIDAR_HEIGHT_M. Less than all of it
# because the tilt comes from the IMU on the front axle, not the cab, and
# the two disagree by a few degrees on rocks. See ground_mask().
GROUND_FRACTION = 0.6


def plane_drop(points, up):
    """How far below the lidar each car-frame point is, metres, if the scan
    plane is tilted as the gravity `up` vector says (device axes).

    The IMU's up vector is in the OAK-D's axes: +x points DOWN when the
    truck is level (up = (-1, 0, 0)), +z forward and +y left (the quality
    pass's reading of it; it called the signs weakly supported, the
    magnitudes solid). So the up vector's car-frame components are
    forward = up_z, left = up_y, and a point at (x, y) in the scan plane sits
    x * up_z + y * up_y above the lidar. Positive return = below.
    """
    ux, uy, uz = (float(v) for v in up)
    n = (ux * ux + uy * uy + uz * uz) ** 0.5
    if not n > 0.5:
        return np.zeros(len(points))
    return -(points[:, 0] * uz + points[:, 1] * uy) / n


def ground_mask(points, up, height=LIDAR_HEIGHT_M, fraction=GROUND_FRACTION):
    """True for the points that could be the scan plane hitting the ground.

    On flat ground with the truck tilted by t, the plane meets the ground
    height / sin(t) out along the downhill bearing, and that line looks like
    a wall. Two things make the exact line useless as a filter: the tilt is
    the axle's, not the cab's, and the sign convention is only weakly
    known. So this is deliberately blunt: any point the tilted plane puts
    more than fraction * height below the lidar, on EITHER side (|drop|, so a
    sign error cannot turn the guard off), is not written. It is still
    matched against: the guard only keeps it out of the map.

    At 6 degrees, the write gate, that is points past 1.7 m straight down
    the slope; at 3 degrees past 3.4 m; level, nothing inside the 8 m
    writing range.
    """
    if up is None:
        return np.zeros(len(points), bool)
    return np.abs(plane_drop(points, up)) > fraction * height
