"""The truck as the planner sees it: its size, how it steers, how fast it goes.

Size, read off the modules that measured or estimated it first:

  half-width     lidar.HALF_WIDTH_MM, 135.89 mm over the wheels, MEASURED
  rear bumper    lidar.REAR_BUMPER_X_MM, 247.65 mm behind the lidar, MEASURED
  front bumper   lidar.FRONT_BUMPER_X_MM, 270 mm ahead of it, ESTIMATED

The pose is the lidar's. On a counter-phase four-wheel steer with the rear
the mirror of the front, the point midway between the axles moves along the
turning circle with no sideslip, and the lidar sits close to that point (the
bumpers are 248 mm behind it and 270 mm ahead), so the lidar is the point the
curvature is about. A path curvature k is the inverse of the radius of the
circle that point drives, positive to the left, and it is the same circle
forward and in reverse: the steering picks the circle, the throttle picks
which way round it.

How it steers: MEASURED off four drives in the house (2026-10-29), the
lidar SLAM's track against the steering command, cross-checked on the IMU.
Path curvature against the front command u (rear = -u, counter-phase):

  k(u) = B + GL*min(u, 0) + GR*max(u, 0)
           + HL*min(u + KNEE, 0) + HR*max(u - KNEE, 0)

which is near linear to |u| 0.35 and flatter past it. Two things in it the
code's own estimate (driver.TURN_RADIUS_MM, 590 mm at 0.6) did not have:
right turns are about 14% wider than left (k(+0.6) = -1.33 against k(-0.6)
= +1.56, so 0.75 m and 0.64 m), and at u = 0 the truck drifts left on a
10 m radius (B). front_for_curvature() inverts the measured curve, so both
come out in the wash: a straight path is asked for with the small right
trim that drives straight, and a right turn gets more lock than the same
left one. MAX_CURVATURE is the weaker side's lock at max_turn, so a plan
never asks for a left turn the truck could not mirror to the right.

How much of that the ground gives back varies, by surface and by speed: over
0.5 s steady turns the ratio of driven to predicted curvature is 0.74-1.31
(p05-p95) forward and wider in reverse. The follower learns it as it goes.

How fast: MEASURED, held-throttle samples off the same drives. Forward is
2.28 m/s per unit of throttle past 0.151, so 0.58 m/s at the 0.4 cap, and
nothing at all under about 0.15. Reverse at -0.6 (driver.ESCAPE_SPEED, where
every autonomous reverse goes out) is 0.255 m/s.
"""

import math

import numpy as np

from crawler.hardware import lidar as ld

HALF_WIDTH_M = ld.HALF_WIDTH_MM / 1000.0
FRONT_M = ld.FRONT_BUMPER_X_MM / 1000.0
REAR_M = ld.REAR_BUMPER_X_MM / 1000.0      # negative: behind the lidar
LENGTH_M = FRONT_M - REAR_M

# The measured steering curve, u positive is a right turn, k positive left.
CURVE_B = 0.039
CURVE_GL = -2.843
CURVE_GR = -2.550
CURVE_HL = 0.736
CURVE_HR = 0.639
CURVE_KNEE = 0.35

# The servo limit the autopilot steers under (driver.Driver.max_turn).
MAX_TURN = 0.6


def curvature_for_front(front):
    """The measured path curvature, 1/m left positive, for a front command
    (rear = -front). Vectorised."""
    u = np.asarray(front, float)
    k = (CURVE_B + CURVE_GL * np.minimum(u, 0.0) + CURVE_GR * np.maximum(u, 0.0)
         + CURVE_HL * np.minimum(u + CURVE_KNEE, 0.0)
         + CURVE_HR * np.maximum(u - CURVE_KNEE, 0.0))
    return k if k.ndim else float(k)


# The tightest circle either way at max_turn: the right one, the wider.
MAX_CURVATURE = float(min(curvature_for_front(-MAX_TURN),
                          -curvature_for_front(MAX_TURN)))
TURN_RADIUS_M = 1.0 / MAX_CURVATURE

# The inverse, as a table: k(u) falls monotonically in u.
_U = np.linspace(-1.0, 1.0, 2001)
_K = curvature_for_front(_U)


def front_for_curvature(curvature, max_turn=MAX_TURN):
    """The front servo command for a path curvature, in [-max_turn, max_turn].

    The measured curve inverted, so positive (left) curvature comes out as a
    negative (left) command, and 0 comes out as the small right trim that
    drives straight. Past what max_turn reaches it is max_turn. A nan stays a
    nan, as everywhere else on the way to set_steering(), which turns it into
    neutral on both channels.
    """
    k = float(curvature)
    if not math.isfinite(k):
        return float("nan")
    # np.interp wants increasing x: k falls as u rises, so read it backwards.
    u = float(np.interp(k, _K[::-1], _U[::-1]))
    return float(np.clip(u, -max_turn, max_turn))


# The footprint as three circles down the centre line: the rectangle cut into
# three equal lengths, each covered by the circle through its corners. The
# planner checks these against the distance field, which is three lookups
# where a rectangle would be a polygon test.
_SEG = LENGTH_M / 3.0
CIRCLE_X_M = tuple(REAR_M + _SEG * (i + 0.5) for i in range(3))
CIRCLE_R_M = math.hypot(HALF_WIDTH_M, _SEG / 2.0)

# Speed, MEASURED (see above).
FORWARD_M_S_PER_THROTTLE = 2.28
FORWARD_START_THROTTLE = 0.151
REVERSE_TABLE = ((0.40, 0.0), (0.45, 0.15), (0.496, 0.182), (0.60, 0.256),
                 (1.00, 0.392))


def ground_speed(throttle):
    """Signed ground speed, m/s, for a throttle on flat floor."""
    t = float(throttle)
    if not math.isfinite(t):
        return 0.0
    if t >= 0.0:
        return max(0.0, FORWARD_M_S_PER_THROTTLE * (t - FORWARD_START_THROTTLE))
    a = -t
    if a < REVERSE_TABLE[0][0]:
        return 0.0
    xs = [p[0] for p in REVERSE_TABLE]
    vs = [p[1] for p in REVERSE_TABLE]
    return -float(np.interp(a, xs, vs))


def footprint_corners(x, y, yaw):
    """The truck's outline at a pose, map metres, (4, 2), counter-clockwise."""
    c, s = math.cos(yaw), math.sin(yaw)
    local = ((FRONT_M, HALF_WIDTH_M), (REAR_M, HALF_WIDTH_M),
             (REAR_M, -HALF_WIDTH_M), (FRONT_M, -HALF_WIDTH_M))
    return np.array([(x + c * px - s * py, y + s * px + c * py)
                     for px, py in local])


def footprint_points(x, y, yaw, step=0.025):
    """Points filling the truck's outline at a pose, map metres, (n, 2).

    Spaced at most `step` apart, edges included, so a check against a grid
    of that resolution cannot step over an occupied cell."""
    nx = max(2, int(math.ceil(LENGTH_M / step)) + 1)
    ny = max(2, int(math.ceil(2 * HALF_WIDTH_M / step)) + 1)
    px, py = np.meshgrid(np.linspace(REAR_M, FRONT_M, nx),
                         np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, ny),
                         indexing="ij")
    px, py = px.ravel(), py.ravel()
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([x + c * px - s * py, y + s * px + c * py], 1)
