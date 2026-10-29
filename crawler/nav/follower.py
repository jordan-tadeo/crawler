"""Following a planned path: pure pursuit for a counter-phase four-wheel steer.

Every decision (10 Hz, the autopilot's AUTO_TICK_INTERVAL) it takes the pose
fresh and asks one question: which circle through the truck, tangent to its
heading, passes through the point LOOKAHEAD metres further down the path?
That circle's curvature is the steering. The truck's reference point is the
lidar, which on this truck sits close to the midway point between the axles,
and with the rear steered as the mirror of the front that point drives the
circle with no sideslip -- so pure pursuit's geometry holds without a
correction for where the axles are. Reverse is the same circle, driven the
other way round: the target is behind, and the curvature comes out of the
same formula.

Why pure pursuit and not Stanley: Stanley wants the front axle's position and
a front-steer model, and on a car whose rear steers as hard as its front both
are wrong. Pure pursuit only needs the circle, and it degrades gently: an
axle that turns 60% of what it is told makes the truck run wide, the next
decision sees the truck outside the path and asks for more.

The learned gain. Terrain changes how much of a steering command reaches the
ground, and the truck can measure that itself: over each decision, the
curvature it actually drove (the change in yaw over the distance moved)
against the one it commanded. GainEstimate keeps a slow average of the ratio
and the command is divided by it, so a truck on carpet that turns 70% of its
command is soon asking for 1/0.7 of what it wants. Clamped (GAIN_MIN,
GAIN_MAX), and only updated on turns big enough to measure.

The lookahead grows with speed (LOOKAHEAD_S of travel past LOOKAHEAD_MIN_M),
which is what absorbs servo lag: the 0.1-0.3 s a servo takes to get there is
a few centimetres of travel the lookahead is already past.
"""

import math
from typing import NamedTuple

import numpy as np

from crawler.nav import truck as tk

LOOKAHEAD_MIN_M = 0.35
LOOKAHEAD_S = 0.6
LOOKAHEAD_REVERSE_M = 0.3
# The search for the nearest path point runs forward from the last one, no
# further than this, so a path that doubles back is never jumped across.
SEARCH_AHEAD_M = 1.0
# A cusp (where the path changes direction) is reached when the truck is this
# close to it, or has run past it.
CUSP_TOL_M = 0.12
# The end of the path.
GOAL_TOL_M = 0.2

GAIN_MIN = 0.4
GAIN_MAX = 1.3
GAIN_ALPHA = 0.15
GAIN_MIN_CURVATURE = 0.35   # of MAX_CURVATURE: smaller turns are not measured
GAIN_MIN_TRAVEL_M = 0.02

# Speed, as a fraction of the forward cap: down on a tight turn, down near the
# end or a cusp, never below MIN_SPEED_FRACTION while moving.
TURN_SLOWDOWN = 0.5
APPROACH_M = 0.6
MIN_SPEED_FRACTION = 0.5


class Command(NamedTuple):
    """One decision. direction +1, -1 or 0 (stop); speed 0..1 of the cap in
    that direction; curvature 1/m to the left. `done` is the end of the
    path; `cusp` is a stop to change direction."""
    direction: int
    speed: float
    curvature: float
    done: bool = False
    cusp: bool = False
    index: int = 0
    cross_track: float = 0.0
    lookahead: float = 0.0


class GainEstimate:
    """The ratio of driven curvature to commanded, averaged. Starts at 1.

    update() measures the decision just finished against what remember()
    was told went out for it, and returns the gain to use now."""

    def __init__(self, alpha=GAIN_ALPHA, lo=GAIN_MIN, hi=GAIN_MAX):
        self.alpha, self.lo, self.hi = alpha, lo, hi
        self.gain = 1.0
        self._last = None   # (x, y, yaw, curvature sent, direction)

    def reset(self):
        """Forget the last decision (a stop, a cusp, a new path). The gain
        itself is kept: the ground has not changed."""
        self._last = None

    def update(self, pose):
        last = self._last
        if last is None or last[4] == 0:
            return self.gain
        x, y, yaw = pose[:3]
        lx, ly, lyaw, k_sent, d = last
        travel = math.hypot(x - lx, y - ly)
        if travel < GAIN_MIN_TRAVEL_M:
            return self.gain
        if abs(k_sent) < GAIN_MIN_CURVATURE * tk.MAX_CURVATURE:
            return self.gain
        dyaw = (yaw - lyaw + math.pi) % (2 * math.pi) - math.pi
        ratio = (dyaw / (travel * d)) / k_sent
        if not math.isfinite(ratio):
            return self.gain
        ratio = min(max(ratio, self.lo), self.hi)
        self.gain += self.alpha * (ratio - self.gain)
        self.gain = min(max(self.gain, self.lo), self.hi)
        return self.gain

    def remember(self, pose, sent, direction):
        self._last = (pose[0], pose[1], pose[2], float(sent), int(direction))


class Follower:
    """Pure pursuit along one Path, a segment at a time."""

    def __init__(self, path=None, adaptive=True, lookahead_min=LOOKAHEAD_MIN_M,
                 lookahead_s=LOOKAHEAD_S, goal_tol=GOAL_TOL_M):
        self.adaptive = adaptive
        self.lookahead_min = lookahead_min
        self.lookahead_s = lookahead_s
        self.goal_tol = goal_tol
        self.gain = GainEstimate()
        self.set_path(path)

    def set_path(self, path):
        self.path = path
        self.index = 0
        self.segment = 0
        self._segments = path.segments() if path is not None else []

    @property
    def remaining_m(self):
        if self.path is None:
            return 0.0
        return max(0, len(self.path.poses) - 1 - self.index) * 0.05

    def step(self, pose, speed_m_s=0.0, cap_m_s=0.42):
        """The Command for this decision off the pose, (x, y, yaw)."""
        path = self.path
        if path is None or not self._segments:
            return Command(0, 0.0, 0.0, done=True)
        x, y, yaw = (float(v) for v in pose[:3])
        a, b = self._segments[self.segment]
        poses = path.poses
        direction = int(path.dirs[min(a + 1, len(path.dirs) - 1)])
        last_seg = self.segment == len(self._segments) - 1

        # Nearest point, searched forward from where it was.
        i0 = max(self.index, a)
        i1 = min(b, i0 + int(SEARCH_AHEAD_M / 0.05) + 1)
        seg = poses[i0:i1]
        d2 = (seg[:, 0] - x) ** 2 + (seg[:, 1] - y) ** 2
        k = int(np.argmin(d2))
        self.index = i0 + k
        cross = math.sqrt(float(d2[k]))

        end = poses[b - 1]
        to_end = math.hypot(end[0] - x, end[1] - y)
        # Past the end of the segment along its own last heading?
        ex, ey = math.cos(end[2]), math.sin(end[2])
        along = ((x - end[0]) * ex + (y - end[1]) * ey) * direction
        tol = self.goal_tol if last_seg else CUSP_TOL_M
        if to_end <= tol or (along > 0 and to_end < 3 * tol) or self.index >= b - 1:
            if last_seg:
                self.gain.reset()
                return Command(0, 0.0, 0.0, done=True, index=self.index,
                               cross_track=cross)
            self.segment += 1
            self.index = self._segments[self.segment][0]
            self.gain.reset()
            return Command(0, 0.0, 0.0, cusp=True, index=self.index,
                           cross_track=cross)

        if direction > 0:
            look = self.lookahead_min + self.lookahead_s * max(speed_m_s, 0.0)
        else:
            look = LOOKAHEAD_REVERSE_M
        # The target: the first point at least `look` away, else the end,
        # pushed out along the end heading so the circle still reaches it.
        tail = poses[self.index:b]
        dd = np.hypot(tail[:, 0] - x, tail[:, 1] - y)
        far = np.nonzero(dd >= look)[0]
        if len(far):
            tx, ty = tail[far[0], 0], tail[far[0], 1]
        else:
            ext = look - to_end
            tx = end[0] + ex * direction * max(ext, 0.0)
            ty = end[1] + ey * direction * max(ext, 0.0)
        c, s = math.cos(yaw), math.sin(yaw)
        lx = c * (tx - x) + s * (ty - y)
        ly = -s * (tx - x) + c * (ty - y)
        d2t = lx * lx + ly * ly
        kappa = 2.0 * ly / d2t if d2t > 1e-9 else 0.0
        # A target on the wrong side of the truck (behind on a forward
        # segment): full lock toward it, which is what the circle formula
        # also gives once it is beside the truck.
        if lx * direction < 0 and abs(ly) < 1e-6:
            kappa = tk.MAX_CURVATURE
        gain = self.gain.update((x, y, yaw)) if self.adaptive else 1.0
        k_out = float(np.clip(kappa / gain, -tk.MAX_CURVATURE, tk.MAX_CURVATURE))
        self.gain.remember((x, y, yaw), k_out, direction)

        frac = 1.0 - TURN_SLOWDOWN * min(abs(kappa) / tk.MAX_CURVATURE, 1.0)
        frac = min(frac, max(to_end / APPROACH_M, 0.0))
        speed = max(frac, MIN_SPEED_FRACTION)
        return Command(direction, float(speed), k_out, index=self.index,
                       cross_track=cross, lookahead=look)
