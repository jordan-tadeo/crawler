"""The OAK-D's IMU as the mapper reads it: heading rate about gravity, and tilt.

Fed one sample at a time, in time order, as the camera hub drains them
(about 88 a second live, 17-20 a second in a recording's /imu). Asked, once
a scan, how far the truck turned between two times and how tilted it is.

Yaw rate is the gyro projected on the up vector, gyro . up, in the device's
own axes. That is turn about gravity whatever the truck is sitting on, and
it is the sign and the projection the prototype measured against the scan
matcher on four drives (yaw RMS under a degree per 10 s window).

Tilt is the angle of the up vector from the device's level up, (-1, 0, 0).

The IMU is on the camera on the FRONT AXLE and the lidar is on the cab, so
everything here is the axle's attitude standing in for the cab's: on rocks
the two differ by however much the suspension is twisted.
"""

import math
from collections import deque

import numpy as np

LEVEL_UP = (-1.0, 0.0, 0.0)

# Ten seconds of samples is far more than any one scan asks about.
KEEP_S = 10.0

# A gap in the samples longer than this and there is no rate over it.
MAX_GAP_S = 0.3


def tilt_deg(up):
    """Degrees between `up` (device axes) and level."""
    ux, uy, uz = (float(v) for v in up)
    n = math.sqrt(ux * ux + uy * uy + uz * uz)
    if not n > 0.5:
        return float("nan")
    return math.degrees(math.acos(max(-1.0, min(1.0, -ux / n))))


class ImuTrack:
    """Integrated heading and the newest attitude, over the last KEEP_S."""

    def __init__(self):
        self._t = deque()
        self._cum = deque()
        self._rate = 0.0
        self.up = None
        self.up_t = None
        self.samples = 0
        self._arr = None

    def add(self, t, gyro, up=None):
        """One sample: time (monotonic s), gyro rad/s (3,), up (3,) or None.

        A sample with no up vector projects the gyro on the last one seen,
        or on level if there has never been one.
        """
        t = float(t)
        if up is not None:
            ux, uy, uz = (float(v) for v in up)
            n = math.sqrt(ux * ux + uy * uy + uz * uz)
            if n > 0.5 and math.isfinite(n):
                self.up = (ux / n, uy / n, uz / n)
                self.up_t = t
        u = self.up or LEVEL_UP
        try:
            rate = sum(float(g) * c for g, c in zip(gyro, u))
        except (TypeError, ValueError):
            return
        if not math.isfinite(rate):
            return
        if self._t:
            t_prev = self._t[-1]
            if t <= t_prev:
                return
            dt = t - t_prev
            cum = self._cum[-1] + (0.5 * (rate + self._rate) * dt
                                   if dt <= MAX_GAP_S else 0.0)
        else:
            cum = 0.0
        self._t.append(t)
        self._cum.append(cum)
        self._rate = rate
        self.samples += 1
        self._arr = None
        while self._t and t - self._t[0] > KEEP_S:
            self._t.popleft()
            self._cum.popleft()

    def _arrays(self):
        if self._arr is None:
            self._arr = (np.fromiter(self._t, float, len(self._t)),
                         np.fromiter(self._cum, float, len(self._cum)))
        return self._arr

    def covers(self, t0, t1):
        """True if samples span [t0, t1] with no gap over MAX_GAP_S at the
        ends. Inside, a long gap integrates as no turn."""
        if len(self._t) < 2:
            return False
        return (self._t[0] <= t0 + MAX_GAP_S and self._t[-1] >= t1 - MAX_GAP_S)

    def _yaw_at(self, x):
        """Integrated heading at time x. Past the newest sample it carries on
        at the newest rate rather than stopping: a scan always ends after the
        last sample that has arrived (by up to a whole sample interval, 50 ms
        in a 20 Hz recording), and holding the heading there drops that turn
        from the prior. Measured on 050853: holding it took the heading error
        per 10 s window from 0.71 to 2.97 degrees."""
        t, cum = self._arrays()
        if x > t[-1]:
            return float(cum[-1] + self._rate * (x - t[-1]))
        return float(np.interp(x, t, cum))

    def yaw_between(self, t0, t1):
        """Radians turned CCW from t0 to t1, or None if not covered."""
        if not self.covers(t0, t1):
            return None
        return self._yaw_at(t1) - self._yaw_at(t0)

    def rate_between(self, t0, t1, bias=0.0):
        """Mean yaw rate on [t0, t1], rad/s, less `bias`, or None."""
        if t1 - t0 <= 1e-3:
            if not self._t or t1 - self._t[-1] > MAX_GAP_S:
                return None
            return self._rate - bias
        d = self.yaw_between(t0, t1)
        if d is None:
            return None
        return d / (t1 - t0) - bias

    def pitch_roll(self, now=None, max_age=1.0):
        """(|pitch|, |roll|) degrees from the newest up vector, NaN if none.

        In the device axes +z is forward and +y left (scan.plane_drop reads
        them the same way), so pitch is asin(up_z) and roll asin(up_y).
        Absolute values, because the signs are only weakly supported."""
        if self.up is None or (now is not None and now - self.up_t > max_age):
            return float("nan"), float("nan")
        _, uy, uz = self.up
        return (abs(math.degrees(math.asin(max(-1.0, min(1.0, uz))))),
                abs(math.degrees(math.asin(max(-1.0, min(1.0, uy))))))

    def tilt(self, now=None, max_age=1.0):
        """Degrees off level from the newest up vector, NaN if none (or none
        within max_age of `now`)."""
        if self.up is None:
            return float("nan")
        if now is not None and now - self.up_t > max_age:
            return float("nan")
        return tilt_deg(self.up)
