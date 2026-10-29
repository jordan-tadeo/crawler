"""When a revolution really ended, from timestamps that arrive late.

A scan's host time is time.monotonic() when the serial read carrying its
last packet returned. That read is 1024 bytes at the sensor's byte rate,
about 52 ms, so the host time lags the true end by anything from 0 to ~52
ms, and that jitter goes straight into de-skew and the IMU yaw prior.

The sensor stamps every packet with its own millisecond clock (u16, wraps
at 30000). Its ticks are evenly spaced; its zero is unknown. So the offset
host - sensor is measured on every scan and the SMALLEST seen in the last
WINDOW_S is the estimate: the smallest lag is the read that returned right
after its packet, and every other read only adds delay on top. The fitted
time is sensor + that offset.

SensorClock is the live one. HostDejitter does the same lower-envelope trick
with only host times and the revolution period, for recordings made before
the sensor's stamp was recorded.
"""

from collections import deque

STAMP_WRAP_MS = 30000

# Long enough to always hold a read that returned promptly, short enough
# that two crystals drifting by 100 ppm move the offset by 1 ms in it.
WINDOW_S = 10.0


class SensorClock:
    """sensor ms stamps + host seconds -> de-jittered host seconds."""

    def __init__(self, window_s=WINDOW_S):
        self.window_s = window_s
        self._last_stamp = None
        self._sensor_s = 0.0
        self._last_host = None
        self._offsets = deque()   # (host, offset), offsets increasing
        self.resets = 0

    def reset(self):
        self._last_stamp = None
        self._offsets.clear()
        self.resets += 1

    def fit(self, host, stamp_ms):
        """The revolution's end in host seconds. `stamp_ms` None -> host."""
        if stamp_ms is None:
            return host
        stamp_ms = int(stamp_ms) % STAMP_WRAP_MS
        if self._last_stamp is not None:
            d_ms = (stamp_ms - self._last_stamp) % STAMP_WRAP_MS
            d_host = host - self._last_host
            # A stamp that disagrees with the host by more than a quarter of
            # a second is a sensor that restarted or a stamp I misread.
            if d_host < 0 or abs(d_ms / 1000.0 - d_host) > 0.25:
                self.reset()
            else:
                self._sensor_s += d_ms / 1000.0
        if self._last_stamp is None:
            self._sensor_s = 0.0
        self._last_stamp = stamp_ms
        self._last_host = host
        off = host - self._sensor_s
        q = self._offsets
        while q and q[-1][1] >= off:
            q.pop()
        q.append((host, off))
        while q and host - q[0][0] > self.window_s:
            q.popleft()
        return self._sensor_s + q[0][1]


class HostDejitter:
    """Host times alone, for recordings with no sensor stamp.

    Each revolution's end is predicted from the last fitted end plus a whole
    number of periods (a lost revolution is two), and the fit is whichever
    is earlier, that prediction or the host time, plus a small creep so a
    period estimated a little short cannot drag the fit away from the host
    times for ever.
    """

    CREEP_S = 0.0005

    def __init__(self, period=0.1009):
        self.period = period
        self._fit = None
        self._host = None

    def fit(self, host, period=None):
        p = period or self.period
        if self._fit is None or host - self._host > 1.0 or host < self._host:
            self._fit, self._host = host, host
            return host
        n = max(1, int(round((host - self._host) / p)))
        pred = self._fit + n * p + self.CREEP_S
        self._fit = min(host, pred)
        self._host = host
        return self._fit
