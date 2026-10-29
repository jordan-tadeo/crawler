"""The roof lidar the policy sees. One builder, used by the dataset and the car.

The camera only looks forward, so a model fed depth alone has to guess
what is behind it when it chooses to back up. The roof lidar sees all the
way round. This file is the one place a lidar scan becomes the policy's
`lidar` input, for the same reason policy_input.py is the one place a depth
frame becomes its picture: the dataset tool pairs each /footprint tick with
a /lidar row (scan_at) and encodes the lot with encode_many, and
PolicyDriver encodes the live LidarReader.latest() scan with encode(),
which is encode_many over one row. Nothing else turns lidar ranges into
network inputs.

The scan
--------
bins is crawler/hardware/lidar.py's: BINS (72) nearest ranges in mm from the
middle of the car, 5 degrees each, bin i centred on bearing i * 5
counter-clockwise from straight ahead (0 ahead, 18 left, 36 behind, 54
right), 0 where nothing came back. The /lidar row's `bins` is the same
array, so a recording and the car hand this file the same numbers. LAYOUT
names that, and a sidecar carries it: a model trained on another binning is
refused, never fed this one.

The input
---------
(1, SIZE) float32, in this order:

  ranges    BINS values: min(mm, CLIP_MM) / CLIP_MM where a bin has a
            return, 0 where it has none
  returns   BINS values: 1 where the bin has a return, 0 where it has none,
            so "nothing came back" is never read as "touching"
  present   1 when there is a scan at all, 0 when there is none

No scan -- no --lidar, a reader with nothing newer than STALE_S, a reader
that throws, a recording from before the lidar -- is every value 0,
present included. A scan that saw nothing in any bin is still a scan:
present 1, the rest 0. CLIP_MM is 4 m: the rear stop and the camera both
work inside 2 m, and past 4 m one more metre changes nothing the car does
this second.

Pairing a tick with a scan
--------------------------
On the car a tick reads LidarReader.latest(): the newest whole scan, or
None when it is more than lidar.STALE_S old. A recording has the same scans
as /lidar rows at their own log times, between the ticks; scan_at() is
latest() over them: the newest row at or before the tick, and none when it
is more than STALE_S older.

Mirroring
---------
The trainer's flip augmentation mirrors the depth left to right, so it
mirrors the scan too: bearing b goes to -b, bin i to bin (BINS - i) % BINS.
0 (ahead) and 36 (behind) stay where they are. MIRROR is that permutation,
for the ranges and the returns alike; present does not move.

numpy only: the car imports this from policy.py, and the laptop from the
tools.
"""

import bisect

import numpy as np

from crawler.hardware import lidar as _lidar

LAYOUT = "bins72_5deg"
BINS = _lidar.BINS
BIN_DEG = _lidar.BIN_DEG
CLIP_MM = 4000.0
STALE_S = _lidar.STALE_S

RANGES = slice(0, BINS)
RETURNS = slice(BINS, 2 * BINS)
PRESENT = 2 * BINS
SIZE = 2 * BINS + 1

# Bin i seen in a mirror is bin (BINS - i) % BINS; over the whole input,
# the ranges and the returns each permuted that way and present left alone.
_BIN_MIRROR = (BINS - np.arange(BINS)) % BINS
MIRROR = np.concatenate([_BIN_MIRROR, BINS + _BIN_MIRROR, [PRESENT]]).astype(np.int64)


def encode_many(bins, present):
    """(N, BINS) mm and (N,) present -> (N, SIZE) float32, the model's input.

    A row with present false is all zeros whatever its bins say. Bins that
    are not finite or are under 1 mm read as no return.
    """
    b = np.asarray(bins, dtype=np.float64).reshape(-1, BINS)
    on = np.asarray(present, dtype=bool).reshape(-1)
    if len(on) != len(b):
        raise ValueError(f"{len(b)} scans but {len(on)} present flags")
    ret = np.isfinite(b) & (b >= 1.0) & on[:, None]
    out = np.zeros((len(b), SIZE), dtype=np.float32)
    out[:, RANGES] = np.where(ret, np.minimum(np.where(ret, b, 0.0), CLIP_MM) / CLIP_MM, 0.0)
    out[:, RETURNS] = ret
    out[:, PRESENT] = on
    return out


def encode(bins):
    """One scan's bins (or None, no scan) -> (1, SIZE). encode_many over one
    row, so the car's input is the training input by construction."""
    if bins is None:
        return encode_many(np.zeros((1, BINS)), [False])
    b = np.asarray(bins)
    if b.shape != (BINS,):
        # A scan in some other binning is not one this model knows: no scan.
        return encode_many(np.zeros((1, BINS)), [False])
    return encode_many(b[None], [True])


def missing(n=1):
    """(n, SIZE) zeros: no scan."""
    return np.zeros((n, SIZE), dtype=np.float32)


def live_bins(reader, now=None):
    """The newest fresh scan's bins off a LidarReader, or None. Never raises
    and never waits: no reader, no scan, a stale one (latest() already says
    None past STALE_S) or a reader that throws are all None."""
    if reader is None:
        return None
    try:
        scan = reader.latest() if now is None else reader.latest(now)
        bins = getattr(scan, "bins", None) if scan is not None else None
        if bins is None:
            return None
        bins = np.asarray(bins)
        return bins if bins.shape == (BINS,) else None
    except Exception:
        return None


def scan_at(scan_times_ns, t_ns, max_age_s=STALE_S):
    """The index of the scan a tick at t_ns would have read, or -1.

    scan_times_ns sorted ascending. The newest at or before t_ns, as
    LidarReader.latest() returns the newest whole scan; -1 when there is
    none or it is more than max_age_s old, as latest() returns None.
    """
    i = bisect.bisect_right(scan_times_ns, int(t_ns)) - 1
    if i < 0 or (int(t_ns) - int(scan_times_ns[i])) > max_age_s * 1e9:
        return -1
    return i


def mirror(x):
    """The mirror of a batch of inputs, (N, SIZE): numpy or torch."""
    return x[:, MIRROR]


def meta():
    """The lidar half of a sidecar: what policy.check_lidar reads."""
    return {"uses_lidar": True, "lidar_layout": LAYOUT, "lidar_size": SIZE,
            "lidar_clip_mm": CLIP_MM, "lidar_stale_s": STALE_S}
