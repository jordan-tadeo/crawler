"""The footprint probes, widened to a column profile a model can read.

driver.free_by_column() cuts the car's footprint into 9 probes and asks each
one whether it is inside a budget. This asks the same band of rows -- the
envelope footprint_geometry() derives from the camera height and the roof
line -- a softer question, in more columns, across the whole frame: for each
of PROFILE_COLUMNS columns, how near is the nearest valid return, what is the
median, and how much of it is measurable at all.

It runs on the model's own downsampled frame (policy.INPUT_HEIGHT x
INPUT_WIDTH), not the raw one, so the recording, the dataset tool and the
car all compute it from the one picture and none of them can disagree about
what a column saw. Pure and numpy only: the recording side can import it
without the policy, and the policy imports it without the recording.

Whole width, not the footprint span. At 80 pixels the 400 mm footprint is
about 42 of them, and the half of the frame outside it is where the room to
turn away is -- which is the half read_the_gradient() goes looking in.
"""

import numpy as np

from crawler.autopilot import driver as dv

PROFILE_COLUMNS = 32
# nearest, median, valid fraction -- in that order, one row each.
PROFILE_FEATURES = 3
PROFILE_SCALE_MM = 3000.0


def profile_rows(shape):
    """The row band, off the same geometry the footprint probes use."""
    geo = dv.footprint_geometry(shape)
    return geo.row_top, geo.row_bottom


def column_profile(small_mm, columns=PROFILE_COLUMNS):
    """(H, W) uint16 mm, 0 invalid -> (PROFILE_FEATURES, columns) float32.

    Distances are scaled by PROFILE_SCALE_MM and clipped to 1. A column with
    nothing valid in it reports 1.0 (far) for both distances and 0 for the
    valid fraction: the fraction is the evidence, and the model learns what a
    blind column means from it rather than from a distance that pretends.
    """
    small = np.asarray(small_mm)
    top, bottom = profile_rows(small.shape)
    return _distance_stats(small, top, bottom, columns)


def _band_cells(small, top, bottom, columns, scale=None):
    """Rows [top, bottom) cut into columns -> (cells, valid count, sizes).

    cells is (columns, rows * widest column), invalid and padding both nan,
    so one nanmin and one nanmedian do every column at once. A per-column
    Python loop measured 7.4 ms a frame on the Pi; this is the same numbers.
    scale, when given, is (rows,) and each row's readings are mapped through
    it (see lowband_profile) before they are cut.
    """
    index, pad = _layout(small.shape[1], columns)
    band = small[top:bottom].astype(np.float32)
    band[band <= 0] = np.nan
    if scale is not None:
        band = scale(band)
    cells = band[:, index]                       # (rows, columns, widest)
    cells[:, pad] = np.nan
    cells = cells.transpose(1, 0, 2).reshape(columns, -1)
    sizes = (~pad).sum(axis=1) * band.shape[0]
    valid = np.count_nonzero(~np.isnan(cells), axis=1)
    return cells, valid, sizes


def _distance_stats(small, top, bottom, columns):
    cells, valid, sizes = _band_cells(small, top, bottom, columns)
    out = np.ones((PROFILE_FEATURES, columns), dtype=np.float32)
    seen = valid > 0
    if seen.any():
        with np.errstate(all="ignore"):
            out[0, seen] = np.minimum(1.0, np.nanmin(cells[seen], axis=1) / PROFILE_SCALE_MM)
            out[1, seen] = np.minimum(1.0, np.nanmedian(cells[seen], axis=1) / PROFILE_SCALE_MM)
    out[2] = valid / np.maximum(1, sizes)
    return out


# ---- the lower band -----------------------------------------------------------
#
# The bottom third of the frame, as a second profile beside the footprint's:
# the same three numbers a column, off rows the footprint band mostly does
# not reach. Level, at 165 mm, those rows are the floor from about 0.37 to
# 1.07 m ahead (the median over my drives is within 5% of that), so what
# they say is what is ON the floor in front of the wheels: a rock, a step
# down, a lip the footprint band looks over.
#
# Two ways to say it, named by the sidecar's lowband:
#
#   LOWBAND_PLAIN   nearest, median and valid, exactly as column_profile()
#   LOWBAND_FLOOR   the same rows against the depth a flat floor would read
#                   there: (reading - floor) / floor per pixel, clipped to
#                   +-1, then the column's lowest and its median, and valid.
#                   Negative is something standing on the floor, positive
#                   a drop. The floor's depth comes off the camera height
#                   and the pitch: a pinhole tilted down by p sees the floor
#                   at depth CAMERA_HEIGHT_MM / (dy cos p + sin p), dy the
#                   row's offset over fy, the same for every column -- so
#                   the mirror image is the same floor, and the mirror only
#                   flips the column order. The pitch is the camera IMU's
#                   (floor_pitch()); with no up vector it is 0, level. Over
#                   my drives the IMU's pitch halves the floor's error on
#                   the 7% of ticks tilted past 2.3 degrees and changes
#                   nothing level.
#
# A blind column is as column_profile's: valid 0 and the distances at the
# "nothing there" end, 1.0 for the plain band and 0 deviation for the floor.
# LOWBAND_BLANK is what a whole band of nothing reads, which is what the
# scorecard's lowband_use feeds to see how much a model leans on it.
LOWBAND_PLAIN = "lower_third"
LOWBAND_FLOOR = "lower_third_floor"
LOWBAND_LAYOUTS = (LOWBAND_PLAIN, LOWBAND_FLOOR)
LOWBAND_FRACTION = 1.0 / 3.0
LOWBAND_FAR = {LOWBAND_PLAIN: 1.0, LOWBAND_FLOOR: 0.0}


def lowband_rows(shape):
    """The bottom third of the frame: rows [round(2h/3), h)."""
    h = int(shape[0])
    return int(round(h * (1.0 - LOWBAND_FRACTION))), h


def floor_pitch(up_z, valid=True):
    """The camera IMU's up vector's z -> the camera's pitch, radians, nose
    down positive; 0 (level) when there is no reading. The device's z is
    along the lens, so a lens tipped down has up leaning back: up_z < 0."""
    try:
        z = float(up_z)
    except (TypeError, ValueError):
        return 0.0
    if not valid or not np.isfinite(z):
        return 0.0
    return float(-np.arcsin(np.clip(z, -1.0, 1.0)))


def floor_depth_mm(shape, pitch=0.0, rows=None):
    """The depth a flat floor reads on each row of a frame this shape, at
    this pitch; inf on a row that looks at or above the horizon."""
    h = int(shape[0])
    top, bottom = rows if rows is not None else (0, h)
    fy = dv.FY_AT_400 * h / 400.0
    dy = (np.arange(top, bottom, dtype=np.float64) + 0.5 - h / 2.0) / fy
    den = dy * np.cos(pitch) + np.sin(pitch)
    with np.errstate(divide="ignore"):
        return np.where(den > 1e-6, dv.CAMERA_HEIGHT_MM / den, np.inf)


def lowband_profile(small_mm, layout=LOWBAND_PLAIN, pitch=0.0, columns=PROFILE_COLUMNS):
    """(H, W) uint16 mm -> (PROFILE_FEATURES, columns) float32, the bottom
    third of the frame in `layout`. pitch only matters to LOWBAND_FLOOR."""
    small = np.asarray(small_mm)
    top, bottom = lowband_rows(small.shape)
    if layout == LOWBAND_PLAIN:
        return _distance_stats(small, top, bottom, columns)
    if layout != LOWBAND_FLOOR:
        raise ValueError(f"unknown lowband {layout!r}")
    floor = floor_depth_mm(small.shape, float(pitch), (top, bottom))
    level = np.isfinite(floor)
    safe = np.where(level, floor, 1.0).astype(np.float32)[:, None]

    def deviation(band):
        # A row at or above the horizon has no floor in it: anything it
        # reads is nearer than the floor, -1. An invalid reading stays nan.
        d = np.clip((band - safe) / safe, -1.0, 1.0)
        return np.where(level[:, None] | np.isnan(band), d, np.float32(-1.0))

    cells, valid, sizes = _band_cells(small, top, bottom, columns, deviation)
    out = np.zeros((PROFILE_FEATURES, columns), dtype=np.float32)
    seen = valid > 0
    if seen.any():
        with np.errstate(all="ignore"):
            out[0, seen] = np.nanmin(cells[seen], axis=1)
            out[1, seen] = np.nanmedian(cells[seen], axis=1)
    out[2] = valid / np.maximum(1, sizes)
    return out


def lowband_blank(layout, columns=PROFILE_COLUMNS):
    """A lower band with nothing in it: every column blind."""
    out = np.full((PROFILE_FEATURES, columns), LOWBAND_FAR[layout], dtype=np.float32)
    out[2] = 0.0
    return out


def profile_input(small_mm, lowband=None, pitch=0.0):
    """The profile model's x for one frame, without the batch axis: the
    footprint band, (3, columns), with the lower band under it, (6,
    columns), when the model has one."""
    band = column_profile(small_mm)
    if lowband is None:
        return band
    return np.concatenate([band, lowband_profile(small_mm, lowband, pitch)])


_LAYOUTS = {}


def _layout(width, columns):
    """Column index table for a frame width: (columns, widest), plus its padding."""
    key = (width, columns)
    if key not in _LAYOUTS:
        edges = np.linspace(0, width, columns + 1).round().astype(int)
        widest = max(1, int(np.max(np.diff(edges))))
        index = np.zeros((columns, widest), dtype=np.intp)
        pad = np.ones((columns, widest), dtype=bool)
        for i in range(columns):
            span = np.arange(edges[i], edges[i + 1])
            index[i, :span.size] = span
            pad[i, :span.size] = False
        _LAYOUTS[key] = (index, pad)
    return _LAYOUTS[key]
