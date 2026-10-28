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
    index, pad = _layout(small.shape[1], columns)

    # (columns, rows * widest column), invalid and padding both nan, so one
    # nanmin and one nanmedian do every column at once. A per-column Python
    # loop measured 7.4 ms a frame on the Pi; this is the same numbers.
    band = small[top:bottom].astype(np.float32)
    band[band <= 0] = np.nan
    cells = band[:, index]                       # (rows, columns, widest)
    cells[:, pad] = np.nan
    cells = cells.transpose(1, 0, 2).reshape(columns, -1)
    sizes = (~pad).sum(axis=1) * band.shape[0]
    valid = np.count_nonzero(~np.isnan(cells), axis=1)

    out = np.ones((PROFILE_FEATURES, columns), dtype=np.float32)
    seen = valid > 0
    if seen.any():
        with np.errstate(all="ignore"):
            out[0, seen] = np.minimum(1.0, np.nanmin(cells[seen], axis=1) / PROFILE_SCALE_MM)
            out[1, seen] = np.minimum(1.0, np.nanmedian(cells[seen], axis=1) / PROFILE_SCALE_MM)
    out[2] = valid / np.maximum(1, sizes)
    return out


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
