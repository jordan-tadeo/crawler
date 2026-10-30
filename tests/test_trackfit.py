"""A saved track's walls fitted onto a fresh map: mapping/trackfit.py.

Synthetic houses where the answer is known, the ways it must refuse (a
room that looks the same turned round, a mirror image, somewhere else, too
little seen), the thread, and the owner's real tracks from 10-30 when the
case files are on this machine (tools/trackfit_eval.py extract).
"""

import math
import os
import time

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler.mapping import grid as gr  # noqa: E402
from crawler.mapping import runner as rn  # noqa: E402
from crawler.mapping import trackfit as tf  # noqa: E402

RES = 0.1


def segments_points(segs, step=0.05):
    pts = []
    for a, b in segs:
        n = int(math.hypot(b[0] - a[0], b[1] - a[1]) / step) + 1
        pts.append(np.linspace(a, b, n))
    return np.concatenate(pts)


# An L-shaped flat with furniture: no turn but the true one fits it.
HOUSE = [((0, 0), (8, 0)), ((8, 0), (8, 4)), ((8, 4), (4, 4)), ((4, 4), (4, 7)),
         ((4, 7), (0, 7)), ((0, 7), (0, 0)), ((2, 1), (3, 1)), ((3, 1), (3, 2.5)),
         ((6, 2), (6.5, 3)), ((1, 5), (2, 5.5)), ((4, 2), (5, 2))]
# A bare box with a cupboard dead centre: the same turned half round.
BOX = [((0, 0), (6, 0)), ((6, 0), (6, 4)), ((6, 4), (0, 4)), ((0, 4), (0, 0)),
       ((2.5, 1.5), (3.5, 1.5)), ((2.5, 2.5), (3.5, 2.5))]


def view_of(segs, pose=(0.0, 0.0, 0.0), inside=None, seen=None):
    """A MapView of the walls through `pose`, the floor inside `inside`
    (a function of the untransformed point) free, and only what `seen`
    (likewise) says seen."""
    walls = segments_points(segs)
    lo, hi = walls.min(0), walls.max(0)
    g = np.mgrid[lo[0]:hi[0]:0.04, lo[1]:hi[1]:0.04].reshape(2, -1).T
    free = g[[inside(p) for p in g]] if inside is not None else g
    if seen is not None:
        walls = walls[[seen(p) for p in walls]]
        free = free[[seen(p) for p in free]]
    w, f = tf.transform(pose, walls), tf.transform(pose, free)
    both = np.concatenate([w, f])
    x0, y0 = both.min(0) - 1.0
    nx, ny = (np.ceil((both.max(0) + 1.0 - (x0, y0)) / RES)).astype(int)
    codes = np.zeros((nx, ny), np.uint8)
    i, j = np.floor((f - (x0, y0)) / RES).astype(int).T
    codes[i, j] = gr.FREE
    i, j = np.floor((w - (x0, y0)) / RES).astype(int).T
    codes[i, j] = gr.OCCUPIED
    return rn.MapView(t=0.0, version=0, x0=float(x0), y0=float(y0), res=RES, codes=codes)


def in_house(p):
    return not (p[0] > 4 and p[1] > 4)


def saved_walls(segs, inside=None):
    return tf.occupied_points(view_of(segs, inside=inside))


def err_of(fit, truth, pts):
    return float(np.max(np.hypot(*(tf.transform(fit, pts) - tf.transform(truth, pts)).T)))


PROBE = np.array([[1.0, 1.0], [7.0, 1.0], [2.0, 6.0]])


@pytest.mark.parametrize("truth", [(0.0, 0.0, 0.0), (math.radians(37), 5.0, -3.0),
                                   (math.radians(-120), -12.0, 7.5),
                                   (math.radians(179), 0.4, 0.2)])
def test_a_house_turned_and_moved_is_found_to_a_few_centimetres(truth):
    walls = saved_walls(HOUSE, in_house)
    fit = tf.fit(walls, view_of(HOUSE, truth, inside=in_house))
    assert fit.ok, fit.why
    assert abs(math.degrees(tf._wrap(fit.yaw - truth[0]))) < 1.5
    assert err_of(fit, truth, PROBE) < 0.15


def test_part_of_the_house_seen_is_enough_once_it_is_enough():
    truth = (math.radians(70), 2.0, 1.0)
    walls = saved_walls(HOUSE, in_house)
    # The bottom half of the L: two walls and a box, already one answer.
    half = view_of(HOUSE, truth, inside=in_house, seen=lambda p: p[1] < 3.5)
    fit = tf.fit(walls, half)
    assert fit.ok, fit.why
    assert err_of(fit, truth, PROBE) < 0.2
    # A corner of it is not.
    corner = view_of(HOUSE, truth, inside=in_house, seen=lambda p: p[0] < 1.5 and p[1] < 1.5)
    fit = tf.fit(walls, corner)
    assert fit is None or not fit.ok


def test_a_room_that_looks_the_same_turned_round_is_not_placed():
    walls = saved_walls(BOX)
    fit = tf.fit(walls, view_of(BOX, (math.radians(30), 1.0, 2.0)))
    assert not fit.ok and "two places" in fit.why


def test_a_mirror_image_of_the_house_is_not_placed():
    walls = saved_walls(HOUSE, in_house)
    flipped = [((a[0], -a[1]), (b[0], -b[1])) for a, b in HOUSE]
    fit = tf.fit(walls, view_of(flipped, inside=lambda p: in_house((p[0], -p[1]))))
    assert not fit.ok


def test_somewhere_else_is_not_placed():
    walls = saved_walls(HOUSE, in_house)
    rng = np.random.default_rng(3)
    other = [((0, 0), (9, 0)), ((9, 0), (9, 6)), ((9, 6), (0, 6)), ((0, 6), (0, 0))]
    other += [((float(x), float(y)), (float(x) + 0.8, float(y) + float(d)))
              for x, y, d in zip(rng.uniform(1, 7, 6), rng.uniform(1, 5, 6),
                                 rng.uniform(-0.6, 0.6, 6))]
    fit = tf.fit(walls, view_of(other))
    assert not fit.ok


def test_nothing_to_fit_is_none():
    assert tf.fit([], view_of(HOUSE)) is None
    assert tf.fit(saved_walls(HOUSE), None) is None


def test_transform_and_compose_agree():
    a, b = (0.3, 1.0, -2.0), (-1.1, 0.5, 4.0)
    p = np.array([[1.0, 2.0], [-3.0, 0.5]])
    assert np.allclose(tf.transform(tf.compose(a, b), p), tf.transform(a, tf.transform(b, p)))


def test_the_fitter_runs_on_its_thread_and_hands_back_the_tag():
    walls = saved_walls(HOUSE, in_house)
    f = tf.Fitter()
    assert f.submit("one", walls, view_of(HOUSE, (0.5, 1.0, 1.0), inside=in_house))
    assert not f.submit("two", walls, None), "one job at a time"
    deadline = time.monotonic() + 30.0
    while f.busy:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    tag, fit = f.result()
    assert tag == "one" and fit.ok
    assert f.result() is None


# --- the owner's tracks, 10-30 ---------------------------------------------------------

def _case_files():
    try:
        import trackfit_eval as te
        d = te.data_dir()
        return sorted(os.path.join(d, n) for n in os.listdir(d) if n.endswith(".npz"))
    except Exception:
        return []


@pytest.mark.skipif(not _case_files(), reason="no trackfit case files on this machine "
                    "(tools/trackfit_eval.py extract)")
def test_the_real_tracks_land_where_the_whole_next_run_says_and_never_on_a_mirror():
    # Every 4th map of each run, to keep it to seconds: the whole set is
    # tools/trackfit_eval.py evaluate (95 of 174 placed, worst 0.11 m).
    import trackfit_eval as te
    accepted = 0
    for path in _case_files():
        for t, f, ok, mean, worst, mirror_ok, fit in te.evaluate(path, every=4):
            assert not mirror_ok, (path, t)
            if ok:
                accepted += 1
                assert worst < 0.25, (path, t, worst)
    assert accepted >= 15
