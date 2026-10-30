"""A shelved map is only merged back where the links say it is.

After a long loss SLAM starts a fresh map and shelves the old one, and
searches the old one for the truck while it drives on the fresh one
(slam.py, "Never losing the map"). The links every reseed carries
(Result.link: the gyro's heading and the commanded speed since the last good
fit) say roughly where the fresh map sits on the old one; the search is drawn
round that and only a heading within their doubt is merged. Before, the
search was drawn where the truck had frozen, every heading, and in 174641
(2026-10-29, 250 s) it merged the room's 180 degree twin.

A bare rectangle is its own twin turned round its centre, so it is the room
here: a pose and its twin fit it equally well.
"""

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler.mapping import reloc as rl  # noqa: E402
from crawler.mapping import slam as sl  # noqa: E402
from crawler.mapping.imu import ImuTrack  # noqa: E402

from test_mapping import LEVEL, PERIOD, scan_at  # noqa: E402

# 6 x 4 m, nothing in it: symmetric about the origin.
BOX = [((-3.0, -2.0), (3.0, -2.0)), ((3.0, -2.0), (3.0, 2.0)),
       ((3.0, 2.0), (-3.0, 2.0)), ((-3.0, 2.0), (-3.0, -2.0))]

START = np.array([-1.8, -0.3, 0.0])
V_BLIND = 0.35          # m/s, straight on while it is lost
BLIND_S = 12.0
T_LOST = 4.0


def twin(pose):
    return np.array([-pose[0], -pose[1], sl._wrap(pose[2] + math.pi)])


def blind_path(t):
    """Standing at START to T_LOST, then BLIND_S straight on at V_BLIND,
    then standing where that left it: 4.2 m on, out of reach of a search
    drawn where it froze, while its twin is 0.7 m from there."""
    s = min(max(t - T_LOST, 0.0), BLIND_S)
    return START + np.array([V_BLIND * s, 0.0, 0.0])


def clutter(pose, rng):
    """Short walls close round the truck, a new set every scan, hiding the
    box: nothing that fits the map, or the scan before."""
    segs = []
    for _ in range(40):
        cx = pose[0] + rng.uniform(-1.5, 1.5)
        cy = pose[1] + rng.uniform(-1.5, 1.5)
        a, L = rng.uniform(0.0, math.pi), rng.uniform(0.3, 0.8)
        segs.append(((cx, cy), (cx + L * math.cos(a), cy + L * math.sin(a))))
    return segs


def lost_in_the_box(gyro_gap=False, after_s=30.0, slip_deg=0.0):
    """Results and true poses, scan by scan, up to the merge or after_s past
    the blind stretch. The gyro reads no turn throughout (none is made);
    with `gyro_gap` it goes quiet for a second while the truck is lost.
    With `slip_deg` the last scans before the blind stretch are seen from a
    truck turned 8 degrees more each scan, up to slip_deg, that never
    turned: fits that are good, and wrong, on the way to getting lost."""
    rng = np.random.default_rng(3)
    imu = ImuTrack()
    slam = sl.Slam(imu=imu)
    out, truth = [], []
    k = 0
    while True:
        k += 1
        t = k * PERIOD
        quiet = gyro_gap and T_LOST + 3.0 < t <= T_LOST + 4.0
        if not quiet:
            for j in range(10):
                imu.add(t - PERIOD + (j + 1) * 0.01, (0.0, 0.0, 0.0), LEVEL)
        blind = T_LOST < t <= T_LOST + BLIND_S
        pose = blind_path(t)
        segs = clutter(pose, rng) if blind else BOX
        seen = blind_path
        slip = min(slip_deg, 8.0 * round((t - (T_LOST - 0.4)) / PERIOD))
        if slip > 0.0 and not blind and t <= T_LOST + 1e-9:
            def seen(tt, d=math.radians(slip)):
                return blind_path(tt) + np.array([0.0, 0.0, d])
        res = slam.step(t, scan_at(t, seen, segs),
                        speed=V_BLIND if blind else 0.0)
        out.append(res)
        truth.append(pose)
        if res.reseed == "merged" or t > T_LOST + BLIND_S + after_s:
            return slam, out, truth


def yaw_off(a, b):
    return abs(math.degrees(sl._wrap(a - b)))


def test_the_twin_is_not_merged_and_the_truck_is_found_where_it_is():
    slam, out, truth = lost_in_the_box()
    why = [r.reseed for r in out if r.reseed]
    assert "lost" in why, why
    assert why[-1] == "merged", "it found the shelved map again"
    r = out[-1]
    p = on_map(truth[-1], START)
    assert yaw_off(r.yaw, p[2]) < 3.0, "it merged the twin"
    assert math.hypot(r.x - p[0], r.y - p[1]) < 0.15


def test_with_the_gyro_in_doubt_it_still_merges():
    # A second's gap in the gyro while it is lost: the link owns up to not
    # knowing the heading at all, the gate opens to the whole circle, and
    # the shelf is still found.
    slam, out, truth = lost_in_the_box(gyro_gap=True)
    lost = next(r for r in out if r.reseed == "lost")
    assert lost.link[4] == pytest.approx(math.pi)
    assert out[-1].reseed == "merged"


def test_good_fits_the_gyro_never_saw_do_not_turn_the_reckoning():
    # Before the loss, three good fits turn the truck 24 degrees the gyro
    # says it never turned. The reckoning the shelf is held to goes on from
    # before them, so the truck is merged where it is: reckoned from the
    # last of them, the true heading is out of the gate and the twin is too.
    # The fits are held to the gyro now (slam.py GYRO_LOST_DEG): the truck is
    # lost before the third, and not found again on their heading either.
    slam, out, truth = lost_in_the_box(slip_deg=24.0)
    k = next(i for i, r in enumerate(out) if r.reseed == "lost")
    assert slam.gyro_lost == 1
    assert max(yaw_off(r.yaw, 0.0) for r in out[:k] if r.written) \
        <= sl.GYRO_LOST_DEG, "written on the slipped heading"
    assert out[-1].reseed == "merged", "it found the shelved map again"
    p = on_map(truth[-1], START)
    assert yaw_off(out[-1].yaw, p[2]) < 3.0
    assert math.hypot(out[-1].x - p[0], out[-1].y - p[1]) < 0.15


def shelved_at(pose):
    """A Slam that mapped the box from `pose`, then shelved it and started a
    fresh map in the same frame, with the links saying exactly that, and
    its shelf search drawn. (slam, scan points at `pose`)."""
    imu = ImuTrack()
    slam = sl.Slam(imu=imu)
    t = 0.0
    for k in range(30):
        t = (k + 1) * PERIOD
        for j in range(10):
            imu.add(t - PERIOD + (j + 1) * 0.01, (0.0, 0.0, 0.0), LEVEL)
        slam.step(t, scan_at(t, lambda tt: pose, BOX))
    slam._start_over(keep=True)
    sh = slam._shelf
    sh.here = np.array(slam.pose, float)
    sh.err_m, sh.err_yaw, sh.since = 0.2, 0.03, t
    p, nrm, _ = sl.sc.scan_points(scan_at(t, lambda tt: pose, BOX))
    assert slam._shelf_piece(sh, p, nrm, t) is None and sh.search is not None
    return slam, p, nrm, t


def on_map(world, start):
    """A world pose on the map a Slam started at `start` drew (its frame is
    where it began)."""
    xy = sl.mt.rot(-start[2]) @ (np.asarray(world[:2]) - start[:2])
    return np.array([xy[0], xy[1], sl._wrap(world[2] - start[2])])


def test_a_twin_that_fits_better_is_still_never_merged():
    pose = np.array([-1.0, -0.3, 0.2])
    slam, p, nrm, t = shelved_at(pose)
    sh = slam._shelf

    here, there = on_map(pose, pose), on_map(twin(pose), pose)

    def circle(fits):
        sh.search.fits = [(i, sl._between(f, slam.pose)) for i, f in fits]
        return slam._shelf_pick(sh, p, nrm, t)
    # The twin scores a clear margin ahead: alone it is the one answer.
    fits = [(0.95, tuple(there)), (0.85, tuple(here))]
    assert not circle(fits)
    assert circle(fits), "the second circle the same way merges"
    assert yaw_off(slam.pose[2], here[2]) < 2.0, "it merged the twin"
    assert math.hypot(*(slam.pose[:2] - here[:2])) < 0.1


def wall_on(grid, a, b):
    """A wall from a to b written straight into `grid`'s cells."""
    s = np.linspace(0.0, 1.0, 200)[:, None]
    xy = np.asarray(a) + s * (np.asarray(b) - np.asarray(a))
    ix, iy = grid.world_to_cell(xy[:, 0], xy[:, 1])
    grid.lo[ix, iy] = sl.gr.L_OCC + 20


def test_a_merge_the_walls_do_not_bear_out_is_not_made():
    # The scan fits the shelf right where the truck is reckoned, at the
    # reckoned heading, twice. But the fresh map has a wall across what the
    # shelf knows is open floor: that is somewhere else that looks alike
    # from here, and it is not merged. With its walls on the shelf's, it is.
    pose = np.array([-1.0, -0.3, 0.2])
    here = on_map(pose, pose)

    def two_circles(walls):
        slam, p, nrm, t = shelved_at(pose)
        sh = slam._shelf
        for a, b in walls:
            wall_on(slam.grid, on_map(a, pose), on_map(b, pose))
        merged = False
        for _ in range(2):
            sh.search.fits = [(0.9, sl._between(tuple(here), slam.pose))]
            merged = slam._shelf_pick(sh, p, nrm, t)
        return slam, merged

    slam, merged = two_circles([((0.5, -1.5, 0), (0.5, 1.5, 0)),
                                ((-2.0, 0.8, 0), (1.5, 0.8, 0))])
    assert not merged and slam.shelf_refused == 1
    share, n = sl.agreement(slam._shelf.grid, slam.grid, (0.0, 0.0, 0.0))
    assert n >= sl.SHELF_AGREE_MIN and share < 0.2
    slam, merged = two_circles([((3.0, -2.0, 0), (3.0, 2.0, 0)),
                                ((-3.0, 2.0, 0), (3.0, 2.0, 0))])
    assert merged and slam.shelf_refused == 0


def test_the_heading_doubt_grows_with_the_links_and_the_time():
    sh = sl.Shelf(None, 0)
    slam = sl.Slam()
    assert slam._shelf_tol(sh, 0.0) == math.pi, "nothing reckoned, no gate"
    sh.here, sh.err_yaw, sh.since = np.zeros(3), math.radians(2.0), 10.0
    at0 = math.degrees(slam._shelf_tol(sh, 10.0))
    at100 = math.degrees(slam._shelf_tol(sh, 110.0))
    assert at0 == pytest.approx(2.0 + sl.SHELF_DRIFT_DEG)
    assert at100 == pytest.approx(at0 + 100 * sl.SHELF_DRIFT_DEG_S)
    sh.err_yaw = math.pi
    assert slam._shelf_tol(sh, 10.0) == math.pi


def test_a_search_with_a_span_tries_only_the_headings_in_it():
    full = rl.yaw_order(0.0)
    assert len(full) == 90
    some = rl.yaw_order(0.0, span=math.radians(14.0))
    offs = sorted(round(math.degrees(y)) for y in some)
    assert offs == [-16, -12, -8, -4, 0, 4, 8, 12, 16]
    assert rl.yaw_order(0.0, span=math.pi) == full


def test_the_shelf_reckons_the_truck_whatever_the_fresh_map_does():
    # On the shelf the truck heads 90 degrees; the fresh map has it at 0
    # (its frame is turned) and says it moved 1 m along its x: that is 1 m
    # along the shelf's y. Then a fresh-map frame turned round (a twin
    # relocalization, a young map) changes nothing, and a blind second at
    # 0.5 m/s goes on along the gyro's heading.
    slam = sl.Slam()
    slam._shelf = sh = sl.Shelf(None, 0)
    sh.here, sh.err_m, sh.err_yaw = np.array([2.0, 3.0, math.pi / 2]), 0.1, 0.02
    slam._reckon_shelf(0.0, None, 0.1, moved=(1.0, 0.0), fresh_yaw=0.0)
    assert np.allclose(sh.here, [2.0, 4.0, math.pi / 2])
    slam._reckon_shelf(0.0, None, 0.1, moved=(-1.0, 0.0), fresh_yaw=math.pi)
    assert np.allclose(sh.here, [2.0, 5.0, math.pi / 2])
    slam._reckon_shelf(0.0, 0.5, 1.0)
    assert np.allclose(sh.here, [2.0, 5.5, math.pi / 2])
    assert sh.err_m == pytest.approx(0.1 + 2 * sl.SHELF_TRACK_FRAC + 0.5 * sl.LINK_ERR_FRAC)
    slam._reckon_shelf(math.radians(10), 0.0, 1.0)
    assert math.degrees(sh.here[2]) == pytest.approx(100.0)
    assert sh.err_yaw == 0.02
    slam._reckon_shelf(None, 0.0, 0.1)
    assert sh.err_yaw == math.pi, "a gap in the gyro: heading unknown"
