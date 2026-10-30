"""What a lost step costs: the reloc Field's scores, and the candidate fits
thinned and given up early (matcher.icp stride= and give_up=,
slam.CANDIDATE_STRIDE and CANDIDATE_GIVE_UP)."""

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler.mapping import matcher as mt  # noqa: E402
from crawler.mapping import reloc as rl  # noqa: E402
from crawler.mapping import scan as sc  # noqa: E402
from crawler.mapping import slam as sl  # noqa: E402
from crawler.mapping.imu import ImuTrack  # noqa: E402

from test_mapping import drive, path_pose, put_down, scan_at  # noqa: E402


@pytest.fixture(scope="module")
def room():
    imu = ImuTrack()
    slam = sl.Slam(imu=imu)
    drive(slam, 60, imu=imu)
    return slam


def scores_by_pair(f, pts, yaw):
    """Field.scores as it was: every point at every offset looked up on its
    own, off the raster scoring 0."""
    c, s = math.cos(yaw), math.sin(yaw)
    q = pts @ np.array([[c, s], [-s, c]])
    wx = q[None, :, 0] + (f.cx + f.offsets[:, None, 0])
    wy = q[None, :, 1] + (f.cy + f.offsets[:, None, 1])
    i = np.floor((wx - f.x0) / f.res).astype(np.int64)
    j = np.floor((wy - f.y0) / f.res).astype(np.int64)
    ok = (i >= 0) & (i < f.n) & (j >= 0) & (j < f.n)
    v = np.where(ok, f.score[np.where(ok, i, 0), np.where(ok, j, 0)], 0.0)
    return v.mean(1)


def test_the_field_scores_every_offset_as_a_lookup_per_pair_would(room):
    rng = np.random.default_rng(3)
    x, y, _ = room.pose
    for trial in range(12):
        f = rl.Field(room.grid, x + rng.uniform(-1, 1), y + rng.uniform(-1, 1))
        # Points near and far: some land off the raster at every offset.
        pts = rng.uniform(-1, 1, (200, 2)) * rng.choice([1.0, 4.0, 12.0], (200, 1))
        yaw = rng.uniform(-math.pi, math.pi)
        got = f.scores(pts, yaw)
        assert got.shape == (len(f.offsets),) and got.dtype == np.float32
        np.testing.assert_allclose(got, scores_by_pair(f, pts, yaw), atol=1e-6)


def scan_here(slam):
    """The room's points at the truck's last scan, and a tree of the map."""
    p, nrm, _ = sc.scan_points(scan_at(6.0))
    return p, nrm, slam._tree, slam._tgt, tuple(slam.pose)


def test_a_fit_on_every_other_point_lands_where_the_full_one_does(room):
    p, nrm, tree, tgt, truth = scan_here(room)
    assert len(p) >= 2 * mt.STRIDE_MIN_POINTS
    start = (truth[0] + 0.15, truth[1] - 0.1, truth[2] + math.radians(5))
    full = mt.icp(p, nrm, tree, tgt, start, iters=25)
    half = mt.icp(p, nrm, tree, tgt, start, iters=25, stride=2)
    assert full.converged and half.converged
    assert math.dist(half.pose[:2], full.pose[:2]) < 0.01
    assert abs(half.pose[2] - full.pose[2]) < math.radians(0.5)
    # Its inliers and residual are counted on the whole scan.
    assert abs(half.inliers - full.inliers) < 0.02
    assert half.matches > 0.9 * full.matches


def test_a_short_scan_is_not_thinned(room):
    p, nrm, tree, tgt, truth = scan_here(room)
    n = 2 * mt.STRIDE_MIN_POINTS - 1
    a = mt.icp(p[:n], nrm[:n], tree, tgt, truth, iters=25)
    b = mt.icp(p[:n], nrm[:n], tree, tgt, truth, iters=25, stride=2)
    assert a == b


def test_a_candidate_in_the_wrong_place_is_given_up_after_the_mid_gate(room):
    p, nrm, tree, tgt, truth = scan_here(room)
    wrong = (truth[0] + 1.2, truth[1] + 0.9, truth[2] + math.radians(100))
    kept = mt.icp(p, nrm, tree, tgt, wrong, iters=25)
    gone = mt.icp(p, nrm, tree, tgt, wrong, iters=25, give_up=0.3)
    assert kept.iters > 7
    assert not gone.converged and gone.iters == 7
    right = mt.icp(p, nrm, tree, tgt, truth, iters=25, give_up=0.3)
    assert right.converged and right.inliers > 0.9


class CountingTree:
    """A cKDTree that counts the points it is asked about."""

    def __init__(self, tree):
        self.tree, self.asked = tree, 0

    def query(self, q, **kw):
        self.asked += len(q)
        return self.tree.query(q, **kw)


def test_a_lost_search_fits_its_candidates_on_half_the_points(
        monkeypatch):
    imu = ImuTrack()
    slam = sl.Slam(imu=imu)
    drive(slam, 60, imu=imu)
    icp = mt.icp
    cands = []

    def spy(src, nrm, tree, tgt, pose0, *a, **kw):
        if kw.get("iters") != 25 or "give_up" not in kw:
            return icp(src, nrm, tree, tgt, pose0, *a, **kw)
        t = CountingTree(tree)
        f = icp(src, nrm, t, tgt, pose0, *a, **kw)
        cands.append((f, len(src), t.asked, kw))
        return f

    monkeypatch.setattr(sl.mt, "icp", spy)
    put = path_pose(6.0) + np.array([1.5, 0.5, math.radians(120)])
    out = put_down(slam, imu, put, 40, 6.0)
    assert slam.recoveries == 1
    assert any(r.state == sl.LOST for r in out)
    assert len(cands) >= 10
    # The points iterated on, each candidate: half its scan an iteration,
    # and the whole scan once for the count at the end.
    for f, n, asked, kw in cands:
        assert kw["give_up"] == sl.CANDIDATE_GIVE_UP > 0
        if f.converged:
            assert asked <= (n + 1) // 2 * f.iters + n
