"""Scan to map matching: point-to-line Gauss-Newton ICP with a Cauchy kernel.

The scan's points, each with the normal of its local line, are pulled onto
the map's occupied-cell hit means. The error of a pair is the distance from
the map point to the scan point's line, so a wall can slide along itself
for free and only the across-wall offset costs anything. The Cauchy weight
1 / (1 + (e / 3 cm)^2) lets a chair that moved pull on the fit only a little.

The nearest map point comes from scipy's cKDTree, the one dependency this
package has beyond numpy (pyproject.toml says why).

What comes back besides the pose is what a navigation layer needs to decide
how far to trust it (Fit): the inlier fraction, the residual, and the
constraint, the smallest eigenvalue of the translation block of the
weighted, normalised Hessian. For unit normals that block is the weighted
mean of n n^T, whose eigenvalues add up to 1: 0.5 is a room with walls
every way, near 0 is a corridor or an open field where the scan cannot tell
one position along it from the next.
"""

from typing import NamedTuple

import numpy as np

# Cauchy kernel width, metres.
KERNEL_M = 0.03

# Correspondence gates per iteration: wide first so a poor prediction still
# finds its walls, then tight so the clutter around a wall drops out.
WIDE_M, MID_M, TIGHT_M = 0.4, 0.2, 0.1

# Fewer matched points than this and the fit is not a fit.
MIN_MATCHES = 30


class Fit(NamedTuple):
    pose: tuple            # (x, y, yaw) map frame
    inliers: float         # fraction of scan points within 10 cm after the fit
    rms: float             # metres, point-to-line, over those
    iters: int
    converged: bool
    matches: int           # pairs in the last iteration
    constraint: float      # smallest translation eigenvalue, 0..0.5


FAILED = None


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def icp(src, nrm, tree, tgt, pose0, yaw_prior=None, sigma_yaw=None, iters=20):
    """Fit the scan (src points, nrm normals, car frame) to the map.

    tree is a cKDTree over tgt (map points). yaw_prior, if given, is a soft
    pull towards that heading with standard deviation sigma_yaw (radians).
    """
    x, y, th = (float(v) for v in pose0)
    sched = [WIDE_M] * 3 + [MID_M] * 4 + [TIGHT_M] * max(iters - 7, 0)
    it = 0
    k = KERNEL_M
    for it, md in enumerate(sched):
        R = rot(th)
        q = src @ R.T + (x, y)
        d, j = tree.query(q, distance_upper_bound=md, workers=1)
        m = np.isfinite(d)
        if m.sum() < MIN_MATCHES:
            return Fit((x, y, th), 0.0, np.inf, it, False, int(m.sum()), 0.0)
        qm = q[m]
        tm = tgt[j[m]]
        nw = nrm[m] @ R.T
        e = ((qm - tm) * nw).sum(1)
        sp = src[m]
        dq = np.stack([-np.sin(th) * sp[:, 0] - np.cos(th) * sp[:, 1],
                       np.cos(th) * sp[:, 0] - np.sin(th) * sp[:, 1]], 1)
        J = np.stack([nw[:, 0], nw[:, 1], (dq * nw).sum(1)], 1)
        w = 1.0 / (1.0 + (e / k) ** 2)
        Jw = J * w[:, None]
        H = Jw.T @ J
        g = Jw.T @ e
        if yaw_prior is not None:
            wp = (k ** 2) * 50.0 / sigma_yaw ** 2
            dth = (th - yaw_prior + np.pi) % (2 * np.pi) - np.pi
            H[2, 2] += wp
            g[2] += wp * dth
        try:
            dx = -np.linalg.solve(H + 1e-9 * np.eye(3), g)
        except np.linalg.LinAlgError:
            return Fit((x, y, th), 0.0, np.inf, it, False, int(m.sum()), 0.0)
        x += dx[0]
        y += dx[1]
        th += dx[2]
        if it >= 7 and abs(dx[0]) + abs(dx[1]) < 1e-4 and abs(dx[2]) < 1e-4:
            break
    R = rot(th)
    q = src @ R.T + (x, y)
    d, j = tree.query(q, distance_upper_bound=TIGHT_M, workers=1)
    m = np.isfinite(d)
    n = int(m.sum())
    if n == 0:
        return Fit((x, y, th), 0.0, np.inf, it + 1, False, 0, 0.0)
    nw = nrm[m] @ R.T
    e = ((q[m] - tgt[j[m]]) * nw).sum(1)
    return Fit((x, y, th), float(m.mean()), float(np.sqrt(np.mean(e ** 2))),
               it + 1, True, n, constraint(nw, e))


def constraint(normals, errors=None):
    """Smallest eigenvalue of the weighted mean of n n^T, 0..0.5.

    The translation block of the ICP Hessian, normalised by its weight so it
    does not grow with the number of points.
    """
    if len(normals) == 0:
        return 0.0
    if errors is None:
        w = np.ones(len(normals))
    else:
        w = 1.0 / (1.0 + (np.asarray(errors) / KERNEL_M) ** 2)
    a = float((w * normals[:, 0] ** 2).sum())
    b = float((w * normals[:, 0] * normals[:, 1]).sum())
    c = float((w * normals[:, 1] ** 2).sum())
    s = a + c
    if s <= 0:
        return 0.0
    lam = 0.5 * (a + c) - np.sqrt(0.25 * (a - c) ** 2 + b * b)
    return float(max(lam, 0.0) / s)
