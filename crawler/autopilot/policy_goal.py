"""Where the policy is asked to go: a point on the SLAM map, as the model sees it.

One encoder, used by the trainer and the car, as egomap.py is for the crop.
A goal is a point on the map (map metres, on one map frame, egomap.Crop.frame).
Each decision it is put in the truck's frame off the pose the crop going into
the same model run was drawn at (Crop.x, .y, .yaw), so the goal and the map
the model sees are always off one pose.

The input
---------
(1, SIZE) float32:

  present   1 with a goal, 0 without. No goal is every value 0: the "no
            goal" the model is trained on for NO_GOAL_SHARE of its ticks,
            and what it is fed with no click, no map, or a goal on another
            map frame than the one the truck is on now.
  dist      the distance to the goal over FAR_M, at most 1
  sin, cos  of the bearing to it, counter-clockwise from straight ahead:
            left is sin > 0, behind is cos < 0

A vector rather than a mark in the crop: the crop is 3.2 m each way, and a
hindsight goal FAR_S ahead on a drive is often past its edge, where a mark
could only sit clipped to the border with its distance gone. The vector says
the same thing at any range, mirrors by one sign, and leaves the map input,
and every model trained on it, exactly as it was.

Mirroring: the bearing flips, so sin changes sign (MIRROR), as the trainer's
flip augmentation mirrors the crop's columns.

Hindsight goals
---------------
The data has no goals in it, so the trainer makes them (hindsight_goals):
for a tick at crop row r, the goal is the truck's own pose k seconds later on
the same drive, k drawn fresh from [NEAR_S, FAR_S] every time the tick is
used, put in the frame of row r's pose. Whatever the truck did at r is then
a right way of getting there. A tick gets no goal when its crop is not
trusted, when nothing at least k later is in the file, when the pose there is
not trusted, or when it is on another map frame (a reseed between).

Arriving is not learned: the safety layer holds neutral within ARRIVE_M of a
clicked goal (goal.py, ClickGoal), the one hand rule on the way there, by
the owner's ruling. The model drives every metre up to it.

numpy only.
"""

import math

import numpy as np

SIZE = 4
PRESENT, DIST, SIN, COS = 0, 1, 2, 3
# Distances past this all read 1.
FAR_M = 8.0
# The hindsight window: a goal this many seconds ahead of the tick.
NEAR_S = 3.0
FAR_S = 15.0
# The share of goal-able training ticks fed no goal anyway, fresh every
# batch, so the model with no goal is the free driver it was.
NO_GOAL_SHARE = 0.3
# The arrival radius (goal.ClickGoal): within it, neutral until a new goal.
ARRIVE_M = 0.25
LAYOUT = f"goal{SIZE}_far{int(FAR_M)}"
MIRROR = np.array([1.0, 1.0, -1.0, 1.0], np.float32)


def relative(pose, gx, gy):
    """(ahead, left) metres of map point (gx, gy) from pose (x, y, yaw).
    Arrays broadcast."""
    x, y, yaw = pose
    dx = np.asarray(gx, float) - x
    dy = np.asarray(gy, float) - y
    c, s = np.cos(yaw), np.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def encode_many(ahead, left, present):
    """(N,) truck-frame goal points and (N,) present -> (N, SIZE) float32;
    a row that is not present is all zeros."""
    ahead = np.asarray(ahead, float).reshape(-1)
    left = np.asarray(left, float).reshape(-1)
    p = np.asarray(present, bool).reshape(-1) & np.isfinite(ahead) & np.isfinite(left)
    out = np.zeros((len(p), SIZE), np.float32)
    if p.any():
        d = np.hypot(ahead[p], left[p])
        b = np.arctan2(left[p], ahead[p])
        out[p, PRESENT] = 1.0
        out[p, DIST] = np.minimum(d / FAR_M, 1.0)
        out[p, SIN] = np.sin(b)
        out[p, COS] = np.cos(b)
    return out


def missing():
    return np.zeros((1, SIZE), np.float32)


def encode(crop, goal):
    """The car's goal input: `goal` (x, y, frame) on the map against the
    crop the model is fed this decision. No goal unless both are there, the
    crop is present and they are on the same map frame."""
    if goal is None or crop is None or not getattr(crop, "present", False):
        return missing()
    gx, gy, frame = goal
    if int(frame) != int(getattr(crop, "frame", 0)):
        return missing()
    a, l = relative((crop.x, crop.y, crop.yaw), gx, gy)
    return encode_many([a], [l], [True])


def mirror(g):
    return np.asarray(g, np.float32) * MIRROR


def distance_bearing(crop, goal):
    """(metres, radians CCW) from the crop's pose to the goal, or None."""
    if goal is None or crop is None:
        return None
    gx, gy, frame = goal
    if int(frame) != int(getattr(crop, "frame", 0)):
        return None
    a, l = relative((crop.x, crop.y, crop.yaw), gx, gy)
    return float(math.hypot(a, l)), float(math.atan2(l, a))


def hindsight_goals(rows, k_s, poses):
    """Hindsight goals for crop rows `rows` (-1: none) k_s seconds ahead.

    poses: dict of the crops' arrays, concatenated over sessions in order
    (policy_train.load_maps): log_t, x, y, yaw, frame, present, sess (the
    session each row is from). -> (ahead, left, ok), (N,) each."""
    rows = np.asarray(rows, np.int64)
    k_s = np.broadcast_to(np.asarray(k_s, float), rows.shape)
    n = len(poses["log_t"])
    ok = rows >= 0
    r = np.where(ok, rows, 0)
    # Rows are time-ordered within a session and sessions follow one
    # another, so one key orders them all: the session, then the time.
    key = poses["key"]
    j = np.searchsorted(key, key[r] + k_s, side="left")
    ok &= j < n
    j = np.where(j < n, j, 0)
    sess, frame, present = poses["sess"], poses["frame"], poses["present"]
    ok &= (sess[j] == sess[r]) & (frame[j] == frame[r]) & present[r] & present[j]
    a, l = relative((poses["x"][r], poses["y"][r], poses["yaw"][r]),
                    poses["x"][j], poses["y"][j])
    return np.where(ok, a, 0.0), np.where(ok, l, 0.0), ok


def sort_key(sess, log_t):
    """The one order hindsight_goals searches: session, then time (seconds;
    a session is never 1e6 s long)."""
    sess = np.asarray(sess, np.int64)
    t = np.asarray(log_t, float)
    t0 = np.zeros(int(sess.max()) + 1 if len(sess) else 1)
    for s in np.unique(sess):
        t0[s] = t[sess == s].min()
    return sess * 1e6 + (t - t0[sess])


def meta():
    return {"goal_layout": LAYOUT, "goal_size": SIZE, "goal_far_m": FAR_M,
            "goal_near_s": NEAR_S, "goal_far_s": FAR_S,
            "goal_no_goal_share": NO_GOAL_SHARE}
