"""Goals that look like the car's: along the path, and the one it was fed.

The planner feeds the model a point pathgoal.LOOKAHEAD_M along its path, and
behind the truck as often as not. A hindsight goal k seconds ahead is mostly
in front and further off. path_goals draws the first kind off the truck's own
path; fed_goals reads the point the car really had out of the recording.
"""

import math

import numpy as np
import pytest

from crawler.autopilot import policy_goal as pg
import fed_goals as fg
import policy_train as pt


def poses(xs, ys=None, yaw=0.0, sess=None, frame=None, dt=0.1):
    n = len(xs)
    g = {"log_t": np.arange(n) * dt, "x": np.asarray(xs, float),
         "y": np.zeros(n) if ys is None else np.asarray(ys, float),
         "yaw": np.full(n, yaw), "present": np.ones(n, bool),
         "sess": np.zeros(n, np.int64) if sess is None else np.asarray(sess, np.int64),
         "frame": np.zeros(n, np.int64) if frame is None else np.asarray(frame, np.int64)}
    g["key"] = pg.sort_key(g["sess"], g["log_t"])
    g["progress"] = pg.path_progress(g)
    return g


def test_a_path_goal_is_the_pose_the_lookahead_further_along():
    g = poses(np.arange(40) * 0.1)            # 10 cm a scan, straight ahead
    a, l, ok = pg.path_goals(np.array([0, 5]), 1.5, g)
    assert ok.all()
    # 1.5 m less the dead band per step: the first pose at least that far on.
    assert a[0] == pytest.approx(1.6, abs=0.11) and a[0] >= 1.5
    assert l == pytest.approx([0.0, 0.0])
    # Too near the end of the drive to have gone that far: no goal.
    assert not pg.path_goals(np.array([30]), 1.5, g)[2][0]


def test_a_truck_backing_up_has_its_path_goal_behind():
    g = poses(-np.arange(40) * 0.1)           # facing +x, going -x
    a, _, ok = pg.path_goals(np.array([0]), 1.5, g)
    assert ok[0] and a[0] < -1.4


def test_a_parked_truck_does_not_walk_to_a_goal_on_its_own_jitter():
    rng = np.random.default_rng(0)
    g = poses(rng.normal(0, 0.003, 2000), rng.normal(0, 0.003, 2000))
    assert not pg.path_goals(np.arange(0, 1900, 100), 1.5, g)[2].any()


def test_a_path_goal_never_crosses_a_session_a_reseed_or_the_time_cap():
    xs = np.r_[np.arange(10) * 0.1, np.arange(30) * 0.1]
    g = poses(xs, sess=np.r_[np.zeros(10), np.ones(30)])
    assert not pg.path_goals(np.array([0]), 1.5, g)[2][0]
    g = poses(np.arange(40) * 0.1, frame=np.r_[np.zeros(10), np.ones(30)])
    assert not pg.path_goals(np.array([0]), 1.5, g)[2][0]
    slow = poses(np.arange(400) * 0.01)      # 1.5 m takes ~30 s at 10 Hz here
    assert not pg.path_goals(np.array([0]), 1.5, slow, max_s=20.0)[2][0]
    assert pg.path_goals(np.array([0]), 1.5, slow, max_s=60.0)[2][0]


# --- the goal the car was fed ---------------------------------------------------

def test_the_fed_aim_is_put_in_the_truck_frame_off_a_fresh_tracking_pose():
    # Truck at (1, 1) facing +y; the aim 1 m further in +y is straight ahead,
    # 1 m in -x is on its left.
    slam = [(0.0, 1.0, 1.0, math.pi / 2, 1), (1.0, 1.0, 1.0, math.pi / 2, 1),
            (2.0, 1.0, 1.0, math.pi / 2, 0)]
    path = [(0.05, 1, 1.0, 2.0), (1.05, 1, 0.0, 1.0), (1.1, 0, 0.0, 0.0),
            (2.05, 1, 1.0, 2.0), (3.0, 1, 1.0, 2.0)]
    out = fg.fed_points(slam, path, [])
    assert str(out["source"]) == "path"
    assert out["ahead"][0] == pytest.approx(1.0) and out["left"][0] == pytest.approx(0.0, abs=1e-9)
    assert out["ahead"][1] == pytest.approx(0.0, abs=1e-9) and out["left"][1] == pytest.approx(1.0)
    # Not fed; SLAM lost; the pose too old.
    assert out["ok"].tolist() == [True, True, False, False, False]


def test_the_planner_aim_wins_over_the_click_and_the_click_is_used_without_one():
    slam = [(0.0, 0.0, 0.0, 0.0, 1)]
    goal = [(0.1, 1, 2.0, 0.0)]
    assert str(fg.fed_points(slam, [(0.1, 1, 1.5, 0.0)], goal)["source"]) == "path"
    out = fg.fed_points(slam, [], goal)
    assert str(out["source"]) == "goal" and out["ahead"][0] == pytest.approx(2.0)


def test_a_tick_takes_the_newest_status_and_none_that_is_stale():
    j = fg.pair([0.0, 1.0, 2.0], [0.5, 1.2, 1.6, -1.0])
    assert j.tolist() == [0, 1, -1, -1]


def test_a_fed_goal_replaces_the_drawn_one_and_still_needs_the_map_shown():
    g = poses(np.arange(40) * 0.1)
    data = {"goal_poses": g, "map_row": np.arange(40),
            "fed_ok": np.zeros(40, bool), "fed_ahead": np.zeros(40), "fed_left": np.zeros(40)}
    data["fed_ok"][2] = True
    data["fed_ahead"][2], data["fed_left"][2] = -1.0, 0.5    # behind, on the left
    rng = np.random.default_rng(0)
    out = pt.goal_inputs(data, np.array([2, 3]), np.array([True, True]), rng,
                         no_goal_share=0.0, path_m=1.5)
    want = pg.encode_many([-1.0], [0.5], [True])[0]
    assert out[0] == pytest.approx(want)
    assert out[1, pg.COS] > 0.99                               # the drawn path goal
    hidden = pt.goal_inputs(data, np.array([2]), np.array([False]), rng, no_goal_share=0.0)
    assert not hidden.any()


def test_the_steer_only_contrast_costs_only_rows_with_a_steer_label():
    present = np.array([True, True, True, False])
    tok = np.array([True, False, True, True])
    sok = np.array([True, True, False, True])
    assert pt.contrast_rows(present, tok, sok).tolist() == [True, True, True, False]
    assert pt.contrast_rows(present, tok, sok, steer_only=True).tolist() == [True, True, False, False]
