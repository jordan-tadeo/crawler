"""Does the model go where it is told: the scorecard's goal adherence.

A hindsight goal is where my hand took the car anyway, so a model can score
well with one and never look at it. These put the same tick in front of the
model with a goal GOAL_PROBE_M off to its left and then to its right, and
grade whether the steer moves toward it -- a question the drive's own goal
cannot ask, because there the goal and everything else agree.
"""

import numpy as np
import pytest

from crawler.autopilot import policy as pl
from crawler.autopilot import policy_goal as pg
import policy_scorecard as sc

from test_policy_scorecard import drive


class GoalStub:
    contract = pl.CONTRACT_DEPTH_SMALL
    representation = pl.PROFILE
    has_mode = uses_prev = uses_imu = uses_lidar = uses_map = False
    uses_goal = True
    chunk, ensemble_m, ensemble_link_s = 1, 0.0, pl.CHUNK_LINK_S

    def __init__(self, steer_of, gear_of=lambda goal: 0):
        self.steer_of, self.gear_of = steer_of, gear_of

    def run(self, x, prev, mode=None, goal=None, **_):
        g = np.asarray(goal, np.float32)[0]
        logits = np.zeros((1, 3), np.float32)
        logits[0, self.gear_of(g)] = 5.0
        return {"logits": logits, "throttle": np.array([[0.5]], np.float32),
                "steer": np.array([[self.steer_of(g)]], np.float32)}


def ticks_of(a):
    return np.flatnonzero(a["decision"])


def test_a_model_that_turns_to_the_goal_follows_it_and_one_that_ignores_it_does_not():
    a = drive()
    t = ticks_of(a)
    # Negative steer is left; a goal on the left has sin > 0.
    follows = sc.goal_adherence(GoalStub(lambda g: -0.5 * g[pg.SIN]), a, t)
    ignores = sc.goal_adherence(GoalStub(lambda g: 0.3), a, t)
    # 0.5 sin(60) each way.
    assert follows["goal_turn"] == 1.0 and follows["goal_gap"] == pytest.approx(np.sqrt(3) / 2)
    assert ignores["goal_turn"] == 0.0 and ignores["goal_gap"] == pytest.approx(0.0)
    # Turning away from it is not following it.
    away = sc.goal_adherence(GoalStub(lambda g: 0.5 * g[pg.SIN]), a, t)
    assert away["goal_turn"] == 0.0 and away["goal_gap"] < 0


def test_the_turn_is_graded_only_where_the_model_drives_forward_both_times():
    a = drive()
    t = ticks_of(a)
    # Reverses whenever the goal is on the right: no forward pair to grade.
    s = sc.goal_adherence(GoalStub(lambda g: -0.5 * g[pg.SIN],
                                   gear_of=lambda g: 1 if g[pg.SIN] < 0 else 0), a, t)
    assert s["goal_turn_ticks"] == 0 and np.isnan(s["goal_turn"])
    # ...and goal_turn_any grades it: the nose goes left either way, so no.
    assert s["goal_turn_any"] == 0.0


def test_backing_round_toward_the_goal_counts_and_driving_away_or_stopping_does_not():
    a = drive()
    t = ticks_of(a)
    rev = lambda g: 1  # noqa: E731
    # Reversing, positive steer turns the nose left: toward a goal on the left.
    toward = sc.goal_adherence(GoalStub(lambda g: 0.5 * g[pg.SIN], gear_of=rev), a, t)
    away = sc.goal_adherence(GoalStub(lambda g: -0.5 * g[pg.SIN], gear_of=rev), a, t)
    stop = sc.goal_adherence(GoalStub(lambda g: -0.5 * g[pg.SIN], gear_of=lambda g: 2), a, t)
    assert toward["goal_turn_any"] == 1.0 and np.isnan(toward["goal_turn"])
    assert away["goal_turn_any"] == 0.0 and stop["goal_turn_any"] == 0.0
    fwd = sc.goal_adherence(GoalStub(lambda g: -0.5 * g[pg.SIN]), a, t)
    assert fwd["goal_turn_any"] == 1.0 == fwd["goal_turn"]


def test_a_goal_behind_that_makes_it_reverse_counts_and_the_blank_goal_is_measured():
    a = drive()
    t = ticks_of(a)
    behind = GoalStub(lambda g: 0.0, gear_of=lambda g: 1 if g[pg.COS] < 0 else 0)
    s = sc.goal_adherence(behind, a, t)
    assert s["goal_back_rev"] == pytest.approx(1.0)
    assert sc.goal_adherence(GoalStub(lambda g: 0.0), a, t)["goal_back_rev"] == 0.0
    # One that reverses whatever the goal is was not sent back by it.
    always = GoalStub(lambda g: 0.0, gear_of=lambda g: 1)
    assert sc.goal_adherence(always, a, t)["goal_back_rev"] == 0.0
    # goal_use: the steer moved by the goal against none at all.
    s = sc.goal_adherence(GoalStub(lambda g: 0.4 * g[pg.PRESENT]), a, t)
    assert s["goal_use"] == pytest.approx(0.4)


# --- training the goal in: the contrast -------------------------------------------

import policy_train as pt  # noqa: E402


def test_a_rotated_goal_keeps_its_distance_and_presence_and_turns_the_bearing():
    g = pg.encode_many([1.0, 0.0], [0.0, 2.0], [True, True])   # ahead; left
    r = pt.rotate_goals(g, np.radians([180.0, 90.0]))
    assert r[:, pg.PRESENT].tolist() == [1.0, 1.0]
    assert r[:, pg.DIST] == pytest.approx(g[:, pg.DIST])
    assert r[0, pg.COS] == pytest.approx(-1.0) and r[0, pg.SIN] == pytest.approx(0.0, abs=1e-6)
    assert r[1, pg.COS] == pytest.approx(-1.0) and r[1, pg.SIN] == pytest.approx(0.0, abs=1e-6)
    # A quarter turn CCW of straight ahead is on the left.
    left = pt.rotate_goals(g[:1], np.radians([90.0]))
    assert left[0, pg.SIN] == pytest.approx(1.0) and left[0, pg.COS] == pytest.approx(0.0, abs=1e-6)
    # No goal stays no goal.
    assert not pt.rotate_goals(np.zeros((1, pg.SIZE), np.float32), np.array([1.0])).any()


def test_the_contrast_costs_only_where_the_wrong_goal_explains_my_driving_as_well():
    true = np.array([0.1, 0.5, 0.1])
    fake = np.array([1.0, 0.5, 0.0])
    got = pt.goal_contrast(true, fake, margin=0.2, xp=np)
    # Clearly better with the real goal: free. As good: the margin. Worse: more.
    assert got == pytest.approx([0.0, 0.2, 0.3])


def test_the_contrast_turns_are_drawn_at_least_a_quarter_turn_off():
    a = pt.contrast_angles(np.random.default_rng(0), 1000)
    assert (np.abs(a) >= np.pi / 2 - 1e-9).all() and (np.abs(a) <= np.pi + 1e-9).all()
    assert (a > 0).any() and (a < 0).any()
