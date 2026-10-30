"""The forward floor is a camera distance, off the lens, and the nav side
sizes things off it in the pose's (lidar's) frame. These pin the mapping: a
probe distance d is LENS_X_MM + d ahead of the pose, which is d less
LENS_TO_FRONT_BUMPER_MM past the bumper."""
import pytest

from crawler.autopilot import driver as dv
from crawler.hardware import lidar as ld
from crawler.nav import driver as nd
from crawler.nav import navigator as nv
from crawler.nav import truck as tk


def test_the_lens_sits_behind_the_bumper_by_the_taped_offset():
    assert tk.LENS_M == pytest.approx(dv.LENS_X_MM / 1000.0)
    assert (tk.FRONT_M - tk.LENS_M) * 1000.0 == pytest.approx(
        dv.LENS_TO_FRONT_BUMPER_MM)
    assert ld.FRONT_BUMPER_X_MM == pytest.approx(
        dv.LENS_X_MM + dv.LENS_TO_FRONT_BUMPER_MM)


def test_the_floor_reach_is_off_the_lens_not_the_bumper():
    """363 mm off the lens is 598 mm ahead of the pose and 335 past the
    bumper. Taking it as off the bumper put the reach 28 mm too far out."""
    assert tk.FLOOR_REACH_M == pytest.approx(
        (dv.LENS_X_MM + dv.FORWARD_FLOOR_MM) / 1000.0)
    assert (tk.FLOOR_REACH_M - tk.FRONT_M) * 1000.0 == pytest.approx(
        dv.FORWARD_FLOOR_MM - dv.LENS_TO_FRONT_BUMPER_MM)
    assert tk.FLOOR_REACH_M == pytest.approx(0.598, abs=0.001)


def test_the_aim_clearance_is_the_reach_less_the_arrival():
    assert nv.AIM_CLEAR_M == pytest.approx(
        tk.FLOOR_REACH_M - nv.ARRIVE_M + 0.07)
    assert nv.AIM_CLEAR_M == pytest.approx(0.418, abs=0.001)


# The veto's reach ahead of the pose, from the driver's own constants.
REACH_M = (dv.LENS_X_MM + dv.FORWARD_FLOOR_MM) / 1000.0


def test_the_planner_probe_box_ends_at_the_reach():
    xs = tk.PROBE_POINTS[:, 0]
    assert xs.min() > tk.FRONT_M
    assert REACH_M - 0.03 < xs.max() <= REACH_M + 1e-9


def test_a_floor_hold_is_marked_where_the_veto_reaches():
    """blocked_ahead takes its distance from the pose: the block's centre is
    the reach plus its radius ahead, not the floor past the bumper."""
    assert nd.FLOOR_AHEAD_M == pytest.approx(REACH_M)
    nav = nv.Navigator()
    nav.blocked_ahead(0.0, (0.0, 0.0, 0.0), nd.FLOOR_AHEAD_M)
    bx, by, _ = nav.blocks[-1]
    assert bx == pytest.approx(REACH_M + nv.BLOCK_R_M)
    assert by == pytest.approx(0.0)
