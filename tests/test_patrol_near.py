"""The near pass (autopilot/patrol.py, "A near miss"): a waypoint the truck
came within near_pass_m of and then left beside or behind it, going away
from it, or deep inside its turning circle counts as passed. Only the goal
fed changes. Never for a waypoint it was not near, never at a stop, and off
at 0."""

import math

import pytest

pytest.importorskip("scipy")

from crawler.autopilot import patrol as pt  # noqa: E402
from crawler.nav import truck  # noqa: E402

from test_goal_reseed import crop_at  # noqa: E402
from test_patrol import Events, cmd, patrol_of  # noqa: E402

# A line along x with a waypoint beside it, and one on beyond.
BESIDE = [(2.0, 0.8), (5.0, 0.0)]


def drive_along(p, y=0.0, x0=0.0, x1=3.0, step=0.05, yaw=0.0):
    """The truck nose first along y from x0 to x1: a control pass a step."""
    n = int(round((x1 - x0) / step))
    for k in range(n + 1):
        p.poll(Events(), crop_at(x0 + k * step, y, yaw))


def events(p):
    out = []
    while True:
        e = p.pop_event()
        if e is None:
            return out
        out.append(e)


# --- the rule ---------------------------------------------------------------------------

def test_near_miss_abeam_away_and_inside_the_turning_circle():
    r = math.radians
    assert pt.near_miss(0.8, r(85), 0.8, 1.0) == "abeam"
    assert pt.near_miss(0.8, r(-170), 0.7, 1.0) == "abeam"
    assert pt.near_miss(0.95, r(20), 0.7, 1.0) == "away"
    assert pt.near_miss(0.9, r(20), 0.7, 1.0) is None, "not yet 0.25 m further"
    # By a full lock circle's centre (0.7 m to the side), a touch ahead and
    # short of abeam: no forward arc gets within PASS_M of it.
    d, b = math.hypot(0.19, 0.67), math.atan2(0.67, 0.19)
    assert math.degrees(b) < pt.NEAR_ABEAM_DEG
    assert pt.near_miss(d, b, d, 1.0) == "circle"
    assert pt.near_miss(d, -b, d, 1.0) == "circle"
    # Ahead and a little off: driven on, it is reached.
    assert pt.near_miss(0.8, r(30), 0.8, 1.0) is None
    assert pt.near_miss(0.6, r(60), 0.6, 1.0) is None


def test_near_miss_never_for_a_waypoint_it_was_not_near_or_with_it_off():
    r = math.radians
    assert pt.near_miss(1.2, r(100), 1.05, 1.0) is None
    assert pt.near_miss(2.0, r(0), 1.05, 1.0) is None
    assert pt.near_miss(0.8, r(100), 0.8, 0.0) is None


def test_the_turning_circle_is_the_one_the_truck_drives():
    assert pt.TURN_RADIUS_M == truck.DRIVEN_TURN_RADIUS_M


# --- the patrol -------------------------------------------------------------------------

def test_driven_past_beside_it_the_waypoint_is_passed_near_with_its_own_event_and_line():
    p, said, _ = patrol_of(points=BESIDE, mode="loop")
    events(p)
    drive_along(p, y=0.0, x0=0.0, x1=2.6)
    assert p.index == 1 and p.passed_near == 1 and p.skipped == 0
    ev = [e for e in events(p) if e["kind"] == "pass"]
    assert len(ev) == 1 and ev[0]["by"] == "near" and ev[0]["index"] == 0
    assert ev[0]["next"] == 1
    lines = [a[2] for a in said if "passed near" in a[2]]
    assert len(lines) == 1 and lines[0].startswith("Waypoint 1 of 2 passed near (")
    st = p.patrol_status()
    assert st["passed_near"] == 1 and st["near_pass_m"] == pytest.approx(1.0)
    assert p.page_state()["near_pass_m"] == pytest.approx(1.0)


def test_off_it_stays_the_goal_as_before():
    p, said, _ = patrol_of(points=BESIDE, mode="loop", near_pass_m=0.0)
    drive_along(p, y=0.0, x0=0.0, x1=3.0)
    assert p.index == 0 and p.passed_near == 0
    assert any("Near passes are off" in a[2] for a in said)


def test_a_waypoint_it_never_came_near_is_never_passed():
    p, _, _ = patrol_of(points=[(2.0, 1.2), (5.0, 0.0)], mode="loop")
    drive_along(p, y=0.0, x0=0.0, x1=4.0)
    assert p.index == 0 and p.passed_near == 0


def test_a_waypoint_it_comes_back_to_is_not_passed_near_on_the_way_in():
    # Straight at it: it is ahead the whole way and passed on PASS_M.
    p, _, _ = patrol_of(points=[(2.0, 0.0), (5.0, 0.0)], mode="loop")
    events(p)
    drive_along(p, y=0.0, x0=0.0, x1=1.6)
    assert p.index == 1 and p.passed_near == 0
    assert [e["by"] for e in events(p) if e["kind"] == "pass"] == ["radius"]


def test_the_last_stop_of_a_once_and_a_dwell_are_never_passed_near():
    p, _, _ = patrol_of(points=[(-3.0, 0.0), (2.0, 0.8)], mode="once")
    p.poll(Events(), crop_at(-3.0, 0.0))
    assert p.index == 1
    drive_along(p, y=0.0, x0=0.0, x1=3.0)
    assert p.index == 1 and not p.finished and p.passed_near == 0
    q, _, _ = patrol_of(points=BESIDE, mode="loop", dwell_s=2.0)
    drive_along(q, y=0.0, x0=0.0, x1=3.0)
    assert q.index == 0 and q.passed_near == 0


def test_not_where_it_stands_as_the_waypoint_becomes_the_goal():
    # The next waypoint is 0.85 m abeam as the first is passed: kept until
    # the truck has moved NEAR_MOVE_M, so a tight bend is not passed in one go.
    p, _, _ = patrol_of(points=[(1.0, 0.0), (0.5, 0.85), (5.0, 5.0)], mode="loop")
    p.poll(Events(), crop_at(0.5, 0.0))
    assert p.index == 1
    for _ in range(5):
        p.poll(Events(), crop_at(0.5, 0.0))
    assert p.index == 1
    p.poll(Events(), crop_at(0.6, 0.0))
    assert p.index == 1, "0.1 m moved"
    p.poll(Events(), crop_at(0.8, 0.0))
    assert p.index == 2 and p.passed_near == 1


def test_the_closest_starts_again_when_the_waypoint_is_moved():
    # Near it, then it is moved 3 m off: further away, but never was near.
    p, _, _ = patrol_of(points=[(1.0, 0.0), (5.0, 5.0)], mode="loop")
    drive_along(p, y=0.0, x0=-1.0, x1=0.2)
    assert p.index == 0 and p._closest < 1.0
    p.poll(Events(cmd("move", i=0, x=4.0, y=0.0)), crop_at(0.2, 0.0))
    assert p.current_place()[:2] == pytest.approx((4.0, 0.0))
    drive_along(p, y=0.0, x0=0.2, x1=0.2)
    assert p.index == 0 and p.passed_near == 0


def test_the_closest_starts_again_when_a_carry_moves_the_waypoint():
    # The snap back on a merge moves a waypoint's place on the same map
    # with nothing else said: the closest from where it was is dropped.
    p, _, _ = patrol_of(points=[(1.0, 0.0), (5.0, 5.0)], mode="loop")
    drive_along(p, y=0.0, x0=-1.0, x1=0.2)
    assert p.index == 0 and p._closest < 1.0
    w = p.points[0]
    w.places[0] = (4.0, 0.0) + tuple(w.places[0][2:])
    drive_along(p, y=0.0, x0=0.2, x1=0.2)
    assert p.current_place()[:2] == pytest.approx((4.0, 0.0))
    assert p.index == 0 and p.passed_near == 0


def test_backed_away_from_it_once_near_it_is_passed():
    p, said, _ = patrol_of(points=[(1.0, 0.0), (5.0, 5.0)], mode="loop")
    drive_along(p, y=0.0, x0=-1.0, x1=0.2)
    assert p.index == 0
    drive_along(p, y=0.0, x0=0.2, x1=-0.1, step=-0.05)
    assert p.index == 1 and any("passed near (away)" in a[2] for a in said)


# --- the setting ------------------------------------------------------------------------

def test_the_radius_is_set_kept_across_a_restart_and_wins_over_the_default(tmp_path):
    settings = pt.Store(str(tmp_path / "patrol_settings.json"), background=False)
    p, said, _ = patrol_of(points=BESIDE, mode="loop", settings=settings)
    assert p.near_pass_m == pytest.approx(pt.NEAR_PASS_M)
    assert any("1.00 m" in a[2] and "the default" in a[2] for a in said)
    p.poll(Events(cmd("skip", on=True), cmd("near", m=0.0)), crop_at(-1.0, -1.0))
    assert p.near_pass_m == 0.0
    assert settings.load() == {"skip_stuck": True, "near_pass_m": 0.0}, "both kept"
    q, said, _ = patrol_of(points=BESIDE, mode="loop", settings=settings, near_pass_m=1.3)
    assert q.near_pass_m == 0.0 and q.skip_stuck
    assert any("Near passes are off" in a[2] and "as last set" in a[2] for a in said)
    q.poll(Events(cmd("near", m=1.3)), crop_at(-1.0, -1.0))
    assert q.near_pass_m == pytest.approx(1.3)
    assert settings.load() == {"skip_stuck": True, "near_pass_m": 1.3}
    for bad in ("1", True, -0.5, 9.0, None, float("nan")):
        assert q.command({"cmd": "near", "m": bad}) is False
    assert q.near_pass_m == pytest.approx(1.3)
    # The flag, as the default.
    from crawler import __main__ as cm
    assert cm.parse_args([]).patrol_near_pass_m == pytest.approx(pt.NEAR_PASS_M)
    assert cm.parse_args(["--patrol-near-pass-m", "0"]).patrol_near_pass_m == 0.0
    r, said, _ = patrol_of(points=BESIDE, mode="loop", near_pass_m=7.0)
    assert r.near_pass_m == pytest.approx(pt.NEAR_PASS_M)
    assert any("not from 0" in a[2] for a in said)
    assert "near" in pt.COMMANDS
