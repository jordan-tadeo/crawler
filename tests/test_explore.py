"""Explore: the truck picks its own goals round the house (autopilot/explore.py,
run inside the patrol as use "explore").

The goal is the search's frontier, then wander, with no person hint; the
model drives to it on its goal input, exactly as to a waypoint. These hold
the switch on /patrol/cmd and its exclusion with the patrol, the hold with
nowhere to go, the target's clock running on auto's decisions only, the carry
across a reseed, the status and /footprint fields that tell an explore drive
from a patrol one, the map page's state, and a simulated drive through the
real PolicyDriver with a stub model that only reads its goal input.
"""

import math
import os

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler.autopilot import footprint as fp  # noqa: E402
from crawler.autopilot import patrol as pt  # noqa: E402
from crawler.autopilot import search as sr  # noqa: E402
from crawler.telemetry import telemetry as tm  # noqa: E402

from test_goal_reseed import crop_at, link  # noqa: E402
from test_patrol import SQUARE, Clock, Events, GoalSteerStub, cmd, pose  # noqa: E402
from test_pathgoal import world  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A room x -2..6, y -3..3, seen only up to x = 3: the edge of the map is the
# line x = 3 across the room (test_search.py's).
HALF_SEEN = world(known=(-2.0, -3.0, 3.0, 3.0))


def explorer(view=HALF_SEEN, points=(), t=0.0):
    said = []
    p = pt.Patrol(say=lambda *a: said.append(a), clock=Clock(), map_view=lambda: view)
    ev = Events()
    if points:
        ev.add(cmd("edit"), *[pose(*xy) for xy in points], cmd("mode", mode="loop"))
    p.poll(ev, crop_at(t=t))
    return p, said


def test_explore_on_heads_for_the_edge_of_the_map_and_is_the_goal_fed():
    p, said = explorer()
    assert p.use == "goal" and p.explore.target is None
    p.poll(Events(cmd("explore")), crop_at(t=0.1))
    assert p.use == "explore" and p.state() == "exploring"
    assert p.explore.kind == "frontier"
    assert p.goal == p.explore.target and p.goal[0] == pytest.approx(2.95, abs=0.15)
    assert not p.reached(), "a target to drive to is no hold"
    np.testing.assert_allclose(p.encode(crop_at(t=0.1)), pt.pg.encode(crop_at(t=0.1), p.goal))
    assert p.ahead == [] and p.passed(p.goal) is False, "no waypoints after it, nothing to pass"
    # Said in the journal under its own topic, and on the map page's toasts.
    topics = [a[1] for a in said]
    assert "Patrol" in topics and "Explore" in topics
    assert any("Exploring: heading for a frontier point" in a[2] for a in said)
    assert any("Exploring" in text for _, text in p.said)


def test_explore_and_the_patrol_are_never_both_on():
    p, said = explorer(points=SQUARE)
    p.poll(Events(cmd("start")), crop_at(-1.0, -1.0))
    assert p.use == "patrol" and p.running
    # Explore on: the patrol stops, its list and place kept.
    p.poll(Events(cmd("explore", on=True)), crop_at(t=0.1))
    assert p.use == "explore" and not p.running and len(p.points) == 4
    events = []
    while (e := p.pop_event()) is not None:
        events.append((e["kind"], e["by"]))
    assert events == [("start", "fresh"), ("pause", "explore")], "a lap report sees it stop"
    # Start: the patrol has the goal again and explore is off, said so.
    p.poll(Events(cmd("start")), crop_at(t=0.2))
    assert p.use == "patrol" and p.running and not p.explore.on
    assert p.goal[:2] == SQUARE[0]
    assert any(a[2] == "Explore off - the patrol has the goal" for a in said)
    # On again, then the toggle: off, holding neutral.
    p.poll(Events(cmd("explore")), crop_at(t=0.3))
    assert p.use == "explore" and not p.running
    p.poll(Events(cmd("explore")), crop_at(t=0.4))
    assert not p.explore.on and p.use == "patrol" and not p.running
    assert p.goal is None and p.reached(), "off is the hold, not the old clicked goal"
    assert said[-1][2] == "Explore off - holding neutral"
    # Pause and a single goal stop it too; on twice is still on once.
    for stop in (cmd("pause"), cmd("goal")):
        p.poll(Events(cmd("explore", on=True), cmd("explore", on=True)), crop_at(t=0.5))
        assert p.use == "explore" and p.explore.started >= 2
        p.poll(Events(stop), crop_at(t=0.6))
        assert not p.explore.on and p.explore.target is None and p.use != "explore"


def test_a_bad_explore_command_is_refused_and_said():
    p, said = explorer()
    p.poll(Events(("cmd", {"cmd": "explore", "on": "yes", "id": "a1"})), crop_at(t=0.1))
    assert p.use == "goal"
    assert p.acks[-1] == ["a1", False, "Explore refused: on is 'yes', not true or false"]
    assert "explore" in pt.COMMANDS and "explore" in pt.MAP_COMMANDS


def test_drawing_while_exploring_keeps_exploring_and_the_stroke_waits_for_start():
    p, _ = explorer()
    p.poll(Events(cmd("explore")), crop_at(t=0.1))
    p.poll(Events(("cmd", {"cmd": "stroke", "points": [[1, 0], [1, 1], [0, 1]]})), crop_at(t=0.2))
    assert p.use == "explore" and len(p.points) == 3 and not p.running


def test_with_nowhere_to_go_it_holds_neutral_and_says_so_once():
    said = []
    p = pt.Patrol(say=lambda *a: said.append(a), clock=Clock(), map_view=lambda: None)
    p.poll(Events(cmd("explore")), crop_at(t=0.0))
    for t in np.arange(0.1, 5.0, 0.1):
        p.poll(Events(), crop_at(t=float(t)))
        p.observe(crop_at(t=float(t)))
    assert p.use == "explore" and p.state() == "holding"
    assert p.goal is None and p.reached(), "the arrival hold: neutral"
    assert sum("nowhere to go" in a[2] for a in said) == 1


def test_the_targets_clock_runs_on_auto_decisions_only():
    p, _ = explorer()
    p.poll(Events(cmd("explore")), crop_at(t=0.0))
    first = p.explore.target
    # Manual: a minute of passes standing still never gives up on it.
    for t in np.arange(0.1, 60.0, 0.1):
        p.poll(Events(), crop_at(t=float(t)))
    assert p.explore.target == first and p.explore.search.skipped == 0
    # Auto standing still: skipped after search's no-progress window.
    for t in np.arange(60.0, 60.0 + sr.NO_PROGRESS_S + 1.0, 0.1):
        p.poll(Events(), crop_at(t=float(t)))
        p.observe(crop_at(t=float(t)))
    assert p.explore.search.skipped == 1 and p.explore.target != first
    # Arrival rotates it on.
    tx, ty, _ = p.explore.target
    p.observe(crop_at(tx, ty, t=90.0))
    assert p.explore.search.arrived == 1


def test_the_target_rides_a_reseed_link():
    p, _ = explorer()
    p.poll(Events(cmd("explore")), crop_at(t=0.0))
    x, y, _ = p.explore.target
    p.poll(Events(), crop_at(frame=1, links=(link(1, 0, 1, (1.0, 0.0, math.pi / 2)),), t=0.1))
    assert p.explore.target == pytest.approx((1.0 - y, x, 1))


def test_the_status_and_the_footprint_row_say_explore_on_their_schemas():
    p, _ = explorer(points=SQUARE)
    p.poll(Events(cmd("explore")), crop_at(t=0.0))
    gx, gy, _ = p.goal
    row = p.row_fields()
    assert set(row) == set(pt.ROW_FIELDS)
    assert row["patrol_use"] == "explore" and row["patrol_state"] == "exploring"
    assert set(row) <= set(fp.AUTOPILOT_FIELDS)
    st = p.patrol_status()
    props = tm.JSON_SCHEMAS["patrol_status"]["schema"]["properties"]
    assert set(st) == set(props)
    assert st["use"] == "explore" and st["state"] == "exploring" and not st["running"]
    assert (st["x"], st["y"]) == (gx, gy) and st["on_map"]
    assert st["distance_m"] == pytest.approx(math.hypot(gx, gy))
    g = p.status()
    assert set(g) == set(tm.JSON_SCHEMAS["goal_status"]["schema"]["properties"])
    assert g["goal_set"] and (g["x"], g["y"]) == (gx, gy) and not g["arrived"]
    assert p.marker()["current"] is None, "the list is drawn, none of it lit"
    page = p.page_state()["explore"]
    assert page["on"] and page["kind"] == "frontier"
    assert page["target"] == [round(gx, 3), round(gy, 3)]
    assert page["distance_m"] == pytest.approx(math.hypot(gx, gy), abs=0.01)


def test_on_the_autopilot_explore_draws_its_goal_disc_and_is_what_auto_drives():
    from test_policy_goal import _pilot
    events = Events()
    mn, log, slam, pilot = _pilot(events)
    pilot.patrol.map_view = lambda: HALF_SEEN
    pilot.patrol.explore.search.map_view = lambda: HALF_SEEN
    events.add(cmd("explore"))
    pilot.goal_step()
    g = pilot.goal.goal
    assert pilot.goal.use == "explore" and g is not None
    assert pilot.goal_marker() == {"goal": (g[0], g[1])}
    state = mn.map_page_state(pilot)
    assert state["patrol"]["explore"]["target"] == [round(g[0], 3), round(g[1], 3)]
    assert pilot.resume() is True and pilot.driver.goal is pilot.goal


def test_a_simulated_drive_explores_on_the_goal_input_alone_and_records_it():
    from test_policy_goal import PosedSlam
    from test_policy_map import T0, driver
    slam = PosedSlam()
    model = GoalSteerStub()
    d = driver(model, slam)
    clock = Clock(T0)
    p = pt.Patrol(clock=clock, map_view=lambda: HALF_SEEN)
    d.goal = p
    slam.c = slam.c._replace(t=T0)
    p.poll(Events(cmd("explore")), slam.crop())
    x, y, yaw, t, dt = 0.0, 0.0, 0.0, T0, 0.1
    kinds, rows = set(), []
    while t < T0 + 120.0:
        t = round(t + dt, 6)
        clock.t = t
        slam.c = slam.c._replace(x=x, y=y, yaw=yaw, t=t)
        p.poll(Events(), slam.crop())
        d.tick(t)
        kinds.add(p.explore.kind)
        rows.append(fp._driver_fields(d, t))
        thr = d.controller.throttles[-1] if d.controller.throttles else 0.0
        front = d.controller.steerings[-1][0] if d.controller.steerings else 0.0
        v = 0.8 * thr
        yaw += -v * front * 1.6 * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
    s = p.explore.search
    print(f"\nexplore sim: {s.targets} targets, {s.arrived} reached, {s.skipped} skipped, "
          f"kinds {sorted(k for k in kinds if k)}")
    assert s.arrived >= 2 and "frontier" in kinds and "wander" in kinds
    assert model.fed > 0
    assert all(r["patrol_use"] == "explore" for r in rows)
    assert any(r["patrol_state"] == "exploring" for r in rows)


def test_nothing_in_explore_can_drive():
    text = open(os.path.join(ROOT, "crawler", "autopilot", "explore.py")).read()
    for word in ("set_throttle", "set_steering", "vecon", "controller", "pigpio",
                 "ESC_GPIO_PIN", "return_neutral", "crawler.nav"):
        assert word not in text, word
