"""An ordered list of waypoints the goal model drives through: once, in laps,
or back and forth.

autopilot/patrol.py keeps the list on the SLAM map and hands the model the
current waypoint through the goal input it already has (policy_goal.encode);
nothing in it steers. These hold the editing (append, undo, clear, a traced
path), the three orders, the advance on arrival, the skip of a waypoint that
cannot be reached, the carry across a reseed and the snap back on a merge,
the save across a restart, /patrol/status and /patrol/marker, the commands
from Foxglove, and a simulated drive of a four-point loop through the real
PolicyDriver with a stub model that only reads its goal input.
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler.autopilot import goal as gl  # noqa: E402
from crawler.autopilot import patrol as pt  # noqa: E402
from crawler.autopilot import policy_goal as pg  # noqa: E402
from crawler.mapping import grid as gr  # noqa: E402
from crawler.mapping import runner as rn  # noqa: E402
from crawler.nav import click as ck  # noqa: E402
from crawler.telemetry import patrol_view as pv  # noqa: E402
from crawler.telemetry import telemetry as tm  # noqa: E402

from test_goal_reseed import apply, click, crop_at, link  # noqa: E402
from test_lidar_view import decode  # noqa: E402


class Events:
    """click.ClickGoals as the patrol reads it: everything, in order."""

    def __init__(self, *events):
        self.events = list(events)

    def add(self, *events):
        self.events.extend(events)
        return self

    def take_events(self):
        out, self.events = self.events, []
        return out


def pose(x, y):
    return ("goal", click(x, y))


def cmd(name, **kw):
    return ("cmd", dict(cmd=name, **kw))


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


SQUARE = [(2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)]


def patrol_of(points=SQUARE, mode="loop", start=True, **kw):
    said = []
    clock = kw.pop("clock", Clock())
    p = pt.Patrol(say=lambda *a: said.append(a), clock=clock, **kw)
    ev = Events(cmd("edit"), *[pose(*xy) for xy in points], cmd("mode", mode=mode))
    if start:
        ev.add(cmd("start"))
    p.poll(ev, crop_at(-1.0, -1.0))
    return p, said, clock


def visit(p, xy, frame=0, links=()):
    """The truck at xy: one control pass."""
    p.poll(Events(), crop_at(xy[0], xy[1], frame=frame, links=links))


def order_of(p, n):
    seen = [p.index]
    for _ in range(n):
        target = p.current_place()
        visit(p, target[:2])
        seen.append(p.index)
    return seen


# --- editing ---------------------------------------------------------------------------

def test_clicks_in_edit_append_in_order_undo_drops_the_last_and_clear_empties():
    p, said, _ = patrol_of(start=False)
    assert p.use == "patrol" and [w[:2] for w in p.waypoints()] == SQUARE
    assert p.single.goal is None, "an edit click is a waypoint, never the goal"
    p.poll(Events(cmd("undo")), crop_at())
    assert [w[:2] for w in p.waypoints()] == SQUARE[:3]
    p.poll(Events(cmd("clear")), crop_at())
    assert p.waypoints() == [] and not p.running
    p.poll(Events(cmd("undo")), crop_at())
    assert p.waypoints() == []
    assert any("cleared" in a[2].lower() for a in said)


def test_single_goal_use_is_the_click_goal_and_leaves_the_list_alone():
    p, said, _ = patrol_of(start=False)
    p.poll(Events(cmd("goal"), pose(5.0, 1.0)), crop_at())
    assert p.use == "goal" and p.single.goal == (5.0, 1.0, 0)
    assert len(p.waypoints()) == 4
    assert p.goal == (5.0, 1.0, 0)
    np.testing.assert_allclose(p.encode(crop_at()), pg.encode(crop_at(), (5.0, 1.0, 0)))
    # Back to the patrol: the list is where it was, paused.
    p.poll(Events(cmd("edit")), crop_at())
    assert p.use == "patrol" and not p.running and p.reached()


def test_a_command_it_does_not_know_is_refused_and_changes_nothing():
    p, said, _ = patrol_of(start=False)
    p.poll(Events(cmd("explode"), cmd("mode", mode="zigzag")), crop_at())
    assert p.mode == "loop" and len(p.waypoints()) == 4
    assert sum("not a patrol command" in a[2] or "no mode" in a[2] for a in said) == 2


def test_tracing_drops_a_waypoint_every_metre_driven():
    p = pt.Patrol(clock=Clock())
    p.poll(Events(cmd("trace")), crop_at(0.0, 0.0))
    assert p.tracing and p.use == "patrol"
    for k in range(1, 31):
        visit(p, (k * 0.1, 0.0))
    p.poll(Events(cmd("trace")), crop_at(3.0, 0.0))
    assert not p.tracing
    xs = [w[0] for w in p.waypoints()]
    assert xs == pytest.approx([0.0, 1.0, 2.0, 3.0], abs=0.051)


# --- the orders ------------------------------------------------------------------------

def test_once_goes_in_order_and_holds_at_the_last():
    p, said, _ = patrol_of(mode="once")
    assert not p.reached()
    assert order_of(p, 3) == [0, 1, 2, 3]
    assert not p.reached()
    visit(p, SQUARE[3])
    assert p.reached() and p.state() == "finished" and p.laps == 1
    np.testing.assert_array_equal(p.encode(crop_at()), pg.missing())
    # Start again from the top.
    p.poll(Events(cmd("start")), crop_at(0.0, 0.0))
    assert p.index == 0 and not p.reached()


def test_the_last_waypoint_of_once_wants_the_arrival_radius_the_others_the_pass_one():
    p, _, _ = patrol_of(points=[(1.0, 0.0), (3.0, 0.0)], mode="once")
    visit(p, (1.0 - pt.PASS_M + 0.05, 0.0))
    assert p.index == 1, "passed within PASS_M"
    visit(p, (3.0 - pt.PASS_M + 0.05, 0.0))
    assert not p.reached(), "the last one is the arrival hold's ARRIVE_M"
    visit(p, (3.0 - pg.ARRIVE_M + 0.05, 0.0))
    assert p.reached()


def test_loop_goes_round_and_counts_laps():
    p, said, _ = patrol_of(mode="loop")
    assert order_of(p, 9) == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    assert p.laps == 2 and not p.reached()
    assert sum("lap" in a[2].lower() for a in said) == 2


def test_patrol_goes_there_and_back():
    p, said, _ = patrol_of(mode="patrol")
    assert order_of(p, 8) == [0, 1, 2, 3, 2, 1, 0, 1, 2]
    assert p.laps == 1


def test_one_waypoint_in_a_loop_is_arrived_at_once_not_lapped_forever():
    p, _, _ = patrol_of(points=[(1.0, 0.0)], mode="loop")
    visit(p, (1.0, 0.0))
    visit(p, (1.0, 0.0))
    assert p.reached() and p.laps <= 1


def test_on_arrival_the_model_is_fed_the_next_waypoint_not_a_hold():
    p, _, _ = patrol_of(mode="loop")
    here = crop_at(2.0, 0.0)
    p.poll(Events(), here)
    assert p.index == 1 and not p.reached()
    np.testing.assert_allclose(p.encode(here), pg.encode(here, (2.0, 2.0, 0)))
    assert p.encode(here)[0, pg.SIN] == pytest.approx(1.0), "(2, 2) is on its left"


def test_a_dwell_holds_at_each_waypoint_then_drives_on():
    clock = Clock()
    p, _, _ = patrol_of(mode="loop", dwell_s=2.0, clock=clock)
    visit(p, (2.0, 0.0))
    assert p.reached() and p.state() == "dwell"
    clock.t += 2.1
    visit(p, (2.0, 0.0))
    assert not p.reached() and p.index == 1


def test_paused_holds_and_start_carries_on_from_the_same_waypoint():
    p, _, _ = patrol_of(mode="loop")
    visit(p, (2.0, 0.0))
    p.poll(Events(cmd("pause")), crop_at(1.0, 1.0))
    assert p.reached() and p.state() == "paused"
    np.testing.assert_array_equal(p.encode(crop_at()), pg.missing())
    p.poll(Events(cmd("start")), crop_at(1.0, 1.0))
    assert p.index == 1 and not p.reached()


def test_start_with_no_waypoints_is_refused():
    said = []
    p = pt.Patrol(say=lambda *a: said.append(a), clock=Clock())
    p.poll(Events(cmd("start")), crop_at())
    assert not p.running and any("no waypoints" in a[2] for a in said)


# --- a waypoint it cannot get to -------------------------------------------------------

def drive_ticks(p, clock, seconds, xy=(0.0, 0.0), dt=0.1):
    for _ in range(int(round(seconds / dt))):
        clock.t += dt
        p.observe(crop_at(*xy))


def test_no_progress_for_the_window_skips_the_waypoint_with_a_journal_line():
    p, said, clock = patrol_of(mode="loop", skip_stuck=True)
    drive_ticks(p, clock, pt.NO_PROGRESS_S - 1.0, xy=(-1.0, -1.0))
    assert p.index == 0
    drive_ticks(p, clock, 1.5, xy=(-1.0, -1.0))
    assert p.index == 1 and p.skipped == 1
    assert any("skipped" in a[2].lower() and "waypoint 1" in a[2].lower() for a in said)


def test_by_default_a_stuck_waypoint_is_kept_said_once_and_reported():
    p, said, clock = patrol_of(mode="loop")
    assert not p.skip_stuck
    drive_ticks(p, clock, pt.TIMEOUT_S + 30.0, xy=(-1.0, -1.0))
    assert p.index == 0 and p.skipped == 0 and p.running
    stuck = [a[2] for a in said if "stuck" in a[2]]
    assert len(stuck) == 2 and "kept" in stuck[0], "once at 20 s, once at 120 s"
    st = p.patrol_status()
    assert st["stuck_s"] == pytest.approx(pt.TIMEOUT_S + 30.0, abs=0.3)
    assert st["skip_stuck"] is False and p.page_state()["skip_stuck"] is False
    # Passing it still moves on (PASS_M, unchanged).
    visit(p, SQUARE[0])
    assert p.index == 1 and p.patrol_status()["stuck_s"] == 0.0


def test_the_skip_is_toggled_and_kept_across_a_restart(tmp_path):
    settings = pt.Store(str(tmp_path / "patrol_settings.json"), background=False)
    p, said, clock = patrol_of(mode="loop", settings=settings)
    assert any("kept (skip off, the default)" in a[2] for a in said)
    p.poll(Events(cmd("skip", on=True)), crop_at(-1.0, -1.0))
    assert p.skip_stuck and settings.load() == {"skip_stuck": True}
    drive_ticks(p, clock, pt.NO_PROGRESS_S + 0.5, xy=(-1.0, -1.0))
    assert p.index == 1, "on: skipped as before"
    # Next boot: the toggle wins over the default either way.
    q, said, _ = patrol_of(mode="loop", settings=settings)
    assert q.skip_stuck and any("as last set" in a[2] for a in said)
    q.poll(Events(cmd("skip")), crop_at(-1.0, -1.0))
    assert not q.skip_stuck
    r, _, _ = patrol_of(mode="loop", settings=settings, skip_stuck=True)
    assert not r.skip_stuck
    assert p.command({"cmd": "skip", "on": "yes"}) is False
    # With no settings file, the default given: off unless the flag says.
    assert patrol_of(mode="loop", skip_stuck=True)[0].skip_stuck
    from crawler import __main__ as cm
    assert cm.parse_args([]).patrol_skip_stuck is False
    assert cm.parse_args(["--patrol-skip-stuck"]).patrol_skip_stuck is True


def test_progress_resets_the_window_but_the_timeout_still_skips():
    p, said, clock = patrol_of(points=[(50.0, 0.0), (0.0, 0.0)], mode="loop",
                               skip_stuck=True)
    x, t_end = -1.0, clock.t + pt.TIMEOUT_S + 1.0
    while clock.t < t_end and p.index == 0:
        clock.t += 0.1
        x += 0.02                 # 0.2 m/s: progress all the way, never there
        p.observe(crop_at(x, 0.0))
    assert p.index == 1 and p.skipped == 1
    assert clock.t - 100.0 == pytest.approx(pt.TIMEOUT_S, abs=0.3)


def test_the_clock_runs_only_on_decisions_in_auto_not_on_manual_passes():
    p, said, clock = patrol_of(mode="loop", skip_stuck=True)
    for _ in range(int(pt.NO_PROGRESS_S * 10) + 50):
        clock.t += 0.1
        p.poll(Events(), crop_at(-1.0, -1.0))
    assert p.index == 0 and p.skipped == 0
    # A gap between decisions (auto off for a minute) is not a minute driven.
    p.observe(crop_at(-1.0, -1.0))
    clock.t += 60.0
    p.observe(crop_at(-1.0, -1.0))
    assert p.index == 0 and p.skipped == 0


def test_every_waypoint_skipped_in_a_row_pauses_the_patrol():
    p, said, clock = patrol_of(mode="loop", skip_stuck=True)
    drive_ticks(p, clock, 4 * (pt.NO_PROGRESS_S + 0.5) + 1.0, xy=(-5.0, -5.0))
    assert not p.running and p.skipped == 4 and p.reached()
    assert any("every waypoint" in a[2].lower() for a in said)


# --- across a reseed -------------------------------------------------------------------

def test_a_reseed_carries_every_waypoint_and_a_merge_snaps_them_back():
    p, said, _ = patrol_of(mode="loop")
    said.clear()
    T = (0.5, -0.2, math.radians(20))
    lost = link(1, 0, 1, T, err_m=0.3, err_yaw=math.radians(5), at=(1.0, 1.0))
    visit(p, (0.0, 0.0), frame=1, links=[lost])
    st = p.patrol_status()
    assert st["frame_now"] == 1 and st["on_map"] and st["approx"]
    for (x, y), w in zip(SQUARE, p.waypoints()):
        assert w[2] == 1 and w[:2] == pytest.approx(apply(T, (x, y)))
    want = pg.encode(crop_at(frame=1), (*apply(T, SQUARE[0]), 1))
    np.testing.assert_allclose(p.encode(crop_at(frame=1)), want)
    carried = [a[2] for a in said if "carried" in a[2]]
    assert len(carried) == 1 and "4 waypoints" in carried[0], "one line, not four"
    back = link(2, 1, 0, (-0.55, 0.25, -math.radians(19)), why="merged")
    visit(p, (0.0, 0.0), frame=0, links=[lost, back])
    assert [w[:2] for w in p.waypoints()] == SQUARE
    assert not p.patrol_status()["approx"]
    assert len([a for a in said if "exactly" in a[2]]) == 1


def test_a_waypoint_added_on_the_fresh_map_rides_the_merge_onto_the_old_one():
    p, _, _ = patrol_of(points=[(1.0, 0.0)], mode="loop", start=False)
    lost = link(1, 0, 1, (0.0, 0.0, 0.0), err_m=0.2)
    visit(p, (0.0, 0.0), frame=1, links=[lost])
    p.poll(Events(pose(3.0, 0.0)), crop_at(frame=1, links=[lost]))
    T = (1.0, 0.0, 0.0)
    m = link(2, 1, 0, T, why="merged")
    visit(p, (0.0, 0.0), frame=0, links=[lost, m])
    assert [w[:2] for w in p.waypoints()] == [(1.0, 0.0), pytest.approx((4.0, 0.0))]


def test_a_reseed_does_not_burn_the_clock_while_the_waypoint_is_off_the_map():
    p, said, clock = patrol_of(mode="loop", skip_stuck=True)
    drive_ticks(p, clock, pt.NO_PROGRESS_S - 2.0, xy=(-1.0, -1.0))
    # Lost: the crop is not present, the model gets no goal, nothing counts.
    for _ in range(100):
        clock.t += 0.1
        p.observe(crop_at(-1.0, -1.0, present=False))
    assert p.index == 0


# --- across a restart ------------------------------------------------------------------

def room_view(shift=(0.0, 0.0), yaw=0.0):
    """A MapView of a 6 x 4 m room with furniture off to the sides, drawn
    round the boot pose, turned by yaw and moved by shift."""
    from crawler.mapping import trackfit as tf
    res = 0.1
    walls, free = [], []
    for i in range(60):
        for j in range(40):
            p = (-2.0 + (i + 0.5) * res, -2.0 + (j + 0.5) * res)
            edge = i in (0, 59) or j in (0, 39) or (i == 12 and 8 <= j < 28) \
                or (j == 30 and 40 <= i < 52) or (i == 50 and 2 <= j < 18) \
                or (j == 8 and 20 <= i < 35)
            (walls if edge else free).append(p)
    w = tf.transform((yaw,) + tuple(shift), walls)
    f = tf.transform((yaw,) + tuple(shift), free)
    x0, y0 = np.concatenate([w, f]).min(0) - 0.5
    codes = np.zeros((90, 90), np.uint8)
    for pts, code in ((f, gr.FREE), (w, gr.OCCUPIED)):
        i, j = np.floor((pts - (x0, y0)) / res).astype(int).T
        codes[i, j] = code
    return rn.MapView(t=0.0, version=1, x0=float(x0), y0=float(y0), res=res, codes=codes)


def saved_patrol(tmp_path, view=None):
    store = pt.Store(str(tmp_path / "patrol.json"), background=False)
    p, said, _ = patrol_of(mode="patrol", start=False, store=store,
                           map_view=lambda: view or room_view())
    visit(p, (0.0, 0.0))
    return store, p


def test_the_list_is_saved_and_comes_back_on_the_same_map(tmp_path):
    store, _ = saved_patrol(tmp_path)
    data = json.loads((tmp_path / "patrol.json").read_text())
    assert data["version"] == 2 and len(data["points"]) == 4 and data["walls"]
    said = []
    q = pt.Patrol(say=lambda *a: said.append(a), store=store, map_view=room_view,
                  clock=Clock())
    assert q.waypoints() == [], "nothing until it is fitted"
    visit(q, (0.3, 0.1))
    got = [w[:2] for w in q.waypoints()]
    assert np.allclose(got, SQUARE, atol=0.1) and q.mode == "patrol"
    assert q.use == "patrol" and not q.running, "placed paused: Start drives it"
    assert q.fit_state["state"] == "placed"
    assert any("placed" in a[2] for a in said)


def test_a_restart_with_the_map_turned_and_moved_puts_the_list_where_it_was_in_the_room(
        tmp_path):
    from crawler.mapping import trackfit as tf
    store, _ = saved_patrol(tmp_path)
    boot = (math.radians(-75), 1.5, -2.0)     # where the old origin is on the new map
    q = pt.Patrol(store=store, map_view=lambda: room_view(shift=boot[1:], yaw=boot[0]),
                  clock=Clock())
    visit(q, (0.0, 0.0))
    got = np.array([w[:2] for w in q.waypoints()])
    assert np.allclose(got, tf.transform(boot, SQUARE), atol=0.12)


def test_a_restart_somewhere_else_keeps_the_track_unplaced_and_the_file_as_it_was(tmp_path):
    store, _ = saved_patrol(tmp_path)
    before = (tmp_path / "patrol.json").read_text()
    said = []
    clock = Clock()
    elsewhere = rn.MapView(t=0.0, version=1, x0=-3.0, y0=-3.0, res=0.1,
                           codes=np.pad(np.full((58, 58), gr.FREE, np.uint8), 1,
                                        constant_values=gr.OCCUPIED))
    q = pt.Patrol(say=lambda *a: said.append(a), store=store, map_view=lambda: elsewhere,
                  clock=clock)
    for _ in range(5):
        visit(q, (0.0, 0.0))
        clock.t += pt.FIT_STALE_S
    assert q.waypoints() == [] and q.fit_state["state"] == "unsure"
    assert q.fit_state["tries"] >= 2, "tried again"
    assert q.page_state()["pending"]["points"] == [list(xy) for xy in SQUARE]
    assert (tmp_path / "patrol.json").read_text() == before, "never written over"
    assert sum("not placed yet" in a[2] for a in said) == 1, "said once, not every try"


def test_an_unplaced_track_is_placed_by_hand_through_the_transform_given(tmp_path):
    store, _ = saved_patrol(tmp_path)
    q = pt.Patrol(store=store, map_view=lambda: None, clock=Clock())
    visit(q, (0.0, 0.0))
    assert q.waypoints() == []
    q.poll(Events(cmd("place", yaw=math.pi / 2, x=1.0, y=0.0)), crop_at(0.0, 0.0))
    got = [w[:2] for w in q.waypoints()]
    assert np.allclose(got, [(1.0, 2.0), (-1.0, 2.0), (-1.0, 0.0), (1.0, 0.0)])
    assert q.fit_state["state"] == "placed" and not q.running
    assert json.loads((tmp_path / "patrol.json").read_text())["points"] == [
        list(map(float, xy)) for xy in got]


def test_a_version_1_file_is_read_and_one_on_another_frame_keeps_its_points(tmp_path):
    view = room_view()
    from crawler.autopilot.patrol import _occupied_points
    walls = _occupied_points(view).tolist()
    path = tmp_path / "patrol.json"
    path.write_text(json.dumps({"version": 1, "frame": 0, "mode": "loop",
                                "points": [list(xy) for xy in SQUARE], "walls": walls}))
    q = pt.Patrol(store=pt.Store(str(path), background=False), map_view=lambda: view,
                  clock=Clock())
    visit(q, (0.0, 0.0))
    assert np.allclose([w[:2] for w in q.waypoints()], SQUARE, atol=0.1)
    path.write_text(json.dumps({"version": 1, "frame": 3, "mode": "loop",
                                "points": [list(xy) for xy in SQUARE], "walls": walls}))
    q = pt.Patrol(store=pt.Store(str(path), background=False), map_view=lambda: view,
                  clock=Clock())
    visit(q, (0.0, 0.0))
    assert q.waypoints() == [] and q.fit_state["state"] == "no walls"
    assert len(q.page_state()["pending"]["points"]) == 4, "kept, to place by hand"


def test_a_store_that_cannot_be_read_is_an_empty_patrol(tmp_path):
    (tmp_path / "patrol.json").write_text("{not json")
    store = pt.Store(str(tmp_path / "patrol.json"), background=False)
    q = pt.Patrol(store=store, map_view=room_view, clock=Clock())
    visit(q, (0.0, 0.0))
    assert q.waypoints() == []


def test_the_walls_ride_a_reseed_with_the_waypoints(tmp_path):
    store, p = saved_patrol(tmp_path)
    lost = link(1, 0, 1, (2.0, 1.0, math.pi / 2), err_m=0.1)
    visit(p, (0.0, 0.0), frame=1, links=[lost])
    data = json.loads((tmp_path / "patrol.json").read_text())
    old = room_view()
    from crawler.autopilot.patrol import _occupied_points
    from crawler.mapping import trackfit as tf
    want = tf.transform((math.pi / 2, 2.0, 1.0), _occupied_points(old))
    assert np.allclose(np.array(data["walls"])[:5], want[:5], atol=1e-3)
    assert np.allclose(data["points"], tf.transform((math.pi / 2, 2.0, 1.0), SQUARE))


# --- telemetry -------------------------------------------------------------------------

def test_the_status_fits_its_schema_on_its_topic():
    p, _, _ = patrol_of(mode="loop")
    visit(p, (2.0, 0.0))
    st = p.patrol_status()
    props = tm.JSON_SCHEMAS["patrol_status"]["schema"]["properties"]
    assert set(st) == set(props)
    assert tm.JSON_TOPICS["patrol_status"] == "/patrol/status"
    assert st["mode"] == "loop" and st["index"] == 1 and st["count"] == 4
    assert (st["x"], st["y"]) == (2.0, 2.0) and st["distance_m"] == pytest.approx(2.0)
    assert st["number"] == 2, "the one a person counts, from 1"
    assert (st["truck_x"], st["truck_y"]) == (2.0, 0.0), "the pose it is measured from"
    for k, v in st.items():
        kind = props[k]["type"]
        ok = {"string": str, "boolean": bool, "integer": int, "number": float}[kind]
        assert isinstance(v, ok), k
    # /goal/status is still the goal the model is fed, in its own schema.
    g = p.status()
    assert set(g) == set(tm.JSON_SCHEMAS["goal_status"]["schema"]["properties"])
    assert (g["x"], g["y"]) == (2.0, 2.0) and g["goal_set"]


def entities(update):
    return {e.id: e for e in decode(update).entities}


def test_the_markers_number_the_waypoints_join_them_in_order_and_light_the_current_one():
    assert tm.NAV_CHANNELS["patrol_marker"] == "/patrol/marker"
    p, _, _ = patrol_of(mode="loop")
    visit(p, (2.0, 0.0))
    scene = pv.scene(p.marker(), 0)
    ents = entities(scene)
    texts = sorted(t.text for e in ents.values() for t in e.texts)
    assert texts == ["1", "2", "3", "4"]
    line = ents[pv.LINE_ID].lines[0]
    pts = [(q.x, q.y) for q in line.points]
    assert pts == SQUARE + [SQUARE[0]], "a loop closes"
    # What the publisher thread sends on /patrol/marker.
    sent = entities(tm.Telemetry.lidar_view_message("patrol_marker", p.marker(), 0))
    assert set(sent) == set(ents)
    cur = ents[pv.CURRENT_ID].cylinders[0]
    assert (cur.pose.position.x, cur.pose.position.y) == (2.0, 2.0)
    assert cur.size.x > ents[pv.point_id(0)].cylinders[0].size.x
    assert cur.color.b != ents[pv.point_id(0)].cylinders[0].color.b


def test_once_and_patrol_lines_do_not_close_and_a_carried_list_draws_its_disc():
    p, _, _ = patrol_of(mode="patrol")
    line = entities(pv.scene(p.marker(), 0))[pv.LINE_ID].lines[0]
    assert len(line.points) == 4
    lost = link(1, 0, 1, (0.0, 0.0, 0.0), err_m=0.8)
    visit(p, (0.0, 0.0), frame=1, links=[lost])
    ents = entities(pv.scene(p.marker(), 0))
    assert ents[pv.APPROX_ID].cylinders[0].size.x == pytest.approx(2 * 0.8 + 2 * 0.0, abs=0.2)


def test_an_empty_patrol_clears_every_marker_it_drew():
    p, _, _ = patrol_of(mode="loop")
    first = pv.scene(p.marker(), 0)
    p.poll(Events(cmd("clear")), crop_at())
    scene = decode(pv.scene(p.marker(), 0))
    assert len(scene.entities) == 0
    gone = {d.id for d in scene.deletions}
    assert set(entities(first)) <= gone


def test_a_list_on_another_map_frame_is_not_drawn():
    p, _, _ = patrol_of(mode="loop")
    visit(p, (0.0, 0.0), frame=5)
    assert len(decode(pv.scene(p.marker(), 0)).entities) == 0


# --- the commands from Foxglove ----------------------------------------------------------

def test_the_listener_queues_clicks_and_commands_in_the_order_they_came():
    listener = ck.ClickGoals()
    c = SimpleNamespace(id=1)
    listener.on_client_advertise(c, SimpleNamespace(
        id=1, topic="/move_base_simple/goal", schema_name="foxglove.PoseInFrame",
        encoding="json"))
    listener.on_client_advertise(c, SimpleNamespace(
        id=2, topic=ck.PATROL_TOPIC, schema_name="crawler.PatrolCommand", encoding="json"))

    def send(ch, msg):
        listener.on_message_data(c, ch, json.dumps(msg).encode())

    pose_msg = {"frame_id": "map", "pose": {"position": {"x": 1.0, "y": 2.0, "z": 0.0},
                                            "orientation": {"w": 1.0}}}
    send(2, {"cmd": "edit"})
    send(1, pose_msg)
    send(1, dict(pose_msg, pose={"position": {"x": 3.0, "y": 4.0}}))
    send(2, {"cmd": "start"})
    send(2, {"nothing": 1})
    events = listener.take_events()
    assert [e[0] for e in events] == ["cmd", "goal", "goal", "cmd"]
    assert events[0][1]["cmd"] == "edit" and (events[2][1].x, events[2][1].y) == (3.0, 4.0)
    assert listener.take_events() == []
    assert listener.rejected == 1
    # The navigator's take() still gets the newest click.
    assert listener.take().x == 3.0


def test_the_drive_layout_has_a_button_for_every_patrol_command_and_draws_the_list():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    layout = json.load(open(os.path.join(root, "deploy", "foxglove-drive.json")))
    cfg = layout["configById"]
    # The Follow buttons are on /follow/cmd (test_follow.py checks them).
    buttons = {k: v for k, v in cfg.items() if k.startswith("Publish!")
               and v["topicName"] != ck.FOLLOW_TOPIC}
    sent = set()
    for key, b in buttons.items():
        assert b["topicName"] == ck.PATROL_TOPIC, key
        msg = json.loads(b["value"])
        assert ck.parse_command(json.dumps(msg).encode()) is not None, key
        assert pt.Patrol(clock=Clock()).command(msg) is not None, key
        sent.add(msg["cmd"])
    assert sent == set(pt.BUTTON_COMMANDS)
    assert cfg["3D!slam"]["topics"]["/patrol/marker"]["visible"]


# --- crawler/__main__.py ---------------------------------------------------------------

def test_with_goal_the_autopilot_keeps_a_patrol_and_a_patrol_draws_no_goal_disc():
    from test_policy_goal import _pilot
    events = Events(pose(2.0, 0.0))
    mn, log, slam, pilot = _pilot(events)
    assert isinstance(pilot.goal, pt.Patrol)
    pilot.goal_step()
    assert pilot.goal_marker() == {"goal": (2.0, 0.0)}
    events.add(cmd("edit"), pose(1.0, 0.0))
    pilot.goal_step()
    assert pilot.goal_marker() == {}, "the patrol's own markers draw it"
    assert pilot.resume() is True and pilot.driver.goal is pilot.goal


# --- a simulated drive -----------------------------------------------------------------

class GoalSteerStub:
    """A stand-in for a goal model: forward, steering at the goal it is fed.
    It reads nothing but its goal input, as the real one reads its inputs."""

    contract = None
    representation = None
    has_mode = False
    uses_prev = False
    uses_imu = False
    uses_lidar = True
    uses_map = True
    uses_goal = True

    def __init__(self):
        from crawler.autopilot import policy as pl
        self.contract, self.representation = pl.CONTRACT_DEPTH_SMALL, pl.PROFILE
        self.fed = 0

    def run(self, x, prev, mode=None, imu=None, lidar=None, map=None, goal=None):
        g = goal[0]
        self.fed += int(g[pg.PRESENT] > 0)
        bearing = math.atan2(g[pg.SIN], g[pg.COS]) if g[pg.PRESENT] > 0 else 0.0
        steer = float(np.clip(-1.5 * bearing, -1.0, 1.0))   # positive is right
        return {"logits": np.array([[5.0, 0.0, 0.0]], np.float32),
                "throttle": np.array([[0.6]], np.float32),
                "steer": np.array([[steer]], np.float32)}


def simulate(seconds, reseed_at=None, merge_at=None, T=(0.4, -0.3, math.radians(15))):
    """The truck as a unicycle under the real PolicyDriver at 10 Hz, on a
    4-point loop. The SLAM crop is the true pose, or after reseed_at the
    true pose on a fresh map frame 1 related by T, which the patrol is told
    with err_m 0.3 off (it carries the waypoints by a link 0.1 m wrong),
    and from merge_at back on frame 0."""
    from test_policy_goal import PosedSlam
    from test_policy_map import T0, driver
    slam = PosedSlam()
    model = GoalSteerStub()
    d = driver(model, slam)
    clock = Clock(T0)
    p = pt.Patrol(clock=clock)
    d.goal = p
    p.poll(Events(cmd("edit"), *[pose(*xy) for xy in SQUARE], cmd("mode", mode="loop"),
                  cmd("start")), slam.crop())
    x, y, yaw = -0.5, -0.5, 0.0
    t, dt = T0, 0.1
    lost = link(1, 0, 1, (T[0] + 0.1, T[1], T[2]), err_m=0.3, err_yaw=math.radians(3))
    merged = link(2, 1, 0, (-T[0], 0.0, -T[2]), why="merged")   # snaps back anyway
    min_d = [math.inf] * len(SQUARE)
    lap_times = []
    laps = 0
    while t < T0 + seconds:
        t = round(t + dt, 6)
        clock.t = t
        frame, links, (cx, cy, cyaw) = 0, (), (x, y, yaw)
        if reseed_at is not None and t >= T0 + reseed_at:
            frame, links = 1, (lost,)
            cx, cy = apply(T, (x, y))
            cyaw = yaw + T[2]
        if merge_at is not None and t >= T0 + merge_at:
            frame, links, (cx, cy, cyaw) = 0, (lost, merged), (x, y, yaw)
        slam.c = slam.c._replace(x=cx, y=cy, yaw=cyaw, frame=frame, links=links, t=t)
        p.poll(Events(), slam.crop())
        d.tick(t)
        thr = d.controller.throttles[-1] if d.controller.throttles else 0.0
        front = d.controller.steerings[-1][0] if d.controller.steerings else 0.0
        v = 0.8 * thr             # 0.24, auto's forward, is about 0.2 m/s
        yaw += -v * front * 1.6 * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        for k, (wx, wy) in enumerate(SQUARE):
            min_d[k] = min(min_d[k], math.hypot(x - wx, y - wy))
        if p.laps > laps:
            laps = p.laps
            lap_times.append(t - T0)
    return SimpleNamespace(patrol=p, laps=p.laps, skipped=p.skipped, min_d=min_d,
                           lap_times=lap_times, fed=model.fed, driver=d)


def test_a_simulated_drive_laps_a_four_point_loop_on_the_goal_input_alone():
    r = simulate(180.0)
    print(f"\nsim: {r.laps} laps in 180 s, lap ends at {[round(s, 1) for s in r.lap_times]} s, "
          f"closest pass per waypoint {[round(v, 2) for v in r.min_d]} m, "
          f"{r.skipped} skipped")
    assert r.laps >= 3 and r.skipped == 0
    assert max(r.min_d) <= pt.PASS_M
    assert r.fed > 0 and r.driver.policy_fields()["policy_goal_present"]


def test_a_simulated_drive_keeps_lapping_through_a_reseed_and_a_merge():
    r = simulate(180.0, reseed_at=20.0, merge_at=60.0)
    print(f"\nsim with a reseed at 20 s and a merge at 60 s: {r.laps} laps, "
          f"lap ends at {[round(s, 1) for s in r.lap_times]} s, closest pass "
          f"{[round(v, 2) for v in r.min_d]} m, {r.skipped} skipped")
    assert r.laps >= 3 and r.skipped == 0
    assert not r.patrol.patrol_status()["approx"]
