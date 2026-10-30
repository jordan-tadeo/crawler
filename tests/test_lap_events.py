"""What the car logs for the lap report (tools/lap_report.py): a
/patrol/event the decision a waypoint is passed or skipped, the patrol
started or paused; a track id that changes with every edit of the list and
is the same for a saved track on every boot; and the patrol's place on
every /footprint row."""

from types import SimpleNamespace

import pytest

pytest.importorskip("scipy")

from crawler.autopilot import follow as fo  # noqa: E402
from crawler.autopilot import footprint as fp  # noqa: E402
from crawler.autopilot import patrol as pt  # noqa: E402
from crawler.autopilot import tracks as tk  # noqa: E402
from crawler.telemetry import telemetry as tm  # noqa: E402

from test_goal_reseed import crop_at  # noqa: E402
from test_patrol import SQUARE, Clock, Events, drive_ticks, patrol_of, visit  # noqa: E402
from test_tracks import cmd, lib, run  # noqa: E402
from test_tracks import patrol as library_patrol  # noqa: E402


def events(p):
    out = []
    while True:
        e = p.pop_event()
        if e is None:
            return out
        out.append(e)


def test_each_pass_is_an_event_at_its_decision_with_the_lap_after_it():
    p, _, clock = patrol_of(mode="loop")
    first = events(p)
    assert [(e["kind"], e["by"], e["index"]) for e in first] == [("start", "fresh", 0)]
    got = []
    for i in range(5):
        clock.t += 1.0
        visit(p, SQUARE[i % 4])
        got += events(p)
    assert [(e["kind"], e["index"], e["next"], e["lap"]) for e in got] == [
        ("pass", 0, 1, 0), ("pass", 1, 2, 0), ("pass", 2, 3, 0), ("pass", 3, 0, 1),
        ("pass", 0, 1, 1)]
    assert [e["mono_s"] for e in got] == [101.0, 102.0, 103.0, 104.0, 105.0]
    assert [e["seq"] for e in first + got] == [1, 2, 3, 4, 5, 6]
    assert all(e["by"] == "radius" and e["count"] == 4 and e["mode"] == "loop"
               and e["track"] == p.track_id() for e in got)
    assert p.pop_event() is None


def test_a_skip_and_a_path_pass_say_so_and_carry_the_waypoint_clock():
    p, _, clock = patrol_of(mode="loop", skip_stuck=True)
    events(p)
    drive_ticks(p, clock, pt.NO_PROGRESS_S + 0.5, xy=(-1.0, -1.0))
    (skip,) = events(p)
    assert (skip["kind"], skip["by"], skip["index"], skip["next"]) == ("skip", "stuck", 0, 1)
    assert skip["stuck_s"] == pytest.approx(pt.NO_PROGRESS_S, abs=0.11)
    assert skip["wp_s"] == pytest.approx(pt.NO_PROGRESS_S, abs=0.11)
    assert p.passed(p.goal, crop_at(2.0, 1.0))
    (by_path,) = events(p)
    assert (by_path["kind"], by_path["by"], by_path["index"]) == ("pass", "path", 1)


def test_pause_and_resume_are_events_and_a_finished_once_says_finish():
    p, _, clock = patrol_of(mode="once")
    events(p)
    p.poll(Events(("cmd", {"cmd": "pause"})), crop_at(-1.0, -1.0))
    p.poll(Events(("cmd", {"cmd": "pause"})), crop_at(-1.0, -1.0))
    p.poll(Events(("cmd", {"cmd": "start"})), crop_at(-1.0, -1.0))
    p.poll(Events(("cmd", {"cmd": "start"})), crop_at(-1.0, -1.0))
    assert [(e["kind"], e["by"]) for e in events(p)] == [("pause", "pause"),
                                                         ("start", "resume")]
    for xy in SQUARE:
        visit(p, xy)
    assert [e["kind"] for e in events(p)] == ["pass", "pass", "pass", "finish"]


def test_every_edit_of_the_list_is_a_new_track_id_and_a_clear_starts_again():
    p, _, _ = patrol_of(start=False)
    a = p.track_id()
    assert a.startswith("unsaved@n") and a.endswith("+4")
    p.poll(Events(("cmd", {"cmd": "undo"})), crop_at())
    b = p.track_id()
    assert b != a and b.split("+")[0] == a.split("+")[0]
    p.poll(Events(("cmd", {"cmd": "mode", "mode": "patrol"})), crop_at())
    assert p.track_id() not in (a, b)
    p.poll(Events(("cmd", {"cmd": "clear"})), crop_at())
    assert p.track_id() == ""
    q, _, _ = patrol_of(start=False)
    assert q.track_id().split("+")[0] != a.split("+")[0], "two fresh lists never share one"


def test_a_saved_track_has_the_same_id_on_every_boot_until_it_is_edited(tmp_path):
    p, _ = library_patrol(tmp_path)
    run(p, cmd("stroke", points=SQUARE))
    run(p, cmd("save", name="kitchen"))
    saved = p.track_id()
    assert saved == f"kitchen@{tk.base_of(lib(tmp_path).load('kitchen'))}"
    assert lib(tmp_path).load("kitchen")["base"] == saved.split("@")[1]
    ids = []
    for boot in (tmp_path / "a", tmp_path / "b"):
        boot.mkdir()
        (boot / "tracks").symlink_to(tmp_path / "tracks")
        q, _ = library_patrol(boot)
        run(q, cmd("load", name="kitchen"))
        visit(q, (0.0, 0.0))
        assert q.fit_state["state"] == "placed"
        ids.append(q.track_id())
    assert ids == [saved, saved]
    run(q, cmd("move", i=1, x=2.2, y=2.0))
    edited = q.track_id()
    assert edited == saved + "+1"
    # The working file keeps the edited id across a restart.
    r = pt.Patrol(store=q.store, clock=Clock())
    assert r.track_id() == edited


def test_a_track_saved_before_ids_gets_one_off_its_points():
    old = {"version": 2, "name": "hall", "mode": "loop", "points": [[1, 2], [3, 4]],
           "walls": [], "saved_at": 1.0}
    t = tk.migrate(old)
    assert t["base"] is None and t["edits"] == 0
    assert tk.base_of(t) == tk.base_of(tk.migrate(dict(old, saved_at=9.0)))
    assert tk.base_of(t) != tk.base_of(dict(t, points=[[1, 2], [3, 4.1]]))
    assert tk.migrate(dict(old, base="abc123", edits=2))["edits"] == 2
    assert tk.migrate(dict(old, edits=True))["edits"] == 0


def test_every_footprint_row_says_where_the_patrol_is():
    p, _, clock = patrol_of(mode="loop")
    visit(p, SQUARE[0])
    driver = SimpleNamespace(goal=p, policy_fields=lambda: {})
    row = fp.autopilot_fields(driver, 0.0)
    assert {k: row[k] for k in pt.ROW_FIELDS} == {
        "patrol_use": "patrol", "patrol_state": "driving", "patrol_track": p.track_id(),
        "patrol_index": 1, "patrol_count": 4, "patrol_lap": 0, "patrol_stuck_s": 0.0}
    assert {k: fp.autopilot_fields(None)[k] for k in pt.ROW_FIELDS} == pt.ROW_FIELDS
    follow = fo.FollowSwitch(p, SimpleNamespace(active=True))
    assert follow.row_fields()["patrol_use"] == "follow"
    events(p)
    visit(p, SQUARE[1])
    assert follow.pop_event()["index"] == 1


def test_the_event_and_the_status_fit_their_schemas():
    p, _, _ = patrol_of(mode="loop")
    visit(p, SQUARE[0])
    props = tm.JSON_SCHEMAS["patrol_event"]["schema"]["properties"]
    for e in events(p):
        assert set(e) == set(props)
    assert tm.JSON_TOPICS["patrol_event"] == "/patrol/event"
    assert tm.INTERVALS["patrol_event"] == 0.0
    assert p.patrol_status()["track"] == p.track_id()
    assert set(p.patrol_status()) <= set(tm.JSON_SCHEMAS["patrol_status"]["schema"]["properties"])
    assert set(pt.ROW_FIELDS) <= set(tm.JSON_SCHEMAS["footprint"]["schema"]["properties"])


def test_the_control_loop_sends_each_event_once_stamped_at_its_decision(monkeypatch):
    import time

    import test_main

    class Tele(test_main.FakeTele):
        def offer_now(self, key, payload, now):
            if key not in self.channels:
                return False
            self.offers.append((key, dict(payload, at=now)))
            return True

    p, _, clock = patrol_of(mode="loop")
    events(p)
    clock.t = time.monotonic() - 0.2      # the patrol's clock is the loop's
    passed_at = clock.t
    visit(p, SQUARE[0])
    visit(p, SQUARE[1])

    class Pilot(test_main.FakeAutopilot):
        clicks = object()
        goal = p

    monkeypatch.setattr(test_main, "FakeAutopilot", Pilot)
    _, _, _, tele = test_main.run_loop(test_main.FakePad(), test_main.FakeLogger(), [(), (), ()],
                                       monkeypatch=monkeypatch,
                                       tele=Tele(channels=("patrol_event",)))
    sent = tele.of("patrol_event")
    assert [(e["index"], e["at"]) for e in sent] == [(0, passed_at), (1, passed_at)]
    assert p.pop_event() is None
