"""Named tracks (autopilot/tracks.py) and the patrol's map-page editing:
drawn strokes, undoing one, moving and deleting a waypoint, save, load and
delete by name, and the rule that nothing is ever lost without the owner
asking: a track about to be replaced is kept in the library first."""

import json
import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler.autopilot import patrol as pt  # noqa: E402
from crawler.autopilot import tracks as tk  # noqa: E402

from test_goal_reseed import crop_at  # noqa: E402
from test_patrol import SQUARE, Clock, Events, room_view, visit  # noqa: E402


def cmd(what, **kw):
    return ("cmd", dict(kw, cmd=what))


def lib(tmp_path):
    return tk.Library(str(tmp_path / "tracks"), synchronous=True)


def patrol(tmp_path, **kw):
    said = []
    store = pt.Store(str(tmp_path / "patrol.json"), background=False)
    p = pt.Patrol(say=lambda *a: said.append(a), clock=Clock(), store=store,
                  library=lib(tmp_path), map_view=room_view, **kw)
    visit(p, (0.0, 0.0))
    return p, said


def run(p, *events):
    p.poll(Events(*events), crop_at(0.0, 0.0))


def xy(p):
    return [tuple(round(v, 6) for v in w[:2]) for w in p.waypoints()]


# --- the library -----------------------------------------------------------------------

def test_names_are_cleaned_and_two_names_never_share_a_file(tmp_path):
    assert tk.clean_name("  living   room\tloop ") == "living room loop"
    assert tk.clean_name("   ") is None and tk.clean_name(None) is None
    assert tk.file_of("Loop!") != tk.file_of("loop?")
    assert tk.file_of("../../etc/passwd").startswith("etc_passwd")
    lb = lib(tmp_path)
    assert lb.save("Loop!", {"points": [[1, 2]], "walls": []})
    assert lb.save("loop?", {"points": [[3, 4]], "walls": []})
    assert lb.load("Loop!")["points"] == [[1.0, 2.0]]
    assert lb.load("loop?")["points"] == [[3.0, 4.0]]
    assert [r["name"] for r in lb.names()] == ["loop?", "Loop!"], "newest first"
    assert lb.delete("Loop!") and lb.load("Loop!") is None and not lb.delete("Loop!")


def test_migrate_reads_version_1_and_drops_walls_off_map_0():
    v1 = {"version": 1, "frame": 0, "mode": "patrol", "points": [[1, 2], [3, "x"]],
          "walls": [[0, 0]]}
    t = tk.migrate(v1)
    assert t["version"] == 2 and t["mode"] == "patrol" and t["points"] == [[1.0, 2.0]]
    assert t["walls"] == [[0.0, 0.0]] and t["name"] is None
    assert tk.migrate(dict(v1, frame=2))["walls"] == []
    assert tk.migrate("nope") is None and tk.migrate(None) is None


# --- drawing --------------------------------------------------------------------------

def test_a_stroke_appends_its_points_as_one_step_and_makes_a_loop(tmp_path):
    p, _ = patrol(tmp_path)
    run(p, cmd("mode", mode="once"), cmd("stroke", points=[[0, 0], [1, 0], [2, 0]]))
    assert xy(p) == [(0, 0), (1, 0), (2, 0)] and p.mode == "loop" and p.use == "patrol"
    assert not p.running, "drawing never starts it"
    run(p, cmd("stroke", points=[[2, 1], [2, 2]]))
    assert len(p.points) == 5 and p.marker()["closed"], "the loop closes 5 -> 1"
    run(p, cmd("unstroke"))
    assert xy(p) == [(0, 0), (1, 0), (2, 0)]
    run(p, cmd("unstroke"))
    assert xy(p) == []
    assert p.command({"cmd": "unstroke"}) is False


def test_a_stroke_of_junk_is_refused_whole(tmp_path):
    p, _ = patrol(tmp_path)
    assert p.command({"cmd": "stroke", "points": [[0, 0], [1, "a"]]}, crop_at()) is False
    assert p.command({"cmd": "stroke", "points": [[0, 0], [math.inf, 0]]}, crop_at()) is False
    assert p.command({"cmd": "stroke", "points": "0,0"}, crop_at()) is False
    assert p.points == []


def test_a_stroke_stops_at_the_most_waypoints(tmp_path):
    p, _ = patrol(tmp_path)
    run(p, cmd("stroke", points=[[i * 0.1, 0] for i in range(pt.MAX_WAYPOINTS + 7)]))
    assert len(p.points) == pt.MAX_WAYPOINTS


def test_move_and_delete_take_an_index_and_keep_the_place_in_the_list(tmp_path):
    p, _ = patrol(tmp_path)
    run(p, cmd("stroke", points=SQUARE), cmd("start"))
    p.index = 2
    run(p, cmd("move", i=1, x=2.5, y=2.5))
    assert xy(p)[1] == (2.5, 2.5) and p.index == 2
    run(p, cmd("delete", i=0))
    assert len(p.points) == 3 and p.index == 1, "still on the same waypoint"
    assert xy(p)[p.index] == (0.0, 2.0)
    for bad in ({"i": 7}, {"i": -1}, {"i": "0"}, {"i": True}, {}):
        assert p.command(dict(cmd="delete", **bad), crop_at()) is False
    assert p.command({"cmd": "move", "i": 0, "x": "a", "y": 0}, crop_at()) is False
    run(p, cmd("unstroke"))
    assert xy(p) == [], "the stroke went down by the one deleted"


def test_nudging_a_placed_track_moves_every_waypoint_by_the_transform(tmp_path):
    p, _ = patrol(tmp_path)
    run(p, cmd("stroke", points=SQUARE))
    run(p, cmd("place", yaw=math.pi, x=1.0, y=1.0))
    assert np.allclose(xy(p), [(-1, 1), (-1, -1), (1, -1), (1, 1)])


# --- saved tracks ---------------------------------------------------------------------

def test_save_load_and_forget_by_name(tmp_path):
    p, said = patrol(tmp_path)
    run(p, cmd("stroke", points=SQUARE))
    run(p, cmd("save", name="living room loop"))
    lb = lib(tmp_path)
    saved = lb.load("living room loop")
    assert saved["points"] == [list(map(float, q)) for q in SQUARE] and saved["walls"]
    assert p.name == "living room loop" and not p.dirty
    # A second save of the same name needs the page's yes.
    assert p.command({"cmd": "save", "name": "living room loop"}, crop_at()) is False
    assert p.command({"cmd": "save", "name": "living room loop", "overwrite": True},
                     crop_at()) is True
    run(p, cmd("clear"))
    assert p.points == []
    run(p, cmd("load", name="living room loop"))
    visit(p, (0.0, 0.0))
    assert np.allclose(xy(p), SQUARE, atol=0.1), "fitted on the same map"
    assert p.name == "living room loop" and p.fit_state["state"] == "placed"
    run(p, cmd("forget", name="living room loop"))
    assert lb.load("living room loop") is None
    assert p.command({"cmd": "load", "name": "living room loop"}, crop_at()) is False


def test_nothing_unsaved_is_lost_to_a_clear_a_load_or_a_new_stroke(tmp_path):
    lb = lib(tmp_path)
    p, _ = patrol(tmp_path)
    run(p, cmd("stroke", points=SQUARE), cmd("clear"))
    kept = [r["name"] for r in lb.names()]
    assert len(kept) == 1 and kept[0].startswith("track kept")
    assert lb.load(kept[0])["points"] == [list(map(float, q)) for q in SQUARE]
    # A saved one, unchanged, is not kept again.
    run(p, cmd("stroke", points=SQUARE), cmd("save", name="a"), cmd("clear"))
    assert len(lb.names()) == 2


def test_an_unplaced_track_is_kept_when_a_new_one_is_drawn_over_it(tmp_path):
    lb = lib(tmp_path)
    p, _ = patrol(tmp_path)
    run(p, cmd("stroke", points=SQUARE), cmd("save", name="kitchen"))
    # Next boot, somewhere the walls do not fit: not placed.
    q = pt.Patrol(store=pt.Store(str(tmp_path / "patrol.json"), background=False),
                  library=lb, map_view=lambda: None, clock=Clock())
    visit(q, (0.0, 0.0))
    assert q.page_state()["pending"] is not None
    q.poll(Events(cmd("stroke", points=[[5, 5], [6, 5]])), crop_at())
    assert q.page_state()["pending"] is None and xy(q) == [(5, 5), (6, 5)]
    names = [r["name"] for r in lb.names()]
    assert "kitchen" in names and any(n.startswith("kitchen kept") for n in names)
    on_disk = json.loads((tmp_path / "patrol.json").read_text())
    assert on_disk["points"] == [[5.0, 5.0], [6.0, 5.0]]


def test_without_a_library_the_named_commands_are_refused(tmp_path):
    p = pt.Patrol(clock=Clock())
    visit(p, (0.0, 0.0))
    for name in ("save", "load", "forget"):
        assert p.command({"cmd": name, "name": "x"}, crop_at()) is False


def test_the_page_state_is_plain_json(tmp_path):
    p, _ = patrol(tmp_path)
    run(p, cmd("stroke", points=SQUARE))
    st = p.page_state()
    json.dumps(st)
    assert st["points"] == [list(map(float, q)) for q in SQUARE]
    assert st["pose"] == [0.0, 0.0, 0.0] and st["strokes"] == 1
    assert st["said"][-1][1].startswith("Drew waypoints 1-4")
