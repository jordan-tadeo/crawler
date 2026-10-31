"""tools/lap_report.py: laps and segments off a drive, per model and track.

The drives here are the real patrol (autopilot/patrol.py) driven round a
square by a truck that goes straight at the current waypoint, recorded as the
car records it: /patrol/event, /patrol/status at 2 Hz, the /footprint rows at
10 Hz with the patrol's columns, /slam/status and /vehicle. So the report is
held to what the car writes, not to a stream made up to suit it."""

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

import lap_report as lr  # noqa: E402
from crawler.autopilot import patrol as pt  # noqa: E402

from test_goal_reseed import crop_at  # noqa: E402
from test_patrol import SQUARE, Events, patrol_of  # noqa: E402

DT = 0.1
SPEED = 1.0                        # m/s
UP = (0.0, 0.0, 1.0)
SIDE = (0.0, 1.0, 0.0)             # on its side: 90 degrees off upright


def row(p, x, y, **kw):
    out = {"mode": "auto", "policy_model": "kblock candidate fast-1", "policy_fallback": False,
           "policy_map_present": True, "steer_by": "driver", "throttle_by": "driver",
           "throttle": 0.5, "rear_stop": False, "tilt_veto": "",
           "imu_accel_x": 0.0, "imu_accel_y": 0.0, "imu_accel_z": 9.8,
           "imu_gyro_x": 0.0, "imu_gyro_y": 0.0, "imu_gyro_z": 0.0, "imu_age_ms": 0.0}
    out["imu_up_x"], out["imu_up_y"], out["imu_up_z"] = UP
    out.update(p.row_fields())
    out.update(kw)
    return out


def drive(laps=2.0, hooks=None, points=SQUARE, mode="loop", skip_stuck=False,
          sideways=False):
    """(messages, patrol): the truck driven round `points` for `laps`, with
    hooks {decision k: fn(p, k) -> row overrides} for what happens on the
    way. The clock is the log clock, as on the car. sideways: the nose kept
    on +x whichever way it goes, so a waypoint off to the side is passed
    near (patrol.near_miss) instead of on PASS_M."""
    hooks = hooks or {}
    p, _, clock = patrol_of(points=points, mode=mode, skip_stuck=skip_stuck)
    msgs = []
    x, y, yaw = -1.0, -1.0, 0.0
    k = 0
    next_status = clock.t
    while p.laps + (p.index / len(p.points)) < laps and k < 20000:
        k += 1
        clock.t = round(clock.t + DT, 6)
        t = clock.t
        over = hooks.get(k, lambda p, k: {})(p, k) or {}
        stall = over.pop("_stall", False)
        g = p.current_place()
        if g is not None and not stall and p.running:
            dx, dy = g[0] - x, g[1] - y
            d = math.hypot(dx, dy)
            step = min(SPEED * DT, d)
            if d > 0:
                # Nose first, so the waypoint is ahead as on the car (a
                # truck that slid sideways would pass it near).
                x, y = x + dx / d * step, y + dy / d * step
                yaw = 0.0 if sideways else math.atan2(dy, dx)
        p.poll(Events(), crop_at(x, y, yaw))
        p.observe(crop_at(x, y, yaw))
        while True:
            e = p.pop_event()
            if e is None:
                break
            msgs.append(("/patrol/event", e["mono_s"], e))
        msgs.append(("/footprint", t, row(p, x, y, **over)))
        msgs.append(("/slam/status", t, {"state": "tracking", "x": x, "y": y, "reseed": ""}))
        msgs.append(("/vehicle", t, {"mode": over.get("mode", "auto")}))
        if t >= next_status - 1e-9:
            msgs.append(("/patrol/status", t, p.patrol_status()))
            next_status = t + 0.5
    return msgs, p


def coarse(msgs):
    """The same drive as a recording from before /patrol/event: no events,
    no patrol columns on the rows."""
    out = []
    for topic, t, d in msgs:
        if topic == "/patrol/event":
            continue
        if topic == "/footprint":
            d = {k: v for k, v in d.items() if not k.startswith("patrol_")}
        if topic == "/patrol/status":
            d = {k: v for k, v in d.items() if k not in ("track", "stuck_s", "skip_stuck")}
        out.append((topic, t, d))
    return out


def test_a_clean_loop_is_timed_to_the_decision_and_laps_are_full_and_clean():
    msgs, p = drive(laps=3.0)
    r = lr.analyse_run("run", msgs)
    assert not r["coarse"] and not r["warnings"]
    laps = r["laps"]
    assert [lp["lap"] for lp in laps] == [0, 1, 2]
    first, *rest = laps
    assert not first["full"], "the first lap starts with an approach"
    assert all(lp["full"] and lp["clean"] for lp in rest)
    # A lap is the 8 m square with each corner cut at the pass radius, at 1 m/s.
    for lp in rest:
        assert lp["segments"] == 4 and lp["skips"] == 0 and lp["takeover_s"] == 0
        assert lp["time_s"] == pytest.approx(lp["t1"] - lp["t0"], abs=1e-6)
        assert 5.0 < lp["time_s"] < 8.0
        assert lp["path_m"] == pytest.approx(lp["time_s"] * SPEED, abs=1e-6)
        assert lp["model"] == "kblock candidate fast-1"
    seg = [s for s in r["segments"] if s["lap"] == 1][0]
    assert (seg["from"], seg["to"]) == (3, 0) and seg["chained"]
    assert seg["leg_m"] == pytest.approx(2.0)
    assert seg["s_per_m"] == pytest.approx(seg["time_s"] / 2.0)
    events = [e for tp, _, e in msgs if tp == "/patrol/event" and e["kind"] == "pass"]
    assert seg["t1"] == pytest.approx([e["mono_s"] for e in events if e["index"] == 0][1])


def test_near_passes_end_segments_like_any_pass_and_say_so():
    msgs, p = drive(laps=3.0, sideways=True)
    assert p.passed_near > 0
    r = lr.analyse_run("run", msgs)
    assert not r["coarse"] and not r["warnings"]
    rest = r["laps"][1:]
    assert rest and all(lp["full"] and lp["clean"] and lp["segments"] == 4 for lp in rest)
    by = {s["by"] for s in r["segments"] if s["chained"]}
    assert "near" in by, "a segment ended by a near pass says so"
    near = [e for tp, _, e in msgs if tp == "/patrol/event" and e.get("by") == "near"]
    assert near and all(e["kind"] == "pass" for e in near)
    assert len(near) == p.passed_near
    assert sum(lp["near"] for lp in r["laps"]) == sum(s["by"] == "near" for s in r["segments"])
    assert all(lp["near"] > 0 for lp in rest)
    txt = lr.text([r], lr.summarise([r]), segments=True)
    assert " near " in txt and "near" in lr.LAP_HEAD


def test_a_takeover_a_skip_a_tip_and_the_fallback_each_make_a_lap_not_clean():
    def hand(p, k):
        return {"steer_by": "pad"}

    def tip(p, k):
        return {"imu_up_x": SIDE[0], "imu_up_y": SIDE[1], "imu_up_z": SIDE[2]}

    def fallback(p, k):
        return {"policy_fallback": True, "policy_model": "kblock candidate mapless",
                "policy_map_present": False}

    for name, fn, n in (("takeover", hand, 5), ("tip", tip, 5), ("fallback", fallback, 5)):
        base, _ = drive(laps=2.0)
        lap1 = next(lp for lp in lr.analyse_run("run", base)["laps"] if lp["lap"] == 1)
        k0 = int(round((lap1["t0"] - 100.0) / DT)) + 10
        msgs, _ = drive(laps=2.0, hooks={k0 + i: fn for i in range(n)})
        lp = next(lp for lp in lr.analyse_run("run", msgs)["laps"] if lp["lap"] == 1)
        assert lp["full"] and not lp["clean"], name
        if name == "takeover":
            assert lp["takeovers"] == 1 and lp["takeover_s"] == pytest.approx(n * DT, abs=1e-6)
        if name == "tip":
            assert lp["tips"] == 1 and lp["max_tilt_deg"] == pytest.approx(90.0, abs=1.0)
        if name == "fallback":
            assert lp["fallback"] > 0 and lp["map_lost_s"] == pytest.approx(n * DT, abs=1e-6)
            assert lp["model"] == "kblock candidate fast-1", "the fallback never names a lap"
    stalls = {k: (lambda p, k: {"_stall": True}) for k in range(30, 30 + 220)}
    msgs, p = drive(laps=2.0, hooks=stalls, skip_stuck=True)
    r = lr.analyse_run("run", msgs)
    skipped = [s for s in r["segments"] if s["skip"]]
    assert len(skipped) == 1 and skipped[0]["end"] == "skip"
    assert skipped[0]["stuck_s"] == pytest.approx(pt.NO_PROGRESS_S - lr.STUCK_FROM_S, abs=0.3)
    (lap,) = [lp for lp in r["laps"] if lp["chain"] == skipped[0]["chain"]
              and lp["lap"] == skipped[0]["lap"]]
    assert lap["skips"] == 1 and not lap["clean"]


def test_reverse_zero_throttle_rear_stops_and_tilt_blocks_are_counted():
    hooks = {}
    for k in range(40, 45):
        hooks[k] = lambda p, k: {"throttle": -0.4}
    for k in range(50, 53):
        hooks[k] = lambda p, k: {"throttle": 0.0, "rear_stop": True}
    for k in range(60, 62):
        hooks[k] = lambda p, k: {"tilt_veto": "forward"}
    msgs, _ = drive(laps=1.5, hooks=hooks)
    lps = lr.analyse_run("run", msgs)["laps"]
    total = {k: sum(lp[k] for lp in lps) for k in ("reverse_s", "zero_s", "rear_stops",
                                                    "tilts")}
    assert total == pytest.approx({"reverse_s": 0.5, "zero_s": 0.3, "rear_stops": 1,
                                   "tilts": 1})


def test_manual_with_the_patrol_driving_is_takeover_time():
    hooks = {k: (lambda p, k: {"mode": "manual"}) for k in range(70, 80)}
    msgs, _ = drive(laps=2.0, hooks=hooks)
    msgs = [m for m in msgs if not (m[0] == "/footprint" and m[2]["mode"] == "manual")]
    lps = lr.analyse_run("run", msgs)["laps"]
    assert sum(lp["manual_s"] for lp in lps) == pytest.approx(1.0, abs=1e-6)
    assert sum(lp["takeovers"] for lp in lps) == 1


def test_there_and_back_is_a_lap_of_both_ways_and_the_coarse_read_agrees():
    line = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
    msgs, p = drive(laps=3.3, points=line, mode="patrol")
    for r in (lr.analyse_run("run", msgs), lr.analyse_run("run", coarse(msgs))):
        full = [lp for lp in r["laps"] if lp["full"]]
        assert len(full) >= 2 and all(lp["clean"] for lp in full)
        for lp in full:
            assert lp["segments"] == lr.per_lap("patrol", 3) == 4
            assert [(s["from"], s["to"]) for s in r["segments"] if s["lap"] == lp["lap"]
                    and s["chain"] == lp["chain"]] == [(1, 0), (0, 1), (1, 2), (2, 1)]
            # 8 m there and back, less the pass radius at each end, at 1 m/s.
            assert 5.0 < lp["time_s"] < 8.0


def test_an_edit_is_a_new_track_and_its_laps_are_never_merged_with_the_old():
    def edit(p, k):
        p.command({"cmd": "move", "i": 2, "x": 0.0, "y": 2.2}, crop_at())
    msgs, p = drive(laps=4.0, hooks={140: edit})
    r = lr.analyse_run("run", msgs)
    tracks = list(dict.fromkeys(lp["track"] for lp in r["laps"]))
    old, edits = tracks[0].rsplit("+", 1)
    assert tracks == [tracks[0], f"{old}+{int(edits) + 1}"] and p.track_id() == tracks[1]
    # The segment the edit landed in is timed on neither track.
    t_edit = 100.0 + 140 * DT
    assert all(s["t1"] <= t_edit + 1e-6 or s["t0"] >= t_edit - 1e-6 for s in r["segments"])
    assert all(s["track"] == (tracks[0] if s["t1"] <= t_edit + 1e-6 else tracks[1])
               for s in r["segments"])
    first_new = [lp for lp in r["laps"] if lp["track"] == tracks[1]][0]
    assert not first_new["full"]
    assert any(lp["full"] and lp["clean"] for lp in r["laps"] if lp["track"] == tracks[1])


def test_an_old_recording_is_read_off_the_status_to_half_a_second_and_says_so():
    # A little past the third lap: the status has to see the last pass.
    msgs, _ = drive(laps=3.3)
    exact = lr.analyse_run("run", msgs)
    old = lr.analyse_run("run", coarse(msgs))
    assert old["coarse"] and old["warnings"] and lr.COARSE in old["warnings"][0]
    assert not exact["coarse"] and not exact["warnings"]
    assert [lp["lap"] for lp in old["laps"]] == [lp["lap"] for lp in exact["laps"]] == [0, 1, 2, 3]
    assert [lp["full"] for lp in old["laps"]] == [lp["full"] for lp in exact["laps"]] == [
        False, True, True, False]
    for a, b in zip(old["laps"][1:3], exact["laps"][1:3]):
        assert a["time_s"] == pytest.approx(b["time_s"], abs=0.5)
        assert a["clean"] == b["clean"]


def test_status_two_waypoints_on_is_two_passes_and_a_move_while_paused_is_a_reset():
    def st(t, index, laps=0, state="driving", skipped=0):
        return (t, {"use": "patrol", "state": state, "mode": "loop", "index": index,
                    "count": 4, "laps": laps, "skipped": skipped})
    events, ids = lr.events_from_status(
        [st(0.0, 0), st(0.5, 2), st(1.0, 3, skipped=1), st(1.5, 3, state="paused"),
         st(2.0, 0, state="paused"), st(2.5, 0)], "r")
    passes = [(e["kind"], e["index"], e["next"], e["t"]) for e in events
              if e["kind"] in ("pass", "skip")]
    assert passes == [("pass", 0, 1, 0.25), ("pass", 1, 2, 0.25), ("skip", 2, 3, 0.75)]
    # The index went 3 -> 0 while paused: nothing was passed, and the start
    # after it heads for waypoint 0, so no segment runs across the reset.
    assert [(e["kind"], e["index"]) for e in events if e["kind"] == "start"] == [("start", 0)]
    assert len(set(ids)) == 1


def test_the_summary_is_per_model_and_track_with_intervals_over_laps():
    a, _ = drive(laps=4.0)
    slow = {k: (lambda p, k: {"_stall": True}) for k in range(1, 4000, 4)}
    b, _ = drive(laps=4.0, hooks=slow)
    b = [(tp, t, dict(d, policy_model="kblock candidate slow-1") if tp == "/footprint" else d)
         for tp, t, d in b]
    runs = [lr.analyse_run("a", a, 0), lr.analyse_run("b", b, 1)]
    rows = {r["model"]: r for r in lr.summarise(runs, draws=200)}
    fast, slow_ = rows["kblock candidate fast-1"], rows["kblock candidate slow-1"]
    assert fast["clean_laps"] == 3 and slow_["clean_laps"] == 3
    est, lo, hi = fast["clean_lap_s"]
    assert lo <= est <= hi
    assert slow_["clean_lap_s"][0] == pytest.approx(fast["clean_lap_s"][0] * 4 / 3, rel=0.1)
    assert slow_["clean_s_per_m"][0] > fast["clean_s_per_m"][0]
    assert fast["skips_per_lap"][0] == 0 and fast["tips_per_lap"][0] == 0
    models = {r["model"]: r for r in lr.by_model(runs, draws=200)}
    assert models["kblock candidate slow-1"]["clean_s_per_m"][0] > \
        models["kblock candidate fast-1"]["clean_s_per_m"][0]
    out = lr.text(runs, lr.summarise(runs, draws=50), True, lr.by_model(runs, draws=50))
    assert "fast-1" in out and "slow-1" in out and "clean" in out


def test_rates_count_part_laps_by_their_part():
    stat = lr._rate("skips")
    assert stat([(0.5, 1.0), (1.0, 2.0)]) == pytest.approx(2.0)
    assert stat([]) is None


def test_files_are_runs_of_one_process_each():
    names = ["/d/crawler-20261030-220315Z-p4242-001.mcap",
             "/d/crawler-20261030-221426Z-p4242-002.mcap",
             "/d/crawler-20261030-230000Z-p4242-001.mcap",
             "/d/crawler-20261030-231000Z-p1200-002.mcap",
             "/d/crawler-20261030-221426Z-p4242-002.mcap.orig"]
    runs = lr.runs_of(names)
    assert [len(ps) for _, ps in runs] == [2, 1, 1]
    assert runs[0][0] == "p4242 20261030-220315Z"


def test_a_patrol_with_no_pass_is_no_laps():
    p, _, _ = patrol_of(start=False)
    msgs = [("/patrol/status", 100.0, p.patrol_status())]
    assert lr.analyse_run("r", msgs)["laps"] == []
    assert lr.summarise([]) == [] and lr.by_model([]) == []
