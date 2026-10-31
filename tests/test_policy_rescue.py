"""My rescue reverses weighed down (--rescue-reverse-weight, --drive-aways).

tools/drive_aways.py finds where a patrol near-missed its waypoint or drove
away from it, off /patrol/status; policy_train weighs my open-floor reverse
takeovers inside those windows by the flag. Arrays only, never torch.
"""

import argparse

import numpy as np
import pytest

import drive_aways as da
import policy_dataset as pd
import policy_train as pt
from crawler.autopilot import policy as pl
from crawler.autopilot import policy_lidar as pli

AUTO, TAKE = pd.SOURCES.index(pd.SOURCE_AUTO), pd.SOURCES.index(pd.SOURCE_TAKEOVER)
FWD, REV = pl.ACTIONS.index(pl.FORWARD), pl.ACTIONS.index(pl.REVERSE)


def status(dists, dt=0.5, t0=100.0, index=0, **kw):
    """One /patrol/status row each dt s, a patrol driving on the map towards
    waypoint `index`, these distances to it."""
    rows = []
    for k, d in enumerate(dists):
        r = {"_t": t0 + k * dt, "state": "driving", "running": True, "use": "patrol",
             "on_map": True, "distance_m": d, "index": index, "track": "t", "frame_now": 1}
        r.update(kw)
        rows.append(r)
    return rows


def wins(rows):
    t0, t1 = da.windows(da.status_arrays(rows))
    return list(zip(t0.tolist(), t1.tolist()))


# --- the windows ------------------------------------------------------------

def test_a_waypoint_never_got_near_has_no_window():
    assert wins(status([3.0, 2.8, 2.6, 2.4, 2.2, 2.0])) == []


def test_a_near_miss_opens_a_window_until_the_waypoint_changes():
    # 2.0 down to 0.8 (inside NEAR_M, never PASS_M), then away again.
    rows = status([2.0, 1.6, 1.2, 0.8, 1.0, 1.4, 1.9])
    assert wins(rows) == [(101.0, 103.0 + da.TAIL_S)]


def test_driving_away_from_a_waypoint_it_never_got_near_opens_one_too():
    rows = status([2.0, 1.8, 1.6, 1.9, 2.1, 2.4])
    assert wins(rows) == [(102.0, 102.5 + da.TAIL_S)]
    assert wins(status([2.0, 1.8, 1.6, 1.9, 2.0])) == []  # 0.4 m back is not a drive away


def test_a_plain_pass_shuts_its_window_when_the_next_waypoint_is_the_goal():
    rows = status([2.0, 1.0, 0.4]) + status([3.0, 2.8, 2.6], t0=101.5, index=1)
    assert wins(rows) == [(100.5, 101.0 + da.TAIL_S)]


def test_rows_off_a_patrol_or_off_the_map_are_no_window():
    near = [2.0, 1.2, 0.8, 1.5, 2.0]
    for kw in ({"state": "tracing"}, {"running": False}, {"use": "goal"}, {"on_map": False},
               {"distance_m": -1.0}):
        assert wins(status(near, **kw)) == [], kw


def test_a_quiet_patrol_splits_the_run_and_a_new_frame_does_too():
    # The near miss is in the first run; the second starts again from 1.6.
    rows = status([2.0, 1.0]) + status([1.6, 1.7], t0=105.0)
    assert wins(rows) == [(100.5, 100.5 + da.TAIL_S)]
    rows = status([2.0, 1.0]) + status([1.6, 1.7], t0=101.0, frame_now=2)
    assert wins(rows) == [(100.5, 100.5 + da.TAIL_S)]


def test_a_tick_is_inside_a_window_from_its_start_to_its_end():
    got = da.inside([1.0, 5.0], [2.0, 6.0], [0.9, 1.0, 1.5, 2.0, 2.1, 5.5, 7.0])
    assert got.tolist() == [False, True, True, True, False, True, False]
    assert da.inside([], [], [1.0]).tolist() == [False]


# --- the trainer's ticks ----------------------------------------------------

def drive(sources, actions, near_mm=4000, dt=0.1):
    """One tick each dt s, these sources and gears, every scan clear for
    near_mm in every direction."""
    n = len(sources)
    bins = np.full((n, pli.BINS), near_mm, np.uint16)
    return {"t_ns": (np.arange(n) * dt * 1e9).astype(np.int64),
            "session": np.zeros(n, np.int64),
            "source": np.asarray(sources, np.int8),
            "action": np.asarray(actions, np.int64),
            "decision": np.ones(n, bool),
            "lidar_bins": bins, "lidar_present": np.ones(n, bool),
            "meta": {"sessions": ["a", "b"], "sources": list(pd.SOURCES)}}


def test_only_my_open_floor_reverses_inside_a_window_are_rescues():
    d = drive([TAKE] * 6 + [AUTO] * 2, [REV, REV, FWD, REV, REV, REV, REV, REV])
    d["lidar_bins"][4] = 300  # something 0.3 m ahead: a reverse there is no rescue
    d["decision"][5] = False
    win = np.ones(8, bool)
    win[3] = False
    got = pt.rescue_reverses(d, win)
    assert got.tolist() == [True, True, False, False, False, False, False, False]


def test_a_rescue_weighs_the_flag_and_nothing_else_moves():
    d = drive([TAKE] * 4, [REV] * 4)
    d.update(mark=np.zeros(4, np.int8))
    before = pt.tick_weights(dict(d), np.arange(4), 2.0, 1.0)
    d["rescue_w"] = pt.rescue_weights(np.array([True, False, True, False]), 0.3)
    after = pt.tick_weights(d, np.arange(4), 2.0, 1.0)
    assert after.tolist() == pytest.approx([0.3 * before[0], before[1], 0.3 * before[2], before[3]])


def test_windows_are_read_by_session_name_and_a_session_with_none_has_none(tmp_path):
    d = drive([TAKE] * 20, [REV] * 20)
    d["session"][10:] = 1
    np.savez(tmp_path / "a.npz", t0=np.array([0.2]), t1=np.array([0.4]),
             layout=np.array(da.LAYOUT))
    got, found = pt.load_drive_aways(d, str(tmp_path))
    assert found == 1
    assert np.flatnonzero(got).tolist() == [2, 3, 4]


def test_a_windows_file_of_another_layout_is_an_error(tmp_path):
    d = drive([TAKE] * 4, [REV] * 4)
    np.savez(tmp_path / "a.npz", t0=np.array([0.0]), t1=np.array([1.0]), layout=np.array("x"))
    with pytest.raises(ValueError):
        pt.load_drive_aways(d, str(tmp_path))


def test_the_flags_default_to_the_recipe_as_it_was():
    got = {}
    real = argparse.ArgumentParser.parse_args

    def grab(self, argv=None, namespace=None):
        got.update(vars(real(self, argv, namespace)))
        raise SystemExit(0)
    argparse.ArgumentParser.parse_args = grab
    try:
        with pytest.raises(SystemExit):
            pt.main(["data.npz", "out.onnx"])
    finally:
        argparse.ArgumentParser.parse_args = real
    assert got["rescue_reverse_weight"] == 1.0 and got["drive_aways"] is None
