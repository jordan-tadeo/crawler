"""Learning from my hand on an auto drive: takeovers, and manual right after.

The dataset side (which ticks, which groups, what labels), the trainer's
weight on them, and the report that says how much a drive gave. Dicts and
arrays only, never an MCAP and never torch.
"""

import importlib.util
import json
import os

import numpy as np
import pytest

from crawler.autopilot import driver as dv  # noqa: E402
from crawler.autopilot import policy as pl  # noqa: E402
import models as md  # noqa: E402
import policy_adapters as pa  # noqa: E402
import policy_dataset as pd  # noqa: E402
import policy_eval as pe  # noqa: E402
import policy_train as pt  # noqa: E402

from test_policy_tools import S, T0, footprint_session, png  # noqa: E402

FWD, REV = pa.RUNTIME_CAPS


def row(seq, mode="auto", throttle=0.4, front=0.0, steer_by="driver",
        throttle_by="driver", pad_throttle=0.0, **extra):
    """A /footprint row at today's caps, the autopilot's unless told."""
    _, rear = dv.steering_pair(front)
    r = {"seq": seq, "mode": mode, "throttle": throttle, "steering_front": front,
         "steering_rear": rear, "steering_mode": "unified",
         "verdict_action": "forward", "verdict_steer": 0.0,
         "forward_limit": FWD, "reverse_limit": REV, "steer_by": steer_by,
         "throttle_by": throttle_by, "pad_throttle": pad_throttle,
         "autopilot": "policy"}
    r.update(extra)
    return r


def at(rows, dt=0.1):
    """[row] -> [(t_s, row)] at 10 Hz."""
    return [(i * dt, r) for i, r in enumerate(rows)]


def align(session):
    return pd.align(session, rules=pd.IdleRules())


def mine(seq, pad, front, mode="auto", **extra):
    """A row where my hand had both groups: the trigger scaled as record's."""
    throttle = pad * FWD if pad > 0 else pad * -REV
    return row(seq, mode, throttle, front, "pad", "pad", pad, **extra)


# --- which ticks -----------------------------------------------------------

def test_a_pure_auto_drive_has_no_interventions():
    session = footprint_session(at([row(i, front=0.2) for i in range(30)]))
    ticks, _, _ = align(session)
    assert len(ticks) == 30
    assert {t.source for t in ticks} == {pd.SOURCE_AUTO}
    assert not any(t.steer_label or t.throttle_label for t in ticks)
    s = pd.intervention_summary(session)
    assert (s["takeover_ticks"], s["takeover_segments"]) == (0, 0)


def test_both_groups_under_my_hand_label_both_with_what_was_commanded():
    rows = ([row(i) for i in range(5)]
            + [mine(5 + i, -0.5 - 0.1 * i, -0.4) for i in range(5)]
            + [row(10 + i) for i in range(5)])
    ticks, _, _ = align(footprint_session(at(rows)))
    fix = [t for t in ticks if t.source == pd.SOURCE_TAKEOVER]
    assert len(fix) == 5
    assert all(t.steer_label and t.throttle_label for t in fix)
    # The label is the command that reached the wheels, on record's scale:
    # a fraction of the row's cap, and steering in manual's units.
    assert [t.action for t in fix] == [pl.REVERSE] * 5
    assert [t.magnitude for t in fix] == pytest.approx([0.5, 0.6, 0.7, 0.8, 0.9])
    assert all(t.steer == pytest.approx(-0.4 / pa.STEER_SCALE) for t in fix)
    # Every reverse tick of mine is a decision, as on the pad; the
    # autopilot's own around it are untouched.
    assert all(t.decision for t in fix)
    assert {t.source for t in ticks} - {pd.SOURCE_TAKEOVER} == {pd.SOURCE_AUTO}


def test_a_forward_under_my_trigger_is_measured_against_the_forward_cap():
    rows = [row(0)] + [mine(1 + i, 0.5, 0.3) for i in range(3)] + [row(4)]
    ticks, _, _ = align(footprint_session(at(rows)))
    fix = [t for t in ticks if t.source == pd.SOURCE_TAKEOVER]
    assert [t.magnitude for t in fix] == pytest.approx([0.5] * 3)
    assert {t.action for t in fix} == {pl.FORWARD}


def test_one_group_under_my_hand_labels_only_that_group():
    steer = [row(i, front=0.3, steer_by="pad") for i in range(4)]
    throttle = [row(4 + i, throttle=-0.6, throttle_by="pad", pad_throttle=-0.6)
                for i in range(4)]
    ticks, _, _ = align(footprint_session(at([row(99)] + steer + throttle + [row(98)])))
    by = {t.t_ns: t for t in ticks}
    s_ticks = [by[T0 + round((1 + i) * 0.1 * S)] for i in range(4)]
    t_ticks = [by[T0 + round((5 + i) * 0.1 * S)] for i in range(4)]
    assert all(t.steer_label and not t.throttle_label for t in s_ticks)
    assert all(t.throttle_label and not t.steer_label for t in t_ticks)
    assert {t.source for t in s_ticks + t_ticks} == {pd.SOURCE_TAKEOVER}


def test_the_release_tail_is_not_a_label_but_a_stop_inside_the_run_is():
    rows = ([row(0)]
            + [mine(1, -1.0, 0.5), mine(2, -1.0, 0.5)]
            + [mine(3, 0.0, 0.5), mine(4, 0.0, 0.5)]        # a stop, then...
            + [mine(5, 0.5, 0.5), mine(6, 0.5, 0.5)]        # ...forward: a decision
            + [mine(7 + i, 0.0, 0.0) for i in range(5)]     # let go: 0.5 s of rest
            + [row(12), row(13)])
    session = footprint_session(at(rows))
    ticks, _, _ = align(session)
    by_seq = {session.commands[i].seq: t for i, t in enumerate(ticks)}
    assert all(by_seq[s].throttle_label for s in (1, 2, 3, 4, 5, 6))
    assert [by_seq[s].action for s in (3, 4)] == [pl.STOP, pl.STOP]
    assert not any(by_seq[s].steer_label or by_seq[s].throttle_label for s in range(7, 12))
    assert {by_seq[s].source for s in range(7, 12)} == {pd.SOURCE_AUTO}
    s = pd.intervention_summary(session)
    assert (s["takeover_ticks"], s["takeover_segments"]) == (11, 1)
    assert (s["steer_tails"], s["throttle_tails"]) == (5, 5)


def test_a_throttle_held_off_my_trigger_is_labelled_with_my_trigger():
    # A full RT pull the car commanded 0 for: what reached the ESC is not
    # what I asked for, so the label is my trigger on the row's caps.
    held = [row(1 + i, throttle=0.0, front=-0.5, steer_by="pad", throttle_by="pad",
                pad_throttle=0.5 + 0.1 * i) for i in range(4)]
    back = row(5, throttle=0.0, front=-0.5, steer_by="pad", throttle_by="pad",
               pad_throttle=-0.5)
    rows = [row(0)] + held + [back, mine(6, -0.8, 0.5), row(7)]
    session = footprint_session(at(rows))
    ticks, _, _ = align(session)
    held_ticks = ticks[1:5]
    assert all(t.steer_label and t.throttle_label for t in held_ticks)
    assert [t.action for t in held_ticks] == [pl.FORWARD] * 4
    np.testing.assert_allclose([t.magnitude for t in held_ticks], [0.5, 0.6, 0.7, 0.8])
    assert all(t.throttle == 0.0 for t in held_ticks)  # what went out stays on record
    assert ticks[5].action == pl.REVERSE and ticks[5].magnitude == pytest.approx(0.5)
    assert ticks[6].throttle_label and ticks[6].steer_label
    assert pd.intervention_summary(session)["held"] == 5


def test_two_takeovers_apart_are_two_segments():
    rows = ([mine(i, -0.5, 0.3) for i in range(3)] + [row(3 + i) for i in range(5)]
            + [mine(8 + i, 0.5, -0.3) for i in range(3)])
    s = pd.intervention_summary(footprint_session(at(rows)))
    assert (s["takeover_ticks"], s["takeover_segments"], s["both"]) == (6, 2, 6)


# --- manual right after auto ------------------------------------------------

def manual_after(spec, auto_n=10):
    """auto_n auto rows, then manual rows at 10 Hz off spec [(throttle, front)]."""
    rows = [row(i) for i in range(auto_n)]
    # Manual's own range on its rows, as a manual row would carry it.
    rows += [row(auto_n + i, "manual", thr, front, "pad", "pad", 0.0,
                 forward_limit=1.0, reverse_limit=-1.0)
             for i, (thr, front) in enumerate(spec)]
    return footprint_session(at(rows))


def test_the_manual_window_keeps_my_driving_for_five_seconds_after_auto():
    # 7 s of manual: driving all the way, so only the window can end it.
    session = manual_after([(-0.5, 0.3)] * 70)
    ticks, drops, _ = align(session)
    fix = [t for t in ticks if t.source == pd.SOURCE_MANUAL]
    # The switch is the first manual row; 0.0 .. 5.0 s after it is inside.
    assert len(fix) == 51
    assert drops["manual"] == 19
    assert (fix[-1].t_ns - fix[0].t_ns) / S == pytest.approx(pd.MANUAL_AFTER_AUTO_S)
    assert all(t.steer_label and t.throttle_label and t.decision for t in fix)
    assert pd.MANUAL_AFTER_AUTO_S == 5.0


def test_the_manual_window_counts_only_while_i_am_driving():
    spec = [(0.0, 0.0)] * 10 + [(-0.5, 0.0)] * 5 + [(0.0, 0.0)] * 5 + [(0.0, 0.4)] * 5
    ticks, drops, _ = align(manual_after(spec))
    fix = [t for t in ticks if t.source == pd.SOURCE_MANUAL]
    assert len(fix) == 10 and drops["manual idle"] == 15
    assert [t.action for t in fix] == [pl.REVERSE] * 5 + [pl.STOP] * 5


def test_manual_with_no_auto_before_it_is_not_a_correction():
    rows = [row(i, "manual", -0.5, 0.3, "pad", "pad") for i in range(20)]
    ticks, drops, _ = align(footprint_session(at(rows)))
    assert not ticks and drops["manual"] == 20


def test_manual_throttle_is_measured_against_the_auto_caps_it_left():
    # Manual drives the full range: 0.2 forward is half of auto's 0.4, and
    # anything past auto's cap is all of it.
    ticks, _, _ = align(manual_after([(0.2, 0.0), (0.8, 0.0), (-0.5, 0.0)]))
    fix = [t for t in ticks if t.source == pd.SOURCE_MANUAL]
    assert [t.magnitude for t in fix] == pytest.approx([0.2 / FWD, 1.0, 0.5 / -REV])


def test_the_manual_window_off_vehicle_counts_seconds_with_no_rows():
    stream = ([(T0 + i * S // 20, "auto", 0.4, 0.0) for i in range(20)]
              + [(T0 + (20 + i) * S // 20, "manual", -0.5 if 10 <= i < 50 else 0.0, 0.0)
                 for i in range(200)])
    session = footprint_session(at([row(i) for i in range(10)]))._replace(vehicle=tuple(stream))
    s = pd.intervention_summary(session)
    assert s["auto_exits"] == 1 and s["manual_rows"] == 0
    assert s["manual_driven_s"] == pytest.approx(2.0, abs=0.06)


# --- the arrays and the old datasets ------------------------------------------

def test_the_arrays_tag_every_tick_with_its_source():
    rows = [row(i) for i in range(3)] + [mine(3 + i, -0.5, 0.3) for i in range(3)] + [row(6)]
    session = footprint_session(at(rows))
    ticks, _, _ = align(session)
    a = pd.to_arrays([(session, ticks)])
    assert a["source"].dtype == np.int8
    assert [pd.SOURCES[i] for i in a["source"]] == ["auto"] * 3 + ["takeover"] * 3 + ["auto"]
    assert (a["session"] == 0).all()


def record_session(name, n=30):
    rows = [(i * 0.1, row(i, "record", 0.2, 0.3, "pad", "pad", 0.5)) for i in range(n)]
    return footprint_session(rows)._replace(name=name)


def test_a_record_only_dataset_builds_as_it_did(tmp_path, monkeypatch):
    sessions = {"r.mcap": record_session("r")}
    monkeypatch.setattr(pd, "read_session",
                        lambda path, adapter=None: (sessions[os.path.basename(path)], False))
    out = tmp_path / "d.npz"
    assert pd.main([str(out), str(tmp_path / "r.mcap")]) == 0
    a = pe.load(str(out))
    assert (a["source"] == pd.SOURCES.index(pd.SOURCE_RECORD)).all()
    assert a["steer_label"].all() and a["throttle_label"].all()
    assert a["meta"]["drive_logs"] == []
    # Exactly what alignment alone makes of it: nothing about a record
    # session moved.
    ticks, _, _ = pd.align(sessions["r.mcap"])
    np.testing.assert_array_equal(a["magnitude"], [t.magnitude for t in ticks])


def test_drive_logs_are_read_from_their_folder_and_a_manual_only_one_is_skipped(
        tmp_path, monkeypatch, capsys):
    drives = tmp_path / "drives"
    drives.mkdir()
    for n in ("crawler-1-001.mcap", "crawler-2-001.mcap", "crawler-record-3.mcap"):
        (drives / n).write_bytes(b"")
    assert [os.path.basename(p) for p in pd.drive_logs(str(drives))] == [
        "crawler-1-001.mcap", "crawler-2-001.mcap"]
    drive = footprint_session(at([row(i) for i in range(3)]
                                 + [mine(3 + i, -0.5, 0.3) for i in range(4)]
                                 + [row(7)]))._replace(name="crawler-1-001")
    manual_only = pa.LegacyAdapter("crawler-2-001").finish()
    sessions = {"r.mcap": record_session("r"), "crawler-1-001.mcap": drive,
                "crawler-2-001.mcap": manual_only}
    asked = {}

    def read(path, adapter=None):
        asked[os.path.basename(path)] = adapter
        return sessions[os.path.basename(path)], False
    monkeypatch.setattr(pd, "read_session", read)
    out = tmp_path / "d.npz"
    assert pd.main([str(out), str(tmp_path / "r.mcap"), "--drive-logs", str(drives)]) == 0
    printed = capsys.readouterr().out
    # A drive that starts in manual announces /camera/depth and /vehicle
    # first and would settle on legacy: a drive log is read as footprint.
    assert asked == {"r.mcap": None, "crawler-1-001.mcap": pa.FootprintAdapter.name,
                     "crawler-2-001.mcap": pa.FootprintAdapter.name}
    assert "crawler-2-001: no /footprint rows" in printed
    assert "interventions: 4 takeover ticks" in printed
    a = pe.load(str(out))
    assert a["meta"]["sessions"] == ["r", "crawler-1-001"]
    assert a["meta"]["sources"] == list(pd.SOURCES)
    assert a["meta"]["drive_logs"] == ["crawler-1-001"]
    assert int(pt.intervention_ticks(a).sum()) == 4


# --- the trainer ---------------------------------------------------------------

def mixed_arrays(n_record=40, n_fix=10):
    rec = record_session("r", n_record)
    drive = footprint_session(at([row(i) for i in range(2)]
                                 + [mine(2 + i, -0.5, 0.3) for i in range(n_fix)]
                                 + [row(2 + n_fix)]))._replace(name="d")
    pairs = [(s, pd.align(s, rules=None)[0]) for s in (rec, drive)]
    a = pd.to_arrays(pairs)
    a["meta"] = {"sessions": ["r", "d"], "sources": list(pd.SOURCES)}
    return a


def test_the_takeover_weight_lands_on_intervention_ticks_and_nowhere_else():
    a = mixed_arrays()
    idx = np.arange(len(a["t_ns"]))
    fix = pt.intervention_ticks(a)
    assert fix.sum() == 10
    w = pt.tick_weights(a, idx, 2.0, 3.0)
    assert (w[fix] == 3.0).all() and (w[~fix] == 1.0).all()
    assert pt.TAKEOVER_WEIGHT == 3.0
    # It multiplies a GOOD mark rather than replacing it.
    a["mark"][np.flatnonzero(fix)[0]] = pd.MARK_GOOD
    assert pt.tick_weights(a, idx, 2.0, 3.0)[np.flatnonzero(fix)[0]] == 6.0


def test_the_loss_share_is_the_interventions_weight_over_the_heads_labels():
    a = mixed_arrays()
    mask = pt.training_mask(a, np.zeros(len(a["t_ns"]), bool),
                            a["steer_label"] | a["throttle_label"])
    shares = pt.loss_shares(a, mask, "pad", 2.0, 3.0)
    # 40 record ticks at 1, 10 takeover ticks at 3: 30 / 70 of every head.
    for head in ("direction", "throttle", "steering"):
        n, total, share = shares[head]
        assert (n, total) == (10, 50)
        assert share == pytest.approx(30 / 70)
    assert pt.loss_shares(a, mask, "pad", 2.0, 1.0)["steering"][2] == pytest.approx(10 / 50)
    assert "steering 10 of 50 ticks, 42.9%" in pt.format_shares(shares, 3.0)


def test_a_dataset_from_before_the_source_array_trains_as_all_record():
    a = mixed_arrays()
    del a["source"]
    a["meta"] = {"sessions": ["r", "d"]}
    assert not pt.intervention_ticks(a).any()
    assert (pt.tick_weights(a, np.arange(len(a["t_ns"])), 2.0, 3.0) == 1.0).all()


def test_the_folds_hold_out_record_ticks_only():
    a = mixed_arrays()
    record = a["source"] == pd.SOURCES.index(pd.SOURCE_RECORD)
    held = np.array([h for h, _ in pt.time_blocks(a, 5)])
    assert not held[:, ~record].any()
    assert (held.sum(axis=0)[record] == a["decision"][record]).all()
    # The same folds the record session alone makes.
    rec_only = {k: (v[record] if isinstance(v, np.ndarray) and len(v) == len(record) else v)
                for k, v in a.items()}
    alone = np.array([h for h, _ in pt.time_blocks(rec_only, 5)])
    np.testing.assert_array_equal(held[:, record], alone)


# --- the drive report and the registry -----------------------------------------

def _load(name):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tools", name + ".py")
    spec = importlib.util.spec_from_file_location(name + "_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sr = _load("session_report")


def test_the_report_counts_auto_takeovers_and_the_manual_after():
    ticks = ([sr.Tick(i * 0.05, "auto", 0.4, 0.0, 0.0, "unified") for i in range(200)]
             # 10 s auto, then 8 s manual: driving 1..7 s in, so 4 s of it counts
             + [sr.Tick(10 + i * 0.05, "manual", -0.5 if 20 <= i < 140 else 0.0, 0.0, 0.0,
                        "unified") for i in range(160)])
    rows = ([(i * 0.1, "auto", "driver", "driver") for i in range(100)])
    for i in list(range(10, 20)) + list(range(50, 53)):
        rows[i] = (i * 0.1, "auto", "pad", "driver" if i >= 50 else "pad")
    fix = sr.corrections(ticks, rows)
    assert fix["auto_s"] == pytest.approx(10.0)
    assert fix["takeover_segments"] == 2
    assert fix["takeover_s"] == pytest.approx(1.3)
    assert fix["auto_exits"] == 1
    assert fix["manual_after_auto_s"] == pytest.approx(4.0, abs=0.06)
    assert sr.corrections(ticks[:200], rows[:5]) == {
        "auto_s": pytest.approx(10.0), "takeover_segments": 0, "takeover_s": 0.0,
        "auto_exits": 0, "manual_after_auto_s": 0.0}


def test_the_report_and_the_dataset_agree_on_the_window_and_what_driving_is():
    assert sr.MANUAL_AFTER_AUTO_S == pd.MANUAL_AFTER_AUTO_S
    assert sr.DRIVING_THROTTLE == pytest.approx(pa.THROTTLE_DEADBAND)
    assert sr.DRIVING_STEER == pytest.approx(pd.IdleRules().steer_rest)


def test_a_drive_logs_hash_is_found_in_the_drives_folder(tmp_path):
    (tmp_path / "drives").mkdir()
    (tmp_path / "crawler-record-a.mcap").write_bytes(b"a")
    (tmp_path / "drives" / "crawler-b.mcap").write_bytes(b"b")
    got = md.session_hashes(["crawler-record-a", "crawler-b", "gone"], str(tmp_path),
                            hash_file=lambda p: os.path.basename(os.path.dirname(p)))
    assert [g["mcap_sha256"] for g in got] == [os.path.basename(str(tmp_path)), "drives", None]
