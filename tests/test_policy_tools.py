"""The laptop side of the policy: adapters, alignment, labels and grading.

These run on dicts and arrays, never an MCAP and never torch, so they are
the contract the dataset tool keeps with what record mode writes
(TRAINING_DATA.md) and with what the trainer and the eval read back.
"""

import json
import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tools"))

import driver as dv  # noqa: E402
import policy as pl  # noqa: E402
import policy_adapters as pa  # noqa: E402
import policy_dataset as pd  # noqa: E402
import policy_eval as pe  # noqa: E402
import policy_train as pt  # noqa: E402

S = 1_000_000_000  # ns
T0 = 1_800_000_000 * S


def png(small, seq):
    ok, buf = cv2.imencode(".png", small)
    assert ok
    return {"data": buf.tobytes(), "format": "png",
            "frame_id": f"{pa.DEPTH_SMALL_FRAME_ID_PREFIX}{seq}"}


def row(seq, mode="record", throttle=0.2, front=0.0, **extra):
    _, rear = __import__("driver").steering_pair(front)
    r = {"seq": seq, "mode": mode, "throttle": throttle, "steering_front": front,
         "steering_rear": rear, "steering_mode": "unified",
         "verdict_action": "forward", "verdict_steer": 0.0,
         "forward_limit": 0.2, "reverse_limit": -0.6}
    r.update(extra)
    return r


def footprint_session(rows, frames=None, marks=(), connected=()):
    """rows: [(t_s, row dict)]; a depth_small frame on each row unless frames given."""
    ad = pa.FootprintAdapter("s")
    small = np.full((pl.INPUT_HEIGHT, pl.INPUT_WIDTH), 1500, np.uint16)
    for t, r in rows:
        ad.feed(pa.FOOTPRINT_TOPICS["row"], T0 + round(t * S), r)
        if frames is None:
            ad.feed(pa.FOOTPRINT_TOPICS["depth"], T0 + round(t * S), png(small, r["seq"]))
    for t, payload in frames or ():
        ad.feed(pa.FOOTPRINT_TOPICS["depth"], T0 + round(t * S), payload)
    for t, m in marks:
        ad.feed(pa.FOOTPRINT_TOPICS["marks"], T0 + round(t * S), m)
    for t, c in connected:
        ad.feed(pa.FOOTPRINT_TOPICS["controller"], T0 + round(t * S), {"connected": c})
    return ad.finish()


def align(session, **kw):
    """Alignment alone: the not-driving rules have their own tests below."""
    return pd.align(session, rules=None, **kw)


def driving(n, mode="record", throttle=0.2, **extra):
    return [(i * 0.1, row(i, mode, throttle, **extra)) for i in range(n)]


# --- who was driving ----------------------------------------------------------

def test_a_missing_source_is_the_pad_in_record_and_the_autopilot_in_auto():
    assert pa.control_sources("record") == (pa.BY_PAD, pa.BY_PAD)
    assert pa.control_sources("auto") == (pa.BY_DRIVER, pa.BY_DRIVER)
    assert pa.control_sources("auto", "pad", None) == (pa.BY_PAD, pa.BY_DRIVER)
    assert pa.control_sources("record", "driver", "nonsense") == (pa.BY_DRIVER, pa.BY_PAD)


def test_a_takeover_row_labels_only_the_group_the_pad_had():
    session = footprint_session(driving(3, "auto", steer_by="pad", throttle_by="driver"))
    ticks, drops, _ = align(session)
    assert len(ticks) == 3 and not drops
    assert all(t.steer_label and not t.throttle_label for t in ticks)


def test_an_auto_row_from_before_takeover_is_the_autopilot_on_both():
    ticks, _, _ = align(footprint_session(driving(2, "auto")))
    assert not any(t.steer_label or t.throttle_label for t in ticks)


# --- the record-mode picture --------------------------------------------------

def test_depth_small_round_trips_through_its_png_and_joins_on_seq():
    small = (np.arange(pl.INPUT_HEIGHT * pl.INPUT_WIDTH, dtype=np.uint16)
             .reshape(pl.INPUT_HEIGHT, pl.INPUT_WIDTH) * 3)
    got, seq = pa.decode_depth_small(png(small, 41))
    assert seq == 41 and got.dtype == np.uint16
    np.testing.assert_array_equal(got, small)

    # The seq is the join and the log time the check: each row gets its own
    # picture a few ms off, and a pair whose clocks disagree is stale.
    rows = [(0.0, row(7)), (0.1, row(8)), (0.2, row(9))]
    frames = [(0.003, png(small, 7)), (0.098, png(small + 1, 8)), (0.0, png(small, 9))]
    session = footprint_session(rows, frames=frames)
    assert session.by_seq
    ticks, drops, errors = align(session)
    assert [session.frames[t.frame].seq for t in ticks] == [7, 8]
    assert session.frames[ticks[1].frame].small[0, 1] == small[0, 1] + 1
    assert errors == pytest.approx([3.0, 2.0])
    assert drops == {"stale": 1}


def test_a_lossy_or_wrongly_sized_picture_is_refused():
    small = np.zeros((pl.INPUT_HEIGHT, pl.INPUT_WIDTH), np.uint16)
    assert pa.decode_depth_small({**png(small, 1), "format": "jpeg"}) is None
    assert pa.decode_depth_small(png(np.zeros((10, 10), np.uint16), 1)) is None
    assert pa.decode_depth_small(png(small.astype(np.uint8), 1)) is None


def test_pick_adapter_goes_off_the_topics():
    assert pa.pick_adapter(["/policy/depth_small", "/footprint", "/marks"]) is pa.FootprintAdapter
    assert pa.pick_adapter(["/camera/depth", "/vehicle"]) is pa.LegacyAdapter
    assert pa.pick_adapter(["/vehicle"]) is None


def test_a_record_file_that_announces_the_legacy_topics_first_is_not_legacy():
    # Record mode writes /camera/depth and /vehicle too. Until the channel
    # list is settled, those alone must not decide it.
    early = ["/camera/depth", "/vehicle", "/controller", "/imu"]
    assert pa.pick_adapter(early, settled=False) is None
    assert pa.pick_adapter(early + ["/policy/depth_small", "/footprint"],
                           settled=False) is pa.FootprintAdapter
    assert pa.pick_adapter(early, settled=True) is pa.LegacyAdapter


# --- what is dropped ----------------------------------------------------------

def test_align_drops_what_cannot_be_a_label_and_says_why():
    rows = driving(8)
    rows[0][1]["mode"] = "manual"
    rows[1][1]["throttle"] = math.nan
    rows[2][1].update(steering_mode=pa.SPLIT, steering_rear=0.3)  # my right stick
    small = np.full((pl.INPUT_HEIGHT, pl.INPUT_WIDTH), 1500, np.uint16)
    frames = [(t, png(small, r["seq"])) for t, r in rows if r["seq"] != 3]  # one went missing
    connected = [(0.0, True), (0.55, False), (0.65, True)]  # off across tick 6
    session = footprint_session(rows, frames=frames, connected=connected)
    ticks, drops, _ = align(session)
    assert drops == {"mode": 1, "bad command": 1, "split rear": 1, "no frame": 1, "pad": 1}
    assert len(ticks) == 3


def test_a_legacy_frame_older_than_the_limit_is_stale():
    ad = pa.LegacyAdapter("legacy")
    mm = np.full(pl.LEGACY_SHAPE, 2000, np.uint16)
    ad.feed_mm(T0, mm)
    for i in range(6):
        ad.feed(pa.LEGACY_TOPICS["vehicle"], T0 + i * S // 10,
                {"mode": "auto", "throttle": 0.2, "steering_front": 0.0, "steering_rear": 0.0})
    ticks, drops, _ = align(ad.finish())
    assert [round(t.age_ms) for t in ticks] == [0, 100, 200]
    assert drops == {"stale": 3}


# --- labels -------------------------------------------------------------------

def test_the_slowest_creep_the_triggers_make_is_motion_not_a_stop():
    assert pa.action_of(0.05 * 0.2) == pl.FORWARD
    assert pa.action_of(-0.05 * 0.6) == pl.REVERSE
    assert pa.action_of(0.0) == pl.STOP
    assert pa.magnitude_of(0.010) == pytest.approx(0.05)
    assert pa.magnitude_of(-0.3) == pytest.approx(0.5)
    # A row's caps win, and the -1 "no limits" placeholder falls back.
    assert pa.magnitude_of(0.05, forward_cap=0.1) == pytest.approx(0.5)
    assert pa.magnitude_of(0.1, forward_cap=-1, reverse_cap=-1) == pytest.approx(0.5)


def test_in_auto_only_the_tick_that_starts_a_reverse_and_not_its_settle_is_a_decision():
    throttles = [0.2, -0.6, -0.6, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
    rows = [(i * 0.1, row(i, mode="auto", throttle=t)) for i, t in enumerate(throttles)]
    ticks, _, _ = align(footprint_session(rows))
    got = [(t.action, t.decision) for t in ticks]
    assert got[:4] == [(pl.FORWARD, True), (pl.REVERSE, True),
                       (pl.REVERSE, False), (pl.REVERSE, False)]
    settle = int(round(pd.SETTLE_S / 0.1))
    assert all(not d for a, d in got[4:4 + settle])
    assert got[4 + settle:] and all(d for a, d in got[4 + settle:])


def test_on_the_pad_every_reverse_tick_and_the_stop_after_it_is_a_decision():
    throttles = [0.2, -0.6, -0.6, -0.6, 0.0, 0.0, 0.2]
    rows = [(i * 0.1, row(i, throttle=t)) for i, t in enumerate(throttles)]
    ticks, _, _ = align(footprint_session(rows))
    assert [t.action for t in ticks] == [pl.FORWARD] + [pl.REVERSE] * 3 + [pl.STOP] * 2 + [pl.FORWARD]
    assert all(t.decision for t in ticks)


def test_steering_is_a_fraction_of_manuals_full_lock_clipped():
    import main as mn
    assert pa.STEER_SCALE == pl.STEER_SCALE == mn.STEERING_LIMIT_PERC / 100
    rows = [(0.0, row(0, front=0.35)), (0.1, row(1, front=-0.8))]
    ticks, _, _ = align(footprint_session(rows))
    assert [t.steer for t in ticks] == pytest.approx([0.5, -1.0])


def test_a_verdicts_steer_is_put_in_the_same_units_as_a_label():
    """read_the_gradient's steer is a fraction of max_turn (0.6), not 0.7."""
    import driver as dv
    max_turn = dv.Driver(None, None).max_turn
    assert pa.RULE_MAX_TURN == max_turn
    ticks, _, _ = align(footprint_session([(0.0, row(0, verdict_steer=-1.0))]))
    assert ticks[0].rule_steer == pytest.approx(-max_turn / pl.STEER_SCALE)
    mm = np.full((400, 640), 3000, np.uint16)
    # A graze on the left edge probe: the rules swerve at full max_turn.
    mm[:, :dv.footprint_geometry(mm.shape).col_edges[1]] = 380
    _, steer = pa.gradient_rule(mm)
    raw = dv.read_the_gradient(dv.free_by_column(mm)).steer
    assert raw != 0.0 and steer == pytest.approx(raw * max_turn / pl.STEER_SCALE)


def test_a_group_a_learned_policy_drove_is_never_a_label_even_for_any():
    rows = (driving(3, "auto", autopilot="rules")
            + [(0.3 + i * 0.1, row(3 + i, "auto", autopilot="policy",
                                   steer_by="pad" if i == 0 else "driver"))
               for i in range(3)])
    session = footprint_session(rows)
    ticks, _, _ = align(session)
    a = pd.to_arrays([(session, ticks)])
    assert a["policy_drove"].tolist() == [False] * 3 + [True] * 3
    steer_ok, throttle_ok = pe.label_masks(a, "any")
    assert steer_ok.tolist() == [True] * 3 + [True, False, False]
    assert throttle_ok.tolist() == [True] * 3 + [False] * 3
    steer_ok, throttle_ok = pe.label_masks(a, "pad")
    assert steer_ok.tolist() == [False] * 3 + [True, False, False]
    assert not throttle_ok.any()


def test_the_row_says_who_drove_and_which_scan():
    session = footprint_session([(0.0, row(0, "auto", source="tap", autopilot="policy"))])
    cmd = session.commands[0]
    assert (cmd.source, cmd.autopilot) == ("tap", "policy")
    old = footprint_session([(0.0, row(0, "auto"))]).commands[0]
    assert (old.source, old.autopilot) == ("", "")


def test_no_files_reads_the_pulled_recordings(tmp_path, monkeypatch):
    assert pd.DEFAULT_MCAPS == "~/crawler-data/mcap/crawler-record-*.mcap"
    monkeypatch.setenv("HOME", str(tmp_path))
    mcap = tmp_path / "crawler-data" / "mcap"
    mcap.mkdir(parents=True)
    for name in ("crawler-record-2.mcap", "crawler-record-1.mcap", "crawler-x.mcap"):
        (mcap / name).write_bytes(b"")
    assert [os.path.basename(p) for p in pd.default_mcaps()] == [
        "crawler-record-1.mcap", "crawler-record-2.mcap"]
    read = []
    monkeypatch.setattr(pd, "read_session",
                        lambda path, adapter=None: read.append(path) or (_ for _ in ()).throw(
                            SystemExit(0)))
    with pytest.raises(SystemExit):
        pd.main([str(tmp_path / "out.npz")])
    assert read and os.path.basename(read[0]) == "crawler-record-1.mcap"


def test_no_files_and_none_pulled_is_an_error_not_an_empty_set(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(SystemExit) as e:
        pd.main([str(tmp_path / "out.npz")])
    assert e.value.code == 2


def test_the_jet_jpeg_decodes_back_near_its_millimetres():
    mm = np.tile(np.linspace(500, 2800, pl.LEGACY_SHAPE[1]), (pl.LEGACY_SHAPE[0], 1))
    levels = ((mm - pl.LEGACY_NEAR_MM) * pl.LEGACY_LEVELS
              / (pl.LEGACY_FAR_MM - pl.LEGACY_NEAR_MM)).astype(np.uint8)
    ok, jpeg = cv2.imencode(".jpg", cv2.applyColorMap(levels, cv2.COLORMAP_JET),
                            [cv2.IMWRITE_JPEG_QUALITY, 60])
    got = pa.jet_jpeg_to_mm(jpeg.tobytes()).astype(np.float32)
    inner = got[:, 8:-8]
    assert np.median(np.abs(inner - mm[:, 8:-8])) < 40


# --- /marks on the pad's ticks --------------------------------------------------

def mark(kind, at_s, back=4.0, ahead=1.0, mono=500.0):
    return (at_s, {"kind": kind, "mono_s": mono, "window_start_s": mono - back,
                   "window_end_s": mono + ahead})


def test_a_bad_mark_flags_every_pad_tick_in_its_own_window():
    # A 2 s / 0.5 s window, so a hardcoded 4 s / 1 s would flag the wrong ticks.
    session = footprint_session(driving(40), marks=[mark("bad", 3.0, back=2.0, ahead=0.5)])
    ticks, _, _ = align(session)
    flags = pd.mark_flags(ticks, session.marks)
    flagged = [round(t.t_ns / 1e8 - T0 / 1e8) for t, f in zip(ticks, flags) if f == pd.MARK_BAD]
    assert flagged == list(range(10, 36))
    assert not (flags == pd.MARK_GOOD).any()


def test_good_flags_its_window_and_bad_wins_an_overlap():
    session = footprint_session(driving(40), marks=[mark("good", 2.0, back=1.0, ahead=1.0),
                                                    mark("bad", 3.5, back=1.0, ahead=0.0)])
    ticks, _, _ = align(session)
    flags = pd.mark_flags(ticks, session.marks)
    assert flags[9] == pd.MARK_NONE
    assert (flags[10:25] == pd.MARK_GOOD).all()
    assert (flags[25:36] == pd.MARK_BAD).all()
    assert (flags[36:] == pd.MARK_NONE).all()


def test_a_mark_on_the_autopilots_driving_is_not_a_label_verdict():
    rows = driving(30, "auto")
    for t, r in rows[20:]:
        r["steer_by"] = "pad"  # a takeover from 2.0 s on
    session = footprint_session(rows, marks=[mark("bad", 2.4)])
    ticks, _, _ = align(session)
    flags = pd.mark_flags(ticks, session.marks)
    assert not flags[:20].any()
    assert (flags[20:25] == pd.MARK_BAD).all()


def test_a_mark_with_no_window_flags_nothing():
    session = footprint_session(driving(10), marks=[(0.5, {"kind": "bad", "mono_s": 1.0}),
                                                    (0.5, {"kind": "bad", "mono_s": 1.0,
                                                           "window_start_s": "x",
                                                           "window_end_s": 2.0})])
    ticks, _, _ = align(session)
    assert not pd.mark_flags(ticks, session.marks).any()


def marked_arrays():
    session = footprint_session(driving(40), marks=[mark("bad", 1.0, back=0.5, ahead=0.5),
                                                    mark("good", 3.0, back=0.5, ahead=0.5)])
    ticks, _, _ = align(session)
    a = pd.to_arrays([(session, ticks)])
    a["meta"] = {"sessions": ["s"], "marks": {}}
    return a


def test_the_trainer_leaves_bad_ticks_out_and_weights_good_ones_up():
    a = marked_arrays()
    assert (a["mark"] == pd.MARK_BAD).sum() == 11 and (a["mark"] == pd.MARK_GOOD).sum() == 11
    mask = pt.training_mask(a, np.zeros(40, bool), np.ones(40, bool))
    assert not mask[a["mark"] == pd.MARK_BAD].any()
    assert mask[a["mark"] != pd.MARK_BAD].all()
    w = pt.tick_weights(a, np.arange(40), 3.0)
    assert (w[a["mark"] == pd.MARK_GOOD] == 3.0).all()
    assert (w[a["mark"] != pd.MARK_GOOD] == 1.0).all()
    assert pt.GOOD_WEIGHT == 2.0


class Forward:
    """A model that always says forward at full speed, straight."""
    representation = pl.IMAGE
    uses_prev = False

    def run(self, x, prev):
        return {"logits": np.array([[5.0, 0.0, 0.0]], np.float32),
                "throttle": np.array([[1.0]], np.float32),
                "steer": np.array([[0.0]], np.float32)}


def test_the_val_numbers_leave_bad_ticks_out_too():
    a = marked_arrays()
    a["action"][a["mark"] == pd.MARK_BAD] = pl.ACTIONS.index(pl.REVERSE)
    s = pe.score(Forward(), a, [0], "pad", np.ones(40, bool))
    assert s["direction_ticks"] == 29
    assert s["direction_accuracy"] == 1.0
    assert s["bad_marked_left_out"] == 11


def test_eval_grades_against_the_baselines():
    a = marked_arrays()
    a["mark"][:] = 0
    a["action"][:10] = pl.ACTIONS.index(pl.STOP)
    a["steer"][:] = 0.5
    s = pe.score(Forward(), a, [0], "pad", np.ones(40, bool))
    assert s["direction_accuracy"] == pytest.approx(0.75)
    assert s["always_forward_accuracy"] == pytest.approx(0.75)
    assert s["steer_mae"] == pytest.approx(0.5)
    assert s["steer_straight_mae"] == pytest.approx(0.5)
    assert s["steer_rule_mae"] == pytest.approx(0.5)
    assert s["confusion"][2] == [10, 0, 0]


# --- not driving: parked, idle, the reach for Y, the car in my hands --------

R = pd.IdleRules()


def record(spec):
    """[(seconds, throttle, front, gyro)] -> 10 Hz record rows, in order."""
    rows, i = [], 0
    for seconds, throttle, front, gyro in spec:
        for _ in range(int(round(seconds * 10))):
            extra = {"imu_age_ms": 5.0, "imu_accel_x": 0.0, "imu_accel_y": 0.0,
                     "imu_accel_z": 0.02, "imu_gyro_x": gyro, "imu_gyro_y": 0.0,
                     "imu_gyro_z": 0.0}
            rows.append((i * 0.1, row(i, "record", throttle, front, **extra)))
            i += 1
    return rows


def verdicts(spec, rules=R):
    timeline = []
    ticks, drops, _ = pd.align(footprint_session(record(spec)), rules=rules, timeline=timeline)
    return [w for _, w in timeline], ticks, drops


def test_before_the_first_and_after_the_last_throttle_is_idle():
    why, ticks, drops = verdicts([(3, 0.0, 0.0, 0.0), (5, 0.2, 0.0, 0.1), (3, 0.0, 0.3, 0.0)])
    assert why[:30] == ["idle lead"] * 30
    assert why[80:] == ["idle trail"] * 30
    assert drops["idle lead"] == 30 and drops["idle trail"] == 30
    # Throttle is what ends the lead; steering on its own is still setting up.


def test_the_second_before_the_run_ends_is_the_reach_for_y():
    why, _, drops = verdicts([(6, 0.2, 0.0, 0.1)])
    assert why[:50] == [""] * 50
    assert why[50:] == ["exit reach"] * 10
    assert drops["exit reach"] == 10


def test_a_long_stop_is_parked_but_its_edges_and_a_short_stop_are_kept():
    spec = [(2, 0.2, 0.0, 0.1), (1.4, 0.0, 0.0, 0.0), (2, 0.2, 0.0, 0.1),
            (4.0, 0.0, 0.05, 0.0), (2, -0.6, 0.0, 0.3), (2, 0.2, 0.0, 0.1)]
    why, ticks, drops = verdicts(spec)
    assert why[20:34] == [""] * 14                      # 1.4 s stop: kept whole
    long_stop = why[54:94]                              # 5.4 s .. 9.3 s
    assert long_stop[:5] == [""] * 5 and long_stop[-5:] == [""] * 5
    assert long_stop[5:35] == ["parked"] * 30
    assert drops["parked"] == 30
    # Dropped means gone from the ticks, not only counted.
    assert not any(T0 + round(5.9 * S) <= t.t_ns <= T0 + round(8.8 * S) for t in ticks)
    # The stop right before the reverse survives, so the reverse onset has its approach.
    kept = [t for t in ticks if t.t_ns >= T0 + round(8.9 * S)]
    assert kept[0].action == pl.STOP and any(t.action == pl.REVERSE and t.decision for t in kept)


def test_steering_outside_the_rest_band_is_input_not_parked():
    spec = [(2, 0.2, 0.0, 0.1), (4.0, 0.0, 0.3, 0.0), (2, 0.2, 0.0, 0.1)]
    why, _, drops = verdicts(spec)
    assert "parked" not in why
    assert R.steer_rest == pytest.approx(0.084)


def test_a_big_imu_spike_with_the_throttle_held_is_driving():
    spec = [(2, 0.2, 0.0, 0.1), (1.0, 0.2, 0.0, 6.0), (2, 0.2, 0.0, 0.1)]
    why, _, drops = verdicts(spec)
    assert why[20:30] == [""] * 10
    assert drops["handled"] == 0


def test_the_same_spike_with_no_input_is_the_car_being_handled():
    spec = [(2, 0.2, 0.0, 0.1), (1.0, 0.0, 0.0, 0.0), (0.3, 0.0, 0.0, 6.0),
            (1.0, 0.0, 0.0, 0.0), (2, 0.2, 0.0, 0.1)]
    why, _, drops = verdicts(spec)
    # Spike at ticks 30-32; the margin covers a second either side, inside the
    # no-input span only, and never reaches the throttle on either side.
    assert why[30:33] == ["handled"] * 3
    assert why[20:30] == ["handled"] * 10 and why[33:43] == ["handled"] * 10
    assert why[19] == "" and why[43] == ""
    assert drops["handled"] == 23


def test_the_coast_right_after_letting_go_is_not_handling():
    spec = [(2, 0.2, 0.0, 0.1), (0.4, 0.0, 0.0, 6.0), (2, 0.2, 0.0, 0.1)]
    why, _, drops = verdicts(spec)
    assert drops["handled"] == 0


def test_a_row_with_no_imu_sample_never_reads_as_handled():
    rows = record([(2, 0.2, 0.0, 0.1), (1.0, 0.0, 0.0, 9.0), (2, 0.2, 0.0, 0.1)])
    for _, r in rows:
        r["imu_age_ms"] = -1.0
    _, drops, _ = pd.align(footprint_session(rows), rules=R)
    assert drops["handled"] == 0
    assert pa.imu_magnitudes({"imu_age_ms": 3, "imu_gyro_x": 0.3, "imu_gyro_y": 0.4,
                              "imu_gyro_z": 0, "imu_accel_x": 0, "imu_accel_y": 0,
                              "imu_accel_z": 2})[1] == pytest.approx(0.5)


def test_the_rules_leave_auto_ticks_alone():
    rows = driving(30, "auto", throttle=0.0)
    ticks, drops, _ = pd.align(footprint_session(rows), rules=R)
    assert len(ticks) == 30 and not drops


def test_the_timeline_is_a_character_a_second():
    timeline = [(T0 + i * S // 10, "" if i < 20 else "parked") for i in range(40)]
    timeline += [(T0 + 6 * S, "exit reach")]
    (line,) = pd.format_timeline(timeline)
    assert line.split() == ["0s", "##pp..x"]


# --- speed caps across sessions ------------------------------------------------

OLD = (0.2, -0.6)


def capped(caps, n=20, throttle=None, mode="record"):
    throttle = caps[0] if throttle is None else throttle
    return footprint_session(driving(n, mode, throttle, forward_limit=caps[0],
                                     reverse_limit=caps[1]))


def test_a_label_is_a_fraction_of_its_own_rows_cap_and_the_runtime_agrees():
    """Full RT is magnitude 1.0 at either cap, and the runtime's caps are
    driver.py's, so 1.0 always replays as today's cap."""
    for caps in (OLD, pa.RUNTIME_CAPS):
        ticks, _, _ = align(capped(caps))
        assert ticks and all(t.magnitude == pytest.approx(1.0) for t in ticks)
        rev, _, _ = align(capped(caps, throttle=caps[1]))
        assert rev and all(t.magnitude == pytest.approx(1.0) for t in rev)
    assert pa.RUNTIME_CAPS == (pytest.approx(dv.FORWARD_SPEED),
                               pytest.approx(dv.REVERSE_SPEED))


def test_a_placeholder_row_is_measured_against_its_sessions_own_caps():
    f, r = pa.RUNTIME_CAPS
    rows = driving(5, "record", f, forward_limit=f, reverse_limit=r)
    rows += [(1.0 + i * 0.1, row(10 + i, "auto", r, forward_limit=-1, reverse_limit=-1))
             for i in range(3)]
    session = footprint_session(rows)
    assert pa.session_caps(session) == [pytest.approx((f, r))]
    ticks, _, _ = align(session)
    autos = [t for t in ticks if t.mode == "auto"]
    assert autos and all(t.magnitude == pytest.approx(1.0) for t in autos)


def test_a_session_at_other_caps_is_left_out_unless_asked_for():
    old = capped(OLD)
    assert pa.session_caps(old) == [OLD]
    line, _ = pd.caps_verdict(old)
    assert line.startswith(pd.CAPS_EXCLUDED)
    line, _ = pd.caps_verdict(old, include_other=True)
    assert line.startswith(pd.CAPS_WARNING)
    assert pd.caps_verdict(capped(pa.RUNTIME_CAPS))[0] == ""
    nothing = footprint_session(driving(5, "auto", 0.3, forward_limit=-1, reverse_limit=-1))
    assert pa.session_caps(nothing) == []
    assert pd.caps_verdict(nothing)[0].startswith(pd.CAPS_WARNING)


def test_the_dataset_tool_drops_the_old_caps_by_default(tmp_path, monkeypatch, capsys):
    sessions = {"old.mcap": capped(OLD)._replace(name="old"),
                "new.mcap": capped(pa.RUNTIME_CAPS)._replace(name="new")}
    monkeypatch.setattr(pd, "read_session",
                        lambda path, adapter=None: (sessions[os.path.basename(path)], False))
    paths = [str(tmp_path / n) for n in sessions]

    out = tmp_path / "default.npz"
    assert pd.main([str(out), *paths, "--no-idle-rules"]) == 0
    meta = json.loads(str(np.load(out)["meta"]))
    assert meta["sessions"] == ["new"]
    assert meta["throttle_unit"] == "fraction_of_row_cap"
    assert "LEFT OUT old" in capsys.readouterr().out

    out = tmp_path / "all.npz"
    assert pd.main([str(out), *paths, "--no-idle-rules", "--include-other-caps"]) == 0
    meta = json.loads(str(np.load(out)["meta"]))
    assert meta["sessions"] == ["old", "new"]
    assert meta["session_caps"]["old"] == [list(OLD)]


# --- the steering mode -------------------------------------------------------

U, SP = pl.STEER_MODES.index(pl.UNIFIED), pl.STEER_MODES.index(pl.SPLIT)


def toggled(n=40, at=(2.0,), gap_after=None):
    """n record rows 0.1 s apart, unified, flipping mode at each time in `at`.
    Split rows carry a straight rear, the way I drive them."""
    rows, mode = [], pl.UNIFIED
    for i in range(n):
        t = i * 0.1 + (1.0 if gap_after is not None and i > gap_after else 0.0)
        if any(abs(t - a) < 1e-6 for a in at):
            mode = pl.SPLIT if mode == pl.UNIFIED else pl.UNIFIED
        r = row(i, front=0.3)
        if mode == pl.SPLIT:
            r.update(steering_mode=pl.SPLIT, steering_rear=0.0)
        rows.append((t, r))
    return footprint_session(rows)


def test_split_with_the_rear_straight_is_kept_and_its_mode_is_the_label():
    ticks, drops, _ = align(toggled())
    assert not drops and len(ticks) == 40
    assert [t.steer_mode for t in ticks[20:]] == [SP] * 20
    assert pd.SPLIT_REAR_REST == 0.05


def test_the_label_leads_a_toggle_and_the_input_is_the_mode_going_in():
    """The toggle at 2.0 s: the half second before it wants split while the
    car is still unified, the toggle tick goes in unified and comes out
    split, and from the next tick on the car is in split."""
    ticks, _, _ = align(toggled())
    t_s = [round((t.t_ns - T0) / S, 1) for t in ticks]
    k = t_s.index(2.0)
    lead = round(pd.MODE_LEAD_S / 0.1)
    assert [t.steer_mode for t in ticks[:k - lead]] == [U] * (k - lead)
    assert [t.steer_mode for t in ticks[k - lead:k]] == [SP] * lead
    assert [t.mode_in for t in ticks[:k + 1]] == [U] * (k + 1)
    assert [t.mode_in for t in ticks[k + 1:]] == [SP] * (len(ticks) - k - 1)
    window = [t.mode_switch for t in ticks]
    assert window[k - 5:k + 6] == [True] * 11
    assert not any(window[:k - 5]) and not any(window[k + 6:])


def test_a_toggle_on_a_dropped_tick_still_counts():
    session = toggled()
    # The toggle's own row has no picture, so it is no tick -- but it was a toggle.
    session = session._replace(frames=[f for f in session.frames if f.seq != 20])
    ticks, drops, _ = align(session)
    assert drops == {"no frame": 1}
    assert pd.toggles_of(session.commands) == [(T0 + 2 * S, SP)]
    after = [t for t in ticks if t.t_ns > T0 + 2 * S]
    assert after[0].mode_in == SP and after[0].steer_mode == SP


def test_a_mode_change_across_a_gap_is_not_a_toggle():
    """Manual writes no rows, so a press in there cannot be placed."""
    session = toggled(at=(2.9 + 1.0,), gap_after=28)
    assert [c.steering_mode for c in session.commands][28:30] == [pl.UNIFIED, pl.SPLIT]
    assert pd.toggles_of(session.commands) == []
    ticks, _, _ = align(session)
    assert not any(t.mode_switch for t in ticks)
    assert all(t.mode_in == t.steer_mode for t in ticks)


def test_the_dataset_arrays_and_meta_carry_the_mode(tmp_path, monkeypatch):
    session = toggled(at=(1.0, 3.0))._replace(name="s")
    session = session._replace(commands=[c._replace(forward_cap=pa.RUNTIME_CAPS[0],
                                                    reverse_cap=pa.RUNTIME_CAPS[1])
                                         for c in session.commands])
    monkeypatch.setattr(pd, "read_session", lambda path, adapter=None: (session, False))
    out = tmp_path / "d.npz"
    assert pd.main([str(out), str(tmp_path / "s.mcap"), "--no-idle-rules"]) == 0
    a = pe.load(str(out))
    assert a["meta"]["steer_modes"] == list(pl.STEER_MODES)
    assert a["meta"]["toggles"]["s"] == [T0 + 1 * S, T0 + 3 * S]
    assert a["steer_mode"].dtype == np.int8 and (a["steer_mode"] == SP).sum() == 20
    assert a["mode_switch"].sum() == 22


def test_the_trainer_weights_the_switch_windows_up():
    ticks, _, _ = align(toggled())
    a = pd.to_arrays([(toggled(), ticks)])
    w = pt.mode_weights(a, np.arange(len(ticks)), 7.0)
    assert a["mode_switch"].sum() == 11
    assert (w[a["mode_switch"]] == 7.0).all() and (w[~a["mode_switch"]] == 1.0).all()
    assert pt.MODE_SWITCH_WEIGHT == 10.0
    np.testing.assert_array_equal(pt.encode_modes(a["mode_in"][[0, -1]]),
                                  np.concatenate([pl.encode_mode(False), pl.encode_mode(True)]))


def test_time_blocks_are_contiguous_cover_every_decision_and_guard_their_edges():
    session = toggled(n=100)
    ticks, _, _ = align(session)
    a = pd.to_arrays([(session, ticks)])
    blocks = pt.time_blocks(a, 5, guard_s=0.45)
    held = np.array([h for h, _ in blocks])
    assert (held.sum(axis=0) == a["decision"]).all()     # every decision held out once
    for h, guard in blocks:
        idx = np.flatnonzero(h)
        assert len(idx) == 20 and (np.diff(idx) == 1).all()   # one stretch of time
        assert (guard >= h).all()
        assert guard.sum() == 20 + 4 * (int(idx[0] > 0) + int(idx[-1] < 99))


class ModeModel:
    """A mode head that copies the mode going in: always-stay, as a model."""
    representation = pl.IMAGE
    uses_prev = False
    has_mode = True
    mode_switch = {"threshold": pl.MODE_THRESHOLD, "dwell_s": pl.MODE_DWELL_S,
                   "min_interval_s": pl.MODE_MIN_INTERVAL_S}

    def run(self, x, prev, mode):
        return {**Forward().run(x, prev), pl.MODE_OUTPUT: mode_logits(bool(mode[0, SP]))}


def mode_logits(split):
    return np.array([[0.0, 6.0 if split else -6.0]], np.float32)


def mode_arrays():
    session = toggled(n=60, at=(3.0,))
    ticks, _, _ = align(session)
    a = pd.to_arrays([(session, ticks)])
    a["meta"] = {"sessions": ["s"], "marks": {},
                 "toggles": {"s": [t for t, _ in pd.toggles_of(session.commands)]}}
    return a


def test_a_head_that_only_copies_its_input_scores_always_stay_and_recalls_nothing():
    a = mode_arrays()
    s = pe.score(ModeModel(), a, [0], "pad", np.ones(60, bool))
    # Six ticks where I wanted the other mode: the five-tick lead and the
    # toggle's own tick, which goes in unified and comes out split.
    assert s["mode_accuracy"] == s["mode_stay_accuracy"] == pytest.approx(54 / 60)
    assert s["mode_switch_ticks"] == 6 and s["mode_switch_recall"] == 0.0
    assert s["mode_latch_switches"] == 0 and s["mode_toggles"] == 1


def test_a_head_that_knows_what_i_wanted_switches_the_latch_on_my_toggle(monkeypatch):
    a = mode_arrays()

    def oracle(model, data, i, split=None):
        return {**Forward().run(None, None),
                pl.MODE_OUTPUT: mode_logits(data["steer_mode"][i] == SP)}

    monkeypatch.setattr(pe, "_run", oracle)
    s = pe.score_ticks(ModeModel(), a, np.arange(60), "pad", 1.0)
    assert s["mode_accuracy"] == 1.0 and s["mode_switch_recall"] == 1.0
    assert s["mode_latch_switches"] == 1
    assert s["mode_latch_toggles_matched"] == 1 and s["mode_latch_false_switches"] == 0
    made = pe.latch_replay(ModeModel(), a, np.arange(60))
    assert made == [(T0 + 3 * S, True)]   # the lead plus the dwell lands on the toggle
