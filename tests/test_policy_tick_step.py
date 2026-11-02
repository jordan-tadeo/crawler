"""The decision step: one number (policy_dataset --tick-s, 0.1 by default)
that every tick-sized number in the dataset and the trainer is made off.

At 0.1 every one of them is the constant it was before the option, so a
default build is the file from before it. At another step (0.0625 for 16 Hz)
they move with it, the dataset and the sidecar say tick_s, and a recording
made at another rate than the build is left out rather than mislabelled.
"""

import argparse
from types import SimpleNamespace as NS

import numpy as np
import pytest

from crawler.autopilot import driver as dv  # noqa: E402
from crawler.autopilot import policy as pl  # noqa: E402
from crawler.autopilot import policy_imu as pimu  # noqa: E402
import models as md  # noqa: E402
import policy_adapters as pa  # noqa: E402
import policy_dataset as pd  # noqa: E402
import policy_eval as pe  # noqa: E402
import policy_train as pt  # noqa: E402

from test_policy_tools import S, T0, footprint_session, row  # noqa: E402

IMU = dict(imu_age_ms=2.0, imu_accel_x=0.1, imu_accel_y=0.0, imu_accel_z=0.0,
           imu_gyro_x=0.3, imu_gyro_y=0.0, imu_gyro_z=0.0)


def session_at(step_s, n=20, name="s"):
    rows = [(k * step_s, row(k, throttle=0.4, forward_limit=pa.RUNTIME_CAPS[0],
                             reverse_limit=pa.RUNTIME_CAPS[1], **IMU)) for k in range(n)]
    return footprint_session(rows)._replace(name=name)


def build(tmp_path, monkeypatch, session, *extra):
    monkeypatch.setattr(pd, "read_session", lambda path, adapter=None: (session, False))
    out = tmp_path / "d.npz"
    code = pd.main([str(out), str(tmp_path / "s.mcap"), "--no-idle-rules", *extra])
    return code, (pe.load(str(out)) if code == 0 else None)


# --- at 0.1 nothing moves -----------------------------------------------------

def test_the_10_hz_step_is_exactly_the_constants_from_before_it():
    """Exact, not approx: a float off by one ulp moves a tick across a gap
    and the default build would stop being the old file."""
    step = pd.step_for(0.1)
    assert step == pd.STEP
    assert step.tick_s == pd.TICK_S == 0.1
    assert step.slack_s == pd.TICK_SLACK_S == 0.02
    assert step.prev_gap_s == pd.PREV_GAP_S == 0.25
    assert step.settle_s == pd.SETTLE_S == dv.SETTLE_SECONDS + 0.1
    assert step.imu_window == pimu.WINDOW == 5
    assert step.imu_gap_s == pimu.GAP_S == 0.25
    assert pt.link_for(0.1) == pl.CHUNK_LINK_S == 0.25
    assert pt.chunk_meta(5, -0.5) == pt.chunk_meta(5, -0.5, 0.1)
    assert pt.imu_meta(True, False) == pt.imu_meta(True, False, tick_s=0.1)


def test_sampling_a_jittered_20_hz_stream_at_0_1_is_the_old_rule():
    rng = np.random.default_rng(3)
    t = np.cumsum(0.05 + rng.uniform(-0.012, 0.012, 400))
    cmds = [NS(t_ns=int(T0 + v * S)) for v in t]

    def old(commands):
        out, last = [], None
        for cmd in commands:
            if last is None or (cmd.t_ns - last) / 1e9 >= 0.1 - 0.02:
                out.append(cmd)
                last = cmd.t_ns
        return out

    assert pd.sample_ticks(cmds) == old(cmds)
    assert pd.sample_ticks(cmds, 0.1) == old(cmds)
    # 5 Hz off the same stream: about half the ticks.
    assert len(pd.sample_ticks(cmds, 0.2)) == pytest.approx(len(old(cmds)) / 2, rel=0.1)


def test_a_default_build_says_0_1_and_the_old_window(tmp_path, monkeypatch):
    code, a = build(tmp_path, monkeypatch, session_at(0.1))
    assert code == 0
    assert a["meta"]["tick_s"] == 0.1 and a["meta"]["imu_window"] == 5
    assert len(a["t_ns"]) == 20
    assert not any("(step)" in line for line in a["meta"]["report"])


# --- another step -------------------------------------------------------------

def test_a_16_hz_step_moves_every_tick_sized_number():
    step = pd.step_for(0.0625)
    assert step.tick_s == 0.0625
    assert step.prev_gap_s == pytest.approx(0.15625)
    assert step.settle_s == pytest.approx(dv.SETTLE_SECONDS + 0.0625)
    assert step.slack_s == pytest.approx(0.0125)
    # The IMU peak keeps its half second: 8 decisions, not 5.
    assert step.imu_window == pimu.window_for(0.0625) == 8
    assert step.imu_gap_s == pytest.approx(pimu.gap_for(0.0625))
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            pd.step_for(bad)


def test_the_prev_action_links_by_steps_not_by_a_tenth_of_a_second():
    """0.2 s is one dropped tick at 10 Hz (linked) and two at 16 Hz (not)."""
    ticks = [NS(t_ns=t, action=pl.FORWARD, magnitude=0.5, steer=0.3)
             for t in (T0, T0 + round(0.2 * S))]
    stop = pl.encode_prev(pl.STOP, 0.0, 0.0)[0]
    assert not np.array_equal(pd.prev_inputs(ticks)[1], stop)
    np.testing.assert_array_equal(pd.prev_inputs(ticks, pd.step_for(0.0625))[1], stop)


def test_the_imu_window_is_the_steps_half_second():
    """A knock six decisions back is out of the 10 Hz window and inside the
    16 Hz one: the same half second, in more decisions."""
    def peaks(step_s):
        rows = []
        for k in range(8):
            extra = dict(IMU, imu_accel_x=9.0 if k == 1 else 0.1)
            rows.append((k * step_s, row(k, throttle=0.4, **extra)))
        cmds = footprint_session(rows).commands
        return [f[pimu.FEATURES.index("accel_peak")]
                for f in pd.imu_features(cmds, step=pd.step_for(step_s))]

    knock = 9.0 / pimu.ACCEL_SCALE
    assert peaks(0.1)[1:6] == pytest.approx([knock] * 5) and peaks(0.1)[6] < knock
    assert peaks(0.0625)[1:8] == pytest.approx([knock] * 7)


def test_a_10_hz_recording_is_left_out_of_a_16_hz_build(tmp_path, monkeypatch, capsys):
    """Today's recordings are one /footprint row a decision at 10 Hz: a 16 Hz
    build cannot resample them, so it says so and writes nothing."""
    code, _ = build(tmp_path, monkeypatch, session_at(0.1), "--tick-s", "0.0625")
    assert code == 1
    out = capsys.readouterr().out
    assert pd.STEP_EXCLUDED in out and "0.1000 s" in out


def test_any_step_keeps_it_with_a_warning_and_the_meta_says_the_step(tmp_path, monkeypatch):
    code, a = build(tmp_path, monkeypatch, session_at(0.1), "--tick-s", "0.0625",
                    "--any-step")
    assert code == 0
    assert a["meta"]["tick_s"] == 0.0625 and a["meta"]["imu_window"] == 8
    assert any(line.startswith(pd.STEP_WARNING) for line in a["meta"]["report"])
    # 0.1 s apart is under the 16 Hz link (0.156), so the prev still chains.
    assert len(a["t_ns"]) == 20


def test_a_16_hz_recording_builds_at_16_hz_and_not_at_10(tmp_path, monkeypatch):
    code, a = build(tmp_path, monkeypatch, session_at(0.0625, n=32), "--tick-s", "0.0625")
    assert code == 0 and len(a["t_ns"]) == 32
    assert not any("(step)" in line for line in a["meta"]["report"])
    code, _ = build(tmp_path, monkeypatch, session_at(0.0625, n=32))
    assert code == 1, "16 Hz rows are 37% off a 10 Hz build"


def test_a_session_too_short_to_measure_is_not_judged():
    assert pd.recorded_step_s(session_at(0.1, n=1)) is None
    assert pd.step_verdict(session_at(0.1, n=1), pd.step_for(0.0625)) == ""
    assert pd.recorded_step_s(session_at(0.1)) == pytest.approx(0.1)


# --- the trainer and the sidecar ----------------------------------------------

def test_the_trainer_takes_the_step_from_the_dataset():
    assert pt.tick_of({}) == 0.1, "a dataset from before tick_s is 10 Hz"
    assert pt.tick_of({"tick_s": 0.0625}) == 0.0625
    for bad in (0, -1, "x"):
        with pytest.raises((ValueError, TypeError)):
            pt.tick_of({"tick_s": bad})

    def meta(tick_s, chunk=5):
        args = argparse.Namespace(representation="profile", prev=False, labels="pad",
                                  good_weight=1.0, epochs=1, seed=0, imu=True,
                                  imu_no_yaw=False, imu_tilt=False, chunk=chunk)
        given = {} if tick_s is None else {"tick_s": tick_s}
        return pt._base_meta({"sessions": ["s"], "input_contract": "c", **given},
                             args, [], [])

    old, now = meta(None), meta(0.0625)
    assert old["tick_s"] == 0.1 and now["tick_s"] == 0.0625
    assert old["imu_window"] == 5 and now["imu_window"] == 8
    assert old["imu_scales"]["gap_s"] == 0.25
    assert old["ensemble_link_s"] == pl.CHUNK_LINK_S
    assert now["ensemble_link_s"] == pytest.approx(0.15625)


def test_chunk_labels_chain_by_steps():
    """At 16 Hz a 0.2 s hole is past the link; at 10 Hz it is one dropped tick."""
    t = (np.array([0.0, 0.0625, 0.2625, 0.325]) * S).astype(np.int64)
    data = {"t_ns": t, "session": np.zeros(4, np.int64), "decision": np.ones(4, bool)}
    assert pt.chunk_index(data, 3)[0].tolist() == [0, 1, 2]
    assert pt.chunk_index(data, 3, pt.link_for(0.0625))[0].tolist() == [0, 1, -1]


def test_a_new_step_is_a_major_and_an_old_sidecar_reads_as_10_hz():
    base = {"input_contract": "c", "representation": "profile", "actions": list(pl.ACTIONS)}
    assert md.contract_of(base) == md.contract_of({**base, **pt.tick_meta(0.1)})
    assert md.contract_diff(base, {**base, **pt.tick_meta(0.0625)}) == ["tick_s"]
