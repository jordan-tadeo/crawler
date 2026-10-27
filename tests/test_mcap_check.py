"""The review half of deploy/mcap_check.py: --marks and --window.

A mark I cannot read back afterwards is a button that does nothing, so this is
the half of the feature that makes the other half worth having. What is tested
here is the reading: which messages belong to a mark's window, what the
listing says, and that a mark index nobody has is an answer rather than a
traceback.

mcap_check runs under `uv run --with mcap` and imports the mcap library at
module scope, which is not installed in this checkout -- the same situation
test_recording.py is in with the foxglove SDK, and the same fix: a stand-in in
sys.modules and the module loaded by path. Nothing below touches the real
library, because nothing below needs to: the reading is mine.
"""

import importlib.util
import json
import os
import sys
import types

import pytest


def _load():
    """deploy/mcap_check.py, with a stand-in mcap under it."""
    for name in ("mcap", "mcap.reader", "mcap.records", "mcap.stream_reader",
                 "mcap.writer"):
        sys.modules.setdefault(name, types.ModuleType(name))

    sys.modules["mcap.reader"].make_reader = lambda handle: None
    for name in ("Channel", "Message", "Schema"):
        setattr(sys.modules["mcap.records"], name, type(name, (), {}))
    sys.modules["mcap.stream_reader"].StreamReader = type("StreamReader", (), {})
    sys.modules["mcap.writer"].Writer = type("Writer", (), {})

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "deploy", "mcap_check.py")
    spec = importlib.util.spec_from_file_location("mcap_check_under_test",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mc = _load()

SECOND = 1_000_000_000


def channel(cid, topic, encoding="json", schema_id=1):
    return types.SimpleNamespace(id=cid, topic=topic,
                                 message_encoding=encoding,
                                 schema_id=schema_id, metadata={})


def message(cid, at_s, payload=None, raw=None):
    data = raw if raw is not None else json.dumps(payload or {}).encode()
    return types.SimpleNamespace(channel_id=cid, log_time=int(at_s * SECOND),
                                 publish_time=int(at_s * SECOND), sequence=0,
                                 data=data)


def a_mark(sequence=1, kind="bad", **extra):
    row = {
        "kind": kind,
        "button": "B" if kind == "bad" else "A",
        "sequence": sequence,
        "utc": "2026-10-27T18:04:11Z",
        "mono_s": 100.0,
        "lookback_s": 4.0,
        "lead_s": 1.0,
        "mode": "auto",
        "throttle": 0.2,
        "steering_front": 0.0,
        "steering_rear": 0.0,
        "esc_pulse_us": 1620,
        "block_reason": "near",
        "near_distance_mm": 412.0,
        "probe_distance_mm": [900.0, 412.0, -1.0],
        "probe_blocked": [False, True, False],
        "scan_blocked": True,
        "scan_ms": 2.1,
        "recovery_phase": "reverse",
        "recovery_remaining_s": 1.4,
        "motion_evidence": "still",
        "gyro_mean": 0.004,
        "accel_mean": 0.012,
        "camera_state": "ready",
    }
    row.update(extra)
    return row


@pytest.fixture
def run():
    """A little run: /vehicle at 1 Hz for 30 s, with two marks in it."""
    channels = {1: channel(1, "/vehicle"), 2: channel(2, "/marks"),
                3: channel(3, "/camera/depth", encoding="protobuf",
                           schema_id=2)}
    schemas = {1: types.SimpleNamespace(id=1, name="crawler.Vehicle"),
               2: types.SimpleNamespace(id=2, name="foxglove.CompressedImage")}
    messages = [message(1, t, {"throttle": 0.2}) for t in range(30)]
    messages += [message(3, t, raw=b"\x00" * 500) for t in range(0, 30, 2)]
    messages.append(message(2, 10.0, a_mark(1, "bad")))
    messages.append(message(2, 20.0, a_mark(2, "good")))
    return schemas, channels, messages


def test_only_the_marks_channel_is_read(run):
    _, channels, messages = run
    marks = mc.decode_marks(channels, messages)
    assert [row["sequence"] for _, row in marks] == [1, 2]
    assert [log_time / SECOND for log_time, _ in marks] == [10.0, 20.0]


def test_a_file_with_no_marks_channel_reads_none():
    channels = {1: channel(1, "/vehicle")}
    assert mc.decode_marks(channels, [message(1, 0.0, {})]) == []


def test_a_mark_that_will_not_decode_keeps_its_timestamp():
    channels = {2: channel(2, "/marks")}
    marks = mc.decode_marks(channels, [message(2, 5.0, raw=b"not json")])
    assert len(marks) == 1
    assert marks[0] == (5 * SECOND, {})


def test_the_window_comes_off_the_mark_itself():
    start, end = mc.mark_window(100 * SECOND, a_mark())
    assert start == 96 * SECOND
    assert end == 101 * SECOND


def test_a_mark_recorded_without_its_bounds_falls_back():
    """An older file still reviews, under the numbers in this file."""
    start, end = mc.mark_window(100 * SECOND, {"kind": "good"})
    assert start == (100 - mc.DEFAULT_LOOKBACK_S) * SECOND
    assert end == (100 + mc.DEFAULT_LEAD_S) * SECOND


def test_the_listing_says_what_the_car_was_doing(run, capsys):
    _, channels, messages = run
    assert mc.marks_report("crawler-x.mcap", channels, messages) == 2
    out = capsys.readouterr().out
    assert "2 marks (1 good, 1 bad)" in out
    assert "mark #0  BAD" in out
    assert "mark #1  GOOD" in out
    # The snapshot, which is the point of listing them at all.
    assert "t+10.0s" in out
    assert "window     t+6.0s .. t+11.0s" in out
    assert "near, nearest 412 mm" in out
    assert "probes 900 412 --" in out
    assert "recovering reverse, 1.4 s left" in out
    assert "esc 1620 us" in out


def test_a_file_with_no_marks_says_so(capsys):
    mc.marks_report("crawler-x.mcap", {1: channel(1, "/vehicle")},
                    [message(1, 0.0, {})])
    assert "no marks in this one" in capsys.readouterr().out


def test_the_window_dump_is_only_the_window(run, capsys):
    schemas, channels, messages = run
    assert mc.window_report("crawler-x.mcap", schemas, channels, messages,
                            0) == 0
    out = capsys.readouterr().out

    # The mark is at t+10, so t+6 through t+11 and nothing else.
    assert "t+6.000s  /vehicle" in out
    assert "t+11.000s  /vehicle" in out
    assert "t+5.000s" not in out
    assert "t+12.000s" not in out
    # Binary payloads say their size and stop, so the dump stays readable.
    assert "<500 bytes of foxglove.CompressedImage>" in out
    assert '{"throttle": 0.2}' in out


def test_a_window_running_past_the_file_says_which_segment(capsys):
    """The recorder rotates, so half a window can be in the next file."""
    channels = {2: channel(2, "/marks")}
    messages = [message(2, 1.0, a_mark())]
    mc.window_report("crawler-x.mcap", {}, channels, messages, 0)
    out = capsys.readouterr().out
    assert "before this file does" in out
    assert "after this file does" in out


def test_an_index_nobody_has_is_an_answer_not_a_traceback(run, capsys):
    schemas, channels, messages = run
    assert mc.window_report("crawler-x.mcap", schemas, channels, messages,
                            7) == 1
    assert "there is no mark #7" in capsys.readouterr().out


def test_the_per_file_summary_counts_the_marks(run, tmp_path, capsys):
    schemas, channels, messages = run
    # report() stats the file for its size, so this one has to exist.
    path = tmp_path / "crawler-x.mcap"
    path.write_bytes(b"")
    mc.report(str(path), schemas, channels, messages)
    out = capsys.readouterr().out
    assert "2 marks (1 good, 1 bad) -- see --marks" in out


def test_the_old_invocation_still_means_check_and_repair():
    """pull-recordings.sh passes a bare directory and must keep working."""
    args = mc.parse_args(["recordings"])
    assert args.targets == ["recordings"]
    assert args.marks is False
    assert args.window is None


def test_the_review_flags_parse():
    assert mc.parse_args(["--marks", "a.mcap"]).marks is True
    assert mc.parse_args(["--window", "3", "a.mcap"]).window == 3


def test_window_wants_one_file(capsys):
    assert mc.review(["--window", "0", "a.mcap", "b.mcap"]) == 2
    assert "wants one file" in capsys.readouterr().out
