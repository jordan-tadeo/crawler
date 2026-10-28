"""deploy/footprint_dump.py: /footprint rows out of a recording, into a table.

The mcap reader is not installed in this checkout, so records() is replaced
with a list of (topic, log_time, data) -- which is exactly what it yields. The
raw depth payloads are the real thing: encoded by the foxglove SDK through
telemetry.Telemetry.depth_png, the same call the car makes.
"""

import csv
import importlib.util
import json
import os

import numpy as np
import pytest

from crawler.autopilot import driver as dv
from crawler.autopilot import footprint as fp
from crawler.autopilot import policy_input
from crawler.telemetry import telemetry as tm

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "deploy", "footprint_dump.py")
_spec = importlib.util.spec_from_file_location("footprint_dump_under_test", _PATH)
fd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fd)


def graze():
    frame = np.full((400, 640), 3000, dtype=np.uint16)
    edges = dv.footprint_geometry(frame.shape).col_edges
    frame[:, :edges[1]] = 380
    return frame


class Tele:
    def __init__(self):
        self.out = []

    def wants(self, key, now):
        return True

    def offer(self, key, payload, now):
        log_time = tm.epoch_ns(now)
        if key in tm.RAW_DEPTH_CHANNELS:
            frame, frame_id = payload if isinstance(payload, tuple) else (payload, "oakd")
            data = tm.Telemetry.depth_png(frame, log_time, frame_id).encode()
        else:
            data = json.dumps(payload).encode()
        topic = tm.JSON_TOPICS.get(key) or tm.RAW_DEPTH_CHANNELS[key]
        self.out.append((topic, log_time, data))


class View:
    def __init__(self, frame):
        self.frame = frame

    def get_depth_frame(self):
        return self.frame


def recorded(frame, rows=10):
    tele = Tele()
    tap = fp.FootprintTap(view=View(frame), tele=tele)
    for i in range(rows):
        tap.after_drive(i * 0.1, fp.RECORD, (0.1, 0.2, -0.2), pad_throttle=0.5,
                        limits=(0.2, -0.6))
    return tele.out


@pytest.fixture
def stream(monkeypatch):
    messages = recorded(graze())
    monkeypatch.setattr(fd, "records", lambda path: iter(messages))
    return messages


def test_every_row_comes_out_flat(stream):
    table, _, _ = fd.dump(["crawler-record-x.mcap"])
    assert len(table) == 10
    row = table[0]
    assert row["file"] == "crawler-record-x.mcap"
    assert row["mode"] == "record"
    assert row["probe_reason_0"] == dv.NEAR
    assert row["probe_distance_mm_8"] == pytest.approx(3000.0)
    assert "probe_distance_mm" not in row
    assert row["throttle"] == pytest.approx(0.1)


def test_the_raw_frames_decode_and_rescan_to_their_own_rows(stream, tmp_path):
    table, checked, mismatched = fd.dump(["x.mcap"], str(tmp_path), check=True)
    assert checked == 2 and mismatched == 0
    saved = sorted(tmp_path.iterdir())
    assert len(saved) == 2
    assert np.array_equal(np.load(saved[0]), graze())


def test_a_frame_that_does_not_match_its_row_is_reported(monkeypatch):
    messages = recorded(graze())
    # Swap the first raw frame for a wall the row never saw.
    wall = np.full((400, 640), 200, dtype=np.uint16)
    for i, (topic, at, _) in enumerate(messages):
        if topic == "/camera/depth_raw":
            messages[i] = (topic, at, tm.Telemetry.depth_png(wall, at).encode())
            break
    monkeypatch.setattr(fd, "records", lambda path: iter(messages))
    _, checked, mismatched = fd.dump(["x.mcap"], check=True)
    assert (checked, mismatched) == (2, 1)


def test_the_png_is_lossless_and_a_payload_that_is_not_depth_is_refused():
    frame = (np.arange(400 * 640, dtype=np.uint32) * 7919 % 65535).astype(
        np.uint16).reshape(400, 640)
    back = fd.decode_depth(tm.Telemetry.depth_png(frame, 0).encode())
    assert back.dtype == np.uint16
    assert np.array_equal(back, frame)

    from foxglove.messages import CompressedImage, Timestamp
    jpeg = CompressedImage(timestamp=Timestamp(sec=0, nsec=0), frame_id="oakd",
                           data=b"\xff\xd8", format="jpeg").encode()
    with pytest.raises(ValueError):
        fd.decode_depth(jpeg)


def test_the_car_refuses_to_send_a_frame_that_is_not_uint16_depth():
    assert tm.Telemetry.depth_png(np.zeros((4, 4), dtype=np.uint8), 0) is None
    assert tm.Telemetry.depth_png(np.zeros((4, 4, 3), dtype=np.uint16), 0) is None
    assert tm.Telemetry.depth_png(None, 0) is None


def test_a_picture_that_is_not_the_frame_shrunk_is_reported(monkeypatch):
    """The drift check: a /policy/depth_small that is not depth_small() of
    its own raw frame fails --verify even when the probes all agree."""
    messages = recorded(graze())
    for i, (topic, at, data) in enumerate(messages):
        if topic == "/policy/depth_small" and fd.frame_id(data) == "footprint/1":
            off = policy_input.depth_small(graze()) // 2
            messages[i] = (topic, at, tm.Telemetry.depth_png(
                off, at, "footprint/1").encode())
    monkeypatch.setattr(fd, "records", lambda path: iter(messages))
    _, checked, mismatched = fd.dump(["x.mcap"], check=True)
    assert (checked, mismatched) == (2, 1)


def test_the_check_waits_for_a_picture_written_after_its_row(monkeypatch):
    """Topics land in no fixed order inside one log time."""
    messages = recorded(graze())
    small = [m for m in messages if m[0] == "/policy/depth_small"]
    rest = [m for m in messages if m[0] != "/policy/depth_small"]
    bad = (small[0][0], small[0][1], tm.Telemetry.depth_png(
        np.zeros(policy_input.SHAPE, np.uint16), small[0][1], "footprint/1").encode())
    monkeypatch.setattr(fd, "records",
                        lambda path: iter(rest + [bad] + small[1:]))
    _, checked, mismatched = fd.dump(["x.mcap"], check=True)
    assert (checked, mismatched) == (2, 1)


def test_the_pictures_come_out_paired_to_their_rows_by_seq(stream, tmp_path):
    table, _, _ = fd.dump(["crawler-record-x.mcap"], small_dir=str(tmp_path))
    data = np.load(tmp_path / "crawler-record-x.npz")
    assert data["depth"].shape == (10,) + policy_input.SHAPE
    assert data["depth"].dtype == np.uint16
    assert list(data["seq"]) == [row["seq"] for row in table]
    assert list(data["log_time_ns"]) == [row["log_time_ns"] for row in table]
    assert np.array_equal(data["depth"][3], policy_input.depth_small(graze()))


def test_a_row_whose_marks_alone_disagree_is_reported():
    row = json.loads(recorded(graze(), rows=1)[-1][2])
    assert fd.verify(row, graze()) == []
    row["probe_marks"] = "." * dv.PROBE_COLUMNS
    assert any(p.startswith("marks") for p in fd.verify(row, graze()))


def test_a_jpeg_is_refused_before_it_is_decoded():
    from foxglove.messages import CompressedImage, Timestamp
    jpeg = CompressedImage(timestamp=Timestamp(sec=0, nsec=0), frame_id="oakd",
                           data=b"\xff\xd8", format="jpeg").encode()
    with pytest.raises(ValueError, match="not a PNG"):
        fd.depth_png_bytes(jpeg)


def test_the_csv_has_one_header_and_every_column(stream, tmp_path):
    table, _, _ = fd.dump(["x.mcap"])
    out = tmp_path / "out.csv"
    fd.write(table, str(out))
    with open(out) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert rows[3]["probe_marks"] == table[3]["probe_marks"]
    assert "probe_valid_fraction_4" in rows[0]
