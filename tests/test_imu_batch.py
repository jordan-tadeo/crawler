"""Every IMU sample goes into the recording, not just the plot's 16 a second.

/imu is gated for a plot and records about 16 samples a second out of the
88 the car's SLAM is handed, which made a replay of any heading fix coarser
than the drive it came from. /imu/batch carries all of them, a tenth of a
second to a message, and tools/slam_replay.py reads it when it is there.
"""

import pytest

from crawler.hardware import camera
from crawler.telemetry import telemetry as tm

import slam_replay as sr


def live_tele(watched=True):
    tele = tm.Telemetry()
    tele.running = True
    for key in tele._watched:
        tele._watched[key] = watched
    return tele


def sample(i, up=True):
    s = {"accel": (0.00390625 * i, -0.5, 9.0),
         "gyro": (0.00106526 * i, 0.002130528911948204, -0.01)}
    if up:
        s["up"] = (-1.0, 0.0, 0.0)
    return s


def drive(hub, tele, n, t0=100.0, period=1 / 88):
    """n samples through the hub at the IMU's live rate, the publisher
    taking each mailbox once a sample (it drains every 10 ms)."""
    got = {"imu": [], "imu_batch": []}
    for i in range(n):
        hub._offer("imu", sample(i, up=i % 3 == 0), t0 + i * period)
        for key in got:
            try:
                got[key].append(tele._mail[key].get_nowait())
            except Exception:
                pass
    return got


def test_every_sample_the_listeners_get_is_recorded_at_their_time():
    tele = live_tele()
    hub = camera.CameraHub(tele=tele)
    got = drive(hub, tele, 880)
    batches = [p for _, p in got["imu_batch"]]
    rows = [r for b in batches for r in sr.imu_batch_rows(b)]
    # All but the few since the last flush, against /imu's 16 a second.
    assert len(rows) >= 880 - 10
    assert len(got["imu"]) < 220
    assert [b["seq"] for b in batches] == list(range(len(batches)))
    assert 95 <= len(batches) <= 101
    off = tm.epoch_offset()
    for i, r in enumerate(rows):
        assert r[0] == pytest.approx(100.0 + i / 88 + off, abs=2e-6)
        assert r[1] == pytest.approx(0.00106526 * i, abs=1e-6)
        assert r[4:7] == ((-1.0, 0.0, 0.0) if i % 3 == 0 else (0.0, 0.0, 0.0))


def test_a_batch_is_stamped_with_its_last_sample_so_a_replay_has_it_by_then():
    tele = live_tele()
    hub = camera.CameraHub(tele=tele)
    got = drive(hub, tele, 200)
    assert len(got["imu_batch"]) >= 20
    for log_time, b in got["imu_batch"]:
        assert max(b["t"]) <= log_time * 1e-9 + 1e-6


def test_nothing_is_kept_while_nobody_takes_the_channel():
    tele = live_tele(watched=False)
    hub = camera.CameraHub(tele=tele)
    drive(hub, tele, 50)
    assert hub.imu_batch._rows == []
    tele = live_tele()
    tele.mute(tm.CAMERA_KEYS)
    hub = camera.CameraHub(tele=tele)
    got = drive(hub, tele, 50)
    assert got["imu_batch"] == [] and hub.imu_batch._rows == []


def test_a_bad_sample_costs_that_sample_only():
    tele = live_tele()
    b = tm.ImuBatcher()
    b.add(tele, 1.0, {"gyro": (0.1, float("nan"), 0.0), "accel": (0, 0, 0)})
    b.add(tele, 1.01, {"gyro": "junk"})
    b.add(tele, 1.02, None)
    assert tele._mail["imu_batch"].empty()
    # Live, mapping/imu.ImuTrack drops a NaN gyro, so the replay never sees
    # one; a NaN accel or up costs that column, not the sample.
    b.add(tele, 1.03, {"gyro": (0.2, 0.0, 0.0), "accel": (float("nan"), 0, 0),
                       "up": (float("inf"), -1.0, 0.0)})
    msg = tele._mail["imu_batch"].get_nowait()[1]
    assert msg["gyro_x"] == [0.2]
    assert msg["accel_x"] == [0.0]
    assert (msg["up_x"], msg["up_y"], msg["up_z"]) == ([0.0], [0.0], [0.0])


def test_the_replay_takes_the_batch_over_the_gated_imu():
    tele = live_tele()
    hub = camera.CameraHub(tele=tele)
    got = drive(hub, tele, 300)
    batch = [r for _, b in got["imu_batch"] for r in sr.imu_batch_rows(b)]
    gated = [(lt * 1e-9, p["gyro_x"], p["gyro_y"], p["gyro_z"],
              p["up_x"], p["up_y"], p["up_z"]) for lt, p in got["imu"]]
    rows = sr.imu_rows(gated, batch)
    assert rows.shape == (len(batch), 7)
    assert (rows[1:, 0] > rows[:-1, 0]).all()
    # A file from before /imu/batch replays off /imu as it always did.
    assert sr.imu_rows(gated, []).shape == (len(gated), 7)
