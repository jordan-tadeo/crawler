"""The SLAM map across a restart: saved, loaded back, found again, and never
in the way of mapping when the file is bad (crawler/mapping/mapfile.py)."""

import math
import os
import signal
import threading
import time

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler.autopilot import patrol as pt  # noqa: E402
from crawler.mapping import mapfile as mf  # noqa: E402
from crawler.mapping import runner as rn  # noqa: E402
from crawler.mapping import slam as sl  # noqa: E402
from crawler.mapping.imu import ImuTrack  # noqa: E402

from test_mapping import (LEVEL, PERIOD, FakeScan, drive,  # noqa: E402
                          err_vs_truth, path_pose, put_down, scan_at)


def mapped(scans=60):
    imu = ImuTrack()
    slam = sl.Slam(imu=imu)
    drive(slam, scans, imu=imu)
    return slam


def booted(snap):
    """A Slam as the worker makes it at boot, the snapshot restored."""
    imu = ImuTrack()
    slam = sl.Slam(imu=imu)
    assert slam.restore(snap)
    return slam, imu


def same_cells(a, b):
    return all(np.array_equal(getattr(a, n), getattr(b, n))
               for n in ("lo", "fx", "fy", "cnt")) and (a.ox, a.oy) == (b.ox, b.oy)


def test_a_saved_map_loads_back_cell_for_cell(tmp_path):
    slam = mapped()
    snap = slam.snapshot()
    path = tmp_path / "map.npz"
    n = mf.save(path, snap)
    assert n == os.path.getsize(path) and not os.path.exists(f"{path}.tmp")
    back, why = mf.load(path)
    assert why == "" and back is not None
    assert same_cells(back.grid(), slam.grid)
    assert back.pose == tuple(float(v) for v in slam.pose)
    assert back.writes == slam._map_writes and back.saved_at > 0
    # Only the observed cells are in the file: the room, not the window.
    assert back.lo.size < 0.05 * slam.grid.lo.size


def corrupt(kind, path, good):
    if kind == "garbage":
        path.write_bytes(b"not a map at all" * 100)
    elif kind == "cut short":
        path.write_bytes(good[: len(good) // 2])
    elif kind == "empty":
        path.write_bytes(b"")
    elif kind in ("version", "outside", "pose", "format"):
        z = dict(np.load(path, allow_pickle=False))
        import json
        meta = json.loads(bytes(z["meta"]).decode())
        if kind == "version":
            meta["version"] = mf.VERSION + 1
        elif kind == "outside":
            meta["i0"] = meta["size"] - 1
        elif kind == "pose":
            meta["pose"] = [float("nan"), 0.0, 0.0]
        else:
            meta["format"] = "something else"
        z["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
        with open(path, "wb") as f:
            np.savez(f, **z)


@pytest.mark.parametrize("kind", ["garbage", "cut short", "empty", "version",
                                  "outside", "pose", "format"])
def test_a_bad_map_file_is_ignored_with_a_reason(tmp_path, kind):
    path = tmp_path / "map.npz"
    mf.save(path, mapped(30).snapshot())
    corrupt(kind, path, path.read_bytes())
    snap, why = mf.load(path)
    assert snap is None and why


def test_a_map_file_over_the_bound_is_not_read(tmp_path):
    path = tmp_path / "map.npz"
    mf.save(path, mapped(30).snapshot())
    snap, why = mf.load(path, max_bytes=100)
    assert snap is None and "over" in why
    assert mf.load(tmp_path / "none.npz") == (None, "no map file")


def test_a_failed_save_leaves_the_old_file_as_it_was(tmp_path, monkeypatch):
    path = tmp_path / "map.npz"
    mf.save(path, mapped(30).snapshot())
    before = path.read_bytes()

    def broken(*a, **k):
        raise OSError("card full")
    monkeypatch.setattr(mf.os, "replace", broken)
    with pytest.raises(OSError):
        mf.save(path, mapped(40).snapshot())
    assert path.read_bytes() == before and not os.path.exists(f"{path}.tmp")


def test_a_truck_that_boots_where_it_stopped_is_back_on_its_old_map():
    slam = mapped()
    old = slam.snapshot()
    new, imu = booted(old)
    # Moved a metre and turned a quarter round while it was off.
    put = path_pose(6.0) + np.array([0.8, -0.4, math.radians(90)])
    out = put_down(new, imu, put, 30, 100.0)
    assert out[0].state == sl.LOST and not out[0].written
    found = next(k for k, r in enumerate(out) if r.state == sl.TRACKING)
    assert found <= 10, found
    assert new.recoveries == 1 and new.reseeds == 0
    assert same_cells(new.grid, old.grid()) or new.grid.version > 1
    e_xy, e_yaw = err_vs_truth(out[-1]._replace(t=0.0),
                               pose_fn=lambda t: put if t == 0.0 else path_pose(t))
    assert e_xy < 0.05 and e_yaw < math.radians(2), "in the old map's frame"


def boxed(x, y, r=0.35):
    """A box round the truck: all a scan sees with someone crouched close
    all round it."""
    return [((x - r, y - r), (x + r, y - r)), ((x + r, y - r), (x + r, y + r)),
            ((x + r, y + r), (x - r, y + r)), ((x - r, y + r), (x - r, y - r))]


def test_a_restored_map_not_found_at_boot_is_shelved_and_merged_back():
    slam = mapped()
    saved = slam.snapshot()
    new, imu = booted(saved)
    restored = new.grid
    put = path_pose(6.0) + np.array([0.3, 0.2, math.radians(25)])
    out = []
    # Boxed in for longer than a loss is allowed: no relocalization, a
    # fresh map, the restored one on the shelf.
    n = int(sl.RESEED_LOST_S / PERIOD) + 10
    for k in range(n):
        t = 100.0 + (k + 1) * PERIOD
        for j in range(10):
            imu.add(t - PERIOD + (j + 1) * 0.01, (0.0, 0.0, 0.0), LEVEL)
        out.append(new.step(t, scan_at(t, lambda tt: put,
                                       boxed(put[0], put[1]))))
    assert "lost" in [r.reseed for r in out]
    assert new._shelf is not None and new._shelf.grid is restored
    assert new.kept()[0] is restored, "the restored map is still the one kept"
    # The box goes: the fresh map is started again on the room, and the
    # shelf's search finds the restored map under it.
    out += put_down(new, imu, put, 300, 100.0 + (n + 1) * PERIOD)
    assert "merged" in [r.reseed for r in out]
    assert new.grid is restored and new._shelf is None
    e_xy, e_yaw = err_vs_truth(out[-1]._replace(t=0.0),
                               pose_fn=lambda t: put if t == 0.0 else path_pose(t))
    assert e_xy < 0.05 and e_yaw < math.radians(2)


def test_after_a_loss_the_map_kept_is_the_shelved_one():
    slam = mapped()
    old, writes = slam.grid, slam._map_writes
    put = np.array([3.5, 2.6, math.radians(-100)])
    put_down(slam, slam.imu, put, int(sl.RESEED_LOST_S / PERIOD) + 20, 6.0)
    assert slam._shelf is not None and slam.grid is not old
    snap = slam.snapshot()
    assert snap.writes == writes and same_cells(snap.grid(), old)


def test_a_map_with_no_walls_near_its_pose_is_not_restored():
    snap = mapped().snapshot()._replace(pose=(40.0, 40.0, 0.0))
    slam = sl.Slam(imu=ImuTrack())
    assert not slam.restore(snap)
    assert not slam.started and slam.grid.bounds() is None


def test_a_saved_track_fits_the_restored_map_where_it_was():
    pytest.importorskip("scipy")
    from crawler.mapping import trackfit as tf
    slam = mapped()
    walls = pt._occupied_points(rn.map_view(slam, 6.0))
    new, imu = booted(slam.snapshot())
    # Before it has found itself, and after: the best fit is no turn and no
    # shift, every wall agreeing. (It is not placed on its own: this test
    # room is near enough the same turned round that a second place scores
    # well too, and that is trackfit's to refuse.)
    for view in (rn.map_view(new, 100.0), None):
        if view is None:
            put_down(new, imu, path_pose(6.0), 20, 100.0)
            assert new.recoveries == 1
            view = rn.map_view(new, 102.0)
        fit = tf.fit(walls, view)
        assert fit is not None and fit.share > 0.95
        assert abs(fit.yaw) < math.radians(1.5) and math.hypot(fit.x, fit.y) < 0.1


class Said:
    def __init__(self):
        self.lines = []

    def log(self, level, subject, message):
        self.lines.append((level, message))


def feed(r, scans, t0=0.0, pose_fn=path_pose):
    for k in range(scans):
        t = t0 + (k + 1) * PERIOD
        r.on_scan(FakeScan(t, scan_at(t, pose_fn)))
        end = time.monotonic() + 5.0
        while r.results + r.dropped < r.scans and time.monotonic() < end:
            time.sleep(0.002)


def wait_for(cond, s=5.0):
    end = time.monotonic() + s
    while not cond() and time.monotonic() < end:
        time.sleep(0.01)
    return cond()


@pytest.mark.parametrize("mode", [rn.MODE_THREAD, rn.MODE_PROCESS])
def test_the_runner_saves_on_close_and_the_next_boot_starts_from_it(
        monkeypatch, tmp_path, mode):
    monkeypatch.setattr(rn, "ranges_of", lambda scan, mount: scan.ranges)
    path = tmp_path / "slam_map.npz"
    r = rn.SlamRunner(mode=mode, map_path=path)
    r.start()
    try:
        feed(r, 40)
        assert r.latest().state == sl.TRACKING
    finally:
        r.close()
    snap, why = mf.load(path)
    assert snap is not None, why
    assert snap.writes >= 30
    said = Said()
    r = rn.SlamRunner(mode=mode, map_path=path, logger=said, restore_map=True)
    r.start()
    try:
        here = path_pose(4.0)
        feed(r, 20, t0=50.0, pose_fn=lambda t: here)
        assert wait_for(lambda: any("Restored the map" in m
                                    for _, m in said.lines)), said.lines
        res = r.latest()
        assert res.state == sl.TRACKING
        # On the first run's map, in its frame: not a fresh map at the
        # origin of this one.
        e_xy, _ = err_vs_truth(
            res._replace(t=0.0),
            pose_fn=lambda t: here if t == 0.0 else path_pose(t))
        assert e_xy < 0.05
        assert r.stats()["reseeds"] == 0
    finally:
        r.close()


def test_a_boot_starts_a_new_map_and_keeps_the_last_one_aside(
        monkeypatch, tmp_path):
    # 10-30: the restored map came back at every restart, and at 11:35 and
    # 11:47 it was one written turned at 11:33. Every boot starts afresh.
    monkeypatch.setattr(rn, "ranges_of", lambda scan, mount: scan.ranges)
    path = tmp_path / "slam_map.npz"
    r = rn.SlamRunner(map_path=path)
    r.start()
    try:
        feed(r, 40)
    finally:
        r.close()
    old = mf.load(path)[0]
    assert old is not None and old.writes >= 30
    said = Said()
    r = rn.SlamRunner(map_path=path, logger=said)
    r.start()
    try:
        here = path_pose(4.0)
        feed(r, 20, t0=50.0, pose_fn=lambda t: here)
        assert wait_for(lambda: any("Starting a new map" in m
                                    for _, m in said.lines)), said.lines
        assert not any("Restored the map" in m for _, m in said.lines)
        res = r.latest()
        assert res.state == sl.TRACKING
        assert math.hypot(res.x, res.y) < 0.05, \
            "at the origin of a map of its own, not on the last run's"
    finally:
        r.close()
    prev = mf.load(mf.previous_path(path))[0]
    assert prev is not None and prev.writes == old.writes, \
        "the last run's map is kept to look at"
    new = mf.load(path)[0]
    assert new is not None and new.writes <= 20


def test_the_runner_maps_from_nothing_past_a_bad_file(monkeypatch, tmp_path):
    monkeypatch.setattr(rn, "ranges_of", lambda scan, mount: scan.ranges)
    path = tmp_path / "slam_map.npz"
    path.write_bytes(b"\x00" * 1000)
    said = Said()
    r = rn.SlamRunner(map_path=path, logger=said, restore_map=True)
    r.start()
    try:
        feed(r, 40)
        res = r.latest()
        assert res.state == sl.TRACKING and err_vs_truth(res)[0] < 0.03
        assert wait_for(lambda: any("The saved map is ignored" in m
                                    for _, m in said.lines)), said.lines
    finally:
        r.close()
    assert mf.load(path)[0] is not None, "the bad file is replaced"


def test_the_worker_saves_on_sigterm(monkeypatch, tmp_path):
    # systemd stops the service by sending SIGTERM to every process in it,
    # the mapping worker as well as main(): before, it died on the spot.
    monkeypatch.setattr(rn, "ranges_of", lambda scan, mount: scan.ranges)
    path = tmp_path / "slam_map.npz"
    r = rn.SlamRunner(mode=rn.MODE_PROCESS, map_path=path)
    r.start()
    try:
        feed(r, 40)
        assert not path.exists(), "nothing saved yet: under SAVE_EVERY_S"
        w = r._worker
        os.kill(w.pid, signal.SIGTERM)
        w.join(5.0)
        assert not w.is_alive() and w.exitcode == 0
        assert mf.load(path)[0] is not None
    finally:
        r.close()


def test_the_worker_saves_every_so_often_without_waiting_on_the_disk(
        monkeypatch, tmp_path):
    slow = threading.Event()
    real = mf.save

    def slow_save(path, snap, **k):
        slow.wait(2.0)
        return real(path, snap, **k)
    monkeypatch.setattr(mf, "save", slow_save)
    slam = mapped(30)
    keeper = rn._MapKeeper(slam, tmp_path / "m.npz", _Box())
    t0 = time.perf_counter()
    keeper.tick(0.0)
    keeper.tick(mf.SAVE_EVERY_S + 0.01)
    assert time.perf_counter() - t0 < 0.5, "the step never waits on a save"
    assert not (tmp_path / "m.npz").exists()
    slow.set()
    assert keeper.saver.flush(5.0) and (tmp_path / "m.npz").exists()
    # Nothing has changed since: no second write.
    keeper.tick(2 * mf.SAVE_EVERY_S + 0.02)
    assert keeper.saver.flush(5.0) and keeper.saver.saves == 1
    drive(slam, 10, t0=3.0, imu=slam.imu)
    keeper.tick(3 * mf.SAVE_EVERY_S + 0.03)
    assert keeper.saver.flush(5.0) and keeper.saver.saves == 2


class _Box:
    def put_nowait(self, item):
        pass

