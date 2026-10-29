"""The live mapper: the grid, the gates, the rollover freeze, de-skew, the
runner's never-block rule, and a synthetic room it has to get the pose of.

No sensor and no recording: a room of line segments is ray cast into the
same 720 slots /lidar/scan carries, each slot at the pose the truck had
when the beam went round to it.
"""

import math
import queue
import time

import numpy as np
import pytest

pytest.importorskip("scipy")

from crawler.hardware import lidar as ld
from crawler.mapping import clock as ck
from crawler.mapping import grid as gr
from crawler.mapping import runner as rn
from crawler.mapping import scan as sc
from crawler.mapping import slam as sl
from crawler.mapping.imu import ImuTrack

LEVEL = (-1.0, 0.0, 0.0)
PERIOD = 0.1


def tilted_up(deg):
    """The device up vector with the truck pitched `deg` nose down."""
    a = math.radians(deg)
    return (-math.cos(a), 0.0, math.sin(a))


# ---- a synthetic room ------------------------------------------------------

# 7 x 5 m, a pillar, a box and a short wall: enough corners that the fit is
# pinned in both directions everywhere on the path.
ROOM = [((-3.0, -2.0), (4.0, -2.0)), ((4.0, -2.0), (4.0, 3.0)),
        ((4.0, 3.0), (-3.0, 3.0)), ((-3.0, 3.0), (-3.0, -2.0)),
        ((1.0, 1.0), (1.6, 1.0)), ((1.6, 1.0), (1.6, 1.4)),
        ((1.6, 1.4), (1.0, 1.4)), ((1.0, 1.4), (1.0, 1.0)),
        ((-1.5, -0.8), (-1.5, 0.4)), ((2.5, -2.0), (2.5, -1.2))]


def ray_cast(x, y, bearing, segs=ROOM):
    """Distance from (x, y) along `bearing` (rad, map) to the nearest segment."""
    dx, dy = math.cos(bearing), math.sin(bearing)
    best = math.inf
    for (ax, ay), (bx, by) in segs:
        ex, ey = bx - ax, by - ay
        den = dx * ey - dy * ex
        if abs(den) < 1e-12:
            continue
        t = ((ax - x) * ey - (ay - y) * ex) / den
        u = ((ax - x) * dy - (ay - y) * dx) / den
        if t > 0 and 0 <= u <= 1:
            best = min(best, t)
    return best


def path_pose(t):
    """A slow figure round the room: 0.25 m/s, turning."""
    return np.array([0.5 * math.sin(0.25 * t) + 0.3 * t / 10.0,
                     0.6 * math.sin(0.5 * t),
                     0.6 * math.sin(0.3 * t)])


def scan_at(t_end, pose_fn=path_pose, segs=ROOM):
    """720 ranges, each slot measured at its own moment in the revolution."""
    out = np.full(sc.SLOTS, np.nan, np.float32)
    for i in range(sc.SLOTS):
        ti = t_end - sc.FRAC_BEFORE_END[i] * PERIOD
        x, y, yaw = pose_fn(ti)
        r = ray_cast(x, y, yaw + sc.CAR_B[i], segs)
        if r < 9.0:
            out[i] = r
    return out


def drive(slam, scans, t0=0.0, pose_fn=path_pose, imu=None, up=LEVEL):
    """Step `scans` revolutions; with an ImuTrack, the true yaw rate at 100 Hz."""
    out = []
    for k in range(scans):
        t = t0 + (k + 1) * PERIOD
        if imu is not None:
            for j in range(10):
                tj = t - PERIOD + (j + 1) * 0.01
                rate = (pose_fn(tj + 1e-3)[2] - pose_fn(tj - 1e-3)[2]) / 2e-3
                imu.add(tj, (-rate, 0.0, 0.0), up)
        out.append(slam.step(t, scan_at(t, pose_fn)))
    return out


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def err_vs_truth(res, pose_fn=path_pose):
    """Pose error against the truth, in the boot frame."""
    p0 = pose_fn(PERIOD)
    truth = pose_fn(res.t)
    c, s = math.cos(-p0[2]), math.sin(-p0[2])
    dx, dy = truth[0] - p0[0], truth[1] - p0[1]
    tx, ty = c * dx - s * dy, s * dx + c * dy
    return math.hypot(res.x - tx, res.y - ty), abs(wrap(res.yaw - (truth[2] - p0[2])))


# ---- the grid --------------------------------------------------------------

def wall_scan(g, wall_x, n_rays=41, half=0.5):
    """Rays from the origin to a wall at x = wall_x, |y| <= half."""
    ys = np.linspace(-half, half, n_rays)
    g.integrate(0.0, 0.0, np.full(n_rays, wall_x), ys)


def test_a_removed_obstacle_clears_in_the_passes_the_clamp_allows():
    g = gr.Grid(size=256, l_max=gr.L_MAX)
    for _ in range(40):
        wall_scan(g, 2.0)
    ix, iy = g.world_to_cell(2.0, 0.0)
    assert g.lo[ix, iy] == gr.L_MAX, "clamped, however often it was seen"
    assert g.occupied()[ix, iy]
    # The wall is taken away: the same rays now go on to 4 m.
    want = math.ceil((gr.L_MAX - gr.L_OCC) / -gr.L_FREE)
    passes = 0
    while g.occupied()[ix, iy] and passes < 50:
        wall_scan(g, 4.0)
        passes += 1
    assert passes == want
    # And it reads as free, not unknown, soon after.
    for _ in range(3):
        wall_scan(g, 4.0)
    assert g.lo[ix, iy] < 0


def test_a_higher_clamp_takes_more_passes():
    def passes_for(l_max):
        g = gr.Grid(size=256, l_max=l_max)
        for _ in range(40):
            wall_scan(g, 2.0)
        ix, iy = g.world_to_cell(2.0, 0.0)
        n = 0
        while g.occupied()[ix, iy] and n < 50:
            wall_scan(g, 4.0)
            n += 1
        return n
    assert passes_for(20) < passes_for(30) < passes_for(70)


def test_a_wall_seen_end_on_is_not_cleared_by_rays_along_it():
    g = gr.Grid(size=256)
    xs = np.linspace(0.5, 3.0, 50)
    for _ in range(10):
        g.integrate(0.0, 0.0, xs, np.full_like(xs, 0.01))
    ix, iy = g.world_to_cell(1.5, 0.01)
    assert g.occupied()[ix, iy]


def test_the_grid_scrolls_and_keeps_what_it_had():
    g = gr.Grid(size=256)
    wall_scan(g, 2.0)
    for _ in range(5):
        wall_scan(g, 2.0)
    before = g.occupied().sum()
    # 256 cells of 5 cm is 12.8 m: 6 m out is within a metre of the edge.
    assert g.scroll_to(6.0, 0.0, margin=1.0)
    assert g.scrolls == 1
    assert g.occupied().sum() == before
    ix, iy = g.world_to_cell(2.0, 0.0)
    assert g.occupied()[ix, iy]


# ---- the clock and the IMU -------------------------------------------------------

def test_the_sensor_clock_takes_the_promptest_read_and_unwraps():
    c = ck.SensorClock()
    rng = np.random.default_rng(1)
    host0 = 100.0
    fits = []
    for k in range(300):
        true_end = host0 + k * 0.1
        stamp = int(round(29000 + k * 100)) % ck.STAMP_WRAP_MS   # wraps at k=10
        host = true_end + 0.004 + rng.uniform(0, 0.05)
        fits.append(c.fit(host, stamp) - true_end)
    late = np.array(fits[50:])
    assert c.resets == 0
    assert np.all(np.abs(late - 0.004) < 0.004)


def test_the_imu_heading_carries_on_past_the_newest_sample():
    imu = ImuTrack()
    for k in range(21):
        imu.add(k * 0.05, (-1.0, 0.0, 0.0), LEVEL)   # 1 rad/s CCW
    # The newest sample is at 1.0 s; a scan ending at 1.04 s still turned.
    assert imu.yaw_between(0.9, 1.04) == pytest.approx(0.14, abs=1e-6)


# ---- de-skew and the ground --------------------------------------------------------

def test_deskew_puts_every_point_where_it_was_at_the_end():
    omega, v = 1.5, np.array([0.4, -0.1])
    wall = (2.0, 0.3)                       # a fixed point, end-frame coordinates
    idx = np.arange(0, sc.SLOTS, 37)
    tau = -sc.FRAC_BEFORE_END[idx] * PERIOD    # seconds before the end, <= 0
    # Where the car was at tau, and the point as it saw it then.
    th = omega * tau
    seen = []
    for a, d in zip(th, tau):
        px, py = wall[0] - v[0] * d, wall[1] - v[1] * d
        c, s = math.cos(-a), math.sin(-a)
        seen.append((c * px - s * py, s * px + c * py))
    p, _ = sc.deskew(np.array(seen), idx, PERIOD, omega, v)
    assert np.allclose(p, wall, atol=1e-9)


def test_the_ground_guard_drops_far_points_down_the_slope_only():
    pts = np.array([[1.0, 0.0], [3.0, 0.0], [-3.0, 0.0], [0.0, 3.0]])
    keep_level = sc.ground_mask(pts, LEVEL)
    assert not keep_level.any()
    m = sc.ground_mask(pts, tilted_up(6.0))
    # 0.6 * 0.30 m at 6 degrees is 1.7 m along the slope, either way.
    assert m.tolist() == [False, True, True, False]


# ---- the gates ----------------------------------------------------------------

def test_a_tilted_truck_tracks_but_does_not_write():
    imu = ImuTrack()
    slam = sl.Slam(imu=imu)
    drive(slam, 20, imu=imu)
    v = slam.grid.version
    out = drive(slam, 10, t0=2.0, imu=imu, up=tilted_up(10.0))
    assert all(r.state == sl.TILTED and not r.written for r in out)
    assert all(r.matched for r in out)
    assert slam.grid.version == v, "nothing went into the map"
    out = drive(slam, 5, t0=3.0, imu=imu)
    assert out[-1].state == sl.TRACKING and out[-1].written


def test_a_rollover_freezes_the_pose_and_recovers_after():
    imu = ImuTrack()
    slam = sl.Slam(imu=imu)
    drive(slam, 30, imu=imu)
    frozen = slam.pose.copy()
    v = slam.grid.version

    def tumbling(t):
        return np.array([0.0, 0.0, 3.0 * t])   # spinning in a hand
    out = []
    for k in range(10):
        t = 3.0 + (k + 1) * PERIOD
        for j in range(10):
            imu.add(t - PERIOD + (j + 1) * 0.01, (-3.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        out.append(slam.step(t, scan_at(t, tumbling)))
    assert all(r.state == sl.ROLLED and not r.written for r in out)
    assert np.allclose(slam.pose, frozen)
    assert slam.grid.version == v
    assert slam.lost, "it does not believe the frozen pose any more"
    # Set back down where it was, level again: found by the recovery search
    # (not by an ordinary match), and tracking from then on.
    back = drive(slam, 30, t0=4.0, imu=imu,
                 pose_fn=lambda t: path_pose(3.0))
    assert slam.recoveries == 1
    assert all(r.state == sl.TRACKING for r in back[1:])
    e_xy, e_yaw = err_vs_truth(back[-1]._replace(t=3.0))
    assert e_xy < 0.05 and e_yaw < math.radians(2)


def test_the_synthetic_room_is_mapped_with_centimetre_pose_error():
    imu = ImuTrack()
    slam = sl.Slam(imu=imu)
    out = drive(slam, 300, imu=imu)
    errs = [err_vs_truth(r) for r in out]
    xy = max(e[0] for e in errs)
    yaw = max(e[1] for e in errs)
    assert xy < 0.03, f"worst position error {xy * 100:.1f} cm"
    assert yaw < math.radians(1.0)
    assert sum(r.written for r in out) > 0.9 * len(out)


def test_the_room_without_an_imu_still_maps():
    slam = sl.Slam(imu=ImuTrack())
    out = drive(slam, 150)
    assert math.isnan(out[-1].tilt_deg)
    assert max(err_vs_truth(r)[0] for r in out) < 0.05


# ---- the runner --------------------------------------------------------------

class FakeScan:
    def __init__(self, t, ranges, stamp_ms=None):
        self.t = t
        self.ranges = ranges
        self.stamp_ms = stamp_ms


class Parked:
    """A worker that never takes anything: the mapper fallen behind."""
    def is_alive(self):
        return True


def test_scans_are_dropped_and_counted_when_the_mapper_is_behind(monkeypatch):
    monkeypatch.setattr(rn, "ranges_of", lambda scan, mount: scan.ranges)
    r = rn.SlamRunner(inbox=2)
    r._inbox = queue.Queue(maxsize=2)
    r._worker = Parked()
    r.on_imu(1.0, {"gyro": (0.0, 0.0, 0.0), "up": LEVEL})
    t0 = time.perf_counter()
    for k in range(5):
        r.on_imu(1.0 + 0.1 * k + 0.05, {"gyro": (0.1, 0.0, 0.0)})
        r.on_scan(FakeScan(1.0 + 0.1 * k, np.zeros(sc.SLOTS, np.float32)))
    took = time.perf_counter() - t0
    assert r.scans == 5 and r.dropped == 3
    assert took < 0.05, "on_scan never waits on the worker"
    first = r._inbox.get_nowait()
    assert len(first[3]) == 2
    # The dropped scans' IMU samples are kept for the next scan, not lost.
    assert len(r._imu) == 3
    assert np.isnan(r._imu[0][4]), "no up vector in the sample: NaN"


@pytest.mark.parametrize("mode", [rn.MODE_THREAD, rn.MODE_PROCESS])
def test_the_runner_maps_the_room_on_its_worker(monkeypatch, mode):
    monkeypatch.setattr(rn, "ranges_of", lambda scan, mount: scan.ranges)
    r = rn.SlamRunner(mode=mode, map_every_s=0.0)
    r.start()
    try:
        n = 40
        for k in range(n):
            t = (k + 1) * PERIOD
            r.on_scan(FakeScan(t, scan_at(t)))
            # Paced so neither worker falls behind; the drop test is above.
            end = time.monotonic() + 5.0
            while r.results + r.dropped < k + 1 and time.monotonic() < end:
                time.sleep(0.002)
        assert r.results == n and r.dropped == 0
        res = r.latest()
        assert res.state == sl.TRACKING
        assert err_vs_truth(res)[0] < 0.03
        view = r.map()
        assert view is not None and (view.codes == gr.OCCUPIED).any()
        assert r.pose() == (res.x, res.y, res.yaw)
        assert r.owns_tf()
        stamp = r.pose_stamped()
        assert (stamp.t, stamp.x) == (res.t, res.x) and stamp.confidence > 0.5
        mid = r.pose_at(res.t - 0.05)
        assert mid is not None and abs(mid.t - (res.t - 0.05)) < 1e-9
    finally:
        r.close()
    assert not r.stats()["alive"]


def test_pose_at_interpolates_between_scans_and_refuses_outside_them():
    P = rn.PoseStamp
    poses = [P(1.0, 0.0, 0.0, math.radians(179), 0.9, "tracking"),
             P(1.1, 0.1, 0.2, math.radians(-179), 0.6, "tracking"),
             P(1.2, 0.2, 0.4, math.radians(-177), 0.8, "lost")]
    p = rn.interpolate(poses, 1.05)
    assert (p.x, p.y) == pytest.approx((0.05, 0.1))
    assert wrap(p.yaw - math.pi) == pytest.approx(0.0, abs=1e-9), "the short way round"
    assert p.confidence == 0.6
    assert rn.interpolate(poses, 1.15) is None, "across a change of state"
    assert rn.interpolate(poses, 0.9) is None, "older than anything kept"
    assert rn.interpolate(poses, 1.3).t == 1.2, "just past the newest: the newest"
    assert rn.interpolate(poses, 1.2 + rn.POSE_MAX_AHEAD_S + 0.01) is None


def test_confidence_is_zero_unless_the_pose_matched():
    res = sl.Slam(imu=ImuTrack()).step(0.1, scan_at(0.1))
    good = res._replace(state=sl.TRACKING, matched=True, inliers=0.9,
                        constraint=0.4)
    assert rn.confidence(good) == pytest.approx(0.9)
    assert rn.confidence(good._replace(constraint=0.03)) == pytest.approx(0.18)
    assert rn.confidence(good._replace(state=sl.TILTED)) == pytest.approx(0.45)
    for state in (sl.LOST, sl.ROLLED, sl.NO_POINTS):
        assert rn.confidence(good._replace(state=state)) == 0.0
    assert rn.confidence(good._replace(matched=False)) == 0.0


# ---- --slam off changes nothing ------------------------------------------------

def test_slam_is_off_unless_asked_for():
    from crawler import __main__ as cm
    args = cm.parse_args([])
    assert args.slam is False
    args = cm.parse_args(["--lidar", "--slam"])
    assert args.slam is True and args.slam_worker == "process"


class AllTele:
    def __init__(self):
        self.rows = []

    def wants(self, key, now):
        return True

    def offer(self, key, payload, now):
        self.rows.append((key, payload))


def scan_of(angles, dists):
    return ld.Scan(t=0.0, angles_deg=np.asarray(angles, np.float32),
                   distances_mm=np.asarray(dists, np.uint16),
                   intensities=np.full(len(angles), 200, np.uint8),
                   speed_dps=3600.0, packets=42)


class FakeRunner:
    def __init__(self, owns):
        self.owns = owns
        self.got = []

    def on_scan(self, scan):
        self.got.append(scan)

    def owns_tf(self, now=None):
        return self.owns


def test_the_reader_hands_scans_to_the_mapper_and_leaves_it_tf():
    for owns, want in ((True, ["lidar", "lidar_scan", "lidar_truck"]),
                       (False, ["lidar", "tf", "lidar_scan", "lidar_truck"])):
        tele = AllTele()
        slam = FakeRunner(owns)
        reader = ld.LidarReader(tele=tele, slam=slam)
        reader._publish(scan_of([270.0], [1000]), 1.0)
        assert [k for k, _ in tele.rows] == want
        assert len(slam.got) == 1


def test_a_mapper_that_raises_costs_the_reader_nothing():
    class Broken(FakeRunner):
        def on_scan(self, scan):
            raise RuntimeError("no")
    tele = AllTele()
    reader = ld.LidarReader(tele=tele, slam=Broken(False))
    reader._publish(scan_of([270.0], [1000]), 1.0)
    assert reader.scans == 1 and [k for k, _ in tele.rows][0] == "lidar"


# ---- the lidar's own stamps ---------------------------------------------------------

def test_a_scan_carries_its_packets_sensor_stamps():
    asm = ld.ScanAssembler()
    scans = []
    k = 0
    for turn in range(3):
        for p in range(40):
            start = p * 900
            pkt = ld.decode_packet(ld.encode_packet(
                3600, start, start + 825, [2000] * 12, timestamp_ms=int(k * 2.5)))
            s = asm.add(pkt, 1.0 + k * 0.0025)
            if s is not None:
                scans.append(s)
            k += 1
    assert len(scans) >= 1
    s = scans[-1]
    assert s.stamp_ms == s.stamps_ms[-1]
    assert len(s.stamps_ms) == s.packets + 1
    assert list(s.stamps_ms) == sorted(s.stamps_ms)
    row = ld.lidar_sample(s, {})
    assert row["stamp_ms"] == s.stamp_ms and row["stamps_ms"] == list(s.stamps_ms)


# ---- the Foxglove side -----------------------------------------------------------

def decode(message):
    """An SDK message -> the protobuf message its own schema describes."""
    pytest.importorskip("google.protobuf")
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    schema = message.get_schema()
    files = descriptor_pb2.FileDescriptorSet.FromString(bytes(schema.data))
    pool = descriptor_pool.DescriptorPool()
    for f in files.file:
        pool.Add(f)
    cls = message_factory.GetMessageClass(pool.FindMessageTypeByName(schema.name))
    return cls.FromString(message.encode())


def test_the_map_grid_puts_each_pixel_where_the_map_has_it():
    pytest.importorskip("foxglove")
    from crawler.telemetry import telemetry as tm
    codes = np.zeros((5, 3), np.uint8)          # 5 along x, 3 along y
    codes[4, 1] = gr.OCCUPIED
    codes[0, 2] = gr.FREE
    view = rn.MapView(t=0.0, version=1, x0=-1.0, y0=2.0, res=0.1, codes=codes)
    g = decode(tm.Telemetry.lidar_view_message("map", view, 10**18))
    assert g.frame_id == "map" and g.column_count == 5
    assert g.row_stride == 20 and g.cell_stride == 4
    assert (g.pose.position.x, g.pose.position.y) == (-1.0, 2.0)
    data = np.frombuffer(g.data, np.uint8).reshape(3, 5, 4)   # [row=y, col=x]
    assert data[1, 4].tolist() == [240, 240, 240, 255]
    assert data[2, 0, 3] > 0 and data[0, 0, 3] == 0, "unknown is clear"


def test_tf_with_slam_hangs_base_link_off_the_map_at_the_scan_time():
    pytest.importorskip("foxglove")
    from crawler.telemetry import telemetry as tm
    msg = decode(tm.Telemetry.lidar_view_message(
        "tf", (ld.MOUNT, (1.0, -2.0, math.pi / 2), 50.0), 10**18))
    a, b = msg.transforms
    assert (a.parent_frame_id, a.child_frame_id) == ("base_link", "lidar")
    assert (b.parent_frame_id, b.child_frame_id) == ("map", "base_link")
    assert (b.translation.x, b.translation.y) == (1.0, -2.0)
    assert b.rotation.z == pytest.approx(math.sqrt(0.5))
    assert b.timestamp.ToNanoseconds() == pytest.approx(
        tm.epoch_ns(50.0), abs=1000)
    # Without SLAM it is the one transform it always was.
    assert len(decode(tm.Telemetry.lidar_view_message(
        "tf", ld.MOUNT, 10**18)).transforms) == 1


def test_the_map_layout_reads_topics_that_exist():
    import json
    import os
    from crawler.telemetry import telemetry as tm
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "deploy", "foxglove-map.json")) as f:
        layout = json.load(f)
    topics = (set(tm.JSON_TOPICS.values()) | set(tm.LIDAR_VIEW_CHANNELS.values())
              | set(tm.SLAM_CHANNELS.values()))
    three = layout["configById"]["3D!slam"]
    assert three["followTf"] == "map"
    assert set(three["topics"]) <= topics
    for key, cfg in layout["configById"].items():
        for p in cfg.get("paths", []):
            topic, field = p["value"].split(".", 1)
            assert topic in topics, key
            if topic == "/slam/status":
                assert field in tm.JSON_SCHEMAS["slam_status"]["schema"]["properties"]


def test_the_status_row_fits_its_schema():
    from crawler.telemetry import telemetry as tm
    res = sl.Slam(imu=ImuTrack()).step(0.1, scan_at(0.1))
    row = rn.status_row(res, {"scans": 3, "dropped": 1})
    props = tm.JSON_SCHEMAS["slam_status"]["schema"]["properties"]
    assert set(row) == set(props)
    kinds = {"number": (int, float), "integer": int, "string": str,
             "boolean": bool}
    for k, v in row.items():
        assert isinstance(v, kinds[props[k]["type"]]), k
    assert row["rms_mm"] == -1.0, "NaN goes out as -1"
