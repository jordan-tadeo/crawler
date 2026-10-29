"""The OAK-D voxel map: mapping/voxel.py, its Foxglove messages, and its wiring.

Most of these render a depth frame of a known scene through the same pinhole
the mapper inverts -- a flat floor seen from the measured lens height, a box
on it, a ramp -- and ask what comes back. No device, no recording.
"""

import math
import sys
import threading
import types

import numpy as np
import pytest

from crawler.mapping import voxel as vx


# ---- a scene renderer ----------------------------------------------------------

def render(boxes=(), ramp_from=None, ramp_deg=0.0, up=None, height=None,
           w=640, h=400):
    """A uint16 mm depth frame of a floor (optionally ramping up from
    `ramp_from` metres ahead at `ramp_deg`) with boxes standing on it.

    Each box is (ahead_m, half_width_m, tall_m): a front face square to the
    camera, `ahead_m` along the floor from the lens, its foot on the floor or
    the ramp under it."""
    up = vx.MOUNT_UP if up is None else np.asarray(up, float)
    up = up / np.linalg.norm(up)
    height = vx.CAMERA_HEIGHT_M if height is None else height
    f, l = vx.ground_axes(up)
    u = (np.arange(w) - vx.CX) / vx.FX
    v = (np.arange(h) - vx.CY) / vx.FX
    U, V = np.meshgrid(u, v)
    d = np.stack([U, V, np.ones_like(U)], -1)
    dn, df, dl = d @ up, d @ f, d @ l
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dn < 0, -height / dn, np.inf)
        if ramp_from is not None:
            # the ramp: height above the floor = (ahead - ramp_from) * tan
            s = math.tan(math.radians(ramp_deg))
            # p.n + height = s * (p.f - ramp_from)  ->  t (dn - s df) = -height - s ramp_from
            tr = (-height - s * ramp_from) / (dn - s * df)
            on = (tr > 0) & (tr * df > ramp_from)
            t = np.where(on & (tr < t), tr, t)
        for ahead, half, tall in boxes:
            base = 0.0
            if ramp_from is not None and ahead > ramp_from:
                base = (ahead - ramp_from) * math.tan(math.radians(ramp_deg))
            tb = ahead / df
            hb = tb * dn + height
            lat = tb * dl
            hit = ((df > 0) & (hb >= base) & (hb <= base + tall)
                   & (np.abs(lat) <= half))
            t = np.where(hit & (tb < t), tb, t)
    z = t                       # d has z = 1, so t is the depth
    z = np.where(np.isfinite(z) & (z < 10.0), z, 0.0)
    return np.round(z * 1000.0).astype(np.uint16)


def level_imu_up():
    """The IMU up vector that IMU_TO_CAMERA maps onto MOUNT_UP."""
    up = np.linalg.solve(vx.IMU_TO_CAMERA, vx.MOUNT_UP)
    return up / np.linalg.norm(up)


def low_cells(res, max_r=vx.LOW_RANGE_M):
    m = (res.cls == vx.LOW) & (res.r <= max_r)
    return m


# ---- the picture ---------------------------------------------------------------

def test_the_downsample_is_the_median_of_the_valid_pixels_not_the_nearest():
    frame = np.full((400, 640), 2000, np.uint16)
    frame[0, 0] = 300                # one speckle of near depth
    frame[1:3, 0:4] = 0              # and 8 holes in the same block
    small = vx.downsample(frame)
    assert small.shape == (100, 160)
    assert small[0, 0] == pytest.approx(2.0)


def test_a_block_with_too_few_valid_pixels_says_nothing():
    frame = np.full((400, 640), 1500, np.uint16)
    frame[0:4, 0:4] = 0
    frame[0, 0] = 1500               # 1 valid of 16
    assert np.isnan(vx.downsample(frame)[0, 0])
    frame[0:2, 0:4] = 1500           # 8 valid of 16
    assert vx.downsample(frame)[0, 0] == pytest.approx(1.5)


def test_a_flat_floor_keeps_every_pixel_through_the_edge_filter():
    z = vx.downsample(render())
    kept = vx.drop_edges(z)
    floor = np.isfinite(z)
    assert np.isfinite(kept[floor]).mean() > 0.99


def test_a_flying_pixel_between_a_leg_and_the_wall_is_dropped():
    z = np.full((10, 10), 2.0, np.float32)
    z[:, 4] = 0.8                    # a leg
    z[:, 5] = 1.4                    # the skirt the hole-filling paints behind it
    kept = vx.drop_edges(z)
    assert np.isnan(kept[3:7, 5]).all()
    assert np.isfinite(kept[3:7, 8]).all()


# ---- the floor -------------------------------------------------------------------

def test_the_floor_fit_finds_the_lens_height_on_a_flat_floor():
    res = vx.process_frame(render(), level_imu_up())
    assert res.floor.fitted
    assert res.floor.height == pytest.approx(vx.CAMERA_HEIGHT_M, abs=0.004)
    assert math.degrees(math.acos(min(1.0, res.floor.up @ vx.MOUNT_UP))) < 0.5


def test_a_bad_imu_is_only_the_seed_and_the_floor_still_wins():
    # The IMU 6 degrees out, pitched: at 2 m that is 21 cm of invented height
    # if it were believed.
    a = math.radians(6.0)
    rot = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)],
                    [0, math.sin(a), math.cos(a)]])
    bad = np.linalg.solve(vx.IMU_TO_CAMERA, rot @ vx.MOUNT_UP)
    res = vx.process_frame(render(), bad)
    assert res.floor.fitted
    assert not low_cells(res).any()


def test_flat_empty_floor_makes_no_low_obstacle():
    res = vx.process_frame(render(), level_imu_up())
    assert not low_cells(res).any()
    assert (res.cls == vx.GROUND).sum() > 3000


def test_a_shoe_sized_box_is_a_low_obstacle_where_it_is():
    res = vx.process_frame(render(boxes=[(0.9, 0.12, 0.08)]), level_imu_up())
    m = low_cells(res)
    assert m.sum() > 20
    # in base_link: the lens is CAMERA_X_M ahead of the lidar
    assert np.median(res.X[m]) == pytest.approx(0.9 + vx.CAMERA_X_M, abs=0.05)
    assert np.abs(np.median(res.Y[m])) < 0.05
    assert res.Z[m].max() == pytest.approx(0.08, abs=0.02)


def test_something_taller_than_the_low_band_is_high_above_it():
    res = vx.process_frame(render(boxes=[(1.0, 0.3, 0.6)]), level_imu_up())
    assert (res.cls == vx.HIGH).sum() > 50
    assert (low_cells(res)).sum() > 0      # its bottom 25 cm is still low


def test_a_ramp_is_ground_and_a_box_on_it_still_stands_up():
    ramp = render(ramp_from=0.6, ramp_deg=15.0)
    res = vx.process_frame(ramp, level_imu_up())
    assert not low_cells(res).any()
    # the plane alone would call the ramp an obstacle: it is 10+ cm up at 1 m
    up_the_ramp = (res.Z > vx.LOW_MIN_M) & (res.r <= vx.LOW_RANGE_M)
    assert up_the_ramp.sum() > 50
    with_box = render(ramp_from=0.6, ramp_deg=15.0, boxes=[(1.1, 0.1, 0.08)])
    res = vx.process_frame(with_box, level_imu_up())
    assert low_cells(res).sum() > 10


def test_the_left_edge_of_the_frame_is_never_mapped():
    # On the recorded floor the leftmost ~17 of 160 columns bend up 5-13 cm:
    # mapped, they were most of the phantom low cells left.
    res = vx.process_frame(render(boxes=[(0.9, 1.0, 0.1)]), level_imu_up())
    assert (res.cls[:, :17] == vx.UNKNOWN).all()
    assert (res.cls[:, 40:] != vx.UNKNOWN).any()


# ---- the maps --------------------------------------------------------------------

def test_one_frame_is_not_enough_for_a_low_cell_and_two_are():
    maps = vx.VoxelMaps()
    res = vx.process_frame(render(boxes=[(0.9, 0.12, 0.08)]), level_imu_up())
    vx.integrate(maps, res, (0.0, 0.0, 0.0))
    assert not maps.low_cells().any()
    vx.integrate(maps, res, (0.0, 0.0, 0.0))
    cells = np.argwhere(maps.low_cells())
    assert len(cells) > 0
    x = (cells[:, 0] + maps.ox + 0.5) * vx.RES_M
    assert np.median(x) == pytest.approx(0.9 + vx.CAMERA_X_M, abs=0.08)


def test_one_hit_is_not_an_occupied_voxel_and_two_are():
    maps = vx.VoxelMaps()
    pts = np.array([[1.0, 0.0, 0.1]])
    maps.integrate_3d(np.zeros(3), pts)
    assert not maps.occupied().any()
    maps.integrate_3d(np.zeros(3), pts)
    assert maps.occupied().sum() == 1


def test_free_space_clears_what_a_ray_passes_through():
    maps = vx.VoxelMaps()
    o = np.array([0.0, 0.0, 0.1])
    for _ in range(3):
        maps.integrate_3d(o, np.array([[1.0, 0.0, 0.1]]))
    assert maps.occupied().sum() == 1
    for _ in range(4):
        maps.integrate_3d(o, np.array([[2.0, 0.0, 0.1]]), free_from=np.array([[2.0, 0.0, 0.1]]))
    i, j = maps.cells(1.0, 0.0)
    k = int((0.1 - vx.Z0_M) / vx.RES_Z_M)
    assert maps.lo[i, j, k] <= vx.L_OCC
    assert maps.occupied()[maps.cells(2.0, 0.0)[0], maps.cells(2.0, 0.0)[1]].any()


def test_a_floor_seen_again_takes_a_phantom_back_out():
    maps = vx.VoxelMaps()
    box = vx.process_frame(render(boxes=[(0.9, 0.12, 0.08)]), level_imu_up())
    empty = vx.process_frame(render(), level_imu_up())
    for _ in range(2):
        vx.integrate(maps, box, (0.0, 0.0, 0.0))
    assert maps.low_cells().any()
    for _ in range(3):
        vx.integrate(maps, empty, (0.0, 0.0, 0.0))
    assert not maps.low_cells().any()


def test_the_window_follows_the_truck_and_keeps_the_world_where_it_was():
    maps = vx.VoxelMaps(size=200)
    maps.integrate_3d(np.zeros(3), np.array([[1.0, 1.0, 0.1]] * 1))
    maps.integrate_3d(np.zeros(3), np.array([[1.0, 1.0, 0.1]]))
    before = maps.occupied_points()
    assert maps.follow(6.0, 0.0) is True
    after = maps.occupied_points()
    np.testing.assert_allclose(before, after)
    assert maps.follow(6.0, 0.0) is False


def test_the_map_is_placed_by_the_pose():
    maps = vx.VoxelMaps()
    res = vx.process_frame(render(boxes=[(0.9, 0.12, 0.08)]), level_imu_up())
    pose = (2.0, -1.0, math.pi / 2)        # facing +y
    for _ in range(2):
        vx.integrate(maps, res, pose)
    cells = np.argwhere(maps.low_cells())
    x = (cells[:, 0] + maps.ox + 0.5) * vx.RES_M
    y = (cells[:, 1] + maps.oy + 0.5) * vx.RES_M
    assert np.median(x) == pytest.approx(2.0, abs=0.08)
    assert np.median(y) == pytest.approx(-1.0 + 0.9 + vx.CAMERA_X_M, abs=0.08)


def test_the_low_view_says_floor_obstacle_and_unseen():
    maps = vx.VoxelMaps()
    res = vx.process_frame(render(boxes=[(0.9, 0.12, 0.08)]), level_imu_up())
    for _ in range(2):
        vx.integrate(maps, res, (0.0, 0.0, 0.0))
    view = vx.low_view(maps, 0.0, 0.0, 1.0)
    vals = set(np.unique(view.codes).tolist())
    assert vals == {vx.LOW_UNSEEN, vx.LOW_FLOOR, vx.LOW_OBSTACLE}


# ---- the live half ---------------------------------------------------------------

class Pose(tuple):
    pass


def stamp(t, confidence=1.0, state="tracking"):
    from collections import namedtuple
    P = namedtuple("P", "t x y yaw confidence state")
    return P(t, 0.0, 0.0, 0.0, confidence, state)


class FakeSlam:
    def __init__(self, pose=None):
        self.pose = pose
        self.asked = []

    def pose_at(self, t):
        self.asked.append(t)
        return self.pose


class FakeTele:
    def __init__(self):
        self.offered = []

    def wants(self, key, now):
        return True

    def offer(self, key, payload, now):
        self.offered.append(key)


def test_a_frame_with_no_pose_worth_trusting_is_not_mapped():
    frame = render()
    for pose in (None, stamp(1.0, confidence=0.3), stamp(1.0, state="tilted")):
        m = vx.VoxelMapper(FakeSlam(pose))
        assert m.step(1.0, frame) is False
        assert m.skipped_pose == 1 and m.sent == 0


def test_a_frame_is_placed_by_its_capture_time_not_its_arrival():
    slam = FakeSlam(stamp(10.0))
    m = vx.VoxelMapper(slam)
    m.on_depth(10.3, render(), capture_t=10.21)
    m.step(*m._slot)
    assert slam.asked == [10.21]
    # a stamp that cannot be right falls back to arrival less the frame age
    slam.asked.clear()
    m.on_depth(10.3, render(), capture_t=500.0)
    m.step(*m._slot)
    assert slam.asked == [pytest.approx(10.3 - vx.FRAME_AGE_S)]


def test_frames_that_come_while_it_is_busy_are_dropped_not_queued():
    m = vx.VoxelMapper(FakeSlam(stamp(1.0)))
    frame = render()
    for k in range(5):
        m.on_depth(1.0 + k, frame, None)
    assert m.frames == 5 and m.dropped == 4
    assert m._slot[0] == pytest.approx(5.0 - vx.FRAME_AGE_S)


@pytest.mark.parametrize("mode", [vx.MODE_THREAD, vx.MODE_PROCESS])
def test_the_worker_maps_and_publishes_and_closes(mode):
    tele = FakeTele()
    m = vx.VoxelMapper(FakeSlam(stamp(1.0)), tele=tele, mode=mode)
    m.start()
    try:
        m.on_imu(1.0, {"up": tuple(level_imu_up())})
        m.on_depth(1.0, render(boxes=[(0.9, 0.12, 0.08)]), 1.0)
        for _ in range(1000):
            if m.mapped and len(tele.offered) >= 2:
                break
            threading.Event().wait(0.01)
    finally:
        m.close()
    assert m.mapped == 1 and m.errors == 0 and m.sent == 1
    assert tele.offered == ["voxel_points", "voxel_low"]
    st = m.stats()
    assert st["alive"] is False and st["mode"] == mode


def test_a_frame_that_breaks_the_worker_costs_only_that_frame():
    m = vx.VoxelMapper(FakeSlam(stamp(1.0)), mode=vx.MODE_THREAD)
    m.start()
    try:
        m.on_depth(1.0, np.zeros((3, 3), np.uint16), 1.0)    # not a frame
        for _ in range(500):
            if m.errors:
                break
            threading.Event().wait(0.01)
        m.on_depth(2.0, render(), 2.0)
        for _ in range(500):
            if m.mapped:
                break
            threading.Event().wait(0.01)
    finally:
        m.close()
    assert m.errors == 1 and m.mapped == 1


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_the_feeder_sends_one_frame_at_a_time_and_gives_up_on_a_lost_one():
    clock = Clock()
    m = vx.VoxelMapper(FakeSlam(stamp(1.0)), clock=clock, mode=vx.MODE_THREAD)
    import queue
    m._inbox = queue.Queue(maxsize=1)           # a worker that never answers
    assert m.step(1.0, render()) is True
    assert m.busy()
    clock.t = vx.IN_FLIGHT_GIVE_UP_S - 0.1
    assert m.busy()
    clock.t = vx.IN_FLIGHT_GIVE_UP_S + 0.1
    assert not m.busy() and m.lost == 1
    m._take(("done", 12.0, True))
    assert m.mapped == 1 and m.ms[-1] == 12.0 and not m.busy()


def test_the_mapper_has_nothing_to_drive_with():
    """It is handed a SLAM runner and a telemetry sink, and nothing else: no
    vehicle controller, no pad, no autopilot reaches it."""
    import inspect
    params = inspect.signature(vx.VoxelMapper.__init__).parameters
    assert set(params) == {"self", "slam", "tele", "logger", "clock", "mode"}
    src = inspect.getsource(vx)
    for word in ("throttle", "steer", "vecon", "set_pulse", "pigpio"):
        assert word not in src.lower().replace("steering", "")


def test_the_imu_up_nearest_the_frame_is_used_and_a_stale_one_is_not():
    m = vx.VoxelMapper(FakeSlam())
    m.on_imu(1.00, {"up": (-1.0, 0.0, 0.0)})
    m.on_imu(1.05, {"up": (-0.9, 0.1, 0.0)})
    m.on_imu(1.06, {"accel": (0, 0, 0)})        # no up: ignored
    assert m.imu_up_at(1.04) == (-0.9, 0.1, 0.0)
    assert m.imu_up_at(2.0) is None


# ---- the camera hands it frames ----------------------------------------------------

def test_the_reader_hands_depth_listeners_the_frame_and_its_capture_time(monkeypatch):
    from crawler.hardware import camera
    frame = np.zeros((400, 640), np.uint16)
    got = []

    class OneFrameOakD:
        def __init__(self, rectified=True):
            self.sent = False

        def get_depth_frame(self):
            if self.sent:
                return None
            self.sent = True
            return frame

        def capture_time(self, label):
            return 123.5 if label == "depth" else None

        def get_disparity_frame(self):
            return None

        get_rectified_left_frame = get_imu_sample = get_disparity_frame

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "crawler.hardware.oakd",
                        types.SimpleNamespace(OakD=OneFrameOakD))
    hub = camera.CameraHub()
    hub.depth_listeners.append(lambda now, f, t: got.append((f, t)))

    def raises(now, f, t):
        raise RuntimeError("a listener must not take the reader down")

    hub.depth_listeners.insert(0, raises)
    hub.start()
    assert hub.wait_ready(5)
    for _ in range(300):
        if got:
            break
        threading.Event().wait(0.01)
    hub.close()
    assert len(got) == 1 and got[0][0] is frame and got[0][1] == 123.5


def test_oakd_keeps_each_packets_capture_time():
    pytest.importorskip("depthai", reason="depthai is not installed here")
    import datetime
    from crawler.hardware import oakd

    class Packet:
        def getTimestamp(self):
            return datetime.timedelta(seconds=42.25)

        def getFrame(self):
            return np.zeros((2, 2), np.uint16)

    class Queue:
        def tryGet(self):
            return Packet()

    dev = oakd.OakD.__new__(oakd.OakD)
    assert dev.capture_time("depth") is None
    dev._try_frame(Queue(), "depth")
    assert dev.capture_time("depth") == pytest.approx(42.25)


# ---- Foxglove -------------------------------------------------------------------------

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


def test_the_points_go_out_in_the_map_frame_as_they_are():
    pytest.importorskip("foxglove", reason="foxglove SDK not installed here")
    from crawler.telemetry import telemetry as tm
    pts = np.array([[1, 2, 0.1], [1.05, 2, 0.1]], np.float32)
    msg = decode(tm.Telemetry.lidar_view_message("voxel_points", pts, 10**18))
    assert msg.frame_id == "map" and msg.point_stride == 12
    np.testing.assert_array_equal(
        np.frombuffer(msg.data, np.float32).reshape(-1, 3), pts)


def test_the_low_grid_puts_an_obstacle_where_the_map_has_it():
    pytest.importorskip("foxglove", reason="foxglove SDK not installed here")
    from crawler.telemetry import telemetry as tm
    codes = np.zeros((5, 3), np.uint8)              # 5 along x, 3 along y
    codes[4, 1] = vx.LOW_OBSTACLE
    codes[0, 2] = vx.LOW_FLOOR
    # Something seen in every row and column, so the crop to what has been
    # seen (see test_live_bandwidth.py) leaves the window as it is.
    codes[1, 0] = vx.LOW_FLOOR
    view = vx.LowView(t=0.0, x0=-1.0, y0=2.0, res=0.05, codes=codes)
    g = decode(tm.Telemetry.lidar_view_message("voxel_low", view, 10**18))
    assert g.frame_id == "map" and g.column_count == 5
    assert (g.pose.position.x, g.pose.position.y) == (-1.0, 2.0)
    data = np.frombuffer(g.data, np.uint8).reshape(3, 5, 4)   # [row=y, col=x]
    assert data[1, 4].tolist() == [255, 140, 0, 255]
    assert data[2, 0, 3] > 0 and data[0, 0, 3] == 0, "unseen is clear"


def test_the_channels_are_registered_as_camera_channels():
    from crawler.telemetry import telemetry as tm
    for key in ("voxel_points", "voxel_low"):
        assert key in tm.ALL_KEYS and key in tm.CAMERA_KEYS
        assert key in tm.INTERVALS


# ---- the flag ----------------------------------------------------------------------------

def test_the_flag_is_off_unless_named():
    from crawler import __main__ as mn
    assert mn.parse_args([]).voxel_map is False
    assert mn.parse_args(["--voxel-map"]).voxel_map is True


def test_no_more_than_map_max_hz_frames_are_sent():
    clock = Clock()
    m = vx.VoxelMapper(FakeSlam(stamp(1.0)), clock=clock, mode=vx.MODE_THREAD)
    import queue
    m._inbox = queue.Queue(maxsize=8)
    assert not m.too_soon()
    m.step(1.0, render())
    clock.t = 0.9 / vx.MAP_MAX_HZ
    assert m.too_soon()
    clock.t = 1.1 / vx.MAP_MAX_HZ
    assert not m.too_soon()
