"""crawler/telemetry/lidar_view.py and its three topics: /tf, /lidar/scan, /lidar/truck.

The geometry tests read the messages back off the wire: the SDK's messages
only encode, so each one is decoded through the protobuf schema the SDK
itself publishes for it, and a point is placed the way the 3D panel places
it -- angle = start + i * (end - start) / (n - 1), then the transform.
"""

import json
import math
import os
import struct

import numpy as np
import pytest

from crawler.hardware import lidar as ld
from crawler.telemetry import lidar_view as lv
from crawler.telemetry import telemetry as tm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYOUT = os.path.join(ROOT, "deploy", "foxglove-drive.json")


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


def rotate(q, v):
    """Vector v by unit quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=np.float64)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def scan_of(angles, dists):
    return ld.Scan(t=0.0, angles_deg=np.asarray(angles, np.float32),
                   distances_mm=np.asarray(dists, np.uint16),
                   intensities=np.full(len(angles), 200, np.uint8),
                   speed_dps=3600.0, packets=42)


def points_in_base_link(scan, mount=ld.MOUNT):
    """What the 3D panel draws, in base_link, metres: decoded off the wire."""
    msg = decode(lv.laser_scan(scan, 0, mount))
    tf = decode(lv.frame_transforms(0, mount)).transforms[0]
    assert msg.frame_id == tf.child_frame_id == "lidar"
    assert tf.parent_frame_id == "base_link"
    q = (tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w)
    t = np.array([tf.translation.x, tf.translation.y, tf.translation.z])
    n = len(msg.ranges)
    inc = (msg.end_angle - msg.start_angle) / (n - 1)
    out = []
    for i, r in enumerate(msg.ranges):
        if not math.isfinite(r):
            continue
        a = msg.start_angle + inc * i
        out.append(rotate(q, [r * math.cos(a), r * math.sin(a), 0.0]) + t)
    return np.array(out)


# ---- the transform ------------------------------------------------------------

def test_the_transform_is_the_mount_yaw_and_nothing_else():
    tf = decode(lv.frame_transforms(1_500_000_000, ld.MOUNT)).transforms[0]
    assert (tf.parent_frame_id, tf.child_frame_id) == ("base_link", "lidar")
    h = math.sqrt(0.5)
    assert (tf.rotation.x, tf.rotation.y) == (0.0, 0.0)
    assert tf.rotation.z == pytest.approx(-h)
    assert tf.rotation.w == pytest.approx(h)
    assert (tf.translation.x, tf.translation.y, tf.translation.z) == (0, 0, 0)
    assert (tf.timestamp.seconds, tf.timestamp.nanos) == (1, 500_000_000)


@pytest.mark.parametrize("raw, where", [
    (270.0, (1.5, 0.0)),     # straight ahead, as measured off the bumper box
    (0.0, (0.0, -1.5)),      # the arrow: the truck's right
    (90.0, (-1.5, 0.0)),     # a quarter turn clockwise from the right: behind
    (180.0, (0.0, 1.5)),     # left
])
def test_a_raw_return_lands_where_lidar_py_says_in_base_link(raw, where):
    pts = points_in_base_link(scan_of([raw], [1500]))
    assert len(pts) == 1
    assert pts[0][:2] == pytest.approx(where, abs=1e-6)
    # And it agrees with lidar.py's own frame, which /lidar's bins use.
    x, y = ld.to_car([raw], [1500])
    assert pts[0][:2] == pytest.approx((x[0] / 1000, y[0] / 1000), abs=1e-6)


def test_the_mount_offset_is_in_the_transform():
    mount = ld.Mount(yaw_deg=-90.0, x_mm=100.0, y_mm=-40.0)
    pts = points_in_base_link(scan_of([270.0], [1000]), mount)
    assert pts[0][:2] == pytest.approx((1.1, -0.04), abs=1e-6)


def test_upside_down_turns_the_other_way_in_the_view_too():
    mount = ld.Mount(yaw_deg=-90.0, x_mm=0.0, y_mm=0.0, upside_down=True)
    pts = points_in_base_link(scan_of([90.0], [1000]), mount)
    x, y = ld.to_car([90.0], [1000], mount)
    assert pts[0][:2] == pytest.approx((x[0] / 1000, y[0] / 1000), abs=1e-6)
    assert pts[0][:2] == pytest.approx((1.0, 0.0), abs=1e-6)   # ahead


# ---- the scan's grid -------------------------------------------------------------

def a_real_turn(hz=9.9, seed=3):
    """One revolution laid down the way the sensor does: 12 points a packet,
    even inside one, packets end to end with the motor wobbling."""
    rng = np.random.default_rng(seed)
    spacing = 360.0 * hz / 5000.0
    angles, start = [], rng.uniform(0, 0.5)
    while start < 360.0 - 12 * spacing:
        step = spacing * rng.uniform(0.97, 1.03)
        angles += [start + step * i for i in range(12)]
        start += step * 12
    dists = rng.integers(200, 6000, len(angles))
    return np.array(angles), dists


def test_every_real_point_is_drawn_within_a_quarter_degree_of_its_own_angle():
    angles, dists = a_real_turn()
    pts = points_in_base_link(scan_of(angles, dists))
    assert len(pts) == len(angles) > 480
    want_x, want_y = ld.to_car(angles, dists)
    want = np.degrees(np.arctan2(want_y, want_x))
    got = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    order_w, order_g = np.argsort(want), np.argsort(got)
    err = np.abs((got[order_g] - want[order_w] + 180.0) % 360.0 - 180.0)
    assert err.max() <= lv.SCAN_STEP_DEG / 2 + 1e-4
    assert np.sort(np.hypot(pts[:, 0], pts[:, 1])) == pytest.approx(
        np.sort(dists) / 1000.0, abs=1e-4)


def test_no_return_is_not_drawn_and_two_in_one_slot_keep_the_nearer():
    ranges = lv.scan_slots([270.0, 270.1, 10.0, 20.0, 30.0],
                           [900, 600, 0, 20, 13000])
    assert len(ranges) == lv.SCAN_SLOTS == 720
    assert np.count_nonzero(np.isfinite(ranges)) == 1
    assert ranges[180] == pytest.approx(0.6)     # bearing 90 off the arrow


def test_the_scan_is_the_same_size_whatever_the_sensor_does():
    small = lv.laser_scan(scan_of([0.0], [500]), 0).encode()
    angles, dists = a_real_turn(hz=6.0)
    big = lv.laser_scan(scan_of(angles, dists), 0).encode()
    assert len(small) == len(big) < 6000


# ---- the truck ------------------------------------------------------------------

def test_the_truck_is_a_closed_outline_on_base_link_with_the_arrow_at_the_front():
    update = decode(lv.truck_scene(0))
    assert len(update.entities) == 1
    e = update.entities[0]
    assert e.frame_id == "base_link" and e.id == "truck"
    assert e.frame_locked
    assert e.lifetime.seconds == 0 and e.lifetime.nanos == 0   # forever
    (line,) = e.lines
    assert line.type == 1                                      # LINE_LOOP
    corners = sorted((round(p.x, 3), round(p.y, 3)) for p in line.points)
    # The measured rear bumper and width, the estimated front bumper.
    assert corners == [(-0.248, -0.136), (-0.248, 0.136),
                       (0.27, -0.136), (0.27, 0.136)]
    (tri,) = e.triangles
    tip, left, right = [(p.x, p.y) for p in tri.points]
    assert tip == pytest.approx((0.27, 0.0))
    assert left[0] == right[0] < tip[0], "it points forward, +x"
    assert left[1] == -right[1] > 0
    assert -0.248 < left[0]


def test_a_measured_width_replaces_the_guess(monkeypatch):
    monkeypatch.setattr(ld, "HALF_WIDTH_MM", 120.0)
    assert {round(y, 3) for _, y in lv.truck_outline_m()} == {0.12, -0.12}


def test_with_no_width_the_outline_falls_back_to_the_guess(monkeypatch):
    monkeypatch.setattr(ld, "HALF_WIDTH_MM", None)
    assert {round(y, 3) for _, y in lv.truck_outline_m()} == {0.15, -0.15}


# ---- the reader, the channels and the recording ---------------------------------

class AllTele:
    def __init__(self):
        self.rows = []

    def wants(self, key, now):
        return True

    def offer(self, key, payload, now):
        self.rows.append((key, payload))


def test_the_reader_hands_over_each_scan_with_its_transform_and_the_truck():
    tele = AllTele()
    reader = ld.LidarReader(tele=tele)
    scan = scan_of([270.0], [1000])
    reader._publish(scan, 1.0)
    keys = [k for k, _ in tele.rows]
    assert keys == ["lidar", "tf", "lidar_scan", "lidar_truck"]
    payload = dict(tele.rows)
    assert payload["tf"] is ld.MOUNT
    sent, mount = payload["lidar_scan"]
    assert mount is ld.MOUNT and sent.distances_mm[0] == 1000


def test_the_truck_goes_out_once_a_second_and_the_scan_every_time():
    tele = tm.Telemetry()
    tele.running = True
    for key in tele._watched:
        tele._watched[key] = True
    reader = ld.LidarReader(tele=tele)
    scan = scan_of([270.0], [1000])
    got = {"lidar_scan": 0, "tf": 0, "lidar_truck": 0}
    for i in range(25):
        reader._publish(scan, 10.0 + i * 0.1)
        for key in got:
            try:
                tele._mail[key].get_nowait()
                got[key] += 1
            except Exception:
                pass
    assert got == {"lidar_scan": 25, "tf": 25, "lidar_truck": 3}


def test_with_no_pad_none_of_it_is_written():
    tele = tm.Telemetry()
    tele.running = True
    for key in tele._watched:
        tele._watched[key] = True
    tele.mute(tm.CAMERA_KEYS)
    reader = ld.LidarReader(tele=tele)
    reader._publish(scan_of([270.0], [1000]), 1.0)
    for key in ("lidar", "lidar_scan", "tf", "lidar_truck"):
        assert tele._mail[key].empty(), key
    tele.unmute()
    reader._publish(scan_of([270.0], [1000]), 2.0)
    for key in ("lidar", "lidar_scan", "tf", "lidar_truck"):
        assert not tele._mail[key].empty(), key


def mcap_records(data):
    """(opcode, body) for each top-level record of an unchunked MCAP."""
    magic = b"\x89MCAP0\r\n"
    assert data[:8] == magic
    i, out = 8, []
    while i < len(data) - 8:
        op = data[i]
        (n,) = struct.unpack_from("<Q", data, i + 1)
        out.append((op, data[i + 9:i + 9 + n]))
        i += 9 + n
    return out


def _string(body, i):
    (n,) = struct.unpack_from("<I", body, i)
    return body[i + 4:i + 4 + n].decode(), i + 4 + n


def test_the_three_topics_are_recorded_in_their_foxglove_schemas(tmp_path):
    """The real SDK, a real MCAP file: what the 3D panel will be handed."""
    foxglove = pytest.importorskip("foxglove")
    from foxglove.mcap import MCAPWriteOptions

    tele = tm.Telemetry()
    assert tele.start(serve=False)
    path = str(tmp_path / "lidar.mcap")
    writer = foxglove.open_mcap(path, writer_options=MCAPWriteOptions(
        compression=None, use_chunks=False))
    try:
        tele._refresh_watched()
        reader = ld.LidarReader(tele=tele)
        angles, dists = a_real_turn()
        for i in range(3):
            reader._publish(scan_of(angles, dists), 100.0 + i * 0.1)
            tele._drain_mail(None)
    finally:
        writer.close()
        tele.stop()

    data = open(path, "rb").read()
    schemas, channels, counts = {}, {}, {}
    for op, body in mcap_records(data):
        if op == 0x03:
            (sid,) = struct.unpack_from("<H", body, 0)
            name, _ = _string(body, 2)
            schemas[sid] = name
        elif op == 0x04:
            cid, sid = struct.unpack_from("<HH", body, 0)
            topic, _ = _string(body, 4)
            channels[cid] = (topic, sid)
        elif op == 0x05:
            (cid,) = struct.unpack_from("<H", body, 0)
            counts[cid] = counts.get(cid, 0) + 1
    by_topic = {topic: (schemas.get(sid), counts.get(cid, 0))
                for cid, (topic, sid) in channels.items()}
    assert by_topic["/lidar/scan"] == ("foxglove.LaserScan", 3)
    assert by_topic["/tf"] == ("foxglove.FrameTransforms", 3)
    assert by_topic["/lidar/truck"] == ("foxglove.SceneUpdate", 1)
    assert by_topic["/lidar"][1] == 3


def test_the_keys_are_wired_through():
    assert tm.LIDAR_VIEW_CHANNELS == {"tf": "/tf", "lidar_scan": "/lidar/scan",
                                      "lidar_truck": "/lidar/truck"}
    for key in tm.LIDAR_VIEW_CHANNELS:
        assert key in tm.ALL_KEYS and key in tm.INTERVALS
        assert key in tm.CAMERA_KEYS


# ---- the layout --------------------------------------------------------------------

def panels(node):
    if isinstance(node, str):
        return [node]
    return panels(node["first"]) + panels(node["second"])


def ortho_screen_axes(camera):
    """Where the 3D panel's 2D camera puts screen up and screen right, in the
    display frame.

    The panel's code (CameraStateSettings.ts, #updateCameras, in the open
    Lichtblick fork of Studio; the same lines as Foxglove Studio's last open
    release): with perspective off the orthographic camera sits on +z
    looking down, and its quaternion is setFromAxisAngle(UNIT_Z, theta) with
    theta = -degToRad(thetaOffset). A three.js camera's up is its +y and its
    right is its +x, so those two rotated by theta about z are screen up and
    screen right.
    """
    theta = -math.radians(camera["thetaOffset"])
    up = (-math.sin(theta), math.cos(theta))
    right = (math.cos(theta), math.sin(theta))
    return up, right


def test_the_drive_layout_has_the_lidar_from_above_beside_the_map():
    layout = json.load(open(LAYOUT))
    names = panels(layout["layout"])
    assert set(names) == set(layout["configById"])
    right = layout["layout"]["second"]
    beside = right["second"]
    # The lidar may share its column with a small panel under it (the goto
    # status), but it is the top of it.
    if isinstance(beside, dict):
        assert beside["direction"] == "column"
        beside = beside["first"]
    assert (right["first"], beside) == ("3D!slam", "3D!lidar")

    cfg = layout["configById"]["3D!lidar"]
    assert cfg["followTf"] == "base_link"
    assert cfg["followMode"] == "follow-pose"
    assert cfg["topics"]["/lidar/scan"]["visible"] is True
    assert cfg["topics"]["/lidar/truck"]["visible"] is True
    grid = cfg["layers"]["grid"]
    assert grid["layerId"] == "foxglove.Grid" and grid["visible"] is True
    assert grid["size"] / grid["divisions"] == 0.5
    assert grid["frameId"] == "base_link"


def test_truck_forward_is_up_on_screen_and_its_left_is_left():
    cam = json.load(open(LAYOUT))["configById"]["3D!lidar"]["cameraState"]
    assert cam["perspective"] is False
    assert cam["phi"] == 0
    assert cam["targetOffset"] == [0, 0, 0]
    up, right = ortho_screen_axes(cam)
    assert up == pytest.approx((1.0, 0.0), abs=1e-9), "base_link +x is up"
    assert right == pytest.approx((0.0, -1.0), abs=1e-9), "+y, left, is left"
    # And the front of the truck outline is what is at the top.
    tip = lv.truck_arrow_m()[0]
    assert np.dot(tip, up) == max(np.dot(p, up) for p in lv.truck_outline_m())
