"""What the live view costs the wifi, and what a slow laptop costs the server.

The drive on 2026-10-29 lagged further and further behind in Foxglove and
then stopped, while the car drove on. Two things did it together. The map
messages grew with the map -- /voxel/points went from 60 to 344 KiB/s over
seven minutes as the voxel map filled up to its 40000 points, and
/voxel/low was a 625 KB RGBA picture every second whatever was in it -- so
with everything open it was over 1.1 MiB/s. And the SDK's server keeps 1024
messages per client before it drops one, so a link that cannot keep up does
not lose frames, it falls behind: a client held to 300 KiB/s against the car
was 24 s behind within half a minute and stayed there.

So: every map message has a byte budget that does not grow with the map,
and a client that cannot keep up loses the oldest messages instead of
queueing them.
"""

import base64
import json
import os
import socket
import struct
import time

import numpy as np
import pytest

from crawler.mapping import grid as gr
from crawler.mapping import runner as rn
from crawler.mapping import voxel as vx

pytest.importorskip("foxglove", reason="foxglove SDK not installed here")

from crawler.telemetry import map_view as mv  # noqa: E402
from crawler.telemetry import telemetry as tm  # noqa: E402
from crawler.telemetry import voxel_view as vv  # noqa: E402


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


def rgba_pixels(msg):
    """A Grid message's data as [row=y, col=x, 4]."""
    return np.frombuffer(bytes(msg.data), np.uint8).reshape(
        -1, msg.column_count, 4)


def world_of_opaque(msg, rgba):
    """Map metres of the centre of every cell drawn in colour `rgba`."""
    px = rgba_pixels(msg)
    rows, cols = np.nonzero((px == np.asarray(rgba, np.uint8)).all(-1))
    res = msg.cell_size.x
    return np.stack([msg.pose.position.x + (cols + 0.5) * res,
                     msg.pose.position.y + (rows + 0.5) * res], 1)


def explored_map(n, res=0.1, x0=-20.0, y0=-20.0):
    """A MapView n x n pixels, all of it seen: free with a wall every 2 m."""
    codes = np.full((n, n), gr.FREE, np.uint8)
    codes[::20, :] = gr.OCCUPIED
    return rn.MapView(t=0.0, version=1, x0=x0, y0=y0, res=res, codes=codes)


# ---- the map pictures ----------------------------------------------------------

@pytest.mark.parametrize("n", [50, 200, 400, 1000])
def test_the_map_message_stays_under_its_budget_however_big_the_map_gets(n):
    msg = decode(mv.grid_message(explored_map(n), 10**18))
    assert len(msg.data) <= 4 * mv.MAX_CELLS


def test_a_map_inside_the_budget_goes_out_as_it_is():
    view = explored_map(120)
    msg = decode(mv.grid_message(view, 10**18))
    assert msg.column_count == 120 and msg.cell_size.x == pytest.approx(0.1)
    assert bytes(msg.data) == mv.rgba(view.codes)
    assert (msg.pose.position.x, msg.pose.position.y) == (-20.0, -20.0)


def test_a_big_map_keeps_every_wall_where_it_was_just_in_coarser_pixels():
    view = explored_map(400)
    msg = decode(mv.grid_message(view, 10**18))
    assert msg.cell_size.x > view.res
    walls = world_of_opaque(msg, mv.COLOURS[gr.OCCUPIED])
    half = msg.cell_size.x / 2 + 1e-6
    # Every wall row of the input (x = x0 + 2 m k) is under a drawn wall.
    for k in range(20):
        x = view.x0 + 2.0 * k + view.res / 2
        assert np.any(np.abs(walls[:, 0] - x) <= half), x


def test_the_unknown_round_the_edge_is_not_sent():
    codes = np.zeros((400, 400), np.uint8)
    codes[100:150, 200:260] = gr.FREE
    codes[120, 230] = gr.OCCUPIED
    view = rn.MapView(t=0.0, version=1, x0=-20.0, y0=-20.0, res=0.1,
                      codes=codes)
    msg = decode(mv.grid_message(view, 10**18))
    assert msg.column_count == 50 and len(msg.data) == 50 * 60 * 4
    (x, y), = world_of_opaque(msg, mv.COLOURS[gr.OCCUPIED])
    assert (x, y) == pytest.approx((-20.0 + 12.05, -20.0 + 23.05))


@pytest.mark.parametrize("n", [100, 400, 1000])
def test_the_low_grid_stays_under_its_budget_however_much_has_been_seen(n):
    codes = np.full((n, n), vx.LOW_FLOOR, np.uint8)
    codes[n // 2, n // 3] = vx.LOW_OBSTACLE
    view = vx.LowView(t=0.0, x0=0.0, y0=0.0, res=0.05, codes=codes)
    msg = decode(vv.low_message(view, 10**18))
    assert len(msg.data) <= 4 * mv.MAX_CELLS
    obstacle = world_of_opaque(msg, vv.COLOURS[vx.LOW_OBSTACLE])
    assert len(obstacle) == 1
    want = ((n // 2 + 0.5) * 0.05, (n // 3 + 0.5) * 0.05)
    assert np.all(np.abs(obstacle[0] - want) <= msg.cell_size.x / 2 + 1e-6)


def test_the_point_cloud_is_capped_however_many_voxels_there_are():
    pts = np.random.default_rng(0).uniform(-10, 10, (vx.POINTS_MAX, 3))
    msg = decode(vv.points_message(pts.astype(np.float32), 10**18))
    out = np.frombuffer(bytes(msg.data), np.float32).reshape(-1, 3)
    assert len(out) <= vv.LIVE_POINTS_MAX
    # A thinning of the cloud, not the first patch of it.
    assert out[:, 0].min() < -9 and out[:, 0].max() > 9


def test_the_maps_together_fit_the_wifi_budget_at_their_worst():
    """Every map channel at the biggest message it can make, at its rate."""
    worst = {
        "map": mv.grid_message(explored_map(int(2 * rn.MAP_VIEW_HALF_M / 0.1)),
                               10**18),
        "voxel_low": vv.low_message(vx.LowView(
            t=0.0, x0=0.0, y0=0.0, res=vx.RES_M,
            codes=np.full((int(2 * vx.LOW_VIEW_HALF_M / vx.RES_M),) * 2,
                          vx.LOW_FLOOR, np.uint8)), 10**18),
        "voxel_points": vv.points_message(
            np.ones((vx.POINTS_MAX, 3), np.float32), 10**18),
        "slam_path": mv.path_scene([(0.1 * i, 0.0) for i in range(rn.PATH_MAX + 1)],
                                   10**18),
    }
    # /map's pictures are spaced by the runner, by their size
    # (runner.map_interval); the live gate under it only stops a burst.
    every = dict(tm.INTERVALS)
    every["map"] = max(every["map"], rn.map_interval(
        explored_map(int(2 * rn.MAP_VIEW_HALF_M / 0.1)).codes))
    rate = {key: len(msg.encode()) / every[key]
            for key, msg in worst.items()}
    assert sum(rate.values()) <= tm.MAP_CHANNELS_BUDGET, {
        k: round(v / 1024) for k, v in rate.items()}


# ---- a slow client -------------------------------------------------------------

def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Client:
    """Just enough of a Foxglove WebSocket client to subscribe and then
    not read, like a laptop on a link that cannot keep up."""

    def __init__(self, port):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024)
        s.settimeout(5.0)
        s.connect(("127.0.0.1", port))
        key = base64.b64encode(os.urandom(16)).decode()
        s.sendall((f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                   "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                   f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
                   "Sec-WebSocket-Protocol: foxglove.sdk.v1\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += s.recv(4096)
        head, self.buf = buf.split(b"\r\n\r\n", 1)
        assert b" 101 " in head.split(b"\r\n")[0], head
        self.s = s

    def _take(self, n):
        while len(self.buf) < n:
            data = self.s.recv(1 << 16)
            if not data:
                raise EOFError
            self.buf += data
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def recv(self):
        b0, b1 = self._take(2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", self._take(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", self._take(8))[0]
        return b0 & 0x0F, self._take(n)

    def send_json(self, obj):
        data = json.dumps(obj).encode()
        mask = os.urandom(4)
        n = len(data)
        head = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else
                                bytes([0x80 | 126]) + struct.pack(">H", n))
        self.s.sendall(head + mask
                       + bytes(c ^ mask[i % 4] for i, c in enumerate(data)))

    def subscribe(self, topic):
        while True:
            op, data = self.recv()
            if op != 1:
                continue
            msg = json.loads(data)
            if msg.get("op") != "advertise":
                continue
            for c in msg["channels"]:
                if c["topic"] == topic:
                    self.send_json({"op": "subscribe", "subscriptions": [
                        {"id": 1, "channelId": c["id"]}]})
                    return

    def drain(self, quiet_s=0.5):
        """Every data message waiting, as their "i"s."""
        self.s.settimeout(quiet_s)
        got = []
        try:
            while True:
                op, data = self.recv()
                if op == 2 and data[0] == 1:          # MessageData
                    got.append(json.loads(data[13:])["i"])
        except (socket.timeout, EOFError):
            pass
        return got


def test_a_client_that_cannot_keep_up_loses_the_oldest_not_falls_behind():
    import foxglove
    port = free_port()
    tele = tm.Telemetry(host="127.0.0.1", port=port)
    assert tele.start() and tele.serving
    flood = foxglove.Channel("/test/flood", message_encoding="json")
    try:
        client = Client(port)
        client.subscribe("/test/flood")
        deadline = time.monotonic() + 5.0
        while not flood.has_sinks() and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.2)

        # Far more than the backlog, each one far bigger than the socket
        # buffers, while the client reads nothing.
        sent = 4 * tm.LIVE_BACKLOG + 200
        pad = "x" * (256 * 1024)
        for i in range(sent):
            flood.log(json.dumps({"i": i, "pad": pad}).encode())
        time.sleep(0.5)

        got = client.drain()
        assert got and got[-1] == sent - 1, "the newest message got through"
        # What is still owed to it is the backlog and what the sockets hold,
        # not everything since it stopped reading.
        assert len(got) <= tm.LIVE_BACKLOG + 16, len(got)
    finally:
        flood.close()
        tele.stop()
