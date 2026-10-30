"""The roof lidar: the wire format, the scans, the car's frame, the reader.

No sensor anywhere: packets are built with lidar.encode_packet and fed
through a fake port. The one packet that is not built here is the LD19
manual's own worked example, which is what pins the CRC table to the real
thing.
"""

import os
import subprocess
import sys
import time

import numpy as np
import pytest

from crawler.hardware import lidar as ld
from crawler.telemetry import telemetry as tm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LD19 Development Manual V2.3, section 3.3, byte for byte.
MANUAL_EXAMPLE = bytes.fromhex(
    "54 2C 68 08 AB 7E E0 00 E4 DC 00 E2 D9 00 E5 D5 00 E3 D3 00 E4 D0 00 E9"
    " CD 00 E4 CA 00 E2 C7 00 E9 C5 00 E5 C2 00 E5 C0 00 E5 BE 82 3A 1A 50")

# 40 packets a turn: 12 points 0.75 degrees apart, packets 9 degrees apart.
STEP_CDEG = 75
PACKETS_PER_TURN = 40
SPEED = 3600   # deg/s, 10 Hz


def packet_at(k, dist_fn=lambda a: 2000, speed=SPEED):
    start = (k % PACKETS_PER_TURN) * 900
    end = start + STEP_CDEG * 11
    angles = [((start + STEP_CDEG * i) / 100.0) % 360 for i in range(12)]
    return ld.encode_packet(speed, start, end, [dist_fn(a) for a in angles])


def turn(dist_fn=lambda a: 2000, first=0, count=PACKETS_PER_TURN):
    return b"".join(packet_at(k, dist_fn) for k in range(first, first + count))


class FakeLog:
    def __init__(self):
        self.lines = []

    def log(self, level, subject, message):
        self.lines.append((level, subject, message))


class FakePort:
    """read() hands out the chunks in order; an exception in the list raises."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.closed = False

    def read(self, n):
        if not self.chunks:
            time.sleep(0.005)
            return b""
        item = self.chunks.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed = True


class FakeTele:
    def __init__(self):
        self.rows = []

    def wants(self, key, now):
        return key == "lidar"

    def offer(self, key, payload, now):
        self.rows.append((key, payload))


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


# ---- the wire ----------------------------------------------------------------

def test_the_crc_table_is_the_manuals():
    # The first row of CrcTable[256], LD19 manual 3.1 and lipkg.cpp.
    assert list(ld.CRC_TABLE[:12]) == [0x00, 0x4D, 0x9A, 0xD7, 0x79, 0x34,
                                        0xE3, 0xAE, 0xF2, 0xBF, 0x68, 0x25]
    assert ld.CRC_TABLE[255] == 0xA8
    assert ld.crc8(MANUAL_EXAMPLE[:-1]) == 0x50


def test_the_manuals_example_packet_decodes():
    p = ld.decode_packet(MANUAL_EXAMPLE)
    assert p is not None
    assert p.speed_dps == 2152
    assert p.start_cdeg == 32427 and p.end_cdeg == 33470
    assert p.distances_mm[0] == 224 and p.intensities[0] == 228
    assert len(p.distances_mm) == 12
    angles = ld.packet_angles(p)
    assert angles[0] == pytest.approx(324.27)
    assert angles[-1] == pytest.approx(334.70)


def test_a_built_packet_round_trips():
    raw = ld.encode_packet(3600, 1000, 1825, list(range(100, 112)),
                           timestamp_ms=1234)
    assert len(raw) == ld.PACKET_LEN
    p = ld.decode_packet(raw)
    assert p.distances_mm == tuple(range(100, 112))
    assert p.timestamp_ms == 1234


def test_a_bad_crc_is_counted_and_dropped():
    raw = bytearray(packet_at(3))
    raw[20] ^= 0x01
    assert ld.decode_packet(bytes(raw)) is None
    parser = ld.PacketParser()
    assert parser.feed(bytes(raw)) == []
    assert parser.crc_errors == 1
    assert parser.packets == 0


def test_garbage_between_packets_is_skipped():
    junk = bytes([0x00, 0x54, 0x11, 0x54, 0x2C, 0x99, 0xFF, 0x54])
    stream = junk + packet_at(1) + junk * 3 + packet_at(2) + junk
    parser = ld.PacketParser()
    got = parser.feed(stream)
    assert [p.start_cdeg for p in got] == [900, 1800]
    assert parser.dropped_bytes > 0


def test_a_good_packet_inside_a_broken_one_is_still_found():
    # A header and 10 bytes of a packet that never finished, then a whole
    # one. Read as 47 bytes the first fails its CRC; the parser has to look
    # again from the next byte, not jump 47 past it.
    stream = packet_at(1)[:12] + packet_at(2)
    parser = ld.PacketParser()
    got = parser.feed(stream)
    assert [p.start_cdeg for p in got] == [1800]
    assert parser.crc_errors == 1


def test_a_packet_split_across_reads_is_the_same_packet():
    stream = turn(first=0, count=5)
    whole = ld.PacketParser().feed(stream)
    parser = ld.PacketParser()
    pieces = []
    for i in range(len(stream)):
        pieces += parser.feed(stream[i:i + 1])
    assert pieces == whole
    assert len(pieces) == 5
    # Split exactly between 0x54 and 0x2C.
    parser = ld.PacketParser()
    raw = packet_at(7)
    assert parser.feed(b"\x01\x02" + raw[:1]) == []
    assert len(parser.feed(raw[1:])) == 1


def test_angles_wrap_through_zero():
    raw = ld.encode_packet(SPEED, 35500, 325, [500] * 12)
    p = ld.decode_packet(raw)
    angles = ld.packet_angles(p)
    assert angles[0] == pytest.approx(355.0)
    assert angles[-1] == pytest.approx(3.25)
    assert all(0.0 <= a < 360.0 for a in angles)
    steps = np.diff(np.unwrap(np.radians(angles)))
    assert np.allclose(np.degrees(steps), 0.75)


def test_a_packet_that_claims_a_quarter_turn_is_rejected():
    raw = ld.encode_packet(SPEED, 0, 9000, [500] * 12)
    parser = ld.PacketParser()
    assert parser.feed(raw) == []
    assert parser.rejected == 1


def test_endless_junk_does_not_grow_the_buffer():
    parser = ld.PacketParser()
    for _ in range(50):
        parser.feed(bytes([0x54, 0x2C]) + bytes(500))
        assert len(parser._buf) < ld.PACKET_LEN


# ---- scans -------------------------------------------------------------------

def test_scans_are_cut_at_zero_and_the_first_part_scan_is_dropped():
    asm = ld.ScanAssembler()
    parser = ld.PacketParser()
    # Half a turn, two whole turns, and the packet that closes the second.
    stream = (turn(first=20, count=20) + turn() + turn()
              + packet_at(0))
    scans = []
    for p in parser.feed(stream):
        s = asm.add(p, now=1.0)
        if s is not None:
            scans.append(s)
    assert len(scans) == 2
    for s in scans:
        assert len(s.distances_mm) == PACKETS_PER_TURN * 12
        assert s.angles_deg.min() == pytest.approx(0.0)
        assert s.angles_deg.max() < 360.0
        assert s.rpm == pytest.approx(600.0)
        assert s.packets == PACKETS_PER_TURN


def test_a_scan_that_never_wraps_is_thrown_away():
    asm = ld.ScanAssembler()
    p = ld.decode_packet(packet_at(5))
    for _ in range(ld.ScanAssembler.MAX_POINTS // 12 + 2):
        assert asm.add(p, now=0.0) is None
    assert len(asm._angles) <= ld.ScanAssembler.MAX_POINTS


# ---- the car's frame ---------------------------------------------------------

def test_the_arrow_is_the_cars_right():
    x, y = ld.to_car([0.0], [1000.0])
    assert x[0] == pytest.approx(0.0, abs=1e-6)
    assert y[0] == pytest.approx(-1000.0)


def test_clockwise_from_the_arrow_goes_right_back_left_front():
    bearings = ld.car_bearing_deg([0.0, 90.0, 180.0, 270.0])
    assert list(bearings) == pytest.approx([-90.0, 180.0, 90.0, 0.0])
    x, y = ld.to_car([270.0], [1000.0])
    assert (x[0], y[0]) == pytest.approx((1000.0, 0.0), abs=1e-6)


def test_raw_270_is_straight_ahead_and_raw_0_is_the_right_as_measured():
    # The box test on the car: front bumper box at raw 260-280, arrow right.
    x, y = ld.to_car([270.0, 0.0], [575.0, 575.0])
    assert (x[0], y[0]) == pytest.approx((575.0, 0.0), abs=1e-6)
    assert (x[1], y[1]) == pytest.approx((0.0, -575.0), abs=1e-6)
    bins = ld.bin_scan(np.arange(260.0, 280.5, 0.72), np.full(29, 575))
    assert ld.nearest_in_sector(bins, "front") == 575
    assert ld.nearest_in_sector(bins, "left") == 0
    assert ld.nearest_in_sector(bins, "right") == 0
    # The box was 305 mm off the bumper, so it put the bumper at 270; the tape
    # (lens 235 + 28) puts it at 263, inside the box's own 575-597 spread.
    assert 575 - 305 - ld.FRONT_BUMPER_X_MM == pytest.approx(0.0, abs=10.0)


def test_upside_down_turns_the_other_way():
    mount = ld.MOUNT._replace(upside_down=True)
    assert ld.car_bearing_deg([0.0, 90.0], mount) == pytest.approx([-90.0, 0.0])


def test_the_mount_offset_moves_every_point():
    mount = ld.Mount(yaw_deg=-90.0, x_mm=100.0, y_mm=20.0)
    x, y = ld.to_car([0.0], [500.0], mount)
    assert (x[0], y[0]) == pytest.approx((100.0, -480.0), abs=1e-6)


# ---- the bins ----------------------------------------------------------------

def test_a_return_at_the_arrow_lands_in_the_right_bin():
    bins = ld.bin_scan([0.0], [500])
    assert bins.dtype == np.uint16 and len(bins) == ld.BINS == 72
    assert bins[54] == 500
    assert np.count_nonzero(bins) == 1
    assert ld.nearest_in_sector(bins, "right") == 500
    assert ld.nearest_in_sector(bins, "front") == 0
    assert ld.nearest_in_sector(bins, "rear") == 0


def test_a_bin_keeps_its_nearest_return():
    bins = ld.bin_scan([270.0, 271.0, 269.0], [900, 400, 700])
    assert bins[0] == 400


def test_returns_out_of_range_are_no_return():
    bins = ld.bin_scan([270.0, 0.0, 90.0, 180.0], [0, 10, 13000, 12000])
    assert np.count_nonzero(bins) == 1
    assert bins[18] == 12000   # sensor 180 is the car's left


def test_straight_behind_is_one_bin_from_both_sides():
    # Car bearings +178 and -178 are both bin 36.
    bins = ld.bin_scan([92.0, 88.0], [800, 600])
    assert bins[36] == 600
    assert np.count_nonzero(bins) == 1


def test_bin_edges():
    # Bin 54 is car -90 +- 2.5; sensor 2.4 is car -92.4, sensor 2.6 is -92.6.
    assert ld.bin_scan([2.4], [500])[54] == 500
    assert ld.bin_scan([2.6], [500])[53] == 500


def test_a_corner_bin_counts_for_both_sides():
    bins = np.zeros(72, dtype=np.uint16)
    bins[9] = 300     # car bearing 45, front-left
    assert ld.nearest_in_sector(bins, "front") == 300
    assert ld.nearest_in_sector(bins, "left") == 300
    assert ld.nearest_in_sector(bins, "right") == 0
    assert ld.nearest_in_sector(bins, (45.0, 1.0)) == 300


# ---- the reader ----------------------------------------------------------------

def test_a_scan_goes_stale():
    clock = [10.0]
    reader = ld.LidarReader(clock=lambda: clock[0])
    assert reader.latest() is None and reader.stale()
    reader._scan = ld.Scan(t=10.0, angles_deg=np.zeros(1),
                           distances_mm=np.zeros(1, np.uint16),
                           intensities=np.zeros(1, np.uint8),
                           speed_dps=3600.0, packets=1)
    clock[0] = 10.0 + ld.STALE_S - 0.01
    assert reader.latest() is not None and not reader.stale()
    clock[0] = 10.0 + ld.STALE_S + 0.01
    assert reader.latest() is None and reader.stale()


def test_an_unplugged_lidar_does_not_hold_up_startup_and_says_so_once():
    log = FakeLog()
    tries = []

    def opener(path):
        tries.append(path)
        raise FileNotFoundError(f"no such file: {path}")

    reader = ld.LidarReader(device="/dev/ttyUSB0", logger=log, opener=opener,
                            retry_s=0.01)
    t0 = time.monotonic()
    reader.start()
    assert time.monotonic() - t0 < 0.1
    assert wait_for(lambda: len(tries) >= 5)
    reader.close()
    assert reader.latest() is None
    assert len(log.lines) == 1
    assert log.lines[0][0] == "warning" and "/dev/ttyUSB0" in log.lines[0][2]


def test_auto_with_no_port_says_so_once_and_keeps_looking():
    log = FakeLog()
    looks = []

    def ports():
        looks.append(1)
        return []

    reader = ld.LidarReader(device="auto", logger=log, ports=ports,
                            opener=lambda p: pytest.fail("nothing to open"),
                            retry_s=0.01)
    reader.start()
    assert wait_for(lambda: len(looks) >= 5)
    reader.close()
    assert len(log.lines) == 1 and "No lidar" in log.lines[0][2]


def test_the_reader_finds_the_arrow_on_the_right_and_records_it():
    near = lambda a: 300 if a < 10.0 or a > 350.0 else 3000   # at the arrow
    stream = [turn(near, first=10, count=30)] + [turn(near)] * 4
    port = FakePort(stream)
    tele = FakeTele()
    log = FakeLog()
    reader = ld.LidarReader(device="auto", logger=log, tele=tele,
                            ports=lambda: ["/dev/ttyUSB0"],
                            opener=lambda p: port)
    reader.start()
    try:
        assert wait_for(lambda: reader.scans >= 3)
        scan = reader.latest()
        assert scan is not None
        assert ld.nearest_in_sector(scan.bins, "right") == 300
        assert ld.nearest_in_sector(scan.bins, "front") == 3000
        assert ld.nearest_in_sector(scan.bins, "left") == 3000
        assert reader.port_path == "/dev/ttyUSB0"
    finally:
        reader.close()
    assert port.closed
    rows = [p for k, p in tele.rows if k == "lidar"]
    assert rows
    row = rows[-1]
    assert row["nearest_right_mm"] == 300
    assert row["rpm"] == pytest.approx(600.0)
    assert len(row["bins"]) == 72 and row["bin_deg"] == 5.0
    assert row["points"] == 480
    assert any("Lidar on /dev/ttyUSB0" in line[2] for line in log.lines)


def test_a_lidar_that_goes_away_is_retried_and_comes_back():
    ports = [FakePort([turn()] * 3 + [OSError("device disconnected")]),
             FakePort([turn()] * 3)]
    opened = []

    def opener(path):
        opened.append(path)
        return ports[len(opened) - 1]

    log = FakeLog()
    reader = ld.LidarReader(device="/dev/ttyUSB0", logger=log, opener=opener,
                            retry_s=0.01)
    reader.start()
    try:
        assert wait_for(lambda: len(opened) == 2 and reader.scans >= 3)
    finally:
        reader.close()
    text = [line[2] for line in log.lines]
    assert sum("Lost /dev/ttyUSB0" in t for t in text) == 1
    assert sum("Lidar on" in t for t in text) == 2


def test_a_port_with_no_lidar_on_it_lets_auto_try_the_next():
    silent = FakePort([])
    talking = FakePort([turn()] * 4)
    opened = []

    def opener(path):
        opened.append(path)
        return silent if path == "/dev/ttyACM0" else talking

    reader = ld.LidarReader(device="auto", opener=opener, logger=FakeLog(),
                            ports=lambda: ["/dev/ttyACM0", "/dev/ttyUSB0"],
                            retry_s=0.01, silent_s=0.05)
    reader.start()
    try:
        assert wait_for(lambda: reader.scans >= 2)
    finally:
        reader.close()
    assert opened[:2] == ["/dev/ttyACM0", "/dev/ttyUSB0"]


def test_auto_looks_at_usb_serial_and_never_the_gpio_uart():
    seen = []

    def globber(pattern):
        seen.append(pattern)
        return {"/dev/ttyUSB*": ["/dev/ttyUSB1", "/dev/ttyUSB0"],
                "/dev/ttyACM*": ["/dev/ttyACM0"]}.get(pattern, [])

    assert ld.candidate_ports(globber, resolve=lambda p: p) == [
        "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0"]
    assert not any("serial0" in p or "ttyAMA" in p or "ttyS" in p for p in seen)


def test_auto_tries_the_cp2102_by_its_by_id_name_first_and_each_tty_once():
    cp = ("/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge"
          "_Controller_0001-if00-port0")
    other = "/dev/serial/by-id/usb-Arduino_Uno_1234-if00"
    links = {cp: "/dev/ttyUSB0", other: "/dev/ttyACM0"}

    def globber(pattern):
        return {"/dev/serial/by-id/*": [other, cp],
                "/dev/ttyUSB*": ["/dev/ttyUSB0", "/dev/ttyUSB1"],
                "/dev/ttyACM*": ["/dev/ttyACM0"]}.get(pattern, [])

    got = ld.candidate_ports(globber, resolve=lambda p: links.get(p, p))
    assert got == [cp, other, "/dev/ttyUSB1"]


def test_a_read_that_says_try_again_is_not_a_lost_lidar():
    port = FakePort([BlockingIOError(11, "EAGAIN")] + [turn()] * 4)
    log = FakeLog()
    reader = ld.LidarReader(device="/dev/ttyUSB0", logger=log,
                            opener=lambda p: port, retry_s=5.0)
    reader.start()
    try:
        assert wait_for(lambda: reader.scans >= 2)
    finally:
        reader.close()
    assert not any("Lost" in line[2] for line in log.lines)


# ---- telemetry and the flag ------------------------------------------------------

def test_lidar_is_a_recorded_json_topic():
    assert tm.JSON_TOPICS["lidar"] == "/lidar"
    assert "lidar" in tm.ALL_KEYS
    assert "lidar" in tm.INTERVALS


def test_a_lidar_row_is_exactly_its_schema():
    scan = ld.Scan(t=0.0, angles_deg=np.array([0.0, 270.0]),
                   distances_mm=np.array([500, 800], np.uint16),
                   intensities=np.array([200, 200], np.uint8),
                   speed_dps=3600.0, packets=40)
    row = ld.lidar_sample(scan, {"packets_per_s": 417.0, "crc_errors": 2,
                                 "crc_error_rate": 0.001,
                                 "rejected_packets": 0})
    props = tm.JSON_SCHEMAS["lidar"]["schema"]["properties"]
    assert set(row) == set(props)
    assert row["nearest_right_mm"] == 500 and row["nearest_front_mm"] == 800
    import json
    assert len(json.dumps(row)) < 1000


def test_the_lidar_flag_is_off_unless_given():
    import crawler.__main__ as car
    assert car.parse_args([]).lidar is None
    assert car.parse_args(["--lidar"]).lidar == "auto"
    assert car.parse_args(["--lidar", "--mode", "auto"]).lidar == "auto"
    assert car.parse_args(["--lidar", "/dev/serial0"]).lidar == "/dev/serial0"


# ---- the bring-up tool -------------------------------------------------------------

def test_the_check_tool_cannot_reach_the_hardware():
    code = ("import sys, runpy; sys.argv=['x', '--help'];\n"
            "try:\n"
            "    runpy.run_path('tools/lidar_check.py', run_name='__main__')\n"
            "except SystemExit:\n"
            "    pass\n"
            "bad = [m for m in ('pigpio', 'pygame', 'board', 'busio', 'depthai',"
            " 'crawler.hardware.vehicle_controller') if m in sys.modules]\n"
            "print('BAD', bad)\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "BAD []" in out.stdout


def test_the_check_tool_reports_the_hand_at_the_right(capsys):
    import lidar_check
    near = lambda a: 250 if a < 2.0 or a > 358.0 else 2500

    class Endless(FakePort):
        def read(self, n):
            time.sleep(0.01)
            return turn(near)

    rc = lidar_check.main(["--seconds", "0.6"], opener=lambda p: Endless([]),
                          ports=lambda: ["/dev/ttyUSB0"])
    out = capsys.readouterr().out
    assert rc == 0
    lines = {l.split()[0]: l for l in out.splitlines()
             if l.strip().split(" ")[0] in ("front", "left", "rear", "right")}
    assert "250 mm" in lines["right"] and "-90 deg" in lines["right"]
    assert "2500 mm" in lines["front"]
    assert "CRC errors:      0" in out


def test_the_check_tool_with_nothing_plugged_in(capsys):
    import lidar_check
    rc = lidar_check.main(["--seconds", "0.2"], ports=lambda: [],
                          opener=lambda p: pytest.fail("nothing to open"))
    assert rc == 1
    assert "no port opened" in capsys.readouterr().out
