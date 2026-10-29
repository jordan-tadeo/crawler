"""The 2D lidar on the roof: an LDRobot STL-19P (the D500 kit), read off a UART.

Off unless `--lidar` is given. The one thing in the control loop that reads
it is the autopilot's rear stop (driver.py, "The rear stop"), via latest().
What this file does is turn the sensor's byte stream into a full 360 degree
scan in the car's own frame, on a thread of its own, and keep the newest one
where anything can pick it up without waiting.

The sensor
----------
Sources, all checked against each other:

  STL-19P datasheet v1.0 (2023-09-28), LDROBOT
  LD19 Development Manual V2.3, LDROBOT -- the STL-19P datasheet sends you to
      "the development manual" for the protocol, and the D500 kit is the
      LD19's successor on the same wire format
  ldlidar_stl_sdk, ldlidar_driver/src/dataprocess/lipkg.cpp, LDRobot's own
      parser

It only talks. The connector is 1 Tx, 2 PWM, 3 GND, 4 5V: there is no Rx pin,
so there is nothing to send and this file never writes a byte. It starts
streaming at 230400 8N1, no flow control, once its motor is up to speed, at
10 Hz and 5000 points a second by default. With PWM tied to ground (the kit's
board does that) it runs its own speed loop and needs nothing from me.

Every packet is 47 bytes, little-endian:

  0x54  0x2C  speed u16 (deg/s)  start u16 (0.01 deg)
  12 x (distance u16 mm, intensity u8)
  end u16 (0.01 deg)  timestamp u16 (ms, wraps at 30000)  crc8

The 12 points are spread evenly from start to end, and a packet can straddle
0: start 355.00, end 3.25 is an 8.25 degree packet, not a -351.75 one. A
distance of 0 is no return.

Which way the angles go
-----------------------
Datasheet section 5.5: the sensor's front is its X axis and its 0 degree
direction, and the angle "increases along the clockwise direction". The
figure there is the puck seen from above, so clockwise from above. The LD19
manual (section 4) says the same in the same words.

The front is the little arrow on the lid: datasheet section 5.2's drawing
points a leader at that mark and labels it "This forward". So the arrow is 0
degrees, and the angle grows clockwise looking down on it.

The car's frame
---------------
Nothing in this repo had a 2D frame before this, so it gets the usual one: x
forward, y left, origin at the lidar (on the cab roof, on the centre line; not
necessarily halfway from the front of the truck to the back), bearings counter-clockwise
from straight ahead seen from above (+90 is left, -90 right, 180 behind).
That is the right-handed frame everything else in robotics uses, and it is
the opposite handedness to the sensor's, which is the one thing to keep in
mind below. (The steering sign is -1 left / +1 right; that is a command, not
a frame, and nothing here touches it.)

See MOUNT for how one turns into the other.
"""

import glob
import os
import struct
import threading
import time
from collections import deque
from typing import NamedTuple, Optional

import numpy as np

# ---- the wire ---------------------------------------------------------------

BAUD = 230400
HEADER = 0x54
# The top three bits are the packet type (1), the low five the point count
# (12). Both fixed, so this byte is a second sync byte for free.
VER_LEN = 0x2C
POINTS_PER_PACKET = 12
PACKET_LEN = 47
_FORMAT = "<BBHH" + "HB" * POINTS_PER_PACKET + "HHB"
assert struct.calcsize(_FORMAT) == PACKET_LEN

# Datasheet 4.1: 0.03 to 12 m on a white target. Anything outside it is not a
# return I trust, and 0 is the sensor saying it had none.
MIN_RANGE_MM = 30
MAX_RANGE_MM = 12000

# One packet's span. At the datasheet's top speed, 13 Hz, 12 points at 5000
# a second cover 11.2 degrees; the manual's own worked example is 10.43 at
# 6 Hz. A packet that passed its CRC and says it covers more than this is a
# false header in noise that happened to checksum (1 in 256), not a packet.
#
# Looser than LDRobot's SDK on purpose: it rejects anything wider than 1.5x
# speed * 12 / 5000, and by that rule the manual's own example packet is
# rejected (10.43 deg against a limit of 7.7 at 2152 deg/s).
MAX_PACKET_SPAN_DEG = 30.0


def _crc_table(poly=0x4D):
    """CRC-8, polynomial 0x4D, init 0, no reflection, no final xor.

    Generated rather than pasted: it is byte for byte the CrcTable[256] in
    the manual and in lipkg.cpp, and tests/test_lidar.py holds it to the
    manual's worked example packet, whose last byte is 0x50.
    """
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = ((c << 1) ^ poly) & 0xFF if c & 0x80 else (c << 1) & 0xFF
        table.append(c)
    return bytes(table)


CRC_TABLE = _crc_table()


def crc8(data) -> int:
    crc = 0
    table = CRC_TABLE
    for byte in data:
        crc = table[crc ^ byte]
    return crc


class Packet(NamedTuple):
    speed_dps: int
    start_cdeg: int
    end_cdeg: int
    timestamp_ms: int
    distances_mm: tuple
    intensities: tuple


def decode_packet(frame) -> Optional[Packet]:
    """47 bytes -> Packet, or None if the header, length or CRC is wrong."""
    if len(frame) != PACKET_LEN or frame[0] != HEADER or frame[1] != VER_LEN:
        return None
    if crc8(frame[:-1]) != frame[-1]:
        return None
    return _unpack(frame)


def _unpack(frame) -> Packet:
    f = struct.unpack(_FORMAT, frame)
    points = f[4:4 + 2 * POINTS_PER_PACKET]
    return Packet(speed_dps=f[2], start_cdeg=f[3],
                  end_cdeg=f[4 + 2 * POINTS_PER_PACKET],
                  timestamp_ms=f[5 + 2 * POINTS_PER_PACKET],
                  distances_mm=tuple(points[0::2]),
                  intensities=tuple(points[1::2]))


def encode_packet(speed_dps, start_cdeg, end_cdeg, distances_mm,
                  intensities=None, timestamp_ms=0) -> bytes:
    """A Packet's 47 bytes with a correct CRC. For tests and fake streams."""
    if intensities is None:
        intensities = [200] * POINTS_PER_PACKET
    fields = [HEADER, VER_LEN, speed_dps, start_cdeg % 36000]
    for d, i in zip(distances_mm, intensities):
        fields += [int(d), int(i)]
    fields += [end_cdeg % 36000, timestamp_ms % 30000]
    body = struct.pack(_FORMAT[:-1], *fields)
    return body + bytes([crc8(body)])


def packet_span_cdeg(packet) -> int:
    return (packet.end_cdeg - packet.start_cdeg) % 36000


def packet_angles(packet) -> list:
    """The 12 points' sensor-frame angles in degrees, [0, 360).

    Linear from start to end (manual 3.2), across the wrap if the packet
    straddles 0, the same way lipkg.cpp does it.
    """
    step = packet_span_cdeg(packet) / (POINTS_PER_PACKET - 1)
    start = packet.start_cdeg
    return [((start + step * i) / 100.0) % 360.0
            for i in range(POINTS_PER_PACKET)]


class PacketParser:
    """Bytes in any size of chunk -> whole, CRC-checked packets.

    Resyncs on its own. Anything before a 0x54 0x2C pair is thrown away and
    counted. A packet whose CRC fails costs one byte, not 47: the search
    starts again at the byte after its header, so a real packet that begins
    inside a corrupt one is still found. lipkg.cpp throws the whole 47 away
    and can lose the packet after a bad one that way.
    """

    def __init__(self):
        self._buf = bytearray()
        self.packets = 0
        self.crc_errors = 0
        self.rejected = 0
        self.dropped_bytes = 0

    def feed(self, data) -> list:
        buf = self._buf
        buf += data
        out = []
        i = 0
        sync = bytes((HEADER, VER_LEN))
        while True:
            j = buf.find(sync, i)
            if j < 0:
                # No header in what is left. Keep a trailing 0x54: its 0x2C
                # may be the next chunk's first byte.
                end = len(buf) - (1 if buf[-1:] == b"\x54" else 0)
                end = max(end, i)
                self.dropped_bytes += end - i
                i = end
                break
            self.dropped_bytes += j - i
            if len(buf) - j < PACKET_LEN:
                i = j
                break
            frame = bytes(buf[j:j + PACKET_LEN])
            if crc8(frame[:-1]) != frame[-1]:
                self.crc_errors += 1
                i = j + 1
                continue
            packet = _unpack(frame)
            i = j + PACKET_LEN
            if packet_span_cdeg(packet) > MAX_PACKET_SPAN_DEG * 100:
                self.rejected += 1
                continue
            self.packets += 1
            out.append(packet)
        # What is left is under one packet: the start of one still arriving,
        # or a lone 0x54. Junk never piles up in here.
        del buf[:i]
        return out


# ---- scans ------------------------------------------------------------------

class Scan(NamedTuple):
    """One revolution, in the sensor's own frame, plus its car-frame bins.

    `t` is time.monotonic() when the packet that closed it arrived, which is
    what staleness is measured against.
    """
    t: float
    angles_deg: np.ndarray       # sensor frame, clockwise from the arrow
    distances_mm: np.ndarray     # uint16, 0 = no return
    intensities: np.ndarray      # uint8
    speed_dps: float             # mean of its packets' speed field
    packets: int
    bins: Optional[np.ndarray] = None   # bin_scan() of it, filled by the reader
    # The sensor's millisecond stamp (u16, wraps at 30000) of the packet that
    # closed the revolution, and of every packet with a point in it, in
    # order. `t` lags the real end by up to one serial read (~52 ms at
    # READ_CHUNK); these do not. crawler/mapping/clock.py turns them into a
    # de-jittered host time.
    stamp_ms: Optional[int] = None
    stamps_ms: tuple = ()

    @property
    def rpm(self) -> float:
        return self.speed_dps / 6.0

    @property
    def valid_points(self) -> int:
        d = self.distances_mm
        return int(np.count_nonzero((d >= MIN_RANGE_MM) & (d <= MAX_RANGE_MM)))


class ScanAssembler:
    """Packets -> whole revolutions, cut where the angle passes 0.

    A revolution ends at the first point whose angle is more than half a
    turn below the one before it. The first one after a start is thrown away,
    because it started wherever the sensor happened to be and is a part
    scan; lipkg.cpp drops its first frame for the same reason.
    """

    # Two revolutions at the slowest speed the datasheet allows (6 Hz, 833
    # points a turn). Past it the angle has stopped wrapping and what is
    # held is not a scan of anything.
    MAX_POINTS = 1700

    def __init__(self):
        self._reset()
        self._first = True

    def _reset(self):
        self._angles = []
        self._dist = []
        self._inten = []
        self._speeds = []
        self._stamps = []
        self._last = None

    def add(self, packet, now) -> Optional[Scan]:
        done = None
        angles = packet_angles(packet)
        for a, d, inten in zip(angles, packet.distances_mm, packet.intensities):
            if self._last is not None and a < self._last - 180.0:
                done = self._finish(now, packet.timestamp_ms)
            self._angles.append(a)
            self._dist.append(d)
            self._inten.append(inten)
            self._last = a
        # The packet that closes a scan counts towards the next one's speed:
        # most of its points are in that one.
        self._speeds.append(packet.speed_dps)
        self._stamps.append(packet.timestamp_ms)
        if len(self._angles) > self.MAX_POINTS:
            self._reset()
            self._first = True
        return done

    def _finish(self, now, stamp_ms=None) -> Optional[Scan]:
        scan = None
        if not self._first and self._angles:
            speeds = self._speeds or [0]
            scan = Scan(t=now,
                        angles_deg=np.asarray(self._angles, dtype=np.float32),
                        distances_mm=np.asarray(self._dist, dtype=np.uint16),
                        intensities=np.asarray(self._inten, dtype=np.uint8),
                        speed_dps=float(sum(speeds)) / len(speeds),
                        packets=len(self._speeds),
                        stamp_ms=stamp_ms,
                        stamps_ms=tuple(self._stamps) + (
                            () if stamp_ms is None else (stamp_ms,)))
        self._first = False
        self._reset()
        return scan


# ---- the mount and the car's frame -------------------------------------------

class Mount(NamedTuple):
    """Where the sensor sits on the car, and which way it faces.

    yaw_deg      the car bearing (counter-clockwise from straight ahead, +90
                 left) the sensor's arrow points along
    x_mm, y_mm   the sensor's spin axis from the middle of the car, x forward,
                 y left
    upside_down  lid facing the ground. Its clockwise is then the car's
                 counter-clockwise, so the angle flips sign. Not how it is
                 mounted; here because it costs one line.
    """
    yaw_deg: float
    x_mm: float
    y_mm: float
    upside_down: bool = False


# The arrow on the lid points at the truck's right.
#
#   Right is bearing -90 in the car's frame, and the arrow is the sensor's 0
#   (datasheet 5.2, "This forward"), so the sensor's 0 lands at car -90:
#   yaw_deg = -90.
#
#   The sensor counts clockwise seen from above (datasheet 5.5, manual 4) and
#   the car counts counter-clockwise, so a point the sensor calls phi is at
#   car bearing  yaw - phi = -90 - phi.
#
#   Checks: phi = 0 -> -90, right, where the arrow is. phi = 90 is a quarter
#   turn clockwise from the right, which looking down is the back: -180.
#   phi = 270 -> -360 = 0, straight ahead. phi = 180 -> -270 = +90, left.
#
# Measured on the car, not only derived: a box 305 mm off the front bumper
# showed up at 575-597 mm in raw angles 260-280, and only there; with it
# gone those went to 1.4-4.7 m and nothing else moved by more than 10 mm.
# So raw 270 is straight ahead, which is what yaw -90 clockwise says.
#
# Where: on the roof of the cab, on the centre line. That is the car frame's
# origin by definition, so the offset is 0, 0. The cab is not necessarily
# halfway along: the front and rear bumpers below are each their own number. Height is the cab top, unmeasured, and does not enter a
# 2D scan.
MOUNT = Mount(yaw_deg=-90.0, x_mm=0.0, y_mm=0.0, upside_down=False)

# The front bumper's x, mm, ESTIMATED off the same box: 575 seen less 305
# from the bumper. Not measured yet.
# The rear bumper's x, mm, MEASURED on the car: centre of the lidar to the
# rear bumper, 9.75 in = 247.65 mm. So the lidar is about 22 mm nearer the
# back than the front, going by the front estimate.
# A clearance stop is a range from the lidar less one of these.
FRONT_BUMPER_X_MM = 270.0
REAR_BUMPER_X_MM = -247.65
# Half the truck's width, mm, MEASURED on the car at the widest point, the
# outside of the wheels: 10.7 in = 271.78 mm, half of that. Only the
# visualiser's outline reads it; nothing steers or stops on side clearance.
HALF_WIDTH_MM = 135.89


def car_bearing_deg(angles_deg, mount=MOUNT):
    """Sensor angle(s) -> car bearing about the sensor, degrees, (-180, 180]."""
    sign = 1.0 if mount.upside_down else -1.0
    b = mount.yaw_deg + sign * np.asarray(angles_deg, dtype=np.float64)
    return 180.0 - np.mod(180.0 - b, 360.0)


def to_car(angles_deg, distances_mm, mount=MOUNT):
    """Sensor polar -> car-frame (x_mm, y_mm), x forward, y left."""
    b = np.radians(car_bearing_deg(angles_deg, mount))
    d = np.asarray(distances_mm, dtype=np.float64)
    return mount.x_mm + d * np.cos(b), mount.y_mm + d * np.sin(b)


# ---- the compact scan ---------------------------------------------------------

# 72 bins of 5 degrees, each the nearest return in it in mm, 0 for none.
#
# Why 5 and not 2: at 10 Hz the sensor lays a point every 0.72 degrees, so a
# 5 degree bin holds about 7 of them and one dropout does not empty it; a 2
# degree bin holds 2 or 3 and every missed return is a hole. 5 degrees is 87
# mm wide at 1 m and 175 mm at 2 m, narrower than the car at the distances a
# clearance stop is about, so a chair leg still shows. 72 numbers is about 300
# bytes of JSON a scan, 3 KB/s at 10 Hz, and a vector small enough to hand a
# model as it is. The minimum and not the mean, because the near thing is the
# one that matters and a mean of a wall and the gap beside it is neither.
#
# Bin i is centred on car bearing i * 5, counter-clockwise: bin 0 dead ahead,
# 18 left, 36 behind, 54 right. Distances are from the middle of the car, so
# MOUNT's offset is already in them.
BINS = 72
BIN_DEG = 360.0 / BINS


def bin_bearings(bins=BINS) -> np.ndarray:
    """Each bin's centre bearing, (-180, 180]."""
    b = np.arange(bins) * (360.0 / bins)
    return 180.0 - np.mod(180.0 - b, 360.0)


def bin_scan(angles_deg, distances_mm, mount=MOUNT, bins=BINS) -> np.ndarray:
    """Sensor polar -> (bins,) uint16, nearest car-frame range per bin, 0 none."""
    d = np.asarray(distances_mm)
    a = np.asarray(angles_deg)
    ok = (d >= MIN_RANGE_MM) & (d <= MAX_RANGE_MM)
    out = np.zeros(bins, dtype=np.uint16)
    if not ok.any():
        return out
    x, y = to_car(a[ok], d[ok], mount)
    r = np.minimum(np.hypot(x, y), 65535.0)
    width = 360.0 / bins
    bearing = np.degrees(np.arctan2(y, x))
    idx = np.floor((bearing + width / 2.0) / width).astype(np.intp) % bins
    nearest = np.full(bins, np.inf)
    np.minimum.at(nearest, idx, r)
    seen = np.isfinite(nearest)
    # Never 0 for a real return: 0 means none.
    out[seen] = np.maximum(1, np.round(nearest[seen])).astype(np.uint16)
    return out


# Each sector is (centre bearing, half width), degrees. Quadrants, with the
# bin on each 45 degree line counted in both of its neighbours: for a
# clearance stop a thing on the corner belongs to both sides.
SECTORS = {
    "front": (0.0, 45.0),
    "left": (90.0, 45.0),
    "rear": (180.0, 45.0),
    "right": (-90.0, 45.0),
}


def sector_mask(sector, bins=BINS) -> np.ndarray:
    centre, half = SECTORS[sector] if isinstance(sector, str) else sector
    diff = bin_bearings(bins) - centre
    diff = 180.0 - np.mod(180.0 - diff, 360.0)
    return np.abs(diff) <= half + 1e-6


def nearest_in_sector(bins, sector) -> int:
    """The nearest return in a sector of a binned scan, mm from the middle of
    the car, 0 if the sector saw nothing.

    `sector` is a SECTORS name or a (centre, half width) pair in degrees.
    Works on a live scan's bins and on a recorded /lidar row's alike. It is
    range from the middle of the car, not clearance from the bodywork: a rear
    stop subtracts the distance from the middle to the back bumper itself.
    """
    b = np.asarray(bins)
    picked = b[sector_mask(sector, len(b))]
    picked = picked[picked > 0]
    return int(picked.min()) if picked.size else 0


# ---- the reader ---------------------------------------------------------------

# A scan older than this is no scan. Three revolutions at 10 Hz: one late
# revolution is jitter, three is a sensor that has stopped.
STALE_S = 0.3

# How long to wait between tries at a port that is missing or went away.
RETRY_S = 2.0

# read() blocks this long at most, so close() is never more than this away.
READ_TIMEOUT_S = 0.05
READ_CHUNK = 1024

# An open port with no packet for this long is the wrong port, or a sensor
# that is not spinning. Said once, and in auto the next candidate gets a go.
SILENT_S = 3.0

# The USB-UART boards. The D500 kit's is a CP2102 ("CP2102 USB to UART
# Bridge Controller", cp210x driver): seen on the car as /dev/ttyUSB0, with a
# /dev/serial/by-id link, owned root:dialout. by-id names come first because
# they name the chip and do not renumber, and a CP2102 goes before anything
# else there; the bare ttyUSB/ttyACM nodes are the fallback for a board udev
# gave no by-id link. The Pi's own GPIO UART, /dev/serial0, is never guessed:
# on a stock Pi it is the login console. Name it with --lidar /dev/serial0 if
# the sensor goes on the header pins.
AUTO_GLOBS = ("/dev/serial/by-id/*", "/dev/ttyUSB*", "/dev/ttyACM*")
PREFERRED = "CP2102"


def candidate_ports(globber=glob.glob, resolve=os.path.realpath) -> list:
    """Ports auto tries, in order, each device once."""
    ports, seen = [], set()
    for pattern in AUTO_GLOBS:
        found = sorted(globber(pattern))
        found.sort(key=lambda p: PREFERRED not in p)   # stable: CP2102 first
        for path in found:
            real = resolve(path)
            if real in seen:
                continue
            seen.add(real)
            ports.append(path)
    return ports


def _open_serial(path):
    import serial  # pyserial; imported here so a missing one costs the lidar only
    # exclusive: a second reader on the same tty would split the byte stream
    # between them and both would see junk. This way the second one fails.
    return serial.Serial(path, BAUD, timeout=READ_TIMEOUT_S, exclusive=True)


class LidarReader:
    """The sensor on a thread of its own, and the newest scan for anyone.

    start() returns at once whether or not anything is plugged in. The thread
    looks for the port, reads, and if the port is missing or goes away it
    says so once and tries again every RETRY_S, for as long as the process
    runs. Nothing here ever blocks a caller: latest() is one attribute read.

    Telemetry, if given one, gets a /lidar row per scan from this thread,
    gated by wants() like the camera's channels, and the same scan for the
    3D panel: /lidar/scan, /tf and /lidar/truck (telemetry/lidar_view.py).
    Only the handing over happens here; the messages are built on the
    telemetry thread.

    `slam`, with --slam, is a crawler.mapping.runner.SlamRunner: every scan
    goes to its on_scan(), which never blocks, and while it is producing
    poses it sends /tf itself (map -> base_link as well as base_link ->
    lidar), so this thread leaves /tf to it.
    """

    def __init__(self, device="auto", logger=None, tele=None, mount=MOUNT,
                 opener=None, ports=None, clock=time.monotonic,
                 retry_s=RETRY_S, silent_s=SILENT_S, slam=None):
        self.device = device or "auto"
        self.logger = logger
        self.tele = tele
        self.mount = mount
        self.slam = slam
        self._opener = opener or _open_serial
        self._ports = ports or candidate_ports
        self._clock = clock
        self.retry_s = retry_s
        self.silent_s = silent_s

        self.parser = PacketParser()
        self._assembler = ScanAssembler()
        self._scan = None
        self.scans = 0
        self.bytes_read = 0
        self.port_path = None
        self.connected = False
        self.error = None

        self._next_candidate = 0
        self._said = set()
        self._window = deque()
        self._window_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    # ---- anyone's half ---------------------------------------------------

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="lidar",
                                        daemon=True)
        self._thread.start()

    def close(self):
        """Stop the thread and let go of the port. Never raises."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)

    def latest(self, now=None, max_age=STALE_S) -> Optional[Scan]:
        """The newest whole scan, or None if there is none newer than max_age."""
        scan = self._scan
        if scan is None:
            return None
        if now is None:
            now = self._clock()
        if now - scan.t > max_age:
            return None
        return scan

    def stale(self, now=None, max_age=STALE_S) -> bool:
        return self.latest(now, max_age) is None

    def stats(self, now=None) -> dict:
        """Packet rates over the last second and the running totals."""
        if now is None:
            now = self._clock()
        p = self.parser
        packets, crc = p.packets, p.crc_errors
        # The reader thread and a bring-up tool can both ask. Held for a
        # deque append, never across I/O, and the control loop never asks.
        with self._window_lock:
            window = self._window
            window.append((now, packets, crc))
            while len(window) > 2 and now - window[1][0] >= 1.0:
                window.popleft()
            t0, p0, c0 = window[0]
        dt = now - t0
        dp, dc = packets - p0, crc - c0
        scan = self._scan
        return {
            "device": self.port_path or "",
            "connected": self.connected,
            "rpm": round(scan.rpm, 1) if scan is not None else 0.0,
            "packets_per_s": round(dp / dt, 1) if dt > 0 else 0.0,
            "crc_error_rate": round(dc / (dp + dc), 4) if dp + dc else 0.0,
            "packets": packets,
            "crc_errors": crc,
            "rejected_packets": p.rejected,
            "dropped_bytes": p.dropped_bytes,
            "scans": self.scans,
        }

    # ---- the thread ------------------------------------------------------

    def _run(self):
        while not self._stop.is_set():
            port = self._open()
            if port is None:
                self._stop.wait(self.retry_s)
                continue
            try:
                self._pump(port)
            except Exception as e:
                self.error = str(e)
                self._say("lost", "warning",
                          f"Lost {self.port_path}: {e}. Retrying every "
                          f"{self.retry_s:g} s")
            finally:
                self.connected = False
                try:
                    port.close()
                except Exception:
                    pass
            self._stop.wait(self.retry_s)

    def _open(self):
        if self.device == "auto":
            ports = list(self._ports())
            if not ports:
                self._say("missing", "warning",
                          f"No lidar: nothing at {' or '.join(AUTO_GLOBS)}. "
                          f"Driving without it, looking again every "
                          f"{self.retry_s:g} s")
                return None
            path = ports[self._next_candidate % len(ports)]
        else:
            path = self.device
        try:
            port = self._opener(path)
        except Exception as e:
            self.error = str(e)
            self._say("missing", "warning",
                      f"No lidar at {path}: {e}. Driving without it, "
                      f"trying again every {self.retry_s:g} s")
            if self.device == "auto":
                self._next_candidate += 1
            return None
        self.port_path = path
        self.connected = True
        self.error = None
        return port

    def _pump(self, port):
        opened = self._clock()
        heard = False
        while not self._stop.is_set():
            try:
                data = port.read(READ_CHUNK)
            except (BlockingIOError, InterruptedError):
                # select() said readable and read() said EAGAIN; seen on the
                # car. pyserial 3.5 swallows it itself; this is in case a
                # different port object does not.
                data = b""
            now = self._clock()
            if not data:
                if not heard and now - opened > self.silent_s:
                    self._say("silent", "warning",
                              f"{self.port_path} is open but no lidar packet "
                              f"in {self.silent_s:.0f} s")
                    if self.device == "auto":
                        self._next_candidate += 1
                    return
                continue
            self.bytes_read += len(data)
            for packet in self.parser.feed(data):
                if not heard:
                    heard = True
                    self._said.difference_update(("missing", "silent", "lost"))
                    self._log("info", "Lidar",
                              f"Lidar on {self.port_path}, "
                              f"{packet.speed_dps / 6.0:.0f} rpm")
                scan = self._assembler.add(packet, now)
                if scan is not None:
                    self._publish(scan, now)
            if not heard and now - opened > self.silent_s:
                self._say("silent", "warning",
                          f"{self.port_path} sends bytes but no lidar packet "
                          f"in {self.silent_s:.0f} s")
                if self.device == "auto":
                    self._next_candidate += 1
                return

    def _publish(self, scan, now):
        bins = bin_scan(scan.angles_deg, scan.distances_mm, self.mount)
        scan = scan._replace(bins=bins)
        self._scan = scan
        self.scans += 1
        slam = self.slam
        if slam is not None:
            try:
                slam.on_scan(scan)
            except Exception:
                pass
        tele = self.tele
        if tele is None:
            return
        try:
            if tele.wants("lidar", now):
                tele.offer("lidar", lidar_sample(scan, self.stats(now)), now)
            if tele.wants("tf", now) and not (
                    slam is not None and slam.owns_tf(now)):
                tele.offer("tf", self.mount, now)
            if tele.wants("lidar_scan", now):
                tele.offer("lidar_scan", (scan, self.mount), now)
            if tele.wants("lidar_truck", now):
                tele.offer("lidar_truck", None, now)
        except Exception:
            pass

    def _say(self, key, level, message):
        if key in self._said:
            return
        self._said.add(key)
        self._log(level, "Lidar", message)

    def _log(self, level, subject, message):
        if self.logger is not None:
            try:
                self.logger.log(level, subject, message)
            except Exception:
                pass


def lidar_sample(scan, stats) -> dict:
    """A scan and the reader's stats as a /lidar row."""
    bins = scan.bins
    if bins is None:
        bins = bin_scan(scan.angles_deg, scan.distances_mm)
    row = {
        "bins": [int(v) for v in bins],
        "bin_deg": 360.0 / len(bins),
        "points": int(len(scan.distances_mm)),
        "valid_points": scan.valid_points,
        "rpm": round(scan.rpm, 1),
        "packets_per_s": float(stats.get("packets_per_s", 0.0)),
        "crc_errors": int(stats.get("crc_errors", 0)),
        "crc_error_rate": float(stats.get("crc_error_rate", 0.0)),
        "rejected_packets": int(stats.get("rejected_packets", 0)),
        "stamp_ms": -1 if scan.stamp_ms is None else int(scan.stamp_ms),
        "stamps_ms": [int(v) for v in scan.stamps_ms],
    }
    for name in SECTORS:
        row[f"nearest_{name}_mm"] = nearest_in_sector(bins, name)
    return row
