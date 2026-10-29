"""Is the roof lidar talking, and is it the right way round.

On the car, from the checkout root, with nothing else reading the lidar (the
service only does with --lidar):

    ./.venv/bin/python tools/lidar_check.py                 # find the port, 5 s
    ./.venv/bin/python tools/lidar_check.py --device /dev/ttyUSB0 --seconds 20
    ./.venv/bin/python tools/lidar_check.py --list          # the ports it would try

It reads through the same LidarReader the car does, prints one line a second
with the nearest thing in each quarter around the truck, and a summary at the
end: rpm, packets a second, CRC error rate, points per scan, and the nearest
range per quarter with the bearing it was at.

The orientation check: hold a hand 20-30 cm off the truck's front, then its
right. The near number has to show up under front, then right. If it shows
up a quarter turn off, lidar.MOUNT's yaw is wrong; --yaw tries another one
without editing anything.

This reads a serial port and nothing else. It never imports pigpio, never
opens the ESC or the servos, and cannot move the car.

Exit status: 0 packets parsed, 1 no port, 2 a port but no lidar packets on it.
"""

import argparse
import os
import sys
import time

# Run by path on the car, possibly out of the old env/ venv that never
# installed this project, so the checkout root goes on the path for `crawler`.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from crawler.hardware import lidar as ld  # noqa: E402

QUARTERS = ("front", "left", "rear", "right")


class _PrintLog:
    def log(self, level, subject, message):
        print(f"[{level}] {subject}: {message}", flush=True)


def _mm(v):
    return f"{int(v):>5d} mm" if v else "    ---  "


def _nearest_with_bearing(bins, sector):
    """(range mm, bearing deg) of the nearest bin in a sector, or (0, None)."""
    bins = np.asarray(bins)
    mask = ld.sector_mask(sector, len(bins)) & (bins > 0)
    if not mask.any():
        return 0, None
    idx = np.flatnonzero(mask)
    i = idx[np.argmin(bins[idx])]
    return int(bins[i]), float(ld.bin_bearings(len(bins))[i])


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="lidar_check.py",
        description="Read the roof lidar for a few seconds and say what it sees.")
    p.add_argument("--device", default="auto",
                   help="serial port, or 'auto' to find the USB-UART board "
                        "(default: auto)")
    p.add_argument("--seconds", type=float, default=5.0,
                   help="how long to read (default: 5)")
    p.add_argument("--yaw", type=float, default=None,
                   help=f"try a different mount yaw in degrees "
                        f"(default: lidar.MOUNT's {ld.MOUNT.yaw_deg:g})")
    p.add_argument("--list", action="store_true",
                   help="print the ports auto would try, and exit")
    return p.parse_args(argv)


def main(argv=None, opener=None, ports=None, clock=time.monotonic,
         sleep=time.sleep):
    args = parse_args(argv)
    ports = ports or ld.candidate_ports

    if args.list:
        found = ports()
        print("\n".join(found) if found else
              f"nothing at {' or '.join(ld.AUTO_GLOBS)}")
        return 0 if found else 1

    mount = ld.MOUNT
    if args.yaw is not None:
        mount = mount._replace(yaw_deg=args.yaw)
    print(f"mount: yaw {mount.yaw_deg:g} deg, offset ({mount.x_mm:g}, "
          f"{mount.y_mm:g}) mm, upside down {mount.upside_down}")
    if args.device == "auto":
        found = ports()
        print(f"candidates: {', '.join(found) if found else 'none'}")

    reader = ld.LidarReader(device=args.device, logger=_PrintLog(),
                            mount=mount, opener=opener, ports=ports,
                            clock=clock, retry_s=0.5,
                            silent_s=max(1.0, args.seconds))
    reader.start()

    start = clock()
    next_line = start + 1.0
    seen = []
    last_t = None
    try:
        while clock() - start < args.seconds:
            scan = reader.latest(max_age=float("inf"))
            if scan is not None and scan.t != last_t:
                last_t = scan.t
                seen.append(scan)
            if clock() >= next_line:
                next_line += 1.0
                _print_live(reader, scan)
            sleep(0.02)
    finally:
        reader.close()

    return _summary(reader, seen, clock() - start)


def _print_live(reader, scan):
    st = reader.stats()
    if scan is None or scan.bins is None:
        print(f"  {st['device'] or '(no port)'}: no scan yet, "
              f"{st['packets']} packets, {st['crc_errors']} CRC errors",
              flush=True)
        return
    near = "  ".join(f"{q} {_mm(ld.nearest_in_sector(scan.bins, q))}"
                     for q in QUARTERS)
    print(f"  {st['rpm']:5.0f} rpm  {st['packets_per_s']:5.0f} pkt/s  {near}",
          flush=True)


def _summary(reader, scans, elapsed):
    st = reader.stats()
    print()
    print(f"port:            {reader.port_path or 'none'}")
    print(f"read for:        {elapsed:.1f} s, {reader.bytes_read} bytes")
    if reader.port_path is None:
        print("no port opened. Is the USB-UART board plugged in? "
              "--list shows what auto looks for")
        return 1
    total = st["packets"] + st["crc_errors"]
    print(f"packets:         {st['packets']} "
          f"({st['packets'] / elapsed if elapsed > 0 else 0:.0f}/s, "
          f"417/s expected at 5000 points/s)")
    print(f"CRC errors:      {st['crc_errors']} "
          f"({100.0 * st['crc_errors'] / total if total else 0:.2f}%)")
    print(f"rejected:        {st['rejected_packets']} packets, "
          f"{st['dropped_bytes']} bytes skipped")
    if not st["packets"]:
        print("no lidar packets on this port. Wrong port, wrong baud, Tx "
              "not wired to the board's Rx, or the motor is not spinning")
        return 2
    if not scans:
        print("packets but no whole revolution yet: read for longer")
        return 0
    rpm = [s.rpm for s in scans]
    pts = [len(s.distances_mm) for s in scans]
    valid = [s.valid_points for s in scans]
    print(f"scans:           {len(scans)} "
          f"({len(scans) / elapsed:.1f} Hz, 10 expected)")
    print(f"rpm:             {np.mean(rpm):.0f} (min {min(rpm):.0f}, "
          f"max {max(rpm):.0f}; 600 expected)")
    print(f"points per scan: {np.mean(pts):.0f} (min {min(pts)}, "
          f"max {max(pts)}), {np.mean(valid):.0f} with a return")
    print("nearest, over every scan (car frame, from the middle of the car):")
    stack = np.stack([s.bins for s in scans])
    stack = np.where(stack > 0, stack, np.iinfo(np.uint16).max)
    best = stack.min(axis=0)
    best = np.where(best == np.iinfo(np.uint16).max, 0, best)
    for q in QUARTERS:
        mm, bearing = _nearest_with_bearing(best, q)
        where = f" at {bearing:+.0f} deg" if bearing is not None else ""
        print(f"  {q:<6} {_mm(mm)}{where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
