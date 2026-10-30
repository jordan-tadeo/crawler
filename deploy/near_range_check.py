"""How near the front can actually see: the OAK-D and the roof lidar against
a tape, parked.

On the car, from the checkout root, parked, with a flat box (a shoebox on its
side is fine, but it has to stand at least 0.35 m tall so the lidar's scan
plane meets it too) and a tape measure:

    ./.venv/bin/python deploy/near_range_check.py
    ./.venv/bin/python deploy/near_range_check.py --at 300,200,150,120,100

It stops the crawler service if it is running (it owns the camera and the
lidar) and starts it again on the way out, whatever happens. Then for each
distance it asks for the box's face square to the front bumper that far off
it, measured from the bumper, waits for Enter, reads two seconds of both
sensors, and prints one line:

    tape 150 | camera centre 178 lens = 150 off bumper, valid 97%, probes NNNNN.... | lidar path 149, centre 151

and writes everything to log/near_range_check-<utc>.json.

What to look for. The camera reads the box at tape + 28 (lens to bumper)
down to about 159 off the lens, 131 off the bumper: that is the stereo
clamp with extended disparity on, which it already is. Nearer than that the
centre goes invalid (valid near 0%) or reads wrong, and the probes go '?'.
The lidar should read the tape within about 15 mm all the way down, to the
bumper if the box is tall enough. If it does, --lidar-near can be trusted
for the creep's last 25 mm; if it reads short (the pan-tilt mount or the
hood in its arc) or long, it cannot, and the number says by how much.

--dry-run runs the whole thing against a simulated box, with no hardware
and no service touched.

This never imports pigpio, the PCA9685 or the ESC (they are blocked in
sys.modules before anything else is imported), so it cannot move the car or
the camera mount. It opens the OAK-D read-only, every queue drained and
non-blocking (oakd.OakD), and the lidar through the same LidarReader the
car uses.

Exit status: 0 ran, 1 a sensor would not open, 2 the service would not stop.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

# Nothing that can move the truck, or the camera mount, gets imported here.
for _mod in ("pigpio", "board", "busio", "adafruit_pca9685", "adafruit_motor",
             "adafruit_motor.servo", "crawler.hardware.vehicle_controller"):
    sys.modules[_mod] = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from crawler.autopilot import driver as dv  # noqa: E402
from crawler.hardware import lidar as ld  # noqa: E402

SERVICE = "crawler"
DEFAULT_AT = (300, 250, 200, 175, 150, 130, 110, 90, 70)
READ_S = 2.0
# The middle of the frame: a third of its width, the rows either side of the
# optical axis. The box fills it at every distance asked for.
CENTRE_COLS = (1 / 3, 2 / 3)
CENTRE_ROWS = (0.4, 0.6)


# --- what one frame and one scan say ------------------------------------------

def depth_reading(frame):
    """One depth frame (mm, 0 invalid): the centre's median, its valid
    fraction, and the nine probes as the driver reads them. Pure."""
    f = np.asarray(frame)
    h, w = f.shape
    c = f[int(h * CENTRE_ROWS[0]):int(h * CENTRE_ROWS[1]),
          int(w * CENTRE_COLS[0]):int(w * CENTRE_COLS[1])]
    valid = c[c > 0]
    scan = dv.free_by_column(f)
    return {
        "centre_mm": float(np.median(valid)) if valid.size else None,
        "centre_valid": float(valid.size) / float(c.size),
        "probes": dv.probe_marks(scan),
        "probe_mm": [float(x) if np.isfinite(x) else None for x in scan.distance_mm],
    }


def lidar_reading(bins):
    """One binned scan: the front arc's clearance over the whole path and
    the middle probe's lane, mm off the bumper (None for nothing). Pure."""
    def mm(v):
        return float(v) if np.isfinite(v) else None
    centre = dv.probe_band_y(dv.PROBE_COLUMNS // 2, dv.CLEARANCE_MM)
    return {
        "path_mm": mm(dv.front_band_clearance_mm(
            bins, -dv.FRONT_PATH_HALF_MM, dv.FRONT_PATH_HALF_MM)),
        "centre_mm": mm(dv.front_band_clearance_mm(bins, *centre)),
    }


def _median(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def summarise(tape_mm, depth, scans):
    """Two seconds of readings at one tape distance, as one record."""
    centre = _median([d["centre_mm"] for d in depth])
    rec = {
        "tape_mm": float(tape_mm),
        "frames": len(depth),
        "scans": len(scans),
        "camera_centre_lens_mm": centre,
        "camera_centre_bumper_mm": (centre - dv.LENS_TO_FRONT_BUMPER_MM
                                    if centre is not None else None),
        "camera_centre_valid": (float(np.median([d["centre_valid"] for d in depth]))
                                if depth else 0.0),
        "probes": depth[-1]["probes"] if depth else "",
        "lidar_path_mm": _median([s["path_mm"] for s in scans]),
        "lidar_centre_mm": _median([s["centre_mm"] for s in scans]),
    }
    for k in ("camera_centre_bumper_mm", "lidar_path_mm", "lidar_centre_mm"):
        v = rec[k]
        rec[k.replace("_mm", "_error_mm")] = (v - tape_mm) if v is not None else None
    return rec


def line(rec):
    def n(v):
        return "---" if v is None else "%.0f" % v
    return ("tape %s | camera centre %s lens = %s off bumper, valid %.0f%%, "
            "probes %s | lidar path %s, centre %s" % (
                n(rec["tape_mm"]), n(rec["camera_centre_lens_mm"]),
                n(rec["camera_centre_bumper_mm"]),
                100 * rec["camera_centre_valid"], rec["probes"],
                n(rec["lidar_path_mm"]), n(rec["lidar_centre_mm"])))


# --- the service ----------------------------------------------------------------

class Service:
    """Stops the crawler service if it is running, and restart() starts it
    again only if this stopped it. `run` is subprocess.run's shape."""

    def __init__(self, run=subprocess.run, name=SERVICE):
        self.run, self.name, self.stopped = run, name, False

    def active(self):
        r = self.run(["systemctl", "is-active", self.name],
                     capture_output=True, text=True)
        return (r.stdout or "").strip() == "active"

    def stop(self):
        if not self.active():
            return True
        print(f"stopping {self.name} (it owns the camera and the lidar)", flush=True)
        r = self.run(["sudo", "-n", "systemctl", "stop", self.name])
        self.stopped = r.returncode == 0
        return self.stopped

    def restart(self):
        if self.stopped:
            print(f"starting {self.name} again", flush=True)
            self.run(["sudo", "-n", "systemctl", "start", self.name])
            self.stopped = False


# --- the sensors ------------------------------------------------------------------

class _PrintLog:
    def log(self, level, subject, message):
        print(f"[{level}] {subject}: {message}", flush=True)


class RealSensors:
    def __init__(self):
        from crawler.hardware import oakd
        self.cam = oakd.OakD(rectified=False)
        self.lidar = ld.LidarReader(device="auto", logger=_PrintLog())
        self.lidar.start()

    def depth(self):
        # Drained every pass, depth and the rest, or XLink jams (oakd.OakD).
        self.cam.get_disparity_frame()
        self.cam.get_imu_sample()
        return self.cam.get_depth_frame()

    def scan(self):
        s = self.lidar.latest(max_age=0.2)
        return s.bins if s is not None else None

    def close(self):
        for f in (self.lidar.close, self.cam.close):
            try:
                f()
            except Exception:
                pass


class SimSensors:
    """A flat box `tape` mm off the bumper. The camera clamps at
    STEREO_NEAR_MM like the real pair; the lidar reads it exactly."""

    def __init__(self):
        self.tape = 300.0

    def depth(self):
        z = self.tape + dv.LENS_TO_FRONT_BUMPER_MM
        f = np.full((400, 640), 3000, np.uint16)
        f[60:340, 120:520] = int(z) if z >= dv.STEREO_NEAR_MM else 0
        return f

    def scan(self):
        b = np.zeros(ld.BINS, np.uint16)
        x = ld.FRONT_BUMPER_X_MM + self.tape
        for k, brg in enumerate(ld.bin_bearings()):
            if abs(brg) < 40:
                b[k] = int(round(x / np.cos(np.radians(brg))))
        return b

    def close(self):
        pass


def read_at(sensors, tape_mm, seconds=READ_S, clock=time.monotonic,
            sleep=time.sleep):
    depth, scans, last = [], [], None
    end = clock() + seconds
    while clock() < end:
        f = sensors.depth()
        if f is not None:
            depth.append(depth_reading(f))
        b = sensors.scan()
        if b is not None and (last is None or not np.array_equal(b, last)):
            last = b
            scans.append(lidar_reading(b))
        sleep(0.02)
    return summarise(tape_mm, depth, scans)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="near_range_check.py",
        description="Read the camera and the lidar against a box at tape distances.")
    p.add_argument("--at", default=",".join(str(x) for x in DEFAULT_AT),
                   help="tape distances off the front bumper, mm, nearest last")
    p.add_argument("--seconds", type=float, default=READ_S)
    p.add_argument("--dry-run", action="store_true",
                   help="a simulated box, no hardware, no service")
    return p.parse_args(argv)


def main(argv=None, ask=input, run=subprocess.run, sensors=None,
         clock=time.monotonic, sleep=time.sleep, out_dir=None):
    args = parse_args(argv)
    at = [float(x) for x in args.at.split(",") if x.strip()]
    service = Service(run=run)
    if not args.dry_run and not service.stop():
        print("could not stop the service (sudo -n): nothing was read")
        return 2
    records = []
    try:
        if sensors is None:
            try:
                sensors = SimSensors() if args.dry_run else RealSensors()
            except Exception as e:
                print(f"a sensor would not open: {e}")
                return 1
        try:
            for tape in at:
                if isinstance(sensors, SimSensors):
                    sensors.tape = tape
                ask(f"box face square to the bumper, {tape:.0f} mm off it. Enter to read ")
                rec = read_at(sensors, tape, args.seconds, clock, sleep)
                records.append(rec)
                print(line(rec), flush=True)
        finally:
            sensors.close()
    finally:
        service.restart()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_dir or os.path.join(ROOT, "log")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"near_range_check-{stamp}.json")
    with open(path, "w") as fh:
        json.dump({"dry_run": args.dry_run, "stereo_near_mm": dv.STEREO_NEAR_MM,
                   "lens_to_bumper_mm": dv.LENS_TO_FRONT_BUMPER_MM,
                   "records": records}, fh, indent=1)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
