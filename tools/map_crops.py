"""The policy's map input for recordings that have none: replay the car's SLAM.

    python tools/map_crops.py OUT_DIR RUN.mcap [RUN.mcap ...]

A recording carries /lidar/scan and /imu but not the map the truck would
have had, so this runs the same crawler.mapping.slam.Slam the car runs with
--slam over each file, scan by scan, through tools/slam_replay.py's reader
and loop, and after every step draws the same egomap.EgoMap crop the
runner's worker draws live. Nothing from later in the file reaches a step.

Writes OUT_DIR/<run>.npz, one row per scan:

  log_t     the scan's log time, seconds: what a dataset tick pairs on
  t         Result.t, the crop's own time
  present   egomap.trusted(Result)
  codes     uint8 (n, SIZE, SIZE)
  trail     uint8 (n, SIZE, SIZE)
  layout    egomap.LAYOUT

policy_train.py --map OUT_DIR pairs each dataset tick with the newest scan
at or before it, no more than egomap.STALE_S old (egomap.pair_ticks), as the car
reads the runner's newest crop. A file whose layout is not egomap.LAYOUT is
refused there.

Needs scipy and the `mcap` package, as slam_replay.py does.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crawler.mapping import egomap as em      # noqa: E402
from crawler.mapping.imu import ImuTrack      # noqa: E402
from crawler.mapping.slam import Slam         # noqa: E402
import slam_replay as sr                      # noqa: E402


def crops_of(d, raw_time=False):
    """Slam and EgoMap over one loaded recording -> the arrays above."""
    imu = ImuTrack()
    slam = Slam(imu=imu)
    ego = em.EgoMap()
    ts, _ = sr.scan_times(d, raw_time)
    rows = d["imu"]
    n = len(ts)
    codes = np.zeros((n, em.SIZE, em.SIZE), np.uint8)
    trail = np.zeros_like(codes)
    present = np.zeros(n, bool)
    t = np.zeros(n)
    k = 0
    for i in range(n):
        log_t = d["t"][i]
        while k < len(rows) and rows[k, 0] <= log_t:
            r = rows[k]
            up = r[4:7]
            imu.add(r[0], r[1:4], up if np.linalg.norm(up) > 0.5 else None)
            k += 1
        res = slam.step(float(ts[i]), d["ranges"][i])
        crop = ego.after_step(slam.grid, res)
        codes[i], trail[i], present[i], t[i] = crop.codes, crop.trail, crop.present, crop.t
    return dict(log_t=np.asarray(d["t"], float), t=t, present=present, codes=codes,
                trail=trail, layout=np.array(em.LAYOUT))


# The pairing lives in egomap (numpy only), so the trainer needs no scipy.
pair_ticks = em.pair_ticks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--raw-time", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for path in args.runs:
        name = os.path.basename(path).rsplit(".", 1)[0]
        out = crops_of(sr.load(path), args.raw_time)
        np.savez_compressed(os.path.join(args.out, name + ".npz"), **out)
        print(f"{name}: {len(out['t'])} scans, {int(out['present'].sum())} present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
