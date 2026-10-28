"""What the IMU actually reads, so GYRO_MOVING_MIN is not a guess.

driver.py decides the car is stuck off the gyro, because LINEAR_ACCELERATION
is gravity compensated and reads the same at a steady creep as it does parked.
GYRO_MOVING_MIN is currently anchored off the parked end of the range only --
I measured that from a desk, with the car sitting still. The number I still
owe it is the other end: how much gyro a creep at forward_speed actually
produces.

Run it twice, same duration both times:

    # 1. the floor. Car parked, powered, ESC armed, hands off.
    sudo systemctl stop crawler
    .venv/bin/python scripts/scratch/scratch_imu_floor.py 60

    # 2. the signal. Wheels up on the stand, or rock the car by hand the
    #    whole time -- anything that is genuinely chassis movement.
    .venv/bin/python scripts/scratch/scratch_imu_floor.py 60
    sudo systemctl start crawler

GYRO_MOVING_MIN wants to sit between the parked max and the moving p05. If
those two overlap, the gyro is not a good enough witness on this chassis and
the stuck check needs a wheel-speed source instead.

Takes the camera to itself, so the service has to be stopped first. Touches
nothing that can move the car: no pigpio, no VehicleController, no throttle.
"""

import csv
import math
import statistics
import sys
import time

from crawler.hardware import camera

TICK = 0.1  # the driver's own tick rate, so this sees what the driver sees


def distribution(name, values, unit):
    values = sorted(values)
    def p(q):
        return values[min(len(values) - 1, int(q * len(values)))]
    print(f"  {name:18s} n={len(values):5d}  min {values[0]:.5f}  p05 {p(0.05):.5f}  "
          f"p50 {p(0.5):.5f}  p95 {p(0.95):.5f}  max {values[-1]:.5f}  "
          f"mean {statistics.fmean(values):.5f}  {unit}")


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    out = sys.argv[2] if len(sys.argv) > 2 else None

    hub = camera.CameraHub(rectified=False)
    hub.start()
    if not hub.wait_ready(40):
        print(f"camera would not open: {hub.state} {hub.error}")
        print("is crawler.service still running? it holds the OAK-D.")
        return 1

    view = hub.view("imu-floor")
    rows = []
    ticks = missing = 0

    print(f"recording {seconds:.0f} s at {1/TICK:.0f} Hz...")
    start = time.monotonic()
    next_tick = start
    while time.monotonic() - start < seconds:
        now = time.monotonic()
        if now >= next_tick:
            next_tick = now + TICK
            ticks += 1
            sample = view.get_imu_sample()
            if sample is None:
                missing += 1
            else:
                ax, ay, az = sample["accel"]
                gx, gy, gz = sample["gyro"]
                rows.append((round(now - start, 4), ax, ay, az, gx, gy, gz))
        time.sleep(0.01)

    hub.close()

    print(f"\n{ticks} ticks, {missing} of them with no IMU packet "
          f"({100 * missing / max(ticks, 1):.2f}%)")
    distribution("accel magnitude", [math.dist((0, 0, 0), r[1:4]) for r in rows], "m/s^2")
    distribution("gyro magnitude", [math.dist((0, 0, 0), r[4:7]) for r in rows], "rad/s")

    if out:
        with open(out, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t", "ax", "ay", "az", "gx", "gy", "gz"])
            writer.writerows(rows)
        print(f"\nsamples written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
