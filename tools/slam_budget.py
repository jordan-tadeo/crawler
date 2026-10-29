"""What --slam costs the driving loop, on the car, with the real auto workload.

The real camera (depth for the driver, IMU for the mapper), the real lidar,
the real auto driver (the rules one, or --policy MODEL.onnx for the learned
one), stepped from a 100 Hz loop shaped like control_loop(): step, then wait
LOOP_INTERVAL. The one thing that is not real is the vehicle controller. It
is a recorder: nothing here imports pigpio, the PCA9685 or the ESC (they are
blocked in sys.modules before anything is imported), so no throttle or
steering command reaches any hardware.

Run with the crawler service stopped (it owns the camera and the lidar):

    sudo -n systemctl stop crawler
    python tools/slam_budget.py --seconds 120 --slam off
    python tools/slam_budget.py --seconds 120 --slam process
    python tools/slam_budget.py --seconds 120 --slam thread
    sudo -n systemctl start crawler

Reports, per run: the auto tick (the passes where the driver decided) and
every pass, p50 / p99 / max ms; the loop period p50 / p99 / max; and with
SLAM on, scans in, dropped, matched, and the mapper's own step ms.
"""

import argparse
import json
import os
import sys
import threading
import time

# Nothing that can move the truck gets imported, by anything, from here on.
for _mod in ("pigpio", "board", "busio", "adafruit_pca9685", "adafruit_motor",
             "adafruit_motor.servo", "crawler.hardware.vehicle_controller"):
    sys.modules[_mod] = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


class RecordingVecon:
    """The VehicleController's surface, recording and nothing else."""

    def __init__(self):
        self.last_throttle = 0.0
        self.last_steering_front = 0.0
        self.last_steering_rear = 0.0
        self.last_esc_pulse = 0
        self.commands = 0

    def set_throttle(self, value):
        self.last_throttle = value
        self.commands += 1

    def set_steering(self, front, rear=None):
        self.last_steering_front = front
        self.last_steering_rear = rear
        self.commands += 1

    def return_neutral(self):
        self.last_throttle = 0.0

    def get_state(self):
        return "bench"

    def close(self):
        pass


class Log:
    def log(self, level, subject, message):
        print(f"[{level}] {subject}: {message}", file=sys.stderr, flush=True)

    def csv(self, *a, **k):
        pass


def stats(a):
    a = np.asarray(a, float)
    if not len(a):
        return None
    return {"n": int(len(a)), "p50": round(float(np.percentile(a, 50)), 2),
            "p99": round(float(np.percentile(a, 99)), 2),
            "max": round(float(a.max()), 2)}


def live_check(slam, results):
    """What the mapper made of the run: its states, writes and step time,
    how far the pose wandered, the map's size, and the pose interface.
    Counts only: the map itself is a picture of the house and stays put."""
    from collections import Counter
    from crawler.mapping import grid as gr
    out = {"results_seen": len(results)}
    if results:
        xs = np.array([r.x for r in results])
        ys = np.array([r.y for r in results])
        out.update(
            states=dict(Counter(r.state for r in results)),
            why=dict(Counter(r.why for r in results)),
            written=int(sum(r.written for r in results)),
            step_ms=stats([r.ms for r in results]),
            pose_spread_m=round(float(max(np.ptp(xs), np.ptp(ys))), 4),
            yaw_spread_deg=round(float(np.degrees(np.ptp(
                [r.yaw for r in results]))), 3))
    view = slam.map()
    if view is not None:
        out.update(map_px=list(view.codes.shape),
                   map_occupied=int((view.codes == gr.OCCUPIED).sum()),
                   map_free=int((view.codes == gr.FREE).sum()))
    p = slam.pose_stamped()
    if p is not None:
        out.update(pose_confidence=round(p.confidence, 3),
                   pose_at_ok=slam.pose_at(p.t - 0.05) is not None)
    st = slam.stats()
    out.update(grid_grew=st.get("grid_grew"), grid_dropped=st.get("grid_dropped"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--slam", choices=("off", "thread", "process"),
                    default="off")
    ap.add_argument("--policy", default=None, help="an .onnx, read only")
    ap.add_argument("--lidar", default="auto")
    a = ap.parse_args(argv)

    from crawler import __main__ as cm
    from crawler.hardware import camera
    from crawler.hardware import lidar as ld

    log = Log()
    hub = camera.CameraHub(logger=log, rectified=False)
    slam = None
    if a.slam != "off":
        from crawler.mapping import runner as rn
        slam = rn.SlamRunner(mode=a.slam, logger=log)
        slam.start()
        hub.imu_listeners.append(slam.on_imu)
    lidar = ld.LidarReader(device=a.lidar, logger=log, slam=slam)
    lidar.start()
    hub.start()
    if not hub.wait_ready():
        print(json.dumps({"error": f"camera: {hub.error or hub.state}"}))
        return 1
    vecon = RecordingVecon()
    auto = cm.AutoPilot(vecon, log, hub=hub, policy_path=a.policy, lidar=lidar)
    end = time.monotonic() + 15.0
    while not auto.resume():
        if auto.unavailable or time.monotonic() > end:
            print(json.dumps({"error": "auto would not start"}))
            return 1
        time.sleep(0.5)
    # Let the lidar and the mapper settle in before counting.
    time.sleep(3.0)

    ticks, passes, periods = [], [], []
    results, seen_t = [], None
    stop = threading.Event()
    clock = time.monotonic
    t_end = clock() + a.seconds
    last = None
    try:
        while clock() < t_end:
            now = clock()
            if last is not None:
                periods.append((now - last) * 1000.0)
            last = now
            due = now >= auto._next_tick
            auto.step(now)
            ms = (clock() - now) * 1000.0
            passes.append(ms)
            if due:
                ticks.append(ms)
            if slam is not None:
                # Outside the timed part: every result the mapper made.
                res = slam.latest()
                if res is not None and res.t != seen_t:
                    seen_t = res.t
                    results.append(res)
            stop.wait(cm.LOOP_INTERVAL)
    finally:
        out = {"slam": a.slam, "policy": os.path.basename(a.policy or "rules"),
               "seconds": a.seconds, "auto_tick_ms": stats(ticks),
               "pass_ms": stats(passes), "period_ms": stats(periods),
               "lidar_scans": lidar.scans, "commands_recorded": vecon.commands}
        if slam is not None:
            st = slam.stats()
            res = slam.latest()
            out.update(slam_scans=st["scans"], slam_dropped=st["dropped"],
                       slam_results=st["results"], slam_errors=st["errors"],
                       imu_samples=st["imu_samples"],
                       slam_state=res.state if res else None)
            out.update(live_check(slam, results))
        try:
            auto.stop()
        except Exception:
            pass
        lidar.close()
        if slam is not None:
            slam.close()
        hub.close()
        print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
