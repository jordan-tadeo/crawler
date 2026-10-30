"""deploy/near_range_check.py, against its simulated box. Nothing here
touches hardware or the service."""

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

from crawler.autopilot import driver as dv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(ROOT, "deploy", "near_range_check.py")
_spec = importlib.util.spec_from_file_location("near_range_check_under_test", _PATH)
nr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nr)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_it_cannot_import_anything_that_moves_the_car():
    out = subprocess.run(
        [sys.executable, "-c",
         "import runpy, sys; runpy.run_path(%r, run_name='not_main');"
         # Blocked, not merely unimported: None in sys.modules makes any
         # later import of them raise.
         "bad = [m for m in ('pigpio', 'adafruit_pca9685', "
         "'crawler.hardware.vehicle_controller') "
         "if m not in sys.modules or sys.modules[m] is not None];"
         "print('BAD', bad) if bad else print('OK')" % _PATH],
        capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert out.stdout.strip().endswith("OK"), out.stdout + out.stderr
    src = open(_PATH).read()
    for word in ("set_throttle", "set_steering", "import pigpio"):
        assert word not in src


def test_the_sim_box_reads_the_tape_on_the_lidar_and_clamps_on_the_camera(tmp_path):
    clock = Clock()
    asked = []
    code = nr.main(["--dry-run", "--at", "300,150,100", "--seconds", "0.2"],
                   ask=asked.append, clock=clock, sleep=clock.sleep,
                   out_dir=str(tmp_path))
    assert code == 0 and len(asked) == 3
    import json
    (f,) = list(tmp_path.iterdir())
    recs = json.load(open(f))["records"]
    far, near, under = recs
    assert far["camera_centre_bumper_mm"] == pytest.approx(300, abs=1)
    assert near["camera_centre_bumper_mm"] == pytest.approx(150, abs=1)
    # Under the clamp the camera has nothing; the lidar still reads the tape.
    assert under["camera_centre_lens_mm"] is None
    assert under["camera_centre_valid"] == 0.0
    for r in recs:
        assert r["lidar_centre_error_mm"] == pytest.approx(0, abs=2)
        assert r["lidar_path_mm"] <= r["tape_mm"] + 1


def test_the_service_is_started_again_only_if_it_was_stopped_and_always_then():
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="active\n")

    class Boom(nr.SimSensors):
        def depth(self):
            raise RuntimeError("X_LINK_ERROR")

    clock = Clock()
    with pytest.raises(RuntimeError):
        nr.main(["--at", "200", "--seconds", "0.1"], ask=lambda _: None, run=run,
                sensors=_Lazy(Boom), clock=clock,
                sleep=clock.sleep)
    assert ["sudo", "-n", "systemctl", "stop", "crawler"] in calls
    assert calls[-1] == ["sudo", "-n", "systemctl", "start", "crawler"]

    calls.clear()
    svc = nr.Service(run=lambda cmd, **kw: (calls.append(cmd), subprocess.CompletedProcess(
        cmd, 0, stdout="inactive\n"))[1])
    assert svc.stop()
    svc.restart()
    assert all("start" not in c for c in calls)


class _Lazy(nr.SimSensors):
    """Sensors handed in, but the service still has to be stopped first:
    main() only skips the service for --dry-run."""

    def __init__(self, cls):
        self.inner = cls()

    def depth(self):
        return self.inner.depth()

    def scan(self):
        return self.inner.scan()


def test_a_frame_reads_the_way_the_driver_reads_it():
    f = np.full((400, 640), 3000, np.uint16)
    f[150:250, 220:420] = 200
    r = nr.depth_reading(f)
    assert r["centre_mm"] == 200
    assert r["probes"] == dv.probe_marks(dv.free_by_column(f))
    assert nr.depth_reading(np.zeros((400, 640), np.uint16))["centre_mm"] is None
    # Invalid pixels are not a distance: most of the centre gone still reads
    # the rest.
    g = np.full((400, 640), 200, np.uint16)
    g[160:210, :] = 0
    r = nr.depth_reading(g)
    assert r["centre_mm"] == 200
    assert r["centre_valid"] == pytest.approx(30 / 80)
