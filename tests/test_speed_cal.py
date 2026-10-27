"""deploy/speed_cal.py, entirely in its dry-run world.

Nothing here touches hardware. The simulated car is driven by the same
Calibrator the real one is, through the same three seams (vehicle, pad,
camera), and the vehicle's command log is what every safety test reads: what
was sent to the ESC, and when.
"""

import importlib.util
import json
import math
import os
import signal
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(ROOT, "deploy", "speed_cal.py")
_spec = importlib.util.spec_from_file_location("speed_cal_under_test", _PATH)
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


def primed(start_mm=1600.0, esc=sc.EscModel()):
    """A sim car and a Calibrator that has already seen a few depth frames."""
    hw = sc.build_sim(start_mm, esc)
    cal = sc.Calibrator(hw.vehicle, hw.pad, hw.camera, hw.clock, carry=hw.carry)
    for _ in range(30):
        cal._read(hw.clock.now())
        hw.clock.sleep(sc.LOOP_S)
    return hw, cal


def throttles(hw):
    return [(t, v) for t, kind, v in hw.vehicle.log if kind == "throttle"]


def last_drive(hw):
    """Time of the last nonzero throttle command, or None."""
    moving = [t for t, v in throttles(hw) if v != 0.0]
    return moving[-1] if moving else None


def first_neutral_after(hw, t_from):
    for t, v in throttles(hw):
        if t >= t_from - 1e-9:
            assert v == 0.0, f"throttle {v} commanded at {t:.3f}, after {t_from:.3f}"
            return t
    return None


def ends_neutral_and_centred(hw):
    log = hw.vehicle.log
    last_t = [i for i, (_, k, _) in enumerate(log) if k == "throttle"][-1]
    last_s = [i for i, (_, k, _) in enumerate(log) if k == "steer"][-1]
    assert log[last_t][2] == 0.0
    assert tuple(log[last_s][2]) == (0.0, 0.0)
    assert hw.world.throttle == 0.0


FWD = sc.LegSpec("fwd", sc.FORWARD, 0.4, 4.0)


# -- the deadman ---------------------------------------------------------------

def _fault_mid_leg(kind, reason):
    hw, cal = primed(1600.0)
    hw.pad.hold()
    t_fault = hw.clock.now() + 0.8
    hw.pad.faults.append((t_fault, math.inf, kind))
    rec = cal.run_leg(FWD)
    assert rec["outcome"] == "aborted" and rec["ended"] == reason
    assert last_drive(hw) < t_fault
    t_neutral = first_neutral_after(hw, t_fault)
    assert t_neutral - t_fault <= sc.LOOP_S + 1e-9
    ends_neutral_and_centred(hw)


def test_letting_go_of_rb_is_neutral_within_one_loop():
    _fault_mid_leg("release", "released")


def test_a_nan_pad_is_neutral_within_one_loop():
    _fault_mid_leg("nan", "pad")


def test_a_lost_pad_is_neutral_within_one_loop():
    _fault_mid_leg("lost", "pad")


def test_b_stops_the_run_at_neutral():
    hw, cal = primed(1600.0)
    hw.pad.hold()
    t_fault = hw.clock.now() + 0.5
    hw.pad.faults.append((t_fault, math.inf, "abort"))
    with pytest.raises(sc.Stop):
        cal.run_leg(FWD)
    assert last_drive(hw) < t_fault
    ends_neutral_and_centred(hw)


def test_a_camera_stall_is_neutral_inside_the_stall_limit():
    hw, cal = primed(1600.0)
    hw.pad.hold()
    hw.world.camera_stalled_from = hw.clock.now() + 0.6
    rec = cal.run_leg(FWD)
    assert rec["outcome"] == "aborted" and rec["ended"] == "camera stall"
    last_rx = cal.last_frame_rx
    assert last_drive(hw) - last_rx <= sc.CAMERA_STALL_S + 1e-9
    assert first_neutral_after(hw, last_drive(hw) + 1e-6) - last_rx <= sc.CAMERA_STALL_S + sc.LOOP_S + 1e-9
    ends_neutral_and_centred(hw)


def test_an_rb_held_over_never_starts_the_next_leg():
    hw, cal = primed(1600.0)
    hw.pad.hold()
    hw.pad.prompt = lambda: None          # an owner who never lets go
    hw.pad.hooks.append((hw.clock.now() + 5.0, lambda: hw.pad.faults.append((0, math.inf, "abort"))))
    with pytest.raises(sc.Stop):
        cal.leg_with_retries(FWD, "go")
    assert last_drive(hw) is None


# -- the bounds ------------------------------------------------------------------

def test_a_forward_leg_ends_before_450_mm():
    hw, cal = primed(1000.0)
    hw.pad.hold()
    rec = cal.run_leg(FWD)
    assert rec["ended"] == "near"
    # Every forward command went out on a reading at or over the bound.
    assert cal.distance() < sc.FORWARD_MIN_MM
    assert hw.world.d > 300.0
    assert rec["drive_s"] < 2.0


def test_a_forward_leg_never_starts_inside_450_mm():
    hw, cal = primed(420.0)
    hw.pad.hold()
    rec = cal.run_leg(FWD)
    assert rec["outcome"] == "refused" and rec["ended"] == "near"
    assert last_drive(hw) is None


def test_a_forward_leg_never_starts_without_a_distance():
    hw, cal = primed(1600.0)
    cal.range = sc.Range(None, 900.0, 900.0)
    hw.world.distance_offset = math.nan     # every frame from here is unreadable
    hw.world.frame = lambda d: np.zeros(hw.world.SHAPE, dtype=np.uint16)
    hw.pad.hold()
    rec = cal.run_leg(FWD)
    assert rec["outcome"] == "refused"
    assert last_drive(hw) is None


def test_a_reverse_leg_never_runs_past_1_2_s_whatever_it_asks_for():
    hw, cal = primed(600.0)
    hw.pad.hold()
    rec = cal.run_leg(sc.LegSpec("rev", sc.REVERSE, -0.4, 5.0))
    assert rec["ended"] == "time"
    moving = [t for t, v in throttles(hw) if v != 0.0]
    assert moving[-1] - moving[0] < sc.REVERSE_MAX_S
    assert moving[-1] - moving[0] > sc.REVERSE_MAX_S - 2 * sc.LOOP_S
    assert hw.world.throttle == 0.0


def test_a_reverse_leg_ends_at_2_2_m():
    hw, cal = primed(1800.0)
    hw.pad.hold()
    rec = cal.run_leg(sc.LegSpec("rev", sc.REVERSE, -1.0, sc.REVERSE_MAX_S))
    assert rec["ended"] == "far"
    assert rec["drive_s"] < sc.REVERSE_MAX_S
    assert cal.distance() >= sc.REVERSE_MAX_MM
    assert hw.world.throttle == 0.0


# -- the exit paths ----------------------------------------------------------------

def _run_main(tmp_path, hw, extra=()):
    return sc.main(["--dry-run", "--log-dir", str(tmp_path), *extra], hardware=hw)


def _assert_clean_exit(hw):
    ends_neutral_and_centred(hw)
    assert hw.vehicle.closed and hw.camera.closed
    # Neutral and centre went out after the last drive command, before close.
    log = hw.vehicle.log
    i_close = [i for i, (_, k, _) in enumerate(log) if k == "close"][-1]
    tail = log[:i_close]
    assert tail[-2][1:] == ("throttle", 0.0) and tail[-1][1] == "steer"


def test_a_camera_exception_mid_leg_exits_neutral_and_centred(tmp_path):
    hw = sc.build_sim()
    hw.camera.raise_at = 4.0
    code = _run_main(tmp_path, hw)
    assert code == 1
    assert last_drive(hw) is not None and last_drive(hw) < 4.0
    _assert_clean_exit(hw)


def test_an_esc_exception_exits_neutral_and_centred(tmp_path):
    hw = sc.build_sim()
    hw.vehicle.raise_on_throttle_at = 6.0
    assert _run_main(tmp_path, hw) == 1
    _assert_clean_exit(hw)


def test_sigterm_mid_leg_exits_neutral_and_centred(tmp_path):
    hw = sc.build_sim()
    before = signal.getsignal(signal.SIGTERM)
    hw.pad.hooks.append((3.0, lambda: os.kill(os.getpid(), signal.SIGTERM)))
    assert _run_main(tmp_path, hw) == 130
    assert 2.0 < last_drive(hw) < 3.1
    _assert_clean_exit(hw)
    assert signal.getsignal(signal.SIGTERM) is before


def test_sighup_is_caught_too(tmp_path):
    hw = sc.build_sim()
    hw.pad.hooks.append((3.0, lambda: os.kill(os.getpid(), signal.SIGHUP)))
    assert _run_main(tmp_path, hw) == 130
    _assert_clean_exit(hw)


def test_ctrl_c_mid_leg_exits_neutral_and_centred(tmp_path):
    hw = sc.build_sim()
    hw.pad.hooks.append((3.0, lambda: os.kill(os.getpid(), signal.SIGINT)))
    assert _run_main(tmp_path, hw) == 130
    _assert_clean_exit(hw)


def test_the_signal_handler_commands_neutral_before_it_unwinds(tmp_path):
    """The first command after the signal comes from inside the handler."""
    import traceback
    hw = sc.build_sim()
    seen = {}

    def hook():
        seen["sent"] = True
        os.kill(os.getpid(), signal.SIGTERM)

    real_throttle = hw.vehicle.throttle

    def spy(value):
        if seen.get("sent") and "first" not in seen:
            names = [f.name for f in traceback.extract_stack()]
            seen["first"] = (value, "on_signal" in names)
        real_throttle(value)

    hw.vehicle.throttle = spy
    hw.pad.hooks.append((3.0, hook))
    _run_main(tmp_path, hw)
    assert seen["first"] == (0.0, True)


# -- the numbers ---------------------------------------------------------------------

def test_the_fit_recovers_a_known_line():
    samples = [(t, 1600.0 - 400.0 * t + (3.0 if i % 2 else -3.0))
               for i, t in enumerate(np.arange(0.0, 2.0, 1 / 16.0))]
    fit = sc.fit_speed(samples, 0.0, sc.FORWARD)
    assert fit.speed_mm_s == pytest.approx(400.0, rel=0.01)
    assert fit.residual_mm == pytest.approx(3.0, abs=0.5)
    assert fit.frames == sum(1 for t, _ in samples if t >= sc.SETTLE_S)
    rev = sc.fit_speed([(t, 800.0 + 250.0 * t) for t, _ in samples], 0.0, sc.REVERSE)
    assert rev.speed_mm_s == pytest.approx(250.0, rel=1e-6)


def test_the_fit_skips_the_settle():
    # Standing still for the settle, then 300 mm/s: a fit that kept the
    # settle would come out well under 300.
    samples = [(t, 1600.0 if t < sc.SETTLE_S else 1600.0 - 300.0 * (t - sc.SETTLE_S))
               for t in np.arange(0.0, 1.5, 1 / 16.0)]
    assert sc.fit_speed(samples, 0.0, sc.FORWARD).speed_mm_s == pytest.approx(300.0, rel=0.01)


def test_the_fit_refuses_too_few_frames():
    four = [(0.5 + 0.15 * i, 1000.0 - 40.0 * i) for i in range(4)]
    assert sc.fit_speed(four, 0.0, sc.FORWARD).speed_mm_s is None
    assert sc.fit_speed(four + [(1.1, 840.0)], 0.0, sc.FORWARD).speed_mm_s is not None


def test_the_sweep_recovers_the_simulated_speeds(tmp_path):
    hw = sc.build_sim()
    assert _run_main(tmp_path, hw, ["--no-lock"]) == 0
    summary = sc.main.last_summary
    esc = hw.world.esc
    straight = [r for r in summary["legs"] if not r["lock"] and r["speed_mm_s"] is not None
                and abs(esc.speed(r["throttle"])) > 100]
    assert len(straight) >= 9
    # One reverse leg is about 11 frames past its settle, and the sim
    # quantises depth to whole disparity levels the way the car does (44 mm
    # a level at 1.2 m, 120 at 2 m), so one leg is held to 12% and the set
    # to 5% on average. The match is what has to be right.
    errors = []
    for r in straight:
        truth = abs(esc.speed(r["throttle"]))
        assert r["speed_mm_s"] == pytest.approx(truth, rel=0.12), r["name"]
        errors.append(abs(r["speed_mm_s"] - truth) / truth)
    assert np.mean(errors) < 0.05
    # A forward leg last, after every reverse leg, so drift is bracketed.
    sweep = [r for r in summary["legs"] if not r["lock"] and r["outcome"] == "done"]
    assert sweep[-1]["direction"] == sc.FORWARD
    assert sweep[-1]["at_s"] > max(r["at_s"] for r in sweep if r["direction"] == sc.REVERSE)
    assert len(summary["forward_speeds_mm_s"]) >= sc.MIN_FORWARD_FITS
    truth = -(esc.rev_dead + esc.speed(0.4) / esc.rev_gain)
    assert summary["matched_reverse"] == pytest.approx(truth, abs=0.03)


def test_the_centre_distance_reads_the_box_where_the_probes_read_the_floor():
    world = sc.SimWorld(1600.0)
    world.quantize = False
    r = sc.measure_range(world.frame(1600.0))
    assert r.centre_mm == pytest.approx(1600.0, abs=15)
    assert r.centre_probe_mm < 1000.0


def test_match_interpolates_between_the_bracketing_legs():
    pts = [(-0.3, 50.0), (-0.4, 160.0), (-0.5, 280.0), (-0.6, 380.0), (-0.7, 490.0)]
    m = sc.match_reverse(pts, 435.0)
    assert m.throttle == pytest.approx(-0.65, abs=1e-9)


def test_match_takes_the_first_crossing():
    pts = [(-0.3, 50.0), (-0.4, 300.0), (-0.5, 280.0), (-0.6, 500.0)]
    assert sc.match_reverse(pts, 290.0).throttle == pytest.approx(-0.396, abs=1e-3)


def test_match_says_so_when_reverse_never_gets_there():
    m = sc.match_reverse([(-0.3, 10.0), (-1.0, 300.0)], 437.0)
    assert m.throttle is None and "never" in m.note


def test_match_at_the_lowest_level():
    m = sc.match_reverse([(-0.3, 500.0), (-0.4, 600.0)], 437.0)
    assert m.throttle == -0.3


def test_match_ignores_legs_that_did_not_fit():
    pts = [(-0.3, 100.0), (-0.5, None), (-0.7, 400.0)]
    assert sc.match_reverse(pts, 300.0).throttle == pytest.approx(-0.3 - 0.4 * 200 / 300)


def test_recommend_takes_the_match_when_lock_is_slower():
    r = sc.recommend(-0.6477, -0.5)
    assert r.reverse_speed == -0.65 and not r.conflict
    assert r.pulse_us == sc.esc_pulse(-0.65) == 1201


def test_recommend_takes_lock_plus_margin_and_calls_the_conflict():
    r = sc.recommend(-0.6477, -0.7)
    assert r.reverse_speed == -0.75 and r.conflict and "CONFLICT" in r.note


def test_recommend_when_lock_never_moves():
    r = sc.recommend(-0.6477, None, lock_tested=True)
    assert r.reverse_speed == -1.0 and r.conflict


def test_recommend_without_a_lock_test():
    r = sc.recommend(-0.6411, None, lock_tested=False)
    assert r.reverse_speed == -0.65 and not r.conflict


def test_recommend_rounds_up_not_to_nearest():
    assert sc.recommend(-0.6012, -0.3).reverse_speed == -0.61
    assert sc.recommend(-0.6000000001, -0.3).reverse_speed == -0.60   # float dust
    assert sc.recommend(-0.60, -0.3).reverse_speed == -0.60


def test_a_full_dry_run_writes_the_json(tmp_path):
    hw = sc.build_sim()
    assert _run_main(tmp_path, hw) == 0
    files = list(tmp_path.glob("speed_cal-*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["complete"] is True
    assert data["lock_reverse_threshold"] == -0.5
    assert data["lock_forward_threshold"] == 0.2
    assert data["recommended_reverse_speed"] == pytest.approx(-0.65, abs=0.03)
    assert data["conflict"] is False
    _assert_clean_exit(hw)


def test_a_dry_run_where_lock_needs_more_than_the_match(tmp_path):
    hw = sc.build_sim(esc=sc.EscModel(lock_rev_dead=0.75))
    assert _run_main(tmp_path, hw) == 0
    s = sc.main.last_summary
    assert s["lock_reverse_threshold"] == -0.8
    assert s["recommended_reverse_speed"] == -0.85 and s["conflict"] is True


@pytest.mark.parametrize("seed", range(4))
def test_a_buzzing_stall_at_lock_is_not_motion(seed):
    """A stalled ESC shakes the chassis; the rest measurement never sees it."""
    hw = sc.build_sim(1200.0, seed=seed)
    cal = sc.Calibrator(hw.vehicle, hw.pad, hw.camera, hw.clock, carry=hw.carry)
    for _ in range(30):
        cal._read(hw.clock.now())
        hw.clock.sleep(sc.LOOP_S)
    hw.world.gyro_buzz = 0.03
    hw.pad.hold()
    cal.calibrate_gyro()
    for level in (-0.3, -0.4):
        rec = cal.run_leg(sc.LegSpec("lock", sc.REVERSE, level, sc.LOCK_STEP_S,
                                     sc.LOCK_STEERING, lock=True))
        assert rec["moved"] is False, (seed, level, rec["peak_yaw_rad_s"])


def test_the_lock_test_does_not_call_rest_motion():
    hw, cal = primed(1200.0)
    hw.pad.hold()
    cal.calibrate_gyro()
    rec = cal.run_leg(sc.LegSpec("lock", sc.REVERSE, -0.3, sc.LOCK_STEP_S,
                                 sc.LOCK_STEERING, lock=True))
    assert rec["moved"] is False and rec["ended"] == "time"
    rec = cal.run_leg(sc.LegSpec("lock", sc.REVERSE, -0.6, sc.LOCK_STEP_S,
                                 sc.LOCK_STEERING, lock=True))
    assert rec["moved"] is True


def test_lock_steering_is_manual_full_stick():
    assert sc.LOCK_STEERING == (0.7, -0.7)


def test_the_dry_run_never_imports_pigpio(tmp_path):
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('sc', {_PATH!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        f"assert m.main(['--dry-run', '--log-dir', {str(tmp_path)!r}]) == 0\n"
        "assert 'pigpio' not in sys.modules, 'pigpio imported'\n"
        "assert 'depthai' not in sys.modules, 'depthai imported'\n"
    )
    out = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]


def test_yaw_says_nothing_until_its_window_is_full():
    hw, cal = primed(1200.0)
    t0 = 10.0
    gyro = [(t0 + 0.15 + 0.01 * i, 0.5) for i in range(30)]   # a big early spike
    early = t0 + sc.LOCK_DETECT_DELAY_S + 0.1
    assert cal._yaw(t0, early, gyro) == 0.0
    full = t0 + sc.LOCK_DETECT_DELAY_S + sc.LOCK_WINDOW_S
    assert cal._yaw(t0, full, gyro) == pytest.approx(0.5)


def test_one_window_over_the_threshold_is_not_motion():
    hw, cal = primed(1200.0)
    cal.gyro_bias, cal.gyro_threshold = 0.0, 0.02
    t0 = 0.0
    start = sc.LOCK_DETECT_DELAY_S + sc.LOCK_WINDOW_S
    steady = [(start - 0.3 + 0.01 * i, 0.05) for i in range(200)]
    # Over for less than LOCK_CONFIRM_S, then back under: never moved.
    short = int(sc.LOCK_CONFIRM_S / sc.LOOP_S) - 2
    for k in range(short):
        assert not cal._moved(t0, start + k * sc.LOOP_S, [], steady)
    assert not cal._moved(t0, start + short * sc.LOOP_S, [], [])
    # Over and staying over: moved, once LOCK_CONFIRM_S has passed.
    cal._yaw_over_since = None
    hits = [cal._moved(t0, start + k * sc.LOOP_S, [], steady) for k in range(20)]
    n = round(sc.LOCK_CONFIRM_S / sc.LOOP_S)
    assert hits.index(True) in (n, n + 1)      # float steps land either side


# -- the car adapters, against fakes ----------------------------------------------

class _FakeJs:
    def __init__(self, axes, buttons, heard=True):
        self.axes, self.buttons, self._h = axes, buttons, heard
        self.connected = heard
        self.reattached = 0

    def update_connection_status(self):
        pass

    def heard(self):
        return self._h

    def get_axis(self, name):
        return self.axes.get(name, 0.0)

    def get_button(self, name):
        return self.buttons.get(name, 0)

    def wait_for_connection(self, poll_interval=1.0):
        self.reattached += 1


class _FakePygame:
    class event:
        @staticmethod
        def pump():
            pass

    class joystick:
        @staticmethod
        def get_count():
            return 1


def _car_pad(js):
    pad = sc.CarPad.__new__(sc.CarPad)
    pad.pygame, pad.js = _FakePygame, js
    return pad


def test_the_car_pad_reads_rb_and_b():
    assert _car_pad(_FakeJs({}, {"RB": 1})).poll() == sc.PadState(True, True, False)
    assert _car_pad(_FakeJs({}, {"B": 1})).poll().abort


def test_the_car_pad_refuses_a_nan_axis_even_with_rb_down():
    s = _car_pad(_FakeJs({"LT": float("nan")}, {"RB": 1})).poll()
    assert not s.ok and not s.rb


def test_the_car_pad_refuses_a_lost_pad_and_tries_to_take_it_back():
    js = _FakeJs({}, {"RB": 1}, heard=False)
    s = _car_pad(js).poll()
    assert not s.ok and not s.rb and js.reattached == 1


def test_the_car_pad_turns_a_read_error_into_not_ok():
    js = _FakeJs({}, {"RB": 1})

    def broken(name):
        raise RuntimeError("stale handle")

    js.get_button = broken
    assert _car_pad(js).poll().ok is False


def test_the_car_vehicle_never_passes_a_nan_on():
    sent = []
    v = sc.CarVehicle.__new__(sc.CarVehicle)
    v.vc = type("VC", (), {"set_throttle": lambda self, x: sent.append(x)})()
    v.throttle(float("nan"))
    v.throttle(float("inf"))
    v.throttle(None)
    assert sent == [0.0, 0.0, 0.0]


def test_the_car_camera_drains_every_queue_and_times_by_capture():
    import datetime as dt

    class Q:
        def __init__(self, items):
            self.items, self.drained = items, 0

        def tryGetAll(self):
            self.drained += 1
            out, self.items = self.items, []
            return out

    class Pkt:
        def __init__(self, age):
            self.age = age

        def getFrame(self):
            return np.full((4, 4), 1000, dtype=np.uint16)

        def getTimestamp(self):
            return dt.timedelta(seconds=100.0 - self.age)

    class Gyro:
        x = 0.25

    class Imu:
        gyroscope = Gyro

    class ImuData:
        packets = [Imu()] * 3

    class Clock:
        @staticmethod
        def now():
            return dt.timedelta(seconds=100.0)

    class Dai:
        pass

    Dai.Clock = Clock
    cam = sc.CarCamera.__new__(sc.CarCamera)
    cam.dai = Dai
    disp = Q([object(), object()])

    class Oak:
        depth_queue = Q([Pkt(0.2), Pkt(0.1)])
        disparity_queue = disp
        imu_queue = Q([ImuData(), ImuData()])

    cam.oak = Oak
    depth, gyro = cam.poll(50.0)
    assert [round(t, 6) for t, _, _ in depth] == [49.8, 49.9]
    assert all(rx == 50.0 for _, rx, _ in depth)
    assert len(gyro) == 6 and gyro[0] == (50.0, 0.25)
    assert disp.drained == 1 and disp.items == []
