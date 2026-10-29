"""Every reverse auto chooses is slow, it does not rock, and it stops short of
what the roof lidar sees behind it.

pigpio holds the last pulse width forever, so a rear stop that only stopped
commanding the reverse would keep reversing: every one here is checked for a
neutral on the wheels. The lidar under test is the real LidarReader with its
clock pinned, so freshness is decided by the code the car runs; no thread is
started. Every loop is a bounded range().
"""

import math

import numpy as np
import pytest

import crawler.__main__ as mn
from crawler.autopilot import driver as dv
from crawler.autopilot import footprint as fp
from crawler.autopilot import policy as pl
from crawler.hardware import lidar as ld

from conftest import FakeCamera, FakeController

T0 = 100.0

FORWARD = (5.0, 0.0, 0.0)
REVERSE = (0.0, 5.0, 0.0)

MOVING_IMU = {"accel": (0.0, 0.0, 0.0), "gyro": (0.33, 0.0, 0.0)}
STILL_IMU = {"accel": (0.0, 0.0, 0.0), "gyro": (0.001, 0.001, 0.001)}


class StubModel:
    contract = pl.CONTRACT_LEGACY_JET
    representation = pl.PROFILE
    uses_prev = False

    def __init__(self, logits=REVERSE, throttle=1.0, steer=0.0):
        self.out = {"logits": np.array([logits], dtype=np.float32),
                    "throttle": np.array([[throttle]], dtype=np.float32),
                    "steer": np.array([[steer]], dtype=np.float32)}

    def say(self, logits):
        self.out["logits"] = np.array([logits], dtype=np.float32)

    def run(self, x, prev):
        return self.out


def clear(mm=3000):
    return np.full((400, 640), mm, dtype=np.uint16)


def centre_band(mm):
    """A wall inside the floor across the middle of the fan, as
    test_policy's: the floor holds any forward into it."""
    depth = clear()
    depth[150:350, 220:420] = mm
    return depth


def blocked_depth():
    return np.full((400, 640), 300, dtype=np.uint16)


def rear_bins(mm, bearing=180.0):
    """A binned scan with one return, `mm` from the middle of the car."""
    bins = np.zeros(ld.BINS, dtype=np.uint16)
    bins[int(round(bearing / ld.BIN_DEG)) % ld.BINS] = mm
    return bins


def near():
    """Inside REAR_STOP_CLEAR_MM of the bumper, dead behind."""
    return rear_bins(int(dv.REAR_SELF_MM + dv.REAR_STOP_CLEAR_MM - 40))


def far():
    return rear_bins(int(dv.REAR_SELF_MM + dv.REAR_STOP_CLEAR_MM + 400))


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


def lidar(bins, age=0.0):
    """A real reader holding one scan `age` seconds old on its own clock."""
    clock = Clock()
    reader = ld.LidarReader(clock=clock)
    reader._scan = ld.Scan(t=clock.t - age, angles_deg=np.zeros(0),
                           distances_mm=np.zeros(0, dtype=np.uint16),
                           intensities=np.zeros(0, dtype=np.uint8),
                           speed_dps=3570.0, packets=40, bins=bins)
    return reader


def policy_driver(model, depth=None, imu=None, reader=None):
    return pl.PolicyDriver(FakeController(), FakeCamera(depth=depth, imu=imu),
                           model, lidar=reader)


def neutral(d):
    return d.controller.throttles[-1] == 0 and d.controller.steerings[-1] == (0, 0)


# --- every autonomous reverse is slow -----------------------------------------

def test_an_autonomous_reverse_is_capped_at_the_escape_speed_and_still_moves():
    f = dv.auto_reverse_throttle
    assert f(-1.0) == dv.ESCAPE_SPEED
    assert f(-0.6) == dv.ESCAPE_SPEED
    # Between the moving floor and the cap it is left alone...
    assert f(-0.42) == pytest.approx(-0.42)
    # ...and under the floor it is raised to the cap, so it moves.
    assert f(-0.2) == dv.ESCAPE_SPEED
    assert f(-1e-6) == dv.ESCAPE_SPEED
    assert f(0.0) == 0.0 and f(0.4) == 0.4
    for bad in (math.nan, math.inf, -math.inf, None, "x"):
        assert f(bad) == 0.0
    assert abs(dv.ESCAPE_SPEED) >= dv.REVERSE_MOVING_FLOOR


def test_every_reverse_the_policy_drives_is_the_escape_speed():
    """The commit, the model's own tick-by-tick reverse past it, at any
    magnitude the model gives, and the back-out a standoff runs."""
    for magnitude in (1.0, 0.6, 0.1):
        model = StubModel(REVERSE, throttle=magnitude, steer=0.3)
        d = policy_driver(model, clear())
        for k in range(20):
            d.tick(T0 + k * 0.1)
        reverses = [t for t in d.controller.throttles if t < 0]
        assert len(reverses) == 20, magnitude
        assert all(t == dv.ESCAPE_SPEED for t in reverses), (magnitude, reverses)
    # The standoff's back-out: never further back than ESCAPE_SPEED.
    model = StubModel(FORWARD, throttle=1.0)
    d = policy_driver(model, centre_band(150), imu=MOVING_IMU)
    for k in range(40):
        d.tick(T0 + k * 0.1)
    assert d._backout_reason == dv.BACKOUT_STANDOFF
    reverses = [t for t in d.controller.throttles if t < 0]
    assert reverses and min(reverses) >= dv.ESCAPE_SPEED


def run_policy(monkeypatch, model, passes, plan, reader=None):
    import test_takeover as tk
    return tk.run_auto(monkeypatch, passes, tk.Camera(clear()), plan=plan,
                       make_driver=lambda command, cam: pl.PolicyDriver(
                           command, cam, model, lidar=reader))


@pytest.mark.parametrize("bins", [far, near])
def test_my_reverse_under_takeover_is_neither_capped_nor_rear_stopped(monkeypatch, bins):
    import test_takeover as tk
    out = run_policy(monkeypatch, StubModel(REVERSE, throttle=1.0), 60,
                     tk.at(20, throttle=-1.0), reader=lidar(bins()))
    driver, s = out["driver"], out["snaps"]
    for snap in s[25:]:
        assert snap["throttle_by"] == mn.PAD
        assert snap["throttle"] == pytest.approx(driver.reverse_speed)
    assert driver.reverse_speed == -1.0
    # My reverse is mine: the rear stop never speaks under my trigger.
    assert not driver.rear_stop
    if bins is far:
        # Before my hand, the model's own reverse: slow.
        assert s[10]["throttle_by"] == mn.DRIVER
        assert s[10]["throttle"] == pytest.approx(dv.ESCAPE_SPEED)


# --- the refractory -----------------------------------------------------------

def commit_then_forward(d, model, forward_ticks):
    """A commit to the end, then the model forward; returns the next time."""
    t = T0
    for k in range(round(pl.REVERSE_COMMIT_S / 0.1)):
        d.tick(t)
        t = round(t + 0.1, 3)
    model.say(FORWARD)
    for k in range(forward_ticks):
        d.tick(t)
        t = round(t + 0.1, 3)
    return t


def test_a_fresh_reverse_straight_after_a_commit_is_driven_forward():
    model = StubModel(REVERSE, throttle=1.0, steer=0.4)
    d = policy_driver(model, clear())
    t = commit_then_forward(d, model, 3)
    model.say(REVERSE)
    d.tick(t)
    assert not d.reverse_commit
    assert d.reverse_refractory and d.policy_fields()["reverse_refractory"] is True
    assert d.controller.throttles[-1] > 0.0
    assert d.last_decision.action == pl.FORWARD
    assert d.policy_fields()["policy_action"] == pl.REVERSE
    # The model's own steering, not the commit's lock.
    assert d.controller.steerings[-1] == pytest.approx(
        dv.steering_pair(0.4 * pl.STEER_SCALE))
    # Past the window from the last reverse, a fresh one commits again.
    last_reverse = T0 + pl.REVERSE_COMMIT_S - 0.1
    k = 0
    for k in range(1, 40):
        t = round(t + 0.1, 3)
        d.tick(t)
        if d.reverse_commit:
            break
    assert d.reverse_commit
    assert t - last_reverse == pytest.approx(pl.REVERSE_REFRACTORY_S, abs=0.11)
    assert d.controller.throttles[-1] == dv.ESCAPE_SPEED


def test_the_refractory_lets_a_reverse_commit_when_the_floor_holds_forward():
    model = StubModel(REVERSE, throttle=1.0)
    d = policy_driver(model, clear())
    t = commit_then_forward(d, model, 2)
    # Now a wall inside the floor: forward is held, reverse is the way out.
    d.camera.depth = centre_band(150)
    d.tick(t)
    assert d.floor_veto.vetoes()
    model.say(REVERSE)
    d.tick(round(t + 0.1, 3))
    assert d.reverse_commit and not d.reverse_refractory
    assert d.controller.throttles[-1] == dv.ESCAPE_SPEED


# --- the rear stop: what counts -----------------------------------------------

def test_the_rear_clearance_is_from_the_bumper_and_drops_the_car_itself():
    assert dv.rear_clearance_mm(rear_bins(400)) == pytest.approx(400 - dv.REAR_SELF_MM)
    # Inside the bumper is the car: a self-hit, dropped, not a 0 clearance.
    assert dv.rear_clearance_mm(rear_bins(200)) is None
    both = rear_bins(200)
    both[ld.BINS // 2 + 1] = 500
    assert dv.rear_clearance_mm(both) == pytest.approx(500 - dv.REAR_SELF_MM)
    # The sector is narrow: a rear wheel's bearing (180 -+ 37) is outside it,
    # and so is the whole stock rear quadrant's edge.
    for bearing in (180 - 37, 180 + 37, 135, 225, 90, 0):
        assert dv.rear_clearance_mm(rear_bins(300, bearing)) is None, bearing
    for bearing in (160, 180, 200):
        assert dv.rear_clearance_mm(rear_bins(300, bearing)) is not None, bearing
    # Nothing at all, and nothing that is a scan.
    assert dv.rear_clearance_mm(np.zeros(ld.BINS)) is None
    for bad in (None, "x", [], np.full(ld.BINS, np.nan), np.zeros((2, 3))):
        assert dv.rear_clearance_mm(bad) is None
    assert not dv.rear_blocked(None)
    assert dv.rear_blocked(dv.REAR_STOP_CLEAR_MM - 1)
    assert not dv.rear_blocked(dv.REAR_STOP_CLEAR_MM)


# --- the rear stop on the policy's reverses -----------------------------------

def test_a_near_return_behind_stops_the_commit_with_neutral():
    reader = lidar(far())
    d = policy_driver(StubModel(REVERSE, throttle=1.0), clear(), reader=reader)
    d.tick(T0)
    assert d.reverse_commit and d.controller.throttles[-1] == dv.ESCAPE_SPEED
    assert not d.rear_stop
    reader._scan = reader._scan._replace(bins=near())
    d.tick(T0 + 0.1)
    assert neutral(d)
    assert not d.reverse_commit
    assert d.last_decision.reason == "rear stop"
    row = fp.autopilot_fields(d, T0 + 0.1)
    assert row["rear_stop"] is True
    assert row["rear_clearance_mm"] == pytest.approx(dv.REAR_STOP_CLEAR_MM - 40)


def test_a_fresh_reverse_into_a_near_return_is_never_commanded():
    d = policy_driver(StubModel(REVERSE, throttle=1.0), clear(), reader=lidar(near()))
    for k in range(10):
        d.tick(T0 + k * 0.1)
        assert d.controller.throttles[-1] >= 0.0, k
        assert all(t >= 0.0 for t in d.controller.throttles), k
    # And with the way forward held too, it sits at neutral.
    d = policy_driver(StubModel(REVERSE, throttle=1.0), centre_band(150),
                      reader=lidar(near()))
    for k in range(10):
        d.tick(T0 + k * 0.1)
    assert neutral(d) and d.rear_stop
    assert all(t >= 0.0 for t in d.controller.throttles)


@pytest.mark.parametrize("name, reader", [
    ("self-hit", lambda: lidar(rear_bins(200))),
    ("stale", lambda: lidar(near(), age=ld.STALE_S + 0.05)),
    ("nothing behind", lambda: lidar(np.zeros(ld.BINS, dtype=np.uint16))),
    ("no lidar", lambda: None),
])
def test_no_fresh_real_return_behind_reverses_as_before(name, reader):
    d = policy_driver(StubModel(REVERSE, throttle=1.0), clear(), reader=reader())
    for k in range(5):
        d.tick(T0 + k * 0.1)
        assert d.controller.throttles[-1] == dv.ESCAPE_SPEED, (name, k)
        assert not d.rear_stop
    assert d.rear_fields()["rear_clearance_mm"] == -1.0


def test_a_lidar_that_raises_is_no_lidar():
    class Broken:
        def latest(self, *a, **k):
            raise OSError("gone")

    d = policy_driver(StubModel(REVERSE, throttle=1.0), clear(), reader=Broken())
    d.tick(T0)
    assert d.controller.throttles[-1] == dv.ESCAPE_SPEED
    assert d.rear_clearance is None


def test_without_a_fresh_scan_the_gyro_bumper_still_cuts_a_commit(monkeypatch):
    monkeypatch.setattr(pl, "REVERSE_COMMIT_S", 3.0)
    d = policy_driver(StubModel(REVERSE, throttle=1.0), clear(), imu=STILL_IMU,
                      reader=lidar(near(), age=1.0))
    for k in range(20):
        d.tick(T0 + k * 0.1)
        if d.last_decision.reason != "model commit":
            break
    assert d.last_decision.reason == "rear bumper"
    assert neutral(d)


# --- the rear stop in the back-out --------------------------------------------

def back_out(d, ticks=40):
    """Drive a back-out against a wall until its reverse ends; the decision
    it ended on, or None."""
    d.camera.depth = blocked_depth()
    d.begin_recover(T0, dv.BACKOUT_STUCK)
    if d._phase != dv._PHASE_REVERSE:
        return 0
    for i in range(1, ticks):
        d.recovering(round(T0 + i * 0.1, 3))
        if d._phase != dv._PHASE_REVERSE:
            return i
    return None


def test_a_back_out_ends_rear_when_something_comes_up_behind():
    reader = lidar(far())
    d = dv.Driver(FakeController(), FakeCamera(imu=MOVING_IMU), lidar=reader)
    d.camera.depth = blocked_depth()
    d.begin_recover(T0, dv.BACKOUT_STUCK)
    for i in range(1, 4):
        d.recovering(round(T0 + i * 0.1, 3))
    assert d._phase == dv._PHASE_REVERSE and d.controller.throttles[-1] < 0
    reader._scan = reader._scan._replace(bins=near())
    d.recovering(T0 + 0.4)
    assert d._backout_exit == dv.EXIT_REAR
    assert d.controller.throttles[-1] == 0
    assert d.rear_fields()["rear_stop"] is True
    assert d._phase == dv._PHASE_SETTLE


def test_the_rear_stop_is_asked_before_the_gyro_bumper():
    """Still IMU: the bumper fires on decision `fires`. A near return shows up
    on that same decision, so both would end it, and the lidar's is the name
    on it."""
    fires = dv.Driver(FakeController(), FakeCamera()).imu_window_size \
        + dv.REVERSE_STALL_SAMPLES - 1
    d = dv.Driver(FakeController(), FakeCamera(imu=STILL_IMU), lidar=lidar(far()))
    d.camera.depth = blocked_depth()
    d.begin_recover(T0, dv.BACKOUT_STUCK)
    for i in range(1, fires):
        d.recovering(round(T0 + i * 0.1, 3))
        assert d._phase == dv._PHASE_REVERSE, i
    d.lidar._scan = d.lidar._scan._replace(bins=near())
    d.recovering(round(T0 + fires * 0.1, 3))
    assert d._backout_exit == dv.EXIT_REAR
    assert d.controller.throttles[-1] == 0


@pytest.mark.parametrize("reader", [None, lambda: lidar(near(), age=1.0),
                                    lambda: lidar(rear_bins(150))])
def test_with_no_fresh_real_return_the_gyro_bumper_is_the_stop(reader):
    d = dv.Driver(FakeController(), FakeCamera(imu=STILL_IMU),
                  lidar=reader() if reader else None)
    ended = back_out(d)
    assert d._backout_exit == dv.EXIT_BUMPER
    assert ended == d.imu_window_size + dv.REVERSE_STALL_SAMPLES - 1
    assert d.controller.throttles[-1] == 0


def test_a_back_out_with_something_already_behind_never_reverses():
    d = dv.Driver(FakeController(), FakeCamera(imu=MOVING_IMU), lidar=lidar(near()))
    ended = back_out(d)
    assert ended == 0
    assert d._backout_exit == dv.EXIT_REAR
    assert all(t >= 0.0 for t in d.controller.throttles)
    assert d.controller.throttles[-1] == 0
    # The settle runs and the wheels come back, as after any exit.
    for i in range(1, 20):
        if not d.recovering(round(T0 + i * 0.1, 3)):
            break
    assert d._phase is None and d.controller.throttles[-1] == 0


def test_the_rules_obstacle_back_out_reads_the_lidar_too():
    d = dv.Driver(FakeController(), FakeCamera(depth=blocked_depth(), imu=MOVING_IMU),
                  lidar=lidar(near()))
    d.tick(T0)
    assert d._backout_exit == dv.EXIT_REAR
    assert all(t >= 0.0 for t in d.controller.throttles)
    assert fp.autopilot_fields(d, T0)["rear_stop"] is True


def test_abort_mid_back_out_with_a_lidar_is_still_one_tick():
    d = dv.Driver(FakeController(), FakeCamera(imu=MOVING_IMU), lidar=lidar(far()))
    d.camera.depth = blocked_depth()
    d.begin_recover(T0)
    d.recovering(T0 + 0.1)
    d.abort()
    assert d.controller.throttles[-1] == 0 and d._phase is None


# --- telemetry and wiring -----------------------------------------------------

def test_rows_carry_the_rear_clearance_and_a_rules_row_without_a_lidar_is_unchanged():
    assert fp.AUTOPILOT_FIELDS["rear_clearance_mm"] == -1.0
    assert fp.AUTOPILOT_FIELDS["rear_stop"] is False
    d = dv.Driver(FakeController(), FakeCamera(depth=clear()), lidar=lidar(rear_bins(600)))
    d.tick(T0)
    row = fp.autopilot_fields(d, T0)
    assert row["rear_clearance_mm"] == pytest.approx(600 - dv.REAR_SELF_MM)
    assert row["rear_stop"] is False


class ReadyHub:
    from crawler.hardware import camera as _camera
    state = _camera.READY
    error = ""

    def view(self, name):
        return FakeCamera()


class Log:
    def log(self, *a):
        pass


def test_the_autopilot_hands_its_lidar_to_the_driver():
    reader = lidar(far())
    pilot = mn.AutoPilot(FakeController(), Log(), hub=ReadyHub(), lidar=reader)
    assert pilot.resume() is True
    assert pilot.driver.lidar is reader
    pilot = mn.AutoPilot(FakeController(), Log(), hub=ReadyHub())
    assert pilot.resume() is True
    assert pilot.driver.lidar is None
