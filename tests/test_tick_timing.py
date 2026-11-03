"""What a decision costs and how old its inputs were: the instrumentation.

The policy's tick timed by stage (policy.TickTimer), the sensor ages at the
decision, the camera views keeping each depth frame's arrival and capture
times, /diagnostics' pass percentiles and pad-to-write latency, and the
/footprint columns that carry them. None of it may change what the car
does, and none of it may stop an older recording decoding.
"""

import json
import math

import numpy as np
import pytest

from crawler.autopilot import footprint as fp
from crawler.autopilot import policy as pl
from crawler.hardware import camera
from crawler.hardware import lidar as ld
from crawler.telemetry import telemetry as tm
import policy_adapters as pa

from conftest import FakeCamera, FakeController
from test_policy import FORWARD, StubModel, T0, clear, policy_driver
import test_policy_map as tpm
from test_policy_tools import footprint_session, row

NEW_FIELDS = ("policy_stage_ms", "policy_stage_cpu_ms", "tick_wall_ms", "tick_cpu_ms",
              "depth_age_ms", "depth_slot_ms", "lidar_age_ms", "crop_age_ms")
AGES = ("depth_age_ms", "depth_slot_ms", "lidar_age_ms", "crop_age_ms")


def stage(fields, key, name):
    return fields[key][pl.POLICY_STAGES.index(name)]


# --- the columns ---------------------------------------------------------------

def test_the_new_columns_are_in_every_row_and_the_schema_and_default_to_none():
    props = tm.JSON_SCHEMAS["footprint"]["schema"]["properties"]
    assert fp.STAGE_COUNT == len(pl.POLICY_STAGES)
    for key in NEW_FIELDS:
        assert key in fp.AUTOPILOT_FIELDS and key in props
        assert key in fp.NULLABLE_FIELDS
    assert fp.AUTOPILOT_FIELDS["policy_stage_ms"] == [None] * len(pl.POLICY_STAGES)
    assert all(fp.AUTOPILOT_FIELDS[k] is None for k in NEW_FIELDS[2:])
    assert props["tick_wall_ms"]["type"] == ["number", "null"]
    # A record row and a rules row say nothing about a policy tick.
    assert fp.autopilot_fields(None) == fp.AUTOPILOT_FIELDS
    d = pl.PolicyDriver(FakeController(), FakeCamera(), StubModel())
    assert set(NEW_FIELDS) <= set(d.policy_fields())


def test_a_policy_tick_times_its_stages_and_skips_the_ones_it_did_not_run():
    d = policy_driver(StubModel(FORWARD), clear())
    d.tick(T0)
    f = fp.autopilot_fields(d, T0)
    assert f["tick_wall_ms"] is not None and f["tick_wall_ms"] >= 0.0
    assert f["tick_cpu_ms"] is not None and f["tick_cpu_ms"] >= 0.0
    for name in ("goal", "frame", "imu", "scan", "encode", "infer", "post"):
        assert stage(f, "policy_stage_ms", name) is not None, name
        assert stage(f, "policy_stage_cpu_ms", name) is not None, name
    # No map, no lidar, no goal input on this model.
    for name in ("crop", "lidar", "path"):
        assert stage(f, "policy_stage_ms", name) is None, name
    assert sum(v for v in f["policy_stage_ms"] if v is not None) <= f["tick_wall_ms"] + 0.05
    # A fake camera, no lidar, no map: no ages, and None, not 0.
    for key in AGES:
        assert f[key] is None, key
    json.dumps(f)   # None is null, and the row still encodes


def test_a_tick_with_no_frame_has_no_frame_ages_and_no_later_stages():
    d = policy_driver(StubModel(FORWARD), None)
    d.tick(T0)
    f = d.policy_fields()
    assert f["tick_wall_ms"] is not None
    assert stage(f, "policy_stage_ms", "frame") is not None
    assert stage(f, "policy_stage_ms", "infer") is None
    assert f["depth_age_ms"] is None and f["depth_slot_ms"] is None


def test_last_ticks_ages_do_not_survive_into_a_tick_that_had_none():
    view = camera.StreamView("driver")
    d = pl.PolicyDriver(FakeController(), view, StubModel(FORWARD))
    view._put("depth", clear(), T0 - 0.02, T0 - 0.09)
    d.tick(T0)
    assert d.policy_fields()["depth_age_ms"] == pytest.approx(90.0, abs=0.01)
    d.tick(T0 + 0.1)   # no new frame
    assert d.policy_fields()["depth_age_ms"] is None


def test_an_exception_still_closes_the_timer():
    d = policy_driver(StubModel(FORWARD, raises=RuntimeError("boom")), clear())
    with pytest.raises(RuntimeError):
        d.tick(T0)
    f = d.policy_fields()
    assert f["tick_wall_ms"] is not None
    assert stage(f, "policy_stage_ms", "infer") is not None


def test_the_timing_changes_nothing_the_tick_decides(monkeypatch):
    """The same drive with the clocks the timer reads frozen and running."""
    def drive():
        d = policy_driver(StubModel(FORWARD, throttle=0.7, steer=0.3), clear())
        for k in range(5):
            d.tick(T0 + 0.1 * k)
        return d.controller.throttles, d.controller.steerings

    live = drive()
    monkeypatch.setattr(pl.time, "thread_time", lambda: 0.0)
    assert drive() == live


# --- the ages ------------------------------------------------------------------

def test_the_depth_ages_are_off_the_capture_stamp_and_the_arrival():
    view = camera.StreamView("driver")
    d = pl.PolicyDriver(FakeController(), view, StubModel(FORWARD))
    view._put("depth", clear(), T0 - 0.030, T0 - 0.110)   # arrived, captured
    d.tick(T0)
    f = fp.autopilot_fields(d, T0)
    assert f["depth_age_ms"] == pytest.approx(110.0, abs=0.01)
    assert f["depth_slot_ms"] == pytest.approx(30.0, abs=0.01)


def test_a_frame_with_no_stamp_has_no_age_but_still_drives():
    view = camera.StreamView("driver")
    d = pl.PolicyDriver(FakeController(), view, StubModel(FORWARD))
    view._put("depth", clear())      # the old two-argument put
    d.tick(T0)
    f = d.policy_fields()
    assert f["depth_age_ms"] is None and f["depth_slot_ms"] is None
    assert d.controller.throttles[-1] > 0


def test_the_lidar_age_is_off_the_newest_scan_stale_or_not():
    r = tpm.reader()
    d = pl.PolicyDriver(FakeController(), FakeCamera(depth=clear()), StubModel(FORWARD),
                        lidar=r)
    d.tick(T0 + 0.25)
    assert d.policy_fields()["lidar_age_ms"] == pytest.approx(250.0, abs=0.01)
    d.tick(T0 + ld.STALE_S + 1.0)   # stale: the model gets none, the age still says
    assert d.policy_fields()["lidar_age_ms"] == pytest.approx((ld.STALE_S + 1.0) * 1000, abs=0.01)


def test_a_lidar_reader_that_throws_is_no_age():
    class Throws:
        def latest(self, *a, **k):
            raise RuntimeError("serial went away")
    assert pl.lidar_scan_time(Throws(), T0) is None
    assert pl.lidar_scan_time(None, T0) is None


def test_the_crop_age_is_the_fed_crops_and_it_times_the_crop_stage():
    model, slam = tpm.MapStub(), tpm.FakeSlam()
    d = tpm.driver(model, slam)
    slam.at(T0 - 0.2)
    d.tick(T0)
    f = d.policy_fields()
    assert f["crop_age_ms"] == pytest.approx(200.0, abs=0.01)
    assert stage(f, "policy_stage_ms", "crop") is not None
    assert stage(f, "policy_stage_ms", "lidar") is not None


def test_age_ms_is_none_safe():
    assert pl.age_ms(1.0, None) is None
    assert pl.age_ms(None, 1.0) is None
    assert pl.age_ms(1.0, float("nan")) is None
    assert pl.age_ms(1.5, 1.0) == pytest.approx(500.0)


def test_autopilot_fields_keeps_none_and_drops_junk_to_none():
    class Driver:
        def policy_fields(self):
            return {"tick_wall_ms": "fast", "tick_cpu_ms": float("inf"),
                    "depth_age_ms": 12.5, "lidar_age_ms": None,
                    "policy_stage_ms": [1.0] * 3,                   # wrong width
                    "policy_stage_cpu_ms": [None, 2.0, "x"] + [None] * 7}
    f = fp.autopilot_fields(Driver())
    assert f["tick_wall_ms"] is None and f["tick_cpu_ms"] is None
    assert f["depth_age_ms"] == 12.5 and f["lidar_age_ms"] is None
    assert f["policy_stage_ms"] == [None] * fp.STAGE_COUNT
    assert f["policy_stage_cpu_ms"] == [None, 2.0] + [None] * 8


# --- the camera keeps every depth frame's times ------------------------------------

def test_a_view_hands_back_the_times_of_the_frame_it_took():
    view = camera.StreamView("x")
    a, b = clear(1000), clear(2000)
    view._put("depth", a, 1.0, 0.9)
    view._put("depth", b, 2.0, 1.9)   # overwrites: one slot, newest wins
    assert view.get_depth_frame() is b
    assert view.last_depth is b and view.last_depth_times == (2.0, 1.9)
    assert view.get_depth_frame() is None
    assert view.last_depth_times == (2.0, 1.9)   # kept after the take, as last_depth


def test_the_reader_stamps_every_depth_packet_with_or_without_listeners():
    class Dev:
        def __init__(self):
            self.n = 0

        def get_depth_frame(self):
            self.n += 1
            return clear() if self.n == 1 else None

        def capture_time(self, label):
            return 41.5 if label == "depth" else None

        def get_disparity_frame(self):
            return None

        get_rectified_left_frame = get_imu_sample = get_disparity_frame

    hub = camera.CameraHub()
    view = hub.view("driver")
    assert not hub.depth_listeners

    class OnePass:
        """The reader's stop event, set after its first pass."""
        def __init__(self):
            self.done = False

        def is_set(self):
            return self.done

        def wait(self, t):
            self.done = True
            return True

    hub._stop = OnePass()
    assert hub._read_loop(Dev()) is None
    assert view.get_depth_frame() is not None
    arrived, captured = view.last_depth_times
    assert captured == 41.5 and arrived is not None and math.isfinite(arrived)


def test_a_device_with_no_capture_time_still_hands_frames_over():
    class Dev:
        def get_depth_frame(self):
            return clear()

        def get_disparity_frame(self):
            return None

        get_rectified_left_frame = get_imu_sample = get_disparity_frame

    hub = camera.CameraHub()
    view = hub.view("driver")

    class OnePass:
        done = False

        def is_set(self):
            return self.done

        def wait(self, t):
            self.done = True

    hub._stop = OnePass()
    hub._read_loop(Dev())
    assert view.get_depth_frame() is not None
    assert view.last_depth_times[1] is None


# --- old recordings still decode --------------------------------------------------

def test_an_old_row_and_a_new_one_decode_to_the_same_command():
    old = row(1, mode="auto", autopilot="policy")
    new = dict(old, **{k: fp.AUTOPILOT_FIELDS[k] for k in NEW_FIELDS})
    new["tick_wall_ms"], new["depth_age_ms"] = 12.25, 95.0
    new["policy_stage_ms"] = [0.5] + [None] * (fp.STAGE_COUNT - 1)
    # Through JSON, as a recording carries it: null and all.
    new = json.loads(json.dumps(new))
    a = footprint_session([(0.0, old)]).commands
    b = footprint_session([(0.0, new)]).commands
    assert len(a) == len(b) == 1
    assert repr(a[0]) == repr(b[0])
    for key in NEW_FIELDS:
        assert key not in pa.FOOTPRINT_ROW.values()


# --- /diagnostics ----------------------------------------------------------------

def test_the_pass_window_gives_percentiles_and_none_when_empty():
    w = tm.PassWindow()
    f = w.fields()
    assert f["loop_passes"] == 0
    assert all(f[k] is None for k in f if k != "loop_passes")
    for ms in range(1, 101):
        w.add_pass(float(ms))
    w.add_pad(0.004)
    w.add_pad(0.006, 0.016)
    w.add_pad(0.002, 0.012)
    f = w.fields()
    assert f["loop_passes"] == 100
    assert f["loop_tick_p50_ms"] == 50.0 and f["loop_tick_p99_ms"] == 99.0
    assert f["pad_to_write_p50_ms"] == 4.0 and f["pad_to_write_p99_ms"] == 6.0
    assert f["pad_to_write_worst_ms"] == 16.0
    w.clear()
    assert w.fields()["loop_passes"] == 0


def test_the_pass_window_is_bounded():
    w = tm.PassWindow(cap=5)
    for ms in range(100):
        w.add_pass(ms)
        w.add_pad(0.001)
    assert w.fields()["loop_passes"] == 5


def test_the_diagnostics_schema_has_the_pass_fields():
    props = tm.JSON_SCHEMAS["diagnostics"]["schema"]["properties"]
    assert set(tm.PassWindow().fields()) <= set(props)


def test_the_control_loop_feeds_the_pass_window():
    import pathlib
    src = (pathlib.Path(__file__).parent.parent / "crawler" / "__main__.py").read_text()
    assert "passes.add_pass(tick_ms)" in src
    assert "**passes.fields()" in src
    assert "passes.add_pad(" in src
