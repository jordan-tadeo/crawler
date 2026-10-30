"""The hood camera's pan/tilt: the mount, the tracker, the detector's decode,
the runner and the calibration tool. No camera, servo or model needed: every
frame here is a few synthetic pixels and every model output is built by hand.
"""

import json
import math
import os
import pathlib
import time

import numpy as np
import pytest

from crawler.hardware import pantilt as pt
from crawler.hardware import vehicle_controller as vc
from crawler.vision import person
from crawler.vision import runner as ptr
from crawler.vision import track as tk
from crawler.vision.track import Detection

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---- fakes -------------------------------------------------------------------

class FakeWriter:
    freq_hz = 50.0

    def __init__(self):
        self.writes = []
        self.released = []

    def write_us(self, channel, us):
        pt.check_channel(channel)
        self.writes.append((channel, us))

    def release(self, channel):
        self.released.append(channel)


class FakeChannel:
    def __init__(self):
        self.duty_cycle = None


class FakePca:
    frequency = 50.0

    def __init__(self):
        self.channels = [FakeChannel() for _ in range(16)]


class FakeI2C:
    """A PCA9685 register file behind busio.I2C's calls."""

    def __init__(self, mode1=0x20, prescale=121):
        self.regs = bytearray(256)
        self.regs[0x00] = mode1
        self.regs[0xFE] = prescale
        self.writes = []
        self.locked = False

    def try_lock(self):
        if self.locked:
            return False
        self.locked = True
        return True

    def unlock(self):
        self.locked = False

    def writeto(self, addr, data):
        assert self.locked
        self.writes.append(bytes(data))
        reg = data[0]
        for i, b in enumerate(data[1:]):
            self.regs[reg + i] = b

    def writeto_then_readfrom(self, addr, out, buf):
        assert self.locked
        reg = out[0]
        for i in range(len(buf)):
            buf[i] = self.regs[reg + i]


class FakeLogger:
    def __init__(self):
        self.lines = []

    def log(self, level, subject, message):
        self.lines.append((level, subject, message))


class FakeTele:
    def __init__(self):
        self.offers = []

    def wants(self, key, now):
        return True

    def offer(self, key, payload, now):
        self.offers.append((key, payload))


class FakeGrabber:
    def __init__(self):
        self.frame = None
        self.up = True
        self.frames = 0
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def latest(self):
        return self.frame

    def close(self):
        self.closed = True


class FakeDetector:
    model_path = "models/person.onnx"
    input_shape = (256, 320)

    def __init__(self):
        self.state = "ready"
        self.error = ""
        self.busy = False
        self.submitted = []
        self.answers = []

    def start(self):
        return True

    def submit(self, seq, frame):
        if self.busy:
            return False
        self.busy = True
        self.submitted.append(seq)
        return True

    def poll(self):
        if self.answers and self.busy:
            self.busy = False
            return self.answers.pop(0)
        return None

    def close(self):
        self.state = "closed"


class FakeIdle:
    def __init__(self, value=None):
        self.value = value

    def sample(self):
        return self.value


def cfg_default():
    return pt.config_from_dict({})


# ---- the shared chip ---------------------------------------------------------

def test_steering_channels_match_the_vehicle_controller():
    assert pt.STEERING_CHANNELS == {vc.FRONT_STEERING_CHANNEL,
                                    vc.REAR_STEERING_CHANNEL}


def test_the_servos_are_on_10_and_11_everywhere():
    """Swept with the main battery in (it feeds the servo rail): channel 10
    turned the camera side to side and 11 up and down. 12 and 13 have nothing
    on them. The code default, the shipped config and the controller agree."""
    assert (pt.PAN_CHANNEL, pt.TILT_CHANNEL) == (10, 11)
    assert (vc.PAN_CHANNEL, vc.TILT_CHANNEL) == (10, 11)
    cfg = cfg_default()
    assert (cfg.mount.pan.channel, cfg.mount.tilt.channel) == (10, 11)
    shipped = pt.load_config(str(ROOT / "deploy" / "pantilt.json"))
    assert (shipped.mount.pan.channel, shipped.mount.tilt.channel) == (10, 11)


@pytest.mark.parametrize("ch", sorted(pt.STEERING_CHANNELS))
def test_a_mount_refuses_a_steering_channel(ch):
    with pytest.raises(pt.ConfigError):
        pt.check_channel(ch)
    with pytest.raises(pt.ConfigError):
        pt.config_from_dict({"mount": {"pan": {"channel": ch}}})
    with pytest.raises(pt.ConfigError):
        FakeWriter().write_us(ch, 1500)


def test_adafruit_channels_write_only_the_asked_channel():
    pca = FakePca()
    w = pt.AdafruitChannels(pca)
    w.write_us(12, 1500)
    # 1500 us at 50 Hz is 307 of 4096, in adafruit's 16-bit duty cycle.
    assert pca.channels[12].duty_cycle == 307 << 4
    assert [c.duty_cycle for i, c in enumerate(pca.channels) if i != 12] \
        == [None] * 15
    with pytest.raises(pt.ConfigError):
        w.write_us(15, 1500)
    assert pca.channels[15].duty_cycle is None


def test_raw_writer_touches_only_led_registers():
    i2c = FakeI2C()
    w = pt.RawPca9685(i2c)
    assert w.freq_hz == pytest.approx(50.03, abs=0.01)
    w.write_us(12, 1500)
    w.write_us(13, 1200)
    w.release(12)
    assert all(len(d) == 5 for d in i2c.writes)
    assert {d[0] for d in i2c.writes} == {0x06 + 4 * 12, 0x06 + 4 * 13}
    assert i2c.regs[0x00] == 0x20 and i2c.regs[0xFE] == 121
    assert w.read_us(12) is None
    assert w.read_us(13) == pytest.approx(1200, abs=5)
    with pytest.raises(pt.ConfigError):
        w.write_us(14, 1500)
    assert all(d[0] != 0x06 + 4 * 14 for d in i2c.writes)


def test_raw_writer_refuses_a_chip_without_auto_increment():
    with pytest.raises(RuntimeError):
        pt.RawPca9685(FakeI2C(mode1=0x00))


# ---- the mount ---------------------------------------------------------------

def test_axis_pulse_and_angle_round_trip_and_invert():
    a = pt.AxisConfig(channel=12, min_us=1000, max_us=2000, centre_us=1500,
                      deg_per_us=0.09)
    assert a.pulse_of(9.0) == pytest.approx(1600)
    assert a.angle_of(1600) == pytest.approx(9.0)
    a.invert = True
    assert a.pulse_of(9.0) == pytest.approx(1400)
    assert a.angle_of(1400) == pytest.approx(9.0)
    assert a.pulse_of(1000.0) == 1000  # clamped to the soft limit
    assert a.angle_limits() == pytest.approx((-45.0, 45.0))


def test_servo_mount_clamps_guards_nan_and_skips_repeats():
    cfg = cfg_default()
    w = FakeWriter()
    m = pt.ServoMount(cfg.mount, w)
    lo, hi = cfg.mount.pan.angle_limits()
    m.set(500.0, float("nan"))
    assert m.pan_deg == pytest.approx(hi)
    assert m.tilt_deg == 0.0
    assert w.writes == [(10, cfg.mount.pan.max_us), (11, 1500.0)]
    m.set(500.0, 0.0)
    assert len(w.writes) == 2
    assert {ch for ch, _ in w.writes} <= {10, 11}


def test_uvc_mount_drives_v4l2_left_positive():
    calls = []
    m = pt.UvcMount(cfg_default().mount, "/dev/video0",
                    run=lambda cmd, **kw: calls.append(cmd))
    m.set(5.0, -3.0)
    assert calls == [["v4l2-ctl", "-d", "/dev/video0", "-c",
                      "pan_absolute=-18000,tilt_absolute=-10800"]]
    m.set(50.0, 0.0)
    assert m.pan_deg == 16


def test_config_rejects_unknown_keys_and_wrong_types():
    with pytest.raises(pt.ConfigError):
        pt.config_from_dict({"mount": {"pan": {"chanel": 12}}})
    with pytest.raises(pt.ConfigError):
        pt.config_from_dict({"mount": {"pan": {"invert": 1}}})
    with pytest.raises(pt.ConfigError):
        pt.config_from_dict({"mount": {"pan": {"min_us": 300}}})
    with pytest.raises(pt.ConfigError):
        pt.config_from_dict({"mount": {"pan": {"channel": 11}}})  # = tilt


@pytest.mark.parametrize("name,kind", [("pantilt.json", "none"),
                                       ("pantilt-servo.json", "servo")])
def test_the_committed_configs_load(tmp_path, name, kind):
    # The fixed windshield camera is the default; the servo mount stays one
    # config away.
    cfg = pt.load_config(str(ROOT / "deploy" / name))
    assert cfg.mount.kind == kind
    out = tmp_path / "c.json"
    pt.save_config(cfg, str(out))
    assert pt.load_config(str(out)) == cfg


def test_the_fixed_camera_is_the_one_on_the_windshield():
    cfg = pt.load_config(str(ROOT / "deploy" / "pantilt.json"))
    assert "Global_Shutter_Camera" in cfg.camera.device
    assert cfg.mount.pitch_deg == pytest.approx(45.0)
    assert isinstance(pt.build_mount(cfg), pt.NullMount)
    # Square pixels: the vfov is the hfov's at 1280x720.
    tv = math.tan(math.radians(cfg.camera.hfov_deg) / 2) * cfg.camera.height / cfg.camera.width
    assert cfg.camera.vfov_deg == pytest.approx(2 * math.degrees(math.atan(tv)), abs=0.1)


def test_a_ray_off_a_pitched_camera():
    # No pitch: the pixel angle, as pixel_angles says.
    b, e = tk.ray(0.75, 0.5, 60, 46)
    assert b == pytest.approx(pixel_left(0.75))
    assert e == pytest.approx(0.0, abs=1e-9)
    # The middle of the frame is the axis, whatever the pitch and yaw.
    assert tk.ray(0.5, 0.5, 105, 72.5, yaw_deg=10, pitch_deg=45) == pytest.approx((10.0, 45.0))
    # Pitched 45 up, a point off to the side on the middle row is further
    # round than its pixel angle: atan(x / cos 45).
    x = (1 - 2 * 0.25) * math.tan(math.radians(52.5))
    b, _ = tk.ray(0.25, 0.5, 105, 72.5, pitch_deg=45)
    assert b == pytest.approx(math.degrees(math.atan(x / math.cos(math.radians(45)))))
    assert b > tk.pixel_angles(0.25, 0.5, 105, 72.5)[0] + 5
    # The bottom row of a camera pitched up by half its vfov is level.
    assert tk.ray(0.5, 1.0, 105, 72.5, pitch_deg=36.25)[1] == pytest.approx(0.0, abs=1e-9)


def test_the_tracker_reports_the_bearing_off_the_whole_ray():
    cfg = pt.config_from_dict({"camera": {"hfov_deg": 105.0, "vfov_deg": 72.5}})
    t = tk.Tracker(cfg.track, cfg.camera, ((0, 0), (0, 0)), 0.0, 45.0)
    t.on_detections([box(0.25, 0.5)], 0.0, 0.0, now=0.0)
    assert t.bearing_deg == pytest.approx(tk.ray(0.25, 0.5, 105, 72.5, 0.0, 45.0)[0])
    # Off to the side on the middle row is lower than the axis, not on it.
    assert t.elevation_deg == pytest.approx(tk.ray(0.25, 0.5, 105, 72.5, 0.0, 45.0)[1])
    assert t.elevation_deg < 45.0 - 5


# ---- the tracker -------------------------------------------------------------

def test_pixel_angles():
    assert pixel_left(0.5) == pytest.approx(0.0)
    assert pixel_left(1.0) == pytest.approx(-30.0)
    assert pixel_left(0.0) == pytest.approx(30.0)
    _, up = tk.pixel_angles(0.5, 0.0, 60, 46)
    assert up == pytest.approx(23.0)


def pixel_left(u):
    return tk.pixel_angles(u, 0.5, 60, 46)[0]


def box(cx, cy, w=0.1, h=0.3, score=0.9):
    return Detection(score, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_pick_biggest_then_sticks_to_the_followed_one():
    small, big = box(0.2, 0.5), box(0.8, 0.5, w=0.3)
    assert tk.pick([small, big]) is big
    assert tk.pick([small, big], prefer=(0.22, 0.5), gate=0.25) is small
    assert tk.pick([small, big], prefer=(0.5, 0.1), gate=0.1) is big
    assert tk.pick([]) is None


def make_tracker(**track):
    cfg = pt.config_from_dict({"track": track})
    return tk.Tracker(cfg.track, cfg.camera, ((-40, 40), (-30, 30)))


def test_a_person_right_of_centre_turns_the_mount_right():
    t = make_tracker(gain=1.0)
    t.on_detections([box(0.75, 0.5)], 0.0, 0.0, now=0.0)
    left = pixel_left(0.75)
    assert left < 0
    assert t.target_pan == pytest.approx(left)
    assert t.bearing_deg == pytest.approx(left)
    assert t.state == "tracking"


def test_a_late_answer_aims_from_the_angle_its_frame_was_taken_at():
    t = make_tracker(gain=1.0)
    t.pan = 25.0  # the mount has moved on since the frame was taken
    t.on_detections([box(0.75, 0.5)], 10.0, 0.0, now=0.0)
    assert t.target_pan == pytest.approx(10.0 + pixel_left(0.75))


def test_deadband_leaves_the_target_alone():
    t = make_tracker(deadband_deg=5.0)
    t.on_detections([box(0.52, 0.5)], 3.0, 0.0, now=0.0)
    assert t.target_pan == 0.0


def test_step_is_rate_limited_and_clamped():
    t = make_tracker(rate_deg_s=100.0, gain=1.0)
    t.target_pan = 80.0
    pan, _ = t.step(now=0.0, dt=0.05)
    assert pan == pytest.approx(5.0)
    for i in range(100):
        pan, _ = t.step(now=0.0, dt=0.05)
    assert pan == pytest.approx(40.0)


def test_lost_goes_home_after_lost_s():
    t = make_tracker(lost_s=2.0, on_lost="centre", gain=1.0)
    t.on_detections([box(0.9, 0.5)], 0.0, 0.0, now=0.0)
    t.step(1.0, 1.0)
    assert t.state == "tracking" and t.pan != 0.0
    t.step(2.5, 1.5)
    assert t.state == "lost" and t.bearing_deg is None
    for i in range(20):
        t.step(3.0 + i, 1.0)
    assert t.pan == 0.0


def test_scan_sweeps_inside_the_limits():
    t = make_tracker(lost_s=0.1, on_lost="scan", scan_deg_s=30.0)
    t.on_detections([box(0.5, 0.5)], 0.0, 0.0, now=0.0)
    seen = []
    for i in range(200):
        pan, _ = t.step(1.0 + 0.1 * i, 0.1)
        seen.append(pan)
    assert max(seen) <= 40 and min(seen) >= -40
    assert max(seen) > 35 and min(seen) < -35


def test_bearing_wraps():
    assert tk.bearing(170.0, 20.0, 0.0) == pytest.approx(-170.0)
    assert tk.bearing(90.0, 90.0, 0.0) == 180.0


# ---- the rate governor -------------------------------------------------------

def gov(**det):
    return tk.RateGovernor(pt.config_from_dict({"detector": det}).detector)


def test_governor_keeps_the_detector_under_its_duty():
    g = gov(base_hz=4.0, max_duty=0.5, min_hz=0.5)
    assert g.update(infer_s=0.05) == pytest.approx(0.25)
    assert g.update(infer_s=0.4) == pytest.approx(0.8)
    assert g.update(infer_s=5.0) == pytest.approx(2.0)


def test_governor_backs_off_when_the_pi_is_busy_and_comes_back():
    g = gov(base_hz=4.0, min_hz=0.5, low_idle=0.15, recover_idle=0.35)
    assert g.update(idle_frac=0.1) == pytest.approx(0.5)
    assert g.update(idle_frac=0.1) == pytest.approx(1.0)
    for _ in range(10):
        g.update(idle_frac=0.05)
    assert g.interval == pytest.approx(2.0)
    g.update(idle_frac=0.25)
    assert g.interval == pytest.approx(2.0)
    for _ in range(40):
        g.update(idle_frac=0.9)
    assert g.interval == pytest.approx(0.25)


def test_cpu_idle_reads_proc_stat(tmp_path):
    p = tmp_path / "stat"
    p.write_text("cpu  100 0 100 800 0 0 0 0 0 0\n")
    c = tk.CpuIdle(str(p))
    assert c.sample() is None
    p.write_text("cpu  150 0 150 900 0 0 0 0 0 0\n")
    assert c.sample() == pytest.approx(0.5)
    assert tk.CpuIdle(str(tmp_path / "none")).sample() is None


# ---- the detector ------------------------------------------------------------

def test_letterbox_pads_a_4_by_3_frame_into_the_input():
    frame = np.zeros((240, 320, 3), np.uint8)
    frame[:, :, 2] = 255  # red in BGR
    t, scale, px, py = person.letterbox(frame, 256, 320)
    assert t.shape == (1, 3, 256, 320) and t.dtype == np.float32
    assert (scale, px, py) == (1.0, 0, 8)
    assert t[0, 0, 100, 100] == pytest.approx(1.0)  # R first
    assert t[0, 0, 0, 0] == pytest.approx(114 / 255)


def yolo_output(rows, classes=80, n=50):
    """A 1 x (4+classes) x n output with the given (cx, cy, w, h, cls, score)
    in input pixels, everything else zero."""
    out = np.zeros((1, 4 + classes, n), np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(rows):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + cls, i] = score
    return out


def test_decode_undoes_the_letterbox_and_keeps_people_only():
    out = yolo_output([(160, 128, 40, 120, 0, 0.9),
                       (162, 129, 40, 118, 0, 0.8),   # the same person again
                       (60, 60, 20, 20, 2, 0.95),     # a car
                       (250, 100, 20, 60, 0, 0.2)])   # under conf
    people = person.decode(out, 320, 240, 1.0, 0, 8, conf=0.4, iou=0.45)
    assert len(people) == 1
    p = people[0]
    assert p.score == pytest.approx(0.9)
    assert (p.x0, p.x1) == pytest.approx((140 / 320, 180 / 320))
    assert (p.y0, p.y1) == pytest.approx(((128 - 60 - 8) / 240,
                                          (128 + 60 - 8) / 240))


def test_letterbox_fits_the_windshield_camera_into_a_16_by_9_input():
    frame = np.zeros((720, 1280, 3), np.uint8)
    t, scale, px, py = person.letterbox(frame, 192, 320)
    assert t.shape == (1, 3, 192, 320)
    assert (scale, px, py) == (0.25, 0, 6)
    out = yolo_output([(160, 96, 40, 120, 0, 0.9)], n=1260)
    p = person.decode(out, 1280, 720, scale, px, py)[0]
    assert (p.x0, p.x1) == pytest.approx((0.4375, 0.5625))
    assert (p.y0, p.y1) == pytest.approx(((36 - 6) / 180, (156 - 6) / 180))


def test_padding_is_what_a_4_by_3_input_wastes_on_a_16_by_9_camera():
    assert person.padding(1280, 720, 256, 320) == pytest.approx(0.296875)
    assert person.padding(1280, 720, 192, 320) == pytest.approx(0.0625)
    assert person.padding(320, 240, 256, 320) == pytest.approx(0.0625)


def test_the_default_rate_is_what_the_int8_model_can_do():
    g = gov()
    assert g.update(infer_s=0.1) == pytest.approx(0.2)   # 5 Hz at 97 ms
    assert g.update(infer_s=0.36) == pytest.approx(0.72)  # the float model
    for name in ("pantilt.json", "pantilt-servo.json"):
        cfg = json.loads((ROOT / "deploy" / name).read_text())
        assert cfg["detector"]["base_hz"] == 5.0, name


def test_a_model_shaped_for_another_camera_is_journalled():
    cfg = pt.config_from_dict({"camera": {"width": 1280, "height": 720}})

    def warnings_for(shape):
        d = FakeDetector()
        d.input_shape = shape
        log = FakeLogger()
        r = ptr.PanTiltRunner(cfg, pt.ServoMount(cfg.mount, FakeWriter()),
                              FakeGrabber(), d, logger=log, clock=Clock(),
                              idle=FakeIdle(0.5))
        r._watch_detector()
        return [m for lvl, _, m in log.lines if lvl == "warning"]
    assert any("30% padding" in m for m in warnings_for((256, 320)))
    assert warnings_for((192, 320)) == []


def test_decode_of_nothing():
    assert person.decode(yolo_output([]), 320, 240, 1.0, 0, 8) == []


class FakeSession:
    class _In:
        name = "images"
        shape = [1, 3, 256, 320]

    def get_inputs(self):
        return [self._In()]

    def run(self, _, feeds):
        assert feeds["images"].shape == (1, 3, 256, 320)
        return [yolo_output([(160, 128, 40, 120, 0, 0.9)])]


def make_fake_detector(path, conf, iou, threads):
    return person.OnnxPersonDetector(path, conf, iou, threads,
                                     session=FakeSession())


def wait_for(fn, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        got = fn()
        if got:
            return got
        time.sleep(0.01)
    return None


def test_detector_client_round_trip_on_a_thread():
    c = person.DetectorClient("unused.onnx", worker="thread",
                              make=make_fake_detector)
    assert c.start()
    wait_for(lambda: c.poll() or c.state == "ready")
    assert c.state == "ready"
    frame = np.zeros((240, 320, 3), np.uint8)
    assert c.submit(1, frame)
    assert not c.submit(2, frame)  # busy
    msg = wait_for(c.poll)
    assert msg[0] == "people" and msg[1] == 1
    assert len(msg[2]) == 1 and msg[2][0].score == pytest.approx(0.9)
    assert not c.busy
    c.close()


def test_detector_client_without_a_model_says_so(tmp_path):
    c = person.DetectorClient(str(tmp_path / "missing.onnx"))
    assert not c.start()
    assert c.state == "failed" and "no model" in c.error


# ---- the runner --------------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make_runner(**track):
    cfg = pt.config_from_dict({"track": dict({"gain": 1.0}, **track)})
    w = FakeWriter()
    mount = pt.ServoMount(cfg.mount, w)
    g, d, tele, log, clock = (FakeGrabber(), FakeDetector(), FakeTele(),
                              FakeLogger(), Clock())
    r = ptr.PanTiltRunner(cfg, mount, g, d, tele=tele, logger=log, clock=clock,
                          idle=FakeIdle(0.5))
    return r, w, g, d, tele, log, clock


def test_runner_centres_on_start_and_follows_a_person():
    r, w, g, d, tele, log, clock = make_runner()
    r.start()
    r._stop.set()
    assert w.writes[:2] == [(10, 1500.0), (11, 1500.0)]
    g.frame = (np.zeros((240, 320, 3), np.uint8), 1.0)
    r.tick()
    assert d.submitted == [1]
    d.answers.append(("people", 1, [box(0.9, 0.5)], 0.2))
    clock.t += 0.05
    r.tick()
    assert r.tracker.state == "tracking"
    assert r.tracker.target_pan == pytest.approx(pixel_left(0.9))
    for _ in range(40):
        clock.t += 0.05
        r.tick()
    assert r.mount.pan_deg == pytest.approx(pixel_left(0.9), abs=0.1)
    assert all(ch in (10, 11) for ch, _ in w.writes)
    assert any("Person at" in m for _, _, m in log.lines)
    r.close()
    assert g.closed and d.state == "closed"


def test_runner_does_not_resubmit_the_same_frame():
    r, w, g, d, tele, log, clock = make_runner()
    g.frame = (np.zeros((24, 32, 3), np.uint8), 1.0)
    r.tick()
    d.answers.append(("people", 1, [], 0.01))
    for _ in range(20):
        clock.t += 1.0
        r.tick()
    assert d.submitted == [1]


def test_runner_status_matches_the_telemetry_schema():
    from crawler.telemetry import telemetry as tm
    r, w, g, d, tele, log, clock = make_runner()
    g.frame = (np.zeros((240, 320, 3), np.uint8), 1.0)
    r.tick()
    d.answers.append(("people", 1, [box(0.4, 0.5)], 0.1))
    clock.t += 0.05
    r.tick()
    status = [p for k, p in tele.offers if k == "pantilt"][-1]
    schema = tm.JSON_SCHEMAS["pantilt"]["schema"]["properties"]
    assert set(status) == set(schema)
    json.dumps(status)
    assert status["person"] and status["boxes"]
    images = [p for k, p in tele.offers if k == "pantilt_image"]
    assert images and images[-1].shape == (240, 320, 3)


def test_runner_survives_a_detector_that_fails():
    r, w, g, d, tele, log, clock = make_runner()
    g.frame = (np.zeros((24, 32, 3), np.uint8), 1.0)
    r.tick()
    d.answers.append(("failed", 1, "boom"))
    clock.t += 0.05
    r.tick()
    d.state = "failed"
    d.error = "the detector process died"
    clock.t += 0.05
    r.tick()
    msgs = [m for _, _, m in log.lines]
    assert any("boom" in m for m in msgs)
    assert any("died" in m for m in msgs)


def test_telemetry_carries_the_pantilt_channels():
    from crawler.telemetry import telemetry as tm
    assert tm.JSON_TOPICS["pantilt"] == "/pantilt"
    assert tm.IMAGE_CHANNELS["pantilt_image"] == "/pantilt/image"
    assert {"pantilt", "pantilt_image"} <= tm.CAMERA_KEYS
    assert tm.IMAGE_FRAME_IDS["pantilt_image"] == "pantilt"


def test_nothing_in_the_pantilt_path_can_drive():
    files = [ROOT / "crawler" / "hardware" / "pantilt.py",
             ROOT / "tools" / "pantilt_cal.py",
             *sorted((ROOT / "crawler" / "vision").glob("*.py"))]
    for f in files:
        text = f.read_text()
        for word in ("import pigpio", "pigpio.", "set_throttle",
                     "set_steering", "return_neutral", "ESC_GPIO_PIN",
                     "import vehicle_controller"):
            assert word not in text, f"{f.name} has {word}"


# ---- main --------------------------------------------------------------------

def test_pantilt_flag():
    from crawler.__main__ import parse_args
    assert parse_args([]).pantilt is None
    assert parse_args(["--pantilt"]).pantilt == "deploy/pantilt.json"
    assert parse_args(["--pantilt", "x.json"]).pantilt == "x.json"


def test_a_bad_config_costs_only_the_pantilt(tmp_path):
    from crawler.__main__ import parse_args, start_pantilt
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"mount": {"pan": {"channel": 15}}}))
    log = FakeLogger()

    class Vecon:
        pca = FakePca()
    assert start_pantilt(parse_args(["--pantilt", str(bad)]), Vecon(), None,
                         log) is None
    assert any("No pan/tilt" in m and "steering" in m for _, _, m in log.lines)
    assert all(c.duty_cycle is None for c in Vecon.pca.channels)
    assert start_pantilt(parse_args([]), None, None, log) is None


# ---- the calibration tool ----------------------------------------------------

def test_calibration_commands():
    import pantilt_cal
    w = FakeWriter()
    cal = pantilt_cal.Cal(cfg_default(), w)
    cal.centre_both()
    assert w.writes == [(10, 1500.0), (11, 1500.0)]
    assert cal.apply("p+") == "pan 1520 us"
    assert cal.apply("pinv") == "pan.invert = True"
    assert cal.apply("p+") == "pan 1500 us"
    cal.apply("p 1200")
    cal.apply("pmin")
    cal.apply("p 1800")
    cal.apply("pmax")
    assert cal.apply("pspan 60") == "pan.deg_per_us = 0.1000"
    assert "outside" in cal.apply("t 2600")
    assert cal.apply("t 2600") and w.writes[-1][0] == 10
    assert cal.apply("q") is None
    assert w.writes[-2:] == [(10, 1500.0), (11, 1500.0)]


def test_calibration_saves_beside_the_model_not_over_the_tracked_config():
    """deploy/pantilt.json is tracked; saving over it on the car would dirty
    the checkout and block the next pull. The cal tool reads its own earlier
    save when there is one and always writes models/pantilt.json."""
    import pantilt_cal
    a = pantilt_cal.parse([])
    assert pantilt_cal.paths(a, exists=lambda p: False) == (
        "deploy/pantilt.json", "models/pantilt.json")
    assert pantilt_cal.paths(a, exists=lambda p: True) == (
        "models/pantilt.json", "models/pantilt.json")
    b = pantilt_cal.parse(["--config", "x.json", "--out", "y.json"])
    assert pantilt_cal.paths(b, exists=lambda p: True) == ("x.json", "y.json")
