"""Record mode: hold Y in manual, drive at auto's speeds, write it all down.

Four things are held here.

The button. Y held for a second in manual enters record, at the threshold and
while it is still down; the release that follows is not a tap; any fresh press
in record leaves. Y does nothing in auto. Driven off a fake clock, because the
whole contract is about time.

The handovers. START from record ends the session and goes to auto; a lost pad
or a stale handle ends it and holds neutral, the way manual always has.

The throttle. Scaled, not clipped: full RT is auto's forward speed and full LT
its reverse speed, read off a Driver and never off a literal.

The one that matters. pigpio latches the last pulse width forever, so the
autopilot's features being computed in record must never be able to reach the
wheels. Every command in a record run is the pad's, scaled, even on frames the
rules would reverse or swerve on -- and footprint.py is never handed the
controller at all.
"""

import pathlib

import numpy as np
import pytest

import camera
import driver as dv
import footprint as fp
import main as mn
import policy_input
import recording
import telemetry as tm

from test_main import FakeLogger, FakeMarkPad, FakePad, FakeVecon, TickLimit


# --- fakes ------------------------------------------------------------------


class FakeClock:
    """time.monotonic, advanced by the loop's own pump."""

    def __init__(self, step=0.01):
        self.t = 0.0
        self.step = step

    def __call__(self):
        return self.t


class SessionTele:
    """A Telemetry as far as record mode can see it."""

    def __init__(self, running=True,
                 channels=("footprint", "depth_raw", "depth_small")):
        self.running = running
        self.channels = set(channels)
        self.offers = []
        self.session = None
        self.begins = 0
        self.ends = 0

    def wants(self, key, now):
        return key in self.channels

    def offer(self, key, payload, now):
        self.offers.append((key, payload, now))

    def recording_status(self):
        return {}

    def begin_session(self, recorder):
        if not self.running:
            return False
        self.begins += 1
        self.session = recorder
        return True

    def end_session(self):
        self.ends += 1
        self.session = None

    def of(self, key):
        return [(p, now) for k, p, now in self.offers if k == key]


class FakeHub:
    def __init__(self, state=camera.OFF):
        self.state = state
        self.starts = 0
        self.releases = 0
        self.open_seconds = 2.5

    def start(self):
        if self.state in (camera.RELEASING, camera.CLOSED):
            return False
        self.starts += 1
        self.state = camera.STARTING
        return True

    def release(self):
        self.releases += 1
        self.state = camera.OFF
        return True


class FakeAutopilot:
    unavailable = False

    def __init__(self, hub=None, driver=None):
        self.hub = hub
        self.driver = driver
        self.resumes = 0
        self.pauses = 0
        self.steps = 0

    def resume(self):
        self.resumes += 1
        return True

    def pause(self):
        self.pauses += 1

    def step(self, now, takeover=None):
        self.steps += 1
        return True

    def stop(self):
        pass

    def camera_state(self):
        return "off"

    def camera_error(self):
        return ""


class BuzzPad(FakeMarkPad):
    def __init__(self):
        self.buzzes = []

    def buzz(self, pulses, now):
        self.buzzes.append(pulses)


class FrameView:
    """A StreamView that always has `frame` waiting."""

    def __init__(self, frame):
        self.frame = frame
        self.takes = 0

    def get_depth_frame(self):
        self.takes += 1
        return self.frame


class ImuView:
    """The tap's own view in auto: IMU yes, a depth take never."""

    def __init__(self, sample=None):
        self.sample = sample

    def get_imu_sample(self):
        sample, self.sample = self.sample, None
        return sample

    def get_depth_frame(self):
        raise AssertionError("auto took a frame from the tap's own view")


class Trap:
    """Anything at all called on this fails the test."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        raise AssertionError(f"{self._name}.{attr} was touched in record")


class TrapDriver:
    """A driver left over from a lap in auto. Its controller is a trap, and
    so is ticking it -- record must read nothing off it and call nothing."""

    forward_speed = 0.2
    reverse_speed = -0.6

    def __init__(self):
        self.controller = Trap("driver.controller")

    def tick(self, now=None):
        raise AssertionError("the driver was ticked in record")

    def begin_recover(self, now):
        raise AssertionError("the driver recovered in record")


def far_frame(mm=3000):
    return np.full((400, 640), mm, dtype=np.uint16)


def wall_frame(mm=200):
    """Every probe near and under the floor: read_the_gradient reverses."""
    return np.full((400, 640), mm, dtype=np.uint16)


def graze_frame(mm=380):
    """The left edge probe near, the rest open: the rules swerve right."""
    frame = far_frame()
    edges = dv.footprint_geometry(frame.shape).col_edges
    frame[:, :edges[1]] = mm
    return frame


# A binary fraction, so a second of passes adds up to exactly 1.0 and the
# threshold test is about the threshold rather than about float rounding.
STEP = 1.0 / 64


def run(pad, script, mode=mn.MANUAL, tele=None, hub=None, driver=None,
        tap=None, monkeypatch=None, step=STEP, markpad=None):
    """control_loop over a scripted pad and a fake clock, one step a pass."""
    import pygame

    clock = FakeClock(step)
    pad.script = list(script)

    def pump():
        pad.step()
        clock.t += step

    monkeypatch.setattr(pygame.event, "pump", pump)
    monkeypatch.setattr(mn, "LOOP_INTERVAL", 0)

    log = FakeLogger()
    tele = tele or SessionTele()
    hub = hub or FakeHub()
    autopilot = FakeAutopilot(hub=hub, driver=driver)
    vecon = FakeVecon()
    modes = mn.ModeSwitch(mode, log)
    markpad = markpad or BuzzPad()
    session = mn.RecordSession(tele, hub, log, make_recorder=lambda: "rec",
                               markpad=markpad)
    tap = tap or fp.FootprintTap(tele=tele)

    mn.control_loop(pad, vecon, autopilot, modes, log, TickLimit(len(script)),
                    tele, markpad, mn.SteerMode(log), session=session,
                    tap=tap, clock=clock)
    return {"vecon": vecon, "modes": modes, "tele": tele, "hub": hub,
            "session": session, "autopilot": autopilot, "log": log,
            "markpad": markpad, "tap": tap}


def hold_y(seconds, step=STEP):
    """Y down for `seconds` of loop time: the first frame stamps the press,
    the last is `seconds` after it."""
    return [(mn.RECORD_BUTTON,)] * (int(round(seconds / step)) + 1)


# The pass on which a one-second hold enters record.
ENTRY = len(hold_y(1.0))


# --- the button, on its own -------------------------------------------------


class YPad:
    def __init__(self):
        self.down = False
        self.raises = False

    def get_button(self, name):
        if self.raises:
            raise RuntimeError("stale handle")
        return self.down and name == mn.RECORD_BUTTON


def test_y_is_the_record_button_and_nothing_else_reads_it():
    import Joystick as js

    assert mn.RECORD_BUTTON == "Y"
    assert js.XBOX_BUTTONS["Y"] == 3
    others = {mn.MODE_TOGGLE_BUTTON, mn.STEER_SPLIT_BUTTON}
    import marks as mk
    others |= set(mk.BUTTON_KINDS)
    assert "Y" not in others


def test_a_hold_enters_at_one_second_and_not_at_point_nine():
    button, pad = mn.RecordButton(), YPad()
    pad.down = True
    assert button.poll(pad, 0.0, mn.MANUAL) is None
    assert button.poll(pad, 0.9, mn.MANUAL) is None
    assert button.poll(pad, 1.0, mn.MANUAL) == mn.ENTER_RECORD


def test_the_hold_fires_while_y_is_still_down_and_only_once():
    button, pad = mn.RecordButton(), YPad()
    pad.down = True
    button.poll(pad, 0.0, mn.MANUAL)
    assert button.poll(pad, 1.0, mn.MANUAL) == mn.ENTER_RECORD
    # Still held and record refused (no telemetry): it must not fire again
    # every tick, which would be a begin attempt and a log line at 100 Hz.
    assert button.poll(pad, 1.2, mn.MANUAL) is None
    # Still held, now in record: neither more of the hold nor the record mode
    # it just entered may read it as a press.
    assert button.poll(pad, 1.5, mn.RECORD) is None
    assert button.poll(pad, 3.0, mn.RECORD) is None


def test_the_release_after_the_hold_is_not_a_tap():
    button, pad = mn.RecordButton(), YPad()
    pad.down = True
    button.poll(pad, 0.0, mn.MANUAL)
    button.poll(pad, 1.0, mn.MANUAL)
    pad.down = False
    assert button.poll(pad, 1.1, mn.RECORD) is None
    assert button.poll(pad, 5.0, mn.RECORD) is None


def test_a_fresh_tap_in_record_exits():
    button, pad = mn.RecordButton(), YPad()
    pad.down = True
    button.poll(pad, 0.0, mn.MANUAL)
    button.poll(pad, 1.0, mn.MANUAL)
    pad.down = False
    button.poll(pad, 1.1, mn.RECORD)
    pad.down = True
    assert button.poll(pad, 2.0, mn.RECORD) == mn.EXIT_RECORD


def test_a_short_press_in_manual_does_nothing():
    button, pad = mn.RecordButton(), YPad()
    pad.down = True
    button.poll(pad, 0.0, mn.MANUAL)
    button.poll(pad, 0.5, mn.MANUAL)
    pad.down = False
    assert button.poll(pad, 0.6, mn.MANUAL) is None
    # And a second short press does not add up with the first.
    pad.down = True
    assert button.poll(pad, 0.7, mn.MANUAL) is None
    assert button.poll(pad, 1.6, mn.MANUAL) is None


def test_y_does_nothing_in_auto_and_a_hold_from_auto_does_not_carry_over():
    button, pad = mn.RecordButton(), YPad()
    pad.down = True
    for t in (0.0, 0.5, 1.0, 2.0):
        assert button.poll(pad, t, mn.AUTO) is None
    # START to manual with Y still down: the press began in auto, it is spent.
    assert button.poll(pad, 2.5, mn.MANUAL) is None
    assert button.poll(pad, 4.0, mn.MANUAL) is None


def test_a_stale_handle_reads_as_released_and_forgets_the_hold():
    button, pad = mn.RecordButton(), YPad()
    pad.down = True
    button.poll(pad, 0.0, mn.MANUAL)
    pad.raises = True
    assert button.poll(pad, 0.8, mn.MANUAL) is None
    pad.raises = False
    # The hold starts again from here, not from 0.0.
    assert button.poll(pad, 1.2, mn.MANUAL) is None
    assert button.poll(pad, 2.2, mn.MANUAL) == mn.ENTER_RECORD


# --- the button, through the loop ------------------------------------------


def test_holding_y_in_the_loop_enters_record_at_one_second(monkeypatch):
    pad = FakePad()
    out = run(pad, hold_y(1.0) + [()], monkeypatch=monkeypatch)
    assert out["modes"].mode == mn.RECORD
    assert out["tele"].begins == 1
    assert out["hub"].starts == 1


def test_holding_y_for_point_nine_seconds_does_not(monkeypatch):
    pad = FakePad()
    out = run(pad, hold_y(0.9) + [()], monkeypatch=monkeypatch)
    assert out["modes"].mode == mn.MANUAL
    assert out["tele"].begins == 0
    assert out["hub"].starts == 0


def test_the_release_after_entering_does_not_leave_and_a_tap_does(monkeypatch):
    pad = FakePad()
    script = hold_y(1.2) + [()] * 20
    out = run(pad, script, monkeypatch=monkeypatch)
    assert out["modes"].mode == mn.RECORD
    assert out["hub"].releases == 0
    assert not any("Switched to manual" in line[2] for line in out["log"].lines)

    pad = FakePad()
    out = run(pad, script + [(mn.RECORD_BUTTON,), ()], monkeypatch=monkeypatch)
    assert out["modes"].mode == mn.MANUAL
    assert out["tele"].ends == 1
    # The session opened the camera, so leaving to manual gives it back.
    assert out["hub"].releases == 1


def test_y_in_auto_does_nothing_in_the_loop(monkeypatch):
    pad = FakePad()
    out = run(pad, hold_y(2.0), mode=mn.AUTO, monkeypatch=monkeypatch)
    assert out["modes"].mode == mn.AUTO
    assert out["tele"].begins == 0


def test_start_from_record_goes_to_auto_and_closes_the_recording(monkeypatch):
    pad = FakePad()
    tele = SessionTele()
    open_at_resume = []
    real_resume = FakeAutopilot.resume

    def resume(self):
        open_at_resume.append(tele.session is not None)
        return real_resume(self)

    # The loop's own finally ends any session too, so the proof that START
    # ended it is that it was already over when auto took the wheels.
    monkeypatch.setattr(FakeAutopilot, "resume", resume)
    script = hold_y(1.0) + [()] * 5 + [(mn.MODE_TOGGLE_BUTTON,), ()]
    out = run(pad, script, tele=tele, monkeypatch=monkeypatch)
    assert out["modes"].mode == mn.AUTO
    assert out["autopilot"].resumes == 1
    assert open_at_resume == [False]
    assert out["tele"].ends == 1
    assert out["tele"].session is None
    assert out["session"].active is False
    # Auto is about to want the camera, so it is not given back.
    assert out["hub"].releases == 0


def test_a_lost_pad_ends_the_recording_and_holds_neutral(monkeypatch):
    class DroppingPad(FakePad):
        def is_connected(self):
            return self.frames < ENTRY + 5

    pad = DroppingPad()
    script = hold_y(1.0) + [()] * 20
    pad.throttle = 1.0
    out = run(pad, script, monkeypatch=monkeypatch)
    assert out["modes"].mode == mn.MANUAL
    assert out["tele"].ends == 1
    assert out["session"].active is False
    assert out["vecon"].neutrals >= 1
    assert out["vecon"].last_throttle == 0.0


def test_a_stale_handle_in_record_ends_it_on_the_manual_safety_path(monkeypatch):
    class StalingPad(FakePad):
        def read_throttle(self):
            if self.frames > ENTRY + 5:
                raise RuntimeError("stale handle")
            return super().read_throttle()

    pad = StalingPad()
    pad.throttle = 1.0
    out = run(pad, hold_y(1.0) + [()] * 10, monkeypatch=monkeypatch)
    assert out["modes"].mode == mn.MANUAL
    assert out["tele"].ends == 1
    assert out["vecon"].throttles[-1] == 0.0


def test_entering_buzzes_twice_and_leaving_once(monkeypatch):
    pad = FakePad()
    script = hold_y(1.0) + [(), (mn.RECORD_BUTTON,), ()]
    out = run(pad, script, monkeypatch=monkeypatch)
    assert out["markpad"].buzzes == [2, 1]


def test_with_telemetry_off_record_refuses_and_stays_manual(monkeypatch):
    pad = FakePad()
    out = run(pad, hold_y(1.0) + [()], tele=SessionTele(running=False),
              monkeypatch=monkeypatch)
    assert out["modes"].mode == mn.MANUAL
    assert out["hub"].starts == 0


# --- the throttle -----------------------------------------------------------


def test_full_rt_is_auto_forward_and_full_lt_is_auto_reverse():
    limits = mn.auto_speed_limits()
    assert mn.scale_throttle(1.0, limits) == pytest.approx(0.2)
    assert mn.scale_throttle(-1.0, limits) == pytest.approx(-0.6)
    assert mn.scale_throttle(0.0, limits) == 0.0


def test_the_trigger_is_scaled_not_clipped():
    limits = mn.auto_speed_limits()
    assert mn.scale_throttle(0.5, limits) == pytest.approx(0.1)
    assert mn.scale_throttle(-0.5, limits) == pytest.approx(-0.3)


def test_the_limits_come_from_the_driver_not_a_literal():
    class Tuned:
        forward_speed = 0.35
        reverse_speed = -0.45

    limits = mn.auto_speed_limits(Tuned())
    assert mn.scale_throttle(1.0, limits) == pytest.approx(0.35)
    assert mn.scale_throttle(-1.0, limits) == pytest.approx(-0.45)
    # And the defaults are whatever a fresh Driver says, so a retune in
    # driver.py is followed with nothing changed here.
    fresh = dv.Driver(None, None)
    assert mn.auto_speed_limits() == (fresh.forward_speed, fresh.reverse_speed)


def test_with_no_live_driver_a_retuned_driver_py_is_followed(monkeypatch):
    class Retuned:
        def __init__(self, controller, camera):
            self.forward_speed = 0.31
            self.reverse_speed = -0.42

    monkeypatch.setattr(dv, "Driver", Retuned)
    assert mn.auto_speed_limits() == (0.31, -0.42)


def test_unusable_limits_are_a_car_that_does_not_move():
    class Broken:
        forward_speed = float("nan")
        reverse_speed = -0.6

    assert mn.auto_speed_limits(Broken()) == (0.0, 0.0)
    assert mn.auto_speed_limits(object()) == (0.0, 0.0)


def test_a_nan_trigger_scales_to_zero_not_to_a_limit():
    limits = mn.auto_speed_limits()
    assert mn.scale_throttle(float("nan"), limits) == 0.0
    assert mn.scale_throttle(float("inf"), limits) == 0.0


def test_record_drives_the_esc_at_the_scaled_throttle(monkeypatch):
    pad = FakePad()
    pad.throttle = 1.0
    out = run(pad, hold_y(1.0) + [()] * 3, monkeypatch=monkeypatch)
    assert out["vecon"].throttles[0] == pytest.approx(1.0)   # still manual
    assert out["vecon"].throttles[-1] == pytest.approx(0.2)  # record

    pad = FakePad()
    pad.throttle = -1.0
    out = run(pad, hold_y(1.0) + [()] * 3, monkeypatch=monkeypatch)
    assert out["vecon"].throttles[-1] == pytest.approx(-0.6)


def test_steering_in_record_is_manual_steering(monkeypatch):
    pad = FakePad()
    pad.axes["LEFT_X"] = 0.5
    out = run(pad, hold_y(1.0) + [()] * 3, monkeypatch=monkeypatch)
    limited = 0.5 * mn.STEERING_LIMIT_PERC / 100
    assert out["vecon"].steerings[-1] == pytest.approx((limited, -limited))


# --- the features drive nothing --------------------------------------------


@pytest.mark.parametrize("frame", [wall_frame(), graze_frame()],
                         ids=["reverse", "evade"])
def test_in_record_the_features_never_reach_the_wheels(monkeypatch, frame):
    """The hazard test. Frames the rules would reverse on and swerve on, a
    driver left over from auto whose controller and tick are traps, and every
    command the car receives is still exactly my sticks, scaled."""
    verdict = dv.read_the_gradient(dv.free_by_column(frame))
    assert verdict.action in (dv.ACT_REVERSE, dv.ACT_EVADE)

    pad = FakePad()
    pad.throttle = 0.5
    pad.axes["LEFT_X"] = -0.3
    tele = SessionTele()
    tap = fp.FootprintTap(view=FrameView(frame), tele=tele)
    out = run(pad, hold_y(1.0) + [()] * 60, tele=tele, tap=tap,
              driver=TrapDriver(), monkeypatch=monkeypatch)

    assert out["modes"].mode == mn.RECORD
    rows = [row for row, _ in tele.of("footprint")]
    assert rows, "no features were recorded"
    assert all(row["verdict_action"] == verdict.action for row in rows)

    steer = -0.3 * mn.STEERING_LIMIT_PERC / 100
    record_throttles = out["vecon"].throttles[ENTRY:]
    record_steerings = out["vecon"].steerings[ENTRY:]
    assert record_throttles and record_steerings
    assert all(t == pytest.approx(0.1) for t in record_throttles)
    assert all(s == pytest.approx((steer, -steer)) for s in record_steerings)


def test_the_tap_is_never_handed_anything_that_can_command():
    """Structural: footprint.py names no command method, and the loop hands
    it a tuple of floats rather than the controller."""
    source = pathlib.Path(fp.__file__).read_text()
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    body = code.split('"""', 2)[2]  # past the module docstring
    for name in ("set_throttle", "set_steering", "return_neutral",
                 ".tick(", "begin_recover", "vecon", ".controller"):
        assert name not in body, name

    main_source = pathlib.Path(mn.__file__).read_text()
    call = main_source.split("tap.after_drive(", 1)[1].split("steering_mode=")[0]
    assert "vecon.last_throttle" in call
    assert "vecon," not in call and "vecon)" not in call


def test_an_exception_in_the_tap_costs_a_row_and_not_a_tick(monkeypatch):
    def boom(depth, threshold_mm=dv.CLEARANCE_MM):
        raise RuntimeError("scan blew up")

    monkeypatch.setattr(dv, "free_by_column", boom)
    pad = FakePad()
    pad.throttle = 1.0
    tele = SessionTele()
    tap = fp.FootprintTap(view=FrameView(far_frame()), tele=tele)
    out = run(pad, hold_y(1.0) + [()] * 30, tele=tele, tap=tap,
              monkeypatch=monkeypatch)
    assert out["modes"].mode == mn.RECORD
    assert out["vecon"].throttles[-1] == pytest.approx(0.2)
    assert tele.of("footprint") == []


# --- the rows ---------------------------------------------------------------


def test_record_rows_are_the_drivers_own_scan_and_rules(monkeypatch):
    """Called, not copied: swap the rules and the rows follow."""
    seen = []
    real_scan = dv.free_by_column

    def scan(depth, threshold_mm=dv.CLEARANCE_MM):
        seen.append(depth.shape)
        return real_scan(depth, threshold_mm)

    monkeypatch.setattr(dv, "free_by_column", scan)
    monkeypatch.setattr(dv, "read_the_gradient",
                        lambda s: dv.SteerVerdict("sentinel", 0.25, "swapped"))

    tele = SessionTele()
    tap = fp.FootprintTap(view=FrameView(graze_frame()), tele=tele)
    row = tap.after_drive(0.0, fp.RECORD, (0.1, 0.2, -0.2),
                          pad_throttle=0.5, limits=(0.2, -0.6))
    assert seen == [(400, 640)]
    assert row["verdict_action"] == "sentinel"
    assert row["verdict_reason"] == "swapped"
    assert row["mode"] == "record"
    assert row["probe_marks"] == dv.probe_marks(real_scan(graze_frame()))
    assert row["probe_reason"][0] == dv.NEAR
    assert len(row["probe_distance_mm"]) == dv.PROBE_COLUMNS


def test_the_row_carries_the_command_from_its_own_tick(monkeypatch):
    pad = FakePad()
    tele = SessionTele()
    tap = fp.FootprintTap(view=FrameView(far_frame()), tele=tele)

    class Ramp(FakePad):
        def read_throttle(self):
            return min(1.0, self.frames / 200.0)

    pad = Ramp()
    out = run(pad, hold_y(1.0) + [()] * 50, tele=tele, tap=tap,
              monkeypatch=monkeypatch)
    rows = [row for row, _ in tele.of("footprint")]
    assert len(rows) >= 4
    for row in rows:
        assert row["throttle"] == pytest.approx(row["pad_throttle"] * 0.2)
        assert row["forward_limit"] == pytest.approx(0.2)
        assert row["reverse_limit"] == pytest.approx(-0.6)
    # And the last row's throttle is one the ESC was actually sent.
    sent = [round(t, 9) for t in out["vecon"].throttles]
    assert all(round(row["throttle"], 9) in sent for row in rows)


def test_rows_come_at_the_decision_rate_not_the_loop_rate():
    tele = SessionTele()
    tap = fp.FootprintTap(view=FrameView(far_frame()), tele=tele)
    for i in range(100):  # one second of a 100 Hz loop
        tap.after_drive(i * 0.01, fp.RECORD, (0.0, 0.0, 0.0))
    assert len(tele.of("footprint")) == 10
    assert fp.INTERVAL == mn.AUTO_TICK_INTERVAL


def test_every_fifth_row_carries_its_raw_frame_at_the_same_time():
    frame = graze_frame()
    tele = SessionTele()
    tap = fp.FootprintTap(view=FrameView(frame), tele=tele)
    for i in range(100):
        tap.after_drive(i * 0.01, fp.RECORD, (0.0, 0.0, 0.0))

    rows = tele.of("footprint")
    raws = tele.of("depth_raw")
    assert len(raws) == 2
    flagged = [now for row, now in rows if row["depth_raw"]]
    assert flagged == [now for _, now in raws]
    # The first row of a session has its frame: rows 1 and 6 of 10.
    assert [row["seq"] for row, _ in rows if row["depth_raw"]] == [1, 6]
    assert all(raw is frame for raw, _ in raws)
    assert [row["seq"] for row, _ in rows] == list(range(1, 11))


def test_no_rows_and_no_frames_in_plain_manual():
    tele = SessionTele()
    view = FrameView(far_frame())
    tap = fp.FootprintTap(view=view, tele=tele)
    for i in range(50):
        tap.after_drive(i * 0.01, mn.MANUAL, (0.0, 0.0, 0.0))
    assert tele.offers == []
    assert view.takes == 0


def test_auto_rows_are_the_drivers_decisions_once_each():
    class Ticked:
        def __init__(self):
            self.last_scan = dv.free_by_column(graze_frame())
            self.last_scan_ms = 2.0
            self.last_verdict = dv.SteerVerdict(dv.ACT_EVADE, 1.0, "holding out")

    tele = SessionTele()
    tap = fp.FootprintTap(view=ImuView(), tele=tele)
    driver = Ticked()
    first = tap.after_drive(0.0, fp.AUTO, (0.2, 0.6, -0.6), driver=driver)
    again = tap.after_drive(0.01, fp.AUTO, (0.2, 0.6, -0.6), driver=driver)
    assert first is not None and again is None

    assert first["mode"] == "auto"
    assert first["source"] == "driver"
    assert first["verdict_action"] == dv.ACT_EVADE
    assert first["committed_reason"] == "holding out"
    assert first["throttle"] == pytest.approx(0.2)
    assert first["scan_ms"] == pytest.approx(2.0)

    driver.last_scan = dv.free_by_column(wall_frame())
    assert tap.after_drive(0.1, fp.AUTO, (0.0, 0.0, 0.0),
                           driver=driver)["verdict_action"] == dv.ACT_REVERSE


def test_every_row_has_its_policy_picture_at_its_own_time_and_seq():
    frame = graze_frame()
    tele = SessionTele()
    tap = fp.FootprintTap(view=FrameView(frame), tele=tele)
    for i in range(100):
        tap.after_drive(i * 0.01, fp.RECORD, (0.0, 0.0, 0.0))

    rows = tele.of("footprint")
    pictures = tele.of("depth_small")
    assert len(rows) == len(pictures) == 10
    expected = policy_input.depth_small(frame)
    for (row, row_now), ((small, frame_id), now) in zip(rows, pictures):
        assert row["depth_small"] is True
        assert now == row_now
        assert frame_id == f"footprint/{row['seq']}"
        assert np.array_equal(small, expected)


def test_an_auto_picture_is_the_frame_the_driver_scanned():
    """Read off the driver's own view after its take, never a second take."""
    scanned = graze_frame()
    auto_view = camera.StreamView("auto")
    auto_view._put("depth", scanned)
    assert auto_view.get_depth_frame() is scanned       # the driver's take
    assert auto_view.get_depth_frame() is None
    assert auto_view.last_depth is scanned

    class Ticked:
        last_scan = dv.free_by_column(scanned)
        last_scan_ms = 1.0
        last_verdict = None

    tele = SessionTele()
    tap = fp.FootprintTap(view=ImuView(), tele=tele, auto_view=auto_view)
    row = tap.after_drive(0.0, fp.AUTO, (0.2, 0.6, -0.6), driver=Ticked())
    (small, frame_id), now = tele.of("depth_small")[0]
    assert row["depth_small"] is True and now == 0.0
    assert frame_id == f"footprint/{row['seq']}"
    assert np.array_equal(small, policy_input.depth_small(scanned))
    assert (row["frame_width"], row["frame_height"]) == (640, 400)


def test_a_frame_the_policy_cannot_take_costs_the_picture_not_the_row():
    tele = SessionTele()
    tap = fp.FootprintTap(view=FrameView(np.full((40, 64), 3000, np.uint16)),
                          tele=tele)
    row = tap.after_drive(0.0, fp.RECORD, (0.0, 0.0, 0.0))
    assert row is not None and row["depth_small"] is False
    assert tele.of("depth_small") == []
    assert len(tele.of("footprint")) == 1


def test_each_row_carries_the_newest_imu_sample_and_its_age():
    class Both(FrameView):
        def __init__(self, frame):
            super().__init__(frame)
            self.samples = []

        def get_imu_sample(self):
            return self.samples.pop(0) if self.samples else None

    view = Both(far_frame())
    tele = SessionTele()
    tap = fp.FootprintTap(view=view, tele=tele)
    first = tap.after_drive(0.0, fp.RECORD, (0.0, 0.0, 0.0))
    assert first["imu_age_ms"] == -1.0 and first["imu_gyro_z"] == 0.0

    view.samples = [{"accel": [0.1, 0.2, 9.8], "gyro": [0.0, 0.0, 0.5]},
                    {"accel": [1.0, 2.0, 3.0], "gyro": [4.0, 5.0, 6.0]}]
    tap.after_drive(0.03, mn.MANUAL, (0.0, 0.0, 0.0))   # polled in manual too
    tap.after_drive(0.05, mn.MANUAL, (0.0, 0.0, 0.0))
    row = tap.after_drive(0.1, fp.RECORD, (0.0, 0.0, 0.0))
    assert (row["imu_accel_x"], row["imu_accel_y"], row["imu_accel_z"]) == (1.0, 2.0, 3.0)
    assert (row["imu_gyro_x"], row["imu_gyro_y"], row["imu_gyro_z"]) == (4.0, 5.0, 6.0)
    assert row["imu_age_ms"] == pytest.approx(50.0)
    assert row["mode"] == "record"


def test_each_row_says_which_way_the_cars_own_sweep_could_turn():
    """Something near on the left: the sweep clears to the right only."""
    scan = dv.free_by_column(graze_frame())
    assert (dv.turn_sweep_clears(scan, -1), dv.turn_sweep_clears(scan, 1)) == (
        False, True)
    tele = SessionTele()
    tap = fp.FootprintTap(view=FrameView(graze_frame()), tele=tele)
    row = tap.after_drive(0.0, fp.RECORD, (0.0, 0.0, 0.0))
    assert (row["sweep_clears_left"], row["sweep_clears_right"]) == (False, True)


def test_the_footprint_schema_covers_every_field_a_row_has():
    tele = SessionTele()
    tap = fp.FootprintTap(view=FrameView(far_frame()), tele=tele)
    row = tap.after_drive(0.0, fp.RECORD, (0.0, 0.0, 0.0))
    properties = tm.JSON_SCHEMAS["footprint"]["schema"]["properties"]
    assert set(row) == set(properties)
    assert tm.JSON_TOPICS["footprint"] == "/footprint"
    assert tm.RAW_DEPTH_CHANNELS["depth_raw"] == "/camera/depth_raw"
    assert tm.RAW_DEPTH_CHANNELS["depth_small"] == "/policy/depth_small"


# --- one file per session ---------------------------------------------------


class FakeRecorder:
    def __init__(self, name):
        self.name = name
        self.events = []
        self.active = False
        self.path = name
        self.bytes_written = 0
        self.error = None

    def start(self):
        self.events.append("start")
        self.active = True
        return True

    def stop(self):
        self.events.append("stop")
        self.active = False

    def tick(self, now):
        pass


def test_a_session_takes_the_file_from_the_ring_and_gives_it_back():
    ring, session = FakeRecorder("ring"), FakeRecorder("session")
    tele = tm.Telemetry(recorder=ring)
    tele.running = True

    assert tele.begin_session(session) is True
    tele._swap_session(ring)
    assert ring.events == ["stop"]
    assert session.events == ["start"]
    assert tele.session_active
    assert tele.recording_status()["recording_session"] is True
    assert tele.recording_status()["recording_file"] == "session"

    tele._swap_session(ring)  # nothing asked, nothing changes
    assert session.events == ["start"]

    tele.end_session()
    tele._swap_session(ring)
    assert session.events == ["start", "stop"]
    assert ring.events == ["stop", "start"]
    assert not tele.session_active


def test_a_session_cannot_begin_without_a_publisher_thread():
    tele = tm.Telemetry()
    assert tele.begin_session(FakeRecorder("session")) is False


def test_session_files_are_named_apart_and_the_ring_never_prunes_them(tmp_path):
    name = recording.segment_name(when=0, pid=7, index=1,
                                  prefix=recording.SESSION_PREFIX)
    assert name.startswith("crawler-record-19700101-000000Z")

    (tmp_path / name).write_bytes(b"x" * 1000)
    ring_file = recording.segment_name(when=0, pid=7, index=1)
    (tmp_path / ring_file).write_bytes(b"x" * 1000)

    ring = recording.Recorder(directory=str(tmp_path), max_total_bytes=10)
    ring._prune()
    assert (tmp_path / name).exists()
    assert not (tmp_path / ring_file).exists()

    session = recording.Recorder(directory=str(tmp_path), max_total_bytes=None,
                                 prefix=recording.SESSION_PREFIX)
    session._prune()
    assert (tmp_path / name).exists()


# --- the camera -------------------------------------------------------------


def test_release_returns_at_once_and_the_hub_can_open_again(monkeypatch):
    import sys
    import threading
    import types

    gate = threading.Event()

    class SlowOakD:
        def __init__(self, rectified=True):
            pass

        def close(self):
            gate.wait(5)

        def get_depth_frame(self):
            return None

        get_disparity_frame = get_rectified_left_frame = get_depth_frame
        get_imu_sample = get_depth_frame

    monkeypatch.setitem(sys.modules, "oakd", types.SimpleNamespace(OakD=SlowOakD))
    hub = camera.CameraHub()
    hub.start()
    assert hub.wait_ready(5)
    assert hub.open_seconds >= 0.0

    assert hub.release() is True
    assert hub.state == camera.RELEASING   # close() is still blocked
    assert hub.start() is False            # and nothing races it
    gate.set()
    for _ in range(500):
        if hub.state == camera.OFF:
            break
        threading.Event().wait(0.01)
    assert hub.state == camera.OFF
    assert hub.start() is True
    assert hub.wait_ready(5)
    hub.close()
    assert hub.state == camera.CLOSED


def test_auto_after_a_released_camera_reopens_it_rather_than_driving_blind():
    hub = FakeHub(state=camera.OFF)
    pilot = mn.AutoPilot(FakeVecon(), FakeLogger(), hub=hub)
    pilot.driver = object()  # left over from an earlier lap
    assert pilot.resume() is False
    assert hub.starts == 1
    assert pilot.unavailable is False


def test_auto_waits_out_a_release_instead_of_latching_unavailable():
    hub = FakeHub(state=camera.RELEASING)
    pilot = mn.AutoPilot(FakeVecon(), FakeLogger(), hub=hub)
    assert pilot.resume() is False
    assert pilot.unavailable is False


# --- the throttle latch -----------------------------------------------------


class TriggerPad(FakePad):
    """A pad whose throttle follows a plan, one value per loop pass."""

    def __init__(self, plan):
        super().__init__()
        self.plan = list(plan)
        self.axes["LEFT_X"] = -0.3

    def step(self):
        super().step()
        if self.plan:
            self.throttle = self.plan.pop(0)


HELD, LET_GO, AGAIN = 12, 3, 5


def after_handover(throttles, index):
    """The throttles commanded from `index` on, split at the first live one."""
    tail = throttles[index:]
    live = next((i for i, t in enumerate(tail) if t != 0.0), len(tail))
    return tail[:live], tail[live:]


def test_leaving_record_with_rt_held_holds_the_throttle_until_it_is_let_go(
        monkeypatch):
    """The 0.2 to full-range jump, gone: Y tap out with RT still at full."""
    head = hold_y(1.0) + [()] * 5 + [(mn.RECORD_BUTTON,), ()]
    script = head + [()] * (HELD + LET_GO + AGAIN)
    plan = [1.0] * (len(head) + HELD) + [0.0] * LET_GO + [1.0] * AGAIN
    pad = TriggerPad(plan)
    out = run(pad, script, monkeypatch=monkeypatch)
    assert out["modes"].mode == mn.MANUAL

    throttles = out["vecon"].throttles
    last_record = max(i for i, t in enumerate(throttles) if t == pytest.approx(0.2))
    held, live = after_handover(throttles, last_record + 1)
    assert len(held) >= HELD + LET_GO
    assert live == [1.0] * AGAIN
    # Steering was never held: the sticks drove every one of those passes.
    steer = -0.3 * mn.STEERING_LIMIT_PERC / 100
    assert out["vecon"].steerings[-(HELD + LET_GO + AGAIN):] == [
        pytest.approx(dv.steering_pair(steer))] * (HELD + LET_GO + AGAIN)


def test_out_of_auto_into_manual_the_trigger_has_to_let_go_first(monkeypatch):
    head = [()] * 3 + [(mn.MODE_TOGGLE_BUTTON,), ()]
    script = head + [()] * (HELD + LET_GO + AGAIN)
    plan = [1.0] * (len(head) + HELD) + [0.0] * LET_GO + [1.0] * AGAIN
    pad = TriggerPad(plan)
    out = run(pad, script, mode=mn.AUTO, monkeypatch=monkeypatch)
    assert out["modes"].mode == mn.MANUAL
    held, live = after_handover(out["vecon"].throttles, 0)
    assert len(held) >= HELD + LET_GO
    assert live == [1.0] * AGAIN


def test_a_pad_that_comes_back_after_record_is_held_too(monkeypatch):
    class Dropping(TriggerPad):
        def is_connected(self):
            return not (ENTRY + 3 <= self.frames < ENTRY + 8)

    script = hold_y(1.0) + [()] * (5 + 5 + HELD + LET_GO + AGAIN)
    plan = [1.0] * (len(script) - LET_GO - AGAIN) + [0.0] * LET_GO + [1.0] * AGAIN
    pad = Dropping(plan)
    out = run(pad, script, monkeypatch=monkeypatch)
    assert out["modes"].mode == mn.MANUAL
    throttles = out["vecon"].throttles
    last_record = max(i for i, t in enumerate(throttles) if t == pytest.approx(0.2))
    held, live = after_handover(throttles, last_record + 1)
    assert len(held) >= HELD
    assert live == [1.0] * AGAIN


def test_manual_from_boot_is_not_held(monkeypatch):
    pad = TriggerPad([1.0] * 10)
    out = run(pad, [()] * 10, monkeypatch=monkeypatch)
    assert out["vecon"].throttles == [1.0] * 10


def test_a_stale_read_never_counts_as_let_go():
    class Stale:
        def triggers_at_rest(self):
            raise RuntimeError("stale handle")

    latch = mn.ThrottleLatch()
    latch.arm(mn.RECORD)
    assert latch.gate(Stale(), 1.0) == 0.0
    assert latch.held is True


def test_triggers_at_rest_reads_both_triggers_against_their_rest():
    import Joystick as js

    class Handle:
        def __init__(self):
            self.axes = {js.XBOX_AXES["RT"]: -1.0, js.XBOX_AXES["LT"]: -1.0}

        def get_axis(self, index):
            return self.axes[index]

    pad = object.__new__(js.Joystick)
    pad.joystick = Handle()
    pad._trigger_rest = {"RT": -1.0, "LT": -1.0}
    assert pad.triggers_at_rest() is True
    pad.joystick.axes[js.XBOX_AXES["LT"]] = 0.0     # LT half pulled
    assert pad.triggers_at_rest() is False
    pad.joystick.axes[js.XBOX_AXES["LT"]] = -1.0
    pad.joystick.axes[js.XBOX_AXES["RT"]] = -0.85   # RT at 7.5%, past 5%
    assert pad.triggers_at_rest() is False
    pad.joystick = None                              # a stale handle
    assert pad.triggers_at_rest() is False
