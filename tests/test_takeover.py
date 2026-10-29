"""Takeover: correcting auto one control at a time.

In auto the pad's two groups -- steering (LEFT_X, plus RIGHT_X in split) and
throttle (the triggers) -- are each taken off the autopilot by moving them past
their deadzone, and handed back after half a second inside it. These run the
real control loop over a real AutoPilot and a real Driver, a fake clock and a
fake pad, and hold:

  each group is taken alone, and the autopilot keeps the other;
  the handback is at 0.5 s and not before, and puts the autopilot's own
  standing command back, never a straight or a stop of its own;
  a manoeuvre under a taken-over group: my hand keeps it, the driver keeps
  the rest;
  at most one set_steering and one set_throttle reach the hardware a pass;
  no floor holds my trigger either way, as none does in manual, and my
  hand wins both groups on its first pass through any manoeuvre;
  a lost pad, a pad fault or a nan ends at neutral;
  every /footprint row says whose each group was, next to the command that
  actually went out.
"""

import math
import random

import numpy as np
import pytest

from crawler.autopilot import driver as dv
from crawler.autopilot import footprint as fp
import crawler.__main__ as mn

from test_main import FakeLogger, FakePad, FakeVecon, TickLimit
from test_record import (FakeClock, FrameView, SessionTele, far_frame,
                         graze_frame, wall_frame)

# A binary fraction, so half a second is exactly 32 passes.
STEP = 1.0 / 64
HALF = int(mn.TAKEOVER_RELEASE_SECONDS / STEP)

MOVING = {"accel": [0.3, 0.0, 0.0], "gyro": [0.05, 0.0, 0.0]}
STILL = {"accel": [0.0, 0.0, 0.0], "gyro": [0.0, 0.0, 0.0]}


def limited(raw):
    return mn.stick_steer(raw)


class Camera:
    """The driver's view. Whatever it was last loaded with, every take."""

    def __init__(self, frame, imu=MOVING):
        self.frame = frame
        self.imu = imu

    def get_depth_frame(self):
        return self.frame

    def get_imu_sample(self):
        return self.imu


class PassVecon(FakeVecon):
    """A FakeVecon that knows which loop pass each write landed on."""

    def __init__(self, pad):
        super().__init__()
        self.pad = pad
        self.writes = []

    # The pump has already run by the time anything is written, so the pass
    # a write belongs to is one less than the pad's frame count.
    def set_throttle(self, value):
        self.writes.append((self.pad.frames - 1, "throttle", value))
        super().set_throttle(value)

    def set_steering(self, front, rear):
        self.writes.append((self.pad.frames - 1, "steering", (front, rear)))
        super().set_steering(front, rear)

    def return_neutral(self):
        self.neutrals += 1
        self.writes.append((self.pad.frames - 1, "neutral", None))
        self.last_throttle = 0.0
        self.last_steering_front = self.last_steering_rear = 0.0


class Snapshots:
    """A MarkPad stand-in that photographs every pass after the drive call."""

    def __init__(self):
        self.passes = []
        self.buzzes = []
        self.takeover = None

    def poll(self, joystick, now, mode, vecon, driver, autopilot):
        t = self.takeover
        self.passes.append({
            "now": now, "mode": mode,
            "steer_by": t.steer_by, "throttle_by": t.throttle_by,
            "throttle": vecon.last_throttle,
            "steering": (vecon.last_steering_front, vecon.last_steering_rear),
            "phase": getattr(driver, "_phase", None),
        })

    def buzz(self, pulses, now):
        self.buzzes.append(pulses)

    def counts(self):
        return {"marks_good": 0, "marks_bad": 0}

    def stop(self):
        pass


def run_auto(monkeypatch, passes, cam, plan=None, pad=None, tap=None,
             tele=None, make_driver=dv.Driver):
    """control_loop in auto over a real AutoPilot and Driver.

    `plan(i, pad, cam)` runs before pass i reads anything, so it is the
    hand on the pad and the scene in front of the car.
    """
    import pygame

    pad = pad or FakePad()
    clock = FakeClock(STEP)

    def pump():
        if plan is not None:
            plan(pad.frames, pad, cam)
        pad.step()
        clock.t += STEP

    monkeypatch.setattr(pygame.event, "pump", pump)
    monkeypatch.setattr(mn, "LOOP_INTERVAL", 0)

    log = FakeLogger()
    vecon = PassVecon(pad)
    pilot = mn.AutoPilot(vecon, log)
    driver = pilot.driver = make_driver(pilot.command, cam)
    assert pilot.resume()
    snaps = Snapshots()
    takeover = mn.Takeover(log, snaps)
    snaps.takeover = takeover
    tele = tele or SessionTele()
    tap = tap or fp.FootprintTap(tele=tele)
    session = mn.RecordSession(tele, None, log)
    modes = mn.ModeSwitch(mn.AUTO, log)

    random.seed(7)  # begin_recover's random turn
    mn.control_loop(pad, vecon, pilot, modes, log, TickLimit(passes), tele,
                    snaps, mn.SteerMode(log), session=session, tap=tap,
                    clock=clock, takeover=takeover)
    return {"vecon": vecon, "snaps": snaps.passes, "buzzes": snaps.buzzes,
            "driver": driver,
            "log": log, "modes": modes, "pilot": pilot, "tele": tele,
            "takeover": takeover}


def at(i, **changes):
    """A plan that applies `changes` to the pad from pass i on."""
    def plan(n, pad, cam):
        if n == i:
            for key, value in changes.items():
                if key == "throttle":
                    pad.throttle = value
                else:
                    pad.axes[key] = value
    return plan


def chain(*plans):
    def plan(n, pad, cam):
        for p in plans:
            p(n, pad, cam)
    return plan


# --- the two groups ---------------------------------------------------------


def test_steering_takeover_leaves_the_throttle_with_the_autopilot(monkeypatch):
    out = run_auto(monkeypatch, 60, Camera(far_frame()),
                   plan=at(20, LEFT_X=0.5))
    s = out["snaps"]
    assert s[10]["steer_by"] == mn.DRIVER and s[10]["throttle_by"] == mn.DRIVER
    for snap in s[20:]:
        assert snap["steer_by"] == mn.PAD
        assert snap["throttle_by"] == mn.DRIVER
        assert snap["steering"] == pytest.approx(dv.steering_pair(limited(0.5)))
        assert snap["throttle"] == pytest.approx(out["driver"].forward_speed)


def test_throttle_takeover_leaves_the_steering_with_the_autopilot(monkeypatch):
    """The reverse: a graze the driver steers around, and my trigger."""
    out = run_auto(monkeypatch, 60, Camera(graze_frame()),
                   plan=at(20, throttle=0.5))
    driver = out["driver"]
    evade = dv.steering_pair(driver.max_turn)
    for snap in out["snaps"][20:]:
        assert snap["steer_by"] == mn.DRIVER
        assert snap["throttle_by"] == mn.PAD
        assert snap["steering"] == pytest.approx(evade)
        assert snap["throttle"] == pytest.approx(0.5 * driver.forward_speed)


def test_full_triggers_are_auto_forward_and_reverse(monkeypatch):
    def plan(n, pad, cam):
        pad.throttle = 1.0 if n < 30 else -1.0

    out = run_auto(monkeypatch, 60, Camera(far_frame()), plan=plan)
    driver = out["driver"]
    assert out["snaps"][20]["throttle"] == pytest.approx(driver.forward_speed)
    assert out["snaps"][50]["throttle"] == pytest.approx(driver.reverse_speed)


def test_the_caps_are_read_off_the_live_driver_not_copied():
    pilot = mn.AutoPilot(FakeVecon(), FakeLogger())
    pilot.driver = dv.Driver(pilot.command, Camera(far_frame()))
    pilot.driver.forward_speed = 0.13
    pilot.driver.reverse_speed = -0.31
    takeover = mn.Takeover()
    pad = FakePad()
    for pull, want in ((1.0, 0.13), (-1.0, -0.31), (0.5, 0.065)):
        pad.throttle = pull
        assert takeover.poll(pad, 0.0)
        assert pilot._pad_throttle(takeover) == pytest.approx(want)


# --- taking and giving back -------------------------------------------------


def test_the_steering_goes_back_at_half_a_second_and_not_at_point_four(
        monkeypatch):
    """Released at pass 30: still mine at 0.4 s and at 31 passes, the
    autopilot's at 32 -- and the handback is the driver's own command, not a
    centred wheel."""
    out = run_auto(monkeypatch, 30 + HALF + 10, Camera(graze_frame()),
                   plan=chain(at(10, LEFT_X=-0.6), at(30, LEFT_X=0.05)))
    s = out["snaps"]
    evade = dv.steering_pair(out["driver"].max_turn)
    assert s[29]["steering"] == pytest.approx(dv.steering_pair(limited(-0.6)))
    point_four = 30 + math.ceil(0.4 / STEP)
    for i in range(30, 30 + HALF):
        assert s[i]["steer_by"] == mn.PAD, i
        assert s[i]["steering"] == pytest.approx(dv.steering_pair(limited(0.05)))
    assert s[point_four]["steer_by"] == mn.PAD
    assert s[30 + HALF]["steer_by"] == mn.DRIVER
    # Written on the handback pass itself, not on the driver's next tick.
    assert s[30 + HALF]["steering"] == pytest.approx(evade)


def test_the_throttle_goes_back_to_the_autopilot_and_not_to_a_stop(
        monkeypatch):
    out = run_auto(monkeypatch, 30 + HALF + 10, Camera(far_frame()),
                   plan=chain(at(10, throttle=-1.0), at(30, throttle=0.0)))
    s = out["snaps"]
    assert s[29]["throttle"] == pytest.approx(dv.REVERSE_SPEED)
    assert s[30 + HALF - 1]["throttle_by"] == mn.PAD
    assert s[30 + HALF]["throttle_by"] == mn.DRIVER
    assert s[30 + HALF]["throttle"] == pytest.approx(dv.FORWARD_SPEED)


def test_rest_noise_never_takes_over_and_the_band_between_never_releases(
        monkeypatch):
    """0.106 is the highest any rest measured. Between the two edges a stick
    neither takes nor gives back."""
    def plan(n, pad, cam):
        pad.axes["LEFT_X"] = (0.106 if n < 20 else 0.18 if n < 30
                              else 0.5 if n < 35 else 0.15)

    out = run_auto(monkeypatch, 35 + 3 * HALF, Camera(far_frame()), plan=plan)
    s = out["snaps"]
    assert all(snap["steer_by"] == mn.DRIVER for snap in s[:30])
    assert all(snap["steer_by"] == mn.PAD for snap in s[30:])


def test_split_puts_the_right_stick_in_the_steering_group(monkeypatch):
    takeover = mn.Takeover()
    pad = FakePad()
    pad.axes["RIGHT_X"] = 0.5
    assert takeover.poll(pad, 0.0, split=False)
    assert not takeover.steering
    assert takeover.poll(pad, 0.1, split=True)
    assert takeover.steering
    assert takeover.pad_steering() == pytest.approx((0.0, limited(0.5)))


def test_left_y_does_nothing():
    takeover = mn.Takeover()

    class YPad(FakePad):
        def get_axis(self, name, limit_perc=100):
            if name == "LEFT_Y":
                return 1.0
            return super().get_axis(name, limit_perc)

    assert takeover.poll(YPad(), 0.0, split=True)
    assert not takeover.steering and not takeover.throttle


def test_take_and_give_back_each_buzz_once_and_say_so(monkeypatch):
    out = run_auto(monkeypatch, 20 + HALF + 5, Camera(far_frame()),
                   plan=chain(at(5, LEFT_X=0.5), at(20, LEFT_X=0.0)))
    assert out["buzzes"] == [1, 1]
    said = [line[2] for line in out["log"].lines if line[1] == "Takeover"]
    assert said == ["Took the steering", "Gave back the steering"]


# --- manoeuvres -------------------------------------------------------------


def test_a_recover_under_my_steering_keeps_my_steering(monkeypatch):
    """A wall: the driver reverses and turns. My stick keeps the wheel for the
    whole reverse, the driver keeps the throttle, and when I let go mid
    manoeuvre the driver's own turn goes straight back on."""
    let_go = 80
    out = run_auto(monkeypatch, let_go + HALF + 10, Camera(wall_frame()),
                   plan=chain(at(0, LEFT_X=0.5), at(let_go, LEFT_X=0.0)))
    s = out["snaps"]
    mine = dv.steering_pair(limited(0.5))
    reversing = [i for i in range(1, let_go) if s[i]["phase"] is not None]
    assert reversing, "the driver never started a recover"
    for i in range(1, let_go):
        assert s[i]["steering"] == pytest.approx(mine), i
    # Letting go: still mine for half a second, at the stick's own centre.
    for i in range(let_go, let_go + HALF):
        assert s[i]["steer_by"] == mn.PAD, i
        assert s[i]["steering"] == pytest.approx((0.0, 0.0)), i
    assert any(s[i]["throttle"] == pytest.approx(dv.REVERSE_SPEED) for i in reversing)

    handback = s[let_go + HALF]
    assert handback["steer_by"] == mn.DRIVER
    assert handback["phase"] is not None
    driver = out["driver"]
    assert handback["steering"] == pytest.approx(
        dv.steering_pair(driver._turn_dir))


def test_stuck_cannot_fire_under_my_throttle(monkeypatch):
    """Open floor, the driver creeping and armed, then the IMU goes still.
    Under the driver's own throttle that is a STUCK and a reverse; under mine
    -- taken at the same moment the car stops -- it is me holding the car,
    even though the driver had already armed before I took it."""
    def scene(mine):
        def plan(n, pad, cam):
            cam.imu = MOVING if n < 150 else STILL
            if mine and n >= 150:
                pad.throttle = 0.5
        return plan

    alone = run_auto(monkeypatch, 450, Camera(far_frame()), plan=scene(False))
    assert any(t == pytest.approx(dv.REVERSE_SPEED) for t in alone["vecon"].throttles)

    mine = run_auto(monkeypatch, 450, Camera(far_frame()), plan=scene(True))
    assert all(snap["phase"] is None for snap in mine["snaps"])
    assert not any(t < 0 for t in mine["vecon"].throttles)


# --- the hardware -----------------------------------------------------------


def test_at_most_one_steering_and_one_throttle_write_a_pass(monkeypatch):
    """Through takeovers, handbacks, an evade and a recover."""
    def plan(n, pad, cam):
        cam.frame = (far_frame() if n < 40 else graze_frame() if n < 100
                     else wall_frame())
        pad.axes["LEFT_X"] = 0.5 if 20 <= n < 60 or 120 <= n < 150 else 0.0
        pad.throttle = 0.5 if 50 <= n < 90 or 130 <= n < 140 else 0.0

    out = run_auto(monkeypatch, 260, Camera(far_frame()), plan=plan)
    per_pass = {}
    for n, kind, _ in out["vecon"].writes:
        per_pass.setdefault(n, []).append(kind)
    assert per_pass
    for n, kinds in per_pass.items():
        assert kinds.count("steering") <= 1, (n, kinds)
        assert kinds.count("throttle") <= 1, (n, kinds)
    # And takeover passes really did write both every pass.
    assert all(per_pass[n] == ["steering", "throttle"]
               for n in range(131, 140))


def test_without_my_hand_auto_writes_what_the_driver_asks(monkeypatch):
    """Byte for byte against the driver commanding a controller directly,
    at the same times, over the same scenes -- except that two steering asks
    in one tick (the end of a recover) reach the servos as the last one."""
    scenes = [far_frame(), graze_frame(), wall_frame(), None, far_frame(400)]

    def scene(n):
        return scenes[(n // 23) % len(scenes)]

    def plan(n, pad, cam):
        cam.frame = scene(n)

    passes = 600
    out = run_auto(monkeypatch, passes, Camera(far_frame()), plan=plan)
    got = [(n, kind, value) for n, kind, value in out["vecon"].writes]

    from conftest import FakeController

    class Stamped(FakeController):
        def __init__(self):
            super().__init__()
            self.calls = []

        def set_throttle(self, value):
            self.calls.append((self.n, "throttle", value))

        def set_steering(self, front, rear):
            self.calls.append((self.n, "steering", (front, rear)))

    ctl = Stamped()
    cam = Camera(far_frame())
    direct = dv.Driver(ctl, cam)
    random.seed(7)
    next_tick = 0.0
    for n in range(passes):
        now = n * STEP
        if now >= next_tick:
            next_tick = now + mn.AUTO_TICK_INTERVAL
            cam.frame = scene(n)
            ctl.n = n
            before = len(ctl.calls)
            direct.tick(now)
            steer = [c for c in ctl.calls[before:] if c[1] == "steering"]
            ctl.calls[before:] = [c for c in ctl.calls[before:]
                                  if c[1] != "steering" or c is steer[-1]]
    assert len(got) > 50
    assert got == ctl.calls


def centre_frame(mm):
    """Probes 3-5 at `mm` (0 is blind), the rest open floor."""
    frame = far_frame()
    edges = dv.footprint_geometry(frame.shape).col_edges
    frame[:, edges[3]:edges[6]] = mm
    return frame


def standing_policy(command, cam):
    """A policy that only ever says stop: it scans every tick and never
    starts a recover, so all that is left holding my trigger is whatever
    holds it in takeover itself."""
    from test_policy import STOP, StubModel
    from crawler.autopilot import policy as pl
    # Short of the goal its stop is driven on, at the magnitude it gave: 0.
    return pl.PolicyDriver(command, cam, StubModel(STOP, throttle=0.0))


def full_rt(n, pad, cam):
    pad.throttle = 1.0


# --- my forward is manual's forward -----------------------------------------
#
# Taken over, the trigger is mine the way it is in manual: no floor, no scan
# it has to wait for. Manual has never held a forward of mine, and the floor
# that used to sit on my trigger here is what kept me from driving out of a
# spot until I had backed up first. The autopilot's own forwards keep their
# floor (driver.ForwardFloorVeto, policy "model floor"); mine do not.


@pytest.mark.parametrize("mm", [340, 174])
def test_my_forward_goes_through_a_probe_under_the_floor(monkeypatch, mm):
    """The drive that showed it: the centre probes at 340 and 174, under the
    363 mm floor, full RT, and 0 on the wheels until I backed up."""
    assert mm < dv.FORWARD_FLOOR_MM
    out = run_auto(monkeypatch, 60, Camera(centre_frame(mm)), plan=full_rt,
                   make_driver=standing_policy)
    assert all(snap["throttle_by"] == mn.PAD for snap in out["snaps"])
    assert all(snap["throttle"] == pytest.approx(dv.FORWARD_SPEED)
               for snap in out["snaps"])


def test_my_forward_and_my_reverse_go_through_a_wall_at_the_bumper(monkeypatch):
    """Everything at 190 mm, which the rules reverse from: my trigger is
    still mine both ways, from the first pass."""
    out = run_auto(monkeypatch, 60, Camera(wall_frame(190)), plan=full_rt)
    assert all(snap["throttle"] == pytest.approx(dv.FORWARD_SPEED)
               for snap in out["snaps"])

    def back(n, pad, cam):
        pad.throttle = -1.0

    out = run_auto(monkeypatch, 60, Camera(wall_frame(190)), plan=back)
    assert all(snap["throttle"] == pytest.approx(dv.REVERSE_SPEED)
               for snap in out["snaps"])


def test_my_forward_goes_through_a_probe_gone_blind_on_the_way_in(monkeypatch):
    """390, 300, 190, then nothing because it is too close. The autopilot's
    floor latches on that; my trigger does not."""
    approach = [390, 300, 190, 0]
    tick = int(round(mn.AUTO_TICK_INTERVAL / STEP)) + 1

    def plan(n, pad, cam):
        pad.throttle = 1.0
        cam.frame = centre_frame(approach[min(n // tick, len(approach) - 1)])

    out = run_auto(monkeypatch, 6 * tick, Camera(centre_frame(390)),
                   plan=plan, make_driver=standing_policy)
    assert out["driver"].last_scan.reason[4] == dv.NO_RETURN
    assert all(snap["throttle"] == pytest.approx(dv.FORWARD_SPEED)
               for snap in out["snaps"])


def test_the_forward_floor_covers_the_rules_floor_and_more():
    """The autopilot's own floor, which my trigger no longer sits under:
    wherever the rules reverse on R2 a forward the rules did not choose is
    held, and it is held further out too, everything under FORWARD_FLOOR_MM."""
    for mm in (150, 190, 199, 200, 230, 300, 360, 370, 380):
        scan = dv.free_by_column(wall_frame(mm))
        held = dv.forward_floor_vetoes(dv.EMPTY_FLOOR_LATCH, scan)
        if dv.read_the_gradient(scan).reason.startswith("floor"):
            assert held, mm
        assert held == (mm < dv.FORWARD_FLOOR_MM), mm
    assert dv.forward_floor_vetoes(dv.EMPTY_FLOOR_LATCH, None)


# --- my hand outranks every manoeuvre, from its first pass -------------------


def _rules_wall(command, cam):
    return dv.Driver(command, cam)


def _policy(logits):
    def make(command, cam):
        from test_policy import StubModel
        from crawler.autopilot import policy as pl
        return pl.PolicyDriver(command, cam, StubModel(logits, throttle=1.0))
    return make


def _chair_leg():
    from test_policy import chair_leg
    return chair_leg()


# (make_driver, the scene, the IMU from pass n, is it running?) for each of
# the manoeuvres that drive the car on their own: the rules' recover off a
# wall, the policy's standoff with its floor, its STUCK, and its reverse
# commit. Each is caught some way in -- past FLOOR_SCAN_MAX_AGE, where a
# recover has made no scan for a while -- and that is when I take both groups.
def _manoeuvres():
    from test_policy import FORWARD, REVERSE

    def recovering(d):
        return d._phase == dv._PHASE_REVERSE

    return {
        "rules recover": (_rules_wall, lambda: wall_frame(200),
                          lambda n: MOVING, recovering),
        "policy standoff": (_policy(FORWARD), _chair_leg,
                            lambda n: MOVING, recovering),
        # STUCK waits out the recover cooldown from t=0, then the car stops.
        "policy stuck": (_policy(FORWARD), lambda: far_frame(),
                         lambda n: MOVING if n < 360 else STILL, recovering),
        "policy reverse commit": (_policy(REVERSE), lambda: far_frame(),
                                  lambda n: MOVING,
                                  lambda d: d.reverse_commit),
    }


def run_caught(monkeypatch, name, hold, passes=None, into=None):
    """Drive into the manoeuvre, take both groups `into` passes after it
    started (full RT, the stick at 0.5), hold them `hold` passes, let go.
    Returns the run and the pass I took them on."""
    make, scene, imu, running = _manoeuvres()[name]
    if into is None:
        # A commit is 0.7 s; catch it before it ends. A recover's reverse is
        # 2 s; catch it once its last scan is past FLOOR_SCAN_MAX_AGE.
        into = 8 if name.endswith("commit") else 32
    box = {}

    def make_driver(command, cam):
        box["d"] = make(command, cam)
        return box["d"]

    def plan(n, pad, cam):
        d = box["d"]
        cam.imu = imu(n) if "began" not in box else MOVING
        if "began" not in box and running(d):
            box["began"] = n
        began = box.get("began")
        if began is not None and n == began + into:
            box["took"] = n
            pad.throttle = 1.0
            pad.axes["LEFT_X"] = 0.5
        if "took" in box and n == box["took"] + hold:
            pad.throttle = 0.0
            pad.axes["LEFT_X"] = 0.0

    out = run_auto(monkeypatch, passes or 700, Camera(scene()), plan=plan,
                   make_driver=make_driver)
    assert "took" in box, f"{name} never started"
    return out, box["took"]


@pytest.mark.parametrize("name", list(_manoeuvres()))
def test_my_hand_wins_both_groups_on_its_first_pass(monkeypatch, name):
    out, took = run_caught(monkeypatch, name, hold=20)
    s = out["snaps"]
    mine = dv.steering_pair(limited(0.5))
    for i in range(took, took + 20):
        assert (s[i]["steer_by"], s[i]["throttle_by"]) == (mn.PAD, mn.PAD), i
        assert s[i]["throttle"] == pytest.approx(dv.FORWARD_SPEED), i
        assert s[i]["steering"] == pytest.approx(mine), i
    # The pass before was still the manoeuvre's: a reverse, not mine.
    assert s[took - 1]["throttle"] < 0.0


@pytest.mark.parametrize("name", ["rules recover", "policy standoff",
                                  "policy stuck"])
def test_letting_go_mid_recover_hands_back_the_recovers_own_command(
        monkeypatch, name):
    """As before the fix: my hand off the triggers and stick for half a
    second puts the recover's standing reverse and turn back on, not a stop."""
    hold = 10
    out, took = run_caught(monkeypatch, name, hold=hold)
    s = out["snaps"]
    back = took + hold + HALF
    for i in range(took + hold, back):
        assert (s[i]["steer_by"], s[i]["throttle_by"]) == (mn.PAD, mn.PAD), i
        assert s[i]["throttle"] == 0.0, i
    handback = s[back]
    assert (handback["steer_by"], handback["throttle_by"]) == (mn.DRIVER,
                                                               mn.DRIVER)
    assert handback["phase"] == dv._PHASE_REVERSE
    assert handback["throttle"] == pytest.approx(dv.REVERSE_SPEED)
    # The recover's turn, as it stood on the pass before I took it.
    assert handback["steering"] == pytest.approx(s[took - 1]["steering"])
    assert handback["steering"] != pytest.approx((0.0, 0.0))


def test_a_lost_pad_under_the_floor_still_parks_at_neutral(monkeypatch):
    """My forward going through under the floor changes nothing about losing
    the pad: both groups back, auto parked, neutral last."""
    class Dropping(FakePad):
        def is_connected(self):
            return self.frames < 30

    out = run_auto(monkeypatch, 40, Camera(centre_frame(300)), plan=full_rt,
                   pad=Dropping(), make_driver=standing_policy)
    assert out["snaps"][20]["throttle"] == pytest.approx(dv.FORWARD_SPEED)
    assert out["modes"].mode == mn.MANUAL
    assert not out["takeover"].steering and not out["takeover"].throttle
    assert out["vecon"].writes[-1][1] == "neutral"
    assert out["vecon"].last_throttle == 0.0

# --- the safe paths ---------------------------------------------------------


def test_a_lost_pad_gives_both_back_and_parks_at_neutral(monkeypatch):
    class Dropping(FakePad):
        def is_connected(self):
            return self.frames < 30

    def plan(n, pad, cam):
        pad.throttle = 1.0
        pad.axes["LEFT_X"] = 0.5

    out = run_auto(monkeypatch, 40, Camera(far_frame()), plan=plan,
                   pad=Dropping())
    assert out["modes"].mode == mn.MANUAL
    assert not out["takeover"].steering and not out["takeover"].throttle
    assert out["vecon"].writes[-1][1] == "neutral"
    assert out["vecon"].last_throttle == 0.0


@pytest.mark.parametrize("group", ["LEFT_X", "throttle"])
def test_a_nan_off_the_pad_parks_auto_at_neutral(monkeypatch, group):
    def plan(n, pad, cam):
        pad.axes["LEFT_X"] = 0.5
        pad.throttle = 0.5
        if n >= 20:
            if group == "throttle":
                pad.throttle = float("nan")
            else:
                pad.axes["LEFT_X"] = float("nan")

    out = run_auto(monkeypatch, 30, Camera(far_frame()), plan=plan)
    assert out["modes"].mode == mn.MANUAL
    assert out["snaps"][19]["mode"] == mn.AUTO
    assert out["snaps"][20]["mode"] == mn.MANUAL
    vecon = out["vecon"]
    assert (20, "neutral", None) in vecon.writes
    # Nothing auto wrote was a nan. (Manual, after, hands a nan stick to
    # set_steering as it always has, and the controller turns it to neutral.)
    for _, kind, value in [w for w in vecon.writes if w[0] <= 20]:
        values = value if kind == "steering" else (value or 0.0,)
        assert all(math.isfinite(v) for v in values), (kind, value)


def test_a_nan_trigger_never_reaches_the_esc_as_a_limit():
    """Belt and braces under poll(): the scaling itself turns nan into 0."""
    pilot = mn.AutoPilot(FakeVecon(), FakeLogger())
    pilot.driver = dv.Driver(pilot.command, Camera(far_frame()))

    class Held:
        throttle = True
        intent = (float("nan"), 0.0, 0.0)

    assert pilot._pad_throttle(Held()) == 0.0


def test_a_pad_read_that_raises_parks_auto_at_neutral(monkeypatch):
    class Staling(FakePad):
        def read_throttle(self):
            if self.frames > 20:
                raise RuntimeError("stale handle")
            return super().read_throttle()

    out = run_auto(monkeypatch, 30, Camera(far_frame()), pad=Staling(),
                   plan=at(5, throttle=0.5))
    assert out["modes"].mode == mn.MANUAL
    assert out["vecon"].last_throttle == 0.0
    assert not out["takeover"].throttle


def test_start_still_leaves_auto_for_manual_behind_the_latch(monkeypatch):
    """RT held through START: manual holds the throttle at 0, as the latch
    always has, and my takeover is gone with auto."""
    def plan(n, pad, cam):
        pad.throttle = 1.0
        if n == 20:
            pad.press(mn.MODE_TOGGLE_BUTTON)
        elif n == 21:
            pad.release()

    class Held(FakePad):
        def triggers_at_rest(self):
            return False

    out = run_auto(monkeypatch, 40, Camera(far_frame()), plan=plan, pad=Held())
    assert out["modes"].mode == mn.MANUAL
    assert not out["takeover"].throttle
    assert all(snap["throttle"] == 0.0 for snap in out["snaps"][21:])


# --- the rows ---------------------------------------------------------------


def test_every_auto_row_says_whose_each_group_was(monkeypatch):
    """And the command in it is the mix that reached the hardware."""
    def plan(n, pad, cam):
        pad.axes["LEFT_X"] = 0.5 if 20 <= n < 40 else 0.0
        pad.throttle = 0.5 if 90 <= n < 100 else 0.0

    tele = SessionTele()
    out = run_auto(monkeypatch, 160, Camera(graze_frame()), plan=plan,
                   tele=tele, tap=fp.FootprintTap(tele=tele))
    rows = [(row, now) for row, now in tele.of("footprint")]
    assert len(rows) > 10
    by_time = {snap["now"]: snap for snap in out["snaps"]}
    seen = set()
    for row, now in rows:
        snap = by_time[now]
        assert row["steer_by"] == snap["steer_by"]
        assert row["throttle_by"] == snap["throttle_by"]
        assert row["throttle"] == pytest.approx(snap["throttle"])
        assert (row["steering_front"], row["steering_rear"]) == pytest.approx(
            snap["steering"])
        seen.add((row["steer_by"], row["throttle_by"]))
        if row["throttle_by"] == mn.PAD:
            # 0.5 while held, 0 for the half second after letting go.
            assert row["pad_throttle"] in (0.5, 0.0)
            assert row["throttle"] == pytest.approx(row["pad_throttle"] * dv.FORWARD_SPEED)
        # Every row carries auto's caps, the ones no trigger drove included.
        assert row["forward_limit"] == pytest.approx(dv.FORWARD_SPEED)
        assert row["reverse_limit"] == pytest.approx(dv.REVERSE_SPEED)
    assert seen == {(mn.DRIVER, mn.DRIVER), (mn.PAD, mn.DRIVER),
                    (mn.DRIVER, mn.PAD)}


def test_a_correction_during_a_recover_is_still_a_row(monkeypatch):
    """The driver makes no scans while it reverses, so a row of its own would
    be missing exactly when my hand is the label. Ours comes off the tap's
    view instead, at the decision rate."""
    tele = SessionTele()
    tap = fp.FootprintTap(view=FrameView(wall_frame()), tele=tele)
    out = run_auto(monkeypatch, 120, Camera(wall_frame()),
                   plan=at(0, LEFT_X=0.5), tele=tele, tap=tap)
    by_time = {snap["now"]: snap for snap in out["snaps"]}
    during = [(row, now) for row, now in tele.of("footprint")
              if by_time[now]["phase"] is not None]
    # The first is the driver's own: the scan that started the reverse.
    first, start = during.pop(0)
    assert first["source"] == "driver"
    assert during
    times = [now for _, now in during]
    assert times[0] - start >= fp.INTERVAL
    assert all(b - a >= fp.INTERVAL for a, b in zip(times, times[1:]))
    during = [row for row, _ in during]
    assert all(row["source"] == "tap" and row["mode"] == "auto"
               and row["steer_by"] == mn.PAD and not row["depth_raw"]
               for row in during)
    assert all(row["depth_small"] for row in during)
    assert tele.of("depth_raw") == []


def test_no_takeover_no_recover_rows():
    """Without my hand a recover is still not a decision."""
    tele = SessionTele()
    tap = fp.FootprintTap(view=FrameView(wall_frame()), tele=tele)

    class Recovering:
        last_scan = None
        _phase = "reverse"

    for i in range(20):
        tap.after_drive(i * 0.05, fp.AUTO, (0.0, 0.0, 0.0),
                        driver=Recovering())
    assert tele.of("footprint") == []


def test_record_rows_are_the_pad_on_both_groups():
    tele = SessionTele()
    tap = fp.FootprintTap(view=FrameView(far_frame()), tele=tele)
    row = tap.after_drive(0.0, fp.RECORD, (0.1, 0.2, -0.2),
                          pad_throttle=0.5, limits=(0.2, -0.6))
    assert row["steer_by"] == "pad" and row["throttle_by"] == "pad"


def test_the_strings_are_the_same_in_both_files():
    assert (fp.DRIVER, fp.PAD) == (mn.DRIVER, mn.PAD)
