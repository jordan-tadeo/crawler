"""No pad: everything runs, nothing drives.

The service used to wait for the pad, twice: Joystick() at boot sat in
wait_for_connection before the control loop ever started, and the loop parked
in it again whenever the pad went. Either way nothing drained the patrol's
commands and the map page never got a state, so a stroke drawn on the phone
went nowhere (the car's journal had no patrol line, patrol.json was never
written, /api/state said ready false). Now the pad only gates driving: with
none, the loop holds neutral every pass and runs the goal, the patrol, the
planner, the map page and the status channels at PADLESS_INTERVAL.
"""

import json

import pytest

import crawler.__main__ as mn
from crawler.autopilot import patrol as pt
from crawler.nav import click as ck
from crawler.telemetry import buttons as bt

from test_goal_reseed import crop_at
from test_main import FakePad, FakeVecon, TickLimit
from test_record import STEP, BuzzPad, FakeAutopilot, FakeClock, FakeHub, SessionTele
from test_main import FakeLogger


class NoPad(FakePad):
    """Never connected. Its buttons read pressed, the way a stale handle can,
    so a loop that read them with no pad would act on them."""

    def __init__(self, back=None):
        super().__init__()
        self.back = back
        self.down = {"START": True, mn.RECORD_BUTTON: True}
        self.throttle = 1.0
        self.axes = {"LEFT_X": 1.0, "RIGHT_X": 1.0}
        self.tries = 0

    def is_connected(self):
        return self.back is not None and self.frames >= self.back

    def try_connect(self, now):
        self.tries += 1


class PatrolPilot(FakeAutopilot):
    """The AutoPilot's goal side with a real Patrol behind it."""

    def __init__(self, clicks, hub=None):
        super().__init__(hub=hub)
        self.clicks = clicks
        self.patrol = pt.Patrol()
        self.goal = self.patrol
        self.goal_steps = 0

    def goal_step(self):
        self.goal_steps += 1
        self.patrol.poll(self.clicks, crop_at(0.0, 0.0, 0.0))

    def goal_marker(self):
        return {}


def run_padless(monkeypatch, pad, passes, mode=mn.MANUAL, pilot=None, page=None,
                tele=None, clicks=None):
    import pygame

    clock = FakeClock(STEP)

    def pump():
        pad.step()
        clock.t += STEP

    monkeypatch.setattr(pygame.event, "pump", pump)
    monkeypatch.setattr(mn, "LOOP_INTERVAL", 0)
    monkeypatch.setattr(mn, "PADLESS_INTERVAL", 0)
    log = FakeLogger()
    clicks = clicks or ck.ClickGoals()
    pilot = pilot or PatrolPilot(clicks, hub=FakeHub())
    tele = tele or SessionTele()
    vecon = FakeVecon()
    modes = mn.ModeSwitch(mode, log)
    markpad = BuzzPad()
    session = mn.RecordSession(tele, pilot.hub, log, make_recorder=lambda: "rec",
                               markpad=markpad)
    mn.control_loop(pad, vecon, pilot, modes, log, TickLimit(passes), tele, markpad,
                    mn.SteerMode(log), session=session, clock=clock, page=page)
    return {"vecon": vecon, "modes": modes, "pilot": pilot, "log": log, "clock": clock,
            "tele": tele}


def test_with_no_pad_the_patrol_takes_the_pages_commands_and_the_page_gets_its_state(
        monkeypatch):
    clicks = ck.ClickGoals()
    clock = [0.0]
    page = bt.ButtonPage(goals=clicks, clock=lambda: clock[0])
    page.state_json()                       # the page is open and asking
    stroke = {"cmd": "stroke", "points": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], "id": "a-1"}
    assert page.command(json.dumps(stroke).encode()) == (True, "")
    pilot = PatrolPilot(clicks, hub=FakeHub())
    pad = NoPad()
    out = run_padless(monkeypatch, pad, 10, pilot=pilot, page=page, clicks=clicks)

    assert pilot.goal_steps == 10
    assert len(pilot.patrol.points) == 3
    st = json.loads(page.state_json())
    assert st["ready"] is True
    assert st["pad"] is False and st["mode"] == mn.MANUAL
    assert st["patrol"]["acks"] == [["a-1", True]]
    assert len(st["patrol"]["points"]) == 3
    assert pad.tries == 10, "a look for the pad every pass"
    assert out["modes"].mode == mn.MANUAL


def test_with_no_pad_start_arms_the_patrol_and_nothing_drives(monkeypatch):
    """Start from the page arms the patrol; the car drives it only in auto,
    and nothing reads START with no pad. Every pass commands neutral."""
    clicks = ck.ClickGoals()
    clicks.push("cmd", {"cmd": "stroke", "points": [[0.0, 0.0], [3.0, 0.0]]})
    clicks.push("cmd", {"cmd": "start", "id": "s-1"})
    pilot = PatrolPilot(clicks, hub=FakeHub())
    out = run_padless(monkeypatch, NoPad(), 30, pilot=pilot, clicks=clicks)
    assert pilot.patrol.running is True
    assert list(pilot.patrol.acks) == [["s-1", True]]
    v = out["vecon"]
    assert pilot.resumes == 0 and pilot.steps == 0
    assert out["modes"].mode == mn.MANUAL
    assert v.neutrals == 30
    assert set(v.throttles) == {0.0} and set(v.steerings) == {(0.0, 0.0)}


def test_with_no_pad_the_status_channels_and_diagnostics_go_out(monkeypatch):
    tele = SessionTele(channels=("patrol_status", "patrol_marker", "goal_status",
                                 "diagnostics"))
    tele.recording_status = lambda: {}
    clicks = ck.ClickGoals()
    run_padless(monkeypatch, NoPad(), 5, tele=tele, clicks=clicks,
                pilot=PatrolPilot(clicks, hub=FakeHub()))
    keys = {o[0] for o in tele.offers}
    assert {"patrol_status", "patrol_marker", "goal_status", "diagnostics"} <= keys
    diag = [o[1] for o in tele.offers if o[0] == "diagnostics"]
    assert diag and all(d["controller_connected"] is False for d in diag)


def test_a_padless_pass_is_never_a_driving_one(monkeypatch):
    """Auto with no pad (not reachable through the loop, which parks it as
    the pad goes) is parked on the first padless pass."""
    clicks = ck.ClickGoals()
    pilot = PatrolPilot(clicks, hub=FakeHub())
    modes = mn.ModeSwitch(mn.AUTO, FakeLogger())
    vecon = FakeVecon()
    mn.padless_pass(NoPad(), vecon, pilot, modes, SessionTele(), None, 0.0)
    assert modes.mode == mn.MANUAL and pilot.pauses == 1 and pilot.steps == 0
    assert vecon.neutrals == 1


def test_losing_the_pad_in_auto_parks_it_and_holds_neutral_every_pass(monkeypatch):
    from test_pad_loss import DroppingPad

    pad = DroppingPad(gone=20)
    pad.throttle = 1.0
    clicks = ck.ClickGoals()
    pilot = PatrolPilot(clicks, hub=FakeHub())
    out = run_padless(monkeypatch, pad, 60, mode=mn.AUTO, pilot=pilot, clicks=clicks)
    assert out["modes"].mode == mn.MANUAL
    assert pilot.pauses >= 1
    steps_at_loss = pilot.steps
    assert 0 < steps_at_loss < 21, "auto ran with the pad and stopped as it went"
    v = out["vecon"]
    assert v.last_throttle == 0.0 and v.last_steering_front == 0.0
    # The lost-pad block's two neutrals, then one for every padless pass.
    assert v.neutrals >= 2 + 60 - 21
    assert pilot.goal_steps == 60, "the goal ran on with no pad"


def test_the_pad_coming_back_is_manual_and_drives_only_from_then(monkeypatch):
    pad = NoPad(back=10)
    pad.down = {}
    clicks = ck.ClickGoals()
    out = run_padless(monkeypatch, pad, 20, clicks=clicks,
                      pilot=PatrolPilot(clicks, hub=FakeHub()))
    v = out["vecon"]
    assert out["modes"].mode == mn.MANUAL
    assert any(line[2] == "Controller back" for line in out["log"].lines)
    # Padless passes wrote neutral only; manual drives off the sticks after.
    assert v.throttles[:9] == [0.0] * 9


@pytest.mark.parametrize("connected, resumes, mode", [
    (False, 0, mn.MANUAL), (True, 1, mn.AUTO)])
def test_boot_in_auto_needs_the_pad(connected, resumes, mode):
    pad = FakePad()
    pad.is_connected = lambda: connected
    pilot = FakeAutopilot()
    modes = mn.ModeSwitch(mn.AUTO, FakeLogger())
    mn.boot_mode(modes, pad, pilot, FakeLogger())
    assert pilot.resumes == resumes and modes.mode == mode


# --- the Joystick ------------------------------------------------------------


def test_joystick_without_wait_comes_back_at_once_and_try_connect_never_waits(monkeypatch):
    import pygame
    from crawler.hardware import joystick as js

    count = [0]
    monkeypatch.setattr(pygame.joystick, "get_init", lambda: True)
    monkeypatch.setattr(pygame.joystick, "get_count", lambda: count[0])

    def no_wait(*a, **k):
        raise AssertionError("waited for a pad")

    monkeypatch.setattr(js.Joystick, "wait_for_connection", no_wait)
    pad = js.Joystick(wait=False)
    assert pad.is_connected() is False
    assert pad.try_connect(0.0) is False

    class Handle:
        def init(self):
            pass

        def get_axis(self, i):
            return -1.0

    opened = []
    monkeypatch.setattr(pygame.joystick, "Joystick", lambda i: opened.append(i) or Handle())
    monkeypatch.setattr(pygame.event, "pump", lambda: None)
    count[0] = 1
    assert pad.try_connect(0.5) is False, "at most once every CONNECT_EVERY_S"
    assert pad.try_connect(1.0) is True and pad.is_connected() and opened == [0]

    def broken(i):
        raise pygame.error("busy")

    pad2 = js.Joystick(wait=False)
    count[0] = 1
    pad2.connected, pad2.joystick, pad2._tried_at = False, None, None
    monkeypatch.setattr(pygame.joystick, "Joystick", broken)
    assert pad2.try_connect(5.0) is False and pad2.is_connected() is False
