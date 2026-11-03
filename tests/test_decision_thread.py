"""The decisions off the control loop: the fixed phase, the hold, the restarts.

AutoPilot makes its decisions on a thread of its own and the control loop
only writes them (main.AutoPilot, "the decision thread"). These hold:

  the phase is fixed: a late pass makes a late decision, never a later
  phase, and a long stall starts the phase again instead of a burst;
  record rows are on the same phase;
  a decision older than DECISION_STALE_S holds the autopilot's wheels at
  neutral and the car stays in auto, and the first fresh one drives again,
  with one journal line each way;
  a tick that raises is restarted at most once a second, still in auto, and
  after DECISION_MAX_RESTARTS in the window it is given up on, loudly, still
  in auto;
  my hand is read and written on the pass it moves, whatever the decisions
  are doing, and losing the pad still parks auto;
  real threads, random tick times: every pass reaches its write, nothing
  but neutral is left latched once the newest decision is stale, and the
  mode never moves.

The deterministic ones run the decisions EXTERNAL: the test makes each
decision on the pass it picks, which is the decision thread's iteration
with the thread taken out. The threaded ones use real time, a few seconds.
"""

import random
import sys
import threading
import time

import numpy as np
import pytest

from crawler.autopilot import driver as dv
from crawler.autopilot import footprint as fp
from crawler.autopilot import policy_input
import crawler.__main__ as mn

from test_main import FakeLogger, FakePad, FakeVecon, TickLimit
from test_record import FakeClock, FrameView, SessionTele, far_frame

STEP = 1.0 / 64


class StubDriver:
    """What AutoPilot needs of a driver: a tick that asks for a command."""

    forward_speed = 0.2
    reverse_speed = -0.25

    def __init__(self, controller, throttle=0.2, steer=0.3, delay=None,
                 raises=None):
        self.controller = controller
        self.throttle = throttle
        self.steer = steer
        self.delay = delay          # callable -> seconds, or None
        self.raises = raises        # callable(n) -> bool
        self.ticks = []
        self.aborts = 0
        self.human_throttle = False
        self.human_steering = False
        self.pad_throttle = None
        self.pad_split = False
        self.last_scan = None
        self._phase = None

    def tick(self, now):
        n = len(self.ticks)
        self.ticks.append(now)
        if self.delay is not None:
            seconds = self.delay()
            if seconds > 0:
                time.sleep(seconds)
        if self.raises is not None and self.raises(n):
            raise RuntimeError(f"tick {n} blew up")
        self.controller.set_steering(self.steer, -self.steer)
        self.controller.set_throttle(self.throttle)

    def abort(self):
        self.aborts += 1
        self.controller.set_throttle(0)


class Watch:
    """A MarkPad stand-in: what the wheels and the mode were after each
    pass's write, and the newest decision the loop could have seen."""

    def __init__(self):
        self.passes = []
        self.pilot = None

    def poll(self, joystick, now, mode, vecon, driver, autopilot):
        seq, at = autopilot.published.peek()
        self.passes.append({"now": now, "mode": mode, "seq": seq, "at": at,
                            "throttle": vecon.last_throttle,
                            "steering": (vecon.last_steering_front,
                                         vecon.last_steering_rear),
                            "holding": autopilot._holding,
                            "status": autopilot.decision_status(now),
                            "wall": time.monotonic()})

    def buzz(self, pulses, now):
        pass

    def counts(self):
        return {"marks_good": 0, "marks_bad": 0}

    def stop(self):
        pass


class DiagTele(SessionTele):
    def __init__(self):
        super().__init__(channels=("diagnostics",))


def run_loop(monkeypatch, passes, decisions, plan=None, driver=None,
             clock=None, pad=None, loop_interval=0, tele=None):
    """control_loop in auto over a real AutoPilot and a StubDriver.

    `plan(i, pad, pilot, now)` runs inside pass i's pump, before anything
    is read: the hand on the pad, and (EXTERNAL) the decisions made."""
    import pygame

    pad = pad or FakePad()
    fake = clock is None
    clock = clock or FakeClock(STEP)
    holder = {}

    def pump():
        if plan is not None:
            plan(pad.frames, pad, holder["pilot"], clock())
        pad.step()
        if fake:
            clock.t += STEP

    monkeypatch.setattr(pygame.event, "pump", pump)
    monkeypatch.setattr(mn, "LOOP_INTERVAL", loop_interval)

    log = FakeLogger()
    vecon = FakeVecon()
    pilot = mn.AutoPilot(vecon, log, decisions=decisions, clock=clock)
    holder["pilot"] = pilot
    pilot.driver = driver = driver or StubDriver(pilot.command)
    driver.controller = pilot.command
    assert pilot.resume()
    watch = Watch()
    takeover = mn.Takeover(log, watch)
    tele = tele or SessionTele()
    modes = mn.ModeSwitch(mn.AUTO, log)
    mn.control_loop(pad, vecon, pilot, modes, log, TickLimit(passes), tele,
                    watch, mn.SteerMode(log),
                    session=mn.RecordSession(tele, None, log),
                    tap=fp.FootprintTap(tele=tele), clock=clock,
                    takeover=takeover)
    # control_loop's shutdown stops the AutoPilot (thread joined, driver
    # let go), so the driver is the one it started with.
    return {"pilot": pilot, "vecon": vecon, "log": log, "passes": watch.passes,
            "modes": modes, "driver": driver, "tele": tele}


def lines(log, text):
    return [line for line in log.lines if text in line[2]]


# --- the fixed phase ------------------------------------------------------------


def test_the_phase_holds_a_tenth_of_a_second_under_jitter():
    """Passes 5-40 ms apart at random: every decision on its slot of the
    phase, never early, never a slot missed, 600 in a minute."""
    rng = random.Random(7)
    due, now, ticks = 0.0, 0.0, []
    while now < 60.0:
        if now >= due:
            ticks.append(now)
            due = dv.next_decision_at(due, now)
        now += rng.uniform(0.005, 0.040)
    # The first decision anchors the phase; every one after is on it.
    t0 = ticks[0]
    for k, t in enumerate(ticks):
        assert t0 + k * 0.1 - 1e-9 <= t < t0 + k * 0.1 + 0.041, (k, t)
    assert len(ticks) == 600
    # The old scheduler on the same passes drifted to under 9 Hz.
    rng = random.Random(7)
    nxt, now, old = 0.0, 0.0, 0
    while now < 60.0:
        if now >= nxt:
            old += 1
            nxt = now + 0.1
        now += rng.uniform(0.005, 0.040)
    assert old < 560


def test_a_long_stall_starts_the_phase_again_and_fires_no_burst():
    due = dv.next_decision_at(0.0, 10.0)
    assert due == pytest.approx(10.1)
    # One period late: the next is still on the phase.
    assert dv.next_decision_at(10.1, 10.25) == pytest.approx(10.2)
    # A second late: from here, one interval on.
    assert dv.next_decision_at(10.1, 11.1) == pytest.approx(11.2)
    passes = [10.0 + STEP * k for k in range(20)] + [12.0 + STEP * k for k in range(40)]
    due, ticks = 0.0, []
    for now in passes:
        if now >= due:
            ticks.append(now)
            due = dv.next_decision_at(due, now)
    after = [t for t in ticks if t >= 12.0]
    assert after[0] == 12.0
    assert all(b - a >= 0.1 - STEP for a, b in zip(after, after[1:]))


def test_inline_auto_decides_ten_times_a_second(monkeypatch):
    """Ten seconds of 1/64 s passes: 100 decisions, where now + 0.1 made one
    every seventh pass, 91."""
    out = run_loop(monkeypatch, 640, mn.INLINE)
    ticks = out["driver"].ticks
    assert len(ticks) == 100
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert all(0.1 - STEP <= g <= 0.1 + STEP for g in gaps)


def test_record_rows_are_on_the_fixed_phase():
    """Record rows off passes 5-40 ms apart: 300 in 30 s, each on its slot."""
    rng = random.Random(3)
    tele = SessionTele()
    tap = fp.FootprintTap(view=FrameView(far_frame()), tele=tele)
    now = 0.0
    while now < 30.0:
        tap.after_drive(now, fp.RECORD, (0.1, 0.0, 0.0), limits=(0.2, -0.2))
        now += rng.uniform(0.005, 0.040)
    times = [at for key, _, at in tele.offers if key == "footprint"]
    assert len(times) == 300
    for k, t in enumerate(times):
        assert k * 0.1 - 1e-9 <= t < k * 0.1 + 0.041


# --- a stalled decision holds neutral, in auto ----------------------------------


def every_due(pilot, now):
    if pilot._due(now):
        pilot.decide(now)


def stalled(between):
    """EXTERNAL decisions on the phase, none on passes in `between`."""
    lo, hi = between

    def plan(n, pad, pilot, now):
        if not lo <= n < hi:
            every_due(pilot, now)
    return plan


def test_a_stalled_decision_holds_neutral_stays_in_auto_and_resumes(monkeypatch):
    """Decisions to pass 40, none to 140, then on again. Neutral from the
    pass the newest goes past 0.3 s old, every pass; auto the whole way; the
    driver's command back on the first fresh one. One line each way."""
    tele = DiagTele()
    out = run_loop(monkeypatch, 220, mn.EXTERNAL, plan=stalled((40, 140)),
                   tele=tele)
    p = out["passes"]
    assert all(s["mode"] == mn.AUTO for s in p)
    assert out["modes"].mode == mn.AUTO
    last = max(s["at"] for s in p[:40] if s["at"] is not None)
    for s in p[40:140]:
        if s["now"] - last > mn.DECISION_STALE_S:
            assert s["throttle"] == 0.0 and s["steering"] == (0.0, 0.0), s
            assert s["holding"] == mn.DECISIONS_LATE
        else:
            assert s["throttle"] == pytest.approx(0.2)
    held = [s for s in p[40:140] if s["holding"]]
    assert held and held[0]["now"] - last <= mn.DECISION_STALE_S + STEP
    # The first pass with a fresh decision drives it, both groups.
    back = p[140]
    assert back["throttle"] == pytest.approx(0.2)
    assert back["steering"] == pytest.approx((0.3, -0.3))
    assert back["holding"] is None
    assert len(lines(out["log"], "Decisions late: holding neutral")) == 1
    assert len(lines(out["log"], "Decisions back")) == 1
    assert out["pilot"].late_episodes == 1
    diag = [payload for key, payload, _ in tele.offers if key == "diagnostics"]
    assert diag[-1]["decisions_late_episodes"] == 1
    assert any(d["decisions_holding"] for d in diag)
    assert diag[-1]["decisions_state"] == mn.DECISIONS_OK


def test_a_hold_writes_the_throttle_neutral_on_every_pass(monkeypatch):
    out = run_loop(monkeypatch, 120, mn.EXTERNAL, plan=stalled((20, 10 ** 6)))
    vecon = out["vecon"]
    held = [s for s in out["passes"] if s["holding"]]
    assert len(held) > 50
    # One zero throttle written on each held pass, nothing else.
    assert vecon.throttles[-len(held):] == [0.0] * len(held)


def test_my_hand_still_drives_through_a_hold(monkeypatch):
    """Stalled from pass 10, and my steering from pass 60: mine on 60, the
    throttle still the autopilot's held neutral."""
    def plan(n, pad, pilot, now):
        if n < 10:
            every_due(pilot, now)
        if n == 60:
            pad.axes["LEFT_X"] = 0.6

    out = run_loop(monkeypatch, 90, mn.EXTERNAL, plan=plan)
    want = dv.steering_pair(mn.stick_steer(0.6))
    for s in out["passes"][60:]:
        assert s["steering"] == pytest.approx(want)
        assert s["throttle"] == 0.0
        assert s["mode"] == mn.AUTO


def test_no_first_decision_is_a_hold_too(monkeypatch):
    """A thread that never answers at all after START: held and said."""
    out = run_loop(monkeypatch, 60, mn.EXTERNAL)
    assert out["passes"][-1]["holding"] == mn.DECISIONS_LATE
    assert out["modes"].mode == mn.AUTO
    assert len(lines(out["log"], "Decisions late: holding neutral")) == 1


# --- a decision that raises -----------------------------------------------------


def test_a_tick_that_raises_is_restarted_and_auto_stays(monkeypatch):
    """Tick 5 raises: neutral at once, auto, a restart a second later, and
    the decisions back."""
    holder = {}

    def plan(n, pad, pilot, now):
        if "d" not in holder:
            holder["d"] = pilot.driver = StubDriver(pilot.command,
                                                    raises=lambda k: k == 5)
        every_due(pilot, now)

    out = run_loop(monkeypatch, 260, mn.EXTERNAL, plan=plan)
    p = out["passes"]
    d = holder["d"]
    crashed_at, restarted_at = d.ticks[5], d.ticks[6]
    assert all(s["mode"] == mn.AUTO for s in p)
    during = [s for s in p if crashed_at <= s["now"] < restarted_at]
    assert during and all(s["throttle"] == 0.0 for s in during)
    assert during[0]["holding"] == mn.DECISIONS_DOWN
    assert out["pilot"].decision_restarts == 1
    # The restart is a second after the start, and drives again.
    assert d.ticks[6] - d.ticks[0] >= mn.DECISION_RESTART_S - STEP
    assert p[-1]["throttle"] == pytest.approx(0.2) and p[-1]["holding"] is None
    assert len(lines(out["log"], "Decisions crashed")) == 1
    assert len(lines(out["log"], "Decisions restarted")) == 1
    assert len(lines(out["log"], "Decisions back")) == 1
    assert d.aborts >= 1, "a restart starts from a clean slate"


def test_a_tick_that_keeps_raising_is_given_up_on_still_in_auto(monkeypatch):
    holder = {}

    def plan(n, pad, pilot, now):
        if "d" not in holder:
            holder["d"] = pilot.driver = StubDriver(pilot.command,
                                                    raises=lambda k: True)
        every_due(pilot, now)

    passes = int(8.0 / STEP)
    out = run_loop(monkeypatch, passes, mn.EXTERNAL, plan=plan)
    pilot = out["pilot"]
    assert pilot._state == mn.DECISIONS_GAVE_UP
    assert pilot.decision_restarts == mn.DECISION_MAX_RESTARTS
    assert all(s["mode"] == mn.AUTO for s in out["passes"])
    assert all(s["throttle"] == 0.0 for s in out["passes"])
    errors = [line for line in out["log"].of("error") if "keep crashing" in line[2]]
    assert len(errors) == 1 and "START" in errors[0][2]
    status = out["passes"][-1]["status"]
    assert status["state"] == mn.DECISIONS_GAVE_UP and "keep crashing" in status["why"]
    # The map page carries it (engaged, as it was before the loop shut down).
    pilot._active = True
    pilot.patrol = type("P", (), {"page_state": lambda self: {}})()
    page = mn.map_page_state(pilot, mode=mn.AUTO, now=1.0)
    assert page["decisions"]["state"] == mn.DECISIONS_GAVE_UP
    # START round again: trying again.
    pilot.pause()
    pilot.driver = holder["d"]
    assert pilot.resume() and pilot._state == mn.DECISIONS_OK


def test_inline_a_raising_tick_is_no_longer_manual(monkeypatch):
    """The old step() dropped to manual on any raise. Not any more."""
    holder = {}

    def plan(n, pad, pilot, now):
        if "d" not in holder:
            holder["d"] = pilot.driver = StubDriver(pilot.command,
                                                    raises=lambda k: k == 3)

    out = run_loop(monkeypatch, 200, mn.INLINE, plan=plan)
    assert out["modes"].mode == mn.AUTO
    assert not out["pilot"].unavailable
    assert out["pilot"].decision_restarts == 1


# --- the pad, whatever the decisions are doing ----------------------------------


def test_takeover_is_written_on_the_pass_my_hand_moves_whatever_the_tick(monkeypatch):
    """Real thread, real clock, a tick that takes 300 ms holding the GIL.
    My stick is written on the very pass that read it, within a few ms."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(mn.NAV_SWITCH_INTERVAL_S)

    def busy():
        end = time.perf_counter() + 0.3
        while time.perf_counter() < end:
            pass
        return 0.0

    moved = {}

    def plan(n, pad, pilot, now):
        if n == 40:
            pad.axes["LEFT_X"] = 0.6
            moved["wall"] = time.monotonic()
            moved["n"] = n

    try:
        out = run_loop(monkeypatch, 80, mn.THREAD, plan=plan,
                       driver=StubDriver(None, delay=busy),
                       clock=time.monotonic, loop_interval=0.01)
    finally:
        sys.setswitchinterval(old)
    want = dv.steering_pair(mn.stick_steer(0.6))
    s = out["passes"][moved["n"]]
    assert s["steering"] == pytest.approx(want)
    assert s["wall"] - moved["wall"] < 0.05
    assert out["driver"].ticks, "the slow driver did tick"
    assert out["modes"].mode == mn.AUTO


def test_losing_the_pad_still_parks_auto(monkeypatch):
    from test_pad_loss import DroppingPad
    monkeypatch.setattr(mn, "PADLESS_INTERVAL", 0)
    out = run_loop(monkeypatch, 60, mn.EXTERNAL,
                   plan=lambda n, pad, pilot, now: every_due(pilot, now),
                   pad=DroppingPad(gone=30))
    assert out["modes"].mode == mn.MANUAL
    assert out["vecon"].last_throttle == 0.0
    assert out["driver"].aborts >= 1
    assert not out["pilot"]._active


def test_pause_never_waits_for_a_tick_in_progress():
    """A tick holding the driver lock: pause() writes neutral and returns at
    once, and the tick's abort lands as it lets go."""
    vecon = FakeVecon()
    pilot = mn.AutoPilot(vecon, FakeLogger(), decisions=mn.EXTERNAL)
    gate = threading.Event()
    pilot.driver = StubDriver(pilot.command, delay=lambda: 0.0)
    assert pilot.resume()

    def tick(now):
        gate.wait(2.0)
    pilot.driver.tick = tick
    t = threading.Thread(target=pilot.decide, args=(0.0,))
    t.start()
    time.sleep(0.05)
    started = time.perf_counter()
    pilot.pause()
    assert time.perf_counter() - started < 0.02
    assert vecon.neutrals == 1
    assert pilot.driver.aborts == 0
    gate.set()
    t.join(2.0)
    assert pilot.driver.aborts == 1
    # And the parked tick published nothing.
    assert pilot.published.peek() == (0, None)


# --- the threads, for real ------------------------------------------------------


def test_threads_for_real_random_tick_times(monkeypatch):
    """Three seconds of the real decision thread, ticks of 0-450 ms (some
    sleeping, some holding the GIL), throttle changing each one. Every pass
    gets to its write; once the newest decision is stale, the throttle on
    the wheels is 0; auto the whole way."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(mn.NAV_SWITCH_INTERVAL_S)
    rng = random.Random(11)

    def delay():
        seconds = rng.choice([0.0, 0.02, 0.06, 0.12, 0.2, 0.45])
        if rng.random() < 0.5:
            end = time.perf_counter() + seconds
            while time.perf_counter() < end:
                pass
            return 0.0
        return seconds

    class Changing(StubDriver):
        def tick(self, now):
            self.throttle = rng.uniform(0.1, 0.3)
            super().tick(now)

    writes = []
    try:
        driver = Changing(None, delay=delay)
        orig_write = mn.AutoPilot._write

        def counted(self, now, takeover, steering_mode=mn.UNIFIED):
            writes.append(now)
            return orig_write(self, now, takeover, steering_mode)
        monkeypatch.setattr(mn.AutoPilot, "_write", counted)
        out = run_loop(monkeypatch, 300, mn.THREAD, driver=driver,
                       clock=time.monotonic, loop_interval=0.01)
    finally:
        sys.setswitchinterval(old)
    p = out["passes"]
    assert len(writes) == len(p) == 300
    assert all(s["mode"] == mn.AUTO for s in p)
    assert out["modes"].mode == mn.AUTO
    stale = 0
    for s in p:
        since = s["at"] if s["at"] is not None else p[0]["now"]
        if s["now"] - since > mn.DECISION_STALE_S:
            stale += 1
            assert s["throttle"] == 0.0, s
    gaps = [b["wall"] - a["wall"] for a, b in zip(p, p[1:])]
    assert max(gaps) < 0.1, max(gaps)
    assert len(driver.ticks) > 5
    # 450 ms ticks were drawn, so there was something to hold.
    assert stale > 0 and out["pilot"].late_episodes >= 1


def test_a_thread_that_raises_is_started_again_for_real(monkeypatch):
    driver = StubDriver(None, raises=lambda n: n == 2)
    out = run_loop(monkeypatch, 220, mn.THREAD, driver=driver,
                   clock=time.monotonic, loop_interval=0.01)
    assert out["pilot"].decision_restarts == 1
    assert out["modes"].mode == mn.AUTO
    assert len(driver.ticks) > 4
    assert out["passes"][-1]["holding"] is None
    assert out["passes"][-1]["throttle"] == pytest.approx(0.2)


# --- 1e: one depth_small a decision -----------------------------------------------


def test_the_row_takes_the_policys_picture_of_the_same_frame():
    depth = far_frame(1500)
    small = policy_input.depth_small(depth)

    class Driver:
        last_small = (depth, small, policy_input.depth_small)

    assert fp.policy_small(Driver(), depth) is small
    # Not the same frame, or not the recording's downsample: shrunk afresh.
    assert fp.policy_small(Driver(), depth.copy()) is None

    class Legacy:
        last_small = (depth, small, lambda f: f)
    assert fp.policy_small(Legacy(), depth) is None


def test_a_policy_tick_leaves_its_picture_for_the_row():
    from test_policy import FORWARD, StubModel
    from crawler.autopilot import policy as pl
    from conftest import FakeController

    class Cam:
        def __init__(self):
            self.frame = far_frame()

        def get_depth_frame(self):
            return self.frame

        def get_imu_sample(self):
            return None

    cam = Cam()
    model = StubModel(FORWARD)
    model.contract = pl.CONTRACT_DEPTH_SMALL
    d = pl.PolicyDriver(FakeController(), cam, model)
    d.tick(1.0)
    src, small, fn = d.last_small
    assert src is cam.frame and fn is policy_input.depth_small
    assert np.array_equal(small, policy_input.depth_small(cam.frame))
