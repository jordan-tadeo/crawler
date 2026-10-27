"""Marking a moment while I am driving, so I can go and look at it afterwards.

Two buttons on the pad. A is "that was the right call", B is "that was wrong".
Press one while the car is doing something interesting and the moment is
written into the MCAP, into the journal and into log/marks.csv, and the pad
buzzes so I know it took. Later, `deploy/mcap_check.py --marks` lists them and
`--window N` dumps everything the car saw around one of them.

Three things about this that are not obvious.

A mark is a WINDOW, not an instant. I press the button after I have seen the
behaviour, judged it and reacted, and the decision that produced the behaviour
came before all of that. A recover is reverse 2.0 s plus settle 0.5 s = 2.5 s,
so a look-back of 4 s covers a whole recovery AND the approach that triggered
it, which is the part I actually want to read. A second of lead covers what
happened straight after. So each mark carries its own window bounds rather
than leaving me to guess them at review time.

A mark is SELF-DESCRIBING. It carries the mode, the commanded throttle and
steering, the footprint scan, the block reason, the recover phase and what the
IMU was saying -- so a row in the CSV is readable on its own and the MCAP
window is for when I want more. Every one of those is read behind its own
guard, because in manual mode there is no driver and no camera and the honest
answer to most of this is "nothing".

Nothing in here is allowed to cost a tick. pigpio latches the last servo pulse
width and emits it forever, so a blocked or crashed control loop is a car that
keeps driving. poll() cannot raise -- its whole body is guarded and a failure
costs the mark and nothing else -- it does no file I/O at all, and the one
pygame call it makes is timed and switched off for the rest of the run if it
ever comes back slow. The CSV row and the journal line go out on a writer
thread of this module's own, so a card that stalls mid-write stalls that thread
and nobody else.
"""

import csv
import os
import queue
import threading
import time

import Logger as lg
import telemetry as tm

# Which button means what.
#
# He asked for this with A and B both described as the good one, which is a
# slip; A is green and means confirm on an Xbox pad and B is red and means
# cancel, so A is the good mark and B is the bad one. Swapping them is these
# two lines and nothing else.
GOOD_BUTTON = "A"
BAD_BUTTON = "B"

GOOD = "good"
BAD = "bad"

# In press order, so a genuinely simultaneous press of both emits a good mark
# and then a bad one rather than one of them at random.
BUTTON_KINDS = {
    GOOD_BUTTON: GOOD,
    BAD_BUTTON: BAD,
}

# The window a mark stands for, in seconds either side of the press.
#
# Look-back: a full recovery cycle is REVERSE_SECONDS 2.0 plus 0.5 s of settle
# = 2.5 s, so 4 s reaches past the whole recovery and into the approach that
# triggered it. That approach is where the tuning lives -- by the time the car
# is reversing, the decision I am judging has already been made.
#
# Lead: one second, which is what it takes to see whether the thing the car did
# next was any better.
MARK_LOOKBACK_SECONDS = 4.0
MARK_LEAD_SECONDS = 1.0

# The human-glanceable copy. The MCAP is the record; this is the one I can cat
# on the car with nothing installed.
MARKS_CSV = "log/marks.csv"

# Column order, written as a header row the first time the file is created and
# never changed afterwards -- the file is append-only across runs, so a reorder
# here would leave one file holding two shapes.
CSV_FIELDS = (
    "utc",
    "kind",
    "button",
    "sequence",
    "mono_s",
    "window_start_s",
    "window_end_s",
    "mode",
    "throttle",
    "steering_front",
    "steering_rear",
    "esc_pulse_us",
    "block_reason",
    "near_distance_mm",
    "scan_blocked",
    "scan_ms",
    "recovery_phase",
    "recovery_remaining_s",
    "motion_evidence",
    "gyro_mean",
    "accel_mean",
    "camera_state",
)

# Rumble. Both motors together at a bit over half, which is a buzz I can feel
# through gloves without it being a noise.
RUMBLE_LOW = 0.6
RUMBLE_HIGH = 0.6

# One pulse for good, two for bad, so the two are distinguishable with my eyes
# on the car. The gap has to be long enough to feel like a gap.
RUMBLE_MS = 160
RUMBLE_GAP_MS = 140
RUMBLE_PULSES = {GOOD: 1, BAD: 2}

# One long pulse: a record session that ended because the pad went quiet or
# went away. Six times a mark's buzz, so it cannot be mistaken for the one
# and two of a mark or of record's own enter and exit.
RUMBLE_ALARM_MS = 1000

# How long a rumble call is allowed to take before I stop making them.
#
# pygame's rumble() wraps SDL_JoystickRumble, which starts an effect and
# returns rather than waiting out the duration -- so this should never fire.
# It is here because "should" is not good enough on this car: the loop runs at
# 100 Hz and a call that sat there for the 160 ms it was asked to play would be
# sixteen ticks with pigpio emitting the last pulse width it was given. One
# slow call switches rumble off for the rest of the run and I lose a buzz.
RUMBLE_SLOW_MS = 20.0

# States the recover phase machine can be in, mirrored out of driver.py as
# strings. Read-only: nothing in here advances it.
NO_PHASE = ""


def _finite(value, fallback=-1.0) -> float:
    """A float that JSON can carry, or the fallback.

    The footprint scan reports inf for a probe that had nothing measurable to
    say, which is the right answer there and not a number any encoder will
    take. -1.0 is the stand-in everywhere below: no real distance is negative,
    so it cannot be confused for one.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    if out != out or out in (float("inf"), float("-inf")):
        return fallback
    return out


def _utc(epoch_nanos: int) -> str:
    """An epoch timestamp as an ISO-8601 Z string, to the second.

    Built off the same monotonic-plus-fixed-offset clock every channel in
    telemetry.py is stamped with, so the wall clock on a mark and the log_time
    of the message carrying it are the same instant by construction. Reading
    time.time() here instead would drift apart from it the moment NTP stepped.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ",
                         time.gmtime(epoch_nanos / 1e9))


class MarkPad:
    """The two mark buttons: edge detection, the snapshot, and the sinks.

    Built once in main() and polled once a tick. The same shape as
    ModeSwitch: a dict of was-it-down-last-tick, and only the press edge
    counts, so holding a button emits exactly one mark.
    """

    def __init__(self, logger=None, tele=None, csv_path=MARKS_CSV):
        self.logger = logger
        self.tele = tele
        self.csv_path = csv_path

        self.good_count = 0
        self.bad_count = 0

        self._was_down = {name: False for name in BUTTON_KINDS}

        # Rumble state. _pulses_left and _next_pulse_at are the whole of the
        # scheduler: poll() runs at 100 Hz anyway, so spacing two pulses is a
        # float compare on a thread that is already awake. There is no sleep
        # anywhere in this file and there is not allowed to be one.
        self._rumble_ok = True
        self._pulses_left = 0
        self._next_pulse_at = 0.0
        # The long pulse. Owed when it could not be delivered -- a pad that
        # has gone cannot feel it -- and paid on the next poll, which is the
        # first pass after the pad is back.
        self._alarm_owed = False

        # Marks waiting for their turn on the /marks channel. The mailbox in
        # telemetry.py is one deep per channel and drop-oldest, which is right
        # for a 20 Hz sample and wrong for an event: two marks offered inside
        # one drain interval would mean the second one replacing the first in
        # the file. So they go out one per tick instead, which only ever
        # matters if I manage to hit A and B on the same tick -- and then the
        # second one is in the file 10 ms late with its own press time intact
        # rather than not in the file at all.
        self._pending = []

        # The off-thread sinks. The thread is started here, in whoever built
        # this -- main(), before the control loop exists -- rather than on the
        # first press, because Thread.start() waits for the new thread to come
        # up and that is not a wait the control loop should ever do. It costs
        # an idle thread on a run where I press nothing; it does not cost a
        # file, because the CSV is only created when there is a row for it.
        self._rows = queue.Queue()
        self._writer = threading.Thread(
            target=self._write_forever, name="marks", daemon=True)
        self._writer.start()

        # Latches, so a failure that happens every tick says so once.
        self._poll_failed = False
        self._write_failed = False

    # ---- the control thread's half ---------------------------------------

    def poll(self, joystick, now, mode, vecon=None, driver=None,
             autopilot=None):
        """Mark on a press edge, and service the rumble. Never raises.

        Returns the marks made this tick, which is almost always none. The
        whole body is inside one guard on purpose: a missing attribute on a
        driver that is mid-teardown, a pad handle that went stale, a telemetry
        mailbox that is gone -- all of it costs the mark, and a mark is not
        worth a tick of the control loop.

        The rumble is serviced after the press and not before it, so the first
        pulse of a new mark goes out on the same tick as the press rather than
        10 ms later. One call covers both jobs: the new mark's first pulse and
        a previous mark's second one.
        """
        try:
            made = []
            for button in self._pressed(joystick):
                kind = BUTTON_KINDS[button]
                mark = self._mark(kind, button, now, mode, vecon, driver,
                                  autopilot)
                if mark is not None:
                    made.append(mark)
            self._offer_one(now)
            self._service_rumble(joystick, now)
            return tuple(made)
        except Exception as e:
            if not self._poll_failed:
                self._poll_failed = True
                self._log("warning", "Mark",
                          f"Dropped a mark and stopped complaining: {e}")
            return ()

    def counts(self) -> dict:
        """The two /diagnostics fields. Two attribute reads, cannot raise."""
        return {
            "marks_good": int(self.good_count),
            "marks_bad": int(self.bad_count),
        }

    def _pressed(self, joystick):
        """Which mark buttons went down this tick. The same edge as ModeSwitch.

        A hold must not retrigger, so the press edge is what counts. A pad
        that is not there yet, or a handle that went stale after a dropout,
        reads as nothing down -- and clears the memory, so the first read off
        a reconnected pad is an edge again rather than a swallowed press.
        """
        pressed = []
        for button in BUTTON_KINDS:
            try:
                down = bool(joystick.get_button(button))
            except Exception:
                self._was_down[button] = False
                continue
            if down and not self._was_down[button]:
                pressed.append(button)
            self._was_down[button] = down
        return pressed

    def _mark(self, kind, button, now, mode, vecon, driver, autopilot):
        """Count it, buzz the pad, publish it, and hand it to the writer."""
        if kind == GOOD:
            self.good_count += 1
        else:
            self.bad_count += 1

        mark = self._snapshot(kind, button, now, mode, vecon, driver,
                              autopilot)

        self._request_rumble(kind, now)

        # Into the queue for /marks, drained one per tick by poll().
        self._pending.append(mark)

        # The CSV row and the journal line, both off this thread. An unbounded
        # queue, so put_nowait cannot block and cannot be full.
        try:
            self._rows.put_nowait(dict(mark))
        except Exception:
            pass

        return mark

    def _offer_one(self, now):
        """Hand the oldest waiting mark to telemetry. Never blocks.

        offer() drops the message into a one-deep mailbox and returns. There
        is no wants() gate in front of it, unlike every sampled channel in the
        control loop: a mark is an event, and a rate gate on an event is a
        lost press.
        """
        if not self._pending:
            return
        tele = self.tele
        if tele is None:
            self._pending.clear()
            return
        mark = self._pending.pop(0)
        try:
            tele.offer("marks", mark, now)
        except Exception:
            pass

    def _snapshot(self, kind, button, now, mode, vecon, driver, autopilot):
        """Everything worth knowing about this instant, each field guarded.

        In manual there is no driver and usually no camera, so most of the
        auto fields come back as their empty value. That is the honest answer
        and it is why marking in manual is still worth having: it says where I
        was and what I was commanding.
        """
        log_time = tm.epoch_ns(now)
        mark = {
            "kind": kind,
            "button": button,
            "sequence": int(self.good_count + self.bad_count),
            "good_count": int(self.good_count),
            "bad_count": int(self.bad_count),
            "utc": _utc(log_time),
            "mono_s": _finite(now, 0.0),
            "window_start_s": _finite(now - MARK_LOOKBACK_SECONDS, 0.0),
            "window_end_s": _finite(now + MARK_LEAD_SECONDS, 0.0),
            "lookback_s": MARK_LOOKBACK_SECONDS,
            "lead_s": MARK_LEAD_SECONDS,
            "mode": str(mode or ""),
            "throttle": 0.0,
            "steering_front": 0.0,
            "steering_rear": 0.0,
            "esc_pulse_us": 0,
            "block_reason": "",
            "near_distance_mm": -1.0,
            "probe_distance_mm": [],
            "probe_blocked": [],
            "scan_blocked": False,
            "scan_ms": 0.0,
            "recovery_phase": NO_PHASE,
            "recovery_remaining_s": 0.0,
            "motion_evidence": "unknown",
            "gyro_mean": -1.0,
            "accel_mean": -1.0,
            "camera_state": "",
        }

        # What the hardware was last told. Recorded by VehicleController as it
        # wrote it, so these are reads of a float and never bus traffic.
        if vecon is not None:
            try:
                mark["throttle"] = _finite(vecon.last_throttle, 0.0)
                mark["steering_front"] = _finite(vecon.last_steering_front, 0.0)
                mark["steering_rear"] = _finite(vecon.last_steering_rear, 0.0)
                mark["esc_pulse_us"] = int(vecon.last_esc_pulse or 0)
            except Exception:
                pass

        if autopilot is not None:
            try:
                mark["camera_state"] = str(autopilot.camera_state())
            except Exception:
                pass

        if driver is not None:
            try:
                mark["block_reason"] = str(driver.last_block_reason or "")
                mark["scan_ms"] = _finite(driver.last_scan_ms, 0.0)
            except Exception:
                pass
            self._add_scan(mark, driver)
            self._add_phase(mark, driver, now)
            self._add_motion(mark, driver)

        return mark

    def _add_scan(self, mark, driver):
        """The footprint the obstacle check last read, per probe and overall."""
        try:
            scan = driver.last_scan
            if scan is None:
                return
            distances = [_finite(v) for v in scan.distance_mm]
            blocked = [bool(v) for v in scan.blocked]
            mark["probe_distance_mm"] = distances
            mark["probe_blocked"] = blocked
            mark["scan_blocked"] = any(blocked)
            real = [d for d in distances if d >= 0.0]
            mark["near_distance_mm"] = min(real) if real else -1.0
        except Exception:
            pass

    def _add_phase(self, mark, driver, now):
        """Which recover phase owns the wheels, if any, and for how much longer.

        driver._phase and _phase_until are the phase machine's own bookkeeping.
        Read, never written: calling recovering() to ask would advance it.
        """
        try:
            phase = getattr(driver, "_phase", None)
            if not phase:
                return
            mark["recovery_phase"] = str(phase)
            until = _finite(getattr(driver, "_phase_until", 0.0), 0.0)
            mark["recovery_remaining_s"] = max(0.0, until - now)
        except Exception:
            pass

    def _add_motion(self, mark, driver):
        """What the IMU is saying, off the windows the driver already keeps.

        Deliberately NOT driver.motion_evidence(): that function appends the
        sample it is given to both smoothing windows, so calling it from here
        would feed the stuck detector an extra sample every time I pressed a
        button. This reads the same two means the same way and decides the
        same question without touching them.
        """
        try:
            import driver as dv

            gyro = list(getattr(driver, "gyro_history", []) or [])
            accel = list(getattr(driver, "imu_history", []) or [])
            if not gyro and not accel:
                return

            gyro_mean = sum(gyro) / len(gyro) if gyro else 0.0
            accel_mean = sum(accel) / len(accel) if accel else 0.0
            mark["gyro_mean"] = _finite(gyro_mean, -1.0)
            mark["accel_mean"] = _finite(accel_mean, -1.0)

            moving = (gyro_mean > dv.GYRO_MOVING_MIN
                      or accel_mean > float(driver.stuck_threshold))
            mark["motion_evidence"] = dv.MOVING if moving else dv.STILL
        except Exception:
            pass

    # ---- rumble ----------------------------------------------------------

    def buzz(self, pulses, now):
        """Queue `pulses` buzzes on the same scheduler marks use. Never raises.

        Here so record mode's enter and exit buzzes go through the one rumble
        scheduler in the process -- two of them would fight over the motor --
        and inherit its slow-call and no-support switches for free. A buzz
        asked for while a mark's pulses are still playing replaces them.
        """
        try:
            if not self._rumble_ok:
                return
            self._pulses_left = int(pulses)
            self._next_pulse_at = now
        except Exception:
            pass

    def alarm(self, joystick, now):
        """One long pulse, now if the pad will take it. Never raises.

        Tried at once, because the loop may be about to park in
        wait_for_connection and stop polling. A pad that has gone does not
        take it, and that is not the pad saying it has no rumble: the attempt
        failing never switches rumble off, it just leaves the pulse owed for
        the first poll after the pad comes back, so I still find out why the
        recording stopped. Whatever short pulses were queued are dropped --
        the one long pulse is the whole message.
        """
        try:
            self._pulses_left = 0
            if not self._rumble_ok:
                return
            took = self._try_alarm(joystick)
            try:
                connected = bool(joystick.is_connected())
            except Exception:
                connected = False
            # A handle that has lost its pad can still say yes, so a pad that
            # is not connected owes it whatever the call returned.
            self._alarm_owed = not (took and connected)
        except Exception:
            pass

    @property
    def alarm_owed(self) -> bool:
        return self._alarm_owed

    def _try_alarm(self, joystick) -> bool:
        """The long pulse, once. True if the pad took it."""
        handle = getattr(joystick, "joystick", None)
        if handle is None:
            return False
        started = time.monotonic()
        try:
            played = handle.rumble(RUMBLE_LOW, RUMBLE_HIGH, RUMBLE_ALARM_MS)
        except Exception:
            return False
        cost = (time.monotonic() - started) * 1000.0
        if cost > RUMBLE_SLOW_MS:
            # The one failure that is about the loop, not the pad.
            self._no_rumble(f"a rumble call cost {cost:.0f} ms, which the "
                            f"control loop cannot pay")
            return True
        return bool(played)

    def _request_rumble(self, kind, now):
        """Queue the pulses for one mark and fire the first of them now."""
        if not self._rumble_ok:
            return
        self._pulses_left = RUMBLE_PULSES.get(kind, 1)
        self._next_pulse_at = now

    def _service_rumble(self, joystick, now):
        """Fire a pulse if one is due. A float compare on an ordinary tick."""
        if self._alarm_owed:
            # poll() only runs on a pass the loop got a pad for, so this is
            # the pad that came back. One go: a pad that is here and still
            # refuses is not going to take it at 100 Hz either.
            self._alarm_owed = False
            if self._rumble_ok:
                self._try_alarm(joystick)
            self._next_pulse_at = now + (RUMBLE_ALARM_MS
                                         + RUMBLE_GAP_MS) / 1000.0
            return
        if self._pulses_left <= 0:
            return
        if now < self._next_pulse_at:
            return
        self._pulses_left -= 1
        self._next_pulse_at = now + (RUMBLE_MS + RUMBLE_GAP_MS) / 1000.0
        self._fire_rumble(joystick)

    def _fire_rumble(self, joystick):
        """One buzz. Any reason it does not work switches it off for the run."""
        if not self._rumble_ok:
            return
        handle = getattr(joystick, "joystick", None)
        if handle is None:
            return

        started = time.monotonic()
        try:
            played = handle.rumble(RUMBLE_LOW, RUMBLE_HIGH, RUMBLE_MS)
        except Exception as e:
            self._no_rumble(f"the pad would not take a rumble ({e})")
            return

        cost = (time.monotonic() - started) * 1000.0
        if cost > RUMBLE_SLOW_MS:
            self._no_rumble(f"a rumble call cost {cost:.0f} ms, which the "
                            f"control loop cannot pay")
            return

        if not played:
            self._no_rumble("this pad reports no rumble support")

    def _no_rumble(self, why):
        """Give up on rumble, once, out loud. Marks carry on regardless."""
        self._rumble_ok = False
        self._pulses_left = 0
        self._log("info", "Mark", f"Marking without a buzz: {why}")

    # ---- the writer thread's half ----------------------------------------

    def _write_forever(self):
        """Sit on the queue and write whatever lands in it.

        The one place in this file that is allowed to touch the card. A stall
        here is a row arriving late, which is what the queue is for; on the
        control thread the same stall would be the throttle waiting on a
        filesystem.
        """
        while True:
            row = self._rows.get()
            try:
                if row is None:
                    return
                self._append(row)
                self._journal(row)
            except Exception as e:
                if not self._write_failed:
                    self._write_failed = True
                    self._log("warning", "Mark",
                              f"Could not write the mark down: {e}")
            finally:
                self._rows.task_done()

    def _append(self, row):
        """One row onto log/marks.csv, header first if the file is new.

        Append-only and never rotated. A mark is about a hundred bytes and I
        press the button a handful of times a run, so this file is measured in
        kilobytes a year -- the rotation Logger.csv does would only ever lose
        me the history.
        """
        path = lg.log_path(self.csv_path)
        fresh = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as handle:
            writer = csv.writer(handle)
            if fresh:
                writer.writerow(CSV_FIELDS)
            writer.writerow([row.get(field, "") for field in CSV_FIELDS])
            handle.flush()

    def _journal(self, row):
        """One INFO line, so a mark is findable with journalctl and no tools."""
        self._log("info", "Mark", describe(row))

    # ---- odds and ends ---------------------------------------------------

    def flush(self, timeout=2.0):
        """Wait for the queued rows to be written. For tests and shutdown."""
        # unfinished_tasks is puts minus task_done()s, so it reaches zero only
        # once the last row is on the card -- which empty() would say a whole
        # write earlier.
        deadline = time.monotonic() + timeout
        while self._rows.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)

    def stop(self):
        """Let the writer thread finish its queue and go. Safe to call twice.

        Called from the control loop's own finally, after the ESC is back at
        neutral. A mark still in the queue at SIGTERM is the only thing in
        this process that shutdown could lose, and it is worth the two
        seconds -- but only after the wheels are dealt with, never before.
        """
        writer, self._writer = self._writer, None
        if writer is None:
            return
        try:
            self._rows.put_nowait(None)
            writer.join(timeout=2.0)
        except Exception:
            pass

    def _log(self, level, subject, message):
        if self.logger is not None:
            try:
                self.logger.log(level, subject, message)
                return
            except Exception:
                pass
        print(f"[{subject.lower()}] {message}")


def describe(row) -> str:
    """One line for a mark. The journal line and the review tool share it."""
    kind = str(row.get("kind", "")).upper()
    bits = [
        f"{kind} #{row.get('sequence', 0)}",
        f"at {row.get('utc', '')}",
        f"in {row.get('mode', '')}",
        "window [{:+.1f}s, {:+.1f}s]".format(
            -float(row.get("lookback_s", MARK_LOOKBACK_SECONDS) or 0.0),
            float(row.get("lead_s", MARK_LEAD_SECONDS) or 0.0)),
        "throttle {:.2f}".format(float(row.get("throttle", 0.0) or 0.0)),
    ]
    reason = row.get("block_reason", "")
    if reason:
        bits.append(f"blocked {reason}")
    near = float(row.get("near_distance_mm", -1.0) or -1.0)
    if near >= 0.0:
        bits.append(f"near {near:.0f} mm")
    phase = row.get("recovery_phase", "")
    if phase:
        bits.append(f"recovering {phase}")
    motion = row.get("motion_evidence", "")
    if motion:
        bits.append(str(motion))
    return ", ".join(bits)
