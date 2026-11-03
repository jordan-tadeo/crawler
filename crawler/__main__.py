"""Entry point for the crawler.

Three drive modes:

  manual  the Xbox controller drives throttle and both steering axles. Default.
  auto    the OAK-D roomba: driver.Driver crawls forward on stereo depth and
          backs out of whatever it runs into.
  record  manual, with the triggers scaled down to auto's own speeds, the
          camera open and the run written to one MCAP of its own -- with the
          autopilot's footprint features next to my command on every decision
          (/footprint). Training data for a learned steering policy. Hold Y
          for a second in manual to enter it, tap Y to leave. footprint.py has
          the rest, including why nothing it computes can reach the wheels.

Press START on the controller to switch between manual and auto without
restarting. START in record ends the recording and goes to auto.

In auto either control group can be taken off the autopilot by moving it:
the steering stick(s) or the triggers. The autopilot keeps the other, and gets
a group back after it has sat in its deadzone for half a second. That is how I
correct it, and the corrections are labelled in the recording. See Takeover.

In manual the left stick steers both axles counter-phase, through the same
driver.steering_pair() every auto steering decision goes through -- one
definition of how this car steers, not one per mode. Click the left stick (L3)
to split the axles across the two sticks instead, each independent, which is
the escape hatch for a crab walk or for posing the car on an obstacle. It is a
toggle, not a hold, and it survives a lap in auto.

A marks the moment as a good call and B marks it as a bad one, in either mode.
That is for the review loop and not for driving: see marks.py.

One thread owns the ESC and the servos: the control loop. Auto's decisions
are made on a thread of their own (AutoPilot, "the decision thread"), which
only ever publishes a command; the write is a step() call on this loop, so a
mode switch can never leave two owners writing opposing commands to the same
hardware, and a slow decision never holds up the pad or a servo write. A
decision that is late holds the autopilot's wheels at neutral, in auto.

The heavy imports (depthai, PyQt5) are deferred on purpose: manual driving is
the mode that has to come up on a headless boot, and it must not depend on
the vision stack being installed or the camera being plugged in.

`driver` is imported up here rather than deferred, which is a change: manual
steering now calls driver.steering_pair(), so the module has to be here before
the first tick. driver imports numpy and nothing else -- no depthai, no device
-- and numpy costs 246 ms once on the car's own interpreter, measured, against
a boot that already waits on pigpiod. A missing numpy is then a loud failure at
startup instead of a per-tick exception that quietly held the wheels straight.

The thing to keep in mind everywhere below: pigpio latches a servo pulse
width and emits it forever. There is no hardware failsafe on this car and no
RC receiver to fall back on. A crashed thread, a blocked loop or a SIGKILL
does not stop it -- so every error path in here has to end at neutral, and
nothing on this thread is allowed to block.
"""

import argparse
import math
import os
import signal
import sys
import threading
import time
import traceback

from crawler.hardware import camera
from crawler.autopilot import driver as dv
from crawler.autopilot import goal
from crawler.autopilot import patrol
from crawler.telemetry import logger as lg
from crawler.telemetry import tags as tg

MANUAL = "manual"
AUTO = "auto"
RECORD = "record"

# Button that flips between the two modes. START and BACK are the two buttons
# that cannot be hit by accident while driving.
MODE_TOGGLE_BUTTON = "START"

# How the two sticks map onto the two steering axles.
#
#   unified  the left stick alone, both axles, counter-phase, through
#            driver.steering_pair(). The default, and the same arithmetic auto
#            drives on -- a hallway corner needs a radius the front axle alone
#            does not have, and that is as true with my hand on the stick as
#            it is under the camera.
#   split    left stick front, right stick rear, each independent and no ratio
#            applied. Crab walking and odd poses, which is exactly the thing a
#            ratio would take away.
UNIFIED = "unified"
SPLIT = "split"

# L3, the left stick click. Nothing else on the pad reads it.
#
# A toggle rather than a hold, on purpose: pressing down on the same stick I am
# trying to deflect precisely is awkward, and split is a mode I go into
# deliberately and stay in while I position the car, not something I dab at for
# half a second.
STEER_SPLIT_BUTTON = "LS"

# Y, the record button. Held for RECORD_HOLD_SECONDS in manual it enters
# record; any press in record leaves. A hold to get in so a thumb brushing it
# does not start a camera and a file; a tap to get out because leaving is the
# safe direction and should never be hard. Nothing else on the pad reads Y.
RECORD_BUTTON = "Y"
RECORD_HOLD_SECONDS = 1.0

# --goal's patrol waypoints, kept across a restart (autopilot/patrol.py says
# when they come back). Under log/, which is untracked.
PATROL_FILE = "log/patrol.json"
# The patrol's settings the map page sets (the stuck-waypoint skip).
PATROL_SETTINGS_FILE = "log/patrol_settings.json"

# --slam's map, saved as it goes for a record of the run and loaded at boot
# only with --slam-map-restore (mapping/mapfile.py says why). Under log/
# too: a map is a floor plan of the house, never for the repo.
SLAM_MAP_FILE = "log/slam_map.npz"

# The camera view the autopilot's driver reads. The footprint tap reads the
# same one (never takes from it) for the frame behind each auto decision.
AUTO_VIEW = "auto"

# A and B are the two mark buttons. Which is which, and everything about what
# a mark is, lives in marks.py -- this file only polls it.

LOOP_INTERVAL = 0.01
# With no pad the loop runs at 10 Hz: the goal, the patrol's commands, the
# map page and the status channels, the wheels held at neutral every pass.
PADLESS_INTERVAL = 0.1
AUTO_TICK_INTERVAL = 0.1

# The decisions, off the control loop (AutoPilot, "the decision thread").
#
# DECISION_STALE_S: how old the newest decision may be before the wheels the
# autopilot owns are held at neutral, in auto, until a fresh one lands. Three
# decisions' worth, as driver.FLOOR_SCAN_MAX_AGE is three ticks of picture
# age. A decision's age is from the `now` it was made at, so a tick that
# takes 150 ms on the fixed phase is 0.25 s old at worst and holds nothing; a
# 455 ms tick (the worst measured) is a hold. Late is never manual: that is
# the owner's ruling, and the pad is the way to take the car.
DECISION_STALE_S = 0.3
# A decision thread that raised or died is started again at most this often,
# the wheels held at neutral in auto meanwhile...
DECISION_RESTART_S = 1.0
# ...and after this many restarts inside DECISION_RESTART_WINDOW_S, no more:
# neutral, still auto, said in the journal and on the map page, until START
# goes round again (out of auto and back in).
DECISION_MAX_RESTARTS = 3
DECISION_RESTART_WINDOW_S = 10.0
# How long the decision thread waits for the control loop to write its
# command before the /footprint row goes without it. A pass or two.
DECISION_ROW_WAIT_S = 0.05

# Who runs the decisions (AutoPilot's `decisions`):
#   thread    the decision thread. What the car runs.
#   inline    a step() that comes due ticks before it writes, on the control
#             loop -- the old way, kept for the tests that drive the loop on a
#             fake clock and want each tick on a known pass.
#   external  nobody: a test calls AutoPilot.decide(now) itself, which is the
#             decision thread's iteration, to stall or crash it on cue.
THREAD = "thread"
INLINE = "inline"
EXTERNAL = "external"

# The decisions' state (AutoPilot.decision_status, /diagnostics, map page).
DECISIONS_OK = "ok"
DECISIONS_LATE = "late"
DECISIONS_DOWN = "down"
DECISIONS_GAVE_UP = "gave up"
# The interpreter's GIL switch interval with --nav (see main()): CPython's
# default is 5 ms.
NAV_SWITCH_INTERVAL_S = 0.001
STEERING_LIMIT_PERC = 70

# The steering sticks' deadzone, in raw axis units, before STEERING_LIMIT_PERC.
# A let-go LEFT_X settles anywhere from 0.02 to 0.07 and stays there (p99
# 0.094, max 0.106 over 44 rests; RIGHT_X up to 0.064), so without one the car
# steers a little every time I let go of the stick, and every "straight" row a
# record session writes is labelled a few percent off straight. 0.12 is the
# number the takeover already uses for "let go" -- see STICK_RELEASE -- so the
# stick means the same thing to manual, record and takeover. Outside it the
# travel is rescaled, so full deflection is still full lock: see stick_steer().
STICK_DEADZONE = 0.12

# How long the pad can go without proving it is there before a record session
# is called off. Not "the sticks did not move": a pad sitting still on the
# table reads the same numbers for minutes on end, which is a working pad.
PAD_SILENCE_SECONDS = 5.0

# How often the loop is allowed to read and log the vehicle state. See the
# comment at the call site -- this is not cosmetic.
STATE_POLL_INTERVAL = 1.0


class ButtonEdge:
    """One pad button, reported once per press.

    START's mode toggle and L3's steering toggle want the identical thing --
    the press edge, a hold that does not retrigger, and a read that cannot
    raise -- so it is written once here instead of twice below. marks.py keeps
    its own because a mark carries a snapshot and a rumble with it and shares
    nothing else with this.

    Never raises. No pad attached yet, or a stale handle after one re-pairs at
    a different SDL index, reads as "not pressed" and forgets whatever it
    thought the button was doing. That second part matters: the first read
    after the pad comes back is an edge again rather than a phantom hold, and
    nothing downstream of here can be left half-applied by a pad fault,
    because nothing downstream of here has been told anything yet.
    """

    def __init__(self, button):
        self.button = button
        self._was_down = False

    def pressed(self, joystick) -> bool:
        try:
            down = bool(joystick.get_button(self.button))
        except Exception:
            self._was_down = False
            return False

        edge = down and not self._was_down
        self._was_down = down
        return edge


class ModeSwitch:
    """The current drive mode, plus edge detection for the toggle button."""

    def __init__(self, mode, logger):
        self.mode = mode
        self.logger = logger
        self._toggle = ButtonEdge(MODE_TOGGLE_BUTTON)

    def poll(self, joystick):
        """Flip the mode on a press of the toggle button. Returns True on flip.

        A hold must not retrigger, so the press edge is what counts.
        """
        if not self._toggle.pressed(joystick):
            return False

        # Record goes to auto, not to manual: START means "auto" from either
        # of the two modes my hands are driving in.
        self.mode = MANUAL if self.mode == AUTO else AUTO
        self.logger.log("info", "Mode", f"Switched to {self.mode}")
        return True


ENTER_RECORD = "enter"
EXIT_RECORD = "exit"


class RecordButton:
    """Y: hold to enter record from manual, any press to leave it.

    The hold fires at the threshold while Y is still down, and that press is
    then spent: the release after it, and the rest of the hold, do nothing. So
    the only thing that can leave record is a fresh press.

    Y does nothing in auto, and a press that began in auto stays spent: hold
    Y, press START to manual, and the hold does not quietly carry over into a
    record session I never asked for.

    Never raises. A read that fails reads as "not down" and forgets the hold,
    same as ButtonEdge.
    """

    def __init__(self, button=RECORD_BUTTON, hold_seconds=RECORD_HOLD_SECONDS):
        self.button = button
        self.hold_seconds = hold_seconds
        self._down_since = None
        self._spent = False

    def poll(self, joystick, now, mode):
        """ENTER_RECORD, EXIT_RECORD or None."""
        try:
            down = bool(joystick.get_button(self.button))
        except Exception:
            down = False

        if not down:
            self._down_since = None
            self._spent = False
            return None

        if self._spent:
            return None

        edge = self._down_since is None
        if edge:
            self._down_since = now

        if mode == RECORD:
            if edge:
                self._spent = True
                return EXIT_RECORD
            return None

        if mode != MANUAL:
            self._spent = True
            return None

        if now - self._down_since >= self.hold_seconds:
            self._spent = True
            return ENTER_RECORD
        return None


class ThrottleLatch:
    """Throttle held at 0 after a handover, until both triggers let go.

    Leaving record with RT still held used to jump the car from auto's
    forward speed (0.2 then, 0.4 now) to full manual range on the very next
    tick: the same finger, a 5x bigger throttle (2.5x now), and nothing to
    tell me. Out of auto it was
    the same, from wherever the trigger happened to be resting. pigpio latches
    the last pulse width, so a jump like that is not a twitch, it is the car
    leaving at full speed.

    So any handover into manual from a mode that drove at a lower cap or
    that the pad did not drive at all -- record, or auto -- arms this, and the
    throttle stays 0 until both triggers have read under TRIGGER_DEADZONE at
    least once since. After that the pad passes straight through. Steering is
    never held: it is not the thing that leaves.

    Which handovers arm it is decided by the control loop from which mode
    drove the previous pass, not at each place that changes the mode, so a
    new way out of record or auto cannot forget to.
    """

    FROM = (RECORD, AUTO)

    def __init__(self, logger=None):
        self.logger = logger
        self.held = False

    def arm(self, came_from):
        if self.held:
            return
        self.held = True
        message = (f"Throttle held at 0 after {came_from}, "
                   f"until both triggers are released")
        print(f"[manual] {message}")
        if self.logger is not None:
            self.logger.log("info", "Manual", message)

    def gate(self, joystick, throttle):
        """`throttle`, or 0.0 while held. Never raises."""
        if not self.held:
            return throttle
        try:
            released = bool(joystick.triggers_at_rest())
        except Exception:
            released = False
        if released:
            self.held = False
            if self.logger is not None:
                self.logger.log("info", "Manual", "Triggers released, throttle live")
            return throttle
        return 0.0


def auto_speed_limits(driver=None):
    """(forward, reverse): the throttles auto drives at, read off a Driver.

    The live driver's if there is one -- that is what auto is actually using
    -- and otherwise a fresh Driver's defaults. Never a literal copied in
    here: the speeds are tuned in driver.py and record follows them.

    Anything wrong with what comes back -- no driver, a missing attribute, a
    forward speed that is not forward -- is (0.0, 0.0), a car that does not
    move. Never a guess.
    """
    try:
        if driver is None:
            driver = dv.Driver(None, None)
        forward = float(driver.forward_speed)
        reverse = float(driver.reverse_speed)
    except Exception:
        return 0.0, 0.0
    if not (0.0 < forward <= 1.0 and -1.0 <= reverse < 0.0):
        return 0.0, 0.0
    return forward, reverse


def scale_throttle(value, limits):
    """A full-range trigger reading scaled into auto's speeds.

    Scaled, not clipped: full RT is the forward limit and full LT the reverse
    limit, so the whole trigger travel is still mine for fine control. A nan
    or an infinity is 0.0 -- the explicit test first, because min() and max()
    hand a nan straight back out as a limit.
    """
    try:
        value = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(value):
        return 0.0
    value = 1.0 if value > 1.0 else -1.0 if value < -1.0 else value
    forward, reverse = limits
    return value * forward if value >= 0.0 else -value * reverse


class RecordSession:
    """The camera and the file for one record session. Never blocks.

    begin() asks telemetry for a session file -- one attribute assignment; the
    publisher thread opens it -- and starts the camera if it is not up, which
    returns at once and opens on the hub's own thread. The car drives,
    speed-capped, from the first tick; rows start when depth does.

    end() hands the file back to the ring and, if this session is the thing
    that opened the camera, gives the camera back too -- on a thread of the
    hub's own, because shut at idle in manual is how this car is meant to sit.
    Going to auto keeps it: auto is about to need it.
    """

    def __init__(self, tele, hub, logger, make_recorder=None, markpad=None, tags=None):
        self.tele = tele
        self.hub = hub
        self.logger = logger
        self.make_recorder = make_recorder
        self.markpad = markpad
        # The recordings' tags (telemetry/tags.SessionTags), said as a
        # session starts. None says nothing.
        self.tags = tags

        self.active = False
        self.limits = (0.0, 0.0)
        self.started_at = 0.0
        self._opened_camera = False
        self._logged_depth = False

    def begin(self, now, driver=None) -> bool:
        if self.active:
            return True
        limits = auto_speed_limits(driver)
        if limits == (0.0, 0.0):
            self._say("warning", "Not recording: no usable auto speeds to cap at")
            return False
        try:
            recorder = self.make_recorder() if self.make_recorder else None
            ok = recorder is not None and self.tele.begin_session(recorder)
        except Exception as e:
            self._say("warning", f"Not recording: {e}")
            return False
        if not ok:
            self._say("warning", "Not recording: telemetry is off in this "
                      "process (--no-telemetry --no-record)")
            return False

        self.active = True
        self.limits = limits
        self.started_at = now
        self._logged_depth = False
        self._opened_camera = False
        self._want_camera()
        self._buzz(2, now)
        self._say("info", f"Recording, throttle capped at "
                  f"{limits[0]:+.2f}/{limits[1]:+.2f}"
                  + (f", tagged {tg.said(self.tags.bits)}" if self.tags is not None else ""))
        return True

    def tick(self, now):
        """Once a pass in record. Two attribute reads on an ordinary tick."""
        if not self.active:
            return
        self._want_camera()
        hub = self.hub
        if (not self._logged_depth and hub is not None
                and hub.state == camera.READY):
            self._logged_depth = True
            self._say("info", f"Depth is up {now - self.started_at:.1f} s "
                      f"into the session (open took "
                      f"{hub.open_seconds:.1f} s)")

    def end(self, now, keep_camera=False, why="", alarm_pad=None) -> None:
        """Close the file, and the camera if this session opened it.

        `alarm_pad` is the Joystick when the session is ending because the pad
        went: then it is the one long pulse, which the pad may well not be
        there to feel, instead of the ordinary one-buzz exit. See
        marks.MarkPad.alarm.
        """
        if not self.active:
            return
        self.active = False
        try:
            self.tele.end_session()
        except Exception:
            pass
        if self._opened_camera and not keep_camera and self.hub is not None:
            try:
                self.hub.release()
            except Exception:
                pass
        self._opened_camera = False
        if alarm_pad is None:
            self._buzz(1, now)
        elif self.markpad is not None:
            try:
                self.markpad.alarm(alarm_pad, now)
            except Exception:
                pass
        self._say("info", f"Stopped recording after "
                  f"{now - self.started_at:.0f} s" + (f": {why}" if why else ""))

    def _want_camera(self):
        """Ask for the camera until it is coming up. Returns at once.

        Asked every pass rather than once, because a release from the last
        session may still be finishing and start() refuses until it has.
        """
        hub = self.hub
        if hub is None:
            return
        try:
            if hub.state == camera.OFF:
                if hub.start():
                    self._opened_camera = True
        except Exception:
            pass

    def _buzz(self, pulses, now):
        if self.markpad is not None:
            try:
                self.markpad.buzz(pulses, now)
            except Exception:
                pass

    def _say(self, level, message):
        print(f"[record] {message}")
        try:
            self.logger.log(level, "Record", message)
        except Exception:
            pass


class SteerMode:
    """Which sticks drive which axles, plus edge detection for L3.

    Its own object and not a field on ModeSwitch, because it outlives a mode
    switch: a lap in auto and back does not forget which way I left the sticks.
    It lives as long as the control loop does, which is as long as the process.
    Starting state is unified, the new default.

    No rumble on the flip. A mark already buzzes once for a good call and twice
    for a bad one, and a third meaning on the same motor would muddy a signal I
    have only just started trusting. The journal line here and the
    steering_mode field on /vehicle are the receipt instead.
    """

    def __init__(self, logger, split=False):
        self.split = split
        self.logger = logger
        self._toggle = ButtonEdge(STEER_SPLIT_BUTTON)

    @property
    def name(self) -> str:
        return SPLIT if self.split else UNIFIED

    def poll(self, joystick) -> bool:
        """Flip unified/split on a press of L3. Returns True on flip.

        Cannot raise -- see ButtonEdge -- so it can be called before the drive
        call rather than inside it, and a pad fault can never leave the pair of
        axles half-reassigned.
        """
        if not self._toggle.pressed(joystick):
            return False

        self.split = not self.split
        self.logger.log("info", "Steering", f"Steering is {self.name}")
        return True


# ---- takeover: correcting auto one control at a time ----------------------
#
# In auto the pad's controls fall into two groups. Steering is LEFT_X, plus
# RIGHT_X while split is on; throttle is the two triggers. Moving a group past
# its deadzone hands that group, and only that group, to my hand. The
# autopilot keeps driving the other one. LEFT_Y does nothing. A group goes back
# once everything in it has sat inside its deadzone for
# TAKEOVER_RELEASE_SECONDS, and letting go never commands straight or stop by
# itself: what goes back to the wheels at the handback is whatever the
# autopilot is asking for right then. The price of that is that I cannot take
# over just to hold the wheel centred -- a centred stick is a let-go stick.
#
# Every /footprint row says which hand had each group (steer_by, throttle_by),
# and a group that was mine is a correction: the label the policy should learn
# from. See docs/TRAINING_DATA.md.

DRIVER = "driver"
PAD = "pad"

# Both are on the raw axis, read alongside the intent and never worked back out
# of it, so STICK_DEADZONE -- which is the same 0.12 as STICK_RELEASE, applied
# to what the sticks command -- does not move either one. They are off a
# recording rather than a guess. Measured on 2026-10-27 over every /controller sample in
# nine of the car's MCAPs (103,517 samples, the pad under xpadneo), taking the
# stretches where a stick sat within 0.01 of itself for a second or more (44
# on LEFT_X, 60 on RIGHT_X):
#
#   LEFT_X  where it comes to rest   p50 0.052   p95 0.072   p99 0.094   max 0.106
#   RIGHT_X where it comes to rest   p50 0.023   p95 0.057   p99 0.061   max 0.064
#   jitter inside one rest           up to 0.017 on either stick
#
# Those are raw axis readings, before STEERING_LIMIT_PERC. The left stick does
# not come back to 0: it comes back to wherever it lands, anywhere from 0.02 to
# 0.07, and stays there. So "inside" is under 0.12, clear of every rest I have
# recorded including its jitter, and "taken over" is past 0.20. The gap between
# them is the hysteresis: a stick dithering at the edge of 0.12 cannot flap
# ownership, because getting it back means a deliberate push past 0.20.
STICK_RELEASE = 0.12
STICK_TAKEOVER = 0.20

# How long a group has to sit inside its deadzone before the autopilot gets it
# back.
TAKEOVER_RELEASE_SECONDS = 0.5

TAKEN = "taken"
RELEASED = "released"
STEERING = "steering"
THROTTLE = "throttle"


def _finite(*values) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in values)
    except Exception:
        return False


class DriverCommand:
    """The autopilot's command, held instead of written. Its controller.

    driver.Driver commands whatever it is handed as `controller`, and it used
    to be handed the VehicleController. Now it is handed one of these, so
    nothing the driver does reaches pigpio directly: AutoPilot.step() reads
    what it asked for, lets my hand replace the groups I own, and writes the
    result in at most one set_steering and one set_throttle a pass. Never a
    write from the driver followed by a second one from the pad, which on a
    latching pulse would leave the driver's command on the wheels for however
    long the gap between the two happened to be.

    Two things per channel: the value the driver last asked for, which is its
    standing intent (the driver commands nothing while a reverse runs and
    relies on the latch), and whether it asked for anything new since the last
    read. Last write wins inside one tick.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """The wheels went to neutral underneath it. Start from there."""
        self.throttle = 0.0
        self.front = 0.0
        self.rear = 0.0
        self.throttle_new = False
        self.steering_new = False

    def set_throttle(self, value):
        self.throttle = value
        self.throttle_new = True

    def set_steering(self, front, rear):
        self.front = front
        self.rear = rear
        self.steering_new = True

    def take(self):
        """(steering_new, throttle_new), and forget both."""
        new = (self.steering_new, self.throttle_new)
        self.steering_new = self.throttle_new = False
        return new


class PublishedCommand:
    """The newest decision as the control loop reads it. Locked.

    DriverCommand is the driver's draft: it writes it all through a tick on
    the decision thread, and nothing else reads it. At the end of a tick the
    decision thread publishes it here in one step, with the decision's
    sequence number and the `now` it was made at, so the control loop never
    sees half a tick (a steering of one decision with the throttle of the
    one before). The new-flags accumulate until the control loop takes them,
    so an ask is never lost between two of its passes.

    `epoch` is which engage of auto this belongs to. pause() and resume()
    move it on with reset(), and publish() refuses a decision from an epoch
    that has gone: a tick still running when START parked auto cannot land
    its command on the next lap.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.epoch = 0
        self._clear()

    def _clear(self):
        self.throttle = 0.0
        self.front = 0.0
        self.rear = 0.0
        self.steering_new = False
        self.throttle_new = False
        self.seq = 0
        self.at = None

    def reset(self, epoch):
        with self._lock:
            self.epoch = epoch
            self._clear()

    def publish(self, draft, seq, at, epoch) -> bool:
        """The draft's command and its new-flags, as decision `seq` made at
        `at`. False, and nothing changed, for an epoch that has gone."""
        with self._lock:
            if epoch != self.epoch:
                return False
            steering_new, throttle_new = draft.take()
            self.throttle = draft.throttle
            self.front = draft.front
            self.rear = draft.rear
            self.steering_new = self.steering_new or steering_new
            self.throttle_new = self.throttle_new or throttle_new
            self.seq = seq
            self.at = at
            return True

    def take(self):
        """(steering_new, throttle_new, throttle, front, rear, seq, at), and
        forget the two flags."""
        with self._lock:
            out = (self.steering_new, self.throttle_new, self.throttle,
                   self.front, self.rear, self.seq, self.at)
            self.steering_new = self.throttle_new = False
            return out

    def peek(self):
        """(seq, at), for the status. Takes nothing."""
        with self._lock:
            return self.seq, self.at


class _Applied:
    """What the control loop wrote for one decision: the command as it went
    out (read back off the controller) and whose each group was. The
    /footprint row for that decision is built off this."""

    __slots__ = ("seq", "command", "steer_by", "throttle_by", "takeover",
                 "pad_throttle", "steering_mode")

    def __init__(self, seq, command, steer_by, throttle_by, takeover,
                 pad_throttle, steering_mode):
        self.seq = seq
        self.command = command
        self.steer_by = steer_by
        self.throttle_by = throttle_by
        self.takeover = takeover
        self.pad_throttle = pad_throttle
        self.steering_mode = steering_mode


class _Group:
    """One group's ownership. Taken past its outer edge, given back after
    release_seconds inside its inner one."""

    def __init__(self):
        self.owned = False
        self._quiet_since = None

    def update(self, beyond, inside, now, release_seconds):
        """TAKEN, RELEASED or None."""
        if not self.owned:
            if beyond:
                self.owned = True
                self._quiet_since = None
                return TAKEN
            return None
        if not inside:
            self._quiet_since = None
            return None
        if self._quiet_since is None:
            self._quiet_since = now
        if now - self._quiet_since >= release_seconds:
            self.drop()
            return RELEASED
        return None

    def drop(self):
        self.owned = False
        self._quiet_since = None


class Takeover:
    """Which of the two groups my hand owns in auto, off one pad read a pass.

    poll() reads the pad once, through read_pad_intent() -- the same three
    readings manual drives on, so a taken-over steering is exactly manual's
    steering (steering_pair, or split) and a taken-over throttle is the
    trigger record drives on, scaled to auto's own caps by AutoPilot.step().

    A pad read that raises, or hands back a nan or an infinity, is a pad
    fault: both groups go back at once, poll() says False, and the control
    loop parks auto on the same neutral the lost-pad path uses. Nothing in
    here commands anything.

    Takeover and handback each get one buzz, through the mark buttons' rumble
    scheduler (the one record's enter and exit go through), and a log line.
    """

    def __init__(self, logger=None, markpad=None,
                 release_seconds=TAKEOVER_RELEASE_SECONDS):
        self.logger = logger
        self.markpad = markpad
        self.release_seconds = release_seconds
        self._steering = _Group()
        self._throttle = _Group()
        self.intent = (0.0, 0.0, 0.0)
        # Groups handed back on the last poll, for AutoPilot.step() to put the
        # autopilot's standing command back on in the same pass.
        self.released = ()

    @property
    def steering(self) -> bool:
        return self._steering.owned

    @property
    def throttle(self) -> bool:
        return self._throttle.owned

    @property
    def steer_by(self) -> str:
        return PAD if self._steering.owned else DRIVER

    @property
    def throttle_by(self) -> str:
        return PAD if self._throttle.owned else DRIVER

    def poll(self, joystick, now, split=False) -> bool:
        """Read the pad and update both groups. False on a pad fault."""
        self.released = ()
        try:
            intent, raw = read_pad(joystick, split)
            intent = tuple(float(v) for v in intent)
            sticks = tuple(float(v) for v in raw if v is not None)
        except Exception as e:
            self.drop(f"pad read failed: {e}")
            return False
        if not _finite(*intent, *sticks):
            self.drop("the pad read a nan")
            return False
        self.intent = intent

        # The raw axes, read once alongside the intent and never worked back
        # out of it: the intent has been through STICK_DEADZONE, and dividing
        # a rescaled number by the steering limit is no longer the stick. So
        # the thresholds above stay the recording's own numbers, untouched by
        # the deadzone. Unified reads LEFT_X alone.
        deflection = max(abs(v) for v in sticks)

        # read_throttle() already applies TRIGGER_DEADZONE and reads 0.0
        # inside it. The triggers rest at exactly -1.0 (98.5% of every
        # recorded sample is that value; the rest are presses), so there is
        # no rest noise for a hysteresis to absorb and the one deadzone serves
        # both edges.
        pulled = intent[0] != 0.0

        changes = []
        for name, group, beyond, inside in (
                (STEERING, self._steering, deflection > STICK_TAKEOVER,
                 deflection < STICK_RELEASE),
                (THROTTLE, self._throttle, pulled, not pulled)):
            change = group.update(beyond, inside, now, self.release_seconds)
            if change is not None:
                changes.append((name, change))

        self.released = tuple(name for name, change in changes
                              if change == RELEASED)
        for name, change in changes:
            self._say(("Took the " if change == TAKEN else "Gave back the ")
                      + name)
            self._buzz(now)
        return True

    def pad_steering(self):
        """(front, rear) off this pass's read. Centre if it is not a number."""
        _, front, rear = self.intent
        return (front, rear) if _finite(front, rear) else (0.0, 0.0)

    def drop(self, why=""):
        """Both groups back to the autopilot, now. No buzz: the usual reason
        is a pad that is not there to feel it."""
        had = self._steering.owned or self._throttle.owned
        self._steering.drop()
        self._throttle.drop()
        self.released = ()
        self.intent = (0.0, 0.0, 0.0)
        if had:
            self._say("Gave back both" + (f": {why}" if why else ""))

    def _say(self, message):
        print(f"[takeover] {message}")
        if self.logger is not None:
            try:
                self.logger.log("info", "Takeover", message)
            except Exception:
                pass

    def _buzz(self, now):
        if self.markpad is not None:
            try:
                self.markpad.buzz(1, now)
            except Exception:
                pass


class AutoPilot:
    """Auto mode, driven off the shared camera.

    It does not own the camera any more. camera.CameraHub does, main builds
    it, and this takes a view() on it -- a private one-deep slot per stream,
    so the driver and telemetry and the dashboard all read the same device
    without competing for the same packets.

    What has NOT changed: a failure to get a camera is logged and swallowed
    rather than raised. An unplugged OAK-D or a missing depthai costs me auto
    mode, never the ability to drive the car by hand.

    What has changed, and it matters: resume() does not block any more. It
    used to call dai.Device(), which takes seconds, on the control loop --
    seconds in which nothing polled the stop signal or the controller while
    pigpio sat there emitting the last pulse width it was given. Now the hub
    opens the device on a thread of its own and resume() just asks whether it
    is up yet. If it is not, auto refuses for now and manual keeps the wheels,
    and pressing START again a couple of seconds later gets me auto.

    The decision thread. driver.tick() used to run inside step(), on the
    control loop, and a policy tick is 86-116 ms p50 and 455 ms at worst on
    the Pi: that long with no pad read and no servo write. Now (decisions
    THREAD, what main() builds) a thread of its own makes the decisions, on
    the fixed phase (driver.next_decision_at), and step() only writes:

      decision thread   waits for the phase, snapshots what the control loop
                        last said about my hand (_inputs), takes the driver
                        lock, ticks, publishes the result (PublishedCommand,
                        stamped with a seq and its `now`), then makes that
                        decision's /footprint row off what the control loop
                        wrote for it.
      control loop      reads the pad and the takeover every pass and writes
                        in _write(), at its own rate. The only thread that
                        touches vecon.

    A decision older than DECISION_STALE_S, or a decision thread that is
    down, is held at NEUTRAL on whatever group the autopilot owns, and auto
    stays auto: the next fresh decision drives again. A thread that raised
    is started again at most once a DECISION_RESTART_S, and after
    DECISION_MAX_RESTARTS in DECISION_RESTART_WINDOW_S it is not, until
    START goes round. Never manual on any of it (the owner's ruling). My hand
    is read and written every pass whatever the decisions are doing.

    Shared with the tick, and how: the driver (driver_lock: the tick, pause's
    abort, a row; the control loop only ever tries it), the goal and the
    patrol (goal_lock, which the policy driver takes round its own goal
    calls; the control loop only ever tries it, in goal_step and the goal
    telemetry), my hand's state (_inputs, one tuple swapped under a lock),
    the command (PublishedCommand), the tap (its own locks).
    """

    def __init__(self, vecon, logger, hub=None, policy_path=None, lidar=None,
                 slam=None, nav=False, map_source=None, goals=None,
                 fallback_path=None, clicks=None, patrol_store=None,
                 lidar_near=False, path=None, person=None, tilt_veto=True,
                 library=None, fitter=None, skip_stuck=False, patrol_settings=None,
                 near_pass_m=patrol.NEAR_PASS_M, tags=None,
                 decisions=INLINE, clock=time.monotonic):
        self.vecon = vecon
        # Who runs the decisions: THREAD, INLINE or EXTERNAL (see THREAD).
        self.decisions = decisions
        # The decision thread's clock, the control loop's: one time base for
        # a decision's stamp and the pass that reads its age.
        self.clock = clock
        self.driver_lock = threading.RLock()
        self.goal_lock = threading.RLock()
        # The newest decision, published (see PublishedCommand), and what the
        # control loop wrote for it, for its /footprint row.
        self.published = PublishedCommand()
        self._applied = None
        self._applied_cond = threading.Condition()
        # The tap the rows go to (control_loop attaches its own), or None.
        self.tap = None
        # What the control loop said about my hand on its last pass:
        # (human_throttle, pad_throttle, human_steering, pad_split). One
        # tuple, swapped under the lock, so a tick reads one pass's worth.
        self._inputs_lock = threading.Lock()
        self._inputs = (False, None, False, False)
        # Engaged or not, and which engage (PublishedCommand.epoch).
        self._active = False
        self._epoch = 0
        # Asked of the driver by pause()/resume() and done under the driver
        # lock, now if it is free and by the decision thread if not.
        self._pending_abort = False
        self._pending_reset = False
        self._seq = 0
        # The phase: when the next decision is due, and for which epoch.
        self._phase_epoch = None
        self._row_seq = None
        # The decisions' health. _state is DECISIONS_OK, _DOWN or _GAVE_UP;
        # late is worked out per pass off the newest decision's age.
        self._state = DECISIONS_OK
        self._crash = None          # the last crash, said once
        self._last_start = None     # when the decisions were last (re)started
        self._restarts = []         # restart times inside the window
        self._engaged_at = None     # first pass of this engage
        self._holding = None        # why the wheels are held, or None
        self._hold_since = None
        self._thread = None
        self._generation = 0
        self._kick = threading.Event()
        self._halt = threading.Event()
        # Counted since boot, for /diagnostics.
        self.late_episodes = 0
        self.down_episodes = 0
        self.decision_restarts = 0
        self.decisions_made = 0
        # The recordings' tags (telemetry/tags.SessionTags), or None: the
        # patrol takes the map page's tags command into it, and the
        # telemetry block puts it on /session/tags. Never read to drive.
        self.tags = tags
        # --no-tilt-veto turns the policy's tilt veto off (driver.TiltVeto).
        # Set on every policy driver this builds.
        self.tilt_veto = bool(tilt_veto)
        # --lidar-near: the creep's stop under the stereo clamp comes off the
        # lidar's front arc (driver.py, "the near field"). Set on every
        # driver this builds.
        self.lidar_near = bool(lidar_near)
        self.logger = logger
        self.driver = None
        self.unavailable = False
        self._next_tick = 0.0
        self.interval = AUTO_TICK_INTERVAL
        # The shared camera, or None if this process was started without one.
        self.hub = hub
        # What the driver is handed as its controller. See DriverCommand: the
        # driver asks, step() writes.
        self.command = DriverCommand()
        # --policy: a learned model's .onnx. None is the hand-written
        # autopilot, which is the default and stays the default.
        self.policy_path = policy_path
        # --fallback-policy: a map-free model a map model drives on while the
        # map is lost, instead of neutral (policy.PolicyDriver, "fallback").
        self.fallback_path = fallback_path
        # Whether the car has arrived, for a learned driver's stop (goal.py).
        # With --goal, the goal clicked in Foxglove, or the current waypoint
        # of a patrol (patrol.Patrol, which keeps the single goal.ClickGoal
        # too): fed to a goal model, and the arrival hold. Else a flag
        # nothing sets. --goal's click listener (nav/click.ClickGoals),
        # drained into it every pass in every mode by goal_step(): a click
        # or a patrol command before auto is what auto drives to.
        # patrol_store (a patrol.Store) keeps the waypoints across a restart;
        # library (tracks.Library) the named tracks; fitter
        # (trackfit.Fitter) fits a saved track onto the fresh map.
        self.clicks = clicks
        self.goal = (patrol.Patrol(say=logger.log, store=patrol_store,
                                   map_view=getattr(slam, "map", None),
                                   library=library, fitter=fitter,
                                   skip_stuck=skip_stuck, settings=patrol_settings,
                                   near_pass_m=near_pass_m, tags=tags)
                     if clicks is not None else goal.Goal())
        # The patrol itself, under the FollowSwitch if there is one.
        self.patrol = self.goal if clicks is not None else None
        # --goal's path planner (autopilot/pathgoal.PathPlanner), on unless
        # --no-path: the goal model is fed a point along a path round the
        # walls to self.goal instead of the goal itself. Planned on its own
        # thread; it never steers. None without --goal.
        self.path = path
        # With --pantilt as well, a person to follow (autopilot/follow.py):
        # the Follow buttons on /follow/cmd switch the goal between the
        # patrol and the person, and nothing else changes.
        if clicks is not None and person is not None:
            from crawler.autopilot import follow
            self.goal = follow.FollowSwitch(self.goal, person)
        # The roof lidar's reader with --lidar, else None: handed to the
        # driver for its rear stop, which only ever asks it latest().
        self.lidar = lidar
        # The SLAM runner with --slam, else None: a map model's input. A map
        # model without it is auto unavailable (policy.PolicyDriver refuses).
        self.slam = slam
        # --nav: the map navigator drives (crawler/nav/driver.py), off the
        # map in map_source. False, the default, never imports crawler.nav.
        self.nav = bool(nav)
        self.map_source = map_source
        # Where the navigator's goals come from: click.ClickGoals, the
        # Foxglove listener, with --nav. None, no goals, so it stands still.
        self.goals = goals

    def goal_step(self):
        """--goal: the newest click onto the goal, placed off the newest crop,
        and the arrival latch. Every pass, any mode. Never raises."""
        if self.clicks is None:
            return
        # Tried, never waited for: a tick on the decision thread holds the
        # lock round its goal calls, and the clicks keep for the next pass.
        if not self.goal_lock.acquire(blocking=False):
            return
        try:
            crop = self.slam.crop() if self.slam is not None else None
            self.goal.poll(self.clicks, crop)
            if self.path is not None:
                # Planned in any mode, so the path is on the map before auto.
                self.path.update(crop, self.goal)
        except Exception as e:
            print(f"[goal] {e}", file=sys.stderr)
        finally:
            self.goal_lock.release()

    def goal_marker(self):
        """/goal/marker's payload: the goal on the map the truck is on now,
        or nothing, which clears the disc. A goal carried across a reseed
        (goal.ClickGoal) says approx_m, how far off it may be, and is drawn
        as wide as that."""
        if getattr(self.goal, "use", "goal") == "patrol":
            return {}           # the patrol's own markers (/patrol/marker)
        st = self.goal.status()
        if st["goal_set"] and st["frame"] == st["frame_now"]:
            if st.get("approx"):
                return {"goal": (st["x"], st["y"]), "approx_m": st["approx_m"]}
            return {"goal": (st["x"], st["y"])}
        return {}

    def resume(self):
        """Start or restart auto mode. True if it is running.

        Never blocks and never raises. False can mean three different things
        and only one of them is permanent:

          no hub at all, or the hub closed      -> unavailable, latched
          the hub is still opening the device   -> try again in a moment
          the camera would not open, or went    -> try again in a moment: the
            away, and the hub is retrying          hub opens it again on its own
          Driver would not construct            -> unavailable, latched
        """
        if self.unavailable:
            return False

        hub = self.hub
        if hub is not None and hub.state == camera.FAILED:
            # Down, and the hub is opening it again on its own (camera.py,
            # RETRY_FIRST_S). It used to latch auto unavailable here until a
            # service restart. start() is a no-op while it retries.
            hub.start()
            self.logger.log("warning", "Mode",
                            f"Camera is down ({hub.error or hub.state}), "
                            "retrying - press START again once it is up")
            print(f"[auto] camera down, retrying: {hub.error or hub.state}",
                  file=sys.stderr)
            return False

        if (self.driver is not None and hub is not None
                and hub.state in (camera.OFF, camera.STARTING,
                                  camera.RELEASING)):
            # A driver from an earlier lap, on a camera that has since been
            # given back -- a record session closes the camera it opened.
            # Same answer as the first open: start it, and START again.
            hub.start()
            self.logger.log("info", "Mode",
                            "Camera is still coming up, press START again")
            return False

        if self.driver is None:
            hub = self.hub
            if hub is None:
                self.unavailable = True
                self.logger.log("error", "Auto mode unavailable",
                                "no camera in this process")
                return False

            # Kick it off if nobody has yet. Returns immediately.
            if hub.state == camera.OFF:
                hub.start()

            if hub.state in (camera.STARTING, camera.RELEASING):
                # Not a failure. RELEASING is a record session's camera
                # still being given back; START again finds it reopening. Do not latch unavailable -- the device takes
                # a few seconds and the next press of START will find it up.
                self.logger.log("info", "Mode",
                                "Camera is still coming up, press START again")
                return False

            if hub.state != camera.READY:
                self.unavailable = True
                self.logger.log("error", "Auto mode unavailable",
                                hub.error or hub.state)
                print(f"[auto] unavailable: {hub.error or hub.state}",
                      file=sys.stderr)
                return False

            try:
                self.driver = self._build_driver(hub.view(AUTO_VIEW))
            except Exception as e:
                self.unavailable = True
                self.logger.log("error", "Auto mode unavailable", str(e))
                print(f"[auto] unavailable: {e}", file=sys.stderr)
                return False

        # Every way into auto has just put the wheels at neutral. A scan left
        # over from an earlier lap is of no known age, so it clears nothing.
        # The draft is the decision thread's, so the reset is done under the
        # driver lock: now if it is free, else before its next tick.
        self._pending_reset = True
        self._try_pending()
        # START again is the try again after the decisions gave up.
        self._state = DECISIONS_OK
        self._crash = None
        self._restarts = []
        self._last_start = None
        self._new_epoch()           # the first decision on the next pass
        self._active = True
        if self.decisions == THREAD:
            self._start_thread()
        return True

    # ---- the decisions ----------------------------------------------------

    def _new_epoch(self, hold_over=False):
        """A fresh engage: nothing published, the phase from scratch, and
        the age of 'no decision yet' counted from the next pass. A restart
        is `hold_over`: the hold it is in goes on until a decision lands."""
        self._epoch += 1
        self.published.reset(self._epoch)
        self._engaged_at = None
        self._row_seq = None
        if not hold_over:
            self._holding = None
            self._hold_since = None
        self._kick.set()

    def _try_pending(self):
        """pause()'s abort and resume()'s reset, under the driver lock if
        it is free this instant. Never waits: a tick holding it does them
        itself once it is done (_apply_pending)."""
        if self.driver_lock.acquire(blocking=False):
            try:
                self._apply_pending()
            finally:
                self.driver_lock.release()

    def _apply_pending(self):
        """With the driver lock held."""
        if self._pending_abort:
            self._pending_abort = False
            if self.driver is not None:
                self.driver.abort()
                self.driver.human_throttle = False
        if self._pending_reset:
            self._pending_reset = False
            self.command.reset()

    def _set_inputs(self, takeover, split):
        human_throttle = bool(takeover is not None and takeover.throttle)
        inputs = (human_throttle,
                  # What my trigger is putting on the wheels, for the tilt
                  # veto's which-way-was-it-going (policy.PolicyDriver).
                  self._pad_throttle(takeover) if human_throttle else None,
                  bool(takeover is not None and takeover.steering),
                  bool(split))
        with self._inputs_lock:
            self._inputs = inputs

    def decide(self, now):
        """One decision as the decision thread makes it, made by the caller:
        decisions EXTERNAL's hook, for a test to make a decision (or stall,
        or crash one) on the pass it chooses. Its row is made on the next
        step(), after the write. The seq, or None."""
        seq = self._decide_safely(now)
        if seq is not None:
            self._row_seq = seq
        return seq

    def _decide(self, now):
        """One decision: the tick, under the driver lock, published. Returns
        its seq, or None when auto is not engaged or was parked while it
        ticked. Raises whatever the tick raised (see _decide_safely)."""
        with self.driver_lock:
            self._apply_pending()
            if not self._active or self.driver is None:
                return None
            epoch = self._epoch
            with self._inputs_lock:
                human_throttle, pad_throttle, human_steering, split = self._inputs
            d = self.driver
            d.human_throttle = human_throttle
            d.pad_throttle = pad_throttle
            # The pad's side of a learned driver's steering mode (see
            # policy.PolicyDriver): my SteerMode, and whether my hand has
            # the steering. The rules driver never reads either.
            d.human_steering = human_steering
            d.pad_split = split
            d.tick(now)
            # pause() or resume() landed during the tick: do what they asked
            # and drop the decision, which belongs to an engage that is over.
            self._apply_pending()
            if epoch != self._epoch or not self._active:
                return None
            seq = self._seq + 1
            if not self.published.publish(self.command, seq, now, epoch):
                return None
            self._seq = seq
            self.decisions_made += 1
            return seq

    def _decide_safely(self, now):
        """decide(), its next one put on the phase, and a raise turned into
        the decisions being down (held at neutral, restarted by
        _supervise). The seq, or None."""
        if self._state != DECISIONS_OK:
            return None
        if self._phase_epoch != self._epoch:
            self._phase_epoch = self._epoch
            self._next_tick = 0.0
        try:
            seq = self._decide(now)
        except Exception as e:
            self._crashed(e, traceback.format_exc())
            return None
        self._next_tick = dv.next_decision_at(self._next_tick, now, self.interval)
        return seq

    def _due(self, now):
        if self._phase_epoch != self._epoch:
            return True
        return now >= self._next_tick

    def _crashed(self, error, report):
        """The tick raised (or the thread died): down until _supervise
        starts it again. Said by whoever has the time -- the decision thread
        at once, the control loop after its write."""
        self._state = DECISIONS_DOWN
        self._crash = (f"{type(error).__name__}: {error}" if error is not None
                       else "the decision thread died", report, False)
        if self.decisions == THREAD:
            self._say_crash()

    def _say_crash(self):
        crash = self._crash
        if crash is None or crash[2]:
            return
        what, report, _ = crash
        self._crash = (what, report, True)
        self.logger.log("error", "Auto",
                        f"Decisions crashed ({what}): holding neutral in auto, "
                        f"restarting in {DECISION_RESTART_S:.0f} s")
        if report:
            print(f"[auto] decisions crashed:\n{report}", file=sys.stderr)

    def _supervise(self, now):
        """The control loop's look at the decisions, every auto pass: a
        thread that died without a word is down, and down is started again
        at most once a DECISION_RESTART_S -- unless it has been restarted
        DECISION_MAX_RESTARTS times in DECISION_RESTART_WINDOW_S, which is
        given up on (neutral, auto, said loudly) until START goes round."""
        if self._last_start is None:
            self._last_start = now
        if (self.decisions == THREAD and self._state == DECISIONS_OK
                and self._active
                and (self._thread is None or not self._thread.is_alive())):
            self._crashed(None, "")
        if self._state != DECISIONS_DOWN:
            return
        if self.decisions != THREAD:
            self._say_crash()
        if now - self._last_start < DECISION_RESTART_S:
            return
        self._restarts = [t for t in self._restarts
                          if now - t < DECISION_RESTART_WINDOW_S]
        if len(self._restarts) >= DECISION_MAX_RESTARTS:
            self._state = DECISIONS_GAVE_UP
            what = self._crash[0] if self._crash else "unknown"
            self.logger.log("error", "Auto",
                            f"Decisions keep crashing ({len(self._restarts)} "
                            f"restarts in {DECISION_RESTART_WINDOW_S:.0f} s, last: "
                            f"{what}): holding neutral in auto. No more restarts; "
                            "START out of auto and back in to try again")
            print(f"[auto] DECISIONS GAVE UP: {what}", file=sys.stderr)
            return
        self._restarts.append(now)
        self._last_start = now
        self.decision_restarts += 1
        # From a clean slate: no recover half done, nothing standing in the
        # draft, the phase from scratch (the wheels have been at neutral).
        self._pending_abort = self._pending_reset = True
        self._try_pending()
        self._new_epoch(hold_over=True)
        self._state = DECISIONS_OK
        self.logger.log("warning", "Auto",
                        f"Decisions restarted ({len(self._restarts)} of "
                        f"{DECISION_MAX_RESTARTS} in {DECISION_RESTART_WINDOW_S:.0f} s)")
        if self.decisions == THREAD:
            self._start_thread()

    # ---- the decision thread ------------------------------------------------

    def _start_thread(self):
        """A decision thread, unless one of this generation is running."""
        if self._thread is not None and self._thread.is_alive():
            if self._state == DECISIONS_OK:
                self._kick.set()
                return
        self._generation += 1
        self._halt.clear()
        self._thread = threading.Thread(target=self._run,
                                        args=(self._generation,),
                                        name="decisions", daemon=True)
        self._thread.start()

    def _run(self, generation):
        """The decision thread. Ends on a raise (the control loop starts the
        next one), on stop(), and when a newer generation has started."""
        while not self._halt.is_set() and generation == self._generation:
            if not self._active or self._state != DECISIONS_OK:
                self._kick.wait(0.1)
                self._kick.clear()
                continue
            now = self.clock()
            if not self._due(now):
                self._kick.wait(max(0.0, self._next_tick - now))
                self._kick.clear()
                continue
            seq = self._decide_safely(now)
            if self._state != DECISIONS_OK:
                return
            if seq is not None:
                self._row(seq, now, wait=True)

    def _row(self, seq, now, wait=False):
        """Decision `seq`'s /footprint row, off what the control loop wrote
        for it. On the decision thread that is a wait of a pass or two; a
        decision never written (auto parked, a hold) makes no row."""
        tap = self.tap
        if tap is None:
            return
        with self._applied_cond:
            if wait:
                self._applied_cond.wait_for(
                    lambda: (self._applied is not None and self._applied.seq >= seq)
                    or not self._active, DECISION_ROW_WAIT_S)
            applied = self._applied
        if applied is None or applied.seq != seq or not self._active:
            return
        try:
            with self.driver_lock, self.goal_lock:
                tap.after_drive(
                    now, AUTO, applied.command, driver=self.driver,
                    pad_throttle=applied.pad_throttle,
                    # Every auto row carries the caps auto is driving at,
                    # whoever had the throttle.
                    limits=auto_speed_limits(self.driver),
                    steering_mode=applied.steering_mode,
                    steer_by=applied.steer_by, throttle_by=applied.throttle_by,
                    takeover=applied.takeover, decision=True)
        except Exception as e:
            print(f"[auto] row: {e}", file=sys.stderr)

    def attach_tap(self, tap):
        """The FootprintTap the auto rows go to. control_loop's."""
        self.tap = tap

    # ---- what the status says ----------------------------------------------

    def decision_status(self, now=None):
        """{"state", "why", "age_s"}: off, ok, late, down or gave up."""
        if not self._active:
            return {"state": "off", "why": "", "age_s": None}
        seq, at = self.published.peek()
        age = None
        if now is not None and at is not None:
            age = round(max(0.0, now - at), 3)
        if self._state == DECISIONS_GAVE_UP:
            why = self._crash[0] if self._crash else ""
            return {"state": DECISIONS_GAVE_UP, "age_s": age,
                    "why": f"Auto decisions keep crashing ({why}). Holding "
                           "neutral in auto; START out of auto and back in to "
                           "try again."}
        if self._state == DECISIONS_DOWN:
            why = self._crash[0] if self._crash else ""
            return {"state": DECISIONS_DOWN, "age_s": age,
                    "why": f"Auto decisions crashed ({why}), restarting. "
                           "Holding neutral."}
        if self._holding is not None:
            # Late, or restarted and nothing fresh yet: either way, waiting.
            return {"state": DECISIONS_LATE, "age_s": age,
                    "why": "Auto decisions are late. Holding neutral until "
                           "one lands."}
        return {"state": DECISIONS_OK, "why": "", "age_s": age}

    def decision_fields(self, now=None):
        """The decisions' part of /diagnostics."""
        st = self.decision_status(now)
        return {
            "decisions_state": st["state"],
            "decisions_holding": self._holding is not None,
            "decisions_age_ms": (None if st["age_s"] is None
                                 else round(st["age_s"] * 1000.0, 1)),
            "decisions_made": int(self.decisions_made),
            "decisions_late_episodes": int(self.late_episodes),
            "decisions_down_episodes": int(self.down_episodes),
            "decisions_restarts": int(self.decision_restarts),
        }

    def _build_driver(self, view):
        """The autopilot, or with --policy the learned one on the same chassis.

        A model that will not load is auto unavailable, the same as a camera
        that will not open -- never a quiet fall back to the autopilot. I asked
        for the model, and a car that drives on something else without saying
        so is a test I cannot read.

        With --nav, that driver drives until a goal is clicked, and the
        navigator from then on (nav/pointnav.py), on the same command.
        """
        if self.nav:
            # Raises with no map source, which resume() logs as auto
            # unavailable: --nav never falls back to driving without a map
            # quietly. Built first, so a missing map costs no model load.
            from crawler.nav.driver import NavDriver
            from crawler.nav.pointnav import PointNavDriver
            nav = NavDriver(self.command, view, self.map_source, lidar=self.lidar)
            try:
                free = self._build_free(view)
            except Exception:
                nav.close()
                raise
            self.logger.log("info", "Mode", "Auto drives to a goal clicked in "
                            "Foxglove, and as before until there is one")
            return PointNavDriver(free, nav, self.goals)
        return self._build_free(view)

    def _build_free(self, view):
        """The auto driver without the navigator: the rules, or the policy."""
        if self.policy_path is None:
            from crawler.autopilot.driver import Driver
            return Driver(self.command, view, lidar=self.lidar)
        from crawler.autopilot import policy
        model = policy.PolicyModel(self.policy_path)
        self.logger.log("info", "Mode",
                        f"Auto is {model.label}, the policy in {self.policy_path} "
                        f"({model.representation}, {model.contract}, "
                        f"sha256 {model.sha256[:12]})")
        print(f"[auto] policy {model.label} sha256 {model.sha256}", file=sys.stderr)
        fallback = None
        if self.fallback_path is not None:
            # Same rule as the policy: one named that will not load is auto
            # unavailable, never a car that parks on a lost map unannounced.
            fallback = policy.PolicyModel(self.fallback_path)
            self.logger.log("info", "Mode",
                            f"Auto falls back to {fallback.label}, the policy in "
                            f"{self.fallback_path}, while the map is lost "
                            f"(sha256 {fallback.sha256[:12]})")
            print(f"[auto] fallback {fallback.label} sha256 {fallback.sha256}",
                  file=sys.stderr)
        if self.clicks is not None and not model.uses_goal:
            self.logger.log("warning", "Goal",
                            f"{model.label} takes no goal: a click is shown and "
                            "the arrival hold still stops the car there, but the "
                            "model drives as it always has")
        d = policy.PolicyDriver(self.command, view, model, goal=self.goal,
                                lidar=self.lidar, slam=self.slam, fallback=fallback)
        d.goal_lock = self.goal_lock
        d.path = self.path if model.uses_goal else None
        if d.path is not None:
            # The lookahead the model's path goals were drawn at, so the point
            # fed is the one it was trained on (policy_goal.lookahead_of).
            from crawler.autopilot import policy_goal as pg
            want = pg.lookahead_of(getattr(model, "meta", None))
            if want != d.path.lookahead_m:
                self.logger.log("info", "Path", f"{model.label} was trained on path goals "
                                f"{want:g} m along: the point fed is {want:g} m along the path")
                d.path.lookahead_m = want
        d.tilt_veto.enabled = self.tilt_veto
        if not self.tilt_veto:
            self.logger.log("warning", "Mode", "The tilt veto is off (--no-tilt-veto)")
        # Only the policy creeps, so only it reads the near field.
        d.lidar_near = self.lidar_near and self.lidar is not None
        if d.lidar_near:
            self.logger.log("info", "Mode", "The creep stops on the lidar under "
                            "the camera's near clamp, where the two agree")
        return d

    def step(self, now, takeover=None, split=False):
        """One pass of auto mode, on the control loop. False means hand the
        wheels back to manual, which is only ever for no driver at all or a
        write to the hardware that raised: never for the decisions.

        The driver decides at AUTO_TICK_INTERVAL, on the fixed phase, into
        its draft (DriverCommand); a decision is published whole
        (PublishedCommand) and this writes it. With decisions THREAD the
        deciding is the decision thread's and this pass only writes; INLINE
        ticks here first when one is due, as the loop always used to.

        `takeover` is a Takeover already polled this pass, or None. A group it
        owns replaces that field of the driver's command, every pass rather
        than every decision, because my hand at 10 Hz would be a hand with
        100 ms of lag in it. A group it handed back this pass gets the
        driver's standing command written straight away, so letting go puts
        the autopilot's own command on the wheels and never a straight or a
        stop of its own -- unless the decisions are late or down, when what
        the autopilot owns is held at neutral.

        `split` is SteerMode.split this pass. A learned driver with a mode
        head starts from it when auto engages, follows it while my hand has
        the steering, and takes it on an L3 press (policy.ModeLatch).
        """
        if self.unavailable or self.driver is None:
            return False

        try:
            if self._engaged_at is None:
                self._engaged_at = now
            self._set_inputs(takeover, split)
            self._supervise(now)
            decided = None
            if (self.decisions == INLINE and self._state == DECISIONS_OK
                    and self._active and self._due(now)):
                decided = self._decide_safely(now)
            self._write(now, takeover, SPLIT if split else UNIFIED)
            if self.decisions != THREAD:
                # A crash on this pass is said after the wheels were told.
                self._say_crash()
                if decided is not None:
                    self._row_seq = decided
                if self._row_seq is not None:
                    seq, self._row_seq = self._row_seq, None
                    self._row(seq, now)
        except Exception as e:
            # Stop the car BEFORE writing anything down. Logging opens, writes
            # and flushes a file on this same thread, and the throttle does
            # not get to wait on the filesystem.
            try:
                self.vecon.set_throttle(0)
            except Exception:
                pass
            self.unavailable = True
            self.logger.log("error", "Auto mode failed", str(e))
            print(f"[auto] failed: {e}", file=sys.stderr)
            return False

        return True

    def _hold_reason(self, now, seq, at):
        """Why the autopilot's groups are held at neutral this pass, or None."""
        if self._state == DECISIONS_GAVE_UP:
            return DECISIONS_GAVE_UP
        if self._state == DECISIONS_DOWN:
            return DECISIONS_DOWN
        # Before this engage's first decision, its age counts from the first
        # pass of the engage: a thread that never answers is late too.
        since = at if seq else self._engaged_at
        if since is not None and now - since > DECISION_STALE_S:
            return DECISIONS_LATE
        return None

    def _write(self, now, takeover, steering_mode=UNIFIED):
        """The mixed command, in at most one set_steering and one
        set_throttle. Steering first, as the driver always ordered them.

        Held (late, down, gave up): my groups are still mine, and the
        autopilot's are neutral -- the throttle at 0 on every pass, the
        steering centred when the hold begins. The first fresh decision after
        a hold writes both of its groups, its standing command."""
        (steering_new, throttle_new, throttle, front, rear,
         seq, at) = self.published.take()
        released = takeover.released if takeover is not None else ()
        mine_steering = takeover is not None and takeover.steering
        mine_throttle = takeover is not None and takeover.throttle

        if self._engaged_at is None:
            self._engaged_at = now
        reason = self._hold_reason(now, seq, at)
        if reason is None and self._holding is not None and not seq:
            # Restarted, and nothing fresh to drive on yet: still held.
            reason = self._holding
        began = reason is not None and self._holding is None
        back = reason is None and self._holding is not None
        held_for = (now - self._hold_since
                    if back and self._hold_since is not None else 0.0)

        if mine_steering:
            self.vecon.set_steering(*takeover.pad_steering())
        elif reason is not None:
            if began or STEERING in released:
                self.vecon.set_steering(0.0, 0.0)
        elif steering_new or back or STEERING in released:
            self.vecon.set_steering(front, rear)

        if mine_throttle:
            self.vecon.set_throttle(self._pad_throttle(takeover))
        elif reason is not None:
            # Every pass: pigpio latches, and this is commanded, not assumed.
            self.vecon.set_throttle(0.0)
        elif throttle_new or back or THROTTLE in released:
            self.vecon.set_throttle(throttle)

        if reason is None and seq:
            applied = _Applied(
                seq,
                (float(self.vecon.last_throttle or 0.0),
                 float(self.vecon.last_steering_front or 0.0),
                 float(self.vecon.last_steering_rear or 0.0)),
                PAD if mine_steering else DRIVER,
                PAD if mine_throttle else DRIVER,
                bool(mine_steering or mine_throttle),
                takeover.intent[0] if mine_throttle else None,
                steering_mode)
            with self._applied_cond:
                if self._applied is None or self._applied.seq != seq:
                    self._applied = applied
                    self._applied_cond.notify_all()

        # Said after the wheels were told, once an episode.
        if began:
            self._holding = reason
            self._hold_since = now
            if reason == DECISIONS_LATE:
                self.late_episodes += 1
                age = now - (at if seq else self._engaged_at)
                self.logger.log("warning", "Auto",
                                f"Decisions late: holding neutral (newest is "
                                f"{age:.2f} s old, the limit is "
                                f"{DECISION_STALE_S:.2f} s)")
            else:
                self.down_episodes += 1
        elif reason is not None and reason != self._holding:
            self._holding = reason
        elif back:
            self._holding = None
            self._hold_since = None
            self.logger.log("info", "Auto",
                            f"Decisions back: driving again after "
                            f"{held_for:.2f} s at neutral")

    def _pad_throttle(self, takeover):
        """My trigger at record's scaling. Never nan.

        Full RT is the driver's forward_speed and full LT its reverse_speed,
        read off the live driver, and that scaling is all that happens to it
        (/footprint carries pad_throttle next to throttle and the caps). No
        floor, in either direction, and no wait for a scan: a taken-over
        throttle is me driving, as it is in manual, and manual has no floor.
        It used to sit under the forward floor (driver.ForwardFloorVeto),
        and a probe inside FORWARD_FLOOR_MM or latched blind held full RT at
        0 until I had backed the car off -- the one time I take the throttle
        is to get out of a spot like that. The floor still holds every
        forward the autopilot chooses itself.
        scale_throttle() turns a nan into 0.0 before any clamp sees it.
        """
        return scale_throttle(takeover.intent[0],
                              auto_speed_limits(self.driver))

    def pause(self):
        """Hand the wheels back. Drops a recover in progress on the floor.

        Neutral first, before anything that could wait. Then the decisions
        are off, any decision still being made is dropped (the epoch moves
        on), and the driver's abort and the draft's reset are done under the
        driver lock -- now if it is free, else by the decision thread the
        moment its tick ends. Never waits for a tick: this is on the control
        loop, and START, a lost pad and a pad fault all come through here."""
        self.vecon.return_neutral()
        self._active = False
        self._new_epoch()
        with self._applied_cond:
            self._applied_cond.notify_all()
        if self.driver is not None:
            self._pending_abort = True
        self._pending_reset = True
        self._try_pending()

    def stop(self):
        """Give up the driver. The camera is not mine to close.

        main owns the hub and closes it, because the camera outlives auto
        mode now -- telemetry is still publishing frames in manual. The
        decision thread is told to end and given a moment; it never writes
        the hardware, so the wheels do not wait on it.
        """
        self._active = False
        self._halt.set()
        self._kick.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        self.driver = None

    # ---- what telemetry asks about the camera ----------------------------
    #
    # Both of these are read off a field the hub's own thread has already set.
    # Neither touches the device, neither can block, and neither raises, so
    # the control loop can call them inside the /diagnostics block.

    def camera_state(self) -> str:
        if self.hub is None:
            return camera.OFF
        return self.hub.state

    def camera_error(self) -> str:
        if self.hub is None:
            return ""
        return self.hub.error or ""


def stick_steer(raw):
    """One raw steering axis as the angle it commands, before steering_pair.

    Zero inside STICK_DEADZONE, then linear from the edge of it to full
    deflection, so 0.12 commands 0 and 1.0 still commands the whole of
    STEERING_LIMIT_PERC -- the deadzone takes the rest drift out without
    taking any lock away. The sign is kept. Past 1.0 is clamped to 1.0.

    Not a number in, the same not-a-number out, never a 0.0: a NaN is a pad
    fault, and Takeover and set_steering() each look for it and act on it.
    A max() would quietly turn it into straight ahead.
    """
    if not math.isfinite(raw):
        return raw
    magnitude = abs(raw)
    if magnitude <= STICK_DEADZONE:
        return 0.0
    magnitude = min(magnitude, 1.0)
    scaled = (magnitude - STICK_DEADZONE) / (1.0 - STICK_DEADZONE)
    scaled *= STEERING_LIMIT_PERC / 100
    return scaled if raw > 0.0 else -scaled


def read_pad(joystick, split=False):
    """(intent, raw sticks): read_pad_intent's answer, and what it came from.

    The raw sticks are (LEFT_X, RIGHT_X) exactly as the pad reads them, with
    RIGHT_X None in unified, where it is not read at all. Takeover decides on
    those and not on the intent, because its thresholds are measured on the
    raw axis and a deadzone in between would move them.
    """
    throttle = joystick.read_throttle()
    left = float(joystick.get_axis("LEFT_X"))
    if split:
        right = float(joystick.get_axis("RIGHT_X"))
        return ((throttle, stick_steer(left), stick_steer(right)),
                (left, right))
    return (throttle,) + dv.steering_pair(stick_steer(left)), (left, None)


def read_pad_intent(joystick, split=False):
    """What the pad is asking for: (throttle, steering front, steering rear).

    Reads only. Pulled out of drive_manual so telemetry can report what the
    sticks are saying while auto has the wheels, without there being a second
    copy of the three readings that could drift away from the ones that
    actually drive the car.

    Unified -- `split` false, the default -- is the left stick alone, handed to
    driver.steering_pair(). That is deliberately the same function every auto
    steering decision goes through, called and not copied: whether this car
    steers counter-phase or crabs is decided in one place, and nobody has yet
    put the wheels off the ground and confirmed which one (f, -f) actually is
    on this linkage. If it comes back mirrored, REAR_STEER_RATIO's sign is the
    single thing that changes and my hand on the stick inherits the fix with
    nothing in this file moving.

    Split reads the right stick for the rear and applies no ratio at all. A
    pose is the whole point of it and a ratio is exactly what would take the
    pose away, so it does not go through steering_pair and must not.

    Both steering axes go through stick_steer(), here and only here, so
    manual, record, a takeover and the /controller channel all see the same
    deadzone.
    """
    return read_pad(joystick, split)[0]


def drive_manual(joystick, vecon, log, split=False, latch=None):
    """drive_pad() at full range, returning only the command."""
    return drive_pad(joystick, vecon, log, split, latch=latch)[0]


def drive_pad(joystick, vecon, log, split=False, limits=None, latch=None):
    """One tick of hand driving. Anything that goes wrong in here means neutral.

    Returns what it commanded, so the caller does not have to read the pad a
    second time to report it.

    `split` is passed in already read, never polled in here. The toggle is a
    button read and a button read off a stale handle raises -- inside this try
    it would be swallowed into "dropped a tick", which would turn an awkward
    thumb press into a silent neutral. It is also why the pair of axles can
    never be half-reassigned: by the time anything is commanded, which mapping
    is in force has already been settled and both numbers go to the hardware in
    the one set_steering call.

    `latch` is a ThrottleLatch or None; while it holds, the throttle
    commanded is 0 and the steering is the sticks'.

    `limits` is record mode's (forward, reverse) and None for manual. Returns
    ((throttle, front, rear) as commanded, the trigger before scaling, ok),
    and ok False means a pad read raised and the car was sent to neutral.
    """
    try:
        pad_throttle, steering_front, steering_rear = read_pad_intent(
            joystick, split)
        throttle = (pad_throttle if limits is None
                    else scale_throttle(pad_throttle, limits))
        if latch is not None:
            throttle = latch.gate(joystick, throttle)

        vecon.set_throttle(throttle)
        vecon.set_steering(steering_front, steering_rear)
        return (throttle, steering_front, steering_rear), pad_throttle, True
    except Exception as e:
        # A pad that re-pairs at a different SDL index leaves a handle whose
        # get_axis raises, or worse, keeps answering with stale values. An
        # exception here used to walk straight out of the loop with the last
        # pulse width still latched.
        try:
            vecon.set_throttle(0)
        except Exception:
            pass
        log.log("warning", "Manual", f"Dropped a tick, holding neutral: {e}")
        return (0.0, 0.0, 0.0), 0.0, False


def control_loop(joystick, vecon, autopilot, modes, log, stop_event, tele=None,
                 markpad=None, steer=None, session=None, tap=None,
                 record_button=None, clock=time.monotonic, takeover=None,
                 page=None):
    """Poll the controller and drive whichever mode is current.

    Only one mode ever has the wheels: auto's write is a step() on this
    thread (its decisions are the decision thread's, which writes nothing)
    and manual writes nothing while auto is up, so a switch can never leave
    the car rolling under two owners. A pass starts every LOOP_INTERVAL.

    `tele` is a telemetry.Telemetry, or None. A Telemetry that failed to start
    is already a complete no-op -- wants() is False for every channel -- so
    there is no third case to handle and no branch in here that depends on
    whether Foxglove is installed, running or being watched.
    """
    import pygame
    from crawler.autopilot import marks as mk
    from crawler.telemetry import telemetry as tm

    if tele is None:
        tele = tm.Telemetry()  # never started, so every call below is a no-op

    # `markpad` is a marks.MarkPad -- the two mark buttons. Same deal as tele:
    # one built here is harmless, because a MarkPad that never sees a press
    # opens no file and starts no thread.
    if markpad is None:
        markpad = mk.MarkPad(logger=log, tele=tele)

    # Unified/split. Built here and not in main() because this loop is the only
    # thing that reads it and the loop runs once for the life of the process,
    # which is exactly how long the toggle is meant to hold its state.
    if steer is None:
        steer = SteerMode(log)

    # Record mode. A session with no recorder factory refuses to begin, and a
    # tap with no view writes nothing, so the defaults are inert.
    from crawler.autopilot import footprint as fp
    if session is None:
        session = RecordSession(tele, getattr(autopilot, "hub", None), log,
                                markpad=markpad)
    if tap is None:
        tap = fp.FootprintTap(tele=tele)
    # Auto's rows are made with the decisions, off the thread that made
    # them; the AutoPilot is handed the tap for that.
    attach_tap = getattr(autopilot, "attach_tap", None)
    if attach_tap is not None:
        attach_tap(tap)
    if record_button is None:
        record_button = RecordButton()

    # Which pad groups my hand has taken off the autopilot. Only auto reads
    # it, and it is dropped on every way out of auto.
    if takeover is None:
        takeover = Takeover(log, markpad)

    # Which mode drove the previous pass, so a handover into manual from
    # record or auto -- by any path, the disconnect wait included -- arms the
    # throttle latch on the first manual pass after it.
    latch = ThrottleLatch(log)
    drove_last = modes.mode

    vehicle_state = None
    state_read_at = 0.0
    next_state_poll = 0.0
    was_connected = joystick.is_connected()
    # When the pad last proved it is there. See PAD_SILENCE_SECONDS.
    heard_at = clock()

    # Diagnostics bookkeeping. tick_max_ms is the number that matters: it is
    # the longest this loop has spent on one pass since the last diagnostics
    # message, and if telemetry ever did block the loop, this is where it
    # would show up. Watching it in Foxglove while driving is the point.
    last_error = ""
    tick_max_ms = 0.0
    tick_count = 0
    window_started = clock()
    # Every pass's time this window, for its p50 and p99 next to the max,
    # and the pad-to-write latency: from the event pump that read the pad
    # to the end of the drive call that commanded off it, and from the pump
    # before (an event can land just after a pump and wait a whole pass).
    # See telemetry.PassWindow.
    passes = tm.PassWindow()
    last_pump = None
    # The fixed period: a pass starts every LOOP_INTERVAL, the wait being
    # what is left of it once the work is done. It used to wait the whole
    # LOOP_INTERVAL after the work, so the period was the work plus 10 ms.
    # A pass that overran starts the next at once, from where it is: no
    # burst of passes to catch up.
    next_pass = None

    try:
        while not stop_event.is_set():
            now = clock()
            if next_pass is None:
                next_pass = now
            pygame.event.pump()

            joystick.update_connection_status()
            if not joystick.is_connected():
                if was_connected:
                    vecon.return_neutral()
                    log.log("warning", "Controller", "Lost the controller, holding neutral")
                    was_connected = False
                    last_error = "controller disconnected"
                    # First, and past the 4 Hz gate: the row that says the
                    # pad went, in whatever file is open right now. Offered
                    # before the session ends so it is the session file's last
                    # row -- telemetry drains the mailboxes into a session
                    # before it closes it. Gated, it was usually eaten: the
                    # loop parks in the wait below and there is no next pass.
                    _offer_connection(tele, now, modes.mode, False, last_error,
                                      autopilot, tick_max_ms, markpad)
                    # Losing the pad means losing the only way to stop the
                    # car, so auto does not get to keep driving without it.
                    takeover.drop("lost the controller")
                    if modes.mode == AUTO:
                        autopilot.pause()
                        modes.mode = MANUAL
                        log.log("warning", "Mode", "Parked auto, back to manual")
                    if modes.mode == RECORD:
                        # A Y tap's ending -- the file closed, the camera
                        # given back if the session opened it -- with the long
                        # pulse instead of the one buzz.
                        session.end(now, why="lost the controller",
                                    alarm_pad=joystick)
                        modes.mode = MANUAL
                    # Pictures with no driving in them. The rolling file
                    # carries on with everything else and the camera stays up
                    # for whoever wanted it; only its channels stop.
                    _mute_camera(tele)
                    vecon.return_neutral()
                # No pad: everything but driving carries on. The wheels are
                # held at neutral on every one of these passes (pigpio keeps
                # the last pulse forever, so it is commanded, not assumed),
                # and the pad is looked for again. Nothing in here reads the
                # pad, so nothing can start auto or record: START is a pad
                # button. This used to be joystick.wait_for_connection(),
                # which parked the whole loop -- the patrol's commands, the
                # map page and every status channel with it.
                padless_pass(joystick, vecon, autopilot, modes, tele, page,
                             now, last_error, markpad)
                # Interruptible, so a systemctl stop does not sit here until
                # TimeoutStopSec runs out and systemd SIGKILLs us past
                # vecon.close().
                stop_event.wait(PADLESS_INTERVAL)
                last_pump = None   # no pad was read; no latency to measure
                next_pass = None
                continue

            if not was_connected:
                log.log("info", "Controller", "Controller back")
                was_connected = True
                heard_at = now
                _unmute_camera(tele)
                _offer_connection(tele, now, modes.mode, True, last_error,
                                  autopilot, tick_max_ms, markpad)

            # A pad that is attached but has stopped answering. Not the sticks
            # sitting still -- a pad left alone on the table reads the same
            # numbers for minutes and is a working pad -- but no proof it is
            # there at all, for PAD_SILENCE_SECONDS. Record is what this ends:
            # in manual the wheels already go to neutral on every failed read.
            if _pad_heard(joystick):
                heard_at = now
            elif (modes.mode == RECORD
                    and now - heard_at >= PAD_SILENCE_SECONDS):
                vecon.return_neutral()
                last_error = (f"record ended: no pad for "
                              f"{PAD_SILENCE_SECONDS:.0f} s")
                log.log("warning", "Controller",
                        f"Nothing from the controller for "
                        f"{now - heard_at:.1f} s, holding neutral")
                _offer_connection(tele, now, modes.mode, False, last_error,
                                  autopilot, tick_max_ms, markpad)
                session.end(now, why="the controller went quiet",
                            alarm_pad=joystick)
                modes.mode = MANUAL
                log.log("info", "Mode", "Switched to manual")

            was_mode = modes.mode
            if modes.poll(joystick):
                # Hand the wheels over cleanly in either direction.
                vecon.return_neutral()
                if was_mode == RECORD:
                    # The session ends here whatever auto says next. The
                    # camera stays: auto is the thing about to want it.
                    session.end(now, keep_camera=True, why="START, to auto")
                if modes.mode == AUTO:
                    if not autopilot.resume():
                        modes.mode = MANUAL
                        log.log("warning", "Mode", "Auto refused to start, staying in manual")
                else:
                    autopilot.pause()

            # Polled in both modes and deliberately not folded into the mode
            # switch above: which way the sticks are mapped is not a property
            # of the mode, so a lap in auto leaves it exactly where I left it,
            # and pressing L3 while auto has the wheels sets up the mapping I
            # want waiting for me when I take them back. Under a --policy
            # model with a mode head the press also sets the mode the model
            # drives with (policy.ModeLatch); the rules driver ignores it.
            #
            # No return_neutral() on a flip, unlike a mode switch. Nobody is
            # changing hands here -- the next tick is 10 ms away and commands
            # the new mapping off the sticks as they are actually sitting, and
            # a neutral in between would be a jolt in the steering for no
            # reason. The throttle is not touched at all.
            steer.poll(joystick)

            # Y. Neither direction touches the wheels: nobody is changing
            # hands, and the next line commands this tick off the sticks under
            # whichever throttle range is now in force.
            action = record_button.poll(joystick, now, modes.mode)
            if action == ENTER_RECORD and modes.mode == MANUAL:
                if session.begin(now, autopilot.driver):
                    modes.mode = RECORD
                    log.log("info", "Mode", "Switched to record")
            elif action == EXIT_RECORD and modes.mode == RECORD:
                session.end(now, why="Y")
                modes.mode = MANUAL
                log.log("info", "Mode", "Switched to manual")

            pad_throttle = None
            if modes.mode == MANUAL and drove_last in ThrottleLatch.FROM:
                latch.arm(drove_last)
            drove_last = modes.mode
            # --goal: a click is taken in any mode and kept until replaced.
            goal_step = getattr(autopilot, "goal_step", None)
            if goal_step is not None:
                goal_step()
            if modes.mode == MANUAL:
                drive_manual(joystick, vecon, log, steer.split, latch)
            elif modes.mode == RECORD:
                _, pad_throttle, ok = drive_pad(joystick, vecon, log,
                                                steer.split, session.limits)
                session.tick(now)
                if not ok:
                    # A stale handle. drive_pad has already held neutral --
                    # the manual path, unchanged -- and the session is over.
                    session.end(now, why="pad read failed")
                    modes.mode = MANUAL
                    last_error = "record ended on a pad fault"
            elif not takeover.poll(joystick, now, steer.split):
                # The pad cannot be read, so the only way to stop the car is
                # not there either: the lost-pad answer, without the wait.
                autopilot.pause()
                modes.mode = MANUAL
                vecon.return_neutral()
                last_error = "auto parked on a pad fault"
                log.log("warning", "Mode", "Pad fault in auto, parked to manual")
            elif not autopilot.step(now, takeover, steer.split):
                takeover.drop("auto gave up")
                modes.mode = MANUAL
                vecon.return_neutral()
                last_error = "auto gave up"
                log.log("warning", "Mode", "Auto gave up, dropping to manual")
            wrote = clock()
            passes.add_pad(wrote - now, None if last_pump is None else wrote - last_pump)
            last_pump = now
            if modes.mode != AUTO:
                takeover.drop()
            elif takeover.throttle:
                pad_throttle = takeover.intent[0]

            # After driving, not before: the snapshot a mark carries is meant
            # to say what the car was being commanded at the moment I pressed
            # the button, and before the drive call that would be last tick's.
            # poll() cannot raise, does no file I/O and takes no lock anything
            # else wants -- see marks.py, which is written around the fact that
            # a mark is never worth a tick of this loop.
            markpad.poll(joystick, now, modes.mode, vecon,
                         autopilot.driver, autopilot)

            # The decision record. Handed the command as three floats read
            # back off the controller AFTER this pass commanded it -- never
            # the controller -- so it is what reached the ESC and nothing in
            # there can send anything back. footprint.py never raises. In
            # auto the AutoPilot makes the rows, one a decision, off what
            # this loop wrote for each; this pass only keeps the IMU.
            if modes.mode == AUTO and attach_tap is not None:
                poll_imu = getattr(tap, "poll_imu", None)
                if poll_imu is not None:
                    poll_imu(now)
            else:
                tap.after_drive(
                    now, modes.mode,
                    (float(vecon.last_throttle or 0.0),
                     float(vecon.last_steering_front or 0.0),
                     float(vecon.last_steering_rear or 0.0)),
                    driver=autopilot.driver if modes.mode == AUTO else None,
                    pad_throttle=pad_throttle,
                    # Every auto row carries the caps auto is driving at, whoever
                    # had the throttle: a row the autopilot drove is measured
                    # against them as much as one my trigger did. -1 is left for
                    # manual, which writes no rows.
                    limits=(session.limits if modes.mode == RECORD
                            else auto_speed_limits(autopilot.driver)
                            if modes.mode == AUTO else None),
                    steering_mode=steer.name,
                    steer_by=takeover.steer_by if modes.mode == AUTO else PAD,
                    throttle_by=takeover.throttle_by if modes.mode == AUTO else PAD,
                    takeover=modes.mode == AUTO and (takeover.steering
                                                     or takeover.throttle))

            # Telemetry. Every one of these is the same shape: ask whether
            # anybody is watching the channel and whether its rate gate has
            # come round, and only then build the message. With Foxglove shut
            # the wants() call is two dict lookups and a float compare, so
            # this whole block costs nothing and allocates nothing.
            #
            # offer() drops the message into a one-deep mailbox and returns.
            # It does not encode, it does not touch a socket, and it cannot
            # wait on the laptop. See telemetry.py.
            if tele.wants("controller", now):
                # Read the pad again rather than reuse what drive_manual
                # commanded, because in auto mode drive_manual did not run
                # and the sticks are still worth seeing. Three cached SDL
                # reads off the event pump that already happened; no syscall.
                try:
                    intent = read_pad_intent(joystick, steer.split)
                except Exception:
                    intent = (0.0, 0.0, 0.0)
                tele.offer("controller",
                           tm.controller_sample(joystick, *intent), now)

            if tele.wants("vehicle", now):
                readback_age = now - state_read_at if state_read_at else -1.0
                tele.offer("vehicle", {
                    "mode": modes.mode,
                    # Unified or split. On /vehicle rather than /controller
                    # because it is a property of how the car is being
                    # commanded, not of what the pad reads -- and next to
                    # steering_front/steering_rear, which is where I will be
                    # looking when the rear number is not what I expected.
                    "steering_mode": steer.name,
                    # What the last command asked for, recorded by
                    # VehicleController as it wrote it. Free; no bus traffic.
                    "throttle": float(vecon.last_throttle or 0.0),
                    "steering_front": float(vecon.last_steering_front or 0.0),
                    "steering_rear": float(vecon.last_steering_rear or 0.0),
                    "esc_pulse_us": int(vecon.last_esc_pulse or 0),
                    # What pigpio says is on GPIO18. A real round trip, so it
                    # is whatever the once-a-second poll below last read and
                    # not a fresh one -- reading it at 20 Hz would put bus I/O
                    # back on the throttle's thread. readback_age_s says how
                    # old it is so the number cannot be mistaken for live.
                    "esc_pulse_us_readback": int((vehicle_state or {}).get("throttle") or 0),
                    "steering_front_angle": float((vehicle_state or {}).get("front_s") or 0.0),
                    "steering_rear_angle": float((vehicle_state or {}).get("rear_s") or 0.0),
                    "readback_age_s": readback_age,
                }, now)

            # The goto: what it is doing and why, and the path it is on.
            # Only a --nav driver has them; built from what the navigator
            # already holds, nothing planned for them.
            nav_driver = getattr(autopilot, "driver", None)
            if (getattr(nav_driver, "nav_status", None) is not None
                    and tele.wants("nav_status", now)):
                tele.offer("nav_status", nav_driver.nav_status(), now)
            if (getattr(nav_driver, "nav_path", None) is not None
                    and tele.wants("nav_path", now)):
                tele.offer("nav_path", nav_driver.nav_path(), now)
            # --goal: the clicked goal, the patrol, the path, the map page and
            # the person followed. The same with or without a pad.
            goal_telemetry(tele, autopilot, page, now, pad=True,
                           mode=modes.mode)

            if tele.wants("diagnostics", now):
                elapsed = now - window_started
                tele.offer("diagnostics", {
                    "mode": modes.mode,
                    "controller_connected": True,
                    "auto_available": not autopilot.unavailable,
                    "camera_state": autopilot.camera_state(),
                    "camera_error": autopilot.camera_error(),
                    "last_error": last_error,
                    "loop_hz": (tick_count / elapsed) if elapsed > 0 else 0.0,
                    "loop_tick_max_ms": tick_max_ms,
                    # The window's passes, their p50 and p99, and the pad's
                    # pump-to-write p50, p99 and worst (telemetry.PassWindow).
                    **passes.fields(),
                    "camera_open_s": _camera_open_s(autopilot),
                    # Four attribute reads off the Recorder. This is how I
                    # check from Studio that the run I am watching is also
                    # being written down, instead of sshing in to look.
                    **tele.recording_status(),
                    # Two ints off the MarkPad. Same idea: the receipt that the
                    # button I just pressed did something.
                    **markpad.counts(),
                    # The decisions: their state, the newest one's age, and
                    # every late, crashed and restarted since boot.
                    **_decision_fields(autopilot, now),
                }, now)
                tick_max_ms = 0.0
                tick_count = 0
                window_started = now
                passes.clear()

            # Reading the state is three pigpio and I2C round trips, and this
            # loop runs at 100 Hz. Doing it every tick was about 54 MB of log
            # an hour: app.log rotating every eleven minutes, journald's rate
            # limit blown inside two, and a file open-write-flush every 10 ms
            # on the same thread that commands the throttle. Once a second,
            # and only when it changed.
            if now >= next_state_poll:
                next_state_poll = now + STATE_POLL_INTERVAL
                new_vehicle_state = vecon.get_state()
                if new_vehicle_state != vehicle_state:
                    log.log("info", "Vehicle State Change", f"New state: {new_vehicle_state}")
                    log.csv("log/vehicle_state_log.csv", "Vehicle State Change", new_vehicle_state)
                vehicle_state = new_vehicle_state
                state_read_at = now

            # How long that pass took, before the wait. This is measured after
            # everything including the telemetry block on purpose: it is the
            # evidence that the telemetry block costs nothing, and if it ever
            # stops costing nothing it shows up here and gets published on
            # /diagnostics as loop_tick_max_ms.
            tick_ms = (clock() - now) * 1000.0
            if tick_ms > tick_max_ms:
                tick_max_ms = tick_ms
            passes.add_pass(tick_ms)
            tick_count += 1

            next_pass += LOOP_INTERVAL
            left = next_pass - clock()
            if left < 0.0:
                next_pass -= left        # overran: from here, no catch-up
                left = 0.0
            stop_event.wait(min(left, LOOP_INTERVAL))
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[Shutdown] Stopping ESC and Servos...")
        autopilot.stop()
        vecon.close()
        # After the wheels. tele.stop() closes whichever file is open, so
        # this is only the bookkeeping and the journal line.
        try:
            session.end(clock(), keep_camera=True, why="shutting down")
        except Exception:
            pass
        # Last, and after the wheels are dealt with: the mark writer may have
        # a row still in hand, and it is the only thing in this process that
        # SIGTERM could cost me.
        markpad.stop()


def boot_mode(modes, joystick, autopilot, log):
    """--mode auto at boot drives only with a pad there to take it back: no
    pad, manual, and START once it is connected. Otherwise as it always was."""
    if modes.mode != AUTO:
        return
    if not joystick.is_connected():
        modes.mode = MANUAL
        log.log("warning", "Mode", "No controller, booting into manual; "
                "START once it is connected")
    elif not autopilot.resume():
        modes.mode = MANUAL
        log.log("warning", "Mode", "Auto refused to start, booting into manual")


def _decision_fields(autopilot, now):
    """AutoPilot.decision_fields for /diagnostics, or nothing for one that
    has none. Never raises."""
    fields = getattr(autopilot, "decision_fields", None)
    if fields is None:
        return {}
    try:
        return fields(now)
    except Exception:
        return {}


def goal_telemetry(tele, autopilot, page, now, pad=True, mode=None):
    """--goal's status channels and the map page's state, off what the goal,
    the patrol and the planner already hold. Every pass, pad or not: none of
    it drives anything. `pad` and `mode` are for the map page, which says
    when there is no controller to drive with.

    The goal and the patrol are read under the AutoPilot's goal lock, tried
    and never waited for: a decision that holds it has them half moved on,
    and the rate gates come round again next pass."""
    lock = getattr(autopilot, "goal_lock", None)
    if lock is None:
        _goal_telemetry(tele, autopilot, page, now, pad, mode)
        return
    if not lock.acquire(blocking=False):
        return
    try:
        _goal_telemetry(tele, autopilot, page, now, pad, mode)
    finally:
        lock.release()


def _goal_telemetry(tele, autopilot, page, now, pad=True, mode=None):
    # The recordings' tags, with or without --goal: once a second, and on
    # the pass one changes past the gate.
    tags = getattr(autopilot, "tags", None)
    if tags is not None:
        if tags.pop_changed():
            tele.offer_now("session_tags", tags.payload(), now)
        elif tele.wants("session_tags", now):
            tele.offer("session_tags", tags.payload(), now)
    # --goal: the clicked goal, from the truck, and its disc.
    if getattr(autopilot, "clicks", None) is not None:
        if tele.wants("goal_status", now):
            tele.offer("goal_status", autopilot.goal.status(), now)
        if tele.wants("goal_marker", now):
            tele.offer("goal_marker", autopilot.goal_marker(), now)
        # The patrol: where it is in the list, and the list on the map.
        if hasattr(autopilot.goal, "patrol_status"):
            if tele.wants("patrol_status", now):
                tele.offer("patrol_status", autopilot.goal.patrol_status(), now)
            if tele.wants("patrol_marker", now):
                tele.offer("patrol_marker", autopilot.goal.marker(), now)
        # Each pass and skip, one a pass, stamped at the decision it
        # happened on (the patrol's clock is this one).
        pop_event = getattr(autopilot.goal, "pop_event", None)
        event = pop_event() if pop_event is not None else None
        if event is not None:
            at = event.get("mono_s", now)
            tele.offer_now("patrol_event", event,
                           at if 0.0 <= now - at < 5.0 else now)
        # The path the goal model is pointed along, and the point fed.
        path = getattr(autopilot, "path", None)
        if path is not None:
            if tele.wants("path_status", now):
                tele.offer("path_status", path.status(), now)
            if tele.wants("path", now):
                tele.offer("path", path.marker(), now)
        # The map page (telemetry/buttons.py), only while it is open.
        if (page is not None and getattr(autopilot, "patrol", None) is not None
                and page.wants_state(now)):
            page.offer_state(map_page_state(autopilot, pad=pad, mode=mode, now=now), now)
        # Following a person: who, where, and the goal input fed.
        if hasattr(autopilot.goal, "follow_status"):
            if tele.wants("follow_status", now):
                crop = (autopilot.slam.crop()
                        if autopilot.slam is not None else None)
                tele.offer("follow_status",
                           autopilot.goal.follow_status(crop), now)
            if tele.wants("follow_marker", now):
                tele.offer("follow_marker", autopilot.goal.follow_marker(), now)


def padless_pass(joystick, vecon, autopilot, modes, tele, page, now,
                 last_error="", markpad=None):
    """One control pass with no pad attached. Never raises.

    The wheels: neutral, commanded every pass. The mode: nothing here
    changes it, and the lost-pad block has already parked auto and ended
    record, so it is manual. Everything else runs as it does with a pad:
    the clicks and patrol commands are drained into the goal (goal_step),
    the planner plans, the map page and the status channels get their
    state. Then one cheap look for a pad (Joystick.try_connect)."""
    if modes.mode != MANUAL:
        # Not reachable through the loop (the lost-pad block parks auto and
        # ends record first), but a pass with no pad is never a driving one.
        try:
            autopilot.pause()
        except Exception:
            pass
        modes.mode = MANUAL
    try:
        vecon.return_neutral()
    except Exception as e:
        print(f"[padless] neutral failed: {e}", file=sys.stderr)
    goal_step = getattr(autopilot, "goal_step", None)
    if goal_step is not None:
        goal_step()
    try:
        goal_telemetry(tele, autopilot, page, now, pad=False, mode=modes.mode)
        if tele.wants("diagnostics", now):
            payload = {
                "mode": modes.mode,
                "controller_connected": False,
                "auto_available": not autopilot.unavailable,
                "camera_state": autopilot.camera_state(),
                "camera_error": autopilot.camera_error(),
                "last_error": last_error,
                **tele.recording_status(),
            }
            if markpad is not None:
                payload.update(markpad.counts())
            tele.offer("diagnostics", payload, now)
    except Exception as e:
        print(f"[padless] {e}", file=sys.stderr)
    try_connect = getattr(joystick, "try_connect", None)
    if try_connect is not None:
        try:
            try_connect(now)
        except Exception as e:
            print(f"[padless] pad: {e}", file=sys.stderr)


def _offer_connection(tele, now, mode, connected, last_error, autopilot,
                      tick_max_ms, markpad):
    """The /diagnostics row for a pad that just went or came back.

    Past the rate gate, because this is an event and not a sample. Never
    raises: it is called on the way to neutral and on the way back.
    """
    try:
        payload = {
            "mode": mode,
            "controller_connected": bool(connected),
            "auto_available": not autopilot.unavailable,
            "camera_state": autopilot.camera_state(),
            "camera_error": autopilot.camera_error(),
            "last_error": last_error,
            "loop_hz": 0.0,
            "loop_tick_max_ms": tick_max_ms,
            "camera_open_s": _camera_open_s(autopilot),
            **tele.recording_status(),
            **markpad.counts(),
        }
        offer_now = getattr(tele, "offer_now", None)
        if offer_now is not None:
            offer_now("diagnostics", payload, now)
        else:
            tele.offer("diagnostics", payload, now)
    except Exception:
        pass


def _mute_camera(tele):
    from crawler.telemetry import telemetry as tm
    try:
        tele.mute(tm.PADLESS_MUTED_KEYS)
    except Exception:
        pass


def _unmute_camera(tele):
    try:
        tele.unmute()
    except Exception:
        pass


def _pad_heard(joystick) -> bool:
    """Did the pad prove it is there on this pass?

    Joystick.heard(). A pad object without one -- an older stand-in -- counts
    as heard; one whose heard() raises is a dead handle and does not.
    """
    heard = getattr(joystick, "heard", None)
    if heard is None:
        return True
    try:
        return bool(heard())
    except Exception:
        return False


def _camera_open_s(autopilot) -> float:
    try:
        return float(autopilot.hub.open_seconds)
    except Exception:
        return 0.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m crawler",
                                     description="Drive the RC rock crawler.")
    parser.add_argument(
        "--mode",
        choices=[MANUAL, AUTO],
        default=MANUAL,
        help="mode to boot into (default: manual)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="show the PyQt depth dashboard (needs a display)",
    )
    # On by default, and the reason is that it has to be on when I want it,
    # not when I remembered to ask for it. The whole value of this is opening
    # Foxglove while something is going wrong, and if it needed a flag I would
    # have to stop the service, edit the unit and restart it first -- by which
    # point whatever I wanted to look at is over. It is also as close to free
    # as I can make it: with nothing connected, every channel's has_sinks() is
    # false and the control loop builds no messages at all.
    #
    # --no-telemetry is the off switch, and it takes the whole thing out: no
    # thread, no socket, no port bound.
    parser.add_argument(
        "--telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="serve live telemetry to Foxglove (default: on)",
    )
    parser.add_argument(
        "--telemetry-port",
        type=int,
        default=8765,
        help="port for the Foxglove WebSocket server (default: 8765)",
    )
    # The button page (telemetry/buttons.py): patrol and follow from a phone
    # browser. On whenever the Foxglove server takes commands (--goal or
    # --nav with telemetry on), because a flag I have to remember is a page I
    # do not have when I want it. 0 is off.
    parser.add_argument(
        "--buttons-port",
        type=int,
        default=8080,
        help="port for the patrol/follow button page, served with --goal or "
             "--nav and telemetry on; 0 for none (default: 8080)",
    )
    # Off by default, and this one I thought about for a while.
    #
    # Turning it on means opening the OAK-D on every boot, including the
    # manual boot the service does -- a USB enumeration and a pipeline upload
    # that takes seconds, about a watt, and a whole new way for the one code
    # path that has to always work to have a bad day. The camera is not worth
    # that to the mode that is just driving the car.
    #
    # So the default leaves boot exactly as it was: no camera in manual, and
    # auto opens it on the first press of START. Ask for this when I want to
    # see what the camera sees while hand driving, which is a thing I do at a
    # desk, not a thing the car does unattended. In auto the camera is up
    # anyway, so /camera and /imu start flowing with or without this flag.
    parser.add_argument(
        "--telemetry-camera",
        action="store_true",
        help="open the OAK-D at startup so camera and IMU telemetry work in "
             "manual mode too (default: off, and auto mode opens it anyway)",
    )
    # On by default, for the same reason --telemetry is, only more so.
    #
    # Foxglove live is a window. The run I want to look at is always the one
    # that just ended badly, and by then the window is shut. A button to
    # start recording is a button I will not have pressed; recording only in
    # auto loses the manual runs, and the pad axis renumbering that cost me
    # an afternoon was a manual run. So every run gets written down, and the
    # disk budget in recording.py is what keeps that from being a problem:
    # the newest 8 GiB of driving is on the card and nothing older is.
    #
    # It is not free the way telemetry-with-nobody-watching is free. An MCAP
    # writer is a sink, so the channels are live whether or not Foxglove is
    # open, which means the publisher thread encodes camera JPEGs whenever
    # the camera is up. That cost is on the publisher thread, measured, and
    # in the README.
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write every run to log/mcap/*.mcap for replay (default: on)",
    )
    parser.add_argument(
        "--record-dir",
        default=None,
        help="where recordings go (default: log/mcap under the checkout)",
    )
    # Off unless named. With it, START hands the wheels to a learned model
    # instead of the hand-written rules -- behind the same floor, stuck and
    # camera vetoes, which the model cannot see past. models/ is gitignored,
    # so a model on the car is a file I copied there, never one git gave me.
    parser.add_argument(
        "--policy",
        default=None,
        metavar="MODEL.onnx",
        help="drive auto with a learned policy (a .onnx with its .json "
             "beside it) instead of the hand-written autopilot",
    )
    # Off unless named. A map --policy drives on this one while its map is
    # lost (a tilted chassis, the mapper down) instead of parking the car.
    parser.add_argument(
        "--fallback-policy",
        default=None,
        metavar="MODEL.onnx",
        help="with a map --policy, the map-free policy (a .onnx with its .json "
             "beside it) to drive on while the map is lost, instead of neutral",
    )
    # Off unless named, so the unit does not change until the roof lidar is
    # plugged in. Bare --lidar finds the USB-UART board on its own; a path
    # (/dev/ttyUSB0, /dev/serial0) names the port. Nothing drives on it yet:
    # it is read, binned in the car's frame and recorded on /lidar. A sensor
    # that is missing or goes away costs one line in the journal and nothing
    # else -- see crawler/hardware/lidar.py.
    parser.add_argument(
        "--lidar",
        nargs="?",
        const="auto",
        default=None,
        metavar="DEVICE",
        help="read the 2D lidar and record it on /lidar; DEVICE is a serial "
             "port or 'auto' (the default when the flag is bare). Off unless "
             "given",
    )
    # Off unless named. Needs --lidar and a map policy: under the stereo pair's
    # near clamp, the creep's stop comes off the lidar's front arc where the
    # lidar sees what the camera does (driver.py, "the near field").
    # On unless named off. A policy driver blocks the direction it was going
    # while the chassis leans past 20 degrees with something at a side
    # (autopilot/driver.py, "the tilt veto"); the other direction stays the
    # model's.
    parser.add_argument(
        "--no-tilt-veto",
        dest="tilt_veto",
        action="store_false",
        help="turn off the policy's tilt veto (on by default)",
    )
    parser.add_argument(
        "--lidar-near",
        action="store_true",
        help="let the policy's creep stop on the lidar (124.7 mm off the "
             "bumper) instead of the camera's clamp (150), where the two "
             "agree; needs --lidar. Off unless given",
    )
    # Off unless named, and mapping only: nothing reads the pose to drive
    # yet. Builds the room's map from the lidar as the truck goes, on a
    # worker of its own, and shows it in Foxglove (/map, /slam/path,
    # /slam/status, map -> base_link on /tf). Needs --lidar. Also starts the
    # camera without waiting, for the IMU's heading and tilt; with the camera
    # missing it maps on the lidar alone. See crawler/mapping/.
    parser.add_argument(
        "--slam",
        action="store_true",
        help="map the room from the lidar as the truck drives (needs "
             "--lidar); shown in Foxglove on /map. Off unless given",
    )
    parser.add_argument(
        "--slam-worker",
        choices=("thread", "process"),
        default="process",
        help="where the mapper runs: its own process (the default, it cannot "
             "hold the control loop's GIL) or a thread",
    )
    parser.add_argument(
        "--no-slam-map-file",
        action="store_true",
        help=f"keep nothing: by default --slam saves the map to "
             f"{SLAM_MAP_FILE} as it goes, a record of the run (the run "
             f"before it moved aside to slam_map.prev.npz)",
    )
    parser.add_argument(
        "--slam-map-restore",
        action="store_true",
        help=f"start from the map saved in {SLAM_MAP_FILE} instead of a "
             f"fresh one. Off by default: the truck may boot somewhere else, "
             f"and a map with a wrong heading written into it came back at "
             f"every boot",
    )
    # Off unless named, and mapping only. The OAK-D's depth as a 3D voxel map
    # placed by the SLAM pose, and the low obstacles the roof lidar passes
    # over (2.5D, under its plane), shown in Foxglove on /voxel/points and
    # /voxel/low. Needs --slam. See crawler/mapping/voxel.py.
    parser.add_argument(
        "--voxel-map",
        action="store_true",
        help="map the OAK-D's depth in 3D and the low obstacles under the "
             "lidar (needs --slam); shown in Foxglove on /voxel/points and "
             "/voxel/low. Off unless given",
    )
    # Off unless named. Point navigation (nav/pointnav.py): auto drives as it
    # would without this (the --policy model, else the rules) until a goal is
    # clicked in Foxglove (/move_base_simple/goal); then the navigator
    # (crawler/nav/) drives there on the live SLAM map (nav/live.py), or to
    # the reachable point nearest it, and holds neutral until the next goal.
    # It pauses neutral on a pose it does not trust. Needs --slam; without
    # it --nav is auto unavailable, said in the journal -- never a truck
    # driving on no map. The safety layer is under both (nav/driver.py).
    parser.add_argument(
        "--nav",
        action="store_true",
        help="drive auto to goals clicked in Foxglove on the SLAM map, and "
             "hold there; until the first goal auto drives as without it "
             "(needs --slam; off unless given)",
    )
    # Off unless named. A goal on the map for a goal model (autopilot/
    # policy_goal.py): Foxglove's Publish pose tool on the 3D panel, in the
    # map frame, on /move_base_simple/goal, taken in any mode and kept until
    # the next click. The model does all the driving to it; the one rule is
    # the arrival hold, neutral within policy_goal.ARRIVE_M until a new goal
    # (autopilot/goal.py). /goal/status and /goal/marker say where it is.
    # Needs --slam. Not with --nav: one thing reads the clicks.
    parser.add_argument(
        "--goal",
        action="store_true",
        help="take goals clicked in Foxglove on the SLAM map for a goal model "
             "to drive to, holding neutral on arrival (needs --slam; off "
             "unless given)",
    )
    # On with --goal unless this is given. The goal model is fed a point
    # PathPlanner picks along a path round the walls to the goal (autopilot/
    # pathgoal.py), planned on the SLAM map on a thread at 2 Hz; with it off
    # the model is fed the goal itself, the straight-line bearing, as it was.
    # Either way the model does all the driving.
    parser.add_argument(
        "--no-path",
        action="store_true",
        help="with --goal, feed the goal model the goal itself instead of a "
             "point along a planned path to it",
    )
    parser.add_argument(
        "--patrol-skip-stuck",
        action="store_true",
        help="with --goal, skip a patrol waypoint that has not been got 0.3 m "
             "closer to in 20 s (or reached in 120 s). Only the default: the "
             "map page's toggle is kept across a restart and wins (off unless "
             "given: a stuck waypoint stays the goal until it is passed)",
    )
    # patrol.near_miss: a waypoint come within this of and then left beside
    # or behind counts as passed (only the goal the model is fed changes).
    parser.add_argument(
        "--patrol-near-pass-m",
        type=float,
        default=patrol.NEAR_PASS_M,
        metavar="M",
        help="with --goal, pass a patrol waypoint the truck came within M "
             f"metres of once it is abeam, behind or going away (default "
             f"{patrol.NEAR_PASS_M:g}; 0 passes within {patrol.PASS_M:g} m only). "
             "Only the default: a {\"cmd\": \"near\", \"m\": M} is kept across "
             "a restart and wins",
    )
    # Off unless named. The OAK-D's low obstacles (--voxel-map) as extra
    # solid cells in the navigator's map, never clearing a lidar cell. Off
    # because how often it sees things that are not there is not settled.
    parser.add_argument(
        "--nav-low-obstacles",
        action="store_true",
        help="with --nav and --voxel-map, keep the navigator off the low "
             "obstacles the camera sees under the lidar (advisory; off "
             "unless given)",
    )
    # Off unless named. The hood camera on its pan/tilt servos turns to keep
    # the nearest person in the middle of the picture (crawler/vision/). It
    # never drives: the only thing it moves is the mount, on the PCA9685's
    # pan and tilt channels. Bare --pantilt reads deploy/pantilt.json; the
    # detector's model is models/person.onnx, see docs/PANTILT.md. A camera,
    # model or config that is missing costs the pan/tilt and nothing else.
    parser.add_argument(
        "--pantilt",
        nargs="?",
        const="deploy/pantilt.json",
        default=None,
        metavar="CONFIG",
        help="point the hood camera at the nearest person; CONFIG is the "
             "mount's JSON (default deploy/pantilt.json). Off unless given",
    )
    args = parser.parse_args(argv)
    if args.goal and args.nav:
        parser.error("--goal and --nav both take the Foxglove clicks; give one")
    return args


def nav_goals(args, log):
    """(goals, server_options) for --nav: the Foxglove click listener and
    the start_server() arguments that let a client publish to it. (None,
    None) without --nav, or if the listener cannot be built -- then --nav
    has no goals and stands still, which the journal says."""
    if not (getattr(args, "nav", False) or getattr(args, "goal", False)):
        return None, None
    try:
        from crawler.nav import click
        goals = click.ClickGoals()
        return goals, goals.server_options()
    except Exception as e:
        log.log("warning", "Nav", f"No Foxglove goals: {e}")
        return None, None


def start_buttons(args, tele, goals, log, slam=None, library=None):
    """The phone pages (telemetry/buttons.ButtonPage), or None. Only when the
    Foxglove server is up and takes client commands, since the buttons go
    through it; --buttons-port 0 is off. The map page as well when there is
    a SLAM map to draw."""
    port = getattr(args, "buttons_port", 0)
    if not port or goals is None or not getattr(tele, "serving", False):
        return None
    from crawler.nav import truck
    from crawler.telemetry import buttons as bt
    page = bt.ButtonPage(logger=log, host=tele.host, port=port, ws_port=tele.port,
                         goals=goals if slam is not None else None,
                         map_source=getattr(slam, "map", None), library=library,
                         turn_radius_m=truck.DRIVEN_TURN_RADIUS_M)
    return page if page.start() else None


TRACKS_DIR = "log/tracks"


def track_library(args, slam, log):
    """--goal's named tracks (autopilot/tracks.Library) and the fitter that
    puts a saved one on the fresh map (mapping/trackfit.Fitter, on a niced
    thread of its own), or (None, None). The fitter needs --slam and
    scipy; without it a saved track waits to be placed by hand."""
    if not getattr(args, "goal", False):
        return None, None
    from crawler.autopilot import tracks
    library = tracks.Library(lg.log_path(TRACKS_DIR))
    fitter = None
    if slam is not None:
        try:
            import scipy.fft, scipy.ndimage, scipy.spatial  # noqa: F401  (trackfit's)
            from crawler.mapping import trackfit
            fitter = trackfit.Fitter()
        except Exception as e:
            log.log("warning", "Patrol", f"No track fitting ({e}): a saved track "
                    "is placed by hand on the map page")
    return library, fitter


def map_page_state(autopilot, pad=True, mode=None, now=None):
    """The map page's snapshot, built on the control loop's thread: the
    patrol's page_state(), the planner's path, and whether there is a pad to
    drive with (the page says so when there is not) and the mode."""
    out = {"patrol": autopilot.patrol.page_state(), "pad": bool(pad), "mode": mode}
    status = getattr(autopilot, "decision_status", None)
    if status is not None:
        # Late, crashed or given up: the page puts it above the map.
        out["decisions"] = status(now)
    path = getattr(autopilot, "path", None)
    if path is not None:
        m = path.marker()
        out["path"] = [[round(x, 2), round(y, 2)] for x, y in m.get("path", [])]
        out["aim"] = list(m["aim"]) if m.get("aim") is not None else None
    return out


def nav_map_source(args, slam, lidar, voxels, log):
    """--nav's map: the live SLAM (nav/live.SlamMap), or None, which is auto
    unavailable. The low-obstacle layer only with --nav-low-obstacles and a
    --voxel-map to take it from."""
    if not getattr(args, "nav", False):
        return None
    if slam is None:
        log.log("warning", "Nav", "--nav needs --slam; auto will be unavailable")
        return None
    low = bool(getattr(args, "nav_low_obstacles", False))
    if low and voxels is None:
        log.log("warning", "Nav", "--nav-low-obstacles needs --voxel-map; "
                "driving on the lidar map alone")
        low = False
    from crawler.nav import live
    return live.SlamMap(slam, lidar=lidar, voxels=voxels, low_obstacles=low)


def follow_person(args, pantilt, lidar, slam, hub, log):
    """With --goal and --pantilt, the person to follow (autopilot/follow.py),
    off until the Follow on button; else None. Never raises."""
    if not getattr(args, "goal", False) or pantilt is None or slam is None:
        return None
    try:
        from crawler.autopilot import follow
        cam = follow.CameraGeometry.from_config(pantilt.cfg)
        oakd = follow.OakdRange()
        if hub is not None:
            hub.depth_listeners.append(oakd.on_depth)
        person = follow.PersonGoal(cam, sightings=pantilt, lidar=lidar,
                                   pose_at=getattr(slam, "pose_at", None),
                                   oakd=oakd, map_view=getattr(slam, "map", None),
                                   say=log.log)
        log.log("info", "Follow", "Follow on and off on /follow/cmd; the "
                f"person is the goal, holding {person.gap_m:g} m off")
        return person
    except Exception as e:
        log.log("warning", "Follow", f"No follow: {type(e).__name__}: {e}")
        return None


def start_pantilt(args, vecon, tele, log):
    """--pantilt's runner, started, or None. Never raises: a bad config, no
    camera or no model costs the pan/tilt, said in the journal."""
    if not getattr(args, "pantilt", None):
        return None
    try:
        from crawler.hardware import pantilt as pt
        from crawler.vision import runner as ptr
        cfg = pt.load_config(args.pantilt)
        runner = ptr.build(cfg, pca=getattr(vecon, "pca", None), tele=tele,
                           logger=log)
        runner.start()
        log.log("info", "PanTilt", f"Pan/tilt on {cfg.camera.device}, "
                f"{cfg.mount.kind} mount, model {cfg.detector.model}")
        return runner
    except Exception as e:
        log.log("warning", "PanTilt", f"No pan/tilt: {type(e).__name__}: {e}")
        return None


def main(argv=None):
    args = parse_args(argv)

    # SDL wants the name of a video driver before it will initialise at all,
    # and there is no display under a boot service. This is NOT what makes
    # the joystick work headless -- I had that wrong. What makes it work is
    # the service user being in group `input` so it can read
    # /dev/input/event*, which install-service.sh already grants. Keeping the
    # variable because SDL still asks for one; it is just not the magic.
    if not args.dashboard:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame
    from crawler.hardware import vehicle_controller as vc
    from crawler.hardware import joystick as js
    from crawler.telemetry import telemetry as tm

    pygame.init()
    log = lg.Logger()

    stop_event = threading.Event()

    # systemd stops the service with SIGTERM. Catch it so the loop unwinds
    # through vecon.close() and the ESC is left disarmed.
    def request_stop(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    # Disarm the car, then go looking for a controller. The old order built
    # the Joystick first, and that blocks waiting for a pad -- so for the
    # whole of that wait a pulse width left over from a hard kill was still
    # being emitted, with no handler installed to get me out of the wait.
    vecon = vc.VehicleController(logger=log)
    vecon.return_neutral()

    modes = ModeSwitch(args.mode, log)

    # Before the joystick, so the server is listening and the recording is
    # running from the start -- "I booted the car and it never found the pad"
    # is exactly the kind of run I want written down. start() never raises: a
    # missing foxglove-sdk, a port somebody else is already on, or a card
    # with no room on it all cost telemetry and nothing else.
    #
    # The recorder is constructed here and opened on the publisher thread.
    # Constructing it touches no disk.
    # The tags I set on the map page (telemetry/tags.py), kept across a
    # restart: before the recorder, whose every file starts with them.
    tags = tg.SessionTags(store=patrol.Store(lg.log_path(tg.SETTINGS_FILE)),
                          say=log.log)

    def tags_metadata():
        return {tg.METADATA_NAME: tags.metadata()}

    recorder = None
    if args.record:
        from crawler.telemetry import recording
        recorder = recording.Recorder(
            logger=log,
            directory=args.record_dir or recording.DEFAULT_DIR,
            metadata=tags_metadata,
        )

    # --nav takes its goals off Foxglove's publish tool, so the server has
    # to let a client publish. Without --nav the server is as it always was.
    goals, server_options = nav_goals(args, log)
    tele = tm.Telemetry(logger=log, port=args.telemetry_port,
                        recorder=recorder, server_options=server_options)

    # One thread for both jobs, so there is nothing to start if neither is
    # wanted: --no-telemetry --no-record is the old no-thread, no-socket,
    # nothing-bound behaviour exactly. --no-telemetry on its own now means
    # "record, but do not listen on the network", which is the right thing
    # for a car driving somewhere there is no laptop.
    channels_live = args.telemetry or args.record
    if channels_live:
        tele.start(serve=args.telemetry)
    # The phone pages start once the autopilot is built (below): the map
    # page draws its patrol and the SLAM map.
    buttons = None

    # One camera for the whole process, and constructing it opens nothing --
    # no USB, no depthai import, no risk. The rectified mono stream is only
    # linked when something is going to carry it, so with both telemetry and
    # recording off the pipeline is byte for byte the one auto mode has
    # always run.
    hub = camera.CameraHub(logger=log, tele=tele if channels_live else None,
                           rectified=channels_live)

    # The lidar is a thread of its own and start() returns at once, plugged
    # in or not: a missing sensor must never hold up the one boot that has to
    # work. Closed after the ESC is at neutral, with the camera.
    lidar = None
    slam = None
    if args.slam and not args.lidar:
        log.log("warning", "SLAM", "--slam needs --lidar; not mapping")
    if args.lidar:
        from crawler.hardware import lidar as ld
        if args.slam:
            # Before the reader, so its first scan has somewhere to go. The
            # worker comes up on its own; nothing here waits for it.
            from crawler.mapping import runner as slam_runner
            from crawler.nav import truck

            def commanded_speed():
                # What the throttle last written asks for on flat floor, for
                # the dead reckoning that carries a goal across a reseed.
                return truck.ground_speed(vecon.last_throttle or 0.0)
            slam = slam_runner.SlamRunner(
                mode=args.slam_worker, logger=log,
                tele=tele if channels_live else None,
                nav_half_m=slam_runner.NAV_HALF_M if args.nav else None,
                speed=commanded_speed,
                map_path=(None if args.no_slam_map_file
                          else lg.log_path(SLAM_MAP_FILE)),
                restore_map=args.slam_map_restore)
            slam.start()
            hub.imu_listeners.append(slam.on_imu)
            log.log("info", "SLAM", f"Mapping on a {args.slam_worker}")
        lidar = ld.LidarReader(device=args.lidar, logger=log,
                               tele=tele if channels_live else None,
                               slam=slam)
        lidar.start()
    voxels = None
    if args.voxel_map and slam is None:
        log.log("warning", "Voxel", "--voxel-map needs --slam; not mapping")
    elif args.voxel_map:
        # Its own process. The reader hands it each depth frame and returns;
        # a frame that comes while it is busy replaces the waiting one.
        from crawler.mapping import voxel as vx
        voxels = vx.VoxelMapper(slam, tele=tele if channels_live else None,
                                logger=log)
        voxels.start()
        hub.depth_listeners.append(voxels.on_depth)
        hub.imu_listeners.append(voxels.on_imu)
        log.log("info", "Voxel", "Mapping the OAK-D's depth")
    # Always, not only with --nav. The navigator plans on a thread of this
    # process (ThreadPlanner), and a search is pure Python for seconds on the
    # Pi, so it holds the GIL against this 100 Hz loop. Measured on the Pi, a
    # pass woke 22 ms late at p99 during a plan at the default 5 ms switch
    # interval and 12 ms at 1 ms, with the plans no slower. Without --nav
    # there are still threads holding it: the path planner, the lidar and
    # camera readers, telemetry; the loop's GIL wait is what holds auto at
    # 17-30 Hz with the Pi a third idle.
    sys.setswitchinterval(NAV_SWITCH_INTERVAL_S)
    if args.goal:
        if slam is None:
            log.log("warning", "Goal", "--goal needs --slam: a click has no map "
                    "to go on, so every one is refused")
        else:
            log.log("info", "Goal", "Taking goals from Foxglove's Publish pose "
                    "tool on /move_base_simple/goal, in the map frame, and patrol "
                    "commands on /patrol/cmd")
    path_planner = None
    if args.goal and slam is not None and not args.no_path:
        from crawler.autopilot import pathgoal
        path_planner = pathgoal.PathPlanner(slam, say=log.log)
        slam.map_listeners.append(path_planner.on_map)
        log.log("info", "Path", "The goal model is fed a point "
                f"{pathgoal.LOOKAHEAD_M:g} m along a path round the walls to "
                "the goal, planned on the SLAM map (--no-path for the goal itself)")
    pantilt = start_pantilt(args, vecon, tele if channels_live else None, log)
    library, fitter = track_library(args, slam, log)
    autopilot = AutoPilot(vecon, log, hub=hub, policy_path=args.policy,
                          decisions=THREAD,
                          fallback_path=args.fallback_policy,
                          lidar=lidar, slam=slam, nav=args.nav,
                          map_source=nav_map_source(args, slam, lidar, voxels, log),
                          goals=goals if args.nav else None,
                          clicks=goals if args.goal else None,
                          patrol_store=(patrol.Store(lg.log_path(PATROL_FILE))
                                        if args.goal else None),
                          lidar_near=args.lidar_near, path=path_planner,
                          tilt_veto=args.tilt_veto,
                          person=follow_person(args, pantilt, lidar, slam, hub, log),
                          library=library, fitter=fitter,
                          skip_stuck=args.patrol_skip_stuck,
                          near_pass_m=args.patrol_near_pass_m,
                          patrol_settings=(patrol.Store(lg.log_path(PATROL_SETTINGS_FILE))
                                           if args.goal else None),
                          tags=tags)
    buttons = start_buttons(args, tele, goals, log, slam=slam, library=library)

    # Who gets the device opened at startup, and who waits for it:
    #
    #   --telemetry-camera   start it now, do not wait. Manual driving comes
    #                        up immediately and the camera joins when it is
    #                        ready. This is the one that must not block.
    #   --mode auto          start it and wait, up to OPEN_TIMEOUT. The
    #                        control loop has not started yet, so waiting
    #                        here is safe and it is what the old code did.
    #   --dashboard          same, since the depth panes are its whole job.
    #   otherwise            nothing. The first press of START starts it.
    #   --slam               start it, do not wait: the mapper wants the IMU,
    #                        and maps on the lidar alone until it is up.
    if (args.telemetry_camera or modes.mode == AUTO or args.dashboard
            or slam is not None):
        hub.start()
    if modes.mode == AUTO or args.dashboard:
        if not hub.wait_ready():
            log.log("warning", "Camera",
                    f"Camera did not come up: {hub.error or hub.state}")

    # The two mark buttons. Built before the joystick for the same reason
    # telemetry is: it costs nothing to exist, and a press has to be able to
    # land the instant there is a pad to press.
    from crawler.autopilot import marks as mk
    markpad = mk.MarkPad(logger=log, tele=tele)

    # Record mode's two halves. Neither opens anything until Y is held: the
    # session's recorder is built on begin, and the tap's view is a one-deep
    # slot on a camera that is shut until somebody asks for it.
    from crawler.autopilot import footprint as fp
    from crawler.telemetry import recording

    def session_recorder():
        return recording.Recorder(
            logger=log,
            directory=args.record_dir or recording.DEFAULT_DIR,
            prefix=recording.SESSION_PREFIX,
            segment_bytes=recording.SESSION_SEGMENT_BYTES,
            max_total_bytes=None,
            metadata=tags_metadata,
        )

    session = RecordSession(tele, hub, log, make_recorder=session_recorder,
                            markpad=markpad, tags=tags)
    # The driver's own view, so auto rows shrink the frame it scanned.
    tap = fp.FootprintTap(view=hub.view("footprint"), tele=tele,
                          auto_view=hub.view(AUTO_VIEW))

    # The pad, if there is one, without waiting for it: everything but
    # driving runs without a pad, and the control loop looks for one at
    # 10 Hz (padless_pass). Booting with no pad is the same as losing one:
    # the camera's pictures are not written until there is a pad.
    joystick = js.Joystick(stop_event=stop_event, wait=False)
    if not joystick.is_connected():
        _mute_camera(tele)
        log.log("warning", "Controller", "No controller: running, holding "
                "neutral; connect it to drive")

    boot_mode(modes, joystick, autopilot, log)

    log.log("info", "Startup", f"Running in {modes.mode} mode")

    if not args.dashboard:
        try:
            control_loop(joystick, vecon, autopilot, modes, log, stop_event,
                         tele, markpad, session=session, tap=tap, page=buttons)
        finally:
            # After control_loop's own finally, which is what disarms the ESC.
            # The camera and the telemetry server go last because they are the
            # two things in this process that do not matter -- and because a
            # USB device that refuses to close must not get between SIGTERM
            # and the ESC reaching neutral.
            if pantilt is not None:
                pantilt.close()
            if lidar is not None:
                lidar.close()
            if voxels is not None:
                voxels.close()
            if slam is not None:
                slam.close()
            hub.close()
            if buttons is not None:
                buttons.stop()
            tele.stop()
        return 0

    # With the dashboard, Qt owns the main thread and the control loop runs
    # beside it.
    from PyQt5.QtWidgets import QApplication
    from crawler.telemetry import dashboard as db

    # The camera is already up by here -- wait_ready() above saw to it -- so
    # the dashboard gets its own view on it and nothing needs resuming. The
    # frame stealing this used to warn about is gone with it: its QTimers used
    # to tryGet() the same depth queue the driver did, and a queue hands each
    # packet to exactly one caller, so in auto the two of them halved each
    # other's rate. One reader in camera.CameraHub, one slot each, no race.
    worker = threading.Thread(
        target=control_loop,
        args=(joystick, vecon, autopilot, modes, log, stop_event, tele,
              markpad),
        kwargs={"session": session, "tap": tap, "page": buttons},
        daemon=True,
    )
    worker.start()

    app = QApplication(sys.argv[:1])
    dashboard = db.Dashboard(oakd=hub.view("dashboard"))
    dashboard.show()
    try:
        return app.exec_()
    finally:
        stop_event.set()
        worker.join(timeout=5)
        if pantilt is not None:
            pantilt.close()
        if lidar is not None:
            lidar.close()
        if voxels is not None:
            voxels.close()
        if slam is not None:
            slam.close()
        hub.close()
        if buttons is not None:
            buttons.stop()
        tele.stop()


if __name__ == "__main__":
    sys.exit(main())
