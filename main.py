"""Entry point for the crawler.

Two drive modes:

  manual  the Xbox controller drives throttle and both steering axles. Default.
  auto    the OAK-D roomba: driver.Driver crawls forward on stereo depth and
          backs out of whatever it runs into.

Press START on the controller to switch between them without restarting.

One thread owns the ESC and the servos. Auto mode is not a thread of its own,
it is a step() call on this loop, so a mode switch can never leave two owners
writing opposing commands to the same hardware.

The heavy imports (depthai, PyQt5) are deferred on purpose: manual driving is
the mode that has to come up on a headless boot, and it must not depend on
the vision stack being installed or the camera being plugged in.

The thing to keep in mind everywhere below: pigpio latches a servo pulse
width and emits it forever. There is no hardware failsafe on this car and no
RC receiver to fall back on. A crashed thread, a blocked loop or a SIGKILL
does not stop it -- so every error path in here has to end at neutral, and
nothing on this thread is allowed to block.
"""

import argparse
import os
import signal
import sys
import threading
import time

import camera
import Logger as lg

MANUAL = "manual"
AUTO = "auto"

# Button that flips between the two modes. START and BACK are the two buttons
# that cannot be hit by accident while driving.
MODE_TOGGLE_BUTTON = "START"

LOOP_INTERVAL = 0.01
AUTO_TICK_INTERVAL = 0.1
STEERING_LIMIT_PERC = 70

# How often the loop is allowed to read and log the vehicle state. See the
# comment at the call site -- this is not cosmetic.
STATE_POLL_INTERVAL = 1.0


class ModeSwitch:
    """The current drive mode, plus edge detection for the toggle button."""

    def __init__(self, mode, logger):
        self.mode = mode
        self.logger = logger
        self._toggle_was_down = False

    def poll(self, joystick):
        """Flip the mode on a press of the toggle button. Returns True on flip.

        A hold must not retrigger, so the press edge is what counts.
        """
        try:
            down = bool(joystick.get_button(MODE_TOGGLE_BUTTON))
        except Exception:
            # No controller attached yet, or a stale handle after a dropout.
            self._toggle_was_down = False
            return False

        pressed = down and not self._toggle_was_down
        self._toggle_was_down = down

        if pressed:
            self.mode = AUTO if self.mode == MANUAL else MANUAL
            self.logger.log("info", "Mode", f"Switched to {self.mode}")
        return pressed


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

    step() still runs on the control loop's own thread. Driver.tick() never
    blocks, so ticking it at 10 Hz from inside a 100 Hz loop is a timestamp
    comparison.
    """

    def __init__(self, vecon, logger, hub=None):
        self.vecon = vecon
        self.logger = logger
        self.driver = None
        self.unavailable = False
        self._next_tick = 0.0
        # The shared camera, or None if this process was started without one.
        self.hub = hub

    def resume(self):
        """Start or restart auto mode. True if it is running.

        Never blocks and never raises. False can mean three different things
        and only one of them is permanent:

          no hub at all, or the hub failed      -> unavailable, latched
          the hub is still opening the device   -> try again in a moment
          Driver would not construct            -> unavailable, latched
        """
        if self.unavailable:
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

            if hub.state == camera.STARTING:
                # Not a failure. Do not latch unavailable -- the device takes
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
                from driver import Driver
                self.driver = Driver(self.vecon, hub.view("auto"))
            except Exception as e:
                self.unavailable = True
                self.logger.log("error", "Auto mode unavailable", str(e))
                print(f"[auto] unavailable: {e}", file=sys.stderr)
                return False

        self._next_tick = 0.0  # tick on the next step
        return True

    def step(self, now):
        """One pass of auto mode. False means hand the wheels back to manual."""
        if self.unavailable or self.driver is None:
            return False

        if now < self._next_tick:
            return True
        self._next_tick = now + AUTO_TICK_INTERVAL

        try:
            self.driver.tick(now)
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

    def pause(self):
        """Hand the wheels back. Drops a recover in progress on the floor."""
        if self.driver is not None:
            self.driver.abort()
        self.vecon.return_neutral()

    def stop(self):
        """Give up the driver. The camera is not mine to close.

        main owns the hub and closes it, because the camera outlives auto
        mode now -- telemetry is still publishing frames in manual.
        """
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


def read_pad_intent(joystick):
    """What the pad is asking for: (throttle, steering front, steering rear).

    Reads only. Pulled out of drive_manual so telemetry can report what the
    sticks are saying while auto has the wheels, without there being a second
    copy of the three readings that could drift away from the ones that
    actually drive the car.
    """
    return (
        joystick.read_throttle(),
        joystick.get_axis("LEFT_X", limit_perc=STEERING_LIMIT_PERC),
        joystick.get_axis("RIGHT_X", limit_perc=STEERING_LIMIT_PERC),
    )


def drive_manual(joystick, vecon, log):
    """One tick of hand driving. Anything that goes wrong in here means neutral.

    Returns what it commanded, so the caller does not have to read the pad a
    second time to report it.
    """
    try:
        throttle, steering_front, steering_rear = read_pad_intent(joystick)

        vecon.set_throttle(throttle)
        vecon.set_steering(steering_front, steering_rear)
        return throttle, steering_front, steering_rear
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
        return 0.0, 0.0, 0.0


def control_loop(joystick, vecon, autopilot, modes, log, stop_event, tele=None):
    """Poll the controller and drive whichever mode is current.

    Only one mode ever has the wheels: auto runs as a step() on this thread
    and manual writes nothing while auto is up, so a switch can never leave
    the car rolling under two owners.

    `tele` is a telemetry.Telemetry, or None. A Telemetry that failed to start
    is already a complete no-op -- wants() is False for every channel -- so
    there is no third case to handle and no branch in here that depends on
    whether Foxglove is installed, running or being watched.
    """
    import pygame
    import telemetry as tm

    if tele is None:
        tele = tm.Telemetry()  # never started, so every call below is a no-op

    vehicle_state = None
    state_read_at = 0.0
    next_state_poll = 0.0
    was_connected = joystick.is_connected()

    # Diagnostics bookkeeping. tick_max_ms is the number that matters: it is
    # the longest this loop has spent on one pass since the last diagnostics
    # message, and if telemetry ever did block the loop, this is where it
    # would show up. Watching it in Foxglove while driving is the point.
    last_error = ""
    tick_max_ms = 0.0
    tick_count = 0
    window_started = time.monotonic()

    try:
        while not stop_event.is_set():
            now = time.monotonic()
            pygame.event.pump()

            joystick.update_connection_status()
            if not joystick.is_connected():
                if was_connected:
                    log.log("warning", "Controller", "Lost the controller, holding neutral")
                    # Losing the pad means losing the only way to stop the
                    # car, so auto does not get to keep driving without it.
                    if modes.mode == AUTO:
                        autopilot.pause()
                        modes.mode = MANUAL
                        log.log("warning", "Mode", "Parked auto, back to manual")
                    vecon.return_neutral()
                    was_connected = False
                    last_error = "controller disconnected"
                # Say so once before parking in the wait below, or the last
                # thing Foxglove ever saw would be a connected pad.
                if tele.wants("diagnostics", now):
                    tele.offer("diagnostics", {
                        "mode": modes.mode,
                        "controller_connected": False,
                        "auto_available": not autopilot.unavailable,
                        "camera_state": autopilot.camera_state(),
                        "camera_error": autopilot.camera_error(),
                        "last_error": last_error,
                        "loop_hz": 0.0,
                        "loop_tick_max_ms": tick_max_ms,
                    }, now)
                # Interruptible, so a systemctl stop does not sit here until
                # TimeoutStopSec runs out and systemd SIGKILLs us past
                # vecon.close().
                joystick.wait_for_connection(stop_event)
                continue

            if not was_connected:
                log.log("info", "Controller", "Controller back")
                was_connected = True

            if modes.poll(joystick):
                # Hand the wheels over cleanly in either direction.
                vecon.return_neutral()
                if modes.mode == AUTO:
                    if not autopilot.resume():
                        modes.mode = MANUAL
                        log.log("warning", "Mode", "Auto refused to start, staying in manual")
                else:
                    autopilot.pause()

            if modes.mode == MANUAL:
                drive_manual(joystick, vecon, log)
            elif not autopilot.step(now):
                modes.mode = MANUAL
                vecon.return_neutral()
                last_error = "auto gave up"
                log.log("warning", "Mode", "Auto gave up, dropping to manual")

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
                    intent = read_pad_intent(joystick)
                except Exception:
                    intent = (0.0, 0.0, 0.0)
                tele.offer("controller",
                           tm.controller_sample(joystick, *intent), now)

            if tele.wants("vehicle", now):
                readback_age = now - state_read_at if state_read_at else -1.0
                tele.offer("vehicle", {
                    "mode": modes.mode,
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
                }, now)
                tick_max_ms = 0.0
                tick_count = 0
                window_started = now

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
            tick_ms = (time.monotonic() - now) * 1000.0
            if tick_ms > tick_max_ms:
                tick_max_ms = tick_ms
            tick_count += 1

            stop_event.wait(LOOP_INTERVAL)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[Shutdown] Stopping ESC and Servos...")
        autopilot.stop()
        vecon.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Drive the RC rock crawler.")
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
    return parser.parse_args(argv)


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
    import VehicleController as vc
    import Joystick as js
    import telemetry as tm

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

    # Before the joystick, because Joystick() blocks waiting for a pad and I
    # want the server already listening while it waits -- that wait is one of
    # the things worth watching from the laptop. start() never raises: a
    # missing foxglove-sdk or a port somebody else is already on costs
    # telemetry and nothing else.
    tele = tm.Telemetry(logger=log, port=args.telemetry_port)
    if args.telemetry:
        tele.start()

    # One camera for the whole process, and constructing it opens nothing --
    # no USB, no depthai import, no risk. The rectified mono stream is only
    # linked when telemetry is going to carry it, so with telemetry off the
    # pipeline is byte for byte the one auto mode has always run.
    hub = camera.CameraHub(logger=log, tele=tele if args.telemetry else None,
                           rectified=args.telemetry)
    autopilot = AutoPilot(vecon, log, hub=hub)

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
    if args.telemetry_camera or modes.mode == AUTO or args.dashboard:
        hub.start()
    if modes.mode == AUTO or args.dashboard:
        if not hub.wait_ready():
            log.log("warning", "Camera",
                    f"Camera did not come up: {hub.error or hub.state}")

    joystick = js.Joystick(stop_event=stop_event)

    if modes.mode == AUTO and not autopilot.resume():
        modes.mode = MANUAL
        log.log("warning", "Mode", "Auto refused to start, booting into manual")

    log.log("info", "Startup", f"Running in {modes.mode} mode")

    if not args.dashboard:
        try:
            control_loop(joystick, vecon, autopilot, modes, log, stop_event, tele)
        finally:
            # After control_loop's own finally, which is what disarms the ESC.
            # The camera and the telemetry server go last because they are the
            # two things in this process that do not matter -- and because a
            # USB device that refuses to close must not get between SIGTERM
            # and the ESC reaching neutral.
            hub.close()
            tele.stop()
        return 0

    # With the dashboard, Qt owns the main thread and the control loop runs
    # beside it.
    from PyQt5.QtWidgets import QApplication
    import Dashboard as db

    # The camera is already up by here -- wait_ready() above saw to it -- so
    # the dashboard gets its own view on it and nothing needs resuming. The
    # frame stealing this used to warn about is gone with it: its QTimers used
    # to tryGet() the same depth queue the driver did, and a queue hands each
    # packet to exactly one caller, so in auto the two of them halved each
    # other's rate. One reader in camera.CameraHub, one slot each, no race.
    worker = threading.Thread(
        target=control_loop,
        args=(joystick, vecon, autopilot, modes, log, stop_event, tele),
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
        hub.close()
        tele.stop()


if __name__ == "__main__":
    sys.exit(main())
