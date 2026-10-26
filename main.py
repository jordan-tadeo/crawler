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
    """Lazy wrapper around the OAK-D driver.

    Nothing is constructed until auto mode is actually asked for, and a
    failure to construct it is logged and swallowed rather than raised. An
    unplugged camera or a missing depthai should cost me auto mode, not the
    ability to drive the car by hand.

    step() runs on the control loop's own thread. Driver.tick() never blocks,
    so ticking it at 10 Hz from inside a 100 Hz loop is just a timestamp
    comparison.
    """

    def __init__(self, vecon, logger):
        self.vecon = vecon
        self.logger = logger
        self.oakd = None
        self.driver = None
        self.unavailable = False
        self._next_tick = 0.0

    def resume(self):
        """Start or restart auto mode. True if it is running."""
        if self.unavailable:
            return False

        if self.driver is None:
            try:
                from oakd import OakD
                from driver import Driver

                self.oakd = OakD()
                self.driver = Driver(self.vecon, self.oakd)
            except Exception as e:
                self.unavailable = True
                self._release()
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
        self.driver = None
        self._release()

    def _release(self):
        if self.oakd is not None:
            try:
                self.oakd.close()
            except Exception as e:
                print(f"[auto] OAK-D close failed: {e}", file=sys.stderr)
            self.oakd = None


def drive_manual(joystick, vecon, log):
    """One tick of hand driving. Anything that goes wrong in here means neutral."""
    try:
        throttle = joystick.read_throttle()
        steering_front = joystick.get_axis("LEFT_X", limit_perc=STEERING_LIMIT_PERC)
        steering_rear = joystick.get_axis("RIGHT_X", limit_perc=STEERING_LIMIT_PERC)

        vecon.set_throttle(throttle)
        vecon.set_steering(steering_front, steering_rear)
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


def control_loop(joystick, vecon, autopilot, modes, log, stop_event):
    """Poll the controller and drive whichever mode is current.

    Only one mode ever has the wheels: auto runs as a step() on this thread
    and manual writes nothing while auto is up, so a switch can never leave
    the car rolling under two owners.
    """
    import pygame

    vehicle_state = None
    next_state_poll = 0.0
    was_connected = joystick.is_connected()

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
                log.log("warning", "Mode", "Auto gave up, dropping to manual")

            # Reading the state is five pigpio and I2C round trips, and this
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

    autopilot = AutoPilot(vecon, log)
    modes = ModeSwitch(args.mode, log)

    joystick = js.Joystick(stop_event=stop_event)

    if modes.mode == AUTO and not autopilot.resume():
        modes.mode = MANUAL
        log.log("warning", "Mode", "Auto refused to start, booting into manual")

    log.log("info", "Startup", f"Running in {modes.mode} mode")

    if not args.dashboard:
        control_loop(joystick, vecon, autopilot, modes, log, stop_event)
        return 0

    # With the dashboard, Qt owns the main thread and the control loop runs
    # beside it.
    from PyQt5.QtWidgets import QApplication
    import Dashboard as db

    # The dashboard's whole job is the depth panes, so bring the camera up
    # even in manual. Worth knowing: its QTimers pull from the same depth
    # queue the driver does, so in auto the two steal frames from each other
    # and neither gets the full rate. It is a debug flag, not a boot path.
    if autopilot.resume() and modes.mode == MANUAL:
        autopilot.pause()

    worker = threading.Thread(
        target=control_loop,
        args=(joystick, vecon, autopilot, modes, log, stop_event),
        daemon=True,
    )
    worker.start()

    app = QApplication(sys.argv[:1])
    dashboard = db.Dashboard(oakd=autopilot.oakd)
    dashboard.show()
    try:
        return app.exec_()
    finally:
        stop_event.set()
        worker.join(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
