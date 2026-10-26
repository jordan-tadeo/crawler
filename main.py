"""Entry point for the crawler.

Two drive modes:

  manual  the Xbox controller drives throttle and both steering axles. Default.
  auto    PersonFollower runs the pan/tilt camera. The wheels stay neutral,
          because nothing in the follower commands throttle or steering yet.

Press START on the controller to switch between them without restarting.

The heavy imports (PyQt5, TensorFlow) are deliberately deferred: manual
driving is the mode that has to come up on a headless boot, and it should not
depend on the vision stack being installed or the model files being in place.
"""

import argparse
import os
import signal
import sys
import threading

import Logger as lg

MANUAL = "manual"
AUTO = "auto"

# Button that flips between the two modes. START and BACK are the two buttons
# that cannot be hit by accident while driving.
MODE_TOGGLE_BUTTON = "START"

LOOP_INTERVAL = 0.01
STEERING_LIMIT_PERC = 70


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
    """Lazy wrapper around PersonFollower.

    Nothing here is constructed until auto mode is actually asked for, and a
    failure to construct it is logged and swallowed rather than raised. A
    missing camera or model file should cost auto mode, not the ability to
    drive the car by hand.
    """

    def __init__(self, vecon, logger, camera_index=0):
        self.vecon = vecon
        self.logger = logger
        self.camera_index = camera_index
        self.camera = None
        self.follower = None
        self.unavailable = False

    def resume(self):
        """Start or unpause the follower. Returns True if it is running."""
        if self.unavailable:
            return False

        if self.follower is None:
            try:
                import USBCamera as uc
                import PersonFollower as pf

                self.camera = uc.USBCamera(camera_index=self.camera_index, fps=30)
                self.follower = pf.PersonFollower(self.vecon, self.camera)
                self.follower.start()
            except Exception as e:
                self.unavailable = True
                self.logger.log("error", "Auto mode unavailable", str(e))
                print(f"[auto] unavailable: {e}", file=sys.stderr)
                return False
        else:
            self.follower.resume()

        return True

    def pause(self):
        if self.follower is not None:
            self.follower.pause()

    def stop(self):
        if self.follower is not None:
            self.follower.stop()
            self.follower = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None


def drive_manual(joystick, vecon):
    """One tick of hand driving."""
    throttle = joystick.read_throttle()
    steering_front = joystick.get_axis("LEFT_X", limit_perc=STEERING_LIMIT_PERC)
    steering_rear = joystick.get_axis("RIGHT_X", limit_perc=STEERING_LIMIT_PERC)

    vecon.set_throttle(throttle)
    vecon.set_steering(steering_front, steering_rear)


def control_loop(joystick, vecon, autopilot, modes, log, stop_event):
    """Poll the controller and drive whichever mode is current.

    Only one mode ever has the wheels: in auto the throttle is held at neutral,
    so a mode switch can never leave the car rolling under two owners.
    """
    import pygame

    vehicle_state = None
    was_connected = True

    try:
        while not stop_event.is_set():
            pygame.event.pump()

            joystick.update_connection_status()
            if not joystick.is_connected():
                if was_connected:
                    # A Bluetooth dropout mid-drive must not leave the throttle
                    # wherever it happened to be.
                    log.log("warning", "Controller", "Lost the controller, holding neutral")
                    vecon.return_neutral()
                    was_connected = False
                joystick.wait_for_connection()
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
                drive_manual(joystick, vecon)

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
        help="show the PyQt camera dashboard (needs a display)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="USB camera index for auto mode (default: 0)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # SDL still wants a video driver to initialise, even when all we read is a
    # joystick. Without this there is no joystick under a boot service.
    if not args.dashboard:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame
    import Joystick as js
    import VehicleController as vc

    pygame.init()
    log = lg.Logger()

    joystick = js.Joystick()
    vecon = vc.VehicleController(logger=log)
    autopilot = AutoPilot(vecon, log, camera_index=args.camera_index)
    modes = ModeSwitch(args.mode, log)

    stop_event = threading.Event()

    # systemd stops the service with SIGTERM. Catch it so the loop unwinds
    # through vecon.close() and the ESC is left at neutral.
    def request_stop(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

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

    if autopilot.follower is None and not autopilot.resume():
        print("[dashboard] auto mode is unavailable, nothing to show", file=sys.stderr)
        return 1
    if modes.mode == MANUAL:
        autopilot.pause()

    worker = threading.Thread(
        target=control_loop,
        args=(joystick, vecon, autopilot, modes, log, stop_event),
        daemon=True,
    )
    worker.start()

    app = QApplication(sys.argv[:1])
    dashboard = db.Dashboard(autopilot.follower)
    dashboard.show()
    try:
        return app.exec_()
    finally:
        stop_event.set()
        worker.join(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
