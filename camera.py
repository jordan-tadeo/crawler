"""One OAK-D, opened off the control thread, read once, handed to everybody.

Why this file exists
--------------------
dai.Device() takes exclusive USB access. One process, one owner, no sharing --
so for as long as the camera was built inside AutoPilot.resume(), the camera
only existed in auto mode. Which meant no camera telemetry while hand driving,
which is the one time I actually want to see what it sees.

It also could not be opened where it was being opened. dai.Device() blocks for
seconds while it uploads the pipeline and the device boots. resume() runs on
the control loop, and the control loop is the only thing that can take the ESC
back to neutral -- pigpio latches a pulse width and emits it forever, so a
loop parked for four seconds is a car driving for four seconds with nothing
watching the stop signal or the controller.

So: main owns one CameraHub. The hub opens the device on a thread of its own
and never on the caller's, reads every stream in one place, and fans the
frames out to as many consumers as ask for them.

The fan-out, and the queue it fixes
-----------------------------------
Before this, the driver and the dashboard both called tryGet() on the same
depth queue. A queue hands each packet to exactly one caller, so the two of
them stole frames from each other and neither got the full rate -- the
dashboard's own docstring says so. Now there is one reader, and each consumer
gets a view() with its own one-deep slot per stream. A view's getters have the
same shape as OakD's: the newest frame, or None if nothing new has landed
since the last call, which is what tryGet() meant and what driver.py is
written against. Nobody competes, and a slow consumer costs nobody else a
frame.

What a broken camera costs
--------------------------
Camera telemetry and auto mode. Never hand driving. depthai not installed, no
camera plugged in, a udev rule missing, the USB link dying mid-run: all of it
ends up as state == "failed" and an error string on /diagnostics. The control
loop does not import this module, does not construct it, and never calls
anything on it that can raise.
"""

import threading
import time

import telemetry as tm

# How often the reader thread asks the device for new packets. The depth
# stream delivers 16.0 fps -- oakd.CAPTURE_FPS, measured over 600 frames on
# the car -- so 5 ms is twelve times faster than it can produce and nothing
# waits on a frame that has already arrived. Each pass is four tryGet() calls,
# every one of which returns immediately.
#
# 30 is what this said for a long time and it was never true. 30 fps is what
# the mono sensors shutter at by default, not what arrives: the depth stream
# is whatever the stereo node can finish, and with the post-processing filters
# this pipeline runs that was 12.25 fps measured. The gap was not harmless --
# the surplus queued inside the device and depth frames arrived 642 ms old.
# oakd.py now caps capture under what the node can finish, which is why the
# number here is both lower than 30 and higher than what the car was actually
# getting. The real constraint is the stereo node's throughput, and the
# reasoning is all written down next to CAPTURE_FPS in oakd.py.
#
# Nothing here has to change if that cap moves. This interval only has to stay
# well under one frame interval, and at 5 ms it has a factor of twelve.
READ_INTERVAL = 0.005

# Longest the opener thread will sit in dai.Device() before giving up on it.
# There is no timeout argument, so this is only a bound on how long I will
# wait for the thread -- not on the call itself.
OPEN_TIMEOUT = 30.0

OFF = "off"
STARTING = "starting"
READY = "ready"
FAILED = "failed"
CLOSED = "closed"
# Being given back by release(), on a thread of its own, after which it is
# OFF again and start() can open it. CLOSED is close(), the end of the process.
RELEASING = "releasing"


class StreamView:
    """One consumer's private view of the shared camera.

    Deliberately the same method names OakD has, so driver.py and Dashboard.py
    take one of these instead of an OakD and neither of them changes. The
    semantics are the same too: every getter is a "newest, or None", never a
    wait.
    """

    def __init__(self, name):
        self.name = name
        # Plain attributes, not queues. A single assignment and a single read
        # of an attribute are each atomic under the GIL, so the reader thread
        # overwriting a slot while a consumer is taking it can cost at worst
        # one dropped frame and can never produce a torn value or a wait. A
        # Lock here would mean the reader thread could make the control loop
        # wait, which is the whole thing this design is avoiding.
        self._depth = None
        self._disparity = None
        self._rectified_left = None
        self._imu = None
        # The last depth frame this view handed out, kept after the take. The
        # driver's view is the only one that uses it: the footprint tap reads
        # it to shrink exactly the frame the driver just scanned, without a
        # second take that would steal the driver's next one.
        self.last_depth = None

        # Per-view, because it is per-view state: the dashboard's smoothing
        # should not depend on how often telemetry asked for a frame.
        self.smoothed_depth = None

    # ---- the hub's side --------------------------------------------------

    def _put(self, stream, frame):
        if stream == "depth":
            self._depth = frame
        elif stream == "disparity":
            self._disparity = frame
        elif stream == "rectified_left":
            self._rectified_left = frame
        elif stream == "imu":
            self._imu = frame

    # ---- the consumer's side ---------------------------------------------

    def _take(self, attr):
        frame = getattr(self, attr)
        if frame is None:
            return None
        setattr(self, attr, None)
        return frame

    def get_depth_frame(self):
        frame = self._take("_depth")
        if frame is not None:
            self.last_depth = frame
        return frame

    def get_disparity_frame(self):
        return self._take("_disparity")

    def get_rectified_left_frame(self):
        return self._take("_rectified_left")

    def get_imu_sample(self):
        return self._take("_imu")

    def get_depth_colormap(self):
        """The dashboard's smoothed depth picture, or None.

        Same arithmetic OakD.get_depth_colormap() did, moved here because the
        smoothing state belongs to whoever is looking, not to the device.
        """
        frame = self.get_depth_frame()
        if frame is None:
            return None
        try:
            import cv2
            import numpy as np
        except Exception:
            return None

        frame = np.clip(frame, 300, 3000).astype(np.float32)
        if self.smoothed_depth is None:
            self.smoothed_depth = frame.copy()
        else:
            cv2.accumulateWeighted(frame, self.smoothed_depth, 0.3)

        norm = ((self.smoothed_depth - 300) / (3000 - 300) * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    def get_disparity_colormap(self):
        frame = self.get_disparity_frame()
        if frame is None:
            return None
        try:
            import cv2
            import numpy as np
        except Exception:
            return None
        norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(np.uint8(norm), cv2.COLORMAP_JET)


class CameraHub:
    """The one OAK-D. Construct it freely -- this opens nothing.

    start() returns immediately and the device comes up on a background
    thread. state says where it got to, and is the only thing the control loop
    ever reads off this object.
    """

    def __init__(self, logger=None, tele=None, rectified=True):
        self.logger = logger
        self.tele = tele
        self.rectified = rectified

        self.state = OFF
        self.error = ""

        self.oakd = None
        self._views = {}
        self._views_lock = threading.Lock()
        self._stop = threading.Event()
        self._opener = None
        self._reader = None
        self._ready = threading.Event()

        # How long the last open took, start() to READY, in seconds. 0.0 until
        # one has finished. A record session waits on this and I want to know
        # what it costs me.
        self.open_seconds = 0.0
        self._start_requested = 0.0
        # close() rather than release(): nothing reopens after this.
        self._final = False

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> bool:
        """Begin opening the camera. Returns at once; does not raise.

        True only means the attempt started. Ask state afterwards.
        """
        if self.state in (STARTING, READY):
            return True
        if self.state in (CLOSED, RELEASING):
            # RELEASING is not a refusal for good: the caller asks again once
            # the release thread has put it back to OFF.
            return False

        self._start_requested = time.monotonic()
        self.state = STARTING
        self.error = ""
        self._stop.clear()
        self._ready.clear()
        self._opener = threading.Thread(
            target=self._open, name="camera-open", daemon=True)
        self._opener.start()
        return True

    def wait_ready(self, timeout=OPEN_TIMEOUT) -> bool:
        """Block until the camera is up or the attempt failed.

        Only safe to call before the control loop starts, or from a thread
        that is not the control loop. main uses it on the `--mode auto` and
        `--dashboard` paths, where the loop has not been started yet -- which
        is exactly where the old code blocked anyway.
        """
        self._ready.wait(timeout)
        return self.state == READY

    def _open(self):
        """The multi-second blocking construct, on a thread nobody waits on."""
        try:
            from oakd import OakD
            oakd = OakD(rectified=self.rectified)
        except BaseException as e:
            # BaseException on purpose. depthai is a native extension and a
            # bad one has taken this process down with things that are not
            # Exception subclasses. Auto mode is not worth a dead car.
            self.state = FAILED
            self.error = f"{type(e).__name__}: {e}"
            self._ready.set()
            self._log("error", "Camera", f"OAK-D would not open: {self.error}")
            return

        if self._stop.is_set():
            # Shut down while we were still booting it.
            try:
                oakd.close()
            except Exception:
                pass
            self.state = CLOSED
            self._ready.set()
            return

        self.oakd = oakd
        self.open_seconds = time.monotonic() - self._start_requested

        # The reader is started BEFORE the hub says READY. The other order
        # left a window where a release() racing the open found a reader
        # Thread that existed and had not been started, and join() raises on
        # that.
        self._reader = threading.Thread(
            target=self._read_loop, name="camera-read", daemon=True)
        self._reader.start()

        self.state = READY
        self._ready.set()
        self._log("info", "Camera", f"OAK-D up in {self.open_seconds:.1f} s"
                  + (" with the rectified stream" if self.rectified else ""))

    def close(self):
        """Release the device. Safe to call twice, never raises."""
        self._final = True
        self._shut(timeout=3.0)
        self.state = CLOSED

    def release(self) -> bool:
        """Give the device back without waiting, so it can be opened again.

        For a record session ending: the camera is shut at idle in manual by
        design, so a session that opened it closes it. dai.Device's close and
        the thread joins take real time, so they happen on a thread of their
        own and the caller -- the control loop -- returns at once. State is
        RELEASING until that is done and OFF after, and start() refuses in
        between rather than racing it.
        """
        if self.state not in (STARTING, READY):
            return False
        self.state = RELEASING
        threading.Thread(target=self._release, name="camera-release",
                         daemon=True).start()
        return True

    def _release(self):
        # No timeout on the joins. This thread is nobody's wait, and an opener
        # still inside dai.Device() has to finish and see the stop before the
        # hub may claim to be OFF -- otherwise it lands READY on a hub that
        # thinks it has no device.
        self._shut(timeout=None)
        if not self._final:
            self.state = OFF

    def _shut(self, timeout):
        self._stop.set()
        self._ready.set()

        for thread in (self._reader, self._opener):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=timeout)
        self._reader = None
        self._opener = None

        oakd, self.oakd = self.oakd, None
        if oakd is not None:
            try:
                oakd.close()
            except Exception as e:
                print(f"[camera] close failed: {e}")

    # ---- consumers -------------------------------------------------------

    def view(self, name) -> StreamView:
        """A private view for one consumer. Same name gets the same view."""
        with self._views_lock:
            view = self._views.get(name)
            if view is None:
                view = StreamView(name)
                # Rebind rather than mutate, so the reader thread is always
                # iterating a list that is not being appended to underneath
                # it. No lock on the read side at all.
                self._views[name] = view
                self._view_list = list(self._views.values())
            return view

    _view_list = ()

    # ---- the reader thread -----------------------------------------------

    def _read_loop(self):
        """Drain every stream once and hand each packet to every view.

        Draining is not optional and is not gated on anybody watching: the
        driver needs depth, and a queue nobody reads fills up and stalls the
        stream behind it. What IS gated is the expensive part -- the colormap
        and the JPEG encode -- and that happens on the telemetry thread, only
        for channels a Foxglove client is subscribed to. This thread hands
        over raw frames and does no image work at all.
        """
        oakd = self.oakd

        while not self._stop.is_set():
            now = time.monotonic()

            for stream, getter in (
                ("depth", oakd.get_depth_frame),
                ("disparity", oakd.get_disparity_frame),
                ("rectified_left", oakd.get_rectified_left_frame),
                ("imu", oakd.get_imu_sample),
            ):
                try:
                    packet = getter()
                except Exception as e:
                    # The link dying must not kill this thread: the views then
                    # go on returning None, driver.tick() zeroes the throttle,
                    # and the whole thing stays diagnosable from /diagnostics.
                    self.error = f"{stream}: {e}"
                    continue

                if packet is None:
                    continue

                for view in self._view_list:
                    view._put(stream, packet)

                self._offer(stream, packet, now)

            self._stop.wait(READ_INTERVAL)

    def _offer(self, stream, packet, now):
        """Hand a frame to telemetry, if telemetry wants one.

        wants() is false unless a Foxglove client is actually subscribed to
        that channel and its rate gate has come round, so with Studio shut
        this is a dict lookup per stream and nothing is encoded, colormapped
        or copied.
        """
        tele = self.tele
        if tele is None:
            return

        if stream == "imu":
            if tele.wants("imu", now):
                sample = tm.imu_sample(packet)
                if sample is not None:
                    tele.offer("imu", sample, now)
            return

        if tele.wants(stream, now):
            tele.offer(stream, packet, now)

    def _log(self, level, subject, message):
        if self.logger is not None:
            try:
                self.logger.log(level, subject, message)
                return
            except Exception:
                pass
        print(f"[camera] {subject}: {message}")
