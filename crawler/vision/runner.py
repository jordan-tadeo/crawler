"""--pantilt: keep the person in front of the truck in the hood camera.

One thread, at track.step_hz (20 Hz). Each pass it collects the detector's
answer if there is one, hands the newest camera frame to the detector if the
detector is free and the rate governor says it is time, steps the tracker,
writes the mount, and offers telemetry. The camera has a thread of its own
(pantilt.FrameGrabber) and the detector a process of its own
(person.DetectorClient), so nothing here waits on either.

What it never does: command throttle or steering, read the pad, or touch
the OAK-D. The mount is the only thing it moves, and a servo mount refuses
the steering channels (pantilt.check_channel). Anything that goes wrong in
here costs the pan/tilt and nothing else: the loop catches it, says so in
the journal once a minute at most, and carries on.

Telemetry: /pantilt (crawler.PanTilt, 10 Hz) has the angles, the person's
bearing and elevation off the truck, the boxes and the detector's rate;
/pantilt/image is the camera at 2 Hz, 320 wide, with the boxes drawn on.
"""

import threading
import time

from crawler.vision import track as tk

# Journal summary of the detector's rate and cost, this often.
SUMMARY_S = 60.0
# How often the Pi's idle fraction is sampled for the governor.
IDLE_SAMPLE_S = 1.0
# The same error is journalled at most this often.
ERROR_LOG_S = 60.0
# Boxes older than this are not drawn on /pantilt/image.
BOX_SHOW_S = 1.0


class PanTiltRunner:
    def __init__(self, cfg, mount, grabber, detector, tele=None, logger=None,
                 clock=time.monotonic, idle=None):
        self.cfg = cfg
        self.mount = mount
        self.grabber = grabber
        self.detector = detector
        self.tele = tele
        self.logger = logger
        self.clock = clock
        self.idle = idle if idle is not None else tk.CpuIdle()
        self.tracker = tk.Tracker(cfg.track, cfg.camera, mount.limits(),
                                  cfg.mount.yaw_deg, cfg.mount.pitch_deg)
        self.governor = tk.RateGovernor(cfg.detector)
        self._stop = threading.Event()
        self._thread = None
        self._seq = 0
        self._pending = {}
        self._last_frame_t = None
        self._next_submit = 0.0
        self._next_idle = 0.0
        self._next_summary = None
        self._errors = {}
        self.idle_frac = None
        self.last_infer_s = None
        self.answers = 0
        self._answers_at_summary = 0
        self._infer_sum = 0.0
        self._last_people = []
        self._last_people_t = None
        self._detector_state = None
        self._backoff_logged = 1.0
        self._last_step = None
        self._last_state = self.tracker.state

    # ---- lifecycle -------------------------------------------------------

    def start(self):
        try:
            self.mount.centre()
            (plo, phi), (tlo, thi) = self.mount.limits()
            self._log("info", f"Mount {self.mount.kind} centred; pan "
                      f"{plo:+.0f}..{phi:+.0f} deg, tilt {tlo:+.0f}..{thi:+.0f} deg")
        except Exception as e:
            self._log("warning", f"Could not centre the mount: {e}")
        self.grabber.start()
        if not self.detector.start():
            self._log("warning", f"No person detection: {self.detector.error}")
        self._thread = threading.Thread(target=self._run, name="pantilt",
                                        daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        for part in (self.detector, self.grabber, self.mount):
            try:
                part.close()
            except Exception as e:
                print(f"[pantilt] close failed: {e}")

    def _run(self):
        period = 1.0 / self.cfg.track.step_hz
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:
                self._error("tick", f"Pan/tilt pass failed: {type(e).__name__}: {e}")
            self._stop.wait(period)

    # ---- one pass --------------------------------------------------------

    def tick(self):
        now = self.clock()
        self._watch_detector()
        self._collect(now)
        self._sample_idle(now)
        self._maybe_submit(now)
        dt = 0.0 if self._last_step is None else now - self._last_step
        self._last_step = now
        pan, tilt = self.tracker.step(now, dt)
        self.mount.set(pan, tilt)
        self._journal_state(now)
        self._offer(now)

    def _watch_detector(self):
        state = self.detector.state
        if state == self._detector_state:
            return
        self._detector_state = state
        if state == "ready":
            shape = getattr(self.detector, "input_shape", None)
            self._log("info", f"Person detector up ({self.detector.model_path}"
                      + (f", input {shape[1]}x{shape[0]}" if shape else "") + ")")
        elif state == "failed":
            self._log("warning", f"Person detector down: {self.detector.error}")

    def _collect(self, now):
        msg = self.detector.poll()
        if msg is None:
            return
        if msg[0] == "failed":
            self._pending.pop(msg[1], None)
            self._error("detect", f"Detection failed: {msg[2]}")
            return
        _, seq, people, infer_s = msg
        at = self._pending.pop(seq, None)
        if at is None:
            return
        pan, tilt = at
        self.answers += 1
        self.last_infer_s = infer_s
        self._infer_sum += infer_s
        self._last_people = people
        self._last_people_t = now
        self.tracker.on_detections(people, pan, tilt, now)
        self.governor.update(infer_s=infer_s, idle_frac=None)

    def _sample_idle(self, now):
        if now < self._next_idle:
            return
        self._next_idle = now + IDLE_SAMPLE_S
        self.idle_frac = self.idle.sample()
        if self.idle_frac is None:
            return
        self.governor.update(infer_s=self.last_infer_s, idle_frac=self.idle_frac)
        b = self.governor.backoff
        if b != self._backoff_logged and (b >= 2 * self._backoff_logged
                                          or b == 1.0):
            self._log("info" if b == 1.0 else "warning",
                      f"Detection at {1.0 / self.governor.interval:.1f} Hz: "
                      f"the Pi is {100 * self.idle_frac:.0f}% idle")
            self._backoff_logged = b

    def _maybe_submit(self, now):
        if now < self._next_submit or self.detector.state != "ready" \
                or self.detector.busy:
            return
        got = self.grabber.latest()
        if got is None:
            return
        frame, t = got
        if t == self._last_frame_t:
            return
        self._seq += 1
        if self.detector.submit(self._seq, frame):
            self._last_frame_t = t
            # The angles the mount had for this frame: the tracker turns the
            # answer into an absolute target off these, however late it is.
            self._pending = {self._seq: (self.mount.pan_deg,
                                         self.mount.tilt_deg)}
            self._next_submit = now + self.governor.interval

    # ---- the journal -----------------------------------------------------

    def _journal_state(self, now):
        state = self.tracker.state
        if state != self._last_state:
            if state == "tracking":
                d = self.tracker.person
                self._log("info", f"Person at {self.tracker.bearing_deg:+.0f} deg "
                          f"(score {d.score:.2f})" if d is not None else "Person")
            elif state == "lost":
                self._log("info", f"Person lost for {self.cfg.track.lost_s:g} s; "
                          f"mount to {self.cfg.track.on_lost}")
            self._last_state = state
        if self._next_summary is None:
            self._next_summary = now + SUMMARY_S
        elif now >= self._next_summary:
            n = self.answers - self._answers_at_summary
            if n:
                mean_ms = 1000.0 * self._infer_sum / n
                idle = ("" if self.idle_frac is None
                        else f", Pi {100 * self.idle_frac:.0f}% idle")
                self._log("info", f"Detection {n / SUMMARY_S:.2f} Hz, "
                          f"{mean_ms:.0f} ms each{idle}")
            self._answers_at_summary = self.answers
            self._infer_sum = 0.0
            self._next_summary = now + SUMMARY_S

    # ---- telemetry -------------------------------------------------------

    def status(self, now):
        tr = self.tracker
        d = tr.person
        fresh = (self._last_people_t is not None
                 and now - self._last_people_t <= BOX_SHOW_S)
        return {
            "state": tr.state,
            "person": d is not None,
            "pan_deg": round(self.mount.pan_deg, 2),
            "tilt_deg": round(self.mount.tilt_deg, 2),
            "target_pan_deg": round(tr.target_pan, 2),
            "target_tilt_deg": round(tr.target_tilt, 2),
            "bearing_deg": round(tr.bearing_deg, 2) if d is not None else 0.0,
            "elevation_deg": round(tr.elevation_deg, 2) if d is not None else 0.0,
            "score": round(d.score, 3) if d is not None else 0.0,
            "box": d.as_list()[:4] if d is not None else [],
            "people": len(self._last_people) if fresh else 0,
            "boxes": [p.as_list() for p in self._last_people] if fresh else [],
            "detector": self.detector.state,
            "det_ms": round(1000.0 * self.last_infer_s, 1)
            if self.last_infer_s is not None else 0.0,
            "interval_s": round(self.governor.interval, 3),
            "cpu_idle": round(self.idle_frac, 3) if self.idle_frac is not None
            else -1.0,
            "camera_up": bool(getattr(self.grabber, "up", False)),
            "frames": int(getattr(self.grabber, "frames", 0)),
            "mount": self.mount.kind,
            "writes": int(getattr(self.mount, "writes", 0)),
            "acquired": tr.acquired,
        }

    def _offer(self, now):
        tele = self.tele
        if tele is None:
            return
        if tele.wants("pantilt", now):
            tele.offer("pantilt", self.status(now), now)
        if tele.wants("pantilt_image", now):
            got = self.grabber.latest()
            if got is not None:
                fresh = (self._last_people_t is not None
                         and now - self._last_people_t <= BOX_SHOW_S)
                tele.offer("pantilt_image",
                           annotate(got[0], self._last_people if fresh else [],
                                    self.tracker.person), now)

    # ---- odds and ends ---------------------------------------------------

    def _error(self, key, message):
        now = self.clock()
        if now < self._errors.get(key, 0.0):
            return
        self._errors[key] = now + ERROR_LOG_S
        self._log("warning", message)

    def _log(self, level, message):
        if self.logger is not None:
            try:
                self.logger.log(level, "PanTilt", message)
                return
            except Exception:
                pass
        print(f"[pantilt] {message}")


def annotate(frame, people, followed=None, max_dim=320):
    """A copy of frame, no wider than max_dim, with each person's box drawn:
    the followed one green, the rest yellow."""
    import cv2
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / float(max(h, w))
        frame = cv2.resize(frame, (int(w * s), int(h * s)),
                           interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
    else:
        frame = frame.copy()
    for p in people:
        colour = (0, 255, 0) if p is followed else (0, 220, 255)
        cv2.rectangle(frame, (int(p.x0 * w), int(p.y0 * h)),
                      (int(p.x1 * w), int(p.y1 * h)), colour, 2)
    return frame


def build(cfg, pca=None, tele=None, logger=None):
    """The runner the config asks for, not started."""
    from crawler.hardware import pantilt as pt
    from crawler.vision import person
    mount = pt.build_mount(cfg, pca=pca)
    grabber = pt.FrameGrabber(cfg.camera, logger=logger)
    d = cfg.detector
    detector = person.DetectorClient(d.model, conf=d.conf, iou=d.iou,
                                     threads=d.threads, nice=d.nice,
                                     worker=d.worker)
    return PanTiltRunner(cfg, mount, grabber, detector, tele=tele, logger=logger)
