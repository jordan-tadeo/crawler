"""/footprint: what the scan saw on each decision, and what the car was told.

What this is for
----------------
Training data. The next steering policy is a small CNN over a downsampled
depth image, and the person whose steering and throttle it should learn is
me. So on every decision in record mode -- hold Y in manual -- the car writes
three things at one log time: the 80x50 depth picture the policy will see
(policy_input.depth_small, on /policy/depth_small), the command my hands put
on the wheels, and the IMU. In auto it writes the same three for the
autopilot's own decisions, off the frame the driver scanned. The nine-probe
footprint scan and the gradient rules' verdict ride along in the same row:
they are what the hand-written driver would have done, which is the baseline
a learned one has to beat. The schema is in docs/TRAINING_DATA.md.

Nothing in here is a second copy of the rules. The scan is
driver.free_by_column() and the verdict is driver.read_the_gradient(), called,
so whatever the rules become, the rows say what the rules said.

Nothing in here can drive the car
---------------------------------
pigpio latches the last pulse width forever and there is no hardware failsafe.
A recording-side computation that could reach set_steering() or set_throttle()
in record mode would be the autopilot driving under my hands without either of
us knowing which one had the wheels. So this file is never handed the vehicle
controller at all. The command it writes is a tuple of three floats the
control loop read off the controller after commanding it, and the driver it
reads in auto is read and never called -- last_scan and last_verdict are
attributes. A test holds both properties.

The rate
--------
10 Hz, which is AUTO_TICK_INTERVAL: the autopilot's decision rate, not the
camera's 16 fps and not the control loop's 100 Hz. A row is a decision, and
a policy trained at one rate and run at another is learning the wrong thing.
Record rows are on the same fixed phase the decisions are
(driver.next_decision_at): a late pass makes a late row, not a later phase,
so a recording is 10 rows a second and not the 8.8-9.3 the old
next = now + 0.1 drifted to. In auto the AutoPilot hands over one call per
decision (decision=True), from the thread that made it.

Threads
-------
Record rows come from the control loop, auto rows from the decision thread
(main.AutoPilot), and the IMU is polled off the control loop every pass. A
lock round each, so a mode change that has both threads in here at once
cannot interleave two rows' bookkeeping. Neither is ever held across
anything that waits.

The cost
--------
In record the scan runs here, on the control loop, at 10 Hz -- the same scan,
on the same thread, at the same rate auto has always run it. The JSON is built
here and encoded on the publisher thread like every other channel. In auto the
scan is the driver's and this adds a read_the_gradient() call, which is a few
dozen comparisons. In record the policy picture is shrunk here, on the control
loop, once a tick: 2.8 ms p50 and 5.0 ms p99 on the Pi 4, measured. In auto
under a policy that drives on depth_small, the policy has already shrunk the
very frame the row is off, and the row takes its picture (the same array,
by identity of the frame) instead of shrinking it a second time. The PNG
encode is on the publisher thread.
"""

import math
import threading
import time

from crawler.autopilot import driver as dv
from crawler.autopilot import patrol as patrol_mod
from crawler.autopilot import policy_imu
from crawler.autopilot import policy_input

# The decision rate. main.AUTO_TICK_INTERVAL is the same number; it is not
# imported from there because main imports this file.
INTERVAL = 0.1

# Every Nth record row also carries the raw depth frame it was computed from,
# lossless, on /camera/depth_raw at the row's exact log time. 5 is 2 Hz:
# enough to re-run a changed scan over a drive and check it against the rows,
# without the frames being most of the card. See the README for the measured
# size.
RAW_DEPTH_EVERY = 5

RECORD = "record"
AUTO = "auto"

# len(policy.POLICY_STAGES). Written here, not imported: policy pulls in
# the goal, lidar and map code, and this file is loaded by every recording,
# policy or not. A test holds the two equal.
STAGE_COUNT = 10

# Who had each group on a row's tick: "pad" is my hand, "driver" the
# autopilot. main.DRIVER and main.PAD are the same strings.
DRIVER = "driver"
PAD = "pad"

# Who was driving auto's "driver" groups, and what a learned one asked for.
# Every row carries all of these, empty when they do not apply, so the
# columns are the same in every row: autopilot is "" in record, "rules" for
# the hand-written driver and "policy" for python -m crawler --policy. The policy_*
# fields are the model's own command after its floor veto -- what it would
# have put on the wheels, which the row's throttle / steering_* are not
# while my hand has a group.
AUTOPILOT_FIELDS = {
    "autopilot": "",
    "policy_model": "",          # the model's name, e.g. "kblock 1.1"
    "policy_model_sha256": "",   # sha256 of the .onnx
    "policy_action": "",         # forward / reverse / stop
    "policy_throttle": 0.0,
    "policy_steering_front": 0.0,
    "policy_reason": "",
    "goal_reached": False,       # goal.Goal said the car has arrived
    "policy_stop_masked": False, # the model said stop and was driven on
    "policy_confidence": 0.0,    # softmax p of the committed action; -1 with none
    "policy_confidence_smoothed": 0.0,  # its EMA over CONF_EMA_DECISIONS
    "policy_speed_scale": 0.0,   # what the throttle magnitude was multiplied by
    "policy_infer_ms": 0.0,
    "policy_steer_mode": "",     # unified / split: the mode the model drove with
    "policy_steer_mode_p": 0.0,  # the mode head's p of that mode; -1 with no head
    "reverse_commit": False,     # a fresh model reverse was being held this tick
    "reverse_refractory": False, # a fresh model reverse, too soon after the last, driven forward
    "standoff_ticks": 0,         # decisions in a row the floor held a model forward
    # The direction head's [left, straight, right]; all -1 for a regression head.
    "policy_steer_dir_p": [0.0, 0.0, 0.0],
    # A lidar model was fed a real roof lidar scan this decision (policy_lidar).
    "policy_lidar_present": False,
    # A map model was fed a trusted, fresh egomap crop this decision.
    "policy_map_present": False,
    # A map model had handed the wheels to its map-free fallback (the map lost).
    "policy_fallback": False,
    # The tilt veto (driver.TiltVeto): the direction it blocked ("" none),
    # the tilt and side gap it read (-1 none), and whether the model's answer
    # was the blocked direction and its next best drove instead.
    "tilt_veto": "",
    "tilt_veto_deg": -1.0,
    "tilt_veto_gap_mm": -1.0,
    "policy_tilt_masked": False,
    # A goal model was fed a goal this decision, and what it was the encoding
    # of: "goal" the goal itself, "subgoal" a point along the path to it
    # (pathgoal.py), "none".
    "policy_goal_present": False,
    "policy_goal_fed": "none",
    # The back-out (driver.Driver.begin_recover), for either autopilot.
    "backout_phase": "",         # reverse / settle / handback, "" with none
    "backout_reason": "",        # stuck / standoff / obstacle: what started it
    "backout_exit": "",          # opened / bumper / cap / no frame / abort
    "backout_elapsed_s": 0.0,    # since the current one began, 0 with none
    "backout_reverse_s": 0.0,    # how long its reverse lasted, once it ended
    "backout_side": 0.0,         # the side it reversed on, -1 left, +1 right
    "backout_speed": 0.0,        # what it put on the throttle this decision
    "backout_level": 0,          # escalation: 0, then 1, 2... at the same spot
    # The rear stop off the roof lidar (driver.py, "The rear stop").
    "rear_clearance_mm": -1.0,   # rear bumper to the nearest return behind; -1 none / no fresh scan
    "rear_stop": False,          # the rear stop ended or refused a reverse this decision
    # --lidar-near (driver.py, "the near field"): the front arc's path clearance.
    "front_lidar_clearance_mm": -1.0,  # front bumper to the nearest return in the path; -1 off / none
    # With --goal: where the patrol is this decision (patrol.ROW_FIELDS), so
    # tools/lap_report.py can put every row in its lap and segment.
    **patrol_mod.ROW_FIELDS,
    # What the policy's tick cost (policy.TickTimer): wall and thread-CPU ms
    # per policy.POLICY_STAGES, and the whole tick's. Then how old each input
    # was at the tick's `now`: the depth frame by the OAK-D's capture stamp
    # and by when the reader handed it over, the roof lidar scan and the
    # egomap crop. All in NULLABLE_FIELDS: None, not 0, when there is none,
    # so a missing number never reads as a fast one.
    "policy_stage_ms": [None] * STAGE_COUNT,
    "policy_stage_cpu_ms": [None] * STAGE_COUNT,
    "tick_wall_ms": None,
    "tick_cpu_ms": None,
    "depth_age_ms": None,
    "depth_slot_ms": None,
    "lidar_age_ms": None,
    "crop_age_ms": None,
}

# The fields above whose "none" is None (JSON null) and not a number.
NULLABLE_FIELDS = frozenset(key for key, value in AUTOPILOT_FIELDS.items()
                            if value is None or (isinstance(value, list)
                                                 and value and value[0] is None))


def _driver_fields(driver, now):
    """The driver's own fields for the row: the back-out's, then the
    policy's when it is one. Each read off a method the driver has, never
    a call that could command anything."""
    out = {}
    backout = getattr(driver, "backout_fields", None)
    if backout is not None:
        out.update(backout(now))
    rear = getattr(driver, "rear_fields", None)
    if rear is not None:
        out.update(rear())
    policy = getattr(driver, "policy_fields", None)
    if policy is not None:
        out.update(policy())
    patrol = getattr(getattr(driver, "goal", None), "row_fields", None)
    if patrol is not None:
        out.update(patrol())
    return out


def autopilot_fields(driver, now=None):
    """AUTOPILOT_FIELDS for a row, off the auto driver (None in record)."""
    out = dict(AUTOPILOT_FIELDS)
    if driver is None:
        return out
    out["autopilot"] = "rules"
    try:
        if getattr(driver, "policy_fields", None) is not None:
            out["autopilot"] = "policy"
        for key, value in _driver_fields(driver, now).items():
            if key in NULLABLE_FIELDS:
                out[key] = _nullable(value, AUTOPILOT_FIELDS[key])
            elif key in out:
                if isinstance(out[key], bool):
                    out[key] = bool(value)
                elif isinstance(out[key], str):
                    out[key] = str(value)
                elif isinstance(out[key], int):
                    out[key] = int(_num(value, 0.0))
                elif isinstance(out[key], list):
                    # A fixed-width list of numbers: always that many, -1
                    # for any that is not one.
                    width = len(AUTOPILOT_FIELDS[key])
                    value = list(value)
                    if len(value) != width:
                        value = [-1.0] * width
                    out[key] = [_num(v, -1.0) for v in value]
                else:
                    out[key] = _num(value, 0.0)
    except Exception:
        out["autopilot"] = ("policy" if getattr(driver, "policy_fields", None)
                            is not None else "rules")
    return out


# Every IMU column a row carries, age included. Named in policy_imu.
IMU_FIELDS = policy_imu.ROW_FIELDS


def _num(value, fallback=-1.0) -> float:
    try:
        value = float(value)
    except Exception:
        return fallback
    return value if math.isfinite(value) else fallback


def _nullable(value, default):
    """A NULLABLE_FIELDS value: a finite number or None, or a fixed-width
    list of them (None for any that is not one, all None at a wrong width)."""
    if isinstance(default, list):
        try:
            value = list(value)
        except Exception:
            return list(default)
        if len(value) != len(default):
            return list(default)
        return [_num(v, None) for v in value]
    return _num(value, None)


def _sweep_clears(scan, side) -> bool:
    """driver.turn_sweep_clears(), and False if it cannot say."""
    try:
        return bool(dv.turn_sweep_clears(scan, side))
    except Exception:
        return False


def feature_row(scan, verdict, command, mode, source, shape, scan_ms=0.0,
                committed=None, pad_throttle=None, limits=None,
                steering_mode="", steer_by=PAD, throttle_by=PAD,
                autopilot=None):
    """One /footprint message. Pure: values in, a dict out.

    `autopilot` is autopilot_fields(driver); None is a record row's."""
    throttle, front, rear = command
    forward, reverse = limits if limits is not None else (-1.0, -1.0)
    return {
        **(autopilot if autopilot is not None else AUTOPILOT_FIELDS),
        "mode": mode,
        "source": source,
        "frame_width": int(shape[1]) if shape is not None else 0,
        "frame_height": int(shape[0]) if shape is not None else 0,
        "probe_distance_mm": [_num(d) for d in scan.distance_mm],
        "probe_valid_fraction": [_num(v, 0.0) for v in scan.valid_fraction],
        "probe_blocked": [bool(b) for b in scan.blocked],
        "probe_reason": [str(r) for r in scan.reason],
        "probe_marks": dv.probe_marks(scan),
        # Could a full-lock turn that way begin now without touching anything
        # the fan sees? The driver's own sweep test, asked both ways: side -1
        # is left and +1 right, as read_the_gradient passes it.
        "sweep_clears_left": _sweep_clears(scan, -1.0),
        "sweep_clears_right": _sweep_clears(scan, 1.0),
        "scan_ms": _num(scan_ms, 0.0),
        "verdict_action": str(verdict.action),
        "verdict_steer": _num(verdict.steer, 0.0),
        "verdict_reason": str(verdict.reason),
        "committed_action": str(committed.action) if committed else "",
        "committed_steer": _num(committed.steer, 0.0) if committed else 0.0,
        "committed_reason": str(committed.reason) if committed else "",
        "throttle": _num(throttle, 0.0),
        "steering_front": _num(front, 0.0),
        "steering_rear": _num(rear, 0.0),
        "steering_mode": str(steering_mode),
        # Who had each group when `command` went out. A "pad" group is a
        # label; a "driver" group is the autopilot, not a demonstration.
        "steer_by": str(steer_by),
        "throttle_by": str(throttle_by),
        "pad_throttle": _num(pad_throttle, 0.0),
        "forward_limit": _num(forward),
        "reverse_limit": _num(reverse),
        "depth_raw": False,
        "depth_small": False,
    }


def policy_small(driver, depth):
    """The policy's own depth_small() of `depth`, if it made one, else None.

    Only when it is the very frame (`is`, not equal) and the policy shrank it
    with policy_input.depth_small itself -- a legacy-contract model's input is
    a different picture, and is never handed back as this one."""
    if depth is None or driver is None:
        return None
    try:
        src, small, fn = getattr(driver, "last_small", None) or (None, None, None)
    except Exception:
        return None
    if src is depth and fn is policy_input.depth_small and small is not None:
        return small
    return None


def imu_fields(sample, sampled_at, now):
    """The IMU part of a row: a camera.get_imu_sample() dict, flattened.

    Zeros and an age of -1 when there has been no sample, so every row has
    the same columns whether or not the IMU has spoken yet. The flattening
    is policy_imu.row_fields(), because a learned driver builds its IMU input
    off exactly these columns and the car rebuilds them live with the same
    call.
    """
    return policy_imu.row_fields(sample, sampled_at, now)


class FootprintTap:
    """The recording side of a decision. Reads; never commands.

    `view` is a camera.StreamView of its own ("footprint"), so taking a frame
    here costs the driver and the telemetry nothing. `tele` is the Telemetry.
    Neither of them can move a wheel, and there is no third thing in here.
    """

    def __init__(self, view=None, tele=None, interval=INTERVAL,
                 raw_every=RAW_DEPTH_EVERY, auto_view=None):
        self.view = view
        self.tele = tele
        # The driver's own camera view, read for its last_depth: the frame
        # the driver scanned on the tick that made its last_scan. Read only;
        # a take here would steal the driver's next frame.
        self.auto_view = auto_view
        self.interval = interval
        self.raw_every = raw_every

        self.seq = 0
        self.rows = 0
        self._next_at = 0.0
        self._record_rows = 0
        self._last_driver_scan = None
        self._failed = False
        self._imu = None
        self._imu_at = 0.0
        # Pictures taken off the policy instead of shrunk here (1e).
        self.reused_small = 0
        self._lock = threading.RLock()
        self._imu_lock = threading.Lock()

    def poll_imu(self, now):
        """Keep the newest IMU sample. The control loop's call on a pass that
        makes no row of its own (auto, where the rows come with the
        decisions). Never raises."""
        try:
            self._poll_imu(now)
        except Exception:
            pass

    def after_drive(self, now, mode, command, driver=None, pad_throttle=None,
                    limits=None, steering_mode="", steer_by=None,
                    throttle_by=None, takeover=False, decision=False):
        """Called by the control loop once a pass, after the drive call.

        `command` is (throttle, front, rear) as the controller holds it after
        this pass's command -- in auto, the mix of the autopilot and whatever
        my hand has taken over, because that is what reached the hardware.
        `steer_by` and `throttle_by` say whose each group was; left out they
        are the pad in record and the driver in auto. `takeover` is True while
        either group is mine in auto. `decision` True says this call is one
        decision's (the AutoPilot's, once per tick), so an auto row needs no
        rate gate of its own. Returns the row offered, or None. Never
        raises: a recording is never worth a tick, and this runs after the
        wheels have already been told what to do.
        """
        try:
            self._poll_imu(now)
            with self._lock:
                if mode == AUTO:
                    return self._auto(now, command, driver, steering_mode,
                                      steer_by or DRIVER, throttle_by or DRIVER,
                                      takeover, pad_throttle, limits,
                                      gated=not decision)
                if mode == RECORD:
                    return self._record(now, command, pad_throttle, limits,
                                        steering_mode, steer_by or PAD,
                                        throttle_by or PAD)
                return None
        except Exception as e:
            if not self._failed:
                self._failed = True
                print(f"[footprint] dropped a row and stopped complaining: {e}")
            return None

    def _poll_imu(self, now):
        """Keep the newest IMU sample. One attribute read a pass."""
        if self.view is None:
            return
        get = getattr(self.view, "get_imu_sample", None)
        with self._imu_lock:
            sample = get() if get is not None else None
            if sample is not None:
                self._imu, self._imu_at = sample, now

    # ---- auto: the driver's own decision ---------------------------------

    def _auto(self, now, command, driver, steering_mode, steer_by,
              throttle_by, takeover, pad_throttle, limits, gated=True):
        """A row on every pass where the driver made a new scan.

        The driver's tick is already gated at 10 Hz, so this is too. A back-out
        scans every decision too (that is how it knows the way opened), so its
        rows are the driver's, with the backout_* fields set. A pass in a
        recover with no new scan of the driver's and my hand on a group still
        makes a row: what my hand does there is a correction, the most useful
        label there is, so the row comes off our own view at the decision rate,
        exactly as in record, marked source "tap". Called once a decision
        (`gated` False) that is every decision's; called every pass, the
        rate gate makes it one a decision.
        """
        if driver is None or self.tele is None:
            return None
        scan = getattr(driver, "last_scan", None)
        if scan is None or scan is self._last_driver_scan:
            if takeover and getattr(driver, "_phase", None) is not None:
                return self._own_row(now, command, AUTO, pad_throttle, limits,
                                     steering_mode, steer_by, throttle_by,
                                     driver, gated=gated)
            return None
        self._last_driver_scan = scan
        # One rate for both kinds of row, so the tap's first row in a recover
        # comes a decision after the driver's last one and not a pass after.
        self._next_at = dv.next_decision_at(self._next_at, now, self.interval)
        if not self.tele.wants("footprint", now):
            return None

        depth = getattr(self.auto_view, "last_depth", None)
        row = feature_row(
            scan, dv.read_the_gradient(scan), command, AUTO, "driver",
            depth.shape if depth is not None else None,
            getattr(driver, "last_scan_ms", 0.0),
            committed=getattr(driver, "last_verdict", None),
            steering_mode=steering_mode, steer_by=steer_by,
            throttle_by=throttle_by, pad_throttle=pad_throttle,
            limits=limits, autopilot=autopilot_fields(driver, now))
        return self._offer(row, now, depth, small=policy_small(driver, depth))

    # ---- record: our own scan, my hands' command -------------------------

    def _record(self, now, command, pad_throttle, limits, steering_mode,
                steer_by, throttle_by):
        return self._own_row(now, command, RECORD, pad_throttle, limits,
                             steering_mode, steer_by, throttle_by)

    def _own_row(self, now, command, mode, pad_throttle, limits,
                 steering_mode, steer_by, throttle_by, driver=None,
                 gated=True):
        """A row off our own view and our own scan, at the decision rate.
        The raw frame rides along on every raw_every-th one in record only."""
        if self.view is None or self.tele is None:
            return None
        if gated and now < self._next_at:
            return None
        depth = self.view.get_depth_frame()
        if depth is None:
            # Camera still coming up, or no frame since the last row. The
            # gate does not advance, so the next pass with a frame is a row.
            return None
        # On the fixed phase: the next row is due one interval after this
        # one was due, not after this pass happened to run.
        self._next_at = dv.next_decision_at(self._next_at, now, self.interval)

        if not self.tele.wants("footprint", now):
            return None

        started = time.perf_counter()
        scan = dv.free_by_column(depth)
        scan_ms = (time.perf_counter() - started) * 1000.0
        verdict = dv.read_the_gradient(scan)

        row = feature_row(scan, verdict, command, mode, "tap", depth.shape,
                          scan_ms, pad_throttle=pad_throttle, limits=limits,
                          steering_mode=steering_mode, steer_by=steer_by,
                          throttle_by=throttle_by,
                          autopilot=autopilot_fields(driver, now))
        if mode != RECORD:
            return self._offer(row, now, depth)

        # The frame first, at the same `now`, so its log time is the row's.
        if (self._record_rows % self.raw_every == 0
                and self.tele.wants("depth_raw", now)):
            self.tele.offer("depth_raw", depth, now)
            row["depth_raw"] = True
        self._record_rows += 1
        return self._offer(row, now, depth)

    def _offer(self, row, now, depth=None, small=None):
        """The row, its IMU snapshot and its policy picture, all at `now`.

        The picture goes first and carries frame_id "footprint/<seq>", so a
        reader can pair it to its row by log time or by seq, and the two can
        never disagree. `small` is that picture already made, off `depth`, by
        the policy (policy_small); None shrinks it here.
        """
        self.seq += 1
        row["seq"] = self.seq
        with self._imu_lock:
            imu, imu_at = self._imu, self._imu_at
        row.update(imu_fields(imu, imu_at, now))
        if depth is not None and self.tele.wants("depth_small", now):
            try:
                if small is not None:
                    self.reused_small += 1
                else:
                    small = policy_input.depth_small(depth)
            except ValueError as e:
                # A frame the policy could not take. The row still goes,
                # saying it has no picture.
                small = None
                if not self._failed:
                    self._failed = True
                    print(f"[footprint] no policy picture: {e}")
            if small is not None:
                self.tele.offer("depth_small",
                                (small, f"footprint/{self.seq}"), now)
                row["depth_small"] = True
        self.tele.offer("footprint", row, now)
        self.rows += 1
        return row
