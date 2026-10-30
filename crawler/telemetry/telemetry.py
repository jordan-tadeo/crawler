"""Live telemetry out to Foxglove, on a thread that cannot stall the car.

What this is for: watching the controller, the vehicle state, the OAK-D IMU
and the depth picture from the laptop while the car is being driven, instead
of reading app.log afterwards. Foxglove Studio connects straight to the Pi --
no broker, no rosbridge, no cloud:

    ws://jtpi.local:8765

By name rather than by address, because the address moved. The lease the car
used to hold is on some other device now, and that device answers a ping and
refuses SSH -- which is the most confusing possible way for this to break,
because everything looks reachable and nothing works.

The one rule this file exists to obey
-------------------------------------
pigpio latches a servo pulse width and emits it forever. The control loop is
the only thing that can take that pulse width back to neutral, so anything
that can delay the loop can run the car into a wall with a dead process at
the wheel. A WebSocket client is the worst possible thing to put near that
loop: it is a socket to a laptop on wifi, it can be slow, it can be a
half-open TCP connection that takes two minutes to notice, and the laptop can
be closed mid-frame.

So the control thread never touches a socket. It calls wants() -- which is a
dict lookup and two float compares -- and offer(), which drops a value into a
one-deep mailbox and returns. Everything after that (JSON encoding, JPEG
encoding, colormapping, the actual write) happens on the publisher thread in
here. If that thread falls behind, the mailboxes overwrite and old samples
are dropped on the floor, which is the correct answer for live telemetry: a
stale frame is worth nothing and a late throttle command is worth less.

Costs nothing when nobody is watching
-------------------------------------
Every channel is gated on Channel.has_sinks() -- note that it is a method on
this SDK and not the property the type stubs make it look like. With no
Foxglove client connected, wants() is False for everything, so the control
loop builds no dicts and the publisher thread encodes no JPEGs. This is a
4-core Pi that is also running a 100 Hz control loop; telemetry is not
allowed to cost anything when it is not being read.

And a recording counts as watching
----------------------------------
recording.py opens an MCAP writer, and an MCAP writer in this SDK is a sink
exactly like a connected client. So "nobody is watching" now means no laptop
AND no recording, and with recording on by default the messages get built
whether or not Foxglove is open -- which is the entire point, because the run
I want is the one I was not watching.

That means the server is no longer what decides whether any of this runs.
start() creates the channels first and starts the socket second, and a socket
that cannot come up -- a port somebody else is already on -- costs the live
view and leaves the recording alone. Everything the control thread touches is
keyed off the channels, not the server.

Failure is always non-fatal
---------------------------
A missing foxglove-sdk, a port already bound by a previous process, a client
that vanished mid-write: all of it is caught in here and logged. start()
returns False and the car drives exactly as it did before. There is no path
through this file that reaches the hardware -- telemetry only ever reads
values the control loop has already computed, and it has no reference to the
VehicleController at all.
"""

import json
import os
import queue
import threading
import time

from crawler.autopilot import policy_imu
from crawler.telemetry import lidar_view

# The Foxglove default, which is what Studio offers in the connection dialog,
# so there is nothing to type but the host.
DEFAULT_PORT = 8765

# 0.0.0.0, not the SDK's default of 127.0.0.1. The whole point is a laptop on
# the other side of the wifi, and the loopback default fails silently -- the
# server comes up, the port is open on the Pi, and the Mac cannot reach it.
DEFAULT_HOST = "0.0.0.0"

# How often the publisher thread drains the mailboxes. 100 Hz, the same rate
# the control loop offers at, so a sample waits at most one tick.
DRAIN_INTERVAL = 0.01

# How often the publisher thread re-reads has_sinks() for every channel into
# the snapshot the control thread looks at. 10 Hz is plenty -- it only decides
# whether a channel is being watched, and being one tenth of a second late to
# notice is invisible.
SINK_POLL_INTERVAL = 0.1

# Per-channel rate limits, in seconds between messages. These are transport
# rates and nothing else: the control loop runs at 100 Hz and the OAK-D IMU
# at 100 Hz, and neither a plot panel nor a wifi link wants all of it.
#
# The three image rates are the ones that cost real CPU. Each frame is a
# colormap plus a JPEG encode of a downscaled frame on the publisher thread,
# so these are the knob to turn if the Pi gets warm.
INTERVALS = {
    "controller": 0.05,   # 20 Hz -- fast enough to watch a trigger move
    "vehicle": 0.05,      # 20 Hz -- commanded values only, see below
    "diagnostics": 0.25,  # 4 Hz, and it only changes on an event anyway
    # Not a rate at all. /marks is an event channel -- the control loop offers
    # a mark the tick the button goes down and never calls wants() on it -- so
    # a gate here could only ever lose me a press. Zero so that if something
    # does start asking wants("marks"), the answer is always yes.
    "marks": 0.0,
    "imu": 0.05,          # 20 Hz, downsampled from the sensor's 100 Hz
    "depth": 0.125,       # 8 Hz
    "disparity": 0.125,   # 8 Hz
    "rectified_left": 0.5,  # 2 Hz -- this is for checking aim, not for driving
    # Both gated by footprint.FootprintTap, which offers at the decision rate
    # and every RAW_DEPTH_EVERY-th decision. A second gate in here could only
    # ever drop a row or split a frame from the row it belongs to.
    "footprint": 0.0,
    "depth_raw": 0.0,
    "depth_small": 0.0,
    # Offered once per revolution by lidar.LidarReader, so the sensor sets
    # the rate: 10 Hz, 13 at the most it can be driven to. One row is about
    # 600 bytes of JSON.
    "lidar": 0.0,
    # The same revolution for the 3D panel, see lidar_view.py, and the
    # transform it is drawn through, once per scan each. The truck outline
    # never changes, so once a second: enough for a client that joins late.
    "lidar_scan": 0.0,
    "tf": 0.0,
    "lidar_truck": 1.0,
    # With --slam. The status and the map -> base_link half of /tf come once
    # per matched scan; the path every two seconds. The map picture is
    # spaced by the runner (runner.map_interval: twice a second, slower when
    # it is big), so this only stops a burst. At 1.0, the runner's own rate
    # then, it ate one picture in fifteen to the scan clock's jitter and the
    # map stood still two seconds (p1201, 2026-10-29).
    "slam_status": 0.0,
    "map": 0.25,
    "slam_path": 2.0,
    # With --voxel-map, every two seconds each. The mapper offers once a
    # second, but these are the two biggest messages there are and they
    # change slowly.
    "voxel_points": 2.0,
    "voxel_low": 2.0,
    # With --nav (nav/pointnav.py): the status twice a second, the path the
    # navigator is following once a second.
    "nav_status": 0.5,
    "nav_path": 1.0,
    # With --goal (autopilot/goal.ClickGoal): the clicked goal, where it is
    # from the truck and whether it has arrived, twice a second, and its
    # marker on the map once a second.
    "goal_status": 0.5,
    "goal_marker": 1.0,
    # With --goal (autopilot/patrol.Patrol): the patrol, twice a second, and
    # its waypoints on the map once a second.
    "patrol_status": 0.5,
    "patrol_marker": 1.0,
    # With --pantilt (vision/runner.py): the mount and the person at 10 Hz,
    # and the hood camera's picture at 2 Hz. The picture is a 320x240 JPEG,
    # about 15 KB, so 30 KiB/s.
    "pantilt": 0.1,
    "pantilt_image": 0.5,
}

# What the map channels (/map, /slam/path, /voxel/points, /voxel/low) may
# cost together at their worst, in bytes a second: each at the biggest
# message map_view.MAX_CELLS, voxel_view.LIVE_POINTS_MAX and the runner's
# PATH_MAX allow, at its rate above. 304 KiB/s comes out of it. On top go
# the cameras and the lidar, about 230 KiB/s, so everything open is under
# 600 KiB/s at its worst -- down from 1.1 MiB/s and climbing, measured on
# the 10-29 drive. Next to the router the car carried 750 KiB/s without
# falling behind, but the house is a mesh, the car roams between nodes as
# it drives, and two rooms away is not next to the router.
MAP_CHANNELS_BUDGET = 384 * 1024

# Messages the WebSocket server holds for one client before it drops the
# oldest. The SDK's default is 1024, which is what made a slow link fall
# behind instead of losing frames: all of it is queued and sent in order, so
# a laptop held to 300 KiB/s against the car was 24 s behind within half a
# minute and stayed there. Live, everything here is worth only its newest
# value, so the oldest is the one to lose, and 64 is about half a second of
# every channel at once. The recording is a sink of its own and loses
# nothing to this.
LIVE_BACKLOG = 64

# Longest side of an image channel after downscaling, in pixels. The mono
# cameras are 640x400, so this is a 2x reduction in each axis and a 4x
# reduction in pixels to encode.
IMAGE_MAX_DIM = 320

# JPEG quality for the image channels. 60 is visibly lossy on a colormap and
# nobody cares; it is a depth picture, not a photograph.
JPEG_QUALITY = 60


# Timestamps. One monotonic source for every channel, so a plot of the
# throttle and a plot of the IMU line up, made into epoch time by adding the
# offset between time.time() and time.monotonic(). Foxglove wants nanoseconds
# since the unix epoch or it draws everything in 1970.
#
# That offset used to be taken once, at import, and never again. The Pi has
# no RTC: the service starts on the clock fake-hwclock saved at the last
# shutdown, and NTP steps it forward a little later. The 10-29 boot imported
# before the step and stamped every message for the whole run 90.1 s behind
# the wall (Goal set at 15:10:26.1 in the journal, /goal/status stamped
# 15:08:56.0 in the MCAP); an earlier one was 58.5 s. Foxglove's live
# connection runs its clock off the viewer's wall time, not off these stamps,
# so every entity, transform and plot was a minute and a half in its past.
#
# So the offset is checked on every stamp and taken again when it has moved
# by more than CLOCK_STEP_S. A step is a step, and the series follows it once;
# what NTP slews by in between is microseconds and is ignored, so the stamps
# never wobble with it.
CLOCK_STEP_S = 1.0
_EPOCH_OFFSET = time.time() - time.monotonic()


def epoch_offset() -> float:
    """time.time() - time.monotonic(), as of the last wall-clock step."""
    global _EPOCH_OFFSET
    now = time.time() - time.monotonic()
    if abs(now - _EPOCH_OFFSET) > CLOCK_STEP_S:
        _EPOCH_OFFSET = now
    return _EPOCH_OFFSET


def epoch_ns(mono: float) -> int:
    """A time.monotonic() reading as nanoseconds since the unix epoch."""
    return int((mono + epoch_offset()) * 1e9)


def _schema(name: str, properties: dict) -> dict:
    """A JSON Schema object with a stable name.

    Passing a bare dict to Channel() works, but the SDK then invents a schema
    name like `schema-l6Hj5PBy`, which is what Studio shows in the topic list.
    Naming them means the panel says `crawler.Controller`.
    """
    return {
        "name": name,
        "schema": {
            "type": "object",
            "title": name,
            "properties": properties,
        },
    }


_NUM = {"type": "number"}
_INT = {"type": "integer"}
_STR = {"type": "string"}
_BOOL = {"type": "boolean"}

# The project-specific channels. There is no well-known Foxglove schema for
# any of this -- and, worth writing down because it is not what I assumed,
# there is no well-known Foxglove IMU schema either. The SDK's schema set has
# CompressedImage, RawImage, LocationFix, PointCloud, Odometry, JointState and
# forty others, and nothing for an accelerometer. So /imu is JSON as well, and
# the panel that reads it is Plot, which is the panel I wanted anyway.
JSON_SCHEMAS = {
    "controller": _schema("crawler.Controller", {
        "connected": _BOOL,
        "name": _STR,
        "num_axes": _INT,
        "num_buttons": _INT,
        # Every axis and every button by raw SDL index. This is the diagnostic
        # half of the channel: the pad renumbers its axes between hid-generic
        # over Bluetooth and xpad/xpadneo over USB, and that renumbering is
        # what once turned a resting thumbstick into half throttle. Reading
        # the indices live is how you see which map you are on.
        "axes": {"type": "array", "items": _NUM},
        "buttons": {"type": "array", "items": _INT},
        # The named view, resolved through Joystick's own maps at runtime.
        # This file does not contain a copy of those maps on purpose.
        "named_axes": {"type": "object", "additionalProperties": _NUM},
        "named_buttons": {"type": "object", "additionalProperties": _BOOL},
        # Calibrated trigger rest, which is the other half of the same story.
        "trigger_rest_rt": _NUM,
        "trigger_rest_lt": _NUM,
        # What the code actually computed from all of the above and handed to
        # the hardware. If this disagrees with the sticks, the mapping is the
        # thing that is wrong.
        "throttle": _NUM,
        "steering_front": _NUM,
        "steering_rear": _NUM,
    }),
    "vehicle": _schema("crawler.Vehicle", {
        "mode": _STR,
        # "unified" or "split": whether the left stick is driving both axles
        # counter-phase or the two sticks own one axle each. Here and not on
        # /controller because it changes what the two steering numbers below
        # mean, and a rear angle that looks wrong is the first place I will
        # check which mapping was in force. Auto always steers unified.
        "steering_mode": _STR,
        "throttle": _NUM,
        "steering_front": _NUM,
        "steering_rear": _NUM,
        # The pulse width set_throttle() computed and wrote, returned by the
        # call itself, so this costs no bus traffic and is good every tick.
        "esc_pulse_us": _INT,
        # What pigpio says is actually on GPIO18, read back off the daemon.
        # This is a real round trip and it is NOT free, so it is whatever the
        # control loop's once-a-second state poll last read, not a fresh read
        # -- reading it at 20 Hz would put I/O on the throttle's thread, which
        # is the exact mistake the state poll exists to avoid. age_s says how
        # stale it is.
        "esc_pulse_us_readback": _INT,
        "steering_front_angle": _NUM,
        "steering_rear_angle": _NUM,
        "readback_age_s": _NUM,
    }),
    "diagnostics": _schema("crawler.Diagnostics", {
        "mode": _STR,
        "controller_connected": _BOOL,
        "auto_available": _BOOL,
        "camera_state": _STR,
        "camera_error": _STR,
        "last_error": _STR,
        "loop_hz": _NUM,
        "loop_tick_max_ms": _NUM,
        # Whether this run is being written to an MCAP file, and which one.
        # In the file itself this is how a replay says who wrote it; live in
        # Studio it is how I check recording is actually on without sshing
        # into the car. recording_error carries why it is not, if it is not.
        "recording": _BOOL,
        "recording_file": _STR,
        "recording_mib": _NUM,
        "recording_error": _STR,
        # How many moments I have marked in this run, by button. Live in
        # Studio this is the receipt that the button is working at all --
        # pressing A and watching the count tick is a check I can do without
        # taking my eyes somewhere useless.
        "marks_good": _INT,
        "marks_bad": _INT,
        # True while a record session (hold Y in manual) owns the file, and
        # how long the camera took to come up the last time anything asked.
        "recording_session": _BOOL,
        "camera_open_s": _NUM,
    }),
    "imu": _schema("crawler.Imu", {
        # Straight off the OAK-D's BNO086: linear acceleration in m/s^2 and
        # raw gyro in rad/s. Enabled as LINEAR_ACCELERATION, so this reads
        # near zero at a constant velocity -- which is the trap driver.py's
        # motion_evidence() warns about and not a bug in here. That is why the
        # gyro channels are worth watching in Foxglove next to the accel ones:
        # the gyro is the witness that survives a steady creep.
        "accel_x": _NUM, "accel_y": _NUM, "accel_z": _NUM,
        "gyro_x": _NUM, "gyro_y": _NUM, "gyro_z": _NUM,
        "accel_magnitude": _NUM,
        # Which way is up, as a unit vector in the IMU's own axes (x is
        # vertical on this car), off GAME_ROTATION_VECTOR -- gravity
        # referenced, so this is the tilt the accel channels cannot show.
        # All three 0 when there is no orientation reading. See policy_imu.
        "up_x": _NUM, "up_y": _NUM, "up_z": _NUM,
    }),
    # One message per press of a mark button. See marks.py for what a mark is
    # and why it is a window rather than an instant; everything in here is a
    # snapshot taken at the press, guarded field by field, so a mark made in
    # manual mode with no camera up is still a complete message -- it just has
    # nothing to say about the footprint.
    "marks": _schema("crawler.Mark", {
        "kind": _STR,            # "good" or "bad"
        "button": _STR,          # "A" or "B"
        "sequence": _INT,        # nth mark of this run, both kinds together
        "good_count": _INT,
        "bad_count": _INT,
        "utc": _STR,
        # Monotonic, and the two window bounds in the same clock. The window
        # is the substantive part: I press the button after I have seen the
        # behaviour and judged it, so the decision worth reading happened
        # before the press.
        "mono_s": _NUM,
        "window_start_s": _NUM,
        "window_end_s": _NUM,
        "lookback_s": _NUM,
        "lead_s": _NUM,
        "mode": _STR,
        "throttle": _NUM,
        "steering_front": _NUM,
        "steering_rear": _NUM,
        "esc_pulse_us": _INT,
        # What auto mode was thinking, where there was an auto mode to ask.
        # -1 in any distance means the probe had nothing measurable to say,
        # which is inf in driver.py and not a number JSON will carry.
        "block_reason": _STR,
        "near_distance_mm": _NUM,
        "probe_distance_mm": {"type": "array", "items": _NUM},
        "probe_blocked": {"type": "array", "items": _BOOL},
        "scan_blocked": _BOOL,
        "scan_ms": _NUM,
        "recovery_phase": _STR,
        "recovery_remaining_s": _NUM,
        "motion_evidence": _STR,
        "gyro_mean": _NUM,
        "accel_mean": _NUM,
        "camera_state": _STR,
    }),
    # One row per decision: what the footprint scan saw, what the gradient
    # rules would have done about it, and what the car was actually commanded
    # on that same tick. Built by footprint.py -- see there for why the rows
    # in record mode drive nothing.
    "footprint": _schema("crawler.Footprint", {
        "seq": _INT,             # per process, so a dropped row is visible
        "mode": _STR,            # manual is never here; record or auto
        "source": _STR,          # "driver": auto's own scan; "tap": ours
        "frame_width": _INT,
        "frame_height": _INT,
        # Per probe, left to right. -1 distance is inf in driver.py.
        "probe_distance_mm": {"type": "array", "items": _NUM},
        "probe_valid_fraction": {"type": "array", "items": _NUM},
        "probe_blocked": {"type": "array", "items": _BOOL},
        "probe_reason": {"type": "array", "items": _STR},
        "probe_marks": _STR,
        # driver.turn_sweep_clears(scan, -1) and (scan, +1).
        "sweep_clears_left": _BOOL,
        "sweep_clears_right": _BOOL,
        "scan_ms": _NUM,
        # read_the_gradient() on this scan, unlatched.
        "verdict_action": _STR,
        "verdict_steer": _NUM,
        "verdict_reason": _STR,
        # What auto actually did with it after the evade latch. Empty in
        # record, where nothing acts on the verdict at all.
        "committed_action": _STR,
        "committed_steer": _NUM,
        "committed_reason": _STR,
        # The command on this tick, as it reached the ESC and the servos.
        "throttle": _NUM,
        "steering_front": _NUM,
        "steering_rear": _NUM,
        "steering_mode": _STR,
        # Whose each group was on this tick: "pad" (my hand, a label) or
        # "driver" (the autopilot). Always "pad" in record.
        "steer_by": _STR,
        "throttle_by": _STR,
        # Record, and auto while the throttle is mine: the trigger before
        # scaling, and the scale.
        "pad_throttle": _NUM,
        "forward_limit": _NUM,
        "reverse_limit": _NUM,
        # A raw /camera/depth_raw frame carries this row's exact log time.
        "depth_raw": _BOOL,
        # A /policy/depth_small frame carries this row's exact log time and
        # frame_id "footprint/<seq>".
        "depth_small": _BOOL,
        # The newest OAK-D IMU sample at this tick: linear acceleration in
        # m/s^2 and gyro in rad/s, the same numbers /imu carries. imu_age_ms
        # is how old it was; -1 and zeros when there has been none.
        "imu_accel_x": _NUM, "imu_accel_y": _NUM, "imu_accel_z": _NUM,
        "imu_gyro_x": _NUM, "imu_gyro_y": _NUM, "imu_gyro_z": _NUM,
        "imu_age_ms": _NUM,
        # The same sample's up vector, as /imu's up_*; all 0 with none.
        "imu_up_x": _NUM, "imu_up_y": _NUM, "imu_up_z": _NUM,
        # Who drove auto's "driver" groups: "" in record, "rules" or
        # "policy" in auto. The policy_* fields are a learned driver's own
        # command after its floor veto, and its model's name ("kblock 1.1")
        # and sha256; empty and zero under the rules.
        "autopilot": _STR,
        "policy_model": _STR,
        "policy_model_sha256": _STR,
        "policy_action": _STR,
        "policy_throttle": _NUM,
        "policy_steering_front": _NUM,
        "policy_reason": _STR,
        # Whether goal.Goal said the car had arrived (the only thing that
        # lets a model stop be committed), and whether the model said stop
        # on this tick and was driven on instead. policy_action is still the
        # model's own choice; committed_action is what was done.
        "goal_reached": _BOOL,
        "policy_stop_masked": _BOOL,
        # How sure the model was of the action committed (softmax p, -1 on
        # a tick it committed none), that smoothed, and the speed scale off
        # it that multiplied the throttle's magnitude.
        "policy_confidence": _NUM,
        "policy_confidence_smoothed": _NUM,
        "policy_speed_scale": _NUM,
        "policy_infer_ms": _NUM,
        # The steering mode the policy drove with, and its mode head's
        # probability of that mode (-1 for a model with no head).
        "policy_steer_mode": _STR,
        "policy_steer_mode_p": _NUM,
        # Whether a reverse commit held the tick (policy.REVERSE_COMMIT_S),
        # and how many decisions in a row the floor has held a model forward
        # (policy.STANDOFF_S runs the recover).
        "reverse_commit": _BOOL,
        # A fresh model reverse inside policy.REVERSE_REFRACTORY_S of the
        # last one, driven as a forward instead.
        "reverse_refractory": _BOOL,
        "standoff_ticks": _INT,
        # A direction-head model's [left, straight, right] probabilities on
        # its last answer; all -1 for a regression head, 0 when no policy drives.
        "policy_steer_dir_p": {"type": "array", "items": _NUM, "minItems": 3, "maxItems": 3},
        # Whether a lidar model was fed a real roof lidar scan this
        # decision (policy_lidar); false for every other model and tick.
        "policy_lidar_present": _BOOL,
        # Whether a map model was fed a trusted, fresh egomap crop, and
        # whether it had handed the wheels to its map-free fallback.
        "policy_map_present": _BOOL,
        "policy_fallback": _BOOL,
        # The back-out, either autopilot (driver.Driver.backout_fields): its
        # phase (reverse / settle / handback, "" with none), what started it,
        # why its reverse ended, time since it began, how long the reverse
        # lasted, its side, this decision's throttle and its escalation level.
        "backout_phase": _STR,
        "backout_reason": _STR,
        "backout_exit": _STR,
        "backout_elapsed_s": _NUM,
        "backout_reverse_s": _NUM,
        "backout_side": _NUM,
        "backout_speed": _NUM,
        "backout_level": _INT,
        # The rear stop off the roof lidar (driver.py): rear bumper to the
        # nearest return behind, mm, -1 with none or no fresh scan; and
        # whether it ended or refused a reverse this decision.
        "rear_clearance_mm": _NUM,
        "rear_stop": _BOOL,
        # --lidar-near: front bumper to the nearest return in the path
        # ahead, mm, -1 with the switch off or no fresh scan.
        "front_lidar_clearance_mm": _NUM,
    }),
    # One revolution of the roof lidar, see crawler/hardware/lidar.py. bins
    # is lidar.BINS nearest ranges in mm from the middle of the car, bin i
    # centred on bearing i * bin_deg counter-clockwise from straight ahead
    # (0 ahead, 18 left, 36 behind, 54 right), 0 where nothing came back.
    "lidar": _schema("crawler.Lidar", {
        "bins": {"type": "array", "items": _INT},
        "bin_deg": _NUM,
        "points": _INT,
        "valid_points": _INT,
        "rpm": _NUM,
        # Over the last second. 5000 points a second is 417 packets.
        "packets_per_s": _NUM,
        "crc_error_rate": _NUM,
        # Since the reader started.
        "crc_errors": _INT,
        "rejected_packets": _INT,
        # lidar.nearest_in_sector() of bins, so a plot panel can watch them.
        "nearest_front_mm": _INT,
        "nearest_left_mm": _INT,
        "nearest_rear_mm": _INT,
        "nearest_right_mm": _INT,
        # The sensor's own millisecond clock (wraps at 30000): the packet
        # that closed the revolution, and every packet in it, in order. The
        # host time a scan arrives lags its real end by up to a 52 ms serial
        # read; these do not, so a replay can de-skew against them.
        "stamp_ms": _INT,
        "stamps_ms": {"type": "array", "items": _INT},
    }),
    # One scan through the mapper, with --slam: crawler.mapping.runner.
    # status_row(). state is tracking / tilted / rolled / lost / no_points;
    # written says whether this scan went into the map and why says why not.
    # inliers, rms_mm, returns and constraint (the smallest eigenvalue of the
    # fit's translation block, 0 in a corridor or an open field, 0.5 in a
    # room) are what to trust the pose by. -1 for a number there is none of.
    "slam_status": _schema("crawler.SlamStatus", {
        "state": _STR,
        "x": _NUM, "y": _NUM, "yaw_deg": _NUM,
        "matched": _BOOL, "written": _BOOL, "why": _STR,
        "inliers": _NUM, "rms_mm": _NUM, "constraint": _NUM,
        "returns": _INT, "points": _INT, "ground_dropped": _INT,
        "tilt_deg": _NUM,
        # |pitch| and |roll| off level, the two write gates; held is how
        # many directions the fit kept at the prediction because the scene
        # did not pin them, banded that the fit was refused for landing too
        # far from the prediction.
        "pitch_deg": _NUM, "roll_deg": _NUM, "held": _INT, "banded": _BOOL,
        "iters": _INT, "ms": _NUM,
        # "young" or "lost" when this scan started a fresh map (a lost one's
        # old map is shelved, not dropped), "merged" when it found its place
        # on the shelved one again and merged the fresh one in (slam.py), ""
        # otherwise; reseeds is how many fresh maps since boot.
        "reseed": _STR, "reseeds": _INT,
        # The runner's counts since boot: scans offered, dropped because the
        # mapper was behind, and steps that raised.
        "scans": _INT, "dropped": _INT, "errors": _INT,
    }),
    # Point navigation, with --nav: nav/pointnav.PointNavDriver.nav_status().
    # state is idle / planning / driving / arrived / closest / paused /
    # failed, reason why (a pause says the SLAM's why), driving who has the
    # wheels (nav, policy or rules). The goal and the aim (the point the
    # path ends at: the reachable one nearest a goal that cannot be reached
    # when closest) in the map frame; distance_m from the truck to the goal
    # and path_m of path left, -1 for none.
    "nav_status": _schema("crawler.NavStatus", {
        "state": _STR, "reason": _STR, "status": _STR, "driving": _STR,
        "has_goal": _BOOL, "goal_x": _NUM, "goal_y": _NUM,
        "has_aim": _BOOL, "aim_x": _NUM, "aim_y": _NUM, "closest": _BOOL,
        "distance_m": _NUM, "path_m": _NUM, "local_map": _BOOL,
        "plans": _INT, "plan_ms": _NUM, "goals": _INT, "dropped": _INT,
    }),
    # The goal a goal model drives to, with --goal: goal.ClickGoal.status().
    # The goal in the map frame and the map frame it was clicked on (frame,
    # -1 with none) against the one the truck is on now (frame_now); on_map
    # whether it can be placed off the newest crop; distance_m (-1 with none)
    # and bearing_deg (counter-clockwise, left positive) from the truck;
    # arrived, the safety layer's neutral hold within arrive_m; clicks
    # taken and refused. approx: the goal was carried onto this map across
    # a reseed, and approx_m how far off it may be (0 when exact).
    "goal_status": _schema("crawler.GoalStatus", {
        "goal_set": _BOOL, "x": _NUM, "y": _NUM, "frame": _INT, "frame_now": _INT,
        "on_map": _BOOL, "distance_m": _NUM, "bearing_deg": _NUM, "arrived": _BOOL,
        "arrive_m": _NUM, "clicks": _INT, "refused": _INT,
        "approx": _BOOL, "approx_m": _NUM,
    }),
    # The patrol, with --goal: patrol.Patrol.patrol_status(). use is "goal"
    # (a click is the goal) or "patrol"; state empty / paused / driving /
    # dwell / finished / tracing (or goal); mode once / loop / patrol;
    # index the current waypoint from 0 and number from 1 (0 with none),
    # count of them, laps done, skipped so far. x, y the current one on the
    # map the truck is on (frame_now), on_map whether it is placed there,
    # distance_m from the truck (-1 with none), approx and approx_m as for
    # the goal. restored: the list came back from before a restart;
    # pending: a saved list is waiting for the map to be checked.
    # truck_x, truck_y, truck_yaw_deg: the truck's map pose off the crop the
    # distance is from, so a recording has the waypoint and the pose in one
    # message on one map frame.
    "patrol_status": _schema("crawler.PatrolStatus", {
        "use": _STR, "state": _STR, "mode": _STR, "running": _BOOL, "tracing": _BOOL,
        "index": _INT, "number": _INT, "count": _INT, "laps": _INT, "skipped": _INT,
        "x": _NUM, "y": _NUM, "frame_now": _INT, "on_map": _BOOL, "distance_m": _NUM,
        "approx": _BOOL, "approx_m": _NUM, "restored": _BOOL, "pending": _BOOL,
        "truck_x": _NUM, "truck_y": _NUM, "truck_yaw_deg": _NUM,
    }),
    # The hood camera's pan/tilt, with --pantilt: vision/runner.status().
    # state idle / tracking / lost; pan and tilt what the mount was told,
    # degrees, pan left positive, tilt up; the target it is slewing to.
    # bearing_deg (left positive, off the truck's straight ahead) and
    # elevation_deg of the followed person, 0 with nobody; box that person
    # and boxes everyone in the last detection, each [x0, y0, x1, y1(,
    # score)] in fractions of the frame. det_ms the last inference,
    # interval_s the governor's wait between detections, cpu_idle the Pi's
    # idle fraction (-1 unknown).
    "pantilt": _schema("crawler.PanTilt", {
        "state": _STR, "person": _BOOL,
        "pan_deg": _NUM, "tilt_deg": _NUM,
        "target_pan_deg": _NUM, "target_tilt_deg": _NUM,
        "bearing_deg": _NUM, "elevation_deg": _NUM, "score": _NUM,
        "box": {"type": "array", "items": _NUM},
        "people": _INT,
        "boxes": {"type": "array", "items": {"type": "array", "items": _NUM}},
        "detector": _STR, "det_ms": _NUM, "interval_s": _NUM,
        "cpu_idle": _NUM, "camera_up": _BOOL, "frames": _INT,
        "mount": _STR, "writes": _INT, "acquired": _INT,
    }),
}

# The image channels, which do get a well-known schema: foxglove.CompressedImage,
# so the Image panel finds them on its own with no layout to build.
IMAGE_CHANNELS = {
    "depth": "/camera/depth",
    "disparity": "/camera/disparity",
    "rectified_left": "/camera/rectified_left",
    # The hood camera with --pantilt, boxes drawn on (vision/runner.annotate).
    "pantilt_image": "/pantilt/image",
}

# How to turn each one's raw frame into something that can be looked at.
#
# This runs on the publisher thread, not on whoever handed the frame over. The
# camera reader offers each frame exactly as it came off the queue -- 16-bit
# millimetres for depth, raw disparity, 8-bit grey for the rectified mono --
# and all the clipping, normalising, colourmapping, downscaling and JPEG
# happens in here, behind the has_sinks() gate. Nobody watching, nothing
# computed, and none of it anywhere near the control thread.
DEPTH_JET = "depth_jet"
DISPARITY_JET = "disparity_jet"
GREY = "grey"
# A BGR frame that is already a picture: downscale and JPEG, nothing else.
COLOUR = "colour"

IMAGE_RENDER = {
    "depth": DEPTH_JET,
    "disparity": DISPARITY_JET,
    "rectified_left": GREY,
    "pantilt_image": COLOUR,
}

# frame_id on an image channel's messages; "oakd" for anything not named.
IMAGE_FRAME_IDS = {
    "pantilt_image": "pantilt",
}

# The window depth is colourmapped across, in millimetres. The same numbers
# the dashboard has always used: past 3 m this car does not care, and the
# bottom of the window is a picture-making choice rather than a sensor limit.
# It was very nearly a sensor limit once -- minimum range was 348 mm on
# subpixel -- but extended disparity reads down past 200 mm (165 measured), so
# 300 now clips real depth off the near end of the colourmap. Only the
# colourmap: the driver reads the raw depth frame and never comes through here.
DEPTH_NEAR_MM = 300
DEPTH_FAR_MM = 3000

JSON_TOPICS = {
    "controller": "/controller",
    "vehicle": "/vehicle",
    "diagnostics": "/diagnostics",
    "imu": "/imu",
    "marks": "/marks",
    "footprint": "/footprint",
    "lidar": "/lidar",
    "slam_status": "/slam/status",
    "nav_status": "/nav/status",
    "goal_status": "/goal/status",
    "patrol_status": "/patrol/status",
    "pantilt": "/pantilt",
}

# Raw depth: the uint16 millimetres exactly as the driver scans them, as a
# 16-bit greyscale PNG in a foxglove.CompressedImage. Lossless, so a recording
# can be re-scanned through driver.free_by_column() later with whatever the
# rules have become by then. The JPEG on /camera/depth is a picture and cannot
# do that.
#
# PNG and not a foxglove.RawImage, and that was measured, not guessed. A
# RawImage hands 512 KB a frame to the SDK, whose protobuf encode and zstd
# chunk compression run with the GIL held: at 2 Hz that took the control
# loop's p99 period from 13.7 ms to 34.1 ms on the car. cv2.imencode releases
# the GIL while it compresses, and what reaches the SDK is about 130 KB that
# zstd waves through.
RAW_DEPTH_CHANNELS = {
    "depth_raw": "/camera/depth_raw",
    # policy_input.depth_small() of the frame each decision used: 80x50, the
    # same 16-bit PNG, about 4.5 KB. Offered as (frame, frame_id).
    "depth_small": "/policy/depth_small",
}

# zlib level 1. Measured on the car over real frames: 133 KB in 24 ms; level 3
# is 123 KB in 36 ms. The 8% is not worth half again the encode.
DEPTH_PNG_LEVEL = 1

# The roof lidar for the 3D panel, in Foxglove's own schemas: a LaserScan of
# every revolution, the base_link -> lidar transform it is drawn through, and
# a truck outline to see it against. lidar_view.py builds all three, on the
# publisher thread. One scan is 720 doubles, 5.8 KB before zstd, and the three
# together came to 10.3 KiB/s (36 MiB an hour) through the SDK's own MCAP
# writer at 10 scans a second of a synthetic room. Not yet measured on the car.
LIDAR_VIEW_CHANNELS = {
    "tf": "/tf",
    "lidar_scan": "/lidar/scan",
    "lidar_truck": "/lidar/truck",
}

# The map with --slam, see map_view.py: a foxglove.Grid and a SceneUpdate,
# both in the "map" frame that the SLAM half of /tf hangs base_link off.
# Measured through the SDK's MCAP writer on a replayed drive: see
# tools/slam_bench.py --bandwidth.
SLAM_CHANNELS = {
    "map": "/map",
    "slam_path": "/slam/path",
}

# The OAK-D's maps with --voxel-map, see voxel_view.py: the occupied voxels
# as a foxglove.PointCloud and the low-obstacle layer as a foxglove.Grid, both
# in the "map" frame, once a second each.
VOXEL_CHANNELS = {
    "voxel_points": "/voxel/points",
    "voxel_low": "/voxel/low",
}

# The navigator's path, goal and aim with --nav, as a SceneUpdate in "map":
# nav_view.py, built on the publisher thread off pointnav.PointNavDriver.nav_path().
NAV_CHANNELS = {
    "nav_path": "/nav/path",
    # The clicked goal (--goal) as a disc in "map", nav_view.path_scene with
    # the goal only; an empty payload clears it.
    "goal_marker": "/goal/marker",
    # The patrol's waypoints (--goal), patrol_view.scene: numbered, joined in
    # order, the current one lit; an empty list clears them.
    "patrol_marker": "/patrol/marker",
}

ALL_KEYS = (tuple(JSON_TOPICS) + tuple(IMAGE_CHANNELS)
            + tuple(RAW_DEPTH_CHANNELS) + tuple(LIDAR_VIEW_CHANNELS)
            + tuple(SLAM_CHANNELS) + tuple(VOXEL_CHANNELS)
            + tuple(NAV_CHANNELS))

# Everything the camera produces, which is what stops being written while
# there is no pad: a file of nothing but pictures and IMU has no driving in it
# to pair them with, and one afternoon of those was 77 MB. The control loop's
# own channels are not in here -- /diagnostics is what says the pad went.
# /lidar is, for the same reason: it is a sensor with no driving in it, and
# so is its 3D view, the transform and the outline with it.
CAMERA_KEYS = frozenset(("imu", "lidar", "slam_status", "pantilt")
                        + tuple(IMAGE_CHANNELS)
                        + tuple(RAW_DEPTH_CHANNELS)
                        + tuple(LIDAR_VIEW_CHANNELS)
                        + tuple(SLAM_CHANNELS)
                        + tuple(VOXEL_CHANNELS))


class Telemetry:
    """The channels, the WebSocket server, the recorder, and the one thread.

    Everything public splits into two halves:

      the control thread calls wants() and offer(), and neither one can block
      on a socket, a file or a lock that is held during I/O;

      the publisher thread owns the server, the channels, the MCAP writer and
      all encoding.

    An instance that failed to start is not a special case anywhere: wants()
    is False for every key, offer() is a no-op, and the car never notices.
    """

    def __init__(self, logger=None, host=DEFAULT_HOST, port=DEFAULT_PORT,
                 recorder=None, server_options=None):
        self.logger = logger
        self.host = host
        self.port = port
        # More start_server() arguments, or None for the server as it always
        # was. nav/click.py's ClickGoals.server_options() goes here to let
        # Foxglove publish a clicked goal; nothing passes it yet.
        self.server_options = dict(server_options or {})

        self.running = False
        self.serving = False
        self.error = None

        # A recording.Recorder, or None for a run that is not being recorded.
        # It is opened and closed on the publisher thread, inside
        # _publish_loop, so that the sink's lifetime is exactly the drain
        # loop's lifetime -- no window where a channel has a file sink but
        # nothing is draining the mailboxes into it.
        self.recorder = recorder

        # A record session's own Recorder. The control thread asks for one
        # with begin_session() -- one attribute assignment -- and the
        # publisher thread does the swap: the ring segment closes, the
        # session file opens, and at no moment are both writing, so the
        # session's minutes are in exactly one file. end_session() reverses
        # it and the ring carries on in a fresh segment.
        self._session_request = None
        self._session = None

        self._server = None
        self._channels = {}
        self._thread = None
        self._stop = threading.Event()

        # One mailbox per channel, one deep. put_nowait into a full queue
        # raises instead of waiting, and the handler below throws the stale
        # item away and keeps the new one -- drop-oldest, which for live
        # telemetry means the laptop always sees the most recent sample it
        # could have seen rather than a backlog of old ones.
        #
        # These do take a lock. It is held for the handful of microseconds it
        # takes to append to a deque and never across any I/O: the publisher
        # thread get_nowait()s into a local and only then encodes and writes.
        # So the worst case for the control thread is waiting on a mutex that
        # is already released, not on a socket.
        self._mail = {key: queue.Queue(maxsize=1) for key in ALL_KEYS}

        # has_sinks(), snapshotted by the publisher thread. Reading a bool out
        # of a dict is the cheapest check I can give the control loop, and it
        # cannot block even in principle.
        self._watched = {key: False for key in ALL_KEYS}

        # Rate gates, advanced in offer() by whichever thread offered.
        self._next_at = {key: 0.0 for key in ALL_KEYS}

        # Channels nobody gets to write to right now, whoever is watching.
        # Replaced whole by mute()/unmute() rather than edited, so the camera
        # thread reading it in wants() sees one set or the other, never half.
        self._muted = frozenset()

        self._cv2 = None
        self._np = None

    # ---- lifecycle, called once from main --------------------------------

    def start(self, serve: bool = True) -> bool:
        """Build the channels and the thread. False means telemetry is off.

        `serve` is whether to also bind the WebSocket port. False gives a
        recording-only run: the channels exist, the thread drains them into
        the MCAP file, and nothing listens on the network.

        The channels come first and the socket second, and the socket failing
        is not the channels failing. Something already sitting on 8765 used to
        take the whole of telemetry out with it; now it costs the live view
        and the recording carries on, which is the right trade when the
        recording is the thing I cannot get back.

        Nothing in here is allowed to raise. A missing foxglove-sdk is an
        ordinary outcome that costs telemetry and nothing else.
        """
        if self.running:
            return True

        try:
            import foxglove
            from foxglove import Channel, Schema
            from foxglove.channels import (CompressedImageChannel,
                                           FrameTransformsChannel,
                                           GridChannel,
                                           LaserScanChannel,
                                           SceneUpdateChannel)

            # Quiet by default. The SDK logs every client connect and
            # disconnect at info, and this process shares a journal with the
            # thing that drives the car.
            try:
                foxglove.set_log_level("WARN")
            except Exception:
                pass

            for key, topic in JSON_TOPICS.items():
                spec = JSON_SCHEMAS[key]
                self._channels[key] = Channel(
                    topic=topic,
                    message_encoding="json",
                    schema=Schema(
                        name=spec["name"],
                        encoding="jsonschema",
                        data=json.dumps(spec["schema"]).encode("utf-8"),
                    ),
                )

            for key, topic in IMAGE_CHANNELS.items():
                self._channels[key] = CompressedImageChannel(topic=topic)

            for key, topic in RAW_DEPTH_CHANNELS.items():
                self._channels[key] = CompressedImageChannel(topic=topic)

            lidar_kinds = {"tf": FrameTransformsChannel,
                           "lidar_scan": LaserScanChannel,
                           "lidar_truck": SceneUpdateChannel}
            for key, topic in LIDAR_VIEW_CHANNELS.items():
                self._channels[key] = lidar_kinds[key](topic=topic)

            slam_kinds = {"map": GridChannel,
                          "slam_path": SceneUpdateChannel}
            for key, topic in SLAM_CHANNELS.items():
                self._channels[key] = slam_kinds[key](topic=topic)

            for key, topic in NAV_CHANNELS.items():
                self._channels[key] = SceneUpdateChannel(topic=topic)

            from foxglove.channels import PointCloudChannel
            voxel_kinds = {"voxel_points": PointCloudChannel,
                           "voxel_low": GridChannel}
            for key, topic in VOXEL_CHANNELS.items():
                self._channels[key] = voxel_kinds[key](topic=topic)

        except Exception as e:
            self.error = str(e)
            self._channels = {}
            self._log("warning", "Telemetry",
                      f"Not starting telemetry, driving as normal: {e}")
            return False

        if serve:
            try:
                import foxglove
                options = {"message_backlog_size": LIVE_BACKLOG}
                options.update(self.server_options)
                self._server = foxglove.start_server(
                    name="crawler",
                    host=self.host,
                    port=self.port,
                    **options,
                )
                self.serving = True
            except Exception as e:
                self.error = str(e)
                self._server = None
                self._log("warning", "Telemetry",
                          f"No Foxglove server, driving as normal: {e}")

        self.running = True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._publish_loop, name="telemetry", daemon=True)
        self._thread.start()
        if self.serving:
            self._log("info", "Telemetry",
                      f"Foxglove server on ws://{self.host}:{self.port}")
        return True

    def stop(self):
        """Shut it all down. Safe to call twice, and never raises."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

        self.running = False
        self.serving = False

        # Belt and braces. _publish_loop's finally has already closed the
        # recorder by the time that join returns; this catches the case where
        # the thread never started, and Recorder.stop() is idempotent.
        if self.recorder is not None:
            try:
                self.recorder.stop()
            except Exception:
                pass

        for channel in self._channels.values():
            try:
                channel.close()
            except Exception:
                pass
        self._channels = {}

        server, self._server = self._server, None
        if server is not None:
            try:
                server.stop()
            except Exception as e:
                print(f"[telemetry] server stop failed: {e}")

    # ---- the control thread's half ---------------------------------------

    def begin_session(self, recorder) -> bool:
        """Ask for `recorder` to own the file. One assignment, never blocks.

        False when there is no publisher thread to do it -- --no-telemetry
        --no-record -- and then there is no session either.
        """
        if not self.running:
            return False
        self._session_request = recorder
        return True

    def end_session(self) -> None:
        """Hand the file back to the ring. One assignment, never blocks."""
        self._session_request = None

    @property
    def session_active(self) -> bool:
        return self._session is not None

    def wants(self, key: str, now: float) -> bool:
        """Is this channel worth building a message for right now?

        False unless a Foxglove client is subscribed to it AND its rate gate
        has come round. Two dict lookups and a float compare; no lock, no
        allocation, no I/O. With Studio closed this is False forever and the
        caller does no work at all.
        """
        if not self.running:
            return False
        if not self._watched.get(key, False):
            return False
        if key in self._muted:
            return False
        return now >= self._next_at.get(key, 0.0)

    def mute(self, keys) -> None:
        """Stop writing `keys` until unmute(). One assignment, never blocks.

        wants() says no to them, so nothing is encoded for them either, and
        offer() drops anything that asked before the swap.
        """
        self._muted = frozenset(keys)

    def unmute(self) -> None:
        self._muted = frozenset()

    @property
    def muted(self) -> frozenset:
        return self._muted

    def offer_now(self, key: str, payload, now: float) -> bool:
        """offer(), past the rate gate. For an event that must not wait.

        A pad that comes or goes is one row I cannot afford to have a 4 Hz
        gate eat: the loop parks in wait_for_connection straight after, so
        there is no next pass for it to go out on. Still nothing unless
        something is listening -- a file or Studio -- and still never blocks.
        True if it was handed over.
        """
        if not self.running or not self._watched.get(key, False):
            return False
        if key in self._muted:
            return False
        self.offer(key, payload, now)
        return True

    def offer(self, key: str, payload, now: float) -> None:
        """Hand a message over and return. Never blocks, never raises.

        A full mailbox means the publisher thread has not drained the previous
        value yet, so the previous value is thrown away and this one takes its
        place. Dropping is the designed behaviour, not a failure.
        """
        if not self.running:
            return
        if key in self._muted:
            return

        mailbox = self._mail.get(key)
        if mailbox is None:
            return

        self._next_at[key] = now + INTERVALS.get(key, 0.1)
        item = (epoch_ns(now), payload)

        try:
            mailbox.put_nowait(item)
            return
        except queue.Full:
            pass

        try:
            mailbox.get_nowait()
        except queue.Empty:
            pass
        try:
            mailbox.put_nowait(item)
        except queue.Full:
            # The publisher thread refilled it between the two calls. Its
            # value is no older than one tick, so this one is the one to drop.
            pass

    # ---- the publisher thread's half -------------------------------------

    def _publish_loop(self):
        next_sink_poll = 0.0

        # The recorder opens here rather than in start() so that the MCAP
        # sink exists only while something is draining the mailboxes into it,
        # and so that a disk problem is handled on this thread -- the one
        # that is allowed to do I/O -- and never on the caller's.
        recorder = self.recorder
        if recorder is not None:
            try:
                recorder.start()
            except Exception as e:
                # Recorder.start() already swallows everything; this is the
                # case where it has a bug. A recording is never worth this
                # thread.
                self.error = str(e)

        try:
            self._drain_forever(recorder)
        finally:
            for r in (self._session, recorder):
                if r is not None:
                    try:
                        r.stop()
                    except Exception:
                        pass
            self._session = None

    def _swap_session(self, ring):
        """Make the file owner match what the control thread asked for.

        Close first, then open, so no message lands in two files. The gap is
        a few milliseconds with no sink, which costs at worst a sample.
        """
        wanted = self._session_request
        if wanted is self._session:
            return
        if self._session is not None:
            # What the control loop offered just before it ended the session
            # belongs in the session's file -- the row that says the pad went
            # above all, offered on the same pass as the end. Without this the
            # swap goes first and that row opens the next ring segment.
            self._drain_mail(self._session)
            try:
                self._session.stop()
            except Exception:
                pass
            self._session = None
            if ring is not None and wanted is None:
                try:
                    ring.start()
                except Exception as e:
                    self.error = str(e)
        if wanted is not None:
            if ring is not None:
                try:
                    ring.stop()
                except Exception:
                    pass
            try:
                wanted.start()
            except Exception as e:
                self.error = str(e)
            self._session = wanted
        # A sink just came or went. Notice now rather than at the next poll,
        # so the first decision of a session is in the session's file.
        self._refresh_watched()

    def _drain_forever(self, ring):
        next_sink_poll = 0.0

        while not self._stop.is_set():
            now = time.monotonic()

            self._swap_session(ring)
            recorder = self._session if self._session is not None else ring

            if now >= next_sink_poll:
                next_sink_poll = now + SINK_POLL_INTERVAL
                self._refresh_watched()

            if recorder is not None:
                # Rotation and pruning, on this thread, behind a float
                # compare. See recording.py: almost every call returns
                # without a syscall.
                recorder.tick(now)

            self._drain_mail(recorder)

            self._stop.wait(DRAIN_INTERVAL)

    def _drain_mail(self, recorder):
        """Emit whatever is in the mailboxes, once each. Publisher thread."""
        for key in ALL_KEYS:
            try:
                log_time, payload = self._mail[key].get_nowait()
            except queue.Empty:
                continue
            except Exception:
                continue

            # Outside the mailbox lock, always. Encoding a JPEG while
            # holding it would hand the control thread a lock that is
            # held for milliseconds, which is the one thing this whole
            # file is built to prevent.
            try:
                self._emit(key, payload, log_time)
            except Exception as e:
                # A client that went away mid-write lands here. It is not
                # worth a log line per frame, and it is certainly not
                # worth stopping for.
                self.error = str(e)
                if recorder is not None:
                    # channel.log() writes to every sink and does not say
                    # which one failed, so the recorder counts this and
                    # forgets the count again as soon as it can see its
                    # own file growing. A card that has genuinely started
                    # refusing writes is the case it is looking for.
                    recorder.note_write_error(e)

    def _refresh_watched(self):
        for key, channel in self._channels.items():
            try:
                # A method on this SDK, not a property, whatever the .pyi
                # stubs suggest. `if channel.has_sinks:` is a bound method and
                # therefore always truthy, which silently defeats the whole
                # don't-encode-when-nobody-is-watching idea.
                self._watched[key] = bool(channel.has_sinks())
            except Exception:
                self._watched[key] = False

    def _emit(self, key, payload, log_time):
        channel = self._channels.get(key)
        if channel is None:
            return

        if key in JSON_TOPICS:
            channel.log(payload, log_time=log_time)
            return

        if (key in LIDAR_VIEW_CHANNELS or key in SLAM_CHANNELS
                or key in VOXEL_CHANNELS or key in NAV_CHANNELS):
            message = self.lidar_view_message(key, payload, log_time)
            if message is not None:
                channel.log(message, log_time=log_time)
            return

        if key in RAW_DEPTH_CHANNELS:
            frame_id = "oakd"
            if isinstance(payload, tuple):
                payload, frame_id = payload
            message = self.depth_png(payload, log_time, frame_id)
            if message is not None:
                channel.log(message, log_time=log_time)
            return

        data = self._encode_jpeg(payload, IMAGE_RENDER.get(key, GREY))
        if data is None:
            return

        from foxglove.messages import CompressedImage, Timestamp

        channel.log(
            CompressedImage(
                timestamp=Timestamp(sec=log_time // 1_000_000_000,
                                    nsec=log_time % 1_000_000_000),
                frame_id=IMAGE_FRAME_IDS.get(key, "oakd"),
                data=data,
                format="jpeg",
            ),
            log_time=log_time,
        )

    @staticmethod
    def lidar_view_message(key, payload, log_time):
        """The SDK message for one of LIDAR_VIEW_CHANNELS, or None.

        What lidar.LidarReader offers: "lidar_scan" a (Scan, Mount) pair,
        "tf" the Mount, "lidar_truck" nothing it needs. With --slam, the
        mapping runner offers "tf" as (Mount, (x, y, yaw), scan time) and
        the SLAM_CHANNELS: "map" a runner.MapView, "slam_path" (x, y) points.
        """
        if key == "lidar_scan":
            scan, mount = payload
            return lidar_view.laser_scan(scan, log_time, mount)
        if key == "tf":
            if isinstance(payload, tuple) and len(payload) == 3:
                from crawler.telemetry import map_view
                mount, pose, t = payload
                return map_view.frame_transforms(log_time, mount, pose,
                                                 epoch_ns(t))
            return lidar_view.frame_transforms(log_time, payload)
        if key == "lidar_truck":
            return lidar_view.truck_scene(log_time)
        if key == "map":
            from crawler.telemetry import map_view
            return map_view.grid_message(payload, log_time)
        if key == "slam_path":
            from crawler.telemetry import map_view
            return map_view.path_scene(payload, log_time)
        if key == "voxel_points":
            from crawler.telemetry import voxel_view
            return voxel_view.points_message(payload, log_time)
        if key == "patrol_marker":
            from crawler.telemetry import patrol_view
            return patrol_view.scene(payload, log_time)
        if key in ("nav_path", "goal_marker"):
            from crawler.telemetry import nav_view
            return nav_view.path_scene(payload, log_time)
        if key == "voxel_low":
            from crawler.telemetry import voxel_view
            return voxel_view.low_message(payload, log_time)
        return None

    @staticmethod
    def depth_png(frame, log_time, frame_id="oakd"):
        """One depth frame as a lossless 16-bit PNG CompressedImage, or None.

        uint16 in, uint16 out: no clip, no scale, no colourmap. Anything that
        is not a 2-D uint16 frame is refused rather than squeezed into one.
        """
        try:
            import cv2
            import numpy as np
        except Exception:
            return None
        if frame is None or frame.ndim != 2 or frame.dtype != np.uint16:
            return None
        ok, buf = cv2.imencode(".png", frame,
                               [int(cv2.IMWRITE_PNG_COMPRESSION),
                                DEPTH_PNG_LEVEL])
        if not ok:
            return None

        from foxglove.messages import CompressedImage, Timestamp

        return CompressedImage(
            timestamp=Timestamp(sec=log_time // 1_000_000_000,
                                nsec=log_time % 1_000_000_000),
            frame_id=str(frame_id),
            data=buf.tobytes(),
            format="png",
        )

    def _encode_jpeg(self, frame, how):
        """Render one raw camera frame to JPEG bytes. None if it cannot be.

        Downscale first, then colourmap. That order is not cosmetic: it is a
        4x reduction in pixels before the expensive part, and INTER_NEAREST on
        a depth map is the right resampler anyway -- averaging two depths that
        are a metre apart invents a surface that is not there.

        cv2 and numpy are imported here rather than at module scope, and
        cached. The controller, vehicle and diagnostics channels have no
        business needing opencv installed, and this file has to import on a
        laptop that has none of the car's libraries on it.
        """
        if frame is None:
            return None

        if self._cv2 is None:
            try:
                import cv2
                import numpy as np
                self._cv2 = cv2
                self._np = np
            except Exception as e:
                self.error = f"no opencv for image telemetry: {e}"
                return None

        cv2, np = self._cv2, self._np

        height, width = frame.shape[:2]
        longest = max(height, width)
        if longest > IMAGE_MAX_DIM:
            scale = IMAGE_MAX_DIM / float(longest)
            frame = cv2.resize(frame, (max(1, int(width * scale)),
                                       max(1, int(height * scale))),
                               interpolation=cv2.INTER_NEAREST)

        if how == DEPTH_JET:
            # Millimetres to a fixed window, so the colours mean the same
            # distance from one frame to the next. A per-frame normalise would
            # make an empty corridor look exactly like a wall at arm's length.
            frame = np.clip(frame, DEPTH_NEAR_MM, DEPTH_FAR_MM)
            frame = ((frame - DEPTH_NEAR_MM)
                     * (255.0 / (DEPTH_FAR_MM - DEPTH_NEAR_MM))).astype(np.uint8)
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        elif how == DISPARITY_JET:
            # Disparity has no fixed unit to key off, so this one does get a
            # per-frame min/max stretch. Same as the dashboard pane.
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame = cv2.applyColorMap(np.uint8(frame), cv2.COLORMAP_JET)
        elif frame.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame = np.uint8(frame)

        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return None
        return buf.tobytes()

    # ---- odds and ends ---------------------------------------------------

    def recording_status(self) -> dict:
        """The four /diagnostics fields that describe the recording.

        Called from the control loop at 4 Hz, so it is attribute reads off an
        object the publisher thread is also touching -- and nothing more. No
        lock: the worst case is a basename and a size from one side of the
        same rotation, which is a cosmetic lie in a diagnostics message. The
        alternative is a lock the control thread has to take, and this file
        does not hand the control thread locks.
        """
        session = self._session
        recorder = session if session is not None else self.recorder
        if recorder is None:
            return {
                "recording": False,
                "recording_file": "",
                "recording_mib": 0.0,
                "recording_error": "",
                "recording_session": False,
            }
        try:
            path = recorder.path or ""
            return {
                "recording": bool(recorder.active),
                "recording_file": os.path.basename(path) if path else "",
                "recording_mib": round(recorder.bytes_written / 1048576.0, 1),
                "recording_error": str(recorder.error or ""),
                "recording_session": session is not None,
            }
        except Exception as e:
            return {
                "recording": False,
                "recording_file": "",
                "recording_mib": 0.0,
                "recording_error": str(e),
                "recording_session": False,
            }

    def _log(self, level, subject, message):
        if self.logger is not None:
            try:
                self.logger.log(level, subject, message)
                return
            except Exception:
                pass
        print(f"[telemetry] {subject}: {message}")


def controller_sample(joystick, throttle, steering_front, steering_rear):
    """Everything the pad is reporting, as a /controller message.

    Reads Joystick's own XBOX_AXES and XBOX_BUTTONS at call time instead of
    carrying a copy. That map is being measured right now against an xpadneo
    driver change that renumbers both, and a second copy of it in here would
    be a second thing to get wrong -- and would make this channel lie about
    exactly the thing it is meant to diagnose.

    Everything is read defensively. A pad that dropped and re-paired at a
    different SDL index leaves a handle whose getters raise, and this is a
    debug channel: a missing field beats an exception on the caller's thread.
    """
    from crawler.hardware import joystick as js

    sample = {
        "connected": False,
        "name": "",
        "num_axes": 0,
        "num_buttons": 0,
        "axes": [],
        "buttons": [],
        "named_axes": {},
        "named_buttons": {},
        "trigger_rest_rt": 0.0,
        "trigger_rest_lt": 0.0,
        "throttle": float(throttle or 0.0),
        "steering_front": float(steering_front or 0.0),
        "steering_rear": float(steering_rear or 0.0),
    }

    try:
        sample["connected"] = bool(joystick.is_connected())
    except Exception:
        return sample

    handle = getattr(joystick, "joystick", None)
    if handle is None:
        return sample

    try:
        sample["name"] = str(handle.get_name())
    except Exception:
        pass

    # Raw first. This is the part that is worth having when the named map is
    # the suspect: if LEFT_X stops steering, the axis that moves in here is
    # the one the map should be pointing at.
    try:
        count = int(handle.get_numaxes())
        sample["num_axes"] = count
        sample["axes"] = [float(handle.get_axis(i)) for i in range(count)]
    except Exception:
        pass

    try:
        count = int(handle.get_numbuttons())
        sample["num_buttons"] = count
        sample["buttons"] = [int(handle.get_button(i)) for i in range(count)]
    except Exception:
        pass

    for name in getattr(js, "XBOX_AXES", {}):
        try:
            sample["named_axes"][name] = float(joystick.get_axis(name))
        except Exception:
            pass

    for name in getattr(js, "XBOX_BUTTONS", {}):
        try:
            sample["named_buttons"][name] = bool(joystick.get_button(name))
        except Exception:
            pass

    rest = getattr(joystick, "_trigger_rest", None) or {}
    try:
        sample["trigger_rest_rt"] = float(rest.get("RT", 0.0))
        sample["trigger_rest_lt"] = float(rest.get("LT", 0.0))
    except Exception:
        pass

    return sample


def imu_sample(reading):
    """An OakD.get_imu_sample() dict as an /imu message, or None."""
    if not reading:
        return None
    try:
        ax, ay, az = (float(v) for v in reading["accel"])
        gx, gy, gz = (float(v) for v in reading["gyro"])
    except Exception:
        return None
    up = policy_imu.row_fields(reading, 0.0, 0.0)
    return {
        "accel_x": ax, "accel_y": ay, "accel_z": az,
        "gyro_x": gx, "gyro_y": gy, "gyro_z": gz,
        "accel_magnitude": (ax * ax + ay * ay + az * az) ** 0.5,
        "up_x": up["imu_up_x"], "up_y": up["imu_up_y"], "up_z": up["imu_up_z"],
    }
