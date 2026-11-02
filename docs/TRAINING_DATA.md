# Training data: what a drive writes down

The learned policy is a small CNN over a downsampled depth image. Every
decision tick the car writes the picture that policy would see and the command
that was actually on the wheels, at one log time, into the MCAP recording.
This file is the contract between the car and the trainer. The code that
holds it is `crawler/autopilot/footprint.py` (the rows), `crawler/autopilot/policy_input.py` (the picture) and
`crawler/telemetry/telemetry.py` (`JSON_SCHEMAS["footprint"]`, `RAW_DEPTH_CHANNELS`).

## Where the data comes from

- **Record mode** is the training drive: in manual, hold **Y for one second**
  (two buzzes) and the car opens the camera and starts a new MCAP of its own.
  It drives exactly like manual except the throttle is scaled: full RT is
  auto's forward speed (`Driver.forward_speed`, 0.4 today, 1745 us), full LT
  is auto's reverse speed (`Driver.reverse_speed`, -1.0 today, 1000 us:
  full reverse, because the ESC needs a much bigger pulse to start backing
  up than going forward). Every recording up to the first record session
  was made at 0.2 / -0.6, and the sessions after it up to the reverse change
  at 0.4 / -0.296.
  Steering is unchanged. Any
  fresh press of Y goes back to manual (one buzz); START ends the session and
  goes to auto; losing the pad ends it at neutral. A and B marks work as
  always. Back in manual, from record or from auto, the throttle stays 0
  until both triggers have been let go once (`main.ThrottleLatch`). Manual
  writes no rows, so this never shows up in the data.
- **Auto** writes the same rows for the autopilot's own decisions, into the
  rolling recording, from the frame the driver actually scanned. Whenever the
  recorder is on (`--record`, the default) these go to disk with their
  `/policy/depth_small` pictures; nothing has to be watching in Foxglove.
- **Takeover, in auto:** moving the left stick (plus the right in split) takes
  the steering off the autopilot, and pulling a trigger takes the throttle, one
  group at a time; the autopilot keeps driving the other. A group goes back
  after 0.5 s inside its deadzone. A taken throttle is scaled exactly like
  record's. The rows' `steer_by` / `throttle_by` say which hand had each group.
- **Manual** writes no rows.

Files: a record session is `log/mcap/crawler-record-<UTC>Z-p<pid>-001.mcap`
(`/home/jt/Documents/py/crawler/log/mcap/` on the car, or `--record-dir`), one
per session, never pruned by the ring; the rolling recording is
`crawler-<UTC>Z-...mcap` in the same directory. `./scripts/pull-recordings.sh` pulls
both, into `~/crawler/data/mcap/` by default; `--records-only` pulls just the
record sessions and leaves the drive logs on the car.

A record session ends on a Y tap, on START, on losing the pad, or when the pad
has not proved it is there for 5 s; the last two buzz one long pulse instead
of one short one. A pad that dies quietly is lost about 5.8 s later: the
Bluetooth link times out at 5 s on every connection
(`deploy/install-pad-lst.sh`; the default is 20 s), and the loop sees the
device go within about 0.8 s. Either way the session's last `/diagnostics` row says
`controller_connected: false`. With no pad connected the camera channels
(`/camera/*`, `/imu`, `/policy/depth_small`) and `/lidar` are not written at
all, in any file, until it is back.

## The tick and how to pair things

One row per decision, at 10 Hz (`footprint.INTERVAL` = auto's
`AUTO_TICK_INTERVAL`), never at the control loop's 100 Hz. On each tick, all
of these carry the **same `log_time`** (ns since the epoch):

| topic | what | every |
|---|---|---|
| `/footprint` | JSON row below | tick |
| `/policy/depth_small` | the policy picture, `frame_id` = `footprint/<seq>` | tick |
| `/camera/depth_raw` | the full 640x400 frame the row was computed from | 5th record tick (2 Hz); never in auto |

Pair by `log_time`, or by `seq` against the `frame_id` suffix; both always
agree. `seq` counts from 1 per process, so a gap in it is a dropped row.

`/lidar` (only with `--lidar`) is not on the tick: it is one row per
revolution of the roof lidar, about 10 Hz on the sensor's own clock. Pair it
with a tick by nearest `log_time`, and treat a row more than 0.3 s older than
the tick as no scan (`lidar.STALE_S`).

## `/lidar` (JSON, schema `crawler.Lidar`)

`bins` is 72 nearest ranges in mm, from the middle of the car, one per 5
degrees: bin `i` is centred on bearing `i * 5` counter-clockwise from straight
ahead, so 0 ahead, 18 left, 36 behind, 54 right. 0 is no return in that bin.
`nearest_front_mm` / `_left_` / `_rear_` / `_right_` are the nearest bin in each
quarter (the 45 degree bins count in both neighbours). `rpm`, `packets_per_s`
(417 expected), `crc_error_rate` and `crc_errors` are the sensor's health.
`crawler/hardware/lidar.py` has the mount, the frame and why 5 degrees.

## `/policy/depth_small`

`foxglove.CompressedImage`, `format` = `"png"`, a **16-bit greyscale PNG,
80 wide x 50 tall, uint16 millimetres**, lossless. Decode with
`cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)` (or Pillow); the array is
`(50, 80)` `uint16`.

It is `policy_input.depth_small(depth_frame)` and nothing else:

- The 640x400 depth frame is cut into 8x8-pixel blocks.
- Each block becomes the **minimum of its valid (non-zero) pixels**: the
  nearest thing in it. A min, not a mean or median, so a chair leg thinner
  than a block keeps its real distance.
- A block with **fewer than 16 of its 64 pixels valid** (`MIN_VALID_FRACTION`
  0.25) is **0**. So 0 means "no reading" exactly as in the raw frame, never
  "touching". 65535 is a real, far reading.
- No clipping, scaling or normalising. That is the trainer's job.

**Inference on the car calls this same function.** Import it from
`crawler/autopilot/policy_input.py`; do not reimplement it. `deploy/footprint_dump.py --verify`
checks, frame by frame, that every recorded picture with a raw frame behind it
equals `depth_small(raw)` byte for byte.

Depth frame facts (`crawler/hardware/oakd.py`): 400P mono pair at 16 fps, depth aligned to the
RGB socket, left-right check on, extended disparity instead of subpixel. So
the nearest valid depth is about 165 mm (closer reads 0), and depth is
quantised to whole disparity levels: about 5 mm steps at 400 mm, 25 mm at
900 mm.

## `/footprint` (JSON, schema `crawler.Footprint`)

| field | type | meaning |
|---|---|---|
| `seq` | int | per process, from 1 |
| `mode` | str | `"record"` or `"auto"` |
| `source` | str | `"tap"` (our own scan: record, or auto in a recover pass with no fresh driver scan while I have a hand in) or `"driver"` (auto: the driver's scan, back-outs included) |
| `frame_width`, `frame_height` | int | depth frame size, 640x400 (0 if unknown) |
| `throttle` | float | **the label.** Throttle that reached the ESC this tick, -1..1 (record: already scaled) |
| `steering_front`, `steering_rear` | float | **the label.** Steering sent to each servo, -0.7..0.7 (the pad's LEFT_X past a 0.12 deadzone, rescaled, x 70%; recordings before the deadzone carry the stick's rest drift, up to about 0.07 raw); `rear` = -`front` in unified mode |
| `steering_mode` | str | `"unified"` or `"split"` (L3 toggles): the pad's mode. See "The steering mode as a label" |
| `steer_by` | str | `"pad"` or `"driver"`: who had the steering when this row's command went out. Always `"pad"` in record |
| `throttle_by` | str | `"pad"` or `"driver"`: same, for the throttle |
| `pad_throttle` | float | record, or auto with `throttle_by` `"pad"`: the trigger before scaling, -1..1. Otherwise 0 |
| `forward_limit`, `reverse_limit` | float | the speed caps in force on this row, e.g. 0.4 / -1.0 (0.2 / -0.6 in the first sessions, 0.4 / -0.296 after): the scale for `pad_throttle` in record and under takeover, and auto's own caps on every auto row. Older recordings carry -1 on auto rows no trigger drove; read those against the session's other caps |
| `imu_accel_x/y/z` | float | newest OAK-D IMU sample at the tick, linear acceleration, m/s^2 |
| `imu_gyro_x/y/z` | float | same sample, rad/s |
| `imu_age_ms` | float | age of that sample at the tick; **-1 and all zeros if there has been none** |
| `imu_up_x/y/z` | float | same sample: the unit vector pointing away from the ground, in the IMU's own axes, off `GAME_ROTATION_VECTOR`. **All three 0 when there is no orientation reading**, which is every recording made before this field existed. See "Tilt" |
| `autopilot` | string | who drove auto's `driver` groups: `""` in record, `rules` or `policy` (`python -m crawler --policy`) |
| `policy_model` | string | which model drove, by name and version, e.g. `kblock 1.1` (the sidecar's `model_name` and `model_version`; a sidecar without them reads as its file's stem, e.g. `policy-20261028-170131Z`). `""` unless `autopilot` is `policy`. **Rows from before the name existed carry the sha256 here instead**: 64 hex characters, look it up in `~/crawler/data/models/registry.json` |
| `policy_model_sha256` | string | sha256 of the `.onnx` driving, the exact model; `""` unless `autopilot` is `policy`. Absent in rows from before the name existed |
| `policy_action` | string | the model's pick: `forward`, `reverse` or `stop` |
| `policy_throttle` | float | the model's throttle after its floor veto; what it would have driven even while my trigger has the group |
| `policy_steering_front` | float | the model's front steering, in manual's units (full lock 0.7) |
| `policy_reason` | string | why it did that: `model`, `model floor` (a forward held by `driver.ForwardFloorVeto`: something under `FORWARD_FLOOR_MM`, or a probe that went blind on the way in), `model creep` (a map model's forward the floor holds at the cap but not at `policy.CREEP_THROTTLE`, driven at no more than that), `model commit` (a fresh reverse held by the reverse commit), `standoff` (the floor held a forward for `policy.STANDOFF_S`, so the driver backed out), `rear bumper`, `blind`, `stuck`, ... |
| `policy_infer_ms` | float | the model's inference time that tick |
| `policy_lidar_present` | bool | a lidar model was fed a real scan this decision (one no older than `lidar.STALE_S`); false when it ran on no scan, on a tick the model did not run, and for every model without the lidar input. Absent in rows from before the lidar input |
| `policy_steer_mode` | string | the steering mode the model drove with, `unified` or `split` (`policy.ModeLatch`); always `unified` for a model with no mode head. Not the row's `steering_mode`, which is the pad's |
| `policy_steer_mode_p` | float | the mode head's probability of `policy_steer_mode` on its last answer, low while it wants to switch; -1 with no head |
| `reverse_commit` | bool | a reverse commit held this tick: a fresh model reverse kept for `policy.REVERSE_COMMIT_S` at `policy.REVERSE_STEER`, whatever the model said. `policy_action` is still the model's own pick |
| `reverse_refractory` | bool | a fresh model reverse came inside `policy.REVERSE_REFRACTORY_S` of the last autonomous reverse, with the floor not holding, and was driven as a forward. `policy_action` is still the model's own pick. Absent in rows from before the refractory |
| `standoff_ticks` | int | decisions in a row the floor has held a model forward; 0 otherwise |
| `backout_phase` | string | the driver's back-out this tick: `reverse`, `settle`, `handback` (the model or the rules drive again, `driver.HANDBACK_S` of steering lean still on), or `""`. The fields below describe the current back-out, or the last one once it is over. Absent in rows from before the back-out |
| `backout_reason` | string | why it started: `stuck`, `standoff` or `obstacle`; `""` before the first |
| `backout_exit` | string | how the last reverse ended: `opened` (the way forward cleared for `driver.BACKOUT_OPEN_DECISIONS` decisions), `rear` (the lidar rear stop), `bumper`, `cap`, `no frame` or `abort`; `""` while it still reverses |
| `backout_elapsed_s` | float | seconds since this back-out started; 0 when `backout_phase` is `""` |
| `backout_reverse_s` | float | how long its reverse ran, once it ended; 0 before |
| `backout_side` | float | the side it reversed toward, -1 or 1 (front steering sign) |
| `backout_speed` | float | the throttle it commanded this tick, never further back than `driver.ESCAPE_SPEED`; 0 outside a reverse |
| `backout_level` | int | 0, or how many back-outs in a row hit the same spot within `driver.ESCALATE_WINDOW_S` |
| `rear_clearance_mm` | float | the rear bumper to the nearest lidar return in `driver.REAR_STOP_SECTOR`, mm, returns inside the car dropped; -1 with nothing there, no fresh scan or no `--lidar`. Absent in rows from before the rear stop |
| `rear_stop` | bool | the rear stop (`driver.REAR_STOP_CLEAR_MM`) ended or refused an autonomous reverse this decision |
| `policy_steer_dir_p` | float[3] | a direction-head model's steering class probabilities on its last answer, `[left, straight, right]`; all -1 for a regression-head model, all 0 when no policy drives. Absent in rows from before the direction head |
| `depth_small` | bool | a `/policy/depth_small` picture carries this row's log time |
| `depth_raw` | bool | a `/camera/depth_raw` frame carries this row's log time |
| `probe_distance_mm` | float[9] | the hand-written scan, left to right; -1 means infinite |
| `probe_valid_fraction` | float[9] | per probe |
| `probe_blocked` | bool[9] | per probe |
| `probe_reason` | str[9] | `"near"`, `"no_return"` or `""` |
| `probe_marks` | str | the nine as one string: `N` near, `?` no return, `.` clear |
| `sweep_clears_left`, `sweep_clears_right` | bool | `driver.turn_sweep_clears(scan, -1)` / `(scan, +1)`: could a full-lock turn that way begin now without the car's swept corner touching anything the fan sees |
| `scan_ms` | float | time the scan took |
| `verdict_action`, `verdict_steer`, `verdict_reason` | str, float, str | what `driver.read_the_gradient()` says about this scan: `forward`, `evade` or `reverse`. `verdict_steer` is -1 (left) or +1 (right) on an evade, otherwise the cruise bias or 0 |
| `committed_action`, `committed_steer`, `committed_reason` | str, float, str | auto only: what the driver actually did after its evade latch. Empty in record |

The reason strings are free text: the first word names the rule, and a number
after it is that rule's measure. As of `crawler/autopilot/driver.py` at `11cecdd`:

- forward: `clear`
- reverse: `no_return`; `floor <mm>` (nearest blocked probe under
  `REVERSE_FLOOR_MM`, 200 mm); `split`; `spans`; `wide <n>`; `middle` (an exact
  tie); `shallow <ratio>`; `tight <mm>` (the sweep cannot clear the turn side)
- evade: `graze <ratio>` (off an edge block); `dodge <ratio>` (around a
  block in the middle)
- committed only: `holding out`, `evade flipped`, `evade timeout`

Split on the first space to get the rule. The list moves with `crawler/autopilot/driver.py`, so
treat an unknown first word as a new rule, not an error.

The probe and verdict fields are the rule-based driver's opinion, recorded as
a baseline to beat. In record **nothing acts on them**; the wheels get only
the pad (a test holds this).

**Which half of a row is a label.** `throttle`, `steering_front` and
`steering_rear` are always what reached the hardware, and in auto that is a
mix: each group is either my hand or the autopilot. A group marked `"pad"` is a
human label -- in auto it is a correction, the most useful kind. A group marked
`"driver"` is NOT a demonstration: it is the rule-based autopilot's own output
and a policy trained on it learns the rules, not me. Mask it out of that
group's loss, or down-weight it hard; never mix it in as if it were mine. A
row can be `"pad"` on one group and `"driver"` on the other, so mask per group,
not per row. For the 0.5 s after I let go a group is still `"pad"` and reads my
stick's resting value (near centre, or a throttle of 0); that is a real thing
my hand did, but it is also where I stopped meaning anything, so it is worth
trimming.

Rows that do not exist: manual mode. A back-out scans every decision, so its
rows are the driver's with the `backout_*` fields set (recordings from before
it have no rows in the old 2.5 s recover). A recover pass with no fresh driver
scan while my hand has a group still makes a row: 10 Hz off the car's own view,
`source` `"tap"`, `mode` `"auto"`, with a `/policy/depth_small` picture and no
raw frame.

## Tilt, and what the IMU columns mean

`imu_accel_*` is `LINEAR_ACCELERATION`: gravity is removed, so a parked car
reads about 0 whatever slope it sits on. Tilt was not recorded at all until
`imu_up_*`, which comes off a second IMU report, `GAME_ROTATION_VECTOR`
(accelerometer and gyro fused, roll and pitch referenced to gravity, no
magnetometer; `oakd.IMU_SENSORS` says why this one). The existing columns did
not change: accel and gyro are still the first packet's numbers, and the up
vector rides in its own report slot.

- Units: none. `imu_up_*` is a unit vector, `(0, 0, 0)` meaning "no reading".
- Axes, measured off the three 0.4 / -1.0 recordings: **x is vertical**
  (`imu_gyro_x` is yaw), **y is lateral**, **z is along the car**. So a level
  car should read `imu_up` near `(+-1, 0, 0)`, and pitch and roll are the
  angles of the up vector away from x toward z and toward y:
  `pitch = atan2(up_z, |up_x|)`, `roll = atan2(up_y, |up_x|)`, in radians.
- The quaternion to up-vector step (`policy_imu.up_from_quaternion`) uses
  Android's rotation-vector convention, which the BNO08x follows. **Not yet
  checked on the car.** The first parked recording should read `|up_x|` near
  1; if it reads near 1 on another axis instead, the mounting or the
  convention is not what this says.
- `/imu` carries the same three as `up_x/y/z`.

## The IMU as a policy input

`crawler/autopilot/policy_imu.py` is the one place IMU numbers become the policy's `imu`
input. The dataset tool pushes each session's `/footprint` rows through
`policy_imu.ImuFeatures` in time order (every row, before any is dropped) and
stores the result as the `.npz`'s `imu` array; `PolicyDriver` builds the
same row live with `policy_imu.row_fields` (the function `crawler/autopilot/footprint.py`
writes the row with) and pushes it through its own `ImuFeatures`. Features,
in order (`policy_imu.FEATURES`): gyro xyz over 2 rad/s, linear accel xyz
over 10 m/s^2, the largest |accel| and |gyro| of the last 5 decisions (0.5 s)
on the same scales, `valid`, up xyz and `tilt_valid`; every value clipped to
+-3. A row with no sample (`imu_age_ms` -1) or one older than 250 ms is all
zeros with `valid` 0. Rows more than 0.25 s apart restart the half-second
window. A model trained with `policy_train.py --imu` declares `uses_imu`,
`imu_features` and `imu_tilt` in its sidecar. `imu_tilt` is true only with
`--imu-tilt`; without it the up columns are zeroed in training and on the
car, even when the dataset has an up vector. `--imu-tilt` is refused on a
dataset with none, and a session recorded before the up vector existed goes
in as zeros with `tilt_valid` 0, never a made-up reading. The trainer prints
how many training ticks had a real up vector. A model without `uses_imu` has
no `imu` input and never gets one.

`--imu-no-yaw` trains without the yaw rate. gyro_x is yaw on this car, and a
model fed it reads back the turn its steering label made; held out, the full
IMU's steering gain was gone with the gyro zeroed. Its features are
`policy_imu.NO_YAW_FEATURES`: no gyro_x, and `pitch_roll_peak` (the gyro peak
over pitch and roll only) in place of `gyro_peak`. The dataset stores them as
`imu_no_yaw`, built by the same `ImuFeatures` with `yaw=False` that the car
builds them with, and the sidecar says `imu_yaw: false` with that list as
`imu_features`. A car on code from before `imu_yaw` refuses such a model.

## The lidar as a policy input

`crawler/autopilot/policy_lidar.py` is the one place a scan becomes the
policy's `lidar` input, on the car and in training. The input is 145 floats:
the 72 bins of `/lidar` over `policy_lidar.CLIP_MM` (4 m, and anything past it
is 4 m), then 72 flags for "this bin had a return" (a 0 bin reads as nothing
there, not as touching), then one `present` flag. No scan is all zeros,
`present` included.

The dataset pairs each tick with the newest `/lidar` row at or before it, and
no row when that one is more than `lidar.STALE_S` (0.3 s) old: the same rule
as `LidarReader.latest()`, which is what the car feeds. A `/lidar` row in
another binning counts as no scan. `policy_dataset.py` stores `lidar_bins`
(the raw mm), `lidar_present` and `lidar_age_ms` (-1 with no scan) per tick,
and says how many kept ticks had one. Recordings from before the lidar have
none, and train as no scan.

A model trained with `policy_train.py --lidar` declares `uses_lidar` and
`lidar_layout` (`bins72_5deg`), and the car refuses one whose layout, size or
clip is not its own, a graph with a `lidar` input the sidecar does not
declare, and any graph input it does not know how to feed. It trains with
sensor dropout (`--lidar-dropout`, 0.3 by default): that share of the ticks
with a scan train as no scan, so the model still drives on the camera when the
lidar is gone, stale or unplugged. The left-right flip mirrors the scan with
it (bin `i` to bin `(72 - i) % 72`). A lidar model is a new contract, so its
first release is a major version. A car on code from before the lidar input
loads one and then fails on its first decision for the missing input.

## The steering mode as a label

Unified drives both axles off the left stick, the rear counter-phase through
`driver.steering_pair()`: four-wheel steering. Split puts the rear on the
right stick, and I have never used it: in every split row of the current
session the rear is exactly 0, so for me split is two-wheel steering. That
is what the policy's mode head learns, and the rear is never a head of its
own: unified is `steering_pair(front)`, split is 0.

**A split row with the rear off centre is not modelled yet.**
`tools/policy_dataset.py` drops it (`split rear`, past
`SPLIT_REAR_REST` = 0.05 of servo travel) and the model's sidecar says so in
`mode_not_modelled`. If I start crabbing or posing with the right stick,
that is the data a rear head would need, and none exists today.

Per tick the dataset carries `mode_in` (the mode going into the tick, the
model's `mode` input), `steer_mode` (the mode I wanted: the tick's own, or
the next toggle's for the `MODE_LEAD_S` = 0.5 s before it) and `mode_switch`
(within 0.5 s of a toggle). All three come off the whole row stream before
any tick is dropped, so a toggle on a dropped row still counts. A mode change
across a gap in the rows is not a toggle: manual writes no rows, so a press
there cannot be placed. The `.npz` meta lists every toggle per session.

## Throttle labels across speed caps

A throttle label is a **fraction of its own row's cap**: `throttle /
forward_limit` going forward, `throttle / reverse_limit` in reverse, which is
how far the trigger was pulled. The policy runtime multiplies a magnitude by
the caps the car drives at now, so a label and a model output are the same
unit whatever caps the session was recorded at, and a model can never ask
for more than today's cap.

The same pull is not the same decision at a different cap, though: at 0.2 I
held RT flat because the car barely moved. So `tools/policy_dataset.py`
leaves out, by default and with a `LEFT OUT` line, any session whose caps
differ from `policy_adapters.RUNTIME_CAPS` (today 0.4 / -1.0), and
`--include-other-caps` keeps them with a `WARNING`. That includes every
legacy file and the first record session, all at 0.2 / -0.6, and every
session recorded at 0.4 / -0.296. A session that
recorded no caps at all is kept with a warning; its throttle is the
autopilot's and never a label. The `.npz` meta carries `throttle_unit`,
`runtime_caps` and each session's `session_caps`.

## Which model drove

Learned models are a family name plus MAJOR.MINOR, like `kblock 1.1`. MAJOR
changes when the car's runtime has to know: a new input (IMU, the previous
action), a different representation or input contract, different outputs.
MINOR is a retrain on the same contract: more data, other weights. The
versions live on the Mac in `~/crawler/data/models/` as `kblock-X.Y.onnx` +
`.json`, listed in `registry.json` with each one's sha256, training sessions
(and their MCAPs' sha256), training commit and command, CV scores and parent.
`./scripts/models.sh list | register | deploy | rollback | status` is the tool;
`tools/models.py` says how it works. Nothing in there is ever deleted.

## Tags: clean

The map page's Clean button (`{"cmd": "tags", "clean": true}` on
`/patrol/cmd`) says what a drive is; `crawler/telemetry/tags.py` says how.
It lives in `log/session_tags.json` across restarts. Every MCAP file opens
with a metadata record `crawler.tags` (the tags then), and `/session/tags`
(`crawler.SessionTags`) goes out once a second and on the pass a tag
changes, so a flip halfway through splits the session there. A recording
with none, every one before this, is messy (not clean).

The dataset's `tags` array is each tick's bit mask (clean 2; bit 1 is kept
for an outdoor tag later). `policy_train.py --include-tags A,B`,
`--exclude-tags A,B` and `--tag-weight TAG=W` take clean or messy, and
filter or weight training, validation, the folds and retrain's grading
alike; `policy_scorecard.py --include-tags / --exclude-tags` grade the same
way. `tools/session_tags.py DATA.npz` (or MCAPs) lists each session's tags,
ticks and minutes.

## Getting it out

```sh
uv run --with mcap deploy/footprint_dump.py ~/crawler/data/mcap/crawler-record-*.mcap -o rows.csv
uv run --with mcap --with pyarrow deploy/footprint_dump.py run.mcap -o rows.parquet
uv run --with mcap --with numpy==1.26.4 --with opencv-python-headless \
    deploy/footprint_dump.py run.mcap --small pictures/ --verify
```

The table has one column per number (arrays spread out as
`probe_distance_mm_0` .. `_8`), plus `file` and `log_time_ns`.
`--small DIR` writes `DIR/<recording>.npz` with `depth` `(N, 50, 80)` uint16,
`seq` and `log_time_ns`, in row order. `--depth DIR` writes the 2 Hz raw
frames as `DIR/<log_time_ns>.npy`.

## Training a model

The whole loop in one command, on the laptop:

```sh
./scripts/retrain.sh                # pull, decode the new recordings, train, grade, register
./scripts/retrain.sh --no-pull -- --lidar   # after --: flags for policy_train.py
```

It decodes each recording once into `~/crawler/data/cache` (again only when
the file or the decode code changes), drops the junk sessions and says why,
trains the five held-out folds and the final model at once, grades the
model the car runs on the same folds, and registers the new one as a
candidate with that table in its note. It never deploys. A run with nothing
new takes about 16 s, against about 5 min for the commands below by hand.
`tools/retrain.py`'s docstring has the details.

By hand, in `~/crawler/train` (see the README):

```sh
python tools/policy_dataset.py data.npz           # every ~/crawler/data/mcap/crawler-record-*.mcap
python tools/policy_train.py data.npz models/policy.onnx --val <one session name>
python tools/policy_train.py data.npz models/policy.onnx --blocks 5   # one session: time blocks
python tools/policy_train.py data.npz models/policy.onnx --val <name> --imu --prev   # + IMU, + last decision
python tools/policy_train.py data.npz models/policy.onnx --blocks 5 --imu-no-yaw    # + IMU without the yaw rate
python tools/policy_train.py data.npz models/policy.onnx --blocks 5 --lidar         # + the roof lidar
python tools/policy_train.py data.npz models/policy.onnx --val <name> --imu-no-yaw --imu-tilt   # + which way is up
python tools/policy_train.py data.npz models/policy.onnx --blocks 5 --steer-head direction   # left/straight/right + how far
python tools/policy_eval.py data.npz models/policy.onnx --out report.md

# + my takeovers on the auto drives, weighted 3x
python tools/policy_dataset.py data.npz --drive-logs ~/crawler/data/mcap/drives
python tools/policy_train.py data.npz models/policy.onnx --blocks 5 --takeover-weight 3
python tools/session_report.py --no-jpeg-depth ~/crawler/data/mcap/drives   # how much each drive gave
```

### Learning from my takeovers

`policy_dataset.py --drive-logs DIR` reads every auto drive in DIR (not the
record sessions) alongside them, and tags every tick with where it came
from (the `.npz` `source` array, `meta["sources"]`): `record`, `auto` (no
hand on it, never a label), `takeover`, or `manual_after_auto`. A drive
with no `/footprint` rows is skipped and said so.

- **Takeover**: an auto row where either group is `"pad"`. Only the group I
  had is a label, as always, and it is the command that went out, on
  record's scale: the trigger is scaled to the same caps. Every one of
  those ticks is a decision, reverses included. Not a label: the release
  tail (the at-rest ticks that end a group's run, before it goes back to
  the autopilot). A throttle held off my trigger (`throttle` is not
  `pad_throttle` scaled by the row's caps, because the floor veto held my
  forward: the car did that until `2b38c7d`) is labelled with what I
  asked for, `pad_throttle` on the row's caps, not with the 0 that went
  out.
- **Manual after auto**: manual rows within `MANUAL_AFTER_AUTO_S` (5 s) of
  a switch out of auto, only while the throttle or steering is off rest,
  the throttle measured against the auto caps (manual drives the full
  range). **The car writes no rows in manual today**, so there are none of
  these yet; the dataset report and `session_report.py` say how many
  seconds of that driving there were off `/vehicle`.

`policy_train.py --takeover-weight W` (3 by default) counts an intervention
tick W times in every head's loss and prints the share of each head's loss
weight that comes off them. `--blocks` holds out record ticks only, so the
folds are the record sessions' folds and every intervention trains in each.

The dataset tool prints, per session, every tick it dropped and why, the
not-driving spans (parked, idle, the reach for Y, the car in my hands) as a
one-character-a-second timeline, and what the `/marks` did: a BAD window on
my own driving comes out of training, a GOOD one counts double
(`--good-weight`). Validate on a whole session, never a random tick. With
only one session, `--blocks 5` holds out five contiguous stretches of it in
turn (with a second either side left out of training) and then trains the
model it exports on all of it.

`--steer-head direction` replaces the one tanh steering output with a
left / straight / right class and a magnitude trained only on the turns
(`policy_train.STRAIGHT_BAND`, 0.1 of full lock, is straight). The single
output averages: backing out of a 3-point turn I go hard left or hard right,
and the flip augmentation makes those scenes exactly half and half, so the
answer that minimises its error is the middle. The sidecar says `steer_head`;
one without it is a regression, and `policy.decode_steer` reads both, on the
car and in the eval. A different head is a different contract, so a MAJOR.

`--steer-head per_gear` is a regression per gear, `steer_forward` and
`steer_reverse`, each as wide as the chunk. I swap lock on 95% of my
reversals, so one steer had to flip sign on exactly the tick the gear did,
and a tick late it backed out on the forward lock. Each head trains only on
its own gear's ticks; a stop trains the head of the gear the car moves in
next within `--stop-gear-s` (2 s), else the one before, else forward
(`policy_train.steer_gears`). The car steers with the head of the gear it
commands, read after the refractory and the rear mask have had their say, so
a reverse the rear stop turns forward steers on the forward head, and a stop
on the head of the better of forward and reverse. The steering I put on
standing still stays a label with its own value: that is where the lock for
the next gear goes on.

`--mirror both` puts every batch in as it is and mirrored, side by side (the
frame, the IMU's lateral signs, the scan's bins, the crop's columns, the
goal's bearing, every steer label's sign), instead of `coin`, one or the
other a batch. On the scorecard, three seeds each, `both` came out worse
than `coin` nearly everywhere (side 0.641 against 0.673, mirror_fail 0.440
against 0.398), so goal3 is `coin`.

`--map-act leaky --optimizer adamw` is the map branch that works. With ReLUs
and Adam's L2 decay the branch died in every map model up to goal2: its
gradient starts about the size of the decay, the units go dead in a few
hundred steps, and the decay walks the weights to zero, so the map changed
nothing (blanking the crop moves goal2's steer by exactly 0). Either fix on
its own still died on a seed in three; both together lived on all three.

`tools/scorecard.txt` names drives no model trains on (`--train-on-scorecard`
to override), and `tools/policy_scorecard.py DATA.npz --map CROPS M.onnx ...`
grades models side by side on them: gear, stop and reverse, steer side, the
lock across my reversals, the mirror test, hindsight goals including ones
behind, the first ticks of each takeover, and `map_use`, how far the steer
moves with the crop blanked. Add drives to the list, never take one off.

The trainer prints the model against always-forward, the hand-written rule
and the median throttle, which is the number that says whether it learned
anything, and the mode head against always-stay: its recall on the ticks
around each L3 press, and how many of my presses its latch reproduces. Old
rolling recordings work too with `--labels any`, but all they can teach is a
copy of the autopilot. A group a learned policy drove is never a label, not
even with `--labels any`.

`policy_train.py --model-name kblock` names the family at training time,
never the number, and every sidecar it writes says the commit and command it
came from. A model trained before the steering moved to manual's units says
`steer_scale` 0.6 in its `.json`, and the car refuses it; retrain it.

### How a model with a mode head drives

It picks four-wheel (unified, the rear counter-phase through
`steering_pair`) or two-wheel (split with the rear straight). It starts in
whatever mode the pad is in when auto engages, and only switches when its
head has wanted the other mode at over 0.7 for 0.5 s and the last switch was
2 s ago, so it cannot flap. L3 while it drives sets its mode to the pad's and
restarts the 2 s, so my press sticks. While my stick has the steering my mode
drives, and when I let go it carries on in my mode. A model with no mode head
drives four-wheel whatever the pad says.

On the Pi 4 the column profile model takes 0.35 ms to run and about 4.5 ms
with its input made, against a 100 ms tick.
