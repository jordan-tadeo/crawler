# Training data: what a drive writes down

The learned policy is a small CNN over a downsampled depth image. Every
decision tick the car writes the picture that policy would see and the command
that was actually on the wheels, at one log time, into the MCAP recording.
This file is the contract between the car and the trainer. The code that
holds it is `footprint.py` (the rows), `policy_input.py` (the picture) and
`telemetry.py` (`JSON_SCHEMAS["footprint"]`, `RAW_DEPTH_CHANNELS`).

## Where the data comes from

- **Record mode** is the training drive: in manual, hold **Y for one second**
  (two buzzes) and the car opens the camera and starts a new MCAP of its own.
  It drives exactly like manual except the throttle is scaled: full RT is
  auto's forward speed (`Driver.forward_speed`, 0.2 today), full LT is auto's
  reverse speed (`Driver.reverse_speed`, -0.6). Steering is unchanged. Any
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

Files: a record session is `recordings/crawler-record-<UTC>Z-p<pid>-001.mcap`
(`/home/jt/Documents/py/crawler/recordings/` on the car), one per session,
never pruned by the ring; the rolling recording is `crawler-<UTC>Z-...mcap`.
`./pull-recordings.sh` pulls both.

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
`policy_input.py`; do not reimplement it. `deploy/footprint_dump.py --verify`
checks, frame by frame, that every recorded picture with a raw frame behind it
equals `depth_small(raw)` byte for byte.

Depth frame facts (`oakd.py`): 400P mono pair at 16 fps, depth aligned to the
RGB socket, left-right check on, extended disparity instead of subpixel. So
the nearest valid depth is about 165 mm (closer reads 0), and depth is
quantised to whole disparity levels: about 5 mm steps at 400 mm, 25 mm at
900 mm.

## `/footprint` (JSON, schema `crawler.Footprint`)

| field | type | meaning |
|---|---|---|
| `seq` | int | per process, from 1 |
| `mode` | str | `"record"` or `"auto"` |
| `source` | str | `"tap"` (our own scan: record, or auto during a recover I have a hand in) or `"driver"` (auto: the driver's scan) |
| `frame_width`, `frame_height` | int | depth frame size, 640x400 (0 if unknown) |
| `throttle` | float | **the label.** Throttle that reached the ESC this tick, -1..1 (record: already scaled) |
| `steering_front`, `steering_rear` | float | **the label.** Steering sent to each servo, -0.7..0.7 (the pad's LEFT_X x 70%); `rear` = -`front` in unified mode |
| `steering_mode` | str | `"unified"` or `"split"` (L3 toggles) |
| `steer_by` | str | `"pad"` or `"driver"`: who had the steering when this row's command went out. Always `"pad"` in record |
| `throttle_by` | str | `"pad"` or `"driver"`: same, for the throttle |
| `pad_throttle` | float | record, or auto with `throttle_by` `"pad"`: the trigger before scaling, -1..1. Otherwise 0 |
| `forward_limit`, `reverse_limit` | float | the scale used for `pad_throttle`, e.g. 0.2 / -0.6. -1 when no trigger was driving |
| `imu_accel_x/y/z` | float | newest OAK-D IMU sample at the tick, linear acceleration, m/s^2 |
| `imu_gyro_x/y/z` | float | same sample, rad/s |
| `imu_age_ms` | float | age of that sample at the tick; **-1 and all zeros if there has been none** |
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
after it is that rule's measure. As of `driver.py` at `11cecdd`:

- forward: `clear`
- reverse: `no_return`; `floor <mm>` (nearest blocked probe under
  `REVERSE_FLOOR_MM`, 200 mm); `split`; `spans`; `wide <n>`; `middle` (an exact
  tie); `shallow <ratio>`; `tight <mm>` (the sweep cannot clear the turn side)
- evade: `graze <ratio>` (off an edge block); `dodge <ratio>` (around a
  block in the middle)
- committed only: `holding out`, `evade flipped`, `evade timeout`

Split on the first space to get the rule. The list moves with `driver.py`, so
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

Rows that do not exist: manual mode, and auto's 2.5 s recover manoeuvre (no
scan, no decision) -- unless my hand has a group during it. Then the rows come
at 10 Hz off the car's own view, `source` `"tap"`, `mode` `"auto"`, with a
`/policy/depth_small` picture and no raw frame.

## Getting it out

```sh
uv run --with mcap deploy/footprint_dump.py recordings/crawler-record-*.mcap -o rows.csv
uv run --with mcap --with pyarrow deploy/footprint_dump.py run.mcap -o rows.parquet
uv run --with mcap --with numpy==1.26.4 --with opencv-python-headless \
    deploy/footprint_dump.py run.mcap --small pictures/ --verify
```

The table has one column per number (arrays spread out as
`probe_distance_mm_0` .. `_8`), plus `file` and `log_time_ns`.
`--small DIR` writes `DIR/<recording>.npz` with `depth` `(N, 50, 80)` uint16,
`seq` and `log_time_ns`, in row order. `--depth DIR` writes the 2 Hz raw
frames as `DIR/<log_time_ns>.npy`.
