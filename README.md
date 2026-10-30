# crawler

Control for my RC rock crawler. It boots into hand driving off a Bluetooth
pad, and START hands the car to the OAK-D for auto mode: either the
hand-written rules or a small learned policy. Every run is recorded, so a
drive can be read back afterwards instead of remembered.

## Hardware

- Raspberry Pi 4, Raspberry Pi OS Bookworm 64-bit, Python 3.11. This is the
  only machine that drives; a laptop checkout is for editing, tests and
  training.
- Quicrun 880 ESC on GPIO 18 through pigpio. Neutral is 1575 us.
- Front and rear steering servos on a PCA9685 over I2C.
- OAK-D: stereo depth and the IMU, computed on the camera.
- LDRobot STL-19P 2D lidar (the D500 kit) on the roof, arrow to the truck's
  right. Read and recorded with `--lidar`; nothing drives on it yet.
- An Xbox pad over Bluetooth, driven by xpadneo.
- A USB webcam (Suyin "HD Camera", 1e45:8022) on the hood, on two pan/tilt
  servos on the same PCA9685. `--pantilt` turns it to follow a person; it
  never drives.

## Layout

```
crawler/            the code that runs on the car, one package
  __main__.py       the app: `python -m crawler`
  hardware/         ESC and servos, the pad, the OAK-D, the camera hub
  autopilot/        the rules driver, the learned policy and its inputs, marks
  vision/           the hood camera's person detector and pan/tilt tracker
  telemetry/        Foxglove channels, MCAP recording, the log, the dashboard
tools/              laptop tools: dataset, training, eval, the models registry
scripts/            run.sh, models.sh, pull-recordings.sh; scratch/ for one-off checks
deploy/             the systemd unit and its installer, the pad timeout, car-side tools
docs/               TRAINING_DATA.md, the recording schema
tests/              pytest, runs on the laptop with no hardware
```

Models (`models/`), logs (`log/`) and recordings never go in the repo. On the
car they sit at the checkout root, untracked.

### Where things go

- Laptop: one folder, `~/crawler/`, holding `repo/` (this checkout), `data/`
  (recordings, models, cache, retrain runs), `train/` (the training venv) and
  `worktrees/<name>`. `tools/data_paths.py` finds `data/` and `train/` beside
  the checkout; `CRAWLER_DATA` and `CRAWLER_TRAIN` override them. A new
  worktree goes in `~/crawler/worktrees/<name>` and comes out
  (`git worktree remove`, `git branch -d`) once it is merged.
- Car: the live checkout stays at `~/Documents/py/crawler`. Any bench, scratch
  copy or unit backup goes under `~/crawler-misc/`, never loose in `~`.

## Setup

Dependencies are in `pyproject.toml`, locked in `uv.lock`. On the car, pigpio
and PyQt5 come from apt, so the venv is built on the system Python and can see
its packages:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt install pigpio python3-pigpio python3-pyqt5
sudo systemctl enable --now pigpiod

uv venv --python /usr/bin/python3 --system-site-packages
uv sync
```

`uv sync` also installs `crawler/` into `.venv` as editable. The car does not
rely on that: run from the checkout root, `python -m crawler` finds the package
in the working directory.

PyQt5 (only for `--dashboard`) has no aarch64 wheel, which is why it comes from
apt and why the venv needs `--system-site-packages`. The comments in
`pyproject.toml` say why each pin is a pin.

## Running it on the car

The service starts the car at boot:

```sh
./deploy/install-service.sh --print-unit        # show the unit, write nothing
sudo ./deploy/install-service.sh --start        # install, enable, start now
journalctl -u crawler -f
```

The unit is `deploy/crawler.service.template`. It runs `python -m crawler
--mode manual` with the checkout as its working directory. The installer fills
in the interpreter, paths, user and groups from the machine it runs on. To
drive a learned model, a drop-in adds `--policy models/policy.onnx`:

```sh
sudo systemctl edit crawler     # ExecStart= then ExecStart=<python> -m crawler --mode manual --policy models/policy.onnx
```

In a shell instead, with the log on screen:

```sh
./scripts/run.sh                         # manual, telemetry on
./scripts/run.sh --telemetry-camera      # also open the OAK-D in manual
./scripts/run.sh --no-telemetry --no-record
./scripts/run.sh --policy models/policy.onnx
```

`run.sh` and the installer pick `env/bin/python` (the old hand-made venv) if it
exists, then `.venv/bin/python`. `CRAWLER_PYTHON=...` or `--python PATH`
overrides that.

The pad's Bluetooth link timeout is cut to 5 s on every connection, so a pad
that dies quietly puts the car at neutral in about 6 s instead of 20:

```sh
sudo ./deploy/install-pad-lst.sh                # once
journalctl -u 'crawler-pad-lst@*'               # one line per connection
```

The OAK-D's udev rule is already on the car and the installer does not touch
it. The comment at the top of `deploy/install-service.sh` says what it is.

## The roof lidar

The STL-19P only transmits: 230400 8N1, no commands, streaming as soon as its
motor is up to speed. Its 4-pin lead goes to the kit's USB-UART board, and
the board to a USB port on the Pi. The board is a CP2102: it shows up as
`/dev/ttyUSB0` with a `/dev/serial/by-id/*CP2102*` link, and bare `--lidar`
finds it by that name. On the Pi's header pins instead it is sensor 1 Tx to GPIO 15
(RXD), 3 GND to a ground, 4 5V to a 5V pin, 2 PWM to ground, with the serial
console off and the UART on in `raspi-config`, and then
`--lidar /dev/serial0`. The Tx line is 3.3 V, so either way is safe for the
Pi. The service user needs group `dialout` for the tty.

Bring-up, with the service left as it is (it does not read the lidar):

```sh
./.venv/bin/python tools/lidar_check.py --list         # the ports auto tries
./.venv/bin/python tools/lidar_check.py --seconds 20   # then hold a hand at the front, then the right
```

Expect about 600 rpm, 417 packets/s, 0% CRC errors and about 480 points a
scan. The mount has been checked once already with a box off the front
bumper: it showed up at raw sensor angle 270, which `MOUNT` puts straight
ahead. The near number has to land under `front`, then `right`. If it lands a
quarter turn off, `--yaw` tries another mount yaw, and `MOUNT` in
`crawler/hardware/lidar.py` is where it goes once it is right. The sensor
sits on the cab roof on the centre line, which is the frame's origin, so its offset is
0, 0. Then add `--lidar` to the
service's ExecStart with `sudo systemctl edit crawler`, and `/lidar` is in
every recording.

## The hood camera

`--pantilt` points the hood camera at the nearest person and keeps them in
the middle of the picture. It never commands throttle or steering: the only
thing it moves is the mount, and it refuses the steering channels. The
detector is a YOLO model on onnxruntime in a process of its own, at a rate
that backs off when the Pi is busy (`crawler/vision/person.py` says why this
one, and what it should cost). `/pantilt` has the angles and the person's
bearing off the truck, `/pantilt/image` the picture with the boxes on, and
the drive layout shows it under the OAK-D's three.

Once, on the Mac, export the model and copy it over (it is not in the repo):

```sh
uvx --python 3.11 --from 'ultralytics==8.3.40' --with onnx --with onnxslim \
    --with 'torch<2.6' --with 'torchvision<0.21' \
    yolo export model=yolo11n.pt format=onnx imgsz=256,320 opset=12
scp yolo11n.onnx jt@jtpi.local:Documents/py/crawler/models/person.onnx
```

On the car, find the servos' centre, limits and direction. It writes only
the two pan/tilt channels and is safe with the service running:

```sh
./.venv/bin/python tools/pantilt_cal.py --pan-channel 10 --tilt-channel 11
```

`save` writes `models/pantilt.json`, which is untracked like the model, so
the checkout stays clean for the next pull. Then add
`--pantilt models/pantilt.json` to the service's ExecStart with
`sudo systemctl edit crawler`.

## Modes and controls

| control | what it does |
| --- | --- |
| RT / LT | throttle forward / reverse |
| left stick | steering, both axles (unified mode) |
| right stick | rear axle, in split mode |
| L3 | toggle unified / split steering |
| START | manual and auto |
| hold Y 1 s | record mode (manual only); any press of Y leaves it |
| A / B | mark the last few seconds good / bad |

- **Manual** is hand driving.
- **Auto** drives on depth. With no `--policy` it is the rules in
  `crawler/autopilot/driver.py`. With `--policy MODEL.onnx` the model decides
  forward, reverse or stop, a throttle and a steer every 100 ms, behind the
  same forward-floor veto and stuck detector. A frame that is missing, stale
  or blind is neutral, and so is any output that is not finite. A model that
  will not load means auto is unavailable; it never falls back to the rules
  quietly. My stick or trigger takes over from either one.
- **Record** drives like manual at auto's speed caps (full RT is auto's
  forward speed) and writes every decision down as training data. START ends
  the session and goes to auto.
- Coming back to manual, the throttle stays at 0 until both triggers have been
  released once.

A mark is a window, 4 s back and 1 s ahead of the press. It goes to the MCAP
on `/marks`, to the journal, and to `log/marks.csv`. The pad buzzes once for
A and twice for B.

## Watching and recording

The car serves Foxglove on port 8765. In Foxglove: Open connection, Foxglove
WebSocket, `ws://jtpi.local:8765`. Use the name, not an address: the DHCP lease
moves. If mDNS fails, `ssh jt@jtpi.local 'hostname -I'` gives the address.

There are two layouts in `deploy/`. In Foxglove: Layouts, Import from file,
once each, then switch between them from the Layouts menu.

- `foxglove-drive.json`: the three cameras down the left, the SLAM map in the
  middle and the live lidar on the right. The map panel's display frame is
  `map`, so the map holds still and the truck outline moves across it on the
  SLAM pose (`map -> base_link` on `/tf`), with the scan drawn on the walls.
  `/voxel/points` and `/voxel/low` are in its topic list, off until ticked.
  Its publish tool is set to a 2D pose on `/move_base_simple/goal`, which is
  where `--goal` takes goals from: pick Publish pose in the map panel's
  toolbar, press where the truck should go, drag, let go. A green disc on
  `/goal/marker` marks it, and the Goal panel (`/goal/status`) says how far
  and which way it is from the truck and whether it has arrived. A click is
  taken in any mode and kept until the next one; a goal model drives to it
  whenever auto drives, and within 0.25 m auto holds neutral until I click
  somewhere else. The lidar panel follows the truck, front up.

  Patrols, also `--goal`. The buttons under the Goal and Patrol panels
  publish on `/patrol/cmd`. **Edit patrol**, then every Publish pose click is
  the next waypoint, numbered on the map in blue and joined in order;
  **Undo** drops the last, **Clear** all of them. **Trace** on, drive the
  route by hand, Trace off, and it has dropped a waypoint every metre.
  **Mode** goes once (stop at the last), loop (laps) and patrol (there and
  back) in turn, and **Start** / **Pause** drive it. The model drives to the
  amber one through the same goal input as a click, and the next one is the
  goal as soon as the truck is within 0.5 m. A waypoint it gets no closer to
  for 20 s, or does not reach in 120 s, is skipped with a journal line.
  **Single goal** goes back to one click being the goal, the list kept. The
  list is saved to `log/patrol.json`, and after a restart it only comes back
  if the new map's walls line up with the old one's: every boot starts a
  fresh map, so that means the truck booted where it did last time.
  Otherwise it is cleared and the journal says so. `autopilot/patrol.py` has
  the rules.

  The SLAM map is saved to `log/slam_map.npz` every 10 s as a record of the
  run, and at boot the last run's file is moved to `log/slam_map.prev.npz`.
  Nothing loads it unless `--slam-map-restore` is given: the truck may boot
  somewhere else, and a map with a wrong heading written into it came back
  at every boot. `--no-slam-map-file` keeps nothing.

  The same buttons, and Follow, are on a phone page the car serves on port
  8080 with `--goal`: `http://jtpi.local:8080` (or `10.42.0.1:8080` on the
  car's hotspot). Nothing to install, and nothing on it loads from the
  internet. It sends the same messages over the same WebSocket, shows the
  patrol and follow state, and **Clear** wants a hold. Waypoints are still
  placed in Foxglove's map. `--buttons-port 0` turns it off.
- `foxglove-telemetry.json`: commands, ESC, pad axes, IMU, loop health, SLAM
  fit and cost, the mode and SLAM state strips, and `/footprint` and
  `/diagnostics` raw.

Pictures, lidar and the map only come while a pad is connected.

Every run is written to `log/mcap/` from service start, in 128 MiB segments,
with the oldest deleted past 8 GiB and recording stopped under 4 GiB free.
Record sessions get their own `crawler-record-*.mcap` files and are never
pruned. Nothing in telemetry or recording can delay a throttle command or stop
the car if it fails.

## Record, pull, train, deploy

1. **Record.** Hold Y in manual and drive.
2. **Pull**, on the Mac. Recordings go to `~/crawler/data/mcap`, never into a
   git tree (the repo is public and a recording is video of my house):

   ```sh
   ./scripts/pull-recordings.sh                  # everything new, checked and repaired
   ./scripts/pull-recordings.sh --records-only   # only the training sessions
   ./scripts/pull-recordings.sh --list           # what the car holds
   ./scripts/pull-recordings.sh --prune --yes    # delete car copies that are safe here
   ```

   Is a drive any good as training data:

   ```sh
   uv run --with mcap --with numpy==1.26.4 --with opencv-python-headless \
       python tools/session_report.py ~/crawler/data/mcap/crawler-record-*.mcap
   ```

   Which model gets round fastest with the least tipping:

   ```sh
   ~/crawler/train/bin/python tools/lap_report.py ~/crawler/data/mcap/crawler-20261030-*.mcap
   ~/crawler/train/bin/python tools/lap_report.py --segments --json laps.json <files>
   ```

   A lap is every segment of one loop (there and back counts both ways) on
   one track, with no break in the chain. A clean lap or segment had no
   takeover, skip, tip or mapless fallback. The table per model gives the
   median clean lap and clean seconds per metre of leg, and skips,
   takeovers, tips and tilt blocks per lap and per 100 m, each with a 95%
   bootstrap over laps. Lower is better on all of them. The track id
   (`name@base+edits`) changes on every edit of the list, so laps from
   before and after an edit are never compared. Recordings from before
   `/patrol/event` are read off the 2 Hz status to +-0.25 s per pass, and
   the report says so.

3. **Train**, in its own venv. Torch never goes near the car or
   `pyproject.toml`:

   ```sh
   python3.11 -m venv ~/crawler/train && ~/crawler/train/bin/pip install -r tools/requirements-train.txt
   python tools/policy_dataset.py data.npz
   python tools/policy_train.py data.npz models/policy.onnx --val <session>
   python tools/policy_train.py data.npz models/policy.onnx --blocks 5 --imu-no-yaw --imu-tilt
   python tools/policy_eval.py data.npz models/policy.onnx --out report.md
   ```

   `docs/TRAINING_DATA.md` has the recording schema and what the dataset tool
   and trainer report.

4. **Deploy** through the models registry, below.

Whether a model drives well is something I find out by driving behind it with
a hand on START. The numbers only say it agrees with me on recorded frames.

## Models registry

Models live on the Mac in `~/crawler/data/models`, listed in `registry.json`.
A new model is a candidate with no number. `promote` gives it the next
`kblock X.Y` only when it beats the latest release on held-out folds and I say
it drove well. `deploy` and `rollback` refuse unless the car is parked.

```sh
./scripts/models.sh list                      # releases, candidates, * = on the car
./scripts/models.sh register ~/crawler/data/models/policy-<stamp>.onnx --tag imu-gyro
./scripts/models.sh deploy candidate-imu-gyro --dry-run   # then without --dry-run
./scripts/models.sh promote imu-gyro --evidence ab.json --approved-on-car --notes "..."
./scripts/models.sh rollback                  # back to what drove before
./scripts/models.sh status                    # the car's link, checked by sha256
./scripts/models.sh note 1.2 "..."            # a dated drive note
```

`deploy` copies the files into the car's `models/`, re-points the
`models/policy.onnx` and `.json` symlinks, restarts the service and checks it
came back. The docstring in `tools/models.py` has the rules.

## Tests

```sh
uv run pytest
```

A few seconds on the laptop. Nothing needs pigpio, a pad or the OAK-D: the
hardware is faked in `tests/conftest.py`. Most tests pin something that has
already gone wrong on the car once.
