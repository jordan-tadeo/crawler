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

## Layout

```
crawler/            the code that runs on the car, one package
  __main__.py       the app: `python -m crawler`
  hardware/         ESC and servos, the pad, the OAK-D, the camera hub
  autopilot/        the rules driver, the learned policy and its inputs, marks
  telemetry/        Foxglove channels, MCAP recording, the log, the dashboard
tools/              laptop tools: dataset, training, eval, the models registry
scripts/            run.sh, models.sh, pull-recordings.sh; scratch/ for one-off checks
deploy/             the systemd unit and its installer, the pad timeout, car-side tools
docs/               TRAINING_DATA.md, the recording schema
tests/              pytest, runs on the laptop with no hardware
```

Models (`models/`), logs (`log/`) and recordings never go in the repo. On the
car they sit at the checkout root, untracked.

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
  where `--nav` takes goals from. The lidar panel follows the truck, front up.
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
2. **Pull**, on the Mac. Recordings go to `~/crawler-data/mcap`, never into a
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
       python tools/session_report.py ~/crawler-data/mcap/crawler-record-*.mcap
   ```

3. **Train**, in its own venv. Torch never goes near the car or
   `pyproject.toml`:

   ```sh
   python3.11 -m venv ~/crawler-train && ~/crawler-train/bin/pip install -r tools/requirements-train.txt
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

Models live on the Mac in `~/crawler-data/models`, listed in `registry.json`.
A new model is a candidate with no number. `promote` gives it the next
`kblock X.Y` only when it beats the latest release on held-out folds and I say
it drove well. `deploy` and `rollback` refuse unless the car is parked.

```sh
./scripts/models.sh list                      # releases, candidates, * = on the car
./scripts/models.sh register ~/crawler-data/models/policy-<stamp>.onnx --tag imu-gyro
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
