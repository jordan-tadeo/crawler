# crawler

RC rock crawler control for a Raspberry Pi 4. It boots into hand driving off a
Bluetooth pad; START hands the car to the OAK-D for auto mode.

The car is a Pi 4 on Raspberry Pi OS Bookworm 64-bit, aarch64, Python 3.11.2.
That is the only machine this actually drives on. A laptop checkout is for
editing and for importing the controller without any hardware attached.

## Setup

Dependencies live in `pyproject.toml` and are locked in `uv.lock`. There is no
`requirements.txt` any more.

[uv](https://docs.astral.sh/uv/) does the installing:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Two things on the car come from apt rather than pip, so the environment has to
be built to see them:

```sh
sudo apt install pigpio python3-pigpio python3-pyqt5
sudo systemctl enable --now pigpiod

uv venv --python /usr/bin/python3 --system-site-packages
uv sync
```

Both of those flags are load-bearing:

- `--python /usr/bin/python3` keeps uv off its own downloaded CPython. apt's
  PyQt5 is built for the system 3.11 and installs into
  `/usr/lib/python3/dist-packages`; a uv-managed interpreter never looks there.
  `requires-python` is pinned to `==3.11.*` for the same reason.
- `--system-site-packages` is what lets the venv see `dist-packages` at all.
  Without it the dashboard has no PyQt5.

`pigpio` itself is in `pyproject.toml` and comes from PyPI, so the lock covers
the ESC driver instead of leaving it to whatever apt happens to have. The apt
`python3-pigpio` is the same 1.78 and is a fine fallback; the pip one shadows
it inside the venv.

### Extras

There was a `vision` extra here for TensorFlow. It is gone, and so are the four
files that were its only consumers: `PersonFollower.py`, `USBCamera.py`,
`scratch_persondetect.py` and `convert_pb_to_tflite.py`, plus the 45 MiB of
SavedModel and `.tflite` weights they loaded.

That whole branch of the project was a whole-image classifier on a USB webcam,
aimed at a pan/tilt gimbal that is not bolted to the car any more. Vision is the
OAK-D now: `oakd.py` and `camera.py` take stereo depth off the camera's own chip
and `driver.py` crawls on it. Nothing left in the tree imports TensorFlow.

The `dashboard` extra is deliberately empty. `main.py --dashboard` needs PyQt5,
and PyQt5 5.15.x publishes **no `linux_aarch64` wheel at all** -- 5.15.9 ships
x86_64 manylinux, x86_64 macOS, win32 and win_amd64, and nothing else -- so pip
falls back to the sdist and tries to compile Qt bindings on a Pi, which wants
the Qt5 dev headers and sip. It comes from `apt install python3-pyqt5`
instead, which is the whole reason the venv needs `--system-site-packages`.

`opencv-contrib-python` is installed on the car but is not a dependency: every
`cv2` call in the tree is in the base module. Having both installed is a known
way to end up with two native extensions fighting over one `cv2` name.

## Tests

```sh
uv run pytest
```

Runs on a laptop in under a second. Nothing in the suite wants pigpio talking
to a daemon, a pad paired, or the OAK-D plugged in -- which is the point. These
cover the arithmetic and the gating that had never run anywhere but on the car,
and every one of them is aimed at something that has already gone wrong:

- **`tests/test_driver.py`** -- the stuck check. The journal used to show
  "Switched to auto" and "STUCK - backing up to recover." on the same second,
  then STUCK again every five seconds on a car that was driving fine. Three
  faults made that: a missing IMU packet filed as "not moving", no check that
  the car had even been asked to move, and `last_recovery_time` stamped on
  every false positive. One test per fault, plus the fail-closed depth checks.
- **`tests/test_vehicle_controller.py`** -- pulse widths and servo angles,
  including that half throttle is still 1787 us, and the nan/inf guard.
  `min()` and `max()` do not clamp a nan, so a nan throttle used to come out of
  the clamp as a clean 1.0 and reach the ESC as 2000 us, full forward.
- **`tests/test_joystick.py`** -- the axis and button map, which was wrong in a
  way that held the rear steering servo at full lock, and the trigger scaling,
  which turned an unreported axis reading of 0.0 into half throttle.
- **`tests/test_logger.py`** -- that `log/` makes itself. It is gitignored, so
  it is not in a fresh clone, and `open(..., 'a')` does not create a missing
  parent -- which with `Restart=on-failure` was a restart loop a few seconds
  into every run.

`conftest.py` at the root is what lets the tests `import driver` the same way
`main.py` does. There is no `src/` layout and no installed package: the flat
module names are load-bearing in `main.py` and in the systemd unit.

## Driving it

```sh
./run.sh                    # hand driving, in a shell, with the log on screen
```

`run.sh` and the installer both look for `env/bin/python` first, then
`.venv/bin/python`, then system `python3`. `env` is the old hand-rolled venv
that the installed service still points at, so on the car it keeps winning
until it is deleted -- re-running the installer never quietly moves a working
service onto a different interpreter. `CRAWLER_PYTHON=... ./run.sh`, or
`--python PATH` on the installer, overrides that.

## Service

```sh
./deploy/install-service.sh --print-unit        # show the unit, write nothing
sudo ./deploy/install-service.sh                # install + enable for boot
sudo ./deploy/install-service.sh --start        # install + enable + start now
journalctl -u crawler -f
```

The unit is `deploy/crawler.service.template`, with `@PLACEHOLDERS@` the
installer fills from the machine it is run on: interpreter, checkout path,
service user and group, the supplementary groups that actually exist, the
`pigs` binary and the `pigpiod` unit. Three lines drop out entirely when what
they need is missing. `--print-unit` renders it without root and without
touching `/etc`, which is how I check it before it goes anywhere.

Auto mode also needs a udev rule for the OAK-D, which the installer does not
write -- see the comment at the top of it.

## Watching it in Foxglove

The car serves the Foxglove WebSocket protocol on port 8765 while it runs.
[Foxglove](https://foxglove.dev) on the laptop opens it directly: **Open
connection -> Foxglove WebSocket ->**

```
ws://192.168.68.56:8765
```

On a Mac that is `brew install --cask foxglove`, and then `open -a Foxglove`
-- the cask installs an app, not a command.

Seven channels. The four JSON ones carry a named schema so Studio labels them:

| topic | schema | rate | what it is |
| --- | --- | --- | --- |
| `/controller` | `crawler.Controller` | 20 Hz | every axis and button by raw SDL index, the named view through `Joystick`'s own maps, the calibrated trigger rest, and the throttle and steering the code derived |
| `/vehicle` | `crawler.Vehicle` | 20 Hz | mode, commanded throttle and steering, the ESC pulse width that was written, and the pigpio read-back with its age |
| `/diagnostics` | `crawler.Diagnostics` | 4 Hz | mode, pad connected, auto available, camera state and error, last error, loop rate and worst tick, and whether this run is being recorded -- which file, how big, and why not if not |
| `/imu` | `crawler.Imu` | 20 Hz | the OAK-D's accelerometer and gyro, downsampled from 100 Hz |
| `/camera/depth` | `foxglove.CompressedImage` | 8 Hz | depth, colourmapped over a fixed 300-3000 mm window |
| `/camera/disparity` | `foxglove.CompressedImage` | 8 Hz | disparity, per-frame stretch |
| `/camera/rectified_left` | `foxglove.CompressedImage` | 2 Hz | the actual mono picture |

The image channels use a well-known Foxglove schema, so the Image panel finds
them on its own with no layout to build. There is no well-known Foxglove IMU
schema -- the SDK ships forty-odd and none of them is an IMU -- so `/imu` is
JSON and the panel for it is Plot.

```sh
./run.sh                             # telemetry on, no camera in manual
./run.sh --telemetry-camera          # also open the OAK-D, so the camera
                                     #   channels work while hand driving
./run.sh --no-telemetry              # no server, no thread, no port bound
./run.sh --telemetry-port 9000       # somewhere else
```

Telemetry is on by default and `--telemetry-camera` is off by default. The
camera one costs a USB enumeration and about a watt on every boot, and the
mode that has to always work does not need a picture. In auto the camera is
up anyway, so the camera channels flow either way.

None of it can delay a throttle command. The control loop calls `wants()`,
which is a couple of dict lookups, and `offer()`, which drops a value into a
one-deep mailbox and returns; the JSON, the colourmapping, the JPEG and the
socket writes are all on the telemetry thread. Every channel is gated on the
server's own subscriber count, so with Foxglove shut nothing is built and
nothing is encoded. Measured on the car with a client subscribed to all seven
and the camera running: p50 23 us, p99 147 us, worst 720 us, against a 10 ms
tick.

And it is never fatal. A missing `foxglove-sdk`, a port already bound, a
laptop that closed mid-frame -- all logged and swallowed. `Not starting
telemetry, driving as normal` in the log means exactly that. A port that
cannot be bound now costs the live view only: the channels are built before
the socket, so the recording below carries on without it.

### Recording it

Every run is written to `log/mcap/`, from service start, whether or not
Foxglove is open. That is the whole point -- the run worth looking at is
always the one that just ended badly, and by then the laptop is shut.

```sh
./pull-recordings.sh                 # newest 5 onto the laptop, checked
./pull-recordings.sh --list          # what the car is holding, copy nothing
./pull-recordings.sh --all
./pull-recordings.sh -n 20 -o ~/Desktop/crawl
```

Then `open -a Foxglove` and drag a file from `recordings/` onto the window.
**Open connection** is for live; a file is a drag, or **File -> Open local
file**. The same panels work on a file as on the socket.

`pull-recordings.sh` also says what is in each file -- per topic, the message
count, the rate and the schema -- which answers "did that run record what I
think it did" without opening anything. A run with no `/camera/depth` in it
was a run with the camera shut, not a bug.

```sh
./run.sh --no-record                 # drive, do not write it down
./run.sh --record-dir /mnt/usb/runs  # somewhere other than log/mcap
./run.sh --no-telemetry              # record, but bind no socket
./run.sh --no-telemetry --no-record  # no thread, no socket, no file
```

The filenames are `crawler-<UTC>Z-p<pid>-<NNN>.mcap`. The pid is in there
because this Pi has no RTC: at boot the clock is whatever the last shutdown
wrote until NTP lands, so two boots genuinely can produce the same second.

#### What it costs

An MCAP writer in this SDK is a *sink*, exactly like a connected Studio
client, so opening one makes `has_sinks()` true and the existing
`wants()`/`offer()` path does the rest. There is no second encode and no
second thread. The consequence is that "nothing is built when nobody is
watching" now means **no laptop and no recording**: with recording on, the
publisher thread colourmaps and JPEG-encodes the camera channels whenever the
camera is up.

Measured on the car, 120 s a condition, a 100 Hz replica of the control
loop's telemetry block, camera up, nothing connected to the socket:

| | p50 | p90 | p99 | max | process CPU |
| --- | --- | --- | --- | --- | --- |
| recording off | 9 us | 13 us | 25 us | 109 us | 16.0% of a core |
| recording on | 11 us | 190 us | 268 us | 831 us | 33.8% of a core |

Camera shut, which is how the service actually boots: 7/8/10/54 us and 2.0%
off, 9/120/151/240 us and 3.0% on.

So about 240 us at p99 against a 10 ms tick, which is 2.4% of it, and the
same order as the p99 147 us already measured for having Studio connected --
for the same reason. The p90 jump is not `offer()` getting slower, it is the
GIL coming back from a publisher thread that is now busy in zstd and cv2. The
18 points of extra CPU is that thread, on a 4-core Pi, so about 4.5% of the
machine, and none of it on the thread that commands the throttle.

#### What it costs on the card

Measured by reading a finished file back over its own span. The camera-up
number is off a real 469 s run with the car being driven, which the service
recorded by itself while I was not watching:

| | bytes/s | per hour |
| --- | --- | --- |
| camera up, all seven channels | 184.9 KiB/s | **650 MiB** |
| camera shut, the three JSON channels | 1.2 KiB/s | **4 MiB** |

The camera is 90% of that and it is the part zstd cannot help with -- the
payload is already JPEG. Whole-file compression came out at 0.87 of the raw
message bytes, the JSON half compressing well and the images not at all. Per
channel: disparity 117.3 KiB/s, depth 60.4, rectified_left 13.9, controller
12.2, vehicle 5.0, imu 3.2, diagnostics 1.3.

**Do not size this off a bench run.** A static scene on a desk measured 71
KiB/s, which is 2.6x low, and the whole gap is one channel:
`/camera/disparity` went from 5 098 bytes a frame to 17 053. It gets a
min/max stretch, so a real driving scene is high-entropy noise and JPEGs
three times larger than a flat close-up of nothing.

A full SD card is a dead car, and worse than dead: a Pi that cannot write is
a Pi whose filesystem corrupts, and the only thing that can take the ESC back
to neutral is this process staying alive. So there are two brakes and they do
not trust each other:

- a new segment every **128 MiB** (about twelve minutes of auto driving, a
  20-second scp), and the oldest segments deleted once `log/mcap/` passes
  **8 GiB** -- 64 segments, 12.6 hours of auto driving, 8% of this card.
  That is a ring buffer at the file level, which is the only shape
  available: this SDK's writer has no ring mode and building one would mean
  re-encoding every message to drop the old ones. `MAX_TOTAL_BYTES` in
  `recording.py` is the one line to change if I want more than 12 hours.
- under **4 GiB free on the volume**, recording stops dead and says so once
  -- even if `log/mcap/` is well inside its own budget, because something
  else filling the card is not something pruning my own directory can fix.

Nothing in `recording.py` can reach the hardware. A disk-full, a write error,
a read-only filesystem, a missing SDK: `start()` returns False, `tick()`
returns, `stop()` returns, and the car drives exactly as it did. `tests/
test_recording.py` pins that, because it is the requirement that matters more
than the rest put together.

#### Two things I had to measure rather than assume

**zstd buffers a whole chunk before writing anything.** Twenty seconds into a
recording the file on the card held 128 bytes, and 24 003 after close -- all
of it still in memory. A run that ends with the battery out never reaches
close(). So the recorder flushes every five seconds, which bounds that to
five seconds of loss. With the camera up, five seconds is already about a
megabyte, so the chunking is unchanged.

**A run that is killed loses its index, not its data.** Only `close()` writes
the summary and the message index. On a recording killed with `SIGKILL`,
every message was still there and streamed back fine with its schemas, but
the *indexed* reader -- the fast path Studio takes -- refused the file
outright, and `disable_seeking` makes no difference. `pull-recordings.sh`
detects that and rewrites the file with an index, which is the `mcap recover`
operation done through the `mcap` Python library so there is no Go toolchain
to install; `uv run --with mcap` builds the environment for it and leaves
nothing behind. The original is kept as `.orig`.

The newest file is normally one of the ones it repairs, and that is not the
car crashing: it is the run happening right now, still open, with its index
not written yet. Pulling it gives a snapshot up to the last flush, which is
at most five seconds behind.

### Two things that have to be true on the car

The OAK-D needs its udev rule or only root can see it -- `depthai` reports
`Cannot find any device with given deviceInfo` to everyone else, which is the
same thing auto mode has always hit:

```sh
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' \
  | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

And the interpreter the service runs has to have the SDK in it. The installed
unit still points at the old `env/`, which `uv sync` does not touch:

```sh
env/bin/pip install foxglove-sdk==0.30.2        # keep the service on env/
# or move the service onto the uv venv instead:
sudo ./deploy/install-service.sh --python "$PWD/.venv/bin/python" --start
```
