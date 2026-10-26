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

```sh
uv sync --extra vision      # tensorflow, tensorflow-hub, tflite-runtime
```

Only `PersonFollower.py` -- retired, imported by nothing -- and the `scratch_*`
/ `convert_pb_to_tflite.py` bench scripts want those. They are the largest and
most aarch64-fragile things in the project, so the install that has to come up
on boot does not carry them.

The `dashboard` extra is deliberately empty. `main.py --dashboard` needs PyQt5,
and PyQt5 5.15.x publishes **no `linux_aarch64` wheel at all** -- 5.15.9 ships
x86_64 manylinux, x86_64 macOS, win32 and win_amd64, and nothing else -- so pip
falls back to the sdist and tries to compile Qt bindings on a Pi, which wants
the Qt5 dev headers and sip. It comes from `apt install python3-pyqt5`
instead, which is the whole reason the venv needs `--system-site-packages`.

`opencv-contrib-python` is installed on the car but is not a dependency: every
`cv2` call in the tree is in the base module. Having both installed is a known
way to end up with two native extensions fighting over one `cv2` name.

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
| `/diagnostics` | `crawler.Diagnostics` | 4 Hz | mode, pad connected, auto available, camera state and error, last error, loop rate and worst tick |
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
telemetry, driving as normal` in the log means exactly that.

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
