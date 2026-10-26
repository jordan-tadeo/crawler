#!/bin/bash
# Install the crawler as a systemd service so the car comes up driveable on
# power-on, with no keyboard, screen or login.
#
# Run it on the Pi, from the checkout:
#
#   sudo ./deploy/install-service.sh          # install + enable, don't start
#   sudo ./deploy/install-service.sh --start  # install + enable + start now
#
# Everything it needs is detected rather than hardcoded, because a wrong path
# or a group that does not exist makes systemd refuse to start the unit at all.
#
# One thing this script deliberately does NOT do: install the udev rule the
# OAK-D needs. Auto mode wants
#
#   /etc/udev/rules.d/80-movidius.rules
#   SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"
#
# followed by `sudo udevadm control --reload-rules && sudo udevadm trigger`,
# or dai.Device() just fails for a non-root service user. That is a one-time
# system change and I want to be the one typing it.

set -euo pipefail

START_NOW=0
[[ "${1:-}" == "--start" ]] && START_NOW=1

if [[ $EUID -ne 0 ]]; then
  echo "Needs root: sudo $0 ${1:-}" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(id -un)}"
RUN_GROUP="$(id -gn "$RUN_USER")"

if [[ "$RUN_USER" == "root" ]]; then
  echo "Refusing to install a root-owned service. Run it with sudo as your own user." >&2
  exit 1
fi

# Prefer the project venv; fall back to the system interpreter.
if [[ -x "$REPO_DIR/env/bin/python" ]]; then
  PYTHON="$REPO_DIR/env/bin/python"
elif [[ -x "$REPO_DIR/env/bin/python3" ]]; then
  PYTHON="$REPO_DIR/env/bin/python3"
else
  PYTHON="$(command -v python3)"
  echo "WARNING: no venv at $REPO_DIR/env, falling back to $PYTHON" >&2
fi

if [[ ! -f "$REPO_DIR/main.py" ]]; then
  echo "No main.py in $REPO_DIR -- is this the checkout?" >&2
  exit 1
fi

# Only ask for groups that actually exist on this system. SupplementaryGroups
# naming a missing group is a hard start failure.
#
# `input` is the load-bearing one: it is what lets the service user read
# /dev/input/event*, which is what makes joystick input work with no login
# session. SDL_VIDEODRIVER below is not what does that.
WANT_GROUPS=(input i2c gpio video bluetooth dialout plugdev render)
HAVE_GROUPS=()
for g in "${WANT_GROUPS[@]}"; do
  getent group "$g" >/dev/null 2>&1 && HAVE_GROUPS+=("$g")
done
GROUP_LINE=""
if [[ ${#HAVE_GROUPS[@]} -gt 0 ]]; then
  GROUP_LINE="SupplementaryGroups=${HAVE_GROUPS[*]}"
fi

# pigpio is a hard requirement -- VehicleController raises if the daemon is
# not up -- but only wire the dependency if the unit is actually installed.
PIGPIO_UNIT=""
if systemctl list-unit-files pigpiod.service >/dev/null 2>&1 &&
   systemctl list-unit-files pigpiod.service | grep -q pigpiod; then
  PIGPIO_UNIT="pigpiod.service"
else
  echo "WARNING: pigpiod.service not found. Install it with: sudo apt install pigpio" >&2
fi

# `pigs` is how the stop hook disarms the ESC when the process did not get to
# do it itself, and how the start hook waits for the daemon.
PIGS="$(command -v pigs || true)"
if [[ -z "$PIGS" ]]; then
  echo "WARNING: no pigs binary found. Installing without the ESC disarm hook." >&2
fi

UNIT=/etc/systemd/system/crawler.service
echo "Writing $UNIT"

{
  echo "[Unit]"
  echo "Description=RC rock crawler control"
  echo "After=network-online.target bluetooth.target ${PIGPIO_UNIT}"
  echo "Wants=network-online.target ${PIGPIO_UNIT}"
  echo ""
  echo "[Service]"
  echo "Type=simple"
  echo "User=${RUN_USER}"
  echo "Group=${RUN_GROUP}"
  [[ -n "$GROUP_LINE" ]] && echo "$GROUP_LINE"
  echo "WorkingDirectory=${REPO_DIR}"
  # SDL wants to be told the name of a video driver before it will initialise
  # at all, and there is no display here. It is NOT what makes the joystick
  # readable headless -- group `input` above is. Keeping it because SDL still
  # asks; it is not the magic.
  echo "Environment=SDL_VIDEODRIVER=dummy"
  echo "Environment=PYTHONUNBUFFERED=1"
  if [[ -n "$PIGS" && -n "$PIGPIO_UNIT" ]]; then
    # After=pigpiod only means the unit was started, not that the socket is
    # listening yet. VehicleController raises outright if it cannot connect,
    # and with Restart=on-failure that turns into a restart loop at boot.
    # No shell variables and no command substitution in here on purpose:
    # systemd expands $NAME and ${NAME} inside Exec lines itself, so a loop
    # counter would get eaten before /bin/sh ever saw it. The bound on this
    # wait is TimeoutStartSec, which is what it is for.
    echo "ExecStartPre=/bin/sh -c 'until \"${PIGS}\" t >/dev/null 2>&1; do sleep 0.25; done'"
  fi
  # Quoted, because an unquoted path with a space in it splits into extra
  # arguments and argparse rejects them.
  echo "ExecStart=\"${PYTHON}\" \"${REPO_DIR}/main.py\" --mode manual"
  if [[ -n "$PIGS" ]]; then
    # The last line of defence. If the process was SIGKILLed, or crashed hard,
    # or TimeoutStopSec ran out, nothing in Python got to disarm anything --
    # and pigpio keeps emitting the last pulse width it was handed, so the car
    # is still driving. This zeroes GPIO18 whatever happened. The leading `-`
    # keeps a failure here from marking the unit failed. Unquoted, because
    # the prefix character has to be the first thing on the line and I am not
    # going to find out the hard way whether systemd takes a quote straight
    # after it. /usr/bin/pigs has no spaces in it.
    echo "ExecStopPost=-${PIGS} servo 18 0"
  fi
  echo "Restart=on-failure"
  echo "RestartSec=5"
  # Give main.py time to unwind through vecon.close() so the ESC is left at
  # neutral instead of holding its last pulse width.
  echo "KillSignal=SIGTERM"
  echo "TimeoutStopSec=15"
  echo ""
  echo "[Install]"
  echo "WantedBy=multi-user.target"
} > "$UNIT"

systemctl daemon-reload

if [[ -n "$PIGPIO_UNIT" ]]; then
  # Non-fatal. This used to abort the script under `set -e` before
  # crawler.service was ever enabled, which left the unit written but not
  # wired to boot -- so the car came up dead and the install looked like it
  # had worked.
  if ! systemctl enable pigpiod.service; then
    echo "WARNING: could not enable pigpiod.service. Carrying on." >&2
  fi
fi
systemctl enable crawler.service

echo
echo "Installed:"
echo "  user    ${RUN_USER}:${RUN_GROUP}"
echo "  groups  ${HAVE_GROUPS[*]:-(none)}"
echo "  python  ${PYTHON}"
echo "  pigs    ${PIGS:-(none)}"
echo "  repo    ${REPO_DIR}"
echo

if [[ $START_NOW -eq 1 ]]; then
  systemctl restart crawler.service
  sleep 3
  systemctl --no-pager --full status crawler.service || true
else
  echo "Enabled for next boot, not started. To start now:"
  echo "  sudo systemctl start crawler.service"
  echo "  journalctl -u crawler -f"
fi
