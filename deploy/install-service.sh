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
  # No display under a boot service, but SDL still needs a video driver to
  # initialise before it will hand over a joystick.
  echo "Environment=SDL_VIDEODRIVER=dummy"
  echo "Environment=PYTHONUNBUFFERED=1"
  echo "ExecStart=${PYTHON} ${REPO_DIR}/main.py --mode manual"
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
  systemctl enable pigpiod.service
fi
systemctl enable crawler.service

echo
echo "Installed:"
echo "  user    ${RUN_USER}:${RUN_GROUP}"
echo "  groups  ${HAVE_GROUPS[*]:-(none)}"
echo "  python  ${PYTHON}"
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
