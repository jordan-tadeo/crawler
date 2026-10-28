#!/bin/bash
# Install the crawler as a systemd service so the car comes up driveable on
# power-on, with no keyboard, screen or login.
#
# Run it on the Pi, from the checkout:
#
#   sudo ./deploy/install-service.sh                  # install + enable, don't start
#   sudo ./deploy/install-service.sh --start          # install + enable + start now
#   ./deploy/install-service.sh --print-unit          # show the unit, touch nothing
#   sudo ./deploy/install-service.sh --python PATH    # name the interpreter myself
#
# The unit itself lives in deploy/crawler.service.template now, as a file I can
# read, instead of forty `echo` lines in here. This script's whole job is to
# work out what goes in the placeholders and fill them in.
#
# Everything it needs is detected rather than hardcoded, because a wrong path
# or a group that does not exist makes systemd refuse to start the unit at all.
# --print-unit is how I check the result before anything gets written.
#
# One thing this script deliberately does NOT do: install the udev rule the
# OAK-D needs. It does not have to. The rule is already on the car and has been
# since 2025-06-06. This is what is actually there, read off the Pi 2026-10-26:
#
#   /etc/udev/rules.d/99-luxonis-usb.rules
#   SUBSYSTEM=="usb", ATTR{idVendor}=="03e7", ATTR{idProduct}=="2485", MODE="0666"
#   ...and one more line each for f63b, f63d, f63f, f640 through f644
#
# Covering every one of those product IDs is the part that matters: the OAK-D
# changes its USB product ID when it boots, 2485 before and f63b after, so a
# rule that matches a single ID only covers half the device's life.
#
# And it works. The device node comes up crw-rw-rw-, and while I was checking,
# the running crawler.service -- user jt, no sudo, straight out of the unit this
# script writes -- had that node open with a live handle on it. dai.Device() is
# fine for a non-root service user on this machine.
#
# This comment block used to assert the exact opposite: that the rule was not
# installed, that auto mode wanted an 80-movidius.rules, and that dai.Device()
# just fails without it. None of that is true here and 80-movidius.rules has
# never existed on this Pi. Do not go installing one.
#
# If a rebuilt Pi ever does come up without a rule, the symptom is specific:
# dai.Device() fails as jt and succeeds under sudo. The fix then is Luxonis'
# own rules file, not a hand-rolled one, and it is still a one-time system
# change made by hand -- which is why it is not in this script.

set -euo pipefail

START_NOW=0
PRINT_ONLY=0
PYTHON_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start)      START_NOW=1; shift ;;
    --print-unit) PRINT_ONLY=1; shift ;;
    --python)     PYTHON_OVERRIDE="${2:?--python needs a path}"; shift 2 ;;
    --python=*)   PYTHON_OVERRIDE="${1#--python=}"; shift ;;
    # Print the header block as the help text. This used to be a hardcoded
    # sed -n '2,28p', which silently stopped matching the header the moment I
    # edited a comment above -- it was already short by the time I noticed.
    # Walk from the shebang to the first line that is not a comment instead, so
    # the range cannot go stale again.
    -h|--help)    awk 'NR>1 { if (!/^#/) exit; sub(/^#+ ?/, ""); print }' "$0"; exit 0 ;;
    *)            echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="$REPO_DIR/deploy/crawler.service.template"
RUN_USER="${SUDO_USER:-$(id -un)}"
RUN_GROUP="$(id -gn "$RUN_USER")"

if [[ ! -f "$TEMPLATE" ]]; then
  echo "No unit template at $TEMPLATE" >&2
  exit 1
fi

if [[ ! -f "$REPO_DIR/crawler/__main__.py" ]]; then
  echo "No crawler/__main__.py in $REPO_DIR -- is this the checkout?" >&2
  exit 1
fi

if [[ "$RUN_USER" == "root" ]]; then
  echo "Refusing to install a root-owned service. Run it with sudo as your own user." >&2
  exit 1
fi

# Interpreter, in this order and for this reason:
#
#   1. --python, for when I am switching on purpose.
#   2. env/bin/python, env/bin/python3 -- the old hand-rolled venv. This is
#      what the installed unit points at today and what the car is running out
#      of, so it keeps winning. Re-running this script on the Pi must never
#      quietly move a working service onto a different interpreter behind my
#      back.
#   3. .venv/bin/python, .venv/bin/python3 -- what `uv sync` makes. A fresh
#      checkout has no env/, so there it gets picked up on its own, which is
#      the point. On the car it only takes over once I delete env/ or pass
#      --python.
#   4. system python3, with a warning.
#
# So the uv path is available everywhere and automatic on anything new, and
# moving the car itself across stays a decision I make with --python.
if [[ -n "$PYTHON_OVERRIDE" ]]; then
  if [[ ! -x "$PYTHON_OVERRIDE" ]]; then
    echo "Not executable: $PYTHON_OVERRIDE" >&2
    exit 1
  fi
  PYTHON="$PYTHON_OVERRIDE"
elif [[ -x "$REPO_DIR/env/bin/python" ]]; then
  PYTHON="$REPO_DIR/env/bin/python"
elif [[ -x "$REPO_DIR/env/bin/python3" ]]; then
  PYTHON="$REPO_DIR/env/bin/python3"
elif [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
  PYTHON="$REPO_DIR/.venv/bin/python"
elif [[ -x "$REPO_DIR/.venv/bin/python3" ]]; then
  PYTHON="$REPO_DIR/.venv/bin/python3"
else
  PYTHON="$(command -v python3)"
  echo "WARNING: no venv at $REPO_DIR/env or $REPO_DIR/.venv, falling back to $PYTHON" >&2
  echo "         Make one with: uv venv --python /usr/bin/python3 --system-site-packages && uv sync" >&2
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
#
# This used to be `systemctl list-unit-files pigpiod.service | grep -q pigpiod`,
# and that is a race I lost roughly twice in every twenty-five runs. `grep -q`
# exits the instant it matches, systemctl gets EPIPE and dies on SIGPIPE with
# 141, and `set -o pipefail` at the top of this script hands that 141 back as
# the status of the whole pipeline -- so a pigpiod that is sitting right there,
# enabled, reads as missing. When it loses, the unit gets written with no
# pigpiod dependency AND no ExecStartPre socket wait, which is precisely the
# boot restart-loop the template warns about: the service comes up before the
# daemon's socket is listening, VehicleController raises, Restart=on-failure
# goes round again. An intermittent dead car out of a clean install.
#
# No pipe, no race. `list-unit-files UNIT --no-legend` prints one line for a
# unit that exists and exits 0, and prints nothing and exits 1 for one that
# does not, so the output is the whole answer.
PIGPIO_UNIT=""
if [[ -n "$(systemctl list-unit-files pigpiod.service --no-legend 2>/dev/null || true)" ]]; then
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

# Fill the template.
#
# Three lines are conditional, exactly as they were when this script echoed
# them one by one: the SupplementaryGroups line goes in only if some of those
# groups exist, ExecStartPre only if there is both a pigs binary and a pigpiod
# unit to wait for, and ExecStopPost only if there is a pigs binary to call.
#
# Substitution is literal, done by hand, rather than sed or awk's gsub -- both
# of those treat characters in the replacement text as special (& most of all),
# and these values are filesystem paths I did not write.
render_unit() {
  local drop_groups=0 drop_pre=0 drop_post=0
  [[ -n "$GROUP_LINE" ]] || drop_groups=1
  [[ -n "$PIGS" && -n "$PIGPIO_UNIT" ]] || drop_pre=1
  [[ -n "$PIGS" ]] || drop_post=1

  awk \
    -v py="$PYTHON" \
    -v repo="$REPO_DIR" \
    -v run_user="$RUN_USER" \
    -v run_group="$RUN_GROUP" \
    -v group_line="$GROUP_LINE" \
    -v pigs="$PIGS" \
    -v pigpio_unit="$PIGPIO_UNIT" \
    -v drop_groups="$drop_groups" \
    -v drop_pre="$drop_pre" \
    -v drop_post="$drop_post" '
    function rep(s, from, to,   out, i) {
      out = ""
      while ((i = index(s, from)) > 0) {
        out = out substr(s, 1, i - 1) to
        s = substr(s, i + length(from))
      }
      return out s
    }
    # Comments in the template document the unit; they are not part of it.
    /^[ \t]*#/       { next }
    /@GROUP_LINE@/   { if (drop_groups) next }
    /^ExecStartPre=/ { if (drop_pre)    next }
    /^ExecStopPost=/ { if (drop_post)   next }
    {
      line = $0
      line = rep(line, "@PYTHON@",      py)
      line = rep(line, "@REPO_DIR@",    repo)
      line = rep(line, "@RUN_USER@",    run_user)
      line = rep(line, "@RUN_GROUP@",   run_group)
      line = rep(line, "@GROUP_LINE@",  group_line)
      line = rep(line, "@PIGS@",        pigs)
      line = rep(line, "@PIGPIO_UNIT@", pigpio_unit)
      # An empty @PIGPIO_UNIT@ would otherwise leave a trailing space on the
      # After= and Wants= lines. Nothing in this unit ends in whitespace.
      sub(/[ \t]+$/, "", line)
      print line
    }
  ' "$TEMPLATE"
}

# Render once, check it, then use it. Anything still looking like a placeholder
# is one I added to the template and forgot to wire up here, and I would much
# rather hear about that than install it.
UNIT_TEXT="$(render_unit)"
if printf '%s\n' "$UNIT_TEXT" | grep -q '@[A-Z_][A-Z_]*@'; then
  echo "Unfilled placeholder in the rendered unit:" >&2
  printf '%s\n' "$UNIT_TEXT" | grep -n '@[A-Z_][A-Z_]*@' >&2
  exit 1
fi

if [[ $PRINT_ONLY -eq 1 ]]; then
  printf '%s\n' "$UNIT_TEXT"
  exit 0
fi

if [[ $EUID -ne 0 ]]; then
  echo "Needs root: sudo $0" >&2
  exit 1
fi

UNIT=/etc/systemd/system/crawler.service
echo "Writing $UNIT"
printf '%s\n' "$UNIT_TEXT" > "$UNIT"

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
