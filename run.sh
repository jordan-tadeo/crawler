#!/bin/bash
# Drive the car by hand from a terminal. The real entry point on the car is
# the systemd unit that deploy/install-service.sh writes -- this is the same
# command, for when I want to watch it in a shell instead.
#
# It used to `cd ~/Documents/py/crawler`, which is not this checkout, and run
# a bare `python3 main.py` with no --mode, so it could start the wrong copy of
# the program under the wrong interpreter. It works out of whatever checkout
# it is sitting in now. start.sh was a byte-for-byte copy of it and is gone.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Same precedence as deploy/install-service.sh, and the same reason: `env` is
# the old hand-rolled venv the installed service still points at, so on the car
# it keeps winning until I delete it. `.venv` is what `uv sync` makes, and on a
# fresh checkout -- which has no env/ -- it is what gets used. CRAWLER_PYTHON
# overrides the lot, for trying one against the other.
if [[ -n "${CRAWLER_PYTHON:-}" ]]; then
  PYTHON="$CRAWLER_PYTHON"
elif [[ -x env/bin/python ]]; then
  PYTHON=env/bin/python
elif [[ -x env/bin/python3 ]]; then
  PYTHON=env/bin/python3
elif [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
elif [[ -x .venv/bin/python3 ]]; then
  PYTHON=.venv/bin/python3
else
  PYTHON="$(command -v python3)"
  echo "No venv at $REPO_DIR/env or $REPO_DIR/.venv, using $PYTHON" >&2
  echo "Make one with: uv venv --python /usr/bin/python3 --system-site-packages && uv sync" >&2
fi

exec "$PYTHON" main.py --mode manual "$@"
