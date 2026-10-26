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

if [[ -x env/bin/python ]]; then
  PYTHON=env/bin/python
elif [[ -x env/bin/python3 ]]; then
  PYTHON=env/bin/python3
else
  PYTHON="$(command -v python3)"
  echo "No venv at $REPO_DIR/env, using $PYTHON" >&2
fi

exec "$PYTHON" main.py --mode manual "$@"
