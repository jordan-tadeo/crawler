#!/bin/bash
# The driving models: the releases that earned a number, the candidates that
# have not yet, which one the car runs, and putting another one on it.
#
# Runs on the Mac. The models live in ~/crawler-data/models (kblock-X.Y.* and
# candidate-<tag>.*, with registry.json), never in this repo; tools/models.py
# is the tool and its docstring says what earns a number and how a deploy
# keeps the car safe.
#
#   ./models.sh list                       # releases, candidates, * = on the car
#   ./models.sh register ~/crawler-data/models/policy-<stamp>.onnx --tag imu-gyro
#   ./models.sh deploy candidate-imu-gyro  # a test drive (or --candidate imu-gyro)
#   ./models.sh promote imu-gyro --evidence ab.json --approved-on-car --notes "..."
#   ./models.sh promote imu-gyro --force-gate "why" --approved-on-car
#   ./models.sh deploy 1.3 --dry-run       # what it would do, touching nothing
#   ./models.sh deploy 1.3                 # copy, verify, re-point, restart, check
#   ./models.sh rollback                   # back to what drove before
#   ./models.sh status                     # the car's symlink, by sha256
#   ./models.sh approve 1.2 --notes "..."  # it drove well on the car
#   ./models.sh demote 2.0 --tag imu-1028 --reason "..."   # give a number back
#
# A new model is a candidate with no number. promote gives it the next one
# only when it beats the latest release on held-out folds (--evidence) and I
# say it drove well (--approved-on-car). deploy and rollback refuse unless the
# car is parked (pigs gpw 18 reads 1575). CRAWLER_HOST=jt@<address> overrides
# jt@jtpi.local, as for pull-recordings.sh.

set -euo pipefail

export CRAWLER_HOST="${CRAWLER_HOST:-jt@jtpi.local}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
  ""|-h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac
exec python3 "$REPO_DIR/tools/models.py" "$@"
