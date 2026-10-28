#!/bin/bash
# The driving models: which versions exist, which one the car runs, and
# putting another one on it.
#
# Runs on the Mac. The versions live in ~/crawler-data/models (kblock-X.Y.onnx
# + .json and registry.json), never in this repo; tools/models.py is the tool
# and its docstring says how a version number is picked and how a deploy
# keeps the car safe.
#
#   ./models.sh list                       # versions, scores, * = on the car
#   ./models.sh register ~/crawler-data/models/policy-<stamp>.onnx --notes "..."
#   ./models.sh register X.onnx --version 2.0     # or --bump major|minor
#   ./models.sh deploy 1.1 --dry-run       # what it would do, touching nothing
#   ./models.sh deploy 1.1                 # copy, verify, re-point, restart, check
#   ./models.sh rollback                   # back to what drove before
#   ./models.sh status                     # the car's symlink, by sha256
#
# deploy and rollback refuse unless the car is parked (pigs gpw 18 reads
# 1575). CRAWLER_HOST=jt@<address> overrides jt@jtpi.local, as for
# pull-recordings.sh.

set -euo pipefail

export CRAWLER_HOST="${CRAWLER_HOST:-jt@jtpi.local}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
  ""|-h|--help) sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac
exec python3 "$REPO_DIR/tools/models.py" "$@"
