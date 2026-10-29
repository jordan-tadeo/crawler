#!/bin/bash
# Retrain the driving policy in one command: pull the recordings, decode only
# the new ones, build the dataset, train the held-out folds and the final model
# at once, grade it against the model the car runs, and register it as a
# candidate. It never deploys; that stays ./scripts/models.sh deploy.
#
#   ./scripts/retrain.sh                    # the whole thing
#   ./scripts/retrain.sh --no-pull          # with what is on this Mac already
#   ./scripts/retrain.sh --dry-run          # everything but the register
#   ./scripts/retrain.sh -- --imu-no-yaw    # after --: flags for policy_train.py
#
# tools/retrain.py's docstring says what each stage does, the junk rules, and
# the cache in ~/crawler-data/cache. The training venv is
# CRAWLER_TRAIN_PYTHON, else ~/crawler-train; the first run makes that one
# from tools/requirements-train.txt if it is not there.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${CRAWLER_TRAIN_PYTHON:-$HOME/crawler-train/bin/python}"

case "${1:-}" in
  -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

if [ ! -x "$PY" ]; then
  if [ -n "${CRAWLER_TRAIN_PYTHON:-}" ]; then
    echo "CRAWLER_TRAIN_PYTHON=$PY is not a python" >&2
    exit 1
  fi
  echo "no training venv at ~/crawler-train; making it from tools/requirements-train.txt"
  if command -v uv >/dev/null; then
    uv venv --python 3.11 "$HOME/crawler-train"
    uv pip install --python "$PY" -r "$REPO_DIR/tools/requirements-train.txt"
  else
    python3.11 -m venv "$HOME/crawler-train"
    "$PY" -m pip install -r "$REPO_DIR/tools/requirements-train.txt"
  fi
fi

exec "$PY" "$REPO_DIR/tools/retrain.py" "$@"
