#!/bin/bash
# Copy the car's recordings to this laptop and make sure they open.
#
# Runs on the Mac, not on the car -- it is the other end of recording.py. The
# car writes log/mcap/*.mcap as it drives; this fetches them, checks each one
# is something Foxglove Studio will actually accept, and repairs the ones that
# are not.
#
#   ./pull-recordings.sh              # everything new into ~/crawler-data/mcap
#   ./pull-recordings.sh -n 20        # only the newest 20
#   ./pull-recordings.sh --records-only   # only the Y-button training sessions, not the drive logs
#   ./pull-recordings.sh --list       # just say what is on the car, and what is live
#   ./pull-recordings.sh -o ~/Desktop/crawl
#   ./pull-recordings.sh --prune      # say which car copies are safe to delete
#   ./pull-recordings.sh --prune --yes
#
# Then: open -a Foxglove, and drag the file in. Open connection is for live;
# a file is File -> Open local file, or just a drag onto the window.
#
# Is a drive any good as training data:
#   uv run --with mcap --with numpy==1.26.4 --with opencv-python-headless \
#       python tools/session_report.py ~/crawler-data/mcap/crawler-record-*.mcap
#
# The copy itself is tools/pull_mcaps.py: it refuses a destination inside any
# git work tree (this repo is public and a recording is video of my house),
# skips the segment the service still has open, checks size and sha256 on
# both ends, and only ever deletes off the car with --prune --yes, after
# re-hashing both ends. Its docstring says how the live file is told apart.
#
# Why there is a repair step at all. Only close() writes an MCAP's summary and
# message index, and a run that ends with the battery out never reaches
# close(). Measured: every message survives and streams back fine with its
# schema, but the indexed reader Studio uses refuses the file. So any file
# without an index gets rewritten into one that has it, which is exactly the
# `mcap recover` operation and is done here with the mcap Python library so
# there is no Go toolchain to install. The original is kept as .orig, and
# --prune counts that .orig as the verified copy.

set -euo pipefail

# By name, not by address. The lease the car used to hold moved, and the device
# that has that address now answers a ping and refuses SSH -- so a hardcoded
# address fails in the one way that looks like the car being broken instead of
# like a wrong address. jtpi.local is mDNS and needs nothing set up.
# CRAWLER_HOST=jt@<address> still overrides it for a network with no mDNS.
CAR="${CRAWLER_HOST:-jt@jtpi.local}"
export CRAWLER_HOST="$CAR"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${CRAWLER_DATA:-$HOME/crawler-data/mcap}"
ARGS=()
LIST_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--count) ARGS+=(--count "$2"); shift 2 ;;
    --all)      shift ;;  # the default now; kept so old muscle memory works
    --records-only) ARGS+=(--records-only); shift ;;
    --list)     LIST_ONLY=1; ARGS+=(--list); shift ;;
    -o|--out)   OUT="$2"; shift 2 ;;
    --prune)    ARGS+=(--prune); shift ;;
    --yes)      ARGS+=(--yes); shift ;;
    -h|--help)  sed -n '2,37p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PULLED="$(mktemp)"
trap 'rm -f "$PULLED"' EXIT
python3 "$REPO_DIR/tools/pull_mcaps.py" --dest "$OUT" --pulled-list "$PULLED" \
  ${ARGS[@]+"${ARGS[@]}"}
[[ "$LIST_ONLY" == 1 ]] && exit 0

# The check and the repair, over only what this run fetched -- re-reading
# every recording ever pulled would take minutes. uv --with builds a
# throwaway environment for the mcap library, so this needs nothing
# installed and leaves nothing behind.
[[ -s "$PULLED" ]] || exit 0
if ! command -v uv >/dev/null 2>&1; then
  echo
  echo "No uv on this Mac, so skipping the index check." >&2
  echo "A file from a run that lost power will not open in Studio until it is" >&2
  echo "repaired. curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 0
fi

echo
FILES=()
while IFS= read -r line; do FILES+=("$line"); done < "$PULLED"
uv run --quiet --with mcap python "$REPO_DIR/deploy/mcap_check.py" "${FILES[@]}"

echo
echo "Open one: open -a Foxglove   then drag a file from $OUT onto the window."
# The summary above already says how many marks each file has. This is the
# command that turns that count into something I can act on, which is the only
# reason the mark buttons are worth having.
echo "Read the marks: uv run --with mcap python deploy/mcap_check.py --marks $OUT"
echo "Then one of them: ... --window N <file>"
echo "Is it training data: uv run --with mcap --with numpy==1.26.4 --with opencv-python-headless python tools/session_report.py <file>"
