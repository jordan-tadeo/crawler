#!/bin/bash
# Copy the car's recordings to this laptop and make sure they open.
#
# Runs on the Mac, not on the car -- it is the other end of recording.py. The
# car writes log/mcap/*.mcap as it drives; this fetches them, checks each one
# is something Foxglove Studio will actually accept, and repairs the ones that
# are not.
#
#   ./pull-recordings.sh              # newest 5 into ./recordings
#   ./pull-recordings.sh -n 20        # newest 20
#   ./pull-recordings.sh --all
#   ./pull-recordings.sh --list       # just say what is on the car
#   ./pull-recordings.sh -o ~/Desktop/crawl
#
# Then: open -a Foxglove, and drag the file in. Open connection is for live;
# a file is File -> Open local file, or just a drag onto the window.
#
# Or read the marks instead of looking at it. Every file's summary below says
# how many moments I marked with A and B while driving, and
# `deploy/mcap_check.py --marks` lists them with the snapshot taken at each
# press. --window N on one of them dumps everything the car saw around it.
#
# Why there is a repair step at all. Only close() writes an MCAP's summary and
# message index, and a run that ends with the battery out never reaches
# close(). Measured: every message survives and streams back fine with its
# schema, but the indexed reader Studio uses refuses the file. So any file
# without an index gets rewritten into one that has it, which is exactly the
# `mcap recover` operation and is done here with the mcap Python library so
# there is no Go toolchain to install. The original is kept as .orig.

set -euo pipefail

# By name, not by address. The lease the car used to hold moved, and the device
# that has that address now answers a ping and refuses SSH -- so a hardcoded
# address fails in the one way that looks like the car being broken instead of
# like a wrong address. jtpi.local is mDNS and needs nothing set up.
# CRAWLER_HOST=jt@<address> still overrides it for a network with no mDNS.
CAR="${CRAWLER_HOST:-jt@jtpi.local}"
REMOTE_DIR="${CRAWLER_REMOTE_DIR:-/home/jt/Documents/py/crawler/log/mcap}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$REPO_DIR/recordings"
COUNT=5
LIST_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--count) COUNT="$2"; shift 2 ;;
    --all)      COUNT=0; shift ;;
    --list)     LIST_ONLY=1; shift ;;
    -o|--out)   OUT="$2"; shift 2 ;;
    -h|--help)  sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# ls -t, newest first, so -n 5 is the last five segments and not five
# arbitrary ones. The guard matters: with no recordings yet, an unguarded ls
# of a missing directory is a non-zero exit and set -e takes the script out
# before it can say anything useful.
REMOTE_LS="ls -t '$REMOTE_DIR'/crawler-*.mcap 2>/dev/null || true"
FILES="$(ssh "$CAR" "$REMOTE_LS")"

if [[ -z "$FILES" ]]; then
  echo "No recordings in $CAR:$REMOTE_DIR"
  echo "Is the service up, and is it running a build that has recording in it?"
  echo "  ssh $CAR 'systemctl is-active crawler; journalctl -u crawler | grep Recording | tail -3'"
  exit 1
fi

echo "On the car, newest first:"
ssh "$CAR" "ls -lht '$REMOTE_DIR'/crawler-*.mcap 2>/dev/null | head -20"
ssh "$CAR" "du -sh '$REMOTE_DIR'; df -h / | tail -1"
[[ "$LIST_ONLY" == 1 ]] && exit 0

if [[ "$COUNT" != 0 ]]; then
  FILES="$(printf '%s\n' "$FILES" | head -n "$COUNT")"
fi

mkdir -p "$OUT"
echo
echo "Into $OUT:"
# One scp per file rather than one invocation with a remote glob, because the
# selection above is already done and a second glob on the far side would
# ignore it.
while IFS= read -r remote; do
  [[ -z "$remote" ]] && continue
  echo "  $(basename "$remote")"
  scp -q "$CAR:$remote" "$OUT/"
done <<< "$FILES"

# The check and the repair. uv --with builds a throwaway environment for the
# mcap library, so this needs nothing installed and leaves nothing behind.
# README says how to get uv; it is the same uv the car's venv comes from.
if ! command -v uv >/dev/null 2>&1; then
  echo
  echo "No uv on this Mac, so skipping the index check." >&2
  echo "A file from a run that lost power will not open in Studio until it is" >&2
  echo "repaired. curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 0
fi

echo
uv run --quiet --with mcap python "$REPO_DIR/deploy/mcap_check.py" "$OUT"

echo
echo "Open one: open -a Foxglove   then drag a file from $OUT onto the window."
# The summary above already says how many marks each file has. This is the
# command that turns that count into something I can act on, which is the only
# reason the mark buttons are worth having.
echo "Read the marks: uv run --with mcap python deploy/mcap_check.py --marks $OUT"
echo "Then one of them: ... --window N <file>"
