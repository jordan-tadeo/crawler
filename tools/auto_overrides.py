"""Which of the car's auto ticks were the model's own call, off a drive's own recording.

    python tools/auto_overrides.py OUT_DIR RUN.mcap [RUN.mcap ...]

An accepted-auto tick (policy_train.py --accepted-weight) is labelled with
what the car sent. On most ticks that is the model's gear, but not on all:
the goal's "goal reached" parks the car whatever the model said, the floor
holds a forward at 0 ("model floor"), the rear stop cuts a reverse, a rear
masked reverse goes out as a forward, the tilt veto zeroes a direction, the
reverse commit holds the wheels for REVERSE_COMMIT_S, a reverse in the
refractory window is driven as a forward, and the stuck and standoff
recovers back out on their own. Those are the car's rules, not the model,
and a "let stand" on one of them says nothing about the model's choice.
/footprint says which it was (policy_reason, reverse_refractory).

Writes OUT_DIR/<run>.npz, one row per /footprint row the policy drove in auto:

  log_t    the row's log time, seconds: what a dataset tick pairs on
  own      the command was the model's own: policy_reason one of OWN_REASONS
           and not a refractory reverse
  reason   policy_reason as recorded
  layout   LAYOUT

policy_train.py --accepted-own OUT_DIR keeps an accepted-auto tick only when
the newest row at or before it, no more than STALE_S old, is own. A row
with no policy_reason (a recording from before the field) is own, as there
is nothing to say otherwise.

Needs the `mcap` package.
"""

import argparse
import json
import os
import sys

import numpy as np

LAYOUT = "own1"
# The model's gear, as it chose it: "model creep" is its forward at no more
# than the creep's throttle, the gear unchanged.
OWN_REASONS = ("model", "model creep")
# A tick pairs with a row no older than this: /footprint is one row a decision.
STALE_S = 0.15


def is_own(row):
    """One /footprint payload -> (policy drove it in auto, the model's own call)."""
    if row.get("autopilot") != "policy" or row.get("mode") != "auto":
        return False, False
    if row.get("throttle_by", "driver") != "driver":
        return False, False
    reason = row.get("policy_reason")
    if reason is None:
        return True, True
    return True, reason in OWN_REASONS and not bool(row.get("reverse_refractory"))


def read(path):
    """One MCAP -> dict log_t, own, reason over the rows the policy drove."""
    from mcap.reader import make_reader
    t, own, why = [], [], []
    with open(path, "rb") as f:
        try:
            for _s, _ch, m in make_reader(f).iter_messages(topics=["/footprint"]):
                try:
                    d = json.loads(m.data)
                except Exception:
                    continue
                drove, mine = is_own(d)
                if not drove:
                    continue
                t.append(m.log_time / 1e9)
                own.append(mine)
                why.append(str(d.get("policy_reason", "")))
        except Exception:
            pass  # a file cut off mid-chunk: what was read stands
    o = np.argsort(np.asarray(t, float), kind="stable")
    return {"log_t": np.asarray(t, float)[o], "own": np.asarray(own, bool)[o],
            "reason": np.asarray(why, dtype=str)[o], "layout": np.array(LAYOUT)}


def pair(row_t, tick_t, stale_s=STALE_S):
    """For each tick time, the index of the newest row at or before it, no
    more than stale_s old, else -1."""
    row_t = np.asarray(row_t, float)
    tick_t = np.asarray(tick_t, float)
    if not len(row_t):
        return np.full(len(tick_t), -1)
    j = np.searchsorted(row_t, tick_t, side="right") - 1
    good = j >= 0
    jj = np.where(good, j, 0)
    good &= (tick_t - row_t[jj]) <= stale_s
    return np.where(good, j, -1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for path in args.runs:
        name = os.path.basename(path).rsplit(".", 1)[0]
        out = read(path)
        if not len(out["log_t"]):
            print(f"{name}: the policy never drove")
            continue
        np.savez_compressed(os.path.join(args.out, name + ".npz"), **out)
        reasons, counts = np.unique(out["reason"][~out["own"]], return_counts=True)
        print(f"{name}: {int(out['own'].sum())} of {len(out['own'])} the model's own; not: "
              + (", ".join(f"{r or '?'} {c}" for r, c in zip(reasons, counts)) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
