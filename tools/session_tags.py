#!/usr/bin/env python3
"""Every session with its tags, ticks and minutes.

    python tools/session_tags.py DATA.npz             # a built dataset
    python tools/session_tags.py REC.mcap [REC.mcap]  # straight off recordings
    python tools/session_tags.py                      # every <data>/mcap/crawler-record-*.mcap

The tags are what I said on the map page (crawler/telemetry/tags.py:
outdoor, clean); a session with none recorded is indoor, messy. One line a
session, and a session whose tags changed partway shows each mix with its
ticks. Then the totals by mix, which is what --include-tags and
--exclude-tags on policy_train.py and policy_scorecard.py will see.

From a dataset: the kept ticks (10 Hz, after the idle rules and the rest),
the labelled decisions, and minutes as ticks / 600. From a recording: its
/footprint rows (one a decision, 10 Hz: record and auto), the tags the
file opened with (its metadata record) and every /session/tags change, and
minutes as the span of its messages. A recording is read whole in stream
order, so a cut-off file reads up to the cut.
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_paths as dp  # noqa: E402
import policy_tags as ptg  # noqa: E402
from crawler.telemetry import tags as tg  # noqa: E402

TICK_S = 0.1
FOOTPRINT = "/footprint"
DEFAULT_MCAPS = os.path.join(dp.DATA_ROOT, "mcap", "crawler-record-*.mcap")


def dataset_rows(data):
    """[(name, {mix: ticks}, ticks, decisions, minutes)] off a dataset."""
    names = data["meta"]["sessions"]
    bits = ptg.tick_tags(data)
    sess = np.asarray(data["session"])
    labelled = ((np.asarray(data["steer_label"], bool) | np.asarray(data["throttle_label"], bool))
                & np.asarray(data["decision"], bool))
    out = []
    for i, name in enumerate(names):
        on = sess == i
        n = int(on.sum())
        out.append((name, ptg.counts(bits, on), n, int((on & labelled).sum()),
                    n * TICK_S / 60.0))
    return out


def recording_row(path):
    """(name, {mix: rows}, rows, None, minutes) off one MCAP: each
    /footprint row takes the tags then (policy_tags.tags_at)."""
    from mcap.records import Channel, Message, Metadata
    from mcap.stream_reader import StreamReader
    channels, entries, rows = {}, [], []
    first = last = None
    with open(path, "rb") as f:
        try:
            for rec in StreamReader(f, skip_magic=False, validate_crcs=False).records:
                if isinstance(rec, Channel):
                    channels[rec.id] = rec.topic
                elif isinstance(rec, Metadata) and rec.name == tg.METADATA_NAME:
                    entries.append((0, tg.bits_of(dict(rec.metadata))))
                elif isinstance(rec, Message):
                    t = int(rec.log_time)
                    first = t if first is None else min(first, t)
                    last = t if last is None else max(last, t)
                    topic = channels.get(rec.channel_id)
                    if topic == FOOTPRINT:
                        rows.append(t)
                    elif topic == tg.TOPIC:
                        try:
                            entries.append((t, tg.bits_of(json.loads(rec.data))))
                        except ValueError:
                            pass
        except Exception:
            pass   # cut off mid-chunk: what was read stands
    name = os.path.splitext(os.path.basename(path))[0]
    minutes = (last - first) / 60e9 if first is not None else 0.0
    if not entries:
        mixes = {tg.said(0) + " (none recorded)": len(rows)}
    else:
        mixes = ptg.counts(ptg.tags_at(entries, rows)) if rows else {
            tg.said(entries[-1][1]): 0}
    return name, mixes, len(rows), None, minutes


def format_rows(rows, unit):
    lines, totals = [], collections.Counter()
    width = max([len(r[0]) for r in rows] + [7])
    for name, mixes, n, decisions, minutes in rows:
        said = "; ".join(f"{k}" if len(mixes) == 1 else f"{k} {v}" for k, v in mixes.items())
        lines.append(f"{name:<{width}}  {n:>7} {unit}"
                     + (f"  {decisions:>6} labelled" if decisions is not None else "")
                     + f"  {minutes:6.1f} min  {said}")
        for k, v in mixes.items():
            totals[k.replace(" (none recorded)", "")] += v
    lines.append("")
    lines.append(f"totals by tags ({unit}):")
    for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:<24} {v:>8}  {v * TICK_S / 60.0:6.1f} min")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*",
                    help=f"one dataset .npz, or recordings (default {DEFAULT_MCAPS})")
    args = ap.parse_args(argv)
    paths = args.paths
    if not paths:
        import glob
        paths = sorted(glob.glob(DEFAULT_MCAPS))
        if not paths:
            print(f"no recordings match {DEFAULT_MCAPS}")
            return 1
    if len(paths) == 1 and paths[0].endswith(".npz"):
        import policy_eval as pe
        print(format_rows(dataset_rows(pe.load(paths[0])), "ticks"))
        return 0
    if any(p.endswith(".npz") for p in paths):
        print("one dataset .npz, or recordings, not both")
        return 2
    print(format_rows([recording_row(p) for p in paths], "rows"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
