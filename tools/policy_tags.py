"""The recordings' tags (crawler/telemetry/tags.py) on the tools' side.

A dataset's "tags" array is each tick's bit mask (tags.BITS: clean 2), the
newest /session/tags at or before the tick, else the file's own metadata
record, else the first /session/tags in it. A file with none (every
recording before the tags) is 0: not clean. A dataset built before the tags
has no array and is all 0 the same.

A filter or a weight names a tag or its opposite (tags.NAMES): clean or
messy. policy_train.py's --include-tags keeps a tick that has every one
named; --exclude-tags drops one that has any; --tag-weight
TAG=W multiplies a tick's loss weight by W for each named tag it has.

No torch: policy_scorecard.py and tools/session_tags.py use this too.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.telemetry import tags as tg  # noqa: E402


def tags_at(entries, t_ns):
    """Per time in t_ns, the bit mask of the newest (t_ns, bits) entry at or
    before it; before the first, the first's; 0 with no entries."""
    t = np.asarray(t_ns, np.int64)
    if not len(entries):
        return np.zeros(len(t), np.uint8)
    entries = sorted(entries, key=lambda e: e[0])
    times = np.array([e[0] for e in entries], np.int64)
    bits = np.array([e[1] for e in entries], np.uint8)
    k = np.searchsorted(times, t, side="right") - 1
    return bits[np.maximum(k, 0)]


def tick_tags(data):
    """The dataset's per-tick masks; all 0 for one built before the tags."""
    n = len(data["t_ns"])
    if "tags" not in data:
        return np.zeros(n, np.uint8)
    return np.asarray(data["tags"], np.uint8)


def has(bits, name):
    """Per tick: has tag `name`, or (an opposite) not its tag."""
    bits = np.asarray(bits, np.uint8)
    for tag in tg.TAGS:
        if name == tag:
            return (bits & tg.BITS[tag]) != 0
        if name == tg.OPPOSITES[tag]:
            return (bits & tg.BITS[tag]) == 0
    raise ValueError(f"no tag {name!r}; one of {', '.join(tg.NAMES)}")


def parse_names(text):
    """"clean,messy" -> ["clean", "messy"]; ValueError on a name that is
    not one. None or "" is none."""
    names = [n.strip().lower() for n in (text or "").split(",") if n.strip()]
    for n in names:
        if n not in tg.NAMES:
            raise ValueError(f"no tag {n!r}; one of {', '.join(tg.NAMES)}")
    return names


def parse_weights(items):
    """["clean=3", "messy=0.5"] -> {"clean": 3.0, "messy": 0.5}."""
    out = {}
    for item in items or ():
        name, eq, value = str(item).partition("=")
        name = name.strip().lower()
        if not eq or name not in tg.NAMES:
            raise ValueError(f"--tag-weight {item!r}: TAG=W, TAG one of {', '.join(tg.NAMES)}")
        try:
            w = float(value)
        except ValueError:
            w = -1.0
        if not w >= 0 or not np.isfinite(w):
            raise ValueError(f"--tag-weight {item!r}: W is a number, 0 or more")
        out[name] = w
    return out


def keep_mask(bits, include=(), exclude=()):
    """Per tick: has every tag in include and none in exclude."""
    keep = np.ones(len(bits), bool)
    for n in include:
        keep &= has(bits, n)
    for n in exclude:
        keep &= ~has(bits, n)
    return keep


def weights(bits, by_tag):
    """Per tick: the product of by_tag's weight over the tags it has."""
    w = np.ones(len(bits), np.float32)
    for name, value in by_tag.items():
        w = np.where(has(bits, name), w * np.float32(value), w)
    return w.astype(np.float32)


def counts(bits, mask=None):
    """{said(bits): ticks} over the ticks in mask, for a report line."""
    bits = np.asarray(bits, np.uint8)
    if mask is not None:
        bits = bits[np.asarray(mask, bool)]
    values, n = np.unique(bits, return_counts=True)
    return {tg.said(int(v)): int(c) for v, c in zip(values, n)}
