"""Say what is in a pulled recording, and repair the ones Studio will refuse.

Run by pull-recordings.sh through `uv run --with mcap`, so the only thing it
is allowed to import beyond the standard library is `mcap`.

Three jobs now:

  Report. Per topic: message count, rate, schema name. Which is also the
  answer to "did that run actually record what I think it did" -- a run with
  no /camera/depth in it was a run with the camera shut, not a bug.

  Repair. Only MCAPWriter.close() writes the summary and the message index,
  and the run I care most about is the one that ended with the battery out,
  which never reached close(). The messages are all there and stream back
  fine; it is the index that is missing, and the indexed reader Studio uses
  refuses the file without it. So a file with no index is read in streaming
  order and written back out through the same library with one, which is
  what `mcap recover` does, minus the Go toolchain. The original is kept
  beside it as .orig until I am happy, because a repair that eats the only
  copy of a bad run is worse than no repair.

  Review. --marks lists the moments I marked with A and B while driving, each
  with its snapshot, and --window dumps every message on every channel inside
  one mark's window. That pair is the whole point of the mark buttons: a mark
  I cannot go and read afterwards is a button that does nothing. See marks.py
  for why a mark is a window either side of the press rather than an instant.

Both review modes are read-only. They never repair, because the file I most
want to read marks out of is the run that just ended badly and I am not
rewriting that until I have looked at it.
"""

import argparse
import json
import os
import sys

from mcap.reader import make_reader
from mcap.records import Channel, Message, Schema
from mcap.stream_reader import StreamReader
from mcap.writer import Writer


def indexed(path):
    """True if the fast path can read this file -- i.e. Studio can open it."""
    try:
        with open(path, "rb") as handle:
            summary = make_reader(handle).get_summary()
        return summary is not None and bool(summary.channels)
    except Exception:
        return False


def scan(path):
    """Every record in file order. Works with or without an index."""
    schemas, channels, messages = {}, {}, []
    truncated = False
    try:
        with open(path, "rb") as handle:
            for record in StreamReader(handle).records:
                if isinstance(record, Schema):
                    schemas[record.id] = record
                elif isinstance(record, Channel):
                    channels[record.id] = record
                elif isinstance(record, Message):
                    messages.append(record)
    except Exception:
        # The padding a half-written file ends in. Everything read up to here
        # is good, which is the whole point of reading it this way.
        truncated = True
    return schemas, channels, messages, truncated


def repair(path, schemas, channels, messages):
    """Rewrite with a summary and a message index. Returns the new path."""
    fixed = path + ".fixed"
    with open(fixed, "wb") as handle:
        writer = Writer(handle)
        writer.start(profile="crawler", library="mcap_check")
        schema_ids, channel_ids = {}, {}
        for old_id, schema in schemas.items():
            schema_ids[old_id] = writer.register_schema(
                name=schema.name, encoding=schema.encoding, data=schema.data)
        for old_id, channel in channels.items():
            channel_ids[old_id] = writer.register_channel(
                topic=channel.topic,
                message_encoding=channel.message_encoding,
                schema_id=schema_ids.get(channel.schema_id, 0),
                metadata=dict(channel.metadata or {}),
            )
        # log_time order, because an index over out-of-order messages is an
        # index that lies. They come off the car in order per channel, but
        # interleaved across channels only roughly.
        for message in sorted(messages, key=lambda m: m.log_time):
            if message.channel_id not in channel_ids:
                continue
            writer.add_message(
                channel_id=channel_ids[message.channel_id],
                log_time=message.log_time,
                publish_time=message.publish_time,
                sequence=message.sequence,
                data=message.data,
            )
        writer.finish()

    os.replace(path, path + ".orig")
    os.replace(fixed, path)
    return path


# The topic marks.py publishes on. Named here rather than imported, because
# this file runs under `uv run --with mcap` on the laptop with none of the
# car's modules importable.
MARKS_TOPIC = "/marks"

# What a mark's window reaches, if the message itself does not say. Every mark
# carries its own lookback_s and lead_s so that a file recorded before these
# numbers were last changed still reviews with the numbers it was recorded
# under; these are only the fallback for a message missing them.
DEFAULT_LOOKBACK_S = 4.0
DEFAULT_LEAD_S = 1.0

# How far past the ends of a file a window has to reach before it is worth
# saying so. The recorder rotates on a size budget, so a mark near a boundary
# genuinely has half its window in the segment next door -- but the live run's
# own last mark always overruns the end by a sliver and that is not news.
OVERRUN_NOTE_S = 0.25


def run_start(messages):
    """log_time of the first message in the file, or 0 for an empty one."""
    return min((m.log_time for m in messages), default=0)


def offset(log_time, t0):
    """A log_time as a +seconds offset into the run, for reading."""
    return (log_time - t0) / 1e9


def stamp(log_time, t0, places=1):
    """An offset written the way I want to read it: t+63.4s, or t-3.0s.

    Signed, because a mark in the first four seconds of a segment has a window
    that starts before the segment does, and "t+-3.0s" is not a timestamp.
    """
    return "t{:+.{}f}s".format(offset(log_time, t0), places)


def decode_marks(channels, messages):
    """Every /marks message as (log_time, dict), oldest first.

    A message that will not decode is kept with whatever did decode, because
    half a mark still tells me when I pressed the button.
    """
    ids = {cid for cid, ch in channels.items() if ch.topic == MARKS_TOPIC}
    if not ids:
        return []

    out = []
    for message in messages:
        if message.channel_id not in ids:
            continue
        try:
            row = json.loads(message.data.decode("utf-8"))
            if not isinstance(row, dict):
                row = {}
        except Exception:
            row = {}
        out.append((message.log_time, row))
    out.sort(key=lambda pair: pair[0])
    return out


def mark_window(log_time, row):
    """The [start, end] log_time bounds of one mark's window, in nanoseconds.

    Off the mark's own log_time rather than its mono_s, so the arithmetic is
    the same two clocks never have to agree.
    """
    try:
        lookback = float(row.get("lookback_s", DEFAULT_LOOKBACK_S))
    except (TypeError, ValueError):
        lookback = DEFAULT_LOOKBACK_S
    try:
        lead = float(row.get("lead_s", DEFAULT_LEAD_S))
    except (TypeError, ValueError):
        lead = DEFAULT_LEAD_S
    return (log_time - int(lookback * 1e9), log_time + int(lead * 1e9))


def _num(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def describe_mark(index, log_time, row, t0):
    """One mark as a few lines of text. The snapshot, laid out to be read."""
    kind = str(row.get("kind", "?")).upper()
    start, end = mark_window(log_time, row)
    lines = [
        f"    mark #{index}  {kind:<4}  {row.get('utc', '?')}  "
        f"{stamp(log_time, t0)}",
        f"      window     {stamp(start, t0)} .. {stamp(end, t0)}  "
        f"(-{_num(row, 'lookback_s', DEFAULT_LOOKBACK_S):.1f}s / "
        f"+{_num(row, 'lead_s', DEFAULT_LEAD_S):.1f}s)",
        f"      mode       {row.get('mode', '?')}"
        + (f", camera {row['camera_state']}" if row.get("camera_state") else ""),
        f"      commanded  throttle {_num(row, 'throttle'):+.2f}  "
        f"steer {_num(row, 'steering_front'):+.2f} / "
        f"{_num(row, 'steering_rear'):+.2f}  "
        f"esc {int(_num(row, 'esc_pulse_us')):d} us",
    ]

    reason = row.get("block_reason") or "clear"
    near = _num(row, "near_distance_mm", -1.0)
    near_text = f"{near:.0f} mm" if near >= 0 else "nothing measurable"
    probes = row.get("probe_distance_mm") or []
    probe_text = ""
    if probes:
        probe_text = "  probes " + " ".join(
            ("--" if _as_float(p) < 0 else f"{_as_float(p):.0f}")
            for p in probes)
    lines.append(f"      footprint  {reason}, nearest {near_text}"
                 f"{probe_text}  ({_num(row, 'scan_ms'):.1f} ms)")

    if row.get("recovery_phase"):
        lines.append(f"      recovering {row['recovery_phase']}, "
                     f"{_num(row, 'recovery_remaining_s'):.1f} s left")

    lines.append(f"      motion     {row.get('motion_evidence', '?')}  "
                 f"gyro {_num(row, 'gyro_mean', -1.0):.3f}  "
                 f"accel {_num(row, 'accel_mean', -1.0):.3f}")
    return lines


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def marks_report(path, channels, messages):
    """List every mark in one file. Returns how many there were."""
    print(f"  {os.path.basename(path)}")
    marks = decode_marks(channels, messages)
    if not marks:
        print("    no marks in this one -- either I pressed nothing, or it "
              "predates the mark buttons")
        return 0

    t0 = run_start(messages)
    good = sum(1 for _, row in marks if row.get("kind") == "good")
    print(f"    {len(marks)} marks ({good} good, {len(marks) - good} bad)")
    for index, (log_time, row) in enumerate(marks):
        print()
        for line in describe_mark(index, log_time, row, t0):
            print(line)
    return len(marks)


def window_report(path, schemas, channels, messages, index):
    """Dump every message inside one mark's window, in time order."""
    marks = decode_marks(channels, messages)
    if not marks:
        print(f"  {os.path.basename(path)}: no marks to window")
        return 1
    if index < 0 or index >= len(marks):
        print(f"  {os.path.basename(path)}: there is no mark #{index}, "
              f"this file has {len(marks)} (0..{len(marks) - 1})")
        return 1

    t0 = run_start(messages)
    log_time, row = marks[index]
    start, end = mark_window(log_time, row)

    print(f"  {os.path.basename(path)}")
    for line in describe_mark(index, log_time, row, t0):
        print(line)
    print()

    # A rotation mid-window means the rest of it is in the next file along.
    # Say so rather than let the dump look like a quiet five seconds.
    first = run_start(messages)
    last = max((m.log_time for m in messages), default=0)
    if (first - start) / 1e9 > OVERRUN_NOTE_S:
        print(f"    note: the window starts {(first - start) / 1e9:.1f} s "
              f"before this file does -- the look-back is in the segment "
              f"before it")
    if (end - last) / 1e9 > OVERRUN_NOTE_S:
        print(f"    note: the window ends {(end - last) / 1e9:.1f} s after "
              f"this file does -- the lead is in the next segment")

    inside = sorted((m for m in messages if start <= m.log_time <= end),
                    key=lambda m: m.log_time)
    topics = {m.channel_id for m in inside}
    print(f"    {len(inside)} messages on {len(topics)} channels "
          f"inside the window")
    print()
    for message in inside:
        channel = channels.get(message.channel_id)
        topic = channel.topic if channel else f"?{message.channel_id}"
        print(f"    {stamp(message.log_time, t0, 3):>12}  {topic:<24} "
              f"{_payload(channel, schemas, message)}")
    return 0


def _payload(channel, schemas, message):
    """One message's body as one line. Binary says its size and stops."""
    encoding = getattr(channel, "message_encoding", "") if channel else ""
    if encoding == "json":
        try:
            return json.dumps(json.loads(message.data.decode("utf-8")),
                              sort_keys=True)
        except Exception:
            pass
    schema = schemas.get(channel.schema_id) if channel else None
    name = schema.name if schema else encoding or "?"
    return f"<{len(message.data)} bytes of {name}>"


def report(path, schemas, channels, messages):
    name = os.path.basename(path)
    size = os.path.getsize(path)
    print(f"  {name}  {size / 1048576.0:.1f} MiB")
    if not messages:
        print("    no messages")
        return
    span = (max(m.log_time for m in messages)
            - min(m.log_time for m in messages)) / 1e9
    print(f"    {len(messages)} messages over {span:.1f} s")
    counts = {}
    for message in messages:
        counts[message.channel_id] = counts.get(message.channel_id, 0) + 1
    for channel_id, count in sorted(
            counts.items(), key=lambda kv: -kv[1]):
        channel = channels.get(channel_id)
        if channel is None:
            continue
        schema = schemas.get(channel.schema_id)
        rate = count / span if span > 0 else 0.0
        print(f"    {channel.topic:<26} {count:>7}  {rate:>5.1f} Hz  "
              f"{schema.name if schema else '?'}")

    # The one line worth pulling out of the topic table above, because it is
    # the line that says whether this file is worth reading: a run I marked is
    # a run I had something to say about.
    marks = decode_marks(channels, messages)
    if marks:
        good = sum(1 for _, row in marks if row.get("kind") == "good")
        print(f"    {len(marks)} marks ({good} good, {len(marks) - good} "
              f"bad) -- see --marks")


def collect(argv):
    """The files named on the command line, a directory counting as its files."""
    targets = []
    for arg in argv or ["."]:
        if os.path.isdir(arg):
            targets += [os.path.join(arg, n) for n in sorted(os.listdir(arg))
                        if n.startswith("crawler-") and n.endswith(".mcap")]
        else:
            targets.append(arg)
    return targets


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Check, repair and review the car's recordings.")
    parser.add_argument("targets", nargs="*",
                        help="mcap files, or directories of them "
                             "(default: the current one)")
    parser.add_argument("--marks", action="store_true",
                        help="list the moments marked with A and B, with the "
                             "snapshot taken at each press")
    parser.add_argument("--window", type=int, default=None, metavar="N",
                        help="dump every message inside mark N's window. "
                             "Implies --marks and wants one file")
    return parser.parse_args(argv)


def review(argv):
    """--marks and --window. Read-only: neither one repairs anything."""
    args = parse_args(argv)
    targets = collect(args.targets)
    if not targets:
        print("No crawler-*.mcap files to read marks out of.")
        return 0

    if args.window is not None and len(targets) != 1:
        print(f"--window wants one file, not {len(targets)}. Name the file.")
        return 2

    total = 0
    for path in targets:
        schemas, channels, messages, _ = scan(path)
        if args.window is not None:
            return window_report(path, schemas, channels, messages,
                                 args.window)
        total += marks_report(path, channels, messages)

    if total:
        print(f"\n{total} marks. Read one with --window N on the file it is "
              f"in; N is the number in the listing above.")
    return 0


def main(argv):
    args = parse_args(argv)
    if args.marks or args.window is not None:
        return review(argv)

    targets = collect(args.targets)

    if not targets:
        print("No crawler-*.mcap files to check.")
        return 0

    broken = 0
    for path in targets:
        schemas, channels, messages, truncated = scan(path)
        if indexed(path):
            report(path, schemas, channels, messages)
            continue

        # Two innocent reasons for no index and one guilty one, and from
        # here they are indistinguishable: the newest file is the run that is
        # happening right now and is still open, and the one before it may be
        # a run that ended with the battery out. Either way the fix is the
        # same, so say both rather than accuse the car of crashing every
        # time I pull the live run.
        broken += 1
        print(f"  {os.path.basename(path)}  no index -- either it is the run "
              f"still being written, or that run did not shut down cleanly"
              + (", and the tail is truncated" if truncated else ""))
        if not messages:
            print("    nothing recoverable in it, leaving it alone")
            continue
        try:
            repair(path, schemas, channels, messages)
        except Exception as e:
            print(f"    could not repair it: {e}")
            continue
        print("    repaired, original kept as "
              f"{os.path.basename(path)}.orig")
        report(path, schemas, channels, messages)

    if broken:
        print(f"\n{broken} file(s) needed an index writing. Studio will open "
              f"them now. A snapshot of the run that is still going is "
              f"normally one of them.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
