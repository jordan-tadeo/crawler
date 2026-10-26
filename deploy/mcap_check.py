"""Say what is in a pulled recording, and repair the ones Studio will refuse.

Run by pull-recordings.sh through `uv run --with mcap`, so the only thing it
is allowed to import beyond the standard library is `mcap`.

Two jobs:

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
"""

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


def main(argv):
    targets = []
    for arg in argv or ["."]:
        if os.path.isdir(arg):
            targets += [os.path.join(arg, n) for n in sorted(os.listdir(arg))
                        if n.startswith("crawler-") and n.endswith(".mcap")]
        else:
            targets.append(arg)

    if not targets:
        print("No crawler-*.mcap files to check.")
        return 0

    broken = 0
    for path in targets:
        schemas, channels, messages, truncated = scan(path)
        if indexed(path):
            report(path, schemas, channels, messages)
            continue

        broken += 1
        print(f"  {os.path.basename(path)}  no index -- that run did not "
              f"shut down cleanly" + (", and the tail is truncated"
                                      if truncated else ""))
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
        print(f"\n{broken} file(s) needed repairing. Studio will open them now.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
