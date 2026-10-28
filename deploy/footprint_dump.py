"""Turn the /footprint rows in a recording into one table, for training.

    uv run --with mcap deploy/footprint_dump.py recordings/crawler-record-*.mcap
    uv run --with mcap deploy/footprint_dump.py run.mcap -o run.csv
    uv run --with mcap --with pyarrow deploy/footprint_dump.py run.mcap -o run.parquet
    uv run --with mcap --with numpy==1.26.4 --with opencv-python-headless \
        deploy/footprint_dump.py run.mcap --verify

One row per decision, one column per number: the nine probes are spread out
as probe_distance_mm_0 .. _8 and so on, left to right, so the table loads
straight into anything. Every file given goes into the one table with its
file name in a column, so a whole weekend of record sessions is one call.

--verify is the check that the raw depth is worth its bytes. Every fifth
record row has a lossless frame on /camera/depth_raw (16-bit PNG) at exactly
its log time;
this re-runs driver.free_by_column() and read_the_gradient() over each of
those frames and says whether the probes and the verdict come out as the car
wrote them down. Needs numpy and opencv (or Pillow), and reads driver.py
from this checkout -- so
after the rules change, the same flag is how to see what they would have done
on the old drives.

It also re-runs policy_input.depth_small() over the same frames and checks
the result against the /policy/depth_small picture of that row, byte for
byte: the check that training and the car shrink a frame the same way.

--depth DIR writes each of those frames out as DIR/<log_time_ns>.npy.

--small DIR writes the policy pictures, one DIR/<recording>.npz per file:
`depth` (N, 50, 80) uint16 mm, `seq` and `log_time_ns` (N,), in row order,
so `depth[i]` pairs with the row whose seq is `seq[i]`. Schema of both
topics: docs/TRAINING_DATA.md.

Reads in streaming order, so a file that never reached close() -- the battery
out mid-session -- dumps the same as one that did.
"""

import argparse
import csv
import json
import os
import struct
import sys

FOOTPRINT = "/footprint"
DEPTH_RAW = "/camera/depth_raw"
DEPTH_SMALL = "/policy/depth_small"

# The array fields of a row, and what each one is called once it is spread
# out into columns.
ARRAYS = ("probe_distance_mm", "probe_valid_fraction", "probe_blocked",
          "probe_reason")


def flatten(row, log_time=0, source=""):
    """One /footprint message as a flat dict of scalars."""
    out = {"file": source, "log_time_ns": int(log_time)}
    for key, value in row.items():
        if key in ARRAYS and isinstance(value, list):
            for i, item in enumerate(value):
                out[f"{key}_{i}"] = item
        else:
            out[key] = value
    return out


def _varint(data, i):
    shift = value = 0
    while True:
        byte = data[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, i
        shift += 7


def _fields(data):
    """A protobuf message as {field number: last value}. Wire types 0/1/2/5."""
    out, i = {}, 0
    while i < len(data):
        key, i = _varint(data, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            out[number], i = _varint(data, i)
        elif wire == 1:
            out[number] = struct.unpack_from("<Q", data, i)[0]
            i += 8
        elif wire == 2:
            length, i = _varint(data, i)
            out[number] = bytes(data[i:i + length])
            i += length
        elif wire == 5:
            out[number] = struct.unpack_from("<I", data, i)[0]
            i += 4
        else:
            raise ValueError(f"protobuf wire type {wire}")
    return out


def depth_png_bytes(data):
    """The PNG inside a foxglove.CompressedImage payload.

    Field numbers from foxglove's CompressedImage.proto: data 2, format 3. A
    tiny decoder rather than a protobuf dependency, because two fields is all
    this needs.
    """
    fields = _fields(data)
    fmt = fields.get(3, b"").decode("utf-8", "replace")
    if fmt != "png":
        raise ValueError(f"not a PNG depth frame: format {fmt!r}")
    return fields.get(2, b"")


def decode_depth(data):
    """A /camera/depth_raw payload as a uint16 millimetre array.

    opencv if it is there, Pillow if not; either way a 16-bit PNG comes back
    exactly as the car's depth frame was.
    """
    import numpy as np

    png = depth_png_bytes(data)
    try:
        import cv2
        frame = cv2.imdecode(np.frombuffer(png, dtype=np.uint8),
                             cv2.IMREAD_UNCHANGED)
    except ImportError:
        import io
        from PIL import Image
        frame = np.asarray(Image.open(io.BytesIO(png))).astype(np.uint16)
    if frame is None or frame.ndim != 2 or frame.dtype != np.uint16:
        raise ValueError("not a 16-bit greyscale depth frame")
    return frame


def records(path):
    """(topic, log_time, data) for every message, in file order."""
    from mcap.records import Channel, Message
    from mcap.stream_reader import StreamReader

    topics = {}
    with open(path, "rb") as handle:
        try:
            for record in StreamReader(handle).records:
                if isinstance(record, Channel):
                    topics[record.id] = record.topic
                elif isinstance(record, Message):
                    topic = topics.get(record.channel_id)
                    if topic in (FOOTPRINT, DEPTH_RAW, DEPTH_SMALL):
                        yield topic, record.log_time, record.data
        except Exception as e:
            # The padding a half-written file ends in. Everything up to here
            # was good.
            print(f"{os.path.basename(path)}: stopped reading at {e!r}",
                  file=sys.stderr)


def _repo():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def frame_id(data):
    """The frame_id of a CompressedImage payload: "footprint/<seq>"."""
    return _fields(data).get(4, b"").decode("utf-8", "replace")


def verify(row, frame, small=None):
    """Re-scan a frame with this checkout's driver.py. Differences, or [].

    `small` is the row's /policy/depth_small picture, if it has one: it must
    be policy_input.depth_small(frame) exactly.
    """
    _repo()
    from crawler.autopilot import driver as dv
    import numpy as np
    from crawler.autopilot import policy_input

    scan = dv.free_by_column(frame)
    verdict = dv.read_the_gradient(scan)
    problems = []
    if dv.probe_marks(scan) != row.get("probe_marks"):
        problems.append(f"marks {dv.probe_marks(scan)} vs {row.get('probe_marks')}")
    distances = [-1.0 if d == float("inf") else float(d) for d in scan.distance_mm]
    if distances != [float(d) for d in row.get("probe_distance_mm", [])]:
        problems.append(f"distances {distances} vs {row.get('probe_distance_mm')}")
    if (verdict.action, verdict.reason) != (row.get("verdict_action"),
                                            row.get("verdict_reason")):
        problems.append(f"verdict {verdict.action}/{verdict.reason} vs "
                        f"{row.get('verdict_action')}/{row.get('verdict_reason')}")
    if small is not None and not np.array_equal(
            policy_input.depth_small(frame), small):
        problems.append("depth_small is not depth_small(depth_raw)")
    return problems


def dump(paths, depth_dir=None, check=False, small_dir=None):
    """Every /footprint row in every file, flattened. Plus the check counts."""
    table = []
    checked = mismatched = 0
    for path in paths:
        name = os.path.basename(path)
        rows_at = {}
        frames = {}
        smalls = {}
        pictures = []
        for topic, log_time, data in records(path):
            if topic == FOOTPRINT:
                row = json.loads(data)
                table.append(flatten(row, log_time, name))
                if row.get("depth_raw"):
                    rows_at[log_time] = row
            elif topic == DEPTH_SMALL:
                if small_dir:
                    seq = frame_id(data).rpartition("/")[2]
                    pictures.append((int(seq) if seq.isdigit() else -1,
                                     log_time, decode_depth(data)))
                if check:
                    smalls[log_time] = data
            elif depth_dir or check:
                frames[log_time] = data
            # Pair a frame with its row (and, checking, with its picture) as
            # soon as all of them have been read, so a long session does not
            # hold every frame in memory at once. The publisher writes topics
            # in no fixed order within one log time.
            ready = [t for t in frames if t in rows_at
                     and (not check or t in smalls
                          or not rows_at[t].get("depth_small"))]
            for at in ready:
                frame = decode_depth(frames.pop(at))
                if depth_dir:
                    import numpy as np
                    os.makedirs(depth_dir, exist_ok=True)
                    np.save(os.path.join(depth_dir, f"{at}.npy"), frame)
                if check:
                    checked += 1
                    small = smalls.pop(at, None)
                    problems = verify(rows_at[at], frame,
                                      decode_depth(small) if small else None)
                    if problems:
                        mismatched += 1
                        print(f"{name} @ {at}: " + "; ".join(problems),
                              file=sys.stderr)
                del rows_at[at]
            # Pictures whose frame never came are no use to the check.
            if check and len(smalls) > 64:
                for at in sorted(smalls)[:-64]:
                    del smalls[at]
        if small_dir and pictures:
            import numpy as np
            os.makedirs(small_dir, exist_ok=True)
            stem = os.path.splitext(name)[0]
            np.savez(os.path.join(small_dir, f"{stem}.npz"),
                     depth=np.stack([p[2] for p in pictures]),
                     seq=np.array([p[0] for p in pictures], dtype=np.int64),
                     log_time_ns=np.array([p[1] for p in pictures],
                                          dtype=np.int64))
    return table, checked, mismatched


def columns(table):
    """Every column any row has, in first-seen order."""
    seen = {}
    for row in table:
        for key in row:
            seen.setdefault(key, None)
    return list(seen)


def write(table, out):
    if out.endswith(".parquet"):
        import pyarrow as pa
        import pyarrow.parquet as pq
        cols = columns(table)
        pq.write_table(pa.table({c: [r.get(c) for r in table] for c in cols}), out)
        return
    handle = sys.stdout if out == "-" else open(out, "w", newline="")
    try:
        writer = csv.DictWriter(handle, fieldnames=columns(table))
        writer.writeheader()
        writer.writerows(table)
    finally:
        if handle is not sys.stdout:
            handle.close()


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Dump /footprint rows from MCAP recordings to a table.")
    parser.add_argument("paths", nargs="+", help="MCAP files")
    parser.add_argument("-o", "--out", default="-",
                        help="CSV path, or .parquet (needs pyarrow); "
                             "default stdout")
    parser.add_argument("--depth", default=None,
                        help="also write each raw depth frame as DIR/<ns>.npy")
    parser.add_argument("--small", default=None,
                        help="write the /policy/depth_small pictures as "
                             "DIR/<recording>.npz (depth, seq, log_time_ns)")
    parser.add_argument("--verify", action="store_true",
                        help="re-scan every raw frame with this checkout's "
                             "driver.py and compare it to its row")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    table, checked, mismatched = dump(args.paths, args.depth, args.verify,
                                      args.small)
    write(table, args.out)
    print(f"{len(table)} rows from {len(args.paths)} file(s)", file=sys.stderr)
    if args.verify:
        print(f"re-scanned {checked} raw frames: "
              f"{checked - mismatched} match, {mismatched} differ",
              file=sys.stderr)
        return 1 if mismatched else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
