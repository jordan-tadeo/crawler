"""Every telemetry channel, written to an MCAP file so a run can be replayed.

What this is for: Foxglove live is a window, not a record. Close the laptop
and the run is gone. The run worth looking at is always the one that just
ended badly -- auto drove into a rock, the pad renumbered its axes mid-drive,
the loop stalled for 190 ms -- and I never know in advance which run that is.
So the car records all of them, from service start, without being asked.

The trick that makes this nearly free
-------------------------------------
An MCAP writer in this SDK is a *sink*, exactly like a connected Foxglove
client. Opening one makes Channel.has_sinks() true, which is already the
single thing telemetry.wants() keys off. So recording needs no second code
path, no second encode and no second thread: the control loop builds the same
message it would build for a watching laptop, offer() drops it in the same
one-deep mailbox, and the publisher thread's channel.log() now lands in the
file as well as the socket. Recording on with Foxglove shut costs the control
loop exactly what Foxglove open costs it, which is a number I have already
measured and accepted.

The consequence is worth writing down because it is a real change: the
"nothing is built and nothing is encoded when nobody is watching" property in
telemetry.py is now "nobody is watching" = "no laptop AND no recording". With
recording on by default, the publisher thread colourmaps and JPEG-encodes the
camera channels whenever the camera is open, all the time. That is the price
of having the picture of what the car hit, and it is paid on the publisher
thread, never on the throttle's thread.

Why always-on and not a button
------------------------------
A button means remembering to press it before the thing I wanted to see.
Auto-mode-only means losing the manual runs, and the pad mapping bug that
cost me an afternoon was a manual run. A true ring buffer -- keep the last N
minutes, discard the rest -- is the right shape, but this SDK's writer has no
ring mode and building one means decoding and re-encoding every message to
drop the old ones, on a Pi that is already missing its loop target.

So the ring is at the file level instead. One segment per SEGMENT_BYTES, a
new file when that fills, and the oldest files deleted when the directory
goes over MAX_TOTAL_BYTES. That is a coarse ring buffer with no per-message
work at all: the newest 8 GiB of driving is always on the card and nothing
older can accumulate.

What it costs, measured on the car
----------------------------------
A 100 Hz replica of main.py's telemetry block, 120 s a condition, camera up,
nothing connected to the socket:

                              block on the control thread        process
                           p50     p90     p99     max           CPU
  recording off            9 us   13 us   25 us   109 us         16.0%
  recording ON            11 us  190 us  268 us   831 us         33.8%

With the camera shut, which is how the service actually boots: 7/8/10/54 us
and 2.0% off, 9/120/151/240 us and 3.0% on.

So the loop pays about 240 us at p99 against a 10 ms tick, which is 2.4% of
it, and the p90 jump is not the offer() calls getting slower -- they are two
dict writes -- it is the GIL coming back from a publisher thread that is now
busy in zstd and cv2. It is the same order as the p99 147 us already measured
and accepted for having Studio connected, and for the same reason: a sink is
a sink.

The 18 points of extra CPU is the publisher thread, on a 4-core Pi, so about
4.5% of the machine. None of it is on the thread that commands the throttle.

It records whatever is published, not a list of topics
-----------------------------------------------------
There is no channel list in this file, and that is deliberate. open_mcap()
with no channel_filter attaches to the SDK's global context, which is the
same context telemetry.py's channels are created in, so the writer takes
every channel in the process -- including ones created *after* it opened.
Measured: a channel built after the writer was already running landed all of
its messages and its schema in the file. So the next topic somebody adds --
/auto, for the driver's own decisions, is the one being asked for -- shows up
in recordings with no change in here at all. telemetry.py's ALL_KEYS is that
file's own registry of mailboxes and rate gates; extending it is all a new
channel needs.

A run that ends badly loses its index, not its data
---------------------------------------------------
Only close() writes the summary and the message index, and a run that ends
with the battery out does not reach close(). Measured on a recording killed
with SIGKILL: every message was still there and read back fine in streaming
order, with schemas intact, but the indexed reader -- the fast path Studio
takes -- refused the file outright. disable_seeking makes no difference.

Hence the flush above, which is what puts the bytes on the card at all, and
hence the recover step in pull-recordings.sh, which rewrites an unindexed
file into one Studio will open. Both of them exist because the run I most
want is the run that ended with the power pulled.

A full SD card is a dead car
----------------------------
Worse than dead: a Pi that cannot write is a Pi whose filesystem corrupts,
and the only thing that can take the ESC back to neutral is this process
staying alive. So there are two independent brakes, and they do not trust
each other:

  MAX_TOTAL_BYTES  caps what *this* directory holds. Checked on every
                   rotation and every CHECK_INTERVAL.

  MIN_FREE_BYTES   caps nothing and measures the filesystem. If free space
                   is under it, recording stops -- even if this directory is
                   well inside its own budget, because something else on the
                   card is filling it and recording is the part of this
                   process that is allowed to lose.

Nothing in here can reach the hardware
--------------------------------------
Same contract telemetry.py has. Every public method swallows everything:
start() returns False, tick() returns, stop() returns, and the car drives
exactly as it did. There is no open file, no exception and no disk state that
can make a tick late or make this process exit, because an exit leaves
pigpio emitting the last pulse width it was given forever.
"""

import os
import time

# Where the files go. Under log/, which Logger already creates on the way to
# writing its CSVs, and which is gitignored whole.
DEFAULT_DIR = "log/mcap"

SUFFIX = ".mcap"
PREFIX = "crawler-"

# The two rates everything below is sized off, both measured on the car over
# 120 s at the real publish rates, read back out of the finished file:
#
#   camera up, all seven channels   71.0 KiB/s   250 MiB/hour
#   camera shut, the three JSON     1.2 KiB/s      4 MiB/hour
#
# The camera is 80% of that, and it is the part zstd cannot help with: the
# payload is already JPEG. Whole-file compression came out at 0.69 of the raw
# message bytes, which is the JSON half compressing well and the images not
# at all. Per channel, raw: depth 32.1 KiB/s, disparity 29.3, rectified_left
# 20.9, controller 12.5, vehicle 4.0, imu 3.3, diagnostics 1.1.
#
# Worth knowing which case is which. The service boots into manual with the
# camera shut, so a boot-and-drive run is the 4 MiB/hour one. 250 MiB/hour is
# auto mode, or manual with --telemetry-camera.

# Rotate a segment at this size. 128 MiB is about half an hour of auto
# driving at the measured 250 MiB/hour, and a file that size is a reasonable
# scp off the car over wifi. Small enough that pruning has somewhere useful
# to cut, large enough that rotations are rare -- each one closes a zstd
# writer, which flushes and writes a summary, and that is milliseconds on the
# publisher thread.
SEGMENT_BYTES = 128 * 1024 * 1024

# What the whole directory is allowed to hold: 64 full segments. At the
# measured camera-up rate that is about 33 hours of auto driving, and at the
# manual rate it is years. It is also 8% of the 99 GB free on this card,
# which leaves the other 92% for everything that is not telemetry. I would
# rather have 33 hours and a card I never think about than 400 hours and a
# number I have to go and check.
MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024

# The filesystem floor. Under this much free on the volume log/ lives on,
# recording stops and says so once. 4 GiB is far more headroom than apt or a
# journal rotation needs; the point is that the number is reached long before
# anything else on the Pi starts failing. This is the brake that works when
# the thing filling the card is not me.
MIN_FREE_BYTES = 4 * 1024 * 1024 * 1024

# How often tick() does any work at all. Everything it does -- two stats and
# a statvfs -- is syscalls on the publisher thread, so it is not something to
# do at the 100 Hz that thread drains at. Five seconds is a rounding error
# against a 256 MiB segment.
CHECK_INTERVAL = 5.0

# Target uncompressed bytes per zstd chunk. The default.
CHUNK_BYTES = 1024 * 1024

# zstd compression threads. The SDK's default is None, which means "as many
# as there are physical CPUs" -- four, on this Pi, on a machine running a
# 100 Hz control loop that already measures 75 Hz against it. One thread. The
# payload is mostly JPEG and already incompressible, so the throughput this
# gives up is throughput spent on nothing.
COMPRESSION_THREADS = 1

# How many consecutive write failures before giving up on recording for the
# rest of the run. A disk that has started refusing writes will refuse the
# next one too, and an error per message for an hour is its own outage.
MAX_WRITE_ERRORS = 20


def _repo_path(path: str) -> str:
    """Resolve a relative path against the checkout, like Logger does.

    Same reason: the unit sets WorkingDirectory, but that is the unit's
    promise and not a property of this code, and a recording that lands in
    whatever directory the process happened to start in is a recording I will
    not find.
    """
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def segment_name(when=None, pid=None, index=0) -> str:
    """The filename for one segment. Unique across runs and across reboots.

    Three parts, and each one is load-bearing:

      the UTC timestamp sorts the files and is the only part I read;

      the pid disambiguates, because this Pi has no RTC. At boot the clock is
      whatever the last shutdown wrote until NTP lands, so two boots can
      genuinely produce the same second, and the service restarting on
      failure can produce the same second as itself;

      the index separates the segments of one run.

    UTC, not local, so an hour of driving across a DST boundary still sorts.
    """
    when = time.gmtime(when if when is not None else time.time())
    pid = os.getpid() if pid is None else pid
    stamp = time.strftime("%Y%m%d-%H%M%S", when)
    return f"{PREFIX}{stamp}Z-p{pid}-{index:03d}{SUFFIX}"


class Recorder:
    """An MCAP sink over the telemetry channels, with a disk budget.

    Owned by Telemetry and driven entirely from its publisher thread:
    start() before the drain loop, tick() inside it, stop() after. The
    control thread never calls anything in here.

    A Recorder that failed to start is not a special case: tick() and stop()
    are no-ops and `active` is False.
    """

    def __init__(self, logger=None, directory=DEFAULT_DIR,
                 segment_bytes=SEGMENT_BYTES,
                 max_total_bytes=MAX_TOTAL_BYTES,
                 min_free_bytes=MIN_FREE_BYTES):
        self.logger = logger
        self.directory = _repo_path(directory)
        self.segment_bytes = segment_bytes
        self.max_total_bytes = max_total_bytes
        self.min_free_bytes = min_free_bytes

        self.active = False
        self.path = None
        self.error = None
        self.bytes_written = 0

        self._writer = None
        self._index = 0
        self._started_at = 0.0
        self._next_check = 0.0
        self._write_errors = 0
        # Set once, when the floor or a fatal open failure has been hit, so
        # the decision is not re-litigated and re-logged every five seconds.
        self._given_up = False

    # ---- lifecycle, on the publisher thread ------------------------------

    def start(self) -> bool:
        """Open the first segment. False means no recording, and that is fine.

        Nothing in here raises. A missing foxglove-sdk, a read-only log
        directory, a card that is already full: all of them cost the
        recording and nothing else.
        """
        if self.active:
            return True
        if self._given_up:
            return False

        try:
            os.makedirs(self.directory, exist_ok=True)
        except Exception as e:
            self._give_up(f"cannot make {self.directory}: {e}")
            return False

        free = self._free_bytes()
        if free is not None and free < self.min_free_bytes:
            self._give_up(
                f"only {_mib(free)} free on the card, not recording")
            return False

        self._prune()
        if not self._open_segment():
            return False

        self._started_at = time.monotonic()
        self._log("info", "Recording", f"Recording this run to {self.path}")
        return True

    def stop(self):
        """Close the current segment. Safe to call twice, and never raises."""
        writer, self._writer = self._writer, None
        self.active = False
        if writer is None:
            return
        try:
            writer.close()
        except Exception as e:
            self.error = str(e)
        if self.path is not None:
            size = _size(self.path) or 0
            seconds = time.monotonic() - self._started_at
            self._log("info", "Recording",
                      f"Closed {os.path.basename(self.path)}, {_mib(size)} "
                      f"over {seconds:.0f} s")

    # ---- the per-drain call ----------------------------------------------

    def tick(self, now: float) -> None:
        """Flush, rotate and prune if it is time. Never raises, never blocks.

        Called from the publisher thread's drain loop at 100 Hz, so the first
        line is the one that matters: almost every call returns on a float
        compare and does no syscall at all.
        """
        if not self.active:
            return
        if now < self._next_check:
            return
        self._next_check = now + CHECK_INTERVAL

        try:
            self._check(now)
        except Exception as e:
            # A stat that raced a prune, a statvfs on a volume that went
            # away. Not worth stopping a recording that is otherwise fine.
            self.error = str(e)

    def _flush(self) -> bool:
        """Close the open chunk, so what is on the card is what happened.

        This is not optional and it is the thing I got wrong first. zstd
        buffers a whole chunk before anything is written: after twenty
        seconds of recording, a measurement said 128 bytes on disk and
        24 003 bytes after close. Everything else was still in memory.

        Which is fine right up until the run ends the way the interesting
        runs end -- the car on its back with the battery out, or me pulling
        power because it would not stop. There is no close() on that path, so
        a whole chunk of the only minute I wanted is gone, and the file has
        no summary section either.

        Flushing every CHECK_INTERVAL bounds that loss to five seconds. The
        cost is a chunk boundary every five seconds instead of every 1 MiB,
        so with the camera up nothing changes at all (5 s is already about a
        megabyte) and with it down the chunks get small and compress a little
        worse -- on 1.2 KiB/s of JSON, which is not a number worth
        protecting.

        Returns whether it worked, and the caller cares: a flush that raised
        means this interval's bytes did NOT land, so the growth check in
        _check must not then forgive the error on the strength of the
        *previous* flush's bytes. Getting that order wrong made a card that
        had started refusing writes look healthy for ever, because every
        failed flush was pardoned by the last successful one. A test caught
        that, which is the only reason I know about it.
        """
        writer = self._writer
        if writer is None:
            return False
        try:
            writer.flush()
            return True
        except Exception as e:
            self.note_write_error(e)
            return False

    def _check(self, now: float) -> None:
        flushed = self._flush()
        if not self.active:
            # The flush was the write error that used up the last of
            # MAX_WRITE_ERRORS. Nothing below has anything to rotate.
            return

        free = self._free_bytes()
        if free is not None and free < self.min_free_bytes:
            # The floor, and it wins over everything. Pruning this directory
            # to nothing would not fix a card being filled by something else,
            # and a recording is the part of this process that is allowed to
            # lose.
            self.stop()
            self._give_up(f"only {_mib(free)} free on the card, stopped recording")
            return

        size = _size(self.path)
        if flushed and size is not None and size > self.bytes_written:
            # The file is growing, so whatever raised on the way through
            # channel.log() was not this. Forget the count: a laptop that
            # closed mid-frame raises on the same call a bad card does, and
            # the SDK does not say which sink failed, so the only honest test
            # is whether the bytes are landing.
            self._write_errors = 0
        self.bytes_written = size or 0
        if size is not None and size >= self.segment_bytes:
            self._rotate()

    def _rotate(self) -> None:
        """Close this segment and open the next one.

        Close first, then prune, then open. The gap between them is a window
        where no sink exists, so has_sinks() is false and the control loop
        builds nothing -- a handful of dropped samples at a segment boundary,
        which is the same answer this pipeline gives for a slow laptop. The
        alternative is opening the new writer before closing the old one, and
        that writes every message in the overlap into both files, which is a
        worse thing to hand a replay.
        """
        previous = self.path
        self.stop()
        self._prune()
        if self._open_segment():
            self._log("info", "Recording",
                      f"Rolled {os.path.basename(previous or '?')} over to "
                      f"{os.path.basename(self.path)}")

    def _open_segment(self) -> bool:
        if self._given_up:
            return False
        try:
            import foxglove
            from foxglove.mcap import MCAPCompression, MCAPWriteOptions
        except Exception as e:
            self._give_up(f"no foxglove-sdk, not recording: {e}")
            return False

        # A fresh index per attempt, so a name that somehow already exists is
        # stepped over rather than overwritten. allow_overwrite stays False:
        # the one thing worse than not recording this run is recording it on
        # top of the run I actually wanted.
        for _ in range(16):
            self._index += 1
            path = os.path.join(self.directory,
                                segment_name(index=self._index))
            if os.path.exists(path):
                continue
            try:
                self._writer = foxglove.open_mcap(
                    path,
                    writer_options=MCAPWriteOptions(
                        compression=MCAPCompression.Zstd,
                        compression_threads=COMPRESSION_THREADS,
                        chunk_size=CHUNK_BYTES,
                        profile="crawler",
                    ),
                )
            except Exception as e:
                self._give_up(f"cannot write {path}: {e}")
                return False

            self.path = path
            self.active = True
            self._write_errors = 0
            self._next_check = time.monotonic() + CHECK_INTERVAL
            return True

        self._give_up("every segment name I tried already exists")
        return False

    # ---- disk ------------------------------------------------------------

    def _segments(self):
        """Every segment in the directory, oldest name first, with its size.

        Sorted by name, which is the timestamp, which is also the order they
        were written -- except across a clock step, where the pid breaks the
        tie arbitrarily. Mtime would be the honest key, but mtime on the file
        being written moves, and pruning is only ever about the ones that are
        not being written.
        """
        out = []
        try:
            names = sorted(os.listdir(self.directory))
        except Exception:
            return out
        for name in names:
            if not name.startswith(PREFIX) or not name.endswith(SUFFIX):
                continue
            path = os.path.join(self.directory, name)
            size = _size(path)
            if size is None:
                continue
            out.append((path, size))
        return out

    def _prune(self) -> None:
        """Delete oldest segments until the directory is inside its budget.

        Never the file currently being written -- self.path is excluded even
        though by the time this runs it is normally already closed, because
        deleting the open file would leave the writer appending to an inode
        with no name and the run would vanish on close.
        """
        segments = self._segments()
        total = sum(size for _, size in segments)
        if total <= self.max_total_bytes:
            return

        freed = 0
        removed = 0
        for path, size in segments:
            if total <= self.max_total_bytes:
                break
            if self.active and path == self.path:
                continue
            try:
                os.remove(path)
            except Exception as e:
                self.error = str(e)
                continue
            total -= size
            freed += size
            removed += 1

        if removed:
            self._log("info", "Recording",
                      f"Pruned {removed} old recording(s), {_mib(freed)} back")

    def _free_bytes(self):
        try:
            stats = os.statvfs(self.directory)
        except Exception:
            return None
        return stats.f_bavail * stats.f_frsize

    # ---- failure ---------------------------------------------------------

    def note_write_error(self, error) -> None:
        """Telemetry calls this when channel.log() raised.

        A write error here is almost always the card, and a card that refused
        one write will refuse the next. Count them and give up rather than
        log one per message for the rest of the run.
        """
        self.error = str(error)
        self._write_errors += 1
        if self._write_errors >= MAX_WRITE_ERRORS:
            self.stop()
            self._give_up(f"{self._write_errors} write errors, stopped "
                          f"recording: {error}")

    def _give_up(self, message) -> None:
        self._given_up = True
        self.active = False
        self.error = message
        self._log("warning", "Recording", message)

    def _log(self, level, subject, message):
        if self.logger is not None:
            try:
                self.logger.log(level, subject, message)
                return
            except Exception:
                pass
        print(f"[recording] {subject}: {message}")


def _size(path):
    if not path:
        return None
    try:
        return os.path.getsize(path)
    except Exception:
        return None


def _mib(n) -> str:
    return f"{n / (1024.0 * 1024.0):.1f} MiB"
