"""The recorder's policy, tested off the car and without the SDK.

The module imports foxglove inside _open_segment and nowhere else, so it
imports fine on a laptop that has never had the SDK. These tests put a fake
one in sys.modules, which means what is under test is the part that is
actually mine: the naming, the rotation, the disk budget, the filesystem
floor, and -- the one that matters more than the rest put together -- that
none of it can raise.

Why that last one is the point. pigpio latches a servo pulse width and emits
it forever, and the control loop is the only thing that can take it back to
neutral. A recorder that throws out of tick() kills the publisher thread; a
recorder that throws out of start() kills the process before the loop exists.
Either one is a car that keeps driving with nothing at the wheel. So every
test below that ends in "does not raise" is a test about the throttle.
"""

import os
import sys
import types

import pytest

import recording


class FakeLogger:
    def __init__(self):
        self.lines = []

    def log(self, level, subject, message):
        self.lines.append((level, subject, message))

    def warnings(self):
        return [line for line in self.lines if line[0] == "warning"]


class FakeWriter:
    """Stands in for foxglove.MCAPWriter. Counts, and can be told to break."""

    def __init__(self, path, fail_flush=False, fail_close=False):
        self.path = path
        self.flushes = 0
        self.closed = False
        self.fail_flush = fail_flush
        self.fail_close = fail_close

    def flush(self):
        self.flushes += 1
        if self.fail_flush:
            raise OSError("No space left on device")

    def close(self):
        self.closed = True
        if self.fail_close:
            raise OSError("broken close")


@pytest.fixture
def fake_sdk(monkeypatch):
    """A foxglove module whose open_mcap touches the real filesystem.

    It creates the file, because rotation and pruning work off
    os.path.getsize of a path that has to exist. Everything else about the
    SDK is irrelevant in here.
    """
    opened = []

    def open_mcap(path, writer_options=None):
        if open_mcap.fail:
            raise open_mcap.fail
        with open(path, "wb") as handle:
            handle.write(b"\x89MCAP0\r\n")
        writer = FakeWriter(path, fail_flush=open_mcap.fail_flush)
        opened.append(writer)
        return writer

    open_mcap.fail = None
    open_mcap.fail_flush = False
    open_mcap.opened = opened

    mcap_module = types.ModuleType("foxglove.mcap")
    mcap_module.MCAPCompression = types.SimpleNamespace(Zstd="zstd")
    mcap_module.MCAPWriteOptions = lambda **kwargs: kwargs

    foxglove_module = types.ModuleType("foxglove")
    foxglove_module.open_mcap = open_mcap
    foxglove_module.mcap = mcap_module

    monkeypatch.setitem(sys.modules, "foxglove", foxglove_module)
    monkeypatch.setitem(sys.modules, "foxglove.mcap", mcap_module)
    return open_mcap


def make(tmp_path, logger=None, **kwargs):
    kwargs.setdefault("directory", str(tmp_path / "mcap"))
    return recording.Recorder(logger=logger or FakeLogger(), **kwargs)


def grow(path, size):
    """Make a file exactly `size` bytes, the way a writer would."""
    with open(path, "wb") as handle:
        handle.truncate(size)


# ---- naming ----------------------------------------------------------------


def test_segment_name_carries_the_pid_so_two_boots_cannot_collide():
    # This Pi has no RTC: at boot the clock is whatever the last shutdown
    # wrote until NTP lands, so two boots genuinely can produce the same
    # second. The pid is what makes the name unique anyway.
    when = 1_700_000_000
    first = recording.segment_name(when=when, pid=1234, index=1)
    second = recording.segment_name(when=when, pid=5678, index=1)
    assert first != second
    assert "p1234" in first and "p5678" in second


def test_segment_names_sort_in_time_order():
    early = recording.segment_name(when=1_700_000_000, pid=1, index=1)
    late = recording.segment_name(when=1_700_003_600, pid=1, index=1)
    assert sorted([late, early]) == [early, late]


def test_segment_name_is_utc_so_a_dst_step_still_sorts():
    # 1700000000 is 2023-11-14 22:13:20 UTC. If this ever reads as local
    # time, an hour of driving across a DST boundary stops sorting.
    assert recording.segment_name(when=1_700_000_000, pid=1, index=0) == \
        "crawler-20231114-221320Z-p1-000.mcap"


# ---- the happy path --------------------------------------------------------


def test_start_opens_a_segment_and_says_so(tmp_path, fake_sdk):
    logger = FakeLogger()
    recorder = make(tmp_path, logger)
    assert recorder.start() is True
    assert recorder.active is True
    assert os.path.exists(recorder.path)
    assert any("Recording this run" in line[2] for line in logger.lines)


def test_stop_closes_the_writer_and_is_safe_twice(tmp_path, fake_sdk):
    recorder = make(tmp_path, FakeLogger())
    recorder.start()
    writer = fake_sdk.opened[-1]
    recorder.stop()
    assert writer.closed is True
    assert recorder.active is False
    recorder.stop()  # must not raise


def test_tick_flushes_so_the_bytes_are_on_the_card(tmp_path, fake_sdk):
    # zstd buffers a whole chunk before writing anything, and a run that ends
    # with the battery out never reaches close(). The flush is the only
    # reason the file holds anything at all when that happens.
    recorder = make(tmp_path, FakeLogger())
    recorder.start()
    writer = fake_sdk.opened[-1]
    assert writer.flushes == 0
    recorder.tick(recorder._next_check + 1.0)
    assert writer.flushes == 1


def test_tick_does_nothing_before_its_interval(tmp_path, fake_sdk):
    # It is called from the publisher thread's 100 Hz drain loop, so the
    # cheap path has to be genuinely cheap: a float compare and return.
    recorder = make(tmp_path, FakeLogger())
    recorder.start()
    writer = fake_sdk.opened[-1]
    recorder.tick(recorder._next_check - 1.0)
    assert writer.flushes == 0


# ---- rotation --------------------------------------------------------------


def test_a_full_segment_rolls_over_to_a_new_file(tmp_path, fake_sdk):
    recorder = make(tmp_path, FakeLogger(), segment_bytes=1024)
    recorder.start()
    first = recorder.path
    grow(first, 2048)
    recorder.tick(recorder._next_check + 1.0)
    assert recorder.active is True
    assert recorder.path != first
    assert os.path.exists(first)
    assert os.path.exists(recorder.path)


def test_a_segment_under_its_cap_is_left_alone(tmp_path, fake_sdk):
    recorder = make(tmp_path, FakeLogger(), segment_bytes=1024)
    recorder.start()
    first = recorder.path
    grow(first, 512)
    recorder.tick(recorder._next_check + 1.0)
    assert recorder.path == first


# ---- the disk budget -------------------------------------------------------


def test_pruning_deletes_the_oldest_until_it_is_inside_the_budget(tmp_path,
                                                                 fake_sdk):
    directory = tmp_path / "mcap"
    directory.mkdir()
    for n in range(1, 6):
        path = directory / recording.segment_name(
            when=1_700_000_000 + n * 60, pid=1, index=n)
        grow(str(path), 1000)

    recorder = make(tmp_path, FakeLogger(), max_total_bytes=2500)
    recorder._prune()

    left = sorted(p.name for p in directory.iterdir())
    assert len(left) == 2, left
    # The two newest survive, and they are the two with the latest stamps.
    assert "-004.mcap" in left[0] and "-005.mcap" in left[1]


def test_pruning_never_deletes_the_file_being_written(tmp_path, fake_sdk):
    recorder = make(tmp_path, FakeLogger(), max_total_bytes=10)
    recorder.start()
    grow(recorder.path, 5000)
    recorder._prune()
    # Deleting the open file would leave the writer appending to an inode
    # with no name, and the run would vanish on close.
    assert os.path.exists(recorder.path)


def test_pruning_ignores_files_that_are_not_mine(tmp_path, fake_sdk):
    directory = tmp_path / "mcap"
    directory.mkdir()
    grow(str(directory / "notes.txt"), 9999)
    grow(str(directory / recording.segment_name(pid=1, index=1)), 9999)
    recorder = make(tmp_path, FakeLogger(), max_total_bytes=10)
    recorder._prune()
    assert (directory / "notes.txt").exists()


def test_pruning_runs_before_the_first_segment_of_a_run(tmp_path, fake_sdk):
    # Otherwise a car that is power-cycled over and over never prunes: every
    # run is short, no segment ever fills, and the directory only grows.
    directory = tmp_path / "mcap"
    directory.mkdir()
    for n in range(1, 4):
        grow(str(directory / recording.segment_name(
            when=1_700_000_000 + n * 60, pid=1, index=n)), 1000)
    recorder = make(tmp_path, FakeLogger(), max_total_bytes=1500)
    recorder.start()
    names = [p.name for p in directory.iterdir()]
    assert len(names) == 2, names  # one survivor plus the new segment


# ---- the filesystem floor --------------------------------------------------


def test_a_card_with_no_room_is_not_recorded_to_at_all(tmp_path, fake_sdk,
                                                       monkeypatch):
    logger = FakeLogger()
    recorder = make(tmp_path, logger)
    monkeypatch.setattr(recorder, "_free_bytes", lambda: 1024)
    assert recorder.start() is False
    assert recorder.active is False
    assert any("free on the card" in line[2] for line in logger.warnings())


def test_the_floor_stops_a_running_recording(tmp_path, fake_sdk, monkeypatch):
    logger = FakeLogger()
    recorder = make(tmp_path, logger)
    recorder.start()
    writer = fake_sdk.opened[-1]
    monkeypatch.setattr(recorder, "_free_bytes", lambda: 1024)
    recorder.tick(recorder._next_check + 1.0)
    assert recorder.active is False
    assert writer.closed is True


def test_the_floor_beats_the_directory_budget(tmp_path, fake_sdk, monkeypatch):
    # Pruning my own directory cannot fix a card being filled by something
    # else, so the floor is checked first and wins. The failure mode this
    # guards against is a recorder that happily prunes to zero and keeps
    # writing while the filesystem dies under it.
    recorder = make(tmp_path, FakeLogger(), max_total_bytes=10 ** 12)
    recorder.start()
    monkeypatch.setattr(recorder, "_free_bytes", lambda: 1024)
    recorder.tick(recorder._next_check + 1.0)
    assert recorder.active is False


def test_giving_up_is_said_once_and_not_retried(tmp_path, fake_sdk,
                                                monkeypatch):
    logger = FakeLogger()
    recorder = make(tmp_path, logger)
    monkeypatch.setattr(recorder, "_free_bytes", lambda: 1024)
    recorder.start()
    recorder.start()
    recorder.start()
    assert len(logger.warnings()) == 1


# ---- write errors ----------------------------------------------------------


def test_enough_write_errors_gives_up_rather_than_logging_forever(tmp_path,
                                                                  fake_sdk):
    logger = FakeLogger()
    recorder = make(tmp_path, logger)
    recorder.start()
    for _ in range(recording.MAX_WRITE_ERRORS):
        recorder.note_write_error(OSError("No space left on device"))
    assert recorder.active is False
    assert any("write errors" in line[2] for line in logger.warnings())


def test_a_few_write_errors_do_not_stop_a_recording(tmp_path, fake_sdk):
    recorder = make(tmp_path, FakeLogger())
    recorder.start()
    for _ in range(recording.MAX_WRITE_ERRORS - 1):
        recorder.note_write_error(OSError("client went away"))
    assert recorder.active is True


def test_a_growing_file_forgets_the_write_errors(tmp_path, fake_sdk):
    # channel.log() writes to every sink and does not say which one failed, so
    # a laptop closing mid-frame raises on the same call a bad card does. The
    # only honest test of which it was is whether the bytes are landing.
    recorder = make(tmp_path, FakeLogger(), segment_bytes=10 ** 9)
    recorder.start()
    for _ in range(recording.MAX_WRITE_ERRORS - 1):
        recorder.note_write_error(OSError("client went away"))
    grow(recorder.path, 50_000)
    recorder.tick(recorder._next_check + 1.0)
    assert recorder._write_errors == 0
    recorder.note_write_error(OSError("client went away again"))
    assert recorder.active is True


def test_a_flush_that_fails_is_counted_not_raised(tmp_path, fake_sdk):
    fake_sdk.fail_flush = True
    recorder = make(tmp_path, FakeLogger())
    recorder.start()
    recorder.tick(recorder._next_check + 1.0)  # must not raise
    assert recorder._write_errors == 1


# ---- nothing may raise -----------------------------------------------------


def test_a_directory_that_cannot_be_made_costs_the_recording_only(tmp_path,
                                                                  fake_sdk):
    # A file where the directory should be. os.makedirs raises NotADirectory,
    # and the car has to keep driving.
    blocker = tmp_path / "mcap"
    blocker.write_text("not a directory")
    recorder = recording.Recorder(logger=FakeLogger(),
                                  directory=str(blocker / "deeper"))
    assert recorder.start() is False
    recorder.tick(1.0)
    recorder.stop()


def test_an_sdk_that_is_not_installed_costs_the_recording_only(tmp_path,
                                                              monkeypatch):
    monkeypatch.setitem(sys.modules, "foxglove", None)
    logger = FakeLogger()
    recorder = make(tmp_path, logger)
    assert recorder.start() is False
    assert any("foxglove" in line[2] for line in logger.warnings())


def test_an_open_that_raises_costs_the_recording_only(tmp_path, fake_sdk):
    fake_sdk.fail = PermissionError("read-only file system")
    logger = FakeLogger()
    recorder = make(tmp_path, logger)
    assert recorder.start() is False
    assert recorder.active is False
    assert any("read-only" in line[2] for line in logger.warnings())


def test_a_close_that_raises_still_leaves_the_recorder_stopped(tmp_path,
                                                               fake_sdk):
    recorder = make(tmp_path, FakeLogger())
    recorder.start()
    fake_sdk.opened[-1].fail_close = True
    recorder.stop()  # must not raise
    assert recorder.active is False


def test_an_undeletable_old_segment_does_not_stop_the_prune(tmp_path,
                                                            fake_sdk,
                                                            monkeypatch):
    directory = tmp_path / "mcap"
    directory.mkdir()
    for n in range(1, 4):
        grow(str(directory / recording.segment_name(
            when=1_700_000_000 + n * 60, pid=1, index=n)), 1000)

    real_remove = os.remove
    calls = []

    def flaky_remove(path):
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError("not yours")
        return real_remove(path)

    monkeypatch.setattr(os, "remove", flaky_remove)
    recorder = make(tmp_path, FakeLogger(), max_total_bytes=1000)
    recorder._prune()
    # It skipped the one it could not delete and carried on to the next.
    assert len(calls) >= 2


def test_a_statvfs_that_raises_does_not_stop_a_recording(tmp_path, fake_sdk,
                                                         monkeypatch):
    recorder = make(tmp_path, FakeLogger())
    recorder.start()

    def boom():
        raise OSError("volume went away")

    monkeypatch.setattr(recorder, "_free_bytes", boom)
    recorder.tick(recorder._next_check + 1.0)  # must not raise
    # Not being able to tell how much room is left is not a reason to throw a
    # run away, so the recording carries on.
    assert recorder.active is True


def test_no_logger_at_all_still_works(tmp_path, fake_sdk, capsys):
    recorder = recording.Recorder(logger=None,
                                  directory=str(tmp_path / "mcap"))
    assert recorder.start() is True
    recorder.stop()
    assert "[recording]" in capsys.readouterr().out


def test_a_relative_directory_is_anchored_to_the_checkout():
    # Same reason Logger._prepare does it: the unit sets WorkingDirectory, but
    # that is the unit's promise and not a property of this code, and a
    # recording that lands wherever the process started is one I cannot find.
    recorder = recording.Recorder(directory="log/mcap")
    assert os.path.isabs(recorder.directory)
    assert recorder.directory.endswith(os.path.join("log", "mcap"))
    assert recorder.directory.startswith(
        os.path.dirname(os.path.abspath(recording.__file__)))
