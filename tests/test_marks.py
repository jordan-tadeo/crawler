"""The mark buttons, tested off the car.

What is worth testing in here splits three ways.

The behaviour he asked for: A is the good mark, B is the bad one, a hold is
one mark and not a stream, and the window a mark stands for is the four
seconds before the press and the one after it.

The snapshot: a mark made in manual mode has no driver and no camera to ask,
and a mark made in auto has to carry the scan, the block reason and the
recover phase -- the things that would let me tune driver.py from a recording
instead of from memory.

And the part that matters more than either, because pigpio latches the last
servo pulse width and emits it forever: none of this is allowed to cost a tick
of the control loop. A pad whose getters raise, a telemetry mailbox that
raises, a rumble the pad refuses, a card that stalls mid-write -- every one of
those costs the mark and nothing else. Every test below that ends in "does not
raise" or "does not block" is a test about the throttle.
"""

import os
import threading
import time
import types

import pytest

import marks as mk


class FakeLogger:
    def __init__(self):
        self.lines = []

    def log(self, level, subject, message):
        self.lines.append((level, subject, message))

    def of(self, level):
        return [line for line in self.lines if line[0] == level]


class FakeHandle:
    """The pygame joystick handle, as far as rumble() is concerned."""

    def __init__(self, result=True, raises=None, cost=0.0):
        self.calls = []
        self.result = result
        self.raises = raises
        self.cost = cost

    def rumble(self, low, high, duration):
        self.calls.append((low, high, duration))
        if self.cost:
            time.sleep(self.cost)
        if self.raises is not None:
            raise self.raises
        return self.result


class FakePad:
    """A controller. Buttons are whatever I last set them to."""

    def __init__(self, handle=None, raises=False):
        self.down = {}
        self.joystick = handle if handle is not None else FakeHandle()
        self.raises = raises

    def get_button(self, name):
        if self.raises:
            raise RuntimeError("stale handle")
        return self.down.get(name, False)

    def press(self, *names):
        self.down = {name: True for name in names}

    def release(self):
        self.down = {}


class FakeVecon:
    def __init__(self):
        self.last_throttle = 0.2
        self.last_steering_front = -0.3
        self.last_steering_rear = 0.1
        self.last_esc_pulse = 1620


class FakeAutopilot:
    def __init__(self, state="ready"):
        self.state = state

    def camera_state(self):
        return self.state


def fake_scan(distances, blocked):
    return types.SimpleNamespace(
        distance_mm=list(distances),
        valid_fraction=[1.0] * len(distances),
        blocked=list(blocked),
        reason=tuple("" for _ in distances),
        geometry=None,
    )


class FakeDriver:
    def __init__(self):
        self.last_scan = None
        self.last_scan_ms = 2.5
        self.last_block_reason = ""
        self._phase = None
        self._phase_until = 0.0
        self.gyro_history = []
        self.imu_history = []
        self.stuck_threshold = 0.2


class FakeTele:
    def __init__(self, raises=False):
        self.offers = []
        self.raises = raises

    def offer(self, key, payload, now):
        if self.raises:
            raise RuntimeError("no mailbox")
        self.offers.append((key, payload, now))


@pytest.fixture
def pad():
    return FakePad()


@pytest.fixture
def markpad(tmp_path):
    """A MarkPad writing its CSV somewhere disposable."""
    board = mk.MarkPad(logger=FakeLogger(), tele=FakeTele(),
                       csv_path=str(tmp_path / "marks.csv"))
    yield board
    board.stop()


# ---- the behaviour he asked for ------------------------------------------


def test_a_is_good_and_b_is_bad(markpad, pad):
    pad.press("A")
    made = markpad.poll(pad, 10.0, "auto")
    assert [m["kind"] for m in made] == [mk.GOOD]
    assert made[0]["button"] == "A"

    pad.release()
    markpad.poll(pad, 10.1, "auto")

    pad.press("B")
    made = markpad.poll(pad, 10.2, "auto")
    assert [m["kind"] for m in made] == [mk.BAD]
    assert made[0]["button"] == "B"


def test_a_held_down_marks_exactly_once(markpad, pad):
    pad.press("A")
    assert len(markpad.poll(pad, 1.0, "auto")) == 1
    for tick in range(50):
        assert markpad.poll(pad, 1.01 + tick * 0.01, "auto") == ()
    assert markpad.good_count == 1


def test_releasing_and_pressing_again_is_a_second_mark(markpad, pad):
    pad.press("A")
    markpad.poll(pad, 1.0, "auto")
    pad.release()
    markpad.poll(pad, 1.1, "auto")
    pad.press("A")
    markpad.poll(pad, 1.2, "auto")
    assert markpad.good_count == 2


def test_both_buttons_in_one_tick_make_both_marks(markpad, pad):
    pad.press("A", "B")
    made = markpad.poll(pad, 1.0, "auto")
    assert [m["kind"] for m in made] == [mk.GOOD, mk.BAD]
    assert markpad.counts() == {"marks_good": 1, "marks_bad": 1}


def test_a_mark_is_a_window_not_an_instant(markpad, pad):
    pad.press("B")
    mark, = markpad.poll(pad, 100.0, "auto")
    assert mark["window_start_s"] == 100.0 - 4.0
    assert mark["window_end_s"] == 100.0 + 1.0
    assert mark["lookback_s"] == 4.0
    assert mark["lead_s"] == 1.0


def test_the_lookback_covers_a_whole_recovery_cycle():
    """reverse 2.0 + settle 0.5, and then the approach that caused it."""
    import driver as dv

    assert mk.MARK_LOOKBACK_SECONDS > dv.REVERSE_SECONDS + 0.5


def test_marking_works_in_manual_with_no_driver(markpad, pad):
    pad.press("A")
    mark, = markpad.poll(pad, 5.0, "manual", FakeVecon(), None, None)
    assert mark["mode"] == "manual"
    assert mark["throttle"] == pytest.approx(0.2)
    assert mark["esc_pulse_us"] == 1620
    # Nothing to say about the camera's half of it, and that is the answer.
    assert mark["block_reason"] == ""
    assert mark["near_distance_mm"] == -1.0
    assert mark["recovery_phase"] == ""
    assert mark["motion_evidence"] == "unknown"


def test_the_mark_records_which_mode_it_was_made_in(markpad, pad):
    pad.press("A")
    mark, = markpad.poll(pad, 1.0, "auto")
    assert mark["mode"] == "auto"


# ---- the snapshot --------------------------------------------------------


def test_an_auto_mark_carries_the_scan_and_the_block_reason(markpad, pad):
    driver = FakeDriver()
    driver.last_block_reason = "near+no_return"
    driver.last_scan = fake_scan([900.0, 412.0, float("inf")],
                                 [False, True, False])

    pad.press("B")
    mark, = markpad.poll(pad, 7.0, "auto", FakeVecon(), driver,
                         FakeAutopilot())

    assert mark["block_reason"] == "near+no_return"
    assert mark["scan_blocked"] is True
    assert mark["near_distance_mm"] == pytest.approx(412.0)
    assert mark["scan_ms"] == pytest.approx(2.5)
    assert mark["camera_state"] == "ready"


def test_an_unmeasurable_probe_becomes_minus_one_not_inf(markpad, pad):
    """inf is the right answer in driver.py and not one JSON will carry."""
    driver = FakeDriver()
    driver.last_scan = fake_scan([float("inf")] * 3, [False] * 3)

    pad.press("A")
    mark, = markpad.poll(pad, 1.0, "auto", None, driver, None)

    assert mark["probe_distance_mm"] == [-1.0, -1.0, -1.0]
    assert mark["near_distance_mm"] == -1.0


def test_a_mark_during_a_recover_says_which_phase(markpad, pad):
    driver = FakeDriver()
    driver._phase = "reverse"
    driver._phase_until = 12.4

    pad.press("B")
    mark, = markpad.poll(pad, 11.0, "auto", None, driver, None)

    assert mark["recovery_phase"] == "reverse"
    assert mark["recovery_remaining_s"] == pytest.approx(1.4)


def test_motion_evidence_does_not_feed_the_stuck_detector(markpad, pad):
    """The whole reason this does not call driver.motion_evidence()."""
    import driver as dv

    driver = FakeDriver()
    driver.gyro_history = [0.5, 0.5]
    driver.imu_history = [0.01, 0.01]

    pad.press("A")
    mark, = markpad.poll(pad, 1.0, "auto", None, driver, None)

    assert mark["motion_evidence"] == dv.MOVING
    assert mark["gyro_mean"] == pytest.approx(0.5)
    # Pressing a button must not add a sample to either window.
    assert driver.gyro_history == [0.5, 0.5]
    assert driver.imu_history == [0.01, 0.01]


def test_a_still_car_reads_still(markpad, pad):
    import driver as dv

    driver = FakeDriver()
    driver.gyro_history = [0.001]
    driver.imu_history = [0.01]

    pad.press("A")
    mark, = markpad.poll(pad, 1.0, "auto", None, driver, None)
    assert mark["motion_evidence"] == dv.STILL


def test_the_mark_carries_both_clocks(markpad, pad):
    pad.press("A")
    mark, = markpad.poll(pad, 42.0, "auto")
    assert mark["mono_s"] == pytest.approx(42.0)
    assert mark["utc"].endswith("Z")
    assert len(mark["utc"]) == 20


# ---- the throttle's half: none of this may cost a tick -------------------


def test_a_pad_that_raises_costs_the_mark_not_the_tick(markpad):
    bad = FakePad(raises=True)
    assert markpad.poll(bad, 1.0, "auto") == ()
    assert markpad.good_count == 0


def test_a_pad_that_starts_raising_mid_hold_does_not_double_mark(markpad):
    pad = FakePad()
    pad.press("A")
    assert len(markpad.poll(pad, 1.0, "auto")) == 1
    pad.raises = True
    markpad.poll(pad, 1.1, "auto")
    pad.raises = False
    # The handle came back with the button still down. One press, one mark --
    # but the memory was cleared, so this is honestly a new edge.
    assert len(markpad.poll(pad, 1.2, "auto")) == 1


def test_telemetry_that_raises_still_counts_and_writes_the_mark(tmp_path, pad):
    board = mk.MarkPad(logger=FakeLogger(), tele=FakeTele(raises=True),
                       csv_path=str(tmp_path / "marks.csv"))
    try:
        pad.press("A")
        made = board.poll(pad, 1.0, "auto")
        assert len(made) == 1
        assert board.good_count == 1
        board.flush()
        assert (tmp_path / "marks.csv").exists()
    finally:
        board.stop()


def test_a_driver_whose_attributes_raise_still_makes_a_mark(markpad, pad):
    class Exploding:
        def __getattr__(self, name):
            raise RuntimeError("mid teardown")

    pad.press("B")
    made = markpad.poll(pad, 1.0, "auto", None, Exploding(), None)
    assert len(made) == 1
    assert made[0]["block_reason"] == ""


def test_poll_does_not_wait_on_the_card(tmp_path, pad):
    """The CSV write is on the writer thread, so a stall is not the loop's."""
    board = mk.MarkPad(logger=FakeLogger(), tele=FakeTele(),
                       csv_path=str(tmp_path / "marks.csv"))
    held = threading.Event()
    board._append = lambda row: held.wait(5.0)
    try:
        pad.press("A")
        started = time.monotonic()
        board.poll(pad, 1.0, "auto")
        cost = time.monotonic() - started
        assert cost < 0.05, f"poll waited {cost * 1000:.0f} ms on the write"
    finally:
        held.set()
        board.stop()


def test_no_file_is_touched_by_a_run_with_no_marks(tmp_path, pad):
    board = mk.MarkPad(logger=FakeLogger(), tele=FakeTele(),
                       csv_path=str(tmp_path / "marks.csv"))
    try:
        for tick in range(100):
            board.poll(pad, tick * 0.01, "manual")
        assert not (tmp_path / "marks.csv").exists()
    finally:
        board.stop()


def test_the_writer_thread_is_not_started_by_the_control_loop(tmp_path):
    """Thread.start() waits for the thread to come up. Not on that thread."""
    board = mk.MarkPad(logger=FakeLogger(), tele=FakeTele(),
                       csv_path=str(tmp_path / "marks.csv"))
    try:
        assert board._writer is not None
        assert board._writer.is_alive()
        assert board._writer.daemon is True
    finally:
        board.stop()


def test_two_marks_in_one_tick_both_reach_the_channel(markpad, pad):
    """telemetry's mailbox is one deep, so they go out a tick apart."""
    pad.press("A", "B")
    markpad.poll(pad, 1.0, "auto")
    assert [p["kind"] for _, p, _ in markpad.tele.offers] == [mk.GOOD]

    pad.release()
    markpad.poll(pad, 1.01, "auto")
    assert [p["kind"] for _, p, _ in markpad.tele.offers] == [mk.GOOD, mk.BAD]
    # And the late one still carries the instant the button actually went down.
    assert markpad.tele.offers[1][1]["mono_s"] == pytest.approx(1.0)


# ---- the CSV sidecar -----------------------------------------------------


def test_the_csv_gets_one_header_and_a_row_per_mark(tmp_path, pad):
    path = tmp_path / "marks.csv"
    board = mk.MarkPad(logger=FakeLogger(), tele=FakeTele(),
                       csv_path=str(path))
    try:
        pad.press("A")
        board.poll(pad, 1.0, "auto")
        pad.release()
        board.poll(pad, 1.1, "auto")
        pad.press("B")
        board.poll(pad, 1.2, "auto")
        board.flush()
    finally:
        board.stop()

    rows = path.read_text().strip().splitlines()
    assert rows[0].split(",") == list(mk.CSV_FIELDS)
    assert len(rows) == 3
    assert ",good," in rows[1]
    assert ",bad," in rows[2]


def test_the_csv_is_append_only_across_runs(tmp_path, pad):
    path = tmp_path / "marks.csv"
    for run in range(2):
        board = mk.MarkPad(logger=FakeLogger(), tele=FakeTele(),
                           csv_path=str(path))
        try:
            pad.press("A")
            board.poll(pad, 1.0 + run, "auto")
            board.flush()
        finally:
            board.stop()
        pad.release()

    rows = path.read_text().strip().splitlines()
    assert len(rows) == 3  # one header, two marks
    assert rows[0].startswith("utc,")


def test_the_csv_folder_makes_itself(tmp_path, pad):
    path = tmp_path / "nested" / "deeper" / "marks.csv"
    board = mk.MarkPad(logger=FakeLogger(), tele=FakeTele(),
                       csv_path=str(path))
    try:
        pad.press("A")
        board.poll(pad, 1.0, "auto")
        board.flush()
    finally:
        board.stop()
    assert path.exists()


def test_a_mark_gets_a_journal_line(tmp_path, pad):
    log = FakeLogger()
    board = mk.MarkPad(logger=log, tele=FakeTele(),
                       csv_path=str(tmp_path / "marks.csv"))
    try:
        pad.press("B")
        board.poll(pad, 1.0, "auto")
        board.flush()
    finally:
        board.stop()

    lines = [line for line in log.of("info") if line[1] == "Mark"]
    assert len(lines) == 1
    assert "BAD #1" in lines[0][2]


# ---- rumble --------------------------------------------------------------


def test_good_buzzes_once(markpad, pad):
    pad.press("A")
    markpad.poll(pad, 1.0, "auto")
    assert len(pad.joystick.calls) == 1
    # And nothing more, however long I keep polling.
    pad.release()
    for tick in range(100):
        markpad.poll(pad, 1.0 + tick * 0.05, "auto")
    assert len(pad.joystick.calls) == 1


def test_bad_buzzes_twice_without_anybody_sleeping(markpad, pad):
    pad.press("B")
    markpad.poll(pad, 1.0, "auto")
    assert len(pad.joystick.calls) == 1, "the first pulse is immediate"

    pad.release()
    # The gap is spaced by the control loop's own clock, not by a sleep.
    markpad.poll(pad, 1.05, "auto")
    assert len(pad.joystick.calls) == 1, "too early for the second"

    markpad.poll(pad, 1.4, "auto")
    assert len(pad.joystick.calls) == 2

    for tick in range(100):
        markpad.poll(pad, 1.5 + tick * 0.05, "auto")
    assert len(pad.joystick.calls) == 2


def test_a_rumble_that_raises_costs_only_the_buzz(tmp_path):
    log = FakeLogger()
    pad = FakePad(handle=FakeHandle(raises=RuntimeError("no haptics")))
    board = mk.MarkPad(logger=log, tele=FakeTele(),
                       csv_path=str(tmp_path / "marks.csv"))
    try:
        pad.press("A")
        assert len(board.poll(pad, 1.0, "auto")) == 1
        assert board.good_count == 1
        assert board._rumble_ok is False

        # Said once, and then never again however many marks follow.
        pad.release()
        board.poll(pad, 1.1, "auto")
        pad.press("B")
        board.poll(pad, 1.2, "auto")
        notes = [line for line in log.of("info")
                 if "without a buzz" in line[2]]
        assert len(notes) == 1
        assert len(pad.joystick.calls) == 1
    finally:
        board.stop()


def test_a_pad_with_no_rumble_support_is_noted_once(tmp_path):
    log = FakeLogger()
    pad = FakePad(handle=FakeHandle(result=False))
    board = mk.MarkPad(logger=log, tele=FakeTele(),
                       csv_path=str(tmp_path / "marks.csv"))
    try:
        pad.press("A")
        board.poll(pad, 1.0, "auto")
        assert board._rumble_ok is False
        assert any("no rumble support" in line[2] for line in log.of("info"))
    finally:
        board.stop()


def test_a_slow_rumble_switches_rumble_off(tmp_path):
    """If rumble() ever blocks, it costs one tick and never a second."""
    log = FakeLogger()
    cost = (mk.RUMBLE_SLOW_MS + 20.0) / 1000.0
    pad = FakePad(handle=FakeHandle(cost=cost))
    board = mk.MarkPad(logger=log, tele=FakeTele(),
                       csv_path=str(tmp_path / "marks.csv"))
    try:
        pad.press("B")
        board.poll(pad, 1.0, "auto")
        assert board._rumble_ok is False
        assert len(pad.joystick.calls) == 1, "the second pulse was dropped"
        assert any("cannot pay" in line[2] for line in log.of("info"))
    finally:
        board.stop()


def test_a_pad_with_no_handle_at_all_still_marks(tmp_path):
    pad = FakePad()
    pad.joystick = None
    board = mk.MarkPad(logger=FakeLogger(), tele=FakeTele(),
                       csv_path=str(tmp_path / "marks.csv"))
    try:
        pad.press("A")
        assert len(board.poll(pad, 1.0, "auto")) == 1
    finally:
        board.stop()


# ---- the channel and the counts ------------------------------------------


def test_the_mark_is_offered_on_the_marks_channel(markpad, pad):
    pad.press("A")
    markpad.poll(pad, 3.0, "auto")
    keys = [key for key, _, _ in markpad.tele.offers]
    assert keys == ["marks"]


def test_marks_is_a_real_telemetry_channel():
    """A channel the recorder picks up needs to be in the registry."""
    import telemetry as tm

    assert tm.JSON_TOPICS["marks"] == "/marks"
    assert "marks" in tm.ALL_KEYS
    assert "marks" in tm.JSON_SCHEMAS
    # Every field a mark carries has to be in the schema, or Studio drops it.
    properties = tm.JSON_SCHEMAS["marks"]["schema"]["properties"]
    board = mk.MarkPad()
    snapshot = board._snapshot(mk.GOOD, "A", 1.0, "auto", None, None, None)
    assert set(snapshot) <= set(properties)


def test_the_counts_are_in_the_diagnostics_schema():
    import telemetry as tm

    properties = tm.JSON_SCHEMAS["diagnostics"]["schema"]["properties"]
    assert set(mk.MarkPad().counts()) <= set(properties)


def test_marks_has_no_rate_gate():
    """A rate gate on an event channel is a lost press."""
    import telemetry as tm

    assert tm.INTERVALS["marks"] == 0.0


def test_the_control_loop_polls_the_mark_buttons():
    """main.py has to actually call it, on every tick, in both modes."""
    source = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "main.py")).read()
    loop = source.split("def control_loop", 1)[1].split("\ndef ", 1)[0]
    assert "markpad.poll(" in loop
    assert "markpad.counts()" in loop
