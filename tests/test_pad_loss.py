"""A pad that goes away, a pad that goes quiet, and what the files say.

Losing the pad in record ends the session the way a Y tap does, with the one
long pulse. Record also ends PAD_SILENCE_SECONDS after the pad stops proving
it is there -- which is not the sticks sitting still. The row that says the
pad went is written past the 4 Hz gate and into the session's own file. And
with no pad the camera's channels are not written at all.
"""

import pytest

import crawler.__main__ as mn
from crawler.telemetry import telemetry as tm

from test_main import FakePad
from test_record import ENTRY, STEP, FakeRecorder, hold_y, run


def diagnostics_rows(tele):
    return [e[2] for e in tele.events
            if e[0] == "offer_now" and e[1] == "diagnostics"]


class DroppingPad(FakePad):
    """Connected until pass `gone`, then not, then (optionally) back."""

    def __init__(self, gone, back=None):
        super().__init__()
        self.gone = gone
        self.back = back

    def is_connected(self):
        if self.frames < self.gone:
            return True
        return self.back is not None and self.frames >= self.back


def test_losing_the_pad_in_record_writes_the_row_first_then_ends_it(
        monkeypatch):
    """The disconnected row goes out past the 4 Hz gate -- this SessionTele
    does not even say it wants diagnostics -- and before the session's end,
    so it lands in the session's own file. Then the end is a Y tap's: file
    closed, camera given back. With the long pulse."""
    pad = DroppingPad(gone=ENTRY + 5)
    pad.throttle = 1.0
    out = run(pad, hold_y(1.0) + [()] * 20, monkeypatch=monkeypatch)
    tele = out["tele"]

    rows = diagnostics_rows(tele)
    assert len(rows) == 1
    assert rows[0]["controller_connected"] is False
    assert rows[0]["mode"] == mn.RECORD
    order = [e[0] for e in tele.events]
    assert order.index("offer_now") < order.index("end_session")

    assert out["modes"].mode == mn.MANUAL
    assert tele.ends == 1
    assert out["hub"].releases == 1
    assert out["markpad"].alarms, "no long pulse"
    assert out["markpad"].buzzes == [2], "the enter buzz and no exit buzz"
    assert out["vecon"].last_throttle == 0.0


def test_only_the_cameras_pictures_stop_while_there_is_no_pad(monkeypatch):
    """With no pad the pictures stop being written (bytes and encoding with
    no driving in them) and nothing else: the map, the lidar, the IMU and
    the status channels carry on, because mapping and editing a patrol do
    not need a controller."""
    pad = DroppingPad(gone=10, back=20)
    out = run(pad, [()] * 40, monkeypatch=monkeypatch)
    tele = out["tele"]
    mutes = [e for e in tele.events if e[0] in ("mute", "unmute")]
    assert mutes == [("mute", tm.PADLESS_MUTED_KEYS), ("unmute",)]
    assert set(tm.PADLESS_MUTED_KEYS) == {"depth", "disparity", "rectified_left",
                                         "pantilt_image", "depth_raw", "depth_small"}
    assert not {"imu", "imu_batch", "lidar", "lidar_scan", "tf", "slam_status", "map",
                "slam_path", "voxel_points", "voxel_low", "pantilt", "patrol_status",
                "patrol_marker", "path", "goal_status"} & set(tm.PADLESS_MUTED_KEYS)
    assert {"imu", "imu_batch", "lidar", "depth", "disparity", "rectified_left",
            "depth_raw", "depth_small", "lidar_scan", "tf",
            "lidar_truck", "slam_status", "map",
            "slam_path", "voxel_points", "voxel_low",
            "pantilt", "pantilt_image"} == set(tm.CAMERA_KEYS)
    assert not {"controller", "vehicle", "diagnostics", "marks",
                "footprint"} & set(tm.CAMERA_KEYS)

    rows = diagnostics_rows(tele)
    assert [r["controller_connected"] for r in rows] == [False, True]


class QuietPad(FakePad):
    """Attached, connected, reads fine -- and stops proving it is there from
    pass `quiet` on. heard() is the only thing that changes."""

    def __init__(self, quiet):
        super().__init__()
        self.quiet = quiet

    def heard(self):
        return self.frames <= self.quiet


QUIET = ENTRY + 10


@pytest.mark.parametrize("seconds, ends", [(4.9, False), (5.0, True)])
def test_record_ends_five_seconds_after_the_pad_goes_quiet(
        monkeypatch, seconds, ends):
    passes = int(seconds / STEP)          # 313 and 320: STEP is 1/64
    pad = QuietPad(QUIET)
    pad.throttle = 1.0
    script = hold_y(1.0) + [()] * (QUIET - ENTRY + passes)
    out = run(pad, script, monkeypatch=monkeypatch)

    if not ends:
        # Still recording when the loop stops. (The loop's own shutdown then
        # closes the session, so tele.ends says nothing here.)
        assert out["modes"].mode == mn.RECORD
        assert out["markpad"].alarms == []
        assert not any("went quiet" in line[2] for line in out["log"].lines)
        return

    assert out["modes"].mode == mn.MANUAL
    assert out["tele"].ends == 1
    assert out["hub"].releases == 1
    # The loop reads its clock before the pump that advances the pad, so
    # pass i runs at (i - 1) * STEP: last heard at QUIET - 1, ended at exactly
    # 5.0 s after that.
    assert out["markpad"].alarms == [
        pytest.approx((QUIET - 1) * STEP + 5.0)]
    assert out["vecon"].last_throttle == 0.0
    assert out["vecon"].neutrals >= 1
    rows = diagnostics_rows(out["tele"])
    assert rows and rows[-1]["controller_connected"] is False
    order = [e[0] for e in out["tele"].events]
    assert order.index("offer_now") < order.index("end_session")


def test_a_pad_sitting_still_never_ends_record(monkeypatch):
    """Thirty seconds of identical sticks and triggers on a pad that is
    there. Values not changing is not silence."""
    class StillPad(FakePad):
        def heard(self):
            return True

    pad = StillPad()
    pad.axes["LEFT_X"] = 0.052
    out = run(pad, hold_y(1.0) + [()] * int(30 / STEP),
              monkeypatch=monkeypatch)
    assert out["modes"].mode == mn.RECORD
    assert out["markpad"].alarms == []


def test_a_heard_that_raises_is_a_dead_handle():
    class Raises:
        def heard(self):
            raise RuntimeError("stale handle")

    assert mn._pad_heard(Raises()) is False
    assert mn._pad_heard(object()) is True


def test_joystick_heard_is_the_handle_answering(monkeypatch):
    import pygame
    from crawler.hardware import joystick as js

    class Handle:
        raises = False

        def get_axis(self, index):
            if self.raises:
                raise RuntimeError("stale")
            return 0.05

    monkeypatch.setattr(pygame.joystick, "get_count", lambda: 1)
    pad = object.__new__(js.Joystick)
    pad.joystick = Handle()
    pad.connected = True
    pad._heard = False
    pad.update_connection_status()
    assert pad.heard() is True
    # The same number every time is still heard.
    for _ in range(5):
        pad.update_connection_status()
        assert pad.heard() is True
    pad.joystick.raises = True
    pad.update_connection_status()
    assert pad.heard() is False
    assert pad.connected is False


# --- telemetry's half --------------------------------------------------------


def live_tele():
    tele = tm.Telemetry()
    tele.running = True
    for key in tele._watched:
        tele._watched[key] = True
    return tele


def test_offer_now_goes_past_the_rate_gate_and_nowhere_unwatched():
    tele = live_tele()
    tele.offer("diagnostics", {"n": 1}, 10.0)
    assert not tele.wants("diagnostics", 10.1), "the gate is closed"
    assert tele.offer_now("diagnostics", {"n": 2}, 10.1) is True
    assert tele._mail["diagnostics"].get_nowait()[1] == {"n": 2}

    tele._watched["diagnostics"] = False
    assert tele.offer_now("diagnostics", {"n": 3}, 10.2) is False
    assert tele._mail["diagnostics"].empty()


def test_muted_channels_are_not_wanted_or_taken_and_the_rest_are():
    tele = live_tele()
    tele.mute(tm.CAMERA_KEYS)
    for key in tm.CAMERA_KEYS:
        assert not tele.wants(key, 0.0)
        tele.offer(key, "frame", 0.0)
        assert tele._mail[key].empty(), key
    assert tele.wants("controller", 0.0)
    tele.offer("controller", {"c": 1}, 0.0)
    assert not tele._mail["controller"].empty()

    tele.unmute()
    assert tele.wants("imu", 0.0)
    tele.offer("imu", {"i": 1}, 0.0)
    assert not tele._mail["imu"].empty()


def test_what_was_offered_before_the_end_goes_in_the_session_file():
    ring, session = FakeRecorder("ring"), FakeRecorder("session")
    tele = live_tele()
    tele.recorder = ring
    emitted = []
    tele._emit = lambda key, payload, log_time: emitted.append(
        (key, payload, session.active))

    tele.begin_session(session)
    tele._swap_session(ring)
    tele.offer_now("diagnostics", {"controller_connected": False}, 1.0)
    tele.end_session()
    tele._swap_session(ring)

    assert emitted == [("diagnostics", {"controller_connected": False}, True)]
    assert session.events == ["start", "stop"]


# --- the steering deadzone ---------------------------------------------------


@pytest.mark.parametrize("raw, want", [
    (0.0, 0.0),
    (0.052, 0.0),       # LEFT_X's median rest
    (0.106, 0.0),       # the worst rest measured
    (0.12, 0.0),        # the edge itself
    (-0.12, 0.0),
    (0.56, 0.35),       # (0.56 - 0.12) / 0.88 * 0.7
    (-0.56, -0.35),
    (1.0, 0.7),         # full deflection is still full lock
    (-1.0, -0.7),
    (1.4, 0.7),         # past the end is clamped, not extrapolated
])
def test_the_steering_deadzone_and_its_rescale(raw, want):
    assert mn.STICK_DEADZONE == 0.12
    assert mn.stick_steer(raw) == pytest.approx(want)


def test_the_curve_is_continuous_at_the_edge_and_never_steps():
    just_past = mn.stick_steer(0.12 + 1e-6)
    assert 0.0 < just_past < 1e-5
    values = [mn.stick_steer(i / 1000) for i in range(1001)]
    assert values == sorted(values)


def test_a_nan_stays_a_nan_and_is_not_turned_into_straight():
    nan = float("nan")
    assert mn.stick_steer(nan) != mn.stick_steer(nan)


@pytest.mark.parametrize("split", [False, True])
def test_a_resting_stick_commands_straight_in_manual(split):
    from test_main import FakeLogger, FakeVecon
    pad = FakePad()
    pad.axes["LEFT_X"] = 0.106
    pad.axes["RIGHT_X"] = -0.064
    vecon = FakeVecon()
    mn.drive_manual(pad, vecon, FakeLogger(), split)
    assert vecon.steerings[-1] == (0.0, 0.0) or vecon.steerings[-1] == (
        0.0, -0.0)


@pytest.mark.parametrize("raw, taken", [(0.19, False), (0.21, True)])
def test_takeover_still_takes_at_point_two_raw_not_at_point_two_commanded(
        raw, taken):
    """0.21 raw commands only 0.072 past the deadzone, which is 0.10 of the
    limit; read back off the intent it would never cross 0.20. Takeover reads
    the raw stick, so its numbers mean what they did."""
    takeover = mn.Takeover()
    pad = FakePad()
    pad.axes["LEFT_X"] = raw
    assert takeover.poll(pad, 0.0)
    assert takeover.steering is taken


def test_takeover_gives_back_under_point_one_two_raw():
    takeover = mn.Takeover()
    pad = FakePad()
    pad.axes["LEFT_X"] = 0.5
    takeover.poll(pad, 0.0)
    assert takeover.steering
    pad.axes["LEFT_X"] = 0.125     # outside the release band: still mine
    takeover.poll(pad, 0.1)
    takeover.poll(pad, 1.0)
    assert takeover.steering
    pad.axes["LEFT_X"] = 0.115     # inside it, for the release time
    takeover.poll(pad, 1.1)
    takeover.poll(pad, 1.1 + mn.TAKEOVER_RELEASE_SECONDS)
    assert not takeover.steering
