"""The goal disc stays up, stamped now.

/goal/marker is a SceneUpdate with one entity, the disc, republished at the
goal_marker rate for as long as a goal is set: stamped with the time it was
published, in the "map" frame, and with no lifetime, which is forever. And the
stamps follow the wall clock across an NTP step: the Pi has no RTC, and the
10-29 boot stamped every message 90.1 s behind the wall for the whole run
because the offset was taken once at import, before the step.
"""

import time

import pytest

from crawler.telemetry import telemetry as tm


class Clocks:
    """time.time and time.monotonic, moved by hand."""

    def __init__(self, wall, mono):
        self.wall, self.mono = wall, mono

    def advance(self, dt):
        self.wall += dt
        self.mono += dt

    def install(self, monkeypatch):
        monkeypatch.setattr(tm.time, "time", lambda: self.wall)
        monkeypatch.setattr(tm.time, "monotonic", lambda: self.mono)


def test_stamps_follow_the_wall_clock_after_it_steps(monkeypatch):
    clocks = Clocks(wall=1_800_000_000.0, mono=100.0)
    clocks.install(monkeypatch)
    monkeypatch.setattr(tm, "_EPOCH_OFFSET", clocks.wall - clocks.mono)
    assert tm.epoch_ns(100.0) == pytest.approx(1_800_000_000.0e9, abs=1e3)

    # NTP steps the clock 90.1 s forward, as on the 10-29 boot.
    clocks.wall += 90.1
    clocks.advance(1.0)
    assert tm.epoch_ns(clocks.mono) == pytest.approx(clocks.wall * 1e9, abs=1e6)
    # A reading from before the step, made now, is on the new clock too, and
    # the series still runs in order.
    assert tm.epoch_ns(100.0) < tm.epoch_ns(clocks.mono)


def test_a_slew_under_the_step_size_does_not_move_the_stamps(monkeypatch):
    clocks = Clocks(wall=1_800_000_000.0, mono=100.0)
    clocks.install(monkeypatch)
    monkeypatch.setattr(tm, "_EPOCH_OFFSET", clocks.wall - clocks.mono)
    before = tm.epoch_ns(100.0)
    clocks.wall += 0.004            # what NTP slews by, not a step
    assert tm.epoch_ns(100.0) == before


class Channel:
    def __init__(self):
        self.logged = []

    def log(self, message, log_time=None):
        self.logged.append((message, log_time))


def live_telemetry():
    t = tm.Telemetry()
    t.running = True
    t._watched["goal_marker"] = True
    t._channels = {"goal_marker": Channel()}
    return t


def test_the_disc_is_republished_stamped_now_in_map_and_forever(monkeypatch):
    pytest.importorskip("foxglove")
    clocks = Clocks(wall=1_800_000_000.0, mono=100.0)
    clocks.install(monkeypatch)
    # Imported before the clock stepped: the offset taken then is 90.1 s
    # short of the wall now.
    monkeypatch.setattr(tm, "_EPOCH_OFFSET", clocks.wall - clocks.mono - 90.1)

    tele = live_telemetry()
    channel = tele._channels["goal_marker"]
    payload = {"goal": (3.51, -0.92)}
    sent = []
    for _ in range(30):                     # 3 s of 100 Hz loop passes
        now = clocks.mono
        if tele.wants("goal_marker", now):
            tele.offer("goal_marker", payload, now)
            tele._drain_mail(None)
            sent.append(clocks.wall)
        clocks.advance(0.1)

    assert len(channel.logged) == len(sent) == 3, "once a second, every second"
    from test_mapping import decode
    for (raw, log_time), wall in zip(channel.logged, sent):
        msg = decode(raw)
        assert log_time == pytest.approx(wall * 1e9, abs=1e6), "stamped now"
        (disc,) = [e for e in msg.entities if e.id == "nav_goal"]
        assert disc.frame_id == "map"
        assert disc.timestamp.ToNanoseconds() == log_time
        assert disc.lifetime.ToNanoseconds() == 0, "forever"
        assert not [d for d in msg.deletions if d.id == "nav_goal"]
        (cyl,) = disc.cylinders
        assert (cyl.pose.position.x, cyl.pose.position.y) == (3.51, -0.92)
