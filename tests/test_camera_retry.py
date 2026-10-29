"""The camera comes back on its own.

A boot where the OAK-D was still re-enumerating when the hub first opened it
failed once with X_LINK_DEVICE_NOT_FOUND and never tried again, so auto was
refused until the service was restarted. The hub now keeps opening it in the
background until it is up, and opens it again if it goes away later. auto
refuses while it is down and takes the next START once it is up.

Every fake here is a stand-in for crawler.hardware.oakd.OakD, swapped in
through sys.modules the way test_record and test_voxel do it.
"""

import sys
import threading
import types

import numpy as np
import pytest

import crawler.__main__ as mn
from crawler.hardware import camera

from conftest import FakeLogger


def wait_for(cond, timeout=5.0):
    end = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if cond():
            return True
        end.wait(0.01)
    return cond()


@pytest.fixture
def fast_retry(monkeypatch):
    monkeypatch.setattr(camera, "RETRY_FIRST_S", 0.02)
    monkeypatch.setattr(camera, "RETRY_MAX_S", 0.05)


def install(monkeypatch, cls):
    monkeypatch.setitem(sys.modules, "crawler.hardware.oakd",
                        types.SimpleNamespace(OakD=cls))


class Device:
    """A healthy OAK-D that streams depth until told it is unplugged."""

    built = []

    def __init__(self, rectified=True):
        self.unplugged = False
        self.closed = False
        Device.built.append(self)

    def get_depth_frame(self):
        return None if self.unplugged else np.zeros((2, 2), np.uint16)

    def get_disparity_frame(self):
        return None

    get_rectified_left_frame = get_imu_sample = get_disparity_frame

    def capture_time(self, label):
        return None

    def lost(self):
        return self.unplugged

    def close(self):
        self.closed = True


def failing_then_up(failures):
    """A device class whose first `failures` constructions raise like the
    boot did."""
    tries = {"n": 0}

    class Flaky(Device):
        def __init__(self, rectified=True):
            tries["n"] += 1
            if tries["n"] <= failures:
                raise RuntimeError("Failed to find device after booting, "
                                   "error message: X_LINK_DEVICE_NOT_FOUND")
            super().__init__(rectified)

    return Flaky, tries


def test_a_failed_open_is_retried_until_the_camera_is_up(monkeypatch, fast_retry):
    Flaky, tries = failing_then_up(4)
    install(monkeypatch, Flaky)
    log = FakeLogger()
    hub = camera.CameraHub(logger=log)
    assert hub.start() is True
    try:
        assert wait_for(lambda: hub.state == camera.READY)
        assert tries["n"] == 5
        assert hub.error == ""
        # One line for the first failure, not one per try, and one for up.
        fails = [m for lvl, s, m in log.lines if "would not open" in m]
        assert len(fails) == 1
        assert any("up in" in m and "4 failed" in m for _, _, m in log.lines)
    finally:
        hub.close()


def test_start_while_retrying_does_not_start_a_second_opener(monkeypatch, fast_retry):
    monkeypatch.setattr(camera, "RETRY_FIRST_S", 0.2)
    monkeypatch.setattr(camera, "RETRY_MAX_S", 0.2)
    Flaky, tries = failing_then_up(1000)
    install(monkeypatch, Flaky)
    hub = camera.CameraHub()
    hub.start()
    try:
        assert wait_for(lambda: hub.state == camera.FAILED)
        opener = hub._opener
        assert hub.start() is True
        assert hub._opener is opener
    finally:
        hub.close()
    assert hub.state == camera.CLOSED


def test_release_stops_the_retries(monkeypatch, fast_retry):
    Flaky, tries = failing_then_up(10 ** 6)
    install(monkeypatch, Flaky)
    hub = camera.CameraHub()
    hub.start()
    assert wait_for(lambda: hub.state == camera.FAILED)
    assert hub.release() is True
    assert wait_for(lambda: hub.state == camera.OFF)
    n = tries["n"]
    threading.Event().wait(0.2)
    assert tries["n"] == n


def test_a_device_that_goes_away_is_opened_again(monkeypatch, fast_retry):
    # The stall rule would catch this too, given 5 s; this is lost() alone.
    monkeypatch.setattr(camera, "STALL_S", 60.0)
    Device.built = []
    install(monkeypatch, Device)
    log = FakeLogger()
    hub = camera.CameraHub(logger=log)
    view = hub.view("auto")
    hub.start()
    try:
        assert wait_for(lambda: hub.state == camera.READY)
        first = Device.built[0]
        first.unplugged = True
        assert wait_for(lambda: len(Device.built) == 2
                        and hub.state == camera.READY)
        assert first.closed
        assert any("lost" in m.lower() for _, _, m in log.lines)
        # The consumer's view survives the reopen and gets frames again.
        view.get_depth_frame()
        assert wait_for(lambda: view.get_depth_frame() is not None)
    finally:
        hub.close()


def test_a_device_that_stops_sending_depth_is_opened_again(monkeypatch, fast_retry):
    monkeypatch.setattr(camera, "STALL_S", 0.2)

    class Silent(Device):
        def get_depth_frame(self):
            return None

        def lost(self):
            return False

    Silent.built = Device.built = []
    install(monkeypatch, Silent)
    hub = camera.CameraHub()
    hub.start()
    try:
        assert wait_for(lambda: len(Device.built) >= 2)
    finally:
        hub.close()


# --- auto ---------------------------------------------------------------------


class Hub:
    def __init__(self, state, error=""):
        self.state = state
        self.error = error
        self.starts = 0

    def start(self):
        self.starts += 1
        return True

    def view(self, name):
        return object()


class Vecon:
    pass


def test_auto_waits_for_a_camera_that_is_retrying_instead_of_latching():
    hub = Hub(camera.FAILED, "RuntimeError: X_LINK_DEVICE_NOT_FOUND")
    log = FakeLogger()
    pilot = mn.AutoPilot(Vecon(), log, hub=hub)
    pilot._build_driver = lambda view: "driver"
    assert pilot.resume() is False
    assert pilot.unavailable is False
    hub.state, hub.error = camera.READY, ""
    assert pilot.resume() is True
    assert pilot.driver == "driver"


def test_auto_with_a_driver_on_a_camera_that_dropped_refuses_for_now():
    hub = Hub(camera.FAILED, "lost: no depth for 5 s")
    pilot = mn.AutoPilot(Vecon(), FakeLogger(), hub=hub)
    pilot.driver = "driver"
    assert pilot.resume() is False
    assert pilot.unavailable is False
    hub.state = camera.READY
    assert pilot.resume() is True


def test_oakd_says_lost_once_depthai_has_closed_the_device():
    pytest.importorskip("depthai", reason="depthai is not installed here")
    from crawler.hardware import oakd

    class Dev:
        shut = False

        def isClosed(self):
            return self.shut

    dev = oakd.OakD.__new__(oakd.OakD)
    dev.device, dev.closed = Dev(), False
    assert dev.lost() is False
    dev.device.shut = True
    assert dev.lost() is True
    dev.closed = True           # closed by me is not lost
    assert dev.lost() is False
