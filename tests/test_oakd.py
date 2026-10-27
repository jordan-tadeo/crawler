"""The one piece of oakd.py that is arithmetic rather than a device call.

Everything else in that file is a dai.Pipeline being wired up or a queue being
polled, and none of it can run without an OAK-D plugged in. frame_from_payload
can: it takes an object with getData/getWidth/getHeight and works out the
pixel layout, so a four-line stub stands in for an ImgFrame.

Importing oakd still pulls in depthai, which is on the car and on the machine
this was written on but is not a given anywhere else, so the import is a skip
rather than a failure. The rest of the suite stays hardware-free.
"""

import numpy as np
import pytest

pytest.importorskip("depthai", reason="depthai is not installed here")

import oakd  # noqa: E402


class FakePacket:
    """An ImgFrame as far as frame_from_payload is concerned."""

    def __init__(self, data, width, height):
        self.data = np.asarray(data, dtype=np.uint8)
        self.width = width
        self.height = height

    def getData(self):
        return self.data

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height


def test_a_one_byte_payload_is_read_as_eight_bit_disparity():
    """What extended disparity actually sends: 640x400 at one byte a pixel,
    stamped RAW16 by depthai 2.30 because the filters push its reported
    maxDisparity to 8170. getFrame() raises on that; this does not."""
    packet = FakePacket(np.arange(256000) % 251, 640, 400)
    frame = oakd.frame_from_payload(packet, "disparity")
    assert frame.dtype == np.uint8
    assert frame.shape == (400, 640)
    assert frame[0, 0] == 0 and frame[0, 5] == 5


def test_a_two_byte_payload_is_still_read_as_sixteen_bit():
    """Depth, and subpixel disparity, are genuinely 16-bit. The same function
    has to leave those alone, because _try_frame only reaches it on an error
    and must not turn a real frame into a differently shaped one."""
    expected = np.array([[1000, 2000, 3000], [4000, 5000, 60000]], dtype=np.uint16)
    packet = FakePacket(expected.view(np.uint8).ravel(), 3, 2)
    frame = oakd.frame_from_payload(packet, "depth")
    assert frame.dtype == np.uint16
    assert np.array_equal(frame, expected)


@pytest.mark.parametrize("data_size, width, height", [
    (256001, 640, 400),   # not a whole number of bytes per pixel
    (0, 640, 400),        # metadata-only transfer
    (256000, 0, 400),     # no geometry to reshape to
    (768000, 640, 400),   # three bytes a pixel, nothing here produces that
])
def test_a_payload_that_makes_no_sense_is_dropped_not_raised(data_size, width, height):
    """The contract the whole file keeps. A missing frame is a case every
    caller already handles -- driver.tick() zeroes the throttle on one -- and
    an exception off the camera thread is not. pigpio latches the last pulse
    width forever, so a crashed reader is a car still driving."""
    packet = FakePacket(np.zeros(data_size), width, height)
    assert oakd.frame_from_payload(packet, "disparity") is None
