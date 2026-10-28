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

from crawler.hardware import oakd  # noqa: E402


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


def _nodes(pipeline, kind):
    return [n for n in pipeline.getAllNodes() if n.getName() == kind]


def test_both_monos_are_capped_at_the_rate_the_stereo_node_can_finish():
    """The latency fix, and it only works as a pair with the shave count.

    Uncapped, the sensors shutter at 30 fps while the stereo node finishes
    far fewer, the surplus queues inside the device, and depth arrives 642 ms
    old -- measured on the car. The cap has to be on BOTH monos: stereo waits
    for a matched pair, so one camera left at 30 goes on filling its own pool.
    """
    pipeline = oakd.build_pipeline()
    monos = _nodes(pipeline, "MonoCamera")
    assert len(monos) == 2
    assert [m.getFps() for m in monos] == [float(oakd.CAPTURE_FPS)] * 2


def test_the_cap_sits_under_what_the_shave_count_can_deliver():
    """The one invariant that makes the cap work rather than make it worse.

    Measured capability, uncapped, 200 frames a config: 12.25 fps at 2
    post-processing shaves, 15.17 at 3, 17.29 at 4, 18.69 at 5, 19.50 at 6. A
    cap ABOVE capability is not a smaller version of the fix, it is no fix at
    all -- 15 fps at 2 shaves measured 622 ms against the 642 it started at.
    So the two settings have to stay a pair, with room to spare for a scene
    harder than the one on the bench.
    """
    assert oakd.POST_PROCESSING_SHAVES >= 3, (
        "the device's own log asks for more than 2, recommended 3")
    capability_fps = {2: 12.25, 3: 15.17, 4: 17.29, 5: 18.69, 6: 19.50}
    measured = capability_fps[oakd.POST_PROCESSING_SHAVES]
    assert oakd.CAPTURE_FPS < measured, (
        f"{oakd.CAPTURE_FPS} fps asked of a pipeline measured at {measured}")


def test_the_pipeline_asks_for_the_shaves_it_says_it_does():
    """setPostProcessingHardwareResources is the lever that mattered: 12.25
    fps to 18.69 on its own. Worth asserting it reaches the node, because it
    is the sort of call that is easy to drop in a refactor and costs nothing
    visible until somebody measures the frame rate again."""
    stereo = _nodes(oakd.build_pipeline(), "StereoDepth")
    assert len(stereo) == 1
    assert (stereo[0].properties.numPostProcessingShaves
            == oakd.POST_PROCESSING_SHAVES)
    assert (stereo[0].properties.numPostProcessingMemorySlices
            == oakd.POST_PROCESSING_SLICES)


def test_the_rectified_stream_is_the_only_thing_the_flag_changes():
    """build_pipeline's one argument. Off, the rectified XLinkOut is not
    created; depth, disparity and the IMU are there either way, because the
    driver needs depth and Foxglove needs disparity, and a linked queue
    nobody drains stalls the stream behind it."""
    with_rect = oakd.build_pipeline(rectified=True)
    without = oakd.build_pipeline(rectified=False)
    assert len(_nodes(with_rect, "XLinkOut")) == 4
    assert len(_nodes(without, "XLinkOut")) == 3
    for pipeline in (with_rect, without):
        assert len(_nodes(pipeline, "MonoCamera")) == 2
        assert len(_nodes(pipeline, "StereoDepth")) == 1
        assert len(_nodes(pipeline, "IMU")) == 1


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
