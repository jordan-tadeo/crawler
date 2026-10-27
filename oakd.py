"""OAK-D stereo depth, read without ever blocking the control thread.

Every getter here is a tryGet: it returns None when no packet has landed
rather than waiting for one. That is the whole point of this file. The
blocking queue.get() it used to use meant that if the USB link wedged, the
control thread parked inside get() forever -- the loop stopped, the stop
signal was never checked, and because pigpio latches a servo pulse width
and keeps emitting it, the car carried on at whatever throttle it had until
the battery died. There is no hardware failsafe and no receiver to fall
back on, so a stall in here is a runaway.

Note for the Pi: a Luxonis device needs a udev rule before a non-root
service user can open it, or dai.Device() just fails. Put this in
/etc/udev/rules.d/80-movidius.rules and reload:

    SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"

I am not installing that from install-service.sh -- it is a one-time
system change and I want to run it myself.

Who owns one of these: camera.CameraHub, and nothing else. dai.Device() takes
exclusive USB access and blocks for seconds while the device boots, so there
is one instance per process, it is constructed on a background thread, and the
hub is the only thing that reads the queues. Consumers take a
camera.StreamView instead of an OakD -- same getters, one slot each, nobody
stealing anybody's frames.

The colourmapping that used to live in here has moved to StreamView for that
reason. get_depth_colormap() pulled a frame off the depth queue as a side
effect of drawing a picture, so a debug pane and the driver were competing for
the same packets. Everything in here now is a raw frame off one queue.
"""

import depthai as dai
import numpy as np


def frame_from_payload(packet, label=""):
    """An ImgFrame's pixels, worked out from the payload instead of the label.

    getFrame() trusts ImgFrame.getType(), and with extended disparity on, that
    label is wrong. depthai 2.30 reports maxDisparity as 8170 rather than 190
    as soon as any post-processing filter is enabled -- spatial, temporal or
    speckle, any one of them is enough -- and because 8170 does not fit in a
    byte the disparity output gets stamped Type.RAW16. The firmware still sends
    one byte per pixel, so getFrame() asks for 512000 bytes, finds 256000 and
    raises. Measured on the car: extended plus filters is RAW16 carrying
    1.00 bytes/px, extended with no filters is RAW8 carrying 1.00, and subpixel
    plus filters is RAW16 genuinely carrying 2.00. So only the combination this
    pipeline now runs is mislabelled, which is why it never showed up before.

    Depth is not affected -- it is real millimetres, RAW16, two bytes a pixel in
    every configuration -- so nothing the driver reads goes through here. It is
    the disparity stream, which only ever becomes a picture for me to look at.

    Returns None rather than raising on anything it cannot make sense of, which
    is the contract every getter in this file keeps: a missing frame is a case
    the callers already handle, an exception on the camera thread is not.
    """
    data = packet.getData()
    width, height = packet.getWidth(), packet.getHeight()
    pixels = width * height
    if pixels <= 0 or data.size == 0 or data.size % pixels:
        print(f"[OakD] {label}: {data.size} bytes for {width}x{height}, dropped.")
        return None
    per_pixel = data.size // pixels
    if per_pixel == 1:
        return data.reshape(height, width)
    if per_pixel == 2:
        return data.view(np.uint16).reshape(height, width)
    print(f"[OakD] {label}: {per_pixel} bytes per pixel, dropped.")
    return None


class OakD:
    def __init__(self, rectified: bool = True):
        self.pipeline = dai.Pipeline()

        # Mono cameras
        cam_left = self.pipeline.create(dai.node.MonoCamera)
        cam_right = self.pipeline.create(dai.node.MonoCamera)
        cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        # Stereo depth node
        stereo = self.pipeline.create(dai.node.StereoDepth)

        # Create IMU node
        imu = self.pipeline.create(dai.node.IMU)
        imu.enableIMUSensor([dai.IMUSensor.LINEAR_ACCELERATION, dai.IMUSensor.GYROSCOPE_RAW], 100)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)

        # IMU output
        imu_out = self.pipeline.create(dai.node.XLinkOut)
        imu_out.setStreamName("imu")
        imu.out.link(imu_out.input)

        # === GUI-EQUIVALENT SETTINGS ===
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setConfidenceThreshold(245)
        stereo.setLeftRightCheck(True)
        # Extended disparity rather than subpixel. These two are mutually
        # exclusive on RVC2, so it is a trade and not an addition: extended
        # runs the stereo engine twice, once at half resolution, and
        # concatenates the result into a 190 level disparity search instead of
        # 95. That halves the minimum measurable depth --
        # fx * baseline / levels = 441.25 * 75 / 190 = 174 mm against 348 --
        # which is what I actually want, because the obstacle trigger is at
        # 400 mm and with subpixel on there were only 52 mm of measurable
        # warning above the blind zone. Now there are 226 mm.
        #
        # What I give up is subpixel interpolation, so depth comes back
        # quantised to whole disparity levels. One level is Z^2 / (fx*baseline)
        # = 4.8 mm at the 400 mm trigger and 24.5 mm at the 900 mm floor clip,
        # where subpixel's eighth-of-a-level gave 0.6 mm and 3.1 mm. Neither
        # matters here: 4.8 mm of jitter on a 400 mm threshold is 1.2%, and the
        # floor clip at 899 is a band edge with 12 degrees of pitch margin
        # behind it, not a measurement anybody reads. 174 mm of range is worth
        # more than a millimetre of precision I never look at.
        #
        # Measured, 300 frames a side, same scene, camera not moved: nearest
        # valid depth went from 319 mm on subpixel to 165 mm on extended, and
        # readings in the 200-250 mm band exist only on extended -- 25130 pixel
        # samples against zero. 12.1 fps and 0.64 s of frame latency either
        # way, so the second stereo pass costs nothing measurable here.
        #
        # One surprise worth writing down: 348 was never the hard floor the
        # arithmetic says. Subpixel returned depth down to 319 mm, roughly 4%
        # of valid pixels sitting under 348, because the spatial and temporal
        # filters below interpolate across the edge of the search range. So
        # the gain is real but it is a floor dropping from ~320 to ~165, not
        # from 348 to 174.
        stereo.setSubpixel(False)
        stereo.setExtendedDisparity(True)
        stereo.setRectifyEdgeFillColor(0)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)  # Optional
        stereo.setPostProcessingHardwareResources(2, 3)

        # stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

        config = stereo.initialConfig.get()
        config.postProcessing.spatialFilter.enable = True
        config.postProcessing.spatialFilter.holeFillingRadius = 2
        config.postProcessing.spatialFilter.numIterations = 1

        config.postProcessing.temporalFilter.enable = True

        config.postProcessing.speckleFilter.enable = True
        config.postProcessing.speckleFilter.speckleRange = 28

        stereo.initialConfig.set(config)
        # === END SETTINGS ===

        # Link stereo to cameras
        cam_left.out.link(stereo.left)
        cam_right.out.link(stereo.right)

        # Depth output
        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        # Disparity output (optional)
        xout_disp = self.pipeline.create(dai.node.XLinkOut)
        xout_disp.setStreamName("disparity")
        stereo.disparity.link(xout_disp.input)

        # The rectified left mono image: the actual camera picture, which
        # neither depth nor disparity shows me. Worth having in Foxglove for
        # checking where the thing is pointed and whether a lens is dirty --
        # a depth map looks much the same either way.
        #
        # Opt-in, because an XLink stream costs USB bandwidth from the moment
        # it is linked whether or not anybody drains it, and auto mode does
        # not want to pay for a picture it does not look at. The hub only asks
        # for it when telemetry is going to carry it.
        self.rectified = rectified
        if rectified:
            xout_rect = self.pipeline.create(dai.node.XLinkOut)
            xout_rect.setStreamName("rectified_left")
            stereo.rectifiedLeft.link(xout_rect.input)

        # Start device
        self.device = dai.Device(self.pipeline)
        self.closed = False
        # self.device.setLogLevel(dai.LogLevel.INFO)
        # self.device.setLogOutputLevel(dai.LogLevel.INFO)

        self.depth_queue = self.device.getOutputQueue("depth", 4, False)
        self.disparity_queue = self.device.getOutputQueue("disparity", 4, False)
        self.imu_queue = self.device.getOutputQueue("imu", 10, False)
        self.rectified_queue = (
            self.device.getOutputQueue("rectified_left", 2, False)
            if rectified else None)

    def get_imu_sample(self):
        if self.imu_queue is None:
            return None
        data = self.imu_queue.tryGet()
        if data is None or not data.packets:
            return None
        packet = data.packets[0]
        accel = packet.acceleroMeter  # correct
        gyro = packet.gyroscope
        return {"accel": (accel.x, accel.y, accel.z),
                "gyro": (gyro.x, gyro.y, gyro.z)}

    # There used to be a reset_pipeline() here that called self.__init__() on
    # an X_LINK error. It is gone on purpose. Rebuilding the pipeline takes
    # seconds, blocks whoever called it, and can raise halfway through and
    # leave the object half-built -- all of which is exactly what must never
    # happen on the thread that commands throttle. If the link dies, the
    # getters return None, driver.tick() zeroes the throttle, and I restart
    # the service.

    def _try_frame(self, queue, label):
        """Newest frame off one queue, or None. Never waits, never raises.

        The X_LINK_ERROR case is a real one and it is not fatal: the link
        drops, this hands back None, and whoever asked treats a missing frame
        the way it already has to -- driver.tick() zeroes the throttle.
        """
        if queue is None:
            return None
        try:
            packet = queue.tryGet()
        except RuntimeError as e:
            if "X_LINK_ERROR" in str(e):
                print(f"[OakD] Stream error - no {label} this tick.")
                return None
            raise
        if packet is None:
            return None
        try:
            return packet.getFrame()
        except RuntimeError as e:
            if "enough data" not in str(e):
                raise
            return frame_from_payload(packet, label)

    def get_depth_frame(self):
        """Latest depth frame in millimetres, or None if none is waiting."""
        return self._try_frame(self.depth_queue, "depth")

    def get_disparity_frame(self):
        """Latest raw disparity frame, or None."""
        return self._try_frame(self.disparity_queue, "disparity")

    def get_rectified_left_frame(self):
        """Latest rectified left mono frame, or None if not enabled."""
        return self._try_frame(self.rectified_queue, "rectified left")

    def close(self):
        """Release the device. Safe to call twice."""
        if self.closed:
            return
        self.closed = True
        self.depth_queue = None
        self.disparity_queue = None
        self.imu_queue = None
        self.rectified_queue = None
        try:
            self.device.close()
        except Exception as e:
            print(f"[OakD] close failed: {e}")
