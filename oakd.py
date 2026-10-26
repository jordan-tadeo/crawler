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
        stereo.setSubpixel(True)
        stereo.setExtendedDisparity(False)
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
        return packet.getFrame()

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
