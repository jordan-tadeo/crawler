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
"""

import depthai as dai
import numpy as np
import cv2


class OakD:
    def __init__(self):
        self.pipeline = dai.Pipeline()

        self.smoothed_depth = None

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

        # Start device
        self.device = dai.Device(self.pipeline)
        self.closed = False
        # self.device.setLogLevel(dai.LogLevel.INFO)
        # self.device.setLogOutputLevel(dai.LogLevel.INFO)

        self.depth_queue = self.device.getOutputQueue("depth", 4, False)
        self.disparity_queue = self.device.getOutputQueue("disparity", 4, False)
        self.imu_queue = self.device.getOutputQueue("imu", 10, False)

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

    def get_depth_frame(self):
        """Latest depth frame, or None if there is not one waiting."""
        if self.depth_queue is None:
            return None
        try:
            packet = self.depth_queue.tryGet()
        except RuntimeError as e:
            if "X_LINK_ERROR" in str(e):
                print("[OakD] Stream error - no depth this tick.")
                return None
            raise
        if packet is None:
            return None
        return packet.getFrame()

    def get_depth_colormap(self):
        frame = self.get_depth_frame()
        if frame is None:
            # np.clip() of None is a TypeError, and this is a debug pane, so
            # no frame just means no picture.
            return None

        frame = np.clip(frame, 300, 3000).astype(np.float32)

        if self.smoothed_depth is None:
            self.smoothed_depth = frame.copy()
        else:
            cv2.accumulateWeighted(frame, self.smoothed_depth, 0.3)  # alpha = 0.3

        norm = ((self.smoothed_depth - 300) / (3000 - 300) * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    def get_disparity_colormap(self):
        if self.disparity_queue is None:
            return None
        try:
            packet = self.disparity_queue.tryGet()
        except RuntimeError as e:
            if "X_LINK_ERROR" in str(e):
                return None
            raise
        if packet is None:
            return None
        disp_frame = packet.getFrame()
        disp_normalized = cv2.normalize(disp_frame, None, 0, 255, cv2.NORM_MINMAX)
        disp_normalized = np.uint8(disp_normalized)
        return cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)

    def close(self):
        """Release the device. Safe to call twice."""
        if self.closed:
            return
        self.closed = True
        self.depth_queue = None
        self.disparity_queue = None
        self.imu_queue = None
        try:
            self.device.close()
        except Exception as e:
            print(f"[OakD] close failed: {e}")
