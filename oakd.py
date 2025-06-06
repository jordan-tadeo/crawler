import depthai as dai
import numpy as np
import cv2

class OakD:
    def __init__(self):
        self.pipeline = dai.Pipeline()

        self.depth_buffer = []
        self.buffer_size = 4  # average over last 4 frames
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

        imu = self.pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 100)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)

        # Create IMU node
        imu = self.pipeline.create(dai.node.IMU)
        imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 100)
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

        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

        config = stereo.initialConfig.get()
        config.postProcessing.spatialFilter.enable = True
        config.postProcessing.spatialFilter.holeFillingRadius = 2
        config.postProcessing.spatialFilter.numIterations = 1

        config.postProcessing.temporalFilter.enable = True

        config.postProcessing.speckleFilter.enable = True
        config.postProcessing.speckleFilter.speckleRange = 50

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
        self.depth_queue = self.device.getOutputQueue("depth", 4, False)
        self.disparity_queue = self.device.getOutputQueue("disparity", 4, False)
        self.imu_queue = self.device.getOutputQueue("imu", 10, False)

    def get_imu_sample(self):
        if self.imu_queue.has():
            data = self.imu_queue.get()
            if not data.packets:
                return None

            packet = data.packets[0]

            accel = packet.acceleroMeter  # yes, capital M
            gyro = packet.gyroscope

            return {
                "accel": (accel.x, accel.y, accel.z),
                "gyro": (gyro.x, gyro.y, gyro.z)
            }
        return None


    def get_depth_frame(self):
        return self.depth_queue.get().getFrame()

    def get_depth_colormap(self):
        frame = self.get_depth_frame()
        frame = np.clip(frame, 300, 3000).astype(np.float32)

        if self.smoothed_depth is None:
            self.smoothed_depth = frame.copy()
        else:
            cv2.accumulateWeighted(frame, self.smoothed_depth, 0.3)  # α = 0.3

        norm = ((self.smoothed_depth - 300) / (3000 - 300) * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    def get_disparity_colormap(self):
        disp_frame = self.disparity_queue.get().getFrame()
        disp_normalized = cv2.normalize(disp_frame, None, 0, 255, cv2.NORM_MINMAX)
        disp_normalized = np.uint8(disp_normalized)
        return cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
