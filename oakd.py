import depthai as dai
import numpy as np
import cv2

class OakD:
    def __init__(self):
        self.pipeline = dai.Pipeline()

        cam_left = self.pipeline.create(dai.node.MonoCamera)
        cam_right = self.pipeline.create(dai.node.MonoCamera)
        cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo = self.pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)

        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(True)
        stereo.initialConfig.setConfidenceThreshold(245)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

        config = stereo.initialConfig.get()
        config.postProcessing.speckleFilter.enable = True
        config.postProcessing.speckleFilter.speckleRange = 25
        config.postProcessing.temporalFilter.enable = True
        config.postProcessing.spatialFilter.enable = True
        config.postProcessing.spatialFilter.holeFillingRadius = 2
        config.postProcessing.spatialFilter.numIterations = 1
        stereo.initialConfig.set(config)

        cam_left.out.link(stereo.left)
        cam_right.out.link(stereo.right)

        xout_disp = self.pipeline.create(dai.node.XLinkOut)
        xout_disp.setStreamName("disparity")
        stereo.disparity.link(xout_disp.input)


        xout = self.pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("depth")
        stereo.depth.link(xout.input)

        self.device = dai.Device(self.pipeline)
        self.disparity_queue = self.device.getOutputQueue("disparity", maxSize=4, blocking=False)
        self.depth_queue = self.device.getOutputQueue("depth", 4, False)

    def get_disparity_colormap(self):
        disp_frame = self.disparity_queue.get().getFrame()  # uint8

        # Normalize to 0–255 range if needed
        disp_normalized = cv2.normalize(disp_frame, None, 0, 255, cv2.NORM_MINMAX)
        disp_normalized = np.uint8(disp_normalized)

        color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
        return color

    def get_depth_frame(self):
        return self.depth_queue.get().getFrame()  # raw depth in mm

    def get_depth_colormap(self):
        frame = self.get_depth_frame()  # millimeters

        # Clip values to a useful display range (e.g. 300mm to 3000mm)
        frame_clipped = np.clip(frame, 300, 3000)

        # Normalize manually to 0–255
        norm = ((frame_clipped - 300) / (3000 - 300) * 255).astype(np.uint8)

        # Apply a JET colormap
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        return color  # BGR image

