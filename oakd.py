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
        cam_left.out.link(stereo.left)
        cam_right.out.link(stereo.right)

        xout = self.pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("depth")
        stereo.depth.link(xout.input)

        self.device = dai.Device(self.pipeline)
        self.depth_queue = self.device.getOutputQueue("depth", 4, False)

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

