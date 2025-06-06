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
        frame = self.get_depth_frame()
        norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        norm = np.uint8(norm)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return color  # BGR, uint8
