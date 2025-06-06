import depthai as dai
import numpy as np

class OakD:
    """Handles comms with the OAK-D"""

    def __init__(self):
        self.pipeline = dai.Pipeline()

        # Create mono cameras
        self.cam_left = self.pipeline.create(dai.node.MonoCamera)
        self.cam_right = self.pipeline.create(dai.node.MonoCamera)
        self.cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        self.cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        self.cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        # Stereo depth node
        self.stereo = self.pipeline.create(dai.node.StereoDepth)
        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        self.cam_left.out.link(self.stereo.left)
        self.cam_right.out.link(self.stereo.right)

        # Output
        self.xout_depth = self.pipeline.create(dai.node.XLinkOut)
        self.xout_depth.setStreamName("depth")
        self.stereo.depth.link(self.xout_depth.input)

        # Start device
        self.device = dai.Device(self.pipeline)
        self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    def get_depth_frame(self):
        """Returns the latest depth frame as a numpy array (in millimeters)"""
        frame = self.depth_queue.get().getFrame()
        return frame  # shape: (H, W), dtype: uint16, unit: mm
