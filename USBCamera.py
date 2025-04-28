import cv2

class USBCamera:
    def __init__(self, camera_index=1, fps=30):
        self.camera_index = camera_index
        self.fps = fps
        self.cap = cv2.VideoCapture(camera_index)
        self.set_fps(fps)

    def set_fps(self, fps):
        self.fps = fps
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from USB camera.")
        return frame

    def focus_sweep (self, focus_value=0):
        """Set manual focus for the camera.
        Args:
            focus_value (int): Focus value (usually between 0 and 255).
        """
        while True:
            if not self.cap.set(cv2.CAP_PROP_FOCUS, focus_value):
                raise RuntimeError("Failed to set manual focus. Ensure your camera supports this feature.")
            focus_value += 1
            if focus_value > 255:
                focus_value = 0

    def release(self):
        self.cap.release()