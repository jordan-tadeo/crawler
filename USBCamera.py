import cv2

class USBCamera:
    def __init__(self, camera_index=0, fps=30):
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

    def release(self):
        self.cap.release()