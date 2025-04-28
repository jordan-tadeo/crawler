import cv2

class USBCamera:
    def __init__(self, camera_index=0, device_path=None, fps=30):
        self.camera_index = camera_index
        self.device_path = device_path
        self.fps = fps

        # Use device path if provided, otherwise use camera index
        if device_path:
            self.cap = cv2.VideoCapture(device_path)
            print(f"Attempting to open camera at device path: {device_path}")
        else:
            self.cap = cv2.VideoCapture(camera_index)
            print(f"Attempting to open camera at index: {camera_index}")

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at {'device path ' + device_path if device_path else 'index ' + str(camera_index)}")

        print(f"Camera initialized successfully at {'device path ' + device_path if device_path else 'index ' + str(camera_index)}")
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