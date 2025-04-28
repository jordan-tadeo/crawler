from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QLabel, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import cv2
from USBCamera import USBCamera

class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dashboard")
        self.setGeometry(0, 0, 800, 600)  # Default window size

        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.grid_layout = QGridLayout(self.central_widget)

        # Create 3x3 grid of labels
        self.labels = [[QLabel(self) for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                self.labels[i][j].setStyleSheet("border: 1px solid black;")
                self.labels[i][j].setScaledContents(True)
                self.grid_layout.addWidget(self.labels[i][j], i, j)

        # USB Camera setup
        self.usb_camera = USBCamera(camera_index=0, fps=30)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_usb_cam)
        self.timer.start(33)  # ~30 FPS

    def update_usb_cam(self):
        try:
            frame = self.usb_camera.get_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.labels[0][0].setPixmap(pixmap)
        except RuntimeError as e:
            print(f"Error capturing frame: {e}")

    def closeEvent(self, event):
        self.usb_camera.release()
        event.accept()