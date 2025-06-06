from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QLabel, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import cv2
import numpy as np
from PersonFollower import PersonFollower
import tensorflow as tf
from oakd import OakD

class Dashboard(QMainWindow):
    def __init__(self, person_follower: PersonFollower = None, oakd: OakD = None):
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

        # PersonFollower instance
        self.person_follower = person_follower

        self.oakd = oakd

        # Timer to update pantilt feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_pantilt_view)
        self.timer.start(17)  # ~30 FPS

        # Timer to update Oak-D depth feed
        self.depth_timer = QTimer(self)
        self.depth_timer.timeout.connect(self.update_depth_view)
        self.depth_timer.start(17)  # ~30 FPS

        # Timer to update disparity feed
        self.disparity_timer = QTimer(self)
        self.disparity_timer.timeout.connect(self.update_disparity_view)
        self.disparity_timer.start(17)

    def update_disparity_view(self):
        if self.oakd is None:
            return

        frame = self.oakd.get_disparity_colormap()
        if frame is None:
            return

        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)

        self.labels[1][2].setPixmap(pixmap)  # Show in right-middle of the grid


    def update_depth_view(self):
        if self.oakd is None:
            return

        frame = self.oakd.get_depth_colormap()  # BGR image
        if frame is None:
            return

        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)

        self.labels[1][0].setPixmap(pixmap)  # Show on left-middle of grid


    def update_model_input_view(self, input_tensor):
        if not self.person_follower:
            return
        # Convert the tensor to a numpy array for visualization
        display_tensor = self.person_follower.get_latest_input_tensor() # Remove unnecessary dimensions

        # Scale the values back to [0,255] for visualization purposes
        display_tensor = (display_tensor * 255).astype(np.uint8)

        # Display the exact input tensor
        height, width = display_tensor.shape
        bytes_per_line = width
        q_image_input = QImage(display_tensor.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap_input = QPixmap.fromImage(q_image_input)
        self.labels[0][1].setPixmap(pixmap_input)

    def update_pantilt_view(self):
        if not self.person_follower:
            return
        # Get the latest frame from the person follower
        frame = self.person_follower.get_latest_frame()

        if frame is None:
            return

        # Convert the frame to a format that can be displayed in the QLabel
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Update the QLabel with the new pixmap
        self.labels[1][1].setPixmap(pixmap)

        # Update the model input feed for debugging
        self.update_model_input_feed(self.person_follower.get_latest_input_tensor())

    def closeEvent(self, event):
        if not self.person_follower:
            return
        # Stop the person follower when closing the dashboard
        self.person_follower.stop()
        event.accept()
