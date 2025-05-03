from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QLabel, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import cv2
import numpy as np
from PersonFollower import PersonFollower
import tensorflow as tf

class Dashboard(QMainWindow):
    def __init__(self, person_follower: PersonFollower):
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

        # Timer to update pantilt feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_pantilt_view)
        self.timer.start(17)  # ~30 FPS

    def update_model_input_view(self, input_tensor):
        # Convert the tensor to a numpy array for visualization
        display_tensor = input_tensor.numpy().squeeze()  # Remove unnecessary dimensions

        # Scale the values back to [0,255] for visualization purposes
        display_tensor = (display_tensor * 255).astype(np.uint8)

        # Display the exact input tensor
        height, width = display_tensor.shape
        bytes_per_line = width
        q_image_input = QImage(display_tensor.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap_input = QPixmap.fromImage(q_image_input)
        self.labels[0][1].setPixmap(pixmap_input)

    def update_pantilt_view(self):
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
        # Stop the person follower when closing the dashboard
        self.person_follower.stop()
        event.accept()
