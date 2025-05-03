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

        # Timer to update YOLO feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_yolo_feed)
        self.timer.start(33)  # ~30 FPS

    def update_yolo_feed(self):
        try:
            # Get the latest processed frame from PersonFollower
            frame = self.person_follower.latest_frame

            if frame is not None:
                # Convert frame to QImage and display it
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.labels[0][0].setPixmap(pixmap)

                # Process the frame to get the model's input view
                input_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
                input_tensor = tf.image.resize(input_tensor, [128, 128])
                input_tensor = tf.image.rgb_to_grayscale(input_tensor)  # Convert to grayscale
                input_tensor = input_tensor / 255.0  # Normalize to [0,1]
                input_tensor = (input_tensor * 255).numpy().astype(np.uint8).squeeze()  # Convert back to uint8 for display

                # Display the model's input view
                height, width = input_tensor.shape
                bytes_per_line = width
                q_image_input = QImage(input_tensor.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                pixmap_input = QPixmap.fromImage(q_image_input)
                self.labels[0][1].setPixmap(pixmap_input)
        except Exception as e:
            print(f"Error updating YOLO feed: {e}")