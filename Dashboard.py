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

                # Normalize the input tensor to [0,1] for the AI model
                input_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
                input_tensor = tf.image.resize(input_tensor, [128, 128])
                input_tensor = tf.image.rgb_to_grayscale(input_tensor)  # Convert to grayscale
                input_tensor = input_tensor / 255.0  # Normalize to [0,1]

                # Prepare the image for display (convert back to uint8 and expand dimensions)
                display_tensor = (input_tensor * 255).numpy().astype(np.uint8).squeeze()  # Convert back to uint8
                display_tensor = np.expand_dims(display_tensor, axis=-1)  # Add a channel dimension
                display_tensor = np.repeat(display_tensor, 3, axis=-1)  # Repeat the channel to simulate RGB

                # Display the adjusted model's input view
                height, width, channel = display_tensor.shape
                bytes_per_line = width * channel
                q_image_input = QImage(display_tensor, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap_input = QPixmap.fromImage(q_image_input)
                self.labels[0][1].setPixmap(pixmap_input)

                # Ensure the grayscale image is expanded to 3 dimensions (RGB)
                input_tensor = np.expand_dims(input_tensor, axis=-1)  # Add a channel dimension
                input_tensor = np.repeat(input_tensor, 3, axis=-1)  # Repeat the channel to simulate RGB

                # Display the adjusted model's input view
                height, width, channel = input_tensor.shape
                bytes_per_line = width * channel
                q_image_input = QImage(input_tensor, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap_input = QPixmap.fromImage(q_image_input)
                self.labels[0][1].setPixmap(pixmap_input)

                # Display the exact input tensor passed to the AI model
                input_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
                input_tensor = tf.image.resize(input_tensor, [128, 128])
                input_tensor = tf.image.rgb_to_grayscale(input_tensor)  # Convert to grayscale
                input_tensor = input_tensor / 255.0  # Normalize to [0,1]

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
        except Exception as e:
            print(f"Error updating YOLO feed: {e}")