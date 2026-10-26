"""Debug panes behind --dashboard. Not on the boot path.

Takes whatever it is given and guards for the rest, so it works with just an
OAK-D, just a PersonFollower, or neither.

Two things to know before trusting what you see here. The depth and disparity
timers pull from the same device queues the driver does, so in auto mode the
dashboard and the driver steal frames from each other and neither gets the
full rate. And nothing imports PersonFollower any more -- the type hint used
to, which dragged the whole of TensorFlow in just to draw a window.
"""

from PyQt5.QtWidgets import QMainWindow, QGridLayout, QLabel, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import numpy as np


class Dashboard(QMainWindow):
    def __init__(self, person_follower=None, oakd=None):
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
        if self.person_follower is None:
            return

        display_tensor = self.person_follower.get_latest_input_tensor()
        if display_tensor is None:
            return

        # Scale the values back to [0,255] for visualization purposes
        display_tensor = (np.asarray(display_tensor) * 255).astype(np.uint8)

        # Display the exact input tensor
        height, width = display_tensor.shape
        bytes_per_line = width
        q_image_input = QImage(display_tensor.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap_input = QPixmap.fromImage(q_image_input)
        self.labels[0][1].setPixmap(pixmap_input)

    def update_pantilt_view(self):
        if self.person_follower is None:
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

        # Update the model input feed for debugging. This called
        # update_model_input_feed, which does not exist.
        self.update_model_input_view(self.person_follower.get_latest_input_tensor())

    def closeEvent(self, event):
        # Accept either way. The version I had bailed out before accepting if
        # there was no PersonFollower, so with only an OAK-D attached closing
        # the window did nothing at all.
        if self.person_follower is not None:
            self.person_follower.stop()
        event.accept()
