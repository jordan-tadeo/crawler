import tensorflow as tf
from USBCamera import USBCamera
from VehicleController import VehicleController
import cv2
import warnings
import threading
import numpy as np
import os
import kagglehub

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

class PersonFollower:
    def __init__(self, vehicle_controller: VehicleController, usb_cam: USBCamera):
        # Load the TensorFlow SavedModel
        model_dir = "/home/jt/.cache/kagglehub/models/google/mobilenet-v2/tensorFlow2/035-128-classification/2"
        print("Loading TensorFlow SavedModel from:", model_dir)
        self.model = tf.saved_model.load(model_dir)

        # Initialize camera and vehicle controller
        self.camera = usb_cam
        self.controller = vehicle_controller

        # Frame dimensions (assume 320x240 for now, adjust dynamically if needed)
        self.frame_width = 320
        self.frame_height = 240
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2

        # Pan/Tilt adjustment parameters
        self.pan_step = 2  # Degrees to adjust per frame
        self.tilt_step = 2

        # Initial pan/tilt angles
        self.pan_angle = 90
        self.tilt_angle = 90
        self.controller.set_pan_tilt(0, 0)  # Set to neutral

        self.person_detected = False

        # Threading setup
        self.thread = threading.Thread(target=self._run_yolo_processing, daemon=True)
        self.stop_event = threading.Event()
        self.latest_frame = None
        self.latest_detections = None

    def start(self):
        '''Start the YOLO processing thread.'''
        self.thread.start()

    def stop(self):
        '''Stop the YOLO processing thread.'''
        self.stop_event.set()
        self.thread.join()

    def _run_yolo_processing(self):
        while not self.stop_event.is_set():
            self.process_and_adjust()

    def process_and_adjust(self):
        '''Process the latest frame and adjust servos accordingly.'''
        try:
            # Capture frame from camera
            frame = self.camera.get_frame()

            # Resize frame to reduce computational load
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))

            # Process frame to detect person
            person_center_x, person_center_y, processed_frame = self.process_frame(frame)

            # Update the latest processed frame
            self.latest_frame = processed_frame

            # Adjust servos to keep person in frame
            self.adjust_servos(person_center_x, person_center_y)

        except Exception as e:
            print(f"Error in process_and_adjust: {e}")

    def process_frame(self, frame):
        # Preprocess the frame to match the model's input requirements
        input_tensor = tf.convert_to_tensor(frame, dtype=tf.uint8)
        input_tensor = tf.image.resize(input_tensor, [128, 128])  # Resize to model's expected input size
        input_tensor = tf.expand_dims(input_tensor, axis=0)  # Add batch dimension

        # Run inference
        detections = self.model(input_tensor)

        # Visualize all raw detections by drawing bounding boxes
        for detection in detections['detection_boxes'][0].numpy():
            y1, x1, y2, x2 = detection
            x1, y1, x2, y2 = int(x1 * self.frame_width), int(y1 * self.frame_height), int(x2 * self.frame_width), int(y2 * self.frame_height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for raw detections

        # Process detections (assuming person class ID is 0)
        person_detections = [d for d in detections['detection_classes'][0].numpy() if d == 0]

        if person_detections:
            # Visualize detections by drawing bounding boxes
            for detection in detections['detection_boxes'][0].numpy():
                y1, x1, y2, x2 = detection
                x1, y1, x2, y2 = int(x1 * self.frame_width), int(y1 * self.frame_height), int(x2 * self.frame_width), int(y2 * self.frame_height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for person detections

            # Get the first detected person
            y1, x1, y2, x2 = detections['detection_boxes'][0].numpy()[0]
            person_center_x = int((x1 + x2) / 2 * self.frame_width)
            person_center_y = int((y1 + y2) / 2 * self.frame_height)

            return person_center_x, person_center_y, frame

        return None, None, frame

    def adjust_servos(self, person_center_x, person_center_y):
        # Calculate pan/tilt adjustments
        if person_center_x is not None and person_center_y is not None:
            if person_center_x < self.frame_center_x - 50:  # Person is left of center
                self.pan_angle += self.pan_step
            elif person_center_x > self.frame_center_x + 50:  # Person is right of center
                self.pan_angle -= self.pan_step

            if person_center_y < self.frame_center_y - 50:  # Person is above center
                self.tilt_angle += self.tilt_step
            elif person_center_y > self.frame_center_y + 50:  # Person is below center
                self.tilt_angle -= self.tilt_step

            # Clamp angles to valid range (0 to 180 degrees)
            self.pan_angle = max(0, min(180, self.pan_angle))
            self.tilt_angle = max(0, min(180, self.tilt_angle))

            # Update pan/tilt servos
            self.controller.set_pan_tilt((self.pan_angle - 90) / 90, (self.tilt_angle - 90) / 90)