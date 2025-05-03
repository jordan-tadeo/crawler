import tensorflow as tf
from USBCamera import USBCamera
from VehicleController import VehicleController
import cv2
import warnings
import threading
import numpy as np
import os
import kagglehub
import tflite_runtime.interpreter as tflite
import tensorflow_hub as hub

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

class PersonFollower:
    def __init__(self, vehicle_controller: VehicleController, usb_cam: USBCamera):
        # Load the MobileNetV2 model using hub.KerasLayer
        # Ensure the model path points to the directory containing `saved_model.pb`
        local_model_path = "/home/jt/Documents/py/crawler/models/mobilenet_v2"
        if not os.path.exists(os.path.join(local_model_path, "saved_model.pb")):
            raise FileNotFoundError(f"Model not found at {local_model_path}. Please ensure the directory contains `saved_model.pb`." )
        print("Loading MobileNetV2 model from local path:", local_model_path)
        self.model = hub.KerasLayer(local_model_path, trainable=False)
        self.model.build([None, 128, 128, 3])  # Batch input shape

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

        # Load ImageNet labels from the file
        labels_path = "/home/jt/Documents/py/crawler/labels/ImageNetLabels.txt"
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found at {labels_path}. Please ensure the file exists.")
        with open(labels_path, "r") as f:
            self.imagenet_labels = [line.strip() for line in f.readlines()]

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
            # frame = cv2.resize(frame, (self.frame_width, self.frame_height))

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
        input_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)  # Convert to float32
        input_tensor = tf.image.resize(input_tensor, [128, 128])  # Resize to 128x128 pixels
        input_tensor = input_tensor / 255.0  # Normalize to [0,1]
        input_tensor = tf.expand_dims(input_tensor, axis=0)  # Add batch dimension

        # Run inference
        logits = self.model(input_tensor)

        # Get the predicted class and confidence
        predicted_class = tf.argmax(logits, axis=-1).numpy()[0]
        confidence = tf.nn.softmax(logits, axis=-1).numpy()[0][predicted_class]

        # Debug: Print the predicted class and confidence
        print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")

        # Map the predicted class to a label (assuming ImageNet labels)
        label = self.imagenet_labels[predicted_class] if predicted_class < len(self.imagenet_labels) else "Unknown"

        # Visualize the prediction on the frame
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # If the predicted class is 'person', return the center of the frame
        if label == "person":
            person_center_x = self.frame_width // 2
            person_center_y = self.frame_height // 2
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