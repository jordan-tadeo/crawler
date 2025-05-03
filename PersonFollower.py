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
        input_tensor = tf.convert_to_tensor(frame, dtype=tf.uint8)
        input_tensor = tf.image.resize(input_tensor, [128, 128])  # Resize to model's expected input size
        input_tensor = tf.expand_dims(input_tensor, axis=0)  # Add batch dimension

        # Run inference
        detections = self.model(input_tensor)

        # Convert TensorFlow tensor to NumPy array
        detection_array = detections.numpy()[0]  # Remove batch dimension

        # Find the class with the highest probability
        predicted_class = np.argmax(detection_array)
        confidence = detection_array[predicted_class]

        # Debug: Print the predicted class and confidence
        print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")

        # Map the predicted class to a label (assuming ImageNet labels)
        imagenet_labels = {
            0: "background", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 12: "stop sign", 13: "parking meter", 14: "bench", 15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep", 20: "cow", 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe", 25: "backpack", 26: "umbrella", 27: "handbag", 28: "tie", 29: "suitcase", 30: "frisbee", 31: "skis", 32: "snowboard", 33: "sports ball", 34: "kite", 35: "baseball bat", 36: "baseball glove", 37: "skateboard", 38: "surfboard", 39: "tennis racket", 40: "bottle", 41: "wine glass", 42: "cup", 43: "fork", 44: "knife", 45: "spoon", 46: "bowl", 47: "banana", 48: "apple", 49: "sandwich", 50: "orange", 51: "broccoli", 52: "carrot", 53: "hot dog", 54: "pizza", 55: "donut", 56: "cake", 57: "chair", 58: "couch", 59: "potted plant", 60: "bed", 61: "dining table", 62: "toilet", 63: "tv", 64: "laptop", 65: "mouse", 66: "remote", 67: "keyboard", 68: "cell phone", 69: "microwave", 70: "oven", 71: "toaster", 72: "sink", 73: "refrigerator", 74: "book", 75: "clock", 76: "vase", 77: "scissors", 78: "teddy bear", 79: "hair drier", 80: "toothbrush"
        }
        label = imagenet_labels.get(predicted_class, "Unknown")

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