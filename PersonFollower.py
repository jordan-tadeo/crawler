import torch
from USBCamera import USBCamera
from VehicleController import VehicleController

class PersonFollower:
    def __init__(self, vehicle_controller: VehicleController, usb_cam: USBCamera):
        # Initialize YOLO model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5 small model

        # Initialize camera and vehicle controller
        self.camera = usb_cam
        self.controller = vehicle_controller

        # Frame dimensions (assume 640x480 for now, adjust dynamically if needed)
        self.frame_width = 640
        self.frame_height = 480
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

    def process_frame(self, frame):
        # Run YOLO model on the frame
        results = self.model(frame)
        detections = results.xyxy[0]  # Get detections

        # Filter for person class (class ID 0 in COCO dataset)
        person_detections = [d for d in detections if int(d[5]) == 0]

        if person_detections:
            # Get the first detected person (largest bounding box can be prioritized)
            x1, y1, x2, y2, conf, cls = person_detections[0]
            person_center_x = int((x1 + x2) / 2)
            person_center_y = int((y1 + y2) / 2)
            self.person_detected = True
            return person_center_x, person_center_y
        
        self.person_detected = False
        return None, None

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

    def process_and_adjust(self):
        try:
            # Capture frame from camera
            frame = self.camera.get_frame()

            # Process frame to detect person
            person_center_x, person_center_y = self.process_frame(frame)

            # Adjust servos to keep person in frame
            self.adjust_servos(person_center_x, person_center_y)

        except Exception as e:
            print(f"Error processing frame: {e}")