import cv2
import numpy as np
import tensorflow as tf
from tflite_runtime.interpreter import Interpreter

# Load the TFLite model
model_path = "models/person_detect.tflite"
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open the USB camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit.")

while True:
    # Capture frame from the camera
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Preprocess the frame for the model
    input_shape = input_details[0]['shape']
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 255.0

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    person_detected = output_data[0] > 0.5  # Assuming the model outputs a confidence score

    # Display the result
    label = "Person Detected" if person_detected else "No Person"
    color = (0, 255, 0) if person_detected else (0, 0, 255)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Person Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()