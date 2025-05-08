import cv2
import numpy as np
import tensorflow as tf
import os

# Load the SavedModel
def load_saved_model(model_dir):
    print(f"Loading SavedModel from: {model_dir}")
    model = tf.saved_model.load(model_dir)
    return model.signatures['serving_default']

# Load the model
saved_model_dir = "./"
movenet = load_saved_model(saved_model_dir)

# Function to preprocess the image
def preprocess_image(image, input_shape):
    height, width = input_shape[1], input_shape[2]
    image = cv2.resize(image, (width, height))
    image = np.expand_dims(image, axis=0)
    image = np.array(image, dtype=np.int32)  # Cast to INT32 as required by the model
    return image

# Function to draw only high-confidence keypoints and skeleton
def draw_pose_estimation(image, outputs):
    keypoint_pairs = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]

    h, w, _ = image.shape
    for person in outputs[0]:
        keypoints = person[:51].reshape((17, 3))  # Extract keypoints
        ymin, xmin, ymax, xmax, score = person[-5:]  # Bounding box and confidence

        # Draw bounding box if confidence is high
        if score > 0.5:  # Reasonable threshold for bounding box
            start_point = (max(0, int(xmin * w)), max(0, int(ymin * h)))
            end_point = (min(w, int(xmax * w)), min(h, int(ymax * h)))
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)  # Green box

        print("Keypoints (filtered and scaled to image):")
        for i, (y, x, confidence) in enumerate(keypoints):
            if confidence > 0.5:  # Show only high-confidence keypoints
                scaled_x, scaled_y = int(x * w), int(y * h)
                print(f"Joint {i}: ({scaled_x}, {scaled_y}), Confidence: {confidence}")
                cv2.circle(image, (scaled_x, scaled_y), 5, (0, 0, 255), -1)  # Red keypoint

        for start, end in keypoint_pairs:
            y1, x1, c1 = keypoints[start]
            y2, x2, c2 = keypoints[end]
            if c1 > 0.5 and c2 > 0.5:  # Show only high-confidence connections
                start_point = (int(x1 * w), int(y1 * h))
                end_point = (int(x2 * w), int(y2 * h))
                cv2.line(image, start_point, end_point, (255, 0, 0), 2)  # Blue skeleton
    return image

# Save preprocessed input for debugging
def save_preprocessed_input(image, filename="preprocessed_input.jpg"):
    cv2.imwrite(filename, image)

# Open a video stream (webcam)
cap = cv2.VideoCapture(0)

# Replace the test rectangle with pose estimation drawing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_shape = (1, 256, 256, 3)  # Update to match SavedModel input
    preprocessed_frame = preprocess_image(frame, input_shape)

    # Run inference using the SavedModel
    inputs = tf.convert_to_tensor(preprocessed_frame, dtype=tf.int32)
    outputs = movenet(inputs)

    # Extract the output tensor
    output_data = outputs['output_0'].numpy()

    # Draw pose estimation on the frame
    frame_with_pose = draw_pose_estimation(frame, output_data)

    # Display the frame
    cv2.imshow('Pose Estimation', frame_with_pose)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()