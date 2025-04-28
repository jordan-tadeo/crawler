import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera")
else:
    ret, frame = cap.read()
    if ret:
        print("Frame captured successfully")
    else:
        print("Failed to capture frame")
cap.release()