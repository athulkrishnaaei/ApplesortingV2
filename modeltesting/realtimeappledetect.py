#!/usr/bin/env python3
"""
realtime_apple_detect.py

Use your trained YOLOv11n model to detect fresh vs. rotten apples
from a webcam feed and draw bounding boxes & labels in real time.

Prerequisites:
  pip install ultralytics opencv-python

Usage:
  Just run:
    python3 realtime_apple_detect.py
"""

import cv2
from ultralytics import YOLO

# === Configuration ===
MODEL_PATH   = "runs/train/apple_yolo11n/weights/best.pt"  # path to your trained .pt
CAMERA_INDEX = 0                                          # 0 = default webcam
CONFIDENCE   = 0.25                                       # detection threshold
STREAM_URL = "http://192.168.178.153:8080/video"
  
# === Load model ===
model = YOLO(MODEL_PATH)      # loads the PyTorch checkpoint
class_names = model.names     # dictionary: {0: 'rottenApple', 1: 'freshApple'}

# === Open IP-Webcam stream ===
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open stream: {STREAM_URL}")


print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference (returns a list of Results; we take the first)
    results = model(frame, conf=CONFIDENCE)[0]

    # Parse boxes, classes, scores
    boxes      = results.boxes.xyxy.cpu().numpy()  # shape (N,4) in [x1,y1,x2,y2]
    scores     = results.boxes.conf.cpu().numpy()  # shape (N,)
    class_ids  = results.boxes.cls.cpu().numpy().astype(int)  # shape (N,)

    # Draw each detection
    for (x1, y1, x2, y2), score, cls in zip(boxes, scores, class_ids):
        label = f"{class_names[cls]} {score:.2f}"
        color = (0,0,255) if cls == 0 else (0,255,0)  # red for rotten, green for fresh

        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show the frame
    cv2.imshow("Apple Sorter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
