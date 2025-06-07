#!/usr/bin/env python3
"""
inspect_bboxes.py

Load a YOLO-format dataset (images + corresponding .txt label files),
draw each bounding box on its image, and display them one by one.

Usage:
    1. Adjust the variables under “=== Configuration ===” below.
    2. Run:
         python3 inspect_bboxes.py
    3. For each displayed image:
         • Press “n” to move to the next image
         • Press “q” to quit immediately
"""

import cv2
import os
from pathlib import Path

# === Configuration ===
# Path to the root of your YOLO-formatted dataset:
# It should contain:
#   images/
#   labels/
DATASET_ROOT = "/home/athul/Work/AppleSorterV2/Rotten Apple Detection 2.v3i.yolov11/test"

# If your dataset has subfolders (e.g., train/valid/test), point to one of them:
# e.g.: DATASET_ROOT = "path/to/your/dataset/train"
IMAGE_DIR = os.path.join(DATASET_ROOT, "images")
LABEL_DIR = os.path.join(DATASET_ROOT, "labels")

# A mapping from class index to a human-readable class name.
# If you have multiple classes, list them here in order:
#   class 0 -> names[0], class 1 -> names[1], etc.
CLASS_NAMES = [
    "normalApple",
    "rottenApple",
    # add more class names if needed
]

# Colors for each class (BGR tuples). If you have fewer colors than classes,
# colors will repeat cyclically.
CLASS_COLORS = [
    (0, 0, 255),    # red for class 0
    (0, 255, 0),    # green for class 1
    (255, 0, 0),    # blue for class 2 (if added)
]

# === End of Configuration ===


def draw_yolo_bboxes(image, label_path, class_names, class_colors):
    """
    Parse the YOLO-format .txt file at label_path and draw each bbox on image.

    YOLO .txt format per line:
        <class_id> <x_center> <y_center> <width> <height>
    where coordinates are normalized [0..1] relative to image width/height.

    Returns the image with drawn boxes.
    """
    h, w = image.shape[:2]
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"  ❌ Skipping invalid line in {label_path!r}: {line}")
                continue
            cls_id, x_c, y_c, bw, bh = parts
            try:
                cls_id = int(cls_id)
                x_c = float(x_c)
                y_c = float(y_c)
                bw = float(bw)
                bh = float(bh)
            except ValueError:
                print(f"  ❌ Could not parse numbers in {label_path!r}: {line}")
                continue

            # Convert normalized to pixel coordinates
            cx = x_c * w
            cy = y_c * h
            bw_px = bw * w
            bh_px = bh * h
            x1 = int(cx - bw_px / 2)
            y1 = int(cy - bh_px / 2)
            x2 = int(cx + bw_px / 2)
            y2 = int(cy + bh_px / 2)

            # Clamp to image bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            # Choose color and label text
            color = class_colors[cls_id % len(class_colors)]
            label_text = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

            # Draw rectangle and label
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text_pos = (x1, y1 - 6 if y1 - 6 > 10 else y1 + 12)
            cv2.putText(
                image,
                f"{label_text}:{cls_id}",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

    return image


def main():
    image_dir = Path(IMAGE_DIR)
    label_dir = Path(LABEL_DIR)

    if not image_dir.is_dir() or not label_dir.is_dir():
        print(f"Error: IMAGE_DIR={IMAGE_DIR!r} or LABEL_DIR={LABEL_DIR!r} does not exist.")
        return

    image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not image_files:
        print(f"No images found in {IMAGE_DIR!r}.")
        return

    print(f"Found {len(image_files)} images. Press 'n' for next, 'q' to quit.")

    idx = 0
    while idx < len(image_files):
        img_path = image_files[idx]
        lbl_path = label_dir / (img_path.stem + ".txt")

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Could not load image: {img_path}. Skipping.")
            idx += 1
            continue

        if lbl_path.exists():
            image_drawn = draw_yolo_bboxes(image.copy(), str(lbl_path), CLASS_NAMES, CLASS_COLORS)
        else:
            image_drawn = image.copy()
            print(f"Label file not found for {img_path.name}, skipping bbox draw.")

        # Resize window if image is too large
        max_dim = 800
        h0, w0 = image_drawn.shape[:2]
        scale = min(max_dim / w0, max_dim / h0, 1.0)
        if scale < 1.0:
            image_disp = cv2.resize(image_drawn, (int(w0 * scale), int(h0 * scale)))
        else:
            image_disp = image_drawn

        cv2.imshow("BBox Inspection", image_disp)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key in (ord("n"), 83, 81, 82, 84):  # 'n' or arrow keys
            idx += 1
            continue
        else:
            # Any other key also moves to next
            idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
