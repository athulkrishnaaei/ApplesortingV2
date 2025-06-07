# train_yolo11n.py

from ultralytics import YOLO

# === Configuration ===
# Path to the YOLOv11n pretrained weights (download from Ultralytics or use your local copy)
PRETRAINED_WEIGHTS = "yolo11n.pt"  

# Path to your dataset’s data.yaml (YOLO format):
# Example data.yaml contents:
#   train: /home/newin/projects/Applesort/train/images
#   val:   /home/newin/projects/Applesort/val/images
#   test:  /home/newin/projects/Applesort/test/images
#   nc: 1
#   names:
#     0: 'rottenApple'
DATA_YAML = "/home/athul/Work/AppleSorterV2/Rotten Apple Detection 2.v3i.yolov11/data.yaml"

# Number of epochs, image size, batch size
EPOCHS = 50
IMGSZ  = 640
BATCH  = 16

# Name for this training run (outputs go to runs/detect/...)
RUN_NAME = "apple_yolo11n"

# === Training ===
if __name__ == "__main__":
    # Create a YOLO object using the nano‐version
    model = YOLO(PRETRAINED_WEIGHTS)  

    # Start training
    model.train(
        data=DATA_YAML,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        name=RUN_NAME,
        device="0",      # or "0" for GPU 0 if available
        project="runs/train", 
        exist_ok=True      # overwrite if run name already exists
    )
