from ultralytics import YOLO

# 1) Load your best weights
model = YOLO('runs/train/apple_yolo11n/weights/best.pt')

# 2) Evaluate on the validation split
#    Prints metrics and returns a Results object
results = model.val(
    data='/home/athul/Work/AppleSorterV2/Rotten Apple Detection 2.v3i.yolov11/data.yaml',
    imgsz=640,
    batch=16,
)

# 3) Run inference on a single image
#    and save the annotated result to disk
# res = model.predict(
#     source='path/to/your/test_image.jpg',
#     imgsz=640,
#     conf=0.25,
#     save=True,
#     save_dir='runs/infer/apple_single',
# )

# 4) Or run inference on a whole folder
res = model.predict(
    source='/home/athul/Work/AppleSorterV2/Rotten Apple Detection 2.v3i.yolov11/train/images',
    imgsz=640,
    conf=0.25,
    save=True,
    save_dir='runs/infer/apple_batch',
)
