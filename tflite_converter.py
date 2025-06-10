# tflite_converter.py

from ultralytics import YOLO

if __name__ == "__main__":
    # 1) Load the model (defaults to CPU)
    model = YOLO("best.pt")

    # 2) Export, telling Ultralytics to use CUDA,
    #    limit calibration (fraction) and use mixed precision
    model.export(
        format      = "tflite",
        int8        = True,
        #activations = "float16",
        data        = "/home/athul/Work/AppleSorterV2/Rotten Apple Detection 2.v3i.yolov11/data.yaml",
        imgsz       = 320,
        batch       = 1,
        fraction    = 0.1,           # only ~2 of 22 images in memory
        device      = "cuda:0"       # <<< move export to your GPU
    )

    print("✅ GPU‐accelerated mixed‐precision TFLite export done.")
