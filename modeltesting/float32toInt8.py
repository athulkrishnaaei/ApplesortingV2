import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image

# Paths
FLOAT_TFLITE = '/home/athul/Work/AppleSorterV2/runs/train/apple_yolo11n/weights/best_saved_model/best_float32.tflite'
QUANT_TFLITE = 'best_int8.tflite'
IMG_DIR      = '/home/athul/Work/AppleSorterV2/Rotten Apple Detection 2.v3i.yolov11/train/images'  # adjust to point at ~100 representative train images
IMG_SIZE     = 640

# Load float32 model
converter = tf.lite.TFLiteConverter.from_saved_model('.',  # or from file:
    # Use from_saved_model if you have a SavedModel dir, otherwise:
    # tf.lite.TFLiteConverter.from_tflite_model_file(FLOAT_TFLITE)
)
# If using file directly:
converter = tf.lite.TFLiteConverter.from_tflite_model_file(FLOAT_TFLITE)

# Setup for full integer quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

def representative_data_gen():
    for i, img_path in enumerate(Path(IMG_DIR).glob('*.jpg')):
        if i >= 100: break
        img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img).astype(np.float32) / 255.0
        yield [np.expand_dims(arr, 0)]

converter.representative_dataset = representative_data_gen

# Convert and save
quant_model = converter.convert()
with open(QUANT_TFLITE, 'wb') as f:
    f.write(quant_model)

print("✅ Wrote fully-quantized model ->", QUANT_TFLITE)
