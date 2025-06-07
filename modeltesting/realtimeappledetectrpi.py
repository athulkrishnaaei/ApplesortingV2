import cv2, numpy as np
from tflite_runtime.interpreter import Interpreter

# === Configuration ===
MODEL_PATH  = "best_int8.tflite"
STREAM_URL  = "http://192.168.178.153:8080/video"
CONFIDENCE  = 0.25
CLASS_NAMES = ["rottenApple","freshApple"]

# === Load TFLite model ===
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

# get input dims
_, H, W, _ = inp_det['shape']
