import cv2
import numpy as np
import tensorflow as tf
from tflite_runtime.interpreter import Interpreter
import os
import logging

def main():
    pb_model_path = "./"
    tflite_model_path = "./model.tflite"

    # Enable TensorFlow Select for unsupported ops
    converter = tf.lite.TFLiteConverter.from_saved_model(pb_model_path)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops
        tf.lite.OpsSet.SELECT_TF_OPS     # Enable TensorFlow Select ops
    ]

    # Debugging the TFLite conversion process
    if not os.path.exists(tflite_model_path) or os.path.getsize(tflite_model_path) == 0:
        print("The .tflite file is missing or empty. Attempting to create it from the .pb file...")
        try:
            print("Starting TFLite conversion with TensorFlow Select enabled...")
            tflite_model = converter.convert()
            with open(tflite_model_path, "wb") as f:
                f.write(tflite_model)
            print(".tflite file created successfully.")
        except Exception as e:
            print("Error during TFLite conversion:", e)
            logging.exception("Exception occurred during TFLite conversion")
            exit(1)

if __name__ == "__main__":
    main()