"""Make the person detector's INT8 model (models/person.onnx), on the Mac.

    uvx --python 3.11 --from 'ultralytics==8.3.40' --with onnx --with onnxslim \
        --with 'torch<2.6' --with 'torchvision<0.21' \
        yolo export model=yolo11n.pt format=onnx imgsz=192,320 opset=12
    uvx --python 3.11 --with onnxruntime --with onnx --with sympy \
        --with opencv-python-headless --with numpy \
        python tools/person_model.py yolo11n.onnx person-192x320-int8.onnx

The first is the float model, shaped for the 16:9 windshield camera (see
crawler/vision/person.py). This quantises it: opset 13 (per-channel
DequantizeLinear needs it), static QDQ, int8 weights per channel, uint8
activations, MinMax calibration on every other coco128 image put in a 16:9
grey canvas and letterboxed exactly as the car does. The detection head
(everything under /model.23/ but its convolutions) stays float, so the
boxes and scores come out as they did. coco128 (7 MB, Ultralytics) is
downloaded into --data the first time. Neither it nor the model goes in the
repo.

On the car this ran 95 ms against the float model's 182 ms at the same size,
and found 93 of coco128's 254 people at conf 0.4 to the float 256x320
model's 89.
"""

import argparse
import glob
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data_paths as dp  # noqa: E402

COCO128 = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
PAD_VALUE = 114


def to_16_9(img):
    """img in the top left of a grey 16:9 canvas, the way a 16:9 camera
    would frame it."""
    import numpy as np
    h, w = img.shape[:2]
    if w * 9 >= h * 16:
        H, W = int(round(w * 9 / 16)), w
    else:
        H, W = h, int(round(h * 16 / 9))
    canvas = np.full((H, W, 3), PAD_VALUE, np.uint8)
    canvas[:h, :w] = img
    return canvas


def calibration_images(data):
    folder = os.path.join(data, "coco128", "images", "train2017")
    if not os.path.isdir(folder):
        os.makedirs(data, exist_ok=True)
        zpath = os.path.join(data, "coco128.zip")
        urllib.request.urlretrieve(COCO128, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(data)
    return sorted(glob.glob(os.path.join(folder, "*.jpg")))[::2]


def quantise(src, dst, files):
    import cv2
    import onnx
    from onnx import version_converter
    from onnxruntime.quantization import (CalibrationDataReader,
                                          CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from crawler.vision.person import letterbox

    op13, pre = dst + ".op13.onnx", dst + ".pre.onnx"
    try:
        onnx.save(version_converter.convert_version(onnx.load(src), 13), op13)
        quant_pre_process(op13, pre)
        model = onnx.load(pre)
        inp = model.graph.input[0]
        _, _, in_h, in_w = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        head = [n.name for n in model.graph.node
                if n.name.startswith("/model.23/") and n.op_type != "Conv"]

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter(files)

            def get_next(self):
                f = next(self.it, None)
                if f is None:
                    return None
                frame = to_16_9(cv2.imread(f))
                return {inp.name: letterbox(frame, in_h, in_w)[0]}

        quantize_static(pre, dst, Reader(), quant_format=QuantFormat.QDQ,
                        per_channel=True, activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax,
                        nodes_to_exclude=head)
    finally:
        for p in (op13, pre):
            if os.path.exists(p):
                os.remove(p)
    return in_h, in_w, len(head)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", help="the float YOLO .onnx from yolo export")
    ap.add_argument("dst", help="where the INT8 .onnx goes")
    ap.add_argument("--data", default=os.path.join(dp.DATA_ROOT, "coco128"),
                    help="where coco128 is, or is downloaded to")
    a = ap.parse_args(argv)
    files = calibration_images(a.data)
    in_h, in_w, kept = quantise(a.src, a.dst, files)
    print(f"{a.dst}: input {in_w}x{in_h}, calibrated on {len(files)} images, "
          f"{kept} head nodes left float")


if __name__ == "__main__":
    main()
