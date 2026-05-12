# inference_py/run_onnx.py

import os
import cv2
import numpy as np
import onnxruntime as ort        # https://onnxruntime.ai/docs/api/python/index.html
from time import perf_counter_ns # https://docs.python.org/3/library/time.html
from pathlib import Path
import yaml


# -------------------------
# Config
# -------------------------
ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "inference.yaml"

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

inference_cfg = CONFIG["inference"]
paths_cfg = CONFIG["paths"]

MODEL_PATH = ROOT / paths_cfg["model_path"]
INPUT_DIR = ROOT / paths_cfg["input_dir"]
OUTPUT_DIR = ROOT / paths_cfg["output_dir"]

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMG_SIZE = inference_cfg["image_size"]
THRESHOLD = inference_cfg["threshold"]


# -------------------------
# Utils
# -------------------------

def preprocess(img):

    img = img.astype(np.float32) / 255.0    # convert and normalize to 0.0-1.0 (float32)

    img = np.expand_dims(img, axis=0)   # Channels
    img = np.expand_dims(img, axis=0)   # Batch

    return img


def overlay(image, mask):

    color = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    color[:, :, 2] = (mask * 255).astype(np.uint8)

    blended = cv2.addWeighted(
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
        0.7,
        color,
        0.3,
        0
    )

    return blended


def get_session():

    providers = []

    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
        print("Using CUDAExecutionProvider")

    providers.append("CPUExecutionProvider")

    return ort.InferenceSession(
        str(MODEL_PATH),
        providers=providers
    )


# -------------------------
# Main
# -------------------------

def main():

    session = get_session()

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    files = sorted(INPUT_DIR.iterdir())

    print("Found", len(files), "images")

    for path in files:

        img = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            continue

        x = preprocess(img)

        start = perf_counter_ns()

        pred = session.run(
            [output_name],
            {input_name: x}
        )[0]

        stop = perf_counter_ns()

        prob = pred[0, 0]

        mask = (prob > THRESHOLD).astype(np.uint8)

        result = overlay(img, mask)

        out_path = OUTPUT_DIR / path.name

        cv2.imwrite(
            str(out_path),
            result
        )

        print(
            "Saved:",
            out_path,
            " dt = ",
            (stop - start) / 1000000,
            " ms"
        )

if __name__ == "__main__":
    main()
