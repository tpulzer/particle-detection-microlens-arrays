import os
import numpy as np
import cv2
from pathlib import Path    # https://docs.python.org/3/library/pathlib.html?utm_source=chatgpt.com

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR   = ROOT / "data" / "generated" / "images"
MASK_DIR    = ROOT / "data" / "generated" / "masks"
OUT_IMAGES  = ROOT / "data" / "npy" / "images.npy"
OUT_MASKS   = ROOT / "data" / "npy" / "masks.npy"


def main():

    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")])
    mask_files  = sorted([f for f in os.listdir(MASK_DIR) if f.endswith(".png")])

    assert len(image_files) == len(mask_files)

    N = len(image_files)

    # Beispielbild laden für Shape
    sample = cv2.imread(os.path.join(IMAGE_DIR, image_files[0]), cv2.IMREAD_GRAYSCALE)
    H, W = sample.shape

    images = np.zeros((N, H, W), dtype=np.float32)
    masks  = np.zeros((N, H, W), dtype=np.float32)

    for i, (img_f, mask_f) in enumerate(zip(image_files, mask_files)):
        if i % 100 == 0:
            print(f"{i}/{N}")

        img = cv2.imread(os.path.join(IMAGE_DIR, img_f), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(MASK_DIR, mask_f), cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        images[i] = img
        masks[i]  = mask

    OUT_IMAGES.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(OUT_IMAGES, images)
    np.save(OUT_MASKS, masks)

    print("Saved:", OUT_IMAGES, OUT_MASKS)


if __name__ == "__main__":
    main()