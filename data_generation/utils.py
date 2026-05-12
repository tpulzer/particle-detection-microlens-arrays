from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "data" / "generated"

IMAGE_DIR = OUTPUT_DIR / "images"
MASK_DIR = OUTPUT_DIR / "masks"


def save_sample(image, mask, idx):

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MASK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_u8 = np.clip(
        image * 255.0,
        0,
        255
    ).astype(np.uint8)

    mask_u8 = (
        (mask > 0)
        .astype(np.uint8)
        * 255
    )

    image_path = IMAGE_DIR / f"img_{idx:04d}.png"

    mask_path = MASK_DIR / f"mask_{idx:04d}.png"

    cv2.imwrite(
        str(image_path),
        image_u8
    )

    cv2.imwrite(
        str(mask_path),
        mask_u8
    )