# training/test_dataset.py

# create and laod dataset
# convert to numpy tensor -> ndarray
# show sample(image, mask) in pyplot

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT))

import matplotlib.pyplot as plt

from training.dataset import MemmapDataset


def main():

    dataset = MemmapDataset(
        img_path=ROOT / "data" / "npy" / "images.npy",
        mask_path=ROOT / "data" / "npy" / "masks.npy"
    )

    print("Number of samples:", len(dataset))

    img, mask = dataset[0]

    print("Image shape:", img.shape)

    print("Mask shape:", mask.shape)

    img_np = img.squeeze().numpy()

    mask_np = mask.squeeze().numpy()

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)

    plt.title("Image")

    plt.imshow(img_np, cmap="gray")

    plt.axis("off")

    plt.subplot(1, 2, 2)

    plt.title("Mask")

    plt.imshow(mask_np, cmap="gray")

    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()