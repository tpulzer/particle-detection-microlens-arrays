# training/preview_augmentation.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT))

import matplotlib.pyplot as plt

from training.dataset import MemmapDataset
from training.augmentations import get_train_transforms

def main():

    transforms = get_train_transforms()

    dataset = MemmapDataset(
        img_path=ROOT / "data" / "npy" / "images.npy",
        mask_path=ROOT / "data" / "npy" / "masks.npy",
        transform=transforms
    )

    for i in range(8):

        img, mask = dataset[i]

        img_np = img.squeeze().numpy()
        mask_np = mask.squeeze().numpy()

        # Overlap-Test
        # Dunkle Partikel + Maske
        overlap = (img_np < 0.4) & (mask_np > 0.5)
        ratio = overlap.sum() / (mask_np > 0.5).sum()

        print(f"Sample {i} - Overlap ratio: {ratio:.3f}")


        plt.figure(figsize=(4,4))

        plt.imshow(img_np, cmap="gray")
        plt.imshow(mask_np, alpha=0.4, cmap="Reds")

        plt.title(f"Sample {i}")
        plt.axis("off")
        plt.show()



if __name__ == "__main__":
    main()
