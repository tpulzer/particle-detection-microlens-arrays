import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader

from training.dataset import MemmapDataset
from training.model import UNet


def find_max_batch(model, dataset, device):

    model.train()

    for bs in [1, 2, 4, 8, 16, 32, 64]:

        try:

            loader = DataLoader(
                dataset,
                batch_size=bs
            )

            imgs, masks = next(iter(loader))

            imgs = imgs.to(device)

            masks = masks.to(device)

            out = model(imgs)

            loss = out.mean()

            model.zero_grad(set_to_none=True)

            loss.backward()

            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3

            print(
                f"Batch {bs}: OK | "
                f"allocated={allocated:.2f} GB | "
                f"reserved={reserved:.2f} GB"
            )

        except RuntimeError as e:

            print(
                f"Batch {bs}: FAIL "
                f"({str(e).splitlines()[0]})"
            )

            torch.cuda.empty_cache()


def main():
    print("Press CTRL+C to abort.")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = MemmapDataset(
        img_path=ROOT / "data" / "npy" / "images.npy",
        mask_path=ROOT / "data" / "npy" / "masks.npy"
    )

    model = UNet().to(device)

    print("Testing batch sizes...")

    find_max_batch(
        model,
        dataset,
        device
    )

if __name__ == "__main__":
    main()