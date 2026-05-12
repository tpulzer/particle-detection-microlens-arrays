# tools/test_model.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT))

import torch
from training.model import UNet

def main():

    model = UNet()

    x = torch.randn(
        1,
        1,
        512,
        512
    )

    y = model(x)

    print("Input:", x.shape)
    print("Output:", y.shape)

if __name__ == "__main__":
    main()
