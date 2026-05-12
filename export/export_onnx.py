# export/export_onnx.py

# export best model to ONNX

import torch
import os

from training.model import UNet


# -----------------------
# Config
# -----------------------

CHECKPOINT = "checkpoints/best_model.pt"
OUT_DIR = "export"
ONNX_NAME = "particle_unet.onnx"

IMG_SIZE = 512


# -----------------------
# Main
# -----------------------

def main():

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("Device:", device)

    out_path = os.path.join(OUT_DIR, ONNX_NAME)

    # Model
    model = UNet().to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Dummy input
    dummy = torch.randn(
        1, 1, IMG_SIZE, IMG_SIZE,
        device=device
    )

    # Export
    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"}
        }
    )

    print("Exported to:", out_path)


if __name__ == "__main__":
    main()
