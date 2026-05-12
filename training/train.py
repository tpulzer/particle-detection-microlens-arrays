# training/train.py
# run this script from container to train model

import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset

import numpy as np
from pathlib import Path
import yaml

from dataset import MemmapDataset

from augmentations import get_train_transforms, get_val_transforms
from model import UNet
from losses import BCEDiceLoss
from metrics import dice_score

import subprocess       # DATALOGGING

import time # DEBUG


# ======================
# Config
# ======================

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "training.yaml"

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)


training_cfg = CONFIG["training"]
resume_cfg = CONFIG["resume"]
paths_cfg = CONFIG["paths"]


BATCH_SIZE = training_cfg["batch_size"]
EPOCHS = training_cfg["epochs"]
LR = training_cfg["learning_rate"]
VAL_SPLIT = training_cfg["validation_split"]

RESUME = resume_cfg["enabled"]

SAVE_DIR = ROOT / paths_cfg["checkpoint_dir"]
RESUME_PATH = ROOT / paths_cfg["resume_checkpoint"]
LOG_FILE = ROOT / paths_cfg["log_file"]



def get_gpu_stats():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True
        )

        util, mem_used, mem_total, temp, power = result.stdout.strip().split(",")

        util = float(util)
        mem_used = float(mem_used)
        mem_total = float(mem_total)
        temp = float(temp)
        power = float(power)

        mem_percent = 100.0 * mem_used / mem_total

        return util, mem_used, mem_total, mem_percent, temp, power

    except:
        return 0, 0, 0, 0, 0, 0



# ======================
# Train / Val Loop
# ======================

def main():

    # CSV Datalogger
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w") as f:
            f.write(
                "epoch,"
                "time_s,"
                "throughput_img_s,"
                "train_loss,"
                "val_loss,"
                "val_dice,"
                "gpu_util_percent,"
                "vram_util_percent,"
                "gpu_temp_c,"
                "gpu_power_w\n"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Memory allocated:", torch.cuda.memory_allocated(0))

        # 🔥 HIER EINFÜGEN
        print("Initializing CUDA...")
        start = time.time()

        _ = torch.zeros(1).cuda()   # zwingt CUDA Init

        print(f"CUDA init took: {time.time() - start:.2f}s")

    else:
        print("Running on CPU!")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # create Dataset
    full_ds = MemmapDataset(
        img_path=ROOT / "data" / "train_data" / "images.npy",
        mask_path=ROOT / "data" / "train_data" / "masks.npy",
        transform=None
    )

    val_ds_full = MemmapDataset(
        img_path=ROOT / "data" / "train_data" / "images.npy",
        mask_path=ROOT / "data" / "train_data" / "masks.npy",
        transform=None
    )

    val_size = int(len(full_ds) * VAL_SPLIT)
    train_size = len(full_ds) - val_size

    indices = np.arange(len(full_ds))
    np.random.shuffle(indices)

    train_ds = Subset(full_ds, indices[:train_size])
    val_ds   = Subset(val_ds_full, indices[train_size:])

    # Loader
    # https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    # https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,      # ggf. erhöhen (siehe unten)
        shuffle=True,
        num_workers=4,              
        pin_memory=True,
        persistent_workers=True,   
        prefetch_factor=2          
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,              # 🔥 weniger als train
        pin_memory=True,
        persistent_workers=True
    )

    num_samples = len(full_ds)

    print(f"Total samples: {num_samples}")
    print(f"Train samples: {train_size}")
    print(f"Val samples:   {val_size}")

    # Model
    model = UNet().to(device)

    # Loss / Optim
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch = 0
    best_dice = 0.0

    # ======================
    # RESUME
    # ======================
    if RESUME and os.path.exists(RESUME_PATH):

        print("Resuming from checkpoint:", RESUME_PATH)

        checkpoint = torch.load( str(RESUME_PATH), map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint["best_dice"]

        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])


    # ======================
    # Training Loop
    # ======================

    scaler = torch.amp.GradScaler("cuda")

    best_dice = 0.0
    patience = 16
    min_delta = 1e-4
    wait = 0

    last_time = time.time()
    for epoch in range(start_epoch, EPOCHS):

        # ======================
        # TRAIN
        # ======================
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:

            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda"):
                preds = model(imgs)
                loss = criterion(preds, masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ======================
        # VALIDATION
        # ======================
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:

                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda"):
                    preds = model(imgs)
                    loss = criterion(preds, masks)

                val_loss += loss.item()
                val_dice += dice_score(preds, masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_dice = float(val_dice)

        # ======================
        # CHECKPOINT (last)
        # ======================
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice
            },
            str(RESUME_PATH)
        )

        now = time.time()
        delta_t = now - last_time
        last_time = now

        util, mem_used, mem_total, mem_percent, temp, power = get_gpu_stats()

        num_samples = len(train_loader.dataset)
        throughput = num_samples / delta_t

        SEP = ";"

        log_msg = (
            f"Epoch {epoch+1:03d} | "
            f"time: {delta_t:5.1f}s | "
            f"throughput: {throughput:6.1f} img/s | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_dice: {val_dice:.4f} | "
            f"GPU: {util:3.0f}% | "
            f"VRAM: {mem_percent:5.1f}% | "
            f"temp: {temp:2.0f}°C | "
            f"power: {power:3.0f}W"
        )

        csv_msg = (
            f"{epoch+1},{delta_t:.1f},{throughput:.1f},"
            f"{train_loss:.4f},{val_loss:.4f},{val_dice:.4f},"
            f"{util:.0f},{mem_percent:.1f},{temp:.0f},{power:.0f}\n"
        )

        with open(LOG_FILE, "a") as f:
            f.write(csv_msg)

        # ======================
        # EARLY STOP + BEST MODEL
        # ======================
        if val_dice > best_dice + min_delta:

            best_dice = val_dice
            wait = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice
                },
                str(SAVE_DIR / "best_model.pt")
            )

            log_msg += " -> model saved"

        else:
            wait += 1

            if wait >= patience:
                log_msg += " -> Early stopping"
                print(log_msg)
                break

        print(log_msg)

if __name__ == "__main__":
    main()
