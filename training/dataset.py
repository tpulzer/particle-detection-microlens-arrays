# training/dataset.py
# laod dataset for training
# dataset format: image, mask

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


#OBSOLETE
class ParticleDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform


        # Alle Bilddateien sammeln
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".png")
        ])

        self.masks = sorted([
            f for f in os.listdir(mask_dir)
            if f.endswith(".png")
        ])

        assert len(self.images) == len(self.masks), \
            "Anzahl Bilder und Masken stimmt nicht überein"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]
        mask_name = self.masks[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Graustufen laden
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"Fehler beim Laden: {img_name}")

        # Normalisieren
        image = image.astype("float32") / 255.0
        mask = (mask > 0).astype("float32")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Tensor: (1, H, W)
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


# OBSOLETE
class NumpyDataset(Dataset):

    def __init__(self, img_path, mask_path, transform=None):
        self.images = np.load(img_path)
        self.masks  = np.load(mask_path)
        self.transform = transform

        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        mask  = self.masks[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = torch.from_numpy(image).unsqueeze(0).float()
        mask  = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask
    


class MemmapDataset(Dataset):

    def __init__(self, img_path, mask_path, transform=None):
        self.images = np.load(img_path, mmap_mode="r")
        self.masks  = np.load(mask_path, mmap_mode="r")
        self.transform = transform

        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        mask  = self.masks[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]


        image = torch.from_numpy(image).unsqueeze(0).float()
        mask  = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask