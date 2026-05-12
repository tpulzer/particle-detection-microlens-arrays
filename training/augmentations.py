# training/augmentations.py

import albumentations as A

def get_train_transforms():

    return A.Compose([

        # Geometrie
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Kleine photometrische Änderungen
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3
        ),

    ])


def get_val_transforms():

    # Keine Augmentierung für Validation
    return A.Compose([])
