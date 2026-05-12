import numpy as np
import cv2
import random

def apply_augmentations(image, config):

    effects_cfg = config["effects"]

    sigma = random.uniform(
        *effects_cfg["blur_sigma_range"]
    )

    if sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), sigma)

    noise = random.uniform(
        *effects_cfg["noise_sigma_range"]
    )

    image += np.random.normal(
        0,
        noise,
        image.shape
    )

    return np.clip(image, 0.0, 1.0)


def apply_illumination(image, config):

    illumination_cfg = config["illumination"]

    size = image.shape[0]

    strength_min, strength_max = (
        illumination_cfg["strength_range"]
    )

    strength = random.uniform(
        strength_min,
        strength_max
    )

    y, x = np.indices((size, size))

    illumination = np.ones_like(image, dtype=float)

    mode = illumination_cfg["type"]

    # --------------------------------------------------
    # LINEAR (mit Rotation)
    # --------------------------------------------------
    if mode in ["linear", "both"]:

        angle = random.uniform(0, 2 * np.pi)

        # Richtung
        dx = np.cos(angle)
        dy = np.sin(angle)

        # Projektionsgradient
        grad = (x * dx + y * dy)

        # normalisieren
        grad -= grad.min()
        grad /= (grad.max() + 1e-8)

        linear = 1.0 + (strength - 1.0) * grad

        illumination *= linear

    # --------------------------------------------------
    # RADIAL (mit zufälligem Zentrum)
    # --------------------------------------------------
    if mode in ["radial", "both"]:

        cx = random.uniform(0, size)
        cy = random.uniform(0, size)

        r = np.sqrt((x - cx)**2 + (y - cy)**2)

        r /= (r.max() + 1e-8)

        radial = 1.0 - r * (1.0 - strength)

        illumination *= radial

    # -------------------------
    # OPTIONAL: Inhomogenität
    # -------------------------
    if illumination_cfg["noise"] > 0:

        illumination += np.random.normal(
            0,
            illumination_cfg["noise"],
            size=illumination.shape
        )

    # -------------------------
    # OPTIONAL: Clamp
    # -------------------------
    if illumination_cfg["clamp"]:

        illumination = np.clip(
            illumination,
            0.2,
            2.0
        )

    # -------------------------
    # Anwendung auf Bild
    # -------------------------
    image = image * illumination

    # -------------------------
    # OPTIONAL: Gamma
    # -------------------------
    if illumination_cfg["gamma_correction"]:

        gmin, gmax = (
            illumination_cfg["gamma_range"]
        )

        gamma = random.uniform(gmin, gmax)

        image = image ** gamma

    return image