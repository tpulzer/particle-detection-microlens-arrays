import numpy as np

from enum import Enum


class StructureType(Enum):
    LINES = "lines"
    RINGS = "rings"
    NONLINEAR_RINGS = "nonlinear_rings"
    FRESNEL_ARRAY = "fresnel_array"


def generate_structure(config):

    general_cfg = config["general"]
    background_cfg = config["background"]

    size = general_cfg["image_size"]

    structure_type = StructureType(
        background_cfg["structure_type"]
    )

    if structure_type == StructureType.LINES:

        line_cfg = background_cfg["line"]

        return generate_lines(
            size,
            line_cfg["period"],
            line_cfg["width"]
        )

    elif structure_type == StructureType.RINGS:

        ring_cfg = background_cfg["ring"]

        return generate_rings(
            size,
            ring_cfg["period"],
            ring_cfg["width"]
        )

    elif structure_type == StructureType.NONLINEAR_RINGS:

        ring_cfg = background_cfg["ring"]

        return generate_nonlinear_rings(
            size,
            ring_cfg["period"],
            ring_cfg["width"],
            ring_cfg["log_scale"]
        )

    elif structure_type == StructureType.FRESNEL_ARRAY:

        ring_cfg = background_cfg["ring"]
        array_cfg = background_cfg["array"]

        return generate_hex_fresnel_voronoi(
            size=size,
            pitch=array_cfg["pitch"],
            ring_period=ring_cfg["period"],
            ring_width=ring_cfg["width"],
            ring_log_scale=ring_cfg["log_scale"]
        )

    else:
        raise ValueError(
            f"Unknown structure_type: {structure_type}"
        )


def generate_lines(size, period, width):
    x = np.arange(size)
    pattern = ((x % period) < width).astype(float)
    image = np.tile(pattern, (size, 1))
    return image


def generate_rings(size, period, width):
    # Koordinatengitter
    y, x = np.indices((size, size))
    cx = cy = size // 2

    # radialer Abstand vom Zentrum
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    # periodisches Ringsignal
    rings = ((r % period) < width).astype(float)

    return rings

def generate_nonlinear_rings(size, ring_period, ring_width, ring_log_scale):
    y, x = np.indices((size, size))
    cx = cy = size // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    # ---- bewusst über den Bildrand hinaus ----
    overshoot = 1.25
    r_max = overshoot * (size // 2)

    # ---- Anzahl der Ringe ----
    num_rings = max(5, int(r_max / ring_period))

    # ---- Parameterraum ----
    t = np.linspace(0.0, 1.0, num_rings)

    # ---- NICHTLINEARE VERTEILUNG (außen dicht, innen weit) ----
    # außen: kleine Abstände
    # innen: große Abstände
    radii = r_max * (1.0 - t ** ring_log_scale)

    radii = radii[radii > ring_width] #filtert den punkt in der mitte heraus

    # ---- Ringe zeichnen ----
    rings = np.zeros_like(r, dtype=bool)
    half_w = ring_width / 2.0

    for rc in radii:
        rings |= np.abs(r - rc) < half_w

    # Hintergrund weiß, Ringe schwarz
    return 1.0 - rings.astype(float)


def generate_hex_fresnel_voronoi(
    size=512,
    pitch=320,
    ring_period=24,
    ring_width=2.0,
    ring_log_scale=1.5,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    # --- Pixelkoordinaten ---
    y, x = np.indices((size, size))

    # --- Rotation um Bildzentrum ---
    angle = np.random.uniform(0, 2 * np.pi)

    cx_img = size / 2
    cy_img = size / 2

    x_shift = x - cx_img
    y_shift = y - cy_img

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    x_rot = cos_a * x_shift - sin_a * y_shift + cx_img
    y_rot = sin_a * x_shift + cos_a * y_shift + cy_img

    # --- globaler Offset (reduziert, damit Struktur im Bild bleibt) ---
    offset_x = np.random.uniform(-pitch * 0.5, pitch * 0.5)
    offset_y = np.random.uniform(-pitch * 0.5, pitch * 0.5)

    # --- Hex-Gitter ---
    dx = pitch
    dy = pitch * np.sqrt(3) / 2

    # etwas größeres Gitter gegen Randartefakte
    n_cols = int(size / dx) + 6
    n_rows = int(size / dy) + 6

    centers = []

    for row in range(n_rows):
        for col in range(n_cols):
            cx = col * dx + (row % 2) * dx / 2 + offset_x
            cy = row * dy + offset_y
            centers.append((cx, cy))

    centers = np.array(centers)  # (N, 2)

    # --- Distanz zu allen Zentren (auf rotierten Koordinaten) ---
    dx_all = x_rot[None, :, :] - centers[:, 0][:, None, None]
    dy_all = y_rot[None, :, :] - centers[:, 1][:, None, None]

    dist_stack = np.sqrt(dx_all**2 + dy_all**2)

    # --- nächstes Zentrum pro Pixel ---
    nearest_idx = np.argmin(dist_stack, axis=0)

    # --- Distanz zum nächsten Zentrum ---
    r = np.take_along_axis(dist_stack, nearest_idx[None, :, :], axis=0)[0]

    # --- Ringsystem ---
    r_max = pitch * 0.5

    num_rings = max(5, int(r_max / ring_period))
    t = np.linspace(0.0, 1.0, num_rings)

    radii = r_max * (1.0 - t ** ring_log_scale)
    radii = radii[radii > ring_width]

    rings = np.zeros_like(r, dtype=bool)
    half_w = ring_width / 2.0

    for rc in radii:
        rings |= np.abs(r - rc) < half_w

    return 1.0 - rings.astype(float)