import numpy as np
import cv2
import random
import math

def add_particles(image, config):

    particle_cfg = config["particles"]

    mask = np.zeros_like(image)

    num = random.randint(*particle_cfg["count_range"])

    size = image.shape[0]

    particle_types = list(
        particle_cfg["type_weights"].keys()
    )

    particle_weights = list(
        particle_cfg["type_weights"].values()
    )

    for _ in range(num):

        particle_type = random.choices(
            particle_types,
            weights=particle_weights,
            k=1
        )[0]

        if particle_type == "circle":
            add_circle(image, mask, size, particle_cfg)

        elif particle_type == "fiber":
            add_fiber(image, mask, size, particle_cfg)

        elif particle_type == "spline":
            add_spline(image, mask, size, particle_cfg)

        elif particle_type == "freeform":
            add_freeform(image, mask, size, particle_cfg)

    return image, mask


def add_circle(image, mask, size, config):
    x = random.randint(0, size - 1)
    y = random.randint(0, size - 1)
    r = random.randint(*config["radius_range"])

    cv2.circle(image, (x, y), r, 0.0, -1)
    cv2.circle(mask,  (x, y), r, 1.0, -1)

def add_fiber(image, mask, size, config):
    x0 = random.randint(0, size - 1)
    y0 = random.randint(0, size - 1)

    length = random.randint(15, 60)
    angle  = random.uniform(0, 2 * math.pi)
    thickness = random.randint(1, 2)

    x1 = int(x0 + length * math.cos(angle))
    y1 = int(y0 + length * math.sin(angle))

    cv2.line(image, (x0, y0), (x1, y1), 0.0, thickness)
    cv2.line(mask,  (x0, y0), (x1, y1), 1.0, thickness)

def bezier_curve(p0, p1, p2, p3, num_points):
    t = np.linspace(0.0, 1.0, num_points)[:, None]

    curve = (
        (1 - t)**3 * p0 +
        3 * (1 - t)**2 * t * p1 +
        3 * (1 - t) * t**2 * p2 +
        t**3 * p3
    )

    return curve.astype(np.int32)

def add_spline(image, mask, size, config):
    # Startpunkt
    x = random.randint(0, size - 1)
    y = random.randint(0, size - 1)

    # --- neue Parameter ---
    num_segments = random.randint(1, 4)              # mehrere Teilstücke
    base_angle = random.uniform(0, 2 * math.pi)

    thickness = random.randint(1, 4)

    points = [(x, y)]

    angle = base_angle

    for _ in range(num_segments):
        # Länge pro Segment variabel
        length = random.randint(20, 80)

        # Richtungsänderung (macht es organisch)
        angle += random.uniform(-0.8, 0.8)

        x += length * math.cos(angle)
        y += length * math.sin(angle)

        points.append((x, y))

    # --- glatte Kurve durch Punkte (Bezier-Ketten) ---
    curve_pts = []

    for i in range(len(points) - 1):
        p0 = np.array(points[i])
        p3 = np.array(points[i + 1])

        # Kontrollpunkte zufällig
        direction = p3 - p0
        length = np.linalg.norm(direction) + 1e-6
        dir_norm = direction / length

        normal = np.array([-dir_norm[1], dir_norm[0]])

        bend_scale = length * random.uniform(0.1, 0.5)

        p1 = p0 + dir_norm * length * 0.3 + normal * bend_scale * random.uniform(-1, 1)
        p2 = p0 + dir_norm * length * 0.7 + normal * bend_scale * random.uniform(-1, 1)

        segment = bezier_curve(p0, p1, p2, p3, num_points=random.randint(20, 80))

        curve_pts.append(segment)

    if not curve_pts:
        return

    pts = np.vstack(curve_pts)

    # --- optional: jitter für "rauere" Partikel ---
    if random.random() < 0.3:
        jitter = np.random.normal(0, 0.8, pts.shape)
        pts = pts + jitter

    pts = pts.astype(np.int32)

    # --- Zeichnen ---
    for i in range(len(pts) - 1):
        p0 = tuple(pts[i])
        p1 = tuple(pts[i + 1])

        cv2.line(image, p0, p1, 0.0, thickness)
        cv2.line(mask,  p0, p1, 1.0, thickness)



def add_freeform(image, mask, size, config):
    cx = random.randint(0, size - 1)
    cy = random.randint(0, size - 1)

    num_points = random.randint(5, 10)
    radius = random.randint(*config["radius_range"])

    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        r = radius * random.uniform(0.5, 1.2)
        x = int(cx + r * math.cos(angle))
        y = int(cy + r * math.sin(angle))
        points.append([x, y])

    pts = np.array(points, dtype=np.int32)

    cv2.fillPoly(image, [pts], 0.0)
    cv2.fillPoly(mask,  [pts], 1.0)
