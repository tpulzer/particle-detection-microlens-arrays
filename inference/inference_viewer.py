import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import onnxruntime as ort
from pathlib import Path
import yaml

# =========================
# CONFIG
# =========================

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "inference.yaml"

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

inference_cfg = CONFIG["inference"]

paths_cfg = CONFIG["paths"]

MODEL_PATH = ROOT / paths_cfg["model_path"]
INPUT_DIR = ROOT / paths_cfg["input_dir"]
MASK_DIR = ROOT / paths_cfg["mask_dir"]
SAVE_DIR = ROOT / paths_cfg["preview_dir"]

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMG_SIZE = inference_cfg["image_size"]


# =========================
# ONNX
# =========================

def get_session():
    print("Using CPUExecutionProvider")
    return ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )


# =========================
# Utils
# =========================

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return img


def overlay(image, mask):
    color = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    color[:, :, 2] = (mask * 255).astype(np.uint8)

    blended = cv2.addWeighted(
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
        0.7,
        color,
        0.3,
        0
    )
    return blended


# =========================
# Metrics (Pixel)
# =========================

def compute_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    union = tp + fp + fn

    iou = tp / union if union > 0 else 0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return iou, dice, precision, recall


# =========================
# Metrics (Particle Level)
# =========================

def particle_metrics(pred, gt):

    num_gt, gt_labels = cv2.connectedComponents(gt.astype(np.uint8))
    num_pred, pred_labels = cv2.connectedComponents(pred.astype(np.uint8))

    matched = 0

    for gt_id in range(1, num_gt):
        gt_mask = (gt_labels == gt_id)
        overlap = pred[gt_mask].sum()

        if overlap > 0:
            matched += 1

    detection_rate = matched / (num_gt - 1) if num_gt > 1 else 0

    return {
        "gt_particles": num_gt - 1,
        "pred_particles": num_pred - 1,
        "matched": matched,
        "detection_rate": detection_rate
    }


# =========================
# GUI
# =========================

class App:

    def __init__(self, root):
        self.root = root
        self.root.title("Particle Inference Viewer")

        os.makedirs(SAVE_DIR, exist_ok=True)

        self.session = get_session()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.files = sorted([
            f for f in os.listdir(INPUT_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ])
        self.index = 0

        # Store current images
        self.current_img = None
        self.current_mask = None
        self.current_overlay = None
        self.current_name = None

        # Panels
        self.panel_left = tk.Label(root)
        self.panel_left.grid(row=0, column=0)

        self.panel_mid = tk.Label(root)
        self.panel_mid.grid(row=0, column=1)

        self.panel_right = tk.Label(root)
        self.panel_right.grid(row=0, column=2)

        # Info
        self.info = tk.Label(root, text="", font=("Arial", 12))
        self.info.grid(row=1, column=0, columnspan=3)

        # Buttons
        btn_prev = tk.Button(root, text="<< Prev", command=self.prev_image)
        btn_prev.grid(row=2, column=0)

        btn_save = tk.Button(root, text="Save Images", command=self.save_images)
        btn_save.grid(row=2, column=1)

        btn_next = tk.Button(root, text="Next >>", command=self.next_image)
        btn_next.grid(row=2, column=2)

        btn_load = tk.Button(root, text="Load Folder", command=self.load_folder)
        btn_load.grid(row=3, column=1)

        self.show_image()

    def load_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.files = sorted([
                f for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
            ])
            global INPUT_DIR
            INPUT_DIR = folder
            self.index = 0
            self.show_image()

    def next_image(self):
        self.index = (self.index + 1) % len(self.files)
        self.show_image()

    def prev_image(self):
        self.index = (self.index - 1) % len(self.files)
        self.show_image()

    def save_images(self):

        if self.current_img is None:
            return

        base = os.path.splitext(self.current_name)[0]

        cv2.imwrite(
            os.path.join(SAVE_DIR, f"{base}_input.png"),
            self.current_img
        )

        cv2.imwrite(
            os.path.join(SAVE_DIR, f"{base}_mask.png"),
            self.current_mask * 255
        )

        cv2.imwrite(
            os.path.join(SAVE_DIR, f"{base}_overlay.png"),
            self.current_overlay
        )

        print(f"Saved images for: {base}")

    def show_image(self):

        name = self.files[self.index]
        path = os.path.join(INPUT_DIR, name)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return

        # Inference
        x = preprocess(img)

        pred = self.session.run(
            [self.output_name],
            {self.input_name: x}
        )[0]

        logits = pred[0, 0]
        mask = (logits > 0).astype(np.uint8)

        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
        overlay_img = overlay(img, mask_resized)

        # Store current images
        self.current_img = img
        self.current_mask = mask_resized
        self.current_overlay = overlay_img
        self.current_name = name

        # Load GT
        gt_path = os.path.join(MASK_DIR, name)
        metrics_text = ""

        if os.path.exists(gt_path):

            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]))
            gt = (gt > 127).astype(np.uint8)

            iou, dice, precision, recall = compute_metrics(mask_resized, gt)
            pm = particle_metrics(mask_resized, gt)

            metrics_text = (
                f"IoU: {iou:.3f} | Dice: {dice:.3f} | "
                f"P: {precision:.3f} | R: {recall:.3f} | "
                f"Particles: {pm['matched']}/{pm['gt_particles']} "
                f"({pm['detection_rate']:.2%})"
            )

        # Convert for Tkinter
        img_tk = ImageTk.PhotoImage(Image.fromarray(img))
        mask_tk = ImageTk.PhotoImage(Image.fromarray(mask_resized * 255))

        overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
        overlay_tk = ImageTk.PhotoImage(Image.fromarray(overlay_rgb))

        self.panel_left.config(image=img_tk)
        self.panel_left.image = img_tk

        self.panel_mid.config(image=mask_tk)
        self.panel_mid.image = mask_tk

        self.panel_right.config(image=overlay_tk)
        self.panel_right.image = overlay_tk

        self.info.config(text=f"{name} | {metrics_text}")


# =========================
# START
# =========================

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()