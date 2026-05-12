import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class NPYViewer:
    def __init__(self, img_path, mask_path):
        self.images = np.load(img_path, mmap_mode="r")
        self.masks = np.load(mask_path, mmap_mode="r")

        assert len(self.images) == len(self.masks), "Images und Masks haben unterschiedliche Länge!"

        self.idx = 0
        self.n = len(self.images)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.update()
        plt.show()

    def update(self):
        self.ax1.clear()
        self.ax2.clear()

        img = self.images[self.idx]
        mask = self.masks[self.idx]

        # Bild
        if img.ndim == 2:
            self.ax1.imshow(img, cmap="gray")
        else:
            self.ax1.imshow(img)
        self.ax1.set_title("Image")

        # Maske
        self.ax2.imshow(mask, cmap="gray")
        self.ax2.set_title("Mask")

        self.fig.suptitle(f"{self.idx+1}/{self.n}")

        self.ax1.axis("off")
        self.ax2.axis("off")

        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == "right":
            self.idx = (self.idx + 1) % self.n
        elif event.key == "left":
            self.idx = (self.idx - 1) % self.n
        elif event.key == "escape":
            plt.close(self.fig)
            return

        self.update()


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "data" / "train_data"

    img_path = data_dir / "images.npy"
    mask_path = data_dir / "masks.npy"

    if not img_path.exists() or not mask_path.exists():
        print("FEHLER: Dateien nicht gefunden!")
        return

    NPYViewer(img_path, mask_path)


if __name__ == "__main__":
    main()