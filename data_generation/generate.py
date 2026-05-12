from pathlib import Path
import yaml

from structures import generate_structure
from particles import add_particles
from augmentations import apply_augmentations, apply_illumination
from utils import save_sample


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "data_generation.yaml"


with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)


def main():

    print("generate data")

    structure_type = CONFIG["background"]["structure_type"]
    num_samples = CONFIG["general"]["num_samples"]

    print(structure_type)
    print("num", num_samples)

    for i in range(num_samples):

        print(f"\rCounter: {i + 1}", end="", flush=True)

        image = generate_structure(CONFIG)

        image, mask = add_particles(image, CONFIG)

        image = apply_illumination(image, CONFIG)

        image = apply_augmentations(image, CONFIG)

        save_sample(image, mask, i)

    print(" done")


if __name__ == "__main__":
    main()
