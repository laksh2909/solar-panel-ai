"""
Script to visualize domain-specific solar panel augmentations.
Saves a comparison grid of original vs. augmented samples to:
results/plots/augmentation_samples.png
"""

import sys
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing.pipeline import load_image_rgb
from src.preprocessing.augmentation import get_visualization_augmentation
from src.utils.config import get_project_root, load_config
from src.utils.logger import setup_logger

logger = setup_logger("visualize_augmentation")


def generate_augmentation_samples(output_filename: str = "augmentation_samples.png"):
    root = get_project_root()
    train_dir = root / "data" / "train"
    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / output_filename

    classes = [
        "Physical-damage",
        "Bird-drop",
        "Dusty",
        "Clean",
        "Electrical-damage",
        "Snow-Covered",
    ]

    vis_aug = get_visualization_augmentation(image_size=(224, 224))

    fig, axes = plt.subplots(len(classes), 4, figsize=(14, 18), dpi=300)
    fig.suptitle(
        "Solar Panel Domain-Specific Augmentation Samples (Albumentations)\n"
        "Original vs. Realistic Environmental & Sensor Variations",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for r_idx, c_name in enumerate(classes):
        c_folder = train_dir / c_name
        sample_img_path = None
        for f in sorted(c_folder.iterdir()):
            if f.is_file() and f.suffix.lower() in valid_exts:
                sample_img_path = f
                break

        if sample_img_path is None:
            continue

        orig_rgb = load_image_rgb(str(sample_img_path))
        # Resize original for display
        import cv2
        orig_resized = cv2.resize(orig_rgb, (224, 224), interpolation=cv2.INTER_AREA)

        # Original
        axes[r_idx, 0].imshow(orig_resized)
        axes[r_idx, 0].set_title(f"{c_name}\n[Original]", fontsize=10, fontweight="bold")
        axes[r_idx, 0].axis("off")

        # 3 Augmented variants
        for c_idx in range(1, 4):
            aug_out = vis_aug(image=orig_rgb)["image"]
            axes[r_idx, c_idx].imshow(aug_out)
            axes[r_idx, c_idx].set_title(f"Augmented Variant #{c_idx}", fontsize=10)
            axes[r_idx, c_idx].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Augmentation visualization saved to: {out_path}")

    # Copy to brain artifact directory if accessible
    brain_plot_dir = Path(r"C:\Users\laksh\.gemini\antigravity-ide\brain\e6bc9245-4294-41a5-86b7-ca0f7d9f5939\plots")
    brain_plot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, brain_plot_dir / output_filename)
    logger.info(f"Copied to artifact directory: {brain_plot_dir / output_filename}")

    return out_path


if __name__ == "__main__":
    generate_augmentation_samples()
