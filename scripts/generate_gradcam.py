r"""
CLI Script to Generate Grad-CAM Visualizations for Solar Panel Fault Detection.

Usage:
    # Generate for representative images across all 6 classes (at least 2 per class)
    python scripts/generate_gradcam.py

    # Generate for a single specific image
    python scripts/generate_gradcam.py --image "data/test/Physical-damage/Physical (1).jpg"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.explainability.gradcam import GradCAM
from src.models.efficientnet import build_efficientnet_b0
from src.preprocessing.pipeline import load_image_rgb
from src.preprocessing.augmentation import get_test_pipeline
from src.utils.config import get_project_root, load_config
from src.utils.logger import setup_logger

logger = setup_logger("generate_gradcam_cli")


def generate_for_image(
    image_path: Path,
    model: nn.Module,
    cam: GradCAM,
    pipeline,
    classes: List[str],
    device: torch.device,
    output_dir: Path,
    true_class: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates original, heatmap, and overlay images for a single image.
    """
    # Canonical Preprocessing
    raw_rgb = load_image_rgb(image_path)
    # Standardize to 224x224 for uniform overlay rendering
    orig_224 = cv2.resize(raw_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    transformed = pipeline(image=raw_rgb)
    tensor = transformed["image"].unsqueeze(0).to(device)

    # Generate Grad-CAM Heatmap
    heatmap, pred_idx, conf = cam.generate_heatmap(tensor, target_size=(224, 224))
    pred_class = classes[pred_idx]

    # Generate Overlay
    overlay = cam.overlay_heatmap(orig_224, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET)

    # Convert heatmap to colored RGB for saving
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Determine save directory based on true_class or pred_class
    target_folder_name = true_class if true_class is not None else pred_class
    class_dir = output_dir / target_folder_name
    class_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
    orig_save_path = class_dir / f"original_{stem}.png"
    heatmap_save_path = class_dir / f"heatmap_{stem}.png"
    overlay_save_path = class_dir / f"overlay_{stem}.png"

    # Save images using PIL (RGB)
    Image.fromarray(orig_224).save(orig_save_path)
    Image.fromarray(heatmap_rgb).save(heatmap_save_path)
    Image.fromarray(overlay).save(overlay_save_path)

    target_layer_name = str(cam.target_layer.__class__.__name__)
    is_correct = (pred_class == true_class) if true_class is not None else True

    logger.info(
        f"File: {image_path.name:<20} | Pred: {pred_class:<18} | Conf: {conf*100:>5.1f}% | "
        f"True: {str(true_class):<18} | Target Layer: {target_layer_name}"
    )

    return {
        "filename": image_path.name,
        "true_class": true_class if true_class else "Unknown",
        "predicted_class": pred_class,
        "confidence": round(conf, 4),
        "target_layer": target_layer_name,
        "correct_prediction": is_correct,
        "paths": {
            "original": str(orig_save_path),
            "heatmap": str(heatmap_save_path),
            "overlay": str(overlay_save_path),
        },
        "orig_img": orig_224,
        "heatmap_img": heatmap_rgb,
        "overlay_img": overlay,
    }


def create_sample_grid_plot(results: List[Dict[str, Any]], output_path: Path):
    """
    Creates a 6-row grid showing representative Grad-CAM examples across all 6 classes.
    Columns: Original Image, Grad-CAM Heatmap, Attention Overlay.
    """
    # Group results by true_class and pick 1 best representative per class
    classes_order = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-damage", "Snow-Covered"]
    selected_by_class = {}
    for r in results:
        t_cls = r["true_class"]
        if t_cls in classes_order and t_cls not in selected_by_class:
            selected_by_class[t_cls] = r

    num_rows = len(selected_by_class)
    fig, axes = plt.subplots(num_rows, 3, figsize=(11, 3.2 * num_rows), dpi=300)

    col_titles = ["Original (224×224)", "Grad-CAM Heatmap", "Overlay Explanation"]

    for row_idx, cls_name in enumerate(classes_order):
        if cls_name not in selected_by_class:
            continue
        r = selected_by_class[cls_name]

        # 1. Original
        ax_orig = axes[row_idx, 0]
        ax_orig.imshow(r["orig_img"])
        ax_orig.set_ylabel(f"{cls_name}\n({r['filename']})", fontsize=10, fontweight="bold", labelpad=10)
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])
        if row_idx == 0:
            ax_orig.set_title(col_titles[0], fontsize=12, fontweight="bold", pad=8)

        # 2. Heatmap
        ax_hm = axes[row_idx, 1]
        ax_hm.imshow(r["heatmap_img"])
        ax_hm.set_xticks([])
        ax_hm.set_yticks([])
        if row_idx == 0:
            ax_hm.set_title(col_titles[1], fontsize=12, fontweight="bold", pad=8)

        # 3. Overlay
        ax_ov = axes[row_idx, 2]
        ax_ov.imshow(r["overlay_img"])
        status = "Correct" if r["correct_prediction"] else f"Pred: {r['predicted_class']}"
        color = "#2e7d32" if r["correct_prediction"] else "#c62828"
        ax_ov.set_title(f"{status} ({r['confidence']*100:.1f}%)", fontsize=10, fontweight="bold", color=color, pad=4)
        ax_ov.set_xticks([])
        ax_ov.set_yticks([])
        if row_idx == 0:
            ax_ov.set_title(col_titles[2] + f"\n{status} ({r['confidence']*100:.1f}%)", fontsize=11, fontweight="bold", pad=8)

    plt.suptitle("EfficientNet-B0 Grad-CAM Explainability — Six Solar Panel Classes", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Grad-CAM sample grid plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explainability visualizations.")
    parser.add_argument("--image", type=str, default=None, help="Path to single image to process")
    parser.add_argument("--samples-per-class", type=int, default=2, help="Number of test samples per class (default: 2)")
    args = parser.parse_args()

    root = get_project_root()
    config = load_config()
    classes = sorted(config.get("dataset", {}).get("classes", []))
    device = torch.device("cpu")

    ckpt_path = root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Load canonical checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_efficientnet_b0(num_classes=len(classes), pretrained=False, dropout=0.2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    pipeline = get_test_pipeline()
    output_dir = root / "results" / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    with GradCAM(model) as cam:
        target_layer_str = str(cam.target_layer)
        logger.info(f"Grad-CAM initialized on target layer: {cam.target_layer.__class__.__name__}")

        results = []

        if args.image:
            img_p = Path(args.image)
            if not img_p.exists():
                raise FileNotFoundError(f"Image not found: {img_p}")
            res = generate_for_image(img_p, model, cam, pipeline, classes, device, output_dir)
            # Remove image arrays before saving JSON
            res_json = {k: v for k, v in res.items() if not k.endswith("_img")}
            results.append(res_json)
        else:
            # Deterministic selection: pick `samples_per_class` from each of the 6 classes in data/test
            test_dir = root / "data" / "test"
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

            for cls_name in classes:
                cls_folder = test_dir / cls_name
                if not cls_folder.is_dir():
                    continue
                # Sort filenames deterministically
                all_images = sorted([f for f in cls_folder.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
                selected = all_images[: args.samples_per_class]

                for img_p in selected:
                    res = generate_for_image(
                        img_p, model, cam, pipeline, classes, device, output_dir, true_class=cls_name
                    )
                    results.append(res)

            # Generate sample grid plot
            grid_plot_path = root / "results" / "plots" / "gradcam_sample_grid.png"
            create_sample_grid_plot(results, grid_plot_path)

    # Save examples JSON (strip numpy image arrays)
    clean_results = []
    for r in results:
        clean_r = {k: v for k, v in r.items() if not k.endswith("_img")}
        clean_results.append(clean_r)

    json_path = root / "results" / "metrics" / "gradcam_examples.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)
    logger.info(f"Saved Grad-CAM examples JSON ({len(clean_results)} examples) to: {json_path}")

    print("\n" + "=" * 80)
    print(" PHASE 8: GRAD-CAM EXPLAINABILITY GENERATION COMPLETE")
    print("=" * 80)
    print(f"Target Layer: {cam.target_layer.__class__.__name__} (1280 feature channels)")
    print(f"Total Examples Generated: {len(clean_results)}")
    print(f"Output Directory: {output_dir}")
    print(f"Metrics JSON: {json_path}")
    print(f"Sample Grid Plot: results/plots/gradcam_sample_grid.png")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
