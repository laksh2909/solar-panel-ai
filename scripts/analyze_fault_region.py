"""
CLI Tool for Approximate Visual Fault-Region Analysis.

Extracts high-activation saliency regions, bounding boxes, centroids,
and spatial coverage metrics from Grad-CAM heatmaps.

Usage:
    # Batch run on the 12 representative Phase 8 test images
    python scripts/analyze_fault_region.py

    # Run on a single image with custom threshold
    python scripts/analyze_fault_region.py --image "data/test/Physical-damage/Physical (19).jpg" --threshold 0.60
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
from src.explainability.fault_region import FaultRegionExtractor
from src.models.efficientnet import build_efficientnet_b0
from src.preprocessing.pipeline import load_image_rgb
from src.preprocessing.augmentation import get_test_pipeline
from src.utils.config import get_project_root, load_config
from src.utils.logger import setup_logger

logger = setup_logger("analyze_fault_region")


def analyze_single_image(
    image_path: Path,
    model: nn.Module,
    cam: GradCAM,
    extractor: FaultRegionExtractor,
    pipeline,
    classes: List[str],
    device: torch.device,
    threshold: float,
    output_dir: Path,
    true_class: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Performs full fault-region analysis on a single image and saves visualizations.
    """
    # 1. Canonical Preprocessing
    raw_rgb = load_image_rgb(image_path)
    orig_224 = cv2.resize(raw_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    transformed = pipeline(image=raw_rgb)
    tensor = transformed["image"].unsqueeze(0).to(device)

    # 2. EfficientNet Prediction & Grad-CAM Heatmap
    heatmap, pred_idx, conf = cam.generate_heatmap(tensor, target_size=(224, 224))
    pred_class = classes[pred_idx]

    # 3. Fault-Region Extraction
    region_data = extractor.extract_regions(heatmap, threshold=threshold)

    # 4. Visualization Generation
    # Mask Image (uint8 0 or 255)
    mask = region_data["mask"]

    # Heatmap RGB
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Region Overlay with Bounding Box and Centroid
    overlay = extractor.draw_region_overlay(
        image_rgb=orig_224,
        mask=mask,
        bbox=region_data,
        centroid=region_data,
        mask_color=(255, 87, 34),
        bbox_color=(0, 230, 118),
        centroid_color=(213, 0, 0),
        alpha=0.35,
    )

    # 5. Save Outputs
    target_folder = true_class if true_class is not None else pred_class
    class_dir = output_dir / target_folder
    class_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
    orig_save_path = class_dir / f"original_{stem}.png"
    heatmap_save_path = class_dir / f"heatmap_{stem}.png"
    mask_save_path = class_dir / f"mask_{stem}.png"
    overlay_save_path = class_dir / f"region_overlay_{stem}.png"

    Image.fromarray(orig_224).save(orig_save_path)
    Image.fromarray(heatmap_rgb).save(heatmap_save_path)
    Image.fromarray(mask).save(mask_save_path)
    Image.fromarray(overlay).save(overlay_save_path)

    # 6. Format and Print Diagnostics
    bx = region_data["bbox_x"]
    by = region_data["bbox_y"]
    bw = region_data["bbox_width"]
    bh = region_data["bbox_height"]
    cx = region_data["centroid_x"]
    cy = region_data["centroid_y"]
    area_px = region_data["region_area_pixels"]
    area_pct = region_data["region_area_percent"]
    mean_act = region_data["mean_activation"]
    max_act = region_data["max_activation"]
    num_regs = region_data["num_regions"]

    print("-" * 75)
    print(f"Filename               : {image_path.name}")
    print(f"True Class             : {true_class or 'Unknown'}")
    print(f"Predicted Class        : {pred_class} (Confidence: {conf*100:.1f}%)")
    print(f"Grad-CAM Threshold     : {threshold:.2f}")
    print(f"Number of Regions      : {num_regs}")
    print(f"Dominant Bounding Box  : [x={bx}, y={by}, w={bw}, h={bh}]")
    print(f"Dominant Centroid      : (cx={cx}, cy={cy})")
    print(f"Dominant Region Area   : {area_px:,} pixels ({area_pct:.2f}% of image)")
    print(f"Mean Activation (ROI)  : {mean_act:.4f}")
    print(f"Max Activation (ROI)   : {max_act:.4f}")

    return {
        "filename": image_path.name,
        "true_class": true_class if true_class else "Unknown",
        "predicted_class": pred_class,
        "confidence": round(conf, 4),
        "threshold": threshold,
        "num_regions": num_regs,
        "bbox_x": bx,
        "bbox_y": by,
        "bbox_width": bw,
        "bbox_height": bh,
        "centroid_x": cx,
        "centroid_y": cy,
        "region_area_pixels": area_px,
        "region_area_percent": area_pct,
        "mean_activation": mean_act,
        "max_activation": max_act,
        "paths": {
            "original": str(orig_save_path),
            "heatmap": str(heatmap_save_path),
            "mask": str(mask_save_path),
            "region_overlay": str(overlay_save_path),
        },
        "orig_img": orig_224,
        "heatmap_img": heatmap_rgb,
        "mask_img": mask,
        "overlay_img": overlay,
    }


def create_sample_grid_plot(results: List[Dict[str, Any]], output_path: Path):
    """
    Creates a 6-row by 4-column grid:
    Original | Grad-CAM Heatmap | Binary Region Mask | Region Overlay
    """
    classes_order = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-damage", "Snow-Covered"]
    selected_by_class = {}
    for r in results:
        t_cls = r["true_class"]
        if t_cls in classes_order and t_cls not in selected_by_class:
            selected_by_class[t_cls] = r

    num_rows = len(selected_by_class)
    fig, axes = plt.subplots(num_rows, 4, figsize=(14, 3.2 * num_rows), dpi=300)

    col_titles = ["Original (224×224)", "Grad-CAM Heatmap", "Binary Mask (τ=0.60)", "Visual Region Overlay"]

    for row_idx, cls_name in enumerate(classes_order):
        if cls_name not in selected_by_class:
            continue
        r = selected_by_class[cls_name]

        # 1. Original
        ax0 = axes[row_idx, 0]
        ax0.imshow(r["orig_img"])
        ax0.set_ylabel(f"{cls_name}\n({r['filename']})", fontsize=10, fontweight="bold", labelpad=8)
        ax0.set_xticks([])
        ax0.set_yticks([])
        if row_idx == 0:
            ax0.set_title(col_titles[0], fontsize=11, fontweight="bold", pad=8)

        # 2. Heatmap
        ax1 = axes[row_idx, 1]
        ax1.imshow(r["heatmap_img"])
        ax1.set_xticks([])
        ax1.set_yticks([])
        if row_idx == 0:
            ax1.set_title(col_titles[1], fontsize=11, fontweight="bold", pad=8)

        # 3. Mask
        ax2 = axes[row_idx, 2]
        ax2.imshow(r["mask_img"], cmap="gray")
        ax2.set_xticks([])
        ax2.set_yticks([])
        if row_idx == 0:
            ax2.set_title(col_titles[2], fontsize=11, fontweight="bold", pad=8)

        # 4. Region Overlay
        ax3 = axes[row_idx, 3]
        ax3.imshow(r["overlay_img"])
        pct = r["region_area_percent"]
        ax3.set_title(f"Area: {pct:.1f}% | Pred: {r['predicted_class']}", fontsize=9, fontweight="bold", pad=4)
        ax3.set_xticks([])
        ax3.set_yticks([])
        if row_idx == 0:
            ax3.set_title(col_titles[3] + f"\nArea: {pct:.1f}%", fontsize=11, fontweight="bold", pad=8)

    plt.suptitle("Approximate Visual Fault-Region Analysis Across Six Solar Panel Classes", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved fault-region sample grid plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze approximate visual fault regions.")
    parser.add_argument("--image", type=str, default=None, help="Path to single image")
    parser.add_argument("--threshold", type=float, default=0.60, help="Saliency threshold (default: 0.60)")
    args = parser.parse_args()

    root = get_project_root()
    config = load_config()
    classes = sorted(config.get("dataset", {}).get("classes", []))
    device = torch.device("cpu")

    ckpt_path = root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Verify Checkpoint SHA256
    import hashlib
    with open(ckpt_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()
    logger.info(f"Loaded checkpoint SHA256: {ckpt_sha256}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_efficientnet_b0(num_classes=len(classes), pretrained=False, dropout=0.2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    pipeline = get_test_pipeline()
    extractor = FaultRegionExtractor(default_threshold=args.threshold, min_area_pixels=25)
    output_dir = root / "results" / "fault_regions"
    output_dir.mkdir(parents=True, exist_ok=True)

    with GradCAM(model) as cam:
        results = []

        if args.image:
            img_p = Path(args.image)
            if not img_p.exists():
                raise FileNotFoundError(f"Image not found: {img_p}")
            res = analyze_single_image(
                img_p, model, cam, extractor, pipeline, classes, device, args.threshold, output_dir
            )
            results.append(res)
        else:
            # Run on the SAME 12 representative images used in Phase 8
            # Discover from Phase 8's gradcam_examples.json if available
            p8_json_path = root / "results" / "metrics" / "gradcam_examples.json"
            target_images = []
            if p8_json_path.exists():
                with open(p8_json_path, "r", encoding="utf-8") as f:
                    p8_data = json.load(f)
                test_dir = root / "data" / "test"
                for item in p8_data:
                    t_cls = item["true_class"]
                    fname = item["filename"]
                    img_path = test_dir / t_cls / fname
                    if img_path.exists():
                        target_images.append((img_path, t_cls))

            if not target_images:
                # Fallback: select 2 per class
                test_dir = root / "data" / "test"
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
                for cls_name in classes:
                    folder = test_dir / cls_name
                    if folder.is_dir():
                        imgs = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])[:2]
                        for img_p in imgs:
                            target_images.append((img_p, cls_name))

            print("\n" + "=" * 75)
            print(" APPROXIMATE VISUAL FAULT-REGION ANALYSIS (12 SAMPLES)")
            print("=" * 75)

            for img_path, true_cls in target_images:
                res = analyze_single_image(
                    img_path, model, cam, extractor, pipeline, classes, device, args.threshold, output_dir, true_class=true_cls
                )
                results.append(res)

            # Generate sample grid plot
            grid_plot_path = root / "results" / "plots" / "fault_region_sample_grid.png"
            create_sample_grid_plot(results, grid_plot_path)

    # Save metrics JSON (strip numpy arrays)
    clean_results = []
    for r in results:
        clean_r = {k: v for k, v in r.items() if not k.endswith("_img")}
        clean_results.append(clean_r)

    json_path = root / "results" / "metrics" / "fault_region_examples.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)
    logger.info(f"Saved fault region examples JSON ({len(clean_results)} examples) to: {json_path}")

    # Re-verify Checkpoint SHA256 after execution
    with open(ckpt_path, "rb") as f:
        ckpt_sha256_after = hashlib.sha256(f.read()).hexdigest()
    assert ckpt_sha256 == ckpt_sha256_after, "Checkpoint was modified during analysis!"
    logger.info(f"Checkpoint integrity verified after run: {ckpt_sha256_after}")

    print("\n" + "=" * 75)
    print(" PHASE 9: FAULT-REGION ANALYSIS COMPLETE")
    print("=" * 75)
    print(f"Threshold Applied      : {args.threshold:.2f}")
    print(f"Total Images Processed : {len(clean_results)}")
    print(f"Metrics Output JSON    : {json_path}")
    print(f"Sample Grid Plot       : results/plots/fault_region_sample_grid.png")
    print(f"Checkpoint SHA256      : {ckpt_sha256_after} (UNCHANGED)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
