"""
CLI Script for AI-Assisted Solar Panel Maintenance Recommendation Pipeline.

Pipeline:
    Image
    -> Canonical Preprocessing
    -> EfficientNet-B0 Prediction
    -> Grad-CAM Saliency
    -> Fault Region Extraction
    -> Visual Severity Estimation
    -> Maintenance Recommendation

Usage:
    # Run on the 12 representative test samples from Phases 8-10
    python scripts/generate_maintenance_recommendation.py

    # Run on a single image
    python scripts/generate_maintenance_recommendation.py --image "data/test/Electrical-damage/Electrical (1).jpg"
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.explainability.gradcam import GradCAM
from src.explainability.fault_region import FaultRegionExtractor
from src.severity.severity_estimator import SeverityEstimator
from src.maintenance.maintenance_recommender import MaintenanceRecommender
from src.models.efficientnet import build_efficientnet_b0
from src.preprocessing.pipeline import load_image_rgb
from src.preprocessing.augmentation import get_test_pipeline
from src.utils.config import get_project_root, load_config
from src.utils.logger import setup_logger

logger = setup_logger("generate_maintenance_recommendation_cli")

MANDATORY_DISCLAIMER = (
    "Recommendation is AI-assisted workflow guidance based on image analysis. "
    "It does not replace qualified electrical, thermal, structural, or physical inspection."
)


def process_image_maintenance(
    image_path: Path,
    model: nn.Module,
    cam: GradCAM,
    extractor: FaultRegionExtractor,
    estimator: SeverityEstimator,
    recommender: MaintenanceRecommender,
    pipeline,
    classes: List[str],
    device: torch.device,
    true_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs the full 6-stage diagnostic and recommendation pipeline on an image."""
    # 1. Preprocessing
    raw_rgb = load_image_rgb(image_path)
    orig_224 = cv2.resize(raw_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    transformed = pipeline(image=raw_rgb)
    tensor = transformed["image"].unsqueeze(0).to(device)

    # 2. Prediction & Grad-CAM
    heatmap, pred_idx, conf = cam.generate_heatmap(tensor, target_size=(224, 224))
    pred_class = classes[pred_idx]

    # 3. Fault-Region Extraction
    region_info = extractor.extract_regions(heatmap)

    # 4. Severity Estimation
    severity_info = estimator.estimate_severity(
        predicted_class=pred_class,
        confidence=conf,
        region_area_percent=region_info["region_area_percent"],
        mean_activation=region_info["mean_activation"],
        max_activation=region_info["max_activation"],
        num_regions=region_info["num_regions"],
    )

    # 5. Maintenance Recommendation
    rec_info = recommender.get_recommendation(
        predicted_class=pred_class,
        confidence=conf,
        severity=severity_info["severity"],
        region_area_percent=region_info["region_area_percent"],
        manual_inspection_recommended=severity_info["manual_inspection_recommended"],
    )

    # Visual Overlay for plotting
    overlay = extractor.draw_region_overlay(
        image_rgb=orig_224,
        mask=region_info["mask"],
        bbox=region_info,
        centroid=region_info,
    )

    # Print Formatted Report as specified
    print("\n" + "=" * 40)
    print("SOLAR PANEL INSPECTION RECOMMENDATION")
    print("=" * 40)
    print(f"Image: {image_path.name}")
    print(f"Predicted fault: {pred_class}")
    print(f"Confidence: {conf*100:.1f}%")
    print(f"Visual severity: {rec_info['severity']}")
    print(f"Visual region: [x={region_info['bbox_x']}, y={region_info['bbox_y']}, w={region_info['bbox_width']}, h={region_info['bbox_height']}] ({region_info['region_area_percent']:.2f}% of module)")
    print(f"Recommended action: {rec_info['recommended_action']}")
    print(f"Urgency: {rec_info['urgency']}")
    print(f"Manual inspection: {'YES' if rec_info['manual_inspection_recommended'] else 'NO'}")
    print(f"Reason: {rec_info['reason']}")
    print(f"Confidence warning: {rec_info['confidence_warning'] if rec_info['confidence_warning'] else 'None'}")
    print("-" * 40)
    print(f"Disclaimer: {MANDATORY_DISCLAIMER}")
    print("=" * 40)

    return {
        "filename": image_path.name,
        "true_class": true_class if true_class else "Unknown",
        "predicted_class": pred_class,
        "confidence": round(float(conf), 4),
        "region_area_percent": round(float(region_info["region_area_percent"]), 2),
        "severity": rec_info["severity"],
        "recommended_action": rec_info["recommended_action"],
        "urgency": rec_info["urgency"],
        "reason": rec_info["reason"],
        "manual_inspection_recommended": rec_info["manual_inspection_recommended"],
        "confidence_warning": rec_info["confidence_warning"],
        "orig_img": orig_224,
        "overlay_img": overlay,
    }


def create_maintenance_grid_plot(results: List[Dict[str, Any]], output_path: Path):
    """
    Creates a clean, high-resolution 6-row by 3-column summary grid:
    Col 1: Original Image
    Col 2: Predicted Fault & Visual Region Overlay
    Col 3: Severity & Recommended Action Card
    """
    classes_order = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-damage", "Snow-Covered"]
    selected_by_class = {}
    for r in results:
        t_cls = r["true_class"]
        if t_cls in classes_order and t_cls not in selected_by_class:
            selected_by_class[t_cls] = r

    num_rows = len(selected_by_class)
    fig, axes = plt.subplots(num_rows, 3, figsize=(14, 3.2 * num_rows), dpi=300)

    urgency_colors = {
        "ROUTINE": "#2e7d32",          # Green
        "SCHEDULED": "#0288d1",        # Blue
        "PRIORITY": "#f57c00",         # Orange
        "IMMEDIATE_REVIEW": "#c62828", # Red
    }

    severity_badges = {
        "LOW": "[LOW]",
        "MEDIUM": "[MEDIUM]",
        "HIGH": "[HIGH]",
    }

    for row_idx, cls_name in enumerate(classes_order):
        if cls_name not in selected_by_class:
            continue
        r = selected_by_class[cls_name]

        # 1. Original Image
        ax0 = axes[row_idx, 0]
        ax0.imshow(r["orig_img"])
        ax0.set_ylabel(f"Ground Truth:\n{cls_name}", fontsize=10, fontweight="bold", labelpad=10)
        ax0.set_xticks([])
        ax0.set_yticks([])
        if row_idx == 0:
            ax0.set_title("Original Module", fontsize=11, fontweight="bold", pad=8)

        # 2. Predicted Fault & Region Overlay
        ax1 = axes[row_idx, 1]
        ax1.imshow(r["overlay_img"])
        ax1.set_xticks([])
        ax1.set_yticks([])
        pred_title = f"Pred: {r['predicted_class']} ({r['confidence']*100:.1f}%)"
        ax1.set_xlabel(pred_title, fontsize=9.5, fontweight="bold", labelpad=6)
        if row_idx == 0:
            ax1.set_title("Fault Region Overlay", fontsize=11, fontweight="bold", pad=8)

        # 3. Severity & Recommended Action Card
        ax2 = axes[row_idx, 2]
        ax2.axis("off")
        urg = r["urgency"]
        sev = r["severity"]
        action = r["recommended_action"]
        manual = r["manual_inspection_recommended"]
        warning = r["confidence_warning"]

        card_color = urgency_colors.get(urg, "#333333")

        # Format multi-line card
        card_lines = [
            f"CLASS     : {r['predicted_class']}",
            f"SEVERITY  : {sev} (Fault Area: {r['region_area_percent']}%)",
            f"URGENCY   : {urg}",
            f"ACTION    : {action}",
            f"MANUAL REV: {'REQUIRED' if manual else 'ROUTINE MONITOR'}",
        ]
        if warning:
            card_lines.append("WARNING   : Low Confidence (<60%)")

        card_text = "\n".join(card_lines)

        ax2.text(
            0.04, 0.50,
            card_text,
            fontsize=9.0,
            family="monospace",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.7", facecolor="#f8f9fa", edgecolor=card_color, linewidth=2.0),
        )
        if row_idx == 0:
            ax2.set_title("Maintenance Recommendation Card", fontsize=11, fontweight="bold", pad=8)

    plt.suptitle("AI-Assisted Solar Photovoltaic Maintenance Recommendations", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved maintenance sample grid plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate AI-assisted maintenance recommendations.")
    parser.add_argument("--image", type=str, default=None, help="Path to single image")
    args = parser.parse_args()

    root = get_project_root()
    config = load_config()
    classes = sorted(config.get("dataset", {}).get("classes", []))
    device = torch.device("cpu")

    ckpt_path = root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Checkpoint SHA256 integrity check before execution
    with open(ckpt_path, "rb") as f:
        sha_before = hashlib.sha256(f.read()).hexdigest()
    logger.info(f"Checkpoint SHA256 before evaluation: {sha_before}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_efficientnet_b0(num_classes=len(classes), pretrained=False, dropout=0.2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    pipeline = get_test_pipeline()
    extractor = FaultRegionExtractor(default_threshold=0.60, min_area_pixels=25)
    estimator = SeverityEstimator()
    recommender = MaintenanceRecommender(low_confidence_threshold=0.60)

    with GradCAM(model) as cam:
        results = []

        if args.image:
            img_p = Path(args.image)
            if not img_p.exists():
                raise FileNotFoundError(f"Image not found: {img_p}")
            res = process_image_maintenance(
                img_p, model, cam, extractor, estimator, recommender, pipeline, classes, device
            )
            results.append(res)
        else:
            # Use the EXACT SAME 12 representative images from Phases 8-10
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
                test_dir = root / "data" / "test"
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
                for cls_name in classes:
                    folder = test_dir / cls_name
                    if folder.is_dir():
                        imgs = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])[:2]
                        for img_p in imgs:
                            target_images.append((img_p, cls_name))

            for img_path, true_cls in target_images:
                res = process_image_maintenance(
                    img_path, model, cam, extractor, estimator, recommender, pipeline, classes, device, true_class=true_cls
                )
                results.append(res)

            # Generate sample grid plot
            grid_plot_path = root / "results" / "plots" / "maintenance_sample_grid.png"
            create_maintenance_grid_plot(results, grid_plot_path)

    # Save metrics JSON (strip numpy image arrays)
    clean_results = []
    for r in results:
        clean_r = {k: v for k, v in r.items() if not k.endswith("_img")}
        clean_results.append(clean_r)

    json_path = root / "results" / "metrics" / "maintenance_recommendations.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)
    logger.info(f"Saved maintenance recommendations JSON ({len(clean_results)} examples) to: {json_path}")

    # Checkpoint SHA256 integrity check after execution
    with open(ckpt_path, "rb") as f:
        sha_after = hashlib.sha256(f.read()).hexdigest()
    assert sha_before == sha_after, "Checkpoint was modified during maintenance recommendation generation!"
    logger.info(f"Checkpoint SHA256 after evaluation: {sha_after} (UNCHANGED)")


if __name__ == "__main__":
    main()
