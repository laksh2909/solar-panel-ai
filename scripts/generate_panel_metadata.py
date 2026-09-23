"""
Demonstration CLI Script for Phase 12: Panel ID + Location Operational Metadata Layer.

Connects visual inspection results from the existing Phase 11 pipeline with
operational metadata (Panel ID, Location, and Timestamp).

Usage:
    python scripts/generate_panel_metadata.py
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.metadata.panel_metadata import PanelMetadata, create_inspection_record
from src.utils.config import get_project_root
from src.utils.logger import setup_logger

logger = setup_logger("generate_panel_metadata_cli")

# Synthetic demonstration metadata for the 12 representative test samples
DEMO_METADATA = [
    {
        "filename": "13.JPG",
        "panel_id": "SP-HYD-001",
        "location": "Block A - Rooftop 1",
        "timestamp": "2026-09-22T09:00:00Z",
    },
    {
        "filename": "15.JPG",
        "panel_id": "SP-HYD-002",
        "location": "Block A - Rooftop 1",
        "timestamp": "2026-09-22T09:15:00Z",
    },
    {
        "filename": "Clean (102).jpg",
        "panel_id": "SP-HYD-003",
        "location": "Block A - Rooftop 2",
        "timestamp": "2026-09-22T09:30:00Z",
    },
    {
        "filename": "Clean (104).jpg",
        "panel_id": "SP-HYD-004",
        "location": "Block A - Rooftop 2",
        "timestamp": "2026-09-22T09:45:00Z",
    },
    {
        "filename": "Dust (1).jpg",
        "panel_id": "SP-HYD-005",
        "location": "Block B - Ground Array 1",
        "timestamp": "2026-09-22T10:00:00Z",
    },
    {
        "filename": "Dust (100).jpg",
        "panel_id": "SP-HYD-006",
        "location": "Block B - Ground Array 1",
        "timestamp": "2026-09-22T10:15:00Z",
    },
    {
        "filename": "Electrical (1).jpg",
        "panel_id": "SP-HYD-007",
        "location": "Block B - Ground Array 2",
        "timestamp": "2026-09-22T10:30:00Z",
    },
    {
        "filename": "Electrical (10).jpg",
        "panel_id": "SP-HYD-008",
        "location": "Block B - Ground Array 2",
        "timestamp": "2026-09-22T10:45:00Z",
    },
    {
        "filename": "Physical (1).jpg",
        "panel_id": "SP-HYD-009",
        "location": "Block C - Carport Canopy 1",
        "timestamp": "2026-09-22T11:00:00Z",
    },
    {
        "filename": "Physical (19).jpg",
        "panel_id": "SP-HYD-010",
        "location": "Block C - Carport Canopy 1",
        "timestamp": "2026-09-22T11:15:00Z",
    },
    {
        "filename": "Snow (107).jpg",
        "panel_id": "SP-HYD-011",
        "location": "Block C - Incline Mount 3",
        "timestamp": "2026-09-22T11:30:00Z",
    },
    {
        "filename": "Snow (108).jpg",
        "panel_id": "SP-HYD-012",
        "location": "Block C - Incline Mount 3",
        "timestamp": "2026-09-22T11:45:00Z",
    },
]


def create_panel_metadata_plot(
    records: List[Dict[str, Any]],
    image_dir: Path,
    output_path: Path,
):
    """
    Creates an academic, clean visual summary grid demonstrating the metadata layer:
    Displays 6 representative samples (one per class) showing:
    Original Image + Panel ID + Location + Predicted Fault + Severity + Urgency.
    """
    classes_order = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-damage", "Snow-Covered"]
    selected_records = []
    seen_classes = set()

    for r in records:
        cls_name = r["true_class"]
        if cls_name in classes_order and cls_name not in seen_classes:
            selected_records.append(r)
            seen_classes.add(cls_name)

    # Sort in canonical order
    selected_records.sort(key=lambda x: classes_order.index(x["true_class"]))

    num_rows = len(selected_records)
    fig, axes = plt.subplots(num_rows, 2, figsize=(11, 2.5 * num_rows), dpi=300)

    urgency_colors = {
        "ROUTINE": "#2e7d32",          # Dark Green
        "SCHEDULED": "#1976d2",        # Blue
        "PRIORITY": "#e65100",         # Dark Orange
        "IMMEDIATE_REVIEW": "#b71c1c", # Deep Red
    }

    for idx, r in enumerate(selected_records):
        # 1. Image
        ax_img = axes[idx, 0]
        img_path = image_dir / r["true_class"] / r["image_filename"]
        if img_path.exists():
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            ax_img.imshow(img_rgb)
        else:
            # Fallback blank
            ax_img.imshow(np.zeros((224, 224, 3), dtype=np.uint8))

        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_ylabel(f"{r['true_class']}", fontsize=10, fontweight="bold", labelpad=8)
        if idx == 0:
            ax_img.set_title("Inspected Module Image", fontsize=11, fontweight="bold", pad=8)

        # 2. Metadata & Diagnostic Card
        ax_card = axes[idx, 1]
        ax_card.axis("off")

        urg = r["urgency"]
        sev = r["severity"]
        border_color = urgency_colors.get(urg, "#424242")

        info_text = (
            f"PANEL ID        : {r['panel_id']}\n"
            f"LOCATION        : {r['location']}\n"
            f"TIMESTAMP       : {r['inspection_timestamp']}\n"
            f"PREDICTED FAULT : {r['predicted_class']} ({r['confidence']*100:.1f}%)\n"
            f"VISUAL SEVERITY : {sev} (Fault Area: {r['visual_region_area_percent']}%\n"
            f"URGENCY         : {urg}\n"
            f"ACTION          : {r['maintenance_action']}"
        )

        ax_card.text(
            0.04, 0.50,
            info_text,
            fontsize=9.0,
            family="monospace",
            verticalalignment="center",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#fafafa",
                edgecolor=border_color,
                linewidth=1.8,
            ),
        )
        if idx == 0:
            ax_card.set_title("Operational Metadata & Diagnostic Card", fontsize=11, fontweight="bold", pad=8)

    plt.suptitle("Solar Panel Inspection Records (Operational Metadata Layer)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved panel metadata sample grid plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate solar panel inspection records with operational metadata.")
    args = parser.parse_args()

    root = get_project_root()
    ckpt_path = root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Checkpoint SHA256 integrity check before execution
    with open(ckpt_path, "rb") as f:
        sha_before = hashlib.sha256(f.read()).hexdigest()
    logger.info(f"Checkpoint SHA256 before metadata generation: {sha_before}")

    # Load existing Phase 11 maintenance recommendations
    p11_path = root / "results" / "metrics" / "maintenance_recommendations.json"
    if not p11_path.exists():
        raise FileNotFoundError(f"Phase 11 output not found at: {p11_path}")

    with open(p11_path, "r", encoding="utf-8") as f:
        p11_results = json.load(f)

    # Index by filename
    p11_by_filename = {r["filename"]: r for r in p11_results}

    inspection_records = []
    print("\n" + "=" * 50)
    print("SOLAR PANEL INSPECTION RECORDS (METADATA LAYER)")
    print("=" * 50)

    for item in DEMO_METADATA:
        fname = item["filename"]
        diag = p11_by_filename.get(fname)
        if not diag:
            logger.warning(f"No Phase 11 diagnostic found for {fname}, skipping.")
            continue

        # Create and validate metadata
        meta = PanelMetadata(
            panel_id=item["panel_id"],
            location=item["location"],
            inspection_timestamp=item["timestamp"],
        )

        record = create_inspection_record(meta, diag)
        inspection_records.append(record)

        print(f"Panel ID             : {record['panel_id']}")
        print(f"Location             : {record['location']}")
        print(f"Timestamp            : {record['inspection_timestamp']}")
        print(f"Image                : {record['image_filename']}")
        print(f"Predicted Fault      : {record['predicted_class']} ({record['confidence']*100:.1f}%)")
        print(f"Visual Severity      : {record['severity']}")
        print(f"Urgency              : {record['urgency']}")
        print(f"Maintenance Action   : {record['maintenance_action']}")
        print(f"Manual Review        : {'YES' if record['manual_inspection_recommended'] else 'NO'}")
        if record["confidence_warning"]:
            print(f"Confidence Warning   : {record['confidence_warning']}")
        print("-" * 50)

    # Save to results/metrics/panel_metadata_examples.json
    out_json_path = root / "results" / "metrics" / "panel_metadata_examples.json"
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(inspection_records, f, indent=2)
    logger.info(f"Saved {len(inspection_records)} inspection records to: {out_json_path}")

    # Generate academic sample grid plot
    test_dir = root / "data" / "test"
    plot_path = root / "results" / "plots" / "panel_metadata_sample_grid.png"
    create_panel_metadata_plot(inspection_records, test_dir, plot_path)

    # Checkpoint SHA256 integrity check after execution
    with open(ckpt_path, "rb") as f:
        sha_after = hashlib.sha256(f.read()).hexdigest()
    assert sha_before == sha_after, "Checkpoint was modified during metadata script execution!"
    logger.info(f"Checkpoint SHA256 after metadata generation: {sha_after} (UNCHANGED)")


if __name__ == "__main__":
    main()
