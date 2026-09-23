"""
Dataset Inspection and Validation Module.

Performs deep integrity and quality checks on solar panel imagery:
- Detects corrupted, truncated, or unreadable files
- Gathers image dimension distributions (min, max, mean, median, std)
- Verifies color modes and channel counts (RGB vs Grayscale vs RGBA)
- Identifies duplicate filenames and duplicate file content hashes (MD5)
- Exports machine-readable dataset_summary.csv and dataset_report.json
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image

from src.utils.logger import setup_logger
from src.utils.config import get_project_root

logger = setup_logger("dataset_validation")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_file_hash(file_path: Path, block_size: int = 65536) -> str:
    """Calculates MD5 hash of a file to detect bit-exact duplicates."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_single_image(file_path: Path) -> Dict[str, Any]:
    """
    Detailed inspection of an individual image file.
    """
    result: Dict[str, Any] = {
        "file_path": str(file_path.resolve()),
        "filename": file_path.name,
        "extension": file_path.suffix.lower(),
        "file_size_bytes": file_path.stat().st_size if file_path.exists() else 0,
        "is_valid": False,
        "error": None,
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "channels": None,
        "mode": None,
        "format": None,
        "md5_hash": None,
    }

    if not file_path.exists() or result["file_size_bytes"] == 0:
        result["error"] = "File empty or does not exist"
        return result

    if result["extension"] not in SUPPORTED_EXTENSIONS:
        result["error"] = f"Unsupported extension: {result['extension']}"
        return result

    try:
        # Step 1: Verify file structure
        with Image.open(file_path) as img:
            img.verify()

        # Step 2: Decode image to inspect pixel data and metadata
        with Image.open(file_path) as img:
            w, h = img.size
            result["width"] = w
            result["height"] = h
            result["aspect_ratio"] = round(w / h, 4) if h > 0 else 0
            result["mode"] = img.mode
            result["format"] = img.format
            result["channels"] = len(img.getbands())

            # Verify actual pixel loading
            img.load()

        result["md5_hash"] = compute_file_hash(file_path)
        result["is_valid"] = True
    except Exception as e:
        result["error"] = str(e)
        result["is_valid"] = False

    return result


def validate_dataset(
    raw_dir: Path | str,
    output_dir: Optional[Path | str] = None,
    target_classes: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Scans, inspects, and validates the entire raw dataset.
    Generates dataset_summary.csv and dataset_report.json.
    """
    root = get_project_root()
    raw_path = Path(raw_dir)
    if not raw_path.is_absolute():
        raw_path = root / raw_path

    if output_dir is None:
        metrics_dir = root / "results" / "metrics"
    else:
        metrics_dir = Path(output_dir)
        if not metrics_dir.is_absolute():
            metrics_dir = root / metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initiating full dataset inspection on: {raw_path}")
    records: List[Dict[str, Any]] = []

    class_dirs = [d for d in raw_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    for c_dir in sorted(class_dirs, key=lambda x: x.name):
        class_name = c_dir.name
        if target_classes and class_name not in target_classes:
            continue

        for img_file in c_dir.iterdir():
            if img_file.is_file() and not img_file.name.startswith("."):
                res = inspect_single_image(img_file)
                res["class_name"] = class_name
                res["relative_path"] = str(img_file.relative_to(raw_path)).replace("\\", "/")
                records.append(res)

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No images found in dataset directory!")
        return df, {}

    # Duplicate analysis
    filename_counts = df["filename"].value_counts()
    duplicate_filenames = filename_counts[filename_counts > 1].to_dict()

    valid_hashes = df[df["is_valid"]]["md5_hash"].value_counts()
    duplicate_hashes = valid_hashes[valid_hashes > 1].to_dict()

    # Dimension and channel statistics for valid images
    valid_df = df[df["is_valid"]]
    corrupted_df = df[~df["is_valid"]]

    widths = valid_df["width"].dropna().astype(int)
    heights = valid_df["height"].dropna().astype(int)
    aspect_ratios = valid_df["aspect_ratio"].dropna().astype(float)

    dim_stats = {
        "width": {
            "min": int(widths.min()) if len(widths) else 0,
            "max": int(widths.max()) if len(widths) else 0,
            "mean": round(float(widths.mean()), 2) if len(widths) else 0,
            "median": float(widths.median()) if len(widths) else 0,
            "std": round(float(widths.std()), 2) if len(widths) else 0,
        },
        "height": {
            "min": int(heights.min()) if len(heights) else 0,
            "max": int(heights.max()) if len(heights) else 0,
            "mean": round(float(heights.mean()), 2) if len(heights) else 0,
            "median": float(heights.median()) if len(heights) else 0,
            "std": round(float(heights.std()), 2) if len(heights) else 0,
        },
        "aspect_ratio": {
            "min": round(float(aspect_ratios.min()), 4) if len(aspect_ratios) else 0,
            "max": round(float(aspect_ratios.max()), 4) if len(aspect_ratios) else 0,
            "mean": round(float(aspect_ratios.mean()), 4) if len(aspect_ratios) else 0,
            "median": round(float(aspect_ratios.median()), 4) if len(aspect_ratios) else 0,
        }
    }

    # Extensions, formats, and channel modes
    ext_counts = df["extension"].value_counts().to_dict()
    channel_counts = valid_df["channels"].value_counts().to_dict()
    mode_counts = valid_df["mode"].value_counts().to_dict()
    class_counts = df["class_name"].value_counts().to_dict()
    valid_class_counts = valid_df["class_name"].value_counts().to_dict()

    report: Dict[str, Any] = {
        "dataset_name": "Solar Panel Images (Kaggle: pythonafroz/solar-panel-images)",
        "total_files_inspected": len(df),
        "valid_images_count": len(valid_df),
        "corrupted_images_count": len(corrupted_df),
        "corrupted_files": corrupted_df[["filename", "class_name", "error"]].to_dict(orient="records"),
        "duplicate_filename_count": len(duplicate_filenames),
        "duplicate_filenames": duplicate_filenames,
        "duplicate_content_hash_count": len(duplicate_hashes),
        "duplicate_content_hashes": duplicate_hashes,
        "classes_found": sorted(list(class_counts.keys())),
        "total_counts_per_class": class_counts,
        "valid_counts_per_class": valid_class_counts,
        "file_extensions": ext_counts,
        "channel_distribution": {f"{k}_channels": v for k, v in channel_counts.items()},
        "color_modes": mode_counts,
        "dimension_statistics": dim_stats,
    }

    # Save summary CSV and JSON report
    summary_csv_path = metrics_dir / "dataset_summary.csv"
    report_json_path = metrics_dir / "dataset_report.json"

    df.to_csv(summary_csv_path, index=False)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Dataset summary exported to: {summary_csv_path}")
    logger.info(f"Dataset report exported to: {report_json_path}")
    logger.info(f"Validation summary: Total={len(df)}, Valid={len(valid_df)}, Corrupted={len(corrupted_df)}")

    return df, report


if __name__ == "__main__":
    validate_dataset("data/raw")
