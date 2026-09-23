"""
Dataset Acquisition and Archive Extraction Module.

Handles:
- Safe extraction of Kaggle dataset archive (`archive.zip`)
- Normalization of class directory names (e.g., Physical-Damage -> Physical-damage)
- Non-destructive processing: never moves, renames, or deletes the source zip file
- Validation of target classes post-extraction
- Kaggle API download fallback guidance without hardcoding credentials
"""

import os
from pathlib import Path
import shutil
import zipfile
from typing import Dict, List, Optional, Tuple

from src.utils.logger import setup_logger
from src.utils.config import get_project_root

logger = setup_logger("dataset_acquisition")

TARGET_CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-damage",
    "Snow-Covered",
]

# Canonical casing mapping to resolve dataset variances
CLASS_NAME_NORMALIZATION = {
    "bird-drop": "Bird-drop",
    "clean": "Clean",
    "dusty": "Dusty",
    "electrical-damage": "Electrical-damage",
    "physical-damage": "Physical-damage",
    "snow-covered": "Snow-Covered",
}


def check_raw_dataset_exists(raw_dir: Path) -> Tuple[bool, Dict[str, int]]:
    """
    Checks if raw directory already contains images organized by target classes.
    """
    if not raw_dir.exists():
        return False, {}

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    counts: Dict[str, int] = {}
    total_images = 0

    for c in TARGET_CLASSES:
        class_folder = raw_dir / c
        if class_folder.exists() and class_folder.is_dir():
            n = len([f for f in class_folder.iterdir() if f.suffix.lower() in valid_extensions and f.is_file()])
            counts[c] = n
            total_images += n
        else:
            counts[c] = 0

    is_complete = total_images > 0 and all(counts[c] > 0 for c in TARGET_CLASSES)
    return is_complete, counts


def extract_dataset_from_zip(
    zip_path: Path | str,
    output_raw_dir: Path | str,
    force: bool = False
) -> Dict[str, int]:
    """
    Extracts images from the dataset zip archive into `data/raw/<class_name>/`.
    Does NOT delete, rename, or modify the source zip archive.
    """
    zip_file = Path(zip_path).resolve()
    raw_dir = Path(output_raw_dir).resolve()

    if not zip_file.exists():
        raise FileNotFoundError(f"Dataset archive not found at: {zip_file}")

    raw_dir.mkdir(parents=True, exist_ok=True)

    is_complete, existing_counts = check_raw_dataset_exists(raw_dir)
    if is_complete and not force:
        logger.info(f"Raw dataset already populated in {raw_dir}: {existing_counts}")
        return existing_counts

    logger.info(f"Extracting dataset from archive: {zip_file}")
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    extracted_counts: Dict[str, int] = {c: 0 for c in TARGET_CLASSES}

    with zipfile.ZipFile(zip_file, "r") as z:
        for member in z.infolist():
            if member.is_dir():
                continue

            # Normalized path parts
            rel_path = member.filename.replace("\\", "/").strip("/")
            parts = rel_path.split("/")
            
            # Look for matching class name in the path components
            matched_class = None
            for p in parts:
                norm_p = CLASS_NAME_NORMALIZATION.get(p.lower())
                if norm_p:
                    matched_class = norm_p
                    break

            if not matched_class:
                continue

            ext = Path(member.filename).suffix.lower()
            if ext not in valid_extensions:
                continue

            target_class_dir = raw_dir / matched_class
            target_class_dir.mkdir(parents=True, exist_ok=True)

            filename = Path(member.filename).name
            target_file_path = target_class_dir / filename

            # Extract member to target path
            with z.open(member) as source, open(target_file_path, "wb") as target:
                shutil.copyfileobj(source, target)

            extracted_counts[matched_class] += 1

    total_extracted = sum(extracted_counts.values())
    logger.info(
        f"Extraction complete! Total extracted: {total_extracted} images across {len(TARGET_CLASSES)} classes."
    )
    for c, cnt in extracted_counts.items():
        logger.info(f"  - {c}: {cnt} images")

    return extracted_counts


def acquire_dataset(
    archive_zip_path: Optional[Path | str] = None,
    raw_dir: Optional[Path | str] = None,
    force: bool = False
) -> Tuple[Path, Dict[str, int]]:
    """
    Orchestrates dataset acquisition:
    1. Checks if raw dataset already exists in raw_dir.
    2. If not, checks provided or standard archive zip paths.
    3. If archive found, extracts and normalizes.
    4. If not found, raises descriptive error with Kaggle download instructions.
    """
    root = get_project_root()
    if raw_dir is None:
        raw_dir = root / "data" / "raw"
    else:
        raw_dir = Path(raw_dir)

    is_complete, counts = check_raw_dataset_exists(raw_dir)
    if is_complete and not force:
        logger.info(f"Dataset already present in {raw_dir}.")
        return raw_dir, counts

    # Search candidates for archive.zip
    candidates: List[Path] = []
    if archive_zip_path:
        candidates.append(Path(archive_zip_path))

    # Standard desktop / download candidate paths
    home_dir = Path.home()
    candidates.extend([
        home_dir / "OneDrive" / "Desktop" / "archive.zip",
        home_dir / "Desktop" / "archive.zip",
        home_dir / "Downloads" / "archive.zip",
        root / "archive.zip"
    ])

    found_zip: Optional[Path] = None
    for cand in candidates:
        if cand.exists() and cand.is_file():
            found_zip = cand
            break

    if found_zip:
        logger.info(f"Found dataset archive at: {found_zip}")
        counts = extract_dataset_from_zip(found_zip, raw_dir, force=force)
        return raw_dir, counts

    # If archive is not found, report manual download requirement
    instructions = (
        "Dataset archive.zip was not found in standard locations.\n"
        "Please download the dataset from Kaggle:\n"
        "  Kaggle slug: pythonafroz/solar-panel-images\n"
        "  URL: https://www.kaggle.com/datasets/pythonafroz/solar-panel-images\n"
        "Place the downloaded archive.zip at C:\\Users\\laksh\\OneDrive\\Desktop\\archive.zip\n"
        "or directly into the data/raw/ directory."
    )
    logger.error(instructions)
    raise FileNotFoundError(instructions)
