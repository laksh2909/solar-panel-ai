"""
Dataset Ingestion, Validation, and Partitioning Module.

Handles:
- Image file validation and integrity verification
- Stratified Train / Validation / Test dataset partitioning
- Dataset manifest generation (CSV format)
- Class balance analysis and loss-weight computation
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from src.utils.logger import setup_logger
from src.utils.config import load_config, get_project_root

logger = setup_logger("data_ingestion")


def validate_image_file(image_path: Path) -> Tuple[bool, str]:
    """
    Checks if an image file is readable, uncorrupted, and has valid dimensions.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        if not image_path.exists() or image_path.stat().st_size == 0:
            return False, "File empty or non-existent"

        with Image.open(image_path) as img:
            img.verify()  # Fast structural verification

        # Re-open to verify decoding and get shape
        with Image.open(image_path) as img:
            img_format = img.format
            w, h = img.size
            if w <= 0 or h <= 0:
                return False, f"Invalid dimensions: {w}x{h}"
            if img_format not in ["JPEG", "PNG", "WEBP", "TIFF", "BMP"]:
                return False, f"Unsupported format: {img_format}"

        return True, "OK"
    except Exception as e:
        return False, str(e)


def scan_and_validate_dataset(raw_dir: Path | str, allowed_classes: List[str] | None = None) -> pd.DataFrame:
    """
    Scans the raw data directory, validates all images, and returns a metadata DataFrame.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_path}")

    records = []
    logger.info(f"Scanning raw images in: {raw_path}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    # Locate all subdirectories as classes
    class_dirs = [d for d in raw_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not class_dirs:
        logger.warning(f"No class subdirectories found in {raw_path}")

    for c_dir in sorted(class_dirs, key=lambda x: x.name):
        class_name = c_dir.name
        if allowed_classes and class_name not in allowed_classes:
            continue

        for img_file in c_dir.iterdir():
            if img_file.suffix.lower() in valid_extensions:
                is_valid, error_msg = validate_image_file(img_file)
                if is_valid:
                    with Image.open(img_file) as im:
                        w, h = im.size
                        channels = len(im.getbands())
                    records.append({
                        "file_path": str(img_file.resolve()),
                        "relative_path": str(img_file.relative_to(raw_path)),
                        "filename": img_file.name,
                        "class_name": class_name,
                        "width": w,
                        "height": h,
                        "channels": channels,
                        "valid": True
                    })
                else:
                    logger.warning(f"Skipping invalid/corrupt image {img_file}: {error_msg}")

    df = pd.DataFrame(records)
    logger.info(f"Scan complete. Total valid images found: {len(df)}")
    return df


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into stratified Train, Validation, and Test subsets.
    """
    if df.empty:
        raise ValueError("Cannot split an empty dataset DataFrame")

    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    # Step 1: Split into Train and Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["class_name"],
        random_state=seed
    )

    # Step 2: Split Temp into Validation and Test
    val_fraction_of_temp = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_fraction_of_temp),
        stratify=temp_df["class_name"],
        random_state=seed
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    logger.info(f"Dataset split complete: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df


def compute_class_weights(df_train: pd.DataFrame, class_to_idx: Dict[str, int]) -> np.ndarray:
    """
    Calculates balanced inverse class frequency weights for use in loss functions.
    weight_c = N_total / (N_classes * count_c)
    """
    class_counts = df_train["class_name"].value_counts().to_dict()
    total_samples = len(df_train)
    num_classes = len(class_to_idx)
    
    weights = np.zeros(num_classes, dtype=np.float32)
    for class_name, idx in class_to_idx.items():
        count = class_counts.get(class_name, 0)
        if count > 0:
            weights[idx] = total_samples / (num_classes * count)
        else:
            weights[idx] = 1.0

    return weights


def ingest_and_prepare_manifests(config: Dict | None = None) -> Dict[str, pd.DataFrame]:
    """
    High-level orchestrator: scans raw directory, generates splits, writes CSV manifests.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    raw_dir = root / config["data"]["raw_dir"]
    processed_dir = root / config["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    allowed_classes = config["data"]["classes"]
    split_ratios = config["data"]["split_ratios"]
    seed = config["project"]["seed"]

    df = scan_and_validate_dataset(raw_dir, allowed_classes=allowed_classes)
    if df.empty:
        raise RuntimeError(f"No valid images found in {raw_dir}. Ensure dataset is downloaded or synthesized.")

    train_df, val_df, test_df = create_stratified_splits(
        df,
        train_ratio=split_ratios["train"],
        val_ratio=split_ratios["val"],
        test_ratio=split_ratios["test"],
        seed=seed
    )

    # Save manifests
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
    
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df.to_csv(processed_dir / "dataset_manifest.csv", index=False)

    logger.info(f"Manifests successfully saved in {processed_dir}")
    return {"train": train_df, "val": val_df, "test": test_df, "full": full_df}


if __name__ == "__main__":
    ingest_and_prepare_manifests()
