"""
Stratified Dataset Splitting and Partitioning Module.

Features:
- Reproducible 60% Train / 20% Validation / 20% Test partitioning
- Grouped by image content hash (MD5) to guarantee zero data leakage
- Stratified across all six photovoltaic defect classes
- Generates machine-readable `results/metrics/dataset_split.csv`
- Populates physical split directories `data/train/`, `data/val/`, `data/test/`
  with class subfolders, copying files non-destructively
"""

from pathlib import Path
import shutil
from typing import Dict, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import setup_logger
from src.utils.config import load_config, get_project_root

logger = setup_logger("dataset_splitter")


def create_stratified_split(
    summary_df: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Partitions the dataset into train, val, and test splits with stratification
    and hash-based grouping to prevent data leakage.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    valid_df = summary_df[summary_df["is_valid"]].copy()
    if valid_df.empty:
        raise ValueError("No valid images available to split.")

    # Group by md5_hash to ensure identical image contents are placed into the SAME split
    group_df = valid_df.groupby("md5_hash").agg(
        class_name=("class_name", lambda x: x.mode()[0]),
        count=("filename", "count")
    ).reset_index()

    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_hashes, temp_hashes = train_test_split(
        group_df,
        test_size=temp_ratio,
        stratify=group_df["class_name"],
        random_state=seed
    )

    # Second split: Val vs Test
    val_fraction = val_ratio / temp_ratio
    val_hashes, test_hashes = train_test_split(
        temp_hashes,
        test_size=(1.0 - val_fraction),
        stratify=temp_hashes["class_name"],
        random_state=seed
    )

    # Build hash-to-split map
    hash_to_split: Dict[str, str] = {}
    for h in train_hashes["md5_hash"]:
        hash_to_split[h] = "train"
    for h in val_hashes["md5_hash"]:
        hash_to_split[h] = "val"
    for h in test_hashes["md5_hash"]:
        hash_to_split[h] = "test"

    valid_df["split"] = valid_df["md5_hash"].map(hash_to_split)

    # Strict leakage validation
    train_h = set(valid_df[valid_df["split"] == "train"]["md5_hash"])
    val_h = set(valid_df[valid_df["split"] == "val"]["md5_hash"])
    test_h = set(valid_df[valid_df["split"] == "test"]["md5_hash"])

    leak_train_val = train_h & val_h
    leak_train_test = train_h & test_h
    leak_val_test = val_h & test_h

    if leak_train_val or leak_train_test or leak_val_test:
        raise RuntimeError(
            f"Data leakage detected! Overlapping hashes found: "
            f"train-val: {len(leak_train_val)}, train-test: {len(leak_train_test)}, val-test: {len(leak_val_test)}"
        )

    logger.info("Stratified leakage-free split successfully created.")
    return valid_df


def populate_split_directories(
    split_df: pd.DataFrame,
    root_dir: Optional[Path | str] = None,
    copy_files: bool = True
) -> Dict[str, Dict[str, int]]:
    """
    Creates and populates data/train, data/val, data/test directories with class folders.
    Does not delete or move raw images.
    """
    root = Path(root_dir) if root_dir else get_project_root()
    counts: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    for split_name in ["train", "val", "test"]:
        split_subset = split_df[split_df["split"] == split_name]
        split_dir = root / "data" / split_name

        for _, row in split_subset.iterrows():
            src_path = Path(row["file_path"])
            cls_name = row["class_name"]
            dest_dir = split_dir / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / row["filename"]

            if copy_files and not dest_file.exists():
                shutil.copy2(src_path, dest_file)

            counts[split_name][cls_name] = counts[split_name].get(cls_name, 0) + 1

    return counts


def execute_dataset_split(
    config: Optional[Dict] = None,
    copy_to_directories: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    High-level orchestrator: loads summary CSV, performs split, exports CSV,
    and populates data/train, data/val, data/test.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    summary_csv = root / "results" / "metrics" / "dataset_summary.csv"
    output_split_csv = root / "results" / "metrics" / "dataset_split.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(
            f"Summary CSV not found at {summary_csv}. Run validation first."
        )

    summary_df = pd.read_csv(summary_csv)

    ratios = config.get("dataset", {}).get("split_ratios", config.get("data", {}).get("split_ratios", {}))
    train_r = ratios.get("train", 0.60)
    val_r = ratios.get("val", 0.20)
    test_r = ratios.get("test", 0.20)
    seed = config.get("project", {}).get("seed", 42)

    split_df = create_stratified_split(
        summary_df,
        train_ratio=train_r,
        val_ratio=val_r,
        test_ratio=test_r,
        seed=seed
    )

    # Save split manifest
    split_df.to_csv(output_split_csv, index=False)
    logger.info(f"Dataset split manifest saved to: {output_split_csv}")

    counts = populate_split_directories(split_df, root_dir=root, copy_files=copy_to_directories)
    logger.info("Physical split folders (data/train, data/val, data/test) populated.")

    return split_df, counts


if __name__ == "__main__":
    execute_dataset_split()
