"""
CLI Script to partition the dataset into 60% Train, 20% Val, 20% Test.
Usage:
    python scripts/split_dataset.py
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.splitter import execute_dataset_split
from src.utils.logger import setup_logger

logger = setup_logger("split_cli")


def main():
    logger.info("Executing stratified dataset partitioning (60/20/20)...")
    split_df, counts = execute_dataset_split(copy_to_directories=True)

    print("\n" + "=" * 65)
    print(" DATASET PARTITIONING COMPLETE (60% Train / 20% Val / 20% Test)")
    print("=" * 65)
    print(f"Total Images: {len(split_df):,}")
    print(f"  - Train Count : {len(split_df[split_df['split'] == 'train']):>4} ({len(split_df[split_df['split'] == 'train']) / len(split_df) * 100:.1f}%)")
    print(f"  - Val Count   : {len(split_df[split_df['split'] == 'val']):>4} ({len(split_df[split_df['split'] == 'val']) / len(split_df) * 100:.1f}%)")
    print(f"  - Test Count  : {len(split_df[split_df['split'] == 'test']):>4} ({len(split_df[split_df['split'] == 'test']) / len(split_df) * 100:.1f}%)")
    print("-" * 65)
    print("Per-Class Distribution Matrix:")
    crosstab = pd.crosstab(split_df["class_name"], split_df["split"], margins=True)
    print(crosstab)
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
