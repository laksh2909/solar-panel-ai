"""
CLI Script for Dataset Acquisition and Extraction.
Usage:
    python scripts/acquire_dataset.py [--zip-path PATH] [--raw-dir PATH] [--force]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.acquisition import acquire_dataset
from src.utils.logger import setup_logger

logger = setup_logger("acquire_script")


def main():
    parser = argparse.ArgumentParser(description="Acquire and extract the Solar Panel dataset archive.")
    parser.add_argument(
        "--zip-path",
        type=str,
        default=r"C:\Users\laksh\OneDrive\Desktop\archive.zip",
        help="Path to downloaded archive.zip"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Destination directory for raw class folders"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if data exists"
    )
    args = parser.parse_args()

    try:
        raw_path, counts = acquire_dataset(
            archive_zip_path=args.zip_path,
            raw_dir=args.raw_dir,
            force=args.force
        )
        print("\n" + "=" * 60)
        print(" DATASET ACQUISITION SUMMARY")
        print("=" * 60)
        print(f"Destination Path: {raw_path}")
        print(f"Total Extracted : {sum(counts.values())} images")
        for cls_name, count in counts.items():
            print(f"  - {cls_name:<20}: {count:>4} images")
        print("=" * 60 + "\n")
    except Exception as e:
        logger.error(f"Failed to acquire dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
