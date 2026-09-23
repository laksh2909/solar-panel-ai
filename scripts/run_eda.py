"""
CLI Script to execute Exploratory Data Analysis (EDA).
Usage:
    python scripts/run_eda.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.eda import run_eda
from src.utils.logger import setup_logger

logger = setup_logger("run_eda_cli")


def main():
    logger.info("Starting EDA pipeline...")
    outputs = run_eda()
    print("\n" + "=" * 60)
    print(" EXPLORATORY DATA ANALYSIS (EDA) COMPLETE")
    print("=" * 60)
    for plot_name, file_path in outputs.items():
        print(f"  - {plot_name:<22}: {file_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
