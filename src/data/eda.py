"""
Exploratory Data Analysis (EDA) Module for Solar Panel Fault Detection.

Generates publication-quality visualizations:
1. Class distribution bar plot with counts and proportions (results/plots/class_distribution.png)
2. Representative multi-sample grid of solar panel images across all six classes (results/plots/sample_images.png)
3. Image dimension and aspect ratio distribution analysis (results/plots/image_dimensions.png)
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns

from src.utils.logger import setup_logger
from src.utils.config import get_project_root

logger = setup_logger("eda")

# Distinct, professional color palette for solar panel classes
CLASS_PALETTE = {
    "Bird-drop": "#E76F51",
    "Clean": "#2A9D8F",
    "Dusty": "#E9C46A",
    "Electrical-damage": "#D62828",
    "Physical-damage": "#9D4EDD",
    "Snow-Covered": "#4EA8DE",
}


def plot_class_distribution(
    df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """
    Generates a stylish bar plot of image counts per class with annotations.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    counts = df[df["is_valid"]]["class_name"].value_counts().sort_index()
    total_valid = len(df[df["is_valid"]])

    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid", font="sans-serif")

    colors = [CLASS_PALETTE.get(c, "#4A90E2") for c in counts.index]
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=colors, legend=False)

    plt.title(
        f"Solar Panel Defect Class Distribution (Total: {total_valid:,} Valid Images)",
        fontsize=14,
        fontweight="bold",
        pad=15
    )
    plt.xlabel("Fault / Panel Condition Class", fontsize=11, fontweight="semibold", labelpad=10)
    plt.ylabel("Number of Images", fontsize=11, fontweight="semibold", labelpad=10)
    plt.xticks(rotation=15, ha="right", fontsize=10)

    # Annotate exact count and percentage above each bar
    max_val = counts.max()
    for p, c in zip(ax.patches, counts.values):
        pct = (c / total_valid) * 100.0
        ax.annotate(
            f"{c}\n({pct:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            xytext=(0, 4),
            textcoords="offset points"
        )

    plt.ylim(0, max_val * 1.15)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

    logger.info(f"Class distribution plot saved to: {out_file}")
    return out_file


def plot_sample_images(
    df: pd.DataFrame,
    output_path: Path | str,
    samples_per_class: int = 3,
    seed: int = 42
) -> Path:
    """
    Generates a grid of representative sample images from all six classes.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    valid_df = df[df["is_valid"]]
    classes = sorted(valid_df["class_name"].unique())

    fig, axes = plt.subplots(
        nrows=len(classes),
        ncols=samples_per_class,
        figsize=(samples_per_class * 3.8, len(classes) * 3.2),
        dpi=300
    )

    for row_idx, cls_name in enumerate(classes):
        cls_subset = valid_df[valid_df["class_name"] == cls_name]
        chosen = cls_subset.sample(n=min(samples_per_class, len(cls_subset)), random_state=seed)

        for col_idx in range(samples_per_class):
            ax = axes[row_idx, col_idx] if len(classes) > 1 else axes[col_idx]
            if col_idx < len(chosen):
                row = chosen.iloc[col_idx]
                img_path = Path(row["file_path"])
                try:
                    with Image.open(img_path) as img:
                        ax.imshow(img.convert("RGB"))
                    dim_str = f"{row['width']}x{row['height']}"
                    ax.set_title(f"{cls_name} [{dim_str}]", fontsize=10, fontweight="semibold")
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center")
            ax.axis("off")

    plt.suptitle("Solar Panel Fault Detection — Class Visual Samples", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Sample images grid saved to: {out_file}")
    return out_file


def plot_image_dimensions(
    df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """
    Analyzes and plots image width, height, and aspect ratio distributions.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    valid_df = df[df["is_valid"]].copy()
    valid_df["width"] = valid_df["width"].astype(int)
    valid_df["height"] = valid_df["height"].astype(int)
    valid_df["aspect_ratio"] = valid_df["aspect_ratio"].astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    # Panel 1: Width vs Height Scatter Plot
    sns.scatterplot(
        data=valid_df,
        x="width",
        y="height",
        hue="class_name",
        palette=CLASS_PALETTE,
        alpha=0.75,
        s=50,
        ax=axes[0]
    )
    axes[0].set_title("Image Resolution Distribution (Width vs. Height)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Width (pixels)", fontsize=10)
    axes[0].set_ylabel("Height (pixels)", fontsize=10)
    axes[0].legend(title="Class", fontsize=8, title_fontsize=9)

    # Panel 2: Aspect Ratio Histogram & KDE
    sns.histplot(
        data=valid_df,
        x="aspect_ratio",
        kde=True,
        color="#264653",
        bins=25,
        ax=axes[1]
    )
    axes[1].axvline(1.0, color="#E76F51", linestyle="--", label="1:1 (Square)")
    axes[1].set_title("Aspect Ratio Distribution (Width / Height)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Aspect Ratio", fontsize=10)
    axes[1].set_ylabel("Frequency Count", fontsize=10)
    axes[1].legend()

    # Panel 3: Boxplot of Resolution / Pixel Area per Class
    valid_df["pixel_megapixels"] = (valid_df["width"] * valid_df["height"]) / 1e6
    sns.boxplot(
        data=valid_df,
        x="class_name",
        y="pixel_megapixels",
        hue="class_name",
        palette=CLASS_PALETTE,
        legend=False,
        ax=axes[2]
    )
    axes[2].set_title("Image Megapixels by Class", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Class", fontsize=10)
    axes[2].set_ylabel("Megapixels (Width * Height / 10^6)", fontsize=10)
    axes[2].tick_params(axis="x", rotation=25)

    plt.suptitle("Solar Panel Imagery Dimension & Spatial Analysis", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Image dimensions plot saved to: {out_file}")
    return out_file


def run_eda(
    summary_csv_path: Optional[Path | str] = None,
    plots_dir: Optional[Path | str] = None,
    seed: int = 42
) -> Dict[str, Path]:
    """
    Executes all EDA tasks and generates visualizations.
    """
    root = get_project_root()
    if summary_csv_path is None:
        summary_csv = root / "results" / "metrics" / "dataset_summary.csv"
    else:
        summary_csv = Path(summary_csv_path)

    if plots_dir is None:
        plots_path = root / "results" / "plots"
    else:
        plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found at: {summary_csv}. Run dataset validation first.")

    df = pd.read_csv(summary_csv)

    p1 = plot_class_distribution(df, plots_path / "class_distribution.png")
    p2 = plot_sample_images(df, plots_path / "sample_images.png", samples_per_class=3, seed=seed)
    p3 = plot_image_dimensions(df, plots_path / "image_dimensions.png")

    return {
        "class_distribution": p1,
        "sample_images": p2,
        "image_dimensions": p3,
    }


if __name__ == "__main__":
    run_eda()
