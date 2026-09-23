"""
Robustness Evaluation CLI for Solar Panel Fault Classification.

Evaluates the EfficientNet-B0 baseline checkpoint across 10 controlled
robustness conditions on the 177 test images.

Generates:
1. results/metrics/robustness_metrics.json
2. results/predictions/robustness_predictions.csv
3. results/plots/robustness_accuracy_comparison.png
4. results/plots/robustness_macro_f1_comparison.png
5. results/plots/robustness_performance_degradation.png
6. results/plots/robustness_sample_grid.png
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.robustness.evaluator import RobustnessEvaluator
from src.robustness.transforms import (
    ROBUSTNESS_CONDITIONS,
    apply_condition_to_image,
)
from src.utils.config import get_project_root
from src.utils.logger import setup_logger

logger = setup_logger("evaluate_robustness_cli")


def plot_accuracy_comparison(metrics_per_condition: dict, output_path: Path):
    """Generates bar chart of accuracy across all conditions."""
    conditions = list(metrics_per_condition.keys())
    accuracies = [metrics_per_condition[c]["accuracy"] * 100 for c in conditions]
    orig_acc = accuracies[0]

    plt.figure(figsize=(12, 6), dpi=300)
    bars = plt.bar(conditions, accuracies, color="#1f77b4", edgecolor="#0d47a1", alpha=0.85)

    # Highlight original
    bars[0].set_color("#2ca02c")
    bars[0].set_edgecolor("#1b5e20")

    plt.axhline(orig_acc, color="#d62728", linestyle="--", linewidth=1.5, label=f"Original ({orig_acc:.1f}%)")

    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.title("Solar Panel Fault Detection — Accuracy Across Robustness Conditions", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Condition", fontsize=11, fontweight="bold")
    plt.ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    plt.ylim(0, 100)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved accuracy comparison plot to: {output_path}")


def plot_macro_f1_comparison(metrics_per_condition: dict, output_path: Path):
    """Generates bar chart of Macro F1 score across all conditions."""
    conditions = list(metrics_per_condition.keys())
    f1_scores = [metrics_per_condition[c]["macro_f1"] * 100 for c in conditions]
    orig_f1 = f1_scores[0]

    plt.figure(figsize=(12, 6), dpi=300)
    bars = plt.bar(conditions, f1_scores, color="#ff7f0e", edgecolor="#d84315", alpha=0.85)

    # Highlight original
    bars[0].set_color("#2ca02c")
    bars[0].set_edgecolor("#1b5e20")

    plt.axhline(orig_f1, color="#d62728", linestyle="--", linewidth=1.5, label=f"Original ({orig_f1:.1f}%)")

    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.title("Solar Panel Fault Detection — Macro F1 Score Across Robustness Conditions", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Condition", fontsize=11, fontweight="bold")
    plt.ylabel("Macro F1 (%)", fontsize=11, fontweight="bold")
    plt.ylim(0, 100)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved Macro F1 comparison plot to: {output_path}")


def plot_performance_degradation(degradation_summary: dict, output_path: Path):
    """Generates grouped bar chart showing accuracy and macro F1 degradation relative to original."""
    # Exclude ORIGINAL
    conditions = [c for c in degradation_summary.keys() if c != "ORIGINAL"]
    acc_drops = [degradation_summary[c]["accuracy_pct_drop"] for c in conditions]
    f1_drops = [degradation_summary[c]["macro_f1_pct_drop"] for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    plt.figure(figsize=(13, 6), dpi=300)
    rects1 = plt.bar(x - width / 2, acc_drops, width, label="Accuracy Drop (pts)", color="#e53935", alpha=0.85)
    rects2 = plt.bar(x + width / 2, f1_drops, width, label="Macro F1 Drop (pts)", color="#8e24aa", alpha=0.85)

    plt.axhline(0, color="black", linewidth=1.0)

    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        va = "top" if height < 0 else "bottom"
        plt.annotate(
            f"{height:+.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, -8 if height < 0 else 3),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
            fontweight="bold",
        )

    plt.title("Performance Degradation Relative to Pristine Image Acquisition", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Environmental & Sensor Perturbation Condition", fontsize=11, fontweight="bold")
    plt.ylabel("Performance Delta (% points)", fontsize=11, fontweight="bold")
    plt.xticks(x, conditions, rotation=30, ha="right", fontsize=9)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved performance degradation plot to: {output_path}")


def plot_sample_grid(sample_image_path: Path, output_path: Path):
    """Generates a 2x5 grid showing the exact same image under all 10 conditions."""
    with Image.open(sample_image_path) as pil_img:
        base_img = np.array(pil_img.convert("RGB").resize((224, 224)))

    fig, axes = plt.subplots(2, 5, figsize=(16, 7), dpi=300)
    axes = axes.flatten()

    for idx, condition in enumerate(ROBUSTNESS_CONDITIONS):
        transformed = apply_condition_to_image(base_img, condition)
        ax = axes[idx]
        ax.imshow(transformed)
        title = condition.replace("_", " ").title()
        if condition == "ORIGINAL":
            ax.set_title(f"{title} (Baseline)", fontsize=10, fontweight="bold", color="#2e7d32")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2e7d32")
                spine.set_linewidth(2.5)
        else:
            ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Controlled Environmental & Sensor Transformations (Test Sample Grid)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved sample grid to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run robustness evaluation on solar panel classifier.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--test-dir", type=str, default=None, help="Path to test set directory")
    args = parser.parse_args()

    root = get_project_root()
    evaluator = RobustnessEvaluator(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        device="cpu",
    )

    results_dict, predictions_df = evaluator.run_all_conditions()

    # Save metrics JSON
    metrics_path = root / "results" / "metrics" / "robustness_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Saved robustness metrics to: {metrics_path}")

    # Save predictions CSV
    preds_path = root / "results" / "predictions" / "robustness_predictions.csv"
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(preds_path, index=False)
    logger.info(f"Saved robustness predictions ({len(predictions_df)} rows) to: {preds_path}")

    # Generate plots
    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_comparison(results_dict["metrics_per_condition"], plots_dir / "robustness_accuracy_comparison.png")
    plot_macro_f1_comparison(results_dict["metrics_per_condition"], plots_dir / "robustness_macro_f1_comparison.png")
    plot_performance_degradation(results_dict["degradation_relative_to_original"], plots_dir / "robustness_performance_degradation.png")

    # Pick a representative test image (e.g. Physical-damage or Bird-drop)
    sample_img = evaluator.test_samples[0]["path"]
    for s in evaluator.test_samples:
        if s["class_name"] in ["Physical-damage", "Bird-drop", "Dusty"]:
            sample_img = s["path"]
            break
    plot_sample_grid(sample_img, plots_dir / "robustness_sample_grid.png")

    # Print summary table to stdout
    print("\n" + "=" * 90)
    print(" ROBUSTNESS EVALUATION REPORT: EFFICIENTNET-B0 BASELINE")
    print("=" * 90)
    print(f"{'Condition':<20} | {'Accuracy':<9} | {'Macro Prec':<11} | {'Macro Rec':<10} | {'Macro F1':<9} | {'Weighted F1':<11} | {'Avg Conf':<8}")
    print("-" * 90)
    for cond, m in results_dict["metrics_per_condition"].items():
        print(
            f"{cond:<20} | {m['accuracy']*100:>7.2f}% | {m['macro_precision']*100:>9.2f}% | "
            f"{m['macro_recall']*100:>8.2f}% | {m['macro_f1']*100:>7.2f}% | {m['weighted_f1']*100:>9.2f}% | "
            f"{m['average_confidence']*100:>6.1f}%"
        )
    print("=" * 90)
    print(f"Worst Condition by Accuracy : {results_dict['worst_condition_by_accuracy']['condition']} "
          f"({results_dict['worst_condition_by_accuracy']['accuracy']*100:.2f}%, Drop: {results_dict['worst_condition_by_accuracy']['drop_pct']:+.2f}%)")
    print(f"Worst Condition by Macro F1 : {results_dict['worst_condition_by_f1']['condition']} "
          f"({results_dict['worst_condition_by_f1']['macro_f1']*100:.2f}%, Drop: {results_dict['worst_condition_by_f1']['drop_pct']:+.2f}%)")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
