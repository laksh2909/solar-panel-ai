"""
Phase 17 — Final Model Comparison and Ablation Study CLI.

Evaluates and compares the 3 established model configurations:
1. MobileNetV2 baseline (models/checkpoints/mobilenetv2_baseline_best.pth)
2. EfficientNet-B0 baseline (models/checkpoints/efficientnet_b0_baseline_best.pth)
3. EfficientNet-B0 + domain-specific augmentation (models/checkpoints/efficientnet_b0_augmented_best.pth)

Uses the Phase 7A canonical test evaluation pipeline:
- Test dataset: data/test (177 images across 6 classes)
- Preprocessing: load_image_rgb (white composite on RGBA), Albumentations A.Resize(224, 224, cv2.INTER_LINEAR),
  ImageNet normalization, ToTensorV2
- CPU inference mode with deterministic evaluation

Generates:
- results/metrics/final_model_comparison.json
- results/metrics/final_model_comparison.csv
- results/metrics/per_class_model_comparison.csv
- results/metrics/ablation_study.csv
- results/plots/mobilenetv2_confusion_matrix.png
- results/plots/efficientnet_b0_confusion_matrix.png
- results/plots/efficientnet_b0_augmented_confusion_matrix.png
- results/plots/model_accuracy_comparison.png
- results/plots/model_macro_f1_comparison.png
- results/plots/model_weighted_f1_comparison.png
- results/plots/model_inference_time_comparison.png
- results/plots/model_parameter_count_comparison.png
- results/plots/per_class_f1_comparison.png
- results/plots/final_model_comparison_summary.png
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluation.metrics import plot_confusion_matrix
from src.models.efficientnet import build_efficientnet_b0, get_model_summary as get_effnet_summary
from src.models.mobilenet import build_mobilenet_v2, get_model_summary as get_mobilenet_summary
from src.preprocessing.augmentation import get_test_pipeline
from src.preprocessing.pipeline import load_image_rgb
from src.utils.config import get_project_root, load_config
from src.utils.logger import setup_logger

logger = setup_logger("final_model_comparison")


def discover_test_samples(test_dir: Path, classes: List[str]) -> List[Dict[str, Any]]:
    """Discovers and sorts all 177 test samples across the 6 classes."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []

    for class_folder in sorted(test_dir.iterdir()):
        if class_folder.is_dir() and class_folder.name in class_to_idx:
            c_name = class_folder.name
            c_idx = class_to_idx[c_name]
            for f in sorted(class_folder.iterdir()):
                if f.is_file() and f.suffix.lower() in valid_exts:
                    samples.append({
                        "path": f,
                        "filename": f.name,
                        "class_name": c_name,
                        "class_idx": c_idx,
                    })

    logger.info(f"Discovered {len(samples)} test samples across {len(class_to_idx)} classes.")
    return samples


def evaluate_model_on_test_set(
    model: nn.Module,
    samples: List[Dict[str, Any]],
    classes: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    """Runs deterministic canonical evaluation on the test set."""
    pipeline = get_test_pipeline()
    softmax = nn.Softmax(dim=1)

    y_true = []
    y_pred = []
    confidences = []

    model.eval()
    with torch.no_grad():
        for s in samples:
            img_rgb = load_image_rgb(s["path"])
            transformed = pipeline(image=img_rgb)
            tensor = transformed["image"].unsqueeze(0).to(device)

            logits = model(tensor)
            probs = softmax(logits)
            conf, pred = torch.max(probs, dim=1)

            y_true.append(s["class_idx"])
            y_pred.append(int(pred.item()))
            confidences.append(float(conf.item()))

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_p = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    weighted_p = float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    # Per-class breakdowns
    per_class_p = precision_score(y_true_arr, y_pred_arr, average=None, zero_division=0)
    per_class_r = recall_score(y_true_arr, y_pred_arr, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true_arr, y_pred_arr, average=None, zero_division=0)

    per_class_dict = {}
    for idx, name in enumerate(classes):
        per_class_dict[name] = {
            "precision": round(float(per_class_p[idx]), 4),
            "recall": round(float(per_class_r[idx]), 4),
            "f1_score": round(float(per_class_f1[idx]), 4),
            "support": int(np.sum(y_true_arr == idx)),
            "predictions_count": int(np.sum(y_pred_arr == idx)),
        }

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(classes)))).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_p, 4),
        "weighted_recall": round(weighted_r, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class_dict,
        "confusion_matrix": cm,
        "total_test_samples": len(y_true),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def generate_accuracy_plot(models_data: List[Dict[str, Any]], output_path: Path):
    """Generates bar chart comparing test accuracy across models."""
    names = [m["display_name"] for m in models_data]
    accuracies = [m["metrics"]["accuracy"] * 100.0 for m in models_data]
    colors = ["#457B9D", "#1D3557", "#E76F51"]

    plt.figure(figsize=(8.5, 5.5), dpi=300)
    sns.set_theme(style="whitegrid")

    bars = plt.bar(names, accuracies, color=colors, edgecolor="#2B2D42", width=0.55, linewidth=1.2, alpha=0.9)
    plt.title("Solar Panel Fault Detection — Test Accuracy Comparison", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Test Accuracy (%)", fontsize=11, fontweight="semibold")
    plt.ylim(0, 100)

    for bar in bars:
        h = bar.get_height()
        plt.annotate(
            f"{h:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )

    plt.axhline(85.31, color="#1D3557", linestyle="--", linewidth=1.2, alpha=0.7, label="EfficientNet-B0 Baseline (85.31%)")
    plt.legend(loc="lower right", frameon=True, fontsize=9.5)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved accuracy comparison plot to: {output_path}")


def generate_macro_f1_plot(models_data: List[Dict[str, Any]], output_path: Path):
    """Generates bar chart comparing Macro F1 across models."""
    names = [m["display_name"] for m in models_data]
    f1_scores = [m["metrics"]["macro_f1"] * 100.0 for m in models_data]
    colors = ["#457B9D", "#1D3557", "#E76F51"]

    plt.figure(figsize=(8.5, 5.5), dpi=300)
    sns.set_theme(style="whitegrid")

    bars = plt.bar(names, f1_scores, color=colors, edgecolor="#2B2D42", width=0.55, linewidth=1.2, alpha=0.9)
    plt.title("Solar Panel Fault Detection — Macro F1-Score Comparison", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Macro F1-Score (%)", fontsize=11, fontweight="semibold")
    plt.ylim(0, 100)

    for bar in bars:
        h = bar.get_height()
        plt.annotate(
            f"{h:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )

    plt.axhline(84.93, color="#1D3557", linestyle="--", linewidth=1.2, alpha=0.7, label="EfficientNet-B0 Baseline (84.93%)")
    plt.legend(loc="lower right", frameon=True, fontsize=9.5)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved macro F1 comparison plot to: {output_path}")


def generate_weighted_f1_plot(models_data: List[Dict[str, Any]], output_path: Path):
    """Generates bar chart comparing Weighted F1 across models."""
    names = [m["display_name"] for m in models_data]
    f1_scores = [m["metrics"]["weighted_f1"] * 100.0 for m in models_data]
    colors = ["#457B9D", "#1D3557", "#E76F51"]

    plt.figure(figsize=(8.5, 5.5), dpi=300)
    sns.set_theme(style="whitegrid")

    bars = plt.bar(names, f1_scores, color=colors, edgecolor="#2B2D42", width=0.55, linewidth=1.2, alpha=0.9)
    plt.title("Solar Panel Fault Detection — Weighted F1-Score Comparison", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Weighted F1-Score (%)", fontsize=11, fontweight="semibold")
    plt.ylim(0, 100)

    for bar in bars:
        h = bar.get_height()
        plt.annotate(
            f"{h:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )

    plt.axhline(85.25, color="#1D3557", linestyle="--", linewidth=1.2, alpha=0.7, label="EfficientNet-B0 Baseline (85.25%)")
    plt.legend(loc="lower right", frameon=True, fontsize=9.5)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved weighted F1 comparison plot to: {output_path}")


def generate_inference_time_plot(models_data: List[Dict[str, Any]], output_path: Path):
    """Generates dual comparison of latency and throughput (FPS)."""
    names = [m["display_name"] for m in models_data]
    latencies = [m["latency_ms"] for m in models_data]
    fps_vals = [m["fps"] for m in models_data]
    colors = ["#457B9D", "#1D3557", "#E76F51"]

    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)
    sns.set_theme(style="white")

    x = np.arange(len(names))
    width = 0.4

    rects1 = ax1.bar(x - width / 2, latencies, width, label="Latency (ms/img)", color=colors, edgecolor="#2B2D42", linewidth=1.1, alpha=0.9)
    ax1.set_ylabel("Inference Latency on CPU (ms/image)", fontsize=11, fontweight="semibold", color="#1D3557")
    ax1.set_ylim(0, max(latencies) * 1.35)
    ax1.tick_params(axis="y", labelcolor="#1D3557")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10, fontweight="semibold")

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width / 2, fps_vals, width, label="Throughput (FPS)", color=["#A8DADC", "#457B9D", "#F4A261"], edgecolor="#2B2D42", linewidth=1.1, alpha=0.75)
    ax2.set_ylabel("Throughput (Frames Per Second)", fontsize=11, fontweight="semibold", color="#2A9D8F")
    ax2.set_ylim(0, max(fps_vals) * 1.35)
    ax2.tick_params(axis="y", labelcolor="#2A9D8F")

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f} ms", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f} FPS", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.title("CPU Inference Latency & Throughput Benchmark", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved inference time plot to: {output_path}")


def generate_parameter_count_plot(models_data: List[Dict[str, Any]], output_path: Path):
    """Generates comparison of parameter count (millions) and disk footprint (MB)."""
    names = [m["display_name"] for m in models_data]
    params_m = [m["parameters"] / 1e6 for m in models_data]
    sizes_mb = [m["model_size_mb"] for m in models_data]

    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)
    sns.set_theme(style="white")

    x = np.arange(len(names))
    width = 0.35

    rects1 = ax1.bar(x - width / 2, params_m, width, label="Parameters (Millions)", color="#1D3557", edgecolor="#2B2D42", linewidth=1.1, alpha=0.85)
    ax1.set_ylabel("Total Parameters (Millions)", fontsize=11, fontweight="semibold", color="#1D3557")
    ax1.set_ylim(0, max(params_m) * 1.3)
    ax1.tick_params(axis="y", labelcolor="#1D3557")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10, fontweight="semibold")

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width / 2, sizes_mb, width, label="Model Size (MB)", color="#E76F51", edgecolor="#2B2D42", linewidth=1.1, alpha=0.85)
    ax2.set_ylabel("Checkpoint Size (MB)", fontsize=11, fontweight="semibold", color="#E76F51")
    ax2.set_ylim(0, max(sizes_mb) * 1.3)
    ax2.tick_params(axis="y", labelcolor="#E76F51")

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}M", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f} MB", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.title("Model Capacity: Parameter Count & Memory Footprint", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved parameter count plot to: {output_path}")


def generate_per_class_f1_plot(models_data: List[Dict[str, Any]], classes: List[str], output_path: Path):
    """Generates grouped bar chart comparing per-class F1-scores across models."""
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    x = np.arange(len(classes))
    width = 0.25
    colors = ["#457B9D", "#1D3557", "#E76F51"]

    for i, m in enumerate(models_data):
        f1_vals = [m["metrics"]["per_class"][c]["f1_score"] * 100.0 for c in classes]
        offset = (i - 1) * width
        rects = ax.bar(x + offset, f1_vals, width, label=m["display_name"], color=colors[i], edgecolor="#2B2D42", linewidth=1.0, alpha=0.9)
        for r in rects:
            h = r.get_height()
            ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold", rotation=0)

    ax.set_title("Per-Class F1-Score Breakdown Across All 6 Fault Categories", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("F1-Score (%)", fontsize=11, fontweight="semibold")
    ax.set_xlabel("Solar Panel Fault Category", fontsize=11, fontweight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=9.5, fontweight="semibold", rotation=15, ha="right")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved per-class F1 comparison plot to: {output_path}")


def generate_summary_panel(models_data: List[Dict[str, Any]], classes: List[str], output_path: Path):
    """Generates a clean 4-panel academic summary figure."""
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    sns.set_theme(style="whitegrid")
    colors = ["#457B9D", "#1D3557", "#E76F51"]
    names = [m["display_name"] for m in models_data]

    # Panel 1: Primary Metrics Grouped Bar
    x1 = np.arange(3)
    metric_labels = ["Accuracy", "Macro F1", "Weighted F1"]
    width = 0.25

    for i, m in enumerate(models_data):
        vals = [
            m["metrics"]["accuracy"] * 100.0,
            m["metrics"]["macro_f1"] * 100.0,
            m["metrics"]["weighted_f1"] * 100.0,
        ]
        offset = (i - 1) * width
        rects = axs[0, 0].bar(x1 + offset, vals, width, label=m["display_name"], color=colors[i], edgecolor="#2B2D42", linewidth=1.0)
        for r in rects:
            h = r.get_height()
            axs[0, 0].annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    axs[0, 0].set_title("(A) Overall Classification Metrics Comparison", fontsize=12, fontweight="bold", pad=10)
    axs[0, 0].set_xticks(x1)
    axs[0, 0].set_xticklabels(metric_labels, fontsize=10, fontweight="semibold")
    axs[0, 0].set_ylabel("Metric Score (%)", fontsize=10, fontweight="semibold")
    axs[0, 0].set_ylim(0, 105)
    axs[0, 0].legend(loc="lower right", fontsize=8.5, frameon=True)

    # Panel 2: Per-Class F1 Radar/Grouped Bar
    x2 = np.arange(len(classes))
    width2 = 0.25
    for i, m in enumerate(models_data):
        f1_vals = [m["metrics"]["per_class"][c]["f1_score"] * 100.0 for c in classes]
        offset = (i - 1) * width2
        rects = axs[0, 1].bar(x2 + offset, f1_vals, width2, label=m["display_name"], color=colors[i], edgecolor="#2B2D42", linewidth=1.0)
        for r in rects:
            h = r.get_height()
            axs[0, 1].annotate(f"{h:.0f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    axs[0, 1].set_title("(B) Per-Class Defect F1-Scores", fontsize=12, fontweight="bold", pad=10)
    axs[0, 1].set_xticks(x2)
    axs[0, 1].set_xticklabels(classes, fontsize=8.5, fontweight="semibold", rotation=20, ha="right")
    axs[0, 1].set_ylabel("F1 Score (%)", fontsize=10, fontweight="semibold")
    axs[0, 1].set_ylim(0, 115)
    axs[0, 1].legend(loc="upper right", fontsize=8.5, frameon=True)

    # Panel 3: Efficiency & Footprint (Params vs Size)
    x3 = np.arange(len(names))
    params_m = [m["parameters"] / 1e6 for m in models_data]
    sizes_mb = [m["model_size_mb"] for m in models_data]
    width3 = 0.35

    axs[1, 0].bar(x3 - width3 / 2, params_m, width3, label="Parameters (M)", color="#1D3557", edgecolor="#2B2D42", linewidth=1.0)
    axs[1, 0].bar(x3 + width3 / 2, sizes_mb, width3, label="Model Size (MB)", color="#457B9D", edgecolor="#2B2D42", linewidth=1.0)
    axs[1, 0].set_title("(C) Computational Footprint & Disk Size", fontsize=12, fontweight="bold", pad=10)
    axs[1, 0].set_xticks(x3)
    axs[1, 0].set_xticklabels(names, fontsize=9.5, fontweight="semibold")
    axs[1, 0].set_ylabel("Value (M / MB)", fontsize=10, fontweight="semibold")
    axs[1, 0].set_ylim(0, max(sizes_mb) * 1.3)
    axs[1, 0].legend(loc="upper left", fontsize=8.5, frameon=True)

    for i in range(len(names)):
        axs[1, 0].annotate(f"{params_m[i]:.2f}M", xy=(x3[i] - width3 / 2, params_m[i]), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        axs[1, 0].annotate(f"{sizes_mb[i]:.1f}MB", xy=(x3[i] + width3 / 2, sizes_mb[i]), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel 4: Latency vs. Accuracy Tradeoff (Pareto Frontier)
    latencies = [m["latency_ms"] for m in models_data]
    accuracies = [m["metrics"]["accuracy"] * 100.0 for m in models_data]

    for i in range(len(names)):
        axs[1, 1].scatter(latencies[i], accuracies[i], color=colors[i], s=250, edgecolor="#2B2D42", linewidth=1.5, zorder=5, label=names[i])
        axs[1, 1].annotate(
            f" {names[i]}\n ({latencies[i]:.2f} ms, {accuracies[i]:.2f}%)",
            xy=(latencies[i], accuracies[i]),
            xytext=(10, -5 if i != 1 else 10),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color=colors[i],
        )

    # Plot trendline connecting MobileNetV2 and EfficientNet-B0 Baseline
    axs[1, 1].plot([latencies[0], latencies[1]], [accuracies[0], accuracies[1]], linestyle="--", color="#1D3557", alpha=0.6, label="Accuracy Frontier")
    axs[1, 1].set_title("(D) Latency vs. Accuracy Operational Tradeoff", fontsize=12, fontweight="bold", pad=10)
    axs[1, 1].set_xlabel("CPU Inference Latency (ms/image)", fontsize=10, fontweight="semibold")
    axs[1, 1].set_ylabel("Test Accuracy (%)", fontsize=10, fontweight="semibold")
    axs[1, 1].set_xlim(min(latencies) - 2.5, max(latencies) + 5.5)
    axs[1, 1].set_ylim(min(accuracies) - 4.0, max(accuracies) + 4.0)
    axs[1, 1].legend(loc="lower left", fontsize=8.5, frameon=True)

    plt.suptitle("Solar Panel AI Fault Detection — Comprehensive Model Evaluation & Ablation Summary", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comprehensive summary panel to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 17 Final Model Comparison & Ablation Study")
    parser.add_argument("--device", type=str, default="cpu", help="Device for evaluation (cpu recommended)")
    args = parser.parse_args()

    root = get_project_root()
    config = load_config()

    classes = sorted(config.get("dataset", {}).get("classes", config.get("data", {}).get("classes", [])))
    num_classes = len(classes)
    device = torch.device(args.device)

    metrics_dir = root / "results" / "metrics"
    plots_dir = root / "results" / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    test_dir = root / "data" / "test"
    test_samples = discover_test_samples(test_dir, classes)
    if len(test_samples) != 177:
        logger.warning(f"Expected 177 test samples, discovered {len(test_samples)}")

    # Model specifications
    models_config = [
        {
            "id": "mobilenetv2_baseline",
            "display_name": "MobileNetV2 baseline",
            "architecture": "MobileNetV2",
            "checkpoint_path": root / "models" / "checkpoints" / "mobilenetv2_baseline_best.pth",
            "builder": lambda: build_mobilenet_v2(num_classes=num_classes, pretrained=False, dropout=0.2),
            "summary_fn": get_mobilenet_summary,
            "augmentation": "None",
            "canonical_latency_ms": 11.59,
            "canonical_fps": 86.3,
            "interpretation": "Establishes a lightweight baseline with fast CPU inference and low memory footprint.",
        },
        {
            "id": "efficientnet_b0_baseline",
            "display_name": "EfficientNet-B0 baseline",
            "architecture": "EfficientNet-B0",
            "checkpoint_path": root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth",
            "builder": lambda: build_efficientnet_b0(num_classes=num_classes, pretrained=False, dropout=0.2),
            "summary_fn": get_effnet_summary,
            "augmentation": "None",
            "canonical_latency_ms": 16.10,
            "canonical_fps": 62.1,
            "interpretation": "Provides stronger classification metrics across all fault types but requires more parameters and inference time.",
        },
        {
            "id": "efficientnet_b0_augmented",
            "display_name": "EfficientNet-B0 + domain-specific augmentation",
            "architecture": "EfficientNet-B0",
            "checkpoint_path": root / "models" / "checkpoints" / "efficientnet_b0_augmented_best.pth",
            "builder": lambda: build_efficientnet_b0(num_classes=num_classes, pretrained=False, dropout=0.2),
            "summary_fn": get_effnet_summary,
            "augmentation": "Domain-Specific",
            "canonical_latency_ms": 15.72,
            "canonical_fps": 63.6,
            "interpretation": "Domain-specific augmentation was tested as an improvement strategy but reduced performance on this dataset.",
        },
    ]

    evaluated_models: List[Dict[str, Any]] = []

    for cfg in models_config:
        ckpt_path = cfg["checkpoint_path"]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        logger.info(f"Loading {cfg['display_name']} checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model = cfg["builder"]()
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        summary = cfg["summary_fn"](model)
        eval_metrics = evaluate_model_on_test_set(model, test_samples, classes, device)

        evaluated_models.append({
            "id": cfg["id"],
            "display_name": cfg["display_name"],
            "architecture": cfg["architecture"],
            "augmentation": cfg["augmentation"],
            "checkpoint": str(ckpt_path.relative_to(root)),
            "parameters": summary["total_parameters"],
            "trainable_parameters": summary["trainable_parameters"],
            "model_size_mb": summary["model_size_mb"],
            "latency_ms": cfg["canonical_latency_ms"],
            "fps": cfg["canonical_fps"],
            "interpretation": cfg["interpretation"],
            "metrics": eval_metrics,
        })

    # 1. Generate Confusion Matrix plots
    cm_map = {
        "mobilenetv2_baseline": plots_dir / "mobilenetv2_confusion_matrix.png",
        "efficientnet_b0_baseline": plots_dir / "efficientnet_b0_confusion_matrix.png",
        "efficientnet_b0_augmented": plots_dir / "efficientnet_b0_augmented_confusion_matrix.png",
    }
    for m in evaluated_models:
        target_cm_path = cm_map[m["id"]]
        plot_confusion_matrix(
            m["metrics"]["y_true"],
            m["metrics"]["y_pred"],
            classes,
            target_cm_path,
            title=f"{m['display_name']} — Confusion Matrix",
        )

    # 2. Compute Differences
    mobilenet = evaluated_models[0]
    effnet_base = evaluated_models[1]
    effnet_aug = evaluated_models[2]

    diff_effnet_vs_mobile = {
        "comparison": "EfficientNet-B0 Baseline vs MobileNetV2 Baseline",
        "accuracy_diff_pct_pts": round((effnet_base["metrics"]["accuracy"] - mobilenet["metrics"]["accuracy"]) * 100.0, 2),
        "macro_precision_diff_pct_pts": round((effnet_base["metrics"]["macro_precision"] - mobilenet["metrics"]["macro_precision"]) * 100.0, 2),
        "macro_recall_diff_pct_pts": round((effnet_base["metrics"]["macro_recall"] - mobilenet["metrics"]["macro_recall"]) * 100.0, 2),
        "macro_f1_diff_pct_pts": round((effnet_base["metrics"]["macro_f1"] - mobilenet["metrics"]["macro_f1"]) * 100.0, 2),
        "weighted_f1_diff_pct_pts": round((effnet_base["metrics"]["weighted_f1"] - mobilenet["metrics"]["weighted_f1"]) * 100.0, 2),
        "parameter_diff": effnet_base["parameters"] - mobilenet["parameters"],
        "parameter_diff_pct": round((effnet_base["parameters"] - mobilenet["parameters"]) / mobilenet["parameters"] * 100.0, 2),
        "model_size_diff_mb": round(effnet_base["model_size_mb"] - mobilenet["model_size_mb"], 2),
        "latency_diff_ms": round(effnet_base["latency_ms"] - mobilenet["latency_ms"], 2),
    }

    diff_aug_vs_base = {
        "comparison": "EfficientNet-B0 Augmented vs EfficientNet-B0 Baseline",
        "accuracy_diff_pct_pts": round((effnet_aug["metrics"]["accuracy"] - effnet_base["metrics"]["accuracy"]) * 100.0, 2),
        "macro_precision_diff_pct_pts": round((effnet_aug["metrics"]["macro_precision"] - effnet_base["metrics"]["macro_precision"]) * 100.0, 2),
        "macro_recall_diff_pct_pts": round((effnet_aug["metrics"]["macro_recall"] - effnet_base["metrics"]["macro_recall"]) * 100.0, 2),
        "macro_f1_diff_pct_pts": round((effnet_aug["metrics"]["macro_f1"] - effnet_base["metrics"]["macro_f1"]) * 100.0, 2),
        "weighted_f1_diff_pct_pts": round((effnet_aug["metrics"]["weighted_f1"] - effnet_base["metrics"]["weighted_f1"]) * 100.0, 2),
        "parameter_diff": effnet_aug["parameters"] - effnet_base["parameters"],
        "parameter_diff_pct": 0.0,
        "model_size_diff_mb": round(effnet_aug["model_size_mb"] - effnet_base["model_size_mb"], 2),
        "latency_diff_ms": round(effnet_aug["latency_ms"] - effnet_base["latency_ms"], 2),
    }

    # 3. Build CSV Exports
    # Final Model Comparison Table CSV
    final_comparison_rows = []
    for m in evaluated_models:
        final_comparison_rows.append({
            "Model / Configuration": m["display_name"],
            "Accuracy": f"{m['metrics']['accuracy'] * 100.0:.2f}%",
            "Macro Precision": f"{m['metrics']['macro_precision'] * 100.0:.2f}%",
            "Macro Recall": f"{m['metrics']['macro_recall'] * 100.0:.2f}%",
            "Macro F1": f"{m['metrics']['macro_f1'] * 100.0:.2f}%",
            "Weighted F1": f"{m['metrics']['weighted_f1'] * 100.0:.2f}%",
            "Parameters": f"{m['parameters']:,}",
            "Model Size": f"{m['model_size_mb']:.2f} MB",
            "CPU Inference Time": f"{m['latency_ms']:.2f} ms",
            "Approximate FPS": f"{m['fps']:.1f}",
        })
    df_comparison = pd.DataFrame(final_comparison_rows)
    comparison_csv_path = metrics_dir / "final_model_comparison.csv"
    df_comparison.to_csv(comparison_csv_path, index=False)
    logger.info(f"Saved final comparison CSV to: {comparison_csv_path}")

    # Per-Class Comparison Table CSV
    per_class_rows = []
    for c in classes:
        support = effnet_base["metrics"]["per_class"][c]["support"]
        row = {
            "Class": c,
            "Support": support,
            "MobileNetV2 Precision": f"{mobilenet['metrics']['per_class'][c]['precision'] * 100.0:.2f}%",
            "MobileNetV2 Recall": f"{mobilenet['metrics']['per_class'][c]['recall'] * 100.0:.2f}%",
            "MobileNetV2 F1": f"{mobilenet['metrics']['per_class'][c]['f1_score'] * 100.0:.2f}%",
            "EfficientNet-B0 Baseline Precision": f"{effnet_base['metrics']['per_class'][c]['precision'] * 100.0:.2f}%",
            "EfficientNet-B0 Baseline Recall": f"{effnet_base['metrics']['per_class'][c]['recall'] * 100.0:.2f}%",
            "EfficientNet-B0 Baseline F1": f"{effnet_base['metrics']['per_class'][c]['f1_score'] * 100.0:.2f}%",
            "EfficientNet-B0 Augmented Precision": f"{effnet_aug['metrics']['per_class'][c]['precision'] * 100.0:.2f}%",
            "EfficientNet-B0 Augmented Recall": f"{effnet_aug['metrics']['per_class'][c]['recall'] * 100.0:.2f}%",
            "EfficientNet-B0 Augmented F1": f"{effnet_aug['metrics']['per_class'][c]['f1_score'] * 100.0:.2f}%",
        }
        per_class_rows.append(row)
    df_per_class = pd.DataFrame(per_class_rows)
    per_class_csv_path = metrics_dir / "per_class_model_comparison.csv"
    df_per_class.to_csv(per_class_csv_path, index=False)
    logger.info(f"Saved per-class comparison CSV to: {per_class_csv_path}")

    # Ablation Study Table CSV
    ablation_rows = []
    for m in evaluated_models:
        ablation_rows.append({
            "Experiment": m["display_name"],
            "Architecture": m["architecture"],
            "Augmentation": m["augmentation"],
            "Accuracy": f"{m['metrics']['accuracy'] * 100.0:.2f}%",
            "Macro F1": f"{m['metrics']['macro_f1'] * 100.0:.2f}%",
            "Interpretation": m["interpretation"],
        })
    df_ablation = pd.DataFrame(ablation_rows)
    ablation_csv_path = metrics_dir / "ablation_study.csv"
    df_ablation.to_csv(ablation_csv_path, index=False)
    logger.info(f"Saved ablation study CSV to: {ablation_csv_path}")

    # 4. Build and Save JSON Report
    json_clean_models = {}
    for m in evaluated_models:
        json_clean_models[m["id"]] = {
            "display_name": m["display_name"],
            "architecture": m["architecture"],
            "augmentation": m["augmentation"],
            "checkpoint": m["checkpoint"],
            "parameters": m["parameters"],
            "trainable_parameters": m["trainable_parameters"],
            "model_size_mb": m["model_size_mb"],
            "latency_ms": m["latency_ms"],
            "fps": m["fps"],
            "accuracy": m["metrics"]["accuracy"],
            "macro_precision": m["metrics"]["macro_precision"],
            "macro_recall": m["metrics"]["macro_recall"],
            "macro_f1": m["metrics"]["macro_f1"],
            "weighted_precision": m["metrics"]["weighted_precision"],
            "weighted_recall": m["metrics"]["weighted_recall"],
            "weighted_f1": m["metrics"]["weighted_f1"],
            "per_class": m["metrics"]["per_class"],
            "confusion_matrix": m["metrics"]["confusion_matrix"],
            "interpretation": m["interpretation"],
        }

    report_payload = {
        "metadata": {
            "title": "Phase 17 — Final Model Comparison and Ablation Study",
            "generated_at": datetime.datetime.now().isoformat(),
            "evaluation_pipeline": "Phase 7A Canonical Pipeline",
            "test_images_count": len(test_samples),
            "classes": classes,
            "hardware_device": str(device),
        },
        "models": json_clean_models,
        "pairwise_differences": {
            "mobilenetv2_vs_efficientnet_b0": diff_effnet_vs_mobile,
            "efficientnet_b0_baseline_vs_augmented": diff_aug_vs_base,
        },
        "ablation_study": [
            {
                "experiment": m["display_name"],
                "architecture": m["architecture"],
                "augmentation": m["augmentation"],
                "accuracy": m["metrics"]["accuracy"],
                "macro_f1": m["metrics"]["macro_f1"],
                "interpretation": m["interpretation"],
            }
            for m in evaluated_models
        ],
    }

    final_json_path = metrics_dir / "final_model_comparison.json"
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    logger.info(f"Saved machine-readable comparison JSON to: {final_json_path}")

    # 5. Generate Visualizations
    generate_accuracy_plot(evaluated_models, plots_dir / "model_accuracy_comparison.png")
    generate_macro_f1_plot(evaluated_models, plots_dir / "model_macro_f1_comparison.png")
    generate_weighted_f1_plot(evaluated_models, plots_dir / "model_weighted_f1_comparison.png")
    generate_inference_time_plot(evaluated_models, plots_dir / "model_inference_time_comparison.png")
    generate_parameter_count_plot(evaluated_models, plots_dir / "model_parameter_count_comparison.png")
    generate_per_class_f1_plot(evaluated_models, classes, plots_dir / "per_class_f1_comparison.png")
    generate_summary_panel(evaluated_models, classes, plots_dir / "final_model_comparison_summary.png")

    # 6. Print Terminal Report
    print("\n" + "=" * 90)
    print(" PHASE 17 — FINAL MODEL COMPARISON & ABLATION STUDY REPORT")
    print("=" * 90)
    print("\n1. FINAL MODEL COMPARISON TABLE:")
    print(df_comparison.to_string(index=False))

    print("\n2. ABLATION STUDY TABLE:")
    print(df_ablation.to_string(index=False))

    print("\n3. ABSOLUTE DIFFERENCES:")
    print(f" - MobileNetV2 vs EfficientNet-B0 Baseline:")
    print(f"   * Accuracy: {diff_effnet_vs_mobile['accuracy_diff_pct_pts']:+g} % pts")
    print(f"   * Macro F1: {diff_effnet_vs_mobile['macro_f1_diff_pct_pts']:+g} % pts")
    print(f"   * Parameters: {diff_effnet_vs_mobile['parameter_diff']:+,} ({diff_effnet_vs_mobile['parameter_diff_pct']:+g}%)")
    print(f"   * CPU Latency: {diff_effnet_vs_mobile['latency_diff_ms']:+g} ms")
    print(f" - EfficientNet-B0 Baseline vs EfficientNet-B0 Augmented:")
    print(f"   * Accuracy: {diff_aug_vs_base['accuracy_diff_pct_pts']:+g} % pts")
    print(f"   * Macro F1: {diff_aug_vs_base['macro_f1_diff_pct_pts']:+g} % pts")
    print(f"   * Parameters: {diff_aug_vs_base['parameter_diff']:,} (same architecture)")
    print(f"   * CPU Latency: {diff_aug_vs_base['latency_diff_ms']:+g} ms")

    print("\n4. PER-CLASS F1-SCORE BREAKDOWN:")
    for c in classes:
        print(f" - {c:<20}: MobileNet={mobilenet['metrics']['per_class'][c]['f1_score']*100:.1f}%, "
              f"EffNet-Base={effnet_base['metrics']['per_class'][c]['f1_score']*100:.1f}%, "
              f"EffNet-Aug={effnet_aug['metrics']['per_class'][c]['f1_score']*100:.1f}%")

    print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    main()
