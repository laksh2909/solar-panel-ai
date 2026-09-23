"""
Evaluation Metrics, Visualizations, and Misclassification Auditing Module.

Computes:
- Overall Accuracy
- Macro and Weighted Precision, Recall, F1
- Per-class Precision, Recall, F1 breakdown
- Confusion Matrix visualization (results/plots/mobilenetv2_confusion_matrix.png)
- Training & Validation loss/accuracy curves (results/plots/mobilenetv2_training_curves.png)
- Misclassification analysis table (results/predictions/mobilenetv2_misclassifications.csv)
- Metrics export (results/metrics/mobilenetv2_baseline_metrics.json)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.utils.logger import setup_logger
from src.utils.config import get_project_root

logger = setup_logger("evaluation_metrics")


def compute_classification_metrics(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics across all classes.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    weighted_p = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_r = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class breakdowns
    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class_dict = {}
    for idx, name in enumerate(class_names):
        count_true = int(np.sum(y_true == idx))
        count_pred = int(np.sum(y_pred == idx))
        per_class_dict[name] = {
            "precision": round(float(per_class_p[idx]), 4),
            "recall": round(float(per_class_r[idx]), 4),
            "f1_score": round(float(per_class_f1[idx]), 4),
            "support": count_true,
            "predictions_count": count_pred,
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    results = {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_p), 4),
        "weighted_recall": round(float(weighted_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "total_test_samples": len(y_true),
    }

    return results


def plot_confusion_matrix(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
    class_names: List[str],
    output_path: Path | str,
    title: str = "MobileNetV2 Baseline — Confusion Matrix",
) -> Path:
    """
    Renders and saves a high-resolution confusion matrix heatmap.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    # Also calculate row-normalized percentages
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100.0
            annot_matrix[i, j] = f"{count}\n({pct:.1f}%)"

    plt.figure(figsize=(9, 7.5), dpi=300)
    sns.set_theme(style="white")

    sns.heatmap(
        cm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=1.0,
        linecolor="#E0E0E0",
        annot_kws={"fontsize": 9.5, "fontweight": "semibold"}
    )

    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="semibold", labelpad=10)
    plt.ylabel("True Class", fontsize=11, fontweight="semibold", labelpad=10)
    plt.xticks(rotation=20, ha="right", fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

    logger.info(f"Confusion matrix plot saved to: {out_file}")
    return out_file


def plot_training_curves(
    history_df: pd.DataFrame,
    output_path: Path | str,
    model_name: str = "MobileNetV2 Baseline",
) -> Path:
    """
    Plots training & validation loss and accuracy trajectories across epochs.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    epochs = history_df["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    sns.set_theme(style="whitegrid")

    # Panel 1: Loss curves
    axes[0].plot(epochs, history_df["train_loss"], "o-", color="#2A9D8F", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history_df["val_loss"], "s--", color="#E76F51", label="Val Loss", linewidth=2)
    best_val_loss_epoch = history_df.loc[history_df["val_loss"].idxmin(), "epoch"]
    min_loss = history_df["val_loss"].min()
    axes[0].axvline(best_val_loss_epoch, color="#E63946", linestyle=":", alpha=0.8, label=f"Best Loss (Ep {best_val_loss_epoch})")
    axes[0].set_title(f"{model_name} — Loss Trajectory", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10, fontweight="semibold")
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=10, fontweight="semibold")
    axes[0].legend(fontsize=9)

    # Panel 2: Accuracy curves
    axes[1].plot(epochs, history_df["train_acc"] * 100.0, "o-", color="#1D3557", label="Train Acc (%)", linewidth=2)
    axes[1].plot(epochs, history_df["val_acc"] * 100.0, "s--", color="#F4A261", label="Val Acc (%)", linewidth=2)
    if "val_macro_f1" in history_df.columns:
        axes[1].plot(epochs, history_df["val_macro_f1"] * 100.0, "^-.", color="#9D4EDD", label="Val Macro F1 (%)", linewidth=1.8)
    best_acc_epoch = history_df.loc[history_df["val_acc"].idxmax(), "epoch"]
    max_acc = history_df["val_acc"].max() * 100.0
    axes[1].axvline(best_acc_epoch, color="#E63946", linestyle=":", alpha=0.8, label=f"Best Acc {max_acc:.1f}% (Ep {best_acc_epoch})")
    axes[1].set_title(f"{model_name} — Accuracy & F1 Trajectory", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10, fontweight="semibold")
    axes[1].set_ylabel("Percentage (%)", fontsize=10, fontweight="semibold")
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=9)

    plt.suptitle(f"{model_name} Training Dynamics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Training curves plot saved to: {out_file}")
    return out_file


def save_misclassifications(
    y_true: List[int],
    y_pred: List[int],
    confidences: List[float],
    image_paths: List[str],
    class_names: List[str],
    output_path: Path | str,
) -> pd.DataFrame:
    """
    Identifies misclassified images and saves detailed audit table.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for true_idx, pred_idx, conf, path in zip(y_true, y_pred, confidences, image_paths):
        if true_idx != pred_idx:
            records.append({
                "filename": Path(path).name,
                "actual_class": class_names[true_idx],
                "predicted_class": class_names[pred_idx],
                "confidence": round(float(conf), 4),
                "image_path": str(path),
            })

    misclassified_df = pd.DataFrame(records)
    if not misclassified_df.empty:
        # Sort by confidence descending (high-confidence errors are most insightful)
        misclassified_df = misclassified_df.sort_values(by="confidence", ascending=False).reset_index(drop=True)

    misclassified_df.to_csv(out_file, index=False)
    logger.info(f"Misclassification report saved to: {out_file} (Total errors: {len(misclassified_df)})")
    return misclassified_df
