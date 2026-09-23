"""
CLI Script to evaluate the best SparkNet checkpoint on the independent test set.
Usage:
    python scripts/evaluate_sparkneta.py [--checkpoint PATH]
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.sparkneta import (
    build_sparknet,
    get_model_summary,
    benchmark_inference_latency,
)
from src.data.dataset import create_dataloaders
from src.evaluation.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_misclassifications,
)
from src.utils.config import load_config, get_project_root
from src.utils.logger import setup_logger

logger = setup_logger("evaluate_sparkneta_cli")


def evaluate_checkpoint(checkpoint_path: str | Path | None = None):
    root = get_project_root()
    config = load_config()

    if checkpoint_path is None:
        checkpoint_file = root / "models" / "checkpoints" / "sparkneta_baseline_best.pth"
    else:
        checkpoint_file = Path(checkpoint_path)

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_file}")

    classes = config.get("dataset", {}).get("classes", config.get("data", {}).get("classes", []))
    class_names = sorted(classes)
    num_classes = len(class_names)

    device = torch.device("cpu")
    logger.info(f"Loading checkpoint from: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)

    best_epoch = checkpoint.get("epoch", "N/A")
    val_loss = checkpoint.get("val_loss", 0.0)
    val_acc = checkpoint.get("val_acc", 0.0)
    val_f1 = checkpoint.get("val_macro_f1", 0.0)

    model = build_sparknet(num_classes=num_classes, dropout=0.3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    model_summary = get_model_summary(model)
    latency_info = benchmark_inference_latency(model, num_warmup=10, num_runs=50, device="cpu")

    # Load test set (177 images, deterministic preprocessing)
    _, _, test_loader, _ = create_dataloaders(
        config=config,
        batch_size=32,
        num_workers=0,
        use_augmentation=False,
    )

    y_true = []
    y_pred = []
    confidences = []
    image_paths = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = softmax(outputs)
            confs, preds = torch.max(probs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            confidences.extend(confs.cpu().numpy().tolist())
            image_paths.extend(paths)

    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    metrics["model_information"] = model_summary
    metrics["inference_benchmark"] = latency_info
    metrics["best_validation_result"] = {
        "best_epoch": best_epoch,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
    }

    # Save metrics JSON
    metrics_dir = root / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_json_path = metrics_dir / "sparkneta_baseline_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Evaluation metrics saved to: {metrics_json_path}")

    # Visualizations
    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cm_path = plots_dir / "sparkneta_confusion_matrix.png"

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        cm_path,
        title="SparkNet Baseline — Confusion Matrix (Test Set)",
    )

    # Check if history CSV exists to update curves
    history_csv_path = metrics_dir / "sparkneta_baseline_history.csv"
    if history_csv_path.exists():
        history_df = pd.read_csv(history_csv_path)
        curves_path = plots_dir / "sparkneta_training_curves.png"
        plot_training_curves(history_df, curves_path, model_name="SparkNet Baseline")

    # Misclassifications
    preds_dir = root / "results" / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    misclass_csv_path = preds_dir / "sparkneta_misclassifications.csv"
    misclass_df = save_misclassifications(y_true, y_pred, confidences, image_paths, class_names, misclass_csv_path)

    # Print Formatted Report
    print("\n" + "=" * 70)
    print(" SPARKNET BASELINE EVALUATION REPORT")
    print("=" * 70)
    print(f"Evaluated Checkpoint   : {checkpoint_file}")
    print(f"Checkpoint Best Epoch  : {best_epoch}")
    print(f"Val Loss (Best Epoch)  : {val_loss:.4f}")
    print(f"Val Acc  (Best Epoch)  : {val_acc*100:.2f}%")
    print(f"Val F1   (Best Epoch)  : {val_f1*100:.2f}%")
    print("-" * 70)
    print(f"Total Parameters       : {model_summary['total_parameters']:,}")
    print(f"Trainable Parameters   : {model_summary['trainable_parameters']:,}")
    print(f"Model Size             : {model_summary['model_size_mb']} MB")
    print(f"Avg CPU Latency        : {latency_info['average_latency_ms']} ms/image ({latency_info['fps']} FPS)")
    print("-" * 70)
    print(f"Test Accuracy          : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision        : {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall           : {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1 Score         : {metrics['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1 Score      : {metrics['weighted_f1'] * 100:.2f}%")
    print("-" * 70)
    print("Per-Class Metrics:")
    for cls_name, vals in metrics["per_class"].items():
        print(f"  - {cls_name:<20}: Precision={vals['precision']*100:>5.1f}%, Recall={vals['recall']*100:>5.1f}%, F1={vals['f1_score']*100:>5.1f}% (Support: {vals['support']})")
    print("-" * 70)
    print(f"Total Misclassifications: {len(misclass_df)} / {len(y_true)} ({len(misclass_df)/len(y_true)*100:.1f}%)")
    print(f"Saved Metrics JSON     : {metrics_json_path}")
    print(f"Saved Confusion Matrix : {cm_path}")
    print(f"Saved Errors CSV       : {misclass_csv_path}")
    print("=" * 70 + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SparkNet baseline checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
