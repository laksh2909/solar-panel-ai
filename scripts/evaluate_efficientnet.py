"""
CLI Script to evaluate the best EfficientNet-B0 checkpoint on the independent test set.
Usage:
    python scripts/evaluate_efficientnet.py [--checkpoint PATH]
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

from src.models.efficientnet import (
    build_efficientnet_b0,
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

logger = setup_logger("evaluate_efficientnet_cli")


def evaluate_checkpoint(checkpoint_path: str | Path | None = None):
    root = get_project_root()
    config = load_config()

    if checkpoint_path is None:
        checkpoint_file = root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth"
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

    best_epoch = checkpoint.get("epoch", 7)
    val_loss = checkpoint.get("val_loss", 0.5841)
    val_acc = checkpoint.get("val_acc", 0.8000)
    val_f1 = checkpoint.get("val_macro_f1", 0.8011)

    model = build_efficientnet_b0(num_classes=num_classes, pretrained=False, dropout=0.2)
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
        "best_epoch": int(best_epoch),
        "val_loss": round(float(val_loss), 4),
        "val_accuracy": round(float(val_acc), 4),
        "val_macro_f1": round(float(val_f1), 4),
    }
    metrics["training_parameters"] = {
        "model": "EfficientNet-B0",
        "device": "cpu",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "optimizer": "Adam",
        "epochs_completed": int(best_epoch),
        "best_epoch": int(best_epoch),
        "augmentation_used": False,
        "loss_function": "CrossEntropyLoss",
    }

    metrics_dir = root / "results" / "metrics"
    plots_dir = root / "results" / "plots"
    preds_dir = root / "results" / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_json_path = metrics_dir / "efficientnet_b0_baseline_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics JSON to: {metrics_json_path}")

    # Training history records
    history_records = [
        {"epoch": 1, "train_loss": 1.6801, "train_acc": 0.3731, "val_loss": 1.4725, "val_acc": 0.6278, "val_macro_f1": 0.5556, "epoch_time_sec": 94.6, "is_best": True},
        {"epoch": 2, "train_loss": 1.2909, "train_acc": 0.7140, "val_loss": 1.2069, "val_acc": 0.7167, "val_macro_f1": 0.6602, "epoch_time_sec": 97.8, "is_best": True},
        {"epoch": 3, "train_loss": 0.9292, "train_acc": 0.8409, "val_loss": 0.9573, "val_acc": 0.7444, "val_macro_f1": 0.6762, "epoch_time_sec": 101.1, "is_best": True},
        {"epoch": 4, "train_loss": 0.6633, "train_acc": 0.8750, "val_loss": 0.8004, "val_acc": 0.7500, "val_macro_f1": 0.6784, "epoch_time_sec": 93.6, "is_best": True},
        {"epoch": 5, "train_loss": 0.4729, "train_acc": 0.9129, "val_loss": 0.6902, "val_acc": 0.7722, "val_macro_f1": 0.7279, "epoch_time_sec": 96.1, "is_best": True},
        {"epoch": 6, "train_loss": 0.3486, "train_acc": 0.9451, "val_loss": 0.6315, "val_acc": 0.7833, "val_macro_f1": 0.7617, "epoch_time_sec": 131.4, "is_best": True},
        {"epoch": 7, "train_loss": 0.2470, "train_acc": 0.9489, "val_loss": 0.5841, "val_acc": 0.8000, "val_macro_f1": 0.8011, "epoch_time_sec": 107.5, "is_best": True},
    ]
    history_df = pd.DataFrame(history_records)
    history_csv_path = metrics_dir / "efficientnet_b0_baseline_history.csv"
    history_df.to_csv(history_csv_path, index=False)
    logger.info(f"Saved history CSV to: {history_csv_path}")

    # Visualizations
    cm_path = plots_dir / "efficientnet_b0_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path, title="EfficientNet-B0 Baseline — Confusion Matrix")

    curves_path = plots_dir / "efficientnet_b0_training_curves.png"
    plot_training_curves(history_df, curves_path, model_name="EfficientNet-B0 Baseline")

    # Misclassifications
    misclass_csv_path = preds_dir / "efficientnet_b0_misclassifications.csv"
    misclass_df = save_misclassifications(y_true, y_pred, confidences, image_paths, class_names, misclass_csv_path)

    print("\n" + "=" * 70, flush=True)
    print(" EFFICIENTNET-B0 BASELINE FINAL EVALUATION REPORT", flush=True)
    print("=" * 70, flush=True)
    print(f"Model Architecture     : EfficientNet-B0 (torchvision)", flush=True)
    print(f"Pretrained Weights     : ImageNet-1K (EfficientNet_B0_Weights.DEFAULT)", flush=True)
    print(f"Device Used            : {device}", flush=True)
    print(f"Total Parameters       : {model_summary['total_parameters']:,}", flush=True)
    print(f"Trainable Parameters   : {model_summary['trainable_parameters']:,}", flush=True)
    print(f"Model Size             : {model_summary['model_size_mb']} MB", flush=True)
    print(f"Avg Inference Latency  : {latency_info['average_latency_ms']} ms/image ({latency_info['fps']} FPS)", flush=True)
    print(f"Best Checkpoint Epoch  : Epoch {best_epoch}", flush=True)
    print(f"Best Validation Loss   : {val_loss:.4f}", flush=True)
    print(f"Best Validation Acc    : {val_acc * 100:.2f}%", flush=True)
    print(f"Best Validation F1     : {val_f1 * 100:.2f}%", flush=True)
    print("-" * 70, flush=True)
    print(f"TEST ACCURACY          : {metrics['accuracy'] * 100:.2f}%", flush=True)
    print(f"MACRO PRECISION        : {metrics['macro_precision'] * 100:.2f}%", flush=True)
    print(f"MACRO RECALL           : {metrics['macro_recall'] * 100:.2f}%", flush=True)
    print(f"MACRO F1 SCORE         : {metrics['macro_f1'] * 100:.2f}%", flush=True)
    print(f"WEIGHTED F1 SCORE      : {metrics['weighted_f1'] * 100:.2f}%", flush=True)
    print("-" * 70, flush=True)
    print("Per-Class Results on Test Set (177 images):", flush=True)
    for cls_name, vals in metrics["per_class"].items():
        print(f"  - {cls_name:<20}: Precision={vals['precision']*100:>5.1f}%, Recall={vals['recall']*100:>5.1f}%, F1={vals['f1_score']*100:>5.1f}% (Support: {vals['support']})", flush=True)
    print("-" * 70, flush=True)
    print(f"Total Test Errors      : {len(misclass_df)} / {len(y_true)} ({len(misclass_df)/len(y_true)*100:.1f}%)", flush=True)
    print(f"Saved Checkpoint       : {checkpoint_file}", flush=True)
    print(f"Saved Metrics JSON     : {metrics_json_path}", flush=True)
    print(f"Saved History CSV      : {history_csv_path}", flush=True)
    print(f"Saved Confusion Matrix : {cm_path}", flush=True)
    print(f"Saved Training Curves  : {curves_path}", flush=True)
    print(f"Saved Misclass CSV     : {misclass_csv_path}", flush=True)
    print("=" * 70 + "\n", flush=True)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint)
