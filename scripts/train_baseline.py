"""
CLI Script to train and evaluate the MobileNetV2 Baseline Model.
Usage:
    python scripts/train_baseline.py [--epochs 30] [--batch-size 32] [--lr 1e-4] [--device cpu]
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import torch

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.mobilenet import (
    build_mobilenet_v2,
    get_model_summary,
    benchmark_inference_latency,
)
from src.data.dataset import create_dataloaders
from src.training.trainer import BaselineTrainer
from src.evaluation.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_misclassifications,
)
from src.utils.config import load_config, get_project_root
from src.utils.logger import setup_logger

logger = setup_logger("train_baseline_cli")


def run_baseline_pipeline(
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
    patience: int = 8,
):
    root = get_project_root()
    config = load_config()

    classes = config.get("dataset", {}).get("classes", config.get("data", {}).get("classes", []))
    class_names = sorted(classes)
    num_classes = len(class_names)

    logger.info("=" * 65)
    logger.info(" PHASE 3: MOBILENETV2 EXPERIMENTAL BASELINE TRAINING")
    logger.info("=" * 65)
    logger.info(f"Target Classes ({num_classes}): {class_names}")
    logger.info(f"Hyperparameters: Batch Size={batch_size}, LR={learning_rate}, Max Epochs={epochs}, Seed={seed}")

    # 1. Dataloaders (Clean baseline: NO random augmentation)
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        config=config,
        batch_size=batch_size,
        num_workers=0,
        use_augmentation=False,  # Strictly deterministic baseline preprocessing
    )

    # 2. Build MobileNetV2 with ImageNet pretrained weights
    model = build_mobilenet_v2(num_classes=num_classes, pretrained=True, dropout=0.2)
    model_summary = get_model_summary(model)
    logger.info(f"Model Summary: {model_summary}")

    # 3. Benchmark Inference Latency on target device
    bench_dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    latency_info = benchmark_inference_latency(model, num_warmup=5, num_runs=30, device=bench_dev)
    logger.info(f"Inference Latency: {latency_info['average_latency_ms']} ms/image ({latency_info['fps']} FPS)")

    # 4. Trainer setup
    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        learning_rate=learning_rate,
        checkpoint_name="mobilenetv2_baseline_best.pth",
        early_stopping_patience=patience,
        device=bench_dev,
        seed=seed,
    )

    # 5. Fit model
    history_df = trainer.fit(max_epochs=epochs)

    # Save training history CSV
    metrics_dir = root / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_csv_path = metrics_dir / "mobilenetv2_baseline_history.csv"
    history_df.to_csv(history_csv_path, index=False)
    logger.info(f"Training history saved to: {history_csv_path}")

    # 6. Evaluate on Test set using BEST checkpoint
    logger.info("Evaluating BEST checkpoint on Test set...")
    y_true, y_pred, confidences, image_paths = trainer.evaluate_test_set()

    # 7. Compute complete metrics
    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    metrics["model_information"] = model_summary
    metrics["inference_benchmark"] = latency_info
    metrics["training_parameters"] = {
        "max_epochs": epochs,
        "actual_epochs_trained": len(history_df),
        "best_epoch": int(history_df.loc[history_df["is_best"]]["epoch"].max()) if history_df["is_best"].any() else 0,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "device": bench_dev,
        "augmentation_used": False,
    }

    # Save metrics JSON
    metrics_json_path = metrics_dir / "mobilenetv2_baseline_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Final metrics saved to: {metrics_json_path}")

    # 8. Visualizations
    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cm_path = plots_dir / "mobilenetv2_confusion_matrix.png"
    curves_path = plots_dir / "mobilenetv2_training_curves.png"

    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    plot_training_curves(history_df, curves_path, model_name="MobileNetV2 Baseline")

    # 9. Misclassification Analysis
    preds_dir = root / "results" / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    misclass_csv_path = preds_dir / "mobilenetv2_misclassifications.csv"
    misclass_df = save_misclassifications(y_true, y_pred, confidences, image_paths, class_names, misclass_csv_path)

    # 10. Print Final Formatted Report
    print("\n" + "=" * 70)
    print(" MOBILENETV2 BASELINE EVALUATION REPORT")
    print("=" * 70)
    print(f"Device Used            : {bench_dev}")
    print(f"Total Parameters       : {model_summary['total_parameters']:,}")
    print(f"Trainable Parameters   : {model_summary['trainable_parameters']:,}")
    print(f"Model Size             : {model_summary['model_size_mb']} MB")
    print(f"Avg Inference Latency  : {latency_info['average_latency_ms']} ms/image ({latency_info['fps']} FPS)")
    print(f"Epochs Trained         : {len(history_df)} / {epochs} (Best Epoch: {metrics['training_parameters']['best_epoch']})")
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
    print(f"Saved Checkpoint       : {trainer.checkpoint_path}")
    print(f"Saved Metrics JSON     : {metrics_json_path}")
    print(f"Saved Confusion Matrix : {cm_path}")
    print(f"Saved Training Curves  : {curves_path}")
    print(f"Saved Errors CSV       : {misclass_csv_path}")
    print("=" * 70 + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 baseline.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, or auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    args = parser.parse_args()

    run_baseline_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
