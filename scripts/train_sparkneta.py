"""
CLI Script to train and evaluate the SparkNet Baseline Model.

Features:
- Optimized CPU Threading (4 threads) to eliminate thermal throttling and openMP contention
- Pre-flight CPU Safety Benchmark (Model init, forward-pass, single-batch forward/backward, cost estimation)
- Visible batch progress logging (every 4 batches and epoch completion)
- Hierarchical 4-branch CNN with Squeeze-and-Expand Fire Modules
- Deterministic seed (42), Adam optimizer (lr=1e-4), CrossEntropyLoss
- Early stopping monitoring validation Macro F1 (patience=8, max epochs=30)
- Best checkpoint saved to models/checkpoints/sparkneta_baseline_best.pth
- Full test set evaluation on 177 independent images and artifact generation
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
from src.training.trainer import set_seed
from src.evaluation.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_misclassifications,
)
from src.utils.config import load_config, get_project_root
from src.utils.logger import setup_logger

logger = setup_logger("train_sparkneta_cli")


class SparkNetTrainer:
    """
    Dedicated SparkNet Trainer with visible batch progress logging
    and thermal-safe CPU execution.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        class_names: List[str],
        learning_rate: float = 1e-4,
        checkpoint_name: str = "sparkneta_baseline_best.pth",
        early_stopping_patience: int = 8,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.seed = seed
        set_seed(self.seed)

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.num_classes = len(class_names)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        root = get_project_root()
        self.checkpoint_dir = root / "models" / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / checkpoint_name

        self.early_stopping_patience = early_stopping_patience
        self.history: List[Dict[str, Any]] = []

    def train_epoch(self, epoch: int, max_epochs: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        total_batches = len(self.train_loader)

        for b_idx, (images, labels, _) in enumerate(self.train_loader, 1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += images.size(0)

            # Visible batch progress logging
            if b_idx % 4 == 0 or b_idx == total_batches:
                running_loss = total_loss / total_samples
                running_acc = (correct / total_samples) * 100.0
                logger.info(
                    f"Epoch [{epoch:02d}/{max_epochs:02d}] Batch [{b_idx:02d}/{total_batches:02d}] "
                    f"- Running Loss: {running_loss:.4f} | Running Acc: {running_acc:.1f}%"
                )

        epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
        epoch_acc = correct / total_samples if total_samples > 0 else 0.0
        return epoch_loss, epoch_acc

    def validate_epoch(self) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total_samples += images.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        loss = total_loss / total_samples if total_samples > 0 else 0.0
        acc = correct / total_samples if total_samples > 0 else 0.0
        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        return loss, acc, float(macro_f1)

    def fit(self, max_epochs: int = 30) -> pd.DataFrame:
        best_val_score = -1.0
        best_epoch = 0
        patience_counter = 0

        logger.info(f"Starting SparkNet training run for up to {max_epochs} epochs (Early Stopping Patience={self.early_stopping_patience})...")
        total_start = time.time()

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch(epoch, max_epochs)
            val_loss, val_acc, val_f1 = self.validate_epoch()
            epoch_time = time.time() - t0

            is_best = val_f1 > best_val_score
            if is_best:
                best_val_score = val_f1
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_macro_f1": val_f1,
                    "class_names": self.class_names,
                }, self.checkpoint_path)
            else:
                patience_counter += 1

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "val_macro_f1": round(val_f1, 4),
                "is_best": is_best,
                "epoch_time_sec": round(epoch_time, 2),
            }
            self.history.append(record)

            best_marker = " [BEST SAVED]" if is_best else ""
            logger.info(
                f"=== Epoch [{epoch:02d}/{max_epochs:02d}] Summary === "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | "
                f"Val Macro F1: {val_f1*100:.1f}%{best_marker} (Time: {epoch_time:.1f}s)"
            )

            # Early stopping check
            if patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch} (no validation improvement for {patience_counter} epochs). "
                    f"Best validation was at epoch {best_epoch} with Val Macro F1: {best_val_score*100:.2f}%."
                )
                break

        total_elapsed = time.time() - total_start
        logger.info(f"Training completed in {total_elapsed/60.0:.2f} minutes. Best Epoch: {best_epoch} (Val Macro F1: {best_val_score*100:.2f}%).")
        return pd.DataFrame(self.history)

    def evaluate_test_set(self) -> Tuple[List[int], List[int], List[float], List[str]]:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found at: {self.checkpoint_path}")

        logger.info(f"Loading best checkpoint for test evaluation: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        y_true: List[int] = []
        y_pred: List[int] = []
        confidences: List[float] = []
        image_paths: List[str] = []
        softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            for images, labels, paths in self.test_loader:
                images = images.to(self.device, non_blocking=True)
                outputs = self.model(images)
                probs = softmax(outputs)
                confs, preds = torch.max(probs, dim=1)

                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())
                confidences.extend(confs.cpu().numpy().tolist())
                image_paths.extend(paths)

        return y_true, y_pred, confidences, image_paths


def run_cpu_safety_benchmark(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 32,
    num_train_samples: int = 528,
    num_val_samples: int = 180,
) -> Dict[str, Any]:
    """
    Executes mandatory CPU safety benchmark before full training.
    """
    logger.info("=" * 65)
    logger.info(" RUNNING CPU PRE-FLIGHT SAFETY BENCHMARK")
    logger.info("=" * 65)

    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    dummy_target = torch.randint(0, 6, (batch_size,), device=device)

    # 1. Forward-pass test
    t0_fwd = time.perf_counter()
    with torch.no_grad():
        out = model(dummy_input)
    t1_fwd = time.perf_counter()
    fwd_latency_ms = (t1_fwd - t0_fwd) * 1000.0
    assert out.shape == torch.Size([batch_size, 6]), f"Expected shape [32, 6], got {out.shape}"
    logger.info(f" [PASS] Forward pass test: Output shape {list(out.shape)} ({fwd_latency_ms:.2f} ms)")

    # 2. Single-batch forward + backward benchmark
    optimizer.zero_grad()
    t0_step = time.perf_counter()
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()
    optimizer.step()
    t1_step = time.perf_counter()
    step_latency_sec = t1_step - t0_step

    batches_per_epoch = (num_train_samples + batch_size - 1) // batch_size
    val_batches = (num_val_samples + batch_size - 1) // batch_size
    est_train_epoch_sec = step_latency_sec * batches_per_epoch
    est_val_epoch_sec = (fwd_latency_ms / 1000.0) * val_batches
    est_epoch_sec = est_train_epoch_sec + est_val_epoch_sec

    summary = {
        "forward_only_latency_ms": round(fwd_latency_ms, 2),
        "single_batch_train_step_sec": round(step_latency_sec, 3),
        "batches_per_epoch": batches_per_epoch,
        "est_train_epoch_sec": round(est_train_epoch_sec, 2),
        "est_epoch_total_sec": round(est_epoch_sec, 2),
        "est_10_epochs_min": round((est_epoch_sec * 10) / 60.0, 2),
        "est_30_epochs_min": round((est_epoch_sec * 30) / 60.0, 2),
    }

    logger.info(f" [PASS] Single batch forward/backward step: {step_latency_sec:.3f} s")
    logger.info(f" Estimated epoch duration: {est_epoch_sec:.1f} s (~{summary['est_10_epochs_min']} min for 10 epochs, ~{summary['est_30_epochs_min']} min for 30 epochs)")
    logger.info(" CPU safety benchmark complete.\n")

    return summary


def run_sparknet_pipeline(
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device_arg: str = "cpu",
    seed: int = 42,
    patience: int = 8,
    benchmark_only: bool = False,
):
    # Set PyTorch threads to 4 for CPU thermal safety and optimal performance
    torch.set_num_threads(4)

    root = get_project_root()
    config = load_config()

    classes = config.get("dataset", {}).get("classes", config.get("data", {}).get("classes", []))
    class_names = sorted(classes)
    num_classes = len(class_names)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and device_arg != "cpu" else "cpu")

    logger.info("=" * 65)
    logger.info(" PHASE 5: SPARKNET BASELINE REPRODUCTION")
    logger.info("=" * 65)
    logger.info(f"Architecture: SparkNet (4-branch Hierarchical Squeeze-and-Expand CNN)")
    logger.info(f"Target Classes ({num_classes}): {class_names}")
    logger.info(f"Hyperparameters: Batch Size={batch_size}, LR={learning_rate}, Max Epochs={epochs}, Patience={patience}, Seed={seed}")
    logger.info(f"Execution Device: {device} (PyTorch Threads: {torch.get_num_threads()})")

    # 1. Dataloaders (Clean baseline: NO random augmentation, strictly deterministic)
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        config=config,
        batch_size=batch_size,
        num_workers=0,
        use_augmentation=False,
    )

    # 2. Build SparkNet model
    model = build_sparknet(num_classes=num_classes, dropout=0.3)
    model_summary = get_model_summary(model)
    logger.info(f"Model Summary: {model_summary}")

    # 3. CPU Safety Pre-Flight Benchmark
    bench_results = run_cpu_safety_benchmark(
        model=model,
        device=device,
        batch_size=batch_size,
        num_train_samples=len(train_loader.dataset),
        num_val_samples=len(val_loader.dataset),
    )

    if benchmark_only:
        logger.info("Benchmark-only flag set. Halting before training.")
        return bench_results

    # 4. Measure CPU inference latency & throughput
    latency_info = benchmark_inference_latency(model, num_warmup=10, num_runs=50, device="cpu")
    logger.info(f"CPU Inference Latency: {latency_info['average_latency_ms']} ms/image ({latency_info['fps']} FPS)")

    # 5. Trainer setup
    checkpoint_name = "sparkneta_baseline_best.pth"
    trainer = SparkNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        learning_rate=learning_rate,
        checkpoint_name=checkpoint_name,
        early_stopping_patience=patience,
        device=str(device),
        seed=seed,
    )

    # 6. Fit SparkNet model
    logger.info("Launching SparkNet training loop with visible batch progress...")
    history_df = trainer.fit(max_epochs=epochs)

    # Save training history CSV
    metrics_dir = root / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_csv_path = metrics_dir / "sparkneta_baseline_history.csv"
    history_df.to_csv(history_csv_path, index=False)
    logger.info(f"Training history saved to: {history_csv_path}")

    # 7. Evaluate on independent Test set (177 images) using BEST checkpoint
    logger.info("Evaluating BEST SparkNet checkpoint on 177-image Test set...")
    y_true, y_pred, confidences, image_paths = trainer.evaluate_test_set()

    # 8. Compute complete metrics
    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    metrics["model_information"] = model_summary
    metrics["inference_benchmark"] = latency_info
    metrics["safety_benchmark"] = bench_results
    metrics["training_parameters"] = {
        "max_epochs": epochs,
        "actual_epochs_trained": len(history_df),
        "best_epoch": int(history_df.loc[history_df["is_best"]]["epoch"].max()) if history_df["is_best"].any() else 0,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "early_stopping_patience": patience,
        "device": str(device),
        "augmentation_used": False,
    }

    metrics_json_path = metrics_dir / "sparkneta_baseline_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Final metrics saved to: {metrics_json_path}")

    # 9. Visualizations
    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cm_path = plots_dir / "sparkneta_confusion_matrix.png"
    curves_path = plots_dir / "sparkneta_training_curves.png"

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        cm_path,
        title="SparkNet Baseline — Confusion Matrix",
    )
    plot_training_curves(
        history_df,
        curves_path,
        model_name="SparkNet Baseline",
    )

    # 10. Misclassification Analysis
    preds_dir = root / "results" / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    misclass_csv_path = preds_dir / "sparkneta_misclassifications.csv"
    misclass_df = save_misclassifications(y_true, y_pred, confidences, image_paths, class_names, misclass_csv_path)

    # 11. Final Formatted Report
    print("\n" + "=" * 70)
    print(" SPARKNET BASELINE EVALUATION REPORT")
    print("=" * 70)
    print(f"Architecture           : SparkNet (4 Hierarchical FireModule Branches)")
    print(f"Device Used            : {device}")
    print(f"Total Parameters       : {model_summary['total_parameters']:,}")
    print(f"Trainable Parameters   : {model_summary['trainable_parameters']:,}")
    print(f"Model Size             : {model_summary['model_size_mb']} MB")
    print(f"Avg CPU Latency        : {latency_info['average_latency_ms']} ms/image ({latency_info['fps']} FPS)")
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
    parser = argparse.ArgumentParser(description="Train SparkNet baseline reproduction.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, or auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--benchmark-only", action="store_true", help="Run only CPU safety checks")
    args = parser.parse_args()

    run_sparknet_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device_arg=args.device,
        seed=args.seed,
        patience=args.patience,
        benchmark_only=args.benchmark_only,
    )


if __name__ == "__main__":
    main()
