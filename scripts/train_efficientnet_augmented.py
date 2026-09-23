"""
CLI Script to train and evaluate the Improved EfficientNet-B0 Model with Domain-Specific Augmentation.

Phase 5: Improved EfficientNet-B0 with Domain-Specific Augmentation
- Albumentations training augmentation pipeline tailored for solar panels
- PyTorch CPU thermal safety optimization (4 threads)
- Deterministic validation and testing pipelines
- Pre-flight CPU safety benchmark and augmentation sanity check
- Tracking metrics: Loss, Accuracy, Macro F1 across all epochs
- Early stopping with patience=8 monitoring validation Macro F1
- Checkpoint saved to: models/checkpoints/efficientnet_b0_augmented_best.pth
- Full test evaluation on 177 independent images
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
from src.preprocessing.augmentation import (
    get_training_augmentation,
    get_visualization_augmentation,
    get_validation_pipeline,
)
from src.training.trainer import set_seed
from src.evaluation.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_misclassifications,
)
from src.utils.config import load_config, get_project_root
from src.utils.logger import setup_logger

logger = setup_logger("train_efficientnet_augmented_cli")


def run_cpu_safety_benchmark(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Executes mandatory CPU safety benchmark before starting training:
    1. Preprocessing / Augmentation sanity test
    2. Forward pass test on a real batch
    3. Single batch forward + backward training step
    4. Cost and runtime estimation
    """
    logger.info("=" * 65)
    logger.info(" RUNNING CPU PRE-FLIGHT SAFETY BENCHMARK")
    logger.info("=" * 65)

    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 1. Fetch one real augmented batch
    for images, labels, _ in train_loader:
        batch_images = images.to(device)
        batch_labels = labels.to(device)
        break

    batch_size = batch_images.size(0)
    logger.info(f" [PASS] Augmentation sanity check: Loaded batch of {batch_size} images. Tensor shape: {list(batch_images.shape)}")
    assert not torch.isnan(batch_images).any(), "NaN found in augmented images!"

    # 2. Forward pass benchmark
    t0_fwd = time.perf_counter()
    with torch.no_grad():
        out = model(batch_images)
    t1_fwd = time.perf_counter()
    fwd_latency_ms = (t1_fwd - t0_fwd) * 1000.0
    logger.info(f" [PASS] Forward pass test: Output shape {list(out.shape)} ({fwd_latency_ms:.2f} ms for batch of {batch_size})")

    # 3. Single-batch training step
    optimizer.zero_grad()
    t0_step = time.perf_counter()
    output = model(batch_images)
    loss = criterion(output, batch_labels)
    loss.backward()
    optimizer.step()
    t1_step = time.perf_counter()
    step_latency_sec = t1_step - t0_step

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    est_train_epoch_sec = step_latency_sec * num_train_batches
    est_val_epoch_sec = (fwd_latency_ms / 1000.0) * num_val_batches
    est_epoch_sec = est_train_epoch_sec + est_val_epoch_sec

    summary = {
        "batch_size": batch_size,
        "forward_latency_ms": round(fwd_latency_ms, 2),
        "single_batch_train_step_sec": round(step_latency_sec, 3),
        "num_train_batches": num_train_batches,
        "est_train_epoch_sec": round(est_train_epoch_sec, 2),
        "est_epoch_total_sec": round(est_epoch_sec, 2),
        "est_10_epochs_min": round((est_epoch_sec * 10) / 60.0, 2),
        "est_30_epochs_min": round((est_epoch_sec * 30) / 60.0, 2),
    }

    logger.info(f" [PASS] Single batch forward/backward step: {step_latency_sec:.3f} s")
    logger.info(f" Estimated epoch duration: {est_epoch_sec:.1f} s (~{summary['est_10_epochs_min']} min for 10 epochs)")
    logger.info(" CPU safety benchmark complete.\n")

    return summary


class AugmentedTrainer:
    """
    Trainer for EfficientNet-B0 with domain-specific augmentation,
    early stopping, visible batch progress, and CPU thermal management.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        class_names: List[str],
        learning_rate: float = 1e-4,
        checkpoint_name: str = "efficientnet_b0_augmented_best.pth",
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

            # Visible batch logging every 4 batches and at epoch end
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

        logger.info(f"Starting Augmented EfficientNet-B0 training (max {max_epochs} epochs, patience={self.early_stopping_patience})...")
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


def plot_separate_curves(
    history_df: pd.DataFrame,
    loss_path: Path,
    acc_path: Path,
    model_name: str = "EfficientNet-B0 + Augmentation",
):
    """Generates separate high-resolution plots for loss trajectories and accuracy trajectories."""
    epochs = history_df["epoch"]
    sns.set_theme(style="whitegrid")

    # 1. Loss curves
    plt.figure(figsize=(8, 5.5), dpi=300)
    plt.plot(epochs, history_df["train_loss"], "o-", color="#2A9D8F", label="Train Loss", linewidth=2)
    plt.plot(epochs, history_df["val_loss"], "s--", color="#E76F51", label="Val Loss", linewidth=2)
    best_loss_ep = history_df.loc[history_df["val_loss"].idxmin(), "epoch"]
    plt.axvline(best_loss_ep, color="#E63946", linestyle=":", alpha=0.8, label=f"Lowest Val Loss (Ep {best_loss_ep})")
    plt.title(f"{model_name} — Loss Trajectory", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11, fontweight="semibold")
    plt.ylabel("Cross Entropy Loss", fontsize=11, fontweight="semibold")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()
    logger.info(f"Saved loss curves to: {loss_path}")

    # 2. Accuracy & F1 curves
    plt.figure(figsize=(8, 5.5), dpi=300)
    plt.plot(epochs, history_df["train_acc"] * 100, "o-", color="#264653", label="Train Acc (%)", linewidth=2)
    plt.plot(epochs, history_df["val_acc"] * 100, "s--", color="#F4A261", label="Val Acc (%)", linewidth=2)
    plt.plot(epochs, history_df["val_macro_f1"] * 100, "^-.", color="#9B5DE5", label="Val Macro F1 (%)", linewidth=2)
    best_f1_ep = history_df.loc[history_df["val_macro_f1"].idxmax(), "epoch"]
    plt.axvline(best_f1_ep, color="#00BBF9", linestyle=":", alpha=0.8, label=f"Best Val F1 (Ep {best_f1_ep})")
    plt.title(f"{model_name} — Accuracy & F1 Trajectory", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11, fontweight="semibold")
    plt.ylabel("Score (%)", fontsize=11, fontweight="semibold")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300)
    plt.close()
    logger.info(f"Saved accuracy curves to: {acc_path}")


def run_experiment(
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device_arg: str = "cpu",
    seed: int = 42,
    patience: int = 8,
    benchmark_only: bool = False,
):
    # Set PyTorch threads to 4 for CPU thermal safety and efficiency
    torch.set_num_threads(4)

    root = get_project_root()
    config = load_config()

    classes = config.get("dataset", {}).get("classes", config.get("data", {}).get("classes", []))
    class_names = sorted(classes)
    num_classes = len(class_names)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and device_arg != "cpu" else "cpu")

    logger.info("=" * 65)
    logger.info(" PHASE 5: IMPROVED EFFICIENTNET-B0 WITH DATA AUGMENTATION")
    logger.info("=" * 65)
    logger.info(f"Architecture: ImageNet Pretrained EfficientNet-B0 + 6-class head")
    logger.info(f"Target Classes ({num_classes}): {class_names}")
    logger.info(f"Hyperparameters: Batch Size={batch_size}, LR={learning_rate}, Max Epochs={epochs}, Patience={patience}, Seed={seed}")
    logger.info(f"Execution Device: {device} (PyTorch Threads: {torch.get_num_threads()})")

    # 1. Dataloaders: Train set uses DOMAIN-SPECIFIC AUGMENTATION, Val & Test are DETERMINISTIC
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        config=config,
        batch_size=batch_size,
        num_workers=0,
        use_augmentation=True,  # Activated domain-specific training augmentation!
    )

    # 2. Build EfficientNet-B0 with ImageNet pretrained weights
    model = build_efficientnet_b0(num_classes=num_classes, pretrained=True, dropout=0.2)
    model_summary = get_model_summary(model)
    logger.info(f"Model Summary: {model_summary}")

    # 3. CPU Pre-Flight Safety Benchmark
    bench_results = run_cpu_safety_benchmark(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    if benchmark_only:
        logger.info("Benchmark-only flag specified. Terminating before training.")
        return bench_results

    # 4. Measure CPU inference latency
    latency_info = benchmark_inference_latency(model, num_warmup=10, num_runs=50, device="cpu")
    logger.info(f"CPU Inference Latency: {latency_info['average_latency_ms']} ms/image ({latency_info['fps']} FPS)")

    # 5. Trainer setup
    checkpoint_name = "efficientnet_b0_augmented_best.pth"
    trainer = AugmentedTrainer(
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

    # 6. Fit Augmented Model
    logger.info("Starting training loop with visible batch and epoch monitoring...")
    history_df = trainer.fit(max_epochs=epochs)

    # Save training history CSV
    metrics_dir = root / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_csv_path = metrics_dir / "efficientnet_b0_augmented_history.csv"
    history_df.to_csv(history_csv_path, index=False)
    logger.info(f"Training history saved to: {history_csv_path}")

    # 7. Evaluate on independent 177-image Test set using BEST checkpoint
    logger.info("Evaluating BEST Augmented checkpoint on independent 177-image Test set...")
    y_true, y_pred, confidences, image_paths = trainer.evaluate_test_set()

    # 8. Compute complete metrics
    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    metrics["model_information"] = model_summary
    metrics["inference_benchmark"] = latency_info
    metrics["safety_benchmark"] = bench_results

    best_epoch_num = int(history_df.loc[history_df["is_best"]]["epoch"].max()) if history_df["is_best"].any() else 0
    best_record = history_df[history_df["epoch"] == best_epoch_num].iloc[0].to_dict() if best_epoch_num > 0 else {}

    metrics["best_validation_result"] = {
        "best_epoch": best_epoch_num,
        "val_loss": best_record.get("val_loss", 0.0),
        "val_accuracy": best_record.get("val_acc", 0.0),
        "val_macro_f1": best_record.get("val_macro_f1", 0.0),
    }

    metrics["training_parameters"] = {
        "max_epochs": epochs,
        "actual_epochs_trained": len(history_df),
        "best_epoch": best_epoch_num,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "early_stopping_patience": patience,
        "device": str(device),
        "augmentation_used": True,
        "augmentation_library": "Albumentations",
    }

    # Baseline comparison calculations
    mobilenet_acc = 0.8305
    mobilenet_f1 = 0.8366
    effnet_base_acc = 0.8531
    effnet_base_f1 = 0.8493

    acc_improvement_over_baseline = (metrics["accuracy"] - effnet_base_acc) * 100.0
    f1_improvement_over_baseline = (metrics["macro_f1"] - effnet_base_f1) * 100.0

    metrics["baseline_comparison"] = {
        "mobilenet_v2_baseline_acc": mobilenet_acc,
        "mobilenet_v2_baseline_macro_f1": mobilenet_f1,
        "efficientnet_b0_baseline_acc": effnet_base_acc,
        "efficientnet_b0_baseline_macro_f1": effnet_base_f1,
        "augmented_acc": metrics["accuracy"],
        "augmented_macro_f1": metrics["macro_f1"],
        "accuracy_improvement_pct_pts": round(acc_improvement_over_baseline, 2),
        "macro_f1_improvement_pct_pts": round(f1_improvement_over_baseline, 2),
    }

    metrics_json_path = metrics_dir / "efficientnet_b0_augmented_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Final metrics saved to: {metrics_json_path}")

    # 9. Visualizations
    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cm_path = plots_dir / "efficientnet_b0_augmented_confusion_matrix.png"
    combined_curves_path = plots_dir / "efficientnet_b0_augmented_training_curves.png"
    loss_path = plots_dir / "efficientnet_b0_augmented_loss_curves.png"
    acc_path = plots_dir / "efficientnet_b0_augmented_accuracy_curves.png"

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        cm_path,
        title="EfficientNet-B0 + Augmentation — Confusion Matrix",
    )
    plot_training_curves(
        history_df,
        combined_curves_path,
        model_name="EfficientNet-B0 + Augmentation",
    )
    plot_separate_curves(
        history_df,
        loss_path=loss_path,
        acc_path=acc_path,
        model_name="EfficientNet-B0 + Augmentation",
    )

    # 10. Misclassifications
    preds_dir = root / "results" / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    misclass_csv_path = preds_dir / "efficientnet_b0_augmented_misclassifications.csv"
    misclass_df = save_misclassifications(y_true, y_pred, confidences, image_paths, class_names, misclass_csv_path)

    # Copy plots to brain artifacts directory
    brain_plot_dir = Path(r"C:\Users\laksh\.gemini\antigravity-ide\brain\e6bc9245-4294-41a5-86b7-ca0f7d9f5939\plots")
    brain_plot_dir.mkdir(parents=True, exist_ok=True)
    for p in [cm_path, combined_curves_path, loss_path, acc_path]:
        if p.exists():
            shutil.copy2(p, brain_plot_dir / p.name)

    # 11. Final Formatted Report
    print("\n" + "=" * 75)
    print(" EFFICIENTNET-B0 + DOMAIN AUGMENTATION EVALUATION REPORT")
    print("=" * 75)
    print(f"Architecture           : EfficientNet-B0 (with Domain Augmentation)")
    print(f"Device Used            : {device}")
    print(f"Total Parameters       : {model_summary['total_parameters']:,}")
    print(f"Trainable Parameters   : {model_summary['trainable_parameters']:,}")
    print(f"Model Size             : {model_summary['model_size_mb']} MB")
    print(f"Avg CPU Latency        : {latency_info['average_latency_ms']} ms/image ({latency_info['fps']} FPS)")
    print(f"Epochs Trained         : {len(history_df)} / {epochs} (Best Epoch: {best_epoch_num})")
    print("-" * 75)
    print(f"Best Val Accuracy      : {best_record.get('val_acc', 0.0)*100:.2f}%")
    print(f"Best Val Macro F1      : {best_record.get('val_macro_f1', 0.0)*100:.2f}%")
    print("-" * 75)
    print(f"Test Accuracy          : {metrics['accuracy'] * 100:.2f}% (Baseline: 85.31% -> Diff: {acc_improvement_over_baseline:+.2f}%)")
    print(f"Macro Precision        : {metrics['macro_precision'] * 100:.2f}% (Baseline: 87.34%)")
    print(f"Macro Recall           : {metrics['macro_recall'] * 100:.2f}% (Baseline: 83.57%)")
    print(f"Macro F1 Score         : {metrics['macro_f1'] * 100:.2f}% (Baseline: 84.93% -> Diff: {f1_improvement_over_baseline:+.2f}%)")
    print(f"Weighted F1 Score      : {metrics['weighted_f1'] * 100:.2f}% (Baseline: 85.25%)")
    print("-" * 75)
    print("Per-Class Metrics:")
    for cls_name, vals in metrics["per_class"].items():
        print(f"  - {cls_name:<20}: Precision={vals['precision']*100:>5.1f}%, Recall={vals['recall']*100:>5.1f}%, F1={vals['f1_score']*100:>5.1f}% (Support: {vals['support']})")
    print("-" * 75)
    print(f"Total Misclassifications: {len(misclass_df)} / {len(y_true)} ({len(misclass_df)/len(y_true)*100:.1f}%)")
    print(f"Saved Checkpoint       : {trainer.checkpoint_path}")
    print(f"Saved Metrics JSON     : {metrics_json_path}")
    print(f"Saved Confusion Matrix : {cm_path}")
    print(f"Saved Loss Curves      : {loss_path}")
    print(f"Saved Accuracy Curves  : {acc_path}")
    print(f"Saved Errors CSV       : {misclass_csv_path}")
    print("=" * 75 + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Improved EfficientNet-B0 with Augmentation.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, or auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--benchmark-only", action="store_true", help="Run only CPU safety checks")
    args = parser.parse_args()

    run_experiment(
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
