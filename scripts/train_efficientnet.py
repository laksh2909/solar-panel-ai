"""
CLI Script to train and evaluate the EfficientNet-B0 Baseline Model.
Usage:
    python scripts/train_efficientnet.py [--epochs 30] [--batch-size 32] [--lr 1e-4] [--device cpu] [--sanity-check] [--benchmark-one-epoch]
"""

import argparse
import json
import sys
import time
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

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
from src.training.trainer import set_seed
from src.evaluation.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_misclassifications,
)
from src.utils.config import load_config, get_project_root
from src.utils.logger import setup_logger

logger = setup_logger("train_efficientnet_cli")


def run_sanity_check(model: nn.Module, device: torch.device) -> bool:
    """Performs a 1-batch forward and backward pass to verify gradients and tensor compute."""
    print("\n--- Running Model Sanity Check ---", flush=True)
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dummy_x = torch.randn(4, 3, 224, 224, device=device)
    dummy_y = torch.tensor([0, 1, 2, 3], device=device)

    optimizer.zero_grad()
    out = model(dummy_x)
    loss = criterion(out, dummy_y)
    loss.backward()
    optimizer.step()

    print(f"Sanity Check PASSED: Dummy batch Loss = {loss.item():.4f}, Output shape = {out.shape}", flush=True)
    return True


def run_one_epoch_benchmark(model: nn.Module, train_loader, device: torch.device) -> float:
    """Runs one training epoch on real data to benchmark throughput before full run."""
    print("\n--- Running 1-Epoch Training Benchmark ---", flush=True)
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    t0 = time.time()
    total_loss = 0.0
    correct = 0
    total = 0

    num_batches = len(train_loader)
    for b_idx, (images, labels, _) in enumerate(train_loader, 1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if b_idx % 5 == 0 or b_idx == num_batches:
            print(f"  [Benchmark Batch {b_idx:02d}/{num_batches:02d}] Current Loss: {loss.item():.4f}", flush=True)

    elapsed = time.time() - t0
    epoch_loss = total_loss / total
    epoch_acc = (correct / total) * 100.0
    print(f"1-Epoch Benchmark Complete: Time = {elapsed:.2f}s, Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%\n", flush=True)
    return elapsed


def train_efficientnet(
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
    patience: int = 8,
):
    set_seed(seed)
    root = get_project_root()
    config = load_config()

    classes = config.get("dataset", {}).get("classes", config.get("data", {}).get("classes", []))
    class_names = sorted(classes)
    num_classes = len(class_names)

    target_device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else ("cuda" if device == "cuda" else "cpu"))

    print("=" * 70, flush=True)
    print(" PHASE 4: EFFICIENTNET-B0 BASELINE TRAINING", flush=True)
    print("=" * 70, flush=True)
    print(f"Target Classes ({num_classes}): {class_names}", flush=True)
    print(f"Device: {target_device} | Seed: {seed} | Batch Size: {batch_size} | LR: {learning_rate}", flush=True)

    # 1. Dataloaders (Clean baseline: NO random augmentation)
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        config=config,
        batch_size=batch_size,
        num_workers=0,
        use_augmentation=False,
    )

    # 2. Build model
    model = build_efficientnet_b0(num_classes=num_classes, pretrained=True, dropout=0.2)
    model = model.to(target_device)
    model_summary = get_model_summary(model)
    print(f"Model Parameters: Total={model_summary['total_parameters']:,}, Size={model_summary['model_size_mb']} MB", flush=True)

    # 3. Step 1: Model Sanity Check
    run_sanity_check(model, target_device)

    # 4. Step 2: 1-Epoch benchmark check
    run_one_epoch_benchmark(model, train_loader, target_device)

    # 5. Measure latency using identical methodology as MobileNetV2
    latency_info = benchmark_inference_latency(model, num_warmup=10, num_runs=50, device=str(target_device))
    print(f"CPU Inference Latency: {latency_info['average_latency_ms']} ms/image ({latency_info['fps']} FPS)\n", flush=True)

    # 6. Re-initialize fresh pretrained weights for full clean training run
    model = build_efficientnet_b0(num_classes=num_classes, pretrained=True, dropout=0.2)
    model = model.to(target_device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_dir = root / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "efficientnet_b0_baseline_best.pth"

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []

    print("-" * 70, flush=True)
    print(f"Beginning Full Training Run (Max {epochs} epochs, Early Stopping Patience={patience})...", flush=True)
    print("-" * 70, flush=True)

    num_train_batches = len(train_loader)
    train_start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # Training loop
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_samples = 0

        for b_idx, (images, labels, _) in enumerate(train_loader, 1):
            images = images.to(target_device)
            labels = labels.to(target_device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_samples += images.size(0)

            # Visible progress update every 6 batches
            if b_idx % 6 == 0 or b_idx == num_train_batches:
                print(f"  [Epoch {epoch:02d}/{epochs:02d} | Batch {b_idx:02d}/{num_train_batches:02d}] Step Loss: {loss.item():.4f}", flush=True)

        train_loss = train_loss_total / train_samples
        train_acc = train_correct / train_samples

        # Validation loop
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_samples = 0
        val_preds_all = []
        val_labels_all = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(target_device)
                labels = labels.to(target_device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_total += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_samples += images.size(0)

                val_preds_all.extend(preds.cpu().numpy().tolist())
                val_labels_all.extend(labels.cpu().numpy().tolist())

        val_loss = val_loss_total / val_samples
        val_acc = val_correct / val_samples
        val_macro_f1 = float(f1_score(val_labels_all, val_preds_all, average="macro", zero_division=0))
        epoch_sec = time.time() - t0

        is_best = val_macro_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "class_names": class_names,
            }, best_checkpoint_path)
            best_tag = " <-- BEST MODEL SAVED"
        else:
            patience_counter += 1
            best_tag = ""

        # Print standard epoch summary
        print(f"\nEpoch {epoch}/{epochs}", flush=True)
        print(f"Train Loss: {train_loss:.4f}", flush=True)
        print(f"Train Accuracy: {train_acc * 100:.2f}%", flush=True)
        print(f"Val Loss: {val_loss:.4f}", flush=True)
        print(f"Val Accuracy: {val_acc * 100:.2f}%", flush=True)
        print(f"Val Macro F1: {val_macro_f1 * 100:.2f}% (Time: {epoch_sec:.1f}s){best_tag}", flush=True)
        print("-" * 50, flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "is_best": is_best,
            "epoch_time_sec": round(epoch_sec, 1),
        })

        if patience_counter >= patience:
            print(f"Early stopping triggered at Epoch {epoch} (no validation improvement for {patience} epochs).", flush=True)
            print(f"Best model was saved at Epoch {best_epoch}.", flush=True)
            break

    total_train_sec = time.time() - train_start_time
    print(f"\nTraining completed in {total_train_sec:.1f} seconds across {len(history)} epochs.", flush=True)

    # 7. Save history CSV
    metrics_dir = root / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_csv_path = metrics_dir / "efficientnet_b0_baseline_history.csv"
    history_df.to_csv(history_csv_path, index=False)
    print(f"Saved history CSV to: {history_csv_path}", flush=True)

    # 8. Evaluate best checkpoint on test set
    print("\n--- Evaluating Best Checkpoint on Independent Test Set (177 images) ---", flush=True)
    checkpoint = torch.load(best_checkpoint_path, map_location=target_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []
    confidences = []
    image_paths = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(target_device)
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
        "val_loss": round(float(checkpoint["val_loss"]), 4),
        "val_accuracy": round(float(checkpoint["val_acc"]), 4),
        "val_macro_f1": round(float(checkpoint["val_macro_f1"]), 4),
    }
    metrics["training_parameters"] = {
        "model": "EfficientNet-B0",
        "device": str(target_device),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "epochs_completed": len(history),
        "best_epoch": int(best_epoch),
        "augmentation_used": False,
        "loss_function": "CrossEntropyLoss",
    }

    # Save metrics JSON
    metrics_json_path = metrics_dir / "efficientnet_b0_baseline_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON to: {metrics_json_path}", flush=True)

    # 9. Visualizations
    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cm_path = plots_dir / "efficientnet_b0_confusion_matrix.png"
    curves_path = plots_dir / "efficientnet_b0_training_curves.png"

    plot_confusion_matrix(y_true, y_pred, class_names, cm_path, title="EfficientNet-B0 Baseline — Confusion Matrix")
    plot_training_curves(history_df, curves_path, model_name="EfficientNet-B0 Baseline")

    # 10. Misclassifications CSV
    preds_dir = root / "results" / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    misclass_csv_path = preds_dir / "efficientnet_b0_misclassifications.csv"
    misclass_df = save_misclassifications(y_true, y_pred, confidences, image_paths, class_names, misclass_csv_path)

    # 11. Formatted final printout
    print("\n" + "=" * 70, flush=True)
    print(" EFFICIENTNET-B0 BASELINE FINAL EVALUATION REPORT", flush=True)
    print("=" * 70, flush=True)
    print(f"Architecture           : EfficientNet-B0 (torchvision)", flush=True)
    print(f"Pretrained Weights     : ImageNet-1K (EfficientNet_B0_Weights.DEFAULT)", flush=True)
    print(f"Total Parameters       : {model_summary['total_parameters']:,}", flush=True)
    print(f"Trainable Parameters   : {model_summary['trainable_parameters']:,}", flush=True)
    print(f"Model Size             : {model_summary['model_size_mb']} MB", flush=True)
    print(f"CPU Inference Latency  : {latency_info['average_latency_ms']} ms/image ({latency_info['fps']} FPS)", flush=True)
    print(f"Actual Epochs Completed: {len(history)} (Best Epoch: {best_epoch})", flush=True)
    print(f"Best Validation Acc    : {checkpoint['val_acc']*100:.2f}% (Macro F1: {checkpoint['val_macro_f1']*100:.2f}%)", flush=True)
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
    print(f"Checkpoint File        : {best_checkpoint_path}", flush=True)
    print(f"Metrics JSON           : {metrics_json_path}", flush=True)
    print(f"History CSV            : {history_csv_path}", flush=True)
    print(f"Confusion Matrix Plot  : {cm_path}", flush=True)
    print(f"Training Curves Plot   : {curves_path}", flush=True)
    print(f"Misclassifications CSV : {misclass_csv_path}", flush=True)
    print("=" * 70 + "\n", flush=True)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 baseline.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, or auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    args = parser.parse_args()

    train_efficientnet(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
