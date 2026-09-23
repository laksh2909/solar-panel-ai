"""
Training Engine for Solar Panel Fault Detection Models.

Features:
- Reproducible deterministic seeding
- Dynamic device dispatch (CPU / CUDA GPU)
- Epoch training and validation loops with loss and accuracy tracking
- Early stopping based on validation macro F1 and loss
- Checkpoint persistence for the best-performing model state
- Comprehensive training history tracking and CSV export
"""

import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logger import setup_logger
from src.utils.config import get_project_root

logger = setup_logger("trainer")


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across all libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BaselineTrainer:
    """
    Standard PyTorch trainer for deep learning classification models.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        class_names: List[str],
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        checkpoint_dir: Optional[Path | str] = None,
        checkpoint_name: str = "mobilenetv2_baseline_best.pth",
        early_stopping_patience: int = 8,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.seed = seed
        set_seed(self.seed)

        # Device detection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.num_classes = len(class_names)

        # Standard baseline configuration
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        root = get_project_root()
        if checkpoint_dir is None:
            self.checkpoint_dir = root / "models" / "checkpoints"
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / checkpoint_name

        self.early_stopping_patience = early_stopping_patience
        self.history: List[Dict[str, Any]] = []

        logger.info(
            f"Trainer initialized on device '{self.device}' with lr={learning_rate}, "
            f"early_stopping_patience={early_stopping_patience}, checkpoint='{self.checkpoint_path.name}'"
        )

    def train_epoch(self) -> Tuple[float, float]:
        """Runs one full training epoch across the training dataset."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for images, labels, _ in self.train_loader:
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

        epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
        epoch_acc = correct / total_samples if total_samples > 0 else 0.0
        return epoch_loss, epoch_acc

    def validate_epoch(self, loader: Optional[DataLoader] = None) -> Tuple[float, float, float]:
        """Evaluates model performance on validation or test dataset."""
        val_loader = loader if loader is not None else self.val_loader
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
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
        """
        Executes complete training loop with validation checks, early stopping,
        and best model checkpoint saving.
        """
        best_val_score = -1.0  # Monitoring validation Macro F1
        best_epoch = 0
        patience_counter = 0

        logger.info(f"Starting training run for max {max_epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate_epoch()
            epoch_time = time.time() - t0

            # Composite validation score: prioritizing Macro F1 with loss tie-breaker
            current_score = val_f1

            is_best = current_score > best_val_score
            if is_best:
                best_val_score = current_score
                best_epoch = epoch
                patience_counter = 0
                # Save checkpoint
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
                f"Epoch [{epoch:02d}/{max_epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | "
                f"Val F1: {val_f1*100:.1f}%{best_marker} ({epoch_time:.1f}s)"
            )

            # Early stopping check
            if patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch} (no validation improvement for {patience_counter} epochs). "
                    f"Best validation was at epoch {best_epoch}."
                )
                break

        total_time = time.time() - start_time
        logger.info(
            f"Training finished in {total_time:.1f}s. Best Epoch: {best_epoch} with Val Macro F1: {best_val_score*100:.2f}%."
        )

        history_df = pd.DataFrame(self.history)
        return history_df

    def evaluate_test_set(self) -> Tuple[List[int], List[int], List[float], List[str]]:
        """
        Loads the best saved checkpoint and evaluates inference across the test dataset.
        Returns:
            (y_true, y_pred, confidences, image_paths)
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found at: {self.checkpoint_path}")

        logger.info(f"Loading best checkpoint from: {self.checkpoint_path}")
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
