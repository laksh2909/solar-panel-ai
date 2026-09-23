"""
Robustness Evaluator for Solar Panel Fault Classification.

Evaluates a frozen PyTorch model checkpoint across 10 controlled robustness
conditions on the independent 177-image test dataset.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.models.efficientnet import build_efficientnet_b0
from src.robustness.transforms import (
    ROBUSTNESS_CONDITIONS,
    apply_condition_to_image,
    apply_condition_to_tensor,
)
from src.utils.config import get_project_root, load_config
from src.utils.logger import setup_logger

logger = setup_logger("robustness_evaluator")


class RobustnessEvaluator:
    """
    Evaluates model robustness across environmental & sensor degradation conditions.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        test_dir: str | Path | None = None,
        device: str = "cpu",
    ):
        self.root = get_project_root()
        self.config = load_config()

        classes = self.config.get("dataset", {}).get("classes", self.config.get("data", {}).get("classes", []))
        self.class_names = sorted(classes)
        self.num_classes = len(self.class_names)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.class_names)}

        self.device = torch.device(device)

        if checkpoint_path is None:
            self.checkpoint_path = (
                self.root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth"
            )
        else:
            self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        if test_dir is None:
            self.test_dir = self.root / "data" / "test"
        else:
            self.test_dir = Path(test_dir)

        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found at: {self.test_dir}")

        self.model = self._load_model()
        self.test_samples = self._discover_test_images()

    def _load_model(self) -> nn.Module:
        logger.info(f"Loading EfficientNet-B0 baseline checkpoint from: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model = build_efficientnet_b0(num_classes=self.num_classes, pretrained=False, dropout=0.2)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        return model

    def _discover_test_images(self) -> List[Dict[str, Any]]:
        """
        Discovers all test image files in data/test subdirectories.
        Returns a sorted list of sample dicts.
        """
        samples = []
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for class_dir in sorted(self.test_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                continue
            class_idx = self.class_to_idx[class_name]

            for file_path in sorted(class_dir.iterdir()):
                if file_path.suffix.lower() in valid_exts:
                    samples.append({
                        "path": file_path,
                        "name": file_path.name,
                        "class_name": class_name,
                        "class_idx": class_idx,
                    })

        logger.info(f"Discovered {len(samples)} test images across {self.num_classes} classes.")
        if len(samples) != 177:
            logger.warning(f"Expected 177 test samples, found {len(samples)}.")
        return samples

    def evaluate_condition(
        self, condition: str, batch_size: int = 32
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Evaluates the model on all test images under a single robustness condition.
        Returns (metrics_dict, predictions_list).
        """
        logger.info(f"Evaluating condition: {condition} ({len(self.test_samples)} images)...")
        softmax = nn.Softmax(dim=1)

        predictions_records = []
        y_true = []
        y_pred = []
        confidences = []

        # Process in batches for computational efficiency
        for i in range(0, len(self.test_samples), batch_size):
            batch_samples = self.test_samples[i : i + batch_size]
            tensors = []
            for sample in batch_samples:
                # Load image deterministically
                with Image.open(sample["path"]) as pil_img:
                    rgb_img = np.array(pil_img.convert("RGB").resize((224, 224)))
                tensor = apply_condition_to_tensor(rgb_img, condition)
                tensors.append(tensor)

            batch_tensor = torch.stack(tensors, dim=0).to(self.device)

            with torch.no_grad():
                logits = self.model(batch_tensor)
                probs = softmax(logits)
                confs, preds = torch.max(probs, dim=1)

            batch_confs = confs.cpu().numpy().tolist()
            batch_preds = preds.cpu().numpy().tolist()

            for sample, pred_idx, conf in zip(batch_samples, batch_preds, batch_confs):
                true_idx = sample["class_idx"]
                true_name = sample["class_name"]
                pred_name = self.idx_to_class[pred_idx]
                is_correct = bool(true_idx == pred_idx)

                y_true.append(true_idx)
                y_pred.append(pred_idx)
                confidences.append(conf)

                predictions_records.append({
                    "image_name": sample["name"],
                    "true_class": true_name,
                    "condition": condition,
                    "predicted_class": pred_name,
                    "confidence": round(conf, 4),
                    "correct": is_correct,
                })

        # Calculate metrics
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        macro_p = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
        macro_r = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

        # Per-class breakdowns
        per_class_p = precision_score(y_true_arr, y_pred_arr, average=None, zero_division=0)
        per_class_r = recall_score(y_true_arr, y_pred_arr, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true_arr, y_pred_arr, average=None, zero_division=0)

        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(self.num_classes)))

        per_class_metrics = {}
        for idx, cls in enumerate(self.class_names):
            support = int(np.sum(y_true_arr == idx))
            pred_count = int(np.sum(y_pred_arr == idx))
            per_class_metrics[cls] = {
                "precision": round(float(per_class_p[idx]), 4),
                "recall": round(float(per_class_r[idx]), 4),
                "f1_score": round(float(per_class_f1[idx]), 4),
                "support": support,
                "predicted_count": pred_count,
            }

        # Confidence stats
        conf_arr = np.array(confidences)
        correct_mask = (y_true_arr == y_pred_arr)
        avg_conf_all = float(np.mean(conf_arr))
        avg_conf_correct = float(np.mean(conf_arr[correct_mask])) if np.any(correct_mask) else 0.0
        avg_conf_incorrect = float(np.mean(conf_arr[~correct_mask])) if np.any(~correct_mask) else 0.0

        metrics = {
            "condition": condition,
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "average_confidence": round(avg_conf_all, 4),
            "average_confidence_correct": round(avg_conf_correct, 4),
            "average_confidence_incorrect": round(avg_conf_incorrect, 4),
            "per_class": per_class_metrics,
            "confusion_matrix": cm.tolist(),
        }

        return metrics, predictions_records

    def run_all_conditions(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Runs evaluation across all 10 conditions.
        Returns a dictionary of all results and a combined predictions DataFrame.
        """
        all_metrics = {}
        all_predictions = []

        for condition in ROBUSTNESS_CONDITIONS:
            cond_metrics, cond_preds = self.evaluate_condition(condition)
            all_metrics[condition] = cond_metrics
            all_predictions.extend(cond_preds)

        predictions_df = pd.DataFrame(all_predictions)

        # Compute degradation metrics relative to ORIGINAL
        orig_metrics = all_metrics["ORIGINAL"]
        orig_acc = orig_metrics["accuracy"]
        orig_f1 = orig_metrics["macro_f1"]
        orig_phys_recall = orig_metrics["per_class"]["Physical-damage"]["recall"]

        degradation_summary = {}
        for cond, m in all_metrics.items():
            acc_drop = round(m["accuracy"] - orig_acc, 4)
            f1_drop = round(m["macro_f1"] - orig_f1, 4)
            phys_recall = m["per_class"]["Physical-damage"]["recall"]
            phys_recall_drop = round(phys_recall - orig_phys_recall, 4)

            degradation_summary[cond] = {
                "accuracy_delta": acc_drop,
                "macro_f1_delta": f1_drop,
                "physical_damage_recall_delta": phys_recall_drop,
                "accuracy_pct_drop": round(acc_drop * 100, 2),
                "macro_f1_pct_drop": round(f1_drop * 100, 2),
                "physical_damage_recall_pct_drop": round(phys_recall_drop * 100, 2),
            }

        # Identify condition causing largest degradation
        worst_acc_cond = min(all_metrics.keys(), key=lambda c: all_metrics[c]["accuracy"])
        worst_f1_cond = min(all_metrics.keys(), key=lambda c: all_metrics[c]["macro_f1"])

        results_dict = {
            "model": "EfficientNet-B0 Baseline",
            "checkpoint": str(self.checkpoint_path),
            "total_test_samples": len(self.test_samples),
            "conditions_evaluated": ROBUSTNESS_CONDITIONS,
            "metrics_per_condition": all_metrics,
            "degradation_relative_to_original": degradation_summary,
            "worst_condition_by_accuracy": {
                "condition": worst_acc_cond,
                "accuracy": all_metrics[worst_acc_cond]["accuracy"],
                "drop_pct": degradation_summary[worst_acc_cond]["accuracy_pct_drop"],
            },
            "worst_condition_by_f1": {
                "condition": worst_f1_cond,
                "macro_f1": all_metrics[worst_f1_cond]["macro_f1"],
                "drop_pct": degradation_summary[worst_f1_cond]["macro_f1_pct_drop"],
            },
        }

        return results_dict, predictions_df
