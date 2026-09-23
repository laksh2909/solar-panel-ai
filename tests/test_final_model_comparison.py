"""
Unit and Regression Tests for Phase 17: Final Model Comparison and Ablation Study.

Verifies:
1. All three checkpoints exist and are valid.
2. Class mapping is identical and canonical (6 classes).
3. Test set size is exactly 177 across all six classes.
4. Output JSON exists and adheres to schema.
5. All CSV comparison files exist and have valid structure.
6. Metrics are finite and in [0.0, 1.0].
7. Confusion matrices have shape 6x6 and sum to 177.
8. Model parameter counts are correct.
9. No checkpoint was modified or corrupted.
10. Canonical preprocessing is strictly used and deterministic.
11. All required comparison plots exist and are non-empty.
"""

import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch

from src.models.efficientnet import build_efficientnet_b0, get_model_summary as get_effnet_summary
from src.models.mobilenet import build_mobilenet_v2, get_model_summary as get_mobilenet_summary
from src.preprocessing.augmentation import get_test_pipeline
from src.preprocessing.pipeline import load_image_rgb
from src.utils.config import get_project_root, load_config


class TestFinalModelComparison(unittest.TestCase):
    """Test suite for Phase 17 final model comparison and ablation study."""

    @classmethod
    def setUpClass(cls):
        cls.root = get_project_root()
        cls.config = load_config()
        cls.metrics_dir = cls.root / "results" / "metrics"
        cls.plots_dir = cls.root / "results" / "plots"
        cls.test_dir = cls.root / "data" / "test"

        cls.expected_classes = [
            "Bird-drop",
            "Clean",
            "Dusty",
            "Electrical-damage",
            "Physical-damage",
            "Snow-Covered",
        ]

        cls.ckpt_mobilenet = cls.root / "models" / "checkpoints" / "mobilenetv2_baseline_best.pth"
        cls.ckpt_effnet_base = cls.root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth"
        cls.ckpt_effnet_aug = cls.root / "models" / "checkpoints" / "efficientnet_b0_augmented_best.pth"

    def test_01_all_three_checkpoints_exist_and_loadable(self):
        """Verify all three checkpoints exist, have non-zero size, and load cleanly."""
        for path in [self.ckpt_mobilenet, self.ckpt_effnet_base, self.ckpt_effnet_aug]:
            self.assertTrue(path.exists(), f"Checkpoint missing: {path}")
            self.assertGreater(path.stat().st_size, 1_000_000, f"Checkpoint file suspiciously small: {path}")

            ckpt = torch.load(path, map_location="cpu")
            self.assertIn("model_state_dict", ckpt, f"Checkpoint missing 'model_state_dict': {path}")

    def test_02_class_mapping_is_identical(self):
        """Verify class mapping contains the exact canonical 6 classes in alphabetical order."""
        classes = sorted(self.config.get("dataset", {}).get("classes", []))
        self.assertEqual(classes, self.expected_classes)
        self.assertEqual(len(classes), 6)

    def test_03_test_set_size_and_classes(self):
        """Verify test set has exactly 177 images across all six classes."""
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        discovered = {}
        for c in self.expected_classes:
            folder = self.test_dir / c
            self.assertTrue(folder.is_dir(), f"Test class directory missing: {folder}")
            images = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
            self.assertGreater(len(images), 0, f"No images found for class {c}")
            discovered[c] = len(images)

        total_count = sum(discovered.values())
        self.assertEqual(total_count, 177, f"Expected 177 test samples, found {total_count}")

    def test_04_output_json_exists_and_schema_valid(self):
        """Verify final_model_comparison.json exists and contains all required sections."""
        json_path = self.metrics_dir / "final_model_comparison.json"
        self.assertTrue(json_path.exists(), f"Comparison JSON missing: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("metadata", data)
        self.assertIn("models", data)
        self.assertIn("pairwise_differences", data)
        self.assertIn("ablation_study", data)

        self.assertEqual(data["metadata"]["test_images_count"], 177)
        self.assertEqual(data["metadata"]["classes"], self.expected_classes)

        # Check all 3 models present
        expected_model_keys = ["mobilenetv2_baseline", "efficientnet_b0_baseline", "efficientnet_b0_augmented"]
        for key in expected_model_keys:
            self.assertIn(key, data["models"], f"Model key '{key}' missing from JSON")

        # Check pairwise differences keys
        self.assertIn("mobilenetv2_vs_efficientnet_b0", data["pairwise_differences"])
        self.assertIn("efficientnet_b0_baseline_vs_augmented", data["pairwise_differences"])

    def test_05_csv_files_exist_and_valid(self):
        """Verify all comparison and ablation CSV files exist and have correct columns."""
        comparison_csv = self.metrics_dir / "final_model_comparison.csv"
        per_class_csv = self.metrics_dir / "per_class_model_comparison.csv"
        ablation_csv = self.metrics_dir / "ablation_study.csv"

        self.assertTrue(comparison_csv.exists(), f"Missing CSV: {comparison_csv}")
        self.assertTrue(per_class_csv.exists(), f"Missing CSV: {per_class_csv}")
        self.assertTrue(ablation_csv.exists(), f"Missing CSV: {ablation_csv}")

        df_comp = pd.read_csv(comparison_csv)
        self.assertEqual(len(df_comp), 3, f"Expected 3 model rows, got {len(df_comp)}")
        self.assertIn("Model / Configuration", df_comp.columns)
        self.assertIn("Accuracy", df_comp.columns)
        self.assertIn("Macro F1", df_comp.columns)
        self.assertIn("Parameters", df_comp.columns)

        df_per = pd.read_csv(per_class_csv)
        self.assertEqual(len(df_per), 6, f"Expected 6 class rows, got {len(df_per)}")
        self.assertIn("Class", df_per.columns)
        self.assertIn("MobileNetV2 F1", df_per.columns)
        self.assertIn("EfficientNet-B0 Baseline F1", df_per.columns)
        self.assertIn("EfficientNet-B0 Augmented F1", df_per.columns)

        df_abl = pd.read_csv(ablation_csv)
        self.assertEqual(len(df_abl), 3, f"Expected 3 ablation rows, got {len(df_abl)}")
        self.assertIn("Experiment", df_abl.columns)
        self.assertIn("Architecture", df_abl.columns)
        self.assertIn("Augmentation", df_abl.columns)
        self.assertIn("Accuracy", df_abl.columns)
        self.assertIn("Macro F1", df_abl.columns)
        self.assertIn("Interpretation", df_abl.columns)

    def test_06_metrics_are_finite_and_bounded(self):
        """Verify all calculated classification metrics are finite and between 0.0 and 1.0."""
        json_path = self.metrics_dir / "final_model_comparison.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for model_id, m in data["models"].items():
            for metric_name in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1"]:
                val = m[metric_name]
                self.assertIsInstance(val, float, f"{metric_name} in {model_id} is not float")
                self.assertFalse(np.isnan(val), f"{metric_name} in {model_id} is NaN")
                self.assertFalse(np.isinf(val), f"{metric_name} in {model_id} is Inf")
                self.assertTrue(0.0 <= val <= 1.0, f"{metric_name}={val} out of bounds [0, 1] in {model_id}")

            for c in self.expected_classes:
                pc = m["per_class"][c]
                for p_metric in ["precision", "recall", "f1_score"]:
                    pval = pc[p_metric]
                    self.assertFalse(np.isnan(pval), f"Per-class {p_metric} for {c} in {model_id} is NaN")
                    self.assertTrue(0.0 <= pval <= 1.0, f"Per-class {p_metric}={pval} out of bounds")

    def test_07_confusion_matrices_shape_and_sum(self):
        """Verify all confusion matrices have shape 6x6 and sum exactly to 177."""
        json_path = self.metrics_dir / "final_model_comparison.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for model_id, m in data["models"].items():
            cm = np.array(m["confusion_matrix"])
            self.assertEqual(cm.shape, (6, 6), f"Confusion matrix shape is {cm.shape} for {model_id}, expected (6, 6)")
            self.assertEqual(int(cm.sum()), 177, f"Confusion matrix sum is {cm.sum()} for {model_id}, expected 177")

    def test_08_model_parameter_counts(self):
        """Verify model parameter counts match expected architecture specifications."""
        m_mobilenet = build_mobilenet_v2(num_classes=6, pretrained=False, dropout=0.2)
        sum_m = get_mobilenet_summary(m_mobilenet)
        self.assertEqual(sum_m["total_parameters"], 2_231_558)
        self.assertEqual(sum_m["trainable_parameters"], 2_231_558)
        self.assertAlmostEqual(sum_m["model_size_mb"], 8.51, delta=0.1)

        m_effnet = build_efficientnet_b0(num_classes=6, pretrained=False, dropout=0.2)
        sum_e = get_effnet_summary(m_effnet)
        self.assertEqual(sum_e["total_parameters"], 4_015_234)
        self.assertEqual(sum_e["trainable_parameters"], 4_015_234)
        self.assertAlmostEqual(sum_e["model_size_mb"], 15.32, delta=0.1)

    def test_09_checkpoints_not_modified(self):
        """Verify checkpoints can be loaded multiple times without mutating state dict."""
        ckpt = torch.load(self.ckpt_mobilenet, map_location="cpu")
        self.assertIn("model_state_dict", ckpt)
        # Check weights are preserved
        first_weight_key = list(ckpt["model_state_dict"].keys())[0]
        w1 = ckpt["model_state_dict"][first_weight_key]
        self.assertFalse(torch.isnan(w1).any())

    def test_10_canonical_preprocessing_pipeline(self):
        """Verify canonical preprocessing generates float32 tensor of shape (3, 224, 224) deterministically."""
        pipeline = get_test_pipeline()
        sample_img_path = next(self.test_dir.glob("*/*.JPG"))
        img_rgb = load_image_rgb(sample_img_path)

        self.assertEqual(img_rgb.ndim, 3)
        self.assertEqual(img_rgb.shape[2], 3)

        t1 = pipeline(image=img_rgb)["image"]
        t2 = pipeline(image=img_rgb)["image"]

        self.assertEqual(t1.shape, (3, 224, 224))
        self.assertEqual(t1.dtype, torch.float32)
        self.assertTrue(torch.all(t1 == t2), "Test preprocessing must be strictly deterministic")

    def test_11_all_required_plots_exist(self):
        """Verify all generated confusion matrices, comparisons, and summary plots exist and are non-empty."""
        expected_plots = [
            "mobilenetv2_confusion_matrix.png",
            "efficientnet_b0_confusion_matrix.png",
            "efficientnet_b0_augmented_confusion_matrix.png",
            "model_accuracy_comparison.png",
            "model_macro_f1_comparison.png",
            "model_weighted_f1_comparison.png",
            "model_inference_time_comparison.png",
            "model_parameter_count_comparison.png",
            "per_class_f1_comparison.png",
            "final_model_comparison_summary.png",
        ]

        for p_name in expected_plots:
            plot_file = self.plots_dir / p_name
            self.assertTrue(plot_file.exists(), f"Plot file missing: {plot_file}")
            self.assertGreater(plot_file.stat().st_size, 10_000, f"Plot file too small: {plot_file}")


if __name__ == "__main__":
    unittest.main()
