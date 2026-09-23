"""
Unit and Integration Tests for Phase 3 — MobileNetV2 Baseline Model:
- Model initialization with 6-class output
- Forward pass tensor output shape [batch_size, 6] on CPU
- Class mapping correctness
- Checkpoint persistence and state_dict reloading
- Metrics computation and evaluation functions
- Model parameter inspection and latency benchmarking
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from src.models.mobilenet import (
    build_mobilenet_v2,
    get_model_summary,
    benchmark_inference_latency,
)
from src.evaluation.metrics import (
    compute_classification_metrics,
    save_misclassifications,
)
from src.utils.config import load_config, get_project_root

EXPECTED_CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-damage",
    "Snow-Covered",
]


class TestMobileNetV2Baseline(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.num_classes = 6
        self.class_names = sorted(EXPECTED_CLASSES)

    def test_01_model_initialization(self):
        """Verifies MobileNetV2 builds with 6 output units and ImageNet backbone."""
        model = build_mobilenet_v2(num_classes=self.num_classes, pretrained=False, dropout=0.2)
        self.assertIsInstance(model, nn.Module)

        # Check classifier structure
        classifier = model.classifier
        self.assertIsInstance(classifier, nn.Sequential)
        # Final layer must be Linear with out_features == 6
        final_layer = classifier[-1]
        self.assertIsInstance(final_layer, nn.Linear)
        self.assertEqual(final_layer.out_features, 6)
        self.assertEqual(final_layer.in_features, 1280)

    def test_02_forward_pass_output_shape(self):
        """Verifies forward pass produces [batch_size, 6] tensor on CPU without error."""
        model = build_mobilenet_v2(num_classes=self.num_classes, pretrained=False)
        model = model.to(self.device)
        model.eval()

        batch_sizes = [1, 4, 8]
        for bs in batch_sizes:
            dummy_input = torch.randn(bs, 3, 224, 224, device=self.device)
            with torch.no_grad():
                output = model(dummy_input)

            self.assertEqual(output.shape, torch.Size([bs, self.num_classes]))
            self.assertFalse(torch.isnan(output).any().item())
            self.assertFalse(torch.isinf(output).any().item())

    def test_03_class_mapping_alignment(self):
        """Validates that model classes strictly align with dataset configuration."""
        config = load_config()
        configured_classes = config.get("dataset", {}).get("classes", config.get("data", {}).get("classes", []))
        self.assertEqual(sorted(configured_classes), self.class_names)

        class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        idx_to_class = {i: name for name, i in class_to_idx.items()}

        self.assertEqual(len(class_to_idx), 6)
        for i in range(6):
            self.assertEqual(class_to_idx[idx_to_class[i]], i)

    def test_04_checkpoint_save_and_load(self):
        """Verifies model state can be saved to disk and reloaded with identical weights."""
        model_orig = build_mobilenet_v2(num_classes=self.num_classes, pretrained=False)
        model_orig.eval()

        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out_orig = model_orig(dummy_input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            chk_path = Path(tmp_dir) / "test_checkpoint.pth"

            # Save checkpoint
            torch.save({
                "model_state_dict": model_orig.state_dict(),
                "class_names": self.class_names,
            }, chk_path)

            self.assertTrue(chk_path.exists())

            # Load into new model instance
            model_loaded = build_mobilenet_v2(num_classes=self.num_classes, pretrained=False)
            checkpoint = torch.load(chk_path, map_location="cpu")
            model_loaded.load_state_dict(checkpoint["model_state_dict"])
            model_loaded.eval()

            with torch.no_grad():
                out_loaded = model_loaded(dummy_input)

            # Assert identical output tensor values
            self.assertTrue(torch.allclose(out_orig, out_loaded, atol=1e-6))

    def test_05_evaluation_metrics_computation(self):
        """Verifies calculation of accuracy, macro F1, and per-class metrics."""
        # Simulated test ground truth and predictions
        y_true = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
        y_pred = [0, 1, 2, 3, 4, 4, 0, 1, 2, 3]  # One mistake: 5 predicted as 4

        metrics = compute_classification_metrics(y_true, y_pred, self.class_names)

        self.assertIn("accuracy", metrics)
        self.assertEqual(metrics["accuracy"], 0.90)  # 9/10 correct
        self.assertIn("macro_precision", metrics)
        self.assertIn("macro_recall", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("per_class", metrics)
        self.assertEqual(len(metrics["per_class"]), 6)

        # Check misclassification reporter
        confidences = [0.95] * 10
        paths = [f"dummy_{i}.jpg" for i in range(10)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            misclass_csv = Path(tmp_dir) / "misclass.csv"
            df_err = save_misclassifications(y_true, y_pred, confidences, paths, self.class_names, misclass_csv)
            self.assertEqual(len(df_err), 1)
            self.assertEqual(df_err.iloc[0]["actual_class"], self.class_names[5])
            self.assertEqual(df_err.iloc[0]["predicted_class"], self.class_names[4])

    def test_06_model_summary_and_latency(self):
        """Verifies parameter counting and fast inference latency measurement."""
        model = build_mobilenet_v2(num_classes=self.num_classes, pretrained=False)
        summary = get_model_summary(model)

        self.assertEqual(summary["architecture"], "MobileNetV2")
        self.assertGreater(summary["total_parameters"], 2_000_000)
        self.assertGreater(summary["trainable_parameters"], 0)
        self.assertGreater(summary["model_size_mb"], 0.0)

        latency = benchmark_inference_latency(model, num_warmup=2, num_runs=5, device="cpu")
        self.assertIn("average_latency_ms", latency)
        self.assertIn("fps", latency)
        self.assertGreater(latency["average_latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
