"""
Unit and Integration Tests for Phase 5 — SparkNet Baseline Reproduction Model:
1. FireModule initialization and forward pass
2. SparkNet initialization and architecture structure
3. Forward pass output shape [batch_size, 6] on CPU
4. Six-class mapping correctness
5. Checkpoint persistence and reload integrity
6. Evaluation metrics computation and misclassification reporting
7. Parameter counting and latency benchmarking
"""

import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn as nn

from src.models.sparkneta import (
    FireModule,
    SparkNetBranch,
    SparkNet,
    build_sparknet,
    get_model_summary,
    benchmark_inference_latency,
)
from src.evaluation.metrics import (
    compute_classification_metrics,
    save_misclassifications,
)
from src.utils.config import load_config

EXPECTED_CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-damage",
    "Snow-Covered",
]


class TestSparkNetBaseline(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.num_classes = 6
        self.class_names = sorted(EXPECTED_CLASSES)

    def test_01_fire_module_initialization(self):
        """Verifies FireModule initialization, expansion, and output dimensions."""
        fire = FireModule(
            in_channels=64,
            squeeze_channels=16,
            expand1x1_channels=48,
            expand3x3_channels=48,
            use_bn=True,
        )
        self.assertIsInstance(fire, nn.Module)
        self.assertEqual(fire.out_channels, 96)

        # Check forward pass
        dummy_x = torch.randn(2, 64, 28, 28)
        out = fire(dummy_x)
        self.assertEqual(out.shape, torch.Size([2, 96, 28, 28]))
        self.assertFalse(torch.isnan(out).any().item())

    def test_02_sparknet_initialization(self):
        """Verifies SparkNet initializes with 4 hierarchical branches and classification head."""
        model = build_sparknet(num_classes=self.num_classes, dropout=0.3)
        self.assertIsInstance(model, nn.Module)

        # Verify 4 branches
        self.assertTrue(hasattr(model, "branch1"))
        self.assertTrue(hasattr(model, "branch2"))
        self.assertTrue(hasattr(model, "branch3"))
        self.assertTrue(hasattr(model, "branch4"))

        # Verify branch output channels
        self.assertEqual(model.branch1.out_channels, 96)
        self.assertEqual(model.branch2.out_channels, 128)
        self.assertEqual(model.branch3.out_channels, 192)
        self.assertEqual(model.branch4.out_channels, 256)

        # Verify fused channels: 96 + 128 + 192 + 256 = 672
        fused_channels = (
            model.branch1.out_channels
            + model.branch2.out_channels
            + model.branch3.out_channels
            + model.branch4.out_channels
        )
        self.assertEqual(fused_channels, 672)

        # Verify final classifier layer produces 6 logits
        final_layer = model.classifier[-1]
        self.assertIsInstance(final_layer, nn.Linear)
        self.assertEqual(final_layer.out_features, 6)

    def test_03_forward_pass_output_shape(self):
        """Verifies forward pass produces [batch_size, 6] tensor on CPU."""
        model = build_sparknet(num_classes=self.num_classes)
        model = model.to(self.device)
        model.eval()

        batch_sizes = [1, 2, 4]
        for bs in batch_sizes:
            dummy_input = torch.randn(bs, 3, 224, 224, device=self.device)
            with torch.no_grad():
                output = model(dummy_input)

            self.assertEqual(output.shape, torch.Size([bs, self.num_classes]))
            self.assertFalse(torch.isnan(output).any().item())
            self.assertFalse(torch.isinf(output).any().item())

    def test_04_six_class_mapping(self):
        """Validates that model classes strictly align with dataset configuration."""
        config = load_config()
        configured_classes = config.get("dataset", {}).get("classes", config.get("data", {}).get("classes", []))
        self.assertEqual(sorted(configured_classes), self.class_names)

        class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        idx_to_class = {i: name for name, i in class_to_idx.items()}

        self.assertEqual(len(class_to_idx), 6)
        for i in range(6):
            self.assertEqual(class_to_idx[idx_to_class[i]], i)

    def test_05_checkpoint_save_and_load(self):
        """Verifies model state can be saved to disk and reloaded identically."""
        model_orig = build_sparknet(num_classes=self.num_classes)
        model_orig.eval()

        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out_orig = model_orig(dummy_input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            chk_path = Path(tmp_dir) / "test_sparknet_chk.pth"

            # Save checkpoint
            torch.save({
                "model_state_dict": model_orig.state_dict(),
                "class_names": self.class_names,
            }, chk_path)

            self.assertTrue(chk_path.exists())

            # Load into new model instance
            model_loaded = build_sparknet(num_classes=self.num_classes)
            checkpoint = torch.load(chk_path, map_location="cpu")
            model_loaded.load_state_dict(checkpoint["model_state_dict"])
            model_loaded.eval()

            with torch.no_grad():
                out_loaded = model_loaded(dummy_input)

            self.assertTrue(torch.allclose(out_orig, out_loaded, atol=1e-6))

    def test_06_evaluation_metrics(self):
        """Verifies evaluation metrics computation and misclassification reporting."""
        y_true = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
        y_pred = [0, 1, 2, 3, 4, 4, 0, 1, 2, 3]  # One mistake: 5 predicted as 4

        metrics = compute_classification_metrics(y_true, y_pred, self.class_names)

        self.assertIn("accuracy", metrics)
        self.assertEqual(metrics["accuracy"], 0.90)
        self.assertIn("macro_precision", metrics)
        self.assertIn("macro_recall", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("per_class", metrics)
        self.assertEqual(len(metrics["per_class"]), 6)

        # Check misclassification output
        confidences = [0.95] * 10
        paths = [f"test_img_{i}.jpg" for i in range(10)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            misclass_csv = Path(tmp_dir) / "misclass.csv"
            df_err = save_misclassifications(y_true, y_pred, confidences, paths, self.class_names, misclass_csv)
            self.assertEqual(len(df_err), 1)
            self.assertEqual(df_err.iloc[0]["actual_class"], self.class_names[5])
            self.assertEqual(df_err.iloc[0]["predicted_class"], self.class_names[4])

    def test_07_parameter_and_latency_reporting(self):
        """Verifies parameter count and inference latency benchmarking."""
        model = build_sparknet(num_classes=self.num_classes)
        summary = get_model_summary(model)

        self.assertEqual(summary["architecture"], "SparkNet")
        # Ensure parameter count fits the ~300k budget from the paper
        self.assertGreater(summary["total_parameters"], 200_000)
        self.assertLess(summary["total_parameters"], 500_000)
        self.assertGreater(summary["trainable_parameters"], 0)
        self.assertGreater(summary["model_size_mb"], 0.0)

        latency = benchmark_inference_latency(model, num_warmup=2, num_runs=5, device="cpu")
        self.assertIn("average_latency_ms", latency)
        self.assertIn("fps", latency)
        self.assertGreater(latency["average_latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
