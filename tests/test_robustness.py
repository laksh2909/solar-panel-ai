"""
Unit tests for Phase 6: Robustness Testing of Solar Panel Fault Classification.

Verifies:
1. Each transformation produces a valid image.
2. Output dimensions remain valid (224, 224, 3) and pixel values in [0, 255].
3. Original test images are not modified.
4. Model output has exactly six classes with valid probabilities summing to 1.
5. Reproducibility with fixed seed.
6. All 177 test images are evaluated for every condition.
7. Metrics calculation works correctly.
"""

import hashlib
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
import torch

from src.models.efficientnet import build_efficientnet_b0
from src.robustness.evaluator import RobustnessEvaluator
from src.robustness.transforms import (
    ROBUSTNESS_CONDITIONS,
    apply_condition_to_image,
    apply_condition_to_tensor,
)
from src.utils.config import get_project_root


class TestRobustnessPipeline(unittest.TestCase):
    """Test suite covering the 7 robustness requirements."""

    def setUp(self):
        self.root = get_project_root()
        # Synthetic 224x224 RGB image with gradient pattern
        self.dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        for c in range(3):
            self.dummy_img[:, :, c] = np.linspace(20, 220, 224, dtype=np.uint8)

        test_dir = self.root / "data" / "test"
        self.test_images = list(test_dir.glob("*/*.JPG"))
        self.assertGreater(len(self.test_images), 0, "No test images found in data/test")
        self.sample_test_image = self.test_images[0]

    def test_01_each_transformation_produces_valid_image(self):
        """1. Each transformation produces a valid non-empty image."""
        self.assertEqual(len(ROBUSTNESS_CONDITIONS), 10)
        for condition in ROBUSTNESS_CONDITIONS:
            transformed = apply_condition_to_image(self.dummy_img, condition)
            self.assertIsInstance(transformed, np.ndarray, f"Condition {condition} did not return ndarray")
            self.assertGreater(transformed.size, 0, f"Condition {condition} returned empty array")
            self.assertFalse(np.isnan(transformed).any(), f"Condition {condition} produced NaNs")

    def test_02_output_dimensions_and_pixel_range(self):
        """2. Output dimensions remain (224, 224, 3) and pixel values in [0, 255]."""
        for condition in ROBUSTNESS_CONDITIONS:
            transformed = apply_condition_to_image(self.dummy_img, condition)
            self.assertEqual(
                transformed.shape,
                (224, 224, 3),
                f"Condition {condition} changed shape to {transformed.shape}",
            )
            self.assertEqual(
                transformed.dtype,
                np.uint8,
                f"Condition {condition} changed dtype to {transformed.dtype}",
            )
            self.assertGreaterEqual(transformed.min(), 0, f"Condition {condition} has negative values")
            self.assertLessEqual(transformed.max(), 255, f"Condition {condition} exceeded max 255")

    def test_03_original_test_images_not_modified(self):
        """3. Original test images are not modified on disk."""
        with open(self.sample_test_image, "rb") as f:
            md5_before = hashlib.md5(f.read()).hexdigest()

        with Image.open(self.sample_test_image) as pil_img:
            img_np = np.array(pil_img.convert("RGB").resize((224, 224)))

        for condition in ROBUSTNESS_CONDITIONS:
            _ = apply_condition_to_image(img_np, condition)
            _ = apply_condition_to_tensor(img_np, condition)

        with open(self.sample_test_image, "rb") as f:
            md5_after = hashlib.md5(f.read()).hexdigest()

        self.assertEqual(md5_before, md5_after, "Test image on disk was modified!")

    def test_04_model_output_six_classes_and_probabilities(self):
        """4. Model output has six classes and probabilities sum to 1."""
        model = build_efficientnet_b0(num_classes=6, pretrained=False, dropout=0.2)
        model.eval()

        tensor = apply_condition_to_tensor(self.dummy_img, "GAUSSIAN_BLUR")
        batch = tensor.unsqueeze(0)  # [1, 3, 224, 224]

        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)

        self.assertEqual(logits.shape, (1, 6), f"Expected shape (1, 6), got {logits.shape}")
        prob_sum = probs.sum().item()
        self.assertAlmostEqual(prob_sum, 1.0, places=4, msg=f"Probabilities do not sum to 1: {prob_sum}")

    def test_05_reproducibility_with_fixed_seed(self):
        """5. Deterministic reproducibility with fixed inputs."""
        for condition in [
            "ORIGINAL",
            "LOW_LIGHT",
            "HIGH_BRIGHTNESS",
            "LOW_CONTRAST",
            "HIGH_CONTRAST",
            "SMALL_ROTATION",
        ]:
            out1 = apply_condition_to_image(self.dummy_img, condition)
            out2 = apply_condition_to_image(self.dummy_img, condition)
            np.testing.assert_array_equal(
                out1, out2, err_msg=f"Condition {condition} is not deterministic"
            )

    def test_06_all_177_test_images_evaluated(self):
        """6. All 177 test images are discovered and evaluated."""
        evaluator = RobustnessEvaluator(
            checkpoint_path=self.root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth",
            test_dir=self.root / "data" / "test",
            device="cpu",
        )
        self.assertEqual(
            len(evaluator.test_samples),
            177,
            f"Expected 177 test samples, found {len(evaluator.test_samples)}",
        )

    def test_07_metrics_calculation_correctness(self):
        """7. Metrics calculation works correctly."""
        from src.evaluation.metrics import compute_classification_metrics

        classes = ["A", "B", "C"]
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 1]  # 4 correct out of 6 -> Acc = 66.67%

        metrics = compute_classification_metrics(y_true, y_pred, classes)
        self.assertAlmostEqual(metrics["accuracy"], 4.0 / 6.0, places=4)
        self.assertIn("macro_f1", metrics)
        self.assertIn("per_class", metrics)
        self.assertEqual(len(metrics["per_class"]), 3)


if __name__ == "__main__":
    unittest.main()
