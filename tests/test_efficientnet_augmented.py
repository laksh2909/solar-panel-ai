"""
Unit and Integration Tests for Phase 5 — Improved EfficientNet-B0 with Domain-Specific Augmentation:
1. Augmentation pipeline initialization
2. Augmented image shape [3, 224, 224]
3. Augmented image tensor validity (finite values, valid range)
4. Model output shape [batch_size, 6]
5. Checkpoint save and load
6. Metrics calculation
7. Class mapping alignment
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from src.preprocessing.augmentation import (
    get_training_augmentation,
    get_visualization_augmentation,
    get_validation_pipeline,
    get_test_pipeline,
)
from src.models.efficientnet import (
    build_efficientnet_b0,
    get_model_summary,
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


class TestEfficientNetAugmented(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.num_classes = 6
        self.class_names = sorted(EXPECTED_CLASSES)

    def test_01_augmentation_pipeline_initialization(self):
        """Verifies Albumentations augmentation pipeline initializes properly."""
        train_aug = get_training_augmentation()
        self.assertIsNotNone(train_aug)
        self.assertGreater(len(train_aug.transforms), 5)

        vis_aug = get_visualization_augmentation()
        self.assertIsNotNone(vis_aug)

        val_pipe = get_validation_pipeline()
        self.assertIsNotNone(val_pipe)

        test_pipe = get_test_pipeline()
        self.assertIsNotNone(test_pipe)

    def test_02_augmented_image_shape(self):
        """Verifies augmented outputs have exact expected shapes."""
        dummy_rgb = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

        # Visual pipeline -> returns (224, 224, 3) numpy array
        vis_aug = get_visualization_augmentation(image_size=(224, 224))
        vis_out = vis_aug(image=dummy_rgb)["image"]
        self.assertIsInstance(vis_out, np.ndarray)
        self.assertEqual(vis_out.shape, (224, 224, 3))

        # Training tensor pipeline -> returns (3, 224, 224) torch.Tensor
        train_aug = get_training_augmentation(image_size=(224, 224), include_normalization=True)
        train_out = train_aug(image=dummy_rgb)["image"]
        self.assertIsInstance(train_out, torch.Tensor)
        self.assertEqual(train_out.shape, torch.Size([3, 224, 224]))

    def test_03_augmented_tensor_validity(self):
        """Verifies augmented tensor has valid float values, no NaNs, and standard normalized range."""
        dummy_rgb = np.random.randint(0, 256, (250, 250, 3), dtype=np.uint8)
        train_aug = get_training_augmentation(include_normalization=True)

        for _ in range(5):
            tensor = train_aug(image=dummy_rgb)["image"]
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertFalse(torch.isnan(tensor).any().item())
            self.assertFalse(torch.isinf(tensor).any().item())
            # Normalized values should fall reasonably within [-3.0, 3.5]
            self.assertGreater(tensor.min().item(), -5.0)
            self.assertLess(tensor.max().item(), 5.0)

    def test_04_model_output_shape(self):
        """Verifies EfficientNet-B0 forward pass produces [batch_size, 6] tensor."""
        model = build_efficientnet_b0(num_classes=self.num_classes, pretrained=False)
        model.eval()

        batch_sizes = [1, 2, 4]
        for bs in batch_sizes:
            dummy_input = torch.randn(bs, 3, 224, 224)
            with torch.no_grad():
                out = model(dummy_input)
            self.assertEqual(out.shape, torch.Size([bs, self.num_classes]))
            self.assertFalse(torch.isnan(out).any().item())

    def test_05_checkpoint_save_and_load(self):
        """Verifies checkpoint state saving and exact parameter restoration."""
        model_orig = build_efficientnet_b0(num_classes=self.num_classes, pretrained=False)
        model_orig.eval()

        dummy_x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out_orig = model_orig(dummy_x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            chk_path = Path(tmp_dir) / "test_aug_chk.pth"
            torch.save({
                "model_state_dict": model_orig.state_dict(),
                "class_names": self.class_names,
            }, chk_path)

            self.assertTrue(chk_path.exists())

            model_loaded = build_efficientnet_b0(num_classes=self.num_classes, pretrained=False)
            checkpoint = torch.load(chk_path, map_location="cpu")
            model_loaded.load_state_dict(checkpoint["model_state_dict"])
            model_loaded.eval()

            with torch.no_grad():
                out_loaded = model_loaded(dummy_x)

            self.assertTrue(torch.allclose(out_orig, out_loaded, atol=1e-6))

    def test_06_metrics_calculation(self):
        """Verifies classification metrics calculation and error auditing."""
        y_true = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
        y_pred = [0, 1, 2, 3, 4, 4, 0, 1, 2, 3]  # 1 mistake

        metrics = compute_classification_metrics(y_true, y_pred, self.class_names)
        self.assertEqual(metrics["accuracy"], 0.90)
        self.assertIn("macro_f1", metrics)
        self.assertIn("weighted_f1", metrics)
        self.assertIn("per_class", metrics)
        self.assertEqual(len(metrics["per_class"]), 6)

        confidences = [0.95] * 10
        paths = [f"img_{i}.jpg" for i in range(10)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test_misclass.csv"
            df_err = save_misclassifications(y_true, y_pred, confidences, paths, self.class_names, csv_path)
            self.assertEqual(len(df_err), 1)

    def test_07_class_mapping(self):
        """Validates that class mapping strictly matches 6 target classes."""
        config = load_config()
        configured_classes = config.get("dataset", {}).get("classes", config.get("data", {}).get("classes", []))
        self.assertEqual(sorted(configured_classes), self.class_names)


if __name__ == "__main__":
    unittest.main()
