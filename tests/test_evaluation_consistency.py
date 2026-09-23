"""
Unit tests for Phase 7A: Evaluation Consistency Verification.

Verifies:
1. Same checkpoint is loaded.
2. Same six-class mapping is used.
3. Same 177 test images are evaluated.
4. Same preprocessing is applied.
5. Evaluation is deterministic.
"""

from pathlib import Path
import unittest
import numpy as np
import torch

from src.models.efficientnet import build_efficientnet_b0
from src.preprocessing.pipeline import load_image_rgb
from src.preprocessing.augmentation import get_test_pipeline
from src.utils.config import get_project_root, load_config


class TestEvaluationConsistency(unittest.TestCase):
    """Test suite covering the 5 consistency requirements."""

    def setUp(self):
        self.root = get_project_root()
        self.config = load_config()
        self.ckpt_path = self.root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth"
        self.test_dir = self.root / "data" / "test"

    def test_01_same_checkpoint_is_loaded(self):
        """1. Verifies that the baseline checkpoint exists and loads state dict cleanly."""
        self.assertTrue(self.ckpt_path.exists(), f"Checkpoint not found at: {self.ckpt_path}")
        checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        self.assertIn("model_state_dict", checkpoint)
        self.assertEqual(checkpoint.get("epoch"), 7)
        self.assertAlmostEqual(checkpoint.get("val_acc"), 0.8000, places=4)

        model = build_efficientnet_b0(num_classes=6, pretrained=False, dropout=0.2)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(total_params, 4015234)

    def test_02_same_six_class_mapping_is_used(self):
        """2. Verifies that the exact 6 expected classes and sorted indices are used."""
        classes = sorted(self.config.get("dataset", {}).get("classes", []))
        expected_classes = [
            "Bird-drop",
            "Clean",
            "Dusty",
            "Electrical-damage",
            "Physical-damage",
            "Snow-Covered",
        ]
        self.assertEqual(classes, expected_classes)
        self.assertEqual(len(classes), 6)

    def test_03_same_177_test_images_are_evaluated(self):
        """3. Verifies that exactly 177 test images exist and are discoverable in data/test."""
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        discovered_images = []
        for class_dir in sorted(self.test_dir.iterdir()):
            if class_dir.is_dir():
                for f in class_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in valid_exts:
                        discovered_images.append(f)

        self.assertEqual(len(discovered_images), 177, f"Expected 177 test images, found {len(discovered_images)}")

    def test_04_same_canonical_preprocessing_applied(self):
        """4. Verifies that the canonical preprocessing produces [3, 224, 224] tensor."""
        pipeline = get_test_pipeline()
        sample_images = list(self.test_dir.glob("*/*.JPG"))
        self.assertGreater(len(sample_images), 0)

        img_rgb = load_image_rgb(sample_images[0])
        self.assertEqual(img_rgb.ndim, 3)
        self.assertEqual(img_rgb.shape[2], 3)

        transformed = pipeline(image=img_rgb)
        tensor = transformed["image"]
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_05_evaluation_is_deterministic(self):
        """5. Verifies that multiple forward passes on the same image produce identical logits."""
        model = build_efficientnet_b0(num_classes=6, pretrained=False, dropout=0.2)
        checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        pipeline = get_test_pipeline()
        sample_images = list(self.test_dir.glob("*/*.JPG"))
        img_rgb = load_image_rgb(sample_images[0])

        tensor1 = pipeline(image=img_rgb)["image"].unsqueeze(0)
        tensor2 = pipeline(image=img_rgb)["image"].unsqueeze(0)

        with torch.no_grad():
            logits1 = model(tensor1)
            logits2 = model(tensor2)

        self.assertTrue(torch.equal(logits1, logits2), "Forward passes produced non-deterministic outputs")


if __name__ == "__main__":
    unittest.main()
