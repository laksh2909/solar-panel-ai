"""
Unit tests for Phase 8: Grad-CAM Explainability Module.

Verifies:
1. Checkpoint loads correctly and state_dict is verified.
2. Six-class mapping is preserved.
3. Grad-CAM output exists and produces valid heatmaps.
4. Heatmap dimensions are valid and match requested target size.
5. Heatmap values are finite and non-NaN.
6. Heatmap is strictly normalized in [0, 1].
7. Overlay dimensions match original image.
8. Model parameters remain unchanged after Grad-CAM execution (no weight drift).
9. Canonical preprocessing is used.
"""

from pathlib import Path
import unittest
import numpy as np
from PIL import Image
import torch

from src.explainability.gradcam import GradCAM
from src.models.efficientnet import build_efficientnet_b0
from src.preprocessing.pipeline import load_image_rgb
from src.preprocessing.augmentation import get_test_pipeline
from src.utils.config import get_project_root, load_config


class TestGradCAM(unittest.TestCase):
    """Test suite for Grad-CAM explainability module."""

    def setUp(self):
        self.root = get_project_root()
        self.config = load_config()
        self.ckpt_path = self.root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth"
        self.test_dir = self.root / "data" / "test"

        self.assertTrue(self.ckpt_path.exists(), f"Missing checkpoint: {self.ckpt_path}")
        self.checkpoint = torch.load(self.ckpt_path, map_location="cpu")

        self.model = build_efficientnet_b0(num_classes=6, pretrained=False, dropout=0.2)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

        self.pipeline = get_test_pipeline()
        self.sample_images = list(self.test_dir.glob("*/*.JPG"))
        self.assertGreater(len(self.sample_images), 0, "No test images found")

    def test_01_checkpoint_loads_correctly(self):
        """1. Checkpoint loads correctly and parameters match baseline."""
        self.assertEqual(self.checkpoint.get("epoch"), 7)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(total_params, 4015234)

    def test_02_six_class_mapping_preserved(self):
        """2. Six-class mapping is preserved exactly."""
        classes = sorted(self.config.get("dataset", {}).get("classes", []))
        expected = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-damage", "Snow-Covered"]
        self.assertEqual(classes, expected)

    def test_03_gradcam_output_exists(self):
        """3. Grad-CAM generates non-None heatmap, prediction, and confidence."""
        img_rgb = load_image_rgb(self.sample_images[0])
        tensor = self.pipeline(image=img_rgb)["image"].unsqueeze(0)

        with GradCAM(self.model) as cam:
            heatmap, pred_idx, conf = cam.generate_heatmap(tensor)

        self.assertIsNotNone(heatmap)
        self.assertIsInstance(pred_idx, int)
        self.assertGreaterEqual(pred_idx, 0)
        self.assertLess(pred_idx, 6)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_04_heatmap_dimensions_valid(self):
        """4. Heatmap spatial dimensions match input or requested size."""
        img_rgb = load_image_rgb(self.sample_images[0])
        tensor = self.pipeline(image=img_rgb)["image"].unsqueeze(0)

        with GradCAM(self.model) as cam:
            # Default dimension (224, 224)
            heatmap, _, _ = cam.generate_heatmap(tensor)
            self.assertEqual(heatmap.shape, (224, 224))

            # Custom requested target size (e.g. 300, 400) -> (W, H) = (300, 400) -> heatmap (400, 300)
            heatmap_custom, _, _ = cam.generate_heatmap(tensor, target_size=(300, 400))
            self.assertEqual(heatmap_custom.shape, (400, 300))

    def test_05_heatmap_values_finite(self):
        """5. Heatmap values are finite and non-NaN."""
        img_rgb = load_image_rgb(self.sample_images[0])
        tensor = self.pipeline(image=img_rgb)["image"].unsqueeze(0)

        with GradCAM(self.model) as cam:
            heatmap, _, _ = cam.generate_heatmap(tensor)

        self.assertFalse(np.isnan(heatmap).any(), "Heatmap contains NaNs")
        self.assertFalse(np.isinf(heatmap).any(), "Heatmap contains Infs")

    def test_06_heatmap_normalized_zero_to_one(self):
        """6. Heatmap is strictly normalized in [0, 1]."""
        img_rgb = load_image_rgb(self.sample_images[0])
        tensor = self.pipeline(image=img_rgb)["image"].unsqueeze(0)

        with GradCAM(self.model) as cam:
            heatmap, _, _ = cam.generate_heatmap(tensor)

        self.assertGreaterEqual(heatmap.min(), 0.0)
        self.assertLessEqual(heatmap.max(), 1.0)

    def test_07_overlay_dimensions_match_original_image(self):
        """7. Overlay dimensions match original image dimensions."""
        img_rgb = load_image_rgb(self.sample_images[0])
        tensor = self.pipeline(image=img_rgb)["image"].unsqueeze(0)

        with GradCAM(self.model) as cam:
            heatmap, _, _ = cam.generate_heatmap(tensor, target_size=(img_rgb.shape[1], img_rgb.shape[0]))
            overlay = cam.overlay_heatmap(img_rgb, heatmap)

        self.assertEqual(overlay.shape, img_rgb.shape)
        self.assertEqual(overlay.dtype, np.uint8)

    def test_08_model_weights_remain_unchanged(self):
        """8. Model weights do not change after running Grad-CAM."""
        weights_before = [p.clone().detach() for p in self.model.parameters()]

        img_rgb = load_image_rgb(self.sample_images[0])
        tensor = self.pipeline(image=img_rgb)["image"].unsqueeze(0)

        with GradCAM(self.model) as cam:
            _ = cam.generate_heatmap(tensor)
            _ = cam.generate_heatmap(tensor, target_class=3)

        weights_after = [p.clone().detach() for p in self.model.parameters()]

        for w_b, w_a in zip(weights_before, weights_after):
            self.assertTrue(torch.equal(w_b, w_a), "Model weights changed after Grad-CAM backward pass!")

    def test_09_canonical_preprocessing_used(self):
        """9. Canonical preprocessing outputs valid standardized shape [1, 3, 224, 224]."""
        img_rgb = load_image_rgb(self.sample_images[0])
        transformed = self.pipeline(image=img_rgb)["image"]
        self.assertEqual(transformed.shape, (3, 224, 224))
        self.assertEqual(transformed.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
