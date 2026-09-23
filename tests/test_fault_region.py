"""
Unit tests for Phase 9: Approximate Visual Fault-Region Analysis.

Verifies:
1. Heatmap thresholding works with fixed and adaptive thresholds.
2. Binary mask has valid dimensions matching the input heatmap.
3. Mask contains only binary values ({0, 255}).
4. Connected component analysis correctly identifies distinct regions.
5. Tiny/noisy components below the area cutoff are removed.
6. Bounding box coordinates are valid and stay inside image dimensions.
7. Centroid lies inside image dimensions.
8. Region area in pixels is non-negative and area percentage is in [0, 100].
9. Empty / no-region case is handled safely without throwing exceptions.
10. Canonical preprocessing remains unchanged.
"""

from pathlib import Path
import unittest
import numpy as np
import torch

from src.explainability.fault_region import FaultRegionExtractor
from src.preprocessing.pipeline import load_image_rgb
from src.preprocessing.augmentation import get_test_pipeline
from src.utils.config import get_project_root


class TestFaultRegion(unittest.TestCase):
    """Test suite for approximate visual fault region extraction."""

    def setUp(self):
        self.root = get_project_root()
        self.extractor = FaultRegionExtractor(default_threshold=0.60, min_area_pixels=25)

        # Create a synthetic 224x224 heatmap with a known focal Gaussian spot in the center
        y, x = np.ogrid[:224, :224]
        # Spot 1: Center (112, 112), high activation ~ 1.0
        spot1 = np.exp(-((x - 112) ** 2 + (y - 112) ** 2) / (2 * 18 ** 2))
        # Spot 2: Corner tiny noise (10, 10), small radius ~ 2 pixels
        spot2 = np.exp(-((x - 10) ** 2 + (y - 10) ** 2) / (2 * 1 ** 2))

        self.synthetic_heatmap = np.clip(spot1 + spot2, 0.0, 1.0)
        self.empty_heatmap = np.zeros((224, 224), dtype=np.float32)

    def test_01_heatmap_thresholding_works(self):
        """1. Thresholding produces a binary mask where high activations are 255."""
        mask = self.extractor.threshold_heatmap(self.synthetic_heatmap, threshold=0.60)
        self.assertEqual(mask.shape, (224, 224))
        self.assertTrue(np.all(np.isin(mask, [0, 255])))
        self.assertGreater(np.sum(mask == 255), 0)

        # Adaptive thresholding
        mask_adapt = self.extractor.threshold_heatmap(self.synthetic_heatmap, adaptive=True)
        self.assertEqual(mask_adapt.shape, (224, 224))
        self.assertTrue(np.all(np.isin(mask_adapt, [0, 255])))

    def test_02_mask_dimensions_and_binary_values(self):
        """2 & 3. Mask has valid dimensions and contains only binary values {0, 255}."""
        res = self.extractor.extract_regions(self.synthetic_heatmap, threshold=0.60)
        mask = res["mask"]
        self.assertEqual(mask.shape, (224, 224))
        self.assertEqual(mask.dtype, np.uint8)
        unique_vals = set(np.unique(mask))
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_03_connected_components_and_noise_removal(self):
        """4 & 5. Connected components work and tiny components are filtered out."""
        # Uncleaned mask will detect the center spot and the tiny corner noise
        raw_mask = self.extractor.threshold_heatmap(self.synthetic_heatmap, threshold=0.60)
        cleaned_mask, num_regions = self.extractor.clean_mask(raw_mask, min_area=25)

        # The tiny noise at (10, 10) should have area < 25 and be removed
        self.assertEqual(cleaned_mask[10, 10], 0)
        # Center spot should be retained
        self.assertEqual(cleaned_mask[112, 112], 255)
        self.assertEqual(num_regions, 1)

    def test_04_bounding_box_valid_and_within_bounds(self):
        """6. Bounding box coordinates are valid and stay strictly inside [0, 224]."""
        res = self.extractor.extract_regions(self.synthetic_heatmap, threshold=0.60)
        bx = res["bbox_x"]
        by = res["bbox_y"]
        bw = res["bbox_width"]
        bh = res["bbox_height"]

        self.assertGreaterEqual(bx, 0)
        self.assertGreaterEqual(by, 0)
        self.assertGreater(bw, 0)
        self.assertGreater(bh, 0)
        self.assertLessEqual(bx + bw, 224)
        self.assertLessEqual(by + bh, 224)

    def test_05_centroid_inside_dimensions(self):
        """7. Centroid coordinates lie inside image dimensions."""
        res = self.extractor.extract_regions(self.synthetic_heatmap, threshold=0.60)
        cx = res["centroid_x"]
        cy = res["centroid_y"]

        self.assertGreaterEqual(cx, 0.0)
        self.assertLessEqual(cx, 224.0)
        self.assertGreaterEqual(cy, 0.0)
        self.assertLessEqual(cy, 224.0)

        # Center spot should have centroid close to (112, 112)
        self.assertAlmostEqual(cx, 112.0, delta=3.0)
        self.assertAlmostEqual(cy, 112.0, delta=3.0)

    def test_06_region_area_and_percentage(self):
        """8. Area in pixels is non-negative and percentage is in [0, 100]."""
        res = self.extractor.extract_regions(self.synthetic_heatmap, threshold=0.60)
        area = res["region_area_pixels"]
        area_pct = res["region_area_percent"]

        self.assertGreater(area, 0)
        self.assertLessEqual(area, 224 * 224)
        self.assertGreater(area_pct, 0.0)
        self.assertLessEqual(area_pct, 100.0)

    def test_07_empty_heatmap_case_handled_safely(self):
        """9. Empty / zero heatmap handles safely without throwing exceptions."""
        res = self.extractor.extract_regions(self.empty_heatmap, threshold=0.60)
        self.assertEqual(res["num_regions"], 0)
        self.assertEqual(res["bbox_x"], 0)
        self.assertEqual(res["bbox_y"], 0)
        self.assertEqual(res["bbox_width"], 0)
        self.assertEqual(res["bbox_height"], 0)
        self.assertEqual(res["centroid_x"], 0.0)
        self.assertEqual(res["centroid_y"], 0.0)
        self.assertEqual(res["region_area_pixels"], 0)
        self.assertEqual(res["region_area_percent"], 0.0)
        self.assertEqual(res["mean_activation"], 0.0)

        # Drawing overlay on empty mask should succeed
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        overlay = self.extractor.draw_region_overlay(dummy_img, res["mask"], res, res)
        self.assertEqual(overlay.shape, (224, 224, 3))

    def test_08_canonical_preprocessing_remains_unchanged(self):
        """10. Canonical preprocessing outputs valid [3, 224, 224] tensor."""
        pipeline = get_test_pipeline()
        sample_images = list(self.root.glob("data/test/*/*.JPG"))
        self.assertGreater(len(sample_images), 0)

        img_rgb = load_image_rgb(sample_images[0])
        tensor = pipeline(image=img_rgb)["image"]
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
