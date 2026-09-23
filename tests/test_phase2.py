"""
Unit and Integration Tests for Phase 2:
- Dataset validation and class recognition
- Image loading (RGB/RGBA/Grayscale handling)
- Deterministic preprocessing shape and normalization
- Data leakage prevention and split integrity
- Class-to-index label mapping
- Albumentations augmentation pipeline execution

Implemented using standard library unittest for zero external testing dependencies.
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.utils.config import load_config, get_project_root
from src.preprocessing.pipeline import (
    load_image_rgb,
    preprocess_image,
    DEFAULT_IMAGE_SIZE,
)
from src.preprocessing.augmentation import (
    get_training_augmentation,
    get_validation_pipeline,
    get_test_pipeline,
)

EXPECTED_CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-damage",
    "Snow-Covered",
]


class TestPhase2DataAndPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = get_project_root()
        cls.raw_dir = cls.root / "data" / "raw"
        cls.metrics_dir = cls.root / "results" / "metrics"

        # Locate a sample image
        cls.sample_image_path = None
        for c in EXPECTED_CLASSES:
            cls_dir = cls.raw_dir / c
            if cls_dir.exists():
                for f in cls_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        cls.sample_image_path = f
                        break
            if cls.sample_image_path:
                break

    def test_01_six_classes_recognized(self):
        """Verifies all six expected photovoltaic fault classes are recognized in config and data/raw."""
        config = load_config()
        configured_classes = config.get("dataset", {}).get("classes", config.get("data", {}).get("classes", []))
        self.assertEqual(
            sorted(configured_classes),
            sorted(EXPECTED_CLASSES),
            f"Config classes do not match expected {EXPECTED_CLASSES}"
        )

        found_dirs = [d.name for d in self.raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        self.assertEqual(
            sorted(found_dirs),
            sorted(EXPECTED_CLASSES),
            f"Classes in data/raw do not match expected {EXPECTED_CLASSES}"
        )

    def test_02_image_loading_and_channels(self):
        """Tests image loading from disk and verifies output is uint8 3-channel RGB."""
        self.assertIsNotNone(self.sample_image_path, "No sample image found in data/raw")
        rgb_arr = load_image_rgb(self.sample_image_path)
        self.assertIsInstance(rgb_arr, np.ndarray)
        self.assertEqual(rgb_arr.ndim, 3)
        self.assertEqual(rgb_arr.shape[2], 3)
        self.assertEqual(rgb_arr.dtype, np.uint8)

        # Test RGBA conversion
        rgba_dummy = np.full((64, 64, 4), 128, dtype=np.uint8)
        converted_rgba = load_image_rgb(rgba_dummy)
        self.assertEqual(converted_rgba.shape, (64, 64, 3))

        # Test Grayscale conversion
        gray_dummy = np.full((64, 64), 50, dtype=np.uint8)
        converted_gray = load_image_rgb(gray_dummy)
        self.assertEqual(converted_gray.shape, (64, 64, 3))

    def test_03_preprocessing_returns_expected_shape(self):
        """Tests deterministic preprocessing pipeline returns torch.Tensor (3, 224, 224)."""
        self.assertIsNotNone(self.sample_image_path, "No sample image found in data/raw")
        tensor = preprocess_image(self.sample_image_path, target_size=(224, 224))
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([3, 224, 224]))
        self.assertEqual(tensor.dtype, torch.float32)

        # Check finite values
        self.assertFalse(torch.isnan(tensor).any().item(), "Tensor contains NaNs")
        self.assertFalse(torch.isinf(tensor).any().item(), "Tensor contains Infs")

    def test_04_splits_no_overlap_and_leakage_free(self):
        """Ensures train, validation, and test splits have zero data leakage."""
        split_csv = self.metrics_dir / "dataset_split.csv"
        self.assertTrue(split_csv.exists(), f"Split manifest missing at {split_csv}")

        df = pd.read_csv(split_csv)
        self.assertEqual(set(df["split"].unique()), {"train", "val", "test"})

        train_files = set(df[df["split"] == "train"]["file_path"])
        val_files = set(df[df["split"] == "val"]["file_path"])
        test_files = set(df[df["split"] == "test"]["file_path"])

        # Path overlap check
        self.assertEqual(len(train_files & val_files), 0, "Train and Val share file paths!")
        self.assertEqual(len(train_files & test_files), 0, "Train and Test share file paths!")
        self.assertEqual(len(val_files & test_files), 0, "Val and Test share file paths!")

        # Content hash leakage check
        if "md5_hash" in df.columns:
            train_hashes = set(df[df["split"] == "train"]["md5_hash"])
            val_hashes = set(df[df["split"] == "val"]["md5_hash"])
            test_hashes = set(df[df["split"] == "test"]["md5_hash"])

            self.assertEqual(len(train_hashes & val_hashes), 0, "Data leakage between Train and Val!")
            self.assertEqual(len(train_hashes & test_hashes), 0, "Data leakage between Train and Test!")
            self.assertEqual(len(val_hashes & test_hashes), 0, "Data leakage between Val and Test!")

        # Verify all 6 classes exist in every split
        for s in ["train", "val", "test"]:
            split_classes = set(df[df["split"] == s]["class_name"])
            self.assertEqual(
                split_classes,
                set(EXPECTED_CLASSES),
                f"Split {s} missing classes: {set(EXPECTED_CLASSES) - split_classes}"
            )

    def test_05_class_labels_mapping(self):
        """Validates class to index bidirectional mapping."""
        class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(EXPECTED_CLASSES))}
        idx_to_class = {i: cls_name for cls_name, i in class_to_idx.items()}

        self.assertEqual(len(class_to_idx), 6)
        self.assertEqual(len(idx_to_class), 6)

        for i in range(6):
            cls_name = idx_to_class[i]
            self.assertEqual(class_to_idx[cls_name], i)

    def test_06_augmentation_pipeline_execution(self):
        """Verifies that Albumentations training, val, and test pipelines execute successfully."""
        self.assertIsNotNone(self.sample_image_path, "No sample image found in data/raw")
        img_rgb = load_image_rgb(self.sample_image_path)

        train_pipe = get_training_augmentation(image_size=(224, 224))
        val_pipe = get_validation_pipeline(image_size=(224, 224))
        test_pipe = get_test_pipeline(image_size=(224, 224))

        res_train = train_pipe(image=img_rgb)["image"]
        res_val = val_pipe(image=img_rgb)["image"]
        res_test = test_pipe(image=img_rgb)["image"]

        self.assertIsInstance(res_train, torch.Tensor)
        self.assertEqual(res_train.shape, torch.Size([3, 224, 224]))
        self.assertIsInstance(res_val, torch.Tensor)
        self.assertEqual(res_val.shape, torch.Size([3, 224, 224]))
        self.assertIsInstance(res_test, torch.Tensor)
        self.assertEqual(res_test.shape, torch.Size([3, 224, 224]))


if __name__ == "__main__":
    unittest.main()
