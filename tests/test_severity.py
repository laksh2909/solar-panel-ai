"""
Unit tests for Phase 10: AI-Assisted Visual Severity Estimation.

Verifies:
1. Valid structured output with all required keys.
2. Severity level is strictly in {"LOW", "MEDIUM", "HIGH"}.
3. Severity score is a numeric float bounded in [0.0, 100.0].
4. Area percentage scaling is handled correctly.
5. Clean class is handled as benign/low severity.
6. Low-confidence prediction triggers warning and manual inspection flag.
7. Empty/zero region area is handled safely.
8. All six classes are accepted.
9. Configuration thresholds can be dynamically modified.
10. Checkpoint remains completely unmodified before and after test execution.
"""

import hashlib
from pathlib import Path
import unittest

from src.severity.severity_estimator import SeverityEstimator
from src.utils.config import get_project_root


class TestSeverityEstimator(unittest.TestCase):
    """Test suite for the SeverityEstimator module."""

    def setUp(self):
        self.root = get_project_root()
        self.ckpt_path = self.root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth"
        self.estimator = SeverityEstimator()
        self.classes = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-damage", "Snow-Covered"]

    def test_01_valid_severity_output_structure(self):
        """1. Output contains all expected keys with appropriate data types."""
        out = self.estimator.estimate_severity(
            predicted_class="Electrical-damage",
            confidence=0.95,
            region_area_percent=10.0,
        )
        expected_keys = {
            "predicted_class",
            "confidence",
            "region_area_percent",
            "severity",
            "severity_score",
            "explanation",
            "manual_inspection_recommended",
        }
        self.assertTrue(expected_keys.issubset(out.keys()))
        self.assertIsInstance(out["severity"], str)
        self.assertIsInstance(out["severity_score"], float)
        self.assertIsInstance(out["manual_inspection_recommended"], bool)

    def test_02_severity_level_values(self):
        """2. Severity level is strictly LOW, MEDIUM, or HIGH."""
        for c in self.classes:
            for area in [2.0, 10.0, 25.0]:
                out = self.estimator.estimate_severity(c, confidence=0.85, region_area_percent=area)
                self.assertIn(out["severity"], {"LOW", "MEDIUM", "HIGH"})

    def test_03_score_is_numeric_and_bounded(self):
        """3. Score is a float strictly bounded within [0.0, 100.0]."""
        for area in [0.0, 5.0, 15.0, 50.0, 100.0]:
            for conf in [0.2, 0.5, 0.95]:
                out = self.estimator.estimate_severity("Physical-damage", conf, area)
                self.assertGreaterEqual(out["severity_score"], 0.0)
                self.assertLessEqual(out["severity_score"], 100.0)

    def test_04_area_percentage_scaling(self):
        """4. Area bands map correctly to severity levels for localized faults."""
        low_out = self.estimator.estimate_severity("Bird-drop", 0.90, region_area_percent=5.0)
        med_out = self.estimator.estimate_severity("Bird-drop", 0.90, region_area_percent=10.0)
        high_out = self.estimator.estimate_severity("Bird-drop", 0.90, region_area_percent=20.0)

        self.assertEqual(low_out["severity"], "LOW")
        self.assertEqual(med_out["severity"], "MEDIUM")
        self.assertEqual(high_out["severity"], "HIGH")
        self.assertLess(low_out["severity_score"], med_out["severity_score"])
        self.assertLess(med_out["severity_score"], high_out["severity_score"])

    def test_05_clean_class_handling(self):
        """5. Clean class produces LOW severity and benign score."""
        out = self.estimator.estimate_severity("Clean", confidence=0.98, region_area_percent=12.0)
        self.assertEqual(out["severity"], "LOW")
        self.assertLessEqual(out["severity_score"], 10.0)
        self.assertFalse(out["manual_inspection_recommended"])
        self.assertIn("visually clean", out["explanation"])

    def test_06_low_confidence_triggers_warning(self):
        """6. Low confidence (< 0.60) triggers warning and recommends manual inspection."""
        out = self.estimator.estimate_severity("Bird-drop", confidence=0.45, region_area_percent=4.0)
        self.assertTrue(out["manual_inspection_recommended"])
        self.assertIn("Warning: Low model confidence", out["explanation"])

    def test_07_empty_region_handled_safely(self):
        """7. 0% area is handled safely without error."""
        out = self.estimator.estimate_severity("Dusty", confidence=0.80, region_area_percent=0.0)
        self.assertEqual(out["severity"], "LOW")
        self.assertEqual(out["region_area_percent"], 0.0)

    def test_08_all_six_classes_accepted(self):
        """8. All six classes execute without error."""
        for c in self.classes:
            out = self.estimator.estimate_severity(c, confidence=0.90, region_area_percent=10.0)
            self.assertEqual(out["predicted_class"], c)

    def test_09_configurable_thresholds(self):
        """9. Custom thresholds alter severity boundaries."""
        custom_estimator = SeverityEstimator(localized_low_threshold=5.0, localized_high_threshold=10.0)
        # 7.0% is MEDIUM under custom estimator (since low=5), but LOW under default (low=8)
        default_out = self.estimator.estimate_severity("Bird-drop", 0.90, 7.0)
        custom_out = custom_estimator.estimate_severity("Bird-drop", 0.90, 7.0)
        self.assertEqual(default_out["severity"], "LOW")
        self.assertEqual(custom_out["severity"], "MEDIUM")

    def test_10_checkpoint_integrity_unmodified(self):
        """10. Checkpoint remains unmodified."""
        with open(self.ckpt_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(h, "07890dc9964f5162b53ed4c80778ef147c01b977e8bce76d0a15b09c4733fa1e")


if __name__ == "__main__":
    unittest.main()
