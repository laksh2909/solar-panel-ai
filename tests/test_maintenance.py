"""
Unit tests for Phase 11: AI-Assisted Maintenance Recommendation Module.

Verifies:
1. Valid output schema and expected keys.
2. All six classes are supported.
3. All three severity levels (LOW, MEDIUM, HIGH) are supported.
4. Urgency levels belong to {"ROUTINE", "SCHEDULED", "PRIORITY", "IMMEDIATE_REVIEW"}.
5. Bird-drop rules match specification.
6. Dusty rules match specification.
7. Snow-Covered rules match specification.
8. Electrical-damage rules match specification (including IMMEDIATE_REVIEW on HIGH).
9. Physical-damage rules match specification (including IMMEDIATE_REVIEW on HIGH).
10. Clean class handling defaults to ROUTINE monitoring.
11. Low-confidence predictions (< 0.60) trigger confidence warning and manual inspection.
12. Electrical and Physical damage trigger specialist/manual inspection on MEDIUM and HIGH.
13. Checkpoint SHA256 remains completely unmodified.
"""

import hashlib
from pathlib import Path
import unittest

from src.maintenance.maintenance_recommender import MaintenanceRecommender
from src.utils.config import get_project_root


class TestMaintenanceRecommender(unittest.TestCase):
    """Test suite for the MaintenanceRecommender engine."""

    def setUp(self):
        self.root = get_project_root()
        self.ckpt_path = self.root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth"
        self.recommender = MaintenanceRecommender()
        self.classes = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-damage", "Snow-Covered"]
        self.severities = ["LOW", "MEDIUM", "HIGH"]
        self.valid_urgencies = {"ROUTINE", "SCHEDULED", "PRIORITY", "IMMEDIATE_REVIEW"}

    def test_01_valid_output_schema(self):
        """1. Output dictionary matches exact schema and types."""
        res = self.recommender.get_recommendation("Electrical-damage", 0.95, "HIGH", 18.0)
        expected_keys = {
            "predicted_class",
            "severity",
            "recommended_action",
            "urgency",
            "reason",
            "manual_inspection_recommended",
            "confidence_warning",
        }
        self.assertTrue(expected_keys.issubset(res.keys()))
        self.assertIsInstance(res["recommended_action"], str)
        self.assertIsInstance(res["urgency"], str)
        self.assertIsInstance(res["reason"], str)
        self.assertIsInstance(res["manual_inspection_recommended"], bool)

    def test_02_all_six_classes_accepted(self):
        """2. All six classes evaluate cleanly."""
        for c in self.classes:
            res = self.recommender.get_recommendation(c, 0.90, "LOW", 5.0)
            self.assertEqual(res["predicted_class"], c)
            self.assertIn(res["urgency"], self.valid_urgencies)

    def test_03_all_severities_produce_valid_urgency(self):
        """3. All severities map to valid urgency levels."""
        for c in self.classes:
            for s in self.severities:
                res = self.recommender.get_recommendation(c, 0.85, s, 10.0)
                self.assertIn(res["urgency"], self.valid_urgencies)

    def test_04_bird_drop_rules(self):
        """4. Bird-drop escalation rules."""
        low = self.recommender.get_recommendation("Bird-drop", 0.90, "LOW", 5.0)
        med = self.recommender.get_recommendation("Bird-drop", 0.90, "MEDIUM", 10.0)
        high = self.recommender.get_recommendation("Bird-drop", 0.90, "HIGH", 20.0)

        self.assertEqual(low["urgency"], "ROUTINE")
        self.assertEqual(low["recommended_action"], "schedule routine cleaning")
        self.assertEqual(med["urgency"], "SCHEDULED")
        self.assertEqual(med["recommended_action"], "clean panel and visually inspect surface")
        self.assertEqual(high["urgency"], "PRIORITY")
        self.assertEqual(high["recommended_action"], "prioritize cleaning and inspection")

    def test_05_dusty_rules(self):
        """5. Dusty escalation rules."""
        low = self.recommender.get_recommendation("Dusty", 0.90, "LOW", 5.0)
        med = self.recommender.get_recommendation("Dusty", 0.90, "MEDIUM", 15.0)
        high = self.recommender.get_recommendation("Dusty", 0.90, "HIGH", 30.0)

        self.assertEqual(low["urgency"], "ROUTINE")
        self.assertEqual(med["urgency"], "SCHEDULED")
        self.assertEqual(high["urgency"], "PRIORITY")

    def test_06_snow_covered_rules(self):
        """6. Snow-Covered escalation rules."""
        low = self.recommender.get_recommendation("Snow-Covered", 0.90, "LOW", 5.0)
        med = self.recommender.get_recommendation("Snow-Covered", 0.90, "MEDIUM", 15.0)
        high = self.recommender.get_recommendation("Snow-Covered", 0.90, "HIGH", 30.0)

        self.assertEqual(low["urgency"], "ROUTINE")
        self.assertEqual(med["urgency"], "SCHEDULED")
        self.assertEqual(high["urgency"], "PRIORITY")

    def test_07_electrical_damage_rules(self):
        """7. Electrical-damage escalation rules (including IMMEDIATE_REVIEW)."""
        low = self.recommender.get_recommendation("Electrical-damage", 0.90, "LOW", 5.0)
        med = self.recommender.get_recommendation("Electrical-damage", 0.90, "MEDIUM", 10.0)
        high = self.recommender.get_recommendation("Electrical-damage", 0.90, "HIGH", 20.0)

        self.assertEqual(low["urgency"], "SCHEDULED")
        self.assertEqual(med["urgency"], "PRIORITY")
        self.assertEqual(high["urgency"], "IMMEDIATE_REVIEW")
        self.assertTrue(med["manual_inspection_recommended"])
        self.assertTrue(high["manual_inspection_recommended"])

    def test_08_physical_damage_rules(self):
        """8. Physical-damage escalation rules (including IMMEDIATE_REVIEW)."""
        low = self.recommender.get_recommendation("Physical-damage", 0.90, "LOW", 5.0)
        med = self.recommender.get_recommendation("Physical-damage", 0.90, "MEDIUM", 10.0)
        high = self.recommender.get_recommendation("Physical-damage", 0.90, "HIGH", 20.0)

        self.assertEqual(low["urgency"], "SCHEDULED")
        self.assertEqual(med["urgency"], "PRIORITY")
        self.assertEqual(high["urgency"], "IMMEDIATE_REVIEW")
        self.assertTrue(med["manual_inspection_recommended"])
        self.assertTrue(high["manual_inspection_recommended"])

    def test_09_clean_handling(self):
        """9. Clean module produces ROUTINE monitoring."""
        res = self.recommender.get_recommendation("Clean", 0.98, "LOW", 0.0)
        self.assertEqual(res["urgency"], "ROUTINE")
        self.assertFalse(res["manual_inspection_recommended"])
        self.assertIn("continue routine monitoring", res["recommended_action"])

    def test_10_low_confidence_warning_behavior(self):
        """10. Low confidence (< 0.60) forces warning and manual inspection."""
        res = self.recommender.get_recommendation("Dusty", 0.45, "LOW", 5.0)
        self.assertTrue(res["manual_inspection_recommended"])
        self.assertIsNotNone(res["confidence_warning"])
        self.assertIn("below 60%", res["confidence_warning"])

    def test_11_checkpoint_integrity_unmodified(self):
        """11. Checkpoint remains unmodified."""
        with open(self.ckpt_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(h, "07890dc9964f5162b53ed4c80778ef147c01b977e8bce76d0a15b09c4733fa1e")


if __name__ == "__main__":
    unittest.main()
