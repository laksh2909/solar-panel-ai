"""
Unit Tests for Phase 12: Panel ID + Location Operational Metadata Layer.
"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import unittest

from src.metadata.panel_metadata import PanelMetadata, create_inspection_record
from src.utils.config import get_project_root


class TestPanelMetadata(unittest.TestCase):
    """Test suite for operational panel metadata validation and record merging."""

    @classmethod
    def setUpClass(cls):
        cls.root = get_project_root()
        cls.ckpt_path = cls.root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth"
        cls.expected_sha256 = "07890dc9964f5162b53ed4c80778ef147c01b977e8bce76d0a15b09c4733fa1e"

    def test_01_valid_metadata_creation(self):
        """1. Valid panel metadata creates successfully with correct properties."""
        meta = PanelMetadata(panel_id="SP-HYD-001", location="Block A - Rooftop 1")
        self.assertEqual(meta.panel_id, "SP-HYD-001")
        self.assertEqual(meta.location, "Block A - Rooftop 1")
        self.assertIsInstance(meta.inspection_timestamp, str)
        self.assertTrue(len(meta.inspection_timestamp) > 0)

        meta_dict = meta.to_dict()
        self.assertEqual(meta_dict["panel_id"], "SP-HYD-001")
        self.assertEqual(meta_dict["location"], "Block A - Rooftop 1")
        self.assertEqual(meta_dict["inspection_timestamp"], meta.inspection_timestamp)

    def test_02_whitespace_stripping(self):
        """2. Leading and trailing whitespace is stripped from panel_id and location."""
        meta = PanelMetadata(panel_id="   SP-HYD-002   ", location="   Block B - Ground Array 3   ")
        self.assertEqual(meta.panel_id, "SP-HYD-002")
        self.assertEqual(meta.location, "Block B - Ground Array 3")

    def test_03_empty_panel_id_rejection(self):
        """3. Empty or whitespace-only panel_id is rejected with ValueError."""
        with self.assertRaises(ValueError):
            PanelMetadata(panel_id="", location="Block A")
        with self.assertRaises(ValueError):
            PanelMetadata(panel_id="    ", location="Block A")
        with self.assertRaises(TypeError):
            PanelMetadata(panel_id=None, location="Block A")  # type: ignore

    def test_04_empty_location_rejection(self):
        """4. Empty or whitespace-only location is rejected with ValueError."""
        with self.assertRaises(ValueError):
            PanelMetadata(panel_id="SP-001", location="")
        with self.assertRaises(ValueError):
            PanelMetadata(panel_id="SP-001", location="   ")
        with self.assertRaises(TypeError):
            PanelMetadata(panel_id="SP-001", location=None)  # type: ignore

    def test_05_timestamp_auto_generation(self):
        """5. Timestamp is automatically generated when not provided."""
        meta = PanelMetadata(panel_id="SP-HYD-003", location="Block C")
        self.assertIsNotNone(meta.inspection_timestamp)
        # Verify it can be parsed as ISO format
        dt = datetime.fromisoformat(meta.inspection_timestamp.replace("Z", "+00:00"))
        self.assertIsInstance(dt, datetime)

    def test_06_deterministic_timestamp_acceptance(self):
        """6. Deterministic timestamp is accepted and preserved exactly."""
        fixed_ts = "2026-09-22T12:00:00+00:00"
        meta = PanelMetadata(panel_id="SP-HYD-004", location="Block D", inspection_timestamp=fixed_ts)
        self.assertEqual(meta.inspection_timestamp, fixed_ts)

    def test_07_invalid_iso_timestamp_rejection(self):
        """7. Malformed timestamp string is rejected."""
        with self.assertRaises(ValueError):
            PanelMetadata(panel_id="SP-001", location="Block A", inspection_timestamp="not-a-timestamp")
        with self.assertRaises(ValueError):
            PanelMetadata(panel_id="SP-001", location="Block A", inspection_timestamp="")

    def test_08_create_inspection_record_schema(self):
        """8. Inspection record contains all required fields and correct types."""
        meta = PanelMetadata(
            panel_id="SP-HYD-005",
            location="Block A - Rooftop 2",
            inspection_timestamp="2026-09-22T14:30:00+00:00",
        )
        dummy_diagnostic = {
            "image_filename": "Electrical (1).jpg",
            "true_class": "Electrical-damage",
            "predicted_class": "Electrical-damage",
            "confidence": 0.9396,
            "region_area_percent": 10.08,
            "severity": "MEDIUM",
            "urgency": "PRIORITY",
            "recommended_action": "prioritize electrical inspection by qualified personnel",
            "manual_inspection_recommended": True,
            "confidence_warning": None,
        }
        record = create_inspection_record(meta, dummy_diagnostic)

        expected_keys = {
            "panel_id",
            "location",
            "inspection_timestamp",
            "image_filename",
            "true_class",
            "predicted_class",
            "confidence",
            "visual_region_area_percent",
            "severity",
            "urgency",
            "maintenance_action",
            "manual_inspection_recommended",
            "confidence_warning",
        }
        self.assertEqual(set(record.keys()), expected_keys)
        self.assertEqual(record["panel_id"], "SP-HYD-005")
        self.assertEqual(record["location"], "Block A - Rooftop 2")
        self.assertEqual(record["inspection_timestamp"], "2026-09-22T14:30:00+00:00")
        self.assertEqual(record["image_filename"], "Electrical (1).jpg")
        self.assertEqual(record["predicted_class"], "Electrical-damage")
        self.assertEqual(record["severity"], "MEDIUM")
        self.assertEqual(record["urgency"], "PRIORITY")
        self.assertEqual(record["maintenance_action"], "prioritize electrical inspection by qualified personnel")
        self.assertTrue(record["manual_inspection_recommended"])
        self.assertIsNone(record["confidence_warning"])

    def test_09_metadata_isolation_from_model_features(self):
        """9. Verify metadata is purely operational and does not alter diagnostic fields."""
        meta1 = PanelMetadata(panel_id="SP-AAA-001", location="Site Alpha")
        meta2 = PanelMetadata(panel_id="SP-ZZZ-999", location="Site Omega")

        base_diag = {
            "image_filename": "13.JPG",
            "true_class": "Bird-drop",
            "predicted_class": "Bird-drop",
            "confidence": 0.9263,
            "region_area_percent": 36.73,
            "severity": "HIGH",
            "urgency": "PRIORITY",
            "recommended_action": "prioritize cleaning and inspection",
            "manual_inspection_recommended": True,
            "confidence_warning": None,
        }

        rec1 = create_inspection_record(meta1, base_diag)
        rec2 = create_inspection_record(meta2, base_diag)

        # Operational metadata changes
        self.assertNotEqual(rec1["panel_id"], rec2["panel_id"])
        self.assertNotEqual(rec1["location"], rec2["location"])

        # ML diagnostic fields remain identical
        self.assertEqual(rec1["predicted_class"], rec2["predicted_class"])
        self.assertEqual(rec1["confidence"], rec2["confidence"])
        self.assertEqual(rec1["visual_region_area_percent"], rec2["visual_region_area_percent"])
        self.assertEqual(rec1["severity"], rec2["severity"])
        self.assertEqual(rec1["urgency"], rec2["urgency"])
        self.assertEqual(rec1["maintenance_action"], rec2["maintenance_action"])

    def test_10_checkpoint_integrity_unmodified(self):
        """10. EfficientNet-B0 checkpoint remains completely unmodified."""
        self.assertTrue(self.ckpt_path.exists(), f"Missing checkpoint: {self.ckpt_path}")
        with open(self.ckpt_path, "rb") as f:
            actual_sha256 = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(
            actual_sha256,
            self.expected_sha256,
            "Baseline checkpoint was modified during metadata operations!",
        )

    def test_11_all_12_demo_records_valid(self):
        """11. All 12 demo inspection records contain valid operational metadata and diagnostics."""
        json_path = self.root / "results" / "metrics" / "panel_metadata_examples.json"
        self.assertTrue(json_path.exists(), f"Missing panel metadata JSON: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        self.assertEqual(len(records), 12, f"Expected 12 records, found {len(records)}")

        valid_severities = {"LOW", "MEDIUM", "HIGH"}
        valid_urgencies = {"ROUTINE", "SCHEDULED", "PRIORITY", "IMMEDIATE_REVIEW"}

        for idx, rec in enumerate(records):
            # Panel ID
            self.assertIsInstance(rec["panel_id"], str)
            self.assertTrue(len(rec["panel_id"].strip()) > 0)

            # Location
            self.assertIsInstance(rec["location"], str)
            self.assertTrue(len(rec["location"].strip()) > 0)

            # Timestamp format
            self.assertIsInstance(rec["inspection_timestamp"], str)
            dt = datetime.fromisoformat(rec["inspection_timestamp"].replace("Z", "+00:00"))
            self.assertIsInstance(dt, datetime)

            # Image filename
            self.assertIsInstance(rec["image_filename"], str)
            self.assertTrue(len(rec["image_filename"]) > 0)

            # Severity and Urgency
            self.assertIn(rec["severity"], valid_severities)
            self.assertIn(rec["urgency"], valid_urgencies)

            # Maintenance Action
            self.assertIsInstance(rec["maintenance_action"], str)
            self.assertTrue(len(rec["maintenance_action"].strip()) > 0)

            # Confidence
            self.assertTrue(0.0 <= rec["confidence"] <= 1.0)


if __name__ == "__main__":
    unittest.main()

