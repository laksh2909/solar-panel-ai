"""
Unit Tests for Phase 13: Solar Panel Inspection Database Layer.

Tests SQLAlchemy models, database initialization, constraints, and repository
CRUD operations using an isolated in-memory SQLite database.
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import unittest

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.models import Base, Panel, Inspection
from src.database.repository import (
    PanelRepository,
    InspectionRepository,
    create_panel,
    get_panel,
    list_panels,
    create_inspection,
    get_inspection,
    list_inspections,
    list_inspections_by_panel,
)
from src.utils.config import get_project_root


# Enable SQLite Foreign Key Enforcement
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class TestDatabaseLayer(unittest.TestCase):
    """Unit test suite for models and repository with isolated in-memory SQLite."""

    @classmethod
    def setUpClass(cls):
        cls.root = get_project_root()
        cls.ckpt_path = cls.root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth"
        cls.expected_sha256 = "07890dc9964f5162b53ed4c80778ef147c01b977e8bce76d0a15b09c4733fa1e"

    def setUp(self):
        """Creates a fresh in-memory database and session for each test."""
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.session = self.SessionFactory()

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def test_01_panel_creation_and_retrieval(self):
        """1. Panel creation succeeds and retrieves expected attributes."""
        panel = create_panel(self.session, panel_id="SP-HYD-001", location="Block A - Rooftop 1")
        self.assertIsNotNone(panel.id)
        self.assertEqual(panel.panel_id, "SP-HYD-001")
        self.assertEqual(panel.location, "Block A - Rooftop 1")

        # Query via repository
        fetched = get_panel(self.session, "SP-HYD-001")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.location, "Block A - Rooftop 1")

    def test_02_unique_panel_id_enforcement(self):
        """2. Duplicate panel_id is rejected by unique constraint."""
        create_panel(self.session, panel_id="SP-HYD-001", location="Block A")
        with self.assertRaises(IntegrityError):
            create_panel(self.session, panel_id="SP-HYD-001", location="Block B")

    def test_03_empty_panel_fields_rejection(self):
        """3. Empty panel_id is rejected by repository; whitespace-only location is rejected by DB constraint."""
        from sqlalchemy.exc import IntegrityError
        with self.assertRaises(ValueError):
            create_panel(self.session, panel_id="", location="Block A")
        # Whitespace-only location is caught by the DB CHECK constraint (ck_panels_location_nonempty)
        # The repository no longer raises ValueError for location; the DB IntegrityError fires instead
        with self.assertRaises(IntegrityError):
            create_panel(self.session, panel_id="SP-001", location="   ")

    def test_04_inspection_creation_and_foreign_key(self):
        """4. Inspection creation links correctly to parent Panel."""
        create_panel(self.session, panel_id="SP-HYD-002", location="Block A")

        ts = datetime(2026, 9, 22, 10, 0, tzinfo=timezone.utc)
        insp = create_inspection(
            self.session,
            panel_id="SP-HYD-002",
            image_filename="13.JPG",
            predicted_class="Bird-drop",
            confidence=0.9263,
            visual_region_area_percent=36.73,
            severity="HIGH",
            urgency="PRIORITY",
            maintenance_action="prioritize cleaning and inspection",
            inspection_timestamp=ts,
            true_class="Bird-drop",
            manual_inspection_recommended=True,
            confidence_warning=None,
        )
        self.assertIsNotNone(insp.id)
        self.assertEqual(insp.panel_id, "SP-HYD-002")
        self.assertEqual(insp.panel.location, "Block A")

        # Foreign key rejection: panel does not exist
        with self.assertRaises(IntegrityError):
            create_inspection(
                self.session,
                panel_id="NON-EXISTENT-PANEL",
                image_filename="dummy.jpg",
                predicted_class="Clean",
                confidence=0.95,
                visual_region_area_percent=5.0,
                severity="LOW",
                urgency="ROUTINE",
                maintenance_action="routine monitoring",
                inspection_timestamp=ts,
            )

    def test_05_confidence_validation(self):
        """5. Confidence values outside [0, 1] are rejected."""
        create_panel(self.session, panel_id="SP-001", location="Block A")
        ts = datetime.now(timezone.utc)

        with self.assertRaises(ValueError):
            create_inspection(
                self.session,
                panel_id="SP-001",
                image_filename="test.jpg",
                predicted_class="Clean",
                confidence=1.25,  # Invalid
                visual_region_area_percent=10.0,
                severity="LOW",
                urgency="ROUTINE",
                maintenance_action="continue monitoring",
                inspection_timestamp=ts,
            )
        with self.assertRaises(ValueError):
            create_inspection(
                self.session,
                panel_id="SP-001",
                image_filename="test.jpg",
                predicted_class="Clean",
                confidence=-0.1,  # Invalid
                visual_region_area_percent=10.0,
                severity="LOW",
                urgency="ROUTINE",
                maintenance_action="continue monitoring",
                inspection_timestamp=ts,
            )

    def test_06_visual_region_area_validation(self):
        """6. Visual region area percent outside [0, 100] is rejected."""
        create_panel(self.session, panel_id="SP-001", location="Block A")
        ts = datetime.now(timezone.utc)

        with self.assertRaises(ValueError):
            create_inspection(
                self.session,
                panel_id="SP-001",
                image_filename="test.jpg",
                predicted_class="Dusty",
                confidence=0.85,
                visual_region_area_percent=105.0,  # Invalid
                severity="LOW",
                urgency="ROUTINE",
                maintenance_action="clean panel",
                inspection_timestamp=ts,
            )

    def test_07_severity_allowed_values(self):
        """7. Only valid severity values ('LOW', 'MEDIUM', 'HIGH') are accepted."""
        create_panel(self.session, panel_id="SP-001", location="Block A")
        ts = datetime.now(timezone.utc)

        for valid_sev in ["LOW", "MEDIUM", "HIGH"]:
            insp = create_inspection(
                self.session,
                panel_id="SP-001",
                image_filename=f"img_{valid_sev}.jpg",
                predicted_class="Clean",
                confidence=0.90,
                visual_region_area_percent=5.0,
                severity=valid_sev,
                urgency="ROUTINE",
                maintenance_action="monitoring",
                inspection_timestamp=ts,
            )
            self.assertEqual(insp.severity, valid_sev)

        with self.assertRaises(ValueError):
            create_inspection(
                self.session,
                panel_id="SP-001",
                image_filename="invalid.jpg",
                predicted_class="Clean",
                confidence=0.90,
                visual_region_area_percent=5.0,
                severity="CRITICAL",  # Invalid
                urgency="ROUTINE",
                maintenance_action="monitoring",
                inspection_timestamp=ts,
            )

    def test_08_urgency_allowed_values(self):
        """8. Only valid urgency values are accepted."""
        create_panel(self.session, panel_id="SP-001", location="Block A")
        ts = datetime.now(timezone.utc)

        valid_urgencies = ["ROUTINE", "SCHEDULED", "PRIORITY", "IMMEDIATE_REVIEW"]
        for idx, valid_urg in enumerate(valid_urgencies):
            insp = create_inspection(
                self.session,
                panel_id="SP-001",
                image_filename=f"img_urg_{idx}.jpg",
                predicted_class="Clean",
                confidence=0.90,
                visual_region_area_percent=5.0,
                severity="LOW",
                urgency=valid_urg,
                maintenance_action="monitoring",
                inspection_timestamp=ts,
            )
            self.assertEqual(insp.urgency, valid_urg)

        with self.assertRaises(ValueError):
            create_inspection(
                self.session,
                panel_id="SP-001",
                image_filename="invalid_urg.jpg",
                predicted_class="Clean",
                confidence=0.90,
                visual_region_area_percent=5.0,
                severity="LOW",
                urgency="URGENT_NOW",  # Invalid
                maintenance_action="monitoring",
                inspection_timestamp=ts,
            )

    def test_09_list_inspections_and_pagination(self):
        """9. list_inspections correctly supports pagination and ordering."""
        create_panel(self.session, panel_id="SP-001", location="Block A")
        for i in range(5):
            create_inspection(
                self.session,
                panel_id="SP-001",
                image_filename=f"sample_{i}.jpg",
                predicted_class="Dusty",
                confidence=0.80,
                visual_region_area_percent=10.0,
                severity="LOW",
                urgency="ROUTINE",
                maintenance_action="wash panel",
                inspection_timestamp=datetime(2026, 9, 22, 10, i, tzinfo=timezone.utc),
            )

        all_insps = list_inspections(self.session)
        self.assertEqual(len(all_insps), 5)
        # Most recent first
        self.assertEqual(all_insps[0].image_filename, "sample_4.jpg")

        paged = list_inspections(self.session, limit=2, offset=0)
        self.assertEqual(len(paged), 2)

    def test_10_list_inspections_by_panel(self):
        """10. list_inspections_by_panel filters correctly by panel_id."""
        create_panel(self.session, panel_id="PANEL-A", location="Site 1")
        create_panel(self.session, panel_id="PANEL-B", location="Site 2")
        ts = datetime.now(timezone.utc)

        create_inspection(
            self.session,
            panel_id="PANEL-A",
            image_filename="a1.jpg",
            predicted_class="Clean",
            confidence=0.99,
            visual_region_area_percent=0.0,
            severity="LOW",
            urgency="ROUTINE",
            maintenance_action="none",
            inspection_timestamp=ts,
        )
        create_inspection(
            self.session,
            panel_id="PANEL-B",
            image_filename="b1.jpg",
            predicted_class="Dusty",
            confidence=0.85,
            visual_region_area_percent=15.0,
            severity="MEDIUM",
            urgency="SCHEDULED",
            maintenance_action="clean",
            inspection_timestamp=ts,
        )

        insps_a = list_inspections_by_panel(self.session, "PANEL-A")
        self.assertEqual(len(insps_a), 1)
        self.assertEqual(insps_a[0].image_filename, "a1.jpg")

        insps_b = list_inspections_by_panel(self.session, "PANEL-B")
        self.assertEqual(len(insps_b), 1)
        self.assertEqual(insps_b[0].image_filename, "b1.jpg")

    def test_11_checkpoint_integrity_unmodified(self):
        """11. Baseline checkpoint SHA256 remains completely unmodified."""
        self.assertTrue(self.ckpt_path.exists(), f"Missing checkpoint: {self.ckpt_path}")
        with open(self.ckpt_path, "rb") as f:
            actual_sha256 = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(
            actual_sha256,
            self.expected_sha256,
            "Baseline checkpoint was modified during database operations!",
        )


if __name__ == "__main__":
    unittest.main()
