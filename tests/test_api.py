"""
Unit and Integration Tests for Phase 14: FastAPI Backend API.

Tests REST API endpoints:
  - System Health & Root
  - Input Validation (Panel ID, Location, File Types)
  - End-to-End Real ML Inspection (POST /api/inspect)
  - Panel Inventory & Lookup
  - Inspection History & Pagination
  - Checkpoint Invariance
"""

from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import unittest

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.database.database import get_db
from src.database.models import Base
from src.database.repository import create_panel, create_inspection
from src.utils.config import get_project_root


# Enable SQLite Foreign Keys
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class TestFastAPIBackend(unittest.TestCase):
    """Test suite for FastAPI REST endpoints using TestClient and in-memory SQLite."""

    @classmethod
    def setUpClass(cls):
        cls.root = get_project_root()
        cls.ckpt_path = cls.root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth"
        cls.expected_sha256 = "07890dc9964f5162b53ed4c80778ef147c01b977e8bce76d0a15b09c4733fa1e"

        # Locate a representative test image for real inference testing
        test_dir = cls.root / "data" / "test"
        cls.sample_image_path = test_dir / "Bird-drop" / "13.JPG"
        if not cls.sample_image_path.exists():
            # Fallback search
            for p in test_dir.rglob("*.jpg"):
                cls.sample_image_path = p
                break

    def setUp(self):
        # Create isolated in-memory test database
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Override get_db dependency in FastAPI app
        def override_get_db():
            db = self.SessionFactory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def test_01_root_endpoint(self):
        """1. Root endpoint returns service details and links."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("service", data)
        self.assertIn("docs_url", data)

    def test_02_health_endpoint(self):
        """2. Health endpoint returns status 'ok' and service name."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "solar-panel-ai-api")

    def test_03_invalid_panel_metadata_rejected(self):
        """3. Empty panel_id or location returns HTTP 400."""
        dummy_img = io.BytesIO()
        Image.new("RGB", (50, 50), color="blue").save(dummy_img, format="JPEG")
        dummy_img.seek(0)

        # Empty panel_id
        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "   ", "location": "Rooftop 1"},
            files={"file": ("test.jpg", dummy_img.getvalue(), "image/jpeg")},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("panel_id cannot be empty", res.json()["detail"])

        # Empty location
        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "SP-001", "location": ""},
            files={"file": ("test.jpg", dummy_img.getvalue(), "image/jpeg")},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("location cannot be empty", res.json()["detail"])

    def test_04_invalid_image_upload_rejected(self):
        """4. Corrupt file or non-image returns HTTP 400."""
        text_bytes = b"This is not a real image file, just plain text."
        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "SP-001", "location": "Rooftop 1"},
            files={"file": ("malicious.txt", text_bytes, "text/plain")},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("not a valid", res.json()["detail"])

    def test_05_full_ml_inspection_endpoint(self):
        """5. End-to-end inspection with real test image runs full ML pipeline and persists record."""
        self.assertTrue(
            self.sample_image_path.exists(),
            f"Test image not found at {self.sample_image_path}",
        )

        with open(self.sample_image_path, "rb") as f:
            img_bytes = f.read()

        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "SP-TEST-001", "location": "Block A - Rooftop 1"},
            files={"file": (self.sample_image_path.name, img_bytes, "image/jpeg")},
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()

        # Validate response schema
        expected_keys = {
            "inspection_id",
            "panel_id",
            "location",
            "image_filename",
            "inspection_timestamp",
            "predicted_class",
            "confidence",
            "visual_region_area_percent",
            "severity",
            "urgency",
            "maintenance_action",
            "manual_inspection_recommended",
            "confidence_warning",
        }
        self.assertTrue(expected_keys.issubset(set(data.keys())))
        self.assertEqual(data["panel_id"], "SP-TEST-001")
        self.assertEqual(data["location"], "Block A - Rooftop 1")
        self.assertIn(data["severity"], ["LOW", "MEDIUM", "HIGH"])
        self.assertIn(data["urgency"], ["ROUTINE", "SCHEDULED", "PRIORITY", "IMMEDIATE_REVIEW"])
        self.assertTrue(0.0 <= data["confidence"] <= 1.0)
        self.assertTrue(0.0 <= data["visual_region_area_percent"] <= 100.0)

        # Verify record exists in DB
        db = self.SessionFactory()
        insp_id = data["inspection_id"]
        from src.database.repository import get_inspection, get_panel
        db_insp = get_inspection(db, insp_id)
        self.assertIsNotNone(db_insp)
        self.assertEqual(db_insp.panel_id, "SP-TEST-001")
        db_panel = get_panel(db, "SP-TEST-001")
        self.assertIsNotNone(db_panel)
        self.assertEqual(db_panel.location, "Block A - Rooftop 1")
        db.close()

    def test_06_panel_endpoints(self):
        """6. Panel listing, panel lookup, and 404 on missing panel."""
        db = self.SessionFactory()
        create_panel(db, "SP-HYD-001", "Block A")
        create_panel(db, "SP-HYD-002", "Block B")
        db.close()

        # List panels
        res = self.client.get("/api/panels")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 2)
        self.assertEqual(len(data["items"]), 2)

        # Lookup existing panel
        res = self.client.get("/api/panels/SP-HYD-001")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["panel_id"], "SP-HYD-001")
        self.assertEqual(res.json()["location"], "Block A")

        # Missing panel returns 404
        res = self.client.get("/api/panels/SP-DOES-NOT-EXIST")
        self.assertEqual(res.status_code, 404)
        self.assertIn("not found", res.json()["detail"].lower())

    def test_07_panel_inspection_history(self):
        """7. Panel inspection history returns linked records."""
        db = self.SessionFactory()
        create_panel(db, "SP-PANEL-HIST", "Block C")
        create_inspection(
            db,
            panel_id="SP-PANEL-HIST",
            image_filename="sample1.jpg",
            predicted_class="Clean",
            confidence=0.98,
            visual_region_area_percent=2.0,
            severity="LOW",
            urgency="ROUTINE",
            maintenance_action="continue monitoring",
            inspection_timestamp=datetime(2026, 9, 22, 10, 0, tzinfo=timezone.utc),
        )
        db.close()

        res = self.client.get("/api/panels/SP-PANEL-HIST/inspections")
        self.assertEqual(res.status_code, 200)
        items = res.json()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["panel_id"], "SP-PANEL-HIST")
        self.assertEqual(items[0]["predicted_class"], "Clean")

        # Missing panel history returns 404
        res = self.client.get("/api/panels/NON-EXISTENT/inspections")
        self.assertEqual(res.status_code, 404)

    def test_08_inspection_history_and_pagination(self):
        """8. Inspection listing with pagination and single lookup."""
        db = self.SessionFactory()
        create_panel(db, "SP-PAGE", "Block D")
        for i in range(15):
            create_inspection(
                db,
                panel_id="SP-PAGE",
                image_filename=f"img_{i}.jpg",
                predicted_class="Dusty",
                confidence=0.85,
                visual_region_area_percent=12.0,
                severity="MEDIUM",
                urgency="SCHEDULED",
                maintenance_action="wash panel",
                inspection_timestamp=datetime(2026, 9, 22, 10, i, tzinfo=timezone.utc),
            )
        db.close()

        # Page 1, page_size 10
        res = self.client.get("/api/inspections?page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 15)
        self.assertEqual(data["page"], 1)
        self.assertEqual(data["page_size"], 10)
        self.assertEqual(len(data["items"]), 10)

        # Page 2, page_size 10
        res = self.client.get("/api/inspections?page=2&page_size=10")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["items"]), 5)

        # Lookup single inspection
        first_id = data["items"][0]["inspection_id"]
        res = self.client.get(f"/api/inspections/{first_id}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["inspection_id"], first_id)

        # Missing inspection returns 404
        res = self.client.get("/api/inspections/999999")
        self.assertEqual(res.status_code, 404)

    def test_09_checkpoint_integrity_unmodified(self):
        """9. Baseline EfficientNet-B0 checkpoint remains completely unmodified."""
        self.assertTrue(self.ckpt_path.exists(), f"Missing checkpoint: {self.ckpt_path}")
        with open(self.ckpt_path, "rb") as f:
            actual_sha256 = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(
            actual_sha256,
            self.expected_sha256,
            "Baseline checkpoint was modified during FastAPI operations!",
        )


if __name__ == "__main__":
    unittest.main()
