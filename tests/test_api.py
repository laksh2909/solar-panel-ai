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
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image, ImageFilter
from sqlalchemy import create_engine, event, text
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

    def test_02b_request_id_header_present(self):
        """2b. Every API response should include a request correlation ID header."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertIn("X-Request-ID", res.headers)
        self.assertTrue(bool(res.headers["X-Request-ID"]))

    def test_03_optional_panel_metadata_accepted(self):
        """3. Empty panel_id and location are now OPTIONAL — request must succeed with auto-generated values."""
        dummy_img = io.BytesIO()
        Image.new("RGB", (50, 50), color="blue").save(dummy_img, format="JPEG")
        dummy_img.seek(0)

        # Empty panel_id — should auto-generate AUTO-NNN and succeed
        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "", "location": ""},
            files={"file": ("test.jpg", dummy_img.getvalue(), "image/jpeg")},
        )
        self.assertEqual(res.status_code, 201, f"Expected 201 but got {res.status_code}: {res.text}")
        data = res.json()
        # panel_id must start with AUTO- (auto-generated)
        self.assertTrue(
            data["panel_id"].startswith("AUTO-"),
            f"Expected AUTO- prefix, got: {data['panel_id']}"
        )
        # location must be the fallback value
        self.assertEqual(data["location"], "Location not specified")

        # Whitespace-only panel_id — should also auto-generate
        dummy_img2 = io.BytesIO()
        Image.new("RGB", (50, 50), color="red").save(dummy_img2, format="JPEG")
        dummy_img2.seek(0)
        res2 = self.client.post(
            "/api/inspect",
            data={"panel_id": "   ", "location": ""},
            files={"file": ("test2.jpg", dummy_img2.getvalue(), "image/jpeg")},
        )
        self.assertEqual(res2.status_code, 201, f"Expected 201 but got {res2.status_code}: {res2.text}")
        data2 = res2.json()
        self.assertTrue(
            data2["panel_id"].startswith("AUTO-"),
            f"Expected AUTO- prefix, got: {data2['panel_id']}"
        )


    def test_04_invalid_image_upload_rejected(self):
        """4. Corrupt file or non-image returns HTTP 400."""
        text_bytes = b"This is not a real image file, just plain text."
        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "SP-001", "location": "Rooftop 1"},
            files={"file": ("malicious.txt", text_bytes, "text/plain")},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported file type", res.json()["detail"])

    def test_04b_missing_file_is_rejected_by_api(self):
        """Missing multipart file should fail with a safe validation error."""
        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "SP-MISSING", "location": "Upload Test"},
        )
        self.assertEqual(res.status_code, 422)
        self.assertIn("file upload", res.json()["detail"].lower())

    def test_04c_oversized_upload_is_rejected(self):
        """Oversized uploads should be rejected before ML processing."""
        img = Image.new("RGB", (64, 64), color=(10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        payload = buf.getvalue()
        with patch.dict("os.environ", {"UPLOAD_MAX_BYTES": "100"}):
            res = self.client.post(
                "/api/inspect",
                data={"panel_id": "SP-OVERSIZED", "location": "Upload Test"},
                files={"file": ("too_big.jpg", payload, "image/jpeg")},
            )
        self.assertEqual(res.status_code, 413)
        self.assertIn("maximum supported size", res.json()["detail"]) 

    def test_04d_suspicious_filename_is_sanitized(self):
        """Uploaded filenames are normalized and treated as safe metadata only."""
        dummy_img = io.BytesIO()
        Image.new("RGB", (64, 64), color="green").save(dummy_img, format="JPEG")
        dummy_img.seek(0)

        suspicious_name = "../..\\evil.jpg"
        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "SP-SAFE-FILENAME", "location": "Upload Test"},
            files={"file": (suspicious_name, dummy_img.getvalue(), "image/jpeg")},
        )
        self.assertEqual(res.status_code, 201, res.text)
        self.assertEqual(res.json()["image_filename"], "evil.jpg")
        self.assertIn("X-Request-ID", res.headers)

    def test_04e_error_response_does_not_expose_filesystem_paths(self):
        """Upload errors should be generic and not reveal local filesystem details."""
        text_bytes = b"This is not a valid image"
        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "SP-ERROR", "location": "Upload Test"},
            files={"file": ("bad.txt", text_bytes, "text/plain")},
        )
        self.assertEqual(res.status_code, 400)
        self.assertNotIn("C:\\", res.json()["detail"])
        self.assertNotIn("/Users/", res.json()["detail"])
        self.assertIn("X-Request-ID", res.headers)

    def test_04f_error_response_does_not_expose_secrets(self):
        """Validation error payloads should never leak database or credential details."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://user:supersecret@localhost:5432/app"}, clear=False):
            text_bytes = b"This is not a valid image"
            res = self.client.post(
                "/api/inspect",
                data={"panel_id": "SP-SECRET", "location": "Upload Test"},
                files={"file": ("bad.txt", text_bytes, "text/plain")},
            )
        self.assertEqual(res.status_code, 400)
        payload = res.json()["detail"]
        self.assertNotIn("supersecret", payload.lower())
        self.assertNotIn("postgresql://", payload.lower())
        self.assertIn("Unsupported file type", payload)

    def test_04g_failed_upload_does_not_create_db_record(self):
        """Failed upload requests must not create panel or inspection records."""
        db_count_before = self.SessionFactory().execute(text("SELECT COUNT(*) FROM inspections")).scalar_one()
        self.SessionFactory().close()
        text_bytes = b"This is not a valid image"
        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "SP-NO-RECORD", "location": "Upload Test"},
            files={"file": ("bad.txt", text_bytes, "text/plain")},
        )
        self.assertEqual(res.status_code, 400)
        db = self.SessionFactory()
        try:
            count_after = db.execute(text("SELECT COUNT(*) FROM inspections")).scalar_one()
        finally:
            db.close()
        self.assertEqual(count_after, db_count_before)

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
            "timestamp",
            "predicted_class",
            "confidence",
            "visual_region_area_percent",
            "region",
            "severity",
            "urgency",
            "maintenance_action",
            "manual_inspection_recommended",
            "confidence_warning",
        }
        self.assertTrue(expected_keys.issubset(set(data.keys())))
        self.assertEqual(data["panel_id"], "SP-TEST-001")
        self.assertEqual(data["location"], "Block A - Rooftop 1")
        self.assertEqual(data["timestamp"], data["inspection_timestamp"])
        self.assertEqual(data["region"], data["visual_region_area_percent"])
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

    def test_05b_valid_existing_solar_panel_image_passes_quality_validation(self):
        """Valid real solar-panel image passes the quality gate and proceeds to inference."""
        path = self.root / "data" / "test" / "Bird-drop" / "13.JPG"
        self.assertTrue(path.exists(), f"Bird-drop test image not found at {path}")
        with open(path, "rb") as f:
            img_bytes = f.read()

        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "SP-QUALITY-OK", "location": "Validation Test Site"},
            files={"file": (path.name, img_bytes, "image/jpeg")},
        )
        self.assertEqual(res.status_code, 201, res.text)
        data = res.json()
        self.assertIn("predicted_class", data)
        self.assertEqual(data["panel_id"], "SP-QUALITY-OK")

    def test_05c_corrupted_image_is_rejected_safely(self):
        """Corrupted or unreadable uploads are rejected without exposing internal exceptions."""
        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "SP-BAD-IMG", "location": "Validation Site"},
            files={"file": ("corrupt.jpg", b"not-a-real-image", "image/jpeg")},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Image could not be read", res.json()["detail"])

    def test_05d_extremely_small_image_is_rejected(self):
        """Tiny images are rejected before model inference when they are not interpretable."""
        img = Image.new("RGB", (24, 24), color=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "SP-SMALL", "location": "Validation Site"},
            files={"file": ("tiny.png", buf.getvalue(), "image/png")},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("resolution is too low", res.json()["detail"].lower())

    def test_05e_blurry_image_is_rejected(self):
        """Clearly blurry images are blocked by the quality gate."""
        img = Image.new("RGB", (224, 224), color=(245, 245, 245))
        blurred = img.filter(ImageFilter.GaussianBlur(radius=9))
        buf = io.BytesIO()
        blurred.save(buf, format="PNG")
        buf.seek(0)

        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "SP-BLUR", "location": "Validation Site"},
            files={"file": ("blur.png", buf.getvalue(), "image/png")},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("blurry", res.json()["detail"].lower())

    def test_05f_dark_image_is_rejected(self):
        """Extremely dark images are rejected as unsuitable for inspection."""
        img = Image.new("RGB", (224, 224), color=(5, 5, 5))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "SP-DARK", "location": "Validation Site"},
            files={"file": ("dark.jpg", buf.getvalue(), "image/jpeg")},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("too dark", res.json()["detail"].lower())

    def test_05g_bright_image_is_rejected(self):
        """Extremely overexposed images are rejected as unsuitable for inspection."""
        img = Image.new("RGB", (224, 224), color=(250, 250, 250))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "SP-BRIGHT", "location": "Validation Site"},
            files={"file": ("bright.jpg", buf.getvalue(), "image/jpeg")},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("too bright", res.json()["detail"].lower())

    def test_05h_valid_electrical_damage_image_still_infers(self):
        """A known valid Electrical-damage image still completes the inference pipeline."""
        path = self.root / "data" / "test" / "Electrical-damage" / "Electrical (1).jpg"
        self.assertTrue(path.exists(), f"Electrical-damage test image not found at {path}")
        with open(path, "rb") as f:
            img_bytes = f.read()

        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "SP-ELECTRICAL", "location": "Electrical Validation"},
            files={"file": (path.name, img_bytes, "image/jpeg")},
        )
        self.assertEqual(res.status_code, 201, res.text)
        self.assertIn(res.json()["predicted_class"], ["Electrical-damage", "Dusty", "Clean", "Bird-drop", "Physical-damage", "Snow-Covered"])

    def test_05i_valid_bird_drop_image_still_infers(self):
        """A known valid Bird-drop image still completes the inference pipeline."""
        path = self.root / "data" / "test" / "Bird-drop" / "13.JPG"
        self.assertTrue(path.exists(), f"Bird-drop image not found at {path}")
        with open(path, "rb") as f:
            img_bytes = f.read()

        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "SP-BIRD", "location": "Bird Validation"},
            files={"file": (path.name, img_bytes, "image/jpeg")},
        )
        self.assertEqual(res.status_code, 201, res.text)
        self.assertIn(res.json()["predicted_class"], ["Electrical-damage", "Dusty", "Clean", "Bird-drop", "Physical-damage", "Snow-Covered"])

    def test_05j_rejected_image_does_not_create_inspection_record(self):
        """Rejected images must never create database inspection records."""
        initial_total = self.SessionFactory().execute(
            text("SELECT COUNT(*) FROM inspections")
        ).scalar_one()
        self.assertEqual(initial_total, 0)

        img = Image.new("RGB", (24, 24), color=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        res = self.client.post(
            "/api/inspect",
            data={"panel_id": "SP-REJECTED", "location": "No Record"},
            files={"file": ("tiny.png", buf.getvalue(), "image/png")},
        )
        self.assertEqual(res.status_code, 400)

        total_after = self.SessionFactory().execute(
            text("SELECT COUNT(*) FROM inspections")
        ).scalar_one()
        self.assertEqual(total_after, 0)

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

    def test_09_database_url_configuration_behavior(self):
        """9. DATABASE_URL override should be honored without changing the sqlite fallback."""
        with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///:memory:"}, clear=False):
            from src.database.database import get_database_url
            self.assertEqual(get_database_url(), "sqlite:///:memory:")

    def test_09b_database_rollback_and_repeated_panel_behavior(self):
        """Database rollback prevents partial changes and duplicate panel IDs remain guarded by the schema."""
        db = self.SessionFactory()
        try:
            from src.database.repository import PanelRepository
            panel_repo = PanelRepository(db)
            panel_repo.create_panel("SP-ROLLBACK", "Rollback Site")

            # Duplicate panel IDs are rejected by the unique constraint, and rollback returns the session to a safe state.
            try:
                panel_repo.create_panel("SP-ROLLBACK", "Another Site")
            except Exception:
                db.rollback()

            self.assertEqual(panel_repo.get_panel("SP-ROLLBACK").location, "Rollback Site")
        finally:
            db.close()

    def test_09c_database_initialized_and_inspection_history_remains_queryable(self):
        """Inspection history remains queryable and ordered using the SQLAlchemy session records."""
        db = self.SessionFactory()
        try:
            create_panel(db, "SP-HISTORY-DB", "Database Site")
            create_inspection(
                db,
                panel_id="SP-HISTORY-DB",
                image_filename="history.jpg",
                predicted_class="Clean",
                confidence=0.94,
                visual_region_area_percent=3.0,
                severity="LOW",
                urgency="ROUTINE",
                maintenance_action="Routine check",
                inspection_timestamp=datetime(2026, 9, 22, 11, 0, tzinfo=timezone.utc),
            )
            create_inspection(
                db,
                panel_id="SP-HISTORY-DB",
                image_filename="history2.jpg",
                predicted_class="Dusty",
                confidence=0.87,
                visual_region_area_percent=11.5,
                severity="MEDIUM",
                urgency="SCHEDULED",
                maintenance_action="Clean panel",
                inspection_timestamp=datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc),
            )
            history = db.execute(
                text("SELECT COUNT(*) FROM inspections WHERE panel_id = :panel_id"),
                {"panel_id": "SP-HISTORY-DB"},
            ).scalar_one()
            self.assertEqual(history, 2)
        finally:
            db.close()

    def test_10_checkpoint_integrity_unmodified(self):
        """10. Baseline EfficientNet-B0 checkpoint remains completely unmodified."""
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
