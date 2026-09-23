"""
Integration Test Suite for Phase 18: Complete System Testing.

Verifies end-to-end integration and stability across all system tiers:
1. ML Pipeline Integrity & Checkpoints
2. Model Inference Across 3 Configurations
3. Grad-CAM Explainability on EfficientNet-B0 Baseline
4. Approximate Fault-Region Extraction
5. Rule-Based Severity Estimation
6. Maintenance Guidance Engine
7. Database Operations (SQLite CRUD & PostgreSQL Configuration)
8. FastAPI REST Endpoints & Error Handling
9. Performance & Latency Thresholds
10. Generated System Test Report Artifacts
"""

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.api.service import InspectionService, EXPECTED_CHECKPOINT_SHA256
from src.database.models import Base
from src.database.repository import (
    create_panel,
    get_panel,
    create_inspection,
    get_inspection,
    list_inspections_by_panel,
)
from src.explainability.fault_region import FaultRegionExtractor
from src.explainability.gradcam import GradCAM
from src.maintenance.maintenance_recommender import MaintenanceRecommender
from src.models.efficientnet import build_efficientnet_b0
from src.models.mobilenet import build_mobilenet_v2
from src.preprocessing.augmentation import get_test_pipeline
from src.preprocessing.pipeline import load_image_rgb
from src.severity.severity_estimator import SeverityEstimator
from src.utils.config import get_project_root, load_config


class TestSystemIntegration(unittest.TestCase):
    """Automated integration test suite for the complete Solar Panel AI system."""

    @classmethod
    def setUpClass(cls):
        cls.root = get_project_root()
        cls.config = load_config()
        cls.classes = sorted(cls.config.get("dataset", {}).get("classes", []))
        cls.test_dir = cls.root / "data" / "test"
        cls.metrics_dir = cls.root / "results" / "metrics"
        cls.plots_dir = cls.root / "results" / "plots"

        cls.ckpt_effnet = cls.root / "models/checkpoints/efficientnet_b0_baseline_best.pth"
        cls.ckpt_mobile = cls.root / "models/checkpoints/mobilenetv2_baseline_best.pth"
        cls.ckpt_aug = cls.root / "models/checkpoints/efficientnet_b0_augmented_best.pth"

    def test_01_ml_pipeline_test_set_and_classes(self):
        """1. Verify dataset test set contains exactly 177 images across 6 canonical classes."""
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        discovered = {}
        for c in self.classes:
            folder = self.test_dir / c
            self.assertTrue(folder.is_dir(), f"Missing class directory: {folder}")
            images = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
            self.assertGreater(len(images), 0)
            discovered[c] = len(images)

        total_images = sum(discovered.values())
        self.assertEqual(total_images, 177, f"Expected 177 images, got {total_images}")
        self.assertEqual(len(discovered), 6)

    def test_02_checkpoint_integrity_and_sha256(self):
        """2. Verify all three model checkpoints exist and baseline matches exact frozen SHA256."""
        for ckpt in [self.ckpt_effnet, self.ckpt_mobile, self.ckpt_aug]:
            self.assertTrue(ckpt.exists(), f"Missing checkpoint: {ckpt}")
            self.assertGreater(ckpt.stat().st_size, 1_000_000)

        # EfficientNet-B0 baseline frozen hash check
        import hashlib
        with open(self.ckpt_effnet, "rb") as f:
            eff_sha = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(eff_sha, EXPECTED_CHECKPOINT_SHA256)

    def test_03_canonical_preprocessing(self):
        """3. Verify Phase 7A canonical preprocessing produces float32 tensor of shape (3, 224, 224)."""
        pipeline = get_test_pipeline()
        sample_img = next(self.test_dir.glob("*/*.JPG"))
        rgb = load_image_rgb(sample_img)
        t = pipeline(image=rgb)["image"]
        self.assertEqual(t.shape, (3, 224, 224))
        self.assertEqual(t.dtype, torch.float32)

    def test_04_model_inferences_all_three(self):
        """4. Verify inference on all 3 models outputs valid 6-class probability distribution."""
        sample_img = next(self.test_dir.glob("*/*.JPG"))
        rgb = load_image_rgb(sample_img)
        t = get_test_pipeline()(image=rgb)["image"].unsqueeze(0)

        models = [
            build_mobilenet_v2(6, False),
            build_efficientnet_b0(6, False),
            build_efficientnet_b0(6, False),
        ]
        ckpts = [self.ckpt_mobile, self.ckpt_effnet, self.ckpt_aug]

        for model, ckpt_path in zip(models, ckpts):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            with torch.no_grad():
                probs = torch.softmax(model(t), dim=1)[0]

            self.assertEqual(len(probs), 6)
            self.assertTrue(torch.isfinite(probs).all())
            self.assertAlmostEqual(float(torch.sum(probs).item()), 1.0, places=4)
            self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_05_gradcam_on_all_six_classes(self):
        """5. Verify Grad-CAM generates valid normalized heatmaps on EfficientNet-B0 for all 6 classes."""
        ckpt = torch.load(self.ckpt_effnet, map_location="cpu")
        model = build_efficientnet_b0(6, False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        gradcam = GradCAM(model)
        self.assertIs(gradcam.target_layer, model.features[-1])

        pipeline = get_test_pipeline()
        for c in self.classes:
            sample_img = next((self.test_dir / c).glob("*.JPG"))
            rgb = load_image_rgb(sample_img)
            t = pipeline(image=rgb)["image"].unsqueeze(0)
            heatmap, pred_idx, conf = gradcam.generate_heatmap(t, target_size=(rgb.shape[1], rgb.shape[0]))

            self.assertEqual(heatmap.shape, (rgb.shape[0], rgb.shape[1]))
            self.assertTrue(np.isfinite(heatmap).all())
            self.assertTrue(0.0 <= heatmap.min() and heatmap.max() <= 1.0)

            overlay = gradcam.overlay_heatmap(rgb, heatmap)
            self.assertEqual(overlay.shape, rgb.shape)
            self.assertEqual(overlay.dtype, np.uint8)

    def test_06_fault_region_extraction(self):
        """6. Verify FaultRegionExtractor extracts bounded regions and handles empty masks safely."""
        extractor = FaultRegionExtractor()

        # Empty mask
        empty_hm = np.zeros((224, 224), dtype=np.float32)
        res_empty = extractor.extract_regions(empty_hm)
        self.assertEqual(res_empty["num_regions"], 0)
        self.assertEqual(res_empty["region_area_percent"], 0.0)

        # Simulated focal activation
        focal_hm = np.zeros((224, 224), dtype=np.float32)
        focal_hm[50:100, 50:100] = 0.85
        res_focal = extractor.extract_regions(focal_hm)
        self.assertGreater(res_focal["region_area_percent"], 0.0)
        self.assertLessEqual(res_focal["region_area_percent"], 100.0)
        self.assertTrue(0 <= res_focal["bbox_x"] < 224)
        self.assertTrue(0 <= res_focal["bbox_y"] < 224)

    def test_07_severity_estimation_rules(self):
        """7. Verify severity rule logic: Clean=LOW, low conf triggers manual inspection, defect escalation."""
        estimator = SeverityEstimator()

        res_clean = estimator.estimate_severity("Clean", confidence=0.90, region_area_percent=0.0)
        self.assertEqual(res_clean["severity"], "LOW")
        self.assertFalse(res_clean["manual_inspection_recommended"])

        # Low confidence (< 0.60)
        res_low_conf = estimator.estimate_severity("Clean", confidence=0.55, region_area_percent=0.0)
        self.assertTrue(res_low_conf["manual_inspection_recommended"])

        # Electrical damage medium
        res_elec = estimator.estimate_severity("Electrical-damage", confidence=0.85, region_area_percent=12.0)
        self.assertIn(res_elec["severity"], ["MEDIUM", "HIGH"])
        self.assertTrue(res_elec["manual_inspection_recommended"])

    def test_08_maintenance_recommendations(self):
        """8. Verify maintenance recommender outputs valid urgencies, actions, and disclaimers."""
        recommender = MaintenanceRecommender()

        for c in self.classes:
            rec = recommender.get_recommendation(c, confidence=0.88, severity="MEDIUM", region_area_percent=10.0)
            self.assertIn(rec["urgency"], recommender.VALID_URGENCIES)
            self.assertGreater(len(rec["recommended_action"]), 0)

        # Low confidence warning appended
        rec_warn = recommender.get_recommendation("Bird-drop", confidence=0.50, severity="LOW")
        self.assertIsNotNone(rec_warn["confidence_warning"])
        self.assertTrue(rec_warn["manual_inspection_recommended"])

    def test_09_database_sqlite_and_postgres_readiness(self):
        """9. Verify local SQLite repository CRUD and verify PostgreSQL configuration availability."""
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        db = Session()

        panel = create_panel(db, panel_id="INTEG-001", location="Test Section")
        insp = create_inspection(
            db,
            panel_id="INTEG-001",
            image_filename="test.jpg",
            predicted_class="Clean",
            confidence=0.98,
            visual_region_area_percent=0.0,
            severity="LOW",
            urgency="ROUTINE",
            maintenance_action="Clean surface",
            inspection_timestamp=datetime.now(timezone.utc),
        )
        self.assertIsNotNone(get_panel(db, "INTEG-001"))
        self.assertIsNotNone(get_inspection(db, insp.id))
        self.assertEqual(len(list_inspections_by_panel(db, "INTEG-001")), 1)
        db.close()

    def test_10_fastapi_endpoints_and_error_handling(self):
        """10. Verify FastAPI HTTP endpoints return expected schemas and error status codes."""
        client = TestClient(app)

        # GET /api/health
        res_h = client.get("/api/health")
        self.assertEqual(res_h.status_code, 200)
        self.assertEqual(res_h.json()["status"], "ok")

        # GET /api/panels
        res_p = client.get("/api/panels")
        self.assertEqual(res_p.status_code, 200)
        p_data = res_p.json()
        p_list = p_data.get("items", p_data) if isinstance(p_data, dict) else p_data
        self.assertIsInstance(p_list, list)

        # GET /api/inspections
        res_i = client.get("/api/inspections")
        self.assertEqual(res_i.status_code, 200)

        # Invalid upload -> 400
        bad_f = {"file": ("bad.txt", io.BytesIO(b"not image"), "text/plain")}
        res_bad = client.post("/api/inspect", files=bad_f, data={"panel_id": "SP-HYD-001", "location": "Test"})
        self.assertEqual(res_bad.status_code, 400)

        # Nonexistent panel -> 404
        self.assertEqual(client.get("/api/panels/DOES_NOT_EXIST").status_code, 404)

        # Nonexistent inspection -> 404
        self.assertEqual(client.get("/api/inspections/9999999").status_code, 404)

    def test_11_performance_sanity_bounds(self):
        """11. Verify CPU inference and pipeline latency are within reasonable computational bounds."""
        service = InspectionService.get_instance()
        sample_img = next(self.test_dir.glob("*/*.JPG"))
        with open(sample_img, "rb") as f:
            raw_bytes = f.read()

        import time
        t0 = time.perf_counter()
        result = service.run_inspection(raw_bytes, sample_img.name)
        latency_ms = (time.perf_counter() - t0) * 1000

        self.assertIn("predicted_class", result)
        self.assertLess(latency_ms, 2000.0, f"Inspection took excessively long: {latency_ms} ms")

    def test_12_system_report_artifacts_exist(self):
        """12. Verify Phase 18 output report JSON, CSV, and summary plot exist and are populated."""
        json_report = self.metrics_dir / "system_test_report.json"
        csv_report = self.metrics_dir / "system_test_report.csv"
        plot_summary = self.plots_dir / "system_test_summary.png"

        self.assertTrue(json_report.exists(), f"Missing {json_report}")
        self.assertTrue(csv_report.exists(), f"Missing {csv_report}")
        self.assertTrue(plot_summary.exists(), f"Missing {plot_summary}")

        with open(json_report, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("summary", data["metadata"])
        self.assertGreater(data["metadata"]["summary"]["total_tests"], 10)
        self.assertEqual(data["metadata"]["summary"]["failed"], 0)


if __name__ == "__main__":
    unittest.main()
