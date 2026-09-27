"""
ML Inspection Service for FastAPI Backend.

Loads and caches the EfficientNet-B0 diagnostic pipeline once at startup and
executes canonical inference, Grad-CAM, fault-region extraction, severity estimation,
and maintenance recommendations for incoming HTTP requests.
"""

from datetime import datetime, timezone
import hashlib
import io
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from src.explainability.gradcam import GradCAM
from src.explainability.fault_region import FaultRegionExtractor
from src.severity.severity_estimator import SeverityEstimator
from src.maintenance.maintenance_recommender import MaintenanceRecommender
from src.models.efficientnet import build_efficientnet_b0
from src.preprocessing.augmentation import get_test_pipeline
from src.utils.config import get_project_root, load_config
from src.utils.logger import setup_logger

logger = setup_logger("api_inspection_service")

EXPECTED_CHECKPOINT_SHA256 = "07890dc9964f5162b53ed4c80778ef147c01b977e8bce76d0a15b09c4733fa1e"
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
MIN_RESOLUTION_SHORT_EDGE = 32
BLUR_LAPLACIAN_VARIANCE_THRESHOLD = 50.0
DARKNESS_MEAN_LUMINANCE_THRESHOLD = 25.0
BRIGHTNESS_MEAN_LUMINANCE_THRESHOLD = 245.0


class InspectionService:
    """
    Thread-safe singleton service managing the cached EfficientNet-B0 inspection pipeline.
    """

    _instance: Optional["InspectionService"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.root = get_project_root()
        self.config = load_config()
        self.classes: List[str] = sorted(self.config.get("dataset", {}).get("classes", []))
        self.device = torch.device("cpu")

        ckpt_path = self.root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        # Checkpoint SHA256 integrity validation
        with open(ckpt_path, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        if actual_sha != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError(
                f"Checkpoint integrity violation! Expected {EXPECTED_CHECKPOINT_SHA256}, got {actual_sha}"
            )
        logger.info(f"Loaded baseline checkpoint (SHA256 verified: {actual_sha[:16]}...)")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model: nn.Module = build_efficientnet_b0(
            num_classes=len(self.classes), pretrained=False, dropout=0.2
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.pipeline = get_test_pipeline()
        self.extractor = FaultRegionExtractor(default_threshold=0.60, min_area_pixels=25)
        self.estimator = SeverityEstimator()
        self.recommender = MaintenanceRecommender(low_confidence_threshold=0.60)
        self._inference_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "InspectionService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def validate_and_decode_image(self, image_bytes: bytes) -> Tuple[np.ndarray, str]:
        """
        Validates the uploaded file can be read and decoded as a supported image.

        Raises:
            ValueError: If the file is unreadable, unsupported, or structurally invalid.
        """
        if not image_bytes or len(image_bytes) < 16:
            raise ValueError("Image could not be read. File is not a valid or readable image.")

        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.verify()
            pil_img = Image.open(io.BytesIO(image_bytes))
        except Exception:
            raise ValueError("Image could not be read. File is not a valid or readable image.")

        fmt = (pil_img.format or "").upper()
        if fmt == "MPO":
            fmt = "JPEG"
        if fmt not in ALLOWED_IMAGE_FORMATS:
            raise ValueError("Image could not be read. File is not a valid or readable image.")

        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        rgb_array = np.array(pil_img, dtype=np.uint8)
        if rgb_array.size == 0:
            raise ValueError("Image could not be read. File is not a valid or readable image.")
        return rgb_array, fmt

    def validate_image_quality(self, image_bytes: bytes) -> np.ndarray:
        """
        Lightweight quality gate that blocks clearly unusable inputs before ML inference.

        This is not a solar-panel detector or semantic OOD model. It only rejects images that
        are unreadable, undersized, excessively blurry, or clearly too dark/bright for reliable
        visual inspection.
        """
        rgb_array, _ = self.validate_and_decode_image(image_bytes)

        height, width = rgb_array.shape[:2]
        short_edge = min(height, width)
        if short_edge < MIN_RESOLUTION_SHORT_EDGE:
            raise ValueError("Image resolution is too low for reliable inspection.")

        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        mean_luminance = float(gray.mean())
        if mean_luminance < DARKNESS_MEAN_LUMINANCE_THRESHOLD:
            raise ValueError("Image is too dark for reliable inspection.")
        if mean_luminance > BRIGHTNESS_MEAN_LUMINANCE_THRESHOLD:
            raise ValueError("Image is too bright for reliable inspection.")

        if short_edge >= 96:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_variance = float(laplacian.var())
            if blur_variance < BLUR_LAPLACIAN_VARIANCE_THRESHOLD:
                raise ValueError("Image appears excessively blurry.")

        return rgb_array

    def run_inspection(
        self,
        image_bytes: bytes,
        filename: str,
    ) -> Dict[str, Any]:
        """
        Executes the full canonical inspection pipeline:
        Preprocessing -> EfficientNet-B0 -> Grad-CAM -> FaultRegion -> Severity -> Recommendation.
        """
        raw_rgb, _ = self.validate_and_decode_image(image_bytes)

        with self._inference_lock:
            # 1. Canonical preprocessing
            transformed = self.pipeline(image=raw_rgb)
            tensor = transformed["image"].unsqueeze(0).to(self.device)

            # 2. Grad-CAM and Prediction
            with GradCAM(self.model) as cam:
                heatmap, pred_idx, conf = cam.generate_heatmap(tensor, target_size=(224, 224))
            pred_class = self.classes[pred_idx]

        # 3. Fault region extraction
        region_info = self.extractor.extract_regions(heatmap)

        # 4. Severity estimation
        severity_info = self.estimator.estimate_severity(
            predicted_class=pred_class,
            confidence=conf,
            region_area_percent=region_info["region_area_percent"],
            mean_activation=region_info["mean_activation"],
            max_activation=region_info["max_activation"],
            num_regions=region_info["num_regions"],
        )

        # 5. Maintenance recommendation
        rec_info = self.recommender.get_recommendation(
            predicted_class=pred_class,
            confidence=conf,
            severity=severity_info["severity"],
            region_area_percent=region_info["region_area_percent"],
            manual_inspection_recommended=severity_info["manual_inspection_recommended"],
        )

        return {
            "image_filename": filename,
            "predicted_class": pred_class,
            "confidence": round(float(conf), 4),
            "visual_region_area_percent": round(float(region_info["region_area_percent"]), 2),
            "severity": rec_info["severity"],
            "urgency": rec_info["urgency"],
            "maintenance_action": rec_info["recommended_action"],
            "maintenance_actions": rec_info["maintenance_actions"],
            "manual_inspection_recommended": rec_info["manual_inspection_recommended"],
            "confidence_warning": rec_info["confidence_warning"],
            "inspection_timestamp": datetime.now(timezone.utc).isoformat(),
        }
