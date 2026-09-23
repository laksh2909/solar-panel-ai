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
        Validates image magic bytes, format, and decodes to RGB uint8 numpy array.

        Raises:
            ValueError: If file is corrupted or format is unsupported.
        """
        if not image_bytes or len(image_bytes) < 16:
            raise ValueError("Uploaded file is empty or too small to be a valid image.")

        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.verify()  # Verifies file integrity
            # Re-open for decoding after verify()
            pil_img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"File is not a valid or readable image: {e}")

        fmt = (pil_img.format or "").upper()
        # Normalise MPO / JPEG
        if fmt == "MPO":
            fmt = "JPEG"
        if fmt not in ALLOWED_IMAGE_FORMATS:
            raise ValueError(
                f"Unsupported image format: '{fmt}'. Allowed: {', '.join(sorted(ALLOWED_IMAGE_FORMATS))}"
            )

        # Ensure RGB
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        rgb_array = np.array(pil_img, dtype=np.uint8)
        return rgb_array, fmt

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
            "manual_inspection_recommended": rec_info["manual_inspection_recommended"],
            "confidence_warning": rec_info["confidence_warning"],
            "inspection_timestamp": datetime.now(timezone.utc).isoformat(),
        }
