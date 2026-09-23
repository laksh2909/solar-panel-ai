"""
AI-Assisted Visual Severity Estimation Module.

Produces transparent, rule-based visual severity estimates combining classifier
confidence and Grad-CAM fault-region spatial characteristics.

IMPORTANT SCIENTIFIC DISCLAIMER:
This module estimates visual severity from model confidence and Grad-CAM-derived
approximate region characteristics. It does NOT measure electrical output,
temperature, crack depth, structural integrity, or actual percentage power loss.
Actual engineering severity requires electrical, thermal, and physical inspection.
"""

from typing import Any, Dict, Optional


class SeverityEstimator:
    """
    Rule-based visual severity estimator for solar panel fault detections.
    """

    # Class categories
    LOCALIZED_FAULT_CLASSES = {"Bird-drop", "Electrical-damage", "Physical-damage"}
    DIFFUSE_SURFACE_CLASSES = {"Dusty", "Snow-Covered"}
    BENIGN_CLASSES = {"Clean"}

    # Base risk multipliers for severity scoring
    CLASS_RISK_WEIGHTS = {
        "Clean": 0.0,
        "Dusty": 0.55,
        "Bird-drop": 0.70,
        "Snow-Covered": 0.75,
        "Physical-damage": 0.90,
        "Electrical-damage": 1.00,
    }

    def __init__(
        self,
        # Localized fault area thresholds (percent of panel surface)
        localized_low_threshold: float = 8.0,
        localized_high_threshold: float = 15.0,
        # Diffuse surface fault area thresholds
        diffuse_low_threshold: float = 12.0,
        diffuse_high_threshold: float = 25.0,
        # Confidence threshold below which manual inspection is flagged
        low_confidence_threshold: float = 0.60,
    ):
        self.localized_low = localized_low_threshold
        self.localized_high = localized_high_threshold
        self.diffuse_low = diffuse_low_threshold
        self.diffuse_high = diffuse_high_threshold
        self.low_conf_thresh = low_confidence_threshold

    def estimate_severity(
        self,
        predicted_class: str,
        confidence: float,
        region_area_percent: float,
        mean_activation: float = 0.0,
        max_activation: float = 0.0,
        num_regions: int = 1,
    ) -> Dict[str, Any]:
        """
        Estimates visual severity based on class, confidence, and activated region area.

        Returns:
            Dict with:
                - predicted_class: str
                - confidence: float
                - region_area_percent: float
                - severity: "LOW" | "MEDIUM" | "HIGH"
                - severity_score: float in [0.0, 100.0]
                - explanation: str
                - manual_inspection_recommended: bool
        """
        cls_name = predicted_class.strip()
        conf = float(np.clip(confidence, 0.0, 1.0)) if "np" in globals() else float(max(0.0, min(1.0, confidence)))
        area_pct = float(max(0.0, min(100.0, region_area_percent)))

        # 1. Clean module handling
        if cls_name in self.BENIGN_CLASSES:
            severity = "LOW"
            severity_score = round(max(0.0, min(20.0, (1.0 - conf) * 20.0)), 1)
            explanation = (
                f"Panel surface is visually clean (confidence: {conf*100:.1f}%). "
                "No visual fault indicators or localized defects detected."
            )
            manual_inspection = bool(conf < self.low_conf_thresh)
            if manual_inspection:
                explanation += f" [Warning: Low prediction confidence (< {self.low_conf_thresh*100:.0f}%) — manual verification recommended.]"

            return {
                "predicted_class": cls_name,
                "confidence": round(conf, 4),
                "region_area_percent": round(area_pct, 2),
                "severity": severity,
                "severity_score": severity_score,
                "explanation": explanation,
                "manual_inspection_recommended": manual_inspection,
            }

        # 2. Localized fault classes: Bird-drop, Electrical-damage, Physical-damage
        if cls_name in self.LOCALIZED_FAULT_CLASSES:
            if area_pct < self.localized_low:
                severity = "LOW"
                area_desc = f"compact focal region (< {self.localized_low}%)"
            elif area_pct < self.localized_high:
                severity = "MEDIUM"
                area_desc = f"moderate focal footprint ({self.localized_low}%-{self.localized_high}%)"
            else:
                severity = "HIGH"
                area_desc = f"extensive visual anomaly footprint (>= {self.localized_high}%)"

            # Escalate Electrical and Physical damage if moderate/high area due to safety/hotspot risks
            if cls_name in {"Electrical-damage", "Physical-damage"} and severity == "MEDIUM":
                # Severe defect categories with moderate area warrant heightened attention
                manual_inspection = True
            elif severity == "HIGH":
                manual_inspection = True
            else:
                manual_inspection = False

            explanation = (
                f"{cls_name} detected with {conf*100:.1f}% confidence occupying {area_pct:.1f}% "
                f"of visual panel area ({area_desc})."
            )

        # 3. Diffuse surface classes: Dusty, Snow-Covered
        elif cls_name in self.DIFFUSE_SURFACE_CLASSES:
            if area_pct < self.diffuse_low:
                severity = "LOW"
                area_desc = f"light/partial surface accumulation (< {self.diffuse_low}%)"
            elif area_pct < self.diffuse_high:
                severity = "MEDIUM"
                area_desc = f"moderate diffuse coverage ({self.diffuse_low}%-{self.diffuse_high}%)"
            else:
                severity = "HIGH"
                area_desc = f"heavy extensive blanket coverage (>= {self.diffuse_high}%)"

            manual_inspection = bool(severity == "HIGH")
            explanation = (
                f"{cls_name} accumulation detected with {conf*100:.1f}% confidence covering "
                f"{area_pct:.1f}% of visual panel area ({area_desc})."
            )

        # 4. Unknown / fallback class
        else:
            if area_pct < 10.0:
                severity = "LOW"
            elif area_pct < 20.0:
                severity = "MEDIUM"
            else:
                severity = "HIGH"
            manual_inspection = True
            explanation = f"Detected anomaly of class {cls_name} covering {area_pct:.1f}% of panel area."

        # Compute transparent numeric score in [0.0, 100.0]
        # Combines class risk weight (0.0 to 1.0) and visual footprint saturation at 30% area
        risk_weight = self.CLASS_RISK_WEIGHTS.get(cls_name, 0.70)
        area_factor = min(1.0, area_pct / 30.0)
        # Saliency intensity factor (blend of mean activation and confidence)
        act_factor = 0.5 * conf + 0.5 * max(0.5, mean_activation)

        raw_score = (0.50 * area_factor + 0.30 * risk_weight + 0.20 * act_factor) * 100.0 * risk_weight
        severity_score = round(float(max(0.0, min(100.0, raw_score))), 1)

        # Confidence warning flag
        if conf < self.low_conf_thresh:
            manual_inspection = True
            explanation += f" [Warning: Low model confidence (< {self.low_conf_thresh*100:.0f}%) — manual inspection recommended.]"

        return {
            "predicted_class": cls_name,
            "confidence": round(conf, 4),
            "region_area_percent": round(area_pct, 2),
            "severity": severity,
            "severity_score": severity_score,
            "explanation": explanation,
            "manual_inspection_recommended": manual_inspection,
        }
