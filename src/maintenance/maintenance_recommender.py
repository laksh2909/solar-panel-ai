"""
Maintenance Recommendation Engine for Solar Photovoltaic Modules.

Translates visual fault classifications, model confidence, and visual severity
ratings into actionable, transparent maintenance workflow recommendations.

IMPORTANT SCIENTIFIC & SAFETY DISCLAIMER:
The maintenance recommender provides rule-based AI-assisted workflow guidance
derived from the predicted visual fault class, confidence, and visual severity.
It does NOT diagnose electrical faults, certify safety, determine warranty
eligibility, or prescribe repairs. It does not replace qualified electrical,
thermal, structural, or physical inspection.
"""

from typing import Any, Dict, Optional


class MaintenanceRecommender:
    """
    Rule-based recommender mapping visual diagnostic findings to workflow actions.
    """

    VALID_URGENCIES = {"ROUTINE", "SCHEDULED", "PRIORITY", "IMMEDIATE_REVIEW"}
    VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH"}

    def __init__(
        self,
        low_confidence_threshold: float = 0.60,
    ):
        self.low_conf_thresh = low_confidence_threshold

    def get_recommendation(
        self,
        predicted_class: str,
        confidence: float,
        severity: str,
        region_area_percent: float = 0.0,
        manual_inspection_recommended: bool = False,
    ) -> Dict[str, Any]:
        """
        Generates structured maintenance workflow guidance.

        Args:
            predicted_class: Predicted fault class string.
            confidence: Prediction confidence float in [0, 1].
            severity: Visual severity string ("LOW", "MEDIUM", "HIGH").
            region_area_percent: Approximate activated region area percentage.
            manual_inspection_recommended: Prior manual review recommendation flag.

        Returns:
            Dict containing:
                - predicted_class: str
                - severity: str
                - recommended_action: str
                - urgency: str ("ROUTINE" | "SCHEDULED" | "PRIORITY" | "IMMEDIATE_REVIEW")
                - reason: str
                - manual_inspection_recommended: bool
                - confidence_warning: Optional[str]
        """
        cls_name = predicted_class.strip()
        sev = severity.strip().upper()
        if sev not in self.VALID_SEVERITIES:
            sev = "LOW"

        conf = float(max(0.0, min(1.0, confidence)))
        area_pct = float(max(0.0, min(100.0, region_area_percent)))
        manual_flag = bool(manual_inspection_recommended)

        confidence_warning: Optional[str] = None
        if conf < self.low_conf_thresh:
            confidence_warning = (
                f"Model confidence is below {self.low_conf_thresh*100:.0f}%; "
                "manual inspection is recommended before maintenance action."
            )
            manual_flag = True

        # Rule evaluation per class
        if cls_name == "Clean":
            action = "no fault-specific maintenance indicated; continue routine monitoring"
            urgency = "ROUTINE"
            reason = (
                "Module surface is clean with no active fault signatures; "
                "standard scheduled monitoring remains sufficient."
            )

        elif cls_name == "Bird-drop":
            if sev == "LOW":
                action = "schedule routine cleaning"
                urgency = "ROUTINE"
                reason = (
                    f"Localized minor bird dropping detected ({area_pct:.1f}% visual area); "
                    "can be resolved during standard routine maintenance."
                )
            elif sev == "MEDIUM":
                action = "clean panel and visually inspect surface"
                urgency = "SCHEDULED"
                reason = (
                    f"Moderate bird dropping footprint ({area_pct:.1f}% visual area); "
                    "requires targeted cleaning to prevent localized hot-spot formation."
                )
            else:  # HIGH
                action = "prioritize cleaning and inspection"
                urgency = "PRIORITY"
                manual_flag = True
                reason = (
                    f"Extensive bird fouling ({area_pct:.1f}% visual area); "
                    "significant localized shading risks accelerated cell degradation."
                )

        elif cls_name == "Dusty":
            if sev == "LOW":
                action = "monitor and clean during routine maintenance"
                urgency = "ROUTINE"
                reason = (
                    f"Light surface dust accumulation ({area_pct:.1f}% visual area); "
                    "minimal immediate impact on generation."
                )
            elif sev == "MEDIUM":
                action = "schedule panel cleaning"
                urgency = "SCHEDULED"
                reason = (
                    f"Moderate diffuse dust accumulation ({area_pct:.1f}% visual area); "
                    "washing recommended at next maintenance cycle."
                )
            else:  # HIGH
                action = "prioritize cleaning and follow-up inspection"
                urgency = "PRIORITY"
                manual_flag = True
                reason = (
                    f"Heavy dust accumulation ({area_pct:.1f}% visual area); "
                    "substantial soiling loss requires priority array washing."
                )

        elif cls_name == "Snow-Covered":
            if sev == "LOW":
                action = "monitor snow coverage"
                urgency = "ROUTINE"
                reason = (
                    f"Minor snow cover ({area_pct:.1f}% visual area); "
                    "allow natural melting or standard monitoring."
                )
            elif sev == "MEDIUM":
                action = "schedule safe snow-removal inspection"
                urgency = "SCHEDULED"
                reason = (
                    f"Partial snow accumulation ({area_pct:.1f}% visual area); "
                    "assess array clearing based on site conditions."
                )
            else:  # HIGH
                action = "prioritize safe snow-removal/inspection"
                urgency = "PRIORITY"
                manual_flag = True
                reason = (
                    f"Extensive snow blanket ({area_pct:.1f}% visual area); "
                    "severe generation blockage justifies priority intervention."
                )

        elif cls_name == "Electrical-damage":
            if sev == "LOW":
                action = "schedule electrical inspection"
                urgency = "SCHEDULED"
                reason = (
                    f"Focal electrical discoloration detected ({area_pct:.1f}% visual area); "
                    "schedule diagnostic inspection to verify bypass diode and cell integrity."
                )
            elif sev == "MEDIUM":
                action = "prioritize electrical inspection by qualified personnel"
                urgency = "PRIORITY"
                manual_flag = True
                reason = (
                    f"Moderate electrical burn/hotspot footprint ({area_pct:.1f}% visual area); "
                    "potential safety and performance hazard requiring qualified technician."
                )
            else:  # HIGH
                action = "immediate specialist inspection recommended"
                urgency = "IMMEDIATE_REVIEW"
                manual_flag = True
                reason = (
                    f"Extensive burn marks or severe junction degradation ({area_pct:.1f}% visual area); "
                    "immediate electrical safety hazard."
                )

        elif cls_name == "Physical-damage":
            if sev == "LOW":
                action = "schedule physical inspection"
                urgency = "SCHEDULED"
                reason = (
                    f"Minor glass fracture or surface scratch detected ({area_pct:.1f}% visual area); "
                    "schedule inspection to check crack stability."
                )
            elif sev == "MEDIUM":
                action = "prioritize physical inspection"
                urgency = "PRIORITY"
                manual_flag = True
                reason = (
                    f"Moderate structural fracture or glass crack ({area_pct:.1f}% visual area); "
                    "risk of moisture ingress requiring priority technician review."
                )
            else:  # HIGH
                action = "immediate specialist inspection recommended"
                urgency = "IMMEDIATE_REVIEW"
                manual_flag = True
                reason = (
                    f"Severe physical damage or shattered glass ({area_pct:.1f}% visual area); "
                    "high risk of moisture ingress and electrical short requiring immediate replacement review."
                )

        else:
            # Fallback for unexpected class
            action = "schedule visual verification"
            urgency = "SCHEDULED"
            reason = f"Unrecognized category {cls_name}; manual verification required."
            manual_flag = True

        return {
            "predicted_class": cls_name,
            "severity": sev,
            "recommended_action": action,
            "urgency": urgency,
            "reason": reason,
            "manual_inspection_recommended": manual_flag,
            "confidence_warning": confidence_warning,
        }
