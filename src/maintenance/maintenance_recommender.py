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

from typing import Any, Dict, List, Optional


class MaintenanceRecommender:
    """Rule-based recommender mapping visual diagnostic findings to workflow actions."""

    VALID_URGENCIES = {"ROUTINE", "SCHEDULED", "PRIORITY", "IMMEDIATE_REVIEW"}
    VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH"}

    def __init__(self, low_confidence_threshold: float = 0.60):
        self.low_conf_thresh = low_confidence_threshold

    def _build_action_list(
        self,
        predicted_class: str,
        severity: str,
        manual_flag: bool,
    ) -> List[str]:
        """Return a 2-3 item action list tuned to the predicted fault and severity."""
        cls_name = predicted_class.strip()
        sev = severity.strip().upper()

        if cls_name == "Clean":
            base = [
                "Continue routine visual monitoring of the panel surface.",
                "Keep the panel area maintained according to the site's normal cleaning schedule.",
                "Reinspect periodically for newly visible faults or debris accumulation.",
            ]
            if sev == "HIGH":
                base[0] = "Increase routine monitoring of the panel surface."
            return base[:3]

        if cls_name == "Bird-drop":
            if sev == "LOW":
                return [
                    "Schedule routine cleaning of the affected panel surface.",
                    "Inspect the panel after cleaning for any remaining contamination or damage.",
                    "Recheck the panel during the next standard inspection cycle.",
                ]
            if sev == "MEDIUM":
                return [
                    "Arrange targeted cleaning of the affected area.",
                    "Inspect the panel after cleaning for any remaining soiling or visible damage.",
                    "Schedule a follow-up check to confirm the surface remains clear.",
                ]
            return [
                "Prioritize cleaning of the affected panel area using an approved procedure.",
                "Inspect the cleaned surface for any residual contamination or visible degradation.",
                "Document the cleaning outcome and schedule a follow-up review after maintenance.",
            ]

        if cls_name == "Dusty":
            if sev == "LOW":
                return [
                    "Schedule cleaning of the affected panel surface.",
                    "Inspect the panel after cleaning to confirm that the visibility issue is reduced.",
                    "Monitor the panel during the next standard inspection cycle.",
                ]
            if sev == "MEDIUM":
                return [
                    "Arrange a scheduled cleaning of the affected area.",
                    "Inspect the panel after cleaning to confirm the soiling has been removed.",
                    "Prioritize a follow-up inspection to verify output stability after cleaning.",
                ]
            return [
                "Prioritize cleaning of the affected panel surface.",
                "Inspect the panel after cleaning for any remaining dust or surface degradation.",
                "Document the maintenance outcome and conduct a follow-up inspection after service.",
            ]

        if cls_name == "Snow-Covered":
            if sev == "LOW":
                return [
                    "Arrange safe snow removal using an appropriate site procedure.",
                    "Inspect the panel after snow removal for any visible damage.",
                    "Perform a follow-up inspection once the surface is clear.",
                ]
            if sev == "MEDIUM":
                return [
                    "Arrange a safe snow-clearance procedure for the affected area.",
                    "Inspect the panel after clearing to confirm there is no visible damage or residue.",
                    "Schedule a follow-up review once the module surface is fully clear.",
                ]
            return [
                "Prioritize safe snow-removal and visual review for the affected module.",
                "Inspect the panel immediately after clearing to verify the surface remains undamaged.",
                "Perform a follow-up assessment after the site has returned to normal operating conditions.",
            ]

        if cls_name == "Electrical-damage":
            if sev == "LOW":
                return [
                    "Schedule a qualified electrical technician inspection for the affected panel area.",
                    "Inspect visible module connections and nearby components for signs of electrical damage.",
                    "Document the findings and arrange a follow-up assessment after any corrective work.",
                ]
            if sev == "MEDIUM":
                return [
                    "Prioritize a qualified electrical inspection of the affected panel area.",
                    "Inspect visible module connections and affected components for signs of electrical degradation.",
                    "Document the findings and schedule a follow-up review after corrective action is complete.",
                ]
            return [
                "Arrange an immediate qualified electrical specialist inspection for the affected panel area.",
                "Inspect visible module connections and adjacent components for signs of electrical damage or severe degradation.",
                "Document the findings and perform a follow-up review after corrective work is completed.",
            ]

        if cls_name == "Physical-damage":
            if sev == "LOW":
                return [
                    "Schedule a qualified physical inspection of the affected panel.",
                    "Document the visible damaged area and check for additional surface damage.",
                    "Perform a follow-up assessment to determine whether further maintenance is required.",
                ]
            if sev == "MEDIUM":
                return [
                    "Prioritize a qualified physical inspection of the affected panel.",
                    "Document the visible damaged area and check for additional structural or surface damage.",
                    "Schedule a follow-up assessment to verify whether further maintenance is needed.",
                ]
            return [
                "Arrange an immediate specialist inspection of the affected panel area.",
                "Document the visible damage and verify whether adjacent panel sections show additional deterioration.",
                "Perform a follow-up assessment to confirm whether replacement or further corrective action is required.",
            ]

        return [
            "Schedule a qualified visual inspection of the affected panel.",
            "Inspect the panel area for any additional visible faults or anomalies.",
            "Reassess the condition during the next scheduled inspection cycle.",
        ]

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
                - maintenance_actions: List[str]
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

        if cls_name == "Clean":
            action = "continue routine monitoring"
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
            else:
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
            else:
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
            else:
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
            else:
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
            else:
                action = "immediate specialist inspection recommended"
                urgency = "IMMEDIATE_REVIEW"
                manual_flag = True
                reason = (
                    f"Severe physical damage or shattered glass ({area_pct:.1f}% visual area); "
                    "high risk of moisture ingress and electrical short requiring immediate replacement review."
                )

        else:
            action = "schedule visual verification"
            urgency = "SCHEDULED"
            reason = f"Unrecognized category {cls_name}; manual verification required."
            manual_flag = True

        maintenance_actions = self._build_action_list(cls_name, sev, manual_flag)

        return {
            "predicted_class": cls_name,
            "severity": sev,
            "recommended_action": action,
            "maintenance_actions": maintenance_actions,
            "urgency": urgency,
            "reason": reason,
            "manual_inspection_recommended": manual_flag,
            "confidence_warning": confidence_warning,
        }
