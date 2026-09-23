"""
Operational Metadata Layer for Solar Photovoltaic Panel Inspection Records.

This module provides data structures and validation logic for associating
visual inspection results with operational metadata:
  1. Panel ID: Unique identifier for the module (e.g. "SP-HYD-001").
  2. Location: Site/rooftop description provided by the operator (e.g. "Block A - Rooftop 1").
  3. Inspection Timestamp: ISO 8601 timestamp recording when the inspection occurred.

IMPORTANT ARCHITECTURAL & OPERATIONAL BOUNDARIES:
- Metadata is strictly operational and administrative.
- Panel ID and Location are NOT model inputs or features for EfficientNet-B0.
- They do NOT affect model predictions or visual severity calculations.
- Location is an operator-assigned tag, NOT a hardware GPS coordinate or physical telemetry.
- These fields prepare the inspection results for persistence in relational databases (e.g. PostgreSQL).
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional


class PanelMetadata:
    """
    Validates and stores operational metadata for a solar panel inspection.
    """

    def __init__(
        self,
        panel_id: str,
        location: str,
        inspection_timestamp: Optional[str] = None,
    ):
        """
        Initializes and validates panel inspection metadata.

        Args:
            panel_id: Required non-empty string identifier for the panel.
            location: Required non-empty string describing the site/array location.
            inspection_timestamp: Optional ISO 8601 timestamp string. If None,
                                  the current UTC time in ISO 8601 format is generated.

        Raises:
            TypeError: If panel_id or location is not a string.
            ValueError: If panel_id or location is empty after stripping whitespace,
                        or if inspection_timestamp is invalid ISO 8601.
        """
        if not isinstance(panel_id, str):
            raise TypeError(f"panel_id must be a string, got {type(panel_id).__name__}")
        stripped_id = panel_id.strip()
        if not stripped_id:
            raise ValueError("panel_id cannot be empty or whitespace only.")
        self.panel_id = stripped_id

        if not isinstance(location, str):
            raise TypeError(f"location must be a string, got {type(location).__name__}")
        stripped_loc = location.strip()
        if not stripped_loc:
            raise ValueError("location cannot be empty or whitespace only.")
        self.location = stripped_loc

        if inspection_timestamp is not None:
            if not isinstance(inspection_timestamp, str):
                raise TypeError(
                    f"inspection_timestamp must be a string, got {type(inspection_timestamp).__name__}"
                )
            stripped_ts = inspection_timestamp.strip()
            if not stripped_ts:
                raise ValueError("inspection_timestamp cannot be empty string.")
            # Validate ISO 8601 parseability
            try:
                # Handle 'Z' if present for compatibility across Python versions
                iso_to_parse = stripped_ts.replace("Z", "+00:00")
                datetime.fromisoformat(iso_to_parse)
            except Exception as e:
                raise ValueError(
                    f"Invalid ISO 8601 timestamp format: '{inspection_timestamp}'. Error: {e}"
                )
            self.inspection_timestamp = stripped_ts
        else:
            self.inspection_timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, str]:
        """Returns metadata as a dictionary."""
        return {
            "panel_id": self.panel_id,
            "location": self.location,
            "inspection_timestamp": self.inspection_timestamp,
        }

    def __repr__(self) -> str:
        return (
            f"PanelMetadata(panel_id='{self.panel_id}', "
            f"location='{self.location}', "
            f"inspection_timestamp='{self.inspection_timestamp}')"
        )


def create_inspection_record(
    metadata: PanelMetadata,
    diagnostic_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merges operational panel metadata with the diagnostic inspection result.

    Expected Schema:
    {
        "panel_id": "...",
        "location": "...",
        "inspection_timestamp": "...",
        "image_filename": "...",
        "true_class": "...",
        "predicted_class": "...",
        "confidence": ...,
        "visual_region_area_percent": ...,
        "severity": "...",
        "urgency": "...",
        "maintenance_action": "...",
        "manual_inspection_recommended": ...,
        "confidence_warning": "..."
    }

    Args:
        metadata: Validated PanelMetadata instance.
        diagnostic_result: Result dictionary from Phase 11 inspection pipeline.

    Returns:
        Structured inspection record dictionary conforming to the required schema.
    """
    meta_dict = metadata.to_dict()

    # Accommodate naming variations between phase outputs if any
    image_fname = (
        diagnostic_result.get("image_filename")
        or diagnostic_result.get("filename")
        or "unknown_image.jpg"
    )
    true_cls = diagnostic_result.get("true_class", "Unknown")
    pred_cls = diagnostic_result.get("predicted_class", "Unknown")
    conf = float(diagnostic_result.get("confidence", 0.0))
    area_pct = float(
        diagnostic_result.get("visual_region_area_percent")
        or diagnostic_result.get("region_area_percent")
        or 0.0
    )
    severity = diagnostic_result.get("severity", "LOW")
    urgency = diagnostic_result.get("urgency", "ROUTINE")
    action = (
        diagnostic_result.get("maintenance_action")
        or diagnostic_result.get("recommended_action")
        or "continue routine monitoring"
    )
    manual_rec = bool(diagnostic_result.get("manual_inspection_recommended", False))
    warning = diagnostic_result.get("confidence_warning", None)

    return {
        "panel_id": meta_dict["panel_id"],
        "location": meta_dict["location"],
        "inspection_timestamp": meta_dict["inspection_timestamp"],
        "image_filename": image_fname,
        "true_class": true_cls,
        "predicted_class": pred_cls,
        "confidence": round(conf, 4),
        "visual_region_area_percent": round(area_pct, 2),
        "severity": severity,
        "urgency": urgency,
        "maintenance_action": action,
        "manual_inspection_recommended": manual_rec,
        "confidence_warning": warning,
    }
