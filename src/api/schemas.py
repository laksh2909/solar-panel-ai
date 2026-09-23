"""
Pydantic Schemas for FastAPI Solar Panel Inspection REST API.
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """API health status response."""
    status: str = Field(default="ok", description="Service status")
    service: str = Field(default="solar-panel-ai-api", description="Service name")


class PanelResponse(BaseModel):
    """Solar panel entity response."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    panel_id: str
    location: str
    created_at: Optional[str] = None


class PanelListResponse(BaseModel):
    """List of solar panels response."""
    total: int
    items: List[PanelResponse]


class InspectionResponse(BaseModel):
    """Inspection record response matching the Phase 14 required schema."""
    model_config = ConfigDict(from_attributes=True)

    inspection_id: int
    panel_id: str
    location: str
    image_filename: str
    inspection_timestamp: str
    predicted_class: str
    confidence: float
    visual_region_area_percent: float
    severity: str
    urgency: str
    maintenance_action: str
    manual_inspection_recommended: bool
    confidence_warning: Optional[str] = None


class InspectionListResponse(BaseModel):
    """Paginated list of inspection records."""
    total: int
    page: int
    page_size: int
    items: List[InspectionResponse]
