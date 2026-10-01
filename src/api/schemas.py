"""
Pydantic Schemas for FastAPI Solar Panel Inspection REST API.
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    """Inspection record response with canonical fields and backward-compatible aliases."""
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
    maintenance_actions: List[str] = Field(default_factory=list)
    manual_inspection_recommended: bool
    confidence_warning: Optional[str] = None
    timestamp: Optional[str] = None
    region: Optional[float] = None

    @model_validator(mode="after")
    def sync_compatibility_aliases(self):
        """Keep legacy aliases identical to their canonical counterpart."""
        self.timestamp = self.inspection_timestamp
        self.region = self.visual_region_area_percent
        return self


class InspectionListResponse(BaseModel):
    """Paginated list of inspection records."""
    total: int
    page: int
    page_size: int
    items: List[InspectionResponse]
