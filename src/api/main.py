"""
FastAPI Application Entry Point for Solar Panel AI Inspection System.

Exposes REST APIs for:
  - System health and metadata
  - Module inspection with real-time ML inference and Grad-CAM
  - Panel inventory and historical inspection tracking
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from src.api.schemas import (
    HealthResponse,
    InspectionListResponse,
    InspectionResponse,
    PanelListResponse,
    PanelResponse,
)
from src.api.service import InspectionService
from src.database.database import get_db, init_db
from src.database.models import Inspection, Panel
from src.database.repository import (
    InspectionRepository,
    PanelRepository,
    create_inspection,
    create_panel,
    get_inspection,
    get_panel,
    list_inspections,
    list_inspections_by_panel,
    list_panels,
)
from src.utils.logger import setup_logger

logger = setup_logger("fastapi_main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes database tables and pre-warms the cached ML inspection model.
    """
    logger.info("Initializing database schema...")
    try:
        init_db()
    except Exception as e:
        logger.warning(f"Database initialization at startup skipped/failed: {e}")

    logger.info("Pre-warming EfficientNet-B0 ML inspection service...")
    try:
        InspectionService.get_instance()
        logger.info("ML inspection service pre-warmed successfully.")
    except Exception as e:
        logger.error(f"Failed to pre-warm ML inspection service: {e}")
        raise

    yield
    logger.info("Shutting down Solar Panel AI Inspection API.")


app = FastAPI(
    title="Solar Panel AI Inspection API",
    description="REST API for automated visual inspection, fault classification, and maintenance recommendations for solar photovoltaic panels.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for future frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
def get_root():
    """Root endpoint providing service information."""
    return {
        "service": "Solar Panel AI Inspection API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/api/health",
    }


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def get_health():
    """Health check endpoint."""
    return HealthResponse(status="ok", service="solar-panel-ai-api")


@app.post(
    "/api/inspect",
    response_model=InspectionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Inspection"],
)
async def inspect_panel(
    file: UploadFile = File(..., description="Solar panel image (JPEG, PNG, WEBP, BMP)"),
    panel_id: str = Form(default="", description="Unique alphanumeric panel identifier (e.g. SP-HYD-001)"),
    location: str = Form(default="", description="Facility site or array location description"),
    db: Session = Depends(get_db),

):
    """
    Performs visual inspection on an uploaded panel image:
    1. Validates panel metadata and image format.
    2. Runs canonical preprocessing, EfficientNet-B0 prediction, and Grad-CAM.
    3. Calculates approximate fault region and visual severity.
    4. Triages actionable maintenance recommendation and urgency.
    5. Persists inspection record and returns complete diagnostic response.
    """
    # 1. Metadata validation
    stripped_id = (panel_id or "").strip()
    if not stripped_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="panel_id cannot be empty or whitespace only.",
        )

    stripped_loc = (location or "").strip()
    if not stripped_loc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="location cannot be empty or whitespace only.",
        )

    # 2. Image reading and validation
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {e}",
        )

    service = InspectionService.get_instance()
    try:
        service.validate_and_decode_image(image_bytes)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # 3. Execute ML diagnostic pipeline
    try:
        filename = file.filename or "uploaded_panel.jpg"
        diag = service.run_inspection(image_bytes=image_bytes, filename=filename)
    except Exception as e:
        logger.error(f"Inference execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during AI visual inspection processing.",
        )

    # 4. Idempotent Panel retrieval / registration in DB
    panel_repo = PanelRepository(db)
    panel = panel_repo.get_panel(stripped_id)
    if not panel:
        panel = panel_repo.create_panel(panel_id=stripped_id, location=stripped_loc)

    # 5. Persist Inspection record
    insp_repo = InspectionRepository(db)
    ts_val = datetime.fromisoformat(diag["inspection_timestamp"])
    inspection = insp_repo.create_inspection(
        panel_id=panel.panel_id,
        image_filename=diag["image_filename"],
        predicted_class=diag["predicted_class"],
        confidence=diag["confidence"],
        visual_region_area_percent=diag["visual_region_area_percent"],
        severity=diag["severity"],
        urgency=diag["urgency"],
        maintenance_action=diag["maintenance_action"],
        inspection_timestamp=ts_val,
        true_class=None,
        manual_inspection_recommended=diag["manual_inspection_recommended"],
        confidence_warning=diag["confidence_warning"],
    )

    return InspectionResponse(
        inspection_id=inspection.id,
        panel_id=inspection.panel_id,
        location=panel.location,
        image_filename=inspection.image_filename,
        inspection_timestamp=inspection.inspection_timestamp.isoformat(),
        predicted_class=inspection.predicted_class,
        confidence=inspection.confidence,
        visual_region_area_percent=inspection.visual_region_area_percent,
        severity=inspection.severity,
        urgency=inspection.urgency,
        maintenance_action=inspection.maintenance_action,
        manual_inspection_recommended=inspection.manual_inspection_recommended,
        confidence_warning=inspection.confidence_warning,
    )


@app.get("/api/panels", response_model=PanelListResponse, tags=["Panels"])
def get_panels(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Retrieves all registered solar panels."""
    repo = PanelRepository(db)
    total = db.scalar(select(func.count(Panel.id))) or 0
    panels = repo.list_panels(limit=limit, offset=offset)

    items = [
        PanelResponse(
            id=p.id,
            panel_id=p.panel_id,
            location=p.location,
            created_at=p.created_at.isoformat() if p.created_at else None,
        )
        for p in panels
    ]
    return PanelListResponse(total=total, items=items)


@app.get("/api/panels/{panel_id}", response_model=PanelResponse, tags=["Panels"])
def get_panel_by_id(
    panel_id: str,
    db: Session = Depends(get_db),
):
    """Retrieves a single solar panel by its alphanumeric identifier."""
    repo = PanelRepository(db)
    panel = repo.get_panel(panel_id)
    if not panel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Panel '{panel_id}' not found.",
        )
    return PanelResponse(
        id=panel.id,
        panel_id=panel.panel_id,
        location=panel.location,
        created_at=panel.created_at.isoformat() if panel.created_at else None,
    )


@app.get(
    "/api/panels/{panel_id}/inspections",
    response_model=List[InspectionResponse],
    tags=["Panels"],
)
def get_panel_inspections(
    panel_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Retrieves historical visual inspections for a specific panel."""
    panel_repo = PanelRepository(db)
    panel = panel_repo.get_panel(panel_id)
    if not panel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Panel '{panel_id}' not found.",
        )

    insp_repo = InspectionRepository(db)
    inspections = insp_repo.list_inspections_by_panel(
        panel_id=panel_id, limit=limit, offset=offset
    )

    return [
        InspectionResponse(
            inspection_id=i.id,
            panel_id=i.panel_id,
            location=panel.location,
            image_filename=i.image_filename,
            inspection_timestamp=i.inspection_timestamp.isoformat(),
            predicted_class=i.predicted_class,
            confidence=i.confidence,
            visual_region_area_percent=i.visual_region_area_percent,
            severity=i.severity,
            urgency=i.urgency,
            maintenance_action=i.maintenance_action,
            manual_inspection_recommended=i.manual_inspection_recommended,
            confidence_warning=i.confidence_warning,
        )
        for i in inspections
    ]


@app.get("/api/inspections", response_model=InspectionListResponse, tags=["Inspections"])
def get_all_inspections(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db),
):
    """Retrieves paginated visual inspection records across all panels."""
    offset = (page - 1) * page_size
    total = db.scalar(select(func.count(Inspection.id))) or 0

    insp_repo = InspectionRepository(db)
    inspections = insp_repo.list_inspections(limit=page_size, offset=offset)

    panel_repo = PanelRepository(db)
    # Pre-fetch locations to avoid N+1 queries
    panel_ids = {i.panel_id for i in inspections}
    panels = {p.panel_id: p.location for p in [panel_repo.get_panel(pid) for pid in panel_ids] if p}

    items = [
        InspectionResponse(
            inspection_id=i.id,
            panel_id=i.panel_id,
            location=panels.get(i.panel_id, "Unknown Location"),
            image_filename=i.image_filename,
            inspection_timestamp=i.inspection_timestamp.isoformat(),
            predicted_class=i.predicted_class,
            confidence=i.confidence,
            visual_region_area_percent=i.visual_region_area_percent,
            severity=i.severity,
            urgency=i.urgency,
            maintenance_action=i.maintenance_action,
            manual_inspection_recommended=i.manual_inspection_recommended,
            confidence_warning=i.confidence_warning,
        )
        for i in inspections
    ]

    return InspectionListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=items,
    )


@app.get(
    "/api/inspections/{inspection_id}",
    response_model=InspectionResponse,
    tags=["Inspections"],
)
def get_inspection_by_id(
    inspection_id: int,
    db: Session = Depends(get_db),
):
    """Retrieves a single inspection record by its primary key ID."""
    insp_repo = InspectionRepository(db)
    insp = insp_repo.get_inspection(inspection_id)
    if not insp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )

    panel_repo = PanelRepository(db)
    panel = panel_repo.get_panel(insp.panel_id)
    loc = panel.location if panel else "Unknown Location"

    return InspectionResponse(
        inspection_id=insp.id,
        panel_id=insp.panel_id,
        location=loc,
        image_filename=insp.image_filename,
        inspection_timestamp=insp.inspection_timestamp.isoformat(),
        predicted_class=insp.predicted_class,
        confidence=insp.confidence,
        visual_region_area_percent=insp.visual_region_area_percent,
        severity=insp.severity,
        urgency=insp.urgency,
        maintenance_action=insp.maintenance_action,
        manual_inspection_recommended=insp.manual_inspection_recommended,
        confidence_warning=insp.confidence_warning,
    )
