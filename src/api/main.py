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
import re
import time
import uuid
from typing import Any, List, Optional

from src.utils.config import get_project_root

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
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
from src.maintenance.maintenance_recommender import MaintenanceRecommender
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
recommender = MaintenanceRecommender()


def _safe_error_detail(detail: Any, fallback: str = "The request could not be completed.") -> str:
    """Normalize user-facing error messages without exposing internal stack traces or secrets."""
    if isinstance(detail, list):
        candidates = [item for item in detail if isinstance(item, dict)]
        if candidates:
            for item in candidates:
                if isinstance(item.get("msg"), str) and item["msg"].strip():
                    return item["msg"].strip()
        return fallback
    if isinstance(detail, dict):
        detail_value = detail.get("detail")
        return _safe_error_detail(detail_value, fallback)
    text = str(detail or "").strip()
    if not text:
        return fallback
    lowered = text.lower()
    if "traceback" in lowered or "sqlalchemy" in lowered or "sqlite" in lowered:
        return fallback
    if re.search(r"(C:\\|/Users/|/home/|/tmp/|/var/)", text):
        return fallback
    if "password" in lowered or "secret" in lowered or "token" in lowered or "api_key" in lowered:
        return fallback
    return text


def _get_upload_max_bytes() -> int:
    """Maximum allowed upload size for the local prototype API."""
    raw = os.getenv("UPLOAD_MAX_BYTES", "10485760")
    try:
        value = int(raw)
        return max(1, value)
    except (TypeError, ValueError):
        return 10 * 1024 * 1024


def _get_max_image_dimensions() -> int:
    """Reject unexpectedly large image dimensions before they trigger expensive decode work."""
    raw = os.getenv("MAX_IMAGE_DIMENSIONS", "12000")
    try:
        value = int(raw)
        return max(256, value)
    except (TypeError, ValueError):
        return 12000


def _get_cors_allowed_origins() -> List[str]:
    """Read CORS origins from environment with local-development defaults for the prototype app."""
    raw = os.getenv("CORS_ALLOWED_ORIGINS")
    if raw:
        values = [item.strip() for item in raw.split(",") if item.strip()]
        if values:
            return values
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]


def _safe_uploaded_filename(filename: Optional[str]) -> str:
    """Normalize uploaded filenames so they remain metadata only and cannot become filesystem paths."""
    candidate = (filename or "").strip()
    if not candidate:
        return "uploaded_panel.jpg"

    normalized = candidate.replace("\\", "/")
    safe_name = Path(normalized).name
    if not safe_name or safe_name in {".", ".."}:
        return "uploaded_panel.jpg"

    sanitized = "".join(ch for ch in safe_name if ch not in {"\x00"})
    sanitized = sanitized.strip()
    if not sanitized:
        return "uploaded_panel.jpg"
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    return sanitized


def _generate_request_id(provided_id: Optional[str] = None) -> str:
    """Generate or reuse a safe correlation/request identifier."""
    candidate = (provided_id or "").strip()
    if candidate:
        try:
            uuid.UUID(candidate)
            return candidate
        except (TypeError, ValueError):
            pass
    return uuid.uuid4().hex


def _response_maintenance_actions(
    predicted_class: str,
    confidence: float,
    severity: str,
    region_area_percent: float,
    manual_inspection_recommended: bool,
) -> List[str]:
    """Backfill 2-3 maintenance actions for legacy inspection records."""
    rec = recommender.get_recommendation(
        predicted_class=predicted_class,
        confidence=confidence,
        severity=severity,
        region_area_percent=region_area_percent,
        manual_inspection_recommended=manual_inspection_recommended,
    )
    return rec.get("maintenance_actions", [rec.get("recommended_action", "Continue routine monitoring.")])


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


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Attach a request ID, log request lifecycle, and add the ID to the response headers."""
    request_id = _generate_request_id(request.headers.get("X-Request-ID"))
    request.state.request_id = request_id
    started = time.perf_counter()

    logger.info(
        "API request received: method=%s path=%s request_id=%s",
        request.method,
        request.url.path,
        request_id,
    )

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.exception(
            "API request failed: method=%s path=%s status=ERROR duration_ms=%s request_id=%s",
            request.method,
            request.url.path,
            duration_ms,
            request_id,
        )
        raise

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "API request completed: method=%s path=%s status=%s duration_ms=%s request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request_id,
    )
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", _generate_request_id())
    safe_detail = _safe_error_detail(exc.detail)
    logger.warning(
        "HTTP exception: method=%s path=%s status=%s detail=%s request_id=%s",
        request.method,
        request.url.path,
        exc.status_code,
        safe_detail,
        request_id,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": safe_detail},
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", _generate_request_id())
    errors = exc.errors()
    safe_detail = _safe_error_detail(errors, "A file upload is required.")
    if any("file" in str(error.get("loc", [])) for error in errors):
        safe_detail = "A file upload is required."
    logger.warning(
        "Request validation error: method=%s path=%s detail=%s request_id=%s",
        request.method,
        request.url.path,
        safe_detail,
        request_id,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": safe_detail},
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    request_id = getattr(request.state, "request_id", _generate_request_id())
    safe_detail = _safe_error_detail(str(exc), "The uploaded image is invalid.")
    logger.warning(
        "Validation error: method=%s path=%s detail=%s request_id=%s",
        request.method,
        request.url.path,
        safe_detail,
        request_id,
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": safe_detail},
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(SQLAlchemyError)
async def db_exception_handler(request: Request, exc: SQLAlchemyError):
    request_id = getattr(request.state, "request_id", _generate_request_id())
    logger.error(
        "Database error: method=%s path=%s request_id=%s",
        request.method,
        request.url.path,
        request_id,
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "A database error occurred while processing this request."},
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", _generate_request_id())
    logger.exception(
        "Unexpected server error: method=%s path=%s request_id=%s",
        request.method,
        request.url.path,
        request_id,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal server error occurred. Please try again."},
        headers={"X-Request-ID": request_id},
    )


# Enable CORS for future frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_allowed_origins(),
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
    request: Request,
    file: UploadFile = File(..., description="Solar panel image (JPEG, PNG, WEBP, BMP)"),
    panel_id: str = Form(default="", description="Optional panel identifier (e.g. SP-HYD-001). Auto-generated if omitted."),
    location: str = Form(default="", description="Optional facility site or array location description"),
    db: Session = Depends(get_db),
):
    """
    Performs visual inspection on an uploaded panel image:
    1. Validates image format (panel_id and location are OPTIONAL).
    2. Runs canonical preprocessing, EfficientNet-B0 prediction, and Grad-CAM.
    3. Calculates approximate fault region and visual severity.
    4. Triages actionable maintenance recommendation and urgency.
    5. Persists inspection record and returns complete diagnostic response.

    If panel_id is not provided, an AUTO-NNN identifier is generated automatically.
    If location is not provided, 'Location not specified' is used as the fallback.
    """
    request_id = getattr(request.state, "request_id", _generate_request_id())
    safe_filename = _safe_uploaded_filename(file.filename)
    logger.info(
        "Inspection request received: request_id=%s panel_id=%s location=%s filename=%s",
        request_id,
        panel_id or "",
        location or "",
        safe_filename,
    )

    # 1. Optional metadata handling
    stripped_id = (panel_id or "").strip()
    stripped_loc = (location or "").strip() or "Location not specified"

    if not file.filename:
        logger.warning(
            "Upload validation failed: request_id=%s reason=%s",
            request_id,
            "Uploaded file is missing a valid filename.",
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is missing a valid filename.",
        )

    if not file.content_type:
        content_type = ""
    else:
        content_type = file.content_type.lower()
    if content_type and not content_type.startswith("image/"):
        logger.warning(
            "Upload validation failed: request_id=%s reason=%s",
            request_id,
            "Unsupported file type.",
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Upload a valid image.",
        )

    # 2. Read and validate the uploaded image before any DB or inference work
    try:
        image_bytes = await file.read()
    except Exception:
        logger.warning(
            "Upload processing failed: request_id=%s reason=%s",
            request_id,
            "Uploaded image could not be processed.",
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image could not be processed.",
        )

    if not image_bytes:
        logger.warning(
            "Upload validation failed: request_id=%s reason=%s",
            request_id,
            "Uploaded file is empty.",
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(image_bytes) > _get_upload_max_bytes():
        logger.warning(
            "Upload validation failed: request_id=%s reason=%s",
            request_id,
            "Image exceeds upload size limit.",
        )
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Uploaded image exceeds the maximum supported size.",
        )

    service = InspectionService.get_instance()
    try:
        service.validate_image_quality(image_bytes)
    except ValueError as e:
        logger.warning(
            "Image quality validation failed: request_id=%s reason=%s",
            request_id,
            str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # 3. Auto-generate a unique panel ID when the user omits it after the image passes validation.
    if not stripped_id:
        panel_repo_check = PanelRepository(db)
        auto_count = db.scalar(
            select(func.count()).where(Panel.panel_id.like("AUTO-%"))
        ) or 0
        candidate_num = auto_count + 1
        while True:
            candidate_id = f"AUTO-{candidate_num:03d}"
            existing = panel_repo_check.get_panel(candidate_id)
            if not existing:
                break
            candidate_num += 1
        stripped_id = candidate_id
        logger.info(f"No panel_id provided — auto-generated: {stripped_id}")

    # 4. Execute ML diagnostic pipeline
    try:
        filename = safe_filename
        diag = service.run_inspection(image_bytes=image_bytes, filename=filename)
    except Exception as e:
        logger.error(
            "Inference execution failed: request_id=%s panel_id=%s error=%s",
            request_id,
            stripped_id,
            str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during AI visual inspection processing.",
        )

    # 4. Idempotent Panel retrieval / registration in DB
    panel_repo = PanelRepository(db)
    panel = panel_repo.get_panel(stripped_id)
    if not panel:
        try:
            panel = panel_repo.create_panel(panel_id=stripped_id, location=stripped_loc)
        except (ValueError, SQLAlchemyError) as e:
            logger.error(
                "Panel persistence failed: request_id=%s panel_id=%s error=%s",
                request_id,
                stripped_id,
                str(e),
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Panel metadata could not be saved.",
            )

    # 5. Persist Inspection record
    insp_repo = InspectionRepository(db)
    ts_val = datetime.fromisoformat(diag["inspection_timestamp"])
    try:
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
    except (ValueError, SQLAlchemyError) as e:
        logger.error(
            "Inspection persistence failed: request_id=%s panel_id=%s error=%s",
            request_id,
            panel.panel_id,
            str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inspection could not be persisted to storage.",
        )

    logger.info(
        "Inspection completed: request_id=%s inspection_id=%s panel_id=%s predicted_class=%s confidence=%s severity=%s",
        request_id,
        inspection.id,
        inspection.panel_id,
        inspection.predicted_class,
        inspection.confidence,
        inspection.severity,
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
        maintenance_actions=diag["maintenance_actions"],
        manual_inspection_recommended=inspection.manual_inspection_recommended,
        confidence_warning=inspection.confidence_warning,
        timestamp=inspection.inspection_timestamp.isoformat(),
        region=inspection.visual_region_area_percent,
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
            maintenance_actions=_response_maintenance_actions(
                predicted_class=i.predicted_class,
                confidence=i.confidence,
                severity=i.severity,
                region_area_percent=i.visual_region_area_percent,
                manual_inspection_recommended=i.manual_inspection_recommended,
            ),
            manual_inspection_recommended=i.manual_inspection_recommended,
            confidence_warning=i.confidence_warning,
            timestamp=i.inspection_timestamp.isoformat(),
            region=i.visual_region_area_percent,
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
            maintenance_actions=_response_maintenance_actions(
                predicted_class=i.predicted_class,
                confidence=i.confidence,
                severity=i.severity,
                region_area_percent=i.visual_region_area_percent,
                manual_inspection_recommended=i.manual_inspection_recommended,
            ),
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
        maintenance_actions=_response_maintenance_actions(
            predicted_class=insp.predicted_class,
            confidence=insp.confidence,
            severity=insp.severity,
            region_area_percent=insp.visual_region_area_percent,
            manual_inspection_recommended=insp.manual_inspection_recommended,
        ),
        manual_inspection_recommended=insp.manual_inspection_recommended,
        confidence_warning=insp.confidence_warning,
        timestamp=insp.inspection_timestamp.isoformat(),
        region=insp.visual_region_area_percent,
    )
