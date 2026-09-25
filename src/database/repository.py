"""
Data Access Layer (Repository Pattern) for Solar Inspection Database.

Encapsulates CRUD queries for Panel and Inspection entities.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.database.models import Panel, Inspection


class PanelRepository:
    """Repository operations for solar panels."""

    def __init__(self, session: Session):
        self.session = session

    def create_panel(self, panel_id: str, location: str) -> Panel:
        """Creates and persists a new Panel record."""
        p_id = panel_id.strip()
        loc = location.strip()
        if not p_id:
            raise ValueError("panel_id must not be empty.")

        panel = Panel(panel_id=p_id, location=loc)
        self.session.add(panel)
        self.session.commit()
        self.session.refresh(panel)
        return panel

    def get_panel(self, panel_id: str) -> Optional[Panel]:
        """Finds a panel by its unique alphanumeric panel_id."""
        stmt = select(Panel).where(Panel.panel_id == panel_id.strip())
        return self.session.execute(stmt).scalar_one_or_none()

    def list_panels(self, limit: int = 100, offset: int = 0) -> List[Panel]:
        """Lists panels with pagination support."""
        stmt = select(Panel).order_by(Panel.id).limit(limit).offset(offset)
        return list(self.session.execute(stmt).scalars().all())


class InspectionRepository:
    """Repository operations for inspection records."""

    def __init__(self, session: Session):
        self.session = session

    def create_inspection(
        self,
        panel_id: str,
        image_filename: str,
        predicted_class: str,
        confidence: float,
        visual_region_area_percent: float,
        severity: str,
        urgency: str,
        maintenance_action: str,
        inspection_timestamp: datetime,
        true_class: Optional[str] = None,
        manual_inspection_recommended: bool = False,
        confidence_warning: Optional[str] = None,
    ) -> Inspection:
        """Creates and persists an Inspection record."""
        # Validation checks
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {confidence}")
        if not (0.0 <= visual_region_area_percent <= 100.0):
            raise ValueError(
                f"visual_region_area_percent must be in [0, 100], got {visual_region_area_percent}"
            )
        sev = severity.strip().upper()
        if sev not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError(f"Invalid severity: '{severity}'")
        urg = urgency.strip().upper()
        if urg not in {"ROUTINE", "SCHEDULED", "PRIORITY", "IMMEDIATE_REVIEW"}:
            raise ValueError(f"Invalid urgency: '{urgency}'")

        inspection = Inspection(
            panel_id=panel_id.strip(),
            image_filename=image_filename.strip(),
            true_class=true_class.strip() if true_class else None,
            predicted_class=predicted_class.strip(),
            confidence=float(confidence),
            visual_region_area_percent=float(visual_region_area_percent),
            severity=sev,
            urgency=urg,
            maintenance_action=maintenance_action.strip(),
            manual_inspection_recommended=bool(manual_inspection_recommended),
            confidence_warning=confidence_warning.strip() if confidence_warning else None,
            inspection_timestamp=inspection_timestamp,
        )
        self.session.add(inspection)
        self.session.commit()
        self.session.refresh(inspection)
        return inspection

    def get_inspection(self, inspection_id: int) -> Optional[Inspection]:
        """Finds an inspection by primary key ID."""
        stmt = select(Inspection).where(Inspection.id == inspection_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def list_inspections(self, limit: int = 100, offset: int = 0) -> List[Inspection]:
        """Lists inspections with pagination."""
        stmt = (
            select(Inspection)
            .order_by(Inspection.inspection_timestamp.desc(), Inspection.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.execute(stmt).scalars().all())

    def list_inspections_by_panel(
        self, panel_id: str, limit: int = 100, offset: int = 0
    ) -> List[Inspection]:
        """Lists inspections for a specific panel_id."""
        stmt = (
            select(Inspection)
            .where(Inspection.panel_id == panel_id.strip())
            .order_by(Inspection.inspection_timestamp.desc(), Inspection.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.execute(stmt).scalars().all())


# Standalone procedural helpers for ease of use
def create_panel(session: Session, panel_id: str, location: str) -> Panel:
    return PanelRepository(session).create_panel(panel_id, location)


def get_panel(session: Session, panel_id: str) -> Optional[Panel]:
    return PanelRepository(session).get_panel(panel_id)


def list_panels(session: Session, limit: int = 100, offset: int = 0) -> List[Panel]:
    return PanelRepository(session).list_panels(limit=limit, offset=offset)


def create_inspection(session: Session, **kwargs) -> Inspection:
    return InspectionRepository(session).create_inspection(**kwargs)


def get_inspection(session: Session, inspection_id: int) -> Optional[Inspection]:
    return InspectionRepository(session).get_inspection(inspection_id)


def list_inspections(session: Session, limit: int = 100, offset: int = 0) -> List[Inspection]:
    return InspectionRepository(session).list_inspections(limit=limit, offset=offset)


def list_inspections_by_panel(
    session: Session, panel_id: str, limit: int = 100, offset: int = 0
) -> List[Inspection]:
    return InspectionRepository(session).list_inspections_by_panel(
        panel_id, limit=limit, offset=offset
    )
