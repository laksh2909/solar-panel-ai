"""
SQLAlchemy ORM Models for Solar Photovoltaic Inspection Database.

Defines two core entities:
  1. Panel: Represents an inspected solar module identified by panel_id and location.
  2. Inspection: Represents an inspection event recording classification, visual severity,
                 urgency, recommended maintenance action, and metadata.

IMPORTANT ARCHITECTURAL & OPERATIONAL BOUNDARIES:
- The database stores APPLICATION and OPERATIONAL inspection history only.
- It is NOT used for ML model training, feature engineering, or altering inference logic.
- EfficientNet-B0 inference operates independently from database persistence.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Panel(Base):
    """
    Represents a solar photovoltaic panel installed at a facility/site.
    """
    __tablename__ = "panels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    panel_id = Column(String(100), unique=True, nullable=False, index=True)
    location = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    # Relationship to inspections
    inspections = relationship(
        "Inspection",
        back_populates="panel",
        cascade="all, delete-orphan",
        order_by="desc(Inspection.inspection_timestamp)",
    )

    __table_args__ = (
        CheckConstraint("length(trim(panel_id)) > 0", name="ck_panels_panel_id_nonempty"),
        CheckConstraint("length(trim(location)) > 0", name="ck_panels_location_nonempty"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "panel_id": self.panel_id,
            "location": self.location,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<Panel(id={self.id}, panel_id='{self.panel_id}', location='{self.location}')>"


class Inspection(Base):
    """
    Represents a single solar module visual inspection record.
    """
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    panel_id = Column(
        String(100),
        ForeignKey("panels.panel_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    image_filename = Column(String(255), nullable=False)
    true_class = Column(String(100), nullable=True)  # Nullable: field images may lack ground-truth
    predicted_class = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    visual_region_area_percent = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False)
    urgency = Column(String(30), nullable=False)
    maintenance_action = Column(Text, nullable=False)
    manual_inspection_recommended = Column(Boolean, nullable=False, default=False)
    confidence_warning = Column(Text, nullable=True)
    inspection_timestamp = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    # Relationship to parent panel
    panel = relationship("Panel", back_populates="inspections")

    __table_args__ = (
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="ck_inspections_confidence_range",
        ),
        CheckConstraint(
            "visual_region_area_percent >= 0.0 AND visual_region_area_percent <= 100.0",
            name="ck_inspections_area_range",
        ),
        CheckConstraint(
            "severity IN ('LOW', 'MEDIUM', 'HIGH')",
            name="ck_inspections_severity_valid",
        ),
        CheckConstraint(
            "urgency IN ('ROUTINE', 'SCHEDULED', 'PRIORITY', 'IMMEDIATE_REVIEW')",
            name="ck_inspections_urgency_valid",
        ),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "panel_id": self.panel_id,
            "image_filename": self.image_filename,
            "true_class": self.true_class,
            "predicted_class": self.predicted_class,
            "confidence": round(float(self.confidence), 4),
            "visual_region_area_percent": round(float(self.visual_region_area_percent), 2),
            "severity": self.severity,
            "urgency": self.urgency,
            "maintenance_action": self.maintenance_action,
            "manual_inspection_recommended": self.manual_inspection_recommended,
            "confidence_warning": self.confidence_warning,
            "inspection_timestamp": (
                self.inspection_timestamp.isoformat()
                if self.inspection_timestamp
                else None
            ),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return (
            f"<Inspection(id={self.id}, panel_id='{self.panel_id}', "
            f"predicted='{self.predicted_class}', severity='{self.severity}', urgency='{self.urgency}')>"
        )
