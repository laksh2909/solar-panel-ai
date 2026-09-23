"""
Database access module for solar panel inspection persistence.
"""

from src.database.models import Base, Panel, Inspection
from src.database.database import (
    get_database_url,
    create_db_engine,
    get_engine,
    get_session_factory,
    get_db,
    init_db,
)
from src.database.repository import (
    PanelRepository,
    InspectionRepository,
    create_panel,
    get_panel,
    list_panels,
    create_inspection,
    get_inspection,
    list_inspections,
    list_inspections_by_panel,
)

__all__ = [
    "Base",
    "Panel",
    "Inspection",
    "get_database_url",
    "create_db_engine",
    "get_engine",
    "get_session_factory",
    "get_db",
    "init_db",
    "PanelRepository",
    "InspectionRepository",
    "create_panel",
    "get_panel",
    "list_panels",
    "create_inspection",
    "get_inspection",
    "list_inspections",
    "list_inspections_by_panel",
]
