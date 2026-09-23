"""
Database Seeding Script for Solar Panel Inspection System.

Seeds the database with the 12 representative inspection records from
results/metrics/panel_metadata_examples.json using an idempotent strategy.

Usage:
    # Seed using environment-configured DATABASE_URL (or .env)
    python scripts/seed_database.py

    # Seed local SQLite database for verification
    python scripts/seed_database.py --sqlite
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sqlalchemy import select
from src.database.database import (
    create_db_engine,
    get_database_url,
    get_session_factory,
    init_db,
)
from src.database.models import Panel, Inspection
from src.database.repository import PanelRepository, InspectionRepository
from src.utils.config import get_project_root
from src.utils.logger import setup_logger

logger = setup_logger("seed_database_cli")


def mask_db_url(url: str) -> str:
    """Masks password in database URL for safe terminal logging."""
    try:
        parsed = urlparse(url)
        if parsed.password:
            netloc = f"{parsed.username}:*****@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            return parsed._replace(netloc=netloc).geturl()
        return url
    except Exception:
        return "<configured_database_url>"


def seed_records(session, records):
    """
    Idempotently seeds panels and inspections into the active database session.
    Returns (panels_created, inspections_created, inspections_skipped).
    """
    panel_repo = PanelRepository(session)
    insp_repo = InspectionRepository(session)

    panels_created = 0
    inspections_created = 0
    inspections_skipped = 0

    for item in records:
        panel_id = item["panel_id"].strip()
        location = item["location"].strip()

        # 1. Idempotent Panel Insertion
        existing_panel = panel_repo.get_panel(panel_id)
        if not existing_panel:
            panel_repo.create_panel(panel_id=panel_id, location=location)
            panels_created += 1

        # 2. Parse Timestamp
        ts_str = item["inspection_timestamp"].replace("Z", "+00:00")
        dt_val = datetime.fromisoformat(ts_str)

        # 3. Idempotent Inspection Insertion
        # Check if inspection already exists for this panel_id and image_filename
        stmt = select(Inspection).where(
            Inspection.panel_id == panel_id,
            Inspection.image_filename == item["image_filename"].strip(),
        )
        existing_insp = session.execute(stmt).scalar_one_or_none()

        if existing_insp:
            inspections_skipped += 1
        else:
            insp_repo.create_inspection(
                panel_id=panel_id,
                image_filename=item["image_filename"],
                predicted_class=item["predicted_class"],
                confidence=item["confidence"],
                visual_region_area_percent=item["visual_region_area_percent"],
                severity=item["severity"],
                urgency=item["urgency"],
                maintenance_action=item["maintenance_action"],
                inspection_timestamp=dt_val,
                true_class=item.get("true_class"),
                manual_inspection_recommended=item.get(
                    "manual_inspection_recommended", False
                ),
                confidence_warning=item.get("confidence_warning"),
            )
            inspections_created += 1

    return panels_created, inspections_created, inspections_skipped


def main():
    parser = argparse.ArgumentParser(description="Seed database with representative inspection records.")
    parser.add_argument("--url", type=str, default=None, help="Explicit database URL override")
    parser.add_argument("--sqlite", action="store_true", help="Use local SQLite database in data/solar_panel_ai.db")
    args = parser.parse_args()

    root = get_project_root()
    ckpt_path = root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Checkpoint SHA256 integrity check before seeding
    with open(ckpt_path, "rb") as f:
        sha_before = hashlib.sha256(f.read()).hexdigest()

    # Load 12 representative inspection records from Phase 12 JSON
    json_path = root / "results" / "metrics" / "panel_metadata_examples.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Source records file not found at: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if args.sqlite:
        sqlite_db_path = root / "data" / "solar_panel_ai.db"
        sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
        db_url = f"sqlite:///{sqlite_db_path.as_posix()}"
    elif args.url:
        db_url = args.url
    else:
        db_url = get_database_url()

    masked_url = mask_db_url(db_url)
    print("\n" + "=" * 50)
    print("SOLAR PANEL INSPECTION DATABASE SEEDING")
    print("=" * 50)
    print(f"Target Database: {masked_url}")
    print(f"Source JSON    : {json_path.name} ({len(records)} records)")
    print("-" * 50)

    try:
        engine = create_db_engine(db_url)
        # Ensure tables exist
        init_db(engine)

        SessionFactory = get_session_factory(engine)
        with SessionFactory() as session:
            panels_added, insps_added, insps_skipped = seed_records(session, records)

            # Verification Query
            total_panels = len(PanelRepository(session).list_panels())
            total_inspections = len(InspectionRepository(session).list_inspections())

        print("SEEDING SUMMARY:")
        print(f"  Panels Added       : {panels_added}")
        print(f"  Inspections Added  : {insps_added}")
        print(f"  Inspections Skipped: {insps_skipped} (already existed)")
        print(f"  Total Panels in DB : {total_panels}")
        print(f"  Total Insps in DB  : {total_inspections}")
        print("-" * 50)
        print("STATUS: SEEDING COMPLETED SUCCESSFULLY")
        print("=" * 50)
        logger.info(
            f"Seeded {insps_added} inspections into {masked_url} "
            f"(Total: {total_inspections} in DB)"
        )

    except Exception as e:
        print("\nSTATUS: SEEDING FAILED")
        print(f"Error Details: {e}")
        print("-" * 50)
        print("GUIDANCE:")
        print("1. Verify database credentials in .env.")
        print("2. Run setup first: python scripts/setup_database.py")
        print("3. Alternatively, test locally: python scripts/seed_database.py --sqlite")
        print("=" * 50)
        logger.warning(f"Database seeding failed: {e}")

    # Checkpoint SHA256 integrity check after seeding
    with open(ckpt_path, "rb") as f:
        sha_after = hashlib.sha256(f.read()).hexdigest()
    assert sha_before == sha_after, "Checkpoint was modified during seed_database!"


if __name__ == "__main__":
    main()
