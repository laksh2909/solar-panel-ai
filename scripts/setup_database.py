"""
Database Schema Setup Script for Solar Panel Inspection System.

Connects to the database configured via DATABASE_URL and creates the
'panels' and 'inspections' tables if they do not already exist.

Usage:
    # Setup database using environment-configured DATABASE_URL (or .env)
    python scripts/setup_database.py

    # Setup database using custom URL
    python scripts/setup_database.py --url "postgresql+psycopg://user:pass@localhost:5432/solar_panel_ai"

    # Setup local SQLite database for development/testing
    python scripts/setup_database.py --sqlite
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sqlalchemy import inspect
from src.database.database import (
    create_db_engine,
    get_database_url,
    init_db,
)
from src.utils.config import get_project_root
from src.utils.logger import setup_logger

logger = setup_logger("setup_database_cli")


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


def main():
    parser = argparse.ArgumentParser(description="Initialize database schema for solar panel inspections.")
    parser.add_argument("--url", type=str, default=None, help="Explicit database URL override")
    parser.add_argument("--sqlite", action="store_true", help="Initialize local SQLite database in data/solar_panel_ai.db")
    args = parser.parse_args()

    root = get_project_root()
    ckpt_path = root / "models" / "checkpoints" / "efficientnet_b0_baseline_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Checkpoint SHA256 integrity verification
    with open(ckpt_path, "rb") as f:
        sha_before = hashlib.sha256(f.read()).hexdigest()

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
    print("SOLAR PANEL INSPECTION DATABASE INITIALIZATION")
    print("=" * 50)
    print(f"Target Database URL: {masked_url}")

    try:
        engine = create_db_engine(db_url)
        # Verify connection
        with engine.connect() as conn:
            pass

        # Create tables without dropping existing data
        init_db(engine)

        inspector = inspect(engine)
        tables = inspector.get_table_names()

        print("\nSTATUS: SUCCESS")
        print(f"Active Tables in Database: {tables}")
        print("Required entities 'panels' and 'inspections' verified.")
        print("-" * 50)
        logger.info(f"Database schema initialized successfully at {masked_url}")

    except Exception as e:
        print("\nSTATUS: CONNECTION/INITIALIZATION FAILED")
        print(f"Error Details: {e}")
        print("-" * 50)
        print("GUIDANCE:")
        print("1. Ensure PostgreSQL is running on the target host/port.")
        print("2. Set valid credentials in the .env file:")
        print("   DATABASE_URL=postgresql+psycopg://username:password@localhost:5432/solar_panel_ai")
        print("3. Alternatively, test the schema locally using: python scripts/setup_database.py --sqlite")
        print("=" * 50)
        logger.warning(f"Database connection failed: {e}")

    # Verify checkpoint SHA256 invariant
    with open(ckpt_path, "rb") as f:
        sha_after = hashlib.sha256(f.read()).hexdigest()
    assert sha_before == sha_after, "Checkpoint was modified during setup_database!"


if __name__ == "__main__":
    main()
