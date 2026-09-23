"""
Database engine, session factory, and connection initialization.

Uses SQLAlchemy 2.0 with PostgreSQL (psycopg driver) support, configured
via the DATABASE_URL environment variable.
"""

import os
from pathlib import Path
from typing import Generator, Optional
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from src.database.models import Base

# Load environment variables from .env if present
project_root = Path(__file__).resolve().parent.parent.parent
dotenv_path = project_root / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)
else:
    load_dotenv()

def get_database_url() -> str:
    """
    Retrieves the configured database URL from the environment.
    If DATABASE_URL is set in .env or the environment, it is used.
    Otherwise, defaults to local SQLite development database (data/solar_panel_ai.db).
    """
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url
    sqlite_db_path = project_root / "data" / "solar_panel_ai.db"
    return f"sqlite:///{sqlite_db_path.as_posix()}"



def create_db_engine(url: Optional[str] = None, **kwargs) -> Engine:
    """
    Creates and returns a SQLAlchemy Engine for the provided database URL.
    """
    db_url = url or get_database_url()
    
    # SQLite connection requires specific arguments for threading if used
    engine_kwargs = {}
    if db_url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # Standard connection pool tuning for PostgreSQL
        engine_kwargs["pool_pre_ping"] = True

    engine_kwargs.update(kwargs)
    return create_engine(db_url, **engine_kwargs)


# Global default engine and session factory
_default_engine: Optional[Engine] = None
_DefaultSession: Optional[sessionmaker] = None


def get_engine() -> Engine:
    """Returns the singleton application database engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = create_db_engine()
    return _default_engine


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker:
    """Returns a sessionmaker bound to the given or default engine."""
    global _DefaultSession
    eng = engine or get_engine()
    if engine is not None:
        return sessionmaker(autocommit=False, autoflush=False, bind=eng)
    if _DefaultSession is None:
        _DefaultSession = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    return _DefaultSession


def get_db() -> Generator[Session, None, None]:
    """
    Dependency / context utility yielding a transactional database session.
    """
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
    finally:
        session.close()



def init_db(engine: Optional[Engine] = None) -> None:
    """
    Initializes database tables defined in SQLAlchemy Base metadata.
    Creates tables if they do not already exist. Does NOT drop existing tables.
    """
    eng = engine or get_engine()
    Base.metadata.create_all(bind=eng)
