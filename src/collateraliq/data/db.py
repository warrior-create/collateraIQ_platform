"""
Database connection management for CollateralIQ.

Uses SQLAlchemy with SQLite (dev) or PostgreSQL (prod).
Connection string configured via environment variable COLLATERALIQ_DB_URL
or defaults to SQLite at data/collateraliq.db.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

_engines: dict[str, Engine] = {}


def get_engine(db_path: str = "data/collateraliq.db") -> Engine:
    """Return (cached) SQLAlchemy engine."""
    db_url = os.environ.get("COLLATERALIQ_DB_URL")
    if db_url is None:
        if db_path.startswith(("sqlite", "postgres")):
            db_url = db_path
        else:
            # Resolve absolute path for sqlite to avoid cwd issues
            abs_path = Path(db_path).resolve()
            db_url = f"sqlite:///{abs_path}"

    if db_url not in _engines:
        connect_args = {}
        if db_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
        _engines[db_url] = create_engine(
            db_url,
            connect_args=connect_args,
            echo=False,
        )
    return _engines[db_url]


def get_session(db_path: str = "data/collateraliq.db") -> Session:
    """Return a new SQLAlchemy session."""
    engine = get_engine(db_path)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def init_db(db_path: str = "data/collateraliq.db", schema_path: str | None = None) -> None:
    """Create all tables from schema.sql if they don't exist."""
    if schema_path is None:
        schema_path = Path(__file__).parent / "schema.sql"

    engine = get_engine(db_path)
    sql = Path(schema_path).read_text()

    # Split and execute statements individually (SQLite compatibility)
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            try:
                conn.execute(text(stmt))
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise
