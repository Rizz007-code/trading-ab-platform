# data/database/connection.py
"""
Database connection and session management.
Uses SQLAlchemy 2.0 style with context managers.
"""

import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

load_dotenv()

# ─── Build Database URL ────────────────────────────────────────────────────────
def _build_database_url() -> str:
    """Build the database URL from environment variables."""
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    user     = os.getenv("POSTGRES_USER", "trading_user")
    password = os.getenv("POSTGRES_PASSWORD", "trading_pass")
    host     = os.getenv("POSTGRES_HOST", "localhost")
    port     = os.getenv("POSTGRES_PORT", "5432")
    db       = os.getenv("POSTGRES_DB", "trading_db")

    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


DATABASE_URL = _build_database_url()

# ─── Engine ───────────────────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,           # Max persistent connections
    max_overflow=20,        # Extra connections under load
    pool_pre_ping=True,     # Test connection before using (handles dropped conns)
    pool_recycle=3600,      # Recycle connections after 1 hour
    echo=False,             # Set True to log all SQL (debug only)
)

# ─── Session Factory ──────────────────────────────────────────────────────────
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,  # Keep objects usable after commit
)


# ─── Context Manager (for scripts & Airflow tasks) ────────────────────────────
@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Usage:
        with get_db_session() as session:
            session.add(obj)
            session.commit()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


# ─── FastAPI Dependency ───────────────────────────────────────────────────────
def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency injection for database sessions.

    Usage in FastAPI:
        @router.get("/")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── Health Check ─────────────────────────────────────────────────────────────
def check_db_connection() -> bool:
    """Verify the database is reachable. Used in startup events."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection verified.")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


# ─── Table Initializer ────────────────────────────────────────────────────────
def init_db() -> None:
    """
    Create all tables defined in models.py.
    Call this once on first run or in Docker entrypoint.
    """
    from data.database.models import Base  # local import to avoid circular deps
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ All tables created successfully.")


if __name__ == "__main__":
    check_db_connection()
    init_db()
