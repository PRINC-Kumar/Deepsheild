"""
database.py — SQLite via SQLAlchemy 2.0
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import settings

# Normalize database URL
db_url = settings.DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://")

engine = create_engine(
    db_url,
    echo=False,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create all tables on startup."""
    import backend.models  # noqa: F401 - ensure models are registered on Base
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a DB session."""
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
