"""
tests/conftest.py — Shared fixtures for pytest
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

import backend.models  # noqa: F401 - ensure models are registered
from backend.database import Base, get_db
from backend.main import app
from backend.services import ml_adapter

TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(
    bind=test_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def override_get_db():
    db: Session = TestSessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    ml_adapter.load_model()
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def auth_client(client):
    """Authenticated client with a fresh test user."""
    resp = await client.post("/api/auth/login", json={"email": "test@example.com", "password": "testpass123"})
    if resp.status_code != 200:
        resp = await client.post("/api/auth/signup", json={"email": "test@example.com", "password": "testpass123"})
    token = resp.json().get("access_token")
    client.headers["Authorization"] = f"Bearer {token}"
    return client
