"""Shared fixtures for the stock-tracker test suite."""

import pytest
from datetime import date
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.auth import get_password_hash, create_access_token
import app.models as _models  # noqa: F401 — ensure all model classes are registered with Base
from app.models import User

# ---------------------------------------------------------------------------
# In-memory SQLite engine shared by all tests in a session
# ---------------------------------------------------------------------------
TEST_DATABASE_URL = "sqlite://"  # in-memory

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@event.listens_for(engine, "connect")
def _enable_fk(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _create_tables():
    """Create all tables before each test and drop them after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db():
    """Yield a SQLAlchemy session that rolls back nothing (tests use real commits
    against the in-memory DB which is wiped via _create_tables)."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db):
    """FastAPI TestClient with the get_db dependency overridden to use the
    in-memory test database."""
    from fastapi.testclient import TestClient
    from main import app

    def _override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def test_user(db) -> dict:
    """Create a test user in the DB and return a dict with user info + auth token."""
    user = User(
        email="testuser@example.com",
        hashed_password=get_password_hash("testpassword123"),
        role="user",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(data={"sub": user.email})
    return {
        "id": user.id,
        "email": user.email,
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"},
    }


@pytest.fixture()
def second_user(db) -> dict:
    """Create a second test user for isolation tests."""
    user = User(
        email="otheruser@example.com",
        hashed_password=get_password_hash("otherpassword456"),
        role="user",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(data={"sub": user.email})
    return {
        "id": user.id,
        "email": user.email,
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"},
    }
