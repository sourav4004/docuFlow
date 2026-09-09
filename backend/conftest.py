"""Pytest root configuration — shared test database + per-test session cleanup.

The get_db dependency override is a GLOBAL on the FastAPI app. In combined
runs every module's import-time override would fight for it; the module that
happened to be imported last won, corrupting every other module's requests.

Fix: ALL test modules now use the single engine/override from tests.shared_db
(see Step 116), and a session-scoped autouse fixture re-installs that canonical
override after collection so no stray import can displace it.
"""

import pytest

from tests.shared_db import TestingSessionLocal, override_get_db
from app.core.database import get_db
from app.main import app
from app.models.session import UserSession


@pytest.fixture(autouse=True, scope="session")
def _canonical_db_override():
    """Ensure the shared override is authoritative for the whole session."""
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture(autouse=True)
def _clear_db_sessions():
    """Clear database-backed user sessions before each test for isolation."""
    try:
        db = TestingSessionLocal()
        db.query(UserSession).delete()
        db.commit()
        db.close()
    except Exception:
        pass
    yield
    try:
        db = TestingSessionLocal()
        db.query(UserSession).delete()
        db.commit()
        db.close()
    except Exception:
        pass