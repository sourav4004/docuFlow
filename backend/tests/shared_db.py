"""Shared test database — one engine and one get_db override for ALL test modules.

Previously each test module defined its own in-memory engine and installed its
own ``app.dependency_overrides[get_db]`` at import time. Because the override
is a global, the LAST module imported in a combined run served requests for
EVERY module, so fixtures written via a module's own session were invisible to
the API and combined runs produced hundreds of spurious failures.

This module provides the single canonical engine/session/override. Every test
module imports ``TestingSessionLocal`` and ``override_get_db`` from here so
that (a) module fixtures and API requests share the same SQLite connection
(StaticPool) and (b) combined runs behave identically to isolated runs.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


# Create the full schema once at import time. All models must be imported
# first so Base.metadata is complete.
from app.models import *  # noqa: E402,F401,F403  (register all tables)
from app.core.database import Base as _Base  # noqa: E402

_Base.metadata.create_all(bind=engine)