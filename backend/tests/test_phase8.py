"""Phase 8 Test Suite - Production Readiness Validation."""

import pytest
import time
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.config import Settings
from app.core.cache import InProcessCache
from app.core.database import Base, get_db
from app.services.audit_service import AuditService, log_audit_event
from app.services.storage_backend import LocalStorageBackend
from app.models.audit_log import AuditLog
from tests.shared_db import engine, TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture
def client():
    return TestClient(app)


class TestProductionConfiguration:
    def test_development_defaults(self):
        settings = Settings(environment="development")
        assert settings.is_development is True
        assert settings.is_production is False

    def test_production_mode_detection(self):
        settings = Settings(environment="production")
        assert settings.is_production is True

    def test_cors_origins_parsing(self):
        settings = Settings(cors_origins="http://localhost:3000,http://localhost:8000")
        origins = settings.cors_origins_list
        assert "http://localhost:3000" in origins
        assert "http://localhost:8000" in origins

    def test_database_pool_settings(self):
        settings = Settings(db_pool_size=10, db_max_overflow=20)
        assert settings.db_pool_size == 10
        assert settings.db_max_overflow == 20

    def test_cache_settings(self):
        settings = Settings(cache_enabled=True, cache_ttl_seconds=600)
        assert settings.cache_enabled is True
        assert settings.cache_ttl_seconds == 600

    def test_storage_backend_setting(self):
        settings = Settings(storage_backend="local")
        assert settings.storage_backend == "local"

    def test_job_settings(self):
        settings = Settings(job_max_attempts=5)
        assert settings.job_max_attempts == 5


class TestHealthEndpoints:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "database" in data

    def test_liveness_endpoint(self, client):
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_readiness_endpoint(self, client):
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "checks" in data


class TestLocalStorageBackend:
    def setup_method(self):
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.backend = LocalStorageBackend(base_dir=self.temp_dir)

    def test_save_and_retrieve(self):
        data = b"test file content"
        key = self.backend.save(data)
        retrieved = self.backend.retrieve(key)
        assert retrieved == data

    def test_delete_file(self):
        data = b"to be deleted"
        key = self.backend.save(data)
        assert self.backend.delete(key) is True
        assert self.backend.exists(key) is False

    def test_delete_nonexistent(self):
        assert self.backend.delete("nonexistent-key") is False

    def test_path_traversal_protection(self):
        with pytest.raises(ValueError):
            self.backend.save(b"malicious", storage_key="../etc/passwd")


class TestInProcessCache:
    def setup_method(self):
        self.cache = InProcessCache()

    def test_set_and_get(self):
        self.cache.set("key1", "value1")
        assert self.cache.get("key1") == "value1"

    def test_cache_miss(self):
        assert self.cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        self.cache.set("key1", "value1", ttl_seconds=0.1)
        time.sleep(0.2)
        assert self.cache.get("key1") is None

    def test_delete_entry(self):
        self.cache.set("key1", "value1")
        assert self.cache.delete("key1") is True
        assert self.cache.get("key1") is None

    def test_invalidate_pattern(self):
        self.cache.set("doc:1", "data1")
        self.cache.set("doc:2", "data2")
        self.cache.set("user:1", "userdata")
        count = self.cache.invalidate_pattern("doc:")
        assert count == 2
        assert self.cache.get("doc:1") is None
        assert self.cache.get("user:1") == "userdata"

    def test_clear_all(self):
        self.cache.set("key1", "value1")
        self.cache.clear()
        assert self.cache.get("key1") is None


class TestAuditLogging:
    def test_log_event(self, db_session):
        audit = AuditService(db_session)
        entry = audit.log_event(
            event_type="auth",
            event_action="login",
            user_id=1,
        )
        assert entry.event_type == "auth"
        assert entry.event_action == "login"

    def test_sanitize_sensitive_data(self, db_session):
        audit = AuditService(db_session)
        entry = audit.log_event(
            event_type="auth",
            event_action="login",
            details="password=secret123",
        )
        assert "secret123" not in entry.details

    def test_log_with_ip_address(self, db_session):
        audit = AuditService(db_session)
        entry = audit.log_event(
            event_type="auth",
            event_action="login",
            ip_address="192.168.1.1",
        )
        assert entry.ip_address == "192.168.1.1"


class TestObservability:
    def test_request_id_header(self, client):
        response = client.get("/health")
        assert "X-Request-ID" in response.headers

    def test_custom_request_id(self, client):
        response = client.get("/health", headers={"X-Request-ID": "test-123"})
        assert response.headers.get("X-Request-ID") == "test-123"


class TestSecurity:
    def test_security_headers(self, client):
        response = client.get("/health")
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers

    def test_unauthorized_access(self, client):
        response = client.get("/documents")
        assert response.status_code == 401


class TestDatabaseMigration:
    def test_audit_logs_table_exists(self, db_session):
        from sqlalchemy import inspect
        inspector = inspect(db_session.get_bind())
        tables = inspector.get_table_names()
        assert "audit_logs" in tables
